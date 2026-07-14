"""
Step 1b (optional): Download Gaia DR4 epoch astrometry for cross-matched sources.

For each source_id in the HST footprint (identified during cross-matching), this
module fetches the individual CCD-transit along-scan (AL) measurements that will
be available in Gaia DR4.  Two access paths are supported:

  1. DataLink (production DR4): per-source query against the ESA GACS archive
     using ``retrieval_type='EPOCH_ASTROMETRY'``.
  2. Pre-release VOTable (development/testing): a locally-stored VOTable file
     containing ESA's illustrative 12-source pre-release sample.

After fetching, the module re-solves the 5-parameter astrometric solution for
each source using the ``gaiasupdate`` package, producing updated (pmra, pmdec,
parallax) values and covariances that can replace the DR4 summary catalog values
as priors in the BP3M joint solve.

Output is written to ``{output_dir}/Gaia/epoch/``:
  - Per-source cache: ``{source_id}_epoch.csv`` (raw AL transits)
  - Epoch solutions: ``{field_name}_gaia_epoch_solutions.csv``

─────────────────────────────────────────────────────────────────────────────
Future extension — direct AL observations in the BP3M joint solve
─────────────────────────────────────────────────────────────────────────────
The raw AL measurements can be included directly in BP3M's normal-equation
system as additional 1D observation equations alongside HST detections.

The observation equation for one AL transit k of source i is:

    ψ_ik = sin(θ_k) · Δα*_i·cos(δ) + cos(θ_k) · Δδ_i
           + sin(θ_k) · (t_k − t_0) · μα*_i
           + cos(θ_k) · (t_k − t_0) · μδ_i
           + f_al_k · ϖ_i
           [+ colour_factor_k · Δν_i  for 6-parameter model]

where:
  θ_k            = scan_pos_angle (radians)
  f_al_k         = parallax_factor_al
  t_k − t_0      = (obs_time_tcb_years) − ref_epoch_dr4 in Julian years
  Δα*_i, Δδ_i   = BP3M position offsets from the Gaia DR4 reference (mas)
  μα*_i, μδ_i   = proper motions (mas/yr)
  ϖ_i            = parallax (mas)
  Δν_i           = chromaticity offset (6p model only)

This is exactly the design matrix used internally by gaiasupdate.  Unlike HST,
Gaia has no per-epoch "image transformation" to solve for — the scanning
geometry (scan_pos_angle, parallax_factor_al) already encodes the mapping from
source parameters to AL measurement.

Practical considerations for implementation:
  - AGIS down-weighting: use ``used_by_agis_al`` flag; up-weight transits
    rejected by AGIS only after iterative convergence (as in gaiasupdate).
  - Per-source excess noise: ``agis_source_excess_noise`` (mas) inflates per-
    transit uncertainties: σ_eff² = centroid_pos_error_al² + excess_noise².
  - Across-scan (AC) measurements are lower weight (~10 mas vs ~0.1 mas AL)
    and are generally not useful for astrometry; ignore them.
  - Reference epoch for DR4 is 2017.5 TCB (not 2016.0 used in DR3).
  - Timing: obs_time_tcb is a Java Long in nanoseconds from
    2010-01-01T00:00:00 TCB.  Convert to JYear via:
      t_years = 2010.0 + obs_time_tcb_ns / 1e9 / 365.25 / 86400

See solver.py for the planned integration point (_add_gaia_epoch_obs, flagged
with use_gaia_al_obs=True).
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

# Reference epoch for Gaia DR4 (Julian year, TCB).  DR3 used 2016.0.
DR4_REF_EPOCH_JYEAR: float = 2017.5

# DataLink endpoint and data-release string (will be updated to the public DR4
# string when ESA releases it; this value is for the internal pre-release).
_GACS_DATA_SERVER: str = "https://gea.esac.esa.int/"
_DR4_DATA_RELEASE: str = "Gaia DR4"        # update to exact DR4 string at release
_DR4_INT_DATA_RELEASE: str = "Gaia DR4_INT4"  # internal pre-release label

# Conversion: obs_time_tcb Java Long (nanoseconds from 2010-01-01 TCB) → Julian year
_TCB_EPOCH_JD: float = 2455197.5  # JD of 2010-01-01T00:00:00 TCB (approx)
_NS_PER_JYEAR: float = 365.25 * 86400 * 1e9

# Columns that must be present in a valid epoch astrometry DataFrame
_REQUIRED_EPOCH_COLS = [
    "source_id",
    "obs_time_tcb",
    "centroid_pos_al",
    "centroid_pos_error_al",
    "scan_pos_angle",
    "parallax_factor_al",
    "colour_factor_al",
    "used_by_agis_al",
    "agis_source_excess_noise",
    "ra0",
    "dec0",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs_time_to_jyear(obs_time_tcb: np.ndarray) -> np.ndarray:
    """Convert obs_time_tcb (nanoseconds from 2010-01-01 TCB) to Julian year."""
    return 2010.0 + np.asarray(obs_time_tcb, dtype=np.float64) / _NS_PER_JYEAR


def _load_prerelease_votable(votable_path: str | Path) -> pd.DataFrame:
    """Load ESA's pre-release epoch astrometry VOTable into a DataFrame.

    The pre-release ZIP from
      https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/
      gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip
    contains a single VOTable XML file with 12 illustrative sources.

    Returns a DataFrame with source_id as int64.
    """
    from astropy.table import Table
    tbl = Table.read(str(votable_path), format="votable")
    df = tbl.to_pandas()
    # Ensure source_id is int64 (never float — see gaia_ids memory note)
    df["source_id"] = df["source_id"].astype(np.int64)
    return df


# ── DataLink access ───────────────────────────────────────────────────────────

def _datalink_epoch_one(
    source_id: int,
    gaia_data_server: str = _GACS_DATA_SERVER,
    data_release: str = _DR4_DATA_RELEASE,
    credentials_file: str | None = None,
) -> pd.DataFrame:
    """Fetch epoch astrometry for one source via GACS DataLink.

    Requires the ``gaiasupdate`` package (pip install gaiasupdate) and ESA
    archive credentials for non-public releases.

    Parameters
    ----------
    source_id
        Gaia DR4 source_id (int64).
    gaia_data_server
        Base URL of the GACS DataLink server.
    data_release
        DR4 data-release string (e.g. 'Gaia DR4').  For the internal pre-
        release use 'Gaia DR4_INT4'.
    credentials_file
        Path to ESA credentials file.  None for public/anonymous access.

    Returns
    -------
    DataFrame of epoch astrometry transits for this source.
    """
    try:
        from gaiasupdate.epoch_astrometry import GaiaSourceEpochAstrometryArchive
    except ImportError as exc:
        raise ImportError(
            "gaiasupdate is required for DataLink access: pip install gaiasupdate"
        ) from exc

    src = GaiaSourceEpochAstrometryArchive.from_gacs_datalink(
        source_id,
        format="votable",
        gaia_data_server=gaia_data_server,
        credentials_file=credentials_file,
        data_release=data_release,
        data_structure="RAW",
        retrieval_type="EPOCH_ASTROMETRY",
    )
    df = src.epoch_astrometry.copy()
    df["source_id"] = np.int64(source_id)
    return df


# ── Per-source caching ────────────────────────────────────────────────────────

def _cache_path_for_source(epoch_cache_dir: Path, source_id: int) -> Path:
    return epoch_cache_dir / f"{source_id}_epoch.csv"


def _load_cached_epoch(epoch_cache_dir: Path, source_id: int) -> pd.DataFrame | None:
    p = _cache_path_for_source(epoch_cache_dir, source_id)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["source_id"] = df["source_id"].astype(np.int64)
    return df


def _save_epoch_cache(epoch_cache_dir: Path, source_id: int, df: pd.DataFrame) -> None:
    p = _cache_path_for_source(epoch_cache_dir, source_id)
    df.to_csv(p, index=False)


# ── Batch download ────────────────────────────────────────────────────────────

def download_epoch_astrometry(
    source_ids: list[int],
    epoch_cache_dir: str | Path,
    access: str = "datalink",
    prerelease_votable: str | Path | None = None,
    gaia_data_server: str = _GACS_DATA_SERVER,
    data_release: str = _DR4_DATA_RELEASE,
    credentials_file: str | None = None,
    n_workers: int = 4,
    force: bool = False,
    verbose: bool = True,
) -> dict[int, pd.DataFrame]:
    """Download Gaia DR4 epoch astrometry for a list of source_ids.

    Parameters
    ----------
    source_ids
        List of Gaia DR4 source_ids (int64).
    epoch_cache_dir
        Directory for per-source CSV cache files.
    access
        'datalink' (default) — query GACS per source.
        'prerelease' — filter from a local pre-release VOTable.
    prerelease_votable
        Path to the pre-release VOTable file.  Required when access='prerelease'.
    n_workers
        Number of parallel DataLink workers (only used for access='datalink').
    force
        Re-download even if cached.

    Returns
    -------
    Dict mapping source_id (int64) → epoch DataFrame.  Sources for which no
    epoch data could be obtained are absent from the dict.
    """
    epoch_cache_dir = Path(epoch_cache_dir)
    epoch_cache_dir.mkdir(parents=True, exist_ok=True)

    source_ids = [np.int64(s) for s in source_ids]
    results: dict[int, pd.DataFrame] = {}

    if access == "prerelease":
        if prerelease_votable is None:
            raise ValueError("prerelease_votable must be set when access='prerelease'")
        df_all = _load_prerelease_votable(prerelease_votable)
        available = set(df_all["source_id"].values)
        found = 0
        for sid in source_ids:
            cached = _load_cached_epoch(epoch_cache_dir, sid) if not force else None
            if cached is not None:
                results[sid] = cached
                found += 1
                continue
            sub = df_all[df_all["source_id"] == sid].copy()
            if len(sub) == 0:
                continue
            _save_epoch_cache(epoch_cache_dir, sid, sub)
            results[sid] = sub
            found += 1
        if verbose:
            print(f"  Pre-release VOTable: {found}/{len(source_ids)} sources found "
                  f"({len(available)} total in file)")
        return results

    # DataLink path: one request per source, parallelised
    to_fetch = []
    for sid in source_ids:
        cached = _load_cached_epoch(epoch_cache_dir, sid) if not force else None
        if cached is not None:
            results[sid] = cached
        else:
            to_fetch.append(sid)

    if verbose and to_fetch:
        print(f"  Epoch astrometry: {len(results)} cached, {len(to_fetch)} to fetch "
              f"via DataLink ...")

    def _fetch_one(sid):
        try:
            df = _datalink_epoch_one(
                sid, gaia_data_server=gaia_data_server,
                data_release=data_release,
                credentials_file=credentials_file,
            )
            _save_epoch_cache(epoch_cache_dir, sid, df)
            return sid, df
        except Exception as e:
            if verbose:
                print(f"    WARNING: epoch fetch failed for {sid}: {e}", flush=True)
            return sid, None

    if n_workers > 1 and to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as exe:
            futures = {exe.submit(_fetch_one, sid): sid for sid in to_fetch}
            n_done = 0
            for fut in concurrent.futures.as_completed(futures):
                sid, df = fut.result()
                if df is not None:
                    results[sid] = df
                n_done += 1
                if verbose and n_done % 100 == 0:
                    print(f"    {n_done}/{len(to_fetch)} fetched ...", flush=True)
    else:
        for i, sid in enumerate(to_fetch):
            _, df = _fetch_one(sid)
            if df is not None:
                results[sid] = df
            if verbose and (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(to_fetch)} fetched ...", flush=True)

    if verbose:
        print(f"  Epoch astrometry: {len(results)}/{len(source_ids)} sources have data")
    return results


# ── gaiasupdate re-solve ──────────────────────────────────────────────────────

def gaiasupdate_one(
    epoch_df: pd.DataFrame,
    source_id: int,
    summary_row: pd.Series | None = None,
    model: str = "5p_single_source",
    compute_excess_noise: bool = False,
) -> dict | None:
    """Re-solve 5D astrometric parameters for one source using gaiasupdate.

    Parameters
    ----------
    epoch_df
        DataFrame of AL transits for this source (from download_epoch_astrometry).
    source_id
        Source identifier (int64, for labelling only).
    summary_row
        Row from the Gaia DR4 summary catalog (used for nu_eff if not in epoch
        data).  Optional.
    model
        gaiasupdate model string:
          '5p_single_source'              — α, δ, ϖ, μα*, μδ
          '3p_single_source_without_offsets' — ϖ, μα*, μδ only
          '6p_constrained_colour'         — 5p + chromaticity (default in gaiasupdate)
          '6p_perspective_acceleration'   — 5p + perspective acceleration
    compute_excess_noise
        Re-estimate per-source excess noise during the solve.

    Returns
    -------
    Dict with keys:
        source_id, model,
        delta_alpha_mas, delta_delta_mas,
        parallax_mas, parallax_error_mas,
        pmra_maspyr, pmdec_maspyr,
        pmra_error_maspyr, pmdec_error_maspyr,
        pmra_pmdec_corr, pmra_plx_corr, pmdec_plx_corr,
        n_transits_used, n_transits_agis, excess_noise_mas,
        ra0_deg, dec0_deg, ref_epoch_jyear
    or None if the solve failed.
    """
    try:
        from gaiasupdate.epoch_astrometry import GaiaEpochAstrometryArchive
    except ImportError as exc:
        raise ImportError(
            "gaiasupdate is required: pip install gaiasupdate"
        ) from exc

    try:
        result = GaiaEpochAstrometryArchive.supdate(
            epoch_df,
            int(source_id),
            model=model,
            compute_excess_noise=compute_excess_noise,
        )
    except Exception as e:
        return None

    if result is None:
        return None

    # Extract covariance diagonal + key correlations.
    # gaiasupdate returns keys like 'muAlphaStar_maspyr', 'muDelta_maspyr',
    # 'varpi_mas', 'delta_alpha_mas', 'delta_delta_mas' (updates) plus
    # corresponding '_uncertainty' entries and a 'covariance_matrix' (5×5 or 6×6).
    cov = result.get("covariance_matrix")
    pmra_err = pmdec_err = plx_err = np.nan
    pmra_pmdec_corr = pmra_plx_corr = pmdec_plx_corr = np.nan
    if cov is not None:
        cov = np.asarray(cov)
        # Parameter order in gaiasupdate 5p: [Δα*, Δδ, μα*, μδ, ϖ]
        if cov.shape[0] >= 5:
            pmra_err = float(np.sqrt(max(cov[2, 2], 0.0)))
            pmdec_err = float(np.sqrt(max(cov[3, 3], 0.0)))
            plx_err = float(np.sqrt(max(cov[4, 4], 0.0)))
            if pmra_err > 0 and pmdec_err > 0:
                pmra_pmdec_corr = float(cov[2, 3] / (pmra_err * pmdec_err))
            if pmra_err > 0 and plx_err > 0:
                pmra_plx_corr = float(cov[2, 4] / (pmra_err * plx_err))
            if pmdec_err > 0 and plx_err > 0:
                pmdec_plx_corr = float(cov[3, 4] / (pmdec_err * plx_err))

    # Count transits
    n_total = len(epoch_df)
    n_agis = int(epoch_df["used_by_agis_al"].sum()) if "used_by_agis_al" in epoch_df.columns else np.nan

    # Reference position from epoch data
    ra0 = float(epoch_df["ra0"].iloc[0]) if "ra0" in epoch_df.columns else np.nan
    dec0 = float(epoch_df["dec0"].iloc[0]) if "dec0" in epoch_df.columns else np.nan

    return dict(
        source_id=np.int64(source_id),
        model=model,
        delta_alpha_mas=float(result.get("delta_alpha_mas", np.nan)),
        delta_delta_mas=float(result.get("delta_delta_mas", np.nan)),
        parallax_mas=float(result.get("varpi_mas", np.nan)),
        parallax_error_mas=float(result.get("varpi_mas_uncertainty",
                                 result.get("varpi_uncertainty", plx_err))),
        pmra_maspyr=float(result.get("muAlphaStar_maspyr", np.nan)),
        pmdec_maspyr=float(result.get("muDelta_maspyr", np.nan)),
        pmra_error_maspyr=float(result.get("muAlphaStar_maspyr_uncertainty",
                                result.get("muAlphaStar_uncertainty", pmra_err))),
        pmdec_error_maspyr=float(result.get("muDelta_maspyr_uncertainty",
                                 result.get("muDelta_uncertainty", pmdec_err))),
        pmra_pmdec_corr=pmra_pmdec_corr,
        pmra_plx_corr=pmra_plx_corr,
        pmdec_plx_corr=pmdec_plx_corr,
        n_transits_used=n_total,
        n_transits_agis=n_agis,
        excess_noise_mas=float(result.get("excess_noise", np.nan)),
        ra0_deg=ra0,
        dec0_deg=dec0,
        ref_epoch_jyear=DR4_REF_EPOCH_JYEAR,
    )


def batch_epoch_solutions(
    source_ids: list[int],
    epoch_data: dict[int, pd.DataFrame],
    model: str = "5p_single_source",
    compute_excess_noise: bool = False,
    n_workers: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run gaiasupdate for all sources that have epoch data.

    Parameters
    ----------
    source_ids
        All source_ids to attempt (those absent from epoch_data are skipped).
    epoch_data
        Dict from download_epoch_astrometry().
    model
        gaiasupdate model string (see gaiasupdate_one).
    n_workers
        Parallel workers for the gaiasupdate solve (CPU-bound; 1 = serial).

    Returns
    -------
    DataFrame with one row per successfully solved source.  source_id is int64.
    """
    to_solve = [sid for sid in source_ids if sid in epoch_data]
    if verbose:
        print(f"  gaiasupdate re-solve: {len(to_solve)} sources with epoch data "
              f"(model={model}) ...")

    def _solve_one(sid):
        return gaiasupdate_one(
            epoch_data[sid], sid,
            model=model, compute_excess_noise=compute_excess_noise,
        )

    if n_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
            results = list(exe.map(_solve_one, to_solve))
    else:
        results = [_solve_one(sid) for sid in to_solve]

    rows = [r for r in results if r is not None]
    if verbose:
        print(f"  gaiasupdate: {len(rows)}/{len(to_solve)} converged")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["source_id"] = df["source_id"].astype(np.int64)
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def run_gaia_dr4_epoch(
    field_name: str,
    output_dir: str | Path,
    matched_source_ids: list[int],
    access: str = "datalink",
    prerelease_votable: str | Path | None = None,
    gaia_data_server: str = _GACS_DATA_SERVER,
    data_release: str = _DR4_DATA_RELEASE,
    credentials_file: str | None = None,
    model: str = "5p_single_source",
    compute_excess_noise: bool = False,
    n_download_workers: int = 4,
    n_solve_workers: int = 1,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download Gaia DR4 epoch astrometry and re-solve 5D parameters.

    This is the main pipeline entry point.  Call it after cross-matching to
    obtain improved (pmra, pmdec, parallax) priors for all sources in the HST
    footprint.

    Parameters
    ----------
    field_name
        Field name (e.g. 'Leo_I').
    output_dir
        Root output directory (same as for other BP3M steps).
    matched_source_ids
        List of Gaia source_ids (int64) identified in the HST footprint during
        cross-matching.  Obtain from the union of gaia_source_id columns across
        all ``matched_gaia.csv`` files.
    access
        'datalink' (default) or 'prerelease'.
    prerelease_votable
        Path to pre-release VOTable.  Required for access='prerelease'.
    model
        gaiasupdate astrometric model.
    force
        Re-download/re-solve even if cache exists.

    Returns
    -------
    DataFrame with epoch solutions (one row per source).  Saved to
        {output_dir}/Gaia/epoch/{field_name}_gaia_epoch_solutions.csv
    """
    output_dir = Path(output_dir)
    epoch_dir = output_dir / "Gaia" / "epoch"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    out_path = epoch_dir / f"{field_name}_gaia_epoch_solutions.csv"
    if out_path.exists() and not force:
        if verbose:
            print(f"  Epoch solutions: loading existing ({out_path.name})")
        df = pd.read_csv(out_path)
        df["source_id"] = df["source_id"].astype(np.int64)
        return df

    matched_source_ids = [np.int64(s) for s in matched_source_ids]
    if verbose:
        print(f"\n{'='*60}")
        print(f"Gaia DR4 epoch astrometry — {field_name}")
        print(f"{'='*60}")
        print(f"  Sources to process: {len(matched_source_ids)}")
        print(f"  Access method: {access}")

    # ── Step 1: Download epoch transits ──────────────────────────────────────
    epoch_data = download_epoch_astrometry(
        matched_source_ids,
        epoch_cache_dir=epoch_dir,
        access=access,
        prerelease_votable=prerelease_votable,
        gaia_data_server=gaia_data_server,
        data_release=data_release,
        credentials_file=credentials_file,
        n_workers=n_download_workers,
        force=force,
        verbose=verbose,
    )

    if not epoch_data:
        if verbose:
            print("  No epoch data available — skipping re-solve.")
        return pd.DataFrame()

    # ── Step 2: Re-solve 5D parameters with gaiasupdate ──────────────────────
    solutions = batch_epoch_solutions(
        matched_source_ids,
        epoch_data,
        model=model,
        compute_excess_noise=compute_excess_noise,
        n_workers=n_solve_workers,
        verbose=verbose,
    )

    if len(solutions) == 0:
        if verbose:
            print("  gaiasupdate re-solve produced no results.")
        return pd.DataFrame()

    solutions.to_csv(out_path, index=False)
    if verbose:
        print(f"  Saved → {out_path.name}")
        print(f"  Epoch solutions: {len(solutions)} sources")
        if "parallax_mas" in solutions.columns:
            print(f"  Median parallax: {solutions['parallax_mas'].median():.3f} mas")
        if "pmra_maspyr" in solutions.columns:
            print(f"  Median |pmra|:   {solutions['pmra_maspyr'].abs().median():.3f} mas/yr")

    return solutions


# ── Integration with BP3M data loader ────────────────────────────────────────

def merge_epoch_solutions_into_catalog(
    gaia_df: pd.DataFrame,
    epoch_solutions: pd.DataFrame,
    replace_pms: bool = True,
    replace_parallax: bool = True,
) -> pd.DataFrame:
    """Merge epoch-derived 5D solutions into the Gaia summary catalog DataFrame.

    Call this after run_gaia_dr4_epoch() and before passing gaia_df to the
    BP3M data loader / solver.

    Parameters
    ----------
    gaia_df
        Gaia summary catalog as loaded by download_gaia.download_gaia().
        Must have a 'source_id' column (int64).
    epoch_solutions
        DataFrame returned by run_gaia_dr4_epoch().
    replace_pms
        If True, overwrite pmra/pmdec/pmra_error/pmdec_error in gaia_df with
        the epoch-derived values (pmra_maspyr, pmdec_maspyr, ...).
    replace_parallax
        If True, overwrite parallax/parallax_error with epoch-derived values.

    Returns
    -------
    Updated gaia_df with new columns:
        pmra_ep, pmdec_ep, parallax_ep,
        pmra_error_ep, pmdec_error_ep, parallax_error_ep,
        pmra_pmdec_corr_ep, pmra_plx_corr_ep, pmdec_plx_corr_ep,
        n_transits_ep, n_transits_agis_ep, has_epoch_solution
    and optionally overwritten pmra/pmdec/parallax columns.
    """
    if len(epoch_solutions) == 0:
        gaia_df["has_epoch_solution"] = False
        return gaia_df

    gaia_df = gaia_df.copy()
    # Ensure both have int64 source_id for safe merge
    gaia_df["source_id"] = gaia_df["source_id"].astype(np.int64)
    ep = epoch_solutions.copy()
    ep["source_id"] = ep["source_id"].astype(np.int64)

    ep_cols = {
        "pmra_maspyr":        "pmra_ep",
        "pmdec_maspyr":       "pmdec_ep",
        "parallax_mas":       "parallax_ep",
        "pmra_error_maspyr":  "pmra_error_ep",
        "pmdec_error_maspyr": "pmdec_error_ep",
        "parallax_error_mas": "parallax_error_ep",
        "pmra_pmdec_corr":    "pmra_pmdec_corr_ep",
        "pmra_plx_corr":      "pmra_plx_corr_ep",
        "pmdec_plx_corr":     "pmdec_plx_corr_ep",
        "n_transits_used":    "n_transits_ep",
        "n_transits_agis":    "n_transits_agis_ep",
    }
    ep_rename = ep[["source_id"] + [c for c in ep_cols if c in ep.columns]].rename(
        columns=ep_cols
    )
    ep_rename["has_epoch_solution"] = True

    gaia_df = gaia_df.merge(ep_rename, on="source_id", how="left")
    gaia_df["has_epoch_solution"] = gaia_df["has_epoch_solution"].fillna(False)

    if replace_pms:
        mask = gaia_df["has_epoch_solution"] & gaia_df["pmra_ep"].notna()
        if "pmra_ep" in gaia_df.columns:
            gaia_df.loc[mask, "pmra"]       = gaia_df.loc[mask, "pmra_ep"]
            gaia_df.loc[mask, "pmra_error"] = gaia_df.loc[mask, "pmra_error_ep"]
        if "pmdec_ep" in gaia_df.columns:
            gaia_df.loc[mask, "pmdec"]       = gaia_df.loc[mask, "pmdec_ep"]
            gaia_df.loc[mask, "pmdec_error"] = gaia_df.loc[mask, "pmdec_error_ep"]
        if "pmra_pmdec_corr_ep" in gaia_df.columns:
            gaia_df.loc[mask, "pmra_pmdec_corr"] = gaia_df.loc[mask, "pmra_pmdec_corr_ep"]

    if replace_parallax:
        mask = gaia_df["has_epoch_solution"] & gaia_df["parallax_ep"].notna()
        if "parallax_ep" in gaia_df.columns:
            gaia_df.loc[mask, "parallax"]       = gaia_df.loc[mask, "parallax_ep"]
            gaia_df.loc[mask, "parallax_error"] = gaia_df.loc[mask, "parallax_error_ep"]

    return gaia_df


# ── Utility: collect matched source_ids from cross-match output ───────────────

def collect_matched_source_ids(
    output_dir: str | Path,
    bp3m_results_subdir: str = "BP3M_results",
) -> list[int]:
    """Collect all unique Gaia source_ids that were cross-matched in any image.

    Reads all ``matched_gaia.csv`` files under
    ``{output_dir}/HST/mastDownload/HST/*/``.

    Returns list of int64 source_ids.
    """
    output_dir = Path(output_dir)
    hst_dir = output_dir / "HST" / "mastDownload" / "HST"
    source_ids: set[int] = set()

    for matched_csv in hst_dir.glob("*/matched_gaia.csv"):
        try:
            df = pd.read_csv(matched_csv, usecols=["gaia_source_id"])
            ids = df["gaia_source_id"].dropna().astype(np.int64).values
            source_ids.update(ids.tolist())
        except Exception:
            continue

    return sorted(source_ids)


# ── Solver preprocessing ──────────────────────────────────────────────────────

def prepare_epoch_obs_for_solver(
    epoch_data: dict[int, pd.DataFrame],
    gaia_df: pd.DataFrame,
    ref_epoch_jyear: float = DR4_REF_EPOCH_JYEAR,
    use_agis_flag: bool = True,
    min_transits: int = 5,
    excess_noise_floor_mas: float = 0.0,
) -> dict[int, dict]:
    """Precompute per-source AL observation quantities for the BP3M normal equations.

    For each source with epoch data, evaluates the linearized AL observation
    equation from the user's model:

        ΔAL = (Δα* + μα*·Δt)·sin(θ) + (Δδ + μδ·Δt)·cos(θ) + ϖ·P_AL

    which equals  a_k · v_hat  where the design vector is:

        a_k = [sin(θ_k), cos(θ_k), Δt_k·sin(θ_k), Δt_k·cos(θ_k), P_AL_k]

    and v_hat = (Δα*, Δδ, μα*, μδ, ϖ) is BP3M's 5-parameter astrometric vector
    (position offsets in mas, PMs in mas/yr, parallax in mas).

    The "observed" AL for each transit is computed relative to the AGIS
    reference solution so that it is consistent with BP3M's v_hat convention
    (where Δα*=Δδ=0 corresponds to the Gaia reference position):

        y_k = centroid_pos_al_k
              − [μα*_AGIS·Δt_k·sin(θ_k) + μδ_AGIS·Δt_k·cos(θ_k) + ϖ_AGIS·P_AL_k]

    i.e.  y_k = centroid_pos_al_k − a_k[2:]·v_AGIS[2:]   (only PM+parallax terms,
    since v_AGIS[0]=v_AGIS[1]=0 in BP3M's convention).

    Per-transit weight: w_k = 1 / (σ_al_k² + (σ_excess + σ_floor)²).
    Transits with used_by_agis_al=False are optionally down-weighted by a
    factor of 0.1 to match the AGIS soft-rejection scheme.

    Parameters
    ----------
    epoch_data
        Dict source_id → epoch DataFrame from download_epoch_astrometry().
    gaia_df
        Gaia summary catalog (same as passed to the BP3M solver).  Must have
        columns: source_id (int64), pmra, pmdec, parallax.
    ref_epoch_jyear
        Reference epoch for Δt computation (default 2017.5 for DR4).
    use_agis_flag
        Down-weight (×0.1) transits where used_by_agis_al=False (default True).
    min_transits
        Minimum transits after flagging to include a source (default 5).
    excess_noise_floor_mas
        Additional noise floor added in quadrature to all transits (mas).

    Returns
    -------
    Dict mapping source_id (int64) → dict with keys:
        'H_contrib' : (5, 5) ndarray  —  A^T W A
        'h_contrib' : (5,)   ndarray  —  A^T W y
        'n_transits': int              —  number of transits used (weight > 0)
        'n_flagged' : int              —  transits down-weighted by AGIS flag
    Sources with insufficient transits or missing data are absent.
    """
    # Build source_id → (pmra_AGIS, pmdec_AGIS, plx_AGIS) lookup from gaia_df
    gaia_df = gaia_df.copy()
    gaia_df["source_id"] = gaia_df["source_id"].astype(np.int64)
    agis_lookup: dict[int, tuple] = {}
    for _, row in gaia_df.iterrows():
        sid = int(row["source_id"])
        agis_lookup[sid] = (
            float(row.get("pmra",    0.0) or 0.0),
            float(row.get("pmdec",   0.0) or 0.0),
            float(row.get("parallax", 0.0) or 0.0),
        )

    result: dict[int, dict] = {}

    for source_id, ep_df in epoch_data.items():
        sid = int(source_id)

        # ── Required columns ──────────────────────────────────────────────────
        missing = [c for c in _REQUIRED_EPOCH_COLS if c not in ep_df.columns]
        if missing:
            continue

        # ── Timing: obs_time_tcb (nanoseconds) → Julian year ─────────────────
        obs_tcb_ns = ep_df["obs_time_tcb"].to_numpy(dtype=np.float64)
        t_jyear    = _obs_time_to_jyear(obs_tcb_ns)
        dt         = t_jyear - ref_epoch_jyear          # Δt in years

        # ── Scan geometry ─────────────────────────────────────────────────────
        theta_rad = np.radians(ep_df["scan_pos_angle"].to_numpy(dtype=np.float64))
        sin_th    = np.sin(theta_rad)
        cos_th    = np.cos(theta_rad)
        p_al      = ep_df["parallax_factor_al"].to_numpy(dtype=np.float64)

        # ── Design matrix A: (n_transits, 5) ─────────────────────────────────
        # Columns: [sin(θ), cos(θ), Δt·sin(θ), Δt·cos(θ), P_AL]
        A = np.column_stack([sin_th, cos_th, dt * sin_th, dt * cos_th, p_al])

        # ── Observed AL: centroid_pos_al (assumed in mas) ─────────────────────
        centroid_al = ep_df["centroid_pos_al"].to_numpy(dtype=np.float64)

        # Subtract AGIS PM+parallax contribution so that y_k is consistent
        # with BP3M's convention (Δα*=0, Δδ=0 at the Gaia reference position):
        #   y_k = centroid_pos_al - (μα*_AGIS·Δt·sin(θ) + μδ_AGIS·Δt·cos(θ) + ϖ_AGIS·P_AL)
        pmra_agis, pmdec_agis, plx_agis = agis_lookup.get(sid, (0.0, 0.0, 0.0))
        agis_pm_plx_pred = (pmra_agis  * dt * sin_th
                            + pmdec_agis * dt * cos_th
                            + plx_agis   * p_al)
        y = centroid_al - agis_pm_plx_pred

        # ── Per-transit weights ───────────────────────────────────────────────
        sigma_al = ep_df["centroid_pos_error_al"].to_numpy(dtype=np.float64)
        # Per-source excess noise from AGIS (same for all transits of this source)
        excess_noise = float(ep_df["agis_source_excess_noise"].iloc[0]) \
                       if "agis_source_excess_noise" in ep_df.columns else 0.0
        sigma_eff_sq = (sigma_al**2
                        + (excess_noise + excess_noise_floor_mas)**2)
        sigma_eff_sq = np.maximum(sigma_eff_sq, 1e-6)   # numerical floor
        w = 1.0 / sigma_eff_sq

        # Optionally down-weight AGIS-rejected transits by factor 100 in precision
        n_flagged = 0
        if use_agis_flag and "used_by_agis_al" in ep_df.columns:
            agis_used = ep_df["used_by_agis_al"].to_numpy(dtype=bool)
            n_flagged = int((~agis_used).sum())
            w[~agis_used] *= 0.1

        # Check we have enough usable transits
        n_usable = int((w > 0).sum())
        if n_usable < min_transits:
            continue

        # ── Precompute normal-equation contributions ──────────────────────────
        # H_contrib = A^T diag(w) A    shape (5, 5)
        # h_contrib = A^T diag(w) y    shape (5,)
        AW       = A * w[:, None]          # (n, 5) scaled rows
        H_contrib = AW.T @ A               # (5, 5)
        h_contrib = AW.T @ y               # (5,)

        result[sid] = {
            "H_contrib":  H_contrib,
            "h_contrib":  h_contrib,
            "n_transits": n_usable,
            "n_flagged":  n_flagged,
        }

    return result

    return sorted(source_ids)
