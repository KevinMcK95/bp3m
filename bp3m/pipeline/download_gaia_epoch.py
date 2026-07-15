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

# Columns that must be present in a valid epoch astrometry DataFrame.
# These are the columns actually consumed by prepare_epoch_obs_for_solver().
# ra0/dec0 and colour_factor_al are present in the DR4 VOTable but not used
# by the solver; they are not required here so synthetic DataFrames (which
# omit them) pass the check.
_REQUIRED_EPOCH_COLS = [
    "obs_time_tcb",
    "centroid_pos_al",
    "centroid_pos_error_al",
    "scan_pos_angle",
    "parallax_factor_al",
    "used_by_agis_al",
    "agis_source_excess_noise",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs_time_to_jyear(obs_time_tcb: np.ndarray) -> np.ndarray:
    """Convert obs_time_tcb (nanoseconds from 2010-01-01 TCB) to Julian year."""
    return 2010.0 + np.asarray(obs_time_tcb, dtype=np.float64) / _NS_PER_JYEAR


def _explode_epoch_transits(df: pd.DataFrame) -> pd.DataFrame:
    """Explode per-transit epoch DataFrame into per-CCD rows.

    The DR4 VOTable delivers one row per FoV transit with array-valued columns
    (e.g. obs_time_tcb[10], scan_pos_angle[10], centroid_pos_al[10]) holding
    one entry per AF/SM CCD in the focal-plane strip.  Scalar columns such as
    parallax_factor_al and agis_source_excess_noise are the same for all CCDs
    in a given transit.

    Returns a flat DataFrame with one row per (transit × CCD) combination.
    Masked/NaN CCD entries are dropped.  The extra column ``ccd_index`` (0–9)
    records which CCD the row came from.
    """
    # Identify which columns hold arrays vs scalars in the first non-null row
    first_vals = {col: next(
        (v for v in df[col] if v is not None), None
    ) for col in df.columns}
    array_cols = [
        col for col, v in first_vals.items()
        if v is not None and hasattr(v, "__len__") and not isinstance(v, str)
    ]
    scalar_cols = [col for col in df.columns if col not in array_cols]

    # Columns that must be valid for a CCD observation to be included
    _CORE_ARRAY_COLS = {
        "obs_time_tcb", "scan_pos_angle",
        "centroid_pos_al", "centroid_pos_error_al",
    }

    rows = []
    for _, transit in df.iterrows():
        # Determine CCD count from first array column
        n_ccd = None
        for col in array_cols:
            v = transit[col]
            if v is not None and hasattr(v, "__len__"):
                n_ccd = len(v)
                break
        if n_ccd is None:
            continue

        scalar_vals = {col: transit[col] for col in scalar_cols}
        for ccd_idx in range(n_ccd):
            row = dict(scalar_vals)
            row["ccd_index"] = ccd_idx
            skip = False
            for col in array_cols:
                v = transit[col]
                if v is None or not hasattr(v, "__len__") or len(v) <= ccd_idx:
                    row[col] = np.nan
                    continue
                cell = v[ccd_idx]
                is_masked = hasattr(cell, "mask") and np.any(cell.mask)
                if is_masked:
                    if col in _CORE_ARRAY_COLS:
                        skip = True
                        break
                    row[col] = np.nan
                else:
                    row[col] = float(cell) if np.isscalar(cell) else cell
            if not skip:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["ccd_index"])
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_prerelease_votable(votable_path: str | Path) -> pd.DataFrame:
    """Load ESA's pre-release epoch astrometry VOTable into a DataFrame.

    The pre-release ZIP from
      https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/
      gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip
    contains a single VOTable XML file with 12 illustrative sources (one row per
    FoV transit with per-CCD array columns).

    Returns a flat per-CCD DataFrame with source_id as int64.
    """
    from astropy.table import Table
    tbl = Table.read(str(votable_path), format="votable")
    df = tbl.to_pandas()
    # Ensure source_id is int64 (never float — see gaia_ids memory note)
    df["source_id"] = df["source_id"].astype(np.int64)
    # Explode transit-level array columns into per-CCD rows
    df = _explode_epoch_transits(df)
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

    The "observed" AL for each CCD transit is computed relative to the AGIS
    reference solution so that it is consistent with BP3M's v_hat convention
    (where Δα*=Δδ=0 corresponds to the Gaia reference position):

        y_k = centroid_pos_al_k
              + [μα*_AGIS·Δt_k·sin(θ_k) + μδ_AGIS·Δt_k·cos(θ_k) + ϖ_AGIS·P_AL_k]

    i.e.  y_k = centroid_pos_al_k + a_k[2:]·v_AGIS[2:]   (only PM+parallax terms,
    since v_AGIS[0]=v_AGIS[1]=0 in BP3M's convention and centroid_pos_al is
    the AGIS residual: a_k·(v_true − v_AGIS)).

    Each epoch DataFrame must be in per-CCD-row format (one row per CCD
    observation, ~10 rows per FoV transit).  _load_prerelease_votable()
    already returns this format via _explode_epoch_transits().

    Per-CCD weight: w_k = 1 / (σ_al_k² + (σ_excess + σ_floor)²).
    CCDs with used_by_agis_al=False are optionally down-weighted by a
    factor of 0.1 to match the AGIS soft-rejection scheme.

    Parameters
    ----------
    epoch_data
        Dict source_id → per-CCD epoch DataFrame from download_epoch_astrometry().
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
    # Build source_id → (pmra_AGIS, pmdec_AGIS, plx_AGIS) lookup from gaia_df.
    # Use vectorized access — iterrows() on a purely-numeric DataFrame upcasts int64
    # source_ids to float64, silently corrupting IDs > 2^53.
    gaia_df = gaia_df.copy()
    gaia_df["source_id"] = gaia_df["source_id"].astype(np.int64)
    _agis_sids  = gaia_df["source_id"].values                     # int64 array
    _agis_pmra  = np.where(gaia_df["pmra"].notna(),   gaia_df["pmra"].values,   0.0) \
                  if "pmra"     in gaia_df.columns else np.zeros(len(gaia_df))
    _agis_pmdec = np.where(gaia_df["pmdec"].notna(),  gaia_df["pmdec"].values,  0.0) \
                  if "pmdec"    in gaia_df.columns else np.zeros(len(gaia_df))
    _agis_plx   = np.where(gaia_df["parallax"].notna(), gaia_df["parallax"].values, 0.0) \
                  if "parallax" in gaia_df.columns else np.zeros(len(gaia_df))
    agis_lookup: dict[int, tuple] = {
        int(sid): (float(pmra), float(pmdec), float(plx))
        for sid, pmra, pmdec, plx in zip(_agis_sids, _agis_pmra, _agis_pmdec, _agis_plx)
    }

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

        # centroid_pos_al is the AGIS residual measured relative to the
        # per-transit propagated reference position (ra0_k, dec0_k), i.e.:
        #   centroid_pos_al_k = a_k · (v_true − v_AGIS)
        # To get y_k = a_k · v_true (what the normal equations need):
        #   y_k = centroid_pos_al_k + a_k · v_AGIS
        # where v_AGIS = (0, 0, μα*_AGIS, μδ_AGIS, ϖ_AGIS) in BP3M coordinates.
        pmra_agis, pmdec_agis, plx_agis = agis_lookup.get(sid, (0.0, 0.0, 0.0))
        agis_pm_plx_pred = (pmra_agis  * dt * sin_th
                            + pmdec_agis * dt * cos_th
                            + plx_agis   * p_al)
        y = centroid_al + agis_pm_plx_pred

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


# ── Synthetic epoch data generation ──────────────────────────────────────────

def _sigma_al_from_gmag(gmag: float) -> float:
    """Approximate per-CCD AL centroid uncertainty (mas) as a function of G mag.

    Based on the Gaia DR4 per-CCD noise model (Fig. 3 of Lindegren+2021 and
    ESA pre-release documentation).  The floor at bright magnitudes is set by
    attitude noise and calibration residuals; the faint end is photon-noise limited.
    """
    # Rough piecewise log-linear model calibrated to ~0.12 mas at G=16, ~0.06 at G<13
    if gmag <= 13.0:
        return 0.06
    log_sigma = np.interp(
        gmag,
        [13.0, 16.0, 18.0, 19.5, 21.0],
        [np.log10(0.06), np.log10(0.12), np.log10(0.4), np.log10(1.0), np.log10(3.0)],
    )
    return float(10.0 ** log_sigma)


def generate_synthetic_epoch_data(
    source_df: pd.DataFrame,
    n_transits_per_source: int = 80,
    n_ccd_per_transit: int = 9,
    ref_epoch_jyear: float = DR4_REF_EPOCH_JYEAR,
    mission_start_jyear: float = 2014.5,
    mission_end_jyear: float = 2019.5,
    rejection_fraction: float = 0.05,
    seed: int = 42,
) -> dict[int, pd.DataFrame]:
    """Generate synthetic Gaia DR4 epoch astrometry for end-to-end pipeline tests.

    Creates per-CCD AL observations consistent with the model expected by
    prepare_epoch_obs_for_solver().  The ``centroid_pos_al`` values start as pure
    Gaussian noise drawn at the formal CCD precision σ_AL (from G magnitude).
    No AGIS excess noise is added; ``agis_source_excess_noise`` is set to 0.

    To build a self-consistent synthetic catalog, call
    ``compute_epoch_catalog_solutions`` after injecting truth signal.  That
    function solves the 5-parameter AGIS normal equations from the epoch data and
    updates both the epoch DataFrames and the Gaia catalog DataFrame so that the
    catalog values are EXACTLY what the epoch solve recovers.  This guarantees that
    replacing the Gaia summary prior with raw epoch observations produces identical
    results.

    Parameters
    ----------
    source_df
        Per-source parameters.  Required columns: ``source_id`` (int64),
        ``ra``, ``dec``, ``pmra``, ``pmdec``, ``parallax``.  Optional:
        ``phot_g_mean_mag`` (default 18.0 if absent).
    n_transits_per_source
        Number of FoV transits per source (default 80, ~typical for DR4).
    n_ccd_per_transit
        Number of AF CCDs per transit (1-9, default 9 = AF1-9).
    ref_epoch_jyear
        Reference epoch for Δt in the observation model (default 2017.5).
    mission_start_jyear, mission_end_jyear
        Observation window for random transit time draw (default 2014.5–2019.5).
    rejection_fraction
        Fraction of CCD observations to mark as used_by_agis_al=False (default 5%).
    seed
        RNG seed for reproducibility (default 42).

    Returns
    -------
    Dict source_id (int64) → per-CCD epoch DataFrame, compatible with
    ``prepare_epoch_obs_for_solver()`` and ``_save_epoch_cache()``.
    centroid_pos_al = pure noise at σ_AL; call compute_epoch_catalog_solutions
    after truth injection to get exact catalog consistency.
    """
    from bp3m.astro_utils import get_tele_position, get_parallax_factors
    from astropy.time import Time

    rng = np.random.default_rng(seed)
    result: dict[int, pd.DataFrame] = {}

    source_df = source_df.copy()
    source_df["source_id"] = source_df["source_id"].astype(np.int64)
    if "phot_g_mean_mag" not in source_df.columns:
        source_df["phot_g_mean_mag"] = 18.0

    # iterrows() on a purely-numeric DataFrame upcasts int64 source_ids to float64,
    # silently corrupting IDs > 2^53.  Use itertuples() which preserves column dtypes.
    for src in source_df.itertuples(index=False):
        sid  = int(src.source_id)           # np.int64 → Python int, no float roundtrip
        ra   = float(src.ra)
        dec  = float(src.dec)
        gmag = float(src.phot_g_mean_mag)

        sigma_al = _sigma_al_from_gmag(gmag)

        # 2p stars (no Gaia PM measurement) get 2–5 transits to reflect that Gaia
        # could not solve the full 5-parameter system from their limited detections.
        # 5p stars get the full n_transits_per_source.
        pmra_val = getattr(src, 'pmra', float('nan'))
        is_2p = not np.isfinite(float(pmra_val)) if pmra_val is not None else True
        n_transit = int(rng.integers(2, 6)) if is_2p else n_transits_per_source

        # ── Generate random transit times over the mission window ─────────────
        t_jyear = rng.uniform(mission_start_jyear, mission_end_jyear,
                              size=n_transit)
        t_jyear = np.sort(t_jyear)

        # ── Scan angles: approximate Gaia scanning law ────────────────────────
        # Gaia's spin period is ~6 h; over the mission each star is observed at
        # a wide variety of scan angles.  Draw uniform on [0°, 360°).
        theta_deg = rng.uniform(0.0, 360.0, size=n_transit)

        # ── Parallax factors: use Earth's position as Gaia L2 approximation ──
        p_al_arr = np.empty(n_transit)
        for k, t_yr in enumerate(t_jyear):
            t_astropy = Time(t_yr, format="jyear")
            try:
                xyz = get_tele_position(t_astropy, curr_id="earth")
                pf_ra, pf_dec = get_parallax_factors(
                    np.array([ra]), np.array([dec]), xyz
                )
                theta_rad_k = np.radians(theta_deg[k])
                p_al_arr[k] = float(pf_ra[0] * np.sin(theta_rad_k)
                                    + pf_dec[0] * np.cos(theta_rad_k))
            except Exception:
                lam_sun = np.radians(360.0 * (t_yr - 2015.0) % 360.0)
                cos_dec = np.cos(np.radians(dec))
                p_al_arr[k] = float(
                    (-np.sin(lam_sun) * np.sin(theta_deg[k] * np.pi / 180) * cos_dec
                     + np.cos(lam_sun) * np.cos(theta_deg[k] * np.pi / 180) * cos_dec)
                )

        # ── Expand each transit to n_ccd_per_transit CCD rows ─────────────────
        ccd_time_offset_jyr = np.arange(n_ccd_per_transit) * (10.0 / (365.25 * 86400))
        ccd_angle_offset_deg = np.linspace(-0.005, 0.005, n_ccd_per_transit)

        rows = []
        for k in range(n_transit):
            for ccd_idx in range(n_ccd_per_transit):
                t_ccd  = t_jyear[k] + ccd_time_offset_jyr[ccd_idx]
                theta_ccd = theta_deg[k] + ccd_angle_offset_deg[ccd_idx]

                # centroid_pos_al = pure noise at σ_AL.
                # Signal is injected externally (truth injection step).
                # After truth injection, call compute_epoch_catalog_solutions to
                # derive exact catalog values and update centroid_pos_al to the
                # AGIS residual relative to the epoch-solved reference.
                noise = rng.normal(0.0, sigma_al)
                obs_ns = int((t_ccd - 2010.0) * _NS_PER_JYEAR)
                is_rejected = rng.random() < rejection_fraction
                rows.append({
                    "source_id":                 sid,
                    "transit_id":                np.int64(k * 1000 + ccd_idx),
                    "ra0":                       ra,
                    "dec0":                      dec,
                    "obs_time_tcb":              obs_ns,
                    "scan_pos_angle":            theta_ccd,
                    "parallax_factor_al":        float(p_al_arr[k]),
                    "centroid_pos_al":           noise,
                    "centroid_pos_error_al":     sigma_al,
                    "agis_source_excess_noise":  0.0,
                    "used_by_agis_al":           not is_rejected,
                    "ccd_index":                 ccd_idx,
                })

        df = pd.DataFrame(rows)
        df["source_id"] = df["source_id"].astype(np.int64)
        result[sid] = df

    return result


def compute_epoch_catalog_solutions(
    epoch_data: dict[int, pd.DataFrame],
    gaia_df: pd.DataFrame,
    ref_epoch_jyear: float = DR4_REF_EPOCH_JYEAR,
    use_agis_flag: bool = True,
    min_transits: int = 5,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Derive exact Gaia catalog values from epoch data and update both.

    Call this AFTER injecting truth signal into the epoch DataFrames.  For each
    star, the function:

    1. Reconstructs the absolute AL measurements:
           y_k = centroid_pos_al + a_k[2:] · v_AGIS[2:]
       where v_AGIS = (pmra, pmdec, parallax) from the original gaia_df.
    2. Solves the 5-parameter normal equations:
           H = A^T diag(1/σ_AL²) A
           v_hat = H^{-1} A^T W y
           C = H^{-1}
    3. Updates gaia_df with (v_hat, C) — these ARE the "Gaia DR4 catalog
       solution" for this star.
    4. Updates centroid_pos_al in the epoch DataFrame to be the AGIS residual
       relative to v_hat:
           centroid_pos_al_new = y_k - a_k[2:] · v_hat[2:]
       so that prepare_epoch_obs_for_solver (using updated gaia_df) reconstructs
       y_k exactly, guaranteeing that the epoch H/h and the Gaia prior are
       IDENTICALLY the same information.

    Parameters
    ----------
    epoch_data
        Dict source_id → per-CCD DataFrame, after truth injection.
    gaia_df
        Gaia summary catalog (original, before update).  Must have source_id (int64),
        pmra, pmdec, parallax columns.
    ref_epoch_jyear
        Reference epoch (default 2017.5 for DR4).
    use_agis_flag
        Only use transits with used_by_agis_al=True for the solve (default True).
    min_transits
        Minimum active transits required to solve (default 5).

    Returns
    -------
    (updated_epoch_data, updated_gaia_df)
        updated_epoch_data : dict with centroid_pos_al rebaselined to v_hat
        updated_gaia_df    : gaia_df with pmra/pmdec/parallax/errors/correlations
                             replaced by the epoch-derived values for 5p stars,
                             and ra_error/dec_error updated for 2p stars.
    """
    gaia_df = gaia_df.copy()
    gaia_df["source_id"] = gaia_df["source_id"].astype(np.int64)

    # Build lookup of original AGIS catalog values (pmra, pmdec, parallax).
    # Use vectorized access — never iterrows() on numeric DataFrames.
    _sids = gaia_df["source_id"].values
    _pmra  = np.where(gaia_df["pmra"].notna(),     gaia_df["pmra"].values,     0.0) \
             if "pmra"     in gaia_df.columns else np.zeros(len(gaia_df))
    _pmdec = np.where(gaia_df["pmdec"].notna(),    gaia_df["pmdec"].values,    0.0) \
             if "pmdec"    in gaia_df.columns else np.zeros(len(gaia_df))
    _plx   = np.where(gaia_df["parallax"].notna(), gaia_df["parallax"].values, 0.0) \
             if "parallax" in gaia_df.columns else np.zeros(len(gaia_df))
    agis_orig: dict[int, tuple[float, float, float]] = {
        int(s): (float(pm), float(pmd), float(p))
        for s, pm, pmd, p in zip(_sids, _pmra, _pmdec, _plx)
    }

    # Also track which stars are 5p (have measured pmra)
    _has_pmra = gaia_df["pmra"].notna().values if "pmra" in gaia_df.columns \
                else np.zeros(len(gaia_df), dtype=bool)
    is_5p_lookup: dict[int, bool] = {
        int(s): bool(f) for s, f in zip(_sids, _has_pmra)
    }

    # Index gaia_df by source_id (int64) for fast scalar update
    gaia_df = gaia_df.set_index("source_id")

    updated_epoch_data: dict[int, pd.DataFrame] = {}
    n_solved = 0

    for source_id, ep_df in epoch_data.items():
        sid = int(source_id)
        ep_df = ep_df.copy()

        if not all(c in ep_df.columns for c in _REQUIRED_EPOCH_COLS):
            updated_epoch_data[sid] = ep_df
            continue

        # Select AGIS-accepted observations for the solve
        if use_agis_flag and "used_by_agis_al" in ep_df.columns:
            mask_act = ep_df["used_by_agis_al"].values.astype(bool)
        else:
            mask_act = np.ones(len(ep_df), dtype=bool)

        if mask_act.sum() < min_transits:
            updated_epoch_data[sid] = ep_df
            continue

        ep_act = ep_df[mask_act]

        # Timing and geometry for active observations
        obs_tcb = ep_act["obs_time_tcb"].to_numpy(dtype=np.float64)
        t_yr    = _obs_time_to_jyear(obs_tcb)
        dt      = t_yr - ref_epoch_jyear
        theta   = np.radians(ep_act["scan_pos_angle"].to_numpy(dtype=np.float64))
        sin_th  = np.sin(theta)
        cos_th  = np.cos(theta)
        p_al    = ep_act["parallax_factor_al"].to_numpy(dtype=np.float64)

        # Design matrix: [sin θ, cos θ, Δt·sin θ, Δt·cos θ, P_AL]
        A = np.column_stack([sin_th, cos_th, dt * sin_th, dt * cos_th, p_al])

        # Weights: formal CCD noise only (agis_source_excess_noise = 0 for synthetic)
        sigma_al = ep_act["centroid_pos_error_al"].to_numpy(dtype=np.float64)
        sigma_exc = float(ep_act["agis_source_excess_noise"].iloc[0]) \
                    if "agis_source_excess_noise" in ep_act.columns else 0.0
        w = 1.0 / np.maximum(sigma_al**2 + sigma_exc**2, 1e-12)

        # Reconstruct absolute measurements: y = centroid_pos_al + AGIS PM+plx prediction
        pmra_agis, pmdec_agis, plx_agis = agis_orig.get(sid, (0.0, 0.0, 0.0))
        centroid_act = ep_act["centroid_pos_al"].to_numpy(dtype=np.float64)
        agis_corr = pmra_agis * dt * sin_th + pmdec_agis * dt * cos_th + plx_agis * p_al
        y = centroid_act + agis_corr

        # Normal equations: H = A^T W A, h = A^T W y
        AW = A * w[:, None]
        H  = AW.T @ A   # (5, 5)
        h  = AW.T @ y   # (5,)

        try:
            C     = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            updated_epoch_data[sid] = ep_df
            continue

        v_hat = C @ h   # [Δα*, Δδ, μα*, μδ, ϖ]

        # ── Update gaia_df ────────────────────────────────────────────────────
        if sid not in gaia_df.index:
            updated_epoch_data[sid] = ep_df
            continue

        # Helper for safe sqrt
        def _safe_sqrt(x: float) -> float:
            return float(np.sqrt(max(x, 0.0)))

        def _safe_corr(cov_ij: float, var_i: float, var_j: float) -> float:
            d = float(np.sqrt(max(var_i, 1e-40) * max(var_j, 1e-40)))
            return float(np.clip(cov_ij / d, -1.0, 1.0))

        is_5p = is_5p_lookup.get(sid, False)

        if is_5p:
            # ── 5p stars: full 5-parameter epoch solution ──────────────────────
            # H is well-conditioned with ~80 transits; no prior needed.
            # The epoch catalog is a sufficient statistic for the epoch obs:
            #   h_prior = C_inv @ v_survey ≡ h_epoch  when v_survey = v_hat (all 5)
            #             and C = H_epoch^{-1}
            #
            # Centroid rebaseline: remove old AGIS PM+plx prediction and subtract
            # the epoch-derived prediction so prepare_epoch_obs_for_solver recovers
            # y_orig = centroid_new + new_agis (using updated catalog pmra/pmdec/plx).
            obs_tcb_all = ep_df["obs_time_tcb"].to_numpy(dtype=np.float64)
            t_yr_all    = _obs_time_to_jyear(obs_tcb_all)
            dt_all      = t_yr_all - ref_epoch_jyear
            sin_all     = np.sin(np.radians(ep_df["scan_pos_angle"].to_numpy(dtype=np.float64)))
            cos_all     = np.cos(np.radians(ep_df["scan_pos_angle"].to_numpy(dtype=np.float64)))
            p_al_all    = ep_df["parallax_factor_al"].to_numpy(dtype=np.float64)

            centroid_all  = ep_df["centroid_pos_al"].to_numpy(dtype=np.float64)
            agis_corr_all = (pmra_agis * dt_all * sin_all
                             + pmdec_agis * dt_all * cos_all
                             + plx_agis * p_al_all)
            y_all         = centroid_all + agis_corr_all
            new_agis      = (v_hat[2] * dt_all * sin_all
                             + v_hat[3] * dt_all * cos_all
                             + v_hat[4] * p_al_all)
            ep_df["centroid_pos_al"] = y_all - new_agis

            # Update ra/dec (moves linearization point to epoch-derived position).
            if "ra" in gaia_df.columns and "dec" in gaia_df.columns:
                dec_rad = np.radians(float(gaia_df.loc[sid, "dec"]))
                gaia_df.loc[sid, "ra"]  = float(gaia_df.loc[sid, "ra"])  \
                                          + float(v_hat[0]) / (np.cos(dec_rad) * 3.6e6)
                gaia_df.loc[sid, "dec"] = float(gaia_df.loc[sid, "dec"]) \
                                          + float(v_hat[1]) / 3.6e6

            gaia_df.loc[sid, "pmra"]     = float(v_hat[2])
            gaia_df.loc[sid, "pmdec"]    = float(v_hat[3])
            gaia_df.loc[sid, "parallax"] = float(v_hat[4])

            for col, idx in [("pmra_error",     2), ("pmdec_error",     3),
                              ("parallax_error", 4), ("ra_error",        0),
                              ("dec_error",      1)]:
                if col in gaia_df.columns:
                    gaia_df.loc[sid, col] = _safe_sqrt(C[idx, idx])

            corr_pairs = [
                ("pmra_pmdec_corr",      2, 3),
                ("parallax_pmra_corr",   4, 2),
                ("parallax_pmdec_corr",  4, 3),
                ("ra_dec_corr",          0, 1),
                ("ra_parallax_corr",     0, 4),
                ("ra_pmra_corr",         0, 2),
                ("ra_pmdec_corr",        0, 3),
                ("dec_parallax_corr",    1, 4),
                ("dec_pmra_corr",        1, 2),
                ("dec_pmdec_corr",       1, 3),
            ]
            for col, i, j in corr_pairs:
                if col in gaia_df.columns:
                    gaia_df.loc[sid, col] = _safe_corr(C[i, j], C[i, i], C[j, j])

        else:
            # ── 2p stars: position-only update with diffuse PM/parallax prior ───
            # 2p stars have 2–5 transits (insufficient for a 5p solve without a
            # prior).  Apply the same diffuse PM and parallax priors used by the
            # main BP3M solver to regularize, then update only ra/dec and their
            # 2×2 position covariance.  PM/parallax stay NaN (driven by prior, not
            # data).
            #
            # Centroid rebaseline is NOT done for 2p stars: since pmra=NaN → AGIS
            # reference = 0 in prepare_epoch_obs_for_solver, the raw centroid
            # already encodes y = centroid + 0 = y_orig.  Rebaselining with the
            # prior-regularized v_hat would corrupt h in the joint solve.
            from bp3m.astro_utils import michalik_sigma_plx_prior
            _SIGMA_PM_2P = 100.0  # mas/yr — matches solver._SIGMA_PM
            ra_2p  = float(gaia_df.loc[sid, "ra"])
            dec_2p = float(gaia_df.loc[sid, "dec"])
            gmag_2p = float(gaia_df.loc[sid, "phot_g_mean_mag"]) \
                      if "phot_g_mean_mag" in gaia_df.columns else 18.0
            sigma_plx_2p = michalik_sigma_plx_prior(ra_2p, dec_2p, gmag_2p)

            H_prior_2p = np.diag([0.0, 0.0,
                                  _SIGMA_PM_2P**-2, _SIGMA_PM_2P**-2,
                                  float(sigma_plx_2p)**-2])
            try:
                C_2p = np.linalg.inv(H + H_prior_2p)
            except np.linalg.LinAlgError:
                updated_epoch_data[sid] = ep_df
                continue
            v_hat_2p = C_2p @ h

            # Update ra/dec only.
            if "ra" in gaia_df.columns and "dec" in gaia_df.columns:
                dec_rad = np.radians(dec_2p)
                gaia_df.loc[sid, "ra"]  = ra_2p \
                                          + float(v_hat_2p[0]) / (np.cos(dec_rad) * 3.6e6)
                gaia_df.loc[sid, "dec"] = dec_2p \
                                          + float(v_hat_2p[1]) / 3.6e6

            # Update position errors and correlation from the 2×2 subblock.
            if "ra_error" in gaia_df.columns:
                gaia_df.loc[sid, "ra_error"]  = _safe_sqrt(C_2p[0, 0])
            if "dec_error" in gaia_df.columns:
                gaia_df.loc[sid, "dec_error"] = _safe_sqrt(C_2p[1, 1])
            if "ra_dec_corr" in gaia_df.columns:
                gaia_df.loc[sid, "ra_dec_corr"] = _safe_corr(C_2p[0, 1], C_2p[0, 0], C_2p[1, 1])

            # pmra/pmdec/parallax remain NaN — not written to gaia_df.

        updated_epoch_data[sid] = ep_df
        n_solved += 1

    gaia_df = gaia_df.reset_index()
    print(f"  compute_epoch_catalog_solutions: solved {n_solved}/{len(epoch_data)} stars")
    return updated_epoch_data, gaia_df
