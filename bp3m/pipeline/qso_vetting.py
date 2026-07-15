"""
QSO anchor vetting: cross-match Gaia qso_candidates against Quaia and MILLIQUAS,
then apply a 3-sigma astrometric consistency cut using scaled Gaia covariance.

Pipeline
--------
1. Load per-field qso_candidates CSV (from download_gaia_qso_candidates).
2. Load main Gaia CSV filtered to in_qso_candidates=True for astrometry.
3. Quaia cross-match by int64 source_id.
4. MILLIQUAS cross-match by RA/Dec (SkyCoord KD-tree, default radius 1.5 arcsec).
5. 3-sigma Mahalanobis cut on (pmra, pmdec, parallax) vs secular aberration
   prediction + zero parallax, using BP3M-scaled Gaia covariance.
6. Save {field_dir}/Gaia/{field}_qso_anchors.csv and print step-by-step counts.

Output CSV columns (superset of qso_candidates + new flags):
    quaia_match          bool   — matched in Quaia by source_id
    quaia_z              float  — Quaia photometric redshift (NaN if no match)
    milliquas_match      bool   — matched in MILLIQUAS by position
    milliquas_sep_arcsec float  — separation to nearest MILLIQUAS source (arcsec)
    milliquas_z          float  — MILLIQUAS redshift (NaN if no match)
    milliquas_type       str    — MILLIQUAS TYPE code (e.g. 'Q')
    milliquas_gaia_pos   bool   — MILLIQUAS position from Gaia EDR3 ('G' in COMMENT)
    catalog_match        bool   — quaia OR milliquas OR gaia_crf_source
    has_5p_solution      bool   — Gaia 5/6p solution (pmra/pmdec/parallax finite)
    chi2_astrometric     float  — Mahalanobis distance sqrt(chi2) vs aberration+0-plx
    astrometric_pass     bool   — chi2_astrometric < sigma_cut (or False if no 5p)
    is_qso_anchor        bool   — catalog_match AND astrometric_pass

Note on MILLIQUAS-only sources (too faint for Gaia):
    MILLIQUAS contains QSOs with no Gaia source_id.  These may be visible in
    HST images as unmatched detections.  They are NOT handled here (this module
    only vets Gaia-detected sources); future work should match MILLIQUAS directly
    to HST source catalogs for additional Gaia-free anchors.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..astro_utils import GAIA_SYS_DICT
from .secular_aberration import secular_aberration_pm

# Astrometric solution type flags (mirrors solver.py logic)
_MULT = {'5p': GAIA_SYS_DICT['mult_5p'], '6p': GAIA_SYS_DICT['mult_6p']}
_PM_SYS   = GAIA_SYS_DICT['pm_sys_err']    # mas/yr
_PLX_SYS  = GAIA_SYS_DICT['parallax_sys_err']  # mas


# ── Public entry point ────────────────────────────────────────────────────────

def vet_qso_candidates(
    field_name: str,
    output_dir: "Path | str",
    lib_dir: "Path | None" = None,
    search_radius_arcsec: float = 1.5,
    sigma_cut: float = 3.0,
    force_rerun: bool = False,
) -> "pd.DataFrame | None":
    """Vet Gaia qso_candidates for use as astrometric anchors.

    Reads qso_candidates CSV and the main Gaia CSV from the field's Gaia/
    directory.  Requires Quaia and/or MILLIQUAS in lib_dir/qso_catalogs/;
    sources with no external catalog match are excluded.

    Parameters
    ----------
    field_name, output_dir
        Same as used for download_gaia / run_cross_match.
    lib_dir
        BP3M library directory (from bp3m-setup).  If None or catalogs are
        missing, catalog matching is skipped (only gaia_crf_source passes).
    search_radius_arcsec
        MILLIQUAS positional match radius.  Sources flagged as having Gaia
        EDR3 positions ('G' in MILLIQUAS COMMENT) use half this radius.
    sigma_cut
        Mahalanobis distance threshold for (pmra, pmdec, parallax) vs the
        secular aberration + zero-parallax expectation.
    force_rerun
        Re-run even if qso_anchors.csv already exists.

    Returns
    -------
    DataFrame or None
        The vetted QSO anchor catalog with is_qso_anchor column, or None if
        qso_candidates CSV is missing.
    """
    output_dir = Path(output_dir)
    gaia_dir   = output_dir / field_name / "Gaia"

    out_path   = gaia_dir / f"{field_name}_qso_anchors.csv"
    if not force_rerun and out_path.exists():
        print(f"[QSO vetting] Loading cached: {out_path.name}")
        return pd.read_csv(out_path, dtype={'source_id': 'int64'})

    # ── Load qso_candidates CSV ───────────────────────────────────────────────
    qso_paths = sorted(gaia_dir.glob("*_qso_candidates.csv"))
    if not qso_paths:
        print("[QSO vetting] No qso_candidates CSV found — skipping.")
        return None
    qso_df = pd.read_csv(qso_paths[0], dtype={'source_id': 'int64'})
    print(f"\n[QSO vetting] Starting with {len(qso_df)} Gaia qso_candidates")

    # ── Load main Gaia CSV for astrometry ─────────────────────────────────────
    gaia_astro = _load_gaia_astrometry(gaia_dir)

    # ── Quaia cross-match ─────────────────────────────────────────────────────
    quaia_path = _find_catalog(lib_dir, 'quaia_G20.5.fits')
    if quaia_path is not None:
        qso_df = _match_quaia(qso_df, quaia_path)
    else:
        print("  Quaia catalog not found — skipping Quaia cross-match")
        qso_df['quaia_match'] = False
        qso_df['quaia_z']     = np.nan

    n_quaia = qso_df['quaia_match'].sum()
    print(f"  After Quaia match (by source_id):       {n_quaia} / {len(qso_df)}")

    # ── MILLIQUAS cross-match ─────────────────────────────────────────────────
    mq_path = _find_catalog(lib_dir, 'milliquas.fits')
    if mq_path is not None:
        qso_df = _match_milliquas(qso_df, mq_path, search_radius_arcsec)
    else:
        print("  MILLIQUAS catalog not found — skipping MILLIQUAS cross-match")
        qso_df['milliquas_match']      = False
        qso_df['milliquas_sep_arcsec'] = np.nan
        qso_df['milliquas_z']          = np.nan
        qso_df['milliquas_type']       = ''
        qso_df['milliquas_gaia_pos']   = False

    n_mq = qso_df['milliquas_match'].sum()
    print(f"  After MILLIQUAS match (RA/Dec ≤{search_radius_arcsec:.1f}\"): "
          f"{n_mq} / {len(qso_df)}")

    # ── Catalog match flag ────────────────────────────────────────────────────
    crf = qso_df.get('gaia_crf_source', pd.Series(False, index=qso_df.index))
    qso_df['catalog_match'] = (
        qso_df['quaia_match'] | qso_df['milliquas_match'] | crf.fillna(False)
    )
    n_cat = qso_df['catalog_match'].sum()
    print(f"  Catalog match (Quaia OR MILLIQUAS OR CRF3): {n_cat} / {len(qso_df)}")

    # ── Astrometric consistency cut ───────────────────────────────────────────
    qso_df = _astrometric_cut(qso_df, gaia_astro, sigma_cut)
    n_5p   = qso_df['has_5p_solution'].sum()
    n_pass = qso_df['astrometric_pass'].sum()
    print(f"  5/6p Gaia solutions:                    {n_5p} / {len(qso_df)}")
    print(f"  Astrometric pass (Mahal < {sigma_cut}σ):        "
          f"{n_pass} / {n_5p} (of 5/6p sources)")

    # ── Final flag ────────────────────────────────────────────────────────────
    qso_df['is_qso_anchor'] = qso_df['catalog_match'] & qso_df['astrometric_pass']
    n_anchor = qso_df['is_qso_anchor'].sum()
    print(f"  QSO anchors (catalog match AND astrom pass): {n_anchor} / {len(qso_df)}")

    qso_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}  ({n_anchor} anchors)")

    return qso_df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_catalog(lib_dir: "Path | None", filename: str) -> "Path | None":
    if lib_dir is None:
        return None
    p = Path(lib_dir) / "qso_catalogs" / filename
    return p if p.exists() else None


def _load_gaia_astrometry(gaia_dir: Path) -> "pd.DataFrame | None":
    """Load the main Gaia CSV (non-qso/galaxy files) filtered to in_qso_candidates."""
    paths = [p for p in sorted(gaia_dir.glob("*_G*.csv"))
             if 'qso' not in p.name and 'galaxy' not in p.name]
    if not paths:
        return None
    dfs = [pd.read_csv(p, dtype={'source_id': 'int64'}) for p in paths]
    df  = pd.concat(dfs, ignore_index=True)
    # Keep only potential QSOs to save memory during chi2 computation
    if 'in_qso_candidates' in df.columns:
        df = df[df['in_qso_candidates'].fillna(False)]
    return df


def _match_quaia(qso_df: pd.DataFrame, quaia_path: Path) -> pd.DataFrame:
    """Cross-match by int64 source_id.  Always preserves int64 precision."""
    from astropy.io import fits
    with fits.open(quaia_path) as h:
        q_ids  = h[1].data['source_id'].astype(np.int64)
        q_z    = h[1].data['redshift_quaia'].astype(float)

    q_id_set = dict(zip(q_ids, q_z))   # source_id → redshift

    qso_df = qso_df.copy()
    qso_df['quaia_match'] = qso_df['source_id'].isin(q_id_set)
    qso_df['quaia_z']     = qso_df['source_id'].map(q_id_set)
    return qso_df


def _match_milliquas(
    qso_df: pd.DataFrame,
    mq_path: Path,
    search_radius_arcsec: float,
) -> pd.DataFrame:
    """Cross-match by RA/Dec using astropy SkyCoord (builds KD-tree internally)."""
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    with fits.open(mq_path) as h:
        mq_ra      = h[1].data['RA'].astype(float)
        mq_dec     = h[1].data['DEC'].astype(float)
        mq_z       = h[1].data['Z'].astype(float)
        mq_type    = np.array([s.strip() for s in h[1].data['TYPE']])
        mq_comment = np.array([s.strip() for s in h[1].data['COMMENT']])

    mq_gaia_pos = np.array(['G' in c for c in mq_comment])

    cat_sc = SkyCoord(ra=mq_ra * u.deg, dec=mq_dec * u.deg)
    src_sc = SkyCoord(ra=qso_df['ra'].values * u.deg,
                      dec=qso_df['dec'].values * u.deg)

    idx, sep, _ = src_sc.match_to_catalog_sky(cat_sc)
    sep_arcsec  = sep.arcsec

    # Use half radius for sources with Gaia EDR3 positions (sub-arcsec accuracy)
    max_sep = np.where(mq_gaia_pos[idx],
                       search_radius_arcsec * 0.5,
                       search_radius_arcsec)
    matched = sep_arcsec <= max_sep

    qso_df = qso_df.copy()
    qso_df['milliquas_match']      = matched
    qso_df['milliquas_sep_arcsec'] = sep_arcsec
    qso_df['milliquas_z']          = np.where(matched, mq_z[idx].astype(float), np.nan)
    qso_df['milliquas_type']       = np.where(matched, mq_type[idx], '')
    qso_df['milliquas_gaia_pos']   = np.where(matched, mq_gaia_pos[idx], False)
    return qso_df


def _astrometric_cut(
    qso_df: pd.DataFrame,
    gaia_astro: "pd.DataFrame | None",
    sigma_cut: float,
) -> pd.DataFrame:
    """Add has_5p_solution, chi2_astrometric, astrometric_pass columns.

    The expected (pmra, pmdec, parallax) = (secular_aberration, 0).
    Covariance is scaled as in BP3M's solver (GAIA_SYS_DICT).
    For sources without a 5p/6p solution, astrometric_pass = False.
    """
    qso_df = qso_df.copy()
    qso_df['has_5p_solution']    = False
    qso_df['chi2_astrometric']   = np.nan
    qso_df['astrometric_pass']   = False

    if gaia_astro is None or len(gaia_astro) == 0:
        print("  WARNING: no Gaia astrometry loaded — astrometric cut skipped")
        return qso_df

    # Merge astrometry into qso_df
    astro_cols = ['source_id', 'pmra', 'pmdec', 'parallax',
                  'pmra_error', 'pmdec_error', 'parallax_error',
                  'pmra_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr',
                  'astrometric_params_solved']
    astro_cols = [c for c in astro_cols if c in gaia_astro.columns]
    merged = qso_df.merge(
        gaia_astro[astro_cols].rename(columns={'source_id': '_sid'}),
        left_on='source_id', right_on='_sid', how='left'
    ).drop(columns=['_sid'], errors='ignore')

    # 5p/6p: astrometric_params_solved ∈ {31 (5p), 95 (6p)} → both have PM+plx
    aps   = merged.get('astrometric_params_solved', pd.Series(np.nan, index=merged.index))
    is_5p = aps.isin([31])
    is_6p = aps.isin([95])
    has_5p = (is_5p | is_6p) & merged['pmra'].notna() & merged['parallax'].notna()

    merged_5p = merged[has_5p].copy()

    if len(merged_5p) == 0:
        qso_df['has_5p_solution'] = False
        return qso_df

    # Secular aberration prediction (mas/yr) for each source
    pmra_aberr_mas, pmdec_aberr_mas = secular_aberration_pm(
        merged_5p['ra'].values, merged_5p['dec'].values
    )
    pmra_aberr_mas  /= 1000.0   # µas/yr → mas/yr
    pmdec_aberr_mas /= 1000.0

    # Build per-source 3×3 scaled covariance for (pmra, pmdec, parallax)
    pmra_e = merged_5p['pmra_error'].values
    pdec_e = merged_5p['pmdec_error'].values
    plx_e  = merged_5p['parallax_error'].values

    corr_pp  = merged_5p.get('pmra_pmdec_corr',   pd.Series(0.0, index=merged_5p.index)).fillna(0).values
    corr_rp  = merged_5p.get('parallax_pmra_corr', pd.Series(0.0, index=merged_5p.index)).fillna(0).values
    corr_dp  = merged_5p.get('parallax_pmdec_corr',pd.Series(0.0, index=merged_5p.index)).fillna(0).values

    # C_raw[i] = 3×3 covariance (pmra, pmdec, parallax)
    n = len(merged_5p)
    C = np.zeros((n, 3, 3))
    C[:, 0, 0] = pmra_e ** 2
    C[:, 1, 1] = pdec_e ** 2
    C[:, 2, 2] = plx_e  ** 2
    C[:, 0, 1] = C[:, 1, 0] = corr_pp * pmra_e * pdec_e
    C[:, 0, 2] = C[:, 2, 0] = corr_rp * pmra_e * plx_e
    C[:, 1, 2] = C[:, 2, 1] = corr_dp * pdec_e * plx_e

    # Scale by BP3M mult factor (1.05 for 5p, 1.22 for 6p)
    mult = np.where(is_6p[has_5p].values, _MULT['6p'], _MULT['5p'])
    C = C * mult[:, None, None]

    # Add systematic noise floor in quadrature to diagonal
    C[:, 0, 0] += _PM_SYS  ** 2
    C[:, 1, 1] += _PM_SYS  ** 2
    C[:, 2, 2] += _PLX_SYS ** 2

    # Mahalanobis distance
    delta = np.column_stack([
        merged_5p['pmra'].values  - pmra_aberr_mas,
        merged_5p['pmdec'].values - pmdec_aberr_mas,
        merged_5p['parallax'].values,    # expected = 0
    ])
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        C_inv = np.array([np.linalg.pinv(C[i]) for i in range(n)])

    chi2_3d  = np.einsum('ni,nij,nj->n', delta, C_inv, delta)
    mahal    = np.sqrt(np.maximum(chi2_3d, 0.0))

    # Write results back to qso_df using the matched index
    idx_5p = merged_5p.index  # labels in merged (which has same index as qso_df after merge)
    qso_df.loc[idx_5p, 'has_5p_solution']  = True
    qso_df.loc[idx_5p, 'chi2_astrometric'] = mahal
    qso_df.loc[idx_5p, 'astrometric_pass'] = mahal < sigma_cut

    return qso_df
