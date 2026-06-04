"""
Shared utilities for loading pipeline outputs and computing astrometric
quantities for use in both the pipeline and the exploration notebooks.

All functions that work with Gaia covariances apply the same inflation as
bp3m/solver.py so that uncertainties are directly comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .catalog_utils import (
    build_gaia_cov, cov2_geom_sigma,
    pm_uncertainty, pos_uncertainty, GAIA_SYS,
)


# ── Catalogue loaders ────────────────────────────────────────────────────────

def load_gaia_catalog(path: str | Path) -> pd.DataFrame:
    """
    Load a Gaia CSV catalogue produced by download_gaia.

    Ensures ``bp_rp`` is present and casts numeric columns to float.
    Returns the raw DataFrame including quality-flag columns.
    """
    df = pd.read_csv(path)
    if 'bp_rp' not in df.columns:
        if 'bpmag' in df.columns and 'rpmag' in df.columns:
            df['bp_rp'] = df['bpmag'] - df['rpmag']
    return df


def load_bp3m_results(results_dir: str | Path) -> dict:
    """
    Load all outputs produced by run_alignment (BP3M step).

    Parameters
    ----------
    results_dir : path to BP3M_results/ directory

    Returns
    -------
    dict with keys:
        'stars'   : pd.DataFrame  — stellar_astrometry.csv
        'images'  : pd.DataFrame  — image_transformations.csv
        'v_cov'   : (N,5,5) ndarray — marginalised posterior covariance
        'C_vT'    : (N,5,5) ndarray — conditional posterior covariance
    """
    d = Path(results_dir)
    out = {}
    stars_path = d / "stellar_astrometry.csv"
    imgs_path  = d / "image_transformations.csv"
    vcov_path  = d / "v_cov_marginalised.npy"
    cvt_path   = d / "C_vT.npy"

    if stars_path.exists():
        out['stars'] = pd.read_csv(stars_path)
    if imgs_path.exists():
        out['images'] = pd.read_csv(imgs_path)
    if vcov_path.exists():
        out['v_cov'] = np.load(vcov_path)
    if cvt_path.exists():
        out['C_vT'] = np.load(cvt_path)

    if not out:
        raise FileNotFoundError(f"No BP3M results found in {results_dir}")
    return out


def load_cross_match_results(output_dir: str | Path, field_name: str,
                              telescope: str = 'HST',
                              im_type: str = '_flc') -> pd.DataFrame:
    """
    Load all per-image matched_gaia.csv files into a single DataFrame
    with an extra 'image_name' column.
    """
    root = (Path(output_dir) / field_name / telescope.upper()
            / "mastDownload" / telescope.upper())
    frames = []
    for obs_dir in sorted(root.iterdir()) if root.exists() else []:
        if not obs_dir.is_dir():
            continue
        match_csv = obs_dir / "matched_gaia.csv"
        if match_csv.exists():
            df = pd.read_csv(match_csv)
            df.insert(0, 'image_name', obs_dir.name)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No matched_gaia.csv files found under {root}")
    return pd.concat(frames, ignore_index=True)


def load_transformations(output_dir: str | Path, field_name: str,
                          telescope: str = 'HST',
                          im_type: str = '_flc') -> pd.DataFrame:
    """
    Load all per-image transformation.csv files into a single DataFrame
    (wide format, one row per image, one column per parameter).
    """
    root = (Path(output_dir) / field_name / telescope.upper()
            / "mastDownload" / telescope.upper())
    rows = []
    for obs_dir in sorted(root.iterdir()) if root.exists() else []:
        if not obs_dir.is_dir():
            continue
        tran_csv = obs_dir / "transformation.csv"
        if tran_csv.exists():
            df = pd.read_csv(tran_csv).set_index('parameter')['value']
            row = df.to_dict()
            row['image_name'] = obs_dir.name
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No transformation.csv files found under {root}")
    return pd.DataFrame(rows).set_index('image_name')


# ── Covariance utilities ─────────────────────────────────────────────────────

# Re-export from catalog_utils for convenience
build_gaia_cov      = build_gaia_cov
cov2_geom_sigma     = cov2_geom_sigma
pm_uncertainty      = pm_uncertainty
pos_uncertainty     = pos_uncertainty


def bp3m_full_cov(bp3m: dict) -> np.ndarray:
    """
    Return full marginalised 5×5 covariance = v_cov + C_vT.
    Requires both keys in the dict returned by load_bp3m_results().
    """
    return bp3m['v_cov'] + bp3m['C_vT']


def gaia_pm_sigma(df: pd.DataFrame) -> np.ndarray:
    """
    Per-star geometric-mean PM uncertainty from inflated Gaia covariance (mas/yr).
    """
    C = build_gaia_cov(df)
    return pm_uncertainty(C)


def gaia_pos_sigma(df: pd.DataFrame) -> np.ndarray:
    """
    Per-star geometric-mean position uncertainty from inflated Gaia covariance (mas).
    """
    C = build_gaia_cov(df)
    return pos_uncertainty(C)


def bp3m_pm_sigma(bp3m: dict) -> np.ndarray:
    """
    Per-star geometric-mean PM uncertainty from full BP3M posterior (mas/yr).
    """
    return pm_uncertainty(bp3m_full_cov(bp3m))


def bp3m_pos_sigma(bp3m: dict) -> np.ndarray:
    """
    Per-star geometric-mean position uncertainty from BP3M posterior (mas).
    """
    return pos_uncertainty(bp3m_full_cov(bp3m))


# ── Gaia epoch propagation ───────────────────────────────────────────────────

def propagate_gaia(df: pd.DataFrame, target_mjd: float,
                   zero_pm: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate Gaia positions to a target MJD using proper motion and parallax.

    Uses the same formula as fast_cross_match_claude.cross_match_cli.
    Returns (ra_prop, dec_prop) in degrees.
    """
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris

    ref_epoch = df['ref_epoch'].iloc[0] if 'ref_epoch' in df.columns else 2016.0
    t_target  = Time(target_mjd, format='mjd')
    dt        = t_target.jyear - ref_epoch
    n = len(df)

    ra_rad  = np.radians(df['ra'].values)
    dec_rad = np.radians(df['dec'].values)

    if zero_pm:
        plx = pmra = pmdec = np.zeros(n)
    else:
        plx   = df['parallax'].fillna(0.0).values if 'parallax' in df else np.zeros(n)
        pmra  = df['pmra'].fillna(0.0).values     if 'pmra' in df    else np.zeros(n)
        pmdec = df['pmdec'].fillna(0.0).values    if 'pmdec' in df   else np.zeros(n)

    with solar_system_ephemeris.set('builtin'):
        earth = get_body_barycentric('earth', t_target)
    X = earth.x.to_value('au')
    Y = earth.y.to_value('au')
    Z = earth.z.to_value('au')

    p_ra  = X * np.sin(ra_rad) - Y * np.cos(ra_rad)
    p_dec = (X * np.cos(ra_rad) * np.sin(dec_rad)
             + Y * np.sin(ra_rad) * np.sin(dec_rad)
             - Z * np.cos(dec_rad))

    ra_off  = pmra  * dt + plx * p_ra   # mas
    dec_off = pmdec * dt + plx * p_dec  # mas

    ra_prop  = df['ra'].values  + (ra_off  / 3_600_000.0) / np.cos(dec_rad)
    dec_prop = df['dec'].values + (dec_off / 3_600_000.0)
    return ra_prop, dec_prop


# ── Plot helpers ─────────────────────────────────────────────────────────────

def vpd_limits(pmra: np.ndarray, pmdec: np.ndarray,
               n_sigma: float = 4.0) -> tuple[tuple, tuple]:
    """Return (xlim, ylim) centred on the median PM with n_sigma spread."""
    med_ra  = np.nanmedian(pmra)
    med_dec = np.nanmedian(pmdec)
    sig_ra  = 1.4826 * np.nanmedian(np.abs(pmra  - med_ra))
    sig_dec = 1.4826 * np.nanmedian(np.abs(pmdec - med_dec))
    half    = n_sigma * max(sig_ra, sig_dec, 0.1)
    return (med_ra  - half, med_ra  + half), (med_dec - half, med_dec + half)


def sky_extent(ra: np.ndarray, dec: np.ndarray,
               pad_frac: float = 0.05) -> tuple[tuple, tuple]:
    """Return (ra_lim, dec_lim) with fractional padding."""
    ra_min, ra_max   = np.nanmin(ra),  np.nanmax(ra)
    dec_min, dec_max = np.nanmin(dec), np.nanmax(dec)
    pad_ra  = (ra_max  - ra_min)  * pad_frac
    pad_dec = (dec_max - dec_min) * pad_frac
    return ((ra_max  + pad_ra,  ra_min  - pad_ra),   # RA increases right-to-left
            (dec_min - pad_dec, dec_max + pad_dec))
