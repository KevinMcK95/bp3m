"""
Gaia catalogue quality flags, covariance construction, and error inflation.

The inflation constants and logic match bp3m/astro_utils.py exactly so that
any analysis performed here produces identical uncertainties to those used
inside the BP3M solver.
"""

import numpy as np
import pandas as pd

# ── Gaia systematic constants (Vasiliev & Baumgardt 2021, MNRAS 505, 5978) ──
GAIA_SYS = {
    'mult_6p':          1.22,    # covariance multiplier for 6-param solutions
    'mult_5p':          1.05,    # covariance multiplier for 5-param solutions
    'mult_2p':          1.00,
    'parallax_sys_err': 0.011,   # mas
    'pm_sys_err':       0.026,   # mas/yr
}

# Columns that bp3m's data_loader_flc expects in the Gaia CSV
GAIA_REQUIRED_COLS = [
    "source_id", "ra", "dec", "ra_error", "dec_error", "ra_dec_corr",
    "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
    "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
    "parallax", "parallax_error", "parallax_pmra_corr", "parallax_pmdec_corr",
    "pmra", "pmra_error", "pmra_pmdec_corr", "pmdec", "pmdec_error",
    "ref_epoch", "ruwe", "pseudocolour",
    "gmag", "gmag_error", "bpmag", "bpmag_error", "rpmag", "rpmag_error",
    "bp_rp", "bp_rp_error",
]


# ── Gaia quality filtering ───────────────────────────────────────────────────

def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
    """Corrected flux excess factor (Riello et al. 2020 / GaiaHub convention)."""
    bp_rp = np.asarray(bp_rp, dtype=float)
    C = np.asarray(phot_bp_rp_excess_factor, dtype=float)
    corr = np.zeros_like(bp_rp)
    blue  = ~np.isnan(bp_rp) & (bp_rp < 0.5)
    green = ~np.isnan(bp_rp) & (bp_rp >= 0.5) & (bp_rp < 4.0)
    red   = ~np.isnan(bp_rp) & (bp_rp > 4.0)
    corr[blue]  = 1.154360 + 0.033772*bp_rp[blue]  + 0.032277*bp_rp[blue]**2
    corr[green] = 1.162004 + 0.011464*bp_rp[green] + 0.049255*bp_rp[green]**2 \
                           - 0.005879*bp_rp[green]**3
    corr[red]   = 1.057572 + 0.140537*bp_rp[red]
    return C - corr


def clean_astrometry(df, use_5p=False):
    """Boolean mask: True = good astrometry (RUWE, harmonic amp, visibility, noise)."""
    ok = (
        (df['ruwe'] <= 1.4) &
        (df['ipd_gof_harmonic_amplitude'] <= 0.2) &
        (df['visibility_periods_used'] >= 9) &
        (df['astrometric_excess_noise_sig'] <= 2.0)
    )
    if use_5p:
        ok = ok & (df['astrometric_params_solved'] == 31)
    return ok.values


def clean_photometry(gmag, corrected_flux_excess_factor, sigma=3.0):
    """Boolean mask: True = good photometry (flux excess within sigma band)."""
    from matplotlib.path import Path as MPath
    gmag = np.asarray(gmag, dtype=float)
    cfe  = np.asarray(corrected_flux_excess_factor, dtype=float)

    def _sigma_C(g, s):
        return s * (0.0059898 + 8.817481e-12 * g**7.618399)

    nodes = np.linspace(gmag.min() - 0.1, gmag.max() + 0.1, 100)
    up   = list(zip(nodes, _sigma_C(nodes,  sigma)))
    down = list(zip(nodes[::-1], _sigma_C(nodes, -sigma)[::-1]))
    path = MPath(up + down, closed=True)
    return path.contains_points(np.column_stack([gmag, cfe]))


def apply_quality_flags(df, sigma_flux_excess=3.0, use_5p=False):
    """Return df filtered to rows passing both astrometric and photometric cuts."""
    df = df.copy()
    df['corrected_flux_excess_factor'] = correct_flux_excess_factor(
        df['bp_rp'], df['phot_bp_rp_excess_factor'])
    ok_astro  = clean_astrometry(df, use_5p=use_5p)
    ok_photo  = clean_photometry(df['gmag'].values,
                                  df['corrected_flux_excess_factor'].values,
                                  sigma=sigma_flux_excess)
    df['clean_label'] = ok_astro & ok_photo
    return df


# ── Gaia covariance construction ─────────────────────────────────────────────

def build_gaia_cov(df):
    """
    Build inflated 5×5 Gaia covariance matrices for each star.

    Order of axes: (Δα*, Δδ, μα*, μδ, ϖ)  (same as bp3m convention).
    Inflation matches bp3m/solver.py _cache_gaia() exactly:
      - 6-param solutions (pseudocolour finite): C *= mult_6p
      - 5-param solutions (pmra finite, pseudocolour NaN): C *= mult_5p
      - systematic floor added to parallax and PM diagonal entries

    Parameters
    ----------
    df : pd.DataFrame  — Gaia catalogue with standard column names

    Returns
    -------
    C : (N, 5, 5) ndarray  — inflated covariance matrices in (mas, mas/yr) units
    """
    n = len(df)

    def _get(col, default=0.0):
        if col in df.columns:
            return df[col].fillna(default).to_numpy(float)
        return np.full(n, default)

    ra_e    = _get('ra_error',        1e6)
    dec_e   = _get('dec_error',       1e6)
    pmra_e  = _get('pmra_error',      1e3)
    pmdec_e = _get('pmdec_error',     1e3)
    plx_e   = _get('parallax_error',  1e3)

    # Stars with formal PM error > 100 mas/yr have unreliable Gaia astrometry
    pmra_e  = np.where(pmra_e  > 100, np.nan, pmra_e)
    pmdec_e = np.where(pmdec_e > 100, np.nan, pmdec_e)

    sigmas = np.stack([ra_e, dec_e, pmra_e, pmdec_e, plx_e], axis=1)  # (N, 5)

    corr = np.zeros((n, 5, 5))
    for i in range(5):
        corr[:, i, i] = 1.0
    pairs = [
        (0, 1, _get('ra_dec_corr')),
        (0, 2, _get('ra_pmra_corr')),
        (0, 3, _get('ra_pmdec_corr')),
        (0, 4, _get('ra_parallax_corr')),
        (1, 2, _get('dec_pmra_corr')),
        (1, 3, _get('dec_pmdec_corr')),
        (1, 4, _get('dec_parallax_corr')),
        (2, 3, _get('pmra_pmdec_corr')),
        (2, 4, _get('parallax_pmra_corr')),
        (3, 4, _get('parallax_pmdec_corr')),
    ]
    for i, j, arr in pairs:
        corr[:, i, j] = arr
        corr[:, j, i] = arr

    C = sigmas[:, :, None] * corr * sigmas[:, None, :]

    # Inflation (matches bp3m/solver.py exactly — multiplies covariance, not sigma²)
    gaia_6p = np.isfinite(df['pseudocolour'].values)
    gaia_5p = np.isfinite(df['pmra'].values) & ~gaia_6p
    gaia_2p = ~gaia_5p & ~gaia_6p

    C[gaia_6p] *= GAIA_SYS['mult_6p']
    C[gaia_5p] *= GAIA_SYS['mult_5p']
    C[gaia_2p] *= GAIA_SYS['mult_2p']

    # Systematic floor on PM and parallax
    floor = np.diag(np.array([0, 0,
                               GAIA_SYS['pm_sys_err'],
                               GAIA_SYS['pm_sys_err'],
                               GAIA_SYS['parallax_sys_err']])**2)
    C += floor

    return C


def cov2_geom_sigma(C):
    """
    Geometric-mean uncertainty for a set of 2×2 covariance matrices.

    sigma_geom = det(C)^(1/4)

    For uncorrelated (diagonal) C this equals sqrt(sigma_x * sigma_y).

    Parameters
    ----------
    C : (N, 2, 2) ndarray

    Returns
    -------
    sigma : (N,) ndarray  [same units as the covariance matrix's sqrt]
    """
    C = np.asarray(C)
    det = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    return np.where(det > 0, det**0.25, np.nan)


def pm_uncertainty(C_5x5):
    """Geometric-mean PM uncertainty from 5×5 Gaia/BP3M covariance (mas/yr)."""
    return cov2_geom_sigma(C_5x5[:, 2:4, 2:4])


def pos_uncertainty(C_5x5):
    """Geometric-mean position uncertainty from 5×5 Gaia/BP3M covariance (mas)."""
    return cov2_geom_sigma(C_5x5[:, 0:2, 0:2])
