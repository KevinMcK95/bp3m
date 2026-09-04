"""
DELVE proper-motion catalogue ↔ HST cross-matching.

Mirrors process_single_image() from cross_match.py but uses the DELVE CSV
(produced by download_delve.py) as the reference catalogue instead of Gaia.

Key differences from the Gaia cross-match
------------------------------------------
- Reference magnitude  : r_mag  (DES r-band, closest to Gaia G)
- Color                : g_mag − r_mag  (instead of BP-RP)
- All sources have PMs — no 2p / 5p / 6p Gaia solution classification
- No Gaia-style covariance inflation (unknown DELVE calibration; conservative
  systematic floors applied instead: 10 mas position, 1 mas/yr PM)
- Output files         : matched_delve.csv, diagnostic_plots_delve.png,
                         offset_histogram_delve.png
- Discovery tiers      : [all DELVE / HST stars], [all DELVE / all HST]
  (no "clean 5p" tier since all DELVE sources have PM fits)
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from scipy.spatial import KDTree

from .cross_match import (
    get_hst_params,
    _run_4p_discovery,
    _run_affine_refinement,
    _plot_offset_histogram,
    compute_mahalanobis,
    compute_logprob_cost,
    apply_affine,
    rd2x, rd2y,
    FileLogger,
)
from bp3m.instrument_config import SIGMA_ROT_DEG, SIGMA_SCALE, SIGMA_SKEW

def load_delve_data(delve_csv_path: str) -> "pd.DataFrame | None":
    """Load the DELVE CSV produced by download_delve.py and apply quality filters.

    Applies only essential quality cuts:
    - finite astrometry and r_mag
    - r_mag outside the DELVE sentinel range (-99, +99)
    All mtype categories (modest1/2/3, fast) are retained; the cross-match
    magnitude and sigma cuts will naturally reject poorly-constrained sources.
    """
    if not os.path.exists(delve_csv_path):
        print(f'  DELVE CSV not found: {delve_csv_path}')
        return None
    df = pd.read_csv(delve_csv_path, dtype={'source_id': np.int64})
    req = ['ra', 'dec', 'ra_error', 'dec_error', 'pmra', 'pmdec',
           'pmra_error', 'pmdec_error', 'r_mag']
    ok = np.ones(len(df), dtype=bool)
    for c in req:
        if c in df.columns:
            ok &= np.isfinite(df[c])
    # DELVE uses -99 and +99 as sentinels for missing photometry
    if 'r_mag' in df.columns:
        ok &= (df['r_mag'] > -90) & (df['r_mag'] < 50)
    df = df[ok].reset_index(drop=True)
    if 'mtype' in df.columns:
        mtype_counts = df['mtype'].value_counts().to_dict()
        mtype_str = ', '.join(f'{k}:{v:,}' for k, v in sorted(mtype_counts.items()))
    else:
        mtype_str = 'unknown'
    print(f'  DELVE catalogue: {len(df):,} sources  [{mtype_str}]')
    return df


# ── DELVE systematic error floors ────────────────────────────────────────────
# Position systematics: ~10 mas from DECam astrometric calibration residuals.
# PM systematics: ~1 mas/yr — conservative placeholder (DELVE-specific value TBD).
_DELVE_POS_SYS_MAS   = 10.0   # mas
_DELVE_PM_SYS_MASYR  =  1.0   # mas/yr


def _construct_delve_cov(df: pd.DataFrame) -> np.ndarray:
    """Build (N, 5, 5) covariance in (RA, Dec, plx, pmra, pmdec) for DELVE sources.

    Units consistent with Gaia convention: mas for positions/parallax, mas/yr for PM.
    Systematic floors are added in quadrature to all diagonal entries.
    """
    n = len(df)
    sig = np.zeros((n, 5))
    sig[:, 0] = df['ra_error'].values                       # mas
    sig[:, 1] = df['dec_error'].values                      # mas
    sig[:, 2] = df['parallax_error'].fillna(20.0).values   # mas
    sig[:, 3] = df['pmra_error'].values                     # mas/yr
    sig[:, 4] = df['pmdec_error'].values                    # mas/yr

    corr_cols = {
        (0, 1): 'ra_dec_corr',
        (0, 2): 'ra_parallax_corr',  (0, 3): 'ra_pmra_corr',  (0, 4): 'ra_pmdec_corr',
        (1, 2): 'dec_parallax_corr', (1, 3): 'dec_pmra_corr', (1, 4): 'dec_pmdec_corr',
        (2, 3): 'parallax_pmra_corr',(2, 4): 'parallax_pmdec_corr',
        (3, 4): 'pmra_pmdec_corr',
    }
    cov = np.zeros((n, 5, 5))
    for i in range(5):
        cov[:, i, i] = sig[:, i] ** 2
    for (i, j), col in corr_cols.items():
        if col in df.columns:
            v = df[col].fillna(0.0).values * sig[:, i] * sig[:, j]
            cov[:, i, j] = v
            cov[:, j, i] = v

    # Add systematic floors in quadrature
    sys_diag = np.array([_DELVE_POS_SYS_MAS**2,   # RA
                         _DELVE_POS_SYS_MAS**2,   # Dec
                         _DELVE_POS_SYS_MAS**2,   # parallax (use position sys as proxy)
                         _DELVE_PM_SYS_MASYR**2,  # pmra
                         _DELVE_PM_SYS_MASYR**2]) # pmdec
    cov += np.diag(sys_diag)
    return cov


def _propagate_delve(df: pd.DataFrame, target_mjd: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate DELVE positions + 5D covariance to the HST observation epoch.

    Returns (ra_prop, dec_prop, Ct_2x2) where Ct_2x2 is the propagated 2D
    positional covariance in (mas², mas², mas²) — same shape as what
    propagate_gaia_with_cov returns before pixel projection.
    """
    ref_epoch = df['ref_epoch'].iloc[0] if 'ref_epoch' in df.columns else 2016.0
    t_hst  = Time(target_mjd, format='mjd')
    dt     = t_hst.jyear - ref_epoch
    n      = len(df)

    ra_rad  = np.radians(df['ra'].values)
    dec_rad = np.radians(df['dec'].values)
    plx     = df['parallax'].fillna(0.0).values    # mas
    pmra    = df['pmra'].values                    # mas/yr
    pmdec   = df['pmdec'].values                   # mas/yr

    from bp3m.astro_utils import (get_tele_position, get_parallax_factors,
                                  propagate_gaia_positions)
    with solar_system_ephemeris.set('builtin'):
        tele_xyz = get_tele_position(t_hst, curr_id='earth')
    p_ra, p_dec = get_parallax_factors(df['ra'].values, df['dec'].values, tele_xyz)
    ra_prop, dec_prop = propagate_gaia_positions(
        df['ra'].values, df['dec'].values, pmra, pmdec, plx, dt, tele_xyz)

    C0 = _construct_delve_cov(df)   # (N, 5, 5)
    J  = np.zeros((n, 2, 5))
    J[:, 0, 0] = 1.0; J[:, 0, 2] = p_ra;  J[:, 0, 3] = dt
    J[:, 1, 1] = 1.0; J[:, 1, 2] = p_dec; J[:, 1, 4] = dt
    Ct = np.einsum('nij,njk,nlk->nil', J, C0, J)   # (N, 2, 2) positional cov in mas²
    return ra_prop, dec_prop, Ct


def _save_diagnostic_plots_delve(out_dir: str, image_name: str,
                                  matched_df: pd.DataFrame,
                                  rejected_df: pd.DataFrame,
                                  delve_field_df: "pd.DataFrame | None" = None,
                                  label: str = 'delve',
                                  out_suffix: str = '',
                                  gaia_cmd: bool = False,
                                  color_hst_label: str = None,
                                  color_hst_sign: float = 1.0) -> None:
    """10-panel diagnostic figure for the external-catalog cross-match.

    Layout mirrors save_diagnostic_plots() from cross_match.py exactly:
      (0,0) Field map          (0,1) DELVE PM VPD
      (1,0) DELVE CMD (r vs g−r)   (1,1) DELVE vs HST CMD (r vs r−HST)
      (2,0) XY residuals       (2,1) Normalised residuals
      (3,0) Residual vs r mag  (3,1) Sigma histogram
      (4,0) Sigma vs r mag     (4,1) Color-color (g−r vs r−HST)

    Parameters
    ----------
    delve_field_df : full in-field DELVE source table (r_mag, g_mag columns);
                     drawn as grey background population in CMD panels.
    """
    L = label.upper()
    fig, axes = plt.subplots(5, 2, figsize=(14, 24/4*5))
    fig.suptitle(f"DELVE Match Diagnostics: {image_name}", fontsize=18)

    all_df = pd.concat([matched_df, rejected_df], ignore_index=True)
    has_sc = 'hst_is_star' in matched_df.columns
    m_stars    = matched_df[matched_df['hst_is_star'].astype(bool)] if has_sc else matched_df
    m_nonstars = (matched_df[~matched_df['hst_is_star'].astype(bool)]
                  if has_sc else matched_df.iloc[0:0])

    def _scatter_matched(ax, cx, cy, **kw):
        if len(m_nonstars):
            ax.scatter(m_nonstars[cx], m_nonstars[cy],
                       c='orange', alpha=0.6, s=12, label='Matched non-star', **kw)
        if len(m_stars):
            ax.scatter(m_stars[cx], m_stars[cy],
                       c='blue', alpha=0.6, s=10, label='Matched star', **kw)

    mag_lims = None
    if len(all_df):
        mag_min, mag_max = all_df['mag'].min(), all_df['mag'].max()
        mag_pad = (mag_max - mag_min) * 0.05
        mag_lims = (mag_min - mag_pad, mag_max + mag_pad)

    # 1. Field map
    ax = axes[0, 0]
    if delve_field_df is not None:
        ax.scatter(delve_field_df['x'], delve_field_df['y'],
                   c='grey', s=2, alpha=0.2, label=f'{L} field', zorder=1)
    if len(all_df):
        lc = [[(r.x, r.y), (r.hx, r.hy)] for r in all_df.itertuples()]
        ax.add_collection(LineCollection(lc, colors='grey', alpha=0.15, linewidths=0.5, zorder=2))
    ax.scatter(all_df['hx'], all_df['hy'], c='lightgrey', s=2, alpha=0.3, zorder=2)
    if len(rejected_df):
        ax.scatter(rejected_df['x'], rejected_df['y'], c='red', s=5, alpha=0.4, label='Rejected', zorder=3)
    if len(m_nonstars):
        ax.scatter(m_nonstars['x'], m_nonstars['y'], c='orange', alpha=0.8, s=14, label='Matched non-star', zorder=5)
    if len(m_stars):
        ax.scatter(m_stars['x'], m_stars['y'], c='blue', alpha=0.8, s=12, label='Matched star', zorder=5)
    ax.set_xlabel(f'X_{L} (pixels)'); ax.set_ylabel(f'Y_{L} (pixels)')
    ax.set_title('Field Map (Pixels)'); ax.legend(fontsize=7)

    # 2. DELVE PM VPD
    ax = axes[0, 1]
    if delve_field_df is not None:
        ax.scatter(delve_field_df['pmra'], delve_field_df['pmdec'],
                   c='grey', s=2, alpha=0.2, label=f'{L} field', zorder=1)
    if len(rejected_df):
        ax.scatter(rejected_df['pmra'], rejected_df['pmdec'],
                   c='red', s=5, alpha=0.4, label='Rejected', zorder=3)
    _scatter_matched(ax, 'pmra', 'pmdec')
    ax.set_xlabel('PMRA (mas/yr)'); ax.set_ylabel('PMDec (mas/yr)')
    ax.set_title(f'{L} Proper Motions'); ax.legend(fontsize=7)

    # 3. CMD — Gaia G vs BP−RP when gaia_cmd, else r vs g−r
    ax = axes[1, 0]
    if gaia_cmd:
        if delve_field_df is not None and 'gmag' in delve_field_df.columns:
            v = delve_field_df['gmag'].notna() & delve_field_df['bp_rp'].notna()
            ax.scatter(delve_field_df.loc[v, 'bp_rp'],
                       delve_field_df.loc[v, 'gmag'],
                       c='grey', s=3, alpha=0.3, label=f'{L} field (Gaia)',
                       zorder=1)
        for src, col, lbl in [(rejected_df, 'red', 'Rejected'),
                              (m_nonstars, 'orange', 'Matched non-star'),
                              (m_stars, 'blue', 'Matched star')]:
            if len(src) and 'gaia_bp_rp' in src.columns:
                v = src['gaia_bp_rp'].notna() & src['gaia_gmag'].notna()
                ax.scatter(src.loc[v, 'gaia_bp_rp'], src.loc[v, 'gaia_gmag'],
                           c=col, alpha=0.6 if col != 'red' else 0.15,
                           s=12 if col != 'red' else 5, label=lbl, zorder=2)
        ax.invert_yaxis()
        ax.set_xlabel('BP − RP (Gaia, mag)'); ax.set_ylabel('G (Gaia, mag)')
        ax.set_title('Gaia Color-Magnitude Diagram'); ax.legend(fontsize=7)
    elif delve_field_df is not None and 'g_mag' in delve_field_df.columns:
        valid = ((delve_field_df['r_mag'] > -90) & (delve_field_df['r_mag'] < 50) &
                 (delve_field_df['g_mag'] > -90) & (delve_field_df['g_mag'] < 50))
        pop = delve_field_df[valid]
        if len(pop):
            gr_pop = pop['g_mag'].values - pop['r_mag'].values
            ax.scatter(gr_pop, pop['r_mag'].values, c='grey', s=3, alpha=0.3, label=f'{L} field', zorder=1)
    for src, col, lbl in ([] if gaia_cmd else
                          [(rejected_df, 'red', 'Rejected'),
                           (m_nonstars, 'orange', 'Matched non-star'),
                           (m_stars, 'blue', 'Matched star')]):
        if len(src) and 'color' in src.columns:
            valid = (src['color'] > -10) & (src['color'] < 10)
            ax.scatter(src.loc[valid, 'color'], src.loc[valid, 'mag'],
                       c=col, alpha=0.6 if col != 'red' else 0.15, s=12 if col != 'red' else 5,
                       label=lbl, zorder=2)
    if not gaia_cmd:
        ax.invert_yaxis()
        ax.set_xlabel('g − r (DES, mag)'); ax.set_ylabel(f'r_{L} (mag)')
        ax.set_title(f'{L} Color-Magnitude Diagram'); ax.legend(fontsize=7)

    # 4. DELVE vs HST CMD (r vs r − HST)
    ax = axes[1, 1]
    _clbl = color_hst_label or f'r_{L} − HST'
    _ycol = 'gaia_gmag' if (gaia_cmd and 'gaia_gmag' in matched_df.columns) else 'mag'
    _ylbl = 'G (Gaia, mag)' if _ycol == 'gaia_gmag' else f'{L} mag'
    for src, col, lbl in [(rejected_df, 'red', 'Rejected'),
                          (m_nonstars, 'orange', 'Matched non-star'),
                          (m_stars, 'blue', 'Matched star')]:
        if len(src):
            yv = src[_ycol] if _ycol in src.columns else src['mag']
            ax.scatter(color_hst_sign * src['color_hst'], yv, c=col,
                       alpha=0.6 if col != 'red' else 0.15,
                       s=12 if col != 'red' else 5, label=lbl)
    ax.invert_yaxis()
    ax.set_xlabel(f'{_clbl} (mag)'); ax.set_ylabel(_ylbl)
    ax.set_title(f'HST × {L} Color-Magnitude'); ax.legend(fontsize=7)

    # 5. XY residuals
    ax = axes[2, 0]
    if len(rejected_df):
        ax.scatter(rejected_df['dx'], rejected_df['dy'], c='red', s=8, alpha=0.2)
    _scatter_matched(ax, 'dx', 'dy')
    ax.axhline(0, color='black', ls='--', alpha=0.5); ax.axvline(0, color='black', ls='--', alpha=0.5)
    if len(matched_df):
        lim = max(matched_df['dx'].abs().max(), matched_df['dy'].abs().max()) * 2.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('dX (pixels)'); ax.set_ylabel('dY (pixels)'); ax.set_title('XY Residuals')

    # 6. Normalised residuals
    ax = axes[2, 1]
    if len(rejected_df):
        sx, sy = np.sqrt(rejected_df['cxx']), np.sqrt(rejected_df['cyy'])
        ax.scatter(rejected_df['dx'] / sx, rejected_df['dy'] / sy, c='red', alpha=0.15, s=8)
    for sub, col in [(m_nonstars, 'orange'), (m_stars, 'blue')]:
        if len(sub):
            sx, sy = np.sqrt(sub['cxx']), np.sqrt(sub['cyy'])
            ax.scatter(sub['dx'] / sx, sub['dy'] / sy, c=col, alpha=0.5, s=15)
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False, ls='--', alpha=0.5))
    ax.add_artist(plt.Circle((0, 0), 5, color='red', fill=False, ls=':', alpha=0.5))
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_xlabel('dX / sigma_x'); ax.set_ylabel('dY / sigma_y'); ax.set_title('Normalized Residuals')

    # 7. Residual size vs r magnitude
    ax = axes[3, 0]
    if len(rejected_df):
        ax.scatter(rejected_df['mag'], np.sqrt(rejected_df['dx']**2 + rejected_df['dy']**2),
                   c='red', s=5, alpha=0.15, label='Rejected')
    for sub, col, lbl in [(m_nonstars, 'orange', 'Non-star'), (m_stars, 'blue', 'Star')]:
        if len(sub):
            ax.scatter(sub['mag'], np.sqrt(sub['dx']**2 + sub['dy']**2),
                       c=col, s=10, alpha=0.5, label=lbl)
    ax.set_yscale('log')
    ax.set_xlabel(f'r_{L} Magnitude'); ax.set_ylabel('Residual Size (pixels)')
    ax.set_title(f'Residual Magnitude vs {L} r Mag'); ax.legend(fontsize=7)
    if mag_lims: ax.set_xlim(*mag_lims)

    # 8. Sigma histogram
    ax = axes[3, 1]
    bins = np.linspace(0, 10, 50)
    if len(m_nonstars): ax.hist(m_nonstars['sigma'], bins=bins, color='orange', alpha=0.5, label='Non-star')
    if len(m_stars):    ax.hist(m_stars['sigma'],    bins=bins, color='blue',   alpha=0.6, label='Star')
    if len(rejected_df):
        near = rejected_df[rejected_df['sigma'] < 10]
        ax.hist(near['sigma'], bins=bins, color='red', alpha=0.3, label='Rejected (<10σ)')
    ax.axvline(5, color='red', ls='--')
    ax.set_yscale('log'); ax.set_xlabel('Sigma'); ax.set_ylabel('Count (Log)')
    ax.set_title('Sigma Distribution'); ax.legend(fontsize=7)

    # 9. Sigma vs r magnitude
    ax = axes[4, 0]
    if len(rejected_df):
        near = rejected_df[rejected_df['sigma'] < 15]
        ax.scatter(near['mag'], near['sigma'], c='red', s=5, alpha=0.15, label='Rejected (<15σ)')
    for sub, col, lbl in [(m_nonstars, 'orange', 'Non-star'), (m_stars, 'blue', 'Star')]:
        if len(sub):
            ax.scatter(sub['mag'], sub['sigma'], c=col, s=10, alpha=0.5, label=lbl)
    ax.axhline(5, color='red', ls='--', label='Threshold (5σ)')
    ax.set_xlabel(f'r_{L} Magnitude'); ax.set_ylabel('Residual Sigma')
    ax.set_title(f'Sigma vs {L} r Magnitude'); ax.legend(fontsize=7)
    if mag_lims: ax.set_xlim(*mag_lims)

    # 10. Color-color (g−r vs r−HST)
    ax = axes[4, 1]
    has_color = 'color' in matched_df.columns
    if has_color:
        if len(rejected_df) and 'color' in rejected_df.columns:
            v = (rejected_df['color'] > -10) & (rejected_df['color'] < 10)
            ax.scatter(rejected_df.loc[v, 'color'], rejected_df.loc[v, 'color_hst'],
                       c='red', alpha=0.15, s=5, label='Rejected')
        for src, col, lbl in [(m_nonstars, 'orange', 'Matched non-star'),
                               (m_stars, 'blue', 'Matched star')]:
            if len(src) and 'color' in src.columns:
                v = (src['color'] > -10) & (src['color'] < 10)
                ax.scatter(src.loc[v, 'color'], src.loc[v, 'color_hst'],
                           c=col, alpha=0.6, s=12, label=lbl)
        ax.invert_yaxis()
        ax.set_xlabel('g − r (DES, mag)'); ax.set_ylabel(f'r_{L} − HST (mag)')
        ax.set_title('Color-Color Diagram'); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'g-r not available', ha='center', va='center')
        ax.set_title('Color-Color Placeholder')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, f'diagnostic_plots_{label}{out_suffix}.png'), dpi=150)
    plt.close()


def process_single_image_delve(
    hst: dict,
    delve_df: pd.DataFrame,
    hst_pix_floor: float = 0.5,
    label: str = 'delve',
    mag_col: str = 'r_mag',
    out_suffix: str = '',
    gaia_cmd: bool = False,
    color_hst_label: "str | None" = None,
    color_hst_sign: float = 1.0,
    min_matches: int = 3,
    max_mag_diff: float = 5.0,
    scale_sweep: bool = False,
    discovery_max_offset: int = 50,
    use_resid_floor: bool = True,
    sigma_rot_deg: "float | None" = None,
    sigma_scale: "float | None" = None,
    sigma_skew: "float | None" = None,
    init_resid_max: float = 5.0,
) -> None:
    """Cross-match one HST image against the DELVE catalogue.

    Writes matched_delve.csv, diagnostic_plots_delve.png, and
    offset_histogram_delve.png into hst['root'].

    Parameters mirror process_single_image() from cross_match.py.
    Note: max_mag_diff default is 5.0 (larger than Gaia's 3.0) to accommodate
    the larger photometric scatter between DES r and HST magnitudes.
    """
    start_time = time.time()
    image_name   = os.path.basename(hst['flc']).replace('_flc.fits', '')
    log_file     = os.path.join(hst['root'], f'processing_log_{label}{out_suffix}.txt')
    orig_stdout  = sys.stdout
    sys.stdout   = FileLogger(log_file)
    print(f'Starting {label.upper()} cross-match for {image_name}...', file=orig_stdout)

    _sigma_rot = sigma_rot_deg if sigma_rot_deg is not None else SIGMA_ROT_DEG
    _sigma_sc  = sigma_scale   if sigma_scale   is not None else SIGMA_SCALE
    _sigma_sk  = sigma_skew    if sigma_skew    is not None else SIGMA_SKEW

    try:
        print(f'--- {label.upper()} cross-match: {image_name} ---')
        params = get_hst_params(hst['flc'], catalog_file=hst['catalog'])
        if params is None:
            print(f'Finished {image_name}: failed to load HST params.', file=orig_stdout)
            return
        params['min_matches'] = min_matches

        # ── Propagate DELVE positions to HST epoch ────────────────────────────
        ra_prop, dec_prop, Ct = _propagate_delve(delve_df, params['obs_epoch_mjd'])

        # ── Project DELVE sky positions to HST pixel frame ────────────────────
        # DELVE covariance is in mas² (2×2 positional block of the propagated cov).
        # Same pixel-frame projection as Gaia: +X = −RA, then rotate by ORIENTAT.
        dx_deg = rd2x(ra_prop, dec_prop, params['ra_cen'], params['dec_cen'])
        dy_deg = rd2y(ra_prop, dec_prop, params['ra_cen'], params['dec_cen'])
        scale_deg  = params['pixel_scale'] / 3600.0
        mas_to_px  = 1.0 / (params['pixel_scale'] * 1000.0)

        C_pix_dlv = Ct * mas_to_px**2
        C_pix_dlv[:, 0, :] *= -1
        C_pix_dlv[:, :, 0] *= -1
        x_dlv = params['x_cen'] - dx_deg / scale_deg
        y_dlv = params['y_cen'] + dy_deg / scale_deg

        theta_init    = np.radians(-params['orientat'])
        rot_mat       = np.array([[ np.cos(theta_init),  np.sin(theta_init)],
                                   [-np.sin(theta_init),  np.cos(theta_init)]])
        inv_rot_mat   = np.linalg.inv(rot_mat)
        xy_dlv        = (np.einsum('ij,nj->ni', inv_rot_mat,
                                   np.column_stack([x_dlv, y_dlv])
                                   - np.array([params['x_cen'], params['y_cen']]))
                         + np.array([params['x_cen'], params['y_cen']]))
        x_dlv, y_dlv  = xy_dlv[:, 0], xy_dlv[:, 1]
        C_pix_dlv     = np.einsum('ij,njk,lk->nil', inv_rot_mat, C_pix_dlv, inv_rot_mat)
        dlv_err_total  = np.power(np.linalg.det(C_pix_dlv), 0.25)

        # ── Field filter ──────────────────────────────────────────────────────
        margin = 3000
        in_fld = ((np.abs(x_dlv - params['x_cen']) <= margin) &
                  (np.abs(y_dlv - params['y_cen']) <= margin))
        if not np.any(in_fld):
            print(f'Finished {image_name}: no {label.upper()} sources in field.', file=orig_stdout)
            return

        x_d_in  = x_dlv[in_fld];    y_d_in  = y_dlv[in_fld]
        C_d_in  = C_pix_dlv[in_fld]; err_d_in = dlv_err_total[in_fld]
        mag_d_in  = delve_df[mag_col].values[in_fld]
        gmag_d_in = (delve_df['g_mag'].values[in_fld]
                     if 'g_mag' in delve_df.columns
                     else np.full(int(in_fld.sum()), np.nan))
        imag_d_in = (delve_df['i_mag'].values[in_fld]
                     if 'i_mag' in delve_df.columns
                     else np.full(int(in_fld.sum()), np.nan))
        zmag_d_in = (delve_df['z_mag'].values[in_fld]
                     if 'z_mag' in delve_df.columns
                     else np.full(int(in_fld.sum()), np.nan))
        pmra_d_in  = delve_df['pmra'].values[in_fld]
        pmdec_d_in = delve_df['pmdec'].values[in_fld]
        ra_d_in    = ra_prop[in_fld];  dec_d_in = dec_prop[in_fld]
        all_have_pms = np.ones(int(in_fld.sum()), dtype=bool)  # always True for DELVE

        def _col_in(col, default=np.nan):
            arr = (delve_df[col].values[in_fld]
                   if col in delve_df.columns
                   else np.full(int(in_fld.sum()), default))
            return arr.astype(float)

        pmra_error_d_in  = _col_in('pmra_error')
        pmdec_error_d_in = _col_in('pmdec_error')
        parallax_d_in    = _col_in('parallax')
        parallax_error_d_in  = _col_in('parallax_error')
        ra_error_d_in    = _col_in('ra_error')
        dec_error_d_in   = _col_in('dec_error')
        # Catalog positions at reference epoch (2016.0) — for Gaia-DELVE 5D comparison
        ra_cat_d_in  = _col_in('ra')
        dec_cat_d_in = _col_in('dec')
        # All 10 DELVE correlation terms for the full 5×5 covariance
        # DELVE convention: indices (ra=0, dec=1, plx=2, pmra=3, pmdec=4)
        corr_ra_dec_d_in     = _col_in('ra_dec_corr',          0.0)
        corr_ra_plx_d_in     = _col_in('ra_parallax_corr',     0.0)
        corr_ra_pmra_d_in    = _col_in('ra_pmra_corr',         0.0)
        corr_ra_pmdec_d_in   = _col_in('ra_pmdec_corr',        0.0)
        corr_dec_plx_d_in    = _col_in('dec_parallax_corr',    0.0)
        corr_dec_pmra_d_in   = _col_in('dec_pmra_corr',        0.0)
        corr_dec_pmdec_d_in  = _col_in('dec_pmdec_corr',       0.0)
        corr_plx_pmra_d_in   = _col_in('parallax_pmra_corr',   0.0)
        corr_plx_pmdec_d_in  = _col_in('parallax_pmdec_corr',  0.0)
        corr_pmra_pmdec_d_in = _col_in('pmra_pmdec_corr',      0.0)

        # Build the in-field population DataFrame for diagnostic plot backgrounds
        delve_field_df = pd.DataFrame({
            'x':     x_d_in,
            'y':     y_d_in,
            'pmra':  pmra_d_in,
            'pmdec': pmdec_d_in,
            'r_mag': mag_d_in,
            'g_mag': gmag_d_in,
        })

        # ── Load HST catalog ──────────────────────────────────────────────────
        hst_cat = fits.getdata(hst['catalog'])
        if 'is_star_candidate' not in hst_cat.dtype.names:
            print(f'Finished {image_name}: missing is_star_candidate — skipped.',
                  file=orig_stdout)
            return

        _orig_idx = np.arange(len(hst_cat))
        _valid = (np.isfinite(hst_cat['x_gdc'].astype(float)) &
                  np.isfinite(hst_cat['y_gdc'].astype(float)))
        if not _valid.all():
            hst_cat   = hst_cat[_valid]
            _orig_idx = _orig_idx[_valid]

        x_hst       = hst_cat['x_gdc'].astype(float)
        y_hst       = hst_cat['y_gdc'].astype(float)
        mag_hst_gdc = hst_cat['mag_gdc'].astype(float)
        mag_hst     = hst_cat['mag_st_gdc'].astype(float)
        mag_err_hst = (hst_cat['mag_err_gdc'].astype(float)
                       if 'mag_err_gdc' in hst_cat.dtype.names else None)
        mag_ab_hst  = (hst_cat['mag_ab'].astype(float)
                       if 'mag_ab' in hst_cat.dtype.names else None)
        is_star     = hst_cat['is_star_candidate'].astype(bool)
        C_pix_hst   = np.zeros((len(x_hst), 2, 2))
        C_pix_hst[:, 0, 0] = hst_cat['cov_xx_gdc'].astype(float) + hst_pix_floor**2
        C_pix_hst[:, 1, 1] = hst_cat['cov_yy_gdc'].astype(float) + hst_pix_floor**2
        C_pix_hst[:, 0, 1] = hst_cat['cov_xy_gdc'].astype(float)
        C_pix_hst[:, 1, 0] = C_pix_hst[:, 0, 1]

        n_stars = is_star.sum()
        print(f'  HST: {len(x_hst)} sources, {n_stars} star candidates; '
              f'DELVE in field: {in_fld.sum():,}')

        # Scale-adjusted guess positions for seed
        inv_sc = 1.0 / params['initial_scale']
        xd_guess = params['x_cen'] + (x_d_in - params['x_cen']) * inv_sc
        yd_guess = params['y_cen'] + (y_d_in - params['y_cen']) * inv_sc

        delve_field = {
            'x': x_d_in, 'y': y_d_in, 'C': C_d_in, 'mag': mag_d_in,
            'err': err_d_in, 'has_pms': all_have_pms,
            'xguess': xd_guess, 'yguess': yd_guess,
        }

        star_indices  = np.where(is_star)[0]
        hst_data_star = {
            'x': x_hst[is_star], 'y': y_hst[is_star], 'mag': mag_hst[is_star],
            'C': C_pix_hst[is_star],
            'qfit': hst_cat['qfit'].astype(float)[is_star],
            'chi2': hst_cat['chi2'].astype(float)[is_star],
        }
        hst_data_all_disc = {
            'x': x_hst, 'y': y_hst, 'mag': mag_hst, 'C': C_pix_hst,
            'qfit': hst_cat['qfit'].astype(float),
            'chi2': hst_cat['chi2'].astype(float),
        }
        hst_data_all = {'x': x_hst, 'y': y_hst, 'mag': mag_hst, 'C': C_pix_hst}
        tree_delve   = KDTree(np.column_stack([x_d_in, y_d_in]))

        # ── 4P Discovery (two tiers — no Gaia solution-type split) ───────────
        # DELVE has no "clean 5p" concept; all sources have PM fits.
        # Two tiers: HST star candidates first, then all HST sources as fallback.
        _all_dlv = np.ones(len(x_d_in), dtype=bool)
        _disc_tiers = [
            ('all DELVE / HST stars', _all_dlv, hst_data_star,    True),
            ('all DELVE / all HST',   _all_dlv, hst_data_all_disc, False),
        ]
        best, used_tier, _stars_only = None, None, True
        for _label, _seed_mask, _hst_d, _so in _disc_tiers:
            print(f'  Trying 4P discovery [{_label}] '
                  f'({_seed_mask.sum()} DELVE, {len(_hst_d["x"])} HST)...')
            best = _run_4p_discovery(
                _hst_d, delve_field, params, max_mag_diff,
                scale_sweep=scale_sweep,
                discovery_max_offset=discovery_max_offset,
                seed_quality_mask=_seed_mask,
                debug_verbose=True,
                sigma_rot_deg=_sigma_rot,
                sigma_scale=_sigma_sc,
            )
            if best is not None:
                used_tier, _stars_only = _label, _so
                break
            print(f'  4P failed [{_label}] — trying next tier...')

        if best is None:
            print(f'Finished {image_name}: {label.upper()} 4P discovery failed.', file=orig_stdout)
            return
        print(f'  4P succeeded [{used_tier}]: Q<{best["q"]}, Mag<{best["m"]:.1f} '
              f'({best["n_match"]} matches)')

        # Write offset histogram directly to the DELVE-specific filename
        # (avoids touching offset_histogram.png which belongs to the Gaia match)
        if best.get('offset_hist') is not None:
            ds_str = f"ds={best.get('best_ds', 0.0):+.4f}"
            title  = (f'{image_name}  |  DELVE  best tier q<{best["q"]} '
                      f'm<{best["m"]:.1f}  {ds_str}')
            _plot_offset_histogram(
                best['offset_hist'], best['offset_xed'], best['offset_yed'],
                best.get('offset_peaks', []), title,
                os.path.join(hst['root'], f'offset_histogram_{label}{out_suffix}.png'),
            )

        # ── Affine refinement (all sources) ───────────────────────────────────
        if _stars_only:
            best_all = {**best, 'h_v': star_indices[best['h_v']]}
        else:
            best_all = best

        A, B, C, D, xs_o, ys_o, xt_o, yt_o, C_params, resid_cov, zp, h_f, g_f, _irx, _iry = \
            _run_affine_refinement(best_all, hst_data_all, delve_field, tree_delve,
                                   max_mag_diff, use_resid_floor=use_resid_floor,
                                   sigma_rot_deg=_sigma_rot, sigma_scale=_sigma_sc,
                                   sigma_skew=_sigma_sk)

        if max(_irx, _iry) > init_resid_max:
            print(f'Finished {image_name}: Init 6P resid too large '
                  f'({_irx:.2f},{_iry:.2f}px) — spurious seed.', file=orig_stdout)
            return

        M = np.array([[A, B], [C, D]])

        # ── Final match pass ──────────────────────────────────────────────────
        xh_g, yh_g = apply_affine(x_hst, y_hst, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
        ds, g_idxs = tree_delve.query(np.column_stack([xh_g, yh_g]), k=5,
                                       distance_upper_bound=100)
        h_idx_all  = np.repeat(np.arange(len(x_hst)), 5)
        valid      = ds.flatten() < 100
        h_v, g_v   = h_idx_all[valid], g_idxs.flatten()[valid]

        dx_v, dy_v = x_d_in[g_v] - xh_g[h_v], y_d_in[g_v] - yh_g[h_v]
        C_proj      = np.einsum('ij,njk,lk->nil', M, C_pix_hst[h_v], M)
        dxh_v, dyh_v = x_hst[h_v] - xs_o, y_hst[h_v] - ys_o
        J            = np.zeros((len(h_v), 2, 6))
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh_v, dyh_v, 1.0
        J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = dxh_v, dyh_v, 1.0
        C_model   = np.einsum('nij,jk,nlk->nil', J, C_params, J)
        C_total   = C_d_in[g_v] + C_proj + C_model + resid_cov

        sigs_v    = compute_mahalanobis(dx_v, dy_v, C_total)
        costs_v   = compute_logprob_cost(dx_v, dy_v, C_total)
        mag_diffs = mag_d_in[g_v] - mag_hst[h_v]
        costs_v  += ((mag_diffs - zp) / 1.0)**2
        costs_v[np.abs(mag_diffs - zp) > max_mag_diff] = np.inf

        all_mdf = (pd.DataFrame({'h': h_v, 'g': g_v, 's': sigs_v, 'c': costs_v,
                                  'dx': dx_v, 'dy': dy_v, 'mag_diff': mag_diffs,
                                  'cxx': C_total[:, 0, 0], 'cyy': C_total[:, 1, 1]})
                   .sort_values('c').drop_duplicates('g'))
        final_mdf = (all_mdf.drop_duplicates('h')
                     [(all_mdf.drop_duplicates('h')['s'] < 5.0) &
                      (np.abs(all_mdf.drop_duplicates('h')['mag_diff'] - zp) < max_mag_diff)])

        h_final, g_final = final_mdf['h'].values, final_mdf['g'].values
        print(f'  Final DELVE matches: {len(h_final)}')
        if len(h_final) == 0:
            print(f'Finished {image_name}: no final {label.upper()} matches.', file=orig_stdout)
            return

        # ── Build diagnostic DataFrame ────────────────────────────────────────
        # in-field DELVE source index → original delve_df index
        in_fld_idx = np.where(in_fld)[0]

        diag_df = pd.DataFrame({
            'h_idx': all_mdf['h'], 'g_idx': all_mdf['g'],
            'x': x_d_in[all_mdf['g']], 'y': y_d_in[all_mdf['g']],
            'hx': xh_g[all_mdf['h']], 'hy': yh_g[all_mdf['h']],
            'ra': ra_d_in[all_mdf['g']], 'dec': dec_d_in[all_mdf['g']],
            'dx': all_mdf['dx'].values, 'dy': all_mdf['dy'].values,
            'sigma': all_mdf['s'].values, 'cxx': all_mdf['cxx'].values,
            'cyy': all_mdf['cyy'].values,
            'mag': mag_d_in[all_mdf['g']],
            'g_mag': gmag_d_in[all_mdf['g'].values],
            'hst_mag': mag_hst[all_mdf['h'].values],
            'pmra': pmra_d_in[all_mdf['g']],
            'pmdec': pmdec_d_in[all_mdf['g']],
            'hst_is_star': is_star[all_mdf['h'].values],
            'gaia_gmag': _col_in('gmag')[all_mdf['g'].values],
            'gaia_bp_rp': _col_in('bp_rp')[all_mdf['g'].values],
        })
        diag_df['color_hst'] = diag_df['mag'] - diag_df['hst_mag'] - zp
        # g−r color (DELVE DES, analogous to Gaia BP-RP)
        diag_df['color'] = diag_df['g_mag'] - diag_df['mag']
        # mask sentinels
        bad = (diag_df['g_mag'] < -90) | (diag_df['g_mag'] > 50)
        diag_df.loc[bad, 'color'] = np.nan

        final_keys = set(zip(h_final, g_final))
        is_m = diag_df.apply(
            lambda r: (int(r.h_idx), int(r.g_idx)) in final_keys, axis=1)

        _save_diagnostic_plots_delve(hst['root'], image_name,
                                      diag_df[is_m], diag_df[~is_m],
                                      delve_field_df=delve_field_df,
                                      label=label, out_suffix=out_suffix,
                                      gaia_cmd=gaia_cmd,
                                      color_hst_label=color_hst_label,
                                      color_hst_sign=color_hst_sign)

        # ── Save matched_delve.csv ────────────────────────────────────────────
        fm = diag_df[is_m]
        dlv_global_idx = in_fld_idx[fm['g_idx'].values]   # index into delve_df

        output = Table()
        output['hst_index']         = _orig_idx[fm['h_idx'].values]
        output['hst_x_gdc']         = x_hst[fm['h_idx'].values]
        output['hst_y_gdc']         = y_hst[fm['h_idx'].values]
        output['hst_mag_gdc']       = mag_hst_gdc[fm['h_idx'].values]
        if mag_err_hst is not None:
            output['hst_mag_err_gdc']   = mag_err_hst[fm['h_idx'].values]
        output['hst_mag_st_gdc']    = mag_hst[fm['h_idx'].values]
        if mag_ab_hst is not None:
            output['hst_mag_ab']    = mag_ab_hst[fm['h_idx'].values]
        output['hst_is_star']       = is_star[fm['h_idx'].values]
        output[f'{label}_source_id']   = delve_df['source_id'].values[dlv_global_idx]
        output[f'{label}_ra_prop']     = ra_d_in[fm['g_idx'].values]
        output[f'{label}_dec_prop']    = dec_d_in[fm['g_idx'].values]
        output[f'{label}_rmag']        = mag_d_in[fm['g_idx'].values]
        output[f'{label}_gmag']        = gmag_d_in[fm['g_idx'].values]
        output[f'{label}_imag']        = imag_d_in[fm['g_idx'].values]
        output[f'{label}_zmag']        = zmag_d_in[fm['g_idx'].values]
        output[f'{label}_pmra']               = pmra_d_in[fm['g_idx'].values]
        output[f'{label}_pmdec']              = pmdec_d_in[fm['g_idx'].values]
        output['delve_pmra_error']         = pmra_error_d_in[fm['g_idx'].values]
        output['delve_pmdec_error']        = pmdec_error_d_in[fm['g_idx'].values]
        output['delve_parallax']           = parallax_d_in[fm['g_idx'].values]
        output['delve_parallax_error']     = parallax_error_d_in[fm['g_idx'].values]
        output['delve_ra_error']           = ra_error_d_in[fm['g_idx'].values]
        output['delve_dec_error']          = dec_error_d_in[fm['g_idx'].values]
        # Catalog position at 2016.0 (same reference epoch as Gaia DR3)
        output['delve_ra_cat']             = ra_cat_d_in[fm['g_idx'].values]
        output['delve_dec_cat']            = dec_cat_d_in[fm['g_idx'].values]
        # Full 5×5 DELVE correlation terms for the Gaia-DELVE consistency test
        output['delve_corr_ra_dec']        = corr_ra_dec_d_in[fm['g_idx'].values]
        output['delve_corr_ra_plx']        = corr_ra_plx_d_in[fm['g_idx'].values]
        output['delve_corr_ra_pmra']       = corr_ra_pmra_d_in[fm['g_idx'].values]
        output['delve_corr_ra_pmdec']      = corr_ra_pmdec_d_in[fm['g_idx'].values]
        output['delve_corr_dec_plx']       = corr_dec_plx_d_in[fm['g_idx'].values]
        output['delve_corr_dec_pmra']      = corr_dec_pmra_d_in[fm['g_idx'].values]
        output['delve_corr_dec_pmdec']     = corr_dec_pmdec_d_in[fm['g_idx'].values]
        output['delve_corr_plx_pmra']      = corr_plx_pmra_d_in[fm['g_idx'].values]
        output['delve_corr_plx_pmdec']     = corr_plx_pmdec_d_in[fm['g_idx'].values]
        output['delve_corr_pmra_pmdec']    = corr_pmra_pmdec_d_in[fm['g_idx'].values]
        if 'mtype' in delve_df.columns:
            output['delve_mtype']   = delve_df['mtype'].values[dlv_global_idx]
        if 'healpix_pixel' in delve_df.columns:
            output['delve_healpix'] = delve_df['healpix_pixel'].values[dlv_global_idx]
        output['residual_mag']      = output[f'{label}_rmag'] - (output['hst_mag_st_gdc'] + zp)
        output['residual_x']        = fm['dx'].values
        output['residual_y']        = fm['dy'].values
        output['residual_sigma']    = fm['sigma'].values

        # Nullify DELVE astrometric data for entries without a complete, valid
        # 5×5 covariance (any sigma NaN or ≤ 0).  Photometry columns are kept.
        _sig_cols = ['delve_ra_error', 'delve_dec_error', 'delve_pmra_error',
                     'delve_pmdec_error', 'delve_parallax_error']
        _astrom_cols = [
            'delve_pmra', 'delve_pmdec', 'delve_pmra_error', 'delve_pmdec_error',
            'delve_parallax', 'delve_parallax_error',
            'delve_ra_error', 'delve_dec_error',
            'delve_ra_cat', 'delve_dec_cat',
            'delve_corr_ra_dec', 'delve_corr_ra_plx', 'delve_corr_ra_pmra',
            'delve_corr_ra_pmdec', 'delve_corr_dec_plx', 'delve_corr_dec_pmra',
            'delve_corr_dec_pmdec', 'delve_corr_plx_pmra', 'delve_corr_plx_pmdec',
            'delve_corr_pmra_pmdec',
        ]
        _valid_5d = np.ones(len(output), dtype=bool)
        for _c in _sig_cols:
            if _c in output.colnames:
                _sig = output[_c].data.astype(float)
                _valid_5d &= np.isfinite(_sig) & (_sig > 0)
        _n_bad = int((~_valid_5d).sum())
        if _n_bad > 0:
            for _c in _astrom_cols:
                if _c in output.colnames:
                    output[_c][~_valid_5d] = np.nan

        out_csv = os.path.join(hst['root'], f'matched_{label}{out_suffix}.csv')
        output.write(out_csv, format='ascii.csv', overwrite=True)

        # Save transformation parameters (mirrors transformation.csv from Gaia match)
        ratio    = np.sqrt(A*D - B*C)
        rot_deg  = np.degrees(np.arctan2(B - C, A + D))
        on_skew  = 0.5 * (A - D)
        off_skew = 0.5 * (B + C)
        trans_out = Table()
        trans_out['parameter'] = ['A', 'B', 'C', 'D', 'xs_o', 'ys_o', 'xt_o', 'yt_o',
                                   'ratio', 'rot_deg', 'on_skew', 'off_skew', 'zp',
                                   'ra_cen', 'dec_cen', 'x_cen', 'y_cen',
                                   'pixel_scale', 'orientat']
        trans_out['value']     = [A, B, C, D, xs_o, ys_o, xt_o, yt_o,
                                   ratio, rot_deg, on_skew, off_skew, zp,
                                   params['ra_cen'], params['dec_cen'],
                                   params['x_cen'], params['y_cen'],
                                   params['pixel_scale'], params['orientat']]
        trans_out.write(os.path.join(hst['root'], 'transformation_delve.csv'),
                        format='ascii.csv', overwrite=True)

        print(f'Finished {image_name}: {len(fm)} DELVE matches in '
              f'{time.time()-start_time:.1f}s.', file=orig_stdout)

    except Exception as exc:
        import traceback
        print(f'Finished {image_name}: Error — {exc}', file=orig_stdout)
        traceback.print_exc()
    finally:
        sys.stdout = orig_stdout
