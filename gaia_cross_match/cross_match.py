import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from .miracle_match import miracle_match, rd2x, rd2y
from .catalog_matcher import fit_affine_weighted, fit_4p_weighted, apply_affine, compute_mahalanobis, compute_logprob_cost, find_offset, find_scale_and_offset
from bp3m.instrument_config import get_instrument_config, SIGMA_ROT_DEG, SIGMA_SCALE, SIGMA_SKEW
from bp3m.astro_utils import (GAIA_SYS_DICT, get_tele_position,
                              get_parallax_factors, propagate_gaia_positions)

def load_gaia_data(target, data_dir):
    gaia_path = os.path.join(data_dir, target, "Gaia", "*_gaia.csv")
    gaia_files = glob.glob(gaia_path)
    if not gaia_files:
        print(f"No Gaia CSV files found in {gaia_path}"); return None
    print(f"Reading {len(gaia_files)} Gaia CSV files...")
    # Normalise SOURCE_ID → source_id per-file BEFORE concat so that pandas
    # never fills the source_id column with NaN (which would promote int64→float64
    # and silently corrupt Gaia source IDs through floating-point rounding).
    df_list = []
    for f in gaia_files:
        dfi = pd.read_csv(f, dtype={'source_id': np.int64, 'SOURCE_ID': np.int64})
        if 'SOURCE_ID' in dfi.columns and 'source_id' not in dfi.columns:
            dfi = dfi.rename(columns={'SOURCE_ID': 'source_id'})
        df_list.append(dfi)
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates(subset=["source_id"])
    if 'bp_rp' not in df.columns and 'phot_bp_mean_mag' in df.columns and 'phot_rp_mean_mag' in df.columns:
        df['bp_rp'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
    mask = np.isfinite(df['ra']) & np.isfinite(df['dec']) & np.isfinite(df['gmag'])
    if 'bp_rp' in df.columns: mask &= np.isfinite(df['bp_rp'])
    return df[mask]

def find_hst_image_folders(target, data_dir):
    hst_root = os.path.join(data_dir, target, "HST")
    folders = []
    for root, dirs, files in os.walk(hst_root):
        image_name = root.split('/')[-1]
        cat_fname = f"{image_name}_flc_catalog.fits"
        if cat_fname in files:
            flc_files = glob.glob(os.path.join(root, "*_flc.fits"))
            if flc_files:
                folders.append({"root": root, "catalog": os.path.join(root, cat_fname), "flc": flc_files[0]})
    return folders

# Science and DQ extension pairs, and PSF-grid y-offsets, per chip
# (sci_ext, dq_ext, y_offset_for_psf_grid)
_CHIP_CONFIG = {
    ('ACS',  'WFC'):  [(1, 3, 0.0), (4, 6, 2048.0)],
    ('ACS',  'HRC'):  [(1, 2, 0.0)],
    ('ACS',  'SBC'):  [(1, 2, 0.0)],
    ('WFC3', 'UVIS'): [(1, 3, 0.0), (4, 6, 2048.0)],
    ('WFC3', 'IR'):   [(1, 2, 0.0)],
}


def get_chip_config(instrume, detector):
    """Return per-chip configuration for a two-chip instrument.

    Parameters
    ----------
    instrume : str  e.g. 'ACS'
    detector : str  e.g. 'WFC'

    Returns
    -------
    list of (sci_ext, dq_ext, y_offset) tuples, one per chip.
    y_offset is added to image-y to get detector-y for PSF grid lookup.
    """
    key = (instrume.strip().upper(), detector.strip().upper())
    config = _CHIP_CONFIG.get(key)
    if config is None:
        warnings.warn(f"Unknown instrument/detector {key}; assuming single chip at ext 1.")
        return [(1, 2, 0.0)]
    return config


ORIENTAT_FIT_MAX_DEG = 1.0      # never move the orientation further than this from ORIENTAT


def _orientat_in_gdc_frame(catalog_file, ra_cen, dec_cen, x_cen, y_cen, pixel_scale, orientat):
    """Orientation (deg) of the GDC-corrected frame, fitted from the catalogue's header-WCS RA/Dec and
    x_gdc/y_gdc so that the header-only guess has zero rotation.  None if it cannot be determined."""
    t = fits.getdata(catalog_file)
    ra = np.asarray(t['ra'], float); de = np.asarray(t['dec'], float)
    xg = np.asarray(t['x_gdc'], float); yg = np.asarray(t['y_gdc'], float)
    ok = np.isfinite(ra) & np.isfinite(de) & np.isfinite(xg) & np.isfinite(yg)
    if ok.sum() < 10:
        return None
    idx = np.where(ok)[0]
    if len(idx) > 3000:
        idx = idx[np.linspace(0, len(idx) - 1, 3000).astype(int)]
    ra, de, xg, yg = ra[idx], de[idx], xg[idx], yg[idx]
    sdeg = pixel_scale / 3600.0
    dx = rd2x(ra, de, ra_cen, dec_cen); dy = rd2y(ra, de, ra_cen, dec_cen)
    u = -dx / sdeg; v = dy / sdeg                      # sky offsets in unrotated pixel units

    def _rot_for(ori):
        th = np.radians(-ori)
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        g = np.column_stack([u, v]) @ np.linalg.inv(R).T      # header-frame coordinates (minus centre)
        X = np.column_stack([xg - xg.mean(), yg - yg.mean(), np.ones_like(xg)])
        a, b, _ = np.linalg.lstsq(X, g[:, 0], rcond=None)[0]
        c, d, _ = np.linalg.lstsq(X, g[:, 1], rcond=None)[0]
        return float(np.degrees(np.arctan2(b - c, a + d)))

    r0 = _rot_for(orientat)
    best = min((orientat + r0, orientat - r0), key=lambda o: abs(_rot_for(o)))
    if abs(best - orientat) > ORIENTAT_FIT_MAX_DEG or abs(_rot_for(best)) > 0.1 * max(abs(r0), 1e-3) + 1e-3:
        return None
    return float(best)


def get_hst_params(flc_file, catalog_file=None):
    with fits.open(flc_file) as hdul:
        header0 = hdul[0].header
        instrument, detector = header0.get('INSTRUME', ''), header0.get('DETECTOR', '')
        config = _CHIP_CONFIG.get((instrument.upper(), detector.upper()))

        sci_hdrs = {h.header.get('EXTVER', 1): h.header for h in hdul if h.name == 'SCI'}
        if not sci_hdrs and len(hdul) > 1: sci_hdrs = {1: hdul[1].header}
        if not sci_hdrs: return None
        ext_header = list(sci_hdrs.values())[0]
        naxis1, naxis2 = ext_header.get('NAXIS1', 4096), ext_header.get('NAXIS2', 2048)

        # Use the primary chip (y_offset=0 in _CHIP_CONFIG) as the WCS reference.
        # Each chip has its own CRVAL tangent point; averaging CRVAL and CRPIX across
        # chips produces a (ra_cen, dec_cen) that does not correspond to (x_cen, y_cen)
        # in the GDC frame, introducing a systematic positional offset.
        primary_extver = sorted(sci_hdrs.keys())[0]  # default: first chip
        primary_y_offset = 0.0
        if config is not None:
            for extver_idx, (_, _, y_off) in enumerate(config):
                if y_off == 0.0:
                    primary_extver = extver_idx + 1  # EXTVER is 1-based
                    primary_y_offset = 0.0
                    break
        primary_hdr = sci_hdrs.get(primary_extver, ext_header)
        ra_cen  = primary_hdr.get('CRVAL1', 0.0)
        dec_cen = primary_hdr.get('CRVAL2', 0.0)
        x_cen   = primary_hdr.get('CRPIX1', naxis1 / 2.0)
        y_cen   = primary_hdr.get('CRPIX2', naxis2 / 2.0) + primary_y_offset
        orientat = primary_hdr.get('ORIENTAT', 0.0)
        _icfg = get_instrument_config(instrument, detector)
        pixel_scale   = _icfg["pixel_scale"]
        # scale prior centre = instrument constant x this exposure's velocity-aberration factor (as bp3m.data_loader_flc)
        try:
            _vaf = float(primary_hdr.get('VAFACTOR', header0.get('VAFACTOR', 1.0)) or 1.0)
        except (TypeError, ValueError):
            _vaf = 1.0
        if not np.isfinite(_vaf) or abs(_vaf - 1.0) > 5e-4:
            _vaf = 1.0
        initial_scale = _icfg["initial_scale"] * _vaf

        expstart = header0.get('EXPSTART', 51544); obs_epoch_mjd = expstart

    # When catalogs contain CHIP{ext}_CRPIX1_GDC / CHIP{ext}_CRPIX2_GDC keys,
    # override (ra/dec/x/y)_cen with those GDC-corrected positions averaged across
    # chips.  Keys without the _GDC suffix are raw (uncorrected) and must not be used.
    if catalog_file is not None:
        try:
            with fits.open(catalog_file) as cat_hdul:
                cat_hdr = cat_hdul[1].header
                prefixes = sorted({k.split('_CRPIX1_GDC')[0]
                                   for k in cat_hdr.keys()
                                   if k.endswith('_CRPIX1_GDC') and k.startswith('CHIP')})
                x_vals, y_vals, ra_vals, dec_vals = [], [], [], []
                for pfx in prefixes:
                    cx  = cat_hdr.get(f'{pfx}_CRPIX1_GDC')
                    cy  = cat_hdr.get(f'{pfx}_CRPIX2_GDC')
                    ra  = cat_hdr.get(f'{pfx}_CRVAL1')
                    dec = cat_hdr.get(f'{pfx}_CRVAL2')
                    if all(v is not None for v in [cx, cy, ra, dec]):
                        x_vals.append(float(cx));  y_vals.append(float(cy))
                        ra_vals.append(float(ra)); dec_vals.append(float(dec))
                if x_vals:
                    x_cen, y_cen     = np.mean(x_vals),  np.mean(y_vals)
                    ra_cen, dec_cen  = np.mean(ra_vals), np.mean(dec_vals)
        except Exception:
            pass

    # ORIENTAT describes the RAW pixel frame at the image's reference pixel; the matching frame is the
    # GDC-corrected one.  Their rotation differs by the local distortion rotation, which is negligible
    # near the chip centre but 0.2-0.35 deg at the corner reference pixel of a subarray (UVIS2-C512C,
    # -C1K1C, -2K2C: solutions came out 0.21-0.33 deg from ORIENTAT and failed the 4P |rot| < 0.2 deg
    # gate).  The pypass catalogue holds RA/Dec from the FULL header WCS (incl. its distortion terms)
    # and the GDC positions of every detection, so the orientation of the GDC frame is fitted here
    # (2026-10-01; predicts the measured offsets to <= 0.024 deg, full frames to 0.001 deg).  Positions
    # are never corrected with the header distortion -- it only sets this initial orientation.
    orientat_header = orientat
    if catalog_file is not None:
        try:
            _ori = _orientat_in_gdc_frame(catalog_file, ra_cen, dec_cen, x_cen, y_cen, pixel_scale, orientat)
            if _ori is not None:
                orientat = _ori
        except Exception:
            pass

    return {"ra_cen": ra_cen, "dec_cen": dec_cen, "x_cen": x_cen, "y_cen": y_cen,
            "pixel_scale": pixel_scale, "initial_scale": initial_scale,
            "obs_epoch_mjd": obs_epoch_mjd, "orientat": orientat, "orientat_header": orientat_header,
            "naxis1": naxis1, "naxis2": naxis2,
            "instrument": instrument, "detector": detector,
            "chip_dims": {ext: (h.get('NAXIS1'), h.get('NAXIS2')) for ext, h in sci_hdrs.items()}}

def construct_gaia_cov(df, zero_pm=False, fill_pm_plx=None):
    n = len(df)
    errors = np.zeros((n, 5))
    errors[:, 0], errors[:, 1] = df['ra_error'].values, df['dec_error'].values

    if zero_pm:
        errors[:, 2] = 20.0 # 20 mas
        errors[:, 3], errors[:, 4] = 100.0, 100.0 # 100 mas/yr
    else:
        # 2p stars: the error fill sets the effective SEARCH WINDOW after
        # propagation.  The legacy 100 mas/yr made every candidate within
        # ~2 px look consistent, so in dense high-PM fields mismatches
        # survived regardless of where the prediction was centred.  With a
        # field-typical fill available, use 2x the field PM dispersion about
        # the mode instead — a wrong candidate 75 mas off-track becomes a
        # multi-sigma outlier and the star is left unmatched rather than
        # poisoned.
        if fill_pm_plx is not None and "pm_sig" in fill_pm_plx:
            # Width = 2x the field dispersion, combined in quadrature with
            # 0.4x the mode amplitude: guarantees a genuine ZERO-PM star is
            # never more than ~2.5 sigma from the mode-propagated prediction,
            # even in kinematically cold high-PM fields where the clipped
            # dispersion alone would hard-reject real non-members.  The
            # one-to-one cost competition + magnitude term (which did the
            # actual phantom rejection in Omega Cen) arbitrate within the
            # window.
            _mode_amp = float(np.hypot(fill_pm_plx.get("pmra", 0.0),
                                       fill_pm_plx.get("pmdec", 0.0)))
            _pm_fill = float(np.hypot(2.0 * fill_pm_plx["pm_sig"],
                                      0.4 * _mode_amp))
            _plx_fill = max(float(np.hypot(2.0 * fill_pm_plx.get("plx_sig", 10.0),
                                           0.4 * abs(fill_pm_plx.get("plx", 0.0)))),
                            0.1)
        else:
            _pm_fill, _plx_fill = 100.0, 20.0
        errors[:, 2] = df['parallax_error'].fillna(_plx_fill).values
        errors[:, 3], errors[:, 4] = df['pmra_error'].fillna(_pm_fill).values, df['pmdec_error'].fillna(_pm_fill).values

    corrs = {(0, 1): 'ra_dec_corr', (0, 2): 'ra_parallax_corr', (0, 3): 'ra_pmra_corr', (0, 4): 'ra_pmdec_corr',
             (1, 2): 'dec_parallax_corr', (1, 3): 'dec_pmra_corr', (1, 4): 'dec_pmdec_corr',
             (2, 3): 'parallax_pmra_corr', (2, 4): 'parallax_pmdec_corr', (3, 4): 'pmra_pmdec_corr'}
    covs = np.zeros((n, 5, 5))
    for i in range(5): covs[:, i, i] = errors[:, i]**2

    if not zero_pm:
        for (i, j), col in corrs.items():
            if col in df.columns:
                val = df[col].fillna(0.0).values
                c = val * errors[:, i] * errors[:, j]
                covs[:, i, j] = c; covs[:, j, i] = c

    gaia_6p = np.isfinite(df['pseudocolour'])
    gaia_5p = np.isfinite(df['pmra']) & ~gaia_6p
    gaia_2p = np.isfinite(df['ra']) & ~gaia_5p & ~gaia_6p

    #inflate Gaia covs according to literature (mult_* are sigma multipliers)
    covs[gaia_6p] *= GAIA_SYS_DICT['mult_6p']**2
    covs[gaia_5p] *= GAIA_SYS_DICT['mult_5p']**2
    covs[gaia_2p] *= GAIA_SYS_DICT['mult_2p']**2

    #add Gaia systematics according to literature (values in GAIA_SYS_DICT,
    #from E. Vasiliev and H. Baumgardt 2021, MNRAS 505, 5978-6002).
    #Ordering here is the Gaia-archive one: (ra, dec, parallax, pmra, pmdec).
    covs += np.diag(np.array([0, 0, GAIA_SYS_DICT['parallax_sys_err'],
                              GAIA_SYS_DICT['pm_sys_err'], GAIA_SYS_DICT['pm_sys_err']])**2)

    return covs

def propagate_gaia_with_cov(df, target_mjd, zero_pm=False, fill_pm_plx=None):
    ref_epoch = df['ref_epoch'].iloc[0] if 'ref_epoch' in df.columns else 2016.0
    t_hst = Time(target_mjd, format='mjd')
    dt = (t_hst.jyear - ref_epoch)
    n = len(df)
    ra, dec = np.radians(df['ra'].values), np.radians(df['dec'].values)

    if zero_pm:
        plx = np.zeros(n)
        pmra, pmdec = np.zeros(n), np.zeros(n)
    else:
        # Stars without their own solution (2p) are propagated at the
        # field-typical astrometry when provided — NOT at 0 PM, which in
        # dense high-PM fields matches them to the wrong source (the one
        # nearest the un-propagated position).
        _f = fill_pm_plx or {"pmra": 0.0, "pmdec": 0.0, "plx": 0.0}
        plx = df['parallax'].fillna(_f["plx"]).values
        pmra  = df['pmra'].fillna(_f["pmra"]).values
        pmdec = df['pmdec'].fillna(_f["pmdec"]).values

    # Canonical propagation + parallax factors from bp3m.astro_utils — do NOT
    # re-derive this physics inline (see propagate_gaia_positions docstring).
    with solar_system_ephemeris.set('builtin'):
        tele_xyz = get_tele_position(t_hst, curr_id='earth')
    p_ra_cosdec, p_dec = get_parallax_factors(
        df['ra'].values, df['dec'].values, tele_xyz)
    ra_prop, dec_prop = propagate_gaia_positions(
        df['ra'].values, df['dec'].values, pmra, pmdec, plx, dt, tele_xyz)
    C0 = construct_gaia_cov(df, zero_pm=zero_pm, fill_pm_plx=fill_pm_plx)
    J = np.zeros((n, 2, 5))
    J[:, 0, 0], J[:, 0, 2], J[:, 0, 3] = 1.0, p_ra_cosdec, dt
    J[:, 1, 1], J[:, 1, 2], J[:, 1, 4] = 1.0, p_dec, dt
    Ct = np.einsum('nij,njk,nlk->nil', J, C0, J)
    return ra_prop, dec_prop, Ct

def project_gaia_cov_to_pixel(Ct, ra, dec, params):
    """Projects sky-frame covariance (mas²) into 2x2 instrument pixel-frame covariance."""
    n = len(ra)
    mas_to_px = 1.0 / (params['pixel_scale'] * 1000.0)
    theta_init = np.radians(-params['orientat'])

    J_proj = np.zeros((n, 2, 2))
    J_proj[:, 0, 0] =  np.cos(theta_init) * mas_to_px
    J_proj[:, 0, 1] = -np.sin(theta_init) * mas_to_px
    J_proj[:, 1, 0] =  np.sin(theta_init) * mas_to_px
    J_proj[:, 1, 1] =  np.cos(theta_init) * mas_to_px

    C_pix = np.einsum('nij,njk,nlk->nil', J_proj, Ct, J_proj)
    return C_pix

def save_diagnostic_plots(out_dir, image_name, matched_df, rejected_df):
    """Generates diagnostic plots.

    Colour scheme:
      blue   — matched star candidates (hst_is_star == True)
      orange — matched non-star sources (hst_is_star == False)
      red    — rejected / unmatched
    """
    fig, axes = plt.subplots(5, 2, figsize=(14, 24/4*5))
    fig.suptitle(f"Match Diagnostics: {image_name}", fontsize=18)
    all_df = pd.concat([matched_df, rejected_df])

    has_star_col = 'hst_is_star' in matched_df.columns
    if has_star_col:
        m_stars  = matched_df[matched_df['hst_is_star'].astype(bool)]
        m_nonstars = matched_df[~matched_df['hst_is_star'].astype(bool)]
    else:
        m_stars, m_nonstars = matched_df, matched_df.iloc[0:0]

    def _scatter_matched(ax, col_x, col_y, **kwargs):
        if len(m_nonstars) > 0:
            ax.scatter(m_nonstars[col_x], m_nonstars[col_y],
                       c='orange', alpha=0.6, s=12, label='Matched non-star', **kwargs)
        if len(m_stars) > 0:
            ax.scatter(m_stars[col_x], m_stars[col_y],
                       c='blue', alpha=0.6, s=10, label='Matched star', **kwargs)

    # 1. Pixel Positions
    ax = axes[0, 0]
    lines_px = [[(r.x, r.y), (r.hx, r.hy)] for r in all_df.itertuples()]
    ax.add_collection(LineCollection(lines_px, colors='grey', alpha=0.1, linewidths=0.5, zorder=1))
    ax.scatter(all_df['hx'], all_df['hy'], c='grey', s=2, alpha=0.3, zorder=2)
    if len(rejected_df) > 0:
        ax.scatter(rejected_df['x'], rejected_df['y'], c='red', alpha=0.3, s=5, label='Rejected Gaia', zorder=3)
    if len(m_nonstars) > 0:
        ax.scatter(m_nonstars['x'], m_nonstars['y'], c='orange', alpha=0.6, s=12, label='Matched non-star', zorder=4)
    if len(m_stars) > 0:
        ax.scatter(m_stars['x'], m_stars['y'], c='blue', alpha=0.6, s=10, label='Matched star', zorder=5)
    if len(matched_df) > 0:
        x_m, y_m = matched_df['x'], matched_df['y']
        px_p, py_p = (x_m.max()-x_m.min())*0.05, (y_m.max()-y_m.min())*0.05
        ax.set_xlim(x_m.min()-px_p, x_m.max()+px_p); ax.set_ylim(y_m.min()-py_p, y_m.max()+py_p)
    ax.set_xlabel("X_Gaia (pixels)"); ax.set_ylabel("Y_Gaia (pixels)"); ax.set_title("Field Map (Pixels)"); ax.legend(fontsize=7)

    # 2. Gaia CMD (G vs BP-RP)
    ax = axes[1, 0]
    if 'color' in matched_df.columns:
        if len(rejected_df) > 0:
            ax.scatter(rejected_df['color'], rejected_df['mag'], c='red', alpha=0.15, s=5, label='Rejected')
        _scatter_matched(ax, 'color', 'mag')
        ax.invert_yaxis(); ax.set_xlabel("BP - RP (mag)"); ax.set_ylabel("Gaia G (mag)"); ax.set_title("Gaia Color-Magnitude Diagram"); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "BP-RP not available", ha='center', va='center'); ax.set_title("CMD Placeholder")

    # 3. Gaia vs HST CMD (G vs G-HST_mag)
    ax = axes[1, 1]
    if len(rejected_df) > 0:
        ax.scatter(rejected_df['color_hst'], rejected_df['mag'], c='red', alpha=0.15, s=5, label='Rejected')
    _scatter_matched(ax, 'color_hst', 'mag')
    ax.invert_yaxis(); ax.set_xlabel("Gaia G - HST (mag)"); ax.set_ylabel("Gaia G (mag)"); ax.set_title("Gaia G - HST Color-Magnitude"); ax.legend(fontsize=7)

    # 4. XY Residual Scatter
    ax = axes[2, 0]
    if len(rejected_df) > 0:
        ax.scatter(rejected_df['dx'], rejected_df['dy'], c='red', alpha=0.2, s=8)
    _scatter_matched(ax, 'dx', 'dy')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5); ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    if len(matched_df) > 0:
        lim = max(matched_df['dx'].abs().max(), matched_df['dy'].abs().max()) * 2.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("dX (pixels)"); ax.set_ylabel("dY (pixels)"); ax.set_title("XY Residuals")

    # 5. Normalized Residuals
    ax = axes[2, 1]
    if len(rejected_df) > 0:
        rsx, rsy = np.sqrt(rejected_df['cxx']), np.sqrt(rejected_df['cyy'])
        ax.scatter(rejected_df['dx']/rsx, rejected_df['dy']/rsy, c='red', alpha=0.15, s=8)
    for sub_df, col in [(m_nonstars, 'orange'), (m_stars, 'blue')]:
        if len(sub_df) > 0:
            sx, sy = np.sqrt(sub_df['cxx']), np.sqrt(sub_df['cyy'])
            ax.scatter(sub_df['dx']/sx, sub_df['dy']/sy, c=col, alpha=0.5, s=15)
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.5))
    ax.add_artist(plt.Circle((0, 0), 5, color='red', fill=False, linestyle=':', alpha=0.5))
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_xlabel("dX / sigma_x"); ax.set_ylabel("dY / sigma_y"); ax.set_title("Normalized Residuals")

    # Calculate shared magnitude limits for Panels 6 and 8
    if len(all_df) > 0:
        mag_min, mag_max = all_df['mag'].min(), all_df['mag'].max()
        mag_pad = (mag_max - mag_min) * 0.05
        mag_lims = (mag_min - mag_pad, mag_max + mag_pad)
    else:
        mag_lims = None

    # 6. Combined Residual vs Gaia Magnitude (Log-Scaled Y)
    ax = axes[3, 0]
    if len(rejected_df) > 0:
        res_r = np.sqrt(rejected_df['dx']**2 + rejected_df['dy']**2)
        ax.scatter(rejected_df['mag'], res_r, c='red', alpha=0.15, s=5, label='Rejected')
    for sub_df, col, lbl in [(m_nonstars, 'orange', 'Non-star'), (m_stars, 'blue', 'Star')]:
        if len(sub_df) > 0:
            res = np.sqrt(sub_df['dx']**2 + sub_df['dy']**2)
            ax.scatter(sub_df['mag'], res, c=col, alpha=0.5, s=10, label=lbl)
    ax.set_yscale('log'); ax.set_xlabel("Gaia G Magnitude"); ax.set_ylabel("Residual Size (pixels)"); ax.set_title("Residual Magnitude vs Gaia Mag"); ax.legend(fontsize=7)
    if mag_lims: ax.set_xlim(mag_lims)

    # 7. Sigma Histogram
    ax = axes[3, 1]
    bins = np.linspace(0, 10, 50)
    if len(m_nonstars) > 0:
        ax.hist(m_nonstars['sigma'], bins=bins, color='orange', alpha=0.5, label='Non-star')
    if len(m_stars) > 0:
        ax.hist(m_stars['sigma'], bins=bins, color='blue', alpha=0.6, label='Star')
    if len(rejected_df) > 0:
        rej_near = rejected_df[rejected_df['sigma'] < 10.0]
        ax.hist(rej_near['sigma'], bins=bins, color='red', alpha=0.3, label='Rejected (<10s)')
    ax.axvline(5, color='red', linestyle='--'); ax.set_yscale('log'); ax.set_xlabel("Sigma"); ax.set_ylabel("Count (Log)"); ax.set_title("Sigma Distribution"); ax.legend(fontsize=7)

    # 8. Residual Sigma vs Gaia Magnitude
    ax = axes[4, 0]
    if len(rejected_df) > 0:
        rej_near = rejected_df[rejected_df['sigma'] < 15.0]
        ax.scatter(rej_near['mag'], rej_near['sigma'], c='red', alpha=0.15, s=5, label='Rejected (<15s)')
    for sub_df, col, lbl in [(m_nonstars, 'orange', 'Non-star'), (m_stars, 'blue', 'Star')]:
        if len(sub_df) > 0:
            ax.scatter(sub_df['mag'], sub_df['sigma'], c=col, alpha=0.5, s=10, label=lbl)
    ax.axhline(5, color='red', linestyle='--', label='Threshold (5s)')
    ax.set_xlabel("Gaia G Magnitude"); ax.set_ylabel("Residual Sigma"); ax.set_title("Sigma vs Gaia Magnitude"); ax.legend(fontsize=7)
    if mag_lims: ax.set_xlim(mag_lims)

    # 9. Color vs Color
    ax = axes[4, 1]
    if 'color' in matched_df.columns:
        if len(rejected_df) > 0:
            ax.scatter(rejected_df['color'], rejected_df['color_hst'], c='red', alpha=0.15, s=5, label='Rejected')
        _scatter_matched(ax, 'color', 'color_hst')
        ax.invert_yaxis(); ax.set_xlabel("BP - RP (mag)"); ax.set_ylabel("G - HST (mag)"); ax.set_title("Color-Color Diagram"); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "BP-RP not available", ha='center', va='center'); ax.set_title("CMD Placeholder")

    # 10. Gaia Proper Motions (PMRA vs PMDec)
    ax = axes[0, 1]
    if len(rejected_df) > 0:
        ax.scatter(rejected_df['pmra'], rejected_df['pmdec'], c='red', alpha=0.15, s=5, label='Rejected')
    _scatter_matched(ax, 'pmra', 'pmdec')
    ax.set_xlabel("PMRA (mas/yr)"); ax.set_ylabel("PMDec (mas/yr)"); ax.set_title("Gaia Proper Motions"); ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(out_dir, "diagnostic_plots.png"), dpi=150); plt.close()

class FileLogger(object):
    def __init__(self, filename, mode="w"): self.log = open(filename, mode)
    def write(self, message): self.log.write(message)
    def flush(self): self.log.flush()

# ---------------------------------------------------------------------------
# Discovery and refinement helpers
# ---------------------------------------------------------------------------

def _plot_offset_histogram(hist, xed, yed, peaks, title, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    with np.errstate(divide='ignore', invalid='ignore'):
        # log_hist = np.log10(hist.T + 1e-30)
        log_hist = np.log10(hist.T)
        log_hist = hist.T
        log_hist[log_hist == 0] = np.nan
        # finite = log_hist[np.isfinite(log_hist)]
        # log_hist[~np.isfinite(log_hist)] = finite.min() if len(finite) else 0
    im = ax.imshow(log_hist, origin='lower', aspect='equal',
                   extent=[xed[0], xed[-1], yed[0], yed[-1]], cmap='viridis')
    plt.colorbar(im, ax=ax, label='log10(weighted density)')
    for dx, dy, _ in peaks:
        ax.axvline(dx, color='red', lw=0.8, ls='--', alpha=0.7)
        ax.axhline(dy, color='red', lw=0.8, ls='--', alpha=0.7)
    ax.set_xlabel('dx  (HST − Gaia guess, pixels)')
    ax.set_ylabel('dy  (HST − Gaia guess, pixels)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


def _save_offset_histogram(best, image_name, out_dir):
    if best.get('offset_hist') is None:
        return
    ds_str = f"ds={best.get('best_ds', 0.0):+.4f}"
    title = f'{image_name}  |  best tier q<{best["q"]} m<{best["m"]:.1f}  {ds_str}'
    _plot_offset_histogram(best['offset_hist'], best['offset_xed'], best['offset_yed'],
                           best.get('offset_peaks', []), title,
                           os.path.join(out_dir, 'offset_histogram.png'))


def _run_4p_discovery(hst_d, gaia_f, params, max_mag_diff, scale_sweep=False, discovery_max_offset=50,
                      seed_quality_mask=None, debug_verbose=False,
                      sigma_rot_deg=None, sigma_scale=None,
                      forced=None, selection='lowest_cost'):
    """selection: 'lowest_cost' (legacy rule; DELVE / CFHT / other callers) or 'evaluate' (HST,
    2026-10-01: one candidate per agreeing tier cluster, refined and chance-tested by the caller).

    Tier-walks qfit x mag limits to find a physically plausible 4P similarity seed.

    hst_d  keys: x, y, mag, C, qfit, chi2
    gaia_f keys: x, y, C, mag, err, has_pms, xguess, yguess

    Returns best-tier dict or None.
    """
    if hst_d is None or len(hst_d['x']) == 0:
        print('  4P discovery: no HST sources in this tier — skipping')
        return None
    seed_margin = 2000
    near = (np.abs(gaia_f['x'] - params['x_cen']) <= seed_margin) & \
           (np.abs(gaia_f['y'] - params['y_cen']) <= seed_margin)
    field_idx = np.where(near)[0]
    n_subset = 1000
    if len(field_idx) > n_subset:
        seed_idx = field_idx[np.argsort(gaia_f['err'][near])[:n_subset]]
    else:
        seed_idx = field_idx
    # Forced anchors (previously-matched Gaia-common stars): ALWAYS part of
    # the seed sample and of every q/mag tier — trustworthy pairs vote on
    # offsets alongside whatever new (possibly faint) candidates each tier
    # admits. forced = (h_indices_into_hst_d, g_indices_into_gaia_f).
    _fh, _fg = (None, None) if forced is None else forced
    if _fg is not None and len(_fg):
        seed_idx = np.unique(np.concatenate([seed_idx, np.asarray(_fg)]))

    xg_s = gaia_f['xguess'][seed_idx]
    yg_s = gaia_f['yguess'][seed_idx]
    Cg_s = gaia_f['C'][seed_idx]

    tree_hst = KDTree(np.column_stack([hst_d['x'], hst_d['y']]))

    qfit_limits = [0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
    mag_limits = np.arange(hst_d['mag'].min() + 1.0, hst_d['mag'].max() + 0.5, 1.0)
    if len(mag_limits) == 0:
        mag_limits = [hst_d['mag'].max()]
    mag_limits[-1] = hst_d['mag'].max()

    discovered = []
    _dbg = {'valid_seed_fail': 0, 'dedup<3': 0, 'sigma_rej<3': 0, 'scale_rot_fail': 0}
    print(f"  Walking over {len(qfit_limits)*len(mag_limits)} tiers for 4P discovery...")

    for qlim in qfit_limits:
        for mlim in mag_limits:
            curr_q_mag_lims = (hst_d['qfit'] <= qlim) & (hst_d['mag'] <= mlim)
            h_mask = curr_q_mag_lims & (hst_d['qfit'] > 0.0) & (hst_d['chi2'] < 5.0)
            if np.sum(h_mask) < 3:
                h_mask = curr_q_mag_lims & (hst_d['qfit'] >= 0.0) & (hst_d['chi2'] < 5.0)
            if np.sum(h_mask) < 3:
                h_mask = curr_q_mag_lims & (hst_d['qfit'] >= 0.0) & (hst_d['chi2'] < 10.0)
            if _fh is not None and len(_fh):
                h_mask = h_mask.copy(); h_mask[np.asarray(_fh)] = True
            if np.sum(h_mask) < 3:
                continue
            h_idx_tier = np.where(h_mask)[0]

            if seed_quality_mask is not None:
                hist_keep = seed_quality_mask[seed_idx]
            else:
                hist_keep = np.ones(len(xg_s), dtype=bool)
                if np.sum(gaia_f['has_pms'][seed_idx]) >= 3:
                    hist_keep = gaia_f['has_pms'][seed_idx].copy()
            if _fg is not None and len(_fg):
                hist_keep = hist_keep | np.isin(seed_idx, np.asarray(_fg))
            if np.sum(hist_keep) < 3:
                continue

            best_ds, offset_peaks, tier_hist, tier_xed, tier_yed = find_scale_and_offset(
                xg_s[hist_keep], yg_s[hist_keep], gaia_f['err'][seed_idx][hist_keep],
                hst_d['x'][h_idx_tier], hst_d['y'][h_idx_tier], hst_d['mag'][h_idx_tier],
                cov1=Cg_s[hist_keep], cov2=hst_d['C'][h_idx_tier],
                x_cen=params['x_cen'], y_cen=params['y_cen'],
                max_offset=discovery_max_offset, bin_size=1, top_n=3,
                ds_range=(-0.02, 0.02) if scale_sweep else (0.0, 0.0), n_scales=41 if scale_sweep else 1,
                return_histogram=True
            )
            dx_off, dy_off, _peak_score = offset_peaks[0]
            xg_tier = xg_s + best_ds * (xg_s - params['x_cen']) + dx_off
            yg_tier = yg_s + best_ds * (yg_s - params['y_cen']) + dy_off

            # Try 40px (tight prior) then 100px (header fallback)
            valid_seed = False
            for cur_rad in [40.0, 100.0]:
                tree_h_tier = KDTree(np.column_stack([hst_d['x'][h_idx_tier], hst_d['y'][h_idx_tier]]))
                dists, h_idxs_tier = tree_h_tier.query(np.column_stack([xg_tier, yg_tier]), k=1, distance_upper_bound=cur_rad)
                valid_idx = dists < cur_rad
                if np.sum(valid_idx) >= 3:
                    valid_seed = True
                    break
            if not valid_seed:
                _dbg['valid_seed_fail'] += 1
                if debug_verbose and _dbg['valid_seed_fail'] <= 3:
                    n40 = np.sum(dists < 40.0)
                    n100 = np.sum(dists < 100.0)
                    print(f"    DBG valid_seed fail q<{qlim} m<{mlim:.1f}: "
                          f"peak=({dx_off:.1f},{dy_off:.1f}) score={_peak_score:.3f}, "
                          f"n_hst={np.sum(h_mask)}, @40px={n40}, @100px={n100}")
                continue

            # Greedy 1-to-1 cleanup using log-probability cost
            h_v_full = h_idx_tier[h_idxs_tier[valid_idx]]
            g_v_seed = np.where(valid_idx)[0]
            dx = hst_d['x'][h_v_full] - xg_tier[g_v_seed]
            dy = hst_d['y'][h_v_full] - yg_tier[g_v_seed]
            C_tot_seed = Cg_s[g_v_seed] + hst_d['C'][h_v_full]
            costs_seed = compute_logprob_cost(dx, dy, C_tot_seed)
            mdf_seed = pd.DataFrame({'g_seed': g_v_seed, 'h_full': h_v_full, 'c': costs_seed})\
                         .sort_values('c').drop_duplicates('h_full').drop_duplicates('g_seed')
            if len(mdf_seed) < 3:
                _dbg['dedup<3'] += 1
                if debug_verbose and _dbg['dedup<3'] <= 3:
                    print(f"    DBG dedup<3 q<{qlim} m<{mlim:.1f}: {np.sum(valid_idx)} pairs → {len(mdf_seed)} after dedup")
                continue

            h_b_idx = mdf_seed['h_full'].values
            g_b_full_idx = seed_idx[mdf_seed['g_seed'].values]
            xh_b, yh_b = hst_d['x'][h_b_idx], hst_d['y'][h_b_idx]
            xg_b, yg_b = gaia_f['x'][g_b_full_idx], gaia_f['y'][g_b_full_idx]
            mag_diffs = gaia_f['mag'][g_b_full_idx] - hst_d['mag'][h_b_idx]
            curr_keep = np.ones(len(xh_b), dtype=bool)
            if np.sum(gaia_f['has_pms'][g_b_full_idx]) >= 3:
                curr_keep = gaia_f['has_pms'][g_b_full_idx].copy()
            zp = np.median(mag_diffs[curr_keep])
            cur_M = np.eye(2)

            # Iterative 4P fit with sigma rejection
            for _ in range(5):
                res_4p, _, C_params, _ = fit_4p_weighted(
                    xh_b[curr_keep], yh_b[curr_keep],
                    xg_b[curr_keep], yg_b[curr_keep],
                    hst_d['C'][h_b_idx[curr_keep]],
                    gaia_f['C'][g_b_full_idx[curr_keep]],
                    initial_M=cur_M,
                    scale_prior=params['initial_scale'],
                    scale_sigma=sigma_scale if sigma_scale is not None else SIGMA_SCALE,
                    rot_sigma=np.radians(sigma_rot_deg if sigma_rot_deg is not None else SIGMA_ROT_DEG))
                A, B, C, D, xs_o, ys_o, xt_o, yt_o = res_4p
                cur_M = np.array([[A, B], [C, D]])

                xh_p, yh_p = apply_affine(xh_b, yh_b, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
                dx_v, dy_v = xg_b - xh_p, yg_b - yh_p
                C_proj = np.einsum('ij,njk,lk->nil', cur_M, hst_d['C'][h_b_idx], cur_M)
                dxh_v, dyh_v = xh_b - xs_o, yh_b - ys_o
                J = np.zeros((len(dxh_v), 2, 4))
                J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh_v, -dyh_v, 1.0
                J[:, 1, 0], J[:, 1, 1], J[:, 1, 3] = dyh_v, dxh_v, 1.0
                C_model = np.einsum('nij,jk,nlk->nil', J, C_params, J)
                # Add a 1px discovery floor so chi2/cost are on a consistent
                # scale regardless of Gaia quality (5p vs 2p) or star brightness.
                # Correct matches have small pixel residuals → negative costs;
                # false matches have large residuals → positive costs.
                # The floor does not affect sigma rejection (which uses ds_4p).
                _disc_floor = np.eye(2)[np.newaxis] * 1.0**2
                C_tot_v = gaia_f['C'][g_b_full_idx] + C_proj + C_model + _disc_floor
                sigs_v = compute_mahalanobis(dx_v, dy_v, C_tot_v)
                costs_v = compute_logprob_cost(dx_v, dy_v, C_tot_v)
                chi2 = np.sum(sigs_v[curr_keep])
                cost = np.sum(costs_v[curr_keep])

                ds_4p = np.sqrt(dx_v**2 + dy_v**2)
                finite_dists = np.isfinite(ds_4p)
                p16, p50 = np.nanpercentile(ds_4p[finite_dists & curr_keep], [16, 50])
                thresh = min(max(p50 + 3*(p50-p16), 1), cur_rad)
                if not np.isfinite(thresh):
                    thresh = cur_rad

                # Discovery sigma-rejection uses only the pixel-distance threshold.
                # Gaia formal uncertainties (~0.01px) are far smaller than the
                # multi-pixel residuals expected from a noisy 4-5 pair 4P fit, so a
                # Mahalanobis threshold (sigs_v < 5) would always reject all pairs.
                # The scale/rotation sanity check below catches spurious transforms.
                good_v = (ds_4p < thresh) & (np.abs(mag_diffs - zp) < max_mag_diff)
                if np.sum(good_v) < 3:
                    break
                if np.all(curr_keep == good_v):
                    break
                curr_keep[:] = good_v
                zp = np.median(mag_diffs[curr_keep])

            g_b_full_idx = g_b_full_idx[curr_keep]
            h_b_idx = h_b_idx[curr_keep]
            if np.sum(good_v) < 3:
                _dbg['sigma_rej<3'] += 1
                if debug_verbose and _dbg['sigma_rej<3'] <= 3:
                    print(f"    DBG sigma_rej<3 q<{qlim} m<{mlim:.1f}: "
                          f"{len(mdf_seed)} pairs → {np.sum(good_v)} after rejection "
                          f"(dists={np.round(np.sqrt(dx_v**2+dy_v**2),1).tolist()})")
                continue

            scale_fit = np.sqrt(A*D - B*C)
            rot_fit = np.degrees(np.arctan2(B - C, A + D))
            _scale_ok = 0.98*params['initial_scale'] <= scale_fit <= 1.02*params['initial_scale']
            _rot_ok = abs(rot_fit) < 0.2
            if not (_scale_ok and _rot_ok):
                _dbg['scale_rot_fail'] += 1
                if debug_verbose and _dbg['scale_rot_fail'] <= 5:
                    print(f"    DBG scale/rot fail q<{qlim} m<{mlim:.1f}: "
                          f"scale={scale_fit:.4f} (need {0.98*params['initial_scale']:.4f}–{1.02*params['initial_scale']:.4f}), "
                          f"rot={rot_fit:.3f}° (need |rot|<0.2°)")
            if _scale_ok and _rot_ok:
                red_chi2 = chi2 / (2*len(h_b_idx) - 4)
                red_cost = cost / len(h_b_idx)
                zp_tier = np.median(gaia_f['mag'][g_b_full_idx] - hst_d['mag'][h_b_idx])
                discovered.append({
                    'A': A, 'B': B, 'C': C, 'D': D,
                    'xs_o': xs_o, 'ys_o': ys_o, 'xt_o': xt_o, 'yt_o': yt_o,
                    'n_match': len(h_b_idx), 'red_chi2': red_chi2, 'red_cost': red_cost,
                    'q': qlim, 'm': mlim, 'zp': zp_tier,
                    'h_v': h_b_idx, 'g_v': g_b_full_idx,
                    'best_ds': best_ds,
                    'offset_peaks': offset_peaks,
                    'offset_hist': tier_hist, 'offset_xed': tier_xed, 'offset_yed': tier_yed,
                })
                peaks_str = "  |  ".join(f"dx={dx:.1f},dy={dy:.1f}(s={s:.2f})" for dx, dy, s in offset_peaks)
                print(f"    q<{qlim} m<{mlim:.1f}: {len(h_b_idx)} stars, red_chi2={red_chi2:.3f}, "
                      f"red_cost={red_cost:.2f}, zp={zp_tier:.3f}, scale={scale_fit:.6f}, rot={rot_fit:.4f} | "
                      f"offsets(ds={best_ds:+.4f}): {peaks_str}")

    if not discovered:
        if debug_verbose:
            print(f"    DBG failure summary: {_dbg}")
        return None
    if selection == 'evaluate':
        return _select_discovery(discovered, params)
    return min(discovered, key=lambda x: x['red_cost'])


CHANCE_SHIFTS_PX = (25.0, 50.0)   # rings of 8 shifted copies of the final solution
CHANCE_FA_PROB = 1e-3             # accept only if P(N >= N_real | Poisson(lambda_chance)) < this
CHANCE_MAG_WIN = 1.5              # half-width of the test's zero-point window (optical / IR filters)
CHANCE_MAG_WIN_UV = 3.5           # UV filters (PHOTPLAM < 4000 A): G-HST spans several mag (NGC 300 F225W -4..+2)
                                  # Fixed per filter class: an adaptive (3 x MAD) window let a spurious set with
                                  # uniformly spread G-HST widen it to 11 mag (47 Tuc j8c051t9q, 2026-10-01).
CHANCE_SIGMA_MAX = 2.0            # the test counts pairs within this many sigma: real matches sit well
                                  # inside it, chance pairs are uniform over the 5-sigma area (~16% inside)


def _final_pass_count(xh_in_g, yh_in_g, x_g, y_g, M, C_pix_hst, C_g, C_params, xs_o, ys_o, x_hst, y_hst,
                      resid_cov, g_mag, mag_hst, zp, max_mag_diff, zp_win=None, mag_win=CHANCE_MAG_WIN):
    """Number of one-to-one pairs the final pass would accept for these Gaia positions, counting only
    pairs within CHANCE_MAG_WIN of zp_win (default zp)."""
    tree = KDTree(np.column_stack([x_g, y_g]))
    ds, g_idxs = tree.query(np.column_stack([xh_in_g, yh_in_g]), k=5, distance_upper_bound=100)
    h_v = np.repeat(np.arange(len(xh_in_g)), 5); valid = ds.flatten() < 100
    h_v, g_v = h_v[valid], g_idxs.flatten()[valid]
    if len(h_v) == 0:
        return 0
    dx_v, dy_v = x_g[g_v] - xh_in_g[h_v], y_g[g_v] - yh_in_g[h_v]
    C_proj = np.einsum('ij,njk,lk->nil', M, C_pix_hst[h_v], M)
    J = np.zeros((len(h_v), 2, 6)); dxh, dyh = x_hst[h_v] - xs_o, y_hst[h_v] - ys_o
    J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh, dyh, 1.0
    J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = dxh, dyh, 1.0
    C_tot = C_g[g_v] + C_proj + np.einsum('nij,jk,nlk->nil', J, C_params, J) + resid_cov
    sig = compute_mahalanobis(dx_v, dy_v, C_tot); cost = compute_logprob_cost(dx_v, dy_v, C_tot)
    md = g_mag[g_v] - mag_hst[h_v]; cost = cost + ((md - zp) / 1.0) ** 2
    df = pd.DataFrame({'h': h_v, 'g': g_v, 's': sig, 'c': cost, 'md': md}).sort_values('c')
    df = df.drop_duplicates('g').drop_duplicates('h')
    zc = zp if zp_win is None else zp_win
    return int(((df['s'] < CHANCE_SIGMA_MAX) & (np.abs(df['md'] - zp) < max_mag_diff)
                & (np.abs(df['md'] - zc) < min(max_mag_diff, mag_win))).sum())


def chance_significance(n_real, counts):
    """(lambda, false-alarm probability) of n_real matches given the shifted-solution counts."""
    from scipy.stats import poisson
    lam = max(float(np.mean(counts)), 1e-3)
    return lam, float(poisson.sf(n_real - 1, lam))


DISCOVERY_AGREE_PX = 3.0    # two tier solutions agree if they map the footprint within this
MAX_DISCOVERY_CANDIDATES = 4   # distinct discovery clusters evaluated end to end


def _select_discovery(discovered, params):
    """Pick the 4P seed the tiers AGREE on (user 2026-10-01).

    The old rule (lowest red_cost) favours tiny seed sets: on 47 Tuc j8fw01b6q ~40 tiers found the
    same ~960-star solution at offset (0,0) but a lone 3-star tier with scale 0.971 / rot 0.19 deg had
    a lower cost and became a 430-pair chance solution.  Now: cluster tier solutions that map the
    footprint within DISCOVERY_AGREE_PX, take the cluster supported by the most tiers (ties: most
    stars), and inside it the lowest-cost solution among those with >= half the cluster's largest
    seed set.  A single discovered solution is returned unchanged."""
    if len(discovered) == 1:
        return discovered[0]
    gx, gy = np.meshgrid(np.linspace(-1800, 1800, 5), np.linspace(-1800, 1800, 5))
    hx = params['x_cen'] + gx.ravel(); hy = params['y_cen'] + gy.ravel()
    P = np.array([np.column_stack(apply_affine(hx, hy, d['A'], d['B'], d['C'], d['D'],
                                               d['xs_o'], d['ys_o'], d['xt_o'], d['yt_o']))
                  for d in discovered])                                   # (n, 25, 2)
    D = np.max(np.hypot(P[:, None, :, 0] - P[None, :, :, 0], P[:, None, :, 1] - P[None, :, :, 1]), axis=2)
    agree = D < DISCOVERY_AGREE_PX
    support = agree.sum(axis=1)
    nmax = np.array([max(discovered[j]['n_match'] for j in np.where(agree[i])[0]) for i in range(len(discovered))])
    order = sorted(range(len(discovered)), key=lambda i: (support[i], nmax[i]), reverse=True)
    reps, assigned = [], set()
    for i in order:
        if i in assigned:
            continue
        members = [discovered[j] for j in np.where(agree[i])[0]]
        assigned |= set(np.where(agree[i])[0].tolist())
        nm = max(d['n_match'] for d in members)
        reps.append((int(support[i]), min((d for d in members if d['n_match'] >= 0.5 * nm), key=lambda d: d['red_cost'])))
    old = min(discovered, key=lambda d: d['red_cost'])
    # the legacy lowest-cost pick is ALWAYS evaluated (second, after the best-supported cluster): with
    # >= MAX_DISCOVERY_CANDIDATES clusters it used to be cut off (GOODS-N j91we8mgq: its 6-match,
    # 0.06 px solution was never tried)
    _rest = [r for r in reps if r[1] is not old]
    if reps[0][1] is old:
        reps = [reps[0]] + _rest
    else:
        reps = [reps[0], (next((r[0] for r in reps if r[1] is old), 1), old)] + _rest[1:]
    pick = reps[0][1]
    pick['_alternatives'] = [r[1] for r in reps[1:]]
    print(f"  Discovery selection: {len(discovered)} tier solutions in {len(reps)} distinct clusters "
          f"(support {[r[0] for r in reps[:MAX_DISCOVERY_CANDIDATES]]}); candidates go through refinement + "
          f"chance test, the most significant wins")
    return pick


def _run_affine_refinement(best_4p, hst_d, gaia_f, tree_gaia, max_mag_diff, use_resid_floor=True,
                           sigma_rot_deg=None, sigma_scale=None, sigma_skew=None):
    """
    Upgrades 4P seeds to a 6P affine transform and iterates until convergence.

    hst_d  keys: x, y, mag, C
    gaia_f keys: x, y, C, mag

    Returns (A, B, C, D, xs_o, ys_o, xt_o, yt_o, C_params, resid_cov, zp, h_f, g_f).
    """
    A, B, C, D = best_4p['A'], best_4p['B'], best_4p['C'], best_4p['D']
    xs_o, ys_o = best_4p['xs_o'], best_4p['ys_o']
    xt_o, yt_o = best_4p['xt_o'], best_4p['yt_o']
    zp = best_4p['zp']
    h_idx_b, g_idx_b = best_4p['h_v'], best_4p['g_v']
    M = np.array([[A, B], [C, D]])

    # Upgrade 4P seeds to initial 6P affine fit
    xh_b, yh_b = hst_d['x'][h_idx_b], hst_d['y'][h_idx_b]
    xg_b, yg_b = gaia_f['x'][g_idx_b], gaia_f['y'][g_idx_b]
    fit_res, _, C_params, _ = fit_affine_weighted(
        xh_b, yh_b, xg_b, yg_b, hst_d['C'][h_idx_b], gaia_f['C'][g_idx_b],
        initial_M=M,
        sigma_rot_deg=sigma_rot_deg, sigma_scale=sigma_scale, sigma_skew=sigma_skew,
        skew_prior=SIGMA_SKEW if sigma_skew is None else 0)
    A, B, C, D, xs_o, ys_o, xt_o, yt_o = fit_res
    M = np.array([[A, B], [C, D]])

    xh_in_g, yh_in_g = apply_affine(xh_b, yh_b, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
    dx_v, dy_v = xg_b - xh_in_g, yg_b - yh_in_g
    resid_sigma_x = 0.5 * np.diff(np.nanpercentile(dx_v, [16, 84]))[0]
    resid_sigma_y = 0.5 * np.diff(np.nanpercentile(dy_v, [16, 84]))[0]
    init_resid_x, init_resid_y = resid_sigma_x, resid_sigma_y
    resid_cov = np.diag(np.array([resid_sigma_x, resid_sigma_y])**2) if use_resid_floor else np.zeros((2, 2))

    ratio, rot = np.sqrt(A*D-B*C), np.degrees(np.arctan2(B-C, A+D))
    print(f"  Init 6P: {len(xh_b)} seeds, scale={ratio:.6f}, rot={rot:.4f}deg, "
          f"on_skew={0.5*(A-D):.2e}, off_skew={0.5*(B+C):.2e}, "
          f"resid=[{resid_sigma_x:.4f},{resid_sigma_y:.4f}]px, zp={zp:.3f}")

    h_f, g_f = h_idx_b, g_idx_b
    for it in range(10):
        xh_in_g, yh_in_g = apply_affine(hst_d['x'], hst_d['y'], A, B, C, D, xs_o, ys_o, xt_o, yt_o)
        ds, g_idxs = tree_gaia.query(np.column_stack([xh_in_g, yh_in_g]), k=5, distance_upper_bound=100)
        h_idx_all = np.repeat(np.arange(len(hst_d['x'])), 5)
        valid = ds.flatten() < 100
        h_v, g_v = h_idx_all[valid], g_idxs.flatten()[valid]
        if len(h_v) < 3:
            print(f"  Iter {it}: only {len(h_v)} candidates within 100px. Breaking.")
            break

        dx_v, dy_v = gaia_f['x'][g_v] - xh_in_g[h_v], gaia_f['y'][g_v] - yh_in_g[h_v]
        C_proj = np.einsum('ij,njk,lk->nil', M, hst_d['C'][h_v], M)
        dxh_v, dyh_v = hst_d['x'][h_v] - xs_o, hst_d['y'][h_v] - ys_o
        J = np.zeros((len(h_v), 2, 6))
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh_v, dyh_v, 1.0
        J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = dxh_v, dyh_v, 1.0
        C_model = np.einsum('nij,jk,nlk->nil', J, C_params, J)
        C_total = gaia_f['C'][g_v] + C_proj + C_model + resid_cov

        sigs_v = compute_mahalanobis(dx_v, dy_v, C_total)
        costs_v = compute_logprob_cost(dx_v, dy_v, C_total)
        mag_diffs = gaia_f['mag'][g_v] - hst_d['mag'][h_v]
        costs_v += ((mag_diffs - zp) / 1.0)**2
        costs_v[np.abs(mag_diffs - zp) > max_mag_diff] = np.inf

        mdf = pd.DataFrame({'h': h_v, 'g': g_v, 's': sigs_v, 'c': costs_v,
                             'dx': dx_v, 'dy': dy_v})\
                .sort_values('c').drop_duplicates('g').drop_duplicates('h')
        mdf['mag_diff'] = gaia_f['mag'][mdf['g'].values] - hst_d['mag'][mdf['h'].values]
        good = mdf[(mdf['s'] < 5.0) & (np.abs(mdf['mag_diff'] - zp) < max_mag_diff)]
        if len(good) < 3:
            print(f"  Iter {it}: only {len(good)} stars passed sigma<5 filter. Breaking.")
            break

        h_f, g_f = good['h'].values, good['g'].values
        fit_res_new, _, C_params, _ = fit_affine_weighted(
            hst_d['x'][h_f], hst_d['y'][h_f], gaia_f['x'][g_f], gaia_f['y'][g_f],
            hst_d['C'][h_f], gaia_f['C'][g_f], initial_M=M,
            sigma_rot_deg=sigma_rot_deg, sigma_scale=sigma_scale, sigma_skew=sigma_skew,
            skew_prior=SIGMA_SKEW if sigma_skew is None else 0)
        change = np.abs(fit_res_new[0] - A) + np.abs(fit_res_new[1] - B)
        A, B, C, D, xs_o, ys_o, xt_o, yt_o = fit_res_new
        M = np.array([[A, B], [C, D]])

        resid_sigma_x = 0.5 * np.diff(np.nanpercentile(good['dx'], [16, 84]))[0]
        resid_sigma_y = 0.5 * np.diff(np.nanpercentile(good['dy'], [16, 84]))[0]
        resid_cov = np.diag(np.array([resid_sigma_x, resid_sigma_y])**2) if use_resid_floor else np.zeros((2, 2))
        zp = np.median(good['mag_diff'])

        ratio, rot = np.sqrt(A*D-B*C), np.degrees(np.arctan2(B-C, A+D))
        print(f"  Iter {it}: {len(h_f)} matches, scale={ratio:.6f}, rot={rot:.4f}deg, "
              f"resid=[{resid_sigma_x:.4f},{resid_sigma_y:.4f}]px, zp={zp:.3f}")
        if it > 5 and change < 1e-11:
            break

    return A, B, C, D, xs_o, ys_o, xt_o, yt_o, C_params, resid_cov, zp, h_f, g_f, init_resid_x, init_resid_y


# ---------------------------------------------------------------------------
# Main per-image processor
# ---------------------------------------------------------------------------

_PCM_CACHE = {}


PLAUS_SCALE = 6e-4            # |ratio / (header scale x VAFACTOR) - 1| allowed for a final solution
PLAUS_ROT_DEG = 0.15          # |rotation vs header| allowed
PLAUS_SKEW = 1e-3             # |on-/off-axis skew| allowed
PLAUS_MAX_N = 30              # the plausibility gate only applies below this many significant matches (high-N
                              # solutions cannot be bent onto chance pairs; old headers may be > 0.15 deg off)
PLAUS_NSIG = 5.0              # ... plus this many fitted sigmas (low-N solutions are poorly constrained)
CR_SEED_CONC = 1.05          # concentration above which a detection is too sharp for the PSF
CR_SEED_QFIT = 0.10          # ... only when the PSF fit is also poor (qfit above this)
GUESS_SEED_RADIUS_PX = 1.5   # direct association radius around a guess-predicted Gaia position


def _guess_direct_seed(guess_affine, params, hst_d, gaia_f, max_mag_diff, min_matches):
    """Seed pairs straight from a trusted guess transform (visit-group completion): each Gaia
    source's predicted HST position takes the nearest HST source within GUESS_SEED_RADIUS_PX
    (widened by its propagated Gaia error), one-to-one, magnitude-consistent.  Bypasses the
    offset-histogram discovery, which in crowded fields locks onto chance peaks.  Returns a
    discovery-style dict for _run_affine_refinement, or None (caller falls back to discovery)."""
    M, t = np.asarray(guess_affine[0], float), np.asarray(guess_affine[1], float)
    c = np.array([params['x_cen'], params['y_cen']])
    h = np.einsum('ij,nj->ni', np.linalg.inv(M), np.column_stack([gaia_f['x'], gaia_f['y']]) - t) + c
    rad = np.maximum(GUESS_SEED_RADIUS_PX, 3.0 * np.asarray(gaia_f['err'], float))
    tree = KDTree(np.column_stack([hst_d['x'], hst_d['y']]))
    d, j = tree.query(h, k=4, distance_upper_bound=float(np.max(rad)))
    rows = []
    for gi in range(len(h)):
        for k in range(d.shape[1]):
            if np.isfinite(d[gi, k]) and d[gi, k] < rad[gi]:
                rows.append((gi, int(j[gi, k]), float(d[gi, k]), float(gaia_f['mag'][gi] - hst_d['mag'][j[gi, k]])))
    if len(rows) < min_matches:
        return None
    df = pd.DataFrame(rows, columns=['g', 'h', 'd', 'dm'])
    zp = float(np.median(df.sort_values('d').drop_duplicates('g')['dm']))
    df = df[np.abs(df['dm'] - zp) < max_mag_diff]
    df['cost'] = df['d'] ** 2 + (df['dm'] - zp) ** 2          # position (px) + magnitude (mag) consistency
    df = df.sort_values('cost').drop_duplicates('g').drop_duplicates('h')
    if len(df) < min_matches:
        return None
    zp = float(np.median(df['dm']))
    return {'A': M[0, 0], 'B': M[0, 1], 'C': M[1, 0], 'D': M[1, 1],
            'xs_o': c[0], 'ys_o': c[1], 'xt_o': t[0], 'yt_o': t[1],
            'n_match': int(len(df)), 'red_chi2': float('nan'), 'red_cost': float('nan'),
            'q': 'guess', 'm': float('nan'), 'zp': zp,
            'h_v': df['h'].to_numpy(), 'g_v': df['g'].to_numpy(), 'offset_hist': None}


def _pos_corr_applier(spec):
    """Learned GDC-correction applier (bp3m.pos_corr_basis), one per spec per worker process."""
    if spec not in _PCM_CACHE:
        from bp3m.pos_corr_basis import make_pos_corr
        _PCM_CACHE[spec] = make_pos_corr(spec)
    return _PCM_CACHE[spec]


def process_single_image(hst, gaia_df, hst_pix_floor=0.01, min_matches=3, zero_pm=False, max_mag_diff=3.0, scale_sweep=False, discovery_max_offset=50, use_resid_floor=True, sigma_rot_deg=None, sigma_scale=None, sigma_skew=None, init_resid_max=5.0, pos_corr_model=None,
                         guess_affine=None, guess_max_offset=10.0, log_mode="w"):
    """guess_affine: optional (M, t) predicting this image's HST-pixel -> Gaia-frame mapping
    g = M (h - c) + t, c = (x_cen, y_cen) of its header frame (visit-group completion: own previous
    solution or the sibling-transferred header error).  The Gaia guess positions are then taken from
    it and the 4P offset search is narrowed to +-guess_max_offset px; every gate is unchanged."""
    start_time = time.time()
    image_name = os.path.basename(hst['flc']).replace("_flc.fits", "")
    log_file, original_stdout = os.path.join(hst['root'], "processing_log.txt"), sys.stdout
    sys.stdout = FileLogger(log_file, log_mode)
    if log_mode == "a":
        print(f"\n===== retry: discovery_max_offset={discovery_max_offset} px =====")
    print(f"Starting {image_name}...", file=original_stdout)
    try:
        print(f"--- Processing HST image: {image_name} ---")
        params = get_hst_params(hst['flc'], catalog_file=hst['catalog'])
        if params is None:
            print(f"Finished {image_name}: Failed to load parameters.", file=original_stdout)
            return
        params['min_matches'] = min_matches
        try:
            _photplam = float(fits.getval(hst['flc'], 'PHOTPLAM', ext=1))
        except Exception:
            try:
                _photplam = float(fits.getval(hst['flc'], 'PHOTPLAM', ext=0))
            except Exception:
                _photplam = np.nan
        _chance_mag_win = CHANCE_MAG_WIN_UV if (np.isfinite(_photplam) and _photplam < 4000.0) else CHANCE_MAG_WIN

        # --- Propagate Gaia to HST epoch and project to pixel frame ---
        from bp3m.astro_utils import field_typical_astrometry
        _fill = field_typical_astrometry(
            gaia_df['pmra'].values, gaia_df['pmdec'].values,
            plx=gaia_df['parallax'].values if 'parallax' in gaia_df else None)
        if _fill["method"] != "none":
            print(f"  2p propagation fill ({_fill['method']}, n={_fill['n']}): "
                  f"pm=({_fill['pmra']:+.2f},{_fill['pmdec']:+.2f}) "
                  f"±{_fill.get('pm_sig', float('nan')):.2f} mas/yr  "
                  f"plx={_fill['plx']:+.3f} ±{_fill.get('plx_sig', float('nan')):.2f} mas")
        ra_prop, dec_prop, Ct = propagate_gaia_with_cov(
            gaia_df, params['obs_epoch_mjd'], zero_pm=zero_pm,
            fill_pm_plx=_fill)
        dx_deg_full = rd2x(ra_prop, dec_prop, params['ra_cen'], params['dec_cen'])
        dy_deg_full = rd2y(ra_prop, dec_prop, params['ra_cen'], params['dec_cen'])
        scale_deg = params['pixel_scale'] / 3600.0
        mas_to_px = 1.0 / (params['pixel_scale'] * 1000.0)

        # Convert sky covariance (mas²) to pixel frame, accounting for +X = -RA
        C_pix_gaia = Ct * mas_to_px**2
        C_pix_gaia[:, 0, :] *= -1
        C_pix_gaia[:, :, 0] *= -1
        x_gaia_proj = params['x_cen'] - dx_deg_full / scale_deg
        y_gaia_proj = params['y_cen'] + dy_deg_full / scale_deg

        # Rotate Gaia positions into the approximate HST detector frame using ORIENTAT
        theta_init = np.radians(-params['orientat'])
        init_rot_mat    = np.array([[ np.cos(theta_init), np.sin(theta_init)],
                                    [-np.sin(theta_init), np.cos(theta_init)]])
        init_inv_rot_mat = np.linalg.inv(init_rot_mat)
        xy_gaia_proj = np.einsum('ij,nj->ni',
                                  init_inv_rot_mat,
                                  np.column_stack([x_gaia_proj, y_gaia_proj])
                                  - np.array([params['x_cen'], params['y_cen']]))\
                       + np.array([params['x_cen'], params['y_cen']])
        x_gaia_proj, y_gaia_proj = xy_gaia_proj[:, 0], xy_gaia_proj[:, 1]
        C_pix_gaia = np.einsum('ij,njk,lk->nil', init_inv_rot_mat, C_pix_gaia, init_inv_rot_mat)
        gaia_err_total = np.power(np.linalg.det(C_pix_gaia), 0.25)
        has_gaia_pms = np.isfinite(gaia_df['pmra'].to_numpy())

        # --- Filter to stars near the HST field ---
        margin = 3000
        in_field = (np.abs(x_gaia_proj - params['x_cen']) <= margin) & \
                   (np.abs(y_gaia_proj - params['y_cen']) <= margin)
        if not np.any(in_field):
            print(f"Finished {image_name}: No stars in field.", file=original_stdout)
            return

        x_g_in    = x_gaia_proj[in_field]
        y_g_in    = y_gaia_proj[in_field]
        C_g_in    = C_pix_gaia[in_field]
        g_err_in  = gaia_err_total[in_field]
        g_mag_in  = gaia_df['gmag'].values[in_field]
        g_color_in = gaia_df['bp_rp'].values[in_field] if 'bp_rp' in gaia_df.columns else None
        g_pmra_in  = gaia_df['pmra'].values[in_field]
        g_pmdec_in = gaia_df['pmdec'].values[in_field]
        ra_in, dec_in = ra_prop[in_field], dec_prop[in_field]
        in_has_pms = has_gaia_pms[in_field]

        # Gaia quality flag: RUWE ≤ 1.4 (precomputed as clean_label during download).
        if 'clean_label' in gaia_df.columns:
            in_clean = gaia_df['clean_label'].values[in_field].astype(bool)
        elif 'ruwe' in gaia_df.columns:
            _ruwe = gaia_df['ruwe'].values[in_field]
            in_clean = np.isfinite(_ruwe) & (_ruwe <= 1.4)
        else:
            in_clean = np.ones(int(in_field.sum()), dtype=bool)

        # --- Load HST catalog ---
        hst_cat = fits.getdata(hst['catalog'])

        # Require is_star_candidate; skip image if absent
        if 'is_star_candidate' not in hst_cat.dtype.names:
            msg = (f'\n  WARNING: {image_name} catalog is missing the '
                   f'is_star_candidate column — image SKIPPED.\n'
                   f'  Re-run PSF fitting to produce updated catalogs before cross-matching.')
            print(msg)
            print(f'Finished {image_name}: SKIPPED — no is_star_candidate column.', file=original_stdout)
            return

        _orig_row_idx = np.arange(len(hst_cat))
        _valid = (np.isfinite(hst_cat['x_gdc'].astype(float)) &
                  np.isfinite(hst_cat['y_gdc'].astype(float)))
        if not _valid.all():
            print(f'  Dropping {(~_valid).sum()} NaN/inf rows from HST catalog')
            hst_cat = hst_cat[_valid]
            _orig_row_idx = _orig_row_idx[_valid]
        x_hst      = hst_cat['x_gdc'].astype(float)
        y_hst      = hst_cat['y_gdc'].astype(float)
        # Learned GDC correction (same applier and sign as the BP3M loaders: corrected = x_gdc - bias),
        # so the cross-match sees the same frame the alignment will fit (user 2026-09-30).
        if pos_corr_model:
            try:
                _pcm = _pos_corr_applier(pos_corr_model)
                _mb = _pcm.bias(hst_cat, _pcm.read_header(hst['flc']))
                if _mb is not None:
                    x_hst = x_hst - np.asarray(_mb[0], float); y_hst = y_hst - np.asarray(_mb[1], float)
                    print(f'  Learned GDC correction applied ({pos_corr_model}): median |bias| = {np.median(np.hypot(_mb[0], _mb[1])):.4f} px')
                else:
                    print(f'  Learned GDC correction: no model for this instrument/filter — positions unchanged')
            except Exception as _e:
                print(f'  WARNING: learned GDC correction failed ({_e}) — positions unchanged')
        mag_hst_gdc = hst_cat['mag_gdc'].astype(float)
        is_star    = hst_cat['is_star_candidate'].astype(bool)
        mag_err_hst = hst_cat['mag_err_gdc'].astype(float) if 'mag_err_gdc' in hst_cat.dtype.names else None
        if 'mag_st_gdc' not in hst_cat.dtype.names:
            raise ValueError(
                f"Catalog missing 'mag_st_gdc' column — stale py1pass output. "
                f"Delete the catalog so py1pass will re-run."
            )
        mag_hst = hst_cat['mag_st_gdc'].astype(float)
        mag_st_hst  = mag_hst   # kept for output saving below
        mag_ab_hst  = hst_cat['mag_ab'].astype(float)     if 'mag_ab'      in hst_cat.dtype.names else None
        C_pix_hst = np.zeros((len(x_hst), 2, 2))
        C_pix_hst[:, 0, 0] = hst_cat['cov_xx_gdc'].astype(float) + hst_pix_floor**2
        C_pix_hst[:, 1, 1] = hst_cat['cov_yy_gdc'].astype(float) + hst_pix_floor**2
        C_pix_hst[:, 0, 1] = hst_cat['cov_xy_gdc'].astype(float)
        C_pix_hst[:, 1, 0] = C_pix_hst[:, 0, 1]

        n_stars = is_star.sum()
        print(f'  HST catalog: {len(x_hst)} sources, {n_stars} star candidates ({100*n_stars/len(x_hst):.1f}%)')

        # Scale-adjusted Gaia guess positions for seeding.
        # initial_scale is the HST→Gaia scale from the 4P fit, so the inverse
        # (Gaia→HST) is 1/initial_scale — same logic as using init_inv_rot_mat.
        inv_initial_scale = 1.0 / params['initial_scale']
        xg_guess_in = params['x_cen'] + (x_g_in - params['x_cen']) * inv_initial_scale
        yg_guess_in = params['y_cen'] + (y_g_in - params['y_cen']) * inv_initial_scale

        gaia_field = {
            'x': x_g_in, 'y': y_g_in, 'C': C_g_in, 'mag': g_mag_in,
            'err': g_err_in, 'has_pms': in_has_pms,
            'xguess': xg_guess_in, 'yguess': yg_guess_in,
        }
        if guess_affine is not None:
            _M, _t = np.asarray(guess_affine[0], float), np.asarray(guess_affine[1], float)
            _c = np.array([params['x_cen'], params['y_cen']])
            _h = np.einsum('ij,nj->ni', np.linalg.inv(_M), np.column_stack([x_g_in, y_g_in]) - _t) + _c
            _sh = np.hypot(_h[:, 0] - xg_guess_in, _h[:, 1] - yg_guess_in)
            gaia_field['xguess'], gaia_field['yguess'] = _h[:, 0], _h[:, 1]
            discovery_max_offset = int(np.ceil(guess_max_offset))
            print(f"  Guess transform supplied (visit-group completion): Gaia guesses moved by median "
                  f"{np.median(_sh):.2f} px vs the header guess; offset search +-{discovery_max_offset} px")
        # 4P discovery uses high-confidence star candidates only (tight qfit/chi2
        # tiers make the geometric matching more reliable).  The 6P affine
        # refinement and final pass use all sources: non-stars contribute
        # additional positional constraints once a good transform seed exists.
        star_indices = np.where(is_star)[0]   # full-array indices of star candidates
        # CR-like seeds (2026-10-01): in images where AstroDrizzle never flagged cosmic rays (single,
        # un-associated exposures, e.g. GOODS-N ibtm9hbmq: 26.5k detections, mostly CRs), detections
        # sharper than the PSF (1x1 concentration > CR_SEED_CONC) or whose central pixel the PSF fit
        # itself sigma-clipped (concentration NaN although the pixel is not DQ-flagged) are excluded
        # from the DISCOVERY seeds only (qfit -> NaN fails every tier); refinement and the final pass
        # still see every detection.
        _q_disc = hst_cat['qfit'].astype(float).copy()
        _names = hst_cat.dtype.names
        if 'concentration' in _names and 'dq_3x3' in _names:
            _dq3 = np.asarray(hst_cat['dq_3x3'])
            if not np.any((_dq3 & 4096) != 0):
                _conc = np.asarray(hst_cat['concentration'], float)
                _dq1 = np.asarray(hst_cat['dq_1x1']) if 'dq_1x1' in _names else np.zeros(len(_conc), int)
                # ... AND a poor PSF fit: bright real stars can have their core clipped (NaN concentration)
                # yet fit perfectly (pgc_039646 j8hozqkrq: 4 of 6 Gaia stars, qfit 0.02-0.05)
                _qf = hst_cat['qfit'].astype(float)
                _cr_like = ((np.isfinite(_conc) & (_conc > CR_SEED_CONC)) | (~np.isfinite(_conc) & (_dq1 == 0))) \
                           & ~(_qf <= CR_SEED_QFIT)
                if _cr_like.any():
                    _q_disc[_cr_like] = np.nan
                    print(f"  No AstroDrizzle CR flags in this image: {int(_cr_like.sum())} of {len(_conc)} detections "
                          f"look like cosmic rays (concentration > {CR_SEED_CONC} or core clipped by the fit, with qfit > {CR_SEED_QFIT}) — "
                          f"excluded from the discovery seeds")
        hst_data = {
            'x': x_hst[is_star], 'y': y_hst[is_star], 'mag': mag_hst[is_star],
            'C': C_pix_hst[is_star],
            'qfit': _q_disc[is_star],
            'chi2': hst_cat['chi2'].astype(float)[is_star],
        }
        # All HST sources (stars + non-stars) with qfit/chi2 — used in the
        # second round of discovery tiers when the stars-only round fails.
        # Including non-stars gives more histogram pairs in sparse Gaia fields
        # where only a few star-class sources overlap with Gaia positions.
        hst_data_all_disc = {
            'x': x_hst, 'y': y_hst, 'mag': mag_hst,
            'C': C_pix_hst,
            'qfit': _q_disc,
            'chi2': hst_cat['chi2'].astype(float),
        }
        # All sources (stars + non-stars) for 6P refinement and final pass.
        hst_data_all = {
            'x': x_hst, 'y': y_hst, 'mag': mag_hst,
            'C': C_pix_hst,
        }
        tree_gaia_all = KDTree(np.column_stack([x_g_in, y_g_in]))

        # --- 4P Discovery (tiered fallback) ---
        # Within each Gaia quality tier, try HST star candidates first, then
        # all HST sources (stars + non-stars).  Only advance to a looser Gaia
        # quality when both HST variants fail.
        #   Tier 1: clean Gaia 5p / HST stars
        #   Tier 2: clean Gaia 5p / all HST
        #   Tier 3: clean Gaia 5p+2p / HST stars
        #   Tier 4: clean Gaia 5p+2p / all HST
        #   Tier 5: all Gaia / HST stars
        #   Tier 6: all Gaia / all HST  (last resort)
        # In all cases the Gaia seed only controls the offset histogram; affine
        # refinement and the final pass always use the full in-field Gaia catalog.
        _all_gaia = np.ones(len(x_g_in), dtype=bool)
        _disc_tiers = [
            # (label,               Gaia seed mask,         HST data,          stars-only flag)
            ("clean 5p / HST stars",    in_clean & in_has_pms, hst_data,          True),
            ("clean 5p / all HST",      in_clean & in_has_pms, hst_data_all_disc, False),
            ("clean 5p+2p / HST stars", in_clean,              hst_data,          True),
            ("clean 5p+2p / all HST",   in_clean,              hst_data_all_disc, False),
            ("all Gaia / HST stars",    _all_gaia,             hst_data,          True),
            ("all Gaia / all HST",      _all_gaia,             hst_data_all_disc, False),
        ]
        best, used_tier, _used_stars_only = None, None, True
        if guess_affine is not None:
            best = _guess_direct_seed(guess_affine, params, hst_data_all_disc, gaia_field,
                                      max_mag_diff, max(3, int(min_matches)))
            if best is not None:
                used_tier, _used_stars_only = 'guess transform, direct association', False
                print(f"  Guess-seeded association: {best['n_match']} Gaia-HST pairs within "
                      f"{GUESS_SEED_RADIUS_PX} px of their predicted positions (zp={best['zp']:.3f}) "
                      f"-> 6P refinement")
            else:
                print("  Guess-seeded association found < 3 pairs -> narrowed 4P discovery")
        for _tier_name, _seed_mask, _hst_d, _stars_only in ([] if best is not None else _disc_tiers):
            _n_seed = int(_seed_mask.sum())
            if _n_seed < 3:
                print(f"  Skipping tier '{_tier_name}': only {_n_seed} Gaia stars available.")
                continue
            if _hst_d is None or len(_hst_d['x']) == 0:
                print(f"  Skipping tier '{_tier_name}': no HST sources after the tier's cuts.")
                continue
            print(f"  Trying 4P discovery [{_tier_name}] ({_n_seed} Gaia in field, "
                  f"{len(_hst_d['x'])} HST sources)...")
            best = _run_4p_discovery(_hst_d, gaia_field, params, max_mag_diff, selection='evaluate',
                                      scale_sweep=scale_sweep,
                                      discovery_max_offset=discovery_max_offset,
                                      seed_quality_mask=_seed_mask,
                                      debug_verbose=True,
                                      sigma_rot_deg=sigma_rot_deg,
                                      sigma_scale=sigma_scale)
            if best is not None:
                used_tier = _tier_name
                _used_stars_only = _stars_only
                break
            print(f"  4P Discovery failed [{_tier_name}] — trying next tier...")
        if best is None:
            print(f"Finished {image_name}: 4P Discovery failed at all tiers.", file=original_stdout)
            return
        print(f"  4P Discovery Succeeded [{used_tier}]: Best Q<{best['q']}, Mag<{best['m']:.1f} "
              f"({best['n_match']} matches, red_chi2={best['red_chi2']:.2f}, red_cost={best['red_cost']:.2f})")

        def _evaluate(cand):

            # --- Affine Refinement (all sources) ---
            # Seed indices from 4P discovery index into the HST dataset used for that
            # tier.  For the stars-only round, remap to full-array indices; for the
            # all-sources round they are already full-array indices.
            if _used_stars_only:
                best_all = {**cand, 'h_v': star_indices[cand['h_v']]}
            else:
                best_all = cand
            A, B, C, D, xs_o, ys_o, xt_o, yt_o, C_params, resid_cov, zp, h_f, g_f, _init_rx, _init_ry = \
                _run_affine_refinement(best_all, hst_data_all, gaia_field, tree_gaia_all, max_mag_diff, use_resid_floor=use_resid_floor,
                                       sigma_rot_deg=sigma_rot_deg, sigma_scale=sigma_scale, sigma_skew=sigma_skew)

            # Sanity check: if the 4P seed was spurious, the Init 6P residuals
            # (on the seed pairs before any iteration inflates resid_cov) are large.
            # Correct matches have sub-pixel Init 6P residuals; wrong matches have
            # multi-pixel residuals even before the 6P iterates.
            # Since the chance-coincidence test (2026-10-01) the seed-residual gate only warns: a crowded
            # field's real seed set can be >16% mispairs (47 Tuc j8fw01b6q: 10 px at Init 6P, 0.4-0.8 px and
            # 1263 matches after refinement); spurious seeds are rejected by the chance test instead.
            if max(_init_rx, _init_ry) > init_resid_max:
                print(f"  WARNING: Init 6P seed residuals large ({_init_rx:.2f},{_init_ry:.2f}px > {init_resid_max}) — "
                      f"refining anyway; the chance-coincidence test decides")

            M = np.array([[A, B], [C, D]])

            # Plausibility of the refined plate solution vs the header (2026-10-01): with few pairs the 6P fit
            # can bend onto chance coincidences and make them look tight (47 Tuc j8c051t9q: scale -850 ppm,
            # 6 pairs "significant"); genuine solutions (282, 10 fields) sit within -277..+92 ppm, |rot| < 0.06 deg,
            # |skew| < 2.4e-4.
            _ratio = float(np.sqrt(A * D - B * C)); _rot = float(np.degrees(np.arctan2(B - C, A + D)))
            _dsc = _ratio / params['initial_scale'] - 1.0
            _on, _off = 0.5 * (A - D), 0.5 * (B + C)
            # parameter uncertainties from the refinement covariance (order A, B, xt, C, D, yt): a poorly
            # constrained low-N solution may sit far from the header without being wrong
            _V = np.asarray(C_params, float)
            _ss = float(np.sqrt(max(_V[0, 0] + _V[4, 4] + 2 * _V[0, 4], 0.0)) / 2 / max(_ratio, 1e-9))
            _sr = float(np.degrees(np.sqrt(max(_V[1, 1] + _V[3, 3] - 2 * _V[1, 3], 0.0)) / 2))
            _sk = float(np.sqrt(max(_V[0, 0] + _V[4, 4] - 2 * _V[0, 4], _V[1, 1] + _V[3, 3] + 2 * _V[1, 3], 0.0)) / 2)
            print(f"  Plate solution vs header: scale {_dsc * 1e6:+.0f} +- {_ss * 1e6:.0f} ppm, rot {_rot:+.3f} +- {_sr:.3f} deg, "
                  f"skew ({_on:+.1e},{_off:+.1e}) +- {_sk:.1e}")
            _implaus = bool(abs(_dsc) > PLAUS_SCALE + PLAUS_NSIG * _ss or abs(_rot) > PLAUS_ROT_DEG + PLAUS_NSIG * _sr
                            or max(abs(_on), abs(_off)) > PLAUS_SKEW + PLAUS_NSIG * _sk)
            if _implaus:
                print(f"  Plate solution beyond tolerance + {PLAUS_NSIG:.0f} sigma of the header — rejected unless the "
                      f"chance test finds >= {PLAUS_MAX_N} significant matches")

            # --- Final pass: gather all candidates with the converged transform ---
            xh_in_g, yh_in_g = apply_affine(x_hst, y_hst, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
            ds, g_idxs = tree_gaia_all.query(np.column_stack([xh_in_g, yh_in_g]), k=5, distance_upper_bound=100)
            h_idx_all = np.repeat(np.arange(len(x_hst)), 5)
            valid = ds.flatten() < 100
            h_v, g_v = h_idx_all[valid], g_idxs.flatten()[valid]

            dx_v, dy_v = x_g_in[g_v] - xh_in_g[h_v], y_g_in[g_v] - yh_in_g[h_v]
            C_proj = np.einsum('ij,njk,lk->nil', M, C_pix_hst[h_v], M)
            dxh_v, dyh_v = x_hst[h_v] - xs_o, y_hst[h_v] - ys_o
            J = np.zeros((len(h_v), 2, 6))
            J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh_v, dyh_v, 1.0
            J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = dxh_v, dyh_v, 1.0
            C_model = np.einsum('nij,jk,nlk->nil', J, C_params, J)
            C_total = C_g_in[g_v] + C_proj + C_model + resid_cov

            sigs_v = compute_mahalanobis(dx_v, dy_v, C_total)
            costs_v = compute_logprob_cost(dx_v, dy_v, C_total)
            mag_diffs = g_mag_in[g_v] - mag_hst[h_v]
            costs_v += ((mag_diffs - zp) / 1.0)**2
            costs_v[np.abs(mag_diffs - zp) > max_mag_diff] = np.inf

            final_mdf = pd.DataFrame({
                'h': h_v, 'g': g_v, 's': sigs_v, 'c': costs_v,
                'dx': dx_v, 'dy': dy_v, 'mag_diff': mag_diffs,
                'cxx': C_total[:, 0, 0], 'cyy': C_total[:, 1, 1],
            }).sort_values('c')
            all_mdf   = final_mdf.drop_duplicates('g')
            final_mdf = final_mdf.drop_duplicates('g').drop_duplicates('h')
            final_mdf = final_mdf[(final_mdf['s'] < 5.0) & (np.abs(final_mdf['mag_diff'] - zp) < max_mag_diff)]

            h_final, g_final = final_mdf['h'].values, final_mdf['g'].values
            print(f"  Final matches found: {len(h_final)}")
            if len(h_final) == 0:
                return dict(ok=False, reason='Final match filtering removed all stars.')
            if len(h_final) < max(3, int(min_matches)):
                return dict(ok=False, reason=f'only {len(h_final)} final matches (< {max(3, int(min_matches))}) — rejected.')

            # --- Chance-coincidence test: the same solution shifted by tens of px ---
            _md_core = final_mdf['mag_diff'].values[final_mdf['s'].values < CHANCE_SIGMA_MAX]
            if len(_md_core) == 0:
                _md_core = final_mdf['mag_diff'].values
            _zc = float(np.median(_md_core))   # window centre = the matches' own zp
            _mwin = _chance_mag_win
            _counts = []
            for _r in CHANCE_SHIFTS_PX:
                for _a in np.radians(np.arange(0, 360, 45) + (22.5 if _r != CHANCE_SHIFTS_PX[0] else 0.0)):
                    _counts.append(_final_pass_count(
                        xh_in_g, yh_in_g, x_g_in + _r * np.cos(_a), y_g_in + _r * np.sin(_a), M, C_pix_hst, C_g_in,
                        C_params, xs_o, ys_o, x_hst, y_hst, resid_cov, g_mag_in, mag_hst, zp, max_mag_diff, zp_win=_zc,
                        mag_win=_mwin))
            _n_win = int(((np.abs(final_mdf['mag_diff'].values - _zc) < min(max_mag_diff, _mwin))
                          & (final_mdf['s'].values < CHANCE_SIGMA_MAX)).sum())
            _lam, _fap = chance_significance(_n_win, _counts)
            print(f"  Chance-coincidence test: {_n_win} of {len(h_final)} matches within {CHANCE_SIGMA_MAX} sigma and {_mwin:.1f} mag of zp vs {_lam:.2f} expected by chance "
                  f"(shifted copies: median {np.median(_counts):.0f}, max {max(_counts)}) -> false-alarm P = {_fap:.2e}")
            if _implaus and _n_win < PLAUS_MAX_N:
                return dict(ok=False, n_win=_n_win, lam=_lam, fap=_fap,
                            reason=f'implausible plate solution (scale {_dsc * 1e6:+.0f} ppm, rot {_rot:+.3f} deg) with only {_n_win} significant matches — rejected.')
            if _fap >= CHANCE_FA_PROB:
                return dict(ok=False, n_win=_n_win, lam=_lam, fap=_fap,
                            reason=f'match set not significant against chance ({_n_win} vs {_lam:.1f} expected, P={_fap:.2e}) — spurious solution rejected.')
            return dict(ok=True, cand=cand, A=A, B=B, C=C, D=D, xs_o=xs_o, ys_o=ys_o, xt_o=xt_o, yt_o=yt_o, M=M,
                        C_params=C_params, resid_cov=resid_cov, zp=zp, xh_in_g=xh_in_g, yh_in_g=yh_in_g,
                        final_mdf=final_mdf, all_mdf=all_mdf, h_final=h_final, g_final=g_final,
                        _lam=_lam, _fap=_fap, n_win=_n_win)


        _cands = [best] + list(best.get('_alternatives', []))[:MAX_DISCOVERY_CANDIDATES - 1]
        _results = []
        for _k, _cand in enumerate(_cands):
            if len(_cands) > 1:
                print(f"  --- candidate {_k + 1}/{len(_cands)}: q<{_cand['q']} m<{_cand['m']:.1f} ({_cand['n_match']} seeds) ---")
            _results.append(_evaluate(_cand))
        _good = [r for r in _results if r['ok']]
        if not _good:
            print(f"Finished {image_name}: {_results[0]['reason']}", file=original_stdout)
            return
        _pick = min(_good, key=lambda r: (r['_fap'], -r['n_win']))
        if len(_cands) > 1:
            print(f"  Candidate chosen: {_results.index(_pick) + 1}/{len(_cands)} ({_pick['n_win']} matches vs "
                  f"{_pick['_lam']:.1f} by chance, P={_pick['_fap']:.2e})")
        best = _pick['cand']
        A, B, C, D = _pick['A'], _pick['B'], _pick['C'], _pick['D']
        xs_o, ys_o, xt_o, yt_o = _pick['xs_o'], _pick['ys_o'], _pick['xt_o'], _pick['yt_o']
        M, C_params, resid_cov, zp = _pick['M'], _pick['C_params'], _pick['resid_cov'], _pick['zp']
        xh_in_g, yh_in_g = _pick['xh_in_g'], _pick['yh_in_g']
        final_mdf, all_mdf = _pick['final_mdf'], _pick['all_mdf']
        h_final, g_final = _pick['h_final'], _pick['g_final']
        _lam, _fap = _pick['_lam'], _pick['_fap']

        # --- Save offset histogram plot for the chosen discovery tier ---
        _save_offset_histogram(best, image_name, hst['root'])

        # --- Build diagnostic dataframe ---
        diag_df = pd.DataFrame({
            'h_idx': all_mdf['h'], 'g_idx': all_mdf['g'],
            'x': x_g_in[all_mdf['g']], 'y': y_g_in[all_mdf['g']],
            'hx': xh_in_g[all_mdf['h']], 'hy': yh_in_g[all_mdf['h']],
            'ra': ra_in[all_mdf['g']], 'dec': dec_in[all_mdf['g']],
            'dx': all_mdf['dx'], 'dy': all_mdf['dy'],
            'sigma': all_mdf['s'], 'cxx': all_mdf['cxx'], 'cyy': all_mdf['cyy'],
            'mag': g_mag_in[all_mdf['g']], 'mag_hst': mag_hst[all_mdf['h']],
            'pmra': g_pmra_in[all_mdf['g']], 'pmdec': g_pmdec_in[all_mdf['g']],
            'hst_is_star': is_star[all_mdf['h'].values],
        })
        if g_color_in is not None:
            diag_df['color'] = g_color_in[diag_df['g_idx']]
        diag_df['color_hst'] = diag_df['mag'] - diag_df['mag_hst'] - zp

        final_match_keys = set(zip(h_final, g_final))
        is_m = diag_df.apply(lambda r: (int(r.h_idx), int(r.g_idx)) in final_match_keys, axis=1)
        if not np.any(is_m):
            print(f"Finished {image_name}: Final match filtering removed all stars.", file=original_stdout)
            return

        # --- Save outputs ---
        save_diagnostic_plots(hst['root'], image_name, diag_df[is_m], diag_df[~is_m])
        final_matches = diag_df[is_m].copy()

        output = Table()
        output['hst_index']      = _orig_row_idx[final_matches['h_idx'].values]
        output['hst_x_gdc']      = x_hst[final_matches['h_idx'].values]
        output['hst_y_gdc']      = y_hst[final_matches['h_idx'].values]
        output['hst_mag_gdc']    = mag_hst_gdc[final_matches['h_idx'].values]
        if mag_err_hst is not None:
            output['hst_mag_err_gdc']= mag_err_hst[final_matches['h_idx'].values]
        if mag_st_hst is not None:
            output['hst_mag_st_gdc'] = mag_st_hst[final_matches['h_idx'].values]
        if mag_ab_hst is not None:
            output['hst_mag_ab']     = mag_ab_hst[final_matches['h_idx'].values]
        output['gaia_source_id'] = gaia_df.iloc[in_field]['source_id'].to_numpy(dtype=np.int64)[final_matches['g_idx'].values]
        output['has_gaia_pms']   = has_gaia_pms[in_field][final_matches['g_idx'].values]
        output['gaia_ra_prop']   = ra_in[final_matches['g_idx'].values]
        output['gaia_dec_prop']  = dec_in[final_matches['g_idx'].values]
        output['gaia_gmag']      = g_mag_in[final_matches['g_idx'].values]
        # residual_mag uses the same calibrated magnitude that was used for ZP
        # estimation (mag_hst = mag_st_gdc when available, else mag_gdc).
        mag_hst_for_resid = mag_hst[final_matches['h_idx'].values]
        output['residual_mag']   = output['gaia_gmag'] - (mag_hst_for_resid + zp)
        output['residual_x']     = final_matches['dx'].values
        output['residual_y']     = final_matches['dy'].values
        output['residual_sigma'] = final_matches['sigma'].values
        output['hst_is_star']    = is_star[final_matches['h_idx'].values]
        output.write(os.path.join(hst['root'], "matched_gaia.csv"), format='ascii.csv', overwrite=True)

        ratio, rot = np.sqrt(A*D - B*C), np.degrees(np.arctan2(B-C, A+D))
        on_skew, off_skew = 0.5*(A-D), 0.5*(B+C)
        trans_out = Table()
        trans_out['parameter'] = ['A','B','C','D','xs_o','ys_o','xt_o','yt_o',
                                   'ratio','rot_deg','on_skew','off_skew','zp',
                                   'ra_cen','dec_cen','x_cen','y_cen','pixel_scale','orientat']
        trans_out['value'] = [A, B, C, D, xs_o, ys_o, xt_o, yt_o,
                               ratio, rot, on_skew, off_skew, zp,
                               params['ra_cen'], params['dec_cen'],
                               params['x_cen'], params['y_cen'],
                               params['pixel_scale'], params['orientat']]
        trans_out.add_row(['n_chance_expected', _lam]); trans_out.add_row(['chance_fa_prob', _fap])
        trans_out.write(os.path.join(hst['root'], "transformation.csv"), format='ascii.csv', overwrite=True)
        print(f"Finished {image_name}: Found {len(final_matches)} matches in {time.time()-start_time:.2f}s.", file=original_stdout)

    except Exception as e:
        import traceback
        print(f"Finished {image_name}: Error - {e}", file=original_stdout)
        traceback.print_exc(file=original_stdout)
    finally:
        try:
            if hasattr(sys.stdout, 'log'):
                sys.stdout.log.close()   # FileLogger never closed its handle
        except Exception:
            pass
        sys.stdout = original_stdout

def main():
    parser = argparse.ArgumentParser(description="Parallel Gaia-HST catalog cross-matcher with covariance weighting and magnitude rejection.")
    parser.add_argument("--target", required=True,
                        help="Target name (e.g. Fornax_dSph). Expects data in [data_dir]/[target]/Gaia and [data_dir]/[target]/HST")
    parser.add_argument("--data-dir", default="./data",
                        help="Root directory containing target data folders. Default: ./data")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="Number of parallel processing threads. Default: All available cores")
    parser.add_argument("--hst-pix-floor", type=float, default=0.01,
                        help="Minimum HST positional uncertainty (pixels) added in quadrature to reported errors. Default: 0.01")
    parser.add_argument("--min-matches", type=int, default=3,
                        help="Minimum number of seeds required for initial match. Default: 3")
    parser.add_argument("--zero-gaia-pm", action="store_true",
                        help="Set all Gaia PMs and Parallaxes to 0 with large default uncertainties. Useful for debugging.")
    parser.add_argument("--scale-sweep", action="store_true",
                        help="Enable simultaneous scale+offset sweep during 4P discovery (slower but more robust when pixel scale is uncertain).")
    parser.add_argument("--image", type=str, default=None,
                        help="Process only this image (by observation ID). Useful for debugging.")

    args = parser.parse_args()
    gaia_df = load_gaia_data(args.target, args.data_dir)
    if gaia_df is None: return

    hst_folders = find_hst_image_folders(args.target, args.data_dir)
    if args.image:
        hst_folders = [h for h in hst_folders if h['root'].split('/')[-1] == args.image]
        if not hst_folders:
            print(f"No image folder found for '{args.image}'"); return
    print(f"Found {len(hst_folders)} images. Processing with {args.threads} threads...")

    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_single_image, hst, gaia_df, args.hst_pix_floor, args.min_matches, args.zero_gaia_pm, scale_sweep=args.scale_sweep): hst for hst in hst_folders}
        for f in as_completed(futures):
            f.result()
    print("All tasks completed.")

    from .validator import validate_target
    print("\n--- Running cross-image validation ---")
    validate_target(args.target, args.data_dir)

if __name__ == "__main__": main()
