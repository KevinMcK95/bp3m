"""
eGSF (extended-source galaxy catalogue) post-processor.

Uses v1 BP3M alignment outputs and pypass FLC catalogs to:
  1. Group images by epoch: same instrument / detector / filter / PA_V3 (within pa_tol deg)
  2. Identify background galaxy candidates via pypass morphology metrics
     (is_star_candidate=False, high chi2 relative to stellar locus, high concentration)
  3. Cross-match candidates across epochs within each group by sky position
  4. Build a catalogue of multi-epoch galaxy detections with position scatter statistics

Output: {output_dir}/{field_name}_galaxy_candidates.csv
        {output_dir}/{field_name}_epoch_groups.csv

Usage (CLI entry point wired in bp3m_run.py via --run_egsf):
  from bp3m.pipeline.egsf import run_egsf
  run_egsf(field_name, output_dir)

Standalone:
  python -m bp3m.pipeline.egsf Leo_I /path/to/GaiaHub_results/Leo_I
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.signal import fftconvolve  # noqa: F401 (imported here; used in _render_sersic_psf)
import scipy.optimize  # noqa: F401


# ── Constants ─────────────────────────────────────────────────────────────────

_QUALITY_MAG_MIN  = 15.0   # mag_st lower bound (avoid saturated)
_QUALITY_MAG_MAX  = 30.0   # mag_st upper bound
_EXPTIME_MIN      = 1.0    # skip zero-exptime FLCs (bias/dark frames)


# ── Header helpers ─────────────────────────────────────────────────────────────

def _get_filter(h0: fits.Header) -> str:
    """Extract real filter name from FLC primary header (handles ACS CLEAR1L)."""
    inst = h0.get('INSTRUME', '')
    if inst == 'ACS':
        f1 = h0.get('FILTER1', '')
        f2 = h0.get('FILTER2', '')
        return f2 if 'CLEAR' in str(f1).upper() else (f1 if 'CLEAR' in str(f2).upper() else f1)
    return h0.get('FILTER', '')


# ── Image metadata ─────────────────────────────────────────────────────────────

def _get_image_metadata(field_dir: Path, bp3m_results_subdir: str = 'BP3M_results') -> pd.DataFrame:
    """
    Return DataFrame of per-FLC metadata indexed by obs_id.

    Columns: obs_id, flc_path, instrument, detector, filter, pa_v3, exptime
    Only includes obs_ids present in BP3M_results/image_transformations.csv
    (i.e., images that were actually used in the v1 alignment).
    """
    xform_path = field_dir / bp3m_results_subdir / 'image_transformations.csv'
    if not xform_path.exists():
        raise FileNotFoundError(f"image_transformations.csv not found: {xform_path}")

    xform = pd.read_csv(xform_path)
    # image_name is like 'j9gz01orq_hi' or 'j9gz01orq_lo' — strip _hi/_lo suffix
    obs_ids_used = {name.rsplit('_', 1)[0] for name in xform['image_name']}

    hst_dir = field_dir / 'HST' / 'mastDownload' / 'HST'
    rows = []
    for obs_dir in sorted(hst_dir.iterdir()):
        obs_id = obs_dir.name
        if obs_id not in obs_ids_used:
            continue
        flc = list(obs_dir.glob('*_flc.fits'))
        if not flc:
            continue
        with fits.open(flc[0]) as hdul:
            h0 = hdul[0].header
            exptime = float(h0.get('EXPTIME', 0.0))
            if exptime < _EXPTIME_MIN:
                continue
            rows.append(dict(
                obs_id=obs_id,
                flc_path=str(flc[0]),
                instrument=h0.get('INSTRUME', ''),
                detector=h0.get('DETECTOR', ''),
                filter=_get_filter(h0),
                pa_v3=float(h0.get('PA_V3', np.nan)),
                exptime=exptime,
            ))

    return pd.DataFrame(rows)


# ── Epoch grouping ─────────────────────────────────────────────────────────────

def _pa_bin(pa: float, pa_tol: float) -> float:
    """Round PA_V3 to nearest pa_tol-degree bin."""
    return round(pa / pa_tol) * pa_tol


def group_images_by_epoch(
    field_dir: Path,
    bp3m_results_subdir: str = 'BP3M_results',
    pa_tol: float = 5.0,
) -> dict[str, list[str]]:
    """
    Group FLC images into epoch groups by (instrument/detector/filter/PA_V3 bin).

    Returns dict: epoch_key → list of obs_ids.
    epoch_key format: 'ACS/WFC/F814W/PA135'
    """
    meta = _get_image_metadata(field_dir, bp3m_results_subdir)
    groups: dict[str, list[str]] = {}
    for _, row in meta.iterrows():
        pa_b = _pa_bin(row['pa_v3'], pa_tol)
        key = f"{row['instrument']}/{row['detector']}/{row['filter']}/PA{pa_b:.0f}"
        groups.setdefault(key, []).append(row['obs_id'])
    return groups


# ── Gaia-matched source index lookup ──────────────────────────────────────────

def _build_gaia_matched_set(cross_match_path: Path) -> dict[str, set[int]]:
    """
    Parse cross_match_catalog.csv to build {obs_id: set of pypass row indices}
    for all Gaia-matched sources.
    """
    cc = pd.read_csv(cross_match_path)
    gaia_set: dict[str, set[int]] = {}
    for _, row in cc.iterrows():
        imgs = str(row['image_list']).split(',')
        idxs = str(row['hst_index_list']).split(',')
        for img, idx in zip(imgs, idxs):
            img = img.strip()
            try:
                i = int(idx.strip())
            except ValueError:
                continue
            gaia_set.setdefault(img, set()).add(i)
    return gaia_set


# ── Per-image galaxy candidate extraction ─────────────────────────────────────

def _stellar_chi2_cut(chi2: np.ndarray, is_star: np.ndarray, n_sigma: float) -> float:
    """
    Compute chi2 threshold as stellar_median + n_sigma * stellar_mad.
    Falls back to a fixed value if stellar locus is empty.
    """
    star_chi2 = chi2[is_star & np.isfinite(chi2)]
    if len(star_chi2) < 10:
        return 2.0
    med = np.median(star_chi2)
    mad = np.median(np.abs(star_chi2 - med))
    return med + n_sigma * 1.4826 * mad  # 1.4826 converts MAD to sigma


def load_galaxy_candidates(
    flc_path: str,
    obs_id: str,
    gaia_indices: set[int],
    chi2_nsigma: float = 3.0,
    conc_cut: float = 1.3,
) -> pd.DataFrame:
    """
    Load pypass FLC catalog and return galaxy candidate rows.

    Galaxy candidates are sources that are:
      - converged, not saturated, in magnitude range
      - NOT Gaia-matched  OR  classified as non-star by pypass
      - Have is_star_candidate=False OR chi2 > stellar_locus + n_sigma OR concentration > conc_cut

    Returns DataFrame with columns:
      obs_id, pypass_idx, ra, dec, x, y, chip_ext, mag_st,
      chi2, concentration, is_star_candidate, is_gaia_matched,
      sigma_x_model, sigma_y_model, x_gdc, y_gdc
    """
    try:
        t = Table.read(flc_path)
    except Exception as e:
        warnings.warn(f"Cannot read {flc_path}: {e}")
        return pd.DataFrame()

    n = len(t)
    pypass_idx = np.arange(n)

    # Quality mask
    converged = np.asarray(t['converged']).astype(bool)
    n_sat = np.asarray(t['n_sat']).astype(int)
    mag_st = np.asarray(t['mag_st']).astype(float)

    quality = converged & (n_sat == 0) & (mag_st > _QUALITY_MAG_MIN) & (mag_st < _QUALITY_MAG_MAX)

    # Morphology
    chi2 = np.asarray(t['chi2']).astype(float)
    conc = np.asarray(t['concentration']).astype(float)
    is_star = np.asarray(t['is_star_candidate']).astype(bool)

    is_gaia = np.zeros(n, dtype=bool)
    for i in gaia_indices:
        if 0 <= i < n:
            is_gaia[i] = True

    # Stellar locus chi2 threshold (computed from quality-passing stars)
    chi2_thresh = _stellar_chi2_cut(chi2[quality], is_star[quality], chi2_nsigma)

    # Galaxy candidate: morphologically extended relative to stellar locus
    morph_galaxy = (~is_star) | (chi2 > chi2_thresh) | (conc > conc_cut)

    # Select: quality-passing, morphologically extended
    # Prioritise non-Gaia sources, but also include Gaia sources flagged as non-star
    sel = quality & morph_galaxy

    if sel.sum() == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        'obs_id':            obs_id,
        'pypass_idx':        pypass_idx[sel],
        'ra':                np.asarray(t['ra'])[sel],
        'dec':               np.asarray(t['dec'])[sel],
        'x':                 np.asarray(t['x'])[sel],
        'y':                 np.asarray(t['y'])[sel],
        'chip_ext':          np.asarray(t['chip_ext'])[sel],
        'mag_st':            mag_st[sel],
        'chi2':              chi2[sel],
        'concentration':     conc[sel],
        'is_star_candidate': is_star[sel],
        'is_gaia_matched':   is_gaia[sel],
        'sigma_x_model':     np.asarray(t['sigma_x_model'])[sel],
        'sigma_y_model':     np.asarray(t['sigma_y_model'])[sel],
        'x_gdc':             np.asarray(t['x_gdc'])[sel],
        'y_gdc':             np.asarray(t['y_gdc'])[sel],
    })

    return df


# ── Cross-match across epochs ──────────────────────────────────────────────────

def _cross_match_epoch_group(
    epoch_dfs: list[pd.DataFrame],
    match_radius_arcsec: float = 0.5,
    min_detections: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-match galaxy candidates across all images in one epoch group.

    Uses greedy accumulation: start from the richest image, then sequentially
    match each remaining image to the accumulated reference centroids.

    Returns
    -------
    sources_df : per-source summary DataFrame (one row per unique galaxy)
        Columns: source_id, epoch_key (set by caller), ra_med, dec_med,
                 n_detections, ra_rms_mas, dec_rms_mas, pos_rms_mas,
                 chi2_mean, conc_mean, mag_st_mean,
                 is_gaia_matched_any, is_star_candidate_any, obs_ids
    detections_df : per-detection DataFrame (one row per epoch detection)
        Columns: source_id, obs_id, pypass_idx, x, y, chip_ext,
                 ra, dec, chi2, concentration, mag_st,
                 is_gaia_matched, is_star_candidate,
                 sigma_x_model, sigma_y_model, x_gdc, y_gdc
        Raw chip (x, y) and chip_ext are preserved here for 5×5 cutout extraction.
    """
    if not epoch_dfs:
        return pd.DataFrame(), pd.DataFrame()

    # Sort by number of candidates descending so the reference is the richest image
    epoch_dfs = sorted(epoch_dfs, key=len, reverse=True)

    # Each cluster stores full detection rows (preserving x/y/chip_ext for cutouts)
    DET_COLS = ['obs_id', 'pypass_idx', 'x', 'y', 'chip_ext',
                'ra', 'dec', 'chi2', 'concentration', 'mag_st',
                'is_gaia_matched', 'is_star_candidate',
                'sigma_x_model', 'sigma_y_model', 'x_gdc', 'y_gdc']

    clusters: list[list[dict]] = [
        [row[DET_COLS].to_dict()]
        for _, row in epoch_dfs[0].iterrows()
    ]

    for df_j in epoch_dfs[1:]:
        if len(df_j) == 0:
            continue
        ref_ra  = np.array([np.mean([d['ra']  for d in c]) for c in clusters])
        ref_dec = np.array([np.mean([d['dec'] for d in c]) for c in clusters])
        ref_coords = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg)

        new_coords = SkyCoord(
            ra=df_j['ra'].to_numpy() * u.deg,
            dec=df_j['dec'].to_numpy() * u.deg,
        )

        idx, sep, _ = new_coords.match_to_catalog_sky(ref_coords)
        matched = sep.arcsec <= match_radius_arcsec

        new_clusters = []
        for k, (_, row) in enumerate(df_j.iterrows()):
            det = row[DET_COLS].to_dict()
            if matched[k]:
                clusters[idx[k]].append(det)
            else:
                new_clusters.append([det])
        clusters.extend(new_clusters)

    # Build output tables
    source_rows = []
    det_rows = []

    for src_id, cluster in enumerate(clusters):
        if len(cluster) < min_detections:
            continue

        ras  = np.array([d['ra']  for d in cluster])
        decs = np.array([d['dec'] for d in cluster])
        ra_med  = np.median(ras)
        dec_med = np.median(decs)

        cos_dec = np.cos(np.radians(dec_med))
        ra_rms_mas  = np.std(ras)  * 3600e3 * cos_dec
        dec_rms_mas = np.std(decs) * 3600e3
        pos_rms_mas = np.sqrt(ra_rms_mas**2 + dec_rms_mas**2) / np.sqrt(2)

        source_rows.append(dict(
            source_id=src_id,
            ra_med=ra_med,
            dec_med=dec_med,
            n_detections=len(cluster),
            ra_rms_mas=ra_rms_mas,
            dec_rms_mas=dec_rms_mas,
            pos_rms_mas=pos_rms_mas,
            chi2_mean=float(np.nanmean([d['chi2'] for d in cluster])),
            conc_mean=float(np.nanmean([d['concentration'] for d in cluster])),
            mag_st_mean=float(np.nanmean([d['mag_st'] for d in cluster])),
            is_gaia_matched_any=any(d['is_gaia_matched'] for d in cluster),
            is_star_candidate_any=any(d['is_star_candidate'] for d in cluster),
            obs_ids=','.join(sorted({d['obs_id'] for d in cluster})),
        ))

        for det in cluster:
            det_rows.append({'source_id': src_id, **det})

    sources_df    = pd.DataFrame(source_rows)
    detections_df = pd.DataFrame(det_rows)
    return sources_df, detections_df


# ── ePSF constants ────────────────────────────────────────────────────────────

_CUTOUT_HALF = 4    # 9×9 pixel cutout (±4 pixels)
_WEIGHT_SIGMA = 1.5  # Gaussian weight sigma in pixels for moment measurement


# ── ePSF model construction ───────────────────────────────────────────────────

def _load_image_psf(
    obs_dir: Path,
    flc_path: str,
    lib_dir: Path,
) -> tuple | None:
    """
    Load the STDPSF model (+ psf_delta correction) for one FLC image.

    Uses the same logic as psf_fitting.py: find_psf(psf_dir, header) selects
    the correct STDPSF file for the instrument/detector/filter/epoch, then
    loads it with load_stdpsf.  If psf_delta.npy exists in obs_dir it is added
    to every PSF in the cube as a spatially-uniform empirical correction.

    Returns (psf_cube, xs, ys, psf_scale) or None on failure.
    psf_cube : (n_psf, size, size) float64 supersampled PSF array
    xs, ys   : detector coordinates of PSF grid nodes
    psf_scale: integer supersampling factor (typically 4)
    """
    try:
        from pypass.io import load_stdpsf, find_psf, _DETECTOR_PREFIX
    except ImportError as e:
        warnings.warn(f"Cannot import pypass: {e}")
        return None

    try:
        with fits.open(flc_path) as hdul:
            h0 = hdul[0].header
        instrume = h0.get('INSTRUME', '').strip().upper()
        detector = h0.get('DETECTOR', '').strip().upper()

        det_prefix = _DETECTOR_PREFIX.get((instrume, detector))
        if det_prefix is None:
            return None

        psf_dir = lib_dir / 'STDPSFs' / det_prefix
        psf_path = find_psf(str(psf_dir), h0)
        psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)

        # Apply psf_delta correction if it exists for this image
        delta_path = obs_dir / 'psf_delta.npy'
        if delta_path.exists():
            psf_delta = np.load(str(delta_path))
            psf_cube = psf_cube + psf_delta[np.newaxis, :, :]

        return psf_cube, xs, ys, psf_scale

    except Exception as e:
        warnings.warn(f"PSF load failed for {obs_dir.name}: {e}")
        return None


def _eval_psf_on_window(
    psf_cube: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    psf_scale: int,
    x: float,
    y: float,
    sci_shape: tuple[int, int],
    hw: int = _CUTOUT_HALF,
) -> tuple[np.ndarray, int, int, int, int] | None:
    """
    Evaluate the spatially-interpolated STDPSF on a pixel window centred at (x, y).

    Uses the same call sequence as pypass._psf_window:
      interpolate_psf → spline_filter → _eval_psf_grad_fast

    Returns (P, y_lo, y_hi, x_lo, x_hi) where P has shape (y_hi-y_lo, x_hi-x_lo).
    Returns None if the source is too close to the chip edge.
    """
    try:
        from pypass.core import interpolate_psf, _eval_psf_grad_fast, _window_offsets
        from scipy.ndimage import spline_filter as _spline_filter
    except ImportError:
        return None

    ny, nx = sci_shape
    xi = int(round(x))
    yi = int(round(y))
    y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, hw, ny, nx)

    # Reject if the window was clipped (source too close to edge)
    if (y_hi - y_lo) != (2 * hw + 1) or (x_hi - x_lo) != (2 * hw + 1):
        return None

    dx = x - xi
    dy = y - yi

    local_psf = interpolate_psf(psf_cube, xs, ys, x, y)
    coeffs = _spline_filter(local_psf, order=3, output=np.float64)
    P, _, _ = _eval_psf_grad_fast(coeffs, dx, dy, dix, diy, psf_scale)
    return P, y_lo, y_hi, x_lo, x_hi


def _moments_2d(
    image: np.ndarray,
    weight_sigma: float = _WEIGHT_SIGMA,
    inv_var: np.ndarray | None = None,
) -> dict:
    """
    Gaussian-weighted 2nd moments of a square 2D image.

    image     : (n, n) float array — the signal (sky already subtracted)
    inv_var   : (n, n) inverse variance for noise weighting, or None for uniform
    weight_sigma: Gaussian spatial weight sigma in pixels

    Returns dict with M_xx, M_yy, M_xy (px²), total_flux, snr.
    """
    n = image.shape[0]
    c = n // 2
    xi = np.arange(n) - c
    dx, dy = np.meshgrid(xi, xi)  # dx=col offset, dy=row offset

    g = np.exp(-0.5 * (dx**2 + dy**2) / weight_sigma**2)
    w = g if inv_var is None else g * np.maximum(inv_var, 0.0)

    f = np.maximum(image, 0.0)
    wf = w * f
    wf_sum = wf.sum()
    if wf_sum <= 0:
        return {'M_xx': np.nan, 'M_yy': np.nan, 'M_xy': np.nan,
                'total_flux': np.nan, 'snr': np.nan}

    M_xx = float((wf * dx**2).sum() / wf_sum)
    M_yy = float((wf * dy**2).sum() / wf_sum)
    M_xy = float((wf * dx * dy).sum() / wf_sum)
    total_flux = float(image.sum())
    snr = float(total_flux / np.sqrt(max((1.0 / (inv_var + 1e-10)).sum(), 1e-6))) \
          if inv_var is not None else np.nan

    return {'M_xx': M_xx, 'M_yy': M_yy, 'M_xy': M_xy,
            'total_flux': total_flux, 'snr': snr}


def _deconvolve_moments(gal_mom: dict, psf_mom: dict) -> dict:
    """
    Deconvolve PSF moments from galaxy moments (Gaussian approximation).

    For a Gaussian source: M_int = M_obs - M_PSF (quadrature subtraction).
    Ellipticity uses the e1/e2 (Stokes-like) definition.

    Returns dict with sigma_obs, sigma_int, e1_obs, e2_obs, e1_int, e2_int,
                       size_ratio (sqrt(T_int / T_PSF); >1 = spatially resolved).
    """
    Mxx_g = gal_mom['M_xx']; Myy_g = gal_mom['M_yy']; Mxy_g = gal_mom['M_xy']
    Mxx_p = psf_mom['M_xx']; Myy_p = psf_mom['M_yy']; Mxy_p = psf_mom['M_xy']

    T_obs = Mxx_g + Myy_g
    e1_obs = (Mxx_g - Myy_g) / (T_obs + 1e-10)
    e2_obs = 2 * Mxy_g        / (T_obs + 1e-10)

    Mxx_i = Mxx_g - Mxx_p
    Myy_i = Myy_g - Myy_p
    Mxy_i = Mxy_g - Mxy_p
    T_int = Mxx_i + Myy_i
    T_psf = Mxx_p + Myy_p

    e1_int = (Mxx_i - Myy_i) / (T_int + 1e-10) if T_int > 0 else np.nan
    e2_int = 2 * Mxy_i        / (T_int + 1e-10) if T_int > 0 else np.nan
    size_ratio = np.sqrt(max(T_int, 0.0) / (T_psf + 1e-10)) if T_psf > 0 else np.nan

    return dict(
        sigma_obs_x=np.sqrt(max(Mxx_g, 0.0)), sigma_obs_y=np.sqrt(max(Myy_g, 0.0)),
        e1_obs=e1_obs, e2_obs=e2_obs,
        sigma_int_x=np.sqrt(max(Mxx_i, 0.0)), sigma_int_y=np.sqrt(max(Myy_i, 0.0)),
        e1_int=e1_int, e2_int=e2_int,
        size_ratio=size_ratio,
        psf_sigma_x=np.sqrt(max(Mxx_p, 0.0)), psf_sigma_y=np.sqrt(max(Myy_p, 0.0)),
    )


# ── Galaxy moment measurement ──────────────────────────────────────────────────

# ── Extension index for the residual FITS file ────────────────────────────────
# _flc_residual.fits layout:
#   ext 1 = SCI1 (chip_ext=1 residual), ext 2 = VAR1, ext 3 = MASK1
#   ext 4 = SCI4 (chip_ext=4 residual), ext 5 = VAR4, ext 6 = MASK4
# For WFC3 UVIS chip_ext is also 1 and 4 (two-chip detector, same layout).

def _residual_ext_indices(chip_ext: int) -> tuple[int, int, int]:
    """Return (sci_ext, var_ext, mask_ext) for the given chip_ext in residual.fits."""
    if chip_ext == 1:
        return 1, 2, 3
    elif chip_ext == 4:
        return 4, 5, 6
    else:
        # Fall back: assume SCI is at chip_ext
        return chip_ext, chip_ext + 1, chip_ext + 2


def _load_residual_chip(residual_path: str, chip_ext: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load (sci_data, var_data) for one chip from a _flc_residual.fits file.

    Returns None if file not found or extension missing.
    sci_data: neighbor-subtracted residual image (float32)
    var_data: variance image for inverse-variance weighting (float32)
    """
    sci_ext, var_ext, _ = _residual_ext_indices(chip_ext)
    try:
        with fits.open(residual_path) as hdul:
            sci = hdul[sci_ext].data.astype(np.float32)
            var = hdul[var_ext].data.astype(np.float32)
    except Exception:
        return None
    return sci, var


def _batch_extract_cutouts(
    sci: np.ndarray,
    var: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    half_size: int = _CUTOUT_HALF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized extraction of all cutouts from one chip image.

    Parameters
    ----------
    sci, var : (nrow, ncol) arrays
    xs, ys  : pixel coordinates of N sources (float)
    half_size: half-width in pixels (cutout = (2h+1)×(2h+1))

    Returns
    -------
    sci_cutouts : (N, 2h+1, 2h+1) float32 — NaN where out of bounds
    var_cutouts : (N, 2h+1, 2h+1) float32
    valid       : (N,) bool — True where cutout is fully in bounds
    """
    n = 2 * half_size + 1
    N = len(xs)
    sci_out = np.full((N, n, n), np.nan, dtype=np.float32)
    var_out = np.full((N, n, n), np.nan, dtype=np.float32)
    valid   = np.zeros(N, dtype=bool)

    nrow, ncol = sci.shape
    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)

    for i in range(N):
        x0, x1 = xi[i] - half_size, xi[i] + half_size + 1
        y0, y1 = yi[i] - half_size, yi[i] + half_size + 1
        if x0 >= 0 and y0 >= 0 and x1 <= ncol and y1 <= nrow:
            sci_out[i] = sci[y0:y1, x0:x1]
            var_out[i] = var[y0:y1, x0:x1]
            valid[i] = True

    return sci_out, var_out, valid


def _batch_moments(
    sci_cutouts: np.ndarray,
    var_cutouts: np.ndarray,
    weight_sigma: float = _WEIGHT_SIGMA,
) -> dict[str, np.ndarray]:
    """
    Vectorized Gaussian-weighted 2nd moment measurement for a batch of cutouts.

    Uses inverse-variance weights × Gaussian spatial weight so that high-noise
    pixels (e.g. from saturated neighbors) don't dominate.

    Parameters
    ----------
    sci_cutouts : (N, n, n)
    var_cutouts : (N, n, n)  — variance per pixel
    weight_sigma: Gaussian spatial weight sigma in pixels

    Returns dict of (N,) arrays: M_xx, M_yy, M_xy, total_flux, moment_snr
    """
    N, n, _ = sci_cutouts.shape
    c = n // 2
    xi = np.arange(n) - c
    dx, dy = np.meshgrid(xi, xi)  # (n, n), dx=column offset, dy=row offset

    # Spatial Gaussian weight: (n, n) — same for all sources
    g_weight = np.exp(-0.5 * (dx**2 + dy**2) / weight_sigma**2)  # (n, n)

    # Inverse-variance weight: (N, n, n)
    inv_var = np.where(var_cutouts > 0, 1.0 / np.maximum(var_cutouts, 1e-6), 0.0)

    # Combined weight: (N, n, n)
    w = g_weight[np.newaxis, :, :] * inv_var

    # Sky estimate: median of 5×5 border pixels per cutout
    border = np.zeros((N, n, n), dtype=bool)
    border[:, 0, :] = True; border[:, -1, :] = True
    border[:, :, 0] = True; border[:, :, -1] = True
    # Compute per-cutout sky median using border pixels
    sky_arr = np.array([
        np.nanmedian(sci_cutouts[i][border[i]])
        for i in range(N)
    ], dtype=np.float32)
    f = sci_cutouts - sky_arr[:, np.newaxis, np.newaxis]  # (N, n, n)

    wf = w * f  # (N, n, n)
    wf_sum = wf.sum(axis=(1, 2))  # (N,)

    safe = wf_sum > 0
    wf_sum_safe = np.where(safe, wf_sum, 1.0)

    M_xx = (wf * dx[np.newaxis]**2).sum(axis=(1, 2)) / wf_sum_safe
    M_yy = (wf * dy[np.newaxis]**2).sum(axis=(1, 2)) / wf_sum_safe
    M_xy = (wf * dx[np.newaxis] * dy[np.newaxis]).sum(axis=(1, 2)) / wf_sum_safe

    M_xx[~safe] = np.nan
    M_yy[~safe] = np.nan
    M_xy[~safe] = np.nan

    total_flux = f.sum(axis=(1, 2))
    # Noise: sqrt(sum of variance over aperture) as proxy
    var_sum = np.maximum(var_cutouts.sum(axis=(1, 2)), 1e-6)
    moment_snr = total_flux / np.sqrt(var_sum)

    return {'M_xx': M_xx, 'M_yy': M_yy, 'M_xy': M_xy,
            'total_flux': total_flux, 'moment_snr': moment_snr}


def measure_galaxy_morphology(
    field_name: str,
    output_dir: str | Path,
    lib_dir: str | Path | None = None,
    bp3m_results_subdir: str = 'BP3M_results',
    pa_tol: float = 5.0,
    cutout_half: int = _CUTOUT_HALF,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Measure PSF-deconvolved morphology for all galaxy candidates in the eGSF catalogue.

    Requires run_egsf() to have been run first (reads *_galaxy_detections.csv).

    For each galaxy detection per (obs_id, chip_ext):
      1. Load STDPSF model via _load_image_psf (same as psf_fitting.py).
      2. Load residual FITS (_flc_residual.fits) and pypass catalog (for flux).
      3. For each detection:
           - Evaluate spatially-interpolated STDPSF at (x, y) → P via _eval_psf_on_window.
           - Reconstruct galaxy profile: gal = residual_cutout + flux * P
             (restores the PSF-modelled source flux subtracted by pypass, analogous to
             the PSF restoration in pypass multipass later iterations).
           - Measure 2nd moments of gal and of P separately via _moments_2d.
           - Deconvolve: M_int = M_obs − M_PSF → intrinsic shape via _deconvolve_moments.

    Processing is vectorised per (obs_id, chip_ext): each residual FITS and catalog are
    read once; per-detection loop handles the PSF evaluation and reconstruction.

    Parameters
    ----------
    lib_dir
        Path to bp3m lib directory containing STDPSFs/. If None, read from
        bp3m config.toml (set by `bp3m setup`).
    cutout_half
        Half-width of the pixel cutout: cutout size = (2*cutout_half+1)^2.
        Default: _CUTOUT_HALF (9×9). Larger values capture more of extended sources
        but increase runtime and sensitivity to neighbors.

    Output: {output_dir}/egsf/{field_name}_galaxy_morphology.csv
    Columns (in addition to all detections columns):
      psf_sigma_x, psf_sigma_y   (px, STDPSF σ at source position)
      sigma_obs_x, sigma_obs_y   (px, Gaussian-weighted observed width)
      e1_obs, e2_obs              (observed ellipticity)
      sigma_int_x, sigma_int_y   (px, PSF-deconvolved intrinsic width)
      e1_int, e2_int              (PSF-deconvolved ellipticity)
      size_ratio                  (sqrt(T_int/T_PSF); >1 = spatially resolved)
      moment_snr                  (S/N from flux-weighted cutout)
    """
    output_dir = Path(output_dir)
    egsf_dir   = output_dir / 'egsf'
    dets_path  = egsf_dir / f'{field_name}_galaxy_detections.csv'
    out_morph  = egsf_dir / f'{field_name}_galaxy_morphology.csv'

    if out_morph.exists() and not force_rerun:
        print(f"eGSF morphology: loading existing ({out_morph.name})")
        return pd.read_csv(out_morph)

    if not dets_path.exists():
        raise FileNotFoundError(
            f"Galaxy detections not found: {dets_path}\n"
            f"Run run_egsf() first."
        )

    # Resolve lib_dir: read from bp3m config.toml if not specified
    if lib_dir is None:
        try:
            from bp3m.setup import CONFIG_FILE, DEFAULT_LIB_DIR
            if CONFIG_FILE.exists():
                import tomllib  # Python 3.11+
                with open(CONFIG_FILE, 'rb') as _f:
                    _cfg = tomllib.load(_f)
                lib_dir = Path(_cfg.get('lib_dir', str(DEFAULT_LIB_DIR)))
            else:
                lib_dir = DEFAULT_LIB_DIR
        except Exception:
            import bp3m as _bp3m_pkg
            lib_dir = Path(_bp3m_pkg.__file__).parent / 'lib'
    lib_dir = Path(lib_dir)

    dets = pd.read_csv(dets_path)
    print(f"\neGSF morphology: {len(dets)} detections in "
          f"{dets['epoch_key'].nunique()} epoch groups")

    hst_dir = output_dir / 'HST' / 'mastDownload' / 'HST'

    morph_cols = ['psf_sigma_x', 'psf_sigma_y',
                  'sigma_obs_x', 'sigma_obs_y', 'e1_obs', 'e2_obs',
                  'sigma_int_x', 'sigma_int_y', 'e1_int', 'e2_int',
                  'size_ratio', 'moment_snr']
    N = len(dets)
    morph_arrs = {col: np.full(N, np.nan) for col in morph_cols}

    # Cache PSF data per obs_id (STDPSF load is expensive)
    psf_cache: dict[str, tuple | None] = {}

    # Group detection row indices by (obs_id, chip_ext)
    from collections import defaultdict
    img_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(dets.itertuples()):
        img_groups[(row.obs_id, int(row.chip_ext))].append(i)

    n_images_done = 0
    n_groups = len(img_groups)
    print(f"  Processing {n_groups} (obs_id, chip_ext) groups...")

    for (obs_id, chip_ext), row_idxs in sorted(img_groups.items()):
        row_idxs = np.array(row_idxs)
        obs_dir  = hst_dir / obs_id
        flc_path = str(obs_dir / f'{obs_id}_flc.fits')

        # Load PSF data (cached per obs_id)
        if obs_id not in psf_cache:
            psf_cache[obs_id] = _load_image_psf(obs_dir, flc_path, lib_dir)
        psf_data = psf_cache[obs_id]
        if psf_data is None:
            n_images_done += 1
            continue
        psf_cube, xs_psf, ys_psf, psf_scale = psf_data

        # Load residual FITS for this chip
        res_path = str(obs_dir / f'{obs_id}_flc_residual.fits')
        chip_data = _load_residual_chip(res_path, chip_ext)
        if chip_data is None:
            n_images_done += 1
            continue
        sci, var = chip_data

        # Load pypass catalog once for flux values
        cat_path = str(obs_dir / f'{obs_id}_flc_catalog.fits')
        try:
            cat = Table.read(cat_path)
            cat_flux    = np.asarray(cat['flux']).astype(float)
            cat_x       = np.asarray(cat['x']).astype(float)
            cat_y       = np.asarray(cat['y']).astype(float)
            cat_chip    = np.asarray(cat['chip_ext']).astype(int)
        except Exception:
            n_images_done += 1
            continue

        # Detection coordinates and pypass catalog row index for this chip
        det_sub = dets.iloc[row_idxs]
        xs_det  = det_sub['x'].to_numpy(float)
        ys_det  = det_sub['y'].to_numpy(float)

        # Match each detection to its pypass catalog entry (nearest chip position)
        chip_mask = (cat_chip == chip_ext)
        cat_x_c   = cat_x[chip_mask]
        cat_y_c   = cat_y[chip_mask]
        cat_f_c   = cat_flux[chip_mask]

        # Per-detection processing
        for j, (global_idx, x_det, y_det) in enumerate(
            zip(row_idxs, xs_det, ys_det)
        ):
            # Find nearest catalog source on this chip for flux
            if len(cat_x_c) > 0:
                d2 = (cat_x_c - x_det)**2 + (cat_y_c - y_det)**2
                nn = int(np.argmin(d2))
                flux = cat_f_c[nn] if d2[nn] < 4.0 else 0.0  # within 2 px
            else:
                flux = 0.0

            # Evaluate STDPSF at this position → P on (2hw+1)×(2hw+1) window
            result_psf = _eval_psf_on_window(
                psf_cube, xs_psf, ys_psf, psf_scale,
                x_det, y_det, sci.shape, hw=cutout_half,
            )
            if result_psf is None:
                continue
            P, y_lo, y_hi, x_lo, x_hi = result_psf

            # Reconstruct galaxy: residual + flux × PSF (restoring the subtracted source)
            res_cut = sci[y_lo:y_hi, x_lo:x_hi].astype(float)
            var_cut = var[y_lo:y_hi, x_lo:x_hi].astype(float)
            gal_cut = res_cut + flux * P

            # Measure moments of reconstructed galaxy (inv-var weighted)
            inv_var_cut = np.where(var_cut > 0, 1.0 / np.maximum(var_cut, 1e-6), 0.0)
            gal_mom = _moments_2d(gal_cut, weight_sigma=_WEIGHT_SIGMA, inv_var=inv_var_cut)

            # Measure moments of PSF model P (no noise weighting)
            psf_mom = _moments_2d(P, weight_sigma=_WEIGHT_SIGMA)

            if np.isnan(gal_mom['M_xx']) or np.isnan(psf_mom['M_xx']):
                continue

            # PSF-deconvolve
            dec = _deconvolve_moments(gal_mom, psf_mom)

            morph_arrs['psf_sigma_x'][global_idx]  = dec['psf_sigma_x']
            morph_arrs['psf_sigma_y'][global_idx]  = dec['psf_sigma_y']
            morph_arrs['sigma_obs_x'][global_idx]  = dec['sigma_obs_x']
            morph_arrs['sigma_obs_y'][global_idx]  = dec['sigma_obs_y']
            morph_arrs['e1_obs'][global_idx]       = dec['e1_obs']
            morph_arrs['e2_obs'][global_idx]       = dec['e2_obs']
            morph_arrs['sigma_int_x'][global_idx]  = dec['sigma_int_x']
            morph_arrs['sigma_int_y'][global_idx]  = dec['sigma_int_y']
            morph_arrs['e1_int'][global_idx]       = dec['e1_int']
            morph_arrs['e2_int'][global_idx]       = dec['e2_int']
            morph_arrs['size_ratio'][global_idx]   = dec['size_ratio']
            morph_arrs['moment_snr'][global_idx]   = gal_mom['snr']

        n_images_done += 1
        if n_images_done % 10 == 0:
            print(f"    {n_images_done}/{n_groups} image-chip groups processed...")

    # Assemble result DataFrame
    result = dets.copy()
    for col in morph_cols:
        result[col] = morph_arrs[col]

    result.to_csv(out_morph, index=False)

    resolved = (result['size_ratio'] > 1.0).sum()
    good_snr  = (result['moment_snr'] > 3.0).sum()
    print(f"\n  Total detections measured:       {N}")
    print(f"  Resolved (size_ratio > 1):       {resolved}")
    print(f"  Good S/N (moment_snr > 3):       {good_snr}")
    print(f"  Median size_ratio:               {np.nanmedian(result['size_ratio']):.2f}")
    print(f"  Median PSF sigma (px):           "
          f"x={np.nanmedian(result['psf_sigma_x']):.3f}  "
          f"y={np.nanmedian(result['psf_sigma_y']):.3f}")
    print(f"  Median intrinsic sigma (px):     "
          f"x={np.nanmedian(result['sigma_int_x']):.3f}  "
          f"y={np.nanmedian(result['sigma_int_y']):.3f}")
    print(f"  Saved → {out_morph.name}")

    return result


# ── Sérsic forward-model fitting ──────────────────────────────────────────────
#
# Coordinate chain (per image):
#   sky offset (Δα,Δδ arcsec)
#     → pseudo-pixel offset  via gnomonic Jacobian at source position
#     → GDC offset           via inv([[a,b],[c,d]])
#     → raw pixel offset     via inv(J_gdc) at source center
#     → supersampled frame   × psf_scale
#     → convolve with STDPSF → downsample by psf_scale² → raw pixel model
#
# Raw pixels are rectilinear/equal-area, so downsampling is plain averaging.


def _load_stdgdc_full(path: str) -> dict:
    """Load all five STDGDC extensions including reverse maps XCG/YCG (exts 4-5)."""
    from pypass.io import load_stdgdc
    gdc = load_stdgdc(path)
    with fits.open(path) as hdul:
        hdr0 = hdul[0].header
        gdc['ndim_xcg'] = int(hdr0['NDIM_XCG'])
        gdc['ndim_ycg'] = int(hdr0['NDIM_YCG'])
        gdc['xcg'] = np.array(hdul[4].data, dtype=np.float64)
        gdc['ycg'] = np.array(hdul[5].data, dtype=np.float64)
    return gdc


def _apply_inv_gdc(
    x_corr: np.ndarray,
    y_corr: np.ndarray,
    gdc: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse GDC: (x_corr, y_corr) → (x_raw, y_raw) via bilinear on XCG/YCG.

    Out-of-bounds positions (sentinel -20000) are returned as NaN.
    """
    xcg = gdc['xcg']; ycg = gdc['ycg']
    xgc_0 = gdc['xgc_0']; ygc_0 = gdc['ygc_0']
    ndim_xcg = gdc['ndim_xcg']; ndim_ycg = gdc['ndim_ycg']

    x_corr = np.asarray(x_corr, float)
    y_corr = np.asarray(y_corr, float)

    # Float indices into the reverse maps (column=x, row=y)
    fx = x_corr - xgc_0
    fy = y_corr - ygc_0

    ix = np.floor(fx).astype(int)
    iy = np.floor(fy).astype(int)
    dx = fx - ix
    dy = fy - iy

    # Clamp to valid range
    valid = (ix >= 0) & (ix < ndim_xcg - 1) & (iy >= 0) & (iy < ndim_ycg - 1)
    ix = np.clip(ix, 0, ndim_xcg - 2)
    iy = np.clip(iy, 0, ndim_ycg - 2)

    # Bilinear interpolation: xcg[row, col] = xcg[iy, ix]
    x_raw = ((1 - dx) * (1 - dy) * xcg[iy,     ix    ]
           + (    dx) * (1 - dy) * xcg[iy,     ix + 1]
           + (1 - dx) * (    dy) * xcg[iy + 1, ix    ]
           + (    dx) * (    dy) * xcg[iy + 1, ix + 1])
    y_raw = ((1 - dx) * (1 - dy) * ycg[iy,     ix    ]
           + (    dx) * (1 - dy) * ycg[iy,     ix + 1]
           + (1 - dx) * (    dy) * ycg[iy + 1, ix    ]
           + (    dx) * (    dy) * ycg[iy + 1, ix + 1])

    x_raw = np.where(valid, x_raw, np.nan)
    y_raw = np.where(valid, y_raw, np.nan)

    # Sentinel check: the reverse maps use -20000 for masked regions
    x_raw = np.where(x_raw < -1000, np.nan, x_raw)
    y_raw = np.where(y_raw < -1000, np.nan, y_raw)
    return x_raw, y_raw


def _gnomonic_forward(
    ra: float, dec: float, ra0: float, dec0: float, pscale_mas: float
) -> tuple[float, float]:
    """Sky (ra, dec) [deg] → pseudo-pixel (x_p, y_p).

    Inverse of coords.plane_project_inverse.  Sign convention:
    x_p increases to the West (−RA direction), y_p to the North.
    """
    from bp3m.astro_utils import DEG2RAD, RAD2MAS
    ra_r   = np.radians(ra);   dec_r  = np.radians(dec)
    ra0_r  = np.radians(ra0);  dec0_r = np.radians(dec0)
    dra = ra_r - ra0_r
    denom = (np.sin(dec0_r) * np.sin(dec_r)
             + np.cos(dec0_r) * np.cos(dec_r) * np.cos(dra))
    xi  =  np.cos(dec_r) * np.sin(dra) / denom          # +East radians
    eta = (np.cos(dec0_r) * np.sin(dec_r)
           - np.sin(dec0_r) * np.cos(dec_r) * np.cos(dra)) / denom  # +North
    # pseudo-pixel: x = -xi*RAD2MAS/pscale (West positive), y = eta*RAD2MAS/pscale
    x_p = -xi  * RAD2MAS / pscale_mas
    y_p =  eta * RAD2MAS / pscale_mas
    return float(x_p), float(y_p)


def _sky_to_raw_matrix(
    ra_src: float,
    dec_src: float,
    x_raw_src: float,
    y_raw_src: float,
    a: float, b: float, c: float, d: float,
    ra0: float, dec0: float,
    pscale_mas: float,
    Xo: float, Yo: float,
    gdc: dict,
) -> np.ndarray | None:
    """Compute 2×2 matrix M_sky_to_raw such that
    M_sky_to_raw @ (Δα_arcsec, Δδ_arcsec) = (δx_raw, δy_raw).

    Combines the local gnomonic Jacobian, the inverse plate-solution matrix,
    and the inverse GDC Jacobian, all evaluated at the source center.

    Returns (2, 2) ndarray or None on failure.
    """
    try:
        from pypass.io import _gdc_jacobian_batch
    except ImportError:
        return None

    # ── Step 1: sky → pseudo-pixel Jacobian (numerical, 1-mas perturbation) ──
    eps = 1e-3 / 3600.0  # 1 mas in degrees
    xp0, yp0 = _gnomonic_forward(ra_src,       dec_src,       ra0, dec0, pscale_mas)
    xp_a, yp_a = _gnomonic_forward(ra_src + eps, dec_src,       ra0, dec0, pscale_mas)
    xp_d, yp_d = _gnomonic_forward(ra_src,       dec_src + eps, ra0, dec0, pscale_mas)
    # Jacobian: d(x_p,y_p)/d(Δα,Δδ) in pseudo-px per arcsec
    inv_eps_arcsec = 1.0 / (eps * 3600.0)
    J_sky_to_pseudo = np.array([
        [(xp_a - xp0) * inv_eps_arcsec, (xp_d - xp0) * inv_eps_arcsec],
        [(yp_a - yp0) * inv_eps_arcsec, (yp_d - yp0) * inv_eps_arcsec],
    ])  # shape (2, 2): rows=[x_p, y_p], cols=[Δα, Δδ]

    # ── Step 2: pseudo-pixel → GDC offset via inv([[a,b],[c,d]]) ─────────────
    abcd = np.array([[a, b], [c, d]])
    try:
        inv_abcd = np.linalg.inv(abcd)
    except np.linalg.LinAlgError:
        return None
    # Δ(x_gdc, y_gdc) = inv_abcd @ Δ(x_pseudo, y_pseudo)
    J_pseudo_to_gdc = inv_abcd   # (2, 2)

    # ── Step 3: GDC → raw via inv(J_gdc) at source center ───────────────────
    J_gdc = _gdc_jacobian_batch(
        np.array([x_raw_src]), np.array([y_raw_src]), gdc
    )[0]  # (2, 2): d(x_gdc,y_gdc)/d(x_raw,y_raw)
    try:
        J_gdc_inv = np.linalg.inv(J_gdc)  # d(x_raw,y_raw)/d(x_gdc,y_gdc)
    except np.linalg.LinAlgError:
        return None

    # ── Combined: sky (arcsec) → raw pixel ──────────────────────────────────
    M = J_gdc_inv @ J_pseudo_to_gdc @ J_sky_to_pseudo  # (2, 2)
    return M


# ── Sérsic profile ─────────────────────────────────────────────────────────────

def _sersic_bn(n: float) -> float:
    """b_n from MacArthur et al. (2003) approximation (accurate to ~1e-4 for n>0.5)."""
    return (2.0 * n - 1.0 / 3.0
            + 4.0 / (405.0 * n)
            + 46.0 / (25515.0 * n**2)
            + 131.0 / (1148175.0 * n**3))


def _sersic_2d(
    dra: np.ndarray,
    ddec: np.ndarray,
    Re_arcsec: float,
    n: float,
    q: float,
    PA_rad: float,
) -> np.ndarray:
    """Evaluate the unnormalised Sérsic profile at sky offsets (dra, ddec) in arcsec.

    PA_rad is the position angle of the major axis, measured North through East.
    q = semi-minor / semi-major axis ratio.
    Returns S(r) = exp(-b_n * ((r/Re)^{1/n} - 1)).
    """
    # Project onto major/minor axes (N-through-E PA convention)
    cos_pa, sin_pa = np.cos(PA_rad), np.sin(PA_rad)
    y_pa =  dra * sin_pa + ddec * cos_pa   # along major axis
    x_pa =  dra * cos_pa - ddec * sin_pa   # along minor axis

    r = np.sqrt((x_pa / q)**2 + y_pa**2)
    with np.errstate(invalid='ignore'):
        rn = np.where(r > 0, (r / Re_arcsec) ** (1.0 / n), 0.0)
    bn = _sersic_bn(n)
    return np.exp(-bn * (rn - 1.0))


# ── Sérsic forward-model renderer ─────────────────────────────────────────────

def _render_sersic_psf(
    sersic_params: np.ndarray,
    M_raw_to_sky: np.ndarray,
    psf_cube: np.ndarray,
    xs_psf: np.ndarray,
    ys_psf: np.ndarray,
    psf_scale: int,
    x_src: float,
    y_src: float,
    sci_shape: tuple[int, int],
    hw: int = _CUTOUT_HALF,
) -> np.ndarray | None:
    """Render one normalised PSF-convolved Sérsic model cutout in raw pixels.

    sersic_params : (Re_arcsec, n, q, PA_deg)
    M_raw_to_sky  : (2,2) matrix — raw pixel offset → sky offset (arcsec)
    Returns (2*hw+1, 2*hw+1) normalised model (unit total flux) or None on error.
    """
    from scipy.signal import fftconvolve
    try:
        from pypass.core import interpolate_psf
    except ImportError:
        return None

    Re_arcsec, n, q, PA_deg = sersic_params
    PA_rad = np.radians(PA_deg)

    psf_size = psf_cube.shape[-1]   # 101 for ACS/WFC at psf_scale=4
    half_ss  = psf_size // 2        # 50

    # Build a supersampled grid large enough for the convolution.
    # N_ss must accommodate the Sérsic extent plus one full PSF half-width of margin.
    # We use (2*hw+1)*psf_scale + 2*half_ss + 1 (always odd for symmetry).
    raw_pixels = 2 * hw + 1
    ss_core    = raw_pixels * psf_scale
    N_ss       = ss_core + 2 * half_ss   # padded grid in supersampled units
    if N_ss % 2 == 0:
        N_ss += 1
    half_N = N_ss // 2

    # Supersampled pixel offsets from source center, in raw-pixel units
    # Spacing = 1/psf_scale raw px
    k = np.arange(N_ss) - half_N
    dk_raw = k / float(psf_scale)             # raw pixel offset per supersampled pixel
    DX_ss, DY_ss = np.meshgrid(dk_raw, dk_raw, indexing='ij')  # (N_ss, N_ss) in raw px

    # Map raw pixel offsets → sky offsets (arcsec) via M_raw_to_sky
    sky = M_raw_to_sky @ np.stack([DX_ss.ravel(), DY_ss.ravel()])  # (2, N²)
    dra_ss  = sky[0].reshape(N_ss, N_ss)   # East arcsec
    ddec_ss = sky[1].reshape(N_ss, N_ss)   # North arcsec

    # Evaluate Sérsic on the supersampled grid
    S_ss = _sersic_2d(dra_ss, ddec_ss, Re_arcsec, n, q, PA_rad).astype(np.float64)

    # Get STDPSF at source position (interpolated, supersampled)
    psf_raw = interpolate_psf(psf_cube, xs_psf, ys_psf, x_src, y_src)  # (psf_size,psf_size)
    psf_raw = np.asarray(psf_raw, dtype=np.float64)
    psf_norm = psf_raw / psf_raw.sum()   # normalise PSF to unit flux

    # FFT-convolve Sérsic model with PSF (both at 1/psf_scale raw pixel spacing)
    conv = fftconvolve(S_ss, psf_norm, mode='same')

    # Extract central (raw_pixels*psf_scale) × (raw_pixels*psf_scale) region
    c0 = half_N - ss_core // 2
    c1 = c0 + ss_core
    model_ss = conv[c0:c1, c0:c1]   # (ss_core, ss_core) = (raw_pixels*psf_scale)²

    # Downsample: average psf_scale × psf_scale blocks → raw pixel model
    model_raw = (model_ss
                 .reshape(raw_pixels, psf_scale, raw_pixels, psf_scale)
                 .mean(axis=(1, 3)))   # (raw_pixels, raw_pixels)

    total = model_raw.sum()
    if total <= 0:
        return None
    return (model_raw / total).astype(np.float32)


# ── Per-image linear solve (flux + sky analytically at each optimizer step) ───

def _linear_params_and_chi2(
    model: np.ndarray,
    obs: np.ndarray,
    inv_var: np.ndarray,
) -> tuple[float, float, float]:
    """Solve for (flux, sky) minimising sum((obs - flux*model - sky)² * inv_var).

    Returns (flux, sky, chi2).
    """
    w  = inv_var.ravel()
    m  = model.ravel()
    o  = obs.ravel()
    sm = (w * m).sum()
    s1 = w.sum()
    sm2= (w * m * m).sum()
    smo= (w * m * o).sum()
    so = (w * o).sum()
    det = sm2 * s1 - sm * sm
    if det <= 0:
        return 0.0, 0.0, np.inf
    flux = (smo * s1  - so  * sm) / det
    sky  = (so  * sm2 - smo * sm) / det
    resid = o - flux * m - sky
    return float(flux), float(sky), float((w * resid * resid).sum())


# ── Per-source Sérsic fitter ───────────────────────────────────────────────────

def _fit_one_source(
    sersic_init: np.ndarray,
    image_data: list[dict],
    M_raw_to_sky: np.ndarray,
    hw: int,
) -> dict:
    """Fit Sérsic(Re, n, q, PA) jointly over all images for one source.

    sersic_init : (Re_arcsec, n, q, PA_deg) initial guess
    image_data  : list of dicts, one per detection:
        {'psf_cube', 'xs_psf', 'ys_psf', 'psf_scale',
         'x_src', 'y_src', 'sci_shape', 'obs_cut', 'inv_var_cut'}
    M_raw_to_sky: (2,2) matrix (same for all images in one epoch group
                   to a good approximation — recomputed per source below)
    hw          : cutout half-width

    Returns dict with fit results.
    """
    from scipy.optimize import minimize

    # Parameter bounds: Re (0.02"–3"), n (0.3–8), q (0.1–1), PA (0–180 deg)
    bounds = [(0.02, 3.0), (0.3, 8.0), (0.1, 1.0), (0.0, 180.0)]

    def objective(theta):
        Re, n, q, PA = theta
        model_template = _render_sersic_psf(
            theta, M_raw_to_sky,
            image_data[0]['psf_cube'], image_data[0]['xs_psf'],
            image_data[0]['ys_psf'],  image_data[0]['psf_scale'],
            image_data[0]['x_src'],   image_data[0]['y_src'],
            image_data[0]['sci_shape'], hw=hw,
        )
        if model_template is None:
            return 1e30

        total_chi2 = 0.0
        for imd in image_data:
            # Re-render with per-image PSF if PSF data differs
            if imd is not image_data[0]:
                M_i = _render_sersic_psf(
                    theta, M_raw_to_sky,
                    imd['psf_cube'], imd['xs_psf'], imd['ys_psf'], imd['psf_scale'],
                    imd['x_src'], imd['y_src'], imd['sci_shape'], hw=hw,
                )
                if M_i is None:
                    continue
            else:
                M_i = model_template
            _, _, chi2_i = _linear_params_and_chi2(M_i, imd['obs_cut'], imd['inv_var_cut'])
            if np.isfinite(chi2_i):
                total_chi2 += chi2_i
            else:
                total_chi2 += 1e15   # cap to avoid inf→NaN in gradient differences
        return float(total_chi2)

    res = minimize(
        objective,
        x0=np.clip(sersic_init, [b[0] for b in bounds], [b[1] for b in bounds]),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-5},
    )

    theta_best = res.x

    # Recover per-image flux and sky at best-fit params; count contributing images
    fluxes, skies = [], []
    n_pixels_used = 0
    n_images_used = 0
    for imd in image_data:
        M_i = _render_sersic_psf(
            theta_best, M_raw_to_sky,
            imd['psf_cube'], imd['xs_psf'], imd['ys_psf'], imd['psf_scale'],
            imd['x_src'],    imd['y_src'],  imd['sci_shape'], hw=hw,
        )
        if M_i is None:
            fluxes.append(np.nan); skies.append(np.nan)
        else:
            f, s, _ = _linear_params_and_chi2(M_i, imd['obs_cut'], imd['inv_var_cut'])
            fluxes.append(f); skies.append(s)
            n_pixels_used += imd['obs_cut'].size
            n_images_used += 1

    n_dof    = max(n_pixels_used - 4 - 2 * n_images_used, 1)
    chi2_red = res.fun / n_dof

    return dict(
        Re_arcsec=float(theta_best[0]),
        sersic_n=float(theta_best[1]),
        axis_ratio=float(theta_best[2]),
        PA_deg=float(theta_best[3]),
        chi2_red=float(chi2_red),
        n_dof=int(n_dof),
        n_images=len(image_data),
        fit_converged=bool(res.success),
        median_flux=float(np.nanmedian(fluxes)),
        flux_scatter=float(np.nanstd(fluxes)) if len(fluxes) > 1 else np.nan,
    )


# ── Main Sérsic fitting entry point ───────────────────────────────────────────

def fit_egsf_sources(
    field_name: str,
    output_dir: str | Path,
    lib_dir: str | Path | None = None,
    bp3m_results_subdir: str = 'BP3M_results',
    pa_tol: float = 5.0,
    cutout_half: int = _CUTOUT_HALF,
    min_detections: int = 3,
    size_ratio_min: float = 1.2,
    moment_snr_min: float = 5.0,
    n_workers: int = 1,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Fit a PSF-convolved Sérsic profile to each eGSF galaxy candidate.

    Requires measure_galaxy_morphology() to have been run first.

    For each candidate (selected by morphology quality cuts), all images in its
    epoch group (same filter + PA_V3) are used simultaneously in a forward-model
    joint fit.  The Sérsic profile is defined on the sky plane; it is projected
    into each image via the v1 BP3M plate solution + inverse GDC, then convolved
    with that image's spatially-interpolated STDPSF.  Linear parameters (per-image
    flux and sky background) are marginalised analytically at each optimizer step.

    Parameters
    ----------
    min_detections : minimum valid-morphology detections required to attempt a fit
    size_ratio_min : minimum PSF-deconvolved size_ratio (resolved source criterion)
    moment_snr_min : minimum moment S/N per detection
    n_workers      : parallel workers (1 = serial; uses concurrent.futures)
    """
    from collections import defaultdict
    import concurrent.futures

    output_dir = Path(output_dir)
    egsf_dir   = output_dir / 'egsf'
    morph_path = egsf_dir / f'{field_name}_galaxy_morphology.csv'
    out_path   = egsf_dir / f'{field_name}_egsf_catalog.csv'

    if out_path.exists() and not force_rerun:
        print(f"eGSF catalog: loading existing ({out_path.name})")
        return pd.read_csv(out_path)

    if not morph_path.exists():
        raise FileNotFoundError(
            f"Morphology CSV not found: {morph_path}\n"
            f"Run measure_galaxy_morphology() first."
        )

    # ── Resolve lib_dir (same logic as measure_galaxy_morphology) ────────────
    if lib_dir is None:
        try:
            from bp3m.setup import CONFIG_FILE, DEFAULT_LIB_DIR
            if CONFIG_FILE.exists():
                import tomllib
                with open(CONFIG_FILE, 'rb') as _f:
                    _cfg = tomllib.load(_f)
                lib_dir = Path(_cfg.get('lib_dir', str(DEFAULT_LIB_DIR)))
            else:
                lib_dir = DEFAULT_LIB_DIR
        except Exception:
            import bp3m as _bp3m_pkg
            lib_dir = Path(_bp3m_pkg.__file__).parent / 'lib'
    lib_dir = Path(lib_dir)

    # ── Load morphology detections + apply quality cuts ───────────────────────
    morph = pd.read_csv(morph_path)
    print(f"\n{'='*60}")
    print(f"eGSF Sérsic fitting — {field_name}")
    print(f"{'='*60}")
    print(f"  Total detections: {len(morph)}")

    valid = (
        morph['size_ratio'].notna()
        & morph['moment_snr'].notna()
        & (morph['size_ratio'] > size_ratio_min)
        & (morph['moment_snr'] > moment_snr_min)
    )
    morph_q = morph[valid].copy()
    print(f"  Passing quality cuts (size_ratio>{size_ratio_min}, snr>{moment_snr_min}): "
          f"{len(morph_q)}")

    # Count valid detections per source
    src_counts = morph_q.groupby('source_id').size()
    good_srcs  = src_counts[src_counts >= min_detections].index
    morph_fit  = morph_q[morph_q['source_id'].isin(good_srcs)].copy()
    print(f"  Sources with >= {min_detections} valid detections: {len(good_srcs)}")

    if len(good_srcs) == 0:
        print("  No candidates pass cuts — returning empty DataFrame.")
        return pd.DataFrame()

    # ── Load plate solutions from image_transformations.csv ──────────────────
    plates_path = output_dir / bp3m_results_subdir / 'image_transformations.csv'
    plates_df   = pd.read_csv(plates_path)
    # Build dict: obs_id → {hi: row, lo: row}
    plate_map: dict[str, dict] = {}
    for _, row in plates_df.iterrows():
        img_name = str(row['image_name'])
        # e.g. "j9gz01orq_hi" → obs_id="j9gz01orq", suffix="hi"
        if '_hi' in img_name or '_lo' in img_name:
            obs_id = img_name.rsplit('_', 1)[0]
            suffix = img_name.rsplit('_', 1)[1]
            plate_map.setdefault(obs_id, {})[suffix] = row.to_dict()

    # ── Load GDC files (cached per det_prefix) ────────────────────────────────
    from pypass.io import find_gdc, _DETECTOR_PREFIX
    gdc_cache: dict[str, dict] = {}  # det_prefix → gdc dict

    def _get_gdc(instrume: str, detector: str, flt: str, hdr: dict) -> dict | None:
        det_prefix = _DETECTOR_PREFIX.get((instrume, detector))
        if det_prefix is None:
            return None
        if det_prefix not in gdc_cache:
            gdc_dir = lib_dir / 'STDGDCs' / det_prefix
            gdc_path = find_gdc(str(gdc_dir), hdr)
            if gdc_path is None:
                return None
            gdc_cache[det_prefix] = _load_stdgdc_full(gdc_path)
        return gdc_cache[det_prefix]

    # ── Group fitting candidates by epoch_key ────────────────────────────────
    epoch_groups_for_fit = morph_fit.groupby('epoch_key')
    hst_dir = output_dir / 'HST' / 'mastDownload' / 'HST'

    all_results: list[dict] = []

    for epoch_key, epoch_dets in epoch_groups_for_fit:
        print(f"\n  Epoch group: {epoch_key}  ({len(epoch_dets)} detections, "
              f"{epoch_dets['source_id'].nunique()} sources)")

        # ── Pre-load per (obs_id, chip_ext): residual + catalog + PSF + plate ─
        img_key_data: dict[tuple, dict] = {}
        obs_ids_done: set[str] = set()

        for _, det in epoch_dets.drop_duplicates(subset=['obs_id','chip_ext']).iterrows():
            obs_id   = str(det['obs_id'])
            chip_ext = int(det['chip_ext'])
            obs_dir  = hst_dir / obs_id
            flc_path = str(obs_dir / f'{obs_id}_flc.fits')

            # Plate solution: chip_ext=1 → '_lo', chip_ext=4 → '_hi'
            suffix = 'hi' if chip_ext == 4 else 'lo'
            plate = plate_map.get(obs_id, {}).get(suffix)
            if plate is None:
                continue

            # PSF data
            psf_data = _load_image_psf(obs_dir, flc_path, lib_dir)
            if psf_data is None:
                continue

            # GDC (need FITS header for filter-based lookup)
            try:
                with fits.open(flc_path) as hdul:
                    h0 = hdul[0].header
                instrume = h0.get('INSTRUME', '').strip().upper()
                detector = h0.get('DETECTOR', '').strip().upper()
                flt      = _get_filter(h0)
                gdc = _get_gdc(instrume, detector, flt, h0)
            except Exception:
                gdc = None
            if gdc is None:
                continue

            # Residual image
            res_path  = str(obs_dir / f'{obs_id}_flc_residual.fits')
            chip_data = _load_residual_chip(res_path, chip_ext)
            if chip_data is None:
                continue
            sci, var = chip_data

            # Pypass catalog (for flux per source and x_gdc/y_gdc)
            cat_path = str(obs_dir / f'{obs_id}_flc_catalog.fits')
            try:
                cat = Table.read(cat_path)
                cat_flux  = np.asarray(cat['flux']).astype(float)
                cat_x     = np.asarray(cat['x']).astype(float)
                cat_y     = np.asarray(cat['y']).astype(float)
                cat_xgdc  = np.asarray(cat['x_gdc']).astype(float)
                cat_ygdc  = np.asarray(cat['y_gdc']).astype(float)
                cat_chip  = np.asarray(cat['chip_ext']).astype(int)
                chip_mask = cat_chip == chip_ext
                cat_x_c   = cat_x[chip_mask]
                cat_y_c   = cat_y[chip_mask]
                cat_xgdc_c = cat_xgdc[chip_mask]
                cat_ygdc_c = cat_ygdc[chip_mask]
                cat_flux_c = cat_flux[chip_mask]
            except Exception:
                continue

            img_key_data[(obs_id, chip_ext)] = dict(
                psf_data=psf_data, plate=plate, gdc=gdc,
                sci=sci, var=var,
                cat_x=cat_x_c, cat_y=cat_y_c,
                cat_xgdc=cat_xgdc_c, cat_ygdc=cat_ygdc_c,
                cat_flux=cat_flux_c,
            )

        # ── Fit each source ───────────────────────────────────────────────────
        src_groups = epoch_dets.groupby('source_id')
        n_fit = 0
        for source_id, src_dets in src_groups:
            image_data_list: list[dict] = []
            ra_src  = float(src_dets['ra'].iloc[0]  if 'ra'  in src_dets.columns else
                            src_dets['ra_med'].iloc[0])
            dec_src = float(src_dets['dec'].iloc[0] if 'dec' in src_dets.columns else
                            src_dets['dec_med'].iloc[0])

            for _, det in src_dets.iterrows():
                obs_id   = str(det['obs_id'])
                chip_ext = int(det['chip_ext'])
                key      = (obs_id, chip_ext)
                if key not in img_key_data:
                    continue

                d = img_key_data[key]
                psf_cube, xs_psf, ys_psf, psf_scale = d['psf_data']
                sci = d['sci']; var = d['var']
                plate = d['plate']; gdc = d['gdc']

                x_det = float(det['x']); y_det = float(det['y'])

                # Find nearest catalog match for raw pos, gdc pos, and flux
                cat_x = d['cat_x']; cat_y = d['cat_y']
                if len(cat_x) == 0:
                    continue
                d2 = (cat_x - x_det)**2 + (cat_y - y_det)**2
                nn = int(np.argmin(d2))
                if d2[nn] > 4.0:
                    continue
                x_raw = float(cat_x[nn]);     y_raw = float(cat_y[nn])
                x_gdc = float(d['cat_xgdc'][nn]); y_gdc = float(d['cat_ygdc'][nn])
                flux_cat = float(d['cat_flux'][nn])

                # Precompute sky→raw linear map for this image
                M_s2r = _sky_to_raw_matrix(
                    ra_src, dec_src, x_raw, y_raw,
                    plate['a'], plate['b'], plate['c'], plate['d'],
                    plate['ra0_final'], plate['dec0_final'],
                    plate['pixel_scale_mas'],
                    plate['Xo_pivot'], plate['Yo_pivot'],
                    gdc,
                )
                if M_s2r is None:
                    continue
                try:
                    M_r2s = np.linalg.inv(M_s2r)
                except np.linalg.LinAlgError:
                    continue

                # Extract observed cutout (residual + PSF restoration)
                result_psf = _eval_psf_on_window(
                    psf_cube, xs_psf, ys_psf, psf_scale,
                    x_raw, y_raw, sci.shape, hw=cutout_half,
                )
                if result_psf is None:
                    continue
                P, y_lo, y_hi, x_lo, x_hi = result_psf
                res_cut = sci[y_lo:y_hi, x_lo:x_hi].astype(float)
                var_cut = var[y_lo:y_hi, x_lo:x_hi].astype(float)
                obs_cut = res_cut + flux_cat * P

                inv_var_cut = np.where(var_cut > 0,
                                       1.0 / np.maximum(var_cut, 1e-6), 0.0)

                image_data_list.append(dict(
                    psf_cube=psf_cube, xs_psf=xs_psf, ys_psf=ys_psf,
                    psf_scale=psf_scale,
                    x_src=x_raw, y_src=y_raw, sci_shape=sci.shape,
                    obs_cut=obs_cut.astype(np.float32),
                    inv_var_cut=inv_var_cut.astype(np.float32),
                    M_raw_to_sky=M_r2s,
                ))

            if len(image_data_list) < min_detections:
                continue

            # Initial Sérsic params from averaged morphology moments
            # Intrinsic sigma → Re (rough: Re ≈ 2 × sigma_int for n≈1)
            pscale_arcsec = float(src_dets['pixel_scale_mas'].iloc[0]) / 1000.0 \
                            if 'pixel_scale_mas' in src_dets.columns else 0.050
            sig_int_med = float(np.nanmedian(src_dets.get('sigma_int_x', pd.Series([0.5]))))
            Re_init = max(0.03, sig_int_med * pscale_arcsec * 2.0)
            q_init  = float(np.nanmedian(
                src_dets.get('sigma_int_y', pd.Series([1.0])) /
                np.maximum(src_dets.get('sigma_int_x', pd.Series([1.0])), 0.1)
            ))
            q_init = float(np.clip(q_init, 0.15, 1.0))
            PA_init = 0.0
            sersic_init = np.array([Re_init, 1.0, q_init, PA_init])

            # Use M_raw_to_sky from first valid detection
            M_r2s_fit = image_data_list[0]['M_raw_to_sky']

            fit = _fit_one_source(sersic_init, image_data_list, M_r2s_fit, cutout_half)
            fit['source_id']  = source_id
            fit['epoch_key']  = epoch_key
            fit['ra_med']     = ra_src
            fit['dec_med']    = dec_src
            fit['n_det_used'] = len(image_data_list)
            all_results.append(fit)
            n_fit += 1

        print(f"    Fitted {n_fit} sources in this epoch group.")

    result = pd.DataFrame(all_results)
    if len(result) > 0:
        cols_front = ['source_id', 'epoch_key', 'ra_med', 'dec_med',
                      'Re_arcsec', 'sersic_n', 'axis_ratio', 'PA_deg',
                      'chi2_red', 'n_dof', 'n_images', 'n_det_used',
                      'fit_converged', 'median_flux', 'flux_scatter']
        result = result[[c for c in cols_front if c in result.columns]]

    egsf_dir.mkdir(exist_ok=True)
    result.to_csv(out_path, index=False)

    converged = result['fit_converged'].sum() if len(result) > 0 else 0
    print(f"\n  Total sources fitted:    {len(result)}")
    print(f"  Converged:               {converged}")
    if len(result) > 0:
        print(f"  Median Re_arcsec:        {result['Re_arcsec'].median():.3f}\"")
        print(f"  Median chi2_red:         {result['chi2_red'].median():.2f}")
    print(f"  Saved → {out_path.name}")
    return result


# ── Diagnostic cutout plots ────────────────────────────────────────────────────

def plot_egsf_diagnostics(
    field_name: str,
    output_dir: str | Path,
    lib_dir: str | Path | None = None,
    bp3m_results_subdir: str = 'BP3M_results',
    n_sources: int = 24,
    chi2_bins: tuple[float, float, float] = (5.0, 30.0),
    cutout_half: int = _CUTOUT_HALF,
    out_pdf: str | None = None,
) -> Path:
    """Save a PDF of diagnostic cutout panels for a sample of fitted eGSF sources.

    For each source shows (per best detection): observed galaxy, Sérsic model,
    residual (obs − model), and the STDPSF, with fit parameters annotated.

    Sources are stratified into three chi2 tiers:
        good  : chi2_red < chi2_bins[0]
        medium: chi2_bins[0] ≤ chi2_red < chi2_bins[1]
        bad   : chi2_red ≥ chi2_bins[1]

    n_sources are drawn evenly across the three tiers (rounded up).

    Requires fit_egsf_sources() to have been run first.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize
    import warnings

    output_dir = Path(output_dir)
    egsf_dir   = output_dir / 'egsf'
    cat_path   = egsf_dir / f'{field_name}_egsf_catalog.csv'
    morph_path = egsf_dir / f'{field_name}_galaxy_morphology.csv'

    if not cat_path.exists():
        raise FileNotFoundError(f"Run fit_egsf_sources first: {cat_path}")
    if not morph_path.exists():
        raise FileNotFoundError(f"Run measure_galaxy_morphology first: {morph_path}")

    cat   = pd.read_csv(cat_path)
    morph = pd.read_csv(morph_path)

    # ── Resolve lib_dir ───────────────────────────────────────────────────────
    if lib_dir is None:
        try:
            from bp3m.setup import CONFIG_FILE, DEFAULT_LIB_DIR
            if CONFIG_FILE.exists():
                import tomllib
                with open(CONFIG_FILE, 'rb') as _f:
                    _cfg = tomllib.load(_f)
                lib_dir = Path(_cfg.get('lib_dir', str(DEFAULT_LIB_DIR)))
            else:
                lib_dir = DEFAULT_LIB_DIR
        except Exception:
            import bp3m as _bp3m_pkg
            lib_dir = Path(_bp3m_pkg.__file__).parent / 'lib'
    lib_dir = Path(lib_dir)

    # ── Stratified sample ─────────────────────────────────────────────────────
    lo, hi = chi2_bins
    tiers = {
        f'good (χ²<{lo})':   cat[cat['chi2_red'] < lo],
        f'med ({lo}≤χ²<{hi})': cat[(cat['chi2_red'] >= lo) & (cat['chi2_red'] < hi)],
        f'bad (χ²≥{hi})':    cat[cat['chi2_red'] >= hi],
    }
    n_per_tier = max(1, (n_sources + 2) // 3)
    sample_rows = []
    for tier_label, tier_df in tiers.items():
        if len(tier_df) == 0:
            continue
        chosen = tier_df.sample(min(n_per_tier, len(tier_df)),
                                random_state=42).copy()
        chosen['_tier'] = tier_label
        sample_rows.append(chosen)
    sample = pd.concat(sample_rows, ignore_index=True)
    print(f"Diagnostic plots: {len(sample)} sources across {len(tiers)} tiers")

    # ── Load plate solutions ──────────────────────────────────────────────────
    plates_df = pd.read_csv(output_dir / bp3m_results_subdir / 'image_transformations.csv')
    plate_map: dict[str, dict] = {}
    for _, row in plates_df.iterrows():
        img_name = str(row['image_name'])
        if '_hi' in img_name or '_lo' in img_name:
            obs_id = img_name.rsplit('_', 1)[0]
            suffix = img_name.rsplit('_', 1)[1]
            plate_map.setdefault(obs_id, {})[suffix] = row.to_dict()

    # ── GDC cache ─────────────────────────────────────────────────────────────
    from pypass.io import find_gdc, _DETECTOR_PREFIX
    gdc_cache: dict[str, dict] = {}

    def _get_gdc_cached(instrume, detector, hdr):
        det_prefix = _DETECTOR_PREFIX.get((instrume, detector))
        if det_prefix is None:
            return None
        if det_prefix not in gdc_cache:
            gdc_dir  = lib_dir / 'STDGDCs' / det_prefix
            gdc_path = find_gdc(str(gdc_dir), hdr)
            if gdc_path is None:
                return None
            gdc_cache[det_prefix] = _load_stdgdc_full(gdc_path)
        return gdc_cache[det_prefix]

    hst_dir = output_dir / 'HST' / 'mastDownload' / 'HST'

    # ── Build panels ──────────────────────────────────────────────────────────
    n_cols   = 4   # obs / model / residual / PSF
    n_rows   = len(sample)
    fig_w    = n_cols * 2.0 + 1.0
    fig_h    = n_rows * 2.2 + 0.5

    if out_pdf is None:
        out_pdf = str(egsf_dir / f'{field_name}_egsf_diagnostics.pdf')

    with PdfPages(out_pdf) as pdf:
        # Split into pages of 12 rows each
        page_size = 12
        n_pages   = max(1, (len(sample) + page_size - 1) // page_size)

        for page_idx in range(n_pages):
            page_rows = sample.iloc[page_idx * page_size : (page_idx + 1) * page_size]
            n_r = len(page_rows)
            fig, axes = plt.subplots(
                n_r, n_cols,
                figsize=(fig_w, n_r * 2.2 + 0.5),
                squeeze=False,
            )
            col_titles = ['Observed', 'Sérsic model', 'Residual', 'PSF']
            for ci, ct in enumerate(col_titles):
                axes[0, ci].set_title(ct, fontsize=9, pad=3)

            for row_i, (_, src_row) in enumerate(page_rows.iterrows()):
                ax_obs, ax_mod, ax_res, ax_psf = axes[row_i]

                source_id = src_row['source_id']
                tier_lbl  = src_row.get('_tier', '')

                # Find best detection (highest moment_snr) for this source
                src_dets = morph[morph['source_id'] == source_id].copy()
                if len(src_dets) == 0:
                    for ax in (ax_obs, ax_mod, ax_res, ax_psf):
                        ax.set_visible(False)
                    continue
                src_dets = src_dets.sort_values('moment_snr', ascending=False)
                det = src_dets.iloc[0]

                obs_id   = str(det['obs_id'])
                chip_ext = int(det['chip_ext'])
                x_det    = float(det['x'])
                y_det    = float(det['y'])

                obs_dir  = hst_dir / obs_id
                flc_path = str(obs_dir / f'{obs_id}_flc.fits')
                suffix   = 'hi' if chip_ext == 4 else 'lo'
                plate    = plate_map.get(obs_id, {}).get(suffix)

                # Load PSF
                psf_data = _load_image_psf(obs_dir, flc_path, lib_dir)

                # Load GDC
                gdc = None
                try:
                    with fits.open(flc_path) as hdul:
                        h0 = hdul[0].header
                    instrume = h0.get('INSTRUME', '').strip().upper()
                    detector = h0.get('DETECTOR', '').strip().upper()
                    gdc = _get_gdc_cached(instrume, detector, h0)
                except Exception:
                    pass

                # Load residual
                res_path  = str(obs_dir / f'{obs_id}_flc_residual.fits')
                chip_data = _load_residual_chip(res_path, chip_ext)

                # Load catalog for flux and raw position
                cat_path2 = str(obs_dir / f'{obs_id}_flc_catalog.fits')
                flux_cat = None; x_raw = x_det; y_raw = y_det
                try:
                    tbl = Table.read(cat_path2)
                    c_x    = np.asarray(tbl['x']).astype(float)
                    c_y    = np.asarray(tbl['y']).astype(float)
                    c_chip = np.asarray(tbl['chip_ext']).astype(int)
                    c_flux = np.asarray(tbl['flux']).astype(float)
                    c_xgdc = np.asarray(tbl['x_gdc']).astype(float)
                    c_ygdc = np.asarray(tbl['y_gdc']).astype(float)
                    mask   = c_chip == chip_ext
                    if mask.sum() > 0:
                        d2 = (c_x[mask] - x_det)**2 + (c_y[mask] - y_det)**2
                        nn = int(np.argmin(d2))
                        if d2[nn] < 4.0:
                            x_raw    = float(c_x[mask][nn])
                            y_raw    = float(c_y[mask][nn])
                            flux_cat = float(c_flux[mask][nn])
                except Exception:
                    pass

                failed = (psf_data is None or gdc is None
                          or chip_data is None or plate is None
                          or flux_cat is None)

                # ── Reconstruct obs_cut ───────────────────────────────────────
                obs_cut = model_cut = res_cut_img = psf_img = None
                if not failed:
                    sci, var = chip_data
                    psf_cube, xs_psf, ys_psf, psf_scale = psf_data

                    psf_result = _eval_psf_on_window(
                        psf_cube, xs_psf, ys_psf, psf_scale,
                        x_raw, y_raw, sci.shape, hw=cutout_half,
                    )
                    if psf_result is not None:
                        P, y_lo, y_hi, x_lo, x_hi = psf_result
                        res_cut  = sci[y_lo:y_hi, x_lo:x_hi].astype(float)
                        var_cut  = var[y_lo:y_hi, x_lo:x_hi].astype(float)
                        obs_cut  = res_cut + flux_cat * P
                        inv_var  = np.where(var_cut > 0,
                                            1.0 / np.maximum(var_cut, 1e-6), 0.0)

                        # ── Sérsic model ──────────────────────────────────────
                        ra_src  = float(src_row['ra_med'])
                        dec_src = float(src_row['dec_med'])
                        M_s2r   = _sky_to_raw_matrix(
                            ra_src, dec_src, x_raw, y_raw,
                            plate['a'], plate['b'], plate['c'], plate['d'],
                            plate['ra0_final'], plate['dec0_final'],
                            plate['pixel_scale_mas'],
                            plate['Xo_pivot'], plate['Yo_pivot'], gdc,
                        )
                        if M_s2r is not None:
                            try:
                                M_r2s = np.linalg.inv(M_s2r)
                            except np.linalg.LinAlgError:
                                M_r2s = None
                            if M_r2s is not None:
                                theta = np.array([
                                    src_row['Re_arcsec'], src_row['sersic_n'],
                                    src_row['axis_ratio'], src_row['PA_deg'],
                                ])
                                with warnings.catch_warnings():
                                    warnings.simplefilter('ignore')
                                    M_norm = _render_sersic_psf(
                                        theta, M_r2s,
                                        psf_cube, xs_psf, ys_psf, psf_scale,
                                        x_raw, y_raw, sci.shape,
                                        hw=cutout_half,
                                    )
                                if M_norm is not None:
                                    flux_fit, sky_fit, _ = _linear_params_and_chi2(
                                        M_norm, obs_cut.astype(np.float32), inv_var.astype(np.float32)
                                    )
                                    model_cut   = flux_fit * M_norm + sky_fit
                                    res_cut_img = obs_cut - model_cut

                        # PSF postage stamp (same size as cutout)
                        psf_img = P.copy()

                # ── Plot ──────────────────────────────────────────────────────
                def _imshow(ax, img, cmap='gray', norm=None, label=''):
                    if img is None:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                transform=ax.transAxes, fontsize=8)
                        ax.set_xticks([]); ax.set_yticks([])
                        return
                    ax.imshow(img.T, origin='lower', cmap=cmap, norm=norm,
                              interpolation='nearest', aspect='equal')
                    ax.set_xticks([]); ax.set_yticks([])

                # Use symmetric stretch around obs median for all panels
                if obs_cut is not None:
                    vmed = float(np.nanmedian(obs_cut))
                    vstd = float(np.nanstd(obs_cut))
                    vlo  = vmed - 2 * vstd
                    vhi  = vmed + 5 * vstd
                    norm_obs = Normalize(vmin=vlo, vmax=vhi)
                else:
                    norm_obs = None

                _imshow(ax_obs, obs_cut,     cmap='afmhot', norm=norm_obs)
                _imshow(ax_mod, model_cut,   cmap='afmhot', norm=norm_obs)
                _imshow(ax_res, res_cut_img, cmap='RdBu_r',
                        norm=None if res_cut_img is None else
                             Normalize(vmin=-3*vstd, vmax=3*vstd))
                _imshow(ax_psf, psf_img,     cmap='afmhot')

                # Row label on left
                Re   = float(src_row['Re_arcsec'])
                n    = float(src_row['sersic_n'])
                q    = float(src_row['axis_ratio'])
                chi2 = float(src_row['chi2_red'])
                nim  = int(src_row['n_images'])
                lbl  = (f"{source_id}\n"
                        f"Re={Re:.3f}\" n={n:.1f} q={q:.2f}\n"
                        f"χ²={chi2:.1f}  nim={nim}  {tier_lbl}")
                ax_obs.set_ylabel(lbl, fontsize=6, rotation=0, ha='right',
                                  va='center', labelpad=60)

            fig.suptitle(f"{field_name} — eGSF diagnostics (page {page_idx+1}/{n_pages})",
                         fontsize=10, y=1.01)
            fig.tight_layout(rect=[0.18, 0, 1, 1])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"  Page {page_idx+1}/{n_pages} written")

    print(f"Saved → {out_pdf}")
    return Path(out_pdf)


# ── Main entry point ───────────────────────────────────────────────────────────

def run_egsf(
    field_name: str,
    output_dir: str | Path,
    bp3m_results_subdir: str = 'BP3M_results',
    pa_tol: float = 5.0,
    chi2_nsigma: float = 3.0,
    conc_cut: float = 1.3,
    match_radius_arcsec: float = 0.5,
    min_detections: int = 2,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Run eGSF background galaxy candidate identification.

    Parameters
    ----------
    field_name
        Field name (e.g. 'Leo_I').
    output_dir
        Root output directory (e.g. '/path/GaiaHub_results/Leo_I').
    bp3m_results_subdir
        Subdirectory containing BP3M v1 results (default 'BP3M_results').
    pa_tol
        Half-width in degrees for PA_V3 epoch grouping (default 5.0 deg).
    chi2_nsigma
        Number of MAD-sigmas above stellar locus for chi2 galaxy cut (default 3).
    conc_cut
        Concentration threshold for galaxy classification (default 1.3).
    match_radius_arcsec
        Sky matching radius for cross-epoch association (default 0.5").
    min_detections
        Minimum detections across epochs to keep a galaxy candidate (default 2).
    force_rerun
        Overwrite existing output (default False).
    """
    output_dir = Path(output_dir)
    egsf_dir   = output_dir / 'egsf'
    out_gal    = egsf_dir / f'{field_name}_galaxy_candidates.csv'
    out_dets   = egsf_dir / f'{field_name}_galaxy_detections.csv'
    out_epochs = egsf_dir / f'{field_name}_epoch_groups.csv'

    if out_gal.exists() and not force_rerun:
        print(f"eGSF: loading existing catalogue ({out_gal.name})")
        return pd.read_csv(out_gal)

    egsf_dir.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"eGSF: background galaxy candidate identification — {field_name}")
    print(f"{'='*60}")

    # ── 1. Image metadata & epoch groups ──────────────────────────────────────
    print("\nStep 1 — grouping images by epoch")
    epoch_groups = group_images_by_epoch(output_dir, bp3m_results_subdir, pa_tol)

    epoch_rows = []
    for key, obs_ids in sorted(epoch_groups.items()):
        print(f"  {key:45s}: {len(obs_ids):2d} images  {obs_ids}")
        epoch_rows.append({'epoch_key': key, 'n_images': len(obs_ids),
                           'obs_ids': ','.join(obs_ids)})
    pd.DataFrame(epoch_rows).to_csv(out_epochs, index=False)
    print(f"  Saved epoch groups → {out_epochs.name}")

    # ── 2. Gaia-matched source indices ────────────────────────────────────────
    print("\nStep 2 — loading Gaia cross-match catalogue")
    cc_path = output_dir / 'cross_match_catalog.csv'
    gaia_set = _build_gaia_matched_set(cc_path)
    n_gaia_total = sum(len(v) for v in gaia_set.values())
    print(f"  {len(gaia_set)} images with Gaia-matched sources  ({n_gaia_total} source-detections)")

    # ── 3. Load FLC metadata for each obs_id ──────────────────────────────────
    meta = _get_image_metadata(output_dir, bp3m_results_subdir)
    meta_dict = {row.obs_id: row for _, row in meta.iterrows()}
    hst_dir = output_dir / 'HST' / 'mastDownload' / 'HST'

    # ── 4. Per-epoch galaxy candidate extraction & cross-matching ──────────────
    print(f"\nStep 3 — extracting galaxy candidates")
    print(f"  Cuts: chi2 > stellar_median + {chi2_nsigma}*MAD_sigma  OR  "
          f"concentration > {conc_cut}  OR  is_star_candidate=False")

    all_candidates: list[pd.DataFrame] = []
    epoch_summary_rows = []

    for epoch_key, obs_ids in sorted(epoch_groups.items()):
        print(f"\n  {epoch_key}")
        epoch_dfs: list[pd.DataFrame] = []

        for obs_id in obs_ids:
            flc_path = hst_dir / obs_id / f'{obs_id}_flc.fits'
            cat_path = hst_dir / obs_id / f'{obs_id}_flc_catalog.fits'
            if not cat_path.exists():
                print(f"    {obs_id}: catalog not found — skipping")
                continue

            g_set = gaia_set.get(obs_id, set())
            df = load_galaxy_candidates(
                str(cat_path), obs_id, g_set,
                chi2_nsigma=chi2_nsigma, conc_cut=conc_cut,
            )
            n_nongaia = (~df['is_gaia_matched']).sum() if len(df) else 0
            n_gaia_ext = (df['is_gaia_matched'] & ~df['is_star_candidate']).sum() if len(df) else 0
            print(f"    {obs_id}: {len(df):4d} galaxy candidates  "
                  f"({n_nongaia} non-Gaia, {n_gaia_ext} Gaia-extended)")
            epoch_dfs.append(df)

        if not epoch_dfs or sum(len(d) for d in epoch_dfs) == 0:
            print(f"    No candidates in this epoch group")
            continue

        # Cross-match across images in this epoch group
        src_df, det_df = _cross_match_epoch_group(
            epoch_dfs,
            match_radius_arcsec=match_radius_arcsec,
            min_detections=min_detections,
        )
        src_df['epoch_key'] = epoch_key
        det_df['epoch_key'] = epoch_key

        n_images = len([d for d in epoch_dfs if len(d) > 0])
        n_single = sum(len(d) for d in epoch_dfs)
        n_multi  = len(src_df)
        print(f"    → {n_single} single-image candidates → "
              f"{n_multi} cross-matched (≥{min_detections} epochs)")

        all_candidates.append((src_df, det_df))
        epoch_summary_rows.append({
            'epoch_key': epoch_key,
            'n_images_with_candidates': n_images,
            'n_single_image': n_single,
            'n_multi_epoch': n_multi,
        })

    # ── 5. Consolidate and save ───────────────────────────────────────────────
    print(f"\nStep 4 — saving catalogue")
    if not all_candidates:
        print("  WARNING: no multi-epoch galaxy candidates found")
        pd.DataFrame().to_csv(out_gal, index=False)
        return pd.DataFrame()

    all_src_dfs = [pair[0] for pair in all_candidates]
    all_det_dfs = [pair[1] for pair in all_candidates]

    result     = pd.concat(all_src_dfs, ignore_index=True)
    det_result = pd.concat(all_det_dfs, ignore_index=True)

    # Assign globally unique source_ids combining epoch_key + per-epoch source_id
    result['source_id'] = (
        result['epoch_key'].str.replace('/', '_').str.replace(' ', '')
        + '_' + result['source_id'].astype(str)
    )
    det_result['source_id'] = (
        det_result['epoch_key'].str.replace('/', '_').str.replace(' ', '')
        + '_' + det_result['source_id'].astype(str)
    )

    # Sort by number of detections descending, then by chi2
    result = result.sort_values(['n_detections', 'chi2_mean'],
                                ascending=[False, False]).reset_index(drop=True)

    result.to_csv(out_gal, index=False)
    det_result.to_csv(out_dets, index=False)

    n_gaia_in_result = result['is_gaia_matched_any'].sum()
    print(f"  Total multi-epoch galaxy candidates:  {len(result)}")
    print(f"  Of which Gaia-matched (extended):     {n_gaia_in_result}")
    print(f"  Of which non-Gaia (faint background): {len(result) - n_gaia_in_result}")
    print(f"  Saved sources     → {out_gal}")
    print(f"  Saved detections  → {out_dets.name}  ({len(det_result)} rows; includes x/y/chip_ext for cutouts)")
    print(f"  Median pos scatter: {result['pos_rms_mas'].median():.1f} mas")
    print(f"  Median chi2:        {result['chi2_mean'].median():.2f}")
    print(f"  Median concentration: {result['conc_mean'].median():.2f}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run eGSF background galaxy candidate identification')
    parser.add_argument('field_name', help='Field name (e.g. Leo_I)')
    parser.add_argument('output_dir', help='Field root directory')
    parser.add_argument('--bp3m_results', default='BP3M_results',
                        help='BP3M v1 results subdirectory (default: BP3M_results)')
    parser.add_argument('--pa_tol', type=float, default=5.0,
                        help='PA_V3 grouping tolerance in degrees (default: 5.0)')
    parser.add_argument('--chi2_nsigma', type=float, default=3.0,
                        help='Chi2 cut above stellar MAD-sigma locus (default: 3.0)')
    parser.add_argument('--conc_cut', type=float, default=1.3,
                        help='Concentration threshold (default: 1.3)')
    parser.add_argument('--match_radius', type=float, default=0.5,
                        help='Sky match radius in arcsec (default: 0.5)')
    parser.add_argument('--min_detections', type=int, default=2,
                        help='Min epochs to keep a galaxy candidate (default: 2)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output')
    parser.add_argument('--morphology', action='store_true',
                        help='Also run measure_galaxy_morphology after identification')
    parser.add_argument('--lib_dir', default=None,
                        help='bp3m lib directory (for STDPSFs); default: auto-detect')
    parser.add_argument('--cutout_half', type=int, default=_CUTOUT_HALF,
                        help=f'Cutout half-width in pixels (default: {_CUTOUT_HALF}, '
                             f'giving {2*_CUTOUT_HALF+1}×{2*_CUTOUT_HALF+1} px)')
    parser.add_argument('--fit', action='store_true',
                        help='Also run fit_egsf_sources (Sérsic fitting) after morphology')
    parser.add_argument('--fit_min_det', type=int, default=3,
                        help='Min detections for Sérsic fit (default: 3)')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Save diagnostic cutout PDF after fitting')
    parser.add_argument('--diag_n', type=int, default=24,
                        help='Number of sources in diagnostic PDF (default: 24)')
    args = parser.parse_args()

    run_egsf(
        field_name=args.field_name,
        output_dir=args.output_dir,
        bp3m_results_subdir=args.bp3m_results,
        pa_tol=args.pa_tol,
        chi2_nsigma=args.chi2_nsigma,
        conc_cut=args.conc_cut,
        match_radius_arcsec=args.match_radius,
        min_detections=args.min_detections,
        force_rerun=args.force,
    )

    if args.morphology:
        measure_galaxy_morphology(
            field_name=args.field_name,
            output_dir=args.output_dir,
            lib_dir=args.lib_dir,
            bp3m_results_subdir=args.bp3m_results,
            pa_tol=args.pa_tol,
            cutout_half=args.cutout_half,
            force_rerun=args.force,
        )

    if args.fit or args.diagnostics:
        fit_egsf_sources(
            field_name=args.field_name,
            output_dir=args.output_dir,
            lib_dir=args.lib_dir,
            bp3m_results_subdir=args.bp3m_results,
            pa_tol=args.pa_tol,
            cutout_half=args.cutout_half,
            min_detections=args.fit_min_det,
            force_rerun=args.force,
        )

    if args.diagnostics:
        plot_egsf_diagnostics(
            field_name=args.field_name,
            output_dir=args.output_dir,
            lib_dir=args.lib_dir,
            bp3m_results_subdir=args.bp3m_results,
            n_sources=args.diag_n,
            cutout_half=args.cutout_half,
        )


if __name__ == '__main__':
    _cli()
