"""
Data loader for the new HST pipeline whose outputs live under::

    {data_root}/{field_name}/HST/mastDownload/HST/{image_name}/

Expected files per image directory
------------------------------------
{img}_flc.fits              HST image (header-only needed; two SCI extensions)
{img}_flc_catalog.fits      PSF-fit source catalog (positions + covariances)
transformation.csv          Initial alignment onto the Gaia frame
matched_gaia.csv            HST↔Gaia cross-match (hst_index ↔ gaia_source_id)

Gaia catalog
------------
Loaded from {data_root}/{field_name}/Gaia/*_gaia.csv, same location and column
names as the existing load_image_data() loader.

Returns
-------
Same (images, stars_per_image, gaia_catalog) triple as load_image_data(), so
the rest of the BP3M pipeline (solver, run_bp3m.py, etc.) is unchanged.

Also writes a summary CSV
-------------------------
{data_root}/{field_name}/HST/image_transformation_summaries.csv
with one row per usable image, containing all metadata extracted from the
FITS headers and transformation.csv.  This file mimics the columns expected
by load_image_data() so the existing loader can read it if needed.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from bp3m.instrument_config import get_instrument_config

# Minimum position uncertainty floor applied to PSF-fit centroids before they
# reach BP3M.  Prevents astronomically large sigma_resid values for very bright
# stars whose pure-photon-noise PSF uncertainty is negligibly small compared to
# true astrometric error sources.  Overridable via --bp3m_pos_err_floor.
_MIN_POS_ERR_PX: float = 5e-3

# Maximum fraction of the PSF fitting window that may be saturated for a source
# to be eligible as an initial alignment star.  Half-width is read from
# psf_params.json (default 3 → 7×7 = 49 pixels).  Sources above this threshold
# start with use_for_alignment=False but can still be re-admitted by the BP3M
# Phase 2 EM loop through residual-based tests.
_MAX_SAT_FRAC: float = 0.25


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_filter(h0) -> str:
    """Return the science filter name, handling ACS (FILTER1/2) and WFC3 (FILTER)."""
    filt = h0.get("FILTER", "")
    if filt:
        return str(filt).strip()
    # ACS: one slot is CLEAR, the other is the science filter
    f1 = str(h0.get("FILTER1", "")).strip()
    f2 = str(h0.get("FILTER2", "")).strip()
    if "CLEAR" in f1.upper():
        return f2
    return f1


def _pixel_scale_from_cd(cd11, cd12, cd21, cd22) -> float:
    """Return pixel scale in mas/pix from the CD matrix."""
    return np.sqrt(abs(cd11 * cd22 - cd12 * cd21)) * 3600.0 * 1000.0


def _fcm_to_abcd(A, B, C, D, orig_rot_deg: float) -> np.ndarray:
    """
    Convert fast_cross_match transformation parameters to BP3M (a,b,c,d).

    The fast_cross_match affine maps HST pixels → Gaia pixels after applying
    orig_rot (= -orientat).  BP3M's (a,b,c,d) maps HST-centered pixels to the
    unrotated Gaia pseudo-image frame.  The relationship is:

        [[a,b],[c,d]] = R_cw(orig_rot_deg) @ [[A,B],[C,D]]

    where R_cw(θ) = [[cosθ, sinθ], [-sinθ, cosθ]] and Xo=Yo=2048 (BP3M pivot).
    The bulk image offset is now captured by Δα0/Δδ0 (free parameters) rather
    than by pixel translations w/z (removed).
    """
    deg2rad = np.pi / 180.0
    rot_rad = orig_rot_deg * deg2rad
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    R_cw = np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    M    = np.array([[A, B], [C, D]])
    abcd = (R_cw @ M).ravel()
    return np.array([abcd[0], abcd[1], abcd[2], abcd[3]])


def _read_image_meta(img_dir: Path, img_name: str) -> dict | None:
    """
    Extract all metadata needed by BP3MSolver from the FLC FITS header and
    transformation.csv.  Returns None if any required file is missing.
    """
    flc_path  = img_dir / f"{img_name}_flc.fits"
    tran_path = img_dir / "transformation.csv"

    if not flc_path.exists() or not tran_path.exists():
        return None

    # ── FITS header ───────────────────────────────────────────────────────────
    with fits.open(flc_path, memmap=False) as hdu:
        h0 = hdu[0].header
        h1 = hdu["SCI", 1].header

        instrument = str(h0.get("INSTRUME", "")).strip()
        detector   = str(h0.get("DETECTOR", "")).strip()
        filt       = _get_filter(h0)

        # Mid-exposure MJD
        expstart = float(h0["EXPSTART"])
        expend   = float(h0["EXPEND"])
        hst_time_mjd = 0.5 * (expstart + expend)

        # Pixel scale and rotation from primary SCI extension CD matrix
        cd11 = float(h1["CD1_1"]); cd12 = float(h1["CD1_2"])
        cd21 = float(h1["CD2_1"]); cd22 = float(h1["CD2_2"])
        real_pixel_scale_mas = _pixel_scale_from_cd(cd11, cd12, cd21, cd22)

    # ── transformation.csv ───────────────────────────────────────────────────
    tdf = pd.read_csv(tran_path).set_index("parameter")["value"]

    A, B, C, D = float(tdf["A"]),  float(tdf["B"]),  float(tdf["C"]),  float(tdf["D"])
    xs_o = float(tdf["xs_o"]); ys_o = float(tdf["ys_o"])
    xt_o = float(tdf["xt_o"]); yt_o = float(tdf["yt_o"])
    x_cen = float(tdf["x_cen"]); y_cen = float(tdf["y_cen"])
    ra_cen  = float(tdf["ra_cen"]); dec_cen = float(tdf["dec_cen"])
    pscale_arcsec = float(tdf["pixel_scale"])       # arcsec/pix
    orientat      = float(tdf["orientat"])           # degrees, like PA_APER
    rot_deg       = float(tdf["rot_deg"])
    ratio         = float(tdf.get("ratio",   1.0))
    on_skew       = float(tdf.get("on_skew", 0.0))
    off_skew      = float(tdf.get("off_skew", 0.0))

    orig_pixel_scale_mas = pscale_arcsec * 1000.0   # mas/pix

    meta = {
        # Pointing / tangent point
        "ra0":  ra_cen,
        "dec0": dec_cen,
        # Pixel scale and rotation
        # orientat is the PA of the HST image y-axis (N→E), which is the
        # negative of the BP3M rotation angle θ (arctan2(b-c, a+d)).
        # Empirically: orientat_Fornax = +112.48° ↔ orig_rot_Fornax = -112.48°,
        # so orig_rot_deg = -orientat.
        "pixel_scale":     real_pixel_scale_mas,   # mas/pix (from CD matrix)
        "rotation_deg":    -orientat,
        "orig_pixel_scale": orig_pixel_scale_mas,  # mas/pix (nominal, from CSV)
        "orig_rot_deg":    -orientat,
        "pixel_scale_ratio": ratio,
        # Per-instrument config: initial scale ratio (prior mean) and hyperprior
        # sigma widths.  Sourced from instrument_config so cross-match and the
        # BP3M solver always use consistent values.  Solver uses these as
        # per-image defaults; explicit CLI overrides take precedence.
        "initial_scale_ratio": get_instrument_config(instrument, detector)["initial_scale"],
        **{k: v for k, v in get_instrument_config(instrument, detector).items()
           if k.startswith("sigma_")},
        "on_skew":  on_skew,
        "off_skew": off_skew,
        # Timing
        "hst_time_mjd": hst_time_mjd,
        # Instrument
        "instrument": instrument,
        "detector":   detector,
        "filter":     filt,
        # Centering pivot (used as Xo, Yo in X_mat construction)
        "Xo": x_cen,
        "Yo": y_cen,
        # Gaia pseudo-image pivot (target of initial transformation)
        "Wo": xt_o,
        "Zo": yt_o,
        # Initial transformation coefficients
        "AG": A, "BG": B, "CG": C, "DG": D,
        # Source pivot (needed to reconstruct the full initial transform)
        "xs_o": xs_o, "ys_o": ys_o,
        "xt_o": xt_o, "yt_o": yt_o,
        # Pre-converted BP3M (a,b,c,d) from the cross-match solution.
        # Used to initialise r_hat before the first EM iteration; the prior
        # (r_prior, C_r_prior_inv) is still derived from the WCS header and
        # is unaffected by this value.
        "fcm_abcd": _fcm_to_abcd(A, B, C, D, -orientat),
        # Per-chip tangent-point info (lo = lower chip, hi = upper chip).
        # Populated from the catalog FITS header (CHIP{n}_CRPIX{1,2}_GDC /
        # CHIP{n}_CRVAL{1,2}) when the catalog file is present.  None otherwise.
        "chip_pointing_lo": None,
        "chip_pointing_hi": None,
    }

    # ── Per-chip pointing from catalog FITS (optional) ────────────────────────
    cat_path = img_dir / f"{img_name}_flc_catalog.fits"
    if cat_path.exists():
        try:
            with fits.open(cat_path, memmap=False) as cat_hdu:
                cat_hdr = cat_hdu[1].header
            # Collect all CHIP prefixes that have both _CRPIX1_GDC and CRVAL1
            prefixes = sorted({
                k.split("_CRPIX1_GDC")[0]
                for k in cat_hdr.keys()
                if k.endswith("_CRPIX1_GDC") and k.startswith("CHIP")
            })
            chip_pts = []
            for pfx in prefixes:
                xo  = cat_hdr.get(f"{pfx}_CRPIX1_GDC")
                yo  = cat_hdr.get(f"{pfx}_CRPIX2_GDC")
                ra0 = cat_hdr.get(f"{pfx}_CRVAL1")
                dc0 = cat_hdr.get(f"{pfx}_CRVAL2")
                if all(v is not None for v in [xo, yo, ra0, dc0]):
                    chip_pts.append({
                        "Xo": float(xo), "Yo": float(yo),
                        "ra0": float(ra0), "dec0": float(dc0),
                    })
            if len(chip_pts) == 2:
                # Sort by Yo: smaller Yo → lower chip (_lo), larger → upper (_hi)
                chip_pts.sort(key=lambda p: p["Yo"])
                meta["chip_pointing_lo"] = chip_pts[0]
                meta["chip_pointing_hi"] = chip_pts[1]
        except Exception:
            pass  # catalog absent or malformed — fall back to shared pointing

    return meta


def _build_stars_df(img_dir: Path, img_name: str,
                    gaia_float_to_int64: dict | None = None,
                    pos_err_floor: float = _MIN_POS_ERR_PX) -> pd.DataFrame | None:
    """
    Build the per-image source DataFrame expected by BP3MSolver.

    Columns:  Gaia_id, X, Y, X_orig, Y_orig,
              x_hst_err, y_hst_err, xy_hst_corr,
              q_hst, mag, use_for_alignment, use_for_fit

    X, Y       = x_gdc, y_gdc  (GDC-corrected positions, full-frame pixels)
    X_orig, Y_orig = x, y      (raw detector positions,  full-frame pixels)
    Errors derived from the GDC covariance matrix (cov_xx/yy/xy_gdc).

    gaia_float_to_int64 : float64-keyed dict mapping float(source_id) → int64
        source_id.  Required to safely recover exact int64 Gaia source IDs
        from matched_gaia.csv files where gaia_source_id may have been stored
        as float64 (scientific notation), which loses the last ~3 digits of
        the 19-digit integer.  When None, a direct int64 cast is used (safe
        only when the CSV was written with integer formatting).
    """
    cat_path   = img_dir / f"{img_name}_flc_catalog.fits"
    match_path = img_dir / "matched_gaia.csv"

    if not cat_path.exists() or not match_path.exists():
        return None

    # ── Source catalog ────────────────────────────────────────────────────────
    with fits.open(cat_path, memmap=False) as cat_hdu:
        tbl = cat_hdu[1].data
        cat_x      = tbl["x"].astype(float)
        cat_y      = tbl["y"].astype(float)
        cat_xgdc   = tbl["x_gdc"].astype(float)
        cat_ygdc   = tbl["y_gdc"].astype(float)
        cat_cov_xx = tbl["cov_xx_gdc"].astype(float)
        cat_cov_yy = tbl["cov_yy_gdc"].astype(float)
        cat_cov_xy = tbl["cov_xy_gdc"].astype(float)
        cat_mag    = tbl["mag"].astype(float)
        cat_qfit   = tbl["qfit"].astype(float)
        cat_n_sat  = tbl["n_sat"].astype(int)

    # ── Matched Gaia cross-match ──────────────────────────────────────────────
    match = pd.read_csv(match_path)

    hst_idx = match["hst_index"].to_numpy(int)

    # Gaia source IDs may have been written as float64 scientific notation in
    # matched_gaia.csv (e.g. "3.88155130191373e+18"), which loses the last ~3
    # digits of the 19-digit integer.  Detect the column dtype: when pandas
    # reads integer strings it returns int64; when it reads scientific notation
    # it returns float64.  Only use the float→int64 lookup for the latter case
    # to avoid collisions introduced by the float rounding step.
    sid_col = match["gaia_source_id"]
    if sid_col.dtype == np.float64 and gaia_float_to_int64 is not None:
        raw_floats = sid_col.to_numpy(float)
        _sentinel  = np.int64(-1)
        gaia_ids   = np.array(
            [gaia_float_to_int64.get(f, _sentinel) for f in raw_floats],
            dtype=np.int64)
        # Drop rows whose float ID couldn't be mapped (shouldn't happen for
        # stars that originated from this Gaia catalog, but guard just in case)
        valid = gaia_ids != _sentinel
        if not valid.all():
            hst_idx  = hst_idx[valid]
            gaia_ids = gaia_ids[valid]
    else:
        gaia_ids = sid_col.to_numpy(np.int64)

    # Pull catalog values for matched rows
    x_orig = cat_x[hst_idx]
    y_orig = cat_y[hst_idx]
    x_gdc  = cat_xgdc[hst_idx]
    y_gdc  = cat_ygdc[hst_idx]
    cov_xx = cat_cov_xx[hst_idx]
    cov_yy = cat_cov_yy[hst_idx]
    cov_xy = cat_cov_xy[hst_idx]
    mag    = cat_mag[hst_idx]
    qfit   = cat_qfit[hst_idx]
    n_sat  = cat_n_sat[hst_idx]

    # Convert covariance to σ and correlation
    sig_x = np.sqrt(np.maximum(cov_xx, 0.0))
    sig_y = np.sqrt(np.maximum(cov_yy, 0.0))

    # Apply minimum position uncertainty floor
    sig_x = np.maximum(sig_x, pos_err_floor)
    sig_y = np.maximum(sig_y, pos_err_floor)

    denom = sig_x * sig_y
    corr  = np.where(denom > 0, cov_xy / denom, 0.0)
    corr  = np.clip(corr, -0.9999, 0.9999)

    # Saturation fraction criterion: use psf_params.json half_width if available
    psf_params_path = img_dir / "psf_params.json"
    if psf_params_path.exists():
        with open(psf_params_path) as _f:
            _psf_params = json.load(_f)
        half_width = int(_psf_params.get("half_width", 3))
    else:
        half_width = 3
    window_area = (2 * half_width + 1) ** 2
    ok_sat = (n_sat / window_area) < _MAX_SAT_FRAC

    df = pd.DataFrame({
        "Gaia_id":         gaia_ids,
        "X":               x_gdc,
        "Y":               y_gdc,
        "X_orig":          x_orig,
        "Y_orig":          y_orig,
        "x_hst_err":       sig_x,
        "y_hst_err":       sig_y,
        "xy_hst_corr":     corr,
        "q_hst":           qfit,
        "mag":             mag,
        "use_for_alignment": ok_sat,
        "use_for_fit":       ok_sat,
    })
    return df


def _build_delve_only_stars_df(
    img_dir: Path,
    img_name: str,
    gaia_hst_indices: set,
    delve_id_to_gaia_id: dict,
    pos_err_floor: float = _MIN_POS_ERR_PX,
    delve_use_for_align: bool = False,
) -> pd.DataFrame | None:
    """
    Build a stars_df for DELVE-only detections (hst_index not in matched_gaia.csv).
    Reads positions and errors from the PSF catalog by hst_index.

    Parameters
    ----------
    gaia_hst_indices   : set of hst_index values already claimed by matched_gaia.csv
    delve_id_to_gaia_id: dict mapping delve_source_id (int64) → synthetic negative Gaia_id
    """
    cat_path   = img_dir / f"{img_name}_flc_catalog.fits"
    delve_path = img_dir / "matched_delve.csv"
    if not cat_path.exists() or not delve_path.exists():
        return None

    delve = pd.read_csv(delve_path)
    if 'hst_index' not in delve.columns or 'delve_source_id' not in delve.columns:
        return None
    delve_only = delve[~delve['hst_index'].isin(gaia_hst_indices)].copy()
    if len(delve_only) == 0:
        return None

    with fits.open(cat_path, memmap=False) as hdu:
        tbl = hdu[1].data
        cat_x      = tbl["x"].astype(float)
        cat_y      = tbl["y"].astype(float)
        cat_xgdc   = tbl["x_gdc"].astype(float)
        cat_ygdc   = tbl["y_gdc"].astype(float)
        cat_cov_xx = tbl["cov_xx_gdc"].astype(float)
        cat_cov_yy = tbl["cov_yy_gdc"].astype(float)
        cat_cov_xy = tbl["cov_xy_gdc"].astype(float)
        cat_mag    = tbl["mag"].astype(float)
        cat_qfit   = tbl["qfit"].astype(float)
        cat_n_sat  = tbl["n_sat"].astype(int)

    hst_idx       = delve_only['hst_index'].values.astype(int)
    delve_src_ids = delve_only['delve_source_id'].values.astype(np.int64)
    gaia_ids      = np.array([delve_id_to_gaia_id.get(int(did), 0)
                               for did in delve_src_ids], dtype=np.int64)
    # Drop any that have no mapping (shouldn't happen)
    valid = gaia_ids != 0
    if not valid.all():
        hst_idx       = hst_idx[valid]
        gaia_ids      = gaia_ids[valid]

    sig_x = np.maximum(np.sqrt(np.maximum(cat_cov_xx[hst_idx], 0.0)), pos_err_floor)
    sig_y = np.maximum(np.sqrt(np.maximum(cat_cov_yy[hst_idx], 0.0)), pos_err_floor)
    denom = sig_x * sig_y
    corr  = np.where(denom > 0, cat_cov_xy[hst_idx] / denom, 0.0)
    corr  = np.clip(corr, -0.9999, 0.9999)

    half_width  = 3
    window_area = (2 * half_width + 1) ** 2
    ok_sat = (cat_n_sat[hst_idx] / window_area) < _MAX_SAT_FRAC

    return pd.DataFrame({
        "Gaia_id":         gaia_ids,
        "X":               cat_xgdc[hst_idx],
        "Y":               cat_ygdc[hst_idx],
        "X_orig":          cat_x[hst_idx],
        "Y_orig":          cat_y[hst_idx],
        "x_hst_err":       sig_x,
        "y_hst_err":       sig_y,
        "xy_hst_corr":     corr,
        "q_hst":           cat_qfit[hst_idx],
        "mag":             cat_mag[hst_idx],
        "use_for_alignment": ok_sat if delve_use_for_align else np.zeros(len(gaia_ids), dtype=bool),
        "use_for_fit":       ok_sat,
    })


# ── CCD/amplifier split utilities ────────────────────────────────────────────
# Moved here from data_loader.py so that all active pipeline code imports from
# data_loader_flc and data_loader.py (legacy loader) is unused.

_AMP_SPLITS = {
    "ACSWFC":   {"x_split": 2048.0, "y_split": 2048.0},
    "WFC3UVIS": {"x_split": 2048.0, "y_split": 2047.0},
}
_AMP_SPLITS_DEFAULT = {"x_split": 2048.0, "y_split": 2048.0}


def _get_amp_splits(meta: dict) -> dict:
    """Return the x_split / y_split dict for the instrument+detector in *meta*."""
    key = (meta.get("instrument", "") + meta.get("detector", "")).upper()
    return _AMP_SPLITS.get(key, _AMP_SPLITS_DEFAULT)


def split_images_by_ccd(images, stars_per_image, min_stars_per_ccd: int = 20):
    """
    Split each image into two independent CCD halves along the Y boundary.

    For ACS/WFC the chips meet at Y_orig = 2048; for WFC3/UVIS at Y_orig = 2047
    (chip 1 starts at y_combined=2048, so lo: y≤2047, hi: y≥2048).
    The boundary is looked up per-image from ``_AMP_SPLITS`` using the
    ``instrument`` and ``detector`` fields stored in *images*.

    Each half inherits identical pointing/scale metadata from its parent and
    is initialised from the same r_prior.  Fitting them independently gives
    each physical CCD its own r_j vector, which is correct because they have
    independent distortion patterns.

    Naming convention
    -----------------
    ``{img}_lo``  — stars with Y_orig ≤ y_split  (lower chip)
    ``{img}_hi``  — stars with Y_orig >  y_split  (upper chip)

    If all stars in an image fall on one side, only that entry is created.
    If either half has fewer than *min_stars_per_ccd* stars, the image is kept
    whole (not split) so the transformation can still be constrained.

    Parameters
    ----------
    images : dict[str -> dict]
        Image metadata (as returned by load_image_data_flc).
        Must contain ``instrument`` and ``detector`` keys.
    stars_per_image : dict[str -> pd.DataFrame]
        Per-image source tables.  Must contain ``Y_orig`` (falls back to ``Y``).
    min_stars_per_ccd : int
        Minimum stars required on each CCD half to allow splitting.  Images
        where either half falls below this threshold are kept unsplit.
        Default: 20.

    Returns
    -------
    new_images, new_stars_per_image : dicts with ``_lo`` / ``_hi`` suffixes
        for split images, or unchanged keys for images kept whole.
    """
    new_images = {}
    new_spi = {}

    for img in sorted(images.keys()):
        meta   = images[img]
        df     = stars_per_image[img]
        y_col  = 'Y_orig' if 'Y_orig' in df.columns else 'Y'
        y_vals = df[y_col].to_numpy(float)
        y_split = _get_amp_splits(meta)["y_split"]

        lo_mask = y_vals <= y_split
        hi_mask = y_vals >  y_split
        n_lo, n_hi = lo_mask.sum(), hi_mask.sum()

        if n_lo < min_stars_per_ccd or n_hi < min_stars_per_ccd:
            print(f"    {img}: lo={n_lo}, hi={n_hi} stars — below "
                  f"min_stars_per_ccd={min_stars_per_ccd}, keeping unsplit")
            new_images[img] = dict(meta)
            new_spi[img]    = df
            continue

        for suffix, mask in [('_lo', lo_mask), ('_hi', hi_mask)]:
            sub_df = df[mask].reset_index(drop=True)
            if len(sub_df) == 0:
                continue
            sub_meta = dict(meta)
            cp = meta.get(f"chip_pointing{suffix}")  # chip_pointing_lo / _hi
            if cp is not None:
                sub_meta["ra0"]  = cp["ra0"]
                sub_meta["dec0"] = cp["dec0"]
                sub_meta["Xo"]   = cp["Xo"]
                sub_meta["Yo"]   = cp["Yo"]
            new_images[img + suffix] = sub_meta
            new_spi[img + suffix]    = sub_df

    return new_images, new_spi


def build_index_maps(stars_per_image, gaia_catalog):
    """
    Build integer index maps for vectorized access.

    Returns
    -------
    star_id_to_idx : dict[Gaia_id -> int]
    image_names : list[str]   (sorted)
    star_in_image : dict[image_name -> np.ndarray of star indices]
        Maps per-image rows to global star indices.
    """
    star_id_to_idx = {gid: i for i, gid in enumerate(gaia_catalog["Gaia_id"])}
    image_names = sorted(stars_per_image.keys())

    star_in_image = {}
    for img in image_names:
        df = stars_per_image[img]
        idxs = np.array([star_id_to_idx[gid] for gid in df["Gaia_id"]
                         if gid in star_id_to_idx], dtype=int)
        mask = df["Gaia_id"].isin(star_id_to_idx)
        star_in_image[img] = idxs

    return star_id_to_idx, image_names, star_in_image


# ── Public entry point ────────────────────────────────────────────────────────

def load_image_data_flc(data_root, field_name: str,
                        pos_err_floor: float = _MIN_POS_ERR_PX,
                        restrict_images: "set[str] | frozenset[str] | None" = None,
                        gaia_csv: "str | Path | list | None" = None,
                        use_delve: bool = False,
                        delve_use_for_align: bool = False):
    """
    Load BP3M inputs from the new FLC-based pipeline layout.

    Directory layout expected::

        {data_root}/{field_name}/
            HST/mastDownload/HST/
                {img_name}/
                    {img_name}_flc.fits
                    {img_name}_flc_catalog.fits
                    transformation.csv
                    matched_gaia.csv
            Gaia/
                *_gaia.csv   (same format as the existing pipeline)

    Returns
    -------
    images : dict[image_name → dict]
    stars_per_image : dict[image_name → pd.DataFrame]
    gaia_catalog : pd.DataFrame
        Same types and column conventions as load_image_data().
    """
    data_root  = Path(data_root)
    field_path = data_root / field_name
    hst_root   = field_path / "HST" / "mastDownload" / "HST"
    gaia_dir   = field_path / "Gaia"

    if not hst_root.exists():
        raise FileNotFoundError(f"HST image directory not found: {hst_root}")

    # ── Gaia catalog ──────────────────────────────────────────────────────────
    if gaia_csv is not None:
        if isinstance(gaia_csv, (list, tuple)):
            gaia_paths = [Path(p) for p in gaia_csv]
            missing = [p for p in gaia_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Gaia CSV(s) not found: {missing}")
            gaia_frames = [
                pd.read_csv(p).rename(columns={"SOURCE_ID": "source_id"})
                for p in gaia_paths
            ]
            gaia_raw = (pd.concat(gaia_frames, ignore_index=True)
                        .drop_duplicates("source_id"))
            print(f"Loading {len(gaia_paths)} Gaia catalogs "
                  f"({len(gaia_raw)} unique stars)")
        else:
            gaia_csv = Path(gaia_csv)
            if not gaia_csv.exists():
                raise FileNotFoundError(f"Gaia CSV not found: {gaia_csv}")
            print(f"Loading Gaia catalog: {gaia_csv.name}")
            gaia_raw = pd.read_csv(gaia_csv).rename(columns={"SOURCE_ID": "source_id"})
    else:
        gaia_files = sorted(glob.glob(str(gaia_dir / "*_gaia.csv")))
        if not gaia_files:
            raise FileNotFoundError(f"No Gaia catalog files found in {gaia_dir}")
        if len(gaia_files) > 1:
            print(f"Loading {len(gaia_files)} Gaia CSV files (concatenated).")
        gaia_frames = [
            pd.read_csv(f).rename(columns={"SOURCE_ID": "source_id"})
            for f in gaia_files
        ]
        gaia_raw = pd.concat(gaia_frames, ignore_index=True).drop_duplicates("source_id")

    gaia_cols = [
        "source_id", "ra", "dec", "ra_error", "dec_error", "ra_dec_corr",
        "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
        "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
        "parallax", "parallax_error", "parallax_pmra_corr", "parallax_pmdec_corr",
        "pmra", "pmra_error", "pmra_pmdec_corr", "pmdec", "pmdec_error",
        "ref_epoch", "ruwe", "pseudocolour", "gmag", "gmag_error",
        "bpmag", "bpmag_error", "rpmag", "rpmag_error", "bp_rp", "bp_rp_error",
    ]
    gaia_catalog = (
        gaia_raw[[c for c in gaia_cols if c in gaia_raw.columns]]
        .dropna(subset=["ra", "dec"])
        .sort_values("source_id")
        .drop_duplicates("source_id", keep="first")
        .reset_index(drop=True)
    )
    gaia_catalog.rename(
        columns={"source_id": "Gaia_id", "ref_epoch": "Gaia_time"}, inplace=True
    )

    # Build a float64-keyed lookup for recovering exact int64 Gaia IDs from
    # matched_gaia.csv files that stored gaia_source_id as float (scientific
    # notation).  float(int64_id) is the same rounding that pandas applies when
    # writing an int64 column as float, so the dict keys match what the CSV
    # reader returns even when precision is lost.
    gaia_float_to_int64: dict[float, np.int64] = {
        float(gid): np.int64(gid) for gid in gaia_catalog["Gaia_id"].values
    }

    # ── Per-image data ────────────────────────────────────────────────────────
    img_dirs = sorted(p for p in hst_root.iterdir() if p.is_dir())

    images: dict          = {}
    stars_per_image: dict = {}
    summary_rows: list    = []
    keep_gaia_mask        = np.zeros(len(gaia_catalog), dtype=bool)

    gaia_id_set = set(gaia_catalog["Gaia_id"].values)

    skipped = []
    for img_dir in img_dirs:
        img_name = img_dir.name
        if restrict_images is not None and img_name not in restrict_images:
            continue

        meta = _read_image_meta(img_dir, img_name)
        if meta is None:
            skipped.append(img_name)
            continue

        stars_df = _build_stars_df(img_dir, img_name, gaia_float_to_int64, pos_err_floor)
        if stars_df is None or len(stars_df) == 0:
            skipped.append(img_name)
            continue

        # Keep only stars that are in the Gaia catalog
        in_gaia = stars_df["Gaia_id"].isin(gaia_id_set)
        stars_df = stars_df[in_gaia].reset_index(drop=True)
        if len(stars_df) == 0:
            skipped.append(img_name)
            continue

        images[img_name]          = meta
        stars_per_image[img_name] = stars_df

        # Track which Gaia sources are observed
        keep_gaia_mask |= gaia_catalog["Gaia_id"].isin(stars_df["Gaia_id"])

        summary_rows.append({
            "image_name":       img_name,
            "ra":               meta["ra0"],
            "dec":              meta["dec0"],
            "real_img_pix_scale": meta["pixel_scale"],
            "rot":              meta["rotation_deg"],
            "orig_pixel_scale": meta["orig_pixel_scale"],
            "orig_rot":         meta["orig_rot_deg"],
            "pix_scale_ratio":  meta["pixel_scale_ratio"],
            "on_axis_skew":     meta["on_skew"],
            "off_axis_skew":    meta["off_skew"],
            "HST_time":         meta["hst_time_mjd"],
            "instrument":       meta["instrument"],
            "detector":         meta["detector"],
            "filter":           meta["filter"],
            "Xo":               meta["Xo"],
            "Yo":               meta["Yo"],
            "Wo":               meta["Wo"],
            "Zo":               meta["Zo"],
            "AG":               meta["AG"],
            "BG":               meta["BG"],
            "CG":               meta["CG"],
            "DG":               meta["DG"],
            "xs_o":             meta["xs_o"],
            "ys_o":             meta["ys_o"],
            "xt_o":             meta["xt_o"],
            "yt_o":             meta["yt_o"],
            "n_matched":        len(stars_df),
        })

    if skipped:
        print(f"  Skipped {len(skipped)} directories (missing required files): "
              f"{skipped}")

    # ── Cross-match catalog: per-image trustworthiness ────────────────────────
    # cross_match_catalog.csv (one row per star×filter_camera) records which
    # images each star is trustworthy in.  Initial use_for_alignment rules:
    #
    #   Global exclusion (all images):
    #     n_trustworthy <= 1               → excluded (poor cross-match consistency)
    #     is_star_any_image == False        → excluded (classified non-star everywhere)
    #
    #   Per-image exclusion (specific star×image pair):
    #     img in outlier_images            → excluded (cross-match photometric outlier)
    #     (non_star_images is NOT applied per-image: per-exposure PSF-fit
    #      classifications are unreliable for faint stars in ACS data.
    #      The global is_star_any_image=False guard is sufficient.)
    #
    # All excluded pairs remain candidates for re-admission by the Phase 2 EM
    # residual tests — exclusion here only affects the initial alignment sample.
    # The is_star tier (all > any > neither) naturally emerges from how populated
    # non_star_images is: is_star_all_images=True stars have no non_star_images.
    xm_path = field_path / "cross_match_catalog.csv"
    xm = None
    if xm_path.exists():
        xm = pd.read_csv(xm_path, dtype={"gaia_source_id": str})
        has_star_cols = "is_star_any_image" in xm.columns

        def _parse_imglist(raw) -> set:
            if pd.isna(raw) or str(raw).strip() == "":
                return set()
            return {s.strip() for s in str(raw).split(",") if s.strip()}

        # Build (gaia_source_id, img_name) → trusted bool
        trust_lookup: dict[tuple[str, str], bool] = {}
        for _, row in xm.iterrows():
            sid      = str(row["gaia_source_id"])
            img_list = [s.strip() for s in str(row["image_list"]).split(",") if s.strip()]

            outlier_set  = _parse_imglist(row.get("outlier_images", ""))
            # Per-image non-star flags are unreliable: a faint star can fail the
            # PSF-fit quality criterion in one exposure but pass in another.
            # The global is_star_any_image=False check (above) is the conservative
            # guard; per-image non-star exclusion is too aggressive for ACS images
            # where ~30% of detections get misclassified in individual exposures.
            excluded_set = outlier_set

            n_trust   = int(row["n_trustworthy"])
            any_trust = bool(row["any_trustworthy"])
            is_star_any = bool(row["is_star_any_image"]) if has_star_cols else True

            # Global exclusion: never trustworthy OR non-star in every image.
            # n_trust == 0 means the star was an outlier in every image it appeared
            # in — a strong sign of a bad cross-match or chance alignment.
            # n_trust == 1 is allowed: the star was consistent in at least one
            # image, which is the best we can do for sparsely-observed or
            # single-image fields.  Per-image exclusion below handles any
            # image-specific quality issues for such stars.
            globally_excluded = (n_trust == 0) or (not is_star_any)

            for img in img_list:
                if globally_excluded:
                    trusted = False
                elif any_trust:
                    trusted = img not in excluded_set
                else:
                    trusted = False
                trust_lookup[(sid, img)] = trusted

        # Apply to each image's use_for_alignment / use_for_fit
        n_flagged = 0
        for img_name, df in stars_per_image.items():
            for gid in df["Gaia_id"].astype(str):
                trust_lookup.setdefault((gid, img_name), True)

            trusted_mask = np.array([
                trust_lookup[(gid, img_name)]
                for gid in df["Gaia_id"].astype(str)
            ])
            before = int(df["use_for_alignment"].sum())
            df["use_for_alignment"] = df["use_for_alignment"] & trusted_mask
            df["use_for_fit"]       = df["use_for_fit"]       & trusted_mask
            n_flagged += before - int(df["use_for_alignment"].sum())

        print(f"  Cross-match catalog applied: {n_flagged} star-image pairs "
              f"flagged not-trustworthy (use_for_alignment→False)"
              + (f"  [is_star columns present]" if has_star_cols else ""))
    else:
        print(f"  No cross_match_catalog.csv found at {xm_path}; "
              f"skipping trustworthiness filter")

    # ── Trim Gaia catalog to observed stars ───────────────────────────────────
    gaia_catalog = (
        gaia_catalog[keep_gaia_mask]
        .sort_values("Gaia_id")
        .reset_index(drop=True)
    )

    # DELVE-only sources have synthetic negative Gaia IDs.  Exclude them when
    # DELVE is not requested so the Gaia-only analysis is unaffected.
    if not use_delve and 'Gaia_id' in gaia_catalog.columns:
        n_before = len(gaia_catalog)
        gaia_catalog = gaia_catalog[gaia_catalog['Gaia_id'] >= 0].reset_index(drop=True)
        n_removed = n_before - len(gaia_catalog)
        if n_removed > 0:
            print(f"  DELVE-only sources excluded (no --delve_dir): {n_removed} rows removed")

    # ── DELVE prior enrichment ─────────────────────────────────────────────────
    # Merge per-source DELVE PM covariance columns onto gaia_catalog so the
    # solver can tighten the PM prior for Gaia+DELVE matched stars.
    _DELVE_PRIOR_COLS = [
        'delve_pmra', 'delve_pmdec',
        'delve_pmra_error', 'delve_pmdec_error',
        'delve_parallax', 'delve_parallax_error',
        'delve_ra_error', 'delve_dec_error',
        'delve_ra_cat', 'delve_dec_cat',
        'delve_corr_ra_dec', 'delve_corr_ra_plx', 'delve_corr_ra_pmra',
        'delve_corr_ra_pmdec', 'delve_corr_dec_plx', 'delve_corr_dec_pmra',
        'delve_corr_dec_pmdec', 'delve_corr_plx_pmra', 'delve_corr_plx_pmdec',
        'delve_corr_pmra_pmdec',
    ]
    if xm is not None and use_delve:
        xm_gaia = xm[xm['gaia_source_id'].notna()].copy()
        avail = [c for c in _DELVE_PRIOR_COLS if c in xm_gaia.columns]
        if avail:
            xm_gaia['_gid'] = pd.to_numeric(
                xm_gaia['gaia_source_id'], errors='coerce')
            xm_prior = (xm_gaia.dropna(subset=['_gid'])
                        .groupby('_gid')[avail].first().reset_index()
                        .rename(columns={'_gid': 'Gaia_id'}))
            xm_prior['Gaia_id'] = xm_prior['Gaia_id'].astype(np.int64)
            gaia_catalog = gaia_catalog.merge(xm_prior, on='Gaia_id', how='left')
            # Nullify astrometric columns for entries with incomplete 5×5 covariance
            # (handles cached matched_delve.csv that predates the cross-match filter).
            _sig_cols_gc = ['delve_ra_error', 'delve_dec_error', 'delve_pmra_error',
                            'delve_pmdec_error', 'delve_parallax_error']
            _present = [c for c in _sig_cols_gc if c in gaia_catalog.columns]
            if _present:
                _valid = np.ones(len(gaia_catalog), dtype=bool)
                for _c in _present:
                    _s = pd.to_numeric(gaia_catalog[_c], errors='coerce')
                    _valid &= _s.notna() & (_s > 0)
                _astrom_null = [c for c in _DELVE_PRIOR_COLS if c in gaia_catalog.columns]
                gaia_catalog.loc[~_valid, _astrom_null] = np.nan
            n_delve_pm = int(gaia_catalog.get('delve_pmra_error', pd.Series(dtype=float)).notna().sum())
            print(f"  DELVE priors merged: {n_delve_pm} Gaia sources with DELVE PM covariance")

    # ── DELVE-only star loading ────────────────────────────────────────────────
    # Collect DELVE sources (across all images) whose HST detections were never
    # matched to Gaia.  Add them as synthetic catalog rows so the solver can
    # constrain their PMs using the DELVE prior.
    delve_only_rows: list[dict] = []    # one entry per unique delve_source_id
    delve_id_to_gaia_id: dict[int, int] = {}

    for img_dir in (img_dirs if use_delve else []):
        img_name = img_dir.name
        if restrict_images is not None and img_name not in restrict_images:
            continue
        if img_name not in images:
            continue

        delve_path = img_dir / "matched_delve.csv"
        gaia_path  = img_dir / "matched_gaia.csv"
        if not delve_path.exists():
            continue

        delve = pd.read_csv(delve_path)
        if 'hst_index' not in delve.columns or 'delve_source_id' not in delve.columns:
            continue

        gaia_hst_idx: set = set()
        if gaia_path.exists():
            gaia_match = pd.read_csv(gaia_path, usecols=['hst_index'])
            gaia_hst_idx = set(gaia_match['hst_index'].tolist())

        delve_only = delve[~delve['hst_index'].isin(gaia_hst_idx)].copy()
        for _, row in delve_only.iterrows():
            did = int(row['delve_source_id'])
            if did not in delve_id_to_gaia_id:
                # synthetic negative Gaia_id — unique, negative, fits int64
                delve_id_to_gaia_id[did] = -did
                # Collect astrometry for the synthetic catalog entry
                delve_only_rows.append({
                    'delve_source_id': did,
                    'ra':    row.get('delve_ra_prop', np.nan),
                    'dec':   row.get('delve_dec_prop', np.nan),
                    'pmra':  row.get('delve_pmra', np.nan),
                    'pmdec': row.get('delve_pmdec', np.nan),
                    'delve_pmra':              row.get('delve_pmra', np.nan),
                    'delve_pmdec':             row.get('delve_pmdec', np.nan),
                    'delve_pmra_error':        row.get('delve_pmra_error', np.nan),
                    'delve_pmdec_error':       row.get('delve_pmdec_error', np.nan),
                    'delve_parallax':          row.get('delve_parallax', np.nan),
                    'delve_parallax_error':    row.get('delve_parallax_error', np.nan),
                    'delve_ra_error':          row.get('delve_ra_error', np.nan),
                    'delve_dec_error':         row.get('delve_dec_error', np.nan),
                    'delve_ra_cat':            row.get('delve_ra_cat', np.nan),
                    'delve_dec_cat':           row.get('delve_dec_cat', np.nan),
                    'delve_corr_ra_dec':       row.get('delve_corr_ra_dec', 0.0),
                    'delve_corr_ra_plx':       row.get('delve_corr_ra_plx', 0.0),
                    'delve_corr_ra_pmra':      row.get('delve_corr_ra_pmra', 0.0),
                    'delve_corr_ra_pmdec':     row.get('delve_corr_ra_pmdec', 0.0),
                    'delve_corr_dec_plx':      row.get('delve_corr_dec_plx', 0.0),
                    'delve_corr_dec_pmra':     row.get('delve_corr_dec_pmra', 0.0),
                    'delve_corr_dec_pmdec':    row.get('delve_corr_dec_pmdec', 0.0),
                    'delve_corr_plx_pmra':     row.get('delve_corr_plx_pmra', 0.0),
                    'delve_corr_plx_pmdec':    row.get('delve_corr_plx_pmdec', 0.0),
                    'delve_corr_pmra_pmdec':   row.get('delve_corr_pmra_pmdec', 0.0),
                    'gmag':  row.get('delve_rmag', np.nan),  # r ≈ G for red stars
                })

    if delve_only_rows:
        delve_only_df = pd.DataFrame(delve_only_rows)
        delve_only_df['Gaia_id']   = delve_only_df['delve_source_id'].apply(
            lambda did: delve_id_to_gaia_id[int(did)])
        delve_only_df['Gaia_time'] = 2016.0

        # Ensure all Gaia columns exist (NaN for missing ones)
        for col in gaia_catalog.columns:
            if col not in delve_only_df.columns:
                delve_only_df[col] = np.nan

        gaia_catalog = pd.concat(
            [gaia_catalog, delve_only_df[gaia_catalog.columns]],
            ignore_index=True)

        # Add DELVE-only detections to stars_per_image
        n_delve_only_added = 0
        for img_dir in img_dirs:
            img_name = img_dir.name
            if restrict_images is not None and img_name not in restrict_images:
                continue
            if img_name not in images:
                continue

            gaia_path = img_dir / "matched_gaia.csv"
            gaia_hst_idx = set()
            if gaia_path.exists():
                gaia_match = pd.read_csv(gaia_path, usecols=['hst_index'])
                gaia_hst_idx = set(gaia_match['hst_index'].tolist())

            delve_only_stars = _build_delve_only_stars_df(
                img_dir, img_name, gaia_hst_idx, delve_id_to_gaia_id, pos_err_floor,
                delve_use_for_align=delve_use_for_align)
            if delve_only_stars is not None and len(delve_only_stars):
                # Keep only sources that made it into gaia_catalog
                valid_ids = set(gaia_catalog['Gaia_id'].values)
                delve_only_stars = delve_only_stars[
                    delve_only_stars['Gaia_id'].isin(valid_ids)].reset_index(drop=True)
                if len(delve_only_stars):
                    existing = stars_per_image[img_name]
                    stars_per_image[img_name] = pd.concat(
                        [existing, delve_only_stars], ignore_index=True)
                    n_delve_only_added += len(delve_only_stars)

        print(f"  DELVE-only: {len(delve_id_to_gaia_id)} unique sources, "
              f"{n_delve_only_added} per-image detections added")

    # ── Write image summary CSV ───────────────────────────────────────────────
    if summary_rows:
        summary_df  = pd.DataFrame(summary_rows)
        summary_path = hst_root.parent.parent / "image_transformation_summaries.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Image summary written to '{summary_path}'")

    print(f"  {len(images)} images loaded, "
          f"{len(gaia_catalog)} Gaia sources observed")
    for img_name, df in sorted(stars_per_image.items()):
        meta = images[img_name]
        print(f"    {img_name}: {len(df)} matched stars  "
              f"[{meta['instrument']}/{meta['detector']} {meta['filter']}]")

    return images, stars_per_image, gaia_catalog
