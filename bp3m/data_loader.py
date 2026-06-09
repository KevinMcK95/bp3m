"""
Load and organize BP3M input data from GaiaHub source summary CSVs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import glob
import os

def load_image_data(data_root, field_name="Fornax_dSph"):
    """
    Load all per-image source summary CSVs and the image transformation
    summary for a given field.

    Returns
    -------
    images : dict[image_name -> dict]
        Per-image metadata (pixel_scale, rotation, ra0, dec0, hst_time_mjd, ...)
    stars_per_image : dict[image_name -> pd.DataFrame]
        Per-image source table with HST positions and Gaia astrometry.
    gaia_catalog : pd.DataFrame
        One row per unique Gaia source, with ra/dec/pm/parallax and covariance.
        Stars without Gaia astrometry (missing pmra etc.) are included but
        flagged – they contribute only via their position.
    """
    bayesian_pm_dir = Path(data_root) / field_name / "Bayesian_PMs"
    gaia_cat_dir = Path(data_root) / field_name / "Gaia"
    img_summary_path = bayesian_pm_dir / "gaiahub_image_transformation_summaries.csv"
    img_df = pd.read_csv(img_summary_path)
    img_df = img_df.set_index("image_name")

    # Discover image subfolders
    image_dirs = sorted(p for p in bayesian_pm_dir.iterdir() if p.is_dir())

    images = {}
    stars_per_image = {}

    hst_cols = [
        "Gaia_id", "X", "Y", "X_orig", "Y_orig",
        "x_hst_err", "y_hst_err", "xy_hst_corr",
        "q_hst", "mag", 
        "use_for_alignment", "use_for_fit",
    ]

    gaia_files = glob.glob(os.path.join(gaia_cat_dir, "*_gaia.csv"))
    if not gaia_files:
        print(f"Error: No Gaia files found in {gaia_cat_dir}")
        return None
    
    print(f"Concatenating {len(gaia_files)} Gaia file(s)...")
    all_gaia_list = [pd.read_csv(f, dtype={'source_id': np.int64, 'SOURCE_ID': np.int64}).rename(columns={'SOURCE_ID': 'source_id'}, inplace=False) for f in gaia_files]
    gaia_catalog = pd.concat(all_gaia_list, ignore_index=True).drop_duplicates(subset=['source_id'])
    gaia_cols = [
        "source_id", "ra", "dec", "ra_error", "dec_error", "ra_dec_corr",
        "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
        "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
        "parallax", "parallax_error", "parallax_pmra_corr", "parallax_pmdec_corr",
        "pmra", "pmra_error", "pmra_pmdec_corr", "pmdec", "pmdec_error",
        "ref_epoch", "ruwe", "pseudocolour", "gmag", "gmag_error", 
        "bpmag", "bpmag_error", "rpmag", "rpmag_error", "bp_rp", "bp_rp_error",
    ]
    gaia_catalog = (gaia_catalog[gaia_cols].dropna(subset=["ra", "dec"])
                     .sort_values("source_id")
                     .drop_duplicates("source_id", keep="first")
                     .reset_index(drop=True))
    gaia_catalog.rename(columns={'source_id': 'Gaia_id', 'ref_epoch': 'Gaia_time'}, inplace=True)

    keep_gaia_ids = np.zeros(len(gaia_catalog)).astype(bool)

    for img_dir in image_dirs:
        img_name = img_dir.name
        # if img_name not in ["j8fnezfmq"]:
        #     continue
        csv_files = list(img_dir.glob("*_gaiahub_source_summaries.csv"))
        if not csv_files:
            continue

        src_df = pd.read_csv(csv_files[0])[hst_cols]

        # Only use stars flagged for alignment
        # src_df = src_df[src_df["use_for_alignment"].astype(bool)].copy()
        src_df = src_df.reset_index(drop=True)

        if len(src_df) == 0:
            continue

        # Image metadata
        if img_name not in img_df.index:
            print(f"Warning: {img_name} not found in image transformation summary, skipping.")
            continue

        row = img_df.loc[img_name]
        images[img_name] = {
            "ra0": float(row["ra"]),          # degrees, image pointing direction
            "dec0": float(row["dec"]),
            "pixel_scale": float(row["real_img_pix_scale"]),  # mas/pixel
            "rotation_deg": float(row["rot"]),
            "orig_pixel_scale": float(row["orig_pixel_scale"]), # mas/pixel
            "orig_rot_deg": float(row["orig_rot"]),
            "pixel_scale_ratio": float(row["pix_scale_ratio"]),
            "on_skew": float(row.get("on_axis_skew", 0.0)),
            "off_skew": float(row.get("off_axis_skew", 0.0)),
            "hst_time_mjd": float(row["HST_time"]),
            "instrument": str(row["instrument"]),
            "detector": str(row["detector"]),
            "filter": str(row["filter"]),
            # GaiaHub transformation pivot points
            # Xo, Yo = center of rotation in HST raw pixel coordinates
            # Wo, Zo = corresponding center in Gaia pseudo-image coordinates
            "Xo": float(row["Xo"]), "Yo": float(row["Yo"]),
            "Wo": float(row["Wo"]), "Zo": float(row["Zo"]),
            # Initial GaiaHub transformation for reference
            "AG": float(row["AG"]), "BG": float(row["BG"]),
            "CG": float(row["CG"]), "DG": float(row["DG"]),
        }
        stars_per_image[img_name] = src_df
        keep_gaia_ids |= gaia_catalog['Gaia_id'].isin(src_df['Gaia_id'])

    keep_gaia_id_inds = np.where(keep_gaia_ids)[0]
    # Build unified Gaia catalog: one row per unique Gaia_id, keeping first
    # occurrence of Gaia astrometry (it should be the same across images).
    gaia_catalog = (gaia_catalog.iloc[keep_gaia_id_inds]
                    .dropna(subset=["ra", "dec"])
                    .sort_values("Gaia_id")
                    .drop_duplicates("Gaia_id", keep="first")
                    .reset_index(drop=True))

    return images, stars_per_image, gaia_catalog


# ── Amplifier / CCD boundary lookup ──────────────────────────────────────────
# Pixel coordinates of the chip (Y) and amplifier (X) boundaries in raw
# (pre-GDC) detector coordinates, keyed by instrument+detector (upper-case).
#
# ACS/WFC  : two 4096×2048 chips → combined height 4096; amp X boundary at 2048
# WFC3/UVIS: two 4096×2051 chips → combined height 4102; amp X boundary at 2048
#
# The Y value is the chip boundary (used by --split-ccd and --split-amp).
# The X value is the amplifier boundary within each chip (used by --split-amp).

_AMP_SPLITS = {
    "ACSWFC":   {"x_split": 2048.0, "y_split": 2048.0},
    "WFC3UVIS": {"x_split": 2048.0, "y_split": 2051.0},
}
_AMP_SPLITS_DEFAULT = {"x_split": 2048.0, "y_split": 2048.0}


def _get_amp_splits(meta: dict) -> dict:
    """Return the x_split / y_split dict for the instrument+detector in *meta*."""
    key = (meta.get("instrument", "") + meta.get("detector", "")).upper()
    return _AMP_SPLITS.get(key, _AMP_SPLITS_DEFAULT)


def split_images_by_ccd(images, stars_per_image, min_stars_per_ccd: int = 20):
    """
    Split each image into two independent CCD halves along the Y boundary.

    For ACS/WFC the chips meet at Y_orig = 2048; for WFC3/UVIS at Y_orig = 2051.
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
        Image metadata (as returned by load_image_data or load_inputs).
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
            # Too few stars on one side — keep the image whole
            print(f"    {img}: lo={n_lo}, hi={n_hi} stars — below "
                  f"min_stars_per_ccd={min_stars_per_ccd}, keeping unsplit")
            new_images[img] = dict(meta)
            new_spi[img]    = df
            continue

        for suffix, mask in [('_lo', lo_mask), ('_hi', hi_mask)]:
            sub_df = df[mask].reset_index(drop=True)
            if len(sub_df) == 0:
                continue
            new_images[img + suffix] = dict(meta)
            new_spi[img + suffix]    = sub_df

    return new_images, new_spi


def split_images_by_amp(images, stars_per_image):
    """
    Split each image into four independent amplifier quadrants.

    Each physical CCD chip has two amplifiers separated by the X boundary
    (column 2048 for both ACS/WFC and WFC3/UVIS).  Combined with the chip
    (Y) boundary this produces four quadrants, each fitted with an independent
    r_j transformation vector.

    Boundaries (from ``_AMP_SPLITS``, looked up per image):

    =========  ============================  ============================
    Suffix     X condition                   Y condition
    =========  ============================  ============================
    ``_llo``   X_orig ≤ x_split  (left)      Y_orig ≤ y_split  (lower chip)
    ``_rlo``   X_orig >  x_split (right)     Y_orig ≤ y_split  (lower chip)
    ``_lhi``   X_orig ≤ x_split  (left)      Y_orig >  y_split (upper chip)
    ``_rhi``   X_orig >  x_split (right)     Y_orig >  y_split (upper chip)
    =========  ============================  ============================

    If a quadrant contains no stars it is omitted.

    Parameters
    ----------
    images : dict[str -> dict]
        Image metadata.  Must contain ``instrument``, ``detector``,
        ``X_orig`` and ``Y_orig`` columns in the corresponding DataFrame.
    stars_per_image : dict[str -> pd.DataFrame]
        Per-image source tables.  Must contain ``X_orig`` and ``Y_orig``
        (falls back to ``X`` / ``Y`` respectively).

    Returns
    -------
    new_images, new_stars_per_image : dicts with ``_llo``/``_rlo``/``_lhi``/``_rhi``
        suffixes replacing the original keys.
    """
    new_images = {}
    new_spi = {}

    for img in sorted(images.keys()):
        meta    = images[img]
        df      = stars_per_image[img]
        splits  = _get_amp_splits(meta)
        x_split = splits["x_split"]
        y_split = splits["y_split"]

        x_col = 'X_orig' if 'X_orig' in df.columns else 'X'
        y_col = 'Y_orig' if 'Y_orig' in df.columns else 'Y'
        x_vals = df[x_col].to_numpy(float)
        y_vals = df[y_col].to_numpy(float)

        left  = x_vals <= x_split
        right = x_vals >  x_split
        lo    = y_vals <= y_split
        hi    = y_vals >  y_split

        for suffix, mask in [('_llo', left  & lo),
                              ('_rlo', right & lo),
                              ('_lhi', left  & hi),
                              ('_rhi', right & hi)]:
            sub_df = df[mask].reset_index(drop=True)
            if len(sub_df) == 0:
                continue
            new_images[img + suffix] = dict(meta)
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
        # Filter df to only those with Gaia matches
        mask = df["Gaia_id"].isin(star_id_to_idx)
        star_in_image[img] = idxs

    return star_id_to_idx, image_names, star_in_image
