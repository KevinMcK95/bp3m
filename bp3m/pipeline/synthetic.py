"""
Step 6 (optional): Generate synthetic HST observations for BP3M end-to-end testing.

Requires a completed cross-match (Step 4).  No new downloads or PSF fitting.

Workflow
--------
1. generate_synthetic_data()
     Reads real cross-match outputs, forward-models pixel positions from the
     Gaia MAP values (or optionally a draw from the Gaia prior), adds realistic
     measurement noise drawn from the actual PSF-fit positional covariances, and
     writes a parallel directory tree that data_loader_flc can consume unchanged.

2. Run BP3M (via run_alignment) with output_dir=field_path, field_name='synthetic'

3. compare_synthetic_results()
     Loads the truth table and BP3M posteriors, computes residuals and pulls for
     all 5 astrometric parameters plus image transformation parameters, writes
     diagnostic plots and a synthetic_comparison.csv.

Directory layout produced
-------------------------
{output_dir}/{field}/synthetic/
    Gaia/
        {field}_synthetic_gaia.csv
    HST/mastDownload/HST/
        {img}/
            {img}_flc.fits            → symlink to real image
            {img}_flc_catalog.fits    — copy with synthetic x_gdc/y_gdc
            transformation.csv        → symlink to real
            matched_gaia.csv          — same hst_index, int64 gaia_source_id
    truth/
        stellar_truth.csv
        image_truth.csv
    BP3M_results/                     — written by run_alignment after this step
"""

from __future__ import annotations

import os
import sys
import glob as _glob
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits as afits


def _ensure_fcm():
    pass  # gaia_cross_match is installed as a package; no sys.path manipulation needed


def _ensure_bp3m():
    pass  # bp3m is installed as a package; no sys.path manipulation needed


# ── coordinate helpers ────────────────────────────────────────────────────────

def _read_transform(img_dir: Path) -> dict | None:
    """Read transformation.csv into a flat dict. Returns None if file missing."""
    p = img_dir / "transformation.csv"
    if not p.exists():
        return None
    tdf = pd.read_csv(p).set_index("parameter")["value"]
    keys_req = ["A", "B", "C", "D", "xs_o", "ys_o", "xt_o", "yt_o",
                "ra_cen", "dec_cen", "x_cen", "y_cen", "pixel_scale", "orientat"]
    keys_opt = {"ratio": 1.0, "on_skew": 0.0, "off_skew": 0.0}
    try:
        d = {k: float(tdf[k]) for k in keys_req}
    except KeyError as e:
        print(f"    transformation.csv missing key {e}")
        return None
    for k, default in keys_opt.items():
        d[k] = float(tdf[k]) if k in tdf.index else default
    return d



def _mjd_from_flc(flc_path: Path) -> float:
    """Mid-exposure MJD from FLC FITS primary header."""
    with afits.open(flc_path, memmap=False) as hdu:
        h = hdu[0].header
        return 0.5 * (float(h["EXPSTART"]) + float(h["EXPEND"]))


def _abcd_from_physical(rot_deg: float, ratio: float,
                        on_skew: float, off_skew: float) -> dict:
    """
    Construct BP3M (a,b,c,d) analytically from physical alignment parameters.

    The model (from solver.py _make_image_prior Jacobian):
        a =  ratio * cos(θ) + on_skew
        b =  ratio * sin(θ) + off_skew
        c = -ratio * sin(θ) + off_skew
        d =  ratio * cos(θ) - on_skew

    where θ = orig_rot_deg (negative of HST ORIENTAT), ratio is the
    fractional pixel-scale factor (≈ 1.0), and on_skew / off_skew are
    small distortion terms (≈ 0).
    """
    theta = np.deg2rad(rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return dict(
        true_a=float(ratio * cos_t + on_skew),
        true_b=float(ratio * sin_t + off_skew),
        true_c=float(-ratio * sin_t + off_skew),
        true_d=float(ratio * cos_t - on_skew),
    )


def _read_chip_info(flc_path: Path, cat_path: Path) -> dict:
    """
    Read per-chip geometry for the split-CCD forward model.

    Returns a dict with:
        y_split          : Y_orig threshold separating lo/hi chips
        chip_pointing_lo : {Xo, Yo, ra0, dec0} for the lower chip (or None)
        chip_pointing_hi : {Xo, Yo, ra0, dec0} for the upper chip (or None)
    """
    _AMP_SPLITS = {
        ("ACS",  "WFC"):  2048.0,
        ("WFC3", "UVIS"): 2051.0,
    }
    with afits.open(flc_path, memmap=False) as hdu:
        h0 = hdu[0].header
        instrument = str(h0.get("INSTRUME", "")).strip().upper()
        detector   = str(h0.get("DETECTOR", "")).strip().upper()
    y_split = _AMP_SPLITS.get((instrument, detector), 2048.0)

    chip_pointing_lo = None
    chip_pointing_hi = None
    try:
        with afits.open(cat_path, memmap=False) as hdu:
            cat_hdr = hdu[1].header
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
                chip_pts.append({"Xo": float(xo), "Yo": float(yo),
                                  "ra0": float(ra0), "dec0": float(dc0)})
        if len(chip_pts) == 2:
            chip_pts.sort(key=lambda p: p["Yo"])
            chip_pointing_lo = chip_pts[0]
            chip_pointing_hi = chip_pts[1]
    except Exception:
        pass

    return {"y_split": y_split,
            "chip_pointing_lo": chip_pointing_lo,
            "chip_pointing_hi": chip_pointing_hi}


def _propagate(df: pd.DataFrame, mjd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate Gaia catalog positions to target MJD using BP3M's parallax convention.

    BP3M's get_parallax_factors returns NEGATED parallax factors:
        plx_ra_star = -(X*sin(ra) - Y*cos(ra))
        plx_dec     = -(X*cos(ra)*sin(dec) + Y*sin(ra)*sin(dec) - Z*cos(dec))

    These factors enter the U matrix as the parallax column, and the sky offset
    that BP3M models for a star with parallax ϖ at time t is:
        Δ(α*)_mas = pmra*dt + ϖ * plx_ra_star   (where plx_ra_star is NEGATIVE of standard)

    We must use the same convention here so that the synthetic positions fed
    to BP3M are internally consistent with what BP3M's U matrix expects.

    The standard astrometric convention (cross_match, astropy) is:
        parallax_shift_ra  = +X*sin(ra) - Y*cos(ra)  (positive)

    BP3M's convention is the negative of this.  Using cross_match's convention
    here would create a sign-flipped parallax contribution in the synthetic data,
    causing pull widths > 1 and systematic image-parameter bias.
    """
    _ensure_bp3m()
    from bp3m.astro_utils import get_tele_position, get_parallax_factors
    from astropy.time import Time

    t_hst = Time(mjd, format='mjd')
    ref_epoch = (df['ref_epoch'].iloc[0]
                 if 'ref_epoch' in df.columns else 2016.0)
    dt_yr = t_hst.jyear - ref_epoch

    ra  = df['ra'].values
    dec = df['dec'].values
    pmra  = df['pmra'].fillna(0.0).values
    pmdec = df['pmdec'].fillna(0.0).values
    plx   = df['parallax'].fillna(0.0).values

    tele_xyz = get_tele_position(t_hst, curr_id='earth')
    plx_ra, plx_dec = get_parallax_factors(ra, dec, tele_xyz)
    # plx_ra = -(X*sin(ra) - Y*cos(ra))  [BP3M convention, NEGATED vs standard]

    cos_dec = np.cos(np.radians(dec))
    # μα* is already α*=α·cos(δ), so divide by cos(δ) to get Δα in degrees
    ra_off_mas  = pmra  * dt_yr + plx * plx_ra   # mas in α* direction
    dec_off_mas = pmdec * dt_yr + plx * plx_dec  # mas in δ direction

    ra_prop  = ra  + (ra_off_mas  / cos_dec) / 3.6e6  # degrees
    dec_prop = dec + dec_off_mas / 3.6e6               # degrees
    return ra_prop, dec_prop


def _write_synthetic_catalog(src_catalog: Path, dst_catalog: Path,
                              hst_idx: np.ndarray,
                              x_syn: np.ndarray, y_syn: np.ndarray) -> None:
    """
    Write dst_catalog containing ONLY the hst_idx rows from src_catalog, with
    x_gdc/y_gdc replaced by the synthetic pixel coordinates and n_sat cleared.

    Trimming to matched rows only keeps catalog size proportional to detections
    (~500 rows) instead of the full detector catalog (~40k rows).  The row
    indices in the output are 0..N-1, and the hst_index column in matched_gaia.csv
    is rewritten (by the caller) to use these compact indices.
    """
    with afits.open(src_catalog, memmap=False) as src:
        bintable = src[1]
        new_cols = []
        for col in bintable.columns:
            arr = bintable.data[col.name][hst_idx].copy()
            if col.name == "x_gdc":
                arr[:] = x_syn
            elif col.name == "y_gdc":
                arr[:] = y_syn
            elif col.name == "n_sat":
                arr[:] = 0
            new_cols.append(afits.Column(name=col.name, format=col.format,
                                          array=arr, unit=col.unit))
        new_bin  = afits.BinTableHDU.from_columns(new_cols)
        # Copy the catalog header (chip pointing keys etc.) to new extension
        for key, val in src[1].header.items():
            if key not in ("XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                           "PCOUNT", "GCOUNT", "TFIELDS", "END") and not key.startswith("TTYPE") \
                    and not key.startswith("TFORM") and not key.startswith("TUNIT") \
                    and not key.startswith("TDISP"):
                try:
                    new_bin.header[key] = val
                except Exception:
                    pass
        new_hdul = afits.HDUList([src[0].copy(), new_bin]
                                 + [ext.copy() for ext in src[2:]])
        new_hdul.writeto(str(dst_catalog), overwrite=True)


# ── public API ────────────────────────────────────────────────────────────────

def generate_synthetic_data(
    output_dir: Path,
    field_name: str,
    telescope: str = "HST",
    im_type: str = "_flc",
    seed: int = 42,
    draw_from_prior: bool = False,
    zero_parallax: bool = False,
    true_gaia: bool = False,
    jitter_sigma: float = 0.0,
    images: list[str] | None = None,
    force_regenerate: bool = False,
    only_5p: bool = False,
    all_5p_gaia: bool = False,
    rot_sigma: float = 0.01,
    ratio_sigma: float = 1e-4,
    skew_sigma: float = 1e-5,
    pointing_sigma: float = 500.0,
    true_pm_center: tuple[float, float] | None = None,
    true_pm_width: float = 0.1,
    true_parallax_center: float | None = None,
    true_parallax_width: float = 0.1,
    syn_name: str = "synthetic",
    split_ccd: bool = True,
) -> Path:
    """
    Generate synthetic HST observations and write a synthetic data tree.

    Parameters
    ----------
    output_dir       : pipeline root (parent of field_name/)
    field_name       : field subdirectory
    telescope        : 'HST'
    im_type          : '_flc' or '_flt'
    seed             : RNG seed for reproducibility
    draw_from_prior  : if True, draw true stellar params from Gaia prior N(v,C)
                       instead of using the Gaia MAP values directly
    zero_parallax    : if True, set true parallax = 0 for all stars
    true_gaia        : if True, use true values as the Gaia prior (no Gaia noise)
    jitter_sigma     : std dev (in pixel units) of Gaussian perturbation added to
                       the true transformation parameters (A,B,C,D,xs_o,ys_o,xt_o,yt_o).
                       Default 0 = use transformation.csv values unchanged.
    images           : optional explicit list of image names to include
    force_regenerate : regenerate even if synthetic directory already exists
    rot_sigma        : 1σ uncertainty for true rotation draw (degrees, default 0.01)
    ratio_sigma      : 1σ uncertainty for true pixel-scale ratio draw (default 1e-4)
    skew_sigma       : 1σ for true on/off-axis skew draws (default 1e-5)
    pointing_sigma   : 1σ for true pointing offset draw in RA*cos(dec) and Dec (mas, default 500)
    only_5p          : if True, exclude 2-param Gaia stars (no measured PM/parallax)
                       from the synthetic test — useful for isolating whether 2-param
                       stars affect image parameter estimation
    all_5p_gaia      : if True, give 2-param stars synthetic 5-param Gaia measurements
                       (PM + parallax drawn with median uncertainties from real 5-param
                       stars) instead of leaving their catalog entries as NaN. The true
                       PM is still drawn from N(0,10²); the synthetic catalog value is
                       truth+noise and v_true is updated to record the correction from
                       that catalog value (so the comparison remains self-consistent).
    true_pm_center   : if set, override ALL stars' true proper motion by drawing from
                       N(true_pm_center, true_pm_width²) for both pmra and pmdec.
                       Synthetic Gaia catalog is set to truth + per-star noise using
                       real Gaia errors (or median for 2-param stars).  v_true is
                       updated to record truth − catalog = −noise.  Useful for testing
                       BP3M's recovery of bulk population motions.
    true_pm_width    : 1σ width of the PM draw around true_pm_center (mas/yr, default 0.1).
    true_parallax_center : if set, override ALL stars' true parallax by drawing from
                       N(true_parallax_center, true_parallax_width²).  Same
                       catalog-noise treatment as true_pm_center.  Set a positive
                       center to test stars with physically meaningful parallaxes.
    true_parallax_width  : 1σ width of the parallax draw (mas, default 0.1).
    syn_name         : subdirectory name for the synthetic output tree (default 'synthetic').
                       Use a distinct name (e.g. 'synthetic_only5p') to avoid overwriting
                       results from a different configuration.
    split_ccd        : if True (default), generate positions using per-chip tangent points
                       from the catalog header (CHIP{n}_CRVAL/CRPIX), matching BP3M's
                       split-CCD mode.  If False, use the full-frame tangent point
                       (ra_cen/dec_cen from transformation.csv) for ALL stars, matching
                       BP3M's no-split mode.  Must match the split_ccd flag passed to
                       run_alignment / bp3m-synthetic --no_split_ccd.

    Returns
    -------
    Path to {output_dir}/{field_name}/{syn_name}/
    Call run_alignment(output_dir=output_dir/field_name, field_name=syn_name)
    to run BP3M on the synthetic data.
    """
    rng = np.random.default_rng(seed)

    output_dir = Path(output_dir)
    field_path = output_dir / field_name
    tel_upper  = telescope.upper()
    hst_root   = field_path / tel_upper / "mastDownload" / tel_upper
    gaia_dir   = field_path / "Gaia"
    syn_dir    = field_path / syn_name

    # ── Early exit if synthetic data already exists ───────────────────────────
    if syn_dir.exists() and not force_regenerate:
        print(f"\n{'─'*50}")
        print(f"Synthetic data already exists at {syn_dir}")
        print(f"  (pass force_regenerate=True / --force_regenerate to regenerate)")
        print(f"{'─'*50}")
        return syn_dir

    print(f"\n{'─'*50}")
    print("Synthetic: generating synthetic observations")
    print(f"{'─'*50}")

    # ── Load real Gaia catalog ────────────────────────────────────────────────
    gaia_files = sorted(_glob.glob(str(gaia_dir / "*_gaia.csv")))
    if not gaia_files:
        raise FileNotFoundError(f"No Gaia catalog in {gaia_dir}")
    gaia_frames = [pd.read_csv(f, dtype={'source_id': np.int64, 'SOURCE_ID': np.int64})
                   .rename(columns={"SOURCE_ID": "source_id"})
                   for f in gaia_files]
    gaia_raw = (pd.concat(gaia_frames, ignore_index=True)
                  .drop_duplicates("source_id")
                  .dropna(subset=["ra", "dec"])
                  .reset_index(drop=True))
    gaia_float_to_int = {float(sid): np.int64(sid)
                         for sid in gaia_raw["source_id"].values}

    # ── Inventory image directories ───────────────────────────────────────────
    if not hst_root.exists():
        raise FileNotFoundError(f"HST root not found: {hst_root}")
    img_dirs = sorted(p for p in hst_root.iterdir() if p.is_dir())
    if images is not None:
        keep = set(images)
        img_dirs = [d for d in img_dirs if d.name in keep]
    if not img_dirs:
        raise RuntimeError(f"No image directories found under {hst_root}")

    # ── Pass 1: load per-image data, collect observed Gaia IDs ───────────────
    per_image: dict = {}
    all_gaia_ids: set = set()

    for img_dir in img_dirs:
        img_name   = img_dir.name
        flc_path   = img_dir / f"{img_name}{im_type}.fits"
        cat_path   = img_dir / f"{img_name}{im_type}_catalog.fits"
        match_path = img_dir / "matched_gaia.csv"

        missing = [f.name for f in (flc_path, cat_path, match_path)
                   if not f.exists()]
        if missing:
            print(f"  {img_name}: missing {missing} — skipping")
            continue

        t = _read_transform(img_dir)
        if t is None:
            print(f"  {img_name}: bad transformation.csv — skipping")
            continue

        match = pd.read_csv(match_path)
        # Recover exact int64 source IDs if file has float format
        if match["gaia_source_id"].dtype == np.float64:
            match["gaia_source_id"] = (
                match["gaia_source_id"]
                .map(lambda f: gaia_float_to_int.get(f, np.int64(-1)))
                .astype(np.int64))
            match = match[match["gaia_source_id"] != -1].reset_index(drop=True)

        if len(match) == 0:
            print(f"  {img_name}: no matched stars after ID recovery — skipping")
            continue

        with afits.open(cat_path, memmap=False) as hdu:
            tbl = hdu[1].data
            cat_cov_xx = tbl["cov_xx_gdc"].astype(float)
            cat_cov_yy = tbl["cov_yy_gdc"].astype(float)
            cat_cov_xy = tbl["cov_xy_gdc"].astype(float)
            cat_x_gdc  = tbl["x_gdc"].astype(float)
            cat_y_gdc  = tbl["y_gdc"].astype(float)
            cat_y_raw  = tbl["y"].astype(float)    # raw detector Y for chip assignment

        chip_info = _read_chip_info(flc_path, cat_path)

        all_gaia_ids.update(match["gaia_source_id"].values)
        per_image[img_name] = dict(
            img_dir=img_dir, flc_path=flc_path, cat_path=cat_path,
            transform=t, mjd=_mjd_from_flc(flc_path),
            match=match,
            cat_cov_xx=cat_cov_xx, cat_cov_yy=cat_cov_yy, cat_cov_xy=cat_cov_xy,
            cat_x_gdc=cat_x_gdc, cat_y_gdc=cat_y_gdc, cat_y_raw=cat_y_raw,
            **chip_info,  # y_split, chip_pointing_lo, chip_pointing_hi
        )

    if not per_image:
        raise RuntimeError("No usable images for synthetic data generation.")

    # ── Build true stellar parameters ─────────────────────────────────────────
    gaia_obs = (gaia_raw[gaia_raw["source_id"].isin(all_gaia_ids)]
                  .copy().reset_index(drop=True))
    print(f"  {len(gaia_obs)} Gaia stars observed across {len(per_image)} images")

    if only_5p:
        has_pm_all = gaia_obs["pmra"].notna()
        n_drop = (~has_pm_all).sum()
        gaia_obs = gaia_obs[has_pm_all].reset_index(drop=True)
        print(f"  only_5p: dropped {n_drop} 2-param stars → {len(gaia_obs)} remain")

    has_pm  = gaia_obs["pmra"].notna().values   # 5/6-parameter Gaia solutions
    N       = len(gaia_obs)
    cos_dec = np.cos(np.deg2rad(gaia_obs["dec"].values))

    from .catalog_utils import build_gaia_cov
    C_gaia = build_gaia_cov(gaia_obs)           # (N, 5, 5) inflated covariance

    # v_true[i] = (Δα*_i, Δδ_i, μα*_i, μδ_i, ϖ_i)
    # This is the OFFSET of the physical truth from the Gaia catalog values.
    # The Gaia catalog IS the noisy measurement of these true values:
    #   gaia_obs = v_true_physical + gaia_noise
    # → v_true_physical = gaia_obs + v_true_offset
    # → gaia_noise = -v_true_offset
    # BP3M's job: recover v_true_offset from (gaia_obs + HST).
    v_true = np.zeros((N, 5))

    if true_gaia:
        # No draw: truth = exact Gaia catalog values (delta = 0 for all).
        # Useful only to verify BP3M is unbiased when handed perfect data.
        print(f"  true_gaia mode: truth = Gaia catalog values, no prior draw")
    else:
        from scipy.stats import truncnorm as _truncnorm

        # Default: draw the full 5D offset from the Gaia covariance.
        # For 5/6-param stars: full 5D draw from inflated Gaia covariance.
        # Enforce: true parallax (= gaia_plx + v_true[4]) must be positive.
        plx_gaia_5p = gaia_obs["parallax"].values.astype(float)
        n_plx_truncated = 0
        for i in np.where(has_pm)[0]:
            v_true[i] = rng.multivariate_normal(np.zeros(5), C_gaia[i])
            # Truncate parallax component if needed so true_plx > 0
            plx_sigma = float(np.sqrt(max(C_gaia[i, 4, 4], 1e-20)))
            if plx_gaia_5p[i] + v_true[i, 4] <= 0.0:
                a_lo = -plx_gaia_5p[i] / plx_sigma   # lower truncation in σ units
                v_true[i, 4] = float(
                    _truncnorm.rvs(a_lo, np.inf, loc=0.0, scale=plx_sigma,
                                   random_state=rng))
                n_plx_truncated += 1
        if n_plx_truncated:
            print(f"  {n_plx_truncated} 5p star(s) had plx component resampled "
                  f"to enforce true_plx > 0")

        # For 2-param stars: position from 2×2 Gaia covariance; PMs and parallax
        # drawn to match the empirical distribution of the 5p/6p stars' true values.
        no_pm_idx = np.where(~has_pm)[0]
        n_no_pm   = len(no_pm_idx)
        if n_no_pm > 0:
            for i in no_pm_idx:
                v_true[i, :2] = rng.multivariate_normal(np.zeros(2), C_gaia[i, :2, :2])

            # Compute 5p true PM/parallax values (after positivity truncation above)
            pm_idx = np.where(has_pm)[0]
            true_pmra_5p  = gaia_obs["pmra"].values[pm_idx]  + v_true[pm_idx, 2]
            true_pmdec_5p = gaia_obs["pmdec"].values[pm_idx] + v_true[pm_idx, 3]
            true_plx_5p   = plx_gaia_5p[pm_idx]              + v_true[pm_idx, 4]

            # Half 68% width = (84th − 16th pctile) / 2  ≈  σ for a Gaussian
            def _half68(arr):
                return max((np.percentile(arr, 84) - np.percentile(arr, 16)) / 2.0,
                           1e-6)

            pmra_med   = float(np.median(true_pmra_5p))
            pmra_sig   = _half68(true_pmra_5p)
            pmdec_med  = float(np.median(true_pmdec_5p))
            pmdec_sig  = _half68(true_pmdec_5p)
            plx_med    = float(np.median(true_plx_5p))
            plx_sig    = _half68(true_plx_5p)

            # PM draws: unconstrained Gaussian matching 5p distribution
            v_true[no_pm_idx, 2] = rng.normal(pmra_med,  pmra_sig,  n_no_pm)
            v_true[no_pm_idx, 3] = rng.normal(pmdec_med, pmdec_sig, n_no_pm)

            # Parallax: truncated normal with lower bound at 0 (true plx must be > 0)
            # For 2p stars v_true[i, 4] IS the true parallax (no Gaia measurement).
            a_lo_plx = -plx_med / plx_sig
            v_true[no_pm_idx, 4] = _truncnorm.rvs(
                a_lo_plx, np.inf, loc=plx_med, scale=plx_sig,
                size=n_no_pm, random_state=rng)

            print(f"  {n_no_pm} 2-param stars: position from Gaia 2×2 cov; "
                  f"PM from 5p empirical N({pmra_med:.2f},{pmdec_med:.2f}) "
                  f"±({pmra_sig:.2f},{pmdec_sig:.2f}) mas/yr; "
                  f"plx from TN({plx_med:.3f}, {plx_sig:.3f}, >0) mas")

        pos_rms = float(np.sqrt(np.mean(v_true[:, 0]**2 + v_true[:, 1]**2) / 2))
        print(f"  True astrometry offset drawn from Gaia covariance "
              f"(position RMS: {pos_rms:.4f} mas)")
        print(f"  Synthetic Gaia = original catalog (offset from truth by the draw above)")

    # Physical truth: gaia_obs + drawn offset
    gaia_true = gaia_obs.copy()
    gaia_true["ra"]      = gaia_obs["ra"]  + v_true[:, 0] / (cos_dec * 3.6e6)
    gaia_true["dec"]     = gaia_obs["dec"] + v_true[:, 1] / 3.6e6
    # For 5/6-param stars: add offset to existing catalog values
    gaia_true.loc[has_pm, "pmra"]     = gaia_obs["pmra"].values[has_pm]  + v_true[has_pm, 2]
    gaia_true.loc[has_pm, "pmdec"]    = gaia_obs["pmdec"].values[has_pm] + v_true[has_pm, 3]
    gaia_true.loc[has_pm, "parallax"] = (gaia_obs["parallax"].values[has_pm]
                                          + v_true[has_pm, 4])
    # For 2-param stars: set drawn values (no existing Gaia measurement)
    if (~has_pm).any():
        gaia_true.loc[~has_pm, "pmra"]     = v_true[~has_pm, 2]
        gaia_true.loc[~has_pm, "pmdec"]    = v_true[~has_pm, 3]
        gaia_true.loc[~has_pm, "parallax"] = v_true[~has_pm, 4]

    if zero_parallax and true_parallax_center is None:
        gaia_true["parallax"] = 0.0
        v_true[:, 4] = 0.0

    if not true_gaia:
        n_neg_plx = int((gaia_true["parallax"].values < 0).sum())
        if n_neg_plx:
            print(f"  WARNING: {n_neg_plx} star(s) still have negative true parallax "
                  f"after truncation (likely from zero_parallax or override path)")

    # source_id → row index in gaia_obs/gaia_true
    id_to_row = {int(sid): i for i, sid in enumerate(gaia_obs["source_id"].values)}

    # ── Build synthetic Gaia "observed" catalog ───────────────────────────────
    # The Gaia catalog IS the noisy measurement: gaia_obs = gaia_true + noise.
    # No second noise draw is needed — just use the real catalog as-is.
    # 2-param stars keep NaN for pmra/pmdec/parallax (as in real Gaia).
    gaia_syn = gaia_obs.copy()

    # ── Optional: override true PM / parallax with user-specified distributions ──
    # When true_pm_center or true_parallax_center is set:
    #   1. Draw physical truth from N(center, width²) for ALL stars.
    #   2. Draw per-star Gaia catalog noise using real per-star errors.
    #   3. Set gaia_syn (catalog) = truth + noise.
    #   4. Set gaia_true = truth.
    #   5. Set v_true = truth − catalog = −noise  (what BP3M must recover).
    # This gives a fully self-consistent generative model where the catalog IS
    # a draw from N(truth, C_gaia), and all parallaxes are positive when
    # true_parallax_center >> true_parallax_width.
    _pm_parallax_override = (true_pm_center is not None
                             or true_parallax_center is not None)
    if _pm_parallax_override:
        # Per-star PM errors: use real catalog values for 5-param; median for 2-param.
        has_pm_vals = gaia_obs["pmra"].notna()
        pmra_err  = gaia_obs["pmra_error"].values.astype(float).copy()
        pmdec_err = gaia_obs["pmdec_error"].values.astype(float).copy()
        plx_err   = gaia_obs["parallax_error"].values.astype(float).copy()
        med_pmra_err  = float(np.nanmedian(pmra_err[has_pm_vals]))
        med_pmdec_err = float(np.nanmedian(pmdec_err[has_pm_vals]))
        med_plx_err   = float(np.nanmedian(plx_err[has_pm_vals]))
        pmra_err[~has_pm_vals]  = med_pmra_err
        pmdec_err[~has_pm_vals] = med_pmdec_err
        plx_err[~has_pm_vals]   = med_plx_err

        if true_pm_center is not None:
            pm_true_ra  = rng.normal(true_pm_center[0], true_pm_width, N)
            pm_true_dec = rng.normal(true_pm_center[1], true_pm_width, N)
            noise_pmra  = rng.normal(0.0, pmra_err)
            noise_pmdec = rng.normal(0.0, pmdec_err)
            # Physical truth
            gaia_true["pmra"]  = pm_true_ra
            gaia_true["pmdec"] = pm_true_dec
            # Synthetic catalog = truth + noise; v_true = truth − catalog = −noise
            gaia_syn["pmra"]        = pm_true_ra  + noise_pmra
            gaia_syn["pmdec"]       = pm_true_dec + noise_pmdec
            gaia_syn["pmra_error"]  = pmra_err
            gaia_syn["pmdec_error"] = pmdec_err
            v_true[:, 2] = -noise_pmra
            v_true[:, 3] = -noise_pmdec
            # Zero PM cross-correlations for all stars (catalog is fully synthetic)
            for _c in ["ra_pmra_corr", "ra_pmdec_corr",
                       "dec_pmra_corr", "dec_pmdec_corr", "pmra_pmdec_corr"]:
                if _c in gaia_syn.columns:
                    gaia_syn[_c] = 0.0
            print(f"  true_pm_center override: all {N} stars draw PM from "
                  f"N(({true_pm_center[0]:.2f},{true_pm_center[1]:.2f}), "
                  f"{true_pm_width:.3f}²) mas/yr")
            print(f"    true pmra  mean={pm_true_ra.mean():.3f}  std={pm_true_ra.std():.3f}")
            print(f"    true pmdec mean={pm_true_dec.mean():.3f}  std={pm_true_dec.std():.3f}")

        if true_parallax_center is not None:
            plx_true  = rng.normal(true_parallax_center, true_parallax_width, N)
            noise_plx = rng.normal(0.0, plx_err)
            gaia_true["parallax"]       = plx_true
            gaia_syn["parallax"]        = plx_true + noise_plx
            gaia_syn["parallax_error"]  = plx_err
            v_true[:, 4] = -noise_plx
            for _c in ["ra_parallax_corr", "dec_parallax_corr",
                       "parallax_pmra_corr", "parallax_pmdec_corr"]:
                if _c in gaia_syn.columns:
                    gaia_syn[_c] = 0.0
            print(f"  true_parallax_center override: all {N} stars draw parallax from "
                  f"N({true_parallax_center:.2f}, {true_parallax_width:.3f}²) mas")
            print(f"    true parallax mean={plx_true.mean():.3f}  std={plx_true.std():.3f}  "
                  f"min={plx_true.min():.3f}  max={plx_true.max():.3f}")

        # Override zero_parallax if also requested
        if zero_parallax and true_parallax_center is None:
            gaia_true["parallax"] = 0.0
            gaia_syn["parallax"]  = 0.0
            v_true[:, 4] = 0.0

    # Optionally promote 2-param stars to 5-param by assigning synthetic Gaia
    # PM+parallax measurements.  Physical truth is already in gaia_true; we add
    # noise drawn with median errors from real 5-param stars.
    # v_true is updated AFTER the forward model (which uses gaia_true) to record
    # the correction from the NEW catalog value rather than the physical PM.
    # Skip if true_pm_center/true_parallax_center already handled this.
    _all5p_noise = None   # stored for v_true update after forward model
    if all_5p_gaia and (~has_pm).any() and not _pm_parallax_override:
        no_pm_mask = ~has_pm
        has_pm_vals = gaia_obs["pmra"].notna()
        med_pmra_err  = float(gaia_obs.loc[has_pm_vals, "pmra_error"].median())
        med_pmdec_err = float(gaia_obs.loc[has_pm_vals, "pmdec_error"].median())
        med_plx_err   = float(gaia_obs.loc[has_pm_vals, "parallax_error"].median())

        n_no_pm = int(no_pm_mask.sum())
        noise_pmra  = rng.normal(0.0, med_pmra_err,  n_no_pm)
        noise_pmdec = rng.normal(0.0, med_pmdec_err, n_no_pm)
        noise_plx   = rng.normal(0.0, med_plx_err,   n_no_pm)

        # Synthetic catalog = physical truth (in gaia_true) + Gaia measurement noise
        phys_pmra  = gaia_true.loc[no_pm_mask, "pmra"].values
        phys_pmdec = gaia_true.loc[no_pm_mask, "pmdec"].values
        phys_plx   = gaia_true.loc[no_pm_mask, "parallax"].values

        gaia_syn.loc[no_pm_mask, "pmra"]           = phys_pmra  + noise_pmra
        gaia_syn.loc[no_pm_mask, "pmdec"]          = phys_pmdec + noise_pmdec
        gaia_syn.loc[no_pm_mask, "parallax"]       = phys_plx   + noise_plx
        gaia_syn.loc[no_pm_mask, "pmra_error"]     = med_pmra_err
        gaia_syn.loc[no_pm_mask, "pmdec_error"]    = med_pmdec_err
        gaia_syn.loc[no_pm_mask, "parallax_error"] = med_plx_err
        for _col in ["ra_pmra_corr", "ra_pmdec_corr", "ra_parallax_corr",
                     "dec_pmra_corr", "dec_pmdec_corr", "dec_parallax_corr",
                     "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr"]:
            if _col in gaia_syn.columns:
                gaia_syn.loc[no_pm_mask, _col] = 0.0

        # Save noise so we can update v_true after the forward model
        _all5p_noise = (no_pm_mask, noise_pmra, noise_pmdec, noise_plx)
        print(f"  all_5p_gaia: promoted {n_no_pm} 2-param stars to 5-param "
              f"(σ_pmra={med_pmra_err:.3f}, σ_pmdec={med_pmdec_err:.3f}, "
              f"σ_plx={med_plx_err:.3f} mas)")

    # ── Create output directories ─────────────────────────────────────────────
    syn_gaia  = syn_dir / "Gaia"
    syn_hst   = syn_dir / tel_upper / "mastDownload" / tel_upper
    syn_truth = syn_dir / "truth"
    for d in (syn_gaia, syn_hst, syn_truth):
        d.mkdir(parents=True, exist_ok=True)

    # When regenerating with an image filter, remove any stale image dirs from
    # previous runs that are NOT in the current filter set.  Otherwise BP3M
    # would discover them and include them in the fit.
    if images is not None and syn_hst.exists():
        keep_names = set(per_image.keys())
        for old_dir in list(syn_hst.iterdir()):
            if old_dir.is_dir() and old_dir.name not in keep_names:
                import shutil as _shutil
                _shutil.rmtree(old_dir)
                print(f"  Removed stale synthetic image dir: {old_dir.name}")

    # Write synthetic Gaia catalog
    syn_gaia_path = syn_gaia / f"{field_name}_synthetic_gaia.csv"
    gaia_syn.to_csv(syn_gaia_path, index=False)
    print(f"  Wrote synthetic Gaia catalog ({len(gaia_syn)} stars)")

    # ── Pass 2: per-image forward model + write ───────────────────────────────
    n_written = 0
    for img_name, info in per_image.items():
        match    = info["match"].copy()
        t        = info["transform"]
        mjd      = info["mjd"]
        hst_idx  = match["hst_index"].to_numpy(int)
        gaia_ids = match["gaia_source_id"].to_numpy(np.int64)

        rows_true = np.array([id_to_row.get(int(gid), -1) for gid in gaia_ids])
        valid     = rows_true >= 0
        if not valid.all():
            print(f"  {img_name}: {(~valid).sum()} IDs not in truth — dropping")
            match     = match.iloc[valid].reset_index(drop=True)
            hst_idx   = hst_idx[valid]
            gaia_ids  = gaia_ids[valid]
            rows_true = rows_true[valid]

        if jitter_sigma > 0.0:
            import warnings
            warnings.warn(
                "jitter_sigma is deprecated; physical alignment parameters are now always "
                "drawn from their prior distributions.  jitter_sigma is ignored.",
                DeprecationWarning, stacklevel=2)

        # ── Draw true alignment parameters for this image ─────────────────────
        # orig_rot_deg = -orientat  (PA of HST y-axis is negative of BP3M angle)
        orig_rot_deg = -t["orientat"]
        true_rot_deg    = orig_rot_deg + rng.normal(0.0, rot_sigma)
        true_ratio      = 1.0 + rng.normal(0.0, ratio_sigma)
        true_on_skew    = rng.normal(0.0, skew_sigma)
        true_off_skew   = rng.normal(0.0, skew_sigma)
        # Pointing offset in RA*cos(dec) and Dec mas; applied equally to all chips.
        true_delta_racosdec_mas = rng.normal(0.0, pointing_sigma)
        true_delta_dec0_mas     = rng.normal(0.0, pointing_sigma)

        abcd = _abcd_from_physical(true_rot_deg, true_ratio, true_on_skew, true_off_skew)
        M    = np.array([[abcd["true_a"], abcd["true_b"]],
                         [abcd["true_c"], abcd["true_d"]]])
        Minv = np.linalg.inv(M)

        # ── Pointing offset in degrees (shared by both forward model branches) ─
        cos_dec_cen = np.cos(np.deg2rad(t["dec_cen"]))
        ra_off_deg  = true_delta_racosdec_mas / (cos_dec_cen * 3.6e6)
        dec_off_deg = true_delta_dec0_mas / 3.6e6

        # ── True (A,B,C,D) for warm-start transformation.csv ─────────────────
        # Invert _fcm_to_abcd: [[A,B],[C,D]] = R_cw(rot)^T @ [[a,b],[c,d]]
        _rr = np.deg2rad(true_rot_deg)
        _cr, _sr = np.cos(_rr), np.sin(_rr)
        _R_ccw = np.array([[_cr, -_sr], [_sr, _cr]])
        _M_abcd = np.array([[abcd["true_a"], abcd["true_b"]],
                             [abcd["true_c"], abcd["true_d"]]])
        _M_ABCD = _R_ccw @ _M_abcd
        ws_A = float(_M_ABCD[0, 0]); ws_B = float(_M_ABCD[0, 1])
        ws_C = float(_M_ABCD[1, 0]); ws_D = float(_M_ABCD[1, 1])

        # Propagate true stellar positions to HST epoch
        _ensure_bp3m()
        from bp3m.astro_utils import plane_project as _plane_project
        pscale_mas = t["pixel_scale"] * 1000.0   # arcsec/pix → mas/pix
        df_prop = gaia_true.iloc[rows_true].copy().reset_index(drop=True)
        ra_p, dec_p = _propagate(df_prop, mjd)

        chip_lo = info.get("chip_pointing_lo")
        chip_hi = info.get("chip_pointing_hi")

        # Use per-chip tangent points only when split_ccd=True.  When split_ccd=False
        # BP3M uses the full-frame ra_cen/dec_cen from transformation.csv, so the
        # forward model must also use that tangent point to remain consistent.
        if split_ccd and chip_lo is not None and chip_hi is not None:
            # ── Split-CCD forward model ───────────────────────────────────────
            # Both chips share the same drawn alignment params (rotation, scale,
            # skew, pointing offset).  Each chip has its own nominal tangent point
            # (chip_pointing_{lo,hi}["ra0/dec0"]) and pixel pivot (Xo, Yo).
            # The same angular pointing offset is applied to both nominal points.
            ra0_lo  = chip_lo["ra0"]  + ra_off_deg
            dec0_lo = chip_lo["dec0"] + dec_off_deg
            ra0_hi  = chip_hi["ra0"]  + ra_off_deg
            dec0_hi = chip_hi["dec0"] + dec_off_deg

            Xo_lo, Yo_lo = chip_lo["Xo"], chip_lo["Yo"]
            Xo_hi, Yo_hi = chip_hi["Xo"], chip_hi["Yo"]

            y_raw     = info["cat_y_raw"][hst_idx]
            y_split   = info.get("y_split", 2048.0)
            is_lo_mask = y_raw <= y_split
            is_hi_mask = ~is_lo_mask

            x_pred = np.empty(len(hst_idx))
            y_pred = np.empty(len(hst_idx))

            if is_lo_mask.any():
                xs_lo, ys_lo = _plane_project(
                    ra_p[is_lo_mask], dec_p[is_lo_mask], ra0_lo, dec0_lo, pscale_mas)
                dxy_lo = (Minv @ np.vstack([xs_lo, ys_lo])).T
                x_pred[is_lo_mask] = Xo_lo + dxy_lo[:, 0]
                y_pred[is_lo_mask] = Yo_lo + dxy_lo[:, 1]
            if is_hi_mask.any():
                xs_hi, ys_hi = _plane_project(
                    ra_p[is_hi_mask], dec_p[is_hi_mask], ra0_hi, dec0_hi, pscale_mas)
                dxy_hi = (Minv @ np.vstack([xs_hi, ys_hi])).T
                x_pred[is_hi_mask] = Xo_hi + dxy_hi[:, 0]
                y_pred[is_hi_mask] = Yo_hi + dxy_hi[:, 1]

            # Per-chip truth rows: BP3M reports delta_ra0_mas = (ra0_final - ra0_init) * 3.6e6
            # where ra0_init is the chip's nominal ra0 from data_loader_flc.
            truth_rows_this = []
            for suffix, ra0_nom, dec0_nom, ra0_t, dec0_t in [
                ("_lo", chip_lo["ra0"], chip_lo["dec0"], ra0_lo, dec0_lo),
                ("_hi", chip_hi["ra0"], chip_hi["dec0"], ra0_hi, dec0_hi),
            ]:
                truth_rows_this.append(dict(
                    image_name=img_name + suffix,
                    true_rot_deg=true_rot_deg,
                    true_ratio=true_ratio,
                    true_on_skew=true_on_skew,
                    true_off_skew=true_off_skew,
                    true_delta_ra0_mas=(ra0_t - ra0_nom) * 3.6e6,    # raw RA mas
                    true_delta_dec0_mas=(dec0_t - dec0_nom) * 3.6e6, # Dec mas
                    warmstart_delta_ra0_mas=ra_off_deg * 3.6e6,       # baked into CHIP CRVAL
                    warmstart_delta_dec0_mas=dec_off_deg * 3.6e6,
                    **abcd,
                ))

        else:
            # ── Full-frame forward model ──────────────────────────────────────
            ra_cen  = t["ra_cen"]
            dec_cen = t["dec_cen"]
            Xo = t["x_cen"]
            Yo = t["y_cen"]
            ra0_true  = ra_cen  + ra_off_deg
            dec0_true = dec_cen + dec_off_deg

            xs_bp3m, ys_bp3m = _plane_project(ra_p, dec_p, ra0_true, dec0_true, pscale_mas)
            dxy    = (Minv @ np.vstack([xs_bp3m, ys_bp3m])).T
            x_pred = Xo + dxy[:, 0]
            y_pred = Yo + dxy[:, 1]

            truth_rows_this = [dict(
                image_name=img_name,
                true_rot_deg=true_rot_deg,
                true_ratio=true_ratio,
                true_on_skew=true_on_skew,
                true_off_skew=true_off_skew,
                true_delta_ra0_mas=(ra0_true - ra_cen) * 3.6e6,
                true_delta_dec0_mas=(dec0_true - dec_cen) * 3.6e6,
                warmstart_delta_ra0_mas=ra_off_deg * 3.6e6,   # baked into transformation.csv ra_cen
                warmstart_delta_dec0_mas=dec_off_deg * 3.6e6,
                **abcd,
            )]

        # Draw noise from real per-star covariance
        cov_xx = info["cat_cov_xx"][hst_idx]
        cov_yy = info["cat_cov_yy"][hst_idx]
        cov_xy = info["cat_cov_xy"][hst_idx]
        noise_xy = np.zeros((len(hst_idx), 2))
        for k in range(len(hst_idx)):
            C_k = np.array([[max(cov_xx[k], 1e-8), cov_xy[k]],
                             [cov_xy[k],            max(cov_yy[k], 1e-8)]])
            eigv = np.linalg.eigvalsh(C_k)
            if eigv[0] < 0:
                C_k += (-eigv[0] + 1e-10) * np.eye(2)
            noise_xy[k] = rng.multivariate_normal([0., 0.], C_k)

        x_syn = x_pred + noise_xy[:, 0]
        y_syn = y_pred + noise_xy[:, 1]

        # Write synthetic image directory
        syn_img = syn_hst / img_name
        syn_img.mkdir(parents=True, exist_ok=True)

        # Symlink FLC FITS (header only needed; no position data changes)
        dst_flc = syn_img / info["flc_path"].name
        if not dst_flc.exists():
            os.symlink(info["flc_path"].resolve(), dst_flc)

        # Write warm-start transformation.csv with true alignment parameters
        # (instead of symlinking to the original GaiaHub fit) so that the
        # solver starts near the truth and Phase 0 sees only noise-level residuals.
        _orig_tran = pd.read_csv(info["img_dir"] / "transformation.csv",
                                 index_col="parameter")["value"]
        _ws_updates = {
            "A": ws_A, "B": ws_B, "C": ws_C, "D": ws_D,
            "ra_cen":  t["ra_cen"]  + ra_off_deg,
            "dec_cen": t["dec_cen"] + dec_off_deg,
            "orientat": -true_rot_deg,
            "ratio":    true_ratio,
            "on_skew":  true_on_skew,
            "off_skew": true_off_skew,
        }
        if "rot_deg" in _orig_tran.index:
            _ws_updates["rot_deg"] = true_rot_deg
        _new_tran = _orig_tran.copy()
        for _k, _v in _ws_updates.items():
            _new_tran[_k] = _v
        pd.DataFrame({"parameter": _new_tran.index, "value": _new_tran.values}).to_csv(
            syn_img / "transformation.csv", index=False)

        dst_cat = syn_img / info["cat_path"].name
        _write_synthetic_catalog(
            src_catalog=info["cat_path"],
            dst_catalog=dst_cat,
            hst_idx=hst_idx,
            x_syn=x_syn,
            y_syn=y_syn,
        )

        # Update CHIP{n}_CRVAL1/2 in catalog header with drawn pointing offset
        # so split-CCD chips get the correct warm-start tangent point.
        with afits.open(str(dst_cat), mode='update') as _hdu:
            _hdr = _hdu[1].header
            for _key in list(_hdr.keys()):
                if _key.endswith('_CRVAL1'):
                    _hdr[_key] = float(_hdr[_key]) + ra_off_deg
                elif _key.endswith('_CRVAL2'):
                    _hdr[_key] = float(_hdr[_key]) + dec_off_deg
            _hdu.flush()

        # hst_index must be 0..N-1 after catalog trimming (catalog now only has
        # the matched rows in the same order as hst_idx).
        match_out = match.copy()
        match_out["gaia_source_id"] = gaia_ids
        match_out["hst_index"] = np.arange(len(hst_idx), dtype=int)
        match_out.to_csv(syn_img / "matched_gaia.csv", index=False)
        per_image[img_name]["truth_rows"] = truth_rows_this
        n_written += 1

    # ── Update v_true for all_5p_gaia promoted stars ─────────────────────────
    # The forward model (above) used gaia_true which has the physical PM truth.
    # Now update v_true to record the correction from the synthetic catalog value
    # so that compare_synthetic_results computes residual = bp3m_output - truth.
    # For promoted stars: truth_offset = physical_pm - catalog_pm = -noise
    if _all5p_noise is not None:
        no_pm_mask, noise_pmra, noise_pmdec, noise_plx = _all5p_noise
        v_true[no_pm_mask, 2] = -noise_pmra
        v_true[no_pm_mask, 3] = -noise_pmdec
        v_true[no_pm_mask, 4] = -noise_plx

    # ── Write truth tables ────────────────────────────────────────────────────
    pd.DataFrame({
        "gaia_source_id":         gaia_obs["source_id"].values.astype(np.int64),
        "true_delta_racosdec":    v_true[:, 0],
        "true_delta_dec":         v_true[:, 1],
        "true_pmra":              v_true[:, 2],
        "true_pmdec":             v_true[:, 3],
        "true_parallax":          v_true[:, 4],
        "gmag": (gaia_obs["gmag"].values if "gmag" in gaia_obs.columns
                 else np.full(len(gaia_obs), np.nan)),
    }).to_csv(syn_truth / "stellar_truth.csv", index=False)

    # image_truth.csv has one row per CCD half (img_lo / img_hi when split,
    # or one row per base image when unsplit).  Columns:
    #   image_name            — chip-specific name (matches BP3M image_transformations.csv)
    #   true_rot_deg          — drawn rotation (orig_rot + N(0,0.01²))
    #   true_ratio            — drawn pixel-scale ratio (1 + N(0,1e-4²))
    #   true_on_skew          — drawn on-axis skew (N(0,1e-5²))
    #   true_off_skew         — drawn off-axis skew (N(0,1e-5²))
    #   true_a, true_b, true_c, true_d  — BP3M abcd from physical params
    #   true_delta_ra0_mas    — BP3M-equivalent Δα0 = (ra0_true - chip_ra0) × 3.6e6
    #   true_delta_dec0_mas   — BP3M-equivalent Δδ0 = (dec0_true - chip_dec0) × 3.6e6
    img_truth_rows = []
    for img_name, info in per_image.items():
        img_truth_rows.extend(info.get("truth_rows", []))
    pd.DataFrame(img_truth_rows).to_csv(syn_truth / "image_truth.csv", index=False)

    print(f"  Wrote truth tables to {syn_truth}")
    print(f"  Synthetic data complete: {n_written}/{len(per_image)} images")
    print(f"\n  To run BP3M on synthetic data:")
    print(f"    run_alignment(output_dir='{field_path}', field_name='{syn_name}')")
    return syn_dir


def compare_synthetic_results(
    output_dir: Path,
    field_name: str,
    syn_name: str = "synthetic",
) -> pd.DataFrame:
    """
    Compare BP3M posteriors against the synthetic truth table.

    Expects BP3M results at {output_dir}/{field_name}/{syn_name}/BP3M_results/.
    Writes synthetic_comparison.csv and diagnostic plots to the same directory.

    Returns
    -------
    pd.DataFrame with per-star residuals and pulls
    """
    field_path = Path(output_dir) / field_name
    syn_dir    = field_path / syn_name
    bp3m_dir   = syn_dir / "BP3M_results"
    truth_dir  = syn_dir / "truth"

    for p in (bp3m_dir / "stellar_astrometry.csv", truth_dir / "stellar_truth.csv"):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    truth   = pd.read_csv(truth_dir / "stellar_truth.csv")
    results = pd.read_csv(bp3m_dir  / "stellar_astrometry.csv")

    # Normalise ID column names for merge
    if "gaia_source_id" in truth.columns and "Gaia_id" in results.columns:
        truth = truth.rename(columns={"gaia_source_id": "Gaia_id"})

    # Drop truth columns that duplicate results columns (e.g. gmag written for convenience)
    dup_cols = [c for c in truth.columns if c in results.columns and c != "Gaia_id"]
    truth = truth.drop(columns=dup_cols)

    cmp = results.merge(truth, on="Gaia_id", how="inner")
    if len(cmp) == 0:
        raise ValueError("No overlapping Gaia IDs between BP3M results and truth.")
    print(f"\n{'─'*50}")
    print(f"Synthetic comparison: {len(cmp)} stars")
    print(f"{'─'*50}")

    # Residuals (recovered − true).
    # delta_racosdec_bp3m and delta_dec_bp3m are already OFFSETS from Gaia (v_survey=0
    # for position in BP3M), so compare directly to the drawn truth offsets.
    #
    # pmra_bp3m / pmdec_bp3m / parallax_bp3m are ABSOLUTE values (v_survey = gaia
    # catalog values for 5/6-param stars; 0 for 2-param stars).  Convert to a
    # correction-from-catalog before comparing to v_true, which is also an offset.
    has_gaia_pm = cmp["pmra"].notna()
    gaia_pmra     = cmp["pmra"].fillna(0.0)
    gaia_pmdec    = cmp["pmdec"].fillna(0.0)
    gaia_parallax = cmp["parallax"].fillna(0.0)

    cmp["resid_delta_racosdec"] = cmp["delta_racosdec_bp3m"] - cmp["true_delta_racosdec"]
    cmp["resid_delta_dec"]      = cmp["delta_dec_bp3m"]      - cmp["true_delta_dec"]
    cmp["resid_pmra"]           = (cmp["pmra_bp3m"]      - gaia_pmra)     - cmp["true_pmra"]
    cmp["resid_pmdec"]          = (cmp["pmdec_bp3m"]     - gaia_pmdec)    - cmp["true_pmdec"]
    cmp["resid_parallax"]       = (cmp["parallax_bp3m"]  - gaia_parallax) - cmp["true_parallax"]

    # Pulls (residual / marginalised 1σ)
    for key, sig_col in [("delta_racosdec", "sigma_delta_racosdec"),
                          ("delta_dec",      "sigma_delta_dec"),
                          ("pmra",           "sigma_pmra_bp3m"),
                          ("pmdec",          "sigma_pmdec_bp3m"),
                          ("parallax",       "sigma_parallax_bp3m")]:
        sig = cmp[sig_col].replace(0, np.nan)
        cmp[f"pull_{key}"] = cmp[f"resid_{key}"] / sig

    # Summary table
    params = [("delta_racosdec", "Δ(Δα*) [mas]  "),
              ("delta_dec",      "Δ(Δδ)  [mas]  "),
              ("pmra",           "Δμα*   [mas/yr]"),
              ("pmdec",          "Δμδ    [mas/yr]"),
              ("parallax",       "Δϖ     [mas]  ")]
    print(f"  {'Parameter':<22} {'bias':>8} {'RMS':>8} {'pull μ':>8} {'pull σ':>8}  {'N':>5}")
    print(f"  {'─'*60}")
    for key, label in params:
        resid = cmp[f"resid_{key}"].dropna()
        pull  = cmp[f"pull_{key}"].dropna()
        print(f"  {label:<22} {resid.mean():>8.3f} {resid.std():>8.3f} "
              f"{pull.mean():>8.3f} {pull.std():>8.3f}  {len(resid):>5}")

    # 5D chi2 per star using full posterior covariance
    v_cov_marg_path = bp3m_dir / "v_cov_marginalised.npy"
    C_vT_path       = bp3m_dir / "C_vT.npy"
    if v_cov_marg_path.exists() and C_vT_path.exists():
        _compute_stellar_chi2(cmp, v_cov_marg_path, C_vT_path, bp3m_dir)

    # Image transformation comparison
    img_truth_path   = truth_dir  / "image_truth.csv"
    img_results_path = bp3m_dir   / "image_transformations.csv"
    if img_results_path.exists():
        img_results = pd.read_csv(img_results_path).copy()
        # strip _hi/_lo suffixes added by split_ccd before merging with truth
        img_results["image_name_base"] = (img_results["image_name"]
                                          .str.replace(r"_(hi|lo)$", "", regex=True))

        if img_truth_path.exists():
            img_truth = pd.read_csv(img_truth_path)
            # Pass 1: direct name match (new per-chip truth tables where names
            # match exactly: img_lo ↔ img_lo, img_hi ↔ img_hi).
            direct = img_results.merge(
                img_truth, on="image_name", how="left", suffixes=("", "_truth"))
            matched_direct = direct[direct["true_a"].notna()]
            unmatched = direct[direct["true_a"].isna()].drop(
                columns=[c for c in direct.columns if c.endswith("_truth") or c in img_truth.columns[1:]],
                errors="ignore")

            # Pass 2: for BP3M images not matched in pass 1, try:
            #   (a) strip _hi/_lo from the BP3M name and look up the base in truth
            #       (backward compat: truth has single base-name rows)
            #   (b) look up img_lo / img_hi rows in truth when BP3M kept the image
            #       unsplit (BP3M name has no suffix, but truth has lo/hi rows)
            extra_rows = []
            truth_name_set = set(img_truth["image_name"])
            for _, row in unmatched.iterrows():
                bname = row["image_name_base"]
                # case (a): base name directly in truth
                t_rows = img_truth[img_truth["image_name"] == bname]
                if len(t_rows) == 0:
                    # case (b): any lo/hi variant of bname in truth
                    t_rows = img_truth[img_truth["image_name"].str.startswith(bname + "_")]
                if len(t_rows) > 0:
                    merged = {**row.to_dict(),
                              **t_rows.iloc[0].to_dict(),
                              "image_name": row["image_name"]}
                    extra_rows.append(merged)

            parts = [matched_direct]
            if extra_rows:
                parts.append(pd.DataFrame(extra_rows))
            img_cmp = pd.concat(parts, ignore_index=True) if len(parts) > 1 else matched_direct
            # Compare BP3M output a,b,c,d,delta_ra0_mas,delta_dec0_mas to fitted truth.
            # Columns: (result_col, truth_col, sigma_col_in_results)
            #
            # For warm-started runs, delta_ra0_mas and delta_dec0_mas in BP3M are
            # residuals from the warm-started reference point, not from the original
            # ra_cen.  image_truth.csv stores warmstart_delta_ra0_mas so we can
            # compute the correct truth in BP3M's frame:
            #   bp3m_truth = true_delta_ra0 - warmstart_delta_ra0  (≈ 0 for perfect WS)
            if ("warmstart_delta_ra0_mas"  in img_cmp.columns and
                    "warmstart_delta_dec0_mas" in img_cmp.columns):
                img_cmp["bp3m_true_delta_ra0_mas"]  = (img_cmp["true_delta_ra0_mas"]
                                                        - img_cmp["warmstart_delta_ra0_mas"])
                img_cmp["bp3m_true_delta_dec0_mas"] = (img_cmp["true_delta_dec0_mas"]
                                                        - img_cmp["warmstart_delta_dec0_mas"])
                dra_truth_col  = "bp3m_true_delta_ra0_mas"
                ddec_truth_col = "bp3m_true_delta_dec0_mas"
            else:
                dra_truth_col  = "true_delta_ra0_mas"
                ddec_truth_col = "true_delta_dec0_mas"

            bp3m_params = [
                ("a",             "true_a",         "sigma_a"),
                ("b",             "true_b",         "sigma_b"),
                ("c",             "true_c",         "sigma_c"),
                ("d",             "true_d",         "sigma_d"),
                ("delta_ra0_mas",  dra_truth_col,   "sigma_dra0_mas"),
                ("delta_dec0_mas", ddec_truth_col,  "sigma_ddec0_mas"),
            ]
            for bp3m_col, true_col, _ in bp3m_params:
                if bp3m_col in img_cmp and true_col in img_cmp:
                    img_cmp[f"resid_{bp3m_col}"] = img_cmp[bp3m_col] - img_cmp[true_col]
            img_cmp.to_csv(bp3m_dir / "synthetic_image_comparison.csv", index=False)
            print(f"\n  Image transformation residuals "
                  f"({len(img_cmp)} CCD halves, {len(img_truth)} images):")
            print(f"  {'param':<16} {'bias':>10} {'RMS':>10}  {'pull μ':>8} {'pull σ':>8}")
            for bp3m_col, _, sig_col in bp3m_params:
                col = f"resid_{bp3m_col}"
                if col in img_cmp:
                    r = img_cmp[col].dropna()
                    if sig_col in img_cmp:
                        pull = (r / img_cmp[sig_col].dropna().replace(0, np.nan)).dropna()
                        pull_str = f"{pull.mean():>8.3f} {pull.std():>8.3f}"
                    else:
                        pull_str = "     n/a      n/a"
                    print(f"  {bp3m_col:<16} {r.mean():>10.5f} {r.std():>10.5f}  {pull_str}")

            # Per-image chi2 using full 6×6 covariance block from C_r.npy
            C_r_path = bp3m_dir / "C_r.npy"
            if C_r_path.exists():
                _print_image_chi2(img_cmp, C_r_path, bp3m_params, bp3m_dir)

        else:
            img_results.to_csv(bp3m_dir / "synthetic_image_comparison.csv", index=False)
            print(f"  (no image_truth.csv found — skipping transform comparison)")

    # Write comparison CSV + plots
    cmp.to_csv(bp3m_dir / "synthetic_comparison.csv", index=False)
    print(f"\n  Saved: {bp3m_dir / 'synthetic_comparison.csv'}")
    _plot_diagnostics(cmp, bp3m_dir)
    return cmp


def run_conditional_solve(
    output_dir: Path,
    field_name: str,
    syn_name: str = "synthetic",
    split_ccd: bool = True,
    min_stars_split_ccd: int = 20,
    poly_order: int = 1,
    inflate_hst_errors: bool = True,
    bp3m_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run BP3M's stellar solver with image parameters fixed at r_true.

    Calls solver._solve_one_pass(r_true) so the returned a_arr / C_vT are
    the conditional stellar posteriors given the TRUE image parameters.  Pulls
    computed from these posteriors isolate the stellar solver from any errors
    in the image EM step.

    Writes synthetic_comparison_cond.csv and plots_syn_pm_pulls_cond.png to
    the BP3M_results directory.
    """
    # bp3m is installed as a package; no sys.path manipulation needed
    from bp3m.data_loader_flc import load_image_data_flc
    from bp3m.data_loader_flc import build_index_maps, split_images_by_ccd
    from bp3m.solver import BP3MSolver

    field_path = Path(output_dir) / field_name
    syn_dir    = field_path / syn_name
    bp3m_res   = syn_dir / "BP3M_results"
    truth_dir  = syn_dir / "truth"

    print(f"\n{'─'*50}")
    print("Conditional solve: fixing r = r_true")
    print(f"{'─'*50}")

    # ── Build solver from synthetic data ─────────────────────────────────────
    data_root = field_path
    imgs, stars_per_image, gaia_catalog = load_image_data_flc(data_root, syn_name)
    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)

    observed_ids = set()
    for spi in stars_per_image.values():
        observed_ids.update(spi["Gaia_id"].values)
    gaia_catalog = (gaia_catalog[gaia_catalog["Gaia_id"].isin(observed_ids)]
                    .reset_index(drop=True))
    star_id_to_idx = {gid: i for i, gid in enumerate(gaia_catalog["Gaia_id"])}

    if split_ccd:
        imgs, stars_per_image = split_images_by_ccd(
            imgs, stars_per_image, min_stars_per_ccd=min_stars_split_ccd)
        image_names = sorted(stars_per_image.keys())
        star_id_to_idx, image_names, star_in_image = build_index_maps(
            stars_per_image, gaia_catalog)

    solver = BP3MSolver(imgs, stars_per_image, gaia_catalog,
                        star_id_to_idx, image_names, star_in_image,
                        poly_order=poly_order)

    # Run one converged pass so rolling re-linearization accumulates Δα0/Δδ0
    # into ra0_current; after convergence r_hat[4:6]=0 per image (true Δα0=0
    # for synthetic data generated at transformation.csv tangent point).
    r_hat_converged, _, _, _, _, _, _ = solver.fit(n_iter=1, clip_sigma=None,
                                                    inflate_hst_errors=inflate_hst_errors,
                                                    prefilter=False)

    # ── Build r_true vector in BP3M ordering ──────────────────────────────────
    img_truth = pd.read_csv(truth_dir / "image_truth.csv")
    img_truth_map = {row["image_name"]: row
                     for _, row in img_truth.iterrows()}

    nr  = solver.N_R
    # After one convergence pass, _update_geometry has absorbed all Δα0/Δδ0
    # corrections into ra0_current and reset r_hat[4:6] = 0 for each image.
    # The conditional solve evaluates the stellar posterior at the TRUE image
    # params.  For abcd (indices 0-3), we override with the drawn truth.
    # For Δα0/Δδ0 (indices 4,5), the converged r_hat already has them ≈ 0
    # (geometry absorbed), so we also set them to 0 relative to the current
    # tangent point — which after convergence lies close to the true tangent
    # point.  This is correct as long as n_iter=1 convergence moved ra0_current
    # close enough to ra0_true that the residual Δα0/Δδ0 is negligible.
    r_true_vec = r_hat_converged.copy()
    for j_idx, img in enumerate(solver.image_names):
        # Try chip-specific name first (new per-chip truth tables), then strip suffix
        if img in img_truth_map:
            row = img_truth_map[img]
        else:
            base = img.replace("_hi", "").replace("_lo", "")
            if base not in img_truth_map:
                print(f"  WARNING: no truth for image {img} (base={base}) — using r_init")
                continue
            row = img_truth_map[base]
        cs = j_idx * nr
        r_true_vec[cs + 0] = row["true_a"]
        r_true_vec[cs + 1] = row["true_b"]
        r_true_vec[cs + 2] = row["true_c"]
        r_true_vec[cs + 3] = row["true_d"]
        # Δα0/Δδ0 = 0 after convergence (geometry accumulated into ra0_current)

    # ── Conditional stellar solve at r_true ───────────────────────────────────
    _, _, a_arr, _, C_vT = solver._solve_one_pass(r_true_vec)

    # ── Compute pulls against truth ───────────────────────────────────────────
    truth   = pd.read_csv(truth_dir  / "stellar_truth.csv")
    results = pd.read_csv(bp3m_res   / "stellar_astrometry.csv")
    if "gaia_source_id" in truth.columns and "Gaia_id" in results.columns:
        truth = truth.rename(columns={"gaia_source_id": "Gaia_id"})
    dup = [c for c in truth.columns if c in results.columns and c != "Gaia_id"]
    truth = truth.drop(columns=dup)
    cmp = results.merge(truth, on="Gaia_id", how="inner")

    # Build index mapping from Gaia_id → solver row
    gaia_ids_solver = gaia_catalog["Gaia_id"].values
    id_to_solver = {int(gid): i for i, gid in enumerate(gaia_ids_solver)}

    gaia_pmra     = cmp["pmra"].fillna(0.0)
    gaia_pmdec    = cmp["pmdec"].fillna(0.0)
    gaia_parallax = cmp["parallax"].fillna(0.0)
    gaia_ra_off   = np.zeros(len(cmp))
    gaia_dec_off  = np.zeros(len(cmp))

    # v_truth[k] = physical truth as correction from gaia catalog (= v_true)
    v_truth = np.column_stack([
        cmp["true_delta_racosdec"].values,
        cmp["true_delta_dec"].values,
        cmp["true_pmra"].values,
        cmp["true_pmdec"].values,
        cmp["true_parallax"].values,
    ])  # (N_cmp, 5)

    # Gaia catalog values in same 5-param order
    v_gaia = np.column_stack([
        gaia_ra_off,
        gaia_dec_off,
        gaia_pmra.values,
        gaia_pmdec.values,
        gaia_parallax.values,
    ])  # (N_cmp, 5)

    resid_cond = np.full((len(cmp), 5), np.nan)
    sigma_cond = np.full((len(cmp), 5), np.nan)
    n_matched  = 0
    for k, row in enumerate(cmp.itertuples(index=False)):
        gid = int(row.Gaia_id)
        si  = id_to_solver.get(gid, -1)
        if si < 0:
            continue
        n_matched += 1
        # a_arr[si] = conditional posterior mean (absolute, like pmra_bp3m)
        # residual = (a_arr - v_gaia) - v_true
        resid_cond[k] = (a_arr[si] - v_gaia[k]) - v_truth[k]
        sigma_cond[k] = np.sqrt(np.diag(C_vT[si]))

    print(f"  Matched {n_matched}/{len(cmp)} stars to solver")

    pull_cond = resid_cond / sigma_cond

    param_labels = ["Δα*", "Δδ", "μα*", "μδ", "ϖ"]
    param_keys   = ["delta_racosdec", "delta_dec", "pmra", "pmdec", "parallax"]
    print(f"\n  Conditional pulls (r fixed at r_true):")
    print(f"  {'param':<10} {'pull μ':>8} {'pull σ':>8}  {'N':>5}")
    print(f"  {'─'*38}")
    for i, (lbl, key) in enumerate(zip(param_labels, param_keys)):
        p = pull_cond[:, i]
        finite = p[np.isfinite(p)]
        print(f"  {lbl:<10} {finite.mean():>8.3f} {finite.std():>8.3f}  {len(finite):>5}")
        cmp[f"pull_{key}_cond"] = p

    # Also print comparison with marginalised pulls from normal run (if available)
    normal_cmp_path = bp3m_res / "synthetic_comparison.csv"
    if normal_cmp_path.exists():
        normal_cmp = pd.read_csv(normal_cmp_path)
        print(f"\n  Normal (marginalised) pulls for comparison:")
        print(f"  {'param':<10} {'pull μ':>8} {'pull σ':>8}")
        print(f"  {'─'*30}")
        for key, lbl in zip(param_keys, param_labels):
            col = f"pull_{key}"
            if col in normal_cmp.columns:
                p = normal_cmp[col].dropna()
                print(f"  {lbl:<10} {p.mean():>8.3f} {p.std():>8.3f}")

    cmp.to_csv(bp3m_res / "synthetic_comparison_cond.csv", index=False)
    _plot_cond_pulls(pull_cond, param_labels, bp3m_res)
    print(f"\n  Saved: {bp3m_res / 'synthetic_comparison_cond.csv'}")
    return cmp


def _plot_cond_pulls(pull_cond: np.ndarray, param_labels: list, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import norm as _norm
    except ImportError:
        return

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    x_ref = np.linspace(-5, 5, 200)
    for i, (ax, lbl) in enumerate(zip(axes, param_labels)):
        vals = pull_cond[:, i]
        finite = vals[np.isfinite(vals)]
        ax.hist(finite, bins=30, density=True, alpha=0.6, color="steelblue")
        ax.plot(x_ref, _norm.pdf(x_ref), "k--", lw=1.2)
        ax.set_xlabel(f"pull {lbl} | r_true")
        ax.set_xlim(-5, 5)
        mu, sigma = finite.mean(), finite.std()
        ax.set_title(f"μ={mu:.2f}  σ={sigma:.2f}")
    axes[0].set_ylabel("density")
    fig.suptitle("Conditional pulls (r fixed at r_true) — should be ≈ N(0,1)")
    fig.tight_layout()
    fig.savefig(out_dir / "plots_syn_pm_pulls_cond.png", dpi=120)
    plt.close(fig)


# ── stellar 5D chi2 ──────────────────────────────────────────────────────────

def _compute_stellar_chi2(cmp: pd.DataFrame, v_cov_marg_path: Path,
                          C_vT_path: Path, out_dir: Path) -> None:
    """
    Compute per-star 5D chi2 using the full marginalised posterior covariance.

    Δv_i = (resid_delta_racosdec, resid_delta_dec, resid_pmra, resid_pmdec, resid_parallax)
    χ²_i = Δv_i^T @ C_full_i^{-1} @ Δv_i

    where C_full = v_cov_marginalised + C_vT (total posterior covariance).
    Saves chi2 column to cmp and plots a histogram vs chi2(5).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import chi2 as chi2_dist
    except ImportError:
        plt = None
        chi2_dist = None

    v_cov  = np.load(v_cov_marg_path)   # (N, 5, 5)
    C_vT   = np.load(C_vT_path)          # (N, 5, 5)
    C_full = v_cov + C_vT                 # (N, 5, 5)

    resid_cols = ["resid_delta_racosdec", "resid_delta_dec",
                  "resid_pmra", "resid_pmdec", "resid_parallax"]

    # BP3M results are sorted by Gaia_id; truth/cmp order may differ.
    # Align cmp rows to C_full ordering via integer Gaia_id arrays.
    # iterrows() upcasts int64 to float64 when other columns have NaN, which
    # silently corrupts 18-digit Gaia source IDs. Use vectorised numpy lookup.
    astro = pd.read_csv(out_dir / "stellar_astrometry.csv")
    astro_ids = astro["Gaia_id"].values.astype(np.int64)
    id_to_bp3m_row = {int(gid): i for i, gid in enumerate(astro_ids)}

    cmp_ids = cmp["Gaia_id"].values.astype(np.int64)
    resid_mat = cmp[resid_cols].values   # (N_cmp, 5), float64

    chi2_vals = np.full(len(cmp), np.nan)
    for k, gid in enumerate(cmp_ids):
        bp3m_row = id_to_bp3m_row.get(int(gid), -1)
        if bp3m_row < 0:
            continue
        dv = resid_mat[k]
        if np.any(np.isnan(dv)):
            continue
        C_i = C_full[bp3m_row]
        try:
            chi2_vals[k] = float(dv @ np.linalg.inv(C_i) @ dv)
        except np.linalg.LinAlgError:
            pass

    cmp["chi2_5d"] = chi2_vals
    finite_chi2 = cmp["chi2_5d"].dropna()
    print(f"\n  5D stellar chi² (5 dof): N={len(finite_chi2)}, "
          f"median={finite_chi2.median():.2f}, mean={finite_chi2.mean():.2f} "
          f"(expected median={4.352:.2f} for χ²(5))")

    if plt is not None and chi2_dist is not None and len(finite_chi2) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        clip = np.percentile(finite_chi2, 98)
        ax.hist(finite_chi2.clip(upper=clip), bins=30, density=True,
                alpha=0.7, color="steelblue", label=f"N={len(finite_chi2)}")
        x = np.linspace(0, clip, 300)
        ax.plot(x, chi2_dist.pdf(x, df=5), "k--", lw=1.5, label="χ²(5) expected")
        ax.set_xlabel("χ² per star (5 dof: Δα*, Δδ, μα*, μδ, ϖ)")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title("Per-star 5D astrometric χ² (should follow χ²(5))")
        fig.tight_layout()
        fig.savefig(out_dir / "plots_syn_stellar_chi2.png", dpi=120)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'plots_syn_stellar_chi2.png'}")


# ── image chi2 and covariance ─────────────────────────────────────────────────

def _print_image_chi2(img_cmp: pd.DataFrame, C_r_path: Path,
                      bp3m_params: list, out_dir: Path) -> None:
    """
    For each CCD half (row in img_cmp), compute chi2 for the image-parameter
    residual vector using the per-image diagonal block of C_r.

    bp3m_params : list of (result_col, truth_col, sigma_col) triples — only
                  entries with a non-NaN resid_ column in img_cmp are included.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    C_r = np.load(C_r_path)
    n_tot = C_r.shape[0]

    # Determine n_r_per_img from C_r size and image count; fall back to 6.
    n_img_rows = len(img_cmp)
    if n_img_rows > 0 and n_tot % n_img_rows == 0:
        n_r_per_img = n_tot // n_img_rows
    else:
        n_r_per_img = 6   # poly_order=1 default

    # Build the ordered list of result columns that have residuals
    resid_cols = [col for col, _, _ in bp3m_params
                  if f"resid_{col}" in img_cmp.columns]
    n_params = len(resid_cols)
    if n_params == 0:
        print("  Per-image chi²: no residual columns found, skipping")
        return

    # Map result_col → index in the r_j vector (a=0,b=1,c=2,d=3,Δα0=4,Δδ0=5)
    _col_to_ridx = {"a": 0, "b": 1, "c": 2, "d": 3,
                    "delta_ra0_mas": 4, "delta_dec0_mas": 5}

    param_label = ",".join(resid_cols)
    print(f"\n  Per-image chi² ({n_params} parameters: {param_label}):")
    print(f"  {'image':<22} {'chi2':>8}  {'dof':>4}  {'p-val':>7}")

    img_sorted = img_cmp["image_name"].tolist()
    chi2_vals = []
    row_indices_used = []

    for row_idx, row in img_cmp.iterrows():
        img_name = row["image_name"]
        j = img_sorted.index(img_name)
        cs = j * n_r_per_img

        if cs + n_r_per_img > n_tot:
            print(f"  {img_name:<22}  C_r too small for this image, skipping")
            continue

        resid = np.array([row.get(f"resid_{col}", np.nan) for col in resid_cols])
        if np.any(np.isnan(resid)):
            print(f"  {img_name:<22}  missing residuals, skipping")
            continue

        # Extract the sub-block of C_r for these parameters
        ridxs = [_col_to_ridx.get(col, k) for k, col in enumerate(resid_cols)]
        block_idxs = [cs + ri for ri in ridxs]
        C_j = C_r[np.ix_(block_idxs, block_idxs)]

        try:
            C_j_inv = np.linalg.inv(C_j)
            chi2 = float(resid @ C_j_inv @ resid)
        except np.linalg.LinAlgError:
            print(f"  {img_name:<22}  singular C_r block, skipping")
            continue

        from scipy.stats import chi2 as chi2_dist
        pval = 1.0 - chi2_dist.cdf(chi2, df=n_params)
        print(f"  {img_name:<22} {chi2:>8.3f}  {n_params:>4}  {pval:>7.4f}")
        chi2_vals.append(chi2)
        row_indices_used.append(j)

    if chi2_vals and plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(chi2_vals, bins=max(5, len(chi2_vals)//2), alpha=0.7, color="steelblue",
                label=f"N={len(chi2_vals)}")
        x = np.linspace(0, max(chi2_vals) * 1.2, 200)
        from scipy.stats import chi2 as chi2_dist
        ax.plot(x, chi2_dist.pdf(x, df=n_params) * len(chi2_vals) * (x[1]-x[0]),
                "k--", lw=1.5, label=f"χ²({n_params}) expected")
        ax.set_xlabel(f"χ² per image ({n_params} dof: {param_label})")
        ax.set_ylabel("count")
        ax.legend()
        ax.set_title("Per-image transformation χ²")
        fig.tight_layout()
        fig.savefig(out_dir / "plots_syn_image_chi2.png", dpi=120)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'plots_syn_image_chi2.png'}")

    # Global chi2 using full joint covariance block
    resid_all = []
    idx_all = []
    for j in row_indices_used:
        cs = j * n_r_per_img
        row = img_cmp[img_cmp["image_name"] == img_sorted[j]].iloc[0]
        rv = np.array([row.get(f"resid_{col}", np.nan) for col in resid_cols])
        if np.any(np.isnan(rv)):
            continue
        ridxs = [_col_to_ridx.get(col, k) for k, col in enumerate(resid_cols)]
        resid_all.append(rv)
        idx_all.extend([cs + ri for ri in ridxs])

    if len(resid_all) >= 2:
        resid_vec = np.concatenate(resid_all)
        C_full_block = C_r[np.ix_(idx_all, idx_all)]
        try:
            C_inv = np.linalg.inv(C_full_block)
            chi2_global = float(resid_vec @ C_inv @ resid_vec)
            dof_global = len(resid_vec)
            from scipy.stats import chi2 as chi2_dist
            pval_global = 1.0 - chi2_dist.cdf(chi2_global, df=dof_global)
            print(f"\n  Global alignment χ² (full {dof_global}×{dof_global} joint covariance):")
            print(f"    χ² = {chi2_global:.3f}  dof = {dof_global}  "
                  f"χ²/dof = {chi2_global/dof_global:.3f}  p = {pval_global:.4f}")
        except np.linalg.LinAlgError:
            print("\n  Global chi2: joint C_r block is singular, skipping")


# ── diagnostic plots ──────────────────────────────────────────────────────────

def _plot_diagnostics(cmp: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import norm as _norm
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    gmag = cmp.get("gmag", pd.Series(np.nan, index=cmp.index))
    nhst = cmp.get("n_hst_used", pd.Series(1, index=cmp.index))

    # ── PM residuals vs G magnitude ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ax, key, label in [
        (axes[0], "pmra",  "μα* residual [mas/yr]"),
        (axes[1], "pmdec", "μδ  residual [mas/yr]"),
    ]:
        sc = ax.scatter(gmag, cmp[f"resid_{key}"], c=nhst, cmap="viridis",
                        s=8, alpha=0.6, vmin=1)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel(label)
        plt.colorbar(sc, ax=ax, label="N_HST")
    axes[1].set_xlabel("G [mag]")
    fig.suptitle("PM residuals (BP3M − truth) vs G magnitude")
    fig.tight_layout()
    fig.savefig(out_dir / "plots_syn_pm_residuals.png", dpi=120)
    plt.close(fig)

    # ── PM residuals vs true PM value ────────────────────────────────────────
    # physical truth = gaia_catalog + v_true (fillna(0) handles 2p stars)
    phys_pmra  = cmp["pmra"].fillna(0.0) + cmp["true_pmra"]
    phys_pmdec = cmp["pmdec"].fillna(0.0) + cmp["true_pmdec"]
    clipped = nhst == 0   # clipped stars: BP3M returned Gaia prior → resid = -v_true exactly
    obs     = nhst >= 1
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
    for ax, phys, key, xlabel, ylabel in [
        (axes[0], phys_pmra,  "pmra",  "true μα* [mas/yr]", "μα* residual [mas/yr]"),
        (axes[1], phys_pmdec, "pmdec", "true μδ [mas/yr]",  "μδ  residual [mas/yr]"),
    ]:
        resid = cmp[f"resid_{key}"]
        sc = ax.scatter(phys[obs], resid[obs], c=nhst[obs], cmap="viridis",
                        s=8, alpha=0.6, vmin=1)
        # Clipped stars (N_HST=0): resid = -v_true by construction (BP3M returned Gaia prior)
        ax.scatter(phys[clipped], resid[clipped], marker="x", s=30,
                   color="red", linewidths=0.8, zorder=5, label=f"clipped (N_HST=0, n={clipped.sum()})")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(sc, ax=ax, label="N_HST")
        if clipped.sum() > 0:
            ax.legend(fontsize=7)
    fig.suptitle("PM residuals (BP3M − truth) vs true PM\n"
                 "(red × = clipped stars: resid ≡ −v_true by construction)")
    fig.tight_layout()
    fig.savefig(out_dir / "plots_syn_pm_resid_vs_truth.png", dpi=120)
    plt.close(fig)

    # ── VPD of PM residuals ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(cmp.loc[obs, "resid_pmra"], cmp.loc[obs, "resid_pmdec"],
                    c=nhst[obs], cmap="viridis", s=8, alpha=0.6, vmin=1)
    ax.scatter(cmp.loc[clipped, "resid_pmra"], cmp.loc[clipped, "resid_pmdec"],
               marker="x", s=30, color="red", linewidths=0.8, zorder=5,
               label=f"clipped (N_HST=0, n={clipped.sum()})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("μα* residual [mas/yr]")
    ax.set_ylabel("μδ residual [mas/yr]")
    plt.colorbar(sc, ax=ax, label="N_HST")
    if clipped.sum() > 0:
        ax.legend(fontsize=8)
    ax.set_title("VPD of PM residuals (BP3M − truth)")
    fig.tight_layout()
    fig.savefig(out_dir / "plots_syn_pm_resid_vpd.png", dpi=120)
    plt.close(fig)

    # ── Pull histograms ───────────────────────────────────────────────────────
    pull_keys = [("pull_delta_racosdec", "pull Δα*"),
                 ("pull_delta_dec",      "pull Δδ"),
                 ("pull_pmra",           "pull μα*"),
                 ("pull_pmdec",          "pull μδ"),
                 ("pull_parallax",       "pull ϖ")]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    x_ref = np.linspace(-5, 5, 200)
    for ax, (col, label) in zip(axes, pull_keys):
        vals = cmp[col].dropna()
        ax.hist(vals, bins=30, density=True, alpha=0.6, color="steelblue")
        ax.plot(x_ref, _norm.pdf(x_ref), "k--", lw=1.2, label="N(0,1)")
        ax.set_xlabel(label)
        ax.set_xlim(-5, 5)
        mu, sigma = vals.mean(), vals.std()
        ax.set_title(f"μ={mu:.2f}  σ={sigma:.2f}")
    axes[0].set_ylabel("density")
    fig.suptitle("Pull distributions (should be ≈ N(0,1))")
    fig.tight_layout()
    fig.savefig(out_dir / "plots_syn_pm_pulls.png", dpi=120)
    plt.close(fig)

    print(f"  Saved diagnostic plots to {out_dir}")
