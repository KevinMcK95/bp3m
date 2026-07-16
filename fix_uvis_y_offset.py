#!/usr/bin/env python
"""
fix_uvis_y_offset.py
====================
Retroactively fix WFC3/UVIS catalog.fits files produced with the wrong
chip-1 y_offset (2051 instead of the correct 2048).

Background
----------
The STDGDC tables were built by hst1pass placing WFC3/UVIS chip 1 at
y_combined = chip_local_y + 2048, yielding a 4096-row combined frame
that exactly matches the 4096×4096 GDC grid.  bp3m/pypass was using
y_offset=2051, so every chip-4 star's combined-frame y was 3 too high.
This caused:
  - x_gdc, y_gdc shifted by ~3 px for all chip-4 stars
  - Stars at chip-local y ≥ 2045 (combined ≥ 4096) incorrectly got NaN
  - Stars at chip-local y ≥ 2048 (combined ≥ 4096 after fix, was ≥ 4099)
    that were previously valid now correctly become NaN
  - ~3-px inter-chip residual in Gaia cross-matching for fields with
    unbalanced chip counts (e.g. Sag_DIG WFC3/UVIS)

Note on ra/dec columns
-----------------------
The catalog ra, dec, ra_err, dec_err, cov_ra_ra, cov_dec_dec, cov_ra_dec
columns are computed from raw chip-local (x, y) through the chip's FITS
WCS — they are independent of the GDC and y_offset.  They are NOT updated
by this script (their values are unaffected by the fix).

For each catalog with CHIP4_Y_OFFSET == 2051:
  1. Subtract 3 from y for all chip-4 stars (combined-frame y)
  2. Recompute x_gdc, y_gdc, mc at corrected y via apply_gdc
  3. Recompute GDC Jacobian at corrected y → new cov_xx/yy/xy_gdc
  4. Recompute mag_gdc, mag_st_gdc with new mc
  5. Update header: CHIP4_Y_OFFSET, CHIP4_CRPIX2_COMBINED, CHIP4_CRPIX1/2_GDC
  6. Delete matched_gaia.csv and xmatch*.json in the same directory

Usage
-----
    python fix_uvis_y_offset.py [--dry-run] [--root ROOT] [--jobs N]

Options
-------
--dry-run   Print what would be done, modify nothing.
--root      Root directory to search (default: ~/data_bootes/bp3m/GaiaHub_results)
--jobs      Parallel workers (default: 4)
"""
import argparse
import os
import sys
import glob
import shutil
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------------
# Locate lib_dir from bp3m config
# ---------------------------------------------------------------------------

def _get_lib_dir() -> Path:
    config = Path.home() / ".bp3m" / "config.toml"
    if not config.exists():
        # Fallback to bootes location
        fb = Path("/bootes_raid6/users/kmckinnon/.bp3m/config.toml")
        if fb.exists():
            config = fb
        else:
            raise FileNotFoundError("Cannot find bp3m config.toml")
    for line in config.read_text().splitlines():
        line = line.strip()
        if line.startswith("lib_dir"):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            return Path(val)
    raise ValueError(f"lib_dir not found in {config}")


# ---------------------------------------------------------------------------
# GDC utilities (replicated from pypass/io.py to keep this script standalone)
# ---------------------------------------------------------------------------

def _load_stdgdc(path: Path) -> dict:
    with fits.open(path, memmap=False) as hdul:
        h = hdul[0].header
        nx = int(h['NDIM_XGC'])
        ny = int(h['NDIM_YGC'])
        xgc_0 = float(h.get('XGC_0', 0))
        ygc_0 = float(h.get('YGC_0', 0))
        xgc = hdul[1].data.astype(np.float64)   # (ny, nx) after FITS transpose
        ygc = hdul[2].data.astype(np.float64)
        mgc = hdul[3].data.astype(np.float64)
    return {'xgc': xgc, 'ygc': ygc, 'mgc': mgc,
            'ndim_xgc': nx, 'ndim_ygc': ny,
            'xgc_0': xgc_0, 'ygc_0': ygc_0}


def _apply_gdc(x_raw, y_raw, gdc):
    """Forward GDC (SENSE=1) with bilinear interpolation.  Returns x_corr, y_corr, mc."""
    xgc = gdc['xgc']
    ygc = gdc['ygc']
    mgc = gdc['mgc']
    nx  = gdc['ndim_xgc']
    ny  = gdc['ndim_ygc']

    x = np.atleast_1d(np.asarray(x_raw, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y_raw, dtype=np.float64))

    valid = (x >= -1.0) & (x <= nx + 1.0) & (y >= -1.0) & (y <= ny + 1.0)

    ix = np.clip(np.floor(x).astype(int), 0, nx - 2)
    iy = np.clip(np.floor(y).astype(int), 0, ny - 2)
    fx = x - np.floor(x)
    fy = y - np.floor(y)

    def _bilin(arr):
        return ((1 - fx) * (1 - fy) * arr[iy,     ix    ] +
                (1 - fx) *      fy  * arr[iy + 1,  ix    ] +
                     fx  * (1 - fy) * arr[iy,      ix + 1] +
                     fx  *      fy  * arr[iy + 1,  ix + 1])

    x_corr = np.where(valid, _bilin(xgc), np.nan)
    y_corr = np.where(valid, _bilin(ygc), np.nan)

    ix0 = np.floor(x).astype(int)
    iy0 = np.floor(y).astype(int)
    valid_mgc = (ix0 >= 0) & (ix0 <= nx - 2) & (iy0 >= 0) & (iy0 <= ny - 2)
    mc = np.where(valid_mgc, _bilin(mgc), 0.0)

    return x_corr, y_corr, mc


def _gdc_jacobian_batch(x_raw, y_raw, gdc, step=0.1):
    """GDC Jacobian d(x_gdc,y_gdc)/d(x,y) via central differences. Shape (N,2,2)."""
    xp, yp, _ = _apply_gdc(x_raw + step, y_raw, gdc)
    xm, ym, _ = _apply_gdc(x_raw - step, y_raw, gdc)
    xu, yu, _ = _apply_gdc(x_raw, y_raw + step, gdc)
    xd, yd, _ = _apply_gdc(x_raw, y_raw - step, gdc)

    two_step = 2.0 * step
    N = len(np.atleast_1d(x_raw))
    J = np.zeros((N, 2, 2))
    J[:, 0, 0] = (xp - xm) / two_step
    J[:, 0, 1] = (xu - xd) / two_step
    J[:, 1, 0] = (yp - ym) / two_step
    J[:, 1, 1] = (yu - yd) / two_step
    return J


# ---------------------------------------------------------------------------
# GDC file lookup (filter candidates same logic as pypass find_gdc)
# ---------------------------------------------------------------------------

_FILTER_EQUIV = {
    'F350LP': ('F350LP', 'F350L'),
    'F850LP': ('F850LP', 'F850L'),
}

def _filter_candidates(filt):
    return _FILTER_EQUIV.get(filt, (filt,))


def _find_gdc(gdc_dir: Path, flc_path: Path):
    """Return path to best-matching WFC3UV STDGDC file, or None."""
    with fits.open(flc_path, memmap=False) as hdul:
        h = hdul[0].header
        filt = str(h.get('FILTER', '')).strip().upper()
        if not filt:
            filt = str(h.get('FILTER1', '')).strip().upper()
        if filt.startswith('CLEAR') or not filt:
            filt = str(h.get('FILTER2', '')).strip().upper()

    for f in _filter_candidates(filt):
        for name in (f'STDGDC_OFFICIAL_JFRAME_WFC3UV_{f}.fits',
                     f'STDGDC_WFC3UV_{f}.fits'):
            p = gdc_dir / name
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# Per-file fix
# ---------------------------------------------------------------------------

def fix_catalog(cat_path: Path, gdc_dir: Path, dry_run: bool) -> str:
    """
    Fix one *_flc_catalog.fits.  Returns a one-line status string.
    Raises on hard errors.
    """
    cat_path = Path(cat_path)
    img_dir  = cat_path.parent
    base     = cat_path.name.replace('_flc_catalog.fits', '')
    flc_path = img_dir / f"{base}_flc.fits"

    # --- Check if fix is needed ---
    with fits.open(cat_path, memmap=False) as hdul:
        hdr = hdul[1].header
        y_off_chip4 = hdr.get('CHIP4_Y_OFFSET', None)

    if y_off_chip4 is None:
        return f"SKIP  {cat_path.name}: no CHIP4_Y_OFFSET (not WFC3/UVIS or no chip4)"
    if abs(float(y_off_chip4) - 2048.0) < 0.01:
        return f"SKIP  {cat_path.name}: already y_offset=2048"
    if abs(float(y_off_chip4) - 2051.0) > 0.1:
        return f"WARN  {cat_path.name}: unexpected CHIP4_Y_OFFSET={y_off_chip4}, skipping"

    # --- Find GDC file ---
    if not flc_path.exists():
        return f"WARN  {cat_path.name}: FLC not found at {flc_path}, skipping"

    gdc_path = _find_gdc(gdc_dir, flc_path)
    if gdc_path is None:
        return f"WARN  {cat_path.name}: no GDC file found in {gdc_dir}, skipping"

    if dry_run:
        return f"WOULD {cat_path.name}: fix y_offset 2051→2048, GDC={gdc_path.name}"

    # --- Load GDC ---
    gdc = _load_stdgdc(gdc_path)

    # --- Load catalog ---
    with fits.open(cat_path, memmap=False) as hdul:
        hdr  = hdul[1].header.copy()
        data = hdul[1].data.copy()

    chip_ext_col = np.asarray(data['chip_ext'], int)
    mask4 = (chip_ext_col == 4)
    n4 = mask4.sum()

    if n4 == 0:
        return f"SKIP  {cat_path.name}: no chip-4 stars found"

    # --- Extract chip-4 arrays ---
    x_all   = np.asarray(data['x'],      float)
    y_all   = np.asarray(data['y'],      float)   # combined-frame (old: chip_local + 2051)
    mag_all      = np.asarray(data['mag'],      float)
    mag_st_all   = np.asarray(data['mag_st'],   float) if 'mag_st' in data.dtype.names else None
    cov_xx_all   = np.asarray(data['cov_xx'],   float)
    cov_yy_all   = np.asarray(data['cov_yy'],   float)
    cov_xy_all   = np.asarray(data['cov_xy'],   float)

    x4 = x_all[mask4]
    y4_old = y_all[mask4]
    y4_new = y4_old - 3.0          # chip_local + 2048 instead of chip_local + 2051

    cov_xx4 = cov_xx_all[mask4]
    cov_yy4 = cov_yy_all[mask4]
    cov_xy4 = cov_xy_all[mask4]

    # --- Recompute GDC at corrected y ---
    xgdc4, ygdc4, mc4 = _apply_gdc(x4, y4_new, gdc)

    # --- Recompute GDC Jacobian → covariances ---
    J4 = _gdc_jacobian_batch(x4, y4_new, gdc)
    # For each star: cov_gdc = J @ [[cov_xx, cov_xy],[cov_xy, cov_yy]] @ J.T
    cov_gdc_xx4 = np.full(n4, np.nan)
    cov_gdc_yy4 = np.full(n4, np.nan)
    cov_gdc_xy4 = np.full(n4, np.nan)
    valid4 = np.isfinite(xgdc4)
    for i in np.where(valid4)[0]:
        cov_raw = np.array([[cov_xx4[i], cov_xy4[i]],
                            [cov_xy4[i], cov_yy4[i]]])
        cov_g = J4[i] @ cov_raw @ J4[i].T
        cov_gdc_xx4[i] = cov_g[0, 0]
        cov_gdc_yy4[i] = cov_g[1, 1]
        cov_gdc_xy4[i] = cov_g[0, 1]

    # --- Recompute mag_gdc, mag_st_gdc ---
    mag_gdc4    = mag_all[mask4] + mc4
    mag_st_gdc4 = None
    if mag_st_all is not None:
        mag_st_gdc4 = np.where(np.isfinite(mag_st_all[mask4]),
                               mag_st_all[mask4] + mc4, np.nan)

    # --- Propagate to FITS columns ---
    # y (combined-frame): chip-4 stars drop by 3
    y_new_all = y_all.copy()
    y_new_all[mask4] = y4_new

    # x_gdc, y_gdc
    xgdc_new = np.asarray(data['x_gdc'], float).copy()
    ygdc_new = np.asarray(data['y_gdc'], float).copy()
    xgdc_new[mask4] = xgdc4
    ygdc_new[mask4] = ygdc4

    # cov_xx/yy/xy_gdc — preserve NaN for invalid stars (NaN xgdc)
    cov_xx_gdc_new = np.asarray(data['cov_xx_gdc'], float).copy()
    cov_yy_gdc_new = np.asarray(data['cov_yy_gdc'], float).copy()
    cov_xy_gdc_new = np.asarray(data['cov_xy_gdc'], float).copy()
    cov_xx_gdc_new[mask4] = cov_gdc_xx4
    cov_yy_gdc_new[mask4] = cov_gdc_yy4
    cov_xy_gdc_new[mask4] = cov_gdc_xy4

    # mag_gdc
    mag_gdc_new = np.asarray(data['mag_gdc'], float).copy()
    mag_gdc_new[mask4] = mag_gdc4

    # mag_st_gdc
    mag_st_gdc_new = None
    if mag_st_gdc4 is not None and 'mag_st_gdc' in data.dtype.names:
        mag_st_gdc_new = np.asarray(data['mag_st_gdc'], float).copy()
        mag_st_gdc_new[mask4] = mag_st_gdc4

    # --- Update header ---
    hdr['CHIP4_Y_OFFSET'] = 2048.0
    old_combined = hdr.get('CHIP4_CRPIX2_COMBINED', None)
    if old_combined is not None:
        hdr['CHIP4_CRPIX2_COMBINED'] = float(old_combined) - 3.0

    # Recompute CHIP4_CRPIX1_GDC, CHIP4_CRPIX2_GDC
    crpix1_0 = hdr.get('CHIP4_CRPIX1', 2048.0) - 1.0   # FITS 1-indexed → 0-indexed
    crpix2_0 = hdr.get('CHIP4_CRPIX2', 1026.0) - 1.0
    new_y_combined_crpix = crpix2_0 + 2048.0             # correct combined y for reference pixel
    rx, ry, _ = _apply_gdc(np.array([crpix1_0]),
                            np.array([new_y_combined_crpix]), gdc)
    if np.isfinite(rx[0]):
        hdr['CHIP4_CRPIX1_GDC'] = float(rx[0])
        hdr['CHIP4_CRPIX2_GDC'] = float(ry[0])

    # --- Write updated FITS in-place ---
    with fits.open(cat_path, mode='update', memmap=False) as hdul:
        hdul[1].header.update(hdr)
        hdul[1].data['y'][:] = y_new_all
        hdul[1].data['x_gdc'][:] = xgdc_new
        hdul[1].data['y_gdc'][:] = ygdc_new
        hdul[1].data['cov_xx_gdc'][:] = cov_xx_gdc_new
        hdul[1].data['cov_yy_gdc'][:] = cov_yy_gdc_new
        hdul[1].data['cov_xy_gdc'][:] = cov_xy_gdc_new
        hdul[1].data['mag_gdc'][:] = mag_gdc_new
        if mag_st_gdc_new is not None:
            hdul[1].data['mag_st_gdc'][:] = mag_st_gdc_new
        hdul.flush()

    # --- Delete downstream cross-match files so they are regenerated ---
    n_deleted = 0
    for pattern in ('matched_gaia.csv', 'xmatch*.json', 'xmatch_*.json'):
        for f in img_dir.glob(pattern):
            try:
                f.unlink()
                n_deleted += 1
            except Exception:
                pass

    n4_recovered = int(np.sum(np.isfinite(xgdc4) & ~np.isfinite(np.asarray(data['x_gdc'], float)[mask4])))
    n4_lost      = int(np.sum(~np.isfinite(xgdc4) & np.isfinite(np.asarray(data['x_gdc'], float)[mask4])))

    return (f"FIXED {cat_path.name}: {n4} chip-4 stars updated, "
            f"{n4_recovered} NaN→valid, {n4_lost} valid→NaN, "
            f"{n_deleted} xmatch file(s) deleted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done, do not modify files')
    parser.add_argument('--root', default=None,
                        help='Root directory to search (default: ~/data_bootes/bp3m/GaiaHub_results)')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Parallel worker processes (default: 4)')
    args = parser.parse_args()

    root = Path(args.root) if args.root else \
           Path.home() / 'data_bootes' / 'bp3m' / 'GaiaHub_results'
    if not root.exists():
        # Try bootes path
        root = Path('/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results')
    if not root.exists():
        sys.exit(f"Root directory not found: {root}")

    lib_dir = _get_lib_dir()
    gdc_dir = lib_dir / 'STDGDCs' / 'WFC3UV'
    if not gdc_dir.exists():
        sys.exit(f"WFC3UV GDC directory not found: {gdc_dir}")

    print(f"Root   : {root}")
    print(f"GDC dir: {gdc_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Collect only WFC3/UVIS catalogs (have CHIP4_Y_OFFSET)
    all_cats = sorted(root.rglob('*_flc_catalog.fits'))
    print(f"Found {len(all_cats)} total *_flc_catalog.fits files")

    # Pre-filter to those with CHIP4_Y_OFFSET=2051
    to_fix = []
    n_already_ok = 0
    n_no_chip4   = 0
    for p in all_cats:
        try:
            with fits.open(p, memmap=False) as h:
                yo = h[1].header.get('CHIP4_Y_OFFSET', None)
            if yo is None:
                n_no_chip4 += 1
            elif abs(float(yo) - 2048.0) < 0.01:
                n_already_ok += 1
            else:
                to_fix.append(p)
        except Exception:
            pass

    print(f"  {n_no_chip4} non-WFC3/UVIS (no CHIP4_Y_OFFSET) — skipped")
    print(f"  {n_already_ok} already y_offset=2048 — skipped")
    print(f"  {len(to_fix)} need fixing")
    print()

    if not to_fix:
        print("Nothing to do.")
        return

    if args.dry_run:
        for p in to_fix:
            print(f"WOULD fix: {p}")
        return

    results = []
    errors  = []

    if args.jobs <= 1:
        for p in to_fix:
            try:
                msg = fix_catalog(p, gdc_dir, dry_run=False)
                results.append(msg)
                print(msg)
            except Exception as exc:
                err = f"ERROR {p.name}: {exc}"
                errors.append(err)
                print(err)
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(fix_catalog, p, gdc_dir, False): p for p in to_fix}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    msg = fut.result()
                    results.append(msg)
                    print(msg)
                except Exception as exc:
                    err = f"ERROR {p.name}: {exc}"
                    errors.append(err)
                    print(err)

    print()
    n_fixed = sum(1 for r in results if r.startswith('FIXED'))
    print(f"Done: {n_fixed}/{len(to_fix)} fixed, {len(errors)} errors")
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
