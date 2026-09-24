#!/usr/bin/env python3
"""Re-apply the geometric distortion correction to existing pypass catalogues.

Why: until 2026-09-24 the ACS/WFC tables in the bp3m library were Anderson's
VINTAGE_2005 solution (bp3m-setup downloaded that subdirectory), not the
OFFICIAL_JFRAME tables.  The PSF fit itself never uses the GDC, so nothing has
to be refitted: every GDC-derived column can be recomputed from the raw (x, y)
already stored in the catalogue.

What is recomputed (from x, y and the catalogue's own floors / chi2 scaling):
  x_gdc, y_gdc                      forward map of the new table
  jac_??_gdc                        d(x_gdc, y_gdc)/d(x, y) (central differences)
  cov_xx_gdc, cov_yy_gdc, cov_xy_gdc   J C J^T + floor^2, where C is the stored
                                    chi2-inflated (x, y) covariance with the
                                    stored floor removed from the diagonal —
                                    exactly what pypass.io.catalog_to_table does
  mag_gdc, mag_st_gdc               mag + pixel-area term (identical between the
                                    two ACS tables, recomputed for completeness)
  CHIPn_CRPIX1_GDC / CRPIX2_GDC     reference pixel in the corrected frame
  GDC_FILE, GDC_ID, GDC_FIXED       provenance
Not touched: x, y, fluxes, mags, cov_xx/yy/xy, ra/dec (+cov, from the FITS WCS
on raw x, y), everything PSF-related, psf_params.json (the fit cache stays valid).

The catalogue is rewritten IN PLACE (same inode) so hard-linked copies of the
same image in other field directories are fixed at the same time.  The image's
cross-match products (matched_gaia.csv, xmatch_params.json, xmatch_status.json)
are removed unless --keep_xmatch, so the next bp3m run redoes the cross-match
and the indv fit in the new frame.

Idempotent: a catalogue whose GDC_ID already equals the id of the table the
library now provides is skipped (unless --force).

Usage
  bp3m-fix-gdc --field Leo_I [--field ...]        fix every image of a field
  bp3m-fix-gdc --catalog PATH [...]               explicit catalogue files
  bp3m-fix-gdc --all                              every field under --gh_root
  --lib_dir DIR      library to take the tables from (default: bp3m config)
  --out DIR          write fixed copies here instead of in place (validation)
  --dry_run          report what would change, write nothing
  --n_processes N    parallel workers (default 4; I/O bound, keep small)
"""
from __future__ import annotations

import argparse, json, os, sys, time, shutil, tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

GH_ROOT = '/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results'
XMATCH_FILES = ('matched_gaia.csv', 'xmatch_params.json', 'xmatch_status.json')
_GDC_CACHE = {}


def _default_lib_dir():
    try:
        from bp3m_run import _config_lib_dir
        d = _config_lib_dir()
        if d:
            return d
    except Exception:
        pass
    return None


def _load_gdc(path, keep=2):
    """Per-process cache of loaded tables (each ~400 MB as float64), capped."""
    from pypass.io import load_stdgdc, gdc_file_id
    if path not in _GDC_CACHE:
        while len(_GDC_CACHE) >= keep:
            _GDC_CACHE.pop(next(iter(_GDC_CACHE)))
        _GDC_CACHE[path] = (load_stdgdc(path), gdc_file_id(path))
    return _GDC_CACHE[path]


def fix_catalog(cat_path, lib_dir, out_dir=None, dry_run=False, force=False,
                keep_xmatch=False):
    """Fix one catalogue.  Returns (status, message)."""
    from astropy.io import fits
    from astropy.table import Table
    from pypass.io import (apply_gdc, _gdc_jacobian_batch, find_gdc,
                           _DETECTOR_PREFIX)

    cat_path = Path(cat_path)
    img_dir = cat_path.parent
    stem = cat_path.name.replace('_catalog.fits', '')
    flc = img_dir / f'{stem}.fits'
    if not flc.exists():
        return 'error', f'no FLC next to {cat_path.name}'
    hdr = fits.getheader(str(flc), 0)
    instrume = hdr.get('INSTRUME', '').strip().upper()
    detector = hdr.get('DETECTOR', '').strip().upper()
    det_prefix = _DETECTOR_PREFIX.get((instrume, detector))
    if det_prefix is None:
        return 'skip', f'{instrume}/{detector}: no GDC support'
    gdc_dir = Path(lib_dir) / 'STDGDCs' / det_prefix
    gdc_path = find_gdc(str(gdc_dir), hdr) if gdc_dir.is_dir() else None
    if not gdc_path:
        return 'error', f'no GDC table in {gdc_dir} for this filter'
    gdc, gdc_id = _load_gdc(gdc_path)

    t = Table.read(str(cat_path))
    if not force and t.meta.get('GDC_ID') == gdc_id:
        return 'skip', f'already {os.path.basename(gdc_path)} ({gdc_id})'

    x = np.asarray(t['x'], float); y = np.asarray(t['y'], float)   # combined frame
    fx2 = float(t.meta.get('SIGMA_FLOOR_X', 0.0)) ** 2
    fy2 = float(t.meta.get('SIGMA_FLOOR_Y', 0.0)) ** 2
    old_x = np.asarray(t['x_gdc'], float); old_y = np.asarray(t['y_gdc'], float)

    x_gdc, y_gdc, mc = apply_gdc(x, y, gdc)
    J = _gdc_jacobian_batch(x, y, gdc)
    cxx = np.asarray(t['cov_xx'], float) - fx2
    cyy = np.asarray(t['cov_yy'], float) - fy2
    cxy = np.asarray(t['cov_xy'], float)
    # J C J^T, elementwise for N 2x2 matrices
    a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    gxx = a * a * cxx + 2 * a * b * cxy + b * b * cyy
    gyy = c * c * cxx + 2 * c * d * cxy + d * d * cyy
    gxy = a * c * cxx + (a * d + b * c) * cxy + b * d * cyy
    ok = np.isfinite(x_gdc) & np.isfinite(y_gdc) & np.isfinite(gxx) & np.isfinite(gyy)

    t['x_gdc'] = x_gdc; t['y_gdc'] = y_gdc
    t['cov_xx_gdc'] = np.where(ok, gxx + fx2, np.nan)
    t['cov_yy_gdc'] = np.where(ok, gyy + fy2, np.nan)
    t['cov_xy_gdc'] = np.where(ok, gxy, np.nan)
    for name, arr in (('jac_xx_gdc', a), ('jac_xy_gdc', b), ('jac_yx_gdc', c), ('jac_yy_gdc', d)):
        t[name] = np.where(ok, arr, np.nan)
    t['mag_gdc'] = np.asarray(t['mag'], float) + mc
    if 'mag_st_gdc' in t.colnames and 'mag_st' in t.colnames:
        t['mag_st_gdc'] = np.asarray(t['mag_st'], float) + mc
    for col in ('x_gdc', 'y_gdc'):
        t[col].unit = 'pix'
    for col in ('cov_xx_gdc', 'cov_yy_gdc', 'cov_xy_gdc'):
        t[col].unit = 'pix2'

    # reference pixels in the corrected frame (per chip)
    for k in list(t.meta):
        if k.startswith('CHIP') and k.endswith('_CRPIX1_GDC'):
            pfx = k[:-len('_CRPIX1_GDC')]
            cx = t.meta.get(f'{pfx}_CRPIX1'); cy = t.meta.get(f'{pfx}_CRPIX2')
            yoff = t.meta.get(f'{pfx}_Y_OFFSET', 0.0)
            if cx is not None and cy is not None:
                rx, ry, _ = apply_gdc(cx - 1.0, cy - 1.0 + yoff, gdc)
                t.meta[f'{pfx}_CRPIX1_GDC'] = float(rx[0])
                t.meta[f'{pfx}_CRPIX2_GDC'] = float(ry[0])
    old_file = t.meta.get('GDC_FILE', 'unknown')
    t.meta['GDC_FILE'] = os.path.basename(gdc_path)
    t.meta['GDC_ID'] = gdc_id
    t.meta['GDC_FIXED'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    t.meta['GDC_PREV'] = old_file

    dr = np.hypot(x_gdc - old_x, y_gdc - old_y)
    msg = (f'{old_file} -> {os.path.basename(gdc_path)}: {len(t)} rows, '
           f'|dpos| median {np.nanmedian(dr):.3f} max {np.nanmax(dr):.3f} px')
    if dry_run:
        return 'dry', msg

    if out_dir is not None:
        out = Path(out_dir) / cat_path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        t.write(str(out), overwrite=True)
        return 'fixed', msg + f' -> {out}'

    # in place, same inode (hard-linked copies in other fields see the fix too)
    fd, tmp = tempfile.mkstemp(dir=str(img_dir), prefix=f'.{cat_path.name}.', suffix='.tmp')
    os.close(fd)
    t.write(tmp, overwrite=True)
    with open(tmp, 'rb') as src, open(str(cat_path), 'r+b') as dst:
        dst.truncate(0)
        shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
        dst.flush(); os.fsync(dst.fileno())
    os.unlink(tmp)
    if not keep_xmatch:
        for f in XMATCH_FILES:
            p = img_dir / f
            if p.exists():
                p.unlink()
    return 'fixed', msg


def _worker(args):
    cat, lib_dir, out_dir, dry_run, force, keep_xmatch = args
    t0 = time.time()
    try:
        st, msg = fix_catalog(cat, lib_dir, out_dir, dry_run, force, keep_xmatch)
    except Exception as e:
        st, msg = 'error', f'{type(e).__name__}: {e}'
    return str(cat), st, msg, time.time() - t0


def find_catalogs(fields=None, all_fields=False, gh_root=GH_ROOT):
    root = Path(gh_root)
    if all_fields:
        with os.scandir(root) as it:
            fields = sorted(e.name for e in it if e.is_dir() and '_fs_' not in e.name
                            and not e.name.startswith('Leo_I_scheme'))
    cats = []
    for f in fields or []:
        base = root / f / 'HST' / 'mastDownload' / 'HST'
        if not base.is_dir():
            continue
        with os.scandir(base) as it:
            for e in it:
                if e.is_dir():
                    c = Path(e.path) / f'{e.name}_flc_catalog.fits'
                    if c.exists():
                        cats.append(c)
    return cats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', action='append', default=[])
    ap.add_argument('--field', action='append', default=[])
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--gh_root', default=GH_ROOT)
    ap.add_argument('--lib_dir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--keep_xmatch', action='store_true')
    ap.add_argument('--n_processes', type=int, default=4)
    ap.add_argument('--log', default=None, help='append per-catalogue results here')
    a = ap.parse_args(argv)

    lib_dir = a.lib_dir or _default_lib_dir()
    if not lib_dir:
        sys.exit('no --lib_dir and no bp3m config lib_dir')
    cats = [Path(c) for c in a.catalog] + find_catalogs(a.field, a.all, a.gh_root)
    # dedupe hard links: one job per inode
    seen, uniq, n_links = set(), [], 0
    for c in cats:
        try:
            st = os.stat(c)
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            n_links += 1
            continue
        seen.add(key); uniq.append(c)
    print(f'[{time.strftime("%m-%d %H:%M")}] {len(uniq)} catalogues ({n_links} hard-linked duplicates '
          f'skipped), lib_dir={lib_dir}, {"DRY RUN" if a.dry_run else "in place" if not a.out else "-> "+a.out}',
          flush=True)
    jobs = [(c, lib_dir, a.out, a.dry_run, a.force, a.keep_xmatch) for c in uniq]
    counts = {}
    log = open(a.log, 'a') if a.log else None
    t0 = time.time()
    if a.n_processes > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        it = Pool(a.n_processes).imap_unordered(_worker, jobs)
    else:
        it = map(_worker, jobs)
    for i, (cat, st, msg, dt) in enumerate(it, 1):
        counts[st] = counts.get(st, 0) + 1
        line = f'[{time.strftime("%m-%d %H:%M")}] {i}/{len(jobs)} {st:5s} {Path(cat).parent.name}: {msg} ({dt:.1f}s)'
        if log:
            log.write(line + '\n'); log.flush()
        if st in ('error',) or i % 50 == 0 or len(jobs) <= 20:
            print(line, flush=True)
    print(f'[{time.strftime("%m-%d %H:%M")}] done in {time.time()-t0:.0f}s: {counts}', flush=True)
    if log:
        log.close()


if __name__ == '__main__':
    main()
