"""Per-image GDC residual (label) table written by the individual-image fit (user 2026-10-01).

hst_dist_corr/ml/extract_labels.py used to rebuild this table for every image in a second pass,
re-opening the FLC (all SCI headers), the pypass catalogue, detections.npz,
stellar_astrometry.csv and image_transformations.csv over NFS (M31 alone stalled the file server).
The indv fit has all of it at hand, so it now writes the same rows to
<indv dir>/gdc_labels.csv.gz and the label build becomes a concatenation.

Rows/columns reproduce extract_labels.one_image (same 0.75 px catalogue match, same names), with
two differences: ALL Gaia stars are kept (column is_5p marks the 5-parameter ones the label build
used), and the image's jitter summary (jitter_summary.json, written at download) is flattened into
jif_* / jit_* columns.  Hold-out flags are left to the label build (they depend on holdouts.json).

Learned GDC correction provenance (user 2026-10-01): when the fit ran with a pos_corr_model the
residuals dx_gdc/dy_gdc are measured AFTER that correction.  Every row carries the model spec
(gdc_corr_model, '' when none) and the per-detection bias the loader subtracted
(gdc_corr_bx/by, GDC px; corrected = x_gdc - bias), so a later training round can learn either a
residual on top of the applied model or a fresh correction relative to the official GDC by adding
the bias back.  pos_corr_table (the older binned tables) is recorded by spec only.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_FILE = 'gdc_labels.csv.gz'
HDR_FILE = 'gdc_hdr.csv.gz'          # header-frame (model-independent) rows, bp3m.pipeline.header_frame
IMAGE_FILE = 'gdc_image.json'        # per-image record; written LAST, so it marks a complete export
LABEL_VERSION = 2
# every per-image header keyword the hst_dist_corr GDC chain reads (image_features HDR0/HDR1,
# extract_header_frame, chip_geometry, jitter_accumulate), so it never re-opens FLC headers
HDR0_ALL = ['MOONANGL', 'SUNANGLE', 'SUN_ALT', 'POSTARG1', 'POSTARG2', 'FLASHLVL', 'FLASHCUR', 'FGSLOCK', 'GYROMODE',
            'SUBARRAY', 'EXPFLAG', 'PCTECORR', 'APERTURE', 'CCDAMP', 'ATODGNA', 'ATODGNB', 'ATODGNC', 'ATODGND',
            'READNSEA', 'CCDOFSTA', 'BIASLEVA', 'BIASLEVB', 'BIASLEVC', 'BIASLEVD', 'DATE-OBS', 'TIME-OBS', 'EXPSTART',
            'EXPEND', 'EXPTIME', 'DARKTIME', 'PA_V3', 'RA_TARG', 'DEC_TARG', 'ASN_ID', 'OBSTYPE', 'PRIMESI', 'POSTNSTX',
            'POSTNSTY', 'POSTNSTZ', 'CCDGAIN', 'FLASHDUR', 'PCTEFRAC', 'PROPOSID', 'VAFACTOR', 'INSTRUME', 'DETECTOR',
            'FILTER', 'FILTER1', 'FILTER2', 'IMAGETYP', 'TARGNAME', 'ROOTNAME']
HDR1_ALL = ['MDRIZSKY', 'ORIENTAT', 'VAFACTOR', 'CCDCHIP', 'LTV1', 'LTV2', 'BINAXIS1', 'NAXIS1', 'NAXIS2', 'MEANDARK',
            'MEANBLEV', 'MEANFLSH', 'WCSNAME', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'IDCSCALE', 'CD1_1', 'CD1_2',
            'CD2_1', 'CD2_2']
HDR_KEYS = ['EXPSTART', 'EXPTIME', 'PA_V3', 'CCDGAIN', 'FLASHDUR', 'PCTEFRAC', 'SUN_ALT',
            'PROPOSID', 'SUBARRAY', 'POSTARG1', 'POSTARG2', 'VAFACTOR', 'ASN_ID', 'APERTURE']


def _jitter_columns(img_root: Path) -> dict:
    p = img_root / 'jitter_summary.json'
    out = {}
    try:
        j = json.loads(p.read_text())
    except Exception:
        return out
    for grp in ('jif', 'jit'):
        for k, v in (j.get(grp) or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f'{grp}_{k}'] = float(v)
    out['fgslock'] = j.get('fgslock')
    return out


@lru_cache(maxsize=4)
def _pos_corr(spec: str):
    from bp3m.pos_corr_basis import make_pos_corr
    return make_pos_corr(spec)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def export_is_current(output_dir) -> bool:
    """True when <output_dir> holds a complete export of the current LABEL_VERSION that postdates the fit."""
    out_dir = Path(output_dir); rec_p = out_dir / IMAGE_FILE; det = out_dir / 'detections.npz'
    try:
        rec = json.loads(rec_p.read_text())
        return (int(rec.get('label_version', 0)) == LABEL_VERSION
                and (not det.exists() or rec_p.stat().st_mtime >= det.stat().st_mtime))
    except Exception:
        return False


def export_gdc_labels(output_dir, field_name: str, data_root, image_name: str,
                      telescope: str = 'HST', pos_corr_model=None, pos_corr_table=None) -> int:
    """Write the GDC label export of a single-image fit into <output_dir>; returns the Stage-1 row count.

    Files: gdc_labels.csv.gz (Stage-1 residual rows of this fit), gdc_hdr.csv.gz (header-frame,
    model-independent rows), gdc_image.json (per-image record; written last).  Reads only files
    on disk, so it can (re)build the export of an image whose fit is cached and not redone."""
    from astropy.io import fits
    from scipy.spatial import cKDTree
    out_dir = Path(output_dir)
    image_name = str(image_name).replace('_hi', '').replace('_lo', '')
    img_root = Path(data_root) / field_name / telescope / 'mastDownload' / telescope / image_name
    flc = img_root / f'{image_name}_flc.fits'
    cat_p = img_root / f'{image_name}_flc_catalog.fits'
    if not (out_dir / 'detections.npz').exists() or not flc.exists() or not cat_p.exists():
        return 0
    try:
        n_rows, s1_err = _export_stage1(out_dir, field_name, image_name, img_root, flc, cat_p,
                                        pos_corr_model, pos_corr_table), None
    except Exception as exc:
        n_rows, s1_err = 0, f'{type(exc).__name__}: {exc}'
    if n_rows == 0 and (out_dir / LABEL_FILE).exists():
        (out_dir / LABEL_FILE).unlink()               # never leave rows of an earlier export behind
    _export_header_frame_and_record(out_dir, field_name, image_name, img_root, flc, cat_p,
                                    pos_corr_model, pos_corr_table, n_rows, s1_err)
    return n_rows


def _export_header_frame_and_record(out_dir, field_name, image_name, img_root, flc, cat_p,
                                    pos_corr_model, pos_corr_table, n_stage1, stage1_error=None):
    import hashlib
    from astropy.io import fits
    from bp3m.pipeline.header_frame import frame_geometry, header_frame_inputs, trusted_ids
    with fits.open(flc, memmap=True) as hd:
        ph = hd[0].header; h1 = hd[1].header.copy()
        sci = {}
        for e in range(1, len(hd)):
            if hd[e].header.get('EXTNAME') == 'SCI':
                sci[e] = {k: _jsonable(hd[e].header.get(k)) for k in ('CCDCHIP', 'MDRIZSKY', 'LTV1', 'LTV2', 'NAXIS1', 'NAXIS2')}
        h0 = {k.replace('-', '_'): _jsonable(ph.get(k)) for k in HDR0_ALL}
        ph = ph.copy()
    h1d = {k: _jsonable(h1.get(k)) for k in HDR1_ALL}
    with fits.open(cat_p, memmap=True) as hc:
        cat_hdr1 = hc[1].header.copy(); cat = hc[1].data
        geom = frame_geometry(ph, h1, cat_hdr1)
        n_hdr = 0; hdr_err = None
        try:
            mg = img_root / 'matched_gaia.csv'
            if mg.exists():
                matched = pd.read_csv(mg, dtype={'gaia_source_id': np.int64})
                sa = pd.read_csv(out_dir / 'stellar_astrometry.csv', dtype={'Gaia_id': np.int64}, low_memory=False)
                rows = header_frame_inputs(matched, sa, cat, geom, trusted_ids([out_dir]))
                if rows is not None and len(rows):
                    rows.insert(0, 'image', image_name); rows.insert(0, 'field', field_name)
                    rows.to_csv(out_dir / HDR_FILE, index=False); n_hdr = len(rows)
        except Exception as exc:
            hdr_err = f'{type(exc).__name__}: {exc}'
        gdc_id, gdc_file = cat_hdr1.get('GDC_ID'), cat_hdr1.get('GDC_FILE')
    if n_hdr == 0 and (out_dir / HDR_FILE).exists():
        (out_dir / HDR_FILE).unlink()                 # never leave rows of an earlier export behind
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    def _js(p):
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    xs = _js(img_root / 'xmatch_status.json'); rc = _js(out_dir / 'run_config.json')
    try:
        tr = pd.read_csv(img_root / 'transformation.csv').set_index('parameter')['value']
        tr = {k: _jsonable(v) for k, v in tr.items()}
    except Exception:
        tr = {}
    try:
        xf = pd.read_csv(out_dir / 'image_transformations.csv').to_dict(orient='records')
        xf = [{k: _jsonable(v) for k, v in r.items()} for r in xf]
    except Exception:
        xf = []
    geom_j = {k: (v if k == 'M' else _jsonable(v)) for k, v in geom.items()}
    rec = {'field': field_name, 'image': image_name, 'label_version': LABEL_VERSION,
           'n_stage1_rows': int(n_stage1), 'n_hdr_rows': int(n_hdr), 'stage1_error': stage1_error, 'hdr_error': hdr_err,
           'header_frame': geom_j, 'h0': h0, 'h1': h1d, 'sci_ext': {str(k): v for k, v in sci.items()},
           'jitter': {k: _jsonable(v) for k, v in _jitter_columns(img_root).items()},
           'jitter_present': (img_root / 'jitter_summary.json').exists(),
           'catalog_gdc_id': _jsonable(gdc_id), 'catalog_gdc_file': _jsonable(gdc_file),
           'matched_gaia_md5': md5(img_root / 'matched_gaia.csv'),
           'xmatch_status': _jsonable(xs.get('status')), 'xmatch_gdc_id': _jsonable((xs.get('params') or {}).get('gdc_id')),
           'indv_gdc_id': _jsonable(rc.get('gdc_id')), 'indv_matched_gaia_md5': _jsonable(rc.get('matched_gaia_md5')),
           'indv_fit_version': _jsonable(rc.get('indv_fit_version')),
           'gdc_corr_model': str(pos_corr_model) if pos_corr_model else '',
           'gdc_corr_table': str(pos_corr_table) if pos_corr_table else '',
           'pos_err_floor': _jsonable(rc.get('pos_err_floor')),
           'transformation_csv': tr, 'image_transformations': xf}
    tmp = out_dir / (IMAGE_FILE + '.tmp')
    tmp.write_text(json.dumps(rec, indent=1)); tmp.replace(out_dir / IMAGE_FILE)


def _export_stage1(out_dir, field_name, image_name, img_root, flc, cat_p, pos_corr_model, pos_corr_table) -> int:
    from astropy.io import fits
    from scipy.spatial import cKDTree
    det = np.load(out_dir / 'detections.npz', allow_pickle=True)
    keys = [k for k in det.files if k.endswith('_sidx')]
    if not keys:
        return 0
    sa = pd.read_csv(out_dir / 'stellar_astrometry.csv', dtype={'Gaia_id': np.int64},
                     usecols=lambda c: c in ('Gaia_id', 'pmra', 'pmdec', 'parallax', 'pmra_error', 'pmdec_error',
                                             'parallax_error', 'gmag', 'bp_rp', 'ruwe', 'Gaia_time'))
    xf = pd.read_csv(out_dir / 'image_transformations.csv')
    ph = fits.getheader(flc, 0)
    hdr = {k.lower(): ph.get(k) for k in HDR_KEYS}
    filt = ph.get('FILTER') or (ph.get('FILTER1') if str(ph.get('FILTER1', '')).startswith('F') else ph.get('FILTER2'))
    inst, detc = ph.get('INSTRUME'), ph.get('DETECTOR')
    sky_ext = {}
    with fits.open(flc, memmap=True) as hd:
        for e in range(1, len(hd)):
            if hd[e].header.get('EXTNAME') == 'SCI':
                sky_ext[e] = float(hd[e].header.get('MDRIZSKY', np.nan))
    cat = fits.getdata(cat_p, 1)
    cat_xy = np.column_stack([cat['x_gdc'].astype(float), cat['y_gdc'].astype(float)])
    fin = np.isfinite(cat_xy).all(axis=1)
    cat = cat[fin]; cat_xy = cat_xy[fin]
    if len(cat) == 0:
        return 0
    tree = cKDTree(cat_xy)
    names = cat.dtype.names
    pcm, pc_hdr = None, None
    if pos_corr_model:
        pcm = _pos_corr(str(pos_corr_model)); pc_hdr = pcm.read_header(flc)
    rows = []
    for k in keys:
        sub = k[:-5]
        row_xf = xf[xf.image_name.astype(str) == sub]
        if not len(row_xf):
            continue
        r = row_xf.iloc[0]
        sidx = det[f'{sub}_sidx'].astype(int)
        Xc, Yc = det[f'{sub}_X_c'], det[f'{sub}_Y_c']
        dx, dy = det[f'{sub}_dx_gdc'], det[f'{sub}_dy_gdc']
        Ch, Ct = det[f'{sub}_C_hst'], det[f'{sub}_C_gdc_total']
        uff = det[f'{sub}_use_for_fit'].astype(bool)
        ufa = det[f'{sub}_use_for_astrom'].astype(bool) if f'{sub}_use_for_astrom' in det.files else uff
        Xg, Yg = Xc + float(r['Xo_pivot']), Yc + float(r['Yo_pivot'])
        d_, j_ = tree.query(np.column_stack([Xg, Yg]), distance_upper_bound=0.75)
        keep = np.isfinite(d_)
        if not keep.any():
            continue
        s = sa.iloc[sidx]
        p5 = (np.isfinite(s.pmra.to_numpy()) & np.isfinite(s.pmdec.to_numpy())
              & np.isfinite(s.parallax.to_numpy()))
        c = cat[j_[keep]]
        chip = c['chip_ext'].astype(int) if 'chip_ext' in names else np.where(c['y'] >= 2048, 4, 1)
        g = lambda col: s[col].to_numpy()[keep]
        cc = lambda col, dt=float: (c[col].astype(dt) if col in names else np.full(int(keep.sum()), np.nan))
        bxy = pcm.bias(c, pc_hdr) if pcm is not None else None
        if bxy is None:
            bxy = (np.zeros(int(keep.sum())), np.zeros(int(keep.sum())))
        rows.append(pd.DataFrame({
            'field': field_name, 'image': image_name, 'sub_image': sub, 'gaia_id': g('Gaia_id'),
            'is_5p': p5[keep],
            'dx_gdc': dx[keep], 'dy_gdc': dy[keep],
            'chst_xx': Ch[keep, 0, 0], 'chst_yy': Ch[keep, 1, 1], 'chst_xy': Ch[keep, 0, 1],
            'ctot_xx': Ct[keep, 0, 0], 'ctot_yy': Ct[keep, 1, 1], 'ctot_xy': Ct[keep, 0, 1],
            'use_for_fit': uff[keep], 'use_for_astrom': ufa[keep], 'X_c': Xc[keep], 'Y_c': Yc[keep],
            'x': cc('x'), 'y': cc('y'), 'chip_ext': chip, 'x_gdc': cc('x_gdc'), 'y_gdc': cc('y_gdc'),
            'flux': cc('flux'), 'sky': cc('sky'), 'qfit': cc('qfit'), 'chi2': cc('chi2'),
            'psf_frac': cc('psf_frac'), 'n_sat': cc('n_sat'), 'n_neighbors': cc('n_neighbors'),
            'dist_nearest': cc('dist_nearest'), 'dist_nearest_brighter': cc('dist_nearest_brighter'),
            'mag_inst': cc('mag'), 'concentration': cc('concentration'), 'cr_recovered': cc('cr_recovered'),
            'dq_3x3': cc('dq_3x3'),
            'gmag': g('gmag'), 'bp_rp': g('bp_rp'), 'ruwe': g('ruwe'), 'parallax': g('parallax'),
            'pmra': g('pmra'), 'pmdec': g('pmdec'), 'pmra_error': g('pmra_error'), 'pmdec_error': g('pmdec_error'),
            'mdrizsky': [sky_ext.get(int(e), np.nan) for e in chip],
            'a': r['a'], 'b': r['b'], 'c_': r['c'], 'd': r['d'], 'alpha': r.get('alpha', 1.0),
            'n_align': r['n_stars_alignment'], 'match_dist': d_[keep],
            'gdc_corr_bx': np.asarray(bxy[0], float), 'gdc_corr_by': np.asarray(bxy[1], float),
        }))
    if not rows:
        return 0
    df = pd.concat(rows, ignore_index=True)
    const = {**hdr, **_jitter_columns(img_root)}
    df = pd.concat([df, pd.DataFrame({k: [v] * len(df) for k, v in const.items()})], axis=1)
    df['mjd'] = float(ph['EXPSTART']); df['filter'] = filt; df['instrument'] = inst; df['detector'] = detc
    df['year'] = 2000.0 + (df['mjd'] - 51544.5) / 365.25
    df['phase_x'] = np.mod(df['x'], 1.0); df['phase_y'] = np.mod(df['y'], 1.0)
    df['y_chip'] = np.where(df['chip_ext'] == 4, df['y'] - (2051.0 if inst == 'WFC3' else 2048.0), df['y'])
    df['n_sat'] = df['n_sat'].fillna(0).astype(int)
    df['gdc_corr_model'] = str(pos_corr_model) if pos_corr_model else ''
    df['gdc_corr_dir'] = str(getattr(pcm, 'dir', '')) if pcm is not None else ''
    df['gdc_corr_table'] = str(pos_corr_table) if pos_corr_table else ''
    df['label_version'] = LABEL_VERSION
    df = df.copy()
    df.to_csv(out_dir / LABEL_FILE, index=False)
    return len(df)


def _export_one(args):
    """Pool worker: (re)export one indv result dir with the settings ITS fit used (run_config.json)."""
    indv_dir, field_name, data_root = args
    d = Path(indv_dir)
    try:
        rc = json.loads((d / 'run_config.json').read_text())
        n = export_gdc_labels(d, field_name, data_root, d.name,
                              pos_corr_model=rc.get('pos_corr_model'), pos_corr_table=rc.get('pos_corr_table'))
        return d.name, n, None
    except Exception as exc:
        return d.name, 0, f'{type(exc).__name__}: {exc}'


def stale_exports(field_name: str, data_root, images=None, indv_root=None) -> list:
    """Indv result dirs of a field that hold a fit (detections.npz + run_config.json) but no current export."""
    root = Path(indv_root) if indv_root else Path(data_root) / field_name / 'BP3M_indv_results'
    if not root.is_dir():
        return []
    dirs = [root / i for i in images] if images is not None else sorted(p for p in root.iterdir() if p.is_dir())
    return [d for d in dirs if (d / 'detections.npz').exists() and (d / 'run_config.json').exists()
            and not export_is_current(d)]


def export_field(field_name: str, data_root, images=None, workers: int = 4, log=print, indv_root=None) -> dict:
    """(Re)build the GDC export of every indv fit of a field that lacks a current one, WITHOUT refitting
    (2026-10-02: images whose fit is cached must still contribute up-to-date label rows)."""
    import time
    from concurrent.futures import ProcessPoolExecutor
    todo = stale_exports(field_name, data_root, images, indv_root)
    out = {'field': field_name, 'stale': len(todo), 'ok': 0, 'failed': 0, 'rows': 0}
    if not todo:
        log(f"  GDC export: all indv results of {field_name} current (label_version {LABEL_VERSION})")
        return out
    log(f"  GDC export: rebuilding {len(todo)} indv result(s) of {field_name} without refitting "
        f"(label_version {LABEL_VERSION}, {workers} workers)")
    t0 = time.time(); args = [(str(d), field_name, str(data_root)) for d in todo]
    def _run(it):
        for k, (img, n, err) in enumerate(it, 1):
            if err:
                out['failed'] += 1
                log(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] {img} ({k}/{len(todo)}) export FAILED: {err}")
            else:
                out['ok'] += 1; out['rows'] += n
            if k % 100 == 0 or k == len(todo):
                log(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] GDC export {k}/{len(todo)}: {out['ok']} ok, "
                    f"{out['failed']} failed, {out['rows']} rows ({time.time() - t0:.0f}s)")
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=mp.get_context('forkserver')) as ex:
            _run(ex.map(_export_one, args, chunksize=4))
    else:
        _run(map(_export_one, args))
    return out


FIELD_FILES = {'stage1': 'gdc_labels_field.csv.gz', 'hdr': 'gdc_hdr_field.csv.gz', 'images': 'gdc_images_field.csv.gz'}
MANIFEST = 'gdc_export_manifest.json'


def flatten_record(rec: dict) -> dict:
    """One gdc_image.json record -> one flat row (h0_*, h1_*, hf_*, sci<ext>_*, tr_*, jitter, provenance)."""
    row = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
    for k, v in (rec.get('h0') or {}).items(): row[f'h0_{k}'] = v
    for k, v in (rec.get('h1') or {}).items(): row[f'h1_{k}'] = v
    for k, v in (rec.get('header_frame') or {}).items():
        if k == 'M':
            (row['hf_M11'], row['hf_M12']), (row['hf_M21'], row['hf_M22']) = v
        else:
            row[f'hf_{k}'] = v
    for e, d in (rec.get('sci_ext') or {}).items():
        for k, v in d.items(): row[f'sci{e}_{k}'] = v
    for k, v in (rec.get('transformation_csv') or {}).items(): row[f'tr_{k}'] = v
    row.update(rec.get('jitter') or {})
    row['image_transformations_json'] = json.dumps(rec.get('image_transformations') or [])
    return row


def consolidate_field(field_name: str, data_root, indv_root=None, log=print) -> dict:
    """Gather every CURRENT per-image export of a field into one file per table (+ manifest), so the GDC
    work reads three files per field instead of three per image (2026-10-02).

    Incremental: images whose gdc_image.json is unchanged since the last consolidation are carried over
    from the existing field files; changed/new images are re-read; images without a current export
    (stale, failed fit, removed) are dropped and listed in the manifest.  in_selection marks images in
    the field's current MAST selection ({field}_selected_obsids.json)."""
    import time
    root = Path(indv_root) if indv_root else Path(data_root) / field_name / 'BP3M_indv_results'
    if not root.is_dir():
        return {}
    t0 = time.time()
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    cur, stale = {}, []
    for d in dirs:
        rp = d / IMAGE_FILE
        if (d / 'detections.npz').exists() and export_is_current(d):
            cur[d.name] = rp.stat().st_mtime
        elif (d / 'detections.npz').exists():
            stale.append(d.name)
    try:
        sel = set(json.loads((Path(data_root) / field_name / 'HST' / f'{field_name}_selected_obsids.json').read_text()))
    except Exception:
        sel = None
    old_man = {}
    try:
        old_man = json.loads((root / MANIFEST).read_text())
        if int(old_man.get('label_version', 0)) != LABEL_VERSION:
            old_man = {}
    except Exception:
        old_man = {}
    keep = {i for i, t in cur.items() if old_man.get('images', {}).get(i) == t}
    if keep and not all((root / f).exists() for f in FIELD_FILES.values()):
        keep = set()
    redo = [i for i in cur if i not in keep]
    tables = {}
    for key, fname in FIELD_FILES.items():
        parts = []
        if keep:
            old = pd.read_csv(root / fname, dtype={'gaia_id': np.int64} if key != 'images' else None, low_memory=False)
            parts.append(old[old['image'].astype(str).isin(keep)])
        for i in redo:
            d = root / i
            if key == 'images':
                parts.append(pd.DataFrame([flatten_record(json.loads((d / IMAGE_FILE).read_text()))]))
            else:
                f = d / (LABEL_FILE if key == 'stage1' else HDR_FILE)
                if f.exists():
                    parts.append(pd.read_csv(f, dtype={'gaia_id': np.int64}, low_memory=False))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(df) and sel is not None:
            df['in_selection'] = df['image'].astype(str).isin(sel)
        tables[key] = df
        tmp = root / (fname + '.tmp.gz')
        df.to_csv(tmp, index=False); tmp.replace(root / fname)
    man = {'field': field_name, 'label_version': LABEL_VERSION, 'built': time.strftime('%Y-%m-%d %H:%M:%S'),
           'images': cur, 'stale_or_missing_export': stale,
           'n_rows': {k: int(len(v)) for k, v in tables.items()}, 'files': FIELD_FILES}
    tmp = root / (MANIFEST + '.tmp'); tmp.write_text(json.dumps(man, indent=1)); tmp.replace(root / MANIFEST)
    log(f"  GDC export consolidated: {field_name} {len(cur)} images ({len(redo)} re-read, {len(keep)} carried over"
        f"{', ' + str(len(stale)) + ' without a current export' if stale else ''}); rows stage-1 "
        f"{man['n_rows']['stage1']}, header-frame {man['n_rows']['hdr']} -> {root} ({time.time() - t0:.0f}s)")
    return man


def main():
    import argparse, os
    ap = argparse.ArgumentParser(description='(Re)build the per-image GDC label export of finished indv fits '
                                             'without refitting (bp3m.pipeline.residual_export).')
    ap.add_argument('--name', nargs='+', required=True, help='field name(s)')
    ap.add_argument('--output_dir', default=os.getcwd(), help='GaiaHub_results root (default: cwd)')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    for f in a.name:
        r = export_field(f, a.output_dir, workers=a.workers)
        print(f"{f}: {r['stale']} stale, {r['ok']} exported, {r['failed']} failed, {r['rows']} stage-1 rows", flush=True)
        consolidate_field(f, a.output_dir)


if __name__ == '__main__':
    main()
