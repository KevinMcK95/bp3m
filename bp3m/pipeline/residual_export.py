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
LABEL_VERSION = 1
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


def export_gdc_labels(output_dir, field_name: str, data_root, image_name: str,
                      telescope: str = 'HST', pos_corr_model=None, pos_corr_table=None) -> int:
    """Write <output_dir>/gdc_labels.csv.gz for a single-image fit; returns the number of rows."""
    from astropy.io import fits
    from scipy.spatial import cKDTree
    out_dir = Path(output_dir)
    image_name = str(image_name).replace('_hi', '').replace('_lo', '')
    img_root = Path(data_root) / field_name / telescope / 'mastDownload' / telescope / image_name
    flc = img_root / f'{image_name}_flc.fits'
    cat_p = img_root / f'{image_name}_flc_catalog.fits'
    if not (out_dir / 'detections.npz').exists() or not flc.exists() or not cat_p.exists():
        return 0
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
