"""Visit-group pre-alignment for bp3m-v2 (user 2026-10-02): Gaia-free, Sohn-style sibling alignment.

Deep exposures can fail the per-image Gaia cross-match (Leo P: every Gaia star saturated in the
F814W 2014 frames) and are then absent from v1 and from v2 altogether, although a sibling taken
minutes earlier in the same visit (often another filter at the same pointing) was matched fine.

For every selected image WITHOUT a successful Gaia match:
  1. its visit group (bp3m.visit_groups via xmatch_groups.light_groups: same proposal / visit /
     guide stars / roll, contiguous, same WCSNAME) supplies ANCHORS = siblings with a successful
     match (transformation.csv), preferring the ones v1 actually aligned;
  2. initial guess: the target's own header WCS corrected by the anchor's measured header error
     (gaia_cross_match.sibling_seed: within a visit the error agrees to ~1 mas / 1 arcsec);
  3. bright, unsaturated, non-CR-like HST stars of the target are matched to the anchor's stars on
     the sky (the anchor's fitted mapping) and a 6p affine is fitted with sigma clipping
     (radius 5 -> 1.5 -> 0.6 px);
  4. accepted solutions (>= MIN_MATCH stars, rms < MAX_RMS_PX) are written as
       <image>/transformation_prealign.csv   (transformation.csv schema, target header frame)
     and as v1-style rows + covariance blocks in <field>/hst_xmatch/prealign/
     (image_transformations.csv, C_r.npy), which the master cross-match and the v2 solve read
     for images that have no v1 solution.  With --hst_align the shared HST-only stars then couple
     the siblings inside the joint v2 fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PREALIGN_VERSION = 1
MIN_MATCH = 15
MAX_RMS_PX = 0.15
RADII_PX = (5.0, 1.5, 0.6)
N_BRIGHT = 400


def _good_stars(cat, n=N_BRIGHT):
    names = cat.dtype.names
    ok = np.isfinite(cat['x_gdc']) & np.isfinite(cat['y_gdc']) & np.isfinite(cat['flux']) & (cat['flux'] > 0)
    if 'n_sat' in names: ok &= cat['n_sat'] == 0
    if 'qfit' in names: ok &= cat['qfit'] < 0.15
    if 'concentration' in names:
        conc = np.asarray(cat['concentration'], float)
        ok &= ~(np.isfinite(conc) & (conc > 1.05))
    if 'cr_recovered' in names: ok &= ~np.asarray(cat['cr_recovered'], bool)
    idx = np.where(ok)[0]
    return idx[np.argsort(-np.asarray(cat['flux'])[idx])][:n]


def _fit_affine(h, g, w=None):
    """g = M (h - c) + t, c = mean(h); returns (M, t, c, resid, cov6) with cov over (m11,m12,t1,m21,m22,t2)."""
    c = h.mean(axis=0); X = np.column_stack([h - c, np.ones(len(h))])
    px, *_ = np.linalg.lstsq(X, g[:, 0], rcond=None); py, *_ = np.linalg.lstsq(X, g[:, 1], rcond=None)
    M = np.array([[px[0], px[1]], [py[0], py[1]]]); t = np.array([px[2], py[2]])
    res = g - (X @ np.column_stack([px, py]))
    s2 = float(np.mean(res ** 2)) if len(res) > 3 else 1.0
    XtXi = np.linalg.inv(X.T @ X) * s2
    cov6 = np.zeros((6, 6)); cov6[:3, :3] = XtXi; cov6[3:, 3:] = XtXi
    return M, t, c, res, cov6


def _sky_of_anchor(anchor, idx):
    """Sky positions of anchor catalog rows through its fitted cross-match solution."""
    from gaia_cross_match import sibling_seed as ss
    from bp3m.pipeline.xmatch_groups import affine_from_trans
    tr = ss.read_transformation(anchor.root / 'transformation.csv')
    fr = {k: tr[k] for k in ('ra_cen', 'dec_cen', 'x_cen', 'y_cen', 'pixel_scale', 'orientat')}
    M, t = affine_from_trans(tr, [tr['x_cen'], tr['y_cen']])
    h = np.column_stack([anchor.cat['x_gdc'][idx], anchor.cat['y_gdc'][idx]]).astype(float)
    g = (h - np.array([tr['x_cen'], tr['y_cen']])) @ M.T + t
    return frame_to_sky_vec(g, fr)


def frame_to_sky_vec(g, fr):
    from gaia_cross_match import sibling_seed as ss
    return ss.frame_to_sky(g[:, 0], g[:, 1], fr)


def _v1_row(img, fr, M, t, c, cov6, xr, yr, anchor_block):
    """v1-style image_transformations row (+6x6 covariance) for pixel h -> Gaia-frame g = M (h-c) + t."""
    from bp3m.astro_utils import plane_project
    from gaia_cross_match import sibling_seed as ss
    Xo, Yo = float(fr['x_cen']), float(fr['y_cen']); pscale = float(fr['pixel_scale']) * 1000.0
    gx, gy = np.meshgrid(np.linspace(*xr, 9), np.linspace(*yr, 9)); h = np.column_stack([gx.ravel(), gy.ravel()])
    def solve(Mm, tt):
        g = (h - c) @ Mm.T + tt
        ra, dec = ss.frame_to_sky(g[:, 0], g[:, 1], fr)
        g0 = (np.array([[Xo, Yo]]) - c) @ Mm.T + tt
        ra0, dec0 = ss.frame_to_sky(g0[:, 0], g0[:, 1], fr)
        u, v = plane_project(ra, dec, float(ra0[0]), float(dec0[0]), pscale)
        X = np.column_stack([h[:, 0] - Xo, h[:, 1] - Yo])
        pu, *_ = np.linalg.lstsq(X, np.asarray(u), rcond=None); pv, *_ = np.linalg.lstsq(X, np.asarray(v), rcond=None)
        return np.array([pu[0], pu[1], pv[0], pv[1]]), float(ra0[0]), float(dec0[0])
    abcd, ra0, dec0 = solve(M, t)
    # covariance: Monte Carlo of the relative fit, mapped to (a,b,c,d, d_ra0, d_dec0 [mas]) + the anchor's block
    rng = np.random.default_rng(1); S = []
    L = np.linalg.cholesky(cov6 + 1e-18 * np.eye(6))
    for _ in range(100):
        e = L @ rng.standard_normal(6)
        Mm = M + np.array([[e[0], e[1]], [e[3], e[4]]]); tt = t + np.array([e[2], e[5]])
        a2, r2, d2 = solve(Mm, tt)
        S.append(np.r_[a2, (r2 - ra0) * 3.6e6, (d2 - dec0) * 3.6e6])   # solver r[4] convention: no cos(dec)
    C = np.cov(np.array(S).T)
    if anchor_block is not None:
        C = C + anchor_block
    else:
        C[4, 4] += 50.0 ** 2; C[5, 5] += 50.0 ** 2      # anchor without a v1 posterior: 50 mas pointing
    row = dict(image_name=img, a=abcd[0], b=abcd[1], c=abcd[2], d=abcd[3], delta_ra0_mas=0.0, delta_dec0_mas=0.0,
               ra0_final=ra0, dec0_final=dec0, pixel_scale_mas=pscale, Xo_pivot=Xo, Yo_pivot=Yo,
               alpha=1.0, n_stars_alignment=0, n_stars_astrometry_only=0,
               sigma_dra0_mas=float(np.sqrt(C[4, 4])), sigma_ddec0_mas=float(np.sqrt(C[5, 5])), prealign=True)
    return row, C


def run_visit_prealign(field_dir, v1_dir=None, force=False) -> int:
    """Pre-align Gaia-failed images to their visit siblings; returns the number of images aligned."""
    from astropy.io import fits
    from scipy.spatial import cKDTree
    from gaia_cross_match import sibling_seed as ss
    from bp3m.pipeline.cross_match import _find_image_folders
    from bp3m.pipeline.xmatch_groups import Member
    field_dir = Path(field_dir); F = field_dir.name
    v1_dir = Path(v1_dir) if v1_dir else field_dir / 'BP3M_results'
    out = field_dir / 'hst_xmatch' / 'prealign'; out.mkdir(parents=True, exist_ok=True)
    folders = _find_image_folders(field_dir.parent, F)
    try:
        sel = set(json.loads((field_dir / 'HST' / f'{F}_selected_obsids.json').read_text()))
        folders = [f for f in folders if Path(f['root']).name in sel]
    except Exception:
        pass
    v1 = pd.read_csv(v1_dir / 'image_transformations.csv') if (v1_dir / 'image_transformations.csv').exists() else pd.DataFrame()
    C1 = np.load(v1_dir / 'C_r.npy') if (v1_dir / 'C_r.npy').exists() else None
    v1_ok, v1_blk = {}, {}
    if len(v1):
        k1 = C1.shape[0] // len(v1) if C1 is not None else 0
        for j, r in v1.iterrows():
            base = str(r.image_name).replace('_hi', '').replace('_lo', '')
            v1_ok[base] = v1_ok.get(base, 0) + int(r.n_stars_alignment)
            if C1 is not None and k1 >= 6 and base not in v1_blk:
                b = C1[j * k1:j * k1 + 6, j * k1:j * k1 + 6]
                if np.isfinite(b).all() and np.sqrt(abs(b[4, 4])) < 1000.0:
                    v1_blk[base] = b
    # visit groups WITHOUT the WCSNAME split light_groups applies: a Gaia-failed exposure often lacks
    # MAST's a-posteriori WCS that its matched siblings have; such pairs start from a wider radius
    from bp3m.visit_groups import exposure_meta, group_key, GAP_DAYS
    metas, by = {}, {}
    for f in folders:
        try:
            m = exposure_meta(Path(f['flc'])); metas[Path(f['root']).name] = m
        except Exception:
            continue
        k = group_key(m)
        if k is not None:
            by.setdefault(k, []).append((float(m.get('EXPSTART') or 0.0), f))
    groups = []
    for lst in by.values():
        lst.sort(key=lambda q: q[0]); cur, last = [], None
        for tt, f in lst:
            if cur and last is not None and tt - last > GAP_DAYS:
                if len(cur) >= 2: groups.append(cur)
                cur = []
            cur.append(f); last = tt
        if len(cur) >= 2: groups.append(cur)
    rows, blocks, report = [], [], []
    by_name = {Path(f['root']).name: f for f in folders}
    for grp in groups:
        names = [Path(f['root']).name if isinstance(f, dict) else str(f) for f in grp]
        mem = {}
        for n in names:
            if n in by_name:
                try: mem[n] = Member(by_name[n])
                except Exception: pass
        anchors = sorted([m for m in mem.values() if m.ok], key=lambda m: (-v1_ok.get(m.name, 0), -m.n))
        targets = [m for m in mem.values() if not m.ok]
        if not anchors or not targets:
            continue
        for T in targets:
            done = False
            for A in anchors[:3]:
                try:
                    A.cat = fits.getdata(A.catalog, 1); T.cat = fits.getdata(T.catalog, 1)
                    ia, it_ = _good_stars(A.cat), _good_stars(T.cat)
                    if len(ia) < MIN_MATCH or len(it_) < MIN_MATCH: continue
                    ra_a, dec_a = _sky_of_anchor(A, ia)
                    sib = ss.load_sibling(A.flc, A.catalog, A.root / 'transformation.csv',
                                          __import__('gaia_cross_match.cross_match', fromlist=['get_hst_params']).get_hst_params)
                    fr = {k: T.params[k] for k in ('ra_cen', 'dec_cen', 'x_cen', 'y_cen', 'pixel_scale', 'orientat')}
                    hT = np.column_stack([T.cat['x_gdc'][it_], T.cat['y_gdc'][it_]]).astype(float)
                    xr = (float(np.percentile(T.cat['x_gdc'][it_], 1)), float(np.percentile(T.cat['x_gdc'][it_], 99)))
                    yr = (float(np.percentile(T.cat['y_gdc'][it_], 1)), float(np.percentile(T.cat['y_gdc'][it_], 99)))
                    E = ss.measure_header_error(sib['params'], sib['trans'], sib['xr'], sib['yr'])
                    Mg, tg, _ = ss.predicted_affine(T.params, E, xr, yr)
                    cT = np.array([T.params['x_cen'], T.params['y_cen']])
                    gA = np.column_stack(ss.sky_to_frame(ra_a, dec_a, fr))          # anchor stars in T's Gaia frame
                    M, t, c = Mg, tg, cT
                    same_wcs = metas.get(T.name, {}).get('WCSNAME') == metas.get(A.name, {}).get('WCSNAME')
                    radii = RADII_PX if same_wcs else (60.0, 15.0) + RADII_PX
                    for rad in radii:
                        gT = (hT - c) @ M.T + t
                        d, j = cKDTree(gA).query(gT, distance_upper_bound=rad)
                        m = np.isfinite(d)
                        if m.sum() < MIN_MATCH: break
                        M, t, c, res, cov6 = _fit_affine(hT[m], gA[j[m]])
                    else:
                        rms = float(np.sqrt(np.mean(res ** 2)))
                        if rms <= MAX_RMS_PX:
                            row, C = _v1_row(T.name, fr, M, t, c, cov6, xr, yr, v1_blk.get(A.name))
                            row.update(anchor=A.name, n_match=int(m.sum()), rms_px=rms)
                            rows.append(row); blocks.append(C)
                            # transformation.csv-schema file in the target's own header frame
                            A_, B_, C_, D_ = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
                            vals = dict(A=A_, B=B_, C=C_, D=D_, xs_o=c[0], ys_o=c[1], xt_o=t[0], yt_o=t[1],
                                        ratio=float(np.sqrt(A_ * D_ - B_ * C_)), rot_deg=float(np.degrees(np.arctan2(B_ - C_, A_ + D_))),
                                        on_skew=0.5 * (A_ - D_), off_skew=0.5 * (B_ + C_), zp=np.nan,
                                        ra_cen=fr['ra_cen'], dec_cen=fr['dec_cen'], x_cen=fr['x_cen'], y_cen=fr['y_cen'],
                                        pixel_scale=fr['pixel_scale'], orientat=fr['orientat'], prealign_anchor=np.nan,
                                        n_match=int(m.sum()), rms_px=rms)
                            pd.DataFrame({'parameter': list(vals), 'value': list(vals.values())}).to_csv(
                                T.root / 'transformation_prealign.csv', index=False)
                            report.append(dict(image=T.name, anchor=A.name, n_match=int(m.sum()), rms_px=round(rms, 4), status='aligned'))
                            done = True; break
                except Exception as e:
                    report.append(dict(image=T.name, anchor=A.name, status=f'error {type(e).__name__}: {str(e)[:80]}'))
            if not done:
                report.append(dict(image=T.name, anchor=(anchors[0].name if anchors else ''), status='not aligned'))
    pd.DataFrame(report).to_csv(out / 'prealign_report.csv', index=False)
    if rows:
        pd.DataFrame(rows).to_csv(out / 'image_transformations.csv', index=False)
        n = len(blocks); Cb = np.zeros((6 * n, 6 * n))
        for i, b in enumerate(blocks): Cb[6 * i:6 * i + 6, 6 * i:6 * i + 6] = b
        np.save(out / 'C_r.npy', Cb)
    else:
        for f in ('image_transformations.csv', 'C_r.npy'):
            if (out / f).exists(): (out / f).unlink()
    (out / 'prealign_meta.json').write_text(json.dumps({'version': PREALIGN_VERSION, 'n_aligned': len(rows),
                                                        'n_targets': len(report), 'enabled': True}))
    print(f"  visit pre-alignment: {len(rows)} Gaia-failed image(s) aligned to visit siblings on HST stars "
          f"({sum(1 for r in report if r['status'] == 'not aligned')} not aligned)  -> {out}")
    return len(rows)


def set_prealign_enabled(field_dir, enabled: bool) -> None:
    """run_iterate_v2 switches the on-disk prealignment on (--visit_prealign) or off for this run."""
    m = Path(field_dir) / 'hst_xmatch' / 'prealign' / 'prealign_meta.json'
    if m.exists():
        rec = json.loads(m.read_text()); rec['enabled'] = bool(enabled); m.write_text(json.dumps(rec))


def load_prealign(field_dir):
    """(DataFrame of v1-style rows, block-diagonal C_r) or (None, None) when absent or not enabled."""
    out = Path(field_dir) / 'hst_xmatch' / 'prealign'
    p, c = out / 'image_transformations.csv', out / 'C_r.npy'
    if not p.exists() or not c.exists():
        return None, None
    try:
        if not json.loads((out / 'prealign_meta.json').read_text()).get('enabled', False):
            return None, None
    except Exception:
        return None, None
    return pd.read_csv(p), np.load(c)
