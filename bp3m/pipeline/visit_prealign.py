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


def epoch_override_names(pre_df) -> set:
    """Exposures whose prealignment is an epoch anchor: they REPLACE any source-run (v1/v2) solution, which for these
    images is a failed or suspect low-N Gaia cross-match."""
    if pre_df is None or 'anchor' not in pre_df.columns:
        return set()
    return set(pre_df.loc[pre_df['anchor'].astype(str) == 'epoch:v2', 'image_name'].astype(str))


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


# ── epoch anchoring (2026-10-02) ───────────────────────────────────────────────────────────────────────────────
EPOCH_SIG_PM_MAX = 0.5      # mas/yr: reference stars of the previous v2 catalogue
EPOCH_MIN_MATCH = 10
EPOCH_VOTE_PX = 40.0        # header WCS error searched by the offset vote (px; Gaia-tied headers are good to ~1 px)
EPOCH_FRAME_FLOOR_MAS = 2.0


def _epoch_reference(v2_dir, t_obs):
    """Well-measured stars of a v2 catalogue (stellar_astrometry.csv) propagated to epoch t_obs: (ra, dec, sig_mas)."""
    s = pd.read_csv(Path(v2_dir) / 'stellar_astrometry.csv', low_memory=False)
    sp = np.hypot(s.sigma_pmra_bp3m, s.sigma_pmdec_bp3m) / np.sqrt(2)
    s = s[(sp < EPOCH_SIG_PM_MAX) & np.isfinite(s.pmra_bp3m) & np.isfinite(s.delta_racosdec_bp3m)]
    cosd = np.cos(np.radians(s.dec.to_numpy()))
    dt = t_obs - s.Gaia_time.to_numpy(float)
    ra = s.ra.to_numpy() + (s.delta_racosdec_bp3m.to_numpy() + s.pmra_bp3m.to_numpy() * dt) / (cosd * 3.6e6)
    dec = s.dec.to_numpy() + (s.delta_dec_bp3m.to_numpy() + s.pmdec_bp3m.to_numpy() * dt) / 3.6e6
    sig = np.sqrt(0.5 * (s.sigma_delta_racosdec.to_numpy() ** 2 + s.sigma_delta_dec.to_numpy() ** 2)
                  + (np.hypot(s.sigma_pmra_bp3m, s.sigma_pmdec_bp3m).to_numpy() / np.sqrt(2) * dt) ** 2)
    return ra, dec, sig


def run_epoch_anchor(field_dir, v2_dir=None) -> int:
    """Align exposures whose Gaia cross-match failed or is a suspect low-N solution (xmatch_groups.low_n_suspect)
    to the previous v2 catalogue propagated to their epoch, by offset voting in the header frame + clipped affine.
    Writes the same products as run_visit_prealign (rows + covariance in hst_xmatch/prealign, per-image
    transformation_prealign.csv), keeping sibling prealignments only where their anchor is trustworthy."""
    from astropy.io import fits
    from scipy.spatial import cKDTree
    from gaia_cross_match import sibling_seed as ss
    from bp3m.pipeline.cross_match import _find_image_folders
    from bp3m.pipeline.xmatch_groups import Member, low_n_suspect
    field_dir = Path(field_dir); F = field_dir.name
    v2_dir = Path(v2_dir) if v2_dir else field_dir / 'BP3M_v2_results'
    if not (v2_dir / 'stellar_astrometry.csv').exists():
        print(f"  epoch anchor: no v2 catalogue at {v2_dir}; skipped"); return 0
    out = field_dir / 'hst_xmatch' / 'prealign'; out.mkdir(parents=True, exist_ok=True)
    folders = _find_image_folders(field_dir.parent, F)
    try:
        sel = set(json.loads((field_dir / 'HST' / f'{F}_selected_obsids.json').read_text()))
        folders = [f for f in folders if Path(f['root']).name in sel]
    except Exception:
        pass
    mem, suspect = {}, set()
    for f in folders:
        try:
            m = Member(f)
        except Exception:
            continue
        mem[m.name] = m
        if low_n_suspect(m)[0]:
            suspect.add(m.name)
    old_rows, old_C = load_prealign(field_dir) if (out / 'prealign_meta.json').exists() else (None, None)
    if old_rows is None and (out / 'image_transformations.csv').exists():      # present but not enabled
        old_rows, old_C = pd.read_csv(out / 'image_transformations.csv'), np.load(out / 'C_r.npy')
    targets = [m for m in mem.values() if not m.empty and ((not m.ok) or m.name in suspect)]
    if old_rows is not None:      # sibling prealignments anchored on a suspect solution are redone too
        for _, r in old_rows.iterrows():
            if str(r.get('anchor', '')) in suspect and str(r.image_name) in mem and mem[str(r.image_name)] not in targets:
                targets.append(mem[str(r.image_name)])
    rows, blocks, report = [], [], []
    for T in sorted(targets, key=lambda m: m.name):
        try:
            T.cat = fits.getdata(T.catalog, 1); it_ = _good_stars(T.cat, n=3000)
            if len(it_) < EPOCH_MIN_MATCH:
                report.append(dict(image=T.name, status='too few stars')); continue
            t_obs = 2000.0 + (float(fits.getheader(T.flc, 0)['EXPSTART']) - 51544.5) / 365.25
            ra_r, de_r, sig_r = _epoch_reference(v2_dir, t_obs)
            fr = {k: T.params[k] for k in ('ra_cen', 'dec_cen', 'x_cen', 'y_cen', 'pixel_scale', 'orientat')}
            gA = np.column_stack(ss.sky_to_frame(ra_r, de_r, fr))
            hT = np.column_stack([T.cat['x_gdc'][it_], T.cat['y_gdc'][it_]]).astype(float)
            xr = (float(np.percentile(hT[:, 0], 1)), float(np.percentile(hT[:, 0], 99)))
            yr = (float(np.percentile(hT[:, 1], 1)), float(np.percentile(hT[:, 1], 99)))
            E0 = dict(a=1.0, b=0.0, tx=0.0, ty=0.0, ra0=fr['ra_cen'], dec0=fr['dec_cen'])   # header-only guess
            M, t, _ = ss.predicted_affine(T.params, E0, xr, yr); c = np.array([fr['x_cen'], fr['y_cen']])
            gT = (hT - c) @ M.T + t
            inside = (gA[:, 0] > gT[:, 0].min()) & (gA[:, 0] < gT[:, 0].max()) & (gA[:, 1] > gT[:, 1].min()) & (gA[:, 1] < gT[:, 1].max())
            if inside.sum() < EPOCH_MIN_MATCH:
                report.append(dict(image=T.name, status=f'only {int(inside.sum())} reference stars in the field')); continue
            dd = (gA[inside][None, :, :] - gT[:, None, :]).reshape(-1, 2)
            dd = dd[np.hypot(dd[:, 0], dd[:, 1]) < EPOCH_VOTE_PX]
            bins = np.arange(-EPOCH_VOTE_PX, EPOCH_VOTE_PX + 0.2, 0.2)
            Hh, xb, yb = np.histogram2d(dd[:, 0], dd[:, 1], [bins, bins])
            k = np.unravel_index(np.argmax(Hh), Hh.shape); off = np.array([xb[k[0]] + 0.1, yb[k[1]] + 0.1])
            t = t + off; ok_fit = False
            for rad in (2.0, 1.0, 0.6):
                gT = (hT - c) @ M.T + t
                d, j = cKDTree(gA).query(gT, distance_upper_bound=rad); m = np.isfinite(d)
                if m.sum() < EPOCH_MIN_MATCH: break
                M, t, c, res, cov6 = _fit_affine(hT[m], gA[j[m]]); ok_fit = True
            else:
                pass
            if not ok_fit or m.sum() < EPOCH_MIN_MATCH:
                report.append(dict(image=T.name, status=f'not aligned (vote peak {int(Hh[k])})')); continue
            rms = float(np.sqrt(np.mean(res ** 2)))
            if rms > MAX_RMS_PX:
                report.append(dict(image=T.name, status=f'rms {rms:.3f} px too large')); continue
            sig_frame = max(EPOCH_FRAME_FLOOR_MAS, float(np.median(sig_r[j[m]])) / np.sqrt(m.sum()))
            frame_block = np.zeros((6, 6)); frame_block[4, 4] = frame_block[5, 5] = sig_frame ** 2
            row, C = _v1_row(T.name, fr, M, t, c, cov6, xr, yr, frame_block)
            row.update(anchor='epoch:v2', n_match=int(m.sum()), rms_px=rms)
            rows.append(row); blocks.append(C)
            A_, B_, C_, D_ = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
            vals = dict(A=A_, B=B_, C=C_, D=D_, xs_o=c[0], ys_o=c[1], xt_o=t[0], yt_o=t[1],
                        ratio=float(np.sqrt(A_ * D_ - B_ * C_)), rot_deg=float(np.degrees(np.arctan2(B_ - C_, A_ + D_))),
                        on_skew=0.5 * (A_ - D_), off_skew=0.5 * (B_ + C_), zp=np.nan,
                        ra_cen=fr['ra_cen'], dec_cen=fr['dec_cen'], x_cen=fr['x_cen'], y_cen=fr['y_cen'],
                        pixel_scale=fr['pixel_scale'], orientat=fr['orientat'], prealign_anchor=np.nan,
                        n_match=int(m.sum()), rms_px=rms)
            pd.DataFrame({'parameter': list(vals), 'value': list(vals.values())}).to_csv(T.root / 'transformation_prealign.csv', index=False)
            report.append(dict(image=T.name, anchor='epoch:v2', n_match=int(m.sum()), rms_px=round(rms, 4), vote_peak=int(Hh[k]),
                               header_offset_px=np.round(off, 2).tolist(), sigma_frame_mas=round(sig_frame, 2),
                               reason='suspect low-N xmatch' if T.name in suspect else 'xmatch failed', status='aligned'))
        except Exception as e:
            report.append(dict(image=T.name, status=f'error {type(e).__name__}: {str(e)[:80]}'))
    done = {r['image_name'] for r in rows}
    # keep sibling prealignments whose anchor is trustworthy and that were not redone here
    if old_rows is not None:
        for j, r in old_rows.reset_index(drop=True).iterrows():
            n = str(r.image_name)
            if n in done or str(r.get('anchor', '')) in suspect or n in suspect:
                continue
            rows.append(r.to_dict()); blocks.append(old_C[6 * j:6 * j + 6, 6 * j:6 * j + 6]); done.add(n)
    pd.DataFrame(report).to_csv(out / 'epoch_anchor_report.csv', index=False)
    if rows:
        pd.DataFrame(rows).to_csv(out / 'image_transformations.csv', index=False)
        Cb = np.zeros((6 * len(blocks), 6 * len(blocks)))
        for i, b in enumerate(blocks): Cb[6 * i:6 * i + 6, 6 * i:6 * i + 6] = b
        np.save(out / 'C_r.npy', Cb)
    (out / 'prealign_meta.json').write_text(json.dumps({'version': PREALIGN_VERSION, 'n_aligned': len(rows), 'enabled': True,
                                                        'epoch_anchor_from': str(v2_dir),
                                                        'n_epoch_anchored': sum(1 for r in report if r.get('anchor') == 'epoch:v2'),
                                                        'suspect_low_n': sorted(suspect)}))
    n_ep = sum(1 for r in report if r.get('anchor') == 'epoch:v2')
    print(f"  epoch anchor: {n_ep}/{len(targets)} exposure(s) aligned to the v2 catalogue at their epoch "
          f"({len(suspect)} suspect low-N cross-matches); {len(rows)} prealigned in total -> {out}")
    return n_ep
