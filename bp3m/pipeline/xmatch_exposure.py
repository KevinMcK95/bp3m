"""Exposure-level NIRCam Gaia alignment (J5 step 3, 2026-10-23).

A NIRCam exposure is 10 detectors (8 SW + 2 LW) matched one at a time by ``gaia_cross_match``; a 63" SW detector sees
0-16 Gaia stars and in the sparse M31-satellite fields 60-75 % of the frames never get a solution (And XIV 106 of 140),
while the exposure as a whole sees 20-60.  Here every SCA of an exposure is placed on ONE focal plane by the calibrated
layout table (bp3m/data/nircam_sca_layout.csv: per (SCA, SW filter) the affine from pypass x_gdc/y_gdc to the
NRCB1-referenced plane, SIAF + STDGDC->idl + the self-calibrated per-SCA terms with their field-to-field scatter as prior
widths), the exposure's Gaia stars are matched and fitted with ONE transformation (4p similarity below 10 stars, 6p
affine from 10), and the solution is written back per SCA in the ``transformation.csv`` / ``matched_gaia.csv`` /
``xmatch_status.json`` schema of the per-image matcher, so every downstream reader (indv, v1, v2, pop-fit) is unchanged.
Held-out test (ml_jwst/jwst_expo_xmatch_test.py, And XIV): leave-one-SCA-out prediction 9.5 mas rms with the calibrated
layout vs 26 mas with SIAF alone; per-SCA 6p fits impossible on 18/33 SCA-exposures.

Policy: a frame that already has a trusted per-image solution with >= N_OWN_KEEP matches keeps it (it is also one of the
anchors of the pooled fit); every other frame of the exposure (failed, weak, suspect, sibling-prealigned) receives the
pooled solution with its own Gaia stars listed in matched_gaia.csv (even 1-3 of them: the transformation is constrained
by the exposure, and the solver's exposure link prior ties the SCAs).  The previous per-image files are kept as
transformation_single.csv / matched_gaia_single.csv.  Gaia stars saturated in the frame are NOT dropped (the per-image
matcher's n_near_sat guard removes most Gaia stars of a deep frame); their catalogue covariance carries the weight.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

from bp3m.jwst_links import parse_jwst_name

LAYOUT_CSV = Path(__file__).resolve().parents[1] / 'data' / 'nircam_sca_layout.csv'
EXPOSURE_XMATCH_VERSION = 1
R0_MAS, R1_MAS = 400.0, 80.0         # header-WCS seed radius; refined match radius
FLOOR_MAS = 2.0                      # added in quadrature to every star (layout + saturation residuals)
MIN_TOTAL = 5                        # pooled matches needed for a solution
N_6P = 10                            # 6p affine from this many matches, 4p similarity below
N_OWN_KEEP = 8                       # a trusted per-image solution with >= this many matches is kept
PIX_SW_MAS = 31.0
SCAS = ('NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCA5', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4', 'NRCB5')

_LAYOUT = None


def load_layout() -> pd.DataFrame:
    global _LAYOUT
    if _LAYOUT is None:
        _LAYOUT = pd.read_csv(LAYOUT_CSV).set_index(['sca', 'sw_filter'])
    return _LAYOUT


def sca_of(image_name: str):
    p = parse_jwst_name(image_name)
    if p is None:
        return None
    s = p['det'].upper().replace('LONG', '5')
    return s if s in SCAS else None


def _fp(layout, sca, kf, x, y):
    r = layout.loc[(sca, kf)]
    return (r.c0u + r.c1u * (x - 1024.0) + r.c2u * (y - 1024.0), r.c0v + r.c1v * (x - 1024.0) + r.c2v * (y - 1024.0))


def _read_cat(path):
    from astropy.io import fits
    t = fits.getdata(path)
    x = np.asarray(t['x_gdc'], float); y = np.asarray(t['y_gdc'], float)
    sx = np.sqrt(np.clip(np.asarray(t['cov_xx'], float), 0, None)); sy = np.sqrt(np.clip(np.asarray(t['cov_yy'], float), 0, None))
    ok = (np.asarray(t['converged']).astype(bool) & np.asarray(t['is_star_candidate']).astype(bool) & np.isfinite(x) & np.isfinite(y)
          & np.isfinite(np.asarray(t['ra'], float)) & (sx < 2) & (sy < 2) & (np.asarray(t['flux'], float) > 0))
    idx = np.where(ok)[0]
    cols = {c: np.asarray(t[c])[idx] for c in ('x_gdc', 'y_gdc', 'ra', 'dec', 'flux', 'mag_gdc', 'mag_err_gdc', 'mag_st_gdc', 'mag_ab', 'mag_ab_gdc', 'n_sat', 'is_star_candidate') if c in t.dtype.names}
    cols['hst_index'] = idx; cols['e_px'] = np.sqrt(0.5 * (sx[idx] ** 2 + sy[idx] ** 2))
    return pd.DataFrame(cols)


def _fit(X, Y, w, six: bool):
    """weighted LSQ of Y (n,2) = f(X (n,2)); six: 6p affine (c (3,2)); else 4p similarity returned as the same c."""
    W = np.sqrt(w)
    if six:
        A = np.c_[np.ones(len(X)), X]; c = np.linalg.lstsq(A * W[:, None], Y * W[:, None], rcond=None)[0]; return c
    n = len(X); M = np.zeros((2 * n, 4)); yv = np.zeros(2 * n)
    M[:n, 0], M[:n, 1], M[:n, 2] = X[:, 0], -X[:, 1], 1.0; M[n:, 0], M[n:, 1], M[n:, 3] = X[:, 1], X[:, 0], 1.0; yv[:n], yv[n:] = Y[:, 0], Y[:, 1]
    ww = np.r_[W, W]; p = np.linalg.lstsq(M * ww[:, None], yv * ww, rcond=None)[0]
    return np.array([[p[2], p[3]], [p[0], p[1]], [-p[1], p[0]]])


def _apply(c, X): return np.c_[np.ones(len(X)), X] @ c


def _mutual(pa, pb, r):
    from scipy.spatial import cKDTree
    d, j = cKDTree(pb).query(pa, distance_upper_bound=r); d2, i2 = cKDTree(pa).query(pb, distance_upper_bound=r)
    m = np.isfinite(d); m[m] &= i2[j[m]] == np.where(m)[0]; return np.where(m)[0], j[m]


def _pooled_fit(fp_raw, sky0, G, gvar, cvar):
    """fp_raw (n,2) px, sky0 (n,2) header tangent mas, G (m,2) Gaia tangent mas, gvar/cvar variances (mas^2).
    The focal plane is first taken to the sky with the header-implied 6p map L0 (fitted on ALL catalogue rows: roll,
    scale and the V2/V3 -> East/North parity), so that the small-number 4p similarity fit is a near-identity correction.
    -> (ib, ia, c, w, rms_mas) or None;  c maps fp_raw -> tangent mas (L0 composed in)."""
    L0 = _fit(fp_raw, sky0, np.ones(len(fp_raw)), True); fp = _apply(L0, fp_raw)
    ib, ia = _mutual(sky0, G, R0_MAS)
    if len(ib) < MIN_TOTAL:
        return None
    c = None
    for _ in range(3):
        w = 1.0 / (gvar[ia] + cvar[ib] + FLOOR_MAS ** 2); c = _fit(fp[ib], G[ia], w, len(ib) >= N_6P)
        pred = _apply(c, fp); ib, ia = _mutual(pred, G, R1_MAS)
        if len(ib) < MIN_TOTAL:
            return None
        res = np.hypot(*(G[ia] - pred[ib]).T); z = res / np.sqrt(gvar[ia] + cvar[ib] + FLOOR_MAS ** 2)
        ok = z < 3.0 * max(1.0, float(np.median(z)) / 0.83); ib, ia = ib[ok], ia[ok]
        if len(ib) < MIN_TOTAL:
            return None
    w = 1.0 / (gvar[ia] + cvar[ib] + FLOOR_MAS ** 2); c = _fit(fp[ib], G[ia], w, len(ib) >= N_6P)
    res = G[ia] - _apply(c, fp[ib])
    # compose: fp_raw -> (L0) -> fp -> (c) -> sky  as one (3,2) coefficient block
    cc = np.vstack([c[0] + L0[0] @ c[1:], L0[1:] @ c[1:]])
    return ib, ia, cc, w, float(np.sqrt(np.mean(res ** 2)))


def _status(root: Path) -> dict:
    try:
        return json.loads((root / 'xmatch_status.json').read_text())
    except Exception:
        return {}


def _own_solution_kept(root: Path) -> bool:
    st = _status(root)
    if st.get('status') != 'success' or not (root / 'matched_gaia.csv').exists() or not (root / 'transformation.csv').exists():
        return False
    if st.get('params', {}).get('exposure_pooled'):
        return False
    gp = st.get('group_pass', {})
    if gp.get('cls') == 'suspect':
        return False
    return int(st.get('n_matched', 0) or 0) >= N_OWN_KEEP


def run_exposure_pass(folders: list, gaia_df, params_meta: dict, log=print, force: bool = False, out_dir=None) -> int:
    """folders: [{'root','flc','catalog'}, ...] of the field; gaia_df: the field's Gaia table (ra, dec, pm, plx, ...).
    out_dir (tests only): write the per-frame products under out_dir/<image>/ instead of the frame directories.
    Returns the number of frames given a pooled solution."""
    from gaia_cross_match.cross_match import get_hst_params, propagate_gaia_with_cov
    from bp3m.pipeline.cross_match import _gaia_subset_for_image, gaia_unit_tree
    from gaia_cross_match.sibling_seed import sky_to_frame, _untangent
    from gaia_cross_match.miracle_match import rd2x, rd2y
    t0 = time.time(); layout = load_layout()
    by_expo = {}
    for f in folders:
        name = Path(f['root']).name; p = parse_jwst_name(name)
        if p is None or sca_of(name) is None:
            continue
        by_expo.setdefault(p['exposure'], []).append(f)
    tree = gaia_unit_tree(gaia_df) if len(gaia_df) else None
    n_written = n_kept = n_fail = n_expo = 0
    for expo, fs in sorted(by_expo.items()):
        if len(fs) < 2:
            continue
        n_expo += 1
        # per-SCA params / catalogues
        members = []
        for f in fs:
            root = Path(f['root']); name = root.name; sca = sca_of(name)
            try:
                pr = get_hst_params(f['flc'], catalog_file=f['catalog'])
                cat = _read_cat(f['catalog'])
            except Exception as e:
                log(f"  [expo] {name}: skipped ({type(e).__name__}: {e})"); continue
            if pr is None or len(cat) < 5:
                continue
            members.append(dict(f=f, root=root, name=name, sca=sca, params=pr, cat=cat, keep=_own_solution_kept(root) and not force))
        if len(members) < 2:
            continue
        sw = [m['params'].get('filter') or _filter_of(m['f']['flc']) for m in members if not m['sca'].endswith('5')]
        kf = sw[0] if sw else 'F150W'
        if any((m['sca'], kf) not in layout.index for m in members):
            missing = [m['sca'] for m in members if (m['sca'], kf) not in layout.index]
            log(f"  [expo] {expo}: no layout row for {missing} in {kf}; skipped"); continue
        # focal plane + header sky
        pr0 = members[0]['params']; ra_c = float(np.mean([m['params']['ra_cen'] for m in members])); dec_c = float(np.mean([m['params']['dec_cen'] for m in members]))
        fp = []; sky0 = []; cvar = []; owner = []
        for k, m in enumerate(members):
            u, v = _fp(layout, m['sca'], kf, m['cat'].x_gdc.to_numpy(float), m['cat'].y_gdc.to_numpy(float)); fp.append(np.c_[u, v])
            sky0.append(np.c_[rd2x(m['cat'].ra.to_numpy(float), m['cat'].dec.to_numpy(float), ra_c, dec_c), rd2y(m['cat'].ra.to_numpy(float), m['cat'].dec.to_numpy(float), ra_c, dec_c)] * 3.6e6)
            sc = m['params']['pixel_scale'] * 1e3; cvar.append((m['cat'].e_px.to_numpy(float) * sc) ** 2); owner.append(np.full(len(m['cat']), k))
        offs = np.cumsum([0] + [len(a) for a in fp[:-1]]); fp = np.vstack(fp); sky0 = np.vstack(sky0); cvar = np.concatenate(cvar); owner = np.concatenate(owner)
        # Gaia: union of the per-SCA subsets, propagated to the exposure epoch with the first member's ephemeris
        sub = pd.concat([_gaia_subset_for_image(m['f']['flc'], gaia_df, tree=tree) for m in members]).drop_duplicates('source_id') if len(gaia_df) else gaia_df
        if len(sub) < MIN_TOTAL:
            n_fail += 1; continue
        ra_p, dec_p, Ct = propagate_gaia_with_cov(sub, pr0['obs_epoch_mjd'], tele_xyz=pr0.get('tele_xyz'))
        G = np.c_[rd2x(ra_p, dec_p, ra_c, dec_c), rd2y(ra_p, dec_p, ra_c, dec_c)] * 3.6e6; gvar = 0.5 * (Ct[:, 0, 0] + Ct[:, 1, 1])
        sol = _pooled_fit(fp, sky0, G, gvar, cvar)
        if sol is None:
            n_fail += 1; log(f"  [expo] {expo}: no pooled solution ({len(members)} SCAs, {len(sub)} Gaia in field)"); continue
        ib, ia, c, w, rms = sol; pred = _apply(c, fp)
        # per-SCA write-back
        for k, m in enumerate(members):
            mine = owner[ib] == k; n_own = int(mine.sum())
            if m['keep']:
                n_kept += 1; continue
            fr = {kk: m['params'][kk] for kk in ('ra_cen', 'dec_cen', 'x_cen', 'y_cen', 'pixel_scale', 'orientat')}
            cat = m['cat']; xr = (float(np.percentile(cat.x_gdc, 1)), float(np.percentile(cat.x_gdc, 99))); yr = (float(np.percentile(cat.y_gdc, 1)), float(np.percentile(cat.y_gdc, 99)))
            gx, gy = np.meshgrid(np.linspace(*xr, 9), np.linspace(*yr, 9)); hx = gx.ravel(); hy = gy.ravel()
            u, v = _fp(layout, m['sca'], kf, hx, hy); s = _apply(c, np.c_[u, v]); ra_g, dec_g = _untangent(s[:, 0], s[:, 1], ra_c, dec_c); gxf, gyf = sky_to_frame(ra_g, dec_g, fr)
            X = np.column_stack([hx - fr['x_cen'], hy - fr['y_cen'], np.ones_like(hx)]); px = np.linalg.lstsq(X, gxf, rcond=None)[0]; py = np.linalg.lstsq(X, gyf, rcond=None)[0]
            A_, B_, C_, D_ = px[0], px[1], py[0], py[1]; t = (px[2], py[2])
            # matches on this SCA
            gsel = ia[mine]; sc = fr['pixel_scale'] * 1e3; gsel_rows = ib[mine]; sel = gsel_rows - offs[k]      # rows of this SCA's catalogue frame
            res = (G[gsel] - pred[gsel_rows]) / sc; sig = np.sqrt(gvar[gsel] + cvar[gsel_rows] + FLOOR_MAS ** 2) / sc
            mag_hst = cat.mag_ab_gdc.to_numpy(float)[sel] if 'mag_ab_gdc' in cat else cat.mag_gdc.to_numpy(float)[sel]
            gmag = sub.gmag.to_numpy(float)[gsel] if 'gmag' in sub else np.full(len(sel), np.nan)
            zp = float(np.nanmedian(gmag - mag_hst)) if n_own else np.nan
            out = pd.DataFrame(dict(hst_index=cat.hst_index.to_numpy()[sel], hst_x_gdc=cat.x_gdc.to_numpy()[sel], hst_y_gdc=cat.y_gdc.to_numpy()[sel],
                                    hst_mag_gdc=cat.mag_gdc.to_numpy(float)[sel] if 'mag_gdc' in cat else mag_hst, hst_mag_err_gdc=cat.mag_err_gdc.to_numpy(float)[sel] if 'mag_err_gdc' in cat else np.nan,
                                    hst_mag_st_gdc=mag_hst, hst_mag_ab=cat.mag_ab.to_numpy(float)[sel] if 'mag_ab' in cat else mag_hst, hst_mag_ab_gdc=mag_hst, hst_mag_system='AB',
                                    gaia_source_id=sub.source_id.to_numpy(np.int64)[gsel], has_gaia_pms=np.isfinite(sub.pmra.to_numpy(float)[gsel]) if 'pmra' in sub else True,
                                    gaia_ra_prop=ra_p[gsel], gaia_dec_prop=dec_p[gsel], gaia_gmag=gmag, residual_mag=gmag - (mag_hst + zp), residual_x=res[:, 0], residual_y=res[:, 1], residual_sigma=sig,
                                    hst_is_star=cat.is_star_candidate.to_numpy().astype(bool)[sel] if 'is_star_candidate' in cat else True))
            root = m['root']
            if out_dir is not None:
                root = Path(out_dir) / m['name']; root.mkdir(parents=True, exist_ok=True)
            for fn in ('transformation.csv', 'matched_gaia.csv'):
                src = root / fn; dst = root / fn.replace('.csv', '_single.csv')
                if src.exists() and not dst.exists():
                    src.replace(dst)
            ratio = float(np.sqrt(abs(A_ * D_ - B_ * C_))); rot = float(np.degrees(np.arctan2(B_ - C_, A_ + D_)))
            if n_own == 0:
                # no Gaia star on this SCA: a prealign-only frame (loader admits it with zero stars, the exposure link
                # prior ties its geometry); an empty matched_gaia.csv would make the loader drop it
                vals = dict(A=A_, B=B_, C=C_, D=D_, xs_o=fr['x_cen'], ys_o=fr['y_cen'], xt_o=t[0], yt_o=t[1], ratio=ratio, rot_deg=rot, on_skew=0.5 * (A_ - D_), off_skew=0.5 * (B_ + C_), zp=np.nan,
                            ra_cen=fr['ra_cen'], dec_cen=fr['dec_cen'], x_cen=fr['x_cen'], y_cen=fr['y_cen'], pixel_scale=fr['pixel_scale'], orientat=fr['orientat'], prealign_anchor=np.nan,
                            initial_scale=m['params'].get('initial_scale', np.nan), initial_scale_header=m['params'].get('initial_scale_header', np.nan),
                            n_match=0, rms_px=rms / (fr['pixel_scale'] * 1e3), n_siblings=float(len(members) - 1), prealign_source='jwst_exposure_pooled',
                            exposure_pooled=1.0, n_match_expo=float(len(ib)), pooled_rms_mas=rms, exposure_xmatch_version=float(EXPOSURE_XMATCH_VERSION))
                pd.DataFrame({'parameter': list(vals), 'value': list(vals.values())}).to_csv(root / 'transformation_prealign.csv', index=False)
                for fn in ('transformation.csv', 'matched_gaia.csv'):
                    if (root / fn).exists(): (root / fn).unlink()
                st = _status(m['root']); st.update(dict(status='failed', reason='no Gaia star on this SCA; exposure-pooled prealign', n_matched=0, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
                st['params'] = dict(st.get('params') or params_meta, exposure_pooled=True, exposure_xmatch_version=EXPOSURE_XMATCH_VERSION, n_match_expo=int(len(ib)), pooled_rms_mas=round(rms, 3), sw_filter=kf)
                (root / 'xmatch_status.json').write_text(json.dumps(st, indent=2)); n_written += 1
                continue
            out.to_csv(root / 'matched_gaia.csv', index=False)
            vals = dict(A=A_, B=B_, C=C_, D=D_, xs_o=fr['x_cen'], ys_o=fr['y_cen'], xt_o=t[0], yt_o=t[1], ratio=ratio, rot_deg=rot, on_skew=0.5 * (A_ - D_), off_skew=0.5 * (B_ + C_), zp=zp,
                        ra_cen=fr['ra_cen'], dec_cen=fr['dec_cen'], x_cen=fr['x_cen'], y_cen=fr['y_cen'], pixel_scale=fr['pixel_scale'], orientat=fr['orientat'],
                        n_chance_expected=0.0, chance_fa_prob=0.0, initial_scale=m['params'].get('initial_scale', 1.0), initial_scale_header=m['params'].get('initial_scale_header', m['params'].get('initial_scale', 1.0)),
                        exposure_pooled=1.0, n_match_own=float(n_own), n_match_expo=float(len(ib)), n_sca_expo=float(len(members)), pooled_rms_mas=rms, exposure_xmatch_version=float(EXPOSURE_XMATCH_VERSION))
            pd.DataFrame({'parameter': list(vals), 'value': list(vals.values())}).to_csv(root / 'transformation.csv', index=False)
            pre = root / 'transformation_prealign.csv'
            if pre.exists():
                pre.unlink()
            st = _status(m['root']); st.update(dict(status='success', reason='', n_matched=n_own, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
            st['params'] = dict(st.get('params') or params_meta, exposure_pooled=True, exposure_xmatch_version=EXPOSURE_XMATCH_VERSION, n_match_expo=int(len(ib)), pooled_rms_mas=round(rms, 3), sw_filter=kf)
            (root / 'xmatch_status.json').write_text(json.dumps(st, indent=2)); n_written += 1
    log(f"  [{time.strftime('%H:%M:%S')}] JWST exposure-level alignment: {n_expo} exposures, {n_written} frame(s) given the pooled solution, "
        f"{n_kept} kept their own (>= {N_OWN_KEEP} matches), {n_fail} exposure(s) without a pooled solution ({time.time() - t0:.0f}s)")
    return n_written


def _filter_of(flc) -> str:
    from astropy.io import fits
    h = fits.getheader(flc, 0); return str(h.get('FILTER', 'F150W')).strip().upper()
