"""
CTE phase v2 for the population fit (2026-10-06; replaces the stale run_cte_phase_after_popfit for pop-fits).

Why a rewrite (review 2026-10-06): the July CTE phase fitted gamma from members only with the alignment restricted
to members, dropped the QSO/galaxy anchors and soft weights, refit r in frozen-alignment mode, re-used a stale
pre-tangent-update copy of xys, keyed gamma by the _hi/_lo name suffix only (a silent no-op on unsplit master_v2
images, ACS constants applied to UVIS) and threw its mu_pop away.

Model.  Each detection's measured detector position carries a CTE shift
    delta_raw = ( f . gamma_x[g],  f . gamma_y[g] )        [raw px]
    f = yt * [1, xt] (x) [1, m', m'^2] (x) [1, tau]        (12 terms; time_order=0 -> 6)
    yt  = |y_raw - y_readout| / 2048   (0 at the readout, 1 at the inter-chip gap)
    xt  = (x_raw - 2048) / 2048,   m' = (mag_inst - m_med) / m_sd,   tau = (t - t0_cam) / 10 yr
with one gamma block per group g = (camera, chip): ACS WFC1 (raw y > 2048, readout y~4096), ACS WFC2 (readout y~0),
WFC3 UVIS1 (raw y > 2047, readout y~4096), UVIS2 (readout y~0); t0 = 2002.165 (ACS) / 2009.37 (UVIS, SM4).
Chips come from the full-frame raw y of each detection (X_orig/Y_orig of the loaders), never from the image name.

Identifiability.  Within an image the part of the CTE pattern that the image's own transform (X_mat: linear +
pointing [+ poly, + 8p chip offsets]) can represent is degenerate with r.  Per image the displacement design is
therefore projected onto the complement of the alignment design (weighted by the alignment stars), both when
fitting gamma and when applying the correction.  This is identical in frozen-r (pop-fit v2) and free-r modes and
never fights the alignment.

Data.  gamma is estimated from the residuals of ACTIVE detections (use_for_fit | use_for_astrom) of members (tight
population PM prior) and alignment (Gaia) stars; diffuse-prior HST-only non-members are excluded (their own PMs
absorb a time-growing CTE shift) and so are anchors (QSOs / galaxies: different charge profile; galaxies get their
own correction upstream).  The correction is applied to every non-anchor detection by shifting the HST positions
the solver uses (d['X_c'], d['Y_c'] in the GDC-centred frame, via a per-image raw->GDC Jacobian) and rebuilding
X_mat; the originals are kept in d['X_c_nocte'] / d['Y_c_nocte'].

Loop (block coordinate descent, membership frozen): residuals -> gamma increment (projected weighted LSQ) ->
apply total gamma -> the caller's own pop-fit solve (same priors, anchors, z-weights, frozen/free mode) -> repeat.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_CAMS = {('ACS', 'WFC'): ('ACS', 2048.0, 2002.165), ('WFC3', 'UVIS'): ('UVIS', 2047.0, 2009.37)}
_Y_READOUT_TOP, _Y_READOUT_BOT = 4096.0, 0.0
MAX_SHIFT_PX = 1.0          # p99 applied-shift ceiling; CTE shifts are ~0.01-0.3 px


def _year(mjd):
    return 2000.0 + (float(mjd) - 51544.5) / 365.25


def _basis(yt, xt, mp, tau, time_order):
    sp = [yt, yt * xt]
    mg = [np.ones_like(mp), mp, mp * mp]
    tm = [np.ones_like(tau)] + ([tau] if time_order >= 1 else [])
    return np.stack([s * m * t for t in tm for s in sp for m in mg], axis=1)    # (n, nf)


def _setup(solver, image_names, stars_per_image, star_id_to_idx, anchor_sidx, time_order):
    """Per-image static CTE geometry; stores originals of X_c/Y_c once."""
    anchors = set(int(s) for s in (anchor_sidx if anchor_sidx is not None else []))
    per, groups, mags = {}, {}, []
    for img in image_names:
        d = solver._img_data.get(img)
        df = stars_per_image.get(img)
        if d is None or df is None or 'Y_orig' not in df.columns:
            continue
        meta = solver.images[img]
        cam = _CAMS.get((str(meta.get('instrument', '')).upper(), str(meta.get('detector', '')).upper()))
        if cam is None:
            continue
        cname, y_split, t0 = cam
        row_of = {}
        for k, gid in enumerate(df['Gaia_id'].to_numpy()):
            s = star_id_to_idx.get(int(gid))
            if s is not None:
                row_of.setdefault(int(s), k)
        rows = np.array([row_of.get(int(s), -1) for s in d['sidx']])
        ok = rows >= 0
        xr = np.full(d['n'], np.nan); yr = np.full(d['n'], np.nan); mg = np.full(d['n'], np.nan)
        xr[ok] = df['X_orig'].to_numpy(float)[rows[ok]]; yr[ok] = df['Y_orig'].to_numpy(float)[rows[ok]]
        mg[ok] = df['mag'].to_numpy(float)[rows[ok]]
        ok &= np.isfinite(xr) & np.isfinite(yr) & np.isfinite(mg)
        if 'X_c_nocte' not in d:
            d['X_c_nocte'] = np.asarray(d['X_c'], float).copy(); d['Y_c_nocte'] = np.asarray(d['Y_c'], float).copy()
        hi = yr > y_split
        if 'chip_ext' in df.columns:                       # catalogue chip (4 = WFC1/UVIS1 top) when available
            ce = np.full(d['n'], -1); ce[rows >= 0] = df['chip_ext'].to_numpy(int)[rows[rows >= 0]]
            hi = np.where(ce > 0, ce == 4, hi)
        # raw -> GDC-centred Jacobian: per-chip linear fit as fallback, replaced per detection by the catalogue's
        # official-GDC jac_*_gdc where present.  The learned GDC model / ED / pseudo-GDC corrections are smooth
        # additive shifts already contained in X_c (kept untouched here), so whatever GDC corrections the run uses
        # stay applied and the CTE shift is simply added on top.
        Jg = np.tile(np.eye(2), (d['n'], 1, 1))
        for ch in (True, False):
            m = ok & (hi == ch)
            if m.sum() >= 10:
                A = np.c_[xr[m], yr[m], np.ones(m.sum())]
                cx = np.linalg.lstsq(A, d['X_c_nocte'][m], rcond=None)[0]
                cy = np.linalg.lstsq(A, d['Y_c_nocte'][m], rcond=None)[0]
                Jg[m] = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
        n_jac = 0
        if all(c in df.columns for c in ('jac_xx_gdc', 'jac_xy_gdc', 'jac_yx_gdc', 'jac_yy_gdc')):
            jj = np.full((d['n'], 4), np.nan); okr = rows >= 0
            jj[okr] = df[['jac_xx_gdc', 'jac_xy_gdc', 'jac_yx_gdc', 'jac_yy_gdc']].to_numpy(float)[rows[okr]]
            fin = np.isfinite(jj).all(1)
            Jg[fin] = jj[fin].reshape(-1, 2, 2); n_jac = int(fin.sum())
        yt = np.abs(yr - np.where(hi, _Y_READOUT_TOP, _Y_READOUT_BOT)) / 2048.0
        g = np.array([f'{cname}{"1" if h else "2"}' for h in hi], dtype=object)    # WFC1/UVIS1 = top chip
        for gg in set(g[ok]):
            groups.setdefault(gg, 0)
        anc = np.array([int(s) in anchors for s in d['sidx']])
        per[img] = dict(n_jac=n_jac, ok=ok, yt=np.clip(yt, 0, 1.1), xt=(xr - 2048.0) / 2048.0, mag=mg, grp=g, Jg=Jg, anchor=anc,
                        tau=np.full(d['n'], (_year(meta['hst_time_mjd']) - t0) / 10.0), year=_year(meta['hst_time_mjd']))
        mags.append(mg[ok])
    allm = np.concatenate(mags) if mags else np.array([0.0])
    m_med, m_sd = float(np.median(allm)), float(max(np.std(allm), 1e-3))
    nf = _basis(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), time_order).shape[1]
    gl = sorted(groups)
    for p in per.values():
        p['F'] = _basis(p['yt'], np.nan_to_num(p['xt']), (np.nan_to_num(p['mag'], nan=m_med) - m_med) / m_sd, p['tau'], time_order)
        p['gi'] = np.array([gl.index(x) if x in gl else -1 for x in p['grp']])
    return per, gl, nf, m_med, m_sd


def _design_det(p, nf, ngrp, sel):
    """(n_sel, 2, n_gamma) raw-detector displacement design; gamma layout [group][x block | y block][nf]."""
    n = sel.sum(); D = np.zeros((n, 2, ngrp * 2 * nf))
    F = p['F'][sel]; gi = p['gi'][sel]
    for k in range(n):
        if gi[k] < 0:
            continue
        o = gi[k] * 2 * nf
        D[k, 0, o:o + nf] = F[k]; D[k, 1, o + nf:o + 2 * nf] = F[k]
    return D


def _to_sky(solver, img, r, p, D_raw, sel):
    """raw-detector design -> pseudo-sky (solver xys frame) via GDC Jacobian and the image's linear transform."""
    nr = solver.N_R; j = solver.image_names.index(img); rj = r[j * nr:j * nr + nr]
    M = np.array([[rj[0], rj[1]], [rj[2], rj[3]]])
    return np.einsum('ij,njk,nkl->nil', M, p['Jg'][sel], D_raw), M


def _projector(solver, img, r, d, z=None):
    """(X_fit, W_fit, pinv) of the image's alignment design for projecting displacement designs."""
    nr = solver.N_R; j = solver.image_names.index(img)
    uf = d['use_for_fit']
    if uf.sum() < nr + 2:
        return None
    Cs = solver._compute_Cs(img, r[j * nr:j * nr + nr]); W = np.linalg.inv(Cs)
    if z is not None:                                   # soft z-weights exactly as _joint_solve_pop applies them
        W = W * z[:, None, None]
    X = d['X_mat']; Xf, Wf = X[uf], W[uf]
    H = np.einsum('nia,nij,njb->ab', Xf, Wf, Xf)
    try:
        Hi = np.linalg.inv(H + 1e-12 * np.trace(H) / nr * np.eye(nr))
    except np.linalg.LinAlgError:
        return None
    return dict(X=X, W=W, uf=uf, Hi=Hi)


def _project(P, D_all):
    """D_all: (n, 2, k) design over ALL detections of the image -> component orthogonal to the alignment design."""
    Xf, Wf = P['X'][P['uf']], P['W'][P['uf']]
    B = P['Hi'] @ np.einsum('nia,nij,njk->ak', Xf, Wf, D_all[P['uf']])          # (nr, k)
    return D_all - np.einsum('nia,ak->nik', P['X'], B)


def _apply(solver, image_names, per, r, gamma, nf, ngrp, stats=None, z_weights=None):
    """Shift every non-anchor detection by the FULL CTE displacement (HST positions -> X_c/Y_c, X_mat rebuilt) and
    return the per-image transform change B that keeps the alignment stars' fit unchanged up to the non-representable
    part: X(x - delta) (r + B) ~= X(x) r - delta_perp.  In frozen-r mode the caller adds B to the frozen frame (the
    CTE the upstream alignment absorbed); anchors (no stellar CTE) then sit in the undistorted frame."""
    from bp3m.astro_utils import build_X_matrices
    nr = solver.N_R; dr = np.zeros_like(r); n_img = 0
    for img in image_names:
        d = solver._img_data.get(img); p = per.get(img)
        if d is None or p is None:
            continue
        sel = p['ok'] & ~p['anchor'] & (p['gi'] >= 0)
        dXc = np.zeros(d['n']); dYc = np.zeros(d['n'])
        if sel.any():
            Draw = np.zeros((d['n'], 2, ngrp * 2 * nf)); Draw[sel] = _design_det(p, nf, ngrp, sel)
            dsky, M = _to_sky(solver, img, r, p, Draw, np.ones(d['n'], bool))
            dsky = np.einsum('nik,k->ni', dsky, gamma)                            # full sky-frame displacement
            dgdc = np.einsum('ij,nj->ni', np.linalg.inv(M), dsky)
            dXc, dYc = dgdc[:, 0], dgdc[:, 1]
            P = _projector(solver, img, r, d, (z_weights or {}).get(img))
            if P is not None:
                Xf, Wf = P['X'][P['uf']], P['W'][P['uf']]
                j = solver.image_names.index(img)
                dr[j * nr:j * nr + nr] = P['Hi'] @ np.einsum('nia,nij,nj->a', Xf, Wf, dsky[P['uf']])
            n_img += 1
        d['X_c'] = d['X_c_nocte'] - dXc; d['Y_c'] = d['Y_c_nocte'] - dYc
        X = d['X_mat']
        Xn = build_X_matrices(d['X_c'], d['Y_c'], X[:, 0, 4], X[:, 0, 5], X[:, 1, 4], X[:, 1, 5], poly_order=solver.poly_order)
        if getattr(solver, 'fit_chip_offset', False):
            Xn = solver._append_chip_cols(Xn, d.get('chip_hi'))
        d['X_mat'] = Xn
        if stats is not None:
            stats.append(np.hypot(dXc[sel], dYc[sel]) if sel.any() else np.zeros(0))
    return n_img, dr


def _fit_increment(solver, image_names, per, r, a_arr, member_mask, nf, ngrp, z_weights=None, sigma_pm=None,
                   sigma_plx_tot=None, ridge=1e-6):
    """gamma increment from the current residuals, with every eligible star's 5 astrometric parameters marginalised
    (Schur complement).  The star blocks are built HERE from the same z-weighted precisions as the gamma terms plus
    the production priors (Gaia prior, 2p diffuse prior for non-members, population PM/parallax prior for members,
    as _joint_solve_pop), so the reduced system is positive definite by construction.  A CTE shift constant in time
    for a star is absorbed by its position, a time-growing one by its PM unless the PM prior forbids it."""
    ngam = ngrp * 2 * nf
    H = np.zeros((ngam, ngam)); g = np.zeros(ngam); nuse = np.zeros(ngrp, int)
    elig = np.flatnonzero(member_mask)
    for img in image_names:
        d = solver._img_data.get(img)
        if d is not None:
            elig = np.union1d(elig, d['sidx'][d['use_for_fit']])
    cpos = np.full(solver.n_stars, -1); cpos[elig] = np.arange(len(elig))
    cross = np.zeros((len(elig), ngam, 5)); Hd = np.zeros((len(elig), 5, 5))
    nr = solver.N_R
    for img in image_names:
        d = solver._img_data.get(img); p = per.get(img)
        if d is None:
            continue
        z = z_weights.get(img) if z_weights else None
        active = d['use_for_fit'] | d.get('use_for_astrom', d['use_for_fit'])
        j = solver.image_names.index(img); rj = r[j * nr:j * nr + nr]
        W = np.linalg.inv(solver._compute_Cs(img, rj))
        if z is not None:
            W = W * z[:, None, None]
        # data precision of every eligible star from ALL its active detections (as the solve sees them)
        ea = active & (cpos[d['sidx']] >= 0)
        if ea.any():
            JU = d['JU'][ea]
            np.add.at(Hd, cpos[d['sidx'][ea]], np.einsum('nki,nkl,nlj->nij', JU, W[ea], JU))
        if p is None:
            continue
        P = _projector(solver, img, r, d, z)
        if P is None:
            continue
        sel = p['ok'] & ~p['anchor'] & (p['gi'] >= 0) & active & (member_mask[d['sidx']] | d['use_for_fit'])
        if sel.sum() < 5:
            continue
        pred = np.einsum('nij,j->ni', d['X_mat'], rj) - np.einsum('nij,nj->ni', d['JU'], a_arr[d['sidx']])
        ed = solver._ed_disp(img, r)
        if not np.isscalar(ed):
            pred = pred + ed
        res = d['xys'] - pred                                       # = -(sky CTE displacement) + noise
        Draw = np.zeros((d['n'], 2, ngam)); okd = p['ok'] & ~p['anchor'] & (p['gi'] >= 0)
        Draw[okd] = _design_det(p, nf, ngrp, okd)
        Dsky, _ = _to_sky(solver, img, r, p, Draw, np.ones(d['n'], bool))
        Dp = -_project(P, Dsky)                                     # residual = -D_perp gamma + JU dv
        Ds, Ws, rs = Dp[sel], W[sel], res[sel]
        H += np.einsum('nia,nij,njb->ab', Ds, Ws, Ds)
        g += np.einsum('nia,nij,nj->a', Ds, Ws, rs)
        np.add.at(nuse, p['gi'][sel], 1)
        np.add.at(cross, cpos[d['sidx'][sel]], np.einsum('nia,nij,njk->nak', Ds, Ws, d['JU'][sel]))
    if len(elig):
        Hv = solver.C_survey_inv[elig].copy() + Hd
        mem = member_mask[elig]
        nm2p = ~mem & (solver._C_VG_inv_per_star[elig, 2] > 0)
        for k in range(5):
            Hv[nm2p, k, k] += solver._C_VG_inv_per_star[elig[nm2p], k]
        if sigma_pm is not None:
            Hv[mem, 2, 2] += sigma_pm ** -2; Hv[mem, 3, 3] += sigma_pm ** -2
        if sigma_plx_tot is not None:
            Hv[mem, 4, 4] += sigma_plx_tot ** -2
        live = np.abs(cross).reshape(len(elig), -1).max(1) > 0
        tr = np.trace(Hv[live], axis1=1, axis2=2)
        Hvi = np.linalg.inv(Hv[live] + (1e-12 * tr)[:, None, None] * np.eye(5))
        H -= np.einsum('nak,nkl,nbl->ab', cross[live], Hvi, cross[live])
    H = 0.5 * (H + H.T)
    lam = ridge * max(np.trace(H), 1e-30) / ngam
    C = np.linalg.inv(H + lam * np.eye(ngam))
    return C @ g, C, nuse


def run_cte_v2(solver, image_names, stars_per_image, star_id_to_idx, solve_fn, member_sidx, mu_pop, r, a_arr,
               C_vT=None, anchor_sidx=None, sigma_pm=None, sigma_plx_tot=None, fix_r=False, z_weights=None, n_iter=5, time_order=1, output_dir=None, tol_mu=1e-4):
    """Returns (r, mu_pop, C_shared, C_vT, a_arr, info).  solve_fn(member_sidx, mu, r, fix_r_arg, z_weights_arg)
    is the caller's pop-fit solve (run_pop_fit._solve) so every prior/anchor/weight is the production one."""
    t_start = time.time()
    per, groups, nf, m_med, m_sd = _setup(solver, image_names, stars_per_image, star_id_to_idx, anchor_sidx, time_order)
    ngrp = len(groups)
    print("\n" + "-" * 60)
    print(f"  CTE phase v2: groups {groups}, {nf} basis terms x 2 directions per group, time_order={time_order}, "
          f"{len(per)}/{len(image_names)} images with CTE geometry; mag norm ({m_med:.2f}, {m_sd:.2f}); official-GDC "
          f"Jacobian per detection for {sum(p['n_jac'] for p in per.values())} detections (others: per-chip linear fit)")
    if ngrp == 0:
        print("  CTE phase v2: no ACS/WFC or WFC3/UVIS detections with raw coordinates -- skipped")
        return r, mu_pop, None, None, a_arr, dict(skipped=True)
    member_mask = np.zeros(solver.n_stars, bool); member_mask[np.asarray(member_sidx, int)] = True
    gamma = np.zeros(ngrp * 2 * nf); C_gam = None; hist = [dict(iter=0, mu=[float(mu_pop[0]), float(mu_pop[1])])]
    C_shared = None
    r_base = np.array(r, float).copy()          # frozen frame (fix_r) or current start (free r)
    for it in range(1, n_iter + 1):
        dg, C_gam, nuse = _fit_increment(solver, image_names, per, r, a_arr, member_mask, nf, ngrp, z_weights=z_weights,
                                         sigma_pm=sigma_pm, sigma_plx_tot=sigma_plx_tot)
        gamma = gamma + dg
        stats = []
        n_img, dr = _apply(solver, image_names, per, r_base if fix_r else r, gamma, nf, ngrp, stats, z_weights)
        mag = np.concatenate(stats) if stats else np.zeros(1)
        if np.percentile(mag, 99) > MAX_SHIFT_PX:              # divergence guard: revert to the no-CTE solution
            print(f"  WARNING: CTE v2 iteration {it} would shift detections by p99 {np.percentile(mag, 99):.3f} px "
                  f"(> {MAX_SHIFT_PX} px) -- diverging; reverting to the uncorrected positions and frame")
            gamma = np.zeros_like(gamma); _apply(solver, image_names, per, r_base, gamma, nf, ngrp, None, z_weights)
            r = r_base
            _, mu_pop, C_shared, C_vT, a_arr, _, _ = solve_fn(member_sidx, mu_pop, r, fix_r, z_weights)
            hist.append(dict(iter=it, reverted=True)); break
        r_use = r_base + dr if fix_r else r
        r_new, mu_new, C_shared, C_vT, a_arr, _, _ = solve_fn(member_sidx, mu_pop, r_use, fix_r, z_weights)
        dmu = float(np.max(np.abs(np.asarray(mu_new) - np.asarray(mu_pop))))
        r = r_use if fix_r else r_new
        mu_pop = mu_new
        hist.append(dict(iter=it, mu=[float(mu_pop[0]), float(mu_pop[1])], dmu=dmu, max_dgamma=float(np.max(np.abs(dg))),
                         n_det_used=nuse.tolist(), shift_median_px=float(np.median(mag)), shift_p99_px=float(np.percentile(mag, 99))))
        print(f"    CTE iter {it}/{n_iter}: mu_pop=({mu_pop[0]:+.4f}, {mu_pop[1]:+.4f})  dmu={dmu:.2e}  "
              f"|dgamma|max={np.max(np.abs(dg)):.2e}  applied shift median {np.median(mag):.4f} px, p99 "
              f"{np.percentile(mag, 99):.4f} px on {n_img} images; dets/group {dict(zip(groups, nuse.tolist()))}")
        if dmu < tol_mu and it > 1:
            break
    sg = np.sqrt(np.diag(C_gam)) if C_gam is not None else np.zeros_like(gamma)
    info = dict(groups=groups, n_basis=nf, time_order=time_order, basis='yt*[1,xt] x [1,m,m^2] x [1,tau]',
                t0={'ACS': 2002.165, 'UVIS': 2009.37}, tau_unit_yr=10.0, y_readout={'top': _Y_READOUT_TOP, 'bot': _Y_READOUT_BOT},
                y_split={'ACS': 2048.0, 'UVIS': 2047.0}, mag_norm=[m_med, m_sd], gamma=gamma.tolist(), sigma_gamma=sg.tolist(),
                layout='[group][x block | y block][basis]', history=hist, seconds=round(time.time() - t_start, 1),
                note='gamma fitted on members + alignment stars (anchors and diffuse HST-only non-members excluded); '
                     'displacements projected off each image alignment design; correction = minus the toward-readout shift')
    if output_dir is not None:
        od = Path(output_dir); od.mkdir(parents=True, exist_ok=True)
        (od / 'cte_v2.json').write_text(json.dumps(info, indent=1))
        np.savez(od / 'cte_v2_params.npz', gamma=gamma, C_gamma=C_gam if C_gam is not None else np.zeros((1, 1)), groups=np.array(groups),
                 mag_norm=np.array([m_med, m_sd]), time_order=time_order)
        _plot_trends(solver, image_names, per, r, a_arr, member_mask, od / 'plots' / 'cte_v2_trends.png')
    return r, mu_pop, C_shared, C_vT, a_arr, info


def _plot_trends(solver, image_names, per, r, a_arr, member_mask, path):
    """Post-correction residual along the readout direction vs readout distance, per group and magnitude tercile."""
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        rows = []
        nr = solver.N_R
        for img in image_names:
            d = solver._img_data.get(img); p = per.get(img)
            if d is None or p is None:
                continue
            j = solver.image_names.index(img); rj = r[j * nr:j * nr + nr]
            pred = np.einsum('nij,j->ni', d['X_mat'], rj) - np.einsum('nij,nj->ni', d['JU'], a_arr[d['sidx']])
            res = d['xys'] - pred
            M = np.array([[rj[0], rj[1]], [rj[2], rj[3]]])
            rdet = res @ np.linalg.inv(M).T
            sel = p['ok'] & ~p['anchor'] & (p['gi'] >= 0) & member_mask[d['sidx']]
            sgn = np.where(np.char.endswith(p['grp'].astype(str), '1'), 1.0, -1.0)
            for k in np.where(sel)[0]:
                rows.append((p['grp'][k], p['year'], p['yt'][k], p['mag'][k], rdet[k, 1] * sgn[k]))
        if not rows:
            return
        g = np.array([x[0] for x in rows]); yr = np.array([x[1] for x in rows]); yt = np.array([x[2] for x in rows])
        mg = np.array([x[3] for x in rows]); dy = np.array([x[4] for x in rows])
        grs = sorted(set(g)); fig, axs = plt.subplots(1, len(grs), figsize=(4.5 * len(grs), 3.6), squeeze=False)
        for ax, gg in zip(axs[0], grs):
            m = g == gg
            q = np.nanpercentile(mg[m], [33, 67])
            for lab, mm in [('bright', mg < q[0]), ('mid', (mg >= q[0]) & (mg < q[1])), ('faint', mg >= q[1])]:
                s = m & mm
                bins = np.linspace(0, 1, 9); c = 0.5 * (bins[1:] + bins[:-1])
                med = [np.nanmedian(dy[s & (yt >= a) & (yt < b)]) if (s & (yt >= a) & (yt < b)).sum() > 20 else np.nan
                       for a, b in zip(bins[:-1], bins[1:])]
                ax.plot(c, med, 'o-', label=lab)
            ax.axhline(0, color='k', lw=0.6); ax.set_title(f'{gg} (members, after CTE v2)')
            ax.set_xlabel('readout distance yt'); ax.set_ylabel('residual toward readout [px]'); ax.legend(fontsize=7)
        fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)
    except Exception as exc:
        print(f"  WARNING: CTE v2 trend plot failed: {exc}")
