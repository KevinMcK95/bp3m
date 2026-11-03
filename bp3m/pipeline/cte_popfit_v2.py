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


def _setup(solver, image_names, stars_per_image, star_id_to_idx, anchor_sidx, time_order, group_by='chip'):
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
        # group_by (2026-10-15): 'epoch' / 'visit' give each epoch (calendar half-year) / visit (rootname[:6]) its own gamma
        # block per chip, shared by all its sibling images -- CTE residuals change between programmes (Sextans A brightness
        # split: -7..-32 mpx in 2022, -55..-120 mpx in 2025 ACS) while the default 'chip' block is one per field.
        if group_by == 'epoch':
            g = np.array([f'{x}_{np.floor(_year(meta["hst_time_mjd"]) * 2) / 2:.1f}' for x in g], dtype=object)
        elif group_by == 'visit':
            g = np.array([f'{x}_{str(img)[:6]}' for x in g], dtype=object)
        elif group_by == 'filter':      # 2026-11-03: per camera chip x filter (hard rule: CTE models are filter-dependent)
            filt = str(meta.get('filter', '')).upper()
            if not filt:
                raise ValueError(f'CTE v2 group_by=filter: image {img} has no filter in its metadata')
            g = np.array([f'{x}_{filt}' for x in g], dtype=object)
        elif group_by != 'chip':
            raise ValueError(f'CTE v2: unknown group_by {group_by!r}')
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


def _apply(solver, image_names, per, r, gamma, nf, ngrp, stats=None, z_weights=None, const_mask=None):
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
        # stars: the full model (time-constant geometry-like block + time-growing CTE); galaxy anchors: only the
        # time-constant block (chip geometry applies to every source; stellar CTE does not -- galaxies get their own
        # measured differential CTE upstream).  user 2026-10-06
        sel = p['ok'] & ~p['anchor'] & (p['gi'] >= 0)
        anc = p['ok'] & p['anchor'] & (p['gi'] >= 0) if const_mask is not None else np.zeros(d['n'], bool)
        dXc = np.zeros(d['n']); dYc = np.zeros(d['n'])
        if sel.any() or anc.any():
            both = sel | anc
            Draw = np.zeros((d['n'], 2, ngrp * 2 * nf)); Draw[both] = _design_det(p, nf, ngrp, both)
            Dsky, M = _to_sky(solver, img, r, p, Draw, np.ones(d['n'], bool))
            dsky = np.einsum('nik,k->ni', Dsky, gamma)                            # full sky-frame displacement
            if anc.any():
                dsky[anc] = np.einsum('nik,k->ni', Dsky[anc], gamma * const_mask)
            dsky[~both] = 0.0
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
                   sigma_plx_tot=None, gamma_now=None, sigma_gamma=0.03, ridge=1e-6, C_pop_prior_inv=None):
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
        tr = np.trace(Hv, axis1=1, axis2=2)
        Hvi = np.linalg.inv(Hv + (1e-12 * tr)[:, None, None] * np.eye(5))
        H -= np.einsum('nak,nkl,nbl->ab', cross, Hvi, cross)
        # mu_pop marginalised too (user 10-06: faster convergence): members' PM prior N(mu, sigma_pm) couples every
        # member to mu.  theta = (gamma, mu); H_mu,v = -sigma_pm^-2 on the PM components of each member.
        if sigma_pm is not None and mem.any():
            Cm = np.zeros((2, 5)); Cm[0, 2] = Cm[1, 3] = -sigma_pm ** -2
            Hm = Hvi[mem]
            B = -np.einsum('nak,nkl,bl->ab', cross[mem], Hm, Cm)                 # (ngam, 2)
            Dm = mem.sum() * sigma_pm ** -2 * np.eye(2) - np.einsum('ak,nkl,bl->ab', Cm, Hm, Cm)
            if C_pop_prior_inv is not None:
                Dm = Dm + np.asarray(C_pop_prior_inv, float)
            H -= B @ np.linalg.solve(Dm, B.T)
    H = 0.5 * (H + H.T)
    # Gaussian prior on the TOTAL gamma (sigma_gamma px per coefficient).  With star positions marginalised, a CTE
    # pattern that is constant in time for each star is (nearly) degenerate whenever pointings repeat; without the
    # prior the step fills those directions with noise and the outer loop runs away (Leo_P 10-06).
    Lp = np.eye(ngam) / sigma_gamma ** 2
    if gamma_now is not None:
        g = g - Lp @ gamma_now
    lam = ridge * max(np.trace(H), 1e-30) / ngam
    C = np.linalg.inv(H + Lp + lam * np.eye(ngam))
    return C @ g, C, nuse


def _conditioned(solve_fn, member_sidx, mu, r, z_weights):
    """Star posteriors (C_vT, a) conditioned on exactly (r, mu).  _joint_solve_pop returns `a` computed at its INPUT
    (r, mu) while returning UPDATED (r, mu); feeding that stale `a` against the updated frame put every frame / mu change
    into the CTE residuals (the 10-06 runaway).  A fixed-r call at the current values gives the consistent `a`."""
    _, _, _, C_vT, a, _, _ = solve_fn(member_sidx, mu, r, True, z_weights)
    return C_vT, a


def _run_fixed_point(solver, image_names, stars_per_image, star_id_to_idx, solve_fn, member_sidx, mu_pop, r, a_arr,
               C_vT=None, anchor_sidx=None, sigma_pm=None, sigma_plx_tot=None, sigma_gamma=0.03, C_pop_prior_inv=None, fix_r=False, z_weights=None, n_iter=5, time_order=1, output_dir=None, tol_mu=1e-4, group_by='chip'):
    """LEGACY outer fixed-point loop (2026-10-06 .. 11-02; --cte_v2_method fixed_point).  Kept only to reproduce the
    earlier results: it is unstable whenever its model Hessian under-estimates the true curvature by more than x2
    (Sculptor 10-29; see run_cte_v2 / _newton).  Returns (r, mu_pop, C_shared, C_vT, a_arr, info).
    solve_fn(member_sidx, mu, r, fix_r_arg, z_weights_arg) is the caller's pop-fit solve."""
    t_start = time.time()
    per, groups, nf, m_med, m_sd = _setup(solver, image_names, stars_per_image, star_id_to_idx, anchor_sidx, time_order, group_by)
    ngrp = len(groups)
    print("\n" + "-" * 60)
    print(f"  CTE phase v2: groups {groups}, {nf} basis terms x 2 directions per group, time_order={time_order}, "
          f"{len(per)}/{len(image_names)} images with CTE geometry; mag norm ({m_med:.2f}, {m_sd:.2f}); official-GDC "
          f"Jacobian per detection for {sum(p['n_jac'] for p in per.values())} detections (others: per-chip linear fit)")
    if ngrp == 0:
        print("  CTE phase v2: no ACS/WFC or WFC3/UVIS detections with raw coordinates -- skipped")
        return r, mu_pop, None, None, a_arr, dict(skipped=True)
    member_mask = np.zeros(solver.n_stars, bool); member_mask[np.asarray(member_sidx, int)] = True
    import os as _os
    C_vT, a_arr = _conditioned(solve_fn, member_sidx, mu_pop, r, z_weights)
    if _os.environ.get('CTE_V2_SELFTEST') == '1':
        return _selftest(solver, image_names, per, groups, nf, solve_fn, member_sidx, mu_pop, r, a_arr, member_mask,
                         fix_r, z_weights, sigma_pm, sigma_plx_tot)
    gamma = np.zeros(ngrp * 2 * nf); C_gam = None; hist = [dict(iter=0, mu=[float(mu_pop[0]), float(mu_pop[1])])]
    n_const = nf // (time_order + 1)                         # basis order: [time-constant block | tau block]
    const_mask = ((np.arange(ngrp * 2 * nf) % nf) < n_const).astype(float)
    C_shared = None
    r_base = np.array(r, float).copy()          # frozen frame (fix_r) or current start (free r)
    for it in range(1, n_iter + 1):
        dg, C_gam, nuse = _fit_increment(solver, image_names, per, r, a_arr, member_mask, nf, ngrp, z_weights=z_weights,
                                         sigma_pm=sigma_pm, sigma_plx_tot=sigma_plx_tot, gamma_now=gamma,
                                         sigma_gamma=sigma_gamma, C_pop_prior_inv=C_pop_prior_inv)
        gamma = gamma + dg
        stats = []
        n_img, dr = _apply(solver, image_names, per, r_base if fix_r else r, gamma, nf, ngrp, stats, z_weights, const_mask)
        mag = np.concatenate(stats) if stats else np.zeros(1)
        if np.percentile(mag, 99) > MAX_SHIFT_PX:              # divergence guard: revert to the no-CTE solution
            print(f"  WARNING: CTE v2 iteration {it} would shift detections by p99 {np.percentile(mag, 99):.3f} px "
                  f"(> {MAX_SHIFT_PX} px) -- diverging; reverting to the uncorrected positions and frame")
            gamma = np.zeros_like(gamma); _apply(solver, image_names, per, r_base, gamma, nf, ngrp, None, z_weights, const_mask)
            r = r_base
            _, mu_pop, C_shared, _, _, _, _ = solve_fn(member_sidx, mu_pop, r, fix_r, z_weights)
            C_vT, a_arr = _conditioned(solve_fn, member_sidx, mu_pop, r, z_weights)
            hist.append(dict(iter=it, reverted=True)); break
        r_use = r_base + dr if fix_r else r
        r_new, mu_new, C_shared, _, _, _, _ = solve_fn(member_sidx, mu_pop, r_use, fix_r, z_weights)
        dmu = float(np.max(np.abs(np.asarray(mu_new) - np.asarray(mu_pop))))
        r = r_use if fix_r else r_new
        mu_pop = mu_new
        C_vT, a_arr = _conditioned(solve_fn, member_sidx, mu_pop, r, z_weights)
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
                layout='[group][x block | y block][basis]', sigma_gamma_prior_px=sigma_gamma, n_const_basis=n_const,
                anchors_get='time-constant block only', mu_marginalised_in_gamma_step=bool(sigma_pm is not None), history=hist, seconds=round(time.time() - t_start, 1),
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


def _selftest(solver, image_names, per, groups, nf, solve_fn, member_sidx, mu_pop, r, a_arr, member_mask, fix_r,
              z_weights, sigma_pm, sigma_plx_tot):
    """CTE_V2_SELFTEST=1: inject a known gamma into the HST positions of the solved state (with the frozen frame absorbing
    the representable part, as the upstream alignment would), re-solve the stars, and check (1) one gamma step recovers
    gamma_true, (2) after applying it a second step is ~0.  Prints and returns without changing the solution."""
    from bp3m.astro_utils import build_X_matrices
    ngrp = len(groups); ngam = ngrp * 2 * nf
    g_true = np.zeros(ngam)
    for gi in range(ngrp):
        g_true[gi * 2 * nf + nf + 0] = 0.05     # y block, yt term
        g_true[gi * 2 * nf + nf + 1] = 0.03     # y block, yt * m'
    print(f"  CTE v2 SELFTEST: injecting gamma_true (y: yt=0.05 px, yt*m'=0.03 px per group), fix_r={fix_r}")
    keep = {img: (solver._img_data[img]['X_c_nocte'].copy(), solver._img_data[img]['Y_c_nocte'].copy(),
                  solver._img_data[img]['X_mat'].copy()) for img in per if img in solver._img_data}
    r0 = np.array(r, float).copy()
    # baseline: the step on the UNinjected state (the data already contain real CTE)
    g0, _, _ = _fit_increment(solver, image_names, per, r0, a_arr, member_mask, nf, ngrp, z_weights=z_weights,
                              sigma_pm=sigma_pm, sigma_plx_tot=sigma_plx_tot, gamma_now=np.zeros(ngam), sigma_gamma=1.0)
    # positions with CTE: X_c_nocte + delta_true (the apply routine subtracts gamma from X_c_nocte, so a NEGATIVE apply
    # of g_true adds it), then the frame absorbs the representable part: r_true = r0 + dr
    n_img, dr = _apply(solver, image_names, per, r0, -g_true, nf, ngrp, None, z_weights)
    for img in per:
        d = solver._img_data.get(img)
        if d is not None:
            d['X_c_nocte'] = d['X_c'].copy(); d['Y_c_nocte'] = d['Y_c'].copy()
    r_inj = r0 + dr if fix_r else r0
    r1, mu1, _, _, _, _, _ = solve_fn(member_sidx, mu_pop, r_inj, fix_r, z_weights)
    r_inj = r_inj if fix_r else r1
    _, a1 = _conditioned(solve_fn, member_sidx, mu1, r_inj, z_weights)
    g1, C1, nuse = _fit_increment(solver, image_names, per, r_inj, a1, member_mask, nf, ngrp, z_weights=z_weights,
                                  sigma_pm=sigma_pm, sigma_plx_tot=sigma_plx_tot, gamma_now=np.zeros(ngam), sigma_gamma=1.0)
    sg = np.sqrt(np.diag(C1)); dd = g1 - g0
    for gi, gname in enumerate(groups):
        o = gi * 2 * nf + nf; blk = np.delete(np.arange(gi * 2 * nf, (gi + 1) * 2 * nf), [nf, nf + 1])
        print(f"    {gname:6s} n={nuse[gi]:6d}  injected-minus-baseline  y[yt]: true 0.050 rec {dd[o]:+.4f} +- {sg[o]:.4f}   "
              f"y[yt*m]: true 0.030 rec {dd[o + 1]:+.4f} +- {sg[o + 1]:.4f}   other terms max|d|/sigma {np.max(np.abs(dd[blk] / sg[blk])):.1f}")
    # apply the recovered gamma and take a second step
    n_img, dr2 = _apply(solver, image_names, per, r_inj, g1, nf, ngrp, None, z_weights)
    r_cor = r_inj + dr2 if fix_r else r_inj
    r2, mu2, _, _, _, _, _ = solve_fn(member_sidx, mu1, r_cor, fix_r, z_weights)
    r_cor = r_cor if fix_r else r2
    _, a2 = _conditioned(solve_fn, member_sidx, mu2, r_cor, z_weights)
    g2, _, _ = _fit_increment(solver, image_names, per, r_cor, a2, member_mask, nf, ngrp, z_weights=z_weights,
                              sigma_pm=sigma_pm, sigma_plx_tot=sigma_plx_tot, gamma_now=g1, sigma_gamma=1.0)
    z2 = np.abs(g2) / sg
    print(f"    second step after applying the recovered gamma: max|dgamma|/sigma = {z2.max():.2f}, median {np.median(z2):.2f} "
          f"(consistent loop: << 1); max|dgamma| on well-measured terms (sigma<0.02 px) = "
          f"{np.max(np.abs(g2[sg < 0.02])) if (sg < 0.02).any() else float('nan'):.4f} px; "
          f"mu: injected-state {mu1[0]:+.4f},{mu1[1]:+.4f} -> corrected {mu2[0]:+.4f},{mu2[1]:+.4f}")
    # restore
    for img, (xc, yc, X) in keep.items():
        d = solver._img_data[img]; d['X_c_nocte'], d['Y_c_nocte'] = xc, yc; d['X_c'], d['Y_c'] = xc.copy(), yc.copy()
        Xn = build_X_matrices(xc, yc, X[:, 0, 4], X[:, 0, 5], X[:, 1, 4], X[:, 1, 5], poly_order=solver.poly_order)
        if getattr(solver, 'fit_chip_offset', False):
            Xn = solver._append_chip_cols(Xn, d.get('chip_hi'))
        d['X_mat'] = Xn
    _, mu_r, C_sh, C_v, a_r, _, _ = solve_fn(member_sidx, mu_pop, r0, fix_r, z_weights)
    return r0, mu_r, C_sh, C_v, a_r, dict(skipped=True, selftest=True)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Exact linear-response Newton solve (2026-11-03; default --cte_v2_method newton)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#
# Why the fixed-point loop diverged (Sculptor 10-29).  Each outer iteration took gamma += C_model (g - Lp gamma) with a
# HAND-BUILT reduced Hessian: the per-image transform was removed by projecting the displacement design off X with the
# stars held fixed and WITHOUT the image priors (the production 8p priors -- scale 1e-4, skew 5e-5, chip offset 0.015 px
# -- are as tight as the CTE signal itself), the stars were then marginalised with UNprojected star designs, and the field
# modes were absent.  Whenever that model curvature is smaller than the true curvature of the production solve by more
# than x2 in some direction, the iteration map I - C_model (A + Lp) has an eigenvalue < -1 and the loop oscillates with
# growing amplitude (time order 0: x(-2.2) per iteration); the p99 guard then reverted everything.
#
# What replaces it.  With membership, z-weights, alpha and the linearisation frozen (as the CTE phase always did), the
# production pop-fit solve is LINEAR in the detector positions, so the stationarity condition for gamma
#     G(gamma) + Lp gamma = 0,     G(gamma) = sum_sel D_eff^T W res(gamma)          (res at the re-solved r, mu, z, stars)
# is affine in gamma.  Its Jacobian A = dG/dgamma is measured exactly, column by column, by applying one basis shift at a
# time and re-running THE PRODUCTION SOLVE (same priors, anchors, z-weights, Gaia field modes, frozen / free r).  The
# Newton solution gamma* = -(A + Lp)^-1 G(0) is then exact (one step), and is verified by re-solving at gamma* (residual
# step < tol * sigma_gamma, a few refinement steps absorb the tiny bilinear r x gamma term).  No outer fixed point, so
# nothing can drift: directions the data do not constrain (degenerate with the 8p transforms, the star parameters or
# mu_pop) stay at the prior mean 0 with their prior width, and that width is PROPAGATED into mu_pop through the measured
# response S = dmu/dgamma (C_mu += S C_gamma S^T: the joint posterior marginal, no double counting -- gamma, r, mu and
# the stars are one linear-Gaussian model fitted once to the same detections).
#
# Frozen-r mode (pop-fit v2 --frozen_alignment): the frame follows the corrected positions through the per-image
# alignment response WITH the image priors, r = r_base + B gamma, B_j = (X^T W X + C_r_prior^-1)^-1 X^T W D (alignment
# detections, stars fixed); the gradient uses D_eff = D - X B accordingly.  Free-r mode (pop-fit v1/v2 joint phases):
# r is re-solved, D_eff = D.

def _term_names(nf, time_order):
    sp = ['yt', 'yt*xt']; mg = ['', '*m', '*m2']; tm = [''] + (['*tau'] if time_order >= 1 else [])
    names = [f'{s}{m}{t}' for t in tm for s in sp for m in mg]
    assert len(names) == nf
    return names


def _prior_cov(groups, nf, sigma_gamma, sigma_filter, group_by):
    """Prior covariance of gamma.  group_by='filter': hierarchical, gamma_(chip,filter) = gamma_chip + delta_filter with
    gamma_chip ~ N(0, sigma_gamma^2), delta_filter ~ N(0, sigma_filter^2) per coefficient (filters of the same camera chip
    are correlated a priori, so a filter with one epoch borrows the shared time dependence).  Otherwise independent
    N(0, sigma_gamma^2) per coefficient."""
    ng = len(groups); nb = 2 * nf; P = np.zeros((ng * nb, ng * nb))
    base = [g.split('_')[0] for g in groups]
    for i in range(ng):
        for j in range(ng):
            if group_by == 'filter':
                v = (sigma_gamma ** 2 if base[i] == base[j] else 0.0) + (sigma_filter ** 2 if i == j else 0.0)
            else:
                v = sigma_gamma ** 2 if i == j else 0.0
            if v:
                P[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb] += v * np.eye(nb)
    return P


class _NewtonCTE:
    """State + helpers for the exact solve (see the block comment above)."""

    def __init__(self, solver, image_names, per, groups, nf, time_order, solve_fn, member_sidx, mu0, r0, fix_r, z_weights,
                 const_mask, sigma_pm=None, fast_condition=True):
        from bp3m.astro_utils import build_X_matrices
        self.sigma_pm = sigma_pm
        self.fast_condition = bool(fast_condition and (sigma_pm is not None or not len(member_sidx)))
        self.s, self.names, self.per, self.groups, self.nf = solver, image_names, per, groups, nf
        self.ngrp = len(groups); self.ngam = self.ngrp * 2 * nf; self.time_order = time_order
        self.solve_fn, self.members, self.fix_r, self.zw = solve_fn, np.asarray(member_sidx, int), bool(fix_r), z_weights
        self.mu_start = np.array(mu0, float).copy(); self.r_base = np.array(r0, float).copy()
        self.const_mask = const_mask; self.nr = solver.N_R
        self.n_gz = int(getattr(solver, 'n_gz', 0) or 0)
        self.z0 = None if not self.n_gz else np.array(solver.gaia_z, float).copy()
        self.z0_cov = None if not self.n_gz else np.array(getattr(solver, 'gaia_z_cov', np.eye(self.n_gz)), float).copy()
        self.n_solves = 0
        member_mask = np.zeros(solver.n_stars, bool); member_mask[self.members] = True
        self.member_mask = member_mask
        # static per-image pieces: rebuild check, weights, selections, raw->GDC Jacobian rows
        for img, p in per.items():
            d = solver._img_data[img]
            X = d['X_mat']
            Xn = build_X_matrices(d['X_c_nocte'], d['Y_c_nocte'], X[:, 0, 4], X[:, 0, 5], X[:, 1, 4], X[:, 1, 5],
                                  poly_order=solver.poly_order)
            if getattr(solver, 'fit_chip_offset', False):
                Xn = solver._append_chip_cols(Xn, d.get('chip_hi'))
            if Xn.shape != X.shape or not np.array_equal(Xn, X):
                raise RuntimeError(f'CTE v2: rebuilding X_mat from X_c/Y_c does not reproduce the solver design for {img} '
                                   f'(max |diff| {np.max(np.abs(Xn - X)) if Xn.shape == X.shape else "shape"}); the '
                                   f'position shift would change more than the positions -- refusing to run')
            p['X_pointing'] = (X[:, 0, 4].copy(), X[:, 0, 5].copy(), X[:, 1, 4].copy(), X[:, 1, 5].copy())
            j = solver.image_names.index(img)
            W = np.linalg.inv(solver._compute_Cs(img, self.r_base[j * self.nr:(j + 1) * self.nr]))
            z = (z_weights or {}).get(img)
            if z is not None:
                W = W * np.asarray(z)[:, None, None]
            p['W'] = W
            active = d['use_for_fit'] | d.get('use_for_astrom', d['use_for_fit'])
            p['shift'] = p['ok'] & (p['gi'] >= 0)                                   # detections whose position moves
            p['gsel'] = p['shift'] & ~p['anchor'] & active & (member_mask[d['sidx']] | d['use_for_fit'])   # gamma data
            p['mask'] = np.where(p['anchor'][:, None], const_mask[None, :nf], 1.0)      # anchors: time-constant block only
        self.Bfr = self._frame_response() if self.fix_r else None

    # ---- design pieces -------------------------------------------------------------------------------------------
    def _M(self, img, r, rows):
        from bp3m.astro_utils import compute_poly_jacobian
        d = self.s._img_data[img]; j = self.s.image_names.index(img); rj = r[j * self.nr:(j + 1) * self.nr]
        return compute_poly_jacobian(rj, d['X_c'][rows], d['Y_c'][rows], self.s.poly_order)       # (n, 2, 2) sky/GDC

    def _accumulate(self, p, rows, V, out):
        """out[gamma] += sum_n F_n (x) V_n  for the raw-detector design (V: (n, 2) = D_raw^T-side vectors)."""
        nf = self.nf; gi = p['gi'][rows]; F = p['F'][rows] * p['mask'][rows]
        for k in np.unique(gi):
            m = gi == k; o = k * 2 * nf
            out[o:o + nf] += F[m].T @ V[m, 0]; out[o + nf:o + 2 * nf] += F[m].T @ V[m, 1]

    def _frame_response(self):
        """frozen-r: per-image B_j (nr x ngam) = alignment response (with image priors) to the CTE displacement."""
        B = {}
        for img, p in self.per.items():
            d = self.s._img_data[img]; uf = d['use_for_fit'] & p['shift']
            ufa = d['use_for_fit']
            X, W = d['X_mat'], p['W']
            H = np.einsum('nia,nij,njb->ab', X[ufa], W[ufa], X[ufa]) + d['C_r_prior_inv']
            rows = np.flatnonzero(uf)
            if len(rows) == 0:
                B[img] = np.zeros((self.nr, self.ngam)); continue
            L = np.einsum('nij,njk->nik', self._M(img, self.r_base, rows), p['Jg'][rows])        # raw -> sky
            T = np.einsum('nia,nij,njk->nak', X[rows], W[rows], L)                               # (n, nr, 2)
            Q = np.zeros((self.nr, self.ngam))
            for a in range(self.nr):
                col = np.zeros(self.ngam); self._accumulate(p, rows, T[:, a, :], col); Q[a] = col
            B[img] = np.linalg.solve(H, Q)
        return B

    def shift_positions(self, gamma):
        """X_c/Y_c = originals - Jg D_raw gamma (anchors: time-constant block); X_mat rebuilt.  Returns |shift| of the
        gamma-data detections (px, GDC frame)."""
        from bp3m.astro_utils import build_X_matrices
        nf = self.nf; mags = []
        for img, p in self.per.items():
            d = self.s._img_data[img]; rows = np.flatnonzero(p['shift'])
            dx = np.zeros(d['n']); dy = np.zeros(d['n'])
            if len(rows):
                gi = p['gi'][rows]; F = p['F'][rows] * p['mask'][rows]
                G3 = np.asarray(gamma, float).reshape(self.ngrp, 2, nf)
                gx, gy = G3[gi, 0, :], G3[gi, 1, :]
                draw = np.stack([np.einsum('nb,nb->n', F, gx), np.einsum('nb,nb->n', F, gy)], axis=1)
                dg = np.einsum('nij,nj->ni', p['Jg'][rows], draw)
                dx[rows], dy[rows] = dg[:, 0], dg[:, 1]
                mags.append(np.hypot(dg[:, 0], dg[:, 1])[p['gsel'][rows]])
            d['X_c'] = d['X_c_nocte'] - dx; d['Y_c'] = d['Y_c_nocte'] - dy
            Xn = build_X_matrices(d['X_c'], d['Y_c'], *p['X_pointing'], poly_order=self.s.poly_order)
            if getattr(self.s, 'fit_chip_offset', False):
                Xn = self.s._append_chip_cols(Xn, d.get('chip_hi'))
            d['X_mat'] = Xn
        return np.concatenate(mags) if mags else np.zeros(1)

    def frame(self, gamma):
        if not self.fix_r:
            return None
        r = self.r_base.copy()
        for img, Bj in self.Bfr.items():
            j = self.s.image_names.index(img); r[j * self.nr:(j + 1) * self.nr] += Bj @ gamma
        return r

    def gradient(self, r, a):
        """G = sum_sel D_eff^T W res,  res = xys - X r + JU a (the production solve's residual)."""
        G = np.zeros(self.ngam)
        for img, p in self.per.items():
            d = self.s._img_data[img]; rows = np.flatnonzero(p['gsel'])
            if not len(rows):
                continue
            j = self.s.image_names.index(img); rj = r[j * self.nr:(j + 1) * self.nr]
            X = d['X_mat'][rows]
            res = d['xys'][rows] - np.einsum('nij,j->ni', X, rj) + np.einsum('nij,nj->ni', d['JU'][rows], a[d['sidx'][rows]])
            u = np.einsum('nij,nj->ni', p['W'][rows], res)
            L = np.einsum('nij,njk->nik', self._M(img, r, rows), p['Jg'][rows])
            self._accumulate(p, rows, np.einsum('nji,nj->ni', L, u), G)
            if self.fix_r:
                G -= self.Bfr[img].T @ np.einsum('nia,ni->a', X, u)
        return G

    def condition(self, C_vT, a_in, K_img, dr, dmu, dz):
        s = self.s; dh = np.zeros_like(a_in)
        if np.any(dr):
            for j, img in enumerate(s.image_names):
                K = K_img.get(img) if K_img is not None else None
                if K is None:
                    continue
                d = s._img_data[img]; use = d['use_for_fit'] | d.get('use_for_astrom', d['use_for_fit'])
                drj = dr[j * self.nr:(j + 1) * self.nr]
                if use.any() and np.any(drj):
                    np.add.at(dh, d['sidx'][use], np.einsum('nik,k->ni', K[use], drj))
        if self.sigma_pm is not None and len(self.members):
            dh[self.members, 2] += self.sigma_pm ** -2 * dmu[0]; dh[self.members, 3] += self.sigma_pm ** -2 * dmu[1]
        if dz is not None:
            dh += np.einsum('nik,k->ni', s.gaia_Kz, dz)
        return a_in + np.einsum('nij,nj->ni', C_vT, dh)

    # ---- one production solve at a given gamma --------------------------------------------------------------------
    def evaluate(self, gamma, fix_frame_at=None):
        """Apply gamma, re-run the production solve from the reference state, return the re-solved state and G.
        fix_frame_at: keep r fixed there (diagnostic: information with the alignment known)."""
        s = self.s
        shifts = self.shift_positions(gamma)
        if self.n_gz:
            s.gaia_z = self.z0.copy(); s.gaia_z_cov = self.z0_cov.copy()
        if fix_frame_at is not None or self.fix_r:
            r_in = r_use = np.array(fix_frame_at if fix_frame_at is not None else self.frame(gamma), float)
            _, mu, C_sh, C_vT, a_in, _, K_img = self.solve_fn(self.members, self.mu_start, r_use, True, self.zw, True)
        else:
            r_in = self.r_base
            r_use, mu, C_sh, C_vT, a_in, _, K_img = self.solve_fn(self.members, self.mu_start, r_in, False, self.zw, True)
        z = None if not self.n_gz else np.array(s.gaia_z, float).copy()
        gl = getattr(s, '_gaia_last', None) if self.n_gz else None
        if self.fast_condition:
            # the solve returns the star posteriors at its INPUT (r, mu, z); move them to the solved values with the solve's
            # own linear response (h += K dr + sigma_pm^-2 dmu [members] + K_z dz) -- identical to a fixed-r re-solve
            a = self.condition(C_vT, a_in, K_img, r_use - r_in, np.asarray(mu) - self.mu_start,
                               None if not self.n_gz else z - self.z0)
            self.n_solves += 1
        else:
            _, _, _, C_vT, a, _, _ = self.solve_fn(self.members, mu, r_use, True, self.zw, False)   # star posteriors at (r, mu, z)
            self.n_solves += 2
        return dict(gamma=np.array(gamma, float).copy(), r=r_use, mu=np.array(mu, float), C_shared=C_sh, C_vT=C_vT, a=a,
                    z=z, gaia_last=gl, G=self.gradient(r_use, a), shift_p50=float(np.median(shifts)),
                    shift_p99=float(np.percentile(shifts, 99)))


def _newton(ctx, P, probe_px=0.01, tol=1e-3, max_refine=6, diag=False, fp_kwargs=None, log=print):
    """Exact linear-response Newton solve.  Returns (state, info)."""
    ngam = ctx.ngam
    Lp = np.linalg.inv(P); Lp = 0.5 * (Lp + Lp.T)
    t0 = time.time()
    st0 = ctx.evaluate(np.zeros(ngam))
    ctx.r_base = st0['r'].copy() if not ctx.fix_r else ctx.r_base       # probes start at the gamma = 0 optimum
    ctx.mu_start = st0['mu'].copy()
    if ctx.n_gz:
        ctx.z0 = st0['z'].copy()
    st0 = ctx.evaluate(np.zeros(ngam))
    t1 = time.time()
    log(f"    CTE v2 newton: reference solve {t1 - t0:.1f} s; measuring the exact response to {ngam} basis shifts "
        f"({probe_px} px each) with the production solve ...")
    A = np.zeros((ngam, ngam)); S = np.zeros((2, ngam)); Rz = np.zeros((ctx.n_gz, ngam)) if ctx.n_gz else None
    A_fix = np.zeros((ngam, ngam)) if diag else None
    T_fp = np.zeros((ngam, ngam)) if diag else None
    if diag:
        st0f = ctx.evaluate(np.zeros(ngam), fix_frame_at=st0['r'])
        step0 = _fp_step(ctx, st0, np.zeros(ngam), fp_kwargs)
    for k in range(ngam):
        e = np.zeros(ngam); e[k] = probe_px
        st = ctx.evaluate(e)
        A[:, k] = (st['G'] - st0['G']) / probe_px
        S[:, k] = (st['mu'] - st0['mu']) / probe_px
        if ctx.n_gz:
            Rz[:, k] = (st['z'] - st0['z']) / probe_px
        if diag:
            A_fix[:, k] = (ctx.evaluate(e, fix_frame_at=st0['r'])['G'] - st0f['G']) / probe_px
            T_fp[:, k] = e / probe_px + (_fp_step(ctx, st, e, fp_kwargs) - step0) / probe_px
        if (k + 1) % max(1, ngam // 8) == 0:
            log(f"      probes {k + 1}/{ngam}  ({time.time() - t1:.0f} s)")
    H = A + Lp
    Hs = 0.5 * (H + H.T)
    asym = float(np.linalg.norm(A - A.T) / max(np.linalg.norm(A), 1e-300))
    ev = np.linalg.eigvalsh(Hs)
    if not np.all(ev > 0):
        raise RuntimeError(f'CTE v2 newton: (A + prior) is not positive definite (min eigenvalue {ev.min():.3e}); the '
                           f'production solve responds non-linearly or the selection is inconsistent -- refusing to apply')
    sig_prior = np.sqrt(np.diag(P))
    gamma = np.linalg.solve(H, -st0['G'])
    hist = []
    st = None
    for it in range(max_refine):
        st = ctx.evaluate(gamma)
        R = st['G'] + Lp @ gamma
        step = np.linalg.solve(H, -R)
        rel = float(np.max(np.abs(step) / sig_prior))
        hist.append(dict(iter=it, mu=st['mu'].tolist(), max_step_over_sigma_prior=rel, shift_p50_px=st['shift_p50'],
                         shift_p99_px=st['shift_p99']))
        log(f"    CTE v2 newton check {it}: mu_pop=({st['mu'][0]:+.4f}, {st['mu'][1]:+.4f})  residual step "
            f"max|dgamma|/sigma_prior={rel:.2e}  applied shift p50 {st['shift_p50']:.4f} px, p99 {st['shift_p99']:.4f} px")
        if st['shift_p99'] > MAX_SHIFT_PX:
            raise RuntimeError(f'CTE v2 newton: the solution shifts detections by p99 {st["shift_p99"]:.3f} px '
                               f'(> {MAX_SHIFT_PX}) -- not a CTE-sized correction; refusing to apply')
        if rel < tol:
            break
        gamma = gamma + step
    else:
        raise RuntimeError(f'CTE v2 newton: not converged after {max_refine} refinement solves (last residual step '
                           f'{rel:.2e} x sigma_prior) -- the solve is not linear in the positions; refusing to apply')
    C_gam = np.linalg.inv(Hs); C_gam = 0.5 * (C_gam + C_gam.T)
    info = dict(A=A, S=S, Rz=Rz, Lp=Lp, P=P, C_gamma=C_gam, asymmetry=asym, history=hist, n_solves=ctx.n_solves,
                seconds=round(time.time() - t0, 1), mu_nocte=st0['mu'].tolist(), r_nocte=st0['r'],
                A_fixed_frame=A_fix, T_fixed_point=T_fp)
    return st, info


def _fp_step(ctx, st, gamma_now, kw):
    """The legacy fixed-point loop's step at this state (diagnostic only)."""
    dg, _, _ = _fit_increment(ctx.s, ctx.names, ctx.per, st['r'], st['a'], ctx.member_mask, ctx.nf, ctx.ngrp,
                              gamma_now=gamma_now, **kw)
    return dg


def _identifiability(info, groups, nf, time_order, sigma_names=None):
    """Per-coefficient shrinkage, whitened eigen-spectrum, mu_pop sensitivity; JSON-safe summary."""
    A, P, C = info['A'], info['P'], info['C_gamma']
    names = _term_names(nf, time_order)
    labels = [f'{g}:{c}:{t}' for g in groups for c in ('x', 'y') for t in names]
    sp = np.sqrt(np.diag(P)); sq = np.sqrt(np.diag(C))
    shrink = 1.0 - (sq / sp) ** 2
    Lc = np.linalg.cholesky(P)
    As = 0.5 * (A + A.T)
    lam, V = np.linalg.eigh(Lc.T @ As @ Lc)                 # data information in prior units (>1: data-dominated)
    S = info['S']
    C_mu = S @ C @ S.T
    out = dict(labels=labels, sigma_prior=sp.tolist(), sigma_post=sq.tolist(), shrinkage=shrink.tolist(),
               n_data_dominated=int(np.sum(lam > 1.0)), n_prior_dominated=int(np.sum(lam < 0.1)),
               eig_whitened=lam.tolist(), asymmetry=info['asymmetry'],
               C_mu_cte=C_mu.tolist(), sigma_mu_cte=np.sqrt(np.clip(np.diag(C_mu), 0, None)).tolist(),
               mu_sensitivity_top=[])
    # which coefficients carry the mu_pop uncertainty
    contrib = (S ** 2) * (sq ** 2)[None, :]
    for c, comp in enumerate(('ra', 'dec')):
        idx = np.argsort(contrib[c])[::-1][:6]
        out['mu_sensitivity_top'].append({comp: [(labels[i], float(S[c, i]), float(sq[i]), float(shrink[i])) for i in idx]})
    if info.get('A_fixed_frame') is not None:
        Af = info['A_fixed_frame']
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.diag(A) / np.diag(Af)
        out['info_ratio_free_over_fixed_frame'] = ratio.tolist()
    if info.get('T_fixed_point') is not None:
        mu_fp = np.linalg.eigvals(info['T_fixed_point'])
        o = np.argsort(-np.abs(mu_fp))
        out['fixed_point_map_eigs_top'] = [[float(np.real(x)), float(np.imag(x))] for x in mu_fp[o][:10]]
        out['fixed_point_spectral_radius'] = float(np.abs(mu_fp).max())
        w, Vt = np.linalg.eig(info['T_fixed_point'])
        k = int(np.argmax(np.abs(w))); v = np.real(Vt[:, k]); v = v / np.max(np.abs(v))
        top = np.argsort(-np.abs(v))[:8]
        out['fixed_point_unstable_mode'] = [(labels[i], float(v[i])) for i in top]
    return out


def run_cte_v2(solver, image_names, stars_per_image, star_id_to_idx, solve_fn, member_sidx, mu_pop, r, a_arr,
               C_vT=None, anchor_sidx=None, sigma_pm=None, sigma_plx_tot=None, sigma_gamma=0.03, C_pop_prior_inv=None,
               fix_r=False, z_weights=None, n_iter=5, time_order=1, output_dir=None, tol_mu=1e-4, group_by='filter',
               method='newton', sigma_filter=0.015, diag=False, probe_px=0.01, tol=1e-3, fast_condition=True):
    """CTE phase v2.  Returns (r, mu_pop, C_shared, C_vT, a_arr, info).

    solve_fn(member_sidx, mu, r, fix_r_arg, z_weights_arg[, update_z_arg]) is the caller's pop-fit solve
    (run_pop_fit._solve) so every prior / anchor / weight / Gaia field mode is the production one.

    method 'newton' (default): exact linear-response Newton solve with identifiability diagnostics and the gamma
    uncertainty propagated into mu_pop (info['C_mu_cte']); works with the Gaia field-mode block (plan P4).
    method 'fixed_point': the legacy loop (not with field modes)."""
    if method == 'fixed_point':
        if getattr(solver, 'n_gz', 0):
            raise NotImplementedError('--cte_v2_method fixed_point does not handle the Gaia field-mode block (its '
                                      'conditioning solves move z); use the default newton method')
        return _run_fixed_point(solver, image_names, stars_per_image, star_id_to_idx, solve_fn, member_sidx, mu_pop, r,
                                a_arr, C_vT=C_vT, anchor_sidx=anchor_sidx, sigma_pm=sigma_pm, sigma_plx_tot=sigma_plx_tot,
                                sigma_gamma=sigma_gamma, C_pop_prior_inv=C_pop_prior_inv, fix_r=fix_r, z_weights=z_weights,
                                n_iter=n_iter, time_order=time_order, output_dir=output_dir, tol_mu=tol_mu, group_by=group_by)
    if method != 'newton':
        raise ValueError(f'CTE v2: unknown method {method!r}')
    t_start = time.time()
    per, groups, nf, m_med, m_sd = _setup(solver, image_names, stars_per_image, star_id_to_idx, anchor_sidx, time_order, group_by)
    ngrp = len(groups)
    print("\n" + "-" * 60)
    print(f"  CTE phase v2 (newton): groups {groups}, {nf} basis terms x 2 directions per group = {ngrp * 2 * nf} "
          f"coefficients, time_order={time_order}, group_by={group_by}, prior sigma {sigma_gamma} px"
          + (f" shared + {sigma_filter} px per filter" if group_by == 'filter' else '') +
          f"; {len(per)}/{len(image_names)} images with CTE geometry; mag norm ({m_med:.2f}, {m_sd:.2f}); "
          f"{'frozen' if fix_r else 'free'} r; Gaia field modes K={int(getattr(solver, 'n_gz', 0) or 0)}")
    if ngrp == 0:
        print("  CTE phase v2: no ACS/WFC or WFC3/UVIS detections with raw coordinates -- skipped")
        return r, mu_pop, None, None, a_arr, dict(skipped=True)
    n_const = nf // (time_order + 1)
    const_mask = ((np.arange(ngrp * 2 * nf) % nf) < n_const).astype(float)
    ctx = _NewtonCTE(solver, image_names, per, groups, nf, time_order, solve_fn, member_sidx, mu_pop, r, fix_r, z_weights,
                     const_mask, sigma_pm=sigma_pm, fast_condition=fast_condition)
    P = _prior_cov(groups, nf, sigma_gamma, sigma_filter, group_by)
    fp_kw = dict(z_weights=z_weights, sigma_pm=sigma_pm, sigma_plx_tot=sigma_plx_tot, sigma_gamma=sigma_gamma,
                 C_pop_prior_inv=C_pop_prior_inv)
    st, ninfo = _newton(ctx, P, probe_px=probe_px, tol=tol, diag=diag, fp_kwargs=fp_kw)
    ident = _identifiability(ninfo, groups, nf, time_order)
    gamma = st['gamma']; C_gam = ninfo['C_gamma']
    # leave the solver in the final state (positions shifted, z at the gamma* optimum)
    if ctx.n_gz:
        solver.gaia_z = st['z'].copy(); solver._gaia_last = st['gaia_last']
        if st['gaia_last'] is not None:
            _nz = ctx.n_gz; _C = st['gaia_last']['C']; solver.gaia_z_cov = _C[-_nz:, -_nz:].copy()
    mu_new = st['mu']; r_new = st['r']
    smu = ident['sigma_mu_cte']
    print(f"    CTE v2 newton: mu_pop ({ninfo['mu_nocte'][0]:+.4f}, {ninfo['mu_nocte'][1]:+.4f}) -> ({mu_new[0]:+.4f}, "
          f"{mu_new[1]:+.4f}) mas/yr; gamma uncertainty adds ({smu[0]:.4f}, {smu[1]:.4f}) mas/yr in quadrature; "
          f"{ident['n_data_dominated']} / {len(gamma)} directions data-dominated, {ident['n_prior_dominated']} prior-dominated; "
          f"A asymmetry {ident['asymmetry']:.1e}; {ninfo['n_solves']} production solves in {ninfo['seconds']:.0f} s")
    if 'fixed_point_spectral_radius' in ident:
        print(f"    CTE v2 diag: legacy fixed-point map spectral radius {ident['fixed_point_spectral_radius']:.3f} "
              f"(> 1 = diverges); leading eigenvalues {ident['fixed_point_map_eigs_top'][:4]}")
    sg = np.sqrt(np.diag(C_gam))
    info = dict(method='newton', groups=groups, n_basis=nf, time_order=time_order, group_by=group_by,
                basis='yt*[1,xt] x [1,m,m^2] x [1,tau]', term_names=_term_names(nf, time_order),
                t0={'ACS': 2002.165, 'UVIS': 2009.37}, tau_unit_yr=10.0, y_readout={'top': _Y_READOUT_TOP, 'bot': _Y_READOUT_BOT},
                y_split={'ACS': 2048.0, 'UVIS': 2047.0}, mag_norm=[m_med, m_sd], gamma=gamma.tolist(), sigma_gamma=sg.tolist(),
                layout='[group][x block | y block][basis]', sigma_gamma_prior_px=sigma_gamma,
                sigma_filter_prior_px=(sigma_filter if group_by == 'filter' else None), n_const_basis=n_const,
                anchors_get='time-constant block only', frozen_r=bool(fix_r), n_gaia_modes=int(ctx.n_gz),
                mu_nocte=ninfo['mu_nocte'], mu_cte=[float(mu_new[0]), float(mu_new[1])],
                C_mu_cte=ident['C_mu_cte'], sigma_mu_cte=smu, newton_history=ninfo['history'], n_solves=ninfo['n_solves'],
                probe_px=probe_px, tol=tol, identifiability=ident, seconds=round(time.time() - t_start, 1),
                note='gamma = exact Newton solution of the production pop-fit solve linearised in the detector positions '
                     '(membership, z-weights, alpha frozen); fitted on members + alignment stars (anchors and diffuse '
                     'non-members excluded from the gamma data, all non-anchor detections corrected); C_mu_cte = S C_gamma '
                     'S^T is the gamma part of the marginal mu_pop covariance (added to sigma_mu_pop by run_pop_fit)')
    if output_dir is not None:
        od = Path(output_dir); od.mkdir(parents=True, exist_ok=True)
        (od / 'cte_v2.json').write_text(json.dumps(info, indent=1))
        np.savez(od / 'cte_v2_params.npz', gamma=gamma, C_gamma=C_gam, groups=np.array(groups), mag_norm=np.array([m_med, m_sd]),
                 time_order=time_order, A=ninfo['A'], S=ninfo['S'], P=P, labels=np.array(ident['labels']),
                 **({'A_fixed_frame': ninfo['A_fixed_frame'], 'T_fixed_point': ninfo['T_fixed_point']} if diag else {}))
        _plot_trends(solver, image_names, per, r_new, st['a'], ctx.member_mask, od / 'plots' / 'cte_v2_trends.png')
        _plot_identifiability(info, ninfo, od / 'plots' / 'cte_v2_identifiability.png')
    return r_new, mu_new, st['C_shared'], st['C_vT'], st['a'], info


def _plot_identifiability(info, ninfo, path):
    """gamma +- sigma vs the prior band per group / coefficient, and the whitened information spectrum."""
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        ident = info['identifiability']; g = np.array(info['gamma']); sq = np.array(ident['sigma_post'])
        sp = np.array(ident['sigma_prior']); groups = info['groups']; nb = 2 * info['n_basis']
        fig, axs = plt.subplots(2, 1, figsize=(max(8, 0.09 * len(g) + 3), 7.5))
        x = np.arange(len(g))
        axs[0].fill_between(x, -sp, sp, color='0.85', step='mid', label='prior +-1 sigma')
        axs[0].errorbar(x, g, yerr=sq, fmt='o', ms=2.5, lw=0.8, label='posterior')
        for i in range(1, len(groups)):
            axs[0].axvline(i * nb - 0.5, color='k', lw=0.5)
        for i, gg in enumerate(groups):
            axs[0].text(i * nb + nb / 2, axs[0].get_ylim()[1] * 0.92, gg, ha='center', fontsize=7)
        axs[0].set_xlabel('coefficient  [group][x | y][basis]'); axs[0].set_ylabel('gamma [px]'); axs[0].legend(fontsize=7)
        axs[0].set_title(f"CTE v2 newton: mu_pop {info['mu_nocte'][0]:+.4f},{info['mu_nocte'][1]:+.4f} -> "
                         f"{info['mu_cte'][0]:+.4f},{info['mu_cte'][1]:+.4f}; sigma_mu(CTE) {info['sigma_mu_cte'][0]:.4f},"
                         f"{info['sigma_mu_cte'][1]:.4f} mas/yr", fontsize=9)
        lam = np.sort(np.array(ident['eig_whitened']))
        axs[1].semilogy(np.arange(len(lam)), np.clip(lam, 1e-6, None), 'o-', ms=3, label='data information / prior (whitened)')
        axs[1].axhline(1, color='k', lw=0.6)
        if 'fixed_point_map_eigs_top' in ident:
            axs[1].set_title(f"legacy fixed-point map spectral radius {ident['fixed_point_spectral_radius']:.2f}", fontsize=9)
        axs[1].set_xlabel('eigen-direction'); axs[1].set_ylabel('eigenvalue'); axs[1].legend(fontsize=7)
        fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)
    except Exception as exc:
        print(f"  WARNING: CTE v2 identifiability plot failed: {exc}")
