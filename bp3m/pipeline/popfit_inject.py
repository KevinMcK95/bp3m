"""Synthetic-injection test hook for the pop-fit (10-25; TEST ONLY, enabled by $BP3M_POPFIT_INJECT = path to a JSON config).

Replaces the real measurements with draws from the pop-fit's own linearised model at a known truth, right after the
initial (frozen) membership is set, so the full pop-fit (all phases, alpha updates, re-opening, posterior passes,
post-hoc sigma_int) runs unchanged on synthetic data:
  truth   r = the v1 alignment r_hat; members: PM ~ N(mu_true, sigma_int^2 I), parallax = plx_pop, position offset 0;
          non-members: the survey prior mean (Gaia [+DELVE]; 2p stars: PM/parallax 0 = the diffuse-prior centre);
          GDC map error: z_true ~ N(0, z_scale^2 I) through the loaded error modes GM (inject_gdc).
  survey  h = P v_true + P^(1/2) n  (P = C_survey_inv, information form; covers Gaia 5p, 2p and DELVE);
  HST     every detection: X(pix) r_true = xys + JU v_true - R GM z_true + eps,  eps ~ N(0, Cs) (the fit's own C_hst, i.e.
          floor and v1 alpha included); solved for the detector position (Newton, numerical Jacobian; exact for 8p).
Config keys: seed, mu_true [ra, dec] mas/yr, sigma_int (mas/yr; null = the fit's sigma_pm), inject_gdc (bool), z_scale
(default 1), drop_modes (bool: remove GM after injecting, i.e. fit WITHOUT the modes), truth_out (JSON path)."""
import json
import numpy as np


def inject(solver, image_names, r_true, member_sidx, sigma_pm, plx_pop, cfg_path):
    from bp3m.astro_utils import build_X_matrices
    cfg = json.load(open(cfg_path)); rng = np.random.default_rng(int(cfg['seed'])); nr = solver.N_R
    mu_true = np.asarray(cfg['mu_true'], float); s_int = float(cfg['sigma_int']) if cfg.get('sigma_int') is not None else float(sigma_pm)
    print(f"\n  *** SYNTHETIC INJECTION ({cfg_path}): seed {cfg['seed']}, mu_true {mu_true}, sigma_int {s_int}, "
          f"inject_gdc {cfg.get('inject_gdc', False)}, drop_modes {cfg.get('drop_modes', False)} ***")
    for j, img in enumerate(image_names):
        if solver._img_data.get(img) is not None and np.any(np.abs(r_true[j * nr + 4:j * nr + 6]) > 1e-9):
            raise RuntimeError('inject: r_true pointing terms must be zero (call after _update_geometry)')
    # ── truth astrometry ──
    P = solver.C_survey_inv; w, U = np.linalg.eigh(0.5 * (P + np.swapaxes(P, 1, 2))); w = np.clip(w, 0, None)
    v_true = np.asarray(solver.v_prior, float).copy()
    v_true[:, 0:2] = 0.0
    no_pm = (np.abs(P[:, 2, 2]) == 0); v_true[no_pm, 2:5] = 0.0
    m = np.asarray(member_sidx, int)
    v_true[m, 2:4] = mu_true + s_int * rng.standard_normal((len(m), 2)); v_true[m, 4] = plx_pop
    h = np.einsum('nij,nj->ni', P, v_true) + np.einsum('nij,nj->ni', U * np.sqrt(w)[:, None, :], rng.standard_normal(v_true.shape))
    solver.C_survey_inv_dot_v = h
    vp = np.einsum('nij,nj->ni', np.linalg.pinv(P), h); solver.v_prior = vp; solver.v_survey = vp.copy()
    # ── GDC map error ──
    K = 0; z_true = None
    if cfg.get('inject_gdc'):
        gm = [solver._img_data[im].get('GM') for im in image_names if solver._img_data.get(im) is not None]
        if not gm or any(g is None for g in gm): raise RuntimeError('inject_gdc needs the error modes loaded on every image ($BP3M_GDC_COVMODES)')
        K = gm[0].shape[2]; z_true = float(cfg.get('z_scale', 1.0)) * np.random.default_rng(int(cfg['seed']) + 10 ** 6).standard_normal(K)   # own stream: arms share the star/noise draws
    # ── detections ──
    n_det = 0; max_it = 0.0
    for j, img in enumerate(image_names):
        d = solver._img_data.get(img)
        if d is None: continue
        r_j = r_true[j * nr:(j + 1) * nr]; sidx = d['sidx']
        T = d['xys'] + np.einsum('nik,nk->ni', d['JU'], v_true[sidx])
        if K: T = T - np.matmul(solver.R[img], d['GM']) @ z_true
        Cs = solver._compute_Cs(img, r_j); T = T + np.einsum('nij,nj->ni', np.linalg.cholesky(Cs), rng.standard_normal((len(sidx), 2)))
        z0 = np.zeros(len(sidx))
        def f(xc, yc):
            X = build_X_matrices(xc, yc, z0, z0, z0, z0, poly_order=solver.poly_order)
            if solver.fit_chip_offset: X = solver._append_chip_cols(X, d.get('chip_hi'))
            return np.einsum('nkl,l->nk', X, r_j)
        xc = np.asarray(d['X_c'], float).copy(); yc = np.asarray(d['Y_c'], float).copy()
        for _ in range(4):
            f0 = f(xc, yc); fx = f(xc + 1.0, yc) - f0; fy = f(xc, yc + 1.0) - f0
            Jm = np.stack([fx, fy], axis=2); dlt = np.linalg.solve(Jm, (T - f0)[:, :, None])[:, :, 0]
            xc += dlt[:, 0]; yc += dlt[:, 1]
        max_it = max(max_it, float(np.abs(f(xc, yc) - T).max()))
        d['X_c'] = xc; d['Y_c'] = yc; n_det += len(sidx)
    solver._update_geometry(r_true.copy(), solver.v_survey)
    if cfg.get('drop_modes'):
        for im in image_names:
            if solver._img_data.get(im) is not None: solver._img_data[im]['GM'] = None
        solver._gm_z = None
    print(f"  injected {n_det} detections on {sum(solver._img_data.get(im) is not None for im in image_names)} images "
          f"(max inversion residual {max_it:.2e} px), {len(m)} members, K_gdc {K}")
    truth = dict(cfg=cfg, mu_true=mu_true.tolist(), sigma_int=s_int, n_members=int(len(m)), K_gdc=K,
                 mu_members_sample=v_true[m, 2:4].mean(0).tolist(), sd_members_sample=v_true[m, 2:4].std(0, ddof=1).tolist(),
                 z_true_rms=float(np.sqrt(np.mean(z_true ** 2))) if K else None)
    if cfg.get('truth_out'):
        json.dump(truth, open(cfg['truth_out'], 'w'), indent=1)
        ids = np.zeros(solver.n_stars, np.int64)
        for gid, i in solver.star_id_to_idx.items(): ids[i] = int(gid)
        mem = np.zeros(solver.n_stars, bool); mem[m] = True
        np.savez(cfg['truth_out'].replace('.json', '_stars.npz'), Gaia_id=ids, v_true=v_true, member=mem, z_true=z_true if K else np.zeros(0))
    return truth
