"""
bp3m-pop-fit — Population proper motion fitting post-processor.

Called after the standard bp3m run finishes.  Reads the same inputs as bp3m
(FLC pipeline: Bayesian_PMs/ + Gaia/) and the bp3m alignment outputs
(r_hat, alpha, use_for_fit flags from BP3M_results/) to fit the cluster
population mean proper motion μ_pop and jointly refine the per-image alignment.

The data-loading section mirrors run_alignment.py exactly (same loader,
same split_ccd logic, same image-name set taken from BP3M_results/run_config.json).

Steps
-----
1. Load same data as bp3m; split ACS chips; filter to v1 image set.
2. Load bp3m r_hat and alpha values; apply bp3m use_for_fit flags.
3. Estimate initial μ_pop from sigma-clipped Gaia PMs; select initial members.
4. Phase 1 (μ-only): hold r fixed at bp3m values — avoids r–μ degeneracy.
5. Phase 2 (joint):  jointly refine r and μ_pop; iterate member selection.
6. Save results to {target}/BP3M_pop_fit_results/.
7. Plot per-visit residual maps (before / after) in plots/residuals/.

Member prior
------------
Members receive the cluster PM prior  N(μ_pop, σ_pm² I₂) and the LVD parallax
prior N(plx_pop, σ_plx_tot²) on top of their Gaia prior (5p or 2p).
Non-members retain the standard Gaia prior unchanged.

Usage
-----
    bp3m-pop-fit --name "Leo I" \\
        --sigma_pm 0.0075 --plx_pop 0.003873 --sigma_plx_tot 0.0001425 \\
        --mu_pop_prior_sigma 0.5
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── Initial μ_pop estimate from Gaia catalog PMs ─────────────────────────────

def _estimate_mu_pop(
    gaia_catalog: pd.DataFrame,
    n_sigma: float = 3.0,
    n_iter: int = 10,
) -> np.ndarray:
    """Sigma-clipped mean of Gaia proper motions (5p/6p stars only)."""
    pmra  = gaia_catalog['pmra'].to_numpy(float)
    pmdec = gaia_catalog['pmdec'].to_numpy(float)
    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    pmra, pmdec = pmra[finite], pmdec[finite]
    if len(pmra) < 5:
        print("  WARNING: fewer than 5 stars with finite Gaia PMs — using (0, 0)")
        return np.zeros(2)

    keep = np.ones(len(pmra), dtype=bool)
    for _ in range(n_iter):
        if keep.sum() < 5:
            break
        med_ra  = float(np.median(pmra[keep]))
        med_dec = float(np.median(pmdec[keep]))
        dra, ddec = pmra[keep] - med_ra, pmdec[keep] - med_dec
        sigma   = max(float(np.median(np.hypot(dra, ddec))) / 0.6745, 0.01)
        new_keep = np.hypot(pmra - med_ra, pmdec - med_dec) < n_sigma * sigma
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    mu = np.array([float(np.mean(pmra[keep])), float(np.mean(pmdec[keep]))])
    print(f"  Initial μ_pop (Gaia σ-clip, n={keep.sum()}/{len(pmra)}): "
          f"({mu[0]:+.4f}, {mu[1]:+.4f}) mas/yr")
    return mu


# ── Member selection from posterior stellar astrometry ────────────────────────

def _select_members_from_a(
    a_arr: np.ndarray,
    mu_pop: np.ndarray,
    n_hst: np.ndarray,
    sigma_clip: float = 3.0,
    n_iter: int = 5,
    min_members: int = 5,
    init_window_masyr: float = 2.0,
    pm_sys_floor: float = 0.2,
) -> np.ndarray:
    """Sigma-clip on PM distance from mu_pop; only stars with ≥1 HST detection eligible."""
    eidx = np.where(n_hst >= 1)[0]
    if len(eidx) < min_members:
        return eidx

    pmra  = a_arr[eidx, 2]
    pmdec = a_arr[eidx, 3]
    dist  = np.hypot(pmra - mu_pop[0], pmdec - mu_pop[1])

    keep = np.isfinite(dist) & (dist < init_window_masyr)
    if keep.sum() < min_members:
        keep = np.isfinite(dist)

    for _ in range(n_iter):
        if keep.sum() < min_members:
            break
        sigma    = float(np.median(dist[keep])) / 0.6745
        sigma    = max(sigma, pm_sys_floor)
        new_keep = np.isfinite(dist) & (dist < sigma_clip * sigma)
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    return eidx[keep]


# ── Initial member selection from Gaia catalog PMs ───────────────────────────

def _select_initial_members(
    gaia_catalog: pd.DataFrame,
    mu_pop: np.ndarray,
    member_sigma_clip: float,
    sigma_pm: float,
    pm_sys_floor: float = 0.2,
) -> np.ndarray:
    """Select initial member candidates from Gaia catalog PMs."""
    _radius = member_sigma_clip * max(sigma_pm, pm_sys_floor)
    pmra  = gaia_catalog['pmra'].to_numpy(float)
    pmdec = gaia_catalog['pmdec'].to_numpy(float)
    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    dist   = np.where(finite,
                      np.hypot(pmra - mu_pop[0], pmdec - mu_pop[1]),
                      np.inf)
    return np.where(dist < _radius)[0]


# ── Load bp3m outputs from BP3M_results ───────────────────────────────────────

def _load_bp3m_outputs(
    bp3m_dir: Path,
    image_names: list[str],
    nr: int,
    solver,
) -> np.ndarray:
    """
    Read r_hat and alpha from BP3M_results/image_transformations.csv.
    Also applies the v1 alpha inflation to solver._img_data[img]['C_hst'].
    Returns r_hat (n_images * nr,).
    """
    xform_path = bp3m_dir / 'image_transformations.csv'
    xdf = pd.read_csv(xform_path)
    img_to_row = {str(row['image_name']): row for _, row in xdf.iterrows()}

    r_hat = np.zeros(len(image_names) * nr)
    missing = []
    n_alpha_applied = 0

    for j_idx, img in enumerate(image_names):
        row = img_to_row.get(img)
        if row is None:
            missing.append(img)
            continue
        cs = j_idx * nr
        r_hat[cs + 0] = float(row['a'])
        r_hat[cs + 1] = float(row['b'])
        r_hat[cs + 2] = float(row['c'])
        r_hat[cs + 3] = float(row['d'])
        r_hat[cs + 4] = float(row['w'])
        r_hat[cs + 5] = float(row['z'])
        if nr > 6:
            r_hat[cs + 6] = float(row.get('delta_ra0_mas',  0.0)) / 1000.0
        if nr > 7:
            r_hat[cs + 7] = float(row.get('delta_dec0_mas', 0.0)) / 1000.0
        for k in range(8, nr):
            r_hat[cs + k] = float(row.get(f'r_{k}', 0.0))

        # Apply v1 alpha to C_hst (HST position uncertainty inflation)
        alpha = float(row.get('alpha', 1.0))
        d = solver._img_data.get(img)
        if d is not None and alpha != 1.0:
            d['alpha_applied'] = alpha
            d['C_hst'] = alpha ** 2 * d['C_hst_orig']
            n_alpha_applied += 1

    if missing:
        raise RuntimeError(
            f"{len(missing)} solver images missing from "
            f"BP3M_results/image_transformations.csv: {missing[:5]} ..."
        )
    print(f"  r_hat loaded ({len(image_names)} images, {nr} params each); "
          f"alpha applied to {n_alpha_applied} images")
    return r_hat


# ── Apply bp3m use_for_fit / use_for_astrom flags ─────────────────────────────

def _apply_bp3m_flags(
    bp3m_dir: Path,
    solver,
    image_names: list[str],
) -> None:
    """
    Override solver use_for_fit and use_for_astrom from BP3M_results.
    Matches stars by Gaia_id (int64) to avoid float roundtrip corruption.
    """
    _uff_path = bp3m_dir / 'use_for_fit.npz'
    _ufa_path = bp3m_dir / 'use_for_astrom.npz'
    _si_path  = bp3m_dir / 'star_indices.npz'
    _sa_path  = bp3m_dir / 'stellar_astrometry.csv'

    if not all(p.exists() for p in [_uff_path, _si_path, _sa_path]):
        print("  WARNING: use_for_fit.npz / stellar_astrometry.csv not found — "
              "using default quality-cut flags")
        return

    _uff = np.load(_uff_path)
    _ufa = np.load(_ufa_path) if _ufa_path.exists() else None
    _si  = np.load(_si_path)
    _sa  = pd.read_csv(_sa_path, dtype={'Gaia_id': np.int64})
    _bp3m_gids = _sa['Gaia_id'].to_numpy(np.int64)

    # Per-image sets of Gaia_ids that have use_for_fit / use_for_astrom = True in v1
    def _gid_set_from_npz(npz_file):
        out: dict[str, frozenset] = {}
        for _img in npz_file.files:
            _mask = npz_file[_img].astype(bool)
            if _img not in _si:
                out[_img] = frozenset()
                continue
            _sidx = _si[_img]
            _gids = _bp3m_gids[_sidx[_mask]]
            out[_img] = frozenset(int(g) for g in _gids if g > 0)
        return out

    fit_per_img   = _gid_set_from_npz(_uff)
    astrom_per_img = _gid_set_from_npz(_ufa) if _ufa is not None else fit_per_img

    # Build solver star_index → Gaia_id lookup
    _sol_gid = np.zeros(solver.n_stars, dtype=np.int64)
    for _gid, _idx in solver.star_id_to_idx.items():
        _sol_gid[int(_idx)] = np.int64(_gid)

    n_fit_det = 0; n_astrom_det = 0
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        sidx_j = d['sidx']
        gids_j = _sol_gid[sidx_j]
        fit_set    = fit_per_img.get(img, frozenset())
        astrom_set = astrom_per_img.get(img, frozenset())
        d['use_for_fit']    = np.array([int(g) in fit_set    for g in gids_j], dtype=bool)
        d['use_for_astrom'] = np.array([int(g) in astrom_set for g in gids_j], dtype=bool)
        n_fit_det   += int(d['use_for_fit'].sum())
        n_astrom_det += int(d['use_for_astrom'].sum())

    all_fit = set(); all_astrom = set()
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        gids_j = _sol_gid[d['sidx']]
        all_fit.update(int(g) for g in gids_j[d['use_for_fit']])
        all_astrom.update(int(g) for g in gids_j[d['use_for_astrom']])

    has_ufa = _ufa is not None
    print(f"  use_for_fit:    {len(all_fit)} unique stars, {n_fit_det} detections")
    print(f"  use_for_astrom: {len(all_astrom)} unique stars, {n_astrom_det} detections"
          + ("" if has_ufa else " (no use_for_astrom.npz — used use_for_fit)"))


# ── Joint population solve ────────────────────────────────────────────────────

def _joint_solve_pop(
    solver,
    image_names: list[str],
    member_sidx: np.ndarray,
    mu_pop_current: np.ndarray,
    sigma_pm: float,
    plx_pop: float,
    sigma_plx_tot: float,
    C_pop_prior_inv: np.ndarray,
    mu_pop_prior: np.ndarray,
    r_current: np.ndarray,
    fix_r: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One Newton step for (Δr, Δμ_pop) with stellar astrometry marginalised out.

    Population prior
    ----------------
    Members    : H_vv[2:4,2:4] += σ_pm^{-2} I₂  (PM coupled to μ_pop)
                 H_vv[4,4]     += σ_plx^{-2}
    Non-members: Gaia prior (5p: C_survey_inv; 2p: C_survey_inv + diffuse prior)
                 Both 5p and 2p member stars get the population prior.

    Parameters
    ----------
    fix_r : if True solve for Δμ_pop only (Phase 1); else solve jointly (Phase 2).

    Returns
    -------
    r_hat, mu_pop_hat, C_shared, C_vT, a_arr, a_align_arr
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(x, **kw):
            return x

    N_V   = 5          # stellar params: [Δα*, Δδ, μ_α*, μ_δ, plx]
    nr    = solver.N_R
    n_r   = len(image_names) * nr
    n_stars = solver.n_stars

    sigma_pm_inv_sq  = sigma_pm ** -2
    sigma_plx_inv_sq = sigma_plx_tot ** -2

    if fix_r:
        n_shared = 2
    else:
        n_shared = n_r + 2
        idx_r  = slice(0, n_r)
        idx_mu = slice(n_r, n_r + 2)

    # ── H_vv: start from Gaia prior ───────────────────────────────────────────
    H_vv = solver.C_survey_inv.copy()

    # Add diffuse prior diagonal for non-member 2p stars
    _nonmem = np.ones(n_stars, dtype=bool)
    _nonmem[member_sidx] = False
    _nonmem_2p = _nonmem & (solver._C_VG_inv_per_star[:, 2] > 0)
    if _nonmem_2p.any():
        for _k in range(N_V):
            H_vv[_nonmem_2p, _k, _k] += solver._C_VG_inv_per_star[_nonmem_2p, _k]

    # Population prior for member stars (both 5p and 2p)
    H_vv[member_sidx, 2, 2] += sigma_pm_inv_sq
    H_vv[member_sidx, 3, 3] += sigma_pm_inv_sq
    H_vv[member_sidx, 4, 4] += sigma_plx_inv_sq

    # Information vectors: start from Gaia prior contribution
    h_align = solver.C_survey_inv_dot_v.copy()
    h_all   = solver.C_survey_inv_dot_v.copy()

    # Population prior RHS for member stars
    h_align[member_sidx, 2] += sigma_pm_inv_sq * mu_pop_current[0]
    h_align[member_sidx, 3] += sigma_pm_inv_sq * mu_pop_current[1]
    h_all  [member_sidx, 2] += sigma_pm_inv_sq * mu_pop_current[0]
    h_all  [member_sidx, 3] += sigma_pm_inv_sq * mu_pop_current[1]
    h_align[member_sidx, 4] += sigma_plx_inv_sq * plx_pop
    h_all  [member_sidx, 4] += sigma_plx_inv_sq * plx_pop

    # ── Per-image accumulation ─────────────────────────────────────────────────
    K_img       = {}
    XCs_xresid  = {}
    H_rr_block  = np.zeros((n_r, n_r))
    active_glob = np.zeros(n_stars, dtype=bool)

    for j_idx, img in enumerate(_tqdm(image_names, desc='  pop_solve',
                                      unit='img', ncols=90, leave=False)):
        d = solver._img_data.get(img)
        if d is None:
            K_img[img] = None
            continue

        sidx    = d['sidx']
        use_fit = d['use_for_fit']
        use_any = use_fit | d.get('use_for_astrom', use_fit)

        sidx_any = sidx[use_any]
        sidx_fit = sidx[use_fit]
        active_glob[sidx_any] = True

        cs  = j_idx * nr
        r_j = r_current[cs:cs + nr]

        JU  = d['JU']
        X   = d['X_mat']
        xys = d['xys']   # tangent-plane positions at current linearisation point

        Cs     = solver._compute_Cs(img, r_j)
        Cs_inv = np.linalg.inv(Cs)

        x_pred  = np.einsum('nkl,l->nk', X, r_j)
        x_resid = xys - x_pred

        JUT_Cs = np.einsum('nki,nkl->nil', JU, Cs_inv)
        K      = np.einsum('nik,nkl->nil', JUT_Cs, X)
        K_img[img] = K

        np.add.at(H_vv, sidx_any,
                  np.einsum('nik,nkj->nij', JUT_Cs[use_any], JU[use_any]))
        np.subtract.at(h_align, sidx_fit,
                       np.einsum('nik,nk->ni', JUT_Cs[use_fit], x_resid[use_fit]))
        np.subtract.at(h_all, sidx_any,
                       np.einsum('nik,nk->ni', JUT_Cs[use_any], x_resid[use_any]))

        if not fix_r:
            XCsX = np.einsum('nki,nkl,nlj->ij',
                             X[use_fit], Cs_inv[use_fit], X[use_fit])
            H_rr_block[cs:cs + nr, cs:cs + nr] += XCsX + d['C_r_prior_inv']
            XCs_xresid[img] = np.einsum('nki,nkl,nl->ni',
                                         X[use_fit], Cs_inv[use_fit], x_resid[use_fit])

    # ── Stellar posteriors ────────────────────────────────────────────────────
    C_vT = np.zeros_like(H_vv)
    _active_sidx = np.where(active_glob)[0]
    if len(_active_sidx) > 0:
        C_vT[_active_sidx] = np.linalg.inv(H_vv[_active_sidx])
    a_align = np.einsum('nij,nj->ni', C_vT, h_align)
    a       = np.einsum('nij,nj->ni', C_vT, h_all)

    # ── Shared system (μ or r+μ) ───────────────────────────────────────────────
    Lambda = np.zeros((n_shared, n_shared))
    rhs    = np.zeros(n_shared)

    n_mem = len(member_sidx)

    H_mu   = C_pop_prior_inv.copy()
    H_mu  += sigma_pm_inv_sq * n_mem * np.eye(2)
    rhs_mu = (C_pop_prior_inv @ (mu_pop_prior - mu_pop_current)
              - sigma_pm_inv_sq * n_mem * mu_pop_current)

    if not fix_r:
        Lambda[idx_r,  idx_r]  = H_rr_block
        Lambda[idx_mu, idx_mu] = H_mu
        for j_idx, img in enumerate(image_names):
            d = solver._img_data.get(img)
            if d is None:
                continue
            cs = j_idx * nr
            rhs[cs:cs + nr] += d['C_r_prior_inv'] @ (d['r_prior'] - r_current[cs:cs + nr])
            if img in XCs_xresid:
                rhs[cs:cs + nr] += XCs_xresid[img].sum(axis=0)
    else:
        Lambda[:] = H_mu

    # ── Schur correction for μ block ──────────────────────────────────────────
    if n_mem > 0:
        Cv_m = C_vT[member_sidx]
        mu_mu_schur = sigma_pm_inv_sq ** 2 * Cv_m[:, 2:4, 2:4].sum(axis=0)
        if fix_r:
            Lambda -= mu_mu_schur
        else:
            Lambda[idx_mu, idx_mu] -= mu_mu_schur
        rhs_mu += sigma_pm_inv_sq * a[member_sidx, 2:4].sum(axis=0)

    if fix_r:
        rhs[:] = rhs_mu
    else:
        rhs[idx_mu] = rhs_mu

    # ── Per-image Schur corrections (joint solve only) ────────────────────────
    if not fix_r:
        member_set = set(int(s) for s in member_sidx)

        for j_idx, img in enumerate(image_names):
            d = solver._img_data.get(img)
            if d is None or K_img.get(img) is None:
                continue

            cs       = j_idx * nr
            sidx     = d['sidx']
            use_fit  = d['use_for_fit']
            use_fmem = use_fit & np.array([int(s) in member_set for s in sidx], dtype=bool)

            sidx_fit = sidx[use_fit]
            K_fit    = K_img[img][use_fit]
            Cv_fit   = C_vT[sidx_fit]

            CvT_K_fit = np.einsum('nij,njk->nik', Cv_fit, K_fit)
            Lambda[cs:cs + nr, cs:cs + nr] -= np.einsum('nji,njk->ik', K_fit, CvT_K_fit)
            rhs[cs:cs + nr]                += np.einsum('nji,nj->i',   K_fit, a_align[sidx_fit])

            # (r, μ) cross-block
            if use_fmem.any():
                sidx_fm  = sidx[use_fmem]
                K_fm     = K_img[img][use_fmem]
                CvT_M_fm = C_vT[sidx_fm, :, 2:4]
                KT_CvT_M = np.einsum('nji,njk->ik', K_fm, CvT_M_fm)
                Lambda[cs:cs + nr, idx_mu] -= sigma_pm_inv_sq * KT_CvT_M
                Lambda[idx_mu, cs:cs + nr] -= sigma_pm_inv_sq * KT_CvT_M.T

            # Cross-image (r, r) coupling
            for j2_idx, img2 in enumerate(image_names):
                if j2_idx <= j_idx:
                    continue
                d2 = solver._img_data.get(img2)
                if d2 is None or K_img.get(img2) is None:
                    continue
                use2   = d2['use_for_fit']
                sidx2  = d2['sidx'][use2]
                K2     = K_img[img2][use2]

                common, ix1, ix2 = np.intersect1d(sidx_fit, sidx2, return_indices=True)
                if len(common) == 0:
                    continue

                CvT_K2 = np.einsum('nij,njk->nik', C_vT[common], K2[ix2])
                block  = np.einsum('nji,njk->ik', K_fit[ix1], CvT_K2)
                cs2    = j2_idx * nr
                Lambda[cs:cs + nr, cs2:cs2 + nr] -= block
                Lambda[cs2:cs2 + nr, cs:cs + nr] -= block.T

    # ── Solve with diagonal preconditioning ───────────────────────────────────
    d_diag    = np.sqrt(np.maximum(np.abs(np.diag(Lambda)), 1e-30))
    d_inv     = 1.0 / d_diag
    Lambda_sc = d_inv[:, None] * Lambda * d_inv[None, :]
    try:
        C_sc = np.linalg.inv(Lambda_sc)
    except np.linalg.LinAlgError:
        C_sc = np.linalg.pinv(Lambda_sc)
    C_shared = d_inv[:, None] * C_sc * d_inv[None, :]
    delta    = C_shared @ rhs

    if fix_r:
        return r_current.copy(), mu_pop_current + delta, C_shared, C_vT, a, a_align, K_img
    else:
        return (r_current + delta[idx_r],
                mu_pop_current + delta[idx_mu],
                C_shared, C_vT, a, a_align, K_img)


# ── Per-visit residual plots (before / after) ─────────────────────────────────

def _plot_pop_residual_maps(
    output_dir: Path,
    image_names: list[str],
    solver,
    r_before: np.ndarray,
    v_before: np.ndarray,
    r_after: np.ndarray,
    v_after: np.ndarray,
    C_vT_after: np.ndarray | None = None,
    prefix: str = 'final',
) -> None:
    """
    Per-visit 2-row scatter maps (v1 bp3m / pop-fit).
    Columns: dx_gdc, dy_gdc, dx/σ, dy/σ (latter pair when C_vT_after available).

    Geometry (JU, xys) is updated to the appropriate r_hat before each
    compute_gdc_residuals call to ensure correct Jacobians.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict as _defaultdict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_labels = ['bp3m (before)', 'pop-fit (after)']
    nr    = solver.N_R
    saved = 0

    visit_groups: dict[str, list] = _defaultdict(list)
    for img in image_names:
        root = img[:-3] if img.endswith(('_hi', '_lo')) else img
        visit_groups[root].append(img)

    # Pre-compute GDC residuals for both stages.
    # IMPORTANT: JU and xys in each image's data dict are geometry-dependent
    # (_update_geometry writes them).  Must update geometry before each call.
    try:
        solver._update_geometry(r_before, v_before)
        gdc_before = solver.compute_gdc_residuals(r_before, v_before)
    except Exception as _exc:
        print(f"  WARNING: before-residuals failed — {_exc}")
        gdc_before = {}
    try:
        solver._update_geometry(r_after, v_after)
        gdc_after = solver.compute_gdc_residuals(r_after, v_after, C_vT=C_vT_after)
    except Exception as _exc:
        print(f"  WARNING: after-residuals failed — {_exc}")
        gdc_after = {}

    for root, imgs in visit_groups.items():
        rows_x  = [[], []]
        rows_y  = [[], []]
        rows_dx = [[], []]
        rows_dy = [[], []]
        sigma_dx_all: list = []
        sigma_dy_all: list = []
        total_n = 0

        for img in imgs:
            d = solver._img_data.get(img)
            if d is None:
                continue
            use_any = d['use_for_fit'] | d.get('use_for_astrom', d['use_for_fit'])
            if not use_any.any():
                continue

            # detector pixel coordinates for plotting
            xraw = d.get('x_raw', np.zeros(len(d['sidx'])))
            yraw = d.get('y_raw', np.zeros(len(d['sidx'])))

            for si, gdc in enumerate([gdc_before, gdc_after]):
                rd  = gdc.get(img, {})
                dx_all = rd.get('dx_gdc', np.zeros(len(d['sidx'])))
                dy_all = rd.get('dy_gdc', np.zeros(len(d['sidx'])))
                rows_x [si].append(xraw[use_any])
                rows_y [si].append(yraw[use_any])
                rows_dx[si].append(dx_all[use_any])
                rows_dy[si].append(dy_all[use_any])

            if C_vT_after is not None:
                sidx_u = d['sidx'][use_any]
                sigma_dx_all.append(np.sqrt(np.maximum(C_vT_after[sidx_u, 0, 0], 0.0)))
                sigma_dy_all.append(np.sqrt(np.maximum(C_vT_after[sidx_u, 1, 1], 0.0)))

            total_n += int(use_any.sum())

        if total_n == 0:
            continue

        for si in range(2):
            rows_x [si] = np.concatenate(rows_x [si]) if rows_x [si] else np.array([])
            rows_y [si] = np.concatenate(rows_y [si]) if rows_y [si] else np.array([])
            rows_dx[si] = np.concatenate(rows_dx[si]) if rows_dx[si] else np.array([])
            rows_dy[si] = np.concatenate(rows_dy[si]) if rows_dy[si] else np.array([])

        sigma_dx  = np.concatenate(sigma_dx_all) if sigma_dx_all else None
        sigma_dy  = np.concatenate(sigma_dy_all) if sigma_dy_all else None
        has_sigma = (sigma_dx is not None
                     and np.any(np.isfinite(sigma_dx) & (sigma_dx > 0)))

        _vals = np.concatenate([np.abs(rows_dx[0]), np.abs(rows_dy[0])])
        _fin  = _vals[np.isfinite(_vals)]
        _vc   = max(float(np.percentile(_fin, 97)) if len(_fin) > 0 else 0.3, 0.05)
        _vc_sig = 2.0

        n_cols = 4 if has_sigma else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 5, 7),
                                 sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0.08, 'wspace': 0.06})
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f'{root}  n={total_n}', fontsize=10, y=0.99)

        for row_i, stage_lbl in enumerate(stage_labels):
            raw_pairs = [(rows_dx[row_i], 'dx_gdc (px)'),
                         (rows_dy[row_i], 'dy_gdc (px)')]
            if has_sigma:
                _sx = np.where(sigma_dx > 0, sigma_dx, np.nan)
                _sy = np.where(sigma_dy > 0, sigma_dy, np.nan)
                sig_pairs = [(rows_dx[row_i] / _sx, 'dx / σ_x'),
                             (rows_dy[row_i] / _sy, 'dy / σ_y')]
            else:
                sig_pairs = []
            all_pairs = raw_pairs + sig_pairs
            clims = [(-_vc, _vc)] * 2 + [(-_vc_sig, _vc_sig)] * len(sig_pairs)

            for col_i, ((vals, clbl), (vmin, vmax)) in enumerate(zip(all_pairs, clims)):
                ax = axes[row_i, col_i]
                sc = ax.scatter(rows_x[row_i], rows_y[row_i], c=vals,
                                cmap='RdBu_r', vmin=vmin, vmax=vmax,
                                s=1.5, alpha=0.6, linewidths=0, rasterized=True)
                cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
                if row_i == 0:
                    ax.set_title(clbl, fontsize=10, pad=4)
                ax.text(0.02, 0.97, stage_lbl, transform=ax.transAxes,
                        va='top', ha='left', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.75, pad=2, edgecolor='none'))
                ax.tick_params(labelsize=8)
                if col_i == 0:
                    ax.set_ylabel('y_raw (px)', fontsize=8)
                if row_i == 1:
                    ax.set_xlabel('x_raw (px)', fontsize=8)

        plt.savefig(output_dir / f'{prefix}_{root}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        saved += 1

    print(f"  Saved {saved} residual map(s) to {output_dir}/")


# ── Main function ─────────────────────────────────────────────────────────────

def run_pop_fit(
    output_dir: Path,
    field_name: str,
    sigma_pm: float = 0.0075,
    plx_pop: float = 0.003873,
    sigma_plx_tot: float = 0.0001425,
    mu_pop_prior_sigma: float = 0.5,
    n_iter_mu: int = 5,
    n_iter_joint: int = 10,
    member_sigma_clip: float = 3.0,
    poly_order: int | None = None,
    no_plots: bool = False,
) -> Path:
    """
    Run population PM fitting.

    Data loading mirrors run_alignment.py exactly:
      load_image_data_flc → split_images_by_ccd → build_index_maps → BP3MSolver
    v1 r_hat, alpha, and use_for_fit flags are loaded from BP3M_results/.
    """
    from bp3m.data_loader_flc import load_image_data_flc
    from bp3m.data_loader import build_index_maps, split_images_by_ccd
    from bp3m.solver import BP3MSolver

    t_start   = time.time()
    data_root  = Path(output_dir)
    bp3m_dir   = data_root / field_name / 'BP3M_results'
    output_pfr = data_root / field_name / 'BP3M_pop_fit_results'
    output_pfr.mkdir(parents=True, exist_ok=True)

    # ── Read v1 run_config ─────────────────────────────────────────────────────
    _cfg_path = bp3m_dir / 'run_config.json'
    if not _cfg_path.exists():
        raise FileNotFoundError(
            f"BP3M_results/run_config.json not found at {_cfg_path}. Run bp3m first."
        )
    with open(_cfg_path) as _f:
        v1_cfg = json.load(_f)

    v1_image_names      = v1_cfg.get('image_names', [])
    v1_split_ccd        = bool(v1_cfg.get('split_ccd', True))
    min_stars_split_ccd = int(v1_cfg.get('min_stars_split_ccd', 20))
    if poly_order is None:
        poly_order = int(v1_cfg.get('poly_order', 1))

    print("\n" + "─" * 60)
    print("BP3M pop-fit: population PM fitting")
    print("─" * 60)
    print(f"  field={field_name}")
    print(f"  σ_pm={sigma_pm} mas/yr  plx_pop={plx_pop} mas  "
          f"σ_plx_tot={sigma_plx_tot} mas")
    print(f"  μ_pop prior σ={mu_pop_prior_sigma} mas/yr  "
          f"member_sigma_clip={member_sigma_clip}")
    print(f"  poly_order={poly_order}  split_ccd={v1_split_ccd}  "
          f"v1 images={len(v1_image_names)}")

    # ── Load data — mirrors run_alignment.py exactly ───────────────────────────
    print(f"\n  Loading bp3m input data for '{field_name}'...")
    imgs, stars_per_image, gaia_catalog = load_image_data_flc(data_root, field_name)
    if imgs is None or len(imgs) == 0:
        raise RuntimeError(f"No usable images found for '{field_name}'.")

    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)

    # Filter to the base names that v1 used (strip _hi/_lo before split)
    if v1_image_names:
        v1_bases = set()
        for n in v1_image_names:
            base = n[:-3] if n.endswith(('_hi', '_lo')) else n
            v1_bases.add(base)
        image_names = [n for n in image_names if n in v1_bases]
    if not image_names:
        raise RuntimeError(
            "No images remain after filtering to v1 image set."
        )

    filtered_spi = {n: stars_per_image[n] for n in image_names}

    # Filter gaia_catalog to observed stars (same as run_alignment.py)
    observed_ids: set = set()
    for spi in filtered_spi.values():
        observed_ids.update(spi['Gaia_id'].values)
    gaia_catalog = (gaia_catalog[gaia_catalog['Gaia_id'].isin(observed_ids)]
                    .reset_index(drop=True))
    star_id_to_idx = {int(gid): i for i, gid in enumerate(gaia_catalog['Gaia_id'])}

    imgs = {n: imgs[n] for n in image_names}

    # Split ACS chips (same as run_alignment.py)
    if v1_split_ccd:
        imgs, filtered_spi = split_images_by_ccd(
            imgs, filtered_spi, min_stars_per_ccd=min_stars_split_ccd)
        image_names = sorted(filtered_spi.keys())
        star_id_to_idx, image_names, star_in_image = build_index_maps(
            filtered_spi, gaia_catalog)

    # Warn about any mismatch with v1 image set
    if v1_image_names:
        v1_set  = set(v1_image_names)
        our_set = set(image_names)
        extra   = our_set - v1_set
        missing = v1_set - our_set
        if extra:
            print(f"  WARNING: {len(extra)} extra images not in v1 — dropping")
            image_names  = [n for n in image_names if n in v1_set]
            filtered_spi = {n: filtered_spi[n] for n in image_names}
            imgs         = {n: imgs[n] for n in image_names}
            star_id_to_idx, image_names, star_in_image = build_index_maps(
                filtered_spi, gaia_catalog)
        if missing:
            print(f"  WARNING: {len(missing)} v1 images missing from loaded data: "
                  f"{sorted(missing)[:5]} ...")

    print(f"  Images: {len(image_names)}  ", end='')

    # ── Build solver ──────────────────────────────────────────────────────────
    solver = BP3MSolver(imgs, filtered_spi, gaia_catalog,
                        star_id_to_idx, image_names, star_in_image,
                        poly_order=poly_order)
    print(f"Stars: {solver.n_stars}  N_R/image: {solver.N_R}")

    # ── Load v1 r_hat and alpha ────────────────────────────────────────────────
    print("\n  Loading v1 alignment parameters (r_hat, alpha)...")
    r_bp3m = _load_bp3m_outputs(bp3m_dir, image_names, solver.N_R, solver)
    solver._update_R(r_bp3m)
    solver._update_geometry(r_bp3m, solver.v_survey)

    # ── Load v1 bp3m posteriors (for before/after residual plots) ─────────────
    v1_astrom_path = bp3m_dir / 'stellar_astrometry.csv'
    v_bp3m = solver.v_survey.copy()   # fallback: Gaia-only
    if v1_astrom_path.exists():
        try:
            _v1 = pd.read_csv(v1_astrom_path)
            _v1['Gaia_id'] = _v1['Gaia_id'].astype(np.int64)
            _v1_idx = {int(g): i for i, g in enumerate(_v1['Gaia_id'])}
            _cols = ['delta_racosdec_bp3m', 'delta_dec_bp3m',
                     'pmra_bp3m', 'pmdec_bp3m', 'parallax_bp3m']
            if all(c in _v1.columns for c in _cols):
                _v1_arr = _v1[_cols].to_numpy(float)
                for i, gid in enumerate(gaia_catalog['Gaia_id']):
                    j = _v1_idx.get(int(gid))
                    if j is not None:
                        v_bp3m[i] = _v1_arr[j]
        except Exception as _exc:
            print(f"  WARNING: could not load v1 posteriors for plot — {_exc}")

    # ── Apply v1 use_for_fit / use_for_astrom flags ───────────────────────────
    print("\n  Applying v1 detection flags...")
    _apply_bp3m_flags(bp3m_dir, solver, image_names)

    # ── Count HST detections per star ─────────────────────────────────────────
    _n_hst_det = np.zeros(solver.n_stars, dtype=int)
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        _use_a = d.get('use_for_astrom', d['use_for_fit'])
        np.add.at(_n_hst_det, d['sidx'][_use_a], 1)

    # ── Empirical initial μ_pop ────────────────────────────────────────────────
    print("\n  Estimating initial μ_pop from Gaia catalog PMs...")
    mu_pop_est    = _estimate_mu_pop(gaia_catalog)
    mu_pop_prior  = mu_pop_est.copy()
    C_pop_prior_inv = np.eye(2) / mu_pop_prior_sigma ** 2
    mu_pop_current  = mu_pop_prior.copy()
    print(f"  μ_pop prior: ({mu_pop_prior[0]:+.4f}, {mu_pop_prior[1]:+.4f}) ± "
          f"{mu_pop_prior_sigma:.2f} mas/yr")

    # ── Initial member selection from Gaia catalog PMs ────────────────────────
    print("\n  Selecting initial members from Gaia catalog PMs...")
    member_sidx = _select_initial_members(
        gaia_catalog, mu_pop_current, member_sigma_clip, sigma_pm)
    print(f"  Initial members: {len(member_sidx)}")

    # Recompute gaia_n_hst_used to reflect the v1 flags we just applied
    solver.gaia_n_hst_used[:] = 0
    for _img in image_names:
        _d = solver._img_data.get(_img)
        if _d is None:
            continue
        _use_any = _d['use_for_fit'] | _d.get('use_for_astrom', _d['use_for_fit'])
        np.add.at(solver.gaia_n_hst_used, _d['sidx'][_use_any], 1)

    # ── Phase 1: μ-only solve (r fixed at v1 values) ──────────────────────────
    print(f"\n  Phase 1: μ-only solve ({n_iter_mu} iterations, r fixed)...")
    r_current = r_bp3m.copy()
    C_shared_mu = None
    for mu_iter in range(n_iter_mu):
        _, mu_pop_new, C_shared_mu, C_vT, a_arr, _, _ = _joint_solve_pop(
            solver, image_names,
            member_sidx, mu_pop_current,
            sigma_pm, plx_pop, sigma_plx_tot,
            C_pop_prior_inv, mu_pop_prior,
            r_current, fix_r=True,
        )
        delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
        mu_pop_current = mu_pop_new
        member_sidx = _select_members_from_a(
            a_arr, mu_pop_current, _n_hst_det, sigma_clip=member_sigma_clip)
        print(f"    iter {mu_iter + 1}/{n_iter_mu}: "
              f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f}) mas/yr  "
              f"Δμ={delta_mu:.4e}  members={len(member_sidx)}")
        if delta_mu < 1e-6:
            print(f"    Converged.")
            break

    if C_shared_mu is not None:
        sigma_mu_1 = np.sqrt(np.diag(C_shared_mu))
        print(f"  Phase 1 final: μ_pop=({mu_pop_current[0]:+.4f} ± {sigma_mu_1[0]:.4f}, "
              f"{mu_pop_current[1]:+.4f} ± {sigma_mu_1[1]:.4f}) mas/yr")

    # ── Phase 2: joint solve (r + μ_pop) ─────────────────────────────────────
    print(f"\n  Phase 2: joint solve ({n_iter_joint} iterations)...")
    C_shared_joint = None
    for jt_iter in range(n_iter_joint):
        r_new, mu_pop_new, C_shared_joint, C_vT, a_arr, _, _ = _joint_solve_pop(
            solver, image_names,
            member_sidx, mu_pop_current,
            sigma_pm, plx_pop, sigma_plx_tot,
            C_pop_prior_inv, mu_pop_prior,
            r_current, fix_r=False,
        )
        delta_r  = float(np.max(np.abs(r_new - r_current)))
        delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
        r_current      = r_new
        mu_pop_current = mu_pop_new
        member_sidx = _select_members_from_a(
            a_arr, mu_pop_current, _n_hst_det, sigma_clip=member_sigma_clip)
        print(f"    iter {jt_iter + 1}/{n_iter_joint}: "
              f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f})  "
              f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  members={len(member_sidx)}")
        solver._update_R(r_current)
        solver._update_geometry(r_current, a_arr)
        if delta_r < 1e-6 and delta_mu < 1e-6:
            print(f"    Converged.")
            break

    n_r = len(image_names) * solver.N_R
    sigma_mu_joint = (np.sqrt(np.diag(C_shared_joint[n_r:, n_r:]))
                      if C_shared_joint is not None else np.array([np.nan, np.nan]))
    print(f"\n  Final: μ_pop=({mu_pop_current[0]:+.4f} ± {sigma_mu_joint[0]:.4f}, "
          f"{mu_pop_current[1]:+.4f} ± {sigma_mu_joint[1]:.4f}) mas/yr")
    print(f"  Final members: {len(member_sidx)}")

    # ── Final posterior pass at convergence ───────────────────────────────────
    print("\n  Final posterior pass...")
    _, _, C_shared_final, C_vT_final, v_mean, _, K_img_final = _joint_solve_pop(
        solver, image_names,
        member_sidx, mu_pop_current,
        sigma_pm, plx_pop, sigma_plx_tot,
        C_pop_prior_inv, mu_pop_prior,
        r_current, fix_r=False,
    )

    # ── Analytic marginalised posteriors (mirrors run_alignment.py) ──────────
    print("\n  Computing analytic marginalised posteriors...")
    C_r = C_shared_final[:n_r, :n_r]
    v_mean_marg, v_cov = solver.compute_analytic_posteriors(
        r_current, C_r, v_mean, K_img_final, C_vT_final)
    v_cov_full = v_cov + C_vT_final   # full marginal covariance per star

    # ── Save results (mirrors _save_results in run_alignment.py) ─────────────
    print("\n  Saving results...")

    from bp3m.pipeline.run_alignment import compute_chi2_per_star

    # 1. image_transformations.csv — same columns as v1 including sigmas
    _rows = []
    for j_idx, img in enumerate(image_names):
        cs    = j_idx * solver.N_R
        r_j   = r_current[cs:cs + solver.N_R]
        C_j   = C_r[cs:cs + solver.N_R, cs:cs + solver.N_R]
        d_img = solver._img_data.get(img, {}) or {}
        use_ast = d_img.get('use_for_astrom', d_img.get('use_for_fit', np.zeros(0, bool)))
        a, b, c, d = r_j[:4]
        _rows.append(dict(
            image_name=img,
            n_stars_alignment=int(np.sum(d_img.get('use_for_fit', np.zeros(0, bool)))),
            n_stars_astrometry_only=int(np.sum(
                use_ast & ~d_img.get('use_for_fit', np.zeros(0, bool)))),
            a=a, b=b, c=c, d=d,
            w=r_j[4], z=r_j[5],
            delta_ra0_mas=r_j[6] * 1000 if solver.N_R > 6 else 0.0,
            delta_dec0_mas=r_j[7] * 1000 if solver.N_R > 7 else 0.0,
            pixel_scale_mas=(np.sqrt(a * d - b * c)
                             * imgs.get(img, {}).get('orig_pixel_scale', 50.0)),
            rotation_deg=np.degrees(np.arctan2(b - c, a + d)),
            on_skew=(a - d) / 2,
            off_skew=(b + c) / 2,
            sigma_a=np.sqrt(C_j[0, 0]),   sigma_b=np.sqrt(C_j[1, 1]),
            sigma_c=np.sqrt(C_j[2, 2]),   sigma_d=np.sqrt(C_j[3, 3]),
            sigma_w=np.sqrt(C_j[4, 4]),   sigma_z=np.sqrt(C_j[5, 5]),
            sigma_dra0_mas=np.sqrt(C_j[6, 6]) * 1000 if solver.N_R > 6 else 0.0,
            sigma_ddec0_mas=np.sqrt(C_j[7, 7]) * 1000 if solver.N_R > 7 else 0.0,
            alpha=float(d_img.get('alpha_applied', 1.0)),
            **{f'r_{k}': float(r_j[k]) for k in range(8, solver.N_R)},
        ))
    pd.DataFrame(_rows).to_csv(output_pfr / 'image_transformations.csv', index=False)
    print(f"  Saved: image_transformations.csv  ({len(_rows)} images)")

    # 2. stellar_astrometry.csv — same columns as v1 plus is_member
    g = gaia_catalog.copy()
    g['n_hst_used']      = solver.gaia_n_hst_used

    n_align_per_star = np.zeros(solver.n_stars, dtype=int)
    for img in image_names:
        d_img = solver._img_data.get(img)
        if d_img is not None:
            np.add.at(n_align_per_star, d_img['sidx'][d_img['use_for_fit']], 1)
    g['n_hst_alignment'] = n_align_per_star

    chi2_hst, n_chi2 = compute_chi2_per_star(
        solver, r_current, v_mean, image_names, use_key='use_for_astrom')
    g['chi2_hst']     = chi2_hst
    g['n_det_chi2']   = n_chi2
    with np.errstate(invalid='ignore', divide='ignore'):
        g['chi2_hst_red'] = np.where(n_chi2 > 0, chi2_hst / (2 * n_chi2), np.nan)

    g['delta_racosdec_bp3m'] = v_mean_marg[:, 0]
    g['delta_dec_bp3m']      = v_mean_marg[:, 1]
    g['pmra_bp3m']           = v_mean_marg[:, 2]
    g['pmdec_bp3m']          = v_mean_marg[:, 3]
    g['parallax_bp3m']       = v_mean_marg[:, 4]

    g['sigma_delta_racosdec'] = np.sqrt(np.maximum(v_cov_full[:, 0, 0], 0.0))
    g['sigma_delta_dec']      = np.sqrt(np.maximum(v_cov_full[:, 1, 1], 0.0))
    g['sigma_pmra_bp3m']      = np.sqrt(np.maximum(v_cov_full[:, 2, 2], 0.0))
    g['sigma_pmdec_bp3m']     = np.sqrt(np.maximum(v_cov_full[:, 3, 3], 0.0))
    g['sigma_parallax_bp3m']  = np.sqrt(np.maximum(v_cov_full[:, 4, 4], 0.0))

    _sig = np.sqrt(np.maximum(np.diagonal(v_cov_full, axis1=1, axis2=2), 0.0))
    for col, i, j in [
        ('corr_dra_ddec', 0, 1), ('corr_dra_pmra', 0, 2),
        ('corr_dra_pmdec', 0, 3), ('corr_dra_plx', 0, 4),
        ('corr_ddec_pmra', 1, 2), ('corr_ddec_pmdec', 1, 3),
        ('corr_ddec_plx', 1, 4), ('corr_pmra_pmdec', 2, 3),
        ('corr_pmra_plx', 2, 4), ('corr_pmdec_plx', 3, 4),
    ]:
        denom = _sig[:, i] * _sig[:, j]
        g[col] = np.where(denom > 0, v_cov_full[:, i, j] / denom, np.nan)

    # Conditional (MAP alignment fixed) — v_mean is the conditional mean
    g['pmra_bp3m_cond']           = v_mean[:, 2]
    g['pmdec_bp3m_cond']          = v_mean[:, 3]
    g['parallax_bp3m_cond']       = v_mean[:, 4]
    g['sigma_pmra_bp3m_cond']     = np.sqrt(np.maximum(C_vT_final[:, 2, 2], 0.0))
    g['sigma_pmdec_bp3m_cond']    = np.sqrt(np.maximum(C_vT_final[:, 3, 3], 0.0))
    g['sigma_parallax_bp3m_cond'] = np.sqrt(np.maximum(C_vT_final[:, 4, 4], 0.0))

    _mem_mask = np.zeros(solver.n_stars, dtype=bool)
    _mem_mask[member_sidx] = True
    g['is_member'] = _mem_mask

    g.to_csv(output_pfr / 'stellar_astrometry.csv', index=False)
    print(f"  Saved: stellar_astrometry.csv  "
          f"({len(g)} stars, {solver.gaia_n_hst_used.sum()} HST detections)")

    # 3. Covariance arrays
    np.save(output_pfr / 'v_cov_marginalised.npy', v_cov)
    np.save(output_pfr / 'C_vT.npy',              C_vT_final)
    np.save(output_pfr / 'C_r.npy',               C_r)
    np.save(output_pfr / 'C_joint.npy',            C_shared_final)  # (n_r+2) × (n_r+2)
    print(f"  Saved: v_cov_marginalised.npy, C_vT.npy, C_r.npy, C_joint.npy")

    # 4. Detection flags
    _fit_data = {}; _astrom_data = {}; _idx_data = {}
    for img in image_names:
        d_img = solver._img_data.get(img)
        if d_img is None:
            continue
        _fit_data[img]    = d_img['use_for_fit']
        _astrom_data[img] = d_img.get('use_for_astrom', d_img['use_for_fit'])
        _idx_data[img]    = d_img['sidx']
    np.savez(output_pfr / 'use_for_fit.npz',    **_fit_data)
    np.savez(output_pfr / 'use_for_astrom.npz', **_astrom_data)
    np.savez(output_pfr / 'star_indices.npz',   **_idx_data)

    # 5. Per-detection GDC-frame residuals (same keys as v1 detections.npz)
    try:
        gdc_fin = solver.compute_gdc_residuals(r_current, v_mean, C_r=C_r, C_vT=C_vT_final)
        _det_data: dict = {}
        n_det_total = 0
        for img, rd in gdc_fin.items():
            _det_data[f'{img}_X_c']            = rd['X_c']
            _det_data[f'{img}_Y_c']            = rd['Y_c']
            _det_data[f'{img}_dx_gdc']         = rd['dx_gdc']
            _det_data[f'{img}_dy_gdc']         = rd['dy_gdc']
            _det_data[f'{img}_C_hst']          = rd['C_hst']
            _det_data[f'{img}_C_gdc_total']    = rd['C_gdc_total']
            _det_data[f'{img}_sidx']           = rd['sidx']
            _det_data[f'{img}_use_for_fit']    = rd['use_for_fit']
            _det_data[f'{img}_use_for_astrom'] = rd['use_for_astrom']
            n_det_total += len(rd['sidx'])
        np.savez_compressed(output_pfr / 'detections.npz', **_det_data)
        print(f"  Saved: detections.npz  ({len(gdc_fin)} images, {n_det_total} detections)")
    except Exception as _exc:
        print(f"  WARNING: detections.npz failed — {_exc}")

    # 6. mu_pop.json
    mu_result = {
        'mu_pop_ra_masyr':    float(mu_pop_current[0]),
        'mu_pop_dec_masyr':   float(mu_pop_current[1]),
        'sigma_mu_pop_ra':    float(sigma_mu_joint[0]),
        'sigma_mu_pop_dec':   float(sigma_mu_joint[1]),
        'n_members':          int(len(member_sidx)),
        'sigma_pm_masyr':     float(sigma_pm),
        'plx_pop_mas':        float(plx_pop),
        'sigma_plx_tot_mas':  float(sigma_plx_tot),
        'mu_pop_prior_ra':    float(mu_pop_prior[0]),
        'mu_pop_prior_dec':   float(mu_pop_prior[1]),
        'mu_pop_prior_sigma': float(mu_pop_prior_sigma),
    }
    with open(output_pfr / 'mu_pop.json', 'w') as _f:
        json.dump(mu_result, _f, indent=2)

    # 7. run_config.json
    with open(output_pfr / 'run_config.json', 'w') as _f:
        json.dump({
            'poly_order': poly_order, 'n_r_per_image': solver.N_R,
            'n_images': len(image_names),
            'n_stars': solver.n_stars, 'image_names': image_names,
            'sigma_pm': sigma_pm, 'plx_pop': plx_pop,
            'sigma_plx_tot': sigma_plx_tot,
            'mu_pop_prior_sigma': mu_pop_prior_sigma,
            'n_iter_mu': n_iter_mu, 'n_iter_joint': n_iter_joint,
            'member_sigma_clip': member_sigma_clip,
            'mu_pop_ra': float(mu_pop_current[0]),
            'mu_pop_dec': float(mu_pop_current[1]),
            'n_members': int(len(member_sidx)),
            'split_ccd': v1_split_ccd,
        }, _f, indent=2)
    print(f"  Saved: mu_pop.json, run_config.json")

    # 8. Star influence
    try:
        influence_df = solver.compute_star_influence(r_current, C_r, v_mean)
        influence_df.to_csv(output_pfr / 'star_influence.csv', index=False)
        print(f"  Saved: star_influence.csv  ({len(influence_df)} star-image pairs)")
    except Exception as _exc:
        print(f"  WARNING: star_influence.csv failed — {_exc}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not no_plots:
        _plot_dir = output_pfr / 'plots' / 'residuals'
        print(f"\n  Plotting before/after residual maps ({len(image_names)} images)...")
        try:
            _plot_pop_residual_maps(
                _plot_dir, image_names, solver,
                r_before=r_bp3m,   v_before=v_bp3m,
                r_after=r_current, v_after=v_mean,
                C_vT_after=C_vT_final,
                prefix='final',
            )
        except Exception as _exc:
            print(f"  WARNING: residual maps failed — {_exc}")
        # Restore geometry to final state for make_plots
        solver._update_geometry(r_current, v_mean)

        try:
            from bp3m.pipeline.plot_results import make_plots
            print("\n  Generating diagnostic plots...")
            make_plots(solver, imgs, gaia_catalog,
                       r_current, v_mean, v_mean_marg, v_cov, C_vT_final, C_r,
                       output_dir=output_pfr,
                       plot_residuals=False)
        except Exception as _exc:
            print(f"  WARNING: make_plots failed — {_exc}")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Results: {output_pfr}")
    return output_pfr


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog='bp3m-pop-fit',
        description='Population PM fitting post-processor (run after bp3m).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', required=True,
                        help='Target name (must match the field directory from bp3m)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Root output directory (same as passed to bp3m)')
    parser.add_argument('--sigma_pm', type=float, default=0.0075,
                        help='Cluster PM dispersion (mas/yr)')
    parser.add_argument('--plx_pop', type=float, default=0.003873,
                        help='Cluster parallax (mas)')
    parser.add_argument('--sigma_plx_tot', type=float, default=0.0001425,
                        help='Total parallax uncertainty (mas) for pop prior')
    parser.add_argument('--mu_pop_prior_sigma', type=float, default=0.5,
                        help='Gaussian prior width on μ_pop (mas/yr)')
    parser.add_argument('--n_iter_mu', type=int, default=5,
                        help='Phase 1 (μ-only) solve iterations')
    parser.add_argument('--n_iter_joint', type=int, default=10,
                        help='Phase 2 (joint r+μ) solve iterations')
    parser.add_argument('--member_sigma_clip', type=float, default=3.0,
                        help='Sigma threshold for membership selection')
    parser.add_argument('--poly_order', type=int, default=None,
                        help='Polynomial order (default: read from BP3M_results/run_config.json)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip diagnostic plot generation')

    args = parser.parse_args()

    run_pop_fit(
        output_dir=Path(args.output_dir).resolve(),
        field_name=args.name.replace(' ', '_'),
        sigma_pm=args.sigma_pm,
        plx_pop=args.plx_pop,
        sigma_plx_tot=args.sigma_plx_tot,
        mu_pop_prior_sigma=args.mu_pop_prior_sigma,
        n_iter_mu=args.n_iter_mu,
        n_iter_joint=args.n_iter_joint,
        member_sigma_clip=args.member_sigma_clip,
        poly_order=args.poly_order,
        no_plots=args.no_plots,
    )
