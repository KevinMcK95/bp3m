"""
bp3m-pop-fit — Population proper motion fitting post-processor.

Called after the standard bp3m run finishes.  Uses the exact same inputs as
bp3m (FLC pipeline data loader: Bayesian_PMs/ + Gaia/) together with the bp3m
alignment outputs (r_hat, use_for_fit flags from BP3M_results/) to fit the
cluster population mean proper motion μ_pop and jointly refine the per-image
alignment.

Steps
-----
1. Load same data as bp3m (load_image_data_flc: Gaia CSVs + Bayesian_PMs/).
2. Load bp3m outputs: r_hat, use_for_fit, use_for_astrom from BP3M_results/.
3. Estimate initial μ_pop from sigma-clipped Gaia PMs; select initial members.
4. Phase 1 (μ-only): hold r fixed at bp3m values, solve for μ_pop — avoids r–μ
   degeneracy by construction.
5. Phase 2 (joint):  jointly refine r and μ_pop; iterate member selection.
6. Save results to {target}/BP3M_pop_fit_results/.
7. Plot per-visit residual maps (before / after) in plots/residuals/.

Member prior
------------
Members receive the cluster PM prior  N(μ_pop, σ_pm² I₂) and the LVD parallax
prior N(plx_pop, σ_plx_tot²) on top of their Gaia prior.  Non-members retain
the standard Gaia prior unchanged from bp3m.

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
    """
    Sigma-clipped mean of Gaia proper motions (5p/6p stars only).
    Returns (2,) array [pmra_mean, pmdec_mean] in mas/yr.
    """
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
    print(f"  Initial μ_pop estimate (Gaia σ-clip, n={keep.sum()}/{len(pmra)}): "
          f"({mu[0]:+.4f}, {mu[1]:+.4f}) mas/yr")
    return mu


# ── Member selection (from posterior stellar astrometry) ──────────────────────

def _select_members_from_a(
    a_arr: np.ndarray,
    mu_pop: np.ndarray,
    n_hst: np.ndarray,
    sigma_clip: float = 3.0,
    n_iter: int = 5,
    min_members: int = 5,
    init_window_masyr: float = 2.0,
    pm_sys_floor: float = 0.2,
    eligible_sidx: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sigma-clip on PM distance from mu_pop to identify likely cluster members.
    Only stars with ≥ 1 contributing HST detection are eligible.
    """
    eligible = n_hst >= 1
    if eligible_sidx is not None:
        _mask = np.zeros(len(n_hst), bool)
        _mask[eligible_sidx] = True
        eligible = eligible & _mask
    eidx = np.where(eligible)[0]
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
    """
    Select initial member candidates from Gaia catalog PMs.
    Uses a radius of member_sigma_clip × max(σ_pm, pm_sys_floor) around mu_pop.
    Returns global star index array.
    """
    _radius = member_sigma_clip * max(sigma_pm, pm_sys_floor)
    pmra  = gaia_catalog['pmra'].to_numpy(float)
    pmdec = gaia_catalog['pmdec'].to_numpy(float)
    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    dist   = np.where(finite,
                      np.hypot(pmra - mu_pop[0], pmdec - mu_pop[1]),
                      np.inf)
    return np.where(dist < _radius)[0]


# ── Load bp3m r_hat from BP3M_results/image_transformations.csv ───────────────

def _load_bp3m_rhat(
    data_root: Path,
    field_name: str,
    image_names: list[str],
    nr: int,
) -> np.ndarray:
    """
    Read per-image r_hat from BP3M_results/image_transformations.csv.
    Returns (n_images * nr,) array ordered by image_names.
    """
    xform_path = data_root / field_name / 'BP3M_results' / 'image_transformations.csv'
    if not xform_path.exists():
        raise FileNotFoundError(
            f"BP3M_results/image_transformations.csv not found at {xform_path}. "
            "Run bp3m first."
        )
    xdf = pd.read_csv(xform_path)
    img_to_row = {str(row['image_name']): row for _, row in xdf.iterrows()}

    r_hat = np.zeros(len(image_names) * nr)
    missing = []
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
            r_hat[cs + 6] = float(row.get('delta_ra0_mas', 0.0)) / 1000.0
            r_hat[cs + 7] = float(row.get('delta_dec0_mas', 0.0)) / 1000.0
        for k in range(8, nr):
            r_hat[cs + k] = float(row.get(f'r_{k}', 0.0))
    if missing:
        print(f"  WARNING: {len(missing)} images missing from image_transformations.csv; "
              "using r_init for them")
    return r_hat


# ── Override solver use_for_fit/use_for_astrom from BP3M_results ──────────────

def _apply_bp3m_flags(
    data_root: Path,
    field_name: str,
    solver,
    image_names: list[str],
    gaia_catalog: pd.DataFrame,
) -> None:
    """
    Override solver use_for_fit and use_for_astrom flags from BP3M_results.
    Uses Gaia_id matching (int64) to avoid float roundtrip corruption.
    """
    bp3m_dir  = data_root / field_name / 'BP3M_results'
    _uff_path = bp3m_dir / 'use_for_fit.npz'
    _ufa_path = bp3m_dir / 'use_for_astrom.npz'
    _si_path  = bp3m_dir / 'star_indices.npz'
    _sa_path  = bp3m_dir / 'stellar_astrometry.csv'

    if not all(p.exists() for p in [_uff_path, _si_path, _sa_path]):
        print("  WARNING: BP3M_results/use_for_fit.npz not found — "
              "using default quality-cut flags")
        return

    _uff = np.load(_uff_path)
    _ufa = np.load(_ufa_path) if _ufa_path.exists() else None
    _si  = np.load(_si_path)
    _sa  = pd.read_csv(_sa_path, dtype={'Gaia_id': np.int64})
    _bp3m_gids = _sa['Gaia_id'].to_numpy(np.int64)

    def _build_gid_set(npz_file):
        out: dict[str, frozenset] = {}
        for _img in npz_file.files:
            _mask = npz_file[_img].astype(bool)
            _sidx = _si[_img]
            _gids = _bp3m_gids[_sidx[_mask]]
            out[_img] = frozenset(int(g) for g in _gids if g > 0)
        return out

    fit_per_img   = _build_gid_set(_uff)
    astrom_per_img = _build_gid_set(_ufa) if _ufa is not None else fit_per_img

    # Build solver star_index → Gaia_id mapping
    _n_sol = solver.C_survey_inv.shape[0]
    _sol_gid = np.zeros(_n_sol, dtype=np.int64)
    for _gid, _idx in solver.star_id_to_idx.items():
        _sol_gid[_idx] = np.int64(_gid)

    n_fit = 0; n_astrom = 0
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
        n_fit   += int(d['use_for_fit'].sum())
        n_astrom += int(d['use_for_astrom'].sum())

    all_fit_gids: set = set(); all_astrom_gids: set = set()
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        gids_j = _sol_gid[d['sidx']]
        all_fit_gids.update(int(g) for g in gids_j[d['use_for_fit']])
        all_astrom_gids.update(int(g) for g in gids_j[d['use_for_astrom']])

    print(f"  Applied bp3m use_for_fit:    "
          f"{len(all_fit_gids)} unique stars, {n_fit} detections")
    print(f"  Applied bp3m use_for_astrom: "
          f"{len(all_astrom_gids)} unique stars, {n_astrom} detections"
          + ("" if _ufa is not None else " (use_for_astrom.npz not found — using use_for_fit)"))


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

    Population prior structure
    --------------------------
    Members    : H_vv[2:4, 2:4] += σ_pm^{-2} I₂  (PM coupled to μ_pop)
                 H_vv[4, 4]     += σ_plx^{-2}     (parallax independent prior)
    Non-members: unchanged from solver defaults (Gaia prior)

    Parameters
    ----------
    fix_r : if True, only solve for Δμ_pop; r stays at r_current.

    Returns
    -------
    r_hat, mu_pop_hat, C_shared, C_vT, a_arr, a_align_arr
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(x, **kw):
            return x

    N_V   = 5          # [Δα*, Δδ, μ_α*, μ_δ, plx]
    nr    = solver.N_R
    n_img = len(image_names)
    n_r   = n_img * nr
    n_stars = solver.n_stars

    sigma_pm_inv_sq  = sigma_pm ** -2
    sigma_plx_inv_sq = sigma_plx_tot ** -2

    if fix_r:
        n_shared = 2
    else:
        n_shared = n_r + 2
        idx_r  = slice(0, n_r)
        idx_mu = slice(n_r, n_r + 2)

    # ── H_vv: Gaia prior base ─────────────────────────────────────────────────
    H_vv = solver.C_survey_inv.copy()

    # Add diffuse prior for non-member 2p Gaia stars (C_survey_inv[2:,2:]=0 for 2p)
    _nonmem = np.ones(n_stars, dtype=bool)
    _nonmem[member_sidx] = False
    _nonmem_2p = _nonmem & (solver._C_VG_inv_per_star[:, 2] > 0)
    if _nonmem_2p.any():
        for _k in range(N_V):
            H_vv[_nonmem_2p, _k, _k] += solver._C_VG_inv_per_star[_nonmem_2p, _k]

    # Population prior for member stars
    _pos_inv_sq = (1e6) ** -2   # negligible; ensures H_vv invertible for 2p members
    H_vv[member_sidx, 0, 0] += _pos_inv_sq
    H_vv[member_sidx, 1, 1] += _pos_inv_sq
    H_vv[member_sidx, 2, 2] += sigma_pm_inv_sq
    H_vv[member_sidx, 3, 3] += sigma_pm_inv_sq
    H_vv[member_sidx, 4, 4] += sigma_plx_inv_sq

    # Information vectors: start from Gaia prior contribution
    h_align = solver.C_survey_inv_dot_v.copy()
    h_all   = solver.C_survey_inv_dot_v.copy()

    # Linearise population prior at μ_pop_current: h += σ^{-2} μ_pop
    h_align[member_sidx, 2] += sigma_pm_inv_sq * mu_pop_current[0]
    h_align[member_sidx, 3] += sigma_pm_inv_sq * mu_pop_current[1]
    h_all  [member_sidx, 2] += sigma_pm_inv_sq * mu_pop_current[0]
    h_all  [member_sidx, 3] += sigma_pm_inv_sq * mu_pop_current[1]
    h_align[member_sidx, 4] += sigma_plx_inv_sq * plx_pop
    h_all  [member_sidx, 4] += sigma_plx_inv_sq * plx_pop

    # ── Per-image data accumulation ────────────────────────────────────────────
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

        sidx     = d['sidx']
        use_fit  = d['use_for_fit']
        use_any  = use_fit | d.get('use_for_astrom', use_fit)

        sidx_any = sidx[use_any]
        sidx_fit = sidx[use_fit]
        active_glob[sidx_any] = True

        cs  = j_idx * nr
        r_j = r_current[cs:cs + nr]

        JU  = d['JU']
        X   = d['X_mat']
        xys = d.get('xys_orig', d['xys'])

        Cs     = solver._compute_Cs(img, r_j)
        Cs_inv = np.linalg.inv(Cs)

        x_pred  = np.einsum('nkl,l->nk', X, r_j)
        x_resid = xys - x_pred

        JUT_Cs = np.einsum('nki,nkl->nil', JU, Cs_inv)
        K = np.einsum('nik,nkl->nil', JUT_Cs, X)
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

    # ── Invert H_vv → C_vT, compute stellar posteriors ───────────────────────
    C_vT = np.zeros_like(H_vv)
    _active_sidx = np.where(active_glob)[0]
    if len(_active_sidx) > 0:
        C_vT[_active_sidx] = np.linalg.inv(H_vv[_active_sidx])
    a_align = np.einsum('nij,nj->ni', C_vT, h_align)
    a       = np.einsum('nij,nj->ni', C_vT, h_all)

    # ── Assemble Lambda and rhs ───────────────────────────────────────────────
    Lambda = np.zeros((n_shared, n_shared))
    rhs    = np.zeros(n_shared)

    n_mem = len(member_sidx)

    # H_μμ direct: prior precision + Σ σ^{-2} per member
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

    # ── Global Schur correction for μ block (member stars) ────────────────────
    if n_mem > 0:
        Cv_m = C_vT[member_sidx]   # (n_mem, 5, 5)

        # (μ, μ) Schur: -σ^{-4} Σ C_vT[mem, 2:4, 2:4]
        mu_mu_schur = sigma_pm_inv_sq ** 2 * Cv_m[:, 2:4, 2:4].sum(axis=0)
        if fix_r:
            Lambda -= mu_mu_schur
        else:
            Lambda[idx_mu, idx_mu] -= mu_mu_schur

        # μ rhs: +σ^{-2} Σ a[mem, 2:4]
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

            # (r, μ) Schur: -σ^{-2} K_mem^T C_vT[:, :, 2:4]
            if use_fmem.any():
                sidx_fm  = sidx[use_fmem]
                K_fm     = K_img[img][use_fmem]
                CvT_M_fm = C_vT[sidx_fm, :, 2:4]          # (n_fm, 5, 2)
                KT_CvT_M = np.einsum('nji,njk->ik', K_fm, CvT_M_fm)  # (N_R, 2)
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
        return r_current.copy(), mu_pop_current + delta, C_shared, C_vT, a, a_align
    else:
        return (r_current + delta[idx_r],
                mu_pop_current + delta[idx_mu],
                C_shared, C_vT, a, a_align)


# ── Per-visit residual plots (before / after) ─────────────────────────────────

def _plot_pop_residual_maps(
    output_dir: Path,
    image_names: list[str],
    solver,
    filtered_spi: dict,
    arrays_before: dict,
    arrays_after: dict,
    stage_labels: list[str] | None = None,
    prefix: str = 'final',
    vclip: float | None = None,
) -> None:
    """
    Per-visit 2-row detector residual maps (before / after pop-fit).
    Columns: dx_gdc (px), dy_gdc (px), dx/σ_x, dy/σ_y.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.time import Time as _Time
    from collections import defaultdict as _defaultdict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage_labels is None:
        stage_labels = ['Before (bp3m)', 'After (pop-fit)']

    visit_groups: dict[str, list] = _defaultdict(list)
    for img in image_names:
        root = img[:-3] if img.endswith(('_hi', '_lo')) else img
        visit_groups[root].append(img)

    stages = [arrays_before, arrays_after]
    saved  = 0

    for root, imgs in visit_groups.items():
        xr_by_s = [[] for _ in range(2)]
        yr_by_s = [[] for _ in range(2)]
        dx_by_s = [[] for _ in range(2)]
        dy_by_s = [[] for _ in range(2)]
        sx_all  = []
        sy_all  = []
        total_n = 0
        years   = []

        for img in imgs:
            spi = filtered_spi.get(img)
            if spi is None or len(spi) == 0:
                continue
            xc_key = f'{img}_X_c'
            if xc_key not in arrays_before:
                continue

            xc_all = arrays_before[xc_key].astype(float)
            yc_all = arrays_before[f'{img}_Y_c'].astype(float)
            xc_spi = spi['X'].to_numpy(float) - 2048.0
            yc_spi = spi['Y'].to_numpy(float) - 2048.0

            pos_dict = {(round(float(x), 2), round(float(y), 2)): k
                        for k, (x, y) in enumerate(zip(xc_all, yc_all))}
            match_idx = np.array([
                pos_dict.get((round(float(x), 2), round(float(y), 2)), -1)
                for x, y in zip(xc_spi, yc_spi)
            ])
            valid = match_idx >= 0
            if not valid.any():
                continue

            m_idx = match_idx[valid]
            x_raw = (spi['X_orig'].to_numpy(float)[valid]
                     if 'X_orig' in spi.columns else xc_spi[valid] + 2048.0)
            y_raw = (spi['Y_orig'].to_numpy(float)[valid]
                     if 'Y_orig' in spi.columns else yc_spi[valid] + 2048.0)

            for si, arr in enumerate(stages):
                dxk = f'{img}_dx_gdc'; dyk = f'{img}_dy_gdc'
                dx = arr[dxk][m_idx].astype(float) if dxk in arr else np.zeros(m_idx.size)
                dy = arr[dyk][m_idx].astype(float) if dyk in arr else np.zeros(m_idx.size)
                xr_by_s[si].append(x_raw); yr_by_s[si].append(y_raw)
                dx_by_s[si].append(dx);    dy_by_s[si].append(dy)

            _img_d = getattr(solver, '_img_data', {}).get(img)
            _alpha = float(_img_d.get('alpha_applied', 1.0)) if _img_d else 1.0
            if 'x_hst_err' in spi.columns and 'y_hst_err' in spi.columns:
                sx_all.append(spi['x_hst_err'].to_numpy(float)[valid] * _alpha)
                sy_all.append(spi['y_hst_err'].to_numpy(float)[valid] * _alpha)

            total_n += int(valid.sum())
            try:
                years.append(float(_Time(float(solver.images[img]['hst_time_mjd']),
                                         format='mjd').jyear))
            except Exception:
                pass

        if total_n == 0:
            continue

        for si in range(2):
            xr_by_s[si] = np.concatenate(xr_by_s[si]) if xr_by_s[si] else np.array([])
            yr_by_s[si] = np.concatenate(yr_by_s[si]) if yr_by_s[si] else np.array([])
            dx_by_s[si] = np.concatenate(dx_by_s[si]) if dx_by_s[si] else np.array([])
            dy_by_s[si] = np.concatenate(dy_by_s[si]) if dy_by_s[si] else np.array([])

        sigma_x   = np.concatenate(sx_all) if sx_all else None
        sigma_y   = np.concatenate(sy_all) if sy_all else None
        has_sigma = (sigma_x is not None
                     and np.any(np.isfinite(sigma_x) & (sigma_x > 0)))

        if vclip is None:
            _vals = np.concatenate([np.abs(dx_by_s[0]), np.abs(dy_by_s[0])])
            _fin  = _vals[np.isfinite(_vals)]
            _vc   = max(float(np.percentile(_fin, 97)) if len(_fin) > 0 else 0.3, 0.05)
        else:
            _vc = float(vclip)
        _vc_sig = 2.0

        n_cols = 4 if has_sigma else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 5, 7),
                                 sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0.08, 'wspace': 0.06})
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]

        yr_str = (f' ({min(years):.2f}–{max(years):.2f} yr)' if years else '')
        fig.suptitle(f'{root}{yr_str}  n={total_n}', fontsize=10, y=0.99)

        for row_i, stage_lbl in enumerate(stage_labels):
            raw_pairs = [(dx_by_s[row_i], 'dx_gdc (px)'),
                         (dy_by_s[row_i], 'dy_gdc (px)')]
            if has_sigma:
                _sx = np.where(sigma_x > 0, sigma_x, np.nan)
                _sy = np.where(sigma_y > 0, sigma_y, np.nan)
                sig_pairs = [(dx_by_s[row_i] / _sx, 'dx / σ_x'),
                             (dy_by_s[row_i] / _sy, 'dy / σ_y')]
            else:
                sig_pairs = []
            all_pairs = raw_pairs + sig_pairs
            clims = [(-_vc, _vc)] * 2 + [(-_vc_sig, _vc_sig)] * len(sig_pairs)

            for col_i, ((vals, clbl), (vmin, vmax)) in enumerate(zip(all_pairs, clims)):
                ax = axes[row_i, col_i]
                sc = ax.scatter(xr_by_s[row_i], yr_by_s[row_i], c=vals,
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
    Run population PM fitting using bp3m inputs and outputs.

    Reads the same data as bp3m (Bayesian_PMs/ + Gaia/) and uses
    BP3M_results/ outputs (r_hat, use_for_fit flags) as the starting point.
    Saves results to BP3M_pop_fit_results/.
    """
    from bp3m.data_loader_flc import load_image_data_flc
    from bp3m.data_loader import build_index_maps
    from bp3m.solver import BP3MSolver

    t_start = time.time()
    data_root  = Path(output_dir)
    output_pfr = data_root / field_name / 'BP3M_pop_fit_results'
    output_pfr.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 60)
    print("BP3M pop-fit: population PM fitting")
    print("─" * 60)
    print(f"  field={field_name}")
    print(f"  σ_pm={sigma_pm} mas/yr  plx_pop={plx_pop} mas  "
          f"σ_plx_tot={sigma_plx_tot} mas")
    print(f"  μ_pop prior σ={mu_pop_prior_sigma} mas/yr  "
          f"member_sigma_clip={member_sigma_clip}")

    # ── poly_order from run_config.json ───────────────────────────────────────
    if poly_order is None:
        _cfg = data_root / field_name / 'BP3M_results' / 'run_config.json'
        if _cfg.exists():
            with open(_cfg) as _f:
                poly_order = int(json.load(_f).get('poly_order', 1))
            print(f"  poly_order={poly_order} (from BP3M_results/run_config.json)")
        else:
            poly_order = 1
            print(f"  poly_order={poly_order} (default; run_config.json not found)")

    # ── Load data (same as bp3m) ───────────────────────────────────────────────
    print(f"\n  Loading bp3m input data for '{field_name}'...")
    imgs_raw, stars_per_image, gaia_catalog = load_image_data_flc(
        data_root, field_name)
    if imgs_raw is None or len(imgs_raw) == 0:
        raise RuntimeError(f"No usable images found for '{field_name}'.")

    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)
    imgs         = {n: imgs_raw[n] for n in image_names if n in imgs_raw}
    filtered_spi = {n: stars_per_image[n] for n in image_names}

    # Filter gaia_catalog to stars that appear in at least one image
    observed_ids = set()
    for spi in filtered_spi.values():
        observed_ids.update(int(g) for g in spi['Gaia_id'].values)
    gaia_catalog = (gaia_catalog[gaia_catalog['Gaia_id'].isin(observed_ids)]
                    .reset_index(drop=True))
    star_id_to_idx = {int(gid): i for i, gid in enumerate(gaia_catalog['Gaia_id'])}
    star_id_to_idx_rebuild, image_names_rb, star_in_image_rb = build_index_maps(
        filtered_spi, gaia_catalog)
    # Prefer the rebuilt index (consistent with filtered gaia_catalog)
    star_id_to_idx = star_id_to_idx_rebuild
    image_names    = image_names_rb
    star_in_image  = star_in_image_rb

    print(f"  Stars: {len(gaia_catalog)}   Images: {len(image_names)}")

    # ── Initialise solver ─────────────────────────────────────────────────────
    solver = BP3MSolver(
        imgs, filtered_spi, gaia_catalog,
        star_id_to_idx, image_names, star_in_image,
        poly_order=poly_order,
    )
    print(f"  Solver: {solver.n_stars} stars  {solver.N_R} params/image")

    # ── Load bp3m r_hat ────────────────────────────────────────────────────────
    print("\n  Loading bp3m alignment parameters...")
    r_bp3m = _load_bp3m_rhat(data_root, field_name, image_names, solver.N_R)
    solver._update_R(r_bp3m)
    for _img in image_names:
        _d = solver._img_data.get(_img)
        if _d is not None and 'xys_orig' not in _d:
            _d['xys_orig'] = _d['xys'].copy()
    solver._update_geometry(r_bp3m, solver.v_survey)

    # ── Apply bp3m use_for_fit / use_for_astrom flags ─────────────────────────
    print("\n  Applying bp3m detection flags...")
    _apply_bp3m_flags(data_root, field_name, solver, image_names, gaia_catalog)

    # ── Count HST detections per star ──────────────────────────────────────────
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

    # ── "Before" GDC residuals ────────────────────────────────────────────────
    if not no_plots:
        print("\n  Computing bp3m GDC residuals (before)...")
        try:
            arrays_before = solver.compute_gdc_residuals(r_bp3m, solver.v_survey)
            # flatten: solver.compute_gdc_residuals returns dict[img -> dict]
            _arr_b: dict = {}
            for img, rd in arrays_before.items():
                for k, v in rd.items():
                    _arr_b[f'{img}_{k}'] = v
            arrays_before = _arr_b
        except Exception as _exc:
            print(f"  WARNING: before residuals failed — {_exc}")
            arrays_before = None
    else:
        arrays_before = None

    # ── Phase 1: μ-only solve (r fixed at bp3m) ───────────────────────────────
    print(f"\n  Phase 1: μ-only solve ({n_iter_mu} iterations, r fixed)...")
    r_current = r_bp3m.copy()
    C_shared_mu = None
    for mu_iter in range(n_iter_mu):
        _, mu_pop_new, C_shared_mu, C_vT, a_arr, _ = _joint_solve_pop(
            solver, image_names,
            member_sidx, mu_pop_current,
            sigma_pm, plx_pop, sigma_plx_tot,
            C_pop_prior_inv, mu_pop_prior,
            r_current, fix_r=True,
        )
        delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
        mu_pop_current = mu_pop_new
        print(f"    iter {mu_iter + 1}/{n_iter_mu}: "
              f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f}) mas/yr  "
              f"Δμ={delta_mu:.4e}")

        member_sidx = _select_members_from_a(
            a_arr, mu_pop_current, _n_hst_det, sigma_clip=member_sigma_clip)
        print(f"    members: {len(member_sidx)}")

        if delta_mu < 1e-6:
            print(f"    Converged at iteration {mu_iter + 1}")
            break

    if C_shared_mu is not None:
        sigma_mu_1 = np.sqrt(np.diag(C_shared_mu))
        print(f"  Phase 1 final: μ_pop=({mu_pop_current[0]:+.4f} ± {sigma_mu_1[0]:.4f}, "
              f"{mu_pop_current[1]:+.4f} ± {sigma_mu_1[1]:.4f}) mas/yr")

    # ── Phase 2: joint solve (r + μ_pop) ─────────────────────────────────────
    print(f"\n  Phase 2: joint solve ({n_iter_joint} iterations)...")
    C_shared_joint = None
    for jt_iter in range(n_iter_joint):
        r_new, mu_pop_new, C_shared_joint, C_vT, a_arr, _ = _joint_solve_pop(
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
        print(f"    iter {jt_iter + 1}/{n_iter_joint}: "
              f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f})  "
              f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}")

        member_sidx = _select_members_from_a(
            a_arr, mu_pop_current, _n_hst_det, sigma_clip=member_sigma_clip)
        print(f"    members: {len(member_sidx)}")

        if delta_r < 1e-6 and delta_mu < 1e-6:
            print(f"    Converged at iteration {jt_iter + 1}")
            break

    sigma_mu_joint = (np.sqrt(np.diag(C_shared_joint[-2:, -2:]))
                      if C_shared_joint is not None else np.array([np.nan, np.nan]))
    print(f"\n  Final: μ_pop=({mu_pop_current[0]:+.4f} ± {sigma_mu_joint[0]:.4f}, "
          f"{mu_pop_current[1]:+.4f} ± {sigma_mu_joint[1]:.4f}) mas/yr")
    print(f"  Final members: {len(member_sidx)}")

    # ── Final posterior pass at convergence ───────────────────────────────────
    print("\n  Final posterior pass...")
    solver._update_R(r_current)
    solver._update_geometry(r_current, solver.v_survey)
    _, _, C_shared_final, C_vT_final, v_mean, _ = _joint_solve_pop(
        solver, image_names,
        member_sidx, mu_pop_current,
        sigma_pm, plx_pop, sigma_plx_tot,
        C_pop_prior_inv, mu_pop_prior,
        r_current, fix_r=False,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n  Saving results...")

    # image_transformations.csv
    _rows = []
    for j_idx, img in enumerate(image_names):
        cs    = j_idx * solver.N_R
        r_j   = r_current[cs:cs + solver.N_R]
        d_img = solver._img_data.get(img, {})
        _rows.append(dict(
            image_name=img,
            n_stars_alignment=int(np.sum(d_img.get('use_for_fit', np.zeros(0, bool)))),
            n_stars_astrometry_only=int(np.sum(
                d_img.get('use_for_astrom', d_img.get('use_for_fit', np.zeros(0, bool)))
                & ~d_img.get('use_for_fit', np.zeros(0, bool)))),
            a=r_j[0], b=r_j[1], c=r_j[2], d=r_j[3],
            w=r_j[4], z=r_j[5],
            delta_ra0_mas=r_j[6] * 1000 if solver.N_R > 6 else 0.0,
            delta_dec0_mas=r_j[7] * 1000 if solver.N_R > 7 else 0.0,
            alpha=float(d_img.get('alpha_applied', 1.0)),
            **{f'r_{k}': float(r_j[k]) for k in range(8, solver.N_R)},
        ))
    pd.DataFrame(_rows).to_csv(output_pfr / 'image_transformations.csv', index=False)
    print(f"  Saved: image_transformations.csv  ({len(_rows)} images)")

    # stellar_astrometry.csv
    g = gaia_catalog.copy()
    g['pmra_bp3m']           = v_mean[:, 2]
    g['pmdec_bp3m']          = v_mean[:, 3]
    g['parallax_bp3m']       = v_mean[:, 4]
    g['delta_racosdec_bp3m'] = v_mean[:, 0]
    g['delta_dec_bp3m']      = v_mean[:, 1]
    g['sigma_pmra_bp3m']     = np.sqrt(C_vT_final[:, 2, 2])
    g['sigma_pmdec_bp3m']    = np.sqrt(C_vT_final[:, 3, 3])
    g['sigma_parallax_bp3m'] = np.sqrt(C_vT_final[:, 4, 4])
    _mem_mask = np.zeros(solver.n_stars, dtype=bool)
    _mem_mask[member_sidx] = True
    g['is_member'] = _mem_mask
    g.to_csv(output_pfr / 'stellar_astrometry.csv', index=False)
    print(f"  Saved: stellar_astrometry.csv  ({len(g)} stars)")

    # detection flags
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

    # detections.npz (GDC-frame residuals at converged r)
    try:
        gdc_resid = solver.compute_gdc_residuals(r_current, v_mean, C_vT=C_vT_final)
        _det: dict = {}
        for img, rd in gdc_resid.items():
            for k, v in rd.items():
                _det[f'{img}_{k}'] = v
        np.savez_compressed(output_pfr / 'detections.npz', **_det)
        print(f"  Saved: detections.npz  ({len(gdc_resid)} images)")
    except Exception as _exc:
        print(f"  WARNING: detections.npz failed — {_exc}")

    # mu_pop.json
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

    # run_config.json
    with open(output_pfr / 'run_config.json', 'w') as _f:
        json.dump({
            'poly_order': poly_order, 'n_images': len(image_names),
            'n_stars': solver.n_stars, 'image_names': image_names,
            'sigma_pm': sigma_pm, 'plx_pop': plx_pop,
            'sigma_plx_tot': sigma_plx_tot,
            'mu_pop_prior_sigma': mu_pop_prior_sigma,
            'n_iter_mu': n_iter_mu, 'n_iter_joint': n_iter_joint,
            'member_sigma_clip': member_sigma_clip,
            'mu_pop_ra': float(mu_pop_current[0]),
            'mu_pop_dec': float(mu_pop_current[1]),
            'n_members': int(len(member_sidx)),
        }, _f, indent=2)
    print(f"  Saved: mu_pop.json, run_config.json, use_for_fit.npz, star_indices.npz")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not no_plots and arrays_before is not None:
        print("\n  Computing pop-fit GDC residuals (after)...")
        _plot_dir = output_pfr / 'plots' / 'residuals'
        try:
            gdc_after = solver.compute_gdc_residuals(r_current, v_mean, C_vT=C_vT_final)
            _arr_a: dict = {}
            for img, rd in gdc_after.items():
                for k, v in rd.items():
                    _arr_a[f'{img}_{k}'] = v
            arrays_after = _arr_a
            print(f"\n  Plotting residual maps ({len(image_names)} images)...")
            _plot_pop_residual_maps(
                _plot_dir, image_names, solver, filtered_spi,
                arrays_before, arrays_after,
                stage_labels=['bp3m (before)', 'pop-fit (after)'],
                prefix='final',
            )
        except Exception as _exc:
            print(f"  WARNING: residual maps failed — {_exc}")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Results written to: {output_pfr}")
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
