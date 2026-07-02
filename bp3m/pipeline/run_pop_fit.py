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
    C_vT: np.ndarray,
    sigma_pm: float,
    sigma_clip: float = 3.0,
    min_members: int = 5,
    pm_sys_floor: float = 0.2,
) -> np.ndarray:
    """Select members using the same per-star PM uncertainty as _select_initial_members.

    Per-star threshold = sigma_clip × det(C_pm_i + (sigma_pm² + pm_sys_floor²)·I)^(1/4),
    identical to the _select_initial_members formula.  Only stars with ≥1 HST
    detection are eligible.
    """
    eidx = np.where(n_hst >= 1)[0]
    if len(eidx) < min_members:
        return eidx

    pmra  = a_arr[eidx, 2]
    pmdec = a_arr[eidx, 3]
    dist  = np.hypot(pmra - mu_pop[0], pmdec - mu_pop[1])

    extra    = sigma_pm ** 2 + pm_sys_floor ** 2
    C_pm     = C_vT[eidx, 2:4, 2:4]                   # (n_elig, 2, 2)
    c00      = C_pm[:, 0, 0] + extra
    c11      = C_pm[:, 1, 1] + extra
    c01      = C_pm[:, 0, 1]
    det_C    = np.maximum(c00 * c11 - c01 ** 2, 1e-30)
    geom_sig = det_C ** 0.25                            # (n_elig,)

    keep = np.isfinite(dist) & (dist < sigma_clip * geom_sig)
    if keep.sum() < min_members:
        keep = np.isfinite(dist)

    return eidx[keep]


def _compute_free_stellar_posterior(
    a: np.ndarray,
    C_vT: np.ndarray,
    member_sidx: np.ndarray,
    sigma_pm: float,
    sigma_plx_tot: float,
    mu_pop: np.ndarray,
    plx_pop: float,
    C_VG_inv_per_star: np.ndarray,
) -> tuple:
    """Return (a_free, C_vT_free) with the population prior removed for member stars.

    The regular solve bakes in sigma_pm^{-2} for members, pulling a[member, 2:4]
    toward mu_pop regardless of what the data says.  Evaluating membership from
    that posterior means members can never be demoted.  This function undoes the
    population prior contribution so _select_members_from_a sees what Gaia + HST
    alone imply.

    For 2p stars (no Gaia PM), removing the population prior makes H_free
    rank-deficient in the PM/parallax directions.  We add back the same diffuse
    prior that _joint_solve_pop applies to 2p non-members (C_VG_inv_per_star),
    giving exactly the posterior those stars would have as non-members.
    """
    if len(member_sidx) == 0:
        return a, C_vT

    # Inactive members with no data of any kind have C_vT left at zero by
    # _joint_solve_pop (H_vv was singular).  They are excluded from membership
    # evaluation by _select_members_from_a (n_hst == 0) anyway.
    _has_cvT = np.diagonal(C_vT[member_sidx], axis1=1, axis2=2).any(axis=1)
    member_sidx = member_sidx[_has_cvT]
    if len(member_sidx) == 0:
        return a, C_vT

    sigma_pm_inv_sq  = sigma_pm ** -2
    sigma_plx_inv_sq = sigma_plx_tot ** -2

    a_free    = a.copy()
    C_vT_free = C_vT.copy()

    # H_vv = C_vT^{-1}; recover information vector h = H_vv @ a
    H_mem = np.linalg.inv(C_vT[member_sidx])
    h_mem = np.einsum('nij,nj->ni', H_mem, a[member_sidx])

    # Strip the population prior from the normal equations
    H_mem[:, 2, 2] -= sigma_pm_inv_sq
    H_mem[:, 3, 3] -= sigma_pm_inv_sq
    H_mem[:, 4, 4] -= sigma_plx_inv_sq
    h_mem[:, 2]    -= sigma_pm_inv_sq * mu_pop[0]
    h_mem[:, 3]    -= sigma_pm_inv_sq * mu_pop[1]
    h_mem[:, 4]    -= sigma_plx_inv_sq * plx_pop

    # For 2p member stars, add the diffuse prior that non-members get
    # (_joint_solve_pop line 350-355).  These stars have no Gaia PM so
    # H_free would otherwise be rank-deficient in the PM/parallax directions.
    # The diffuse prior has zero mean so only H_mem (not h_mem) changes.
    needs_diffuse = C_VG_inv_per_star[member_sidx, 2] > 0
    if needs_diffuse.any():
        ndx = member_sidx[needs_diffuse]
        for _k in range(5):
            H_mem[needs_diffuse, _k, _k] += C_VG_inv_per_star[ndx, _k]

    C_vT_free[member_sidx] = np.linalg.inv(H_mem)
    a_free[member_sidx]    = np.einsum('nij,nj->ni', C_vT_free[member_sidx], h_mem)

    return a_free, C_vT_free


# ── Initial member selection from Gaia catalog PMs ───────────────────────────

def _select_initial_members(
    pmra: np.ndarray,
    pmdec: np.ndarray,
    sigma_pmra: np.ndarray,
    sigma_pmdec: np.ndarray,
    corr_pm: np.ndarray,
    mu_pop: np.ndarray,
    member_sigma_clip: float,
    sigma_pm: float,
    pm_sys_floor: float = 0.2,
) -> np.ndarray:
    """Select initial member candidates from BP3M v1 (or Gaia fallback) PMs.

    Per-star threshold = member_sigma_clip × geometric mean PM uncertainty,
    where the total covariance is the per-star PM covariance plus
    (sigma_pm² + pm_sys_floor²) on the diagonal.
    Geometric mean sigma = det(C_total)^(1/4).
    """
    extra   = sigma_pm ** 2 + pm_sys_floor ** 2
    var_ra  = np.where(np.isfinite(sigma_pmra),  sigma_pmra  ** 2, 1.0) + extra
    var_dec = np.where(np.isfinite(sigma_pmdec), sigma_pmdec ** 2, 1.0) + extra
    cov_off = np.where(np.isfinite(sigma_pmra) & np.isfinite(sigma_pmdec),
                       np.where(np.isfinite(corr_pm), corr_pm, 0.0)
                       * sigma_pmra * sigma_pmdec, 0.0)

    det_C      = np.maximum(var_ra * var_dec - cov_off ** 2, 1e-30)
    geom_sigma = det_C ** 0.25   # sqrt(sigma_1 * sigma_2)

    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    dist   = np.where(finite, np.hypot(pmra - mu_pop[0], pmdec - mu_pop[1]), np.inf)
    return np.where(dist < member_sigma_clip * geom_sigma)[0]


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
    z_weights: dict | None = None,
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
        if z_weights is not None:
            _z = z_weights.get(img)
            if _z is not None:
                Cs_inv = Cs_inv * _z[:, None, None]

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
    # Also compute C_vT for inactive member stars: Gaia-2p members have Gaia
    # position + population prior in H_vv, making it non-singular even with
    # no active HST detections.  This lets _compute_free_stellar_posterior
    # correctly strip the population prior and apply the diffuse prior.
    _need_cvT_mask = active_glob.copy()
    _need_cvT_mask[member_sidx] = True
    _need_cvT_sidx = np.where(_need_cvT_mask)[0]
    if len(_need_cvT_sidx) > 0:
        # Guard against truly data-less stars (zero H_vv diagonal)
        _hdiag = np.diagonal(H_vv[_need_cvT_sidx], axis1=1, axis2=2)
        _invertible = _hdiag.all(axis=1)
        _safe_sidx = _need_cvT_sidx[_invertible]
        if len(_safe_sidx) > 0:
            C_vT[_safe_sidx] = np.linalg.inv(H_vv[_safe_sidx])
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


# ── Per-image alpha update ────────────────────────────────────────────────────

def _compute_alpha_updates(
    solver,
    image_names: list[str],
    r_current: np.ndarray,
    a_arr: np.ndarray,
) -> list:
    """
    Compute and apply per-image alpha inflation from HST-only residual chi2.

    Mirrors the v1 bp3m logic:
        alpha_raw  = sqrt( median(sigma_resid²) / (2 ln 2) )
        alpha_new  = max(1.0, alpha_prev × alpha_raw)

    where sigma_resid is the 2-D Mahalanobis distance using the currently
    inflated C_hst (no stellar-astrometry or alignment-parameter covariance).

    Returns a list of (img, n_use, n_tot, alpha_prev, alpha_raw, alpha_new).
    Updates solver._img_data[img]['alpha_applied'] and ['C_hst'] in place.
    """
    _MEDIAN_CHI2_2 = 2.0 * np.log(2.0)   # median of chi2(2)
    resid_hst = solver.compute_residuals(r_current, a_arr)   # HST-only, no C_r/C_vT

    info = []
    for img in image_names:
        rd = resid_hst.get(img)
        d  = solver._img_data.get(img)
        if rd is None or d is None:
            continue
        use_fit    = np.asarray(d['use_for_fit'], dtype=bool)
        n_use      = int(use_fit.sum())
        n_tot      = len(use_fit)
        alpha_prev = float(d.get('alpha_applied', 1.0))

        if n_use >= 4:
            chi2_use  = rd['sigma_resid'][use_fit] ** 2
            alpha_raw = float(np.sqrt(np.median(chi2_use) / _MEDIAN_CHI2_2))
        else:
            alpha_raw = 1.0

        alpha_new              = float(max(1.0, alpha_prev * alpha_raw))
        d['alpha_applied']     = alpha_new
        d['C_hst']             = alpha_new ** 2 * d['C_hst_orig']
        info.append((img, n_use, n_tot, alpha_prev, alpha_raw, alpha_new))

    return info


# ── Re-open detections based on Phase 3 residual thresholds ──────────────────

def _reopen_detections(
    solver,
    image_names: list[str],
    r_current: np.ndarray,
    a_arr: np.ndarray,
    adaptive_k: float = 3.0,
) -> list:
    """
    Use the Phase 3 solution to re-evaluate ALL detections against a per-image
    adaptive chi2 threshold derived from the currently-used (use_for_fit=True)
    set.  Any detection—including those excluded by v1 quality filters—is
    admitted to Phase 4 if its sigma_resid² falls below the threshold.

    The threshold mirrors v1's _adapt_thresh formula:
        thresh = max(p50 + k*(p50 - p16),  chi2_floor)
    where the percentiles are computed on sigma_resid² for the currently-used
    detections only, and chi2_floor = chi2.ppf(0.90, df=2) ≈ 4.61.

    Updates d['use_for_fit'] and d['use_for_astrom'] in place.

    Returns list of (img, n_before, n_after, n_added, n_removed, threshold).
    """
    from scipy.stats import chi2 as _chi2

    _FLOOR = float(_chi2.ppf(0.99, df=2))   # ≈ 9.21

    resid_all = solver.compute_residuals(r_current, a_arr)   # ALL detections

    info = []
    for img in image_names:
        rd = resid_all.get(img)
        d  = solver._img_data.get(img)
        if rd is None or d is None:
            continue

        sig_sq   = rd['sigma_resid'] ** 2           # (n,) all detections
        use_fit  = np.asarray(d['use_for_fit'], dtype=bool)
        n_before = int(use_fit.sum())

        # Threshold from currently-used good detections
        good = sig_sq[use_fit]
        if len(good) >= 10:
            p16   = float(np.percentile(good, 16))
            p50   = float(np.median(good))
            thresh = float(max(p50 + adaptive_k * max(p50 - p16, 1e-6), _FLOOR))
        else:
            thresh = _FLOOR

        new_use  = sig_sq < thresh
        n_after  = int(new_use.sum())
        n_added  = int(np.sum(new_use & ~use_fit))
        n_removed = int(np.sum(~new_use & use_fit))

        d['use_for_fit']    = new_use
        d['use_for_astrom'] = new_use.copy()

        n_total = int(d['n'])
        info.append((img, n_total, n_before, n_after, n_added, n_removed, thresh))

    return info


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
        # C_vT intentionally omitted: C_gdc_total gives measurement-only (HST+alpha)
        # uncertainty in GDC pixel units, which is the correct sigma for normalisation.
        gdc_after = solver.compute_gdc_residuals(r_after, v_after)
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
        total_n_possible = 0

        for img in imgs:
            d = solver._img_data.get(img)
            if d is None:
                continue
            use_any = d['use_for_fit'] | d.get('use_for_astrom', d['use_for_fit'])
            total_n_possible += int(d['n'])
            if not use_any.any():
                continue

            n_det = len(d['sidx'])
            for si, gdc in enumerate([gdc_before, gdc_after]):
                rd     = gdc.get(img, {})
                xc_all = rd.get('X_c',    np.zeros(n_det))
                yc_all = rd.get('Y_c',    np.zeros(n_det))
                dx_all = rd.get('dx_gdc', np.zeros(n_det))
                dy_all = rd.get('dy_gdc', np.zeros(n_det))
                rows_x [si].append(xc_all[use_any])
                rows_y [si].append(yc_all[use_any])
                rows_dx[si].append(dx_all[use_any])
                rows_dy[si].append(dy_all[use_any])

            rd_after = gdc_after.get(img, {})
            c_gdc_tot = rd_after.get('C_gdc_total')
            if c_gdc_tot is not None:
                sigma_dx_all.append(np.sqrt(np.maximum(c_gdc_tot[use_any, 0, 0], 0.0)))
                sigma_dy_all.append(np.sqrt(np.maximum(c_gdc_tot[use_any, 1, 1], 0.0)))

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

        # Colorbar limits: use 68% spread (p84 - p16) / 2 of the after residuals,
        # floored at 0.01 px to avoid degenerate colorbars.
        _after_dx = rows_dx[1][np.isfinite(rows_dx[1])]
        _after_dy = rows_dy[1][np.isfinite(rows_dy[1])]
        def _half68(v):
            if len(v) < 4:
                return 0.05
            return max(float((np.percentile(v, 84) - np.percentile(v, 16)) / 2), 0.01)
        _vc = max(_half68(_after_dx), _half68(_after_dy))
        _vc_sig = 2.0

        n_cols = 4 if has_sigma else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 5, 7),
                                 sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0.08, 'wspace': 0.06})
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f'{root}  n={total_n}/{total_n_possible} (used/total Gaia-matched)',
                     fontsize=10, y=0.99)

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
                ax.set_facecolor('#D8D8D8')
                ax.grid(True, which='major', color='white', linewidth=0.7, zorder=0)
                ax.grid(True, which='minor', color='white', linewidth=0.3, zorder=0)
                ax.minorticks_on()
                sc = ax.scatter(rows_x[row_i], rows_y[row_i], c=vals,
                                cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                                s=10, alpha=0.8, linewidths=0,
                                rasterized=True, zorder=2)
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


# ── Soft-weight diagnostic plot ───────────────────────────────────────────────

def _plot_soft_weights_pop(
    z_weights_final: dict,
    p3_active: dict,
    solver,
    image_names: list[str],
    plot_dir: Path,
    student_t_nu: float,
    z_threshold: float = 0.8,
) -> None:
    """
    Two-panel diagnostic for Phase 4 soft-weight IRLS results.

    Left:  histogram of z values split by Phase-3-active vs re-admitted.
    Right: per-image bar chart of N_possible / N_threshold / N_eff,
           sorted by N_eff descending.

    Saved to plot_dir / 'soft_weights_diagnostic.png'.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_z        = []   # z values for threshold-passing detections
    all_p3       = []   # True if the detection was already active in Phase 3
    neff_per     = {}   # {img: (n_possible, n_thresh, n_eff)}

    for img in image_names:
        z = z_weights_final.get(img) if z_weights_final else None
        d = solver._img_data.get(img)
        if z is None or d is None:
            continue

        # Threshold-passing mask: detections that participate (z > 0 or use_for_fit)
        survivors = np.asarray(d['use_for_fit'], dtype=bool) | np.asarray(
            d.get('use_for_astrom', d['use_for_fit']), dtype=bool)
        p3_was_active = np.asarray(
            p3_active.get(img, np.zeros(len(z), dtype=bool)), dtype=bool)

        z_surv  = z[survivors]
        p3_surv = p3_was_active[survivors]
        all_z.extend(z_surv.tolist())
        all_p3.extend(p3_surv.tolist())

        n_possible = int(d['n'])
        n_thresh   = int(survivors.sum())
        n_eff      = float(z_surv.sum())
        neff_per[img] = (n_possible, n_thresh, n_eff)

    all_z  = np.array(all_z)
    all_p3 = np.array(all_p3, dtype=bool)

    n_thresh_total   = len(all_z)
    n_possible_total = sum(v[0] for v in neff_per.values())
    n_eff_total      = float(all_z.sum())
    pct = 100.0 * n_eff_total / max(n_thresh_total, 1)

    # Merge _hi/_lo chip pairs into per-FLC totals for the bar chart
    flc_per = {}   # {root: (n_possible, n_thresh, n_eff)}
    for img, (np_, nt, ne) in neff_per.items():
        root = img[:-3] if img.endswith(('_hi', '_lo')) else img
        if root in flc_per:
            flc_per[root] = (flc_per[root][0] + np_,
                             flc_per[root][1] + nt,
                             flc_per[root][2] + ne)
        else:
            flc_per[root] = (np_, nt, ne)

    flcs_sorted = sorted(flc_per, key=lambda i: flc_per[i][2], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: z histogram ────────────────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    if all_p3.any():
        ax.hist(all_z[all_p3],  bins=bins, alpha=0.6, label='Phase-3 active',
                color='steelblue')
    if (~all_p3).any():
        ax.hist(all_z[~all_p3], bins=bins, alpha=0.6, label='Re-admitted (Phase 4)',
                color='darkorange')
    ax.axvline(z_threshold, color='red', lw=1, ls='--', label=f'z={z_threshold:.2f} threshold')
    ax.set_xlabel('z  (Student-t weight)')
    ax.set_ylabel('N detections')
    ax.set_title('Detection weight distribution')
    ax.legend(fontsize=9)

    # ── Right: N_possible / N_threshold / N_eff per FLC ──────────────────────
    ax2 = axes[1]
    x   = np.arange(len(flcs_sorted))
    nposs_vals   = [flc_per[r][0] for r in flcs_sorted]
    nthresh_vals = [flc_per[r][1] for r in flcs_sorted]
    neff_vals    = [flc_per[r][2] for r in flcs_sorted]
    ax2.bar(x, nposs_vals,   color='lightgrey',      label='N_possible (in solver)')
    ax2.bar(x, nthresh_vals, color='cornflowerblue', alpha=0.8, label='N_threshold')
    ax2.bar(x, neff_vals,    color='steelblue',      alpha=0.9, label='N_eff (Σz)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(flcs_sorted, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Detections (both chips combined)')
    ax2.set_title('N_possible / N_threshold / N_eff per FLC (sorted by N_eff)')
    ax2.legend(fontsize=9)

    fig.suptitle(
        f'Soft-weight IRLS: detection weights (ν={student_t_nu:.0f})\n'
        f'N_possible={n_possible_total}  N_threshold={n_thresh_total}  '
        f'N_eff={n_eff_total:.0f} ({pct:.1f}%)',
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(plot_dir) / 'soft_weights_diagnostic.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")


# ── Main function ─────────────────────────────────────────────────────────────

def run_pop_fit(
    output_dir: Path,
    field_name: str,
    sigma_pm: float = 0.0075,
    plx_pop: float = 0.003873,
    sigma_plx_tot: float = 0.0001425,
    mu_pop_prior_sigma: float = 0.5,
    n_iter_mu: int = 20,
    n_iter_joint: int = 20,
    n_iter_alpha: int = 20,
    n_iter_soft: int = 20,
    student_t_nu: float = 50.0,
    z_threshold: float = 0.8,
    member_sigma_clip: float = 3.0,
    pm_sys_floor: float = 0.2,
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
          f"member_sigma_clip={member_sigma_clip}  pm_sys_floor={pm_sys_floor} mas/yr")
    print(f"  n_iter: μ={n_iter_mu}  joint={n_iter_joint}  alpha={n_iter_alpha}  "
          f"soft={n_iter_soft} (ν={student_t_nu:.1f})")
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

    # ── Load v1 bp3m posteriors (for initial membership + before/after plots) ──
    v1_astrom_path = bp3m_dir / 'stellar_astrometry.csv'
    v_bp3m = solver.v_survey.copy()   # fallback: Gaia-only

    # Initial membership PM arrays — start from Gaia, override with v1 bp3m
    _pmra_init       = gaia_catalog['pmra'].to_numpy(float).copy()
    _pmdec_init      = gaia_catalog['pmdec'].to_numpy(float).copy()
    _sig_pmra_init   = gaia_catalog['pmra_error'].to_numpy(float).copy()
    _sig_pmdec_init  = gaia_catalog['pmdec_error'].to_numpy(float).copy()
    _corr_pm_init    = (gaia_catalog['pmra_pmdec_corr'].to_numpy(float).copy()
                        if 'pmra_pmdec_corr' in gaia_catalog.columns
                        else np.zeros(solver.n_stars))

    _v1_pm_loaded = False
    if v1_astrom_path.exists():
        try:
            _v1 = pd.read_csv(v1_astrom_path)
            _v1['Gaia_id'] = _v1['Gaia_id'].astype(np.int64)
            _v1_idx = {int(g): i for i, g in enumerate(_v1['Gaia_id'])}

            _v_cols = ['delta_racosdec_bp3m', 'delta_dec_bp3m',
                       'pmra_bp3m', 'pmdec_bp3m', 'parallax_bp3m']
            _pm_sig_cols = ['sigma_pmra_bp3m', 'sigma_pmdec_bp3m', 'corr_pmra_pmdec']

            if all(c in _v1.columns for c in _v_cols):
                _v1_arr = _v1[_v_cols].to_numpy(float)
                for i, gid in enumerate(gaia_catalog['Gaia_id']):
                    j = _v1_idx.get(int(gid))
                    if j is not None:
                        v_bp3m[i] = _v1_arr[j]

            if all(c in _v1.columns for c in _pm_sig_cols):
                _v1_pm  = _v1[['pmra_bp3m', 'pmdec_bp3m']].to_numpy(float)
                _v1_sig = _v1[_pm_sig_cols].to_numpy(float)
                for i, gid in enumerate(gaia_catalog['Gaia_id']):
                    j = _v1_idx.get(int(gid))
                    if j is not None:
                        _pmra_init[i]      = _v1_pm[j, 0]
                        _pmdec_init[i]     = _v1_pm[j, 1]
                        _sig_pmra_init[i]  = _v1_sig[j, 0]
                        _sig_pmdec_init[i] = _v1_sig[j, 1]
                        _corr_pm_init[i]   = _v1_sig[j, 2]
                _v1_pm_loaded = True

        except Exception as _exc:
            print(f"  WARNING: could not load v1 posteriors — {_exc}")

    if not _v1_pm_loaded:
        print("  WARNING: v1 stellar_astrometry.csv missing PM sigma columns; "
              "falling back to Gaia PMs for initial membership")

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

    # ── Initial member selection from v1 bp3m posteriors ─────────────────────
    _src = "v1 bp3m" if _v1_pm_loaded else "Gaia"
    print(f"\n  Selecting initial members from {_src} PMs...")
    member_sidx = _select_initial_members(
        _pmra_init, _pmdec_init,
        _sig_pmra_init, _sig_pmdec_init, _corr_pm_init,
        mu_pop_current, member_sigma_clip, sigma_pm, pm_sys_floor)
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
        _a_free, _C_free = _compute_free_stellar_posterior(
            a_arr, C_vT, member_sidx, sigma_pm, sigma_plx_tot, mu_pop_current, plx_pop,
            solver._C_VG_inv_per_star)
        member_sidx = _select_members_from_a(
            _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
            sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor)
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
        _a_free, _C_free = _compute_free_stellar_posterior(
            a_arr, C_vT, member_sidx, sigma_pm, sigma_plx_tot, mu_pop_current, plx_pop,
            solver._C_VG_inv_per_star)
        member_sidx = _select_members_from_a(
            _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
            sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor)
        print(f"    iter {jt_iter + 1}/{n_iter_joint}: "
              f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f})  "
              f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  members={len(member_sidx)}")
        solver._update_R(r_current)
        solver._update_geometry(r_current, a_arr)
        if delta_r < 1e-6 and delta_mu < 1e-6:
            print(f"    Converged.")
            break

    # ── Phase 3: joint solve + per-image alpha update ────────────────────────
    if n_iter_alpha > 0:
        print(f"\n  Phase 3: joint solve + alpha update ({n_iter_alpha} iterations)...")
        C_shared_joint_p3 = C_shared_joint   # may be updated in loop
        for al_iter in range(n_iter_alpha):
            r_new, mu_pop_new, C_shared_joint_p3, C_vT, a_arr, _, _ = _joint_solve_pop(
                solver, image_names,
                member_sidx, mu_pop_current,
                sigma_pm, plx_pop, sigma_plx_tot,
                C_pop_prior_inv, mu_pop_prior,
                r_current, fix_r=False,
            )
            solver._update_R(r_new)
            solver._update_geometry(r_new, a_arr)

            alpha_info = _compute_alpha_updates(solver, image_names, r_new, a_arr)

            delta_r     = float(np.max(np.abs(r_new - r_current)))
            delta_mu    = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
            r_current      = r_new
            mu_pop_current = mu_pop_new
            _a_free, _C_free = _compute_free_stellar_posterior(
                a_arr, C_vT, member_sidx, sigma_pm, sigma_plx_tot, mu_pop_current, plx_pop)
            member_sidx = _select_members_from_a(
                _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
                sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor)

            for img, n_use, n_tot, alpha_prev, alpha_raw, alpha_new in alpha_info:
                tag = '  ← raised' if alpha_new > alpha_prev + 1e-4 else (
                      '  ← lowered' if alpha_new < alpha_prev - 1e-4 else '')
                print(f"    {img}: {n_use:4d}/{n_tot:4d} align  "
                      f"α_prev={alpha_prev:.3f}  α_raw={alpha_raw:.3f}  "
                      f"α_new={alpha_new:.3f}{tag}")

            delta_alpha_max = (max(abs(ai[5] - ai[3]) for ai in alpha_info)
                               if alpha_info else 0.0)
            print(f"    iter {al_iter + 1}/{n_iter_alpha}: "
                  f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f})  "
                  f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  "
                  f"Δα_max={delta_alpha_max:.3e}  members={len(member_sidx)}")
            if delta_r < 1e-6 and delta_mu < 1e-6 and delta_alpha_max < 1e-4:
                print(f"    Converged.")
                break

        C_shared_joint = C_shared_joint_p3

    # ── Re-open detections using Phase 3 residual thresholds ─────────────────
    _p3_active = None
    if n_iter_soft > 0:
        # Save Phase 3 active set for soft-weights plot later
        _p3_active = {
            img: solver._img_data[img]['use_for_fit'].copy()
            for img in image_names
            if solver._img_data.get(img) is not None
        }

        # Re-evaluate ALL detections (including v1-excluded ones) against
        # per-image adaptive thresholds derived from the Phase 3 trusted set.
        # Phase 4 soft weights then handle remaining outliers continuously.
        print(f"\n  Re-evaluating all detections using Phase 3 thresholds...")
        reopen_info = _reopen_detections(
            solver, image_names, r_current, a_arr)
        n_total_added   = sum(r[4] for r in reopen_info)
        n_total_removed = sum(r[5] for r in reopen_info)
        n_grand_total   = sum(r[1] for r in reopen_info)
        n_grand_after   = sum(r[3] for r in reopen_info)
        for img, n_tot, nb, na, nadd, nrem, thr in reopen_info:
            parts = []
            if nadd > 0:
                parts.append(f"+{nadd}")
            if nrem > 0:
                parts.append(f"-{nrem}")
            change_str = f" ({', '.join(parts)})" if parts else ""
            print(f"    {img}: {na}/{n_tot} pass threshold"
                  f"  (p3_used={nb}){change_str}  thresh={thr:.2f}")
        print(f"    Total: {n_grand_after}/{n_grand_total} detections pass threshold  "
              f"(+{n_total_added} re-admitted  -{n_total_removed} removed  "
              f"vs Phase 3 active set)")

        # Recompute _n_hst_det to include newly re-admitted detections so that
        # stars whose all-detections were v1-excluded can now qualify as members.
        _n_hst_det = np.zeros(solver.n_stars, dtype=int)
        for _img in image_names:
            _d = solver._img_data.get(_img)
            if _d is None:
                continue
            _use = _d['use_for_fit'] | _d.get('use_for_astrom', _d['use_for_fit'])
            np.add.at(_n_hst_det, _d['sidx'][_use], 1)

    # ── Phase 4: soft-weight IRLS (alpha frozen from Phase 3) ────────────────
    def _apply_z_threshold(z_dict: dict, thresh: float) -> dict:
        """Zero out z values below thresh so they contribute nothing to the solve."""
        return {img: (z * (z >= thresh) if z is not None else None)
                for img, z in z_dict.items()}

    z_weights_final = None
    if n_iter_soft > 0:
        print(f"\n  Phase 4: soft-weight IRLS  ν={student_t_nu:.1f}  z_threshold={z_threshold:.2f}  "
              f"({n_iter_soft} iterations, α frozen)...")

        # Evaluate z on the Phase 3 active set (before re-opening) to show the
        # warm-start distribution.  Save the re-opened flags, swap in Phase 3
        # flags temporarily, compute z, then restore.
        _reopened_flags = {
            img: solver._img_data[img]['use_for_fit'].copy()
            for img in image_names
            if solver._img_data.get(img) is not None
        }
        for _img in image_names:
            _d = solver._img_data.get(_img)
            if _d is not None and _img in _p3_active:
                _d['use_for_fit']    = _p3_active[_img].copy()
                _d['use_for_astrom'] = _p3_active[_img].copy()
        _z_p3, _, _ = solver._update_soft_weights(r_current, a_arr, student_t_nu)
        for _img in image_names:
            _d = solver._img_data.get(_img)
            if _d is not None and _img in _reopened_flags:
                _d['use_for_fit']    = _reopened_flags[_img]
                _d['use_for_astrom'] = _reopened_flags[_img].copy()
        _p3_z_vals = np.concatenate([
            _z_p3[img][_p3_active[img]]
            for img in image_names
            if _z_p3.get(img) is not None and img in _p3_active
        ])
        _pcts = np.percentile(_p3_z_vals, [0, 16, 50, 84, 100]) if len(_p3_z_vals) else [np.nan]*5
        print(f"  Phase 3 active set z-values (warm start):  "
              f"min={_pcts[0]:.3f}  p16={_pcts[1]:.3f}  p50={_pcts[2]:.3f}  "
              f"p84={_pcts[3]:.3f}  max={_pcts[4]:.3f}  "
              f"(n={len(_p3_z_vals)}, n_below_thresh={int((_p3_z_vals < z_threshold).sum())})")

        # Now compute z on the re-opened set for Phase 4
        _z_raw, n_det_total, n_eff = solver._update_soft_weights(
            r_current, a_arr, student_t_nu)
        z_weights_final = _apply_z_threshold(_z_raw, z_threshold)
        C_shared_joint_sw = C_shared_joint   # fallback if loop never runs

        for sw_iter in range(n_iter_soft):
            r_new, mu_pop_new, C_shared_joint_sw, C_vT_sw, a_arr_sw, _, _ = _joint_solve_pop(
                solver, image_names,
                member_sidx, mu_pop_current,
                sigma_pm, plx_pop, sigma_plx_tot,
                C_pop_prior_inv, mu_pop_prior,
                r_current, fix_r=False,
                z_weights=z_weights_final,
            )
            solver._update_R(r_new)
            solver._update_geometry(r_new, a_arr_sw)

            _z_raw_new, n_det_total, n_eff_new = solver._update_soft_weights(
                r_new, a_arr_sw, student_t_nu)
            z_new = _apply_z_threshold(_z_raw_new, z_threshold)

            delta_r   = float(np.max(np.abs(r_new - r_current)))
            delta_mu  = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
            delta_z   = float(sum(
                np.sum(np.abs(z_new[img] - z_weights_final[img]))
                for img in image_names
                if z_new.get(img) is not None and z_weights_final.get(img) is not None))

            r_current      = r_new
            mu_pop_current = mu_pop_new
            z_weights_final = z_new
            a_arr           = a_arr_sw

            _a_free, _C_free = _compute_free_stellar_posterior(
                a_arr, C_vT_sw, member_sidx, sigma_pm, sigma_plx_tot, mu_pop_current, plx_pop)
            member_sidx = _select_members_from_a(
                _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
                sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor)

            print(f"    iter {sw_iter + 1}/{n_iter_soft}: "
                  f"n_eff={n_eff_new:.0f}/{n_det_total}  "
                  f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f})  "
                  f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  Δz={delta_z:.3e}  "
                  f"members={len(member_sidx)}")
            if delta_r < 1e-6 and delta_mu < 1e-6 and delta_z < 1e-2:
                print(f"    Converged.")
                break

        C_shared_joint = C_shared_joint_sw

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
        z_weights=z_weights_final,
    )

    # ── Analytic marginalised posteriors (joint r + μ_pop propagation) ──────────
    print("\n  Computing analytic marginalised posteriors...")
    C_r     = C_shared_final[:n_r, :n_r]
    C_mu    = C_shared_final[n_r:, n_r:]    # (2, 2) μ_pop posterior covariance
    C_r_mu  = C_shared_final[:n_r, n_r:]    # (n_r, 2) r–μ_pop cross-covariance

    # Step 1: r-only extra covariance (existing analytic formula)
    v_mean_marg, v_cov_r = solver.compute_analytic_posteriors(
        r_current, C_r, v_mean, K_img_final, C_vT_final)

    # Step 2: build K_all (n_stars × 5 × n_r_tot) — same loop as compute_analytic_posteriors
    nr      = solver.N_R
    n_r_tot = nr * solver.n_images
    K_all   = np.zeros((solver.n_stars, 5, n_r_tot))
    for _j, _img in enumerate(solver.image_names):
        if K_img_final.get(_img) is None:
            continue
        _d = solver._img_data[_img]
        _use = _d['use_for_fit'] | _d.get('use_for_astrom', _d['use_for_fit'])
        if not _use.any():
            continue
        _sidx = _d['sidx'][_use]
        np.add.at(K_all[:, :, _j * nr:_j * nr + nr], _sidx, K_img_final[_img][_use])

    # A_i = C_vT_i @ K_all_i  (sensitivity of v_i to r, n_stars × 5 × n_r_tot)
    CvT_K = np.einsum('nij,njk->nik', C_vT_final, K_all)

    # Step 3: μ_pop sensitivity for member stars
    #   ∂v_i/∂μ_pop = σ_pm^{-2} C_vT_i[:, 2:4]  (only for members, 0 for non-members)
    B_all = np.zeros((solver.n_stars, 5, 2))
    B_all[member_sidx] = (sigma_pm ** -2) * C_vT_final[member_sidx, :, 2:4]

    # Step 4: full extra covariance = A C_r A^T + A C_{r,μ} B^T + B C_{μ,r} A^T + B C_μ B^T
    C_extra_mu    = np.einsum('nik,kl,njl->nij', B_all, C_mu, B_all)
    C_extra_cross = (np.einsum('nik,kl,njl->nij', CvT_K, C_r_mu,   B_all) +
                     np.einsum('nik,kl,njl->nij', B_all,  C_r_mu.T, CvT_K))
    v_cov      = v_cov_r + C_extra_cross + C_extra_mu
    v_cov_full = v_cov + C_vT_final   # full marginal covariance per star

    # ── Diffuse-prior stellar posteriors (no population prior for any star) ──────
    print("\n  Computing diffuse-prior (free) stellar posteriors...")
    _, _, _, C_vT_free_sol, v_mean_free_cond, _, K_img_free = _joint_solve_pop(
        solver, image_names,
        np.array([], dtype=int),   # no members → no population prior
        mu_pop_current,
        sigma_pm, plx_pop, sigma_plx_tot,
        C_pop_prior_inv, mu_pop_prior,
        r_current, fix_r=True,
    )
    v_mean_free_marg, v_cov_free_sol = solver.compute_analytic_posteriors(
        r_current, C_r, v_mean_free_cond, K_img_free, C_vT_free_sol)

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

    g['pmra_bp3m_free_cond']     = v_mean_free_cond[:, 2]
    g['pmdec_bp3m_free_cond']    = v_mean_free_cond[:, 3]
    g['parallax_bp3m_free_cond'] = v_mean_free_cond[:, 4]
    g['pmra_bp3m_free']          = v_mean_free_marg[:, 2]
    g['pmdec_bp3m_free']         = v_mean_free_marg[:, 3]
    g['parallax_bp3m_free']      = v_mean_free_marg[:, 4]
    _is_member_arr               = np.zeros(solver.n_stars, dtype=bool)
    if member_sidx is not None and len(member_sidx) > 0:
        _is_member_arr[member_sidx] = True
    g['is_member']               = _is_member_arr

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
    _C_mu = C_shared_final[n_r:, n_r:]   # (2, 2) μ_pop posterior covariance
    _corr_mu = (float(_C_mu[0, 1] / (sigma_mu_joint[0] * sigma_mu_joint[1]))
                if (sigma_mu_joint[0] > 0 and sigma_mu_joint[1] > 0) else 0.0)

    # Gaia-alone pop PM posterior for the final member stars.
    # Uses solver.C_survey (already inflated and floored identically to bp3m fitting)
    # plus sigma_pm^2 on diagonal to marginalise over individual star true PMs.
    _g_pmra  = solver.v_survey[member_sidx, 2]   # pmra in solver frame
    _g_pmdec = solver.v_survey[member_sidx, 3]   # pmdec in solver frame
    # PM 2x2 block of solver.C_survey (indices 2,3 = pmra, pmdec)
    _g_C_pm  = solver.C_survey[member_sidx][:, 2:4, 2:4]   # (n, 2, 2)
    _g_ok    = np.isfinite(_g_pmra) & np.isfinite(_g_pmdec) & (
        np.isfinite(_g_C_pm[:, 0, 0]) & (_g_C_pm[:, 0, 0] > 0) &
        np.isfinite(_g_C_pm[:, 1, 1]) & (_g_C_pm[:, 1, 1] > 0))
    if _g_ok.sum() >= 2:
        _Lambda_g = np.zeros((2, 2))
        _h_g      = np.zeros(2)
        for _k in np.where(_g_ok)[0]:
            # Total per-star covariance = bp3m Gaia cov + intrinsic dispersion
            # (marginalises over individual star true PMs ~ N(mu_pop, sigma_pm^2 I))
            _C_k     = _g_C_pm[_k].copy()
            _C_k[0, 0] += sigma_pm ** 2
            _C_k[1, 1] += sigma_pm ** 2
            _Ci  = np.linalg.inv(_C_k)
            _Lambda_g += _Ci
            _h_g      += _Ci @ np.array([_g_pmra[_k], _g_pmdec[_k]])
        _C_mu_g    = np.linalg.inv(_Lambda_g)
        _mu_g      = _C_mu_g @ _h_g
        _sig_g     = np.sqrt(np.diag(_C_mu_g))
        _corr_mu_g = (float(_C_mu_g[0, 1] / (_sig_g[0] * _sig_g[1]))
                      if (_sig_g[0] > 0 and _sig_g[1] > 0) else 0.0)
    else:
        _mu_g = np.array([np.nan, np.nan])
        _sig_g = np.array([np.nan, np.nan])
        _corr_mu_g = np.nan

    mu_result = {
        'mu_pop_ra_masyr':       float(mu_pop_current[0]),
        'mu_pop_dec_masyr':      float(mu_pop_current[1]),
        'sigma_mu_pop_ra':       float(sigma_mu_joint[0]),
        'sigma_mu_pop_dec':      float(sigma_mu_joint[1]),
        'corr_mu_pop_ra_dec':    _corr_mu,
        'mu_gaia_ra_masyr':      float(_mu_g[0]),
        'mu_gaia_dec_masyr':     float(_mu_g[1]),
        'sigma_mu_gaia_ra':      float(_sig_g[0]),
        'sigma_mu_gaia_dec':     float(_sig_g[1]),
        'corr_mu_gaia_ra_dec':   _corr_mu_g,
        'n_members':             int(len(member_sidx)),
        'n_members_gaia_finite': int(_g_ok.sum()),
        'sigma_pm_masyr':        float(sigma_pm),
        'plx_pop_mas':           float(plx_pop),
        'sigma_plx_tot_mas':     float(sigma_plx_tot),
        'mu_pop_prior_ra':       float(mu_pop_prior[0]),
        'mu_pop_prior_dec':      float(mu_pop_prior[1]),
        'mu_pop_prior_sigma':    float(mu_pop_prior_sigma),
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

        if z_weights_final is not None and _p3_active is not None:
            try:
                print("\n  Plotting soft-weight diagnostic...")
                _plot_soft_weights_pop(
                    z_weights_final, _p3_active,
                    solver, image_names,
                    plot_dir=output_pfr / 'plots',
                    student_t_nu=student_t_nu,
                    z_threshold=z_threshold,
                )
            except Exception as _exc:
                print(f"  WARNING: soft_weights_diagnostic plot failed — {_exc}")

        try:
            from bp3m.pipeline.plot_results import make_plots
            print("\n  Generating diagnostic plots...")
            # Temporarily restore Phase 3 use_for_fit flags so that
            # chi2_hst_distributions uses the tightly-curated Phase 3 sample
            # rather than all re-opened detections (which include borderline
            # ones up to the threshold and inflate the chi2 distribution).
            if _p3_active is not None:
                for _img in image_names:
                    _d = solver._img_data.get(_img)
                    if _d is not None and _img in _p3_active:
                        _d['use_for_fit']    = _p3_active[_img].copy()
                        _d['use_for_astrom'] = _p3_active[_img].copy()
            make_plots(solver, imgs, gaia_catalog,
                       r_current, v_mean, v_mean_marg, v_cov, C_vT_final, C_r,
                       output_dir=output_pfr,
                       plot_residuals=False,
                       member_sidx=member_sidx,
                       mu_pop=mu_pop_current,
                       v_mean_free=v_mean_free_marg,
                       v_cov_free=v_cov_free_sol,
                       C_vT_free=C_vT_free_sol)
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
    parser.add_argument('--n_iter_mu', type=int, default=20,
                        help='Phase 1 (μ-only) solve iterations')
    parser.add_argument('--n_iter_joint', type=int, default=20,
                        help='Phase 2 (joint r+μ) solve iterations')
    parser.add_argument('--n_iter_alpha', type=int, default=20,
                        help='Phase 3 (joint r+μ+alpha) solve iterations (0 to skip)')
    parser.add_argument('--n_iter_soft', type=int, default=20,
                        help='Phase 4 (soft-weight IRLS, frozen alpha) iterations (0 to skip)')
    parser.add_argument('--student_t_nu', type=float, default=50.0,
                        help='Student-t degrees of freedom for soft weights '
                             '(larger = harder, 50 ≈ nearly hard exclusion)')
    parser.add_argument('--z_threshold', type=float, default=0.8,
                        help='Minimum z weight for a detection to contribute to the '
                             'Phase 4 solve; detections below this are hard-excluded '
                             '(default 0.8)')
    parser.add_argument('--member_sigma_clip', type=float, default=3.0,
                        help='Sigma threshold for membership selection')
    parser.add_argument('--pm_sys_floor', type=float, default=0.2,
                        help='Systematic PM floor added in quadrature to per-star '
                             'PM uncertainty for membership radius (mas/yr)')
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
        n_iter_alpha=args.n_iter_alpha,
        n_iter_soft=args.n_iter_soft,
        student_t_nu=args.student_t_nu,
        z_threshold=args.z_threshold,
        member_sigma_clip=args.member_sigma_clip,
        pm_sys_floor=args.pm_sys_floor,
        poly_order=args.poly_order,
        no_plots=args.no_plots,
    )
