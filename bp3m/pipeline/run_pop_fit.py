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


def _estimate_mu_pop_v1(
    pmra: np.ndarray,
    pmdec: np.ndarray,
    n_sigma: float = 3.0,
    n_iter: int = 10,
    mu_init: np.ndarray | None = None,
) -> np.ndarray:
    """Sigma-clipped mean of v1 bp3m proper motions (finite values only).

    If mu_init is given it seeds the first clip iteration; the iterative
    sigma-clip then proceeds as normal from that starting point.
    """
    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    pmra_f, pmdec_f = pmra[finite], pmdec[finite]
    if len(pmra_f) < 5:
        print("  WARNING: fewer than 5 stars with finite v1 PMs — using (0, 0)")
        return np.zeros(2)

    if mu_init is not None:
        c_ra, c_dec = float(mu_init[0]), float(mu_init[1])
        dists  = np.hypot(pmra_f - c_ra, pmdec_f - c_dec)
        sigma  = max(float(np.median(dists)) / 0.6745, 0.01)
        keep   = dists < n_sigma * sigma
        if keep.sum() < 5:
            keep = np.ones(len(pmra_f), dtype=bool)
    else:
        keep = np.ones(len(pmra_f), dtype=bool)
    for _ in range(n_iter):
        if keep.sum() < 5:
            break
        med_ra  = float(np.median(pmra_f[keep]))
        med_dec = float(np.median(pmdec_f[keep]))
        dra, ddec = pmra_f[keep] - med_ra, pmdec_f[keep] - med_dec
        sigma   = max(float(np.median(np.hypot(dra, ddec))) / 0.6745, 0.01)
        new_keep = np.hypot(pmra_f - med_ra, pmdec_f - med_dec) < n_sigma * sigma
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    mu = np.array([float(np.mean(pmra_f[keep])), float(np.mean(pmdec_f[keep]))])
    print(f"  Bootstrap μ_pop (v1 σ-clip, n={keep.sum()}/{len(pmra_f)}): "
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
    pm_sys_floor: float = 0.2,   # retained for call-site compatibility, not used
    max_sigma_free_pm: float = 1.0,
) -> np.ndarray:
    """Select members by 2D Mahalanobis distance from mu_pop.

    C_vT is expected to be the diffuse-prior (free) covariance so the
    population prior does not pull stars into apparent membership.

    C_total = C_vT[star, 2:4, 2:4] + sigma_pm² · I   (measurement + intrinsic dispersion)
    chi2    = Δμ^T C_total^{-1} Δμ
    Kept    ← chi2 < sigma_clip²  AND  RMS free PM sigma < max_sigma_free_pm

    The sigma constraint prevents stars dominated by the diffuse prior (2p stars with
    few HST epochs) from trivially passing chi2 — their C_free is so large that any
    PM is consistent with mu_pop, providing no real membership evidence.

    Only stars with ≥1 HST detection are eligible.
    """
    eidx = np.where(n_hst >= 1)[0]
    if len(eidx) < min_members:
        return eidx

    # Exclude stars with degenerate posteriors (C_vT=0, a_arr=0) — these are
    # stars that were excluded from the active solve by fit_members_only and have
    # no meaningful posterior.  Their a_arr=0 can trivially satisfy chi2<threshold
    # if mu_pop is near zero in the solver's internal frame.
    _has_valid_posterior = (C_vT[eidx, 2, 2] > 0) | (C_vT[eidx, 3, 3] > 0)
    eidx = eidx[_has_valid_posterior]
    if len(eidx) < min_members:
        return eidx

    pmra  = a_arr[eidx, 2]
    pmdec = a_arr[eidx, 3]

    C_pm_sub = C_vT[eidx, 2:4, 2:4].copy()                             # (n, 2, 2)
    C_pm_sub[:, 0, 0] += sigma_pm ** 2
    C_pm_sub[:, 1, 1] += sigma_pm ** 2
    delta_pm = np.column_stack([pmra - mu_pop[0], pmdec - mu_pop[1]])  # (n, 2)
    chi2 = np.einsum('ni,nij,nj->n', delta_pm, np.linalg.inv(C_pm_sub), delta_pm)

    # Require PM to be meaningfully constrained; stars where C_free_pm is dominated
    # by the diffuse prior trivially pass chi2 and provide no membership evidence.
    sig_free = np.sqrt(np.maximum(
        (C_vT[eidx, 2, 2] + C_vT[eidx, 3, 3]) / 2, 0))
    well_constrained = sig_free < max_sigma_free_pm

    keep = np.isfinite(chi2) & (chi2 < sigma_clip ** 2) & well_constrained
    if keep.sum() < min_members:
        keep = np.isfinite(chi2) & well_constrained

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
    # NaN sigma propagates to NaN threshold → star not selected (correct: unknown uncertainty)
    var_ra  = sigma_pmra  ** 2 + extra
    var_dec = sigma_pmdec ** 2 + extra
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
        if nr > 4:
            # r_j[4] = (ra0_current - ra0_true)*3.6e6.  At ra0_current=ra0_orig,
            # ra0_true≈ra0_final, so r_j[4] = (ra0_orig - ra0_final)*3.6e6
            # = -delta_ra0_mas.  Negate the stored offset.
            r_hat[cs + 4] = -float(row.get('delta_ra0_mas',  0.0))
        if nr > 5:
            r_hat[cs + 5] = -float(row.get('delta_dec0_mas', 0.0))
        for k in range(6, nr):
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
    qso_sidx: "np.ndarray | None" = None,
    qso_pmra: "np.ndarray | None" = None,
    qso_pmdec: "np.ndarray | None" = None,
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

    # QSO anchor prior: per-source secular aberration PM + zero parallax.
    # sigma_qso_pm = 0.35 µas/yr = 3.5e-4 mas/yr (σ_κ, Klioner 2021)
    # sigma_qso_plx = 1 µas = 1e-3 mas (cosmological source)
    # Prior is independent of mu_pop (NOT coupled to the free μ_pop parameter).
    if qso_sidx is not None and len(qso_sidx) > 0:
        _sigma_qso_pm_inv_sq  = (3.5e-4) ** -2
        _sigma_qso_plx_inv_sq = (1.0e-3) ** -2
        H_vv[qso_sidx, 2, 2] += _sigma_qso_pm_inv_sq
        H_vv[qso_sidx, 3, 3] += _sigma_qso_pm_inv_sq
        H_vv[qso_sidx, 4, 4] += _sigma_qso_plx_inv_sq
        h_align[qso_sidx, 2] += _sigma_qso_pm_inv_sq * qso_pmra
        h_align[qso_sidx, 3] += _sigma_qso_pm_inv_sq * qso_pmdec
        h_all  [qso_sidx, 2] += _sigma_qso_pm_inv_sq * qso_pmra
        h_all  [qso_sidx, 3] += _sigma_qso_pm_inv_sq * qso_pmdec
        # RHS for parallax prior: 0 (plx_pop_qso = 0, so += 0)

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
    # Compute C_vT for every star whose H_vv is non-degenerate.
    # This covers active stars, inactive members, AND non-active non-member 5p
    # stars (which carry a full Gaia prior and should recover Gaia posteriors
    # when n_hst_used=0).  The _invertible guard skips truly data-less stars.
    _hdiag = np.diagonal(H_vv, axis1=1, axis2=2)
    _invertible = _hdiag.all(axis=1)
    _safe_sidx = np.where(_invertible)[0]
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
    alpha_damp: float = 0.5,
) -> list:
    """
    Compute and apply per-image alpha inflation from HST-only residual chi2.

    Mirrors the v1 bp3m logic with under-relaxation to prevent 2-cycle oscillation:
        alpha_raw  = sqrt( median(sigma_resid²) / (2 ln 2) )
        alpha_new  = max(1.0, alpha_prev × alpha_raw^alpha_damp)

    alpha_damp=1.0 is the full unrelaxed step; alpha_damp=0.5 (default) takes a
    geometric-mean step that damps the 2-cycle without slowing convergence near
    the fixed point.

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

        alpha_new              = float(max(1.0, alpha_prev * alpha_raw ** alpha_damp))
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
    detections only, and chi2_floor = chi2.ppf(0.99, df=2) ≈ 9.21.

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

        ok = sig_sq < thresh

        # Preserve the v1 tier structure instead of flattening it.  The old
        # code set use_for_fit = use_for_astrom = ok, which (a) promoted
        # astrometry-only detections — DELVE-only stars, callback-admitted
        # HST-only stars — into the ALIGNMENT tier they were deliberately kept
        # out of, and (b) bypassed the solver's own hard ceilings.  Alignment
        # eligibility here is: the detection passed the loader's initial
        # alignment quality (use_for_align_init) OR v1's converged fit kept it
        # in alignment (use_fit at entry, i.e. the imported v1 tier) — the
        # pop-fit analogue of the solver's can_enter_fit ratchet.  Any detection
        # that passes the residual test may serve ASTROMETRY, which is the
        # re-opening this function exists for.
        align_eligible = (np.asarray(d.get('use_for_align_init', use_fit),
                                     dtype=bool) | use_fit)
        hard_ok = np.asarray(d.get('use_for_fit_max',
                                   np.ones_like(ok, dtype=bool)), dtype=bool)
        infl = d.get('influence_excl')
        if infl is not None:
            hard_ok = hard_ok & ~np.asarray(infl, dtype=bool)

        new_use    = ok & align_eligible & hard_ok
        new_astrom = ok & hard_ok
        n_after  = int(new_use.sum())
        n_added  = int(np.sum(new_use & ~use_fit))
        n_removed = int(np.sum(~new_use & use_fit))

        d['use_for_fit']    = new_use
        d['use_for_astrom'] = new_astrom

        n_total = int(d['n'])
        info.append((img, n_total, n_before, n_after, n_added, n_removed, thresh))

    return info


# ── Hard-weight Phase 4: population-aware Tests 1+2+3 ────────────────────────

def _hard_update_phase4(
    solver,
    image_names: list[str],
    r_current: np.ndarray,
    a_arr: np.ndarray,
    C_vT: np.ndarray,
    member_sidx: np.ndarray,
    mu_pop: np.ndarray,
    sigma_pm: float,
    plx_pop: float,
    sigma_plx_tot: float,
    ok_star_prev=None,
    adaptive_k: float = 5.0,
    adaptive_delta: float = 0.1,
    a_free: np.ndarray | None = None,
    C_free: np.ndarray | None = None,
    mu_pop_solve: np.ndarray | None = None,
) -> tuple:
    """
    Population-aware hard outlier rejection for Phase 4.

    Test 1  Gaia prior chi2 (adaptive, with hysteresis): identical to v1.
    Test 2  Prior chi2 (posterior+prior covariance):
              members     → compare PM+plx to mu_pop_solve (the value used in the
                            solve that produced a_arr, NOT the updated mu_pop)
              non-members → compare all 5 components to diffuse prior
    Test 3  Per-image position residual chi2 (adaptive, with hysteresis): v1.
    Test 4  Free-posterior PM vs mu_pop (updated) for member stars only.

    Updates solver._img_data[img]['use_for_fit'] and ['use_for_astrom'] in-place.
    Returns (ok_star, n_use_changed, info).
    """
    from scipy.stats import chi2 as chi2_dist

    floor_5 = float(chi2_dist.ppf(0.99, df=5))
    floor_2 = float(chi2_dist.ppf(0.99, df=2))

    def _adapt_thresh(values, k, fallback, floor=0.0):
        if len(values) < 10:
            return float(max(fallback, floor)), float('nan'), float('nan'), float('nan')
        p16 = float(np.percentile(values, 16))
        p50 = float(np.median(values))
        p84 = float(np.percentile(values, 84))
        return float(max(p50 + k * max(p50 - p16, 1e-6), floor)), p16, p50, p84

    observed = solver.gaia_n_hst_used > 0
    obs_5p   = observed & ~solver.gaia_2p
    obs_2p   = observed & solver.gaia_2p

    # ── Test 1: Gaia prior chi2 ───────────────────────────────────────────────
    delta_g    = a_arr - solver.v_survey                   # (n, 5)
    C_comb     = C_vT + solver.C_survey                    # (n, 5, 5)
    C_comb_inv = np.linalg.inv(C_comb)
    chi2_gaia  = np.einsum('ni,nij,nj->n', delta_g, C_comb_inv, delta_g)

    thresh_5a, _, _, _ = _adapt_thresh(chi2_gaia[obs_5p], adaptive_k,
                                        chi2_dist.ppf(0.95, df=5), floor_5)
    thresh_2a, _, _, _ = _adapt_thresh(chi2_gaia[obs_2p], adaptive_k,
                                        chi2_dist.ppf(0.95, df=2), floor_2)
    ok_gaia_admit = np.where(solver.gaia_2p, chi2_gaia < thresh_2a, chi2_gaia < thresh_5a)

    if ok_star_prev is not None and adaptive_delta > 0:
        thresh_5e, _, _, _ = _adapt_thresh(chi2_gaia[obs_5p], adaptive_k + adaptive_delta,
                                            chi2_dist.ppf(0.95, df=5), floor_5)
        thresh_2e, _, _, _ = _adapt_thresh(chi2_gaia[obs_2p], adaptive_k + adaptive_delta,
                                            chi2_dist.ppf(0.95, df=2), floor_2)
        ok_gaia_retain = np.where(solver.gaia_2p, chi2_gaia < thresh_2e, chi2_gaia < thresh_5e)
        ok_gaia = np.where(ok_star_prev, ok_gaia_retain, ok_gaia_admit)
    else:
        ok_gaia = ok_gaia_admit

    # ── Test 2: Prior chi2 (posterior + prior covariance) ─────────────────────
    # C_test = C_vT[relevant] + C_prior — mirrors Test 1's C_comb = C_vT + C_survey
    # IMPORTANT: use mu_pop_solve (the value that constrained a_arr) so that the
    # chi2 is not spuriously inflated by a mu_pop update between iterations.
    is_member = np.zeros(solver.n_stars, dtype=bool)
    is_member[member_sidx] = True

    _mu_t2 = mu_pop_solve if mu_pop_solve is not None else mu_pop

    # Members: PM + parallax vs population prior (df=3)
    delta_pop   = np.column_stack([
        a_arr[:, 2] - _mu_t2[0],
        a_arr[:, 3] - _mu_t2[1],
        a_arr[:, 4] - plx_pop,
    ])  # (n, 3)
    C_prior_pop = np.diag([sigma_pm ** 2, sigma_pm ** 2, sigma_plx_tot ** 2])  # (3,3)
    C_test_pop  = C_vT[:, 2:5, 2:5] + C_prior_pop[np.newaxis]                 # (n,3,3)
    chi2_pop    = np.einsum('ni,nij,nj->n',
                            delta_pop, np.linalg.inv(C_test_pop), delta_pop)

    # Non-members: all 5 components vs diffuse prior (df=5)
    # _sigma_diff_per_star is (n, 5); diffuse prior mean is 0
    sigma_diff_sq = solver._sigma_diff_per_star ** 2                           # (n,5)
    C_test_diff   = C_vT.copy()
    for _i in range(5):
        C_test_diff[:, _i, _i] += sigma_diff_sq[:, _i]
    chi2_diff = np.einsum('ni,nij,nj->n',
                          a_arr, np.linalg.inv(C_test_diff), a_arr)

    thresh_pop  = float(chi2_dist.ppf(0.9545, df=3))   # ≈ 7.8
    thresh_diff = float(chi2_dist.ppf(0.9545, df=5))   # ≈ 11.1

    ok_prior = np.where(is_member, chi2_pop < thresh_pop, chi2_diff < thresh_diff)
    ok_star  = ok_gaia & ok_prior

    n_fail_1 = int((~ok_gaia  & observed).sum())
    n_fail_2 = int((~ok_prior & ok_gaia & observed).sum())
    print(f"    [hard] Tests 1+2 (of {int(observed.sum())} observed): "
          f"{n_fail_1} Gaia-incompatible  {n_fail_2} prior-incompatible  "
          f"(thresh_5p={thresh_5a:.1f}  thresh_2p={thresh_2a:.1f}  "
          f"thresh_pop={thresh_pop:.1f}  thresh_diff={thresh_diff:.1f})")

    # ── Test 3: Per-image position residual chi2 ──────────────────────────────
    resid_hst = solver.compute_residuals(r_current, a_arr)

    _FLOOR          = float(chi2_dist.ppf(0.99, df=2))
    _MEDIAN_CHI2_2  = 2.0 * np.log(2.0)

    solver.gaia_n_hst_used[:] = 0
    n_use_changed = 0
    info = []

    for img in image_names:
        rd = resid_hst.get(img)
        d  = solver._img_data.get(img)
        if rd is None or d is None:
            continue

        sig_sq     = rd['sigma_resid'] ** 2
        sidx_img   = rd['sidx']
        prev_use   = np.asarray(d['use_for_fit'],          dtype=bool)
        align_init = np.asarray(d['use_for_align_init'],   dtype=bool)
        ok_here    = ok_star[sidx_img]

        ok_thresh_ref = ok_here & align_init
        if ok_thresh_ref.sum() < 10:
            ok_thresh_ref = ok_here

        thresh_a, _, _, _ = _adapt_thresh(sig_sq[ok_thresh_ref], adaptive_k,
                                           chi2_dist.ppf(0.95, df=2), floor=_FLOOR)
        thresh_e, _, _, _ = _adapt_thresh(sig_sq[ok_thresh_ref], adaptive_k + adaptive_delta,
                                           chi2_dist.ppf(0.95, df=2), floor=_FLOOR)

        ok_resid = np.where(prev_use, sig_sq < thresh_e, sig_sq < thresh_a)

        new_use  = ok_resid & ok_here
        new_use  = new_use & np.asarray(d['use_for_fit_max'], dtype=bool)
        infl_excl = d.get('influence_excl')
        if infl_excl is not None:
            new_use = new_use & ~np.asarray(infl_excl, dtype=bool)
        can_enter = align_init | prev_use
        new_use   = new_use & can_enter

        n_use_changed += int(np.sum(prev_use != new_use))

        new_use_astrom = np.asarray(d['use_for_astrom'], dtype=bool).copy()
        new_use_astrom[align_init] = new_use[align_init]
        d['use_for_fit']    = new_use
        d['use_for_astrom'] = new_use_astrom

        use_any = new_use | new_use_astrom
        np.add.at(solver.gaia_n_hst_used, sidx_img[use_any], 1)

        n_u = int(new_use.sum())
        chi2_u = sig_sq[new_use]
        alpha_raw = float(np.sqrt(np.median(chi2_u) / _MEDIAN_CHI2_2)) if n_u >= 4 else 1.0
        info.append((img, n_u, len(new_use), thresh_a, alpha_raw))

    # ── Test 4: Free-posterior PM vs pop mean (diagnostic only) ──────────────
    # Computes who would be demoted by the free-PM membership criterion.
    # Does NOT modify ok_star — excluding stars from use_for_fit here causes
    # the plate solution to wander and the loop to diverge.  Demotion is handled
    # by _select_members_from_a (same free posterior, chi2 < sigma_clip²) which
    # switches those stars to the diffuse prior in the next solve instead of
    # removing them from the fit entirely.
    if a_free is not None and C_free is not None and is_member.any():
        delta_free  = np.column_stack([
            a_free[:, 2] - mu_pop[0],
            a_free[:, 3] - mu_pop[1],
        ])  # (n, 2)
        C_free_pm = C_free[:, 2:4, 2:4].copy()
        C_free_pm[:, 0, 0] += sigma_pm ** 2
        C_free_pm[:, 1, 1] += sigma_pm ** 2
        chi2_free = np.einsum('ni,nij,nj->n', delta_free,
                              np.linalg.inv(C_free_pm), delta_free)
        thresh_free = float(chi2_dist.ppf(0.9545, df=2))   # ≈ 6.18  (2σ in 2D)
        n_demote = int((is_member & (chi2_free >= thresh_free) & observed).sum())
        print(f"    [hard] Test 4 (free PM vs pop, df=2, thresh={thresh_free:.2f}): "
              f"{n_demote} member(s) will be demoted by membership update")

    return ok_star, n_use_changed, info


# ── PM vs properties figure (2 rows × 6 cols) ────────────────────────────────

def _plot_pm_vs_properties(
    output_dir,
    solver,
    image_names: list,
    gaia_catalog,
    a_free: np.ndarray,
    C_vT_free: np.ndarray,
    member_sidx: np.ndarray,
    mu_pop: np.ndarray,
    sigma_pm: float,
    field_name: str,
    qso_sidx: "np.ndarray | None" = None,
) -> None:
    """
    2×6 figure: diffuse-prior posterior PMs for member stars vs
    Gaia G magnitude (cols 0-1), mean HST x_raw (cols 2-3),
    mean HST y_raw (cols 4-5).  Row 0 = raw mas/yr; row 1 = normalised
    by individual posterior uncertainty.  Points coloured by 2D Mahalanobis
    distance from pop PM mean (marginalised PM covariance + sigma_pm^2 * I).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if len(member_sidx) == 0:
        return

    # ── Per-member arrays ─────────────────────────────────────────────────────
    # QSOs with tight priors have artificially small posterior σ — exclude from
    # the normalised row (row 1) so the comparison does not blow up.
    _is_qso_member = np.zeros(len(member_sidx), dtype=bool)
    if qso_sidx is not None and len(qso_sidx) > 0:
        _qso_set = set(qso_sidx.tolist())
        _is_qso_member = np.array([int(s) in _qso_set for s in member_sidx])

    pmra_m  = a_free[member_sidx, 2]
    pmdec_m = a_free[member_sidx, 3]
    sig_ra  = np.sqrt(np.maximum(C_vT_free[member_sidx, 2, 2], 0.0))
    sig_dec = np.sqrt(np.maximum(C_vT_free[member_sidx, 3, 3], 0.0))

    # 2D Mahalanobis distance from pop PM mean:
    #   C_total = C_pm_marg + sigma_pm^2 * I  (measurement + intrinsic dispersion)
    #   sigma_from_pop = sqrt(delta^T C_total^{-1} delta)
    C_pm_marg = C_vT_free[member_sidx, 2:4, 2:4].copy()   # (n, 2, 2)
    C_pm_marg[:, 0, 0] += sigma_pm ** 2
    C_pm_marg[:, 1, 1] += sigma_pm ** 2
    C_total_inv = np.linalg.inv(C_pm_marg)
    delta       = np.column_stack([pmra_m - mu_pop[0], pmdec_m - mu_pop[1]])  # (n, 2)
    chi2_pop    = np.einsum('ni,nij,nj->n', delta, C_total_inv, delta)
    sigma_from_pop = np.sqrt(np.maximum(chi2_pop, 0.0))

    gmag_arr = gaia_catalog['gmag'].to_numpy(float)
    gmag_m   = gmag_arr[member_sidx]

    # Mean HST x_raw / y_raw per member star (X_c + 2048, Y_c + 2048)
    _XO = 2048.0
    xsum  = np.zeros(solver.n_stars)
    ysum  = np.zeros(solver.n_stars)
    ncnt  = np.zeros(solver.n_stars, dtype=int)
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        sidx_img = d['sidx']
        use      = np.asarray(d['use_for_fit'], dtype=bool)
        np.add.at(xsum, sidx_img[use], d['X_c'][use] + _XO)
        np.add.at(ysum, sidx_img[use], d['Y_c'][use] + _XO)
        np.add.at(ncnt, sidx_img[use], 1)
    with np.errstate(invalid='ignore'):
        xraw_all = np.where(ncnt > 0, xsum / ncnt, np.nan)
        yraw_all = np.where(ncnt > 0, ysum / ncnt, np.nan)
    xraw_m = xraw_all[member_sidx]
    yraw_m = yraw_all[member_sidx]

    # ── Colour mapping ────────────────────────────────────────────────────────
    cmap  = cm.plasma_r
    vmin, vmax = 0.0, 3.0
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cols  = cmap(norm(sigma_from_pop))   # (n, 4) RGBA per star

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 6, figsize=(22, 7.5), constrained_layout=True)
    fig.suptitle(
        f'{field_name} — diffuse-prior PMs for member stars  '
        f'(N={len(member_sidx)}  μ_pop=({mu_pop[0]:+.3f},{mu_pop[1]:+.3f}) mas/yr  '
        f'colour = 2D σ from pop)',
        fontsize=11,
    )

    xs_list   = [gmag_m,  gmag_m,  xraw_m,  xraw_m,  yraw_m,  yraw_m]
    xlabs     = ['Gaia G (mag)'] * 2 + ['mean x_raw (px)'] * 2 + ['mean y_raw (px)'] * 2
    pm_vals   = [pmra_m,  pmdec_m, pmra_m,  pmdec_m, pmra_m,  pmdec_m]
    pm_errs   = [sig_ra,  sig_dec, sig_ra,  sig_dec, sig_ra,  sig_dec]
    pop_means = [mu_pop[0], mu_pop[1]] * 3
    ylabs_raw = [r'$\mu_{\alpha*}$ (mas/yr)', r'$\mu_\delta$ (mas/yr)'] * 3
    ylabs_nrm = [r'$(\mu_{\alpha*}-\mu_{\rm pop})/\sigma_{\mu_{\alpha*}}$',
                 r'$(\mu_\delta-\mu_{\rm pop})/\sigma_{\mu_\delta}$'] * 3

    def _draw_colored(ax, x, y, yerr, c_rgba, finite):
        """Draw errorbars in star colour, then scatter points on top."""
        for xi, yi, ei, ci in zip(x[finite], y[finite], yerr[finite], c_rgba[finite]):
            ax.errorbar(xi, yi, yerr=ei, fmt='none',
                        ecolor=ci, elinewidth=0.6, capsize=0, alpha=0.7)
        sc = ax.scatter(x[finite], y[finite], c=c_rgba[finite],
                        s=8, zorder=3, linewidths=0)
        return sc

    sc_ref = None
    for col in range(6):
        x      = xs_list[col]
        pm     = pm_vals[col]
        err    = pm_errs[col]
        mu_ref = pop_means[col]
        finite = np.isfinite(x) & np.isfinite(pm) & np.isfinite(err) & (err > 0)
        _xmin  = np.nanmin(x) if np.any(np.isfinite(x)) else 0.0
        _xmax  = np.nanmax(x) if np.any(np.isfinite(x)) else 1.0
        _xpad  = 0.05 * max(_xmax - _xmin, 1e-6)

        # ── Row 0: raw mas/yr ─────────────────────────────────────────────────
        ax0 = axes[0, col]
        if finite.any():
            sc = _draw_colored(ax0, x, pm, err, cols, finite)
            if sc_ref is None:
                sc_ref = sc
        ax0.axhline(mu_ref, color='firebrick', lw=1.2, ls='-', label=r'$\mu_{\rm pop}$')
        ax0.axhspan(mu_ref - sigma_pm, mu_ref + sigma_pm,
                    alpha=0.15, color='firebrick', label=r'$\pm\sigma_{\rm pm}$')
        # y-limits: median ± 5 * MAD of the plotted PMs
        if finite.any():
            _pm_f = pm[finite]
            _med  = float(np.median(_pm_f))
            _mad  = float(np.median(np.abs(_pm_f - _med)))
            _half = max(5.0 * _mad, sigma_pm * 3)
            ax0.set_ylim(_med - _half, _med + _half)
        ax0.set_xlim(_xmin - _xpad, _xmax + _xpad)
        ax0.set_xlabel(xlabs[col], fontsize=8)
        ax0.set_ylabel(ylabs_raw[col], fontsize=8)
        ax0.tick_params(labelsize=7)
        if col == 0:
            ax0.legend(fontsize=7, loc='upper right')

        # ── Row 1: normalised (QSOs excluded — tight prior makes σ meaningless) ─
        ax1 = axes[1, col]
        norm_pm  = (pm - mu_ref) / err
        norm_err = np.ones(len(pm))
        # Exclude QSO members from normalised plot
        finite_nonqso = finite & ~_is_qso_member
        if finite_nonqso.any():
            _draw_colored(ax1, x, norm_pm, norm_err, cols, finite_nonqso)
        ax1.axhline(0,  color='firebrick', lw=1.2, ls='-')
        ax1.axhspan(-1, 1, alpha=0.15, color='firebrick')
        ax1.axhline(-3, color='grey', lw=0.7, ls='--')
        ax1.axhline( 3, color='grey', lw=0.7, ls='--')
        ax1.set_xlim(_xmin - _xpad, _xmax + _xpad)
        ax1.set_xlabel(xlabs[col], fontsize=8)
        ax1.set_ylabel(ylabs_nrm[col], fontsize=8)
        ax1.tick_params(labelsize=7)
        if col == 0 and _is_qso_member.any():
            ax1.set_title(f'QSOs excluded (N={_is_qso_member.sum()})', fontsize=7)

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01, aspect=30)
    cbar.set_label(r'2D $\sigma$ from $\mu_{\rm pop}$  ($\sqrt{\Delta\mu^T\,C_{\rm tot}^{-1}\,\Delta\mu}$)',
                   fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    out_path = output_dir / 'pm_vs_properties.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── QSO anchor diagnostic plots ──────────────────────────────────────────────

def _plot_qso_diagnostics(
    plot_dir,
    solver,
    gaia_catalog,
    a_free: np.ndarray,
    C_vT_free: np.ndarray,
    member_sidx: np.ndarray,
    qso_sidx: np.ndarray,
    qso_pmra_mas: np.ndarray,
    qso_pmdec_mas: np.ndarray,
    mu_pop: np.ndarray,
    field_name: str,
) -> None:
    """
    Three-panel QSO diagnostic figure:
      Panel 1 — All-star VPD with QSO anchors highlighted in gold
      Panel 2 — QSO-only VPD with secular-aberration prediction arrows
      Panel 3 — CMD (BP−RP vs G) with QSO anchors highlighted

    QSOs should appear near (pmra~0, pmdec~0) in the VPD, not at the cluster
    pop mean.  In the CMD they should occupy the blue, faint region (point-like
    AGN) rather than the cluster RGB/MS.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    pmra_all  = a_free[:, 2]
    pmdec_all = a_free[:, 3]

    is_member = np.zeros(solver.n_stars, dtype=bool)
    is_member[member_sidx] = True

    gmag    = gaia_catalog['gmag'].to_numpy(float)
    bp_rp   = (gaia_catalog.get('bp_rp', gaia_catalog.get('bpmag', None)) if True
               else None)
    if bp_rp is None:
        if 'bpmag' in gaia_catalog.columns and 'rpmag' in gaia_catalog.columns:
            bp_rp = gaia_catalog['bpmag'].to_numpy(float) - gaia_catalog['rpmag'].to_numpy(float)
        elif 'bp_rp' in gaia_catalog.columns:
            bp_rp = gaia_catalog['bp_rp'].to_numpy(float)
        else:
            bp_rp = np.full(solver.n_stars, np.nan)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    fig.suptitle(f'{field_name} — QSO anchor diagnostics  (N_qso={len(qso_sidx)})',
                 fontsize=11)

    # ── Panel 1: all-star VPD ─────────────────────────────────────────────────
    ax = axes[0]
    _ok_all    = np.isfinite(pmra_all) & np.isfinite(pmdec_all)
    _ok_mem    = _ok_all & is_member
    _ok_nomem  = _ok_all & ~is_member
    _ok_qso    = np.zeros(solver.n_stars, dtype=bool)
    _ok_qso[qso_sidx] = True
    _ok_qso   &= _ok_all

    ax.scatter(pmra_all[_ok_nomem],  pmdec_all[_ok_nomem],
               s=4, c='#aaaaaa', alpha=0.4, zorder=1, label='non-members')
    ax.scatter(pmra_all[_ok_mem],    pmdec_all[_ok_mem],
               s=6, c='steelblue', alpha=0.7, zorder=2, label=f'members (N={_ok_mem.sum()})')
    if _ok_qso.any():
        ax.scatter(pmra_all[_ok_qso], pmdec_all[_ok_qso],
                   s=60, c='gold', edgecolors='darkorange', linewidths=0.8,
                   zorder=5, marker='*', label=f'QSO anchors (N={_ok_qso.sum()})')
    ax.axhline(0, c='k', lw=0.5, ls='--', alpha=0.3)
    ax.axvline(0, c='k', lw=0.5, ls='--', alpha=0.3)
    ax.scatter(*mu_pop, s=120, c='firebrick', marker='+', linewidths=2.5,
               zorder=6, label=r'$\mu_{\rm pop}$')
    _pm_range = max(abs(mu_pop[0]), abs(mu_pop[1]), 3.0) * 2.5
    ax.set_xlim(-_pm_range, _pm_range)
    ax.set_ylim(-_pm_range, _pm_range)
    ax.set_xlabel(r'$\mu_{\alpha*}$ [mas/yr]', fontsize=9)
    ax.set_ylabel(r'$\mu_\delta$ [mas/yr]', fontsize=9)
    ax.set_title('All-star VPD', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal')

    # ── Panel 2: QSO-only VPD with secular aberration arrows ─────────────────
    ax2 = axes[1]
    qso_pm_ra  = pmra_all[qso_sidx]
    qso_pm_dec = pmdec_all[qso_sidx]
    ok_q = np.isfinite(qso_pm_ra) & np.isfinite(qso_pm_dec)

    if ok_q.any():
        ax2.scatter(qso_pm_ra[ok_q], qso_pm_dec[ok_q],
                    s=50, c='gold', edgecolors='darkorange', linewidths=0.8,
                    zorder=4, label='QSO bp3m PM')
        # Draw arrow from expected aberration to observed bp3m PM
        for ra_obs, dec_obs, ra_exp, dec_exp in zip(
                qso_pm_ra[ok_q], qso_pm_dec[ok_q],
                qso_pmra_mas[ok_q], qso_pmdec_mas[ok_q]):
            ax2.annotate('', xy=(ra_obs, dec_obs), xytext=(ra_exp, dec_exp),
                         arrowprops=dict(arrowstyle='->', color='darkorange',
                                         lw=0.8, alpha=0.7))
        ax2.scatter(qso_pmra_mas[ok_q], qso_pmdec_mas[ok_q],
                    s=30, c='white', edgecolors='darkorange', linewidths=1.0,
                    zorder=5, marker='o', label='secular aberration pred.')

    ax2.axhline(0, c='k', lw=0.5, ls='--', alpha=0.3)
    ax2.axvline(0, c='k', lw=0.5, ls='--', alpha=0.3)
    # Small VPD range centred near 0 (QSO PMs are ~0)
    _q_lim = 1.5
    ax2.set_xlim(-_q_lim, _q_lim)
    ax2.set_ylim(-_q_lim, _q_lim)
    ax2.set_xlabel(r'$\mu_{\alpha*}$ [mas/yr]', fontsize=9)
    ax2.set_ylabel(r'$\mu_\delta$ [mas/yr]', fontsize=9)
    ax2.set_title('QSO-only VPD  (arrows: pred → observed)', fontsize=10)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.set_aspect('equal')

    # ── Panel 3: CMD with QSO markers ─────────────────────────────────────────
    ax3 = axes[2]
    bp_rp_arr = np.asarray(bp_rp, dtype=float)
    has_cmd   = np.isfinite(gmag) & np.isfinite(bp_rp_arr)

    ax3.scatter(bp_rp_arr[has_cmd & _ok_nomem], gmag[has_cmd & _ok_nomem],
                s=4, c='#aaaaaa', alpha=0.4, zorder=1)
    ax3.scatter(bp_rp_arr[has_cmd & _ok_mem], gmag[has_cmd & _ok_mem],
                s=6, c='steelblue', alpha=0.7, zorder=2)
    _cmd_qso = has_cmd & _ok_qso
    if _cmd_qso.any():
        ax3.scatter(bp_rp_arr[_cmd_qso], gmag[_cmd_qso],
                    s=80, c='gold', edgecolors='darkorange', linewidths=0.8,
                    zorder=5, marker='*', label=f'QSO anchors (N={_cmd_qso.sum()})')
        ax3.legend(fontsize=7, loc='lower right')
    ax3.invert_yaxis()
    ax3.set_xlabel('BP − RP [mag]', fontsize=9)
    ax3.set_ylabel('G [mag]', fontsize=9)
    ax3.set_title('CMD with QSO anchors', fontsize=10)

    out_path = plot_dir / 'qso_diagnostics.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


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


def _plot_member_selection_panels(
    gaia_catalog, member_mask, pm_free, plx_free,
    field_dir, out_path, seed_mask=None, mu_pop=None,
    min_pair: int = 20, vpd_zoom: float = 3.0,
):
    """Final-membership panels: VPD, Gaia CMD, HST CMDs, colour-colour,
    parallax vs G — grey non-members, red members, orange = seeded but
    rejected. Mirrors notebook 07_member_selection's panel construction
    (keep the two in sync).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from itertools import combinations

    n = len(gaia_catalog)
    gmag  = gaia_catalog['gmag'].to_numpy(float)
    bp_rp = (gaia_catalog['bp_rp'].to_numpy(float)
             if 'bp_rp' in gaia_catalog.columns else np.full(n, np.nan))

    # HST photometry from the validator catalogue, keyed by Gaia_id
    hst_mags: dict[str, np.ndarray] = {}
    cmc_path = Path(field_dir) / 'cross_match_catalog.csv'
    if cmc_path.exists():
        try:
            _cat  = pd.read_csv(cmc_path, dtype={'gaia_source_id': np.int64})
            _wide = (_cat.pivot_table(index='gaia_source_id',
                                      columns='filter_camera',
                                      values='mag_norm_wmean', aggfunc='first'))
            _gids = gaia_catalog['Gaia_id'].to_numpy(np.int64)
            for band in _wide.columns:
                _ser = _wide[band]
                hst_mags[str(band)] = (
                    pd.Series(_gids).map(_ser).to_numpy(float))
        except Exception as _exc:
            print(f"    (cross_match_catalog.csv unusable for panels: {_exc})")

    _WL = {'F275W': 275, 'F336W': 336, 'F390W': 390, 'F435W': 435,
           'F438W': 438, 'F475W': 475, 'F555W': 555, 'F606W': 606,
           'F625W': 625, 'F775W': 775, 'F814W': 814, 'F850LP': 900,
           'F110W': 1100, 'F125W': 1250, 'F160W': 1600}
    bands = sorted(hst_mags, key=lambda b: _WL.get(b.split('/')[0].upper(), 9999))

    def _n_joint(*arrs):
        m = np.ones(n, bool)
        for a in arrs:
            m &= np.isfinite(a)
        return int(m.sum())

    panels = [('VPD (pop-fit free PM)', pm_free[:, 0], pm_free[:, 1],
               'pmra [mas/yr]', 'pmdec [mas/yr]', False)]
    if np.isfinite(bp_rp).any():
        panels.append(('Gaia CMD', bp_rp, gmag, 'BP − RP', 'G', True))
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            b, r = bands[i], bands[j]
            if _n_joint(hst_mags[b], hst_mags[r]) < min_pair:
                continue
            panels.append((f'{b} − {r} CMD', hst_mags[b] - hst_mags[r],
                           hst_mags[r], f'{b} − {r}', r, True))
    for b1, b2, b3 in combinations(bands, 3):
        if len({b.split('/')[0] for b in (b1, b2, b3)}) < 3:
            continue
        if _n_joint(hst_mags[b1], hst_mags[b2], hst_mags[b3]) < min_pair:
            continue
        panels.append((f'({b1}−{b2}) vs ({b2}−{b3})',
                       hst_mags[b1] - hst_mags[b2], hst_mags[b2] - hst_mags[b3],
                       f'{b1} − {b2}', f'{b2} − {b3}', False))
    panels.append(('Parallax', gmag, plx_free, 'G', 'parallax [mas]', False))

    seed_rej = (seed_mask & ~member_mask) if seed_mask is not None else None
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, x, y, xl, yl, inv) in zip(axes, panels):
        xv = np.asarray(x, float); yv = np.asarray(y, float)
        fin = np.isfinite(xv) & np.isfinite(yv)
        ax.scatter(xv[fin & ~member_mask], yv[fin & ~member_mask],
                   s=4, c='0.75', lw=0, label='non-member')
        if seed_rej is not None and seed_rej.any():
            ax.scatter(xv[fin & seed_rej], yv[fin & seed_rej], s=14,
                       facecolors='none', edgecolors='darkorange', lw=0.8,
                       label='seed, rejected')
        ax.scatter(xv[fin & member_mask], yv[fin & member_mask],
                   s=8, c='crimson', lw=0, label='member')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'{title}  ({int((fin & member_mask).sum())} mem)',
                     fontsize=10)
        if inv:
            ax.invert_yaxis()
        if title.startswith('VPD'):
            _cx, _cy = ((float(mu_pop[0]), float(mu_pop[1]))
                        if mu_pop is not None else (0.0, 0.0))
            ax.plot([_cx], [_cy], marker='+', ms=14, c='k', mew=1.5, zorder=5)
            ax.set_xlim(_cx - vpd_zoom, _cx + vpd_zoom)
            ax.set_ylim(_cy - vpd_zoom, _cy + vpd_zoom)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Final members: {int(member_mask.sum())} of {n} stars',
                 y=1.001)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


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


def _restrict_to_members(solver, image_names: list, member_sidx: np.ndarray) -> None:
    """Set use_for_fit=False for all detections belonging to non-member stars.

    Called each iteration when --fit_members_only is active.  Updates
    gaia_n_hst_used to reflect the new active set.
    """
    is_member = np.zeros(solver.n_stars, dtype=bool)
    is_member[member_sidx] = True
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        sidx_img = d['sidx']
        old_use  = np.asarray(d['use_for_fit'], dtype=bool)
        new_use  = old_use & is_member[sidx_img]
        d['use_for_fit'] = new_use
        if 'use_for_astrom' in d:
            d['use_for_astrom'] = np.asarray(d['use_for_astrom'], dtype=bool) & is_member[sidx_img]
    solver.gaia_n_hst_used[:] = 0
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        use_any = d['use_for_fit'] | d.get('use_for_astrom', d['use_for_fit'])
        np.add.at(solver.gaia_n_hst_used, d['sidx'][use_any], 1)


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
    alpha_damp: float = 0.5,
    n_iter_phase4: int = 0,
    phase4_mode: str = 'hard',
    student_t_nu: float = 50.0,
    z_threshold: float = 0.8,
    member_sigma_clip: float = 3.0,
    pm_sys_floor: float = 0.2,
    max_sigma_free_pm: float = 1.0,
    fit_members_only: bool = False,
    mu_pop_init: tuple[float, float] | None = None,
    mu_pop_init_source: str | None = None,
    member_seed_csv: "Path | str | None" = None,
    use_member_seed: bool = False,
    freeze_member_seed: bool = False,
    freeze_mu_pop_init: bool = False,
    poly_order: int | None = None,
    no_plots: bool = False,
    fit_cte: bool = False,
    cte_mag_poly_order: int = 3,
    cte_spatial_order: int = 2,
    cte_time_poly_order: int = 0,
    cte_n_iter: int = 10,
    lib_dir: "Path | None" = None,
    use_qso_anchors: bool = True,
    qso_anchors_csv: "Path | str | list | None" = None,
) -> Path:
    """
    Run population PM fitting.

    Data loading mirrors run_alignment.py exactly:
      load_image_data_flc → split_images_by_ccd → build_index_maps → BP3MSolver
    v1 r_hat, alpha, and use_for_fit flags are loaded from BP3M_results/.
    """
    from bp3m.data_loader_flc import load_image_data_flc
    from bp3m.data_loader_flc import build_index_maps, split_images_by_ccd
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

    _v1_hp = v1_cfg.get('prior_hyperparams', {})
    v1_prior_sigma_rot_deg       = _v1_hp.get('sigma_rot_deg',           None)
    v1_prior_sigma_scale         = _v1_hp.get('sigma_scale',              None)
    v1_prior_sigma_skew          = _v1_hp.get('sigma_skew',               None)
    v1_prior_sigma_pointing      = _v1_hp.get('sigma_pointing_mas',       None)
    v1_use_pair_prior            = _v1_hp.get('use_pair_prior',             False)
    v1_prior_sigma_pair_rot_deg  = _v1_hp.get('sigma_pair_rot_deg',        None)
    v1_prior_sigma_pair_scale    = _v1_hp.get('sigma_pair_scale',           None)
    v1_prior_sigma_pair_skew     = _v1_hp.get('sigma_pair_skew',            None)
    v1_prior_sigma_pair_pointing = _v1_hp.get('sigma_pair_pointing_mas',    None)

    print("\n" + "─" * 60)
    print("BP3M pop-fit: population PM fitting")
    print("─" * 60)
    print(f"  field={field_name}")
    print(f"  σ_pm={sigma_pm} mas/yr  plx_pop={plx_pop} mas  "
          f"σ_plx_tot={sigma_plx_tot} mas")
    print(f"  μ_pop prior σ={mu_pop_prior_sigma} mas/yr  "
          f"member_sigma_clip={member_sigma_clip}  pm_sys_floor={pm_sys_floor} mas/yr")
    _p4_detail = (f"ν={student_t_nu:.1f}" if phase4_mode == 'soft'
                  else "hard-weight Tests 1+2+3")
    print(f"  n_iter: μ={n_iter_mu}  joint={n_iter_joint}  alpha={n_iter_alpha}  "
          f"phase4={n_iter_phase4} ({phase4_mode}: {_p4_detail})")
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

    # ── QSO anchor loading ────────────────────────────────────────────────────
    _qso_sidx      = None
    _qso_pmra_mas  = None
    _qso_pmdec_mas = None
    _n_qso_anchors = 0

    if use_qso_anchors:
        # Resolve which anchor CSV(s) to load.
        # qso_anchors_csv may be a single path, a list of paths, or None.
        # When None, fall back to globbing the Gaia dir (supports multiple
        # search-parameter sets whose files are concatenated + deduped).
        from .qso_vetting import find_qso_anchors
        _gaia_dir = data_root / field_name / 'Gaia'
        if qso_anchors_csv is not None:
            _anchor_paths = ([Path(qso_anchors_csv)]
                             if not isinstance(qso_anchors_csv, list)
                             else [Path(p) for p in qso_anchors_csv])
        else:
            _p = find_qso_anchors(_gaia_dir, field_name)
            # Also collect any additional parameterised files that exist
            _all = sorted(_gaia_dir.glob(f"{field_name}_*_qso_anchors.csv"),
                          key=lambda p: p.stat().st_mtime)
            _anchor_paths = _all if _all else ([_p] if _p and _p.exists() else [])

        _anchor_paths = [p for p in _anchor_paths if p.exists()]

        if _anchor_paths:
            try:
                _qdfs = [pd.read_csv(p, dtype={'source_id': 'int64'})
                         for p in _anchor_paths]
                _qdf = (pd.concat(_qdfs, ignore_index=True)
                        .drop_duplicates(subset=['source_id'])
                        .reset_index(drop=True)
                        if len(_qdfs) > 1 else _qdfs[0])
                _qdf_anchors = _qdf[_qdf['is_qso_anchor'].fillna(False)]

                _nq_total = len(_qdf)
                _nq_5p    = int(_qdf['has_5p_solution'].sum())
                _nq_astro = int(_qdf['astrometric_pass'].sum())
                _nq_cat   = int(_qdf['catalog_match'].sum())
                _nq_anch  = len(_qdf_anchors)
                print(f"  QSO vetting summary ({len(_anchor_paths)} file(s)):")
                print(f"    Gaia qso_candidates:          {_nq_total}")
                print(f"    With 5p/6p solution:          {_nq_5p}")
                print(f"    Astrometric cut (<3σ):        {_nq_astro}")
                print(f"    Catalog match (Quaia/MILLIQUAS/CRF3): {_nq_cat}")
                print(f"    Vetted QSO anchors:           {_nq_anch}")

                _qso_idx_list = []
                _qso_pmra_list = []
                _qso_pmdec_list = []
                for _, _row in _qdf_anchors.iterrows():
                    _sidx = star_id_to_idx.get(int(_row['source_id']))
                    if _sidx is not None:
                        _qso_idx_list.append(_sidx)
                        _qso_pmra_list.append(float(_row['pmra_aberr_uas']) * 1e-3)
                        _qso_pmdec_list.append(float(_row['pmdec_aberr_uas']) * 1e-3)
                if _qso_idx_list:
                    _qso_sidx      = np.array(_qso_idx_list,  dtype=int)
                    _qso_pmra_mas  = np.array(_qso_pmra_list, dtype=float)
                    _qso_pmdec_mas = np.array(_qso_pmdec_list, dtype=float)
                    _n_qso_anchors = len(_qso_sidx)
                print(f"    In HST field (in solver):     {_n_qso_anchors}  "
                      + ("← tight σ_κ prior applied" if _n_qso_anchors > 0
                         else "(none in HST FOV — prior not applied)"))
            except Exception as _qexc:
                print(f"  WARNING: could not load QSO anchors — {_qexc}")
        else:
            print("  QSO anchor file not found; re-run from Phase 1 to generate it")
    else:
        print("  QSO anchors disabled (--no_qso_anchors)")

    # ── Build solver ──────────────────────────────────────────────────────────
    solver = BP3MSolver(imgs, filtered_spi, gaia_catalog,
                        star_id_to_idx, image_names, star_in_image,
                        poly_order=poly_order,
                        prior_sigma_rot_deg=v1_prior_sigma_rot_deg,
                        prior_sigma_scale=v1_prior_sigma_scale,
                        prior_sigma_skew=v1_prior_sigma_skew,
                        prior_sigma_pointing=v1_prior_sigma_pointing,
                        prior_sigma_pair_rot_deg=v1_prior_sigma_pair_rot_deg,
                        prior_sigma_pair_scale=v1_prior_sigma_pair_scale,
                        prior_sigma_pair_skew=v1_prior_sigma_pair_skew,
                        prior_sigma_pair_pointing=v1_prior_sigma_pair_pointing,
                        use_pair_prior=v1_use_pair_prior)
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
    # Save original Gaia uncertainties before v1 overwrites (used as fallback below)
    _gaia_sig_pmra  = _sig_pmra_init.copy()
    _gaia_sig_pmdec = _sig_pmdec_init.copy()
    _gaia_corr_pm   = _corr_pm_init.copy()

    _v1_pm_loaded = False
    _v1_matched   = np.zeros(solver.n_stars, dtype=bool)
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
                        _v1_matched[i]     = True
                _v1_pm_loaded = True
                # Restore Gaia sigmas wherever v1 wrote NaN (don't lose Gaia covariance)
                _sig_pmra_init  = np.where(np.isfinite(_sig_pmra_init),
                                           _sig_pmra_init,  _gaia_sig_pmra)
                _sig_pmdec_init = np.where(np.isfinite(_sig_pmdec_init),
                                           _sig_pmdec_init, _gaia_sig_pmdec)
                _corr_pm_init   = np.where(np.isfinite(_corr_pm_init),
                                           _corr_pm_init,   _gaia_corr_pm)

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

    # ── Bootstrap μ_pop from v1 PMs ──────────────────────────────────────────
    _mu_init_arr = (np.array([float(mu_pop_init[0]), float(mu_pop_init[1])])
                    if mu_pop_init is not None else None)
    if _v1_pm_loaded:
        _pmra_v1_only  = np.where(_v1_matched, _pmra_init,  np.nan)
        _pmdec_v1_only = np.where(_v1_matched, _pmdec_init, np.nan)
    else:
        _pmra_v1_only  = _pmra_init
        _pmdec_v1_only = _pmdec_init

    if freeze_mu_pop_init and _mu_init_arr is not None:
        print(f"\n  Skipping bootstrap — using --mu_pop_init directly: "
              f"({_mu_init_arr[0]:+.4f}, {_mu_init_arr[1]:+.4f}) mas/yr")
        _mu_boot = _mu_init_arr.copy()
    elif _v1_pm_loaded:
        if _mu_init_arr is not None:
            print(f"\n  Bootstrapping μ_pop from v1 bp3m PMs (sigma-clip seeded at "
                  f"({_mu_init_arr[0]:+.4f}, {_mu_init_arr[1]:+.4f}) mas/yr)...")
        else:
            print("\n  Bootstrapping μ_pop from v1 bp3m PMs (sigma-clip)...")
        _mu_boot = _estimate_mu_pop_v1(_pmra_v1_only, _pmdec_v1_only,
                                       mu_init=_mu_init_arr)
    else:
        print("  WARNING: v1 PMs not loaded; using Gaia sigma-clip for μ_pop bootstrap")
        _mu_boot = _estimate_mu_pop(gaia_catalog)

    # ── Initial member selection using v1 PMs only ────────────────────────────
    # --use_member_seed / --freeze_member_seed: auto-locate the CSV written by
    # notebook 07_member_selection in the field directory.
    if member_seed_csv is None and (use_member_seed or freeze_member_seed):
        member_seed_csv = data_root / field_name / 'member_seed.csv'
        if not Path(member_seed_csv).exists():
            raise FileNotFoundError(
                f"use_member_seed/freeze_member_seed: no seed found at "
                f"{member_seed_csv} — draw and save one with notebook "
                f"07_member_selection first.")
    if member_seed_csv is not None:
        # Hand-drawn seed (e.g. from notebook 07_member_selection): replaces the
        # sigma-clip initial selection. The phases still refine membership.
        _seed_path = Path(member_seed_csv)
        _seed = pd.read_csv(_seed_path, dtype={'gaia_source_id': np.int64})
        if 'trusted' in _seed.columns:
            _seed = _seed[_seed['trusted'].astype(bool)]
        _seed_ids = [int(g) for g in _seed['gaia_source_id'].values]
        _found    = [star_id_to_idx[g] for g in _seed_ids if g in star_id_to_idx]
        _n_miss   = len(_seed_ids) - len(_found)
        member_sidx = np.array(sorted(_found), dtype=int)
        print(f"\n  Initial members from seed CSV {_seed_path.name}: "
              f"{len(member_sidx)} matched"
              + (f"  ({_n_miss} seed IDs not in this field's star list)" if _n_miss else ""))
        if len(member_sidx) < 3:
            raise ValueError(
                f"member_seed_csv matched only {len(member_sidx)} stars — "
                f"check that the IDs come from this field's catalogues.")
    else:
        print("\n  Selecting initial members from v1 bp3m PMs...")
        member_sidx = _select_initial_members(
            _pmra_v1_only, _pmdec_v1_only,
            _sig_pmra_init, _sig_pmdec_init, _corr_pm_init,
            _mu_boot, member_sigma_clip, sigma_pm, pm_sys_floor)
        print(f"  Initial members: {len(member_sidx)}")

    # Freeze semantics: membership refinement in the phases may REMOVE stars
    # from the seed-defined group but can never add stars from outside it.
    _seed_initial_sidx = (np.asarray(member_sidx, int).copy()
                          if member_seed_csv is not None else None)
    _seed_frozen_sidx = None
    if freeze_member_seed:
        _seed_frozen_sidx = np.asarray(member_sidx, int).copy()
        print(f"  Freeze: membership restricted to the {len(_seed_frozen_sidx)} "
              f"seed stars (phases can remove, never add)")

    def _freeze_members(_sidx):
        if _seed_frozen_sidx is None:
            return _sidx
        return np.intersect1d(np.asarray(_sidx, int), _seed_frozen_sidx)

    # ── μ_pop prior: weighted mean of initial members ─────────────────────────
    _extra = sigma_pm ** 2 + pm_sys_floor ** 2
    _mem_pm_ra   = _pmra_v1_only[member_sidx]
    _mem_pm_dec  = _pmdec_v1_only[member_sidx]
    _mem_sig_ra  = _sig_pmra_init[member_sidx]
    _mem_sig_dec = _sig_pmdec_init[member_sidx]
    _fin_m = np.isfinite(_mem_pm_ra) & np.isfinite(_mem_pm_dec)
    if _fin_m.sum() >= 3:
        _wra  = 1.0 / (_mem_sig_ra[_fin_m]  ** 2 + _extra)
        _wdec = 1.0 / (_mem_sig_dec[_fin_m] ** 2 + _extra)
        _mu_ra_v1   = float(np.sum(_wra  * _mem_pm_ra[_fin_m])  / np.sum(_wra))
        _mu_dec_v1  = float(np.sum(_wdec * _mem_pm_dec[_fin_m]) / np.sum(_wdec))
        _unc_ra_v1  = float(1.0 / np.sqrt(np.sum(_wra)))
        _unc_dec_v1 = float(1.0 / np.sqrt(np.sum(_wdec)))
        mu_pop_prior = np.array([_mu_ra_v1, _mu_dec_v1])
    else:
        print("  WARNING: too few members with finite v1 PMs; using bootstrap center as prior")
        mu_pop_prior = _mu_boot.copy()
        _unc_ra_v1 = _unc_dec_v1 = 0.0
    _n_prior_members = int(_fin_m.sum()) if '_fin_m' in dir() else 0
    print(f"  μ_pop prior from v1 members (N={_n_prior_members}): "
          f"({mu_pop_prior[0]:+.4f} ± {_unc_ra_v1:.4f}, "
          f"{mu_pop_prior[1]:+.4f} ± {_unc_dec_v1:.4f}) mas/yr  "
          f"[prior σ = ±{mu_pop_prior_sigma:.2f} mas/yr]")

    C_pop_prior_inv = np.eye(2) / mu_pop_prior_sigma ** 2
    mu_pop_current  = mu_pop_prior.copy()

    # ── Nσ reference: the initial μ_pop, with its provenance ─────────────────
    # Every phase prints the Mahalanobis distance of the current μ_pop from
    # this reference, using that phase's own posterior μ_pop covariance.
    _mu_init_ref = _mu_boot.copy()
    if freeze_mu_pop_init and _mu_init_arr is not None:
        _mu_ref_label = f'--mu_pop_init [{mu_pop_init_source or "user"}]'
    elif _mu_init_arr is not None and mu_pop_init_source:
        _mu_ref_label = f'sigma-clip bootstrap seeded from [{mu_pop_init_source}] init'
    elif _v1_pm_loaded:
        _mu_ref_label = 'empirical v1 sigma-clip bootstrap'
    else:
        _mu_ref_label = 'empirical Gaia sigma-clip bootstrap'
    print(f"  Nσ reference: μ_init=({_mu_init_ref[0]:+.4f}, {_mu_init_ref[1]:+.4f}) mas/yr  "
          f"({_mu_ref_label})")

    def _nsig_init(_mu, _C2):
        """Mahalanobis distance of _mu from μ_init under 2×2 posterior cov _C2."""
        if _C2 is None:
            return np.nan
        try:
            _d = np.asarray(_mu, float) - _mu_init_ref
            return float(np.sqrt(_d @ np.linalg.solve(np.asarray(_C2, float), _d)))
        except Exception:
            return np.nan

    # Print Gaia weighted PM mean for the same initial members (reference only)
    _gaia_pmra_col   = gaia_catalog['pmra'].to_numpy(float)[member_sidx]
    _gaia_pmdec_col  = gaia_catalog['pmdec'].to_numpy(float)[member_sidx]
    _gaia_sig_ra_col = gaia_catalog['pmra_error'].to_numpy(float)[member_sidx]
    _gaia_sig_dec_col = gaia_catalog['pmdec_error'].to_numpy(float)[member_sidx]
    _gaia_fin = np.isfinite(_gaia_pmra_col) & np.isfinite(_gaia_pmdec_col)
    if _gaia_fin.sum() > 0:
        _gw_ra  = 1.0 / (_gaia_sig_ra_col[_gaia_fin]  ** 2 + _extra)
        _gw_dec = 1.0 / (_gaia_sig_dec_col[_gaia_fin] ** 2 + _extra)
        _gmu_ra  = float(np.sum(_gw_ra  * _gaia_pmra_col[_gaia_fin])  / np.sum(_gw_ra))
        _gmu_dec = float(np.sum(_gw_dec * _gaia_pmdec_col[_gaia_fin]) / np.sum(_gw_dec))
        _gunc_ra  = float(1.0 / np.sqrt(np.sum(_gw_ra)))
        _gunc_dec = float(1.0 / np.sqrt(np.sum(_gw_dec)))
        print(f"  Gaia PM mean for initial members (N={_gaia_fin.sum()}, reference only): "
              f"({_gmu_ra:+.4f} ± {_gunc_ra:.4f}, "
              f"{_gmu_dec:+.4f} ± {_gunc_dec:.4f}) mas/yr")

    # Recompute gaia_n_hst_used to reflect the v1 flags we just applied
    solver.gaia_n_hst_used[:] = 0
    for _img in image_names:
        _d = solver._img_data.get(_img)
        if _d is None:
            continue
        _use_any = _d['use_for_fit'] | _d.get('use_for_astrom', _d['use_for_fit'])
        np.add.at(solver.gaia_n_hst_used, _d['sidx'][_use_any], 1)

    # ── Convenience wrapper: inject QSO anchor args into every _joint_solve call
    def _solve(member_sidx_arg, mu_pop_arg, r_arg,
               fix_r_arg=False, z_weights_arg=None):
        return _joint_solve_pop(
            solver, image_names,
            member_sidx_arg, mu_pop_arg,
            sigma_pm, plx_pop, sigma_plx_tot,
            C_pop_prior_inv, mu_pop_prior,
            r_arg, fix_r=fix_r_arg, z_weights=z_weights_arg,
            qso_sidx=_qso_sidx,
            qso_pmra=_qso_pmra_mas,
            qso_pmdec=_qso_pmdec_mas,
        )

    # ── Phase 1: μ-only solve (r fixed at v1 values) ──────────────────────────
    print(f"\n  Phase 1: μ-only solve ({n_iter_mu} iterations, r fixed)...")
    r_current = r_bp3m.copy()
    C_shared_mu = None
    for mu_iter in range(n_iter_mu):
        _, mu_pop_new, C_shared_mu, C_vT, a_arr, _, _ = _solve(
            member_sidx, mu_pop_current, r_current, fix_r_arg=True,
        )
        delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
        _mu_pop_used = mu_pop_current.copy()   # mu_pop baked into a_arr
        mu_pop_current = mu_pop_new
        _a_free, _C_free = _compute_free_stellar_posterior(
            a_arr, C_vT, member_sidx, sigma_pm, sigma_plx_tot, _mu_pop_used, plx_pop,
            solver._C_VG_inv_per_star)
        member_sidx = _freeze_members(_select_members_from_a(
            _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
            sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor,
            max_sigma_free_pm=max_sigma_free_pm))
        print(f"    iter {mu_iter + 1}/{n_iter_mu}: "
              f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f}) mas/yr  "
              f"Δμ={delta_mu:.4e}  Nσ_init={_nsig_init(mu_pop_current, C_shared_mu):.2f}  "
              f"members={len(member_sidx)}")
        if delta_mu < 1e-6:
            print(f"    Converged.")
            break

    if C_shared_mu is not None:
        sigma_mu_1 = np.sqrt(np.diag(C_shared_mu))
        print(f"  Phase 1 final: μ_pop=({mu_pop_current[0]:+.4f} ± {sigma_mu_1[0]:.4f}, "
              f"{mu_pop_current[1]:+.4f} ± {sigma_mu_1[1]:.4f}) mas/yr  "
              f"Nσ_init={_nsig_init(mu_pop_current, C_shared_mu):.2f} "
              f"(cov conditional on r fixed)")

    # ── Phase 2: joint solve (r + μ_pop) ─────────────────────────────────────
    # σ_μpop per iteration, from the JOINT shared covariance: the μ_pop block
    # is C_shared[n_r:, n_r:], i.e. marginal over the alignment r.  (Phase 1's
    # 2×2 C_shared is conditional on r fixed, which is why it is much smaller.)
    _n_r_shared = len(image_names) * solver.N_R

    def _mu_cov_of(_C_shared):
        """2×2 marginal μ_pop covariance block of the joint shared covariance."""
        return None if _C_shared is None else _C_shared[_n_r_shared:, _n_r_shared:]

    def _sigma_mu_of(_C_shared):
        _C2 = _mu_cov_of(_C_shared)
        if _C2 is None:
            return np.nan, np.nan
        _s = np.sqrt(np.diag(_C2))
        return float(_s[0]), float(_s[1])

    print(f"\n  Phase 2: joint solve ({n_iter_joint} iterations)...")
    C_shared_joint = None
    for jt_iter in range(n_iter_joint):
        r_new, mu_pop_new, C_shared_joint, C_vT, a_arr, _, _ = _solve(
            member_sidx, mu_pop_current, r_current,
        )
        delta_r  = float(np.max(np.abs(r_new - r_current)))
        delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
        _mu_pop_used = mu_pop_current.copy()   # mu_pop baked into a_arr
        r_current      = r_new
        mu_pop_current = mu_pop_new
        _a_free, _C_free = _compute_free_stellar_posterior(
            a_arr, C_vT, member_sidx, sigma_pm, sigma_plx_tot, _mu_pop_used, plx_pop,
            solver._C_VG_inv_per_star)
        member_sidx = _freeze_members(_select_members_from_a(
            _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
            sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor,
            max_sigma_free_pm=max_sigma_free_pm))
        if fit_members_only:
            _restrict_to_members(solver, image_names, member_sidx)
        _smu = _sigma_mu_of(C_shared_joint)
        print(f"    iter {jt_iter + 1}/{n_iter_joint}: "
              f"μ_pop=({mu_pop_current[0]:+.4f}±{_smu[0]:.4f}, "
              f"{mu_pop_current[1]:+.4f}±{_smu[1]:.4f})  "
              f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  "
              f"Nσ_init={_nsig_init(mu_pop_current, _mu_cov_of(C_shared_joint)):.2f}  "
              f"members={len(member_sidx)}")
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
            r_new, mu_pop_new, C_shared_joint_p3, C_vT, a_arr, _, _ = _solve(
                member_sidx, mu_pop_current, r_current,
            )
            solver._update_R(r_new)
            solver._update_geometry(r_new, a_arr)

            alpha_info = _compute_alpha_updates(solver, image_names, r_new, a_arr,
                                                 alpha_damp=alpha_damp)

            delta_r     = float(np.max(np.abs(r_new - r_current)))
            delta_mu    = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
            _mu_pop_used = mu_pop_current.copy()   # mu_pop baked into a_arr
            r_current      = r_new
            mu_pop_current = mu_pop_new
            _a_free, _C_free = _compute_free_stellar_posterior(
                a_arr, C_vT, member_sidx, sigma_pm, sigma_plx_tot, _mu_pop_used, plx_pop,
                solver._C_VG_inv_per_star)
            member_sidx = _freeze_members(_select_members_from_a(
                _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
                sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor,
                max_sigma_free_pm=max_sigma_free_pm))
            if fit_members_only:
                _restrict_to_members(solver, image_names, member_sidx)

            for img, n_use, n_tot, alpha_prev, alpha_raw, alpha_new in alpha_info:
                tag = '  ← raised' if alpha_new > alpha_prev + 1e-4 else (
                      '  ← lowered' if alpha_new < alpha_prev - 1e-4 else '')
                print(f"    {img}: {n_use:4d}/{n_tot:4d} align  "
                      f"α_prev={alpha_prev:.3f}  α_raw={alpha_raw:.3f}  "
                      f"α_new={alpha_new:.3f}{tag}")

            delta_alpha_max = (max(abs(ai[5] - ai[3]) for ai in alpha_info)
                               if alpha_info else 0.0)
            _smu = _sigma_mu_of(C_shared_joint_p3)
            print(f"    iter {al_iter + 1}/{n_iter_alpha}: "
                  f"μ_pop=({mu_pop_current[0]:+.4f}±{_smu[0]:.4f}, "
                  f"{mu_pop_current[1]:+.4f}±{_smu[1]:.4f})  "
                  f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  "
                  f"Δα_max={delta_alpha_max:.3e}  "
                  f"Nσ_init={_nsig_init(mu_pop_current, _mu_cov_of(C_shared_joint_p3)):.2f}  "
                  f"members={len(member_sidx)}")
            if delta_r < 1e-6 and delta_mu < 1e-6 and delta_alpha_max < 1e-4:
                print(f"    Converged.")
                break

        C_shared_joint = C_shared_joint_p3

    # ── Re-open detections using Phase 3 residual thresholds ─────────────────
    _p3_active = None
    if n_iter_phase4 > 0:
        # Save Phase 3 active set for diagnostics / soft-weights plot
        _p3_active = {
            img: solver._img_data[img]['use_for_fit'].copy()
            for img in image_names
            if solver._img_data.get(img) is not None
        }

        # Re-evaluate ALL detections (including v1-excluded ones) against
        # per-image adaptive thresholds derived from the Phase 3 trusted set.
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

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    def _apply_z_threshold(z_dict: dict, thresh: float) -> dict:
        return {img: (z * (z >= thresh) if z is not None else None)
                for img, z in z_dict.items()}

    z_weights_final = None
    if n_iter_phase4 > 0 and phase4_mode == 'hard':
        # ── Phase 4 (hard): population-aware Tests 1+2+3, iterate until stable
        print(f"\n  Phase 4: hard-weight outlier rejection  "
              f"({n_iter_phase4} max iterations, α frozen)...")
        C_shared_joint_p4 = C_shared_joint
        ok_star_prev = None

        for h_iter in range(n_iter_phase4):
            r_new, mu_pop_new, C_shared_joint_p4, C_vT_p4, a_arr_p4, _, _ = _solve(
                member_sidx, mu_pop_current, r_current,
            )
            solver._update_R(r_new)
            solver._update_geometry(r_new, a_arr_p4)

            delta_r  = float(np.max(np.abs(r_new - r_current)))
            delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
            _mu_pop_used   = mu_pop_current.copy()
            r_current      = r_new
            mu_pop_current = mu_pop_new
            a_arr          = a_arr_p4

            _a_free, _C_free = _compute_free_stellar_posterior(
                a_arr, C_vT_p4, member_sidx, sigma_pm, sigma_plx_tot, _mu_pop_used, plx_pop,
                solver._C_VG_inv_per_star)

            ok_star, n_use_changed, uff_info = _hard_update_phase4(
                solver, image_names, r_current, a_arr, C_vT_p4,
                member_sidx, mu_pop_current,
                sigma_pm, plx_pop, sigma_plx_tot,
                ok_star_prev=ok_star_prev,
                a_free=_a_free, C_free=_C_free,
                mu_pop_solve=_mu_pop_used,
            )
            ok_star_prev = ok_star

            member_sidx = _freeze_members(_select_members_from_a(
                _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
                sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor,
                max_sigma_free_pm=max_sigma_free_pm))
            if fit_members_only:
                _restrict_to_members(solver, image_names, member_sidx)

            n_use_img = sum(d['use_for_fit'].sum()
                            for d in (solver._img_data.get(img) for img in image_names)
                            if d is not None)
            _smu = _sigma_mu_of(C_shared_joint_p4)
            print(f"    iter {h_iter + 1}/{n_iter_phase4}: "
                  f"μ_pop=({mu_pop_current[0]:+.4f}±{_smu[0]:.4f}, "
                  f"{mu_pop_current[1]:+.4f}±{_smu[1]:.4f})  "
                  f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  "
                  f"Δuse={n_use_changed}  "
                  f"Nσ_init={_nsig_init(mu_pop_current, _mu_cov_of(C_shared_joint_p4)):.2f}  "
                  f"members={len(member_sidx)}")
            if delta_r < 1e-6 and delta_mu < 1e-6 and n_use_changed == 0:
                print(f"    Converged.")
                break

        C_shared_joint = C_shared_joint_p4

    elif n_iter_phase4 > 0 and phase4_mode == 'soft':
        # ── Phase 4 (soft): Student-t IRLS ───────────────────────────────────
        print(f"\n  Phase 4: soft-weight IRLS  ν={student_t_nu:.1f}  z_threshold={z_threshold:.2f}  "
              f"({n_iter_phase4} iterations, α frozen)...")

        # Evaluate z on the Phase 3 active set (before re-opening) to show the
        # warm-start distribution.
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

        _z_raw, n_det_total, n_eff = solver._update_soft_weights(
            r_current, a_arr, student_t_nu)
        z_weights_final = _apply_z_threshold(_z_raw, z_threshold)
        C_shared_joint_sw = C_shared_joint

        for sw_iter in range(n_iter_phase4):
            r_new, mu_pop_new, C_shared_joint_sw, C_vT_sw, a_arr_sw, _, _ = _solve(
                member_sidx, mu_pop_current, r_current, z_weights_arg=z_weights_final,
            )
            solver._update_R(r_new)
            solver._update_geometry(r_new, a_arr_sw)

            _z_raw_new, n_det_total, n_eff_new = solver._update_soft_weights(
                r_new, a_arr_sw, student_t_nu)
            z_new = _apply_z_threshold(_z_raw_new, z_threshold)

            delta_r  = float(np.max(np.abs(r_new - r_current)))
            delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
            delta_z  = float(sum(
                np.sum(np.abs(z_new[img] - z_weights_final[img]))
                for img in image_names
                if z_new.get(img) is not None and z_weights_final.get(img) is not None))

            _mu_pop_used   = mu_pop_current.copy()
            r_current      = r_new
            mu_pop_current = mu_pop_new
            z_weights_final = z_new
            a_arr           = a_arr_sw

            _a_free, _C_free = _compute_free_stellar_posterior(
                a_arr, C_vT_sw, member_sidx, sigma_pm, sigma_plx_tot, _mu_pop_used, plx_pop,
                solver._C_VG_inv_per_star)
            member_sidx = _freeze_members(_select_members_from_a(
                _a_free, mu_pop_current, _n_hst_det, _C_free, sigma_pm,
                sigma_clip=member_sigma_clip, pm_sys_floor=pm_sys_floor,
                max_sigma_free_pm=max_sigma_free_pm))
            if fit_members_only:
                _restrict_to_members(solver, image_names, member_sidx)

            _smu = _sigma_mu_of(C_shared_joint_sw)
            print(f"    iter {sw_iter + 1}/{n_iter_phase4}: "
                  f"n_eff={n_eff_new:.0f}/{n_det_total}  "
                  f"μ_pop=({mu_pop_current[0]:+.4f}±{_smu[0]:.4f}, "
                  f"{mu_pop_current[1]:+.4f}±{_smu[1]:.4f})  "
                  f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  Δz={delta_z:.3e}  "
                  f"Nσ_init={_nsig_init(mu_pop_current, _mu_cov_of(C_shared_joint_sw)):.2f}  "
                  f"members={len(member_sidx)}")
            if delta_r < 1e-6 and delta_mu < 1e-6 and delta_z < 1e-2:
                print(f"    Converged.")
                break

        C_shared_joint = C_shared_joint_sw

    n_r = len(image_names) * solver.N_R
    sigma_mu_joint = (np.sqrt(np.diag(C_shared_joint[n_r:, n_r:]))
                      if C_shared_joint is not None else np.array([np.nan, np.nan]))
    print(f"\n  Final: μ_pop=({mu_pop_current[0]:+.4f} ± {sigma_mu_joint[0]:.4f}, "
          f"{mu_pop_current[1]:+.4f} ± {sigma_mu_joint[1]:.4f}) mas/yr  "
          f"Nσ_init={_nsig_init(mu_pop_current, C_shared_joint[n_r:, n_r:] if C_shared_joint is not None else None):.2f}")
    print(f"  Final members: {len(member_sidx)}")

    # ── Final posterior pass at convergence ───────────────────────────────────
    print("\n  Final posterior pass...")
    _, _, C_shared_final, C_vT_final, v_mean, _, K_img_final = _solve(
        member_sidx, mu_pop_current, r_current, z_weights_arg=z_weights_final,
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
    _, _, _, C_vT_free_sol, v_mean_free_cond, _, K_img_free = _solve(
        np.array([], dtype=int),   # no members → no population prior
        mu_pop_current, r_current, fix_r_arg=True,
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
        _meta      = imgs.get(img, {})
        _ra0_orig  = _meta.get('ra0',  float('nan'))
        _dec0_orig = _meta.get('dec0', float('nan'))
        _ra0_final  = _meta.get('ra0_final',  _ra0_orig)
        _dec0_final = _meta.get('dec0_final', _dec0_orig)
        _rows.append(dict(
            image_name=img,
            n_stars_alignment=int(np.sum(d_img.get('use_for_fit', np.zeros(0, bool)))),
            n_stars_astrometry_only=int(np.sum(
                use_ast & ~d_img.get('use_for_fit', np.zeros(0, bool)))),
            a=a, b=b, c=c, d=d,
            delta_ra0_mas=(_ra0_final - _ra0_orig) * 3_600_000.0 if solver.N_R > 4 else 0.0,
            delta_dec0_mas=(_dec0_final - _dec0_orig) * 3_600_000.0 if solver.N_R > 5 else 0.0,
            ra0_final=_ra0_final,
            dec0_final=_dec0_final,
            pixel_scale_mas=(np.sqrt(a * d - b * c)
                             * imgs.get(img, {}).get('orig_pixel_scale', 50.0)),
            rotation_deg=np.degrees(np.arctan2(b - c, a + d)),
            on_skew=(a - d) / 2,
            off_skew=(b + c) / 2,
            sigma_a=np.sqrt(C_j[0, 0]),   sigma_b=np.sqrt(C_j[1, 1]),
            sigma_c=np.sqrt(C_j[2, 2]),   sigma_d=np.sqrt(C_j[3, 3]),
            sigma_dra0_mas=np.sqrt(C_j[4, 4]) if solver.N_R > 4 else 0.0,
            sigma_ddec0_mas=np.sqrt(C_j[5, 5]) if solver.N_R > 5 else 0.0,
            alpha=float(d_img.get('alpha_applied', 1.0)),
            **{f'r_{k}': float(r_j[k]) for k in range(6, solver.N_R)},
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
    from bp3m.solver import _SIGMA_ROT_DEG, _SIGMA_SCALE, _SIGMA_SKEW, _SIGMA_POINTING
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
            'prior_hyperparams': {
                'sigma_rot_deg':           solver._prior_sigma_rot_deg,
                'sigma_scale':             solver._prior_sigma_scale,
                'sigma_skew':              solver._prior_sigma_skew,
                'sigma_pointing_mas':      solver._prior_sigma_pointing,
                'use_pair_prior':          solver._use_pair_prior,
                'sigma_pair_rot_deg':      solver._prior_sigma_pair_rot_deg,
                'sigma_pair_scale':        solver._prior_sigma_pair_scale,
                'sigma_pair_skew':         solver._prior_sigma_pair_skew,
                'sigma_pair_pointing_mas': solver._prior_sigma_pair_pointing,
            },
            'image_priors': {
                img: {
                    'r_prior':       solver._img_data[img]['r_prior'].tolist(),
                    'C_r_prior_inv': solver._img_data[img]['C_r_prior_inv'].tolist(),
                }
                for img in image_names
                if img in solver._img_data
            },
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
        try:
            _mem_mask = np.zeros(solver.n_stars, bool)
            _mem_mask[np.asarray(member_sidx, int)] = True
            _seed_mask = None
            if _seed_initial_sidx is not None:
                _seed_mask = np.zeros(solver.n_stars, bool)
                _seed_mask[_seed_initial_sidx] = True
            _panels_png = _plot_member_selection_panels(
                gaia_catalog, _mem_mask,
                v_mean_free_marg[:, 2:4], v_mean_free_marg[:, 4],
                field_dir=data_root / field_name,
                out_path=output_pfr / 'plots' / 'member_selection_panels.png',
                seed_mask=_seed_mask, mu_pop=mu_pop_current,
            )
            print(f"  Saved: {_panels_png}")
        except Exception as _exc:
            print(f"  WARNING: member_selection_panels.png failed — {_exc}")

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

        try:
            _plot_dir = output_pfr / 'plots'
            _plot_dir.mkdir(parents=True, exist_ok=True)
            _plot_pm_vs_properties(
                _plot_dir, solver, image_names, gaia_catalog,
                v_mean_free_marg, C_vT_free_sol,
                member_sidx, mu_pop_current, sigma_pm, field_name,
                qso_sidx=_qso_sidx,
            )
        except Exception as _exc:
            print(f"  WARNING: _plot_pm_vs_properties failed — {_exc}")

        # QSO-only VPD + CMD diagnostic plot
        if _qso_sidx is not None and len(_qso_sidx) > 0:
            try:
                _plot_qso_diagnostics(
                    _plot_dir, solver, gaia_catalog,
                    v_mean_free_marg, C_vT_free_sol,
                    member_sidx, _qso_sidx, _qso_pmra_mas, _qso_pmdec_mas,
                    mu_pop_current, field_name,
                )
            except Exception as _exc:
                print(f"  WARNING: _plot_qso_diagnostics failed — {_exc}")

        # ── Residual maps — last ───────────────────────────────────────────────
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
        solver._update_geometry(r_current, v_mean)

    # ── Optional CTE phase ────────────────────────────────────────────────────
    if fit_cte:
        from .run_alignment_cte import run_cte_phase_after_popfit
        cte_params, r_cte, mu_pop_cte, _ = run_cte_phase_after_popfit(
            solver=solver,
            image_names=image_names,
            stars_per_image=filtered_spi,
            gaia_catalog=gaia_catalog,
            r_hat=r_current,
            mu_pop_hat=mu_pop_current,
            member_sidx=member_sidx,
            sigma_pm=sigma_pm,
            plx_pop=plx_pop,
            sigma_plx_tot=sigma_plx_tot,
            mu_pop_prior=mu_pop_prior,
            C_pop_prior_inv=C_pop_prior_inv,
            mag_poly_order=cte_mag_poly_order,
            spatial_order=cte_spatial_order,
            time_poly_order=cte_time_poly_order,
            n_iter_cte=cte_n_iter,
            output_dir=output_pfr,
        )
        import numpy as _np
        _cte_out = {}
        for chip in ('hi', 'lo'):
            p = cte_params.get(chip)
            if p is not None:
                _cte_out[f'{chip}_gamma_x'] = p.gamma_x
                _cte_out[f'{chip}_gamma_y'] = p.gamma_y
        _cte_path = output_pfr / 'cte_params.npz'
        _np.savez(_cte_path, **_cte_out)
        print(f"  Saved CTE params: {_cte_path}")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Results: {output_pfr}")
    return output_pfr


# ── CLI entry point ───────────────────────────────────────────────────────────

def _lookup_lvd(lvd_dir: Path, key: str) -> dict:
    """Look up an LVD entry by key and return derived pop-fit parameters.

    Searches dwarf_all.csv then gc_harris.csv.  Returns a dict with any
    subset of:
      plx_pop          — cluster parallax (mas), from distance_modulus
      sigma_plx_tot    — parallax prior width (mas), from distance uncertainty
      mu_pop_init      — (pmra, pmdec) tuple (mas/yr), from pmra/pmdec columns
      mu_pop_prior_sigma — prior width on μ_pop (mas/yr), 3× max(PM errors)
      sigma_pm         — internal PM dispersion (mas/yr), vlos_sigma/(4.7405 d_kpc)
      d_kpc            — distance (kpc), for informational printing
    """
    import numpy as np
    import pandas as pd

    lvd_dir = Path(lvd_dir)
    for csv_name in ('dwarf_all.csv', 'gc_harris.csv'):
        csv = lvd_dir / csv_name
        if not csv.exists():
            continue
        df = pd.read_csv(csv, low_memory=False)
        if 'key' not in df.columns:
            continue
        rows = df[df['key'] == key]
        if len(rows) == 0:
            continue

        row = rows.iloc[0]
        params: dict = {}

        mu = float(row.get('distance_modulus', np.nan))
        if np.isfinite(mu):
            d_kpc = 10 ** (mu / 5 + 1) / 1000
            params['plx_pop'] = 1.0 / d_kpc   # mas
            params['d_kpc']   = d_kpc

            mu_ep = float(row.get('distance_modulus_ep', np.nan))
            mu_em = float(row.get('distance_modulus_em', np.nan))
            errs  = [v for v in (mu_ep, mu_em) if np.isfinite(v)]
            if errs:
                err    = float(np.mean(errs))
                d_hi   = 10 ** ((mu + err) / 5 + 1) / 1000
                d_lo   = 10 ** ((mu - err) / 5 + 1) / 1000
                sigma_d = (d_hi - d_lo) / 2.0
                params['sigma_plx_tot'] = sigma_d / d_kpc ** 2   # mas

            # Internal PM dispersion from the line-of-sight velocity dispersion,
            # assuming isotropy: sigma_pm [mas/yr] = sigma_vlos [km/s] / (4.7405 d [kpc])
            vlos_sig = float(row.get('vlos_sigma', np.nan))
            if np.isfinite(vlos_sig) and vlos_sig > 0:
                params['sigma_pm'] = vlos_sig / (4.740470463533348 * d_kpc)

        pmra  = float(row.get('pmra',  np.nan))
        pmdec = float(row.get('pmdec', np.nan))
        if np.isfinite(pmra) and np.isfinite(pmdec):
            params['mu_pop_init'] = (pmra, pmdec)

            pm_errs = [float(row.get(c, np.nan))
                       for c in ('pmra_ep', 'pmra_em', 'pmdec_ep', 'pmdec_em')]
            pm_errs = [v for v in pm_errs if np.isfinite(v) and v > 0]
            if pm_errs:
                params['mu_pop_prior_sigma'] = 3.0 * float(max(pm_errs))

        return params

    raise ValueError(f"LVD key {key!r} not found in {lvd_dir} "
                     f"(searched dwarf_all.csv and gc_harris.csv)")


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
    parser.add_argument('--sigma_pm', type=float, default=None,
                        help='Cluster PM dispersion (mas/yr). Derived from --lvd_key '
                             'vlos_sigma and distance if both exist; otherwise defaults '
                             'to 0.0075 (Leo I).')
    parser.add_argument('--plx_pop', type=float, default=None,
                        help='Cluster parallax (mas). Derived from --lvd_key distance if '
                             'not given; otherwise defaults to 0.003873 (Leo I).')
    parser.add_argument('--sigma_plx_tot', type=float, default=None,
                        help='Total parallax uncertainty (mas) for pop prior. Derived from '
                             '--lvd_key distance uncertainty if not given; otherwise '
                             'defaults to 0.0001425 (Leo I).')
    parser.add_argument('--mu_pop_prior_sigma', type=float, default=None,
                        help='Gaussian prior width on μ_pop (mas/yr). If --lvd_key provides '
                             'a literature PM uncertainty, set to 3× that value; otherwise '
                             'defaults to 0.5.')
    parser.add_argument('--lvd_key', type=str, default=None,
                        help='Local Volume Database key (e.g. "leo_1", "ngc_0055") to '
                             'automatically derive plx_pop, sigma_plx_tot, mu_pop_init, and '
                             'mu_pop_prior_sigma. Explicit CLI values always override LVD '
                             'values.')
    parser.add_argument('--lvd_dir', type=str, default=None,
                        help='Path to the LVD data directory containing dwarf_all.csv and '
                             'gc_harris.csv. Falls back to $BP3M_LVD_DIR, then '
                             '~/data_bootes/bp3m/local_volume_database/data/')
    parser.add_argument('--n_iter_mu', type=int, default=20,
                        help='Phase 1 (μ-only) solve iterations')
    parser.add_argument('--n_iter_joint', type=int, default=20,
                        help='Phase 2 (joint r+μ) solve iterations')
    parser.add_argument('--n_iter_alpha', type=int, default=20,
                        help='Phase 3 (joint r+μ+alpha) solve iterations (0 to skip)')
    parser.add_argument('--alpha_damp', type=float, default=0.5,
                        help='Under-relaxation for alpha update: alpha_new = alpha_prev * '
                             'alpha_raw^alpha_damp (0.5=geometric mean; 1.0=full step, may oscillate)')
    parser.add_argument('--n_iter_phase_4', type=int, default=0,
                        help='Phase 4 iterations (default 0 = skip; Phase 4 is '
                             'experimental and can diverge)')
    parser.add_argument('--phase4_mode', type=str, default='hard',
                        choices=['hard', 'soft'],
                        help='Phase 4 mode: hard=population-aware Tests 1+2+3 (default), '
                             'soft=Student-t IRLS')
    parser.add_argument('--student_t_nu', type=float, default=50.0,
                        help='[soft mode only] Student-t degrees of freedom '
                             '(larger = harder, 50 ≈ nearly hard exclusion)')
    parser.add_argument('--z_threshold', type=float, default=0.8,
                        help='[soft mode only] Minimum z weight for a detection to contribute; '
                             'detections below this are hard-excluded (default 0.8)')
    parser.add_argument('--member_sigma_clip', type=float, default=3.0,
                        help='Mahalanobis sigma threshold for membership selection (default 3.0)')
    parser.add_argument('--max_sigma_free_pm', type=float, default=1.0,
                        help='Maximum free-posterior RMS PM sigma (mas/yr) for membership '
                             'eligibility; stars dominated by diffuse prior (2p, few epochs) '
                             'are excluded (default 1.0)')
    parser.add_argument('--pm_sys_floor', type=float, default=0.2,
                        help='Systematic PM floor added in quadrature to per-star '
                             'PM uncertainty for membership radius (mas/yr)')
    parser.add_argument('--poly_order', type=int, default=None,
                        help='Polynomial order (default: read from BP3M_results/run_config.json)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip diagnostic plot generation')
    parser.add_argument('--fit_members_only', action='store_true',
                        help='When set, only identified members are used in the astrometric '
                             'fit from Phase 2 onwards; non-members are excluded from '
                             'use_for_fit each iteration')
    parser.add_argument('--mu_pop_init', type=float, nargs=2,
                        metavar=('PMRA', 'PMDEC'), default=None,
                        help='Initial μ_pop estimate in mas/yr (pmra pmdec). Seeds the '
                             'sigma-clip bootstrap as the starting centre. Combine with '
                             '--freeze_mu_pop_init to skip the bootstrap entirely. '
                             '(e.g. --mu_pop_init -0.06 -0.11)')
    parser.add_argument('--use_member_seed', action='store_true',
                        help='Use <field>/member_seed.csv (saved by notebook '
                             '07_member_selection) as the initial member selection.')
    parser.add_argument('--freeze_member_seed', action='store_true',
                        help='Like --use_member_seed, but membership refinement can only '
                             'REMOVE stars from the seed set — stars outside the seed can '
                             'never become members.')
    parser.add_argument('--member_seed_csv', type=str, default=None,
                        help='CSV with gaia_source_id (+ optional trusted bool) from the '
                             'interactive selection notebook (07_member_selection). '
                             'Replaces the sigma-clip initial member selection; the '
                             'fit phases still refine membership from there.')
    parser.add_argument('--freeze_mu_pop_init', action='store_true',
                        help='Skip the sigma-clip bootstrap and use --mu_pop_init directly '
                             'as the starting μ_pop and prior centre. Useful for sparse '
                             'fields where the bootstrap wanders from the literature value.')
    parser.add_argument('--fit_cte', action='store_true',
                        help='After pop-fit convergence, run a CTE phase: warm-start the '
                             'CTE model (γ only, r and μ_pop fixed), then jointly fit '
                             '(r, γ, μ_pop) with alpha and membership frozen.')
    parser.add_argument('--cte_mag_poly_order', type=int, default=3,
                        help='CTE magnitude polynomial order (default 3)')
    parser.add_argument('--cte_spatial_order', type=int, default=2,
                        help='CTE spatial polynomial order (default 2)')
    parser.add_argument('--cte_time_poly_order', type=int, default=0,
                        help='CTE time polynomial order (default 0; 1 adds constant-in-time block)')
    parser.add_argument('--cte_n_iter', type=int, default=10,
                        help='CTE joint loop iterations (default 10)')
    parser.add_argument('--lib_dir', type=str, default=None,
                        help='BP3M library directory (from bp3m-setup).  Used to run QSO '
                             'vetting on demand if {field}/Gaia/{field}_qso_anchors.csv '
                             'is not yet present.  Reads config.toml if omitted.')
    parser.add_argument('--no_qso_anchors', action='store_true',
                        help='Disable QSO anchor priors.  By default the solver loads '
                             '{field}/Gaia/{field}_*_qso_anchors.csv (produced at Phase 1) '
                             'and applies tight secular-aberration '
                             'PM + zero-parallax priors to vetted QSOs.')
    parser.add_argument('--qso_anchors_csv', type=str, default=None, nargs='+',
                        help='Explicit path(s) to qso_anchors CSV file(s).  When multiple '
                             'paths are given they are concatenated and deduped by source_id.  '
                             'Overrides the default glob in {field}/Gaia/.')

    args = parser.parse_args()

    # ── Resolve LVD-derived parameters ───────────────────────────────────────
    _sigma_pm         = args.sigma_pm          # None if not given by user
    _plx_pop          = args.plx_pop           # None if not given by user
    _sigma_plx_tot    = args.sigma_plx_tot     # None if not given by user
    _mu_pop_prior_sigma = args.mu_pop_prior_sigma  # None if not given by user
    _mu_pop_init      = (tuple(args.mu_pop_init) if args.mu_pop_init is not None
                         else None)
    _mu_init_src      = 'user' if args.mu_pop_init is not None else None

    if args.lvd_key is not None:
        import os
        _lvd_dir = (args.lvd_dir
                    or os.environ.get('BP3M_LVD_DIR')
                    or str(Path.home() / 'data_bootes' / 'bp3m'
                           / 'local_volume_database' / 'data'))
        try:
            _lvd = _lookup_lvd(Path(_lvd_dir), args.lvd_key)
            print(f"LVD lookup for key={args.lvd_key!r} (dir={_lvd_dir}):")
            if 'd_kpc' in _lvd:
                print(f"  distance   = {_lvd['d_kpc']:.2f} kpc")
            if 'plx_pop' in _lvd:
                print(f"  plx_pop    = {_lvd['plx_pop']:.4e} mas")
            if 'sigma_plx_tot' in _lvd:
                print(f"  σ_plx_tot  = {_lvd['sigma_plx_tot']:.4e} mas")
            if 'mu_pop_init' in _lvd:
                print(f"  μ_pop_init = {_lvd['mu_pop_init']} mas/yr")
            if 'mu_pop_prior_sigma' in _lvd:
                print(f"  μ_pop σ    = {_lvd['mu_pop_prior_sigma']:.4f} mas/yr")
            if 'sigma_pm' in _lvd:
                print(f"  σ_pm       = {_lvd['sigma_pm']:.4f} mas/yr  "
                      f"(from vlos_sigma / 4.7405 d)")
            # Only fill in values the user did not supply explicitly
            if _sigma_pm is None:
                _sigma_pm = _lvd.get('sigma_pm')
            if _plx_pop is None:
                _plx_pop = _lvd.get('plx_pop')
            if _sigma_plx_tot is None:
                _sigma_plx_tot = _lvd.get('sigma_plx_tot')
            if _mu_pop_init is None and 'mu_pop_init' in _lvd:
                _mu_pop_init = _lvd['mu_pop_init']
                _mu_init_src = 'LVD'
            if _mu_pop_prior_sigma is None and 'mu_pop_prior_sigma' in _lvd:
                _mu_pop_prior_sigma = _lvd['mu_pop_prior_sigma']
        except Exception as exc:
            print(f"WARNING: LVD lookup failed ({exc}); falling back to defaults.")

    # Apply fallback defaults for any still-unset parameters
    if _sigma_pm is None:
        _sigma_pm = 0.0075
    if _plx_pop is None:
        _plx_pop = 0.003873
    if _sigma_plx_tot is None:
        _sigma_plx_tot = 0.0001425
    if _mu_pop_prior_sigma is None:
        _mu_pop_prior_sigma = 0.5

    # Resolve lib_dir: CLI arg > config.toml
    _lib_dir_arg = args.lib_dir
    if _lib_dir_arg is None:
        try:
            import re, os
            _cfg = Path(os.environ.get('BP3M_HOME', Path.home() / '.bp3m')) / 'config.toml'
            if _cfg.exists():
                _m = re.search(r'^lib_dir\s*=\s*["\']([^"\']+)["\']',
                               _cfg.read_text(), re.MULTILINE)
                if _m:
                    _lib_dir_arg = _m.group(1)
        except Exception:
            pass

    run_pop_fit(
        output_dir=Path(args.output_dir).resolve(),
        field_name=args.name.replace(' ', '_'),
        sigma_pm=_sigma_pm,
        plx_pop=_plx_pop,
        sigma_plx_tot=_sigma_plx_tot,
        mu_pop_prior_sigma=_mu_pop_prior_sigma,
        n_iter_mu=args.n_iter_mu,
        n_iter_joint=args.n_iter_joint,
        n_iter_alpha=args.n_iter_alpha,
        alpha_damp=args.alpha_damp,
        n_iter_phase4=args.n_iter_phase_4,
        phase4_mode=args.phase4_mode,
        student_t_nu=args.student_t_nu,
        z_threshold=args.z_threshold,
        member_sigma_clip=args.member_sigma_clip,
        pm_sys_floor=args.pm_sys_floor,
        max_sigma_free_pm=args.max_sigma_free_pm,
        fit_members_only=args.fit_members_only,
        mu_pop_init=_mu_pop_init,
        mu_pop_init_source=_mu_init_src,
        member_seed_csv=args.member_seed_csv,
        use_member_seed=args.use_member_seed,
        freeze_member_seed=args.freeze_member_seed,
        freeze_mu_pop_init=args.freeze_mu_pop_init,
        poly_order=args.poly_order,
        no_plots=args.no_plots,
        fit_cte=args.fit_cte,
        cte_mag_poly_order=args.cte_mag_poly_order,
        cte_spatial_order=args.cte_spatial_order,
        cte_time_poly_order=args.cte_time_poly_order,
        cte_n_iter=args.cte_n_iter,
        lib_dir=Path(_lib_dir_arg) if _lib_dir_arg else None,
        use_qso_anchors=not args.no_qso_anchors,
        qso_anchors_csv=[Path(p) for p in args.qso_anchors_csv] if args.qso_anchors_csv else None,
    )

    # Save the command only on successful completion so interrupted runs
    # do not overwrite the record of the last successful invocation.
    import sys as _sys, shlex as _shlex
    from datetime import datetime as _datetime
    _cmd_file = (Path(args.output_dir).resolve() / args.name.replace(' ', '_')
                 / 'BP3M_pop_fit_results' / 'bp3m_pop_fit_command.txt')
    _cmd_file.parent.mkdir(parents=True, exist_ok=True)
    _cmd_file.write_text(
        f"# {_datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + ' '.join(_shlex.quote(a) for a in _sys.argv) + '\n'
    )
