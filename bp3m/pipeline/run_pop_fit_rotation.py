"""
Population PM fitting with a disk rotation model — NGC 55 and NGC 300 only.

Extends bp3m-pop-fit by replacing the isotropic σ_pm prior with a
position-dependent rotation prediction for each candidate member star.

Rotation model
--------------
For a star at sky offset (Δα cosδ, Δδ) from the galaxy centre:

    ξ  =  Δα cosδ · sin(PA + θ) + Δδ · cos(PA + θ)   [along major axis, arcsec]
    η  =  Δα cosδ · cos(PA + θ) − Δδ · sin(PA + θ)   [along minor axis, arcsec]

    X  =  ξ · (d_kpc · 4.84814e-6)                   [deproject, kpc]
    Y  = (η / cos i) · (d_kpc · 4.84814e-6)           [deproject, kpc]
    R  =  sqrt(X² + Y²)                               [deprojected radius, kpc]
    φ  =  atan2(Y, X)                                  [azimuthal angle in disk]

    v_ξ  = −f · V_rot(R) · sin φ                       [km/s, along major axis]
    v_η  = +f · V_rot(R) · cos φ · cos i               [km/s, along minor axis]

    Δμ_ra*  = (v_ξ · sin(PA+θ) + v_η · cos(PA+θ)) / (d_kpc · 4.74047)   [mas/yr]
    Δμ_dec  = (v_ξ · cos(PA+θ) − v_η · sin(PA+θ)) / (d_kpc · 4.74047)   [mas/yr]

Free parameters (fitted alongside μ_pop)
-----------------------------------------
  f           : stellar-to-HI rotation speed ratio (asymmetric drift).
                Prior: N(f0, σ_f²).  f0=1 for no prior correction; σ_f≈0.2.
  theta_offset: kinematic PA offset from literature photometric PA (radians).
                Prior: N(0, σ_θ²).  σ_θ ≈ deg2rad(10°).

Galaxy parameters (hardcoded, literature values)
-------------------------------------------------
NGC 55:
  Centre      : (3.7233, −39.1967) deg
  Distance    : 1932 kpc  (DM=26.43, σ_DM=0.12)
  PA          : 108 deg   (receding major axis, N through E)
  Inclination : 84 deg
  V_rot       : flat at 90.6 km/s beyond R_turn = 1.0 kpc
  f0 prior    : 1.0 ± 0.2   (σ_f)
  θ prior     : 0.0 ± 10°   (σ_θ in deg)

NGC 300:
  Centre      : (13.7229, −37.6844) deg
  Distance    : 2089 kpc  (DM=26.60, σ_DM=0.06)
  PA          : 290 deg   (receding major axis)
  Inclination : 42 deg
  V_rot       : flat at 90.0 km/s beyond R_turn = 1.5 kpc
  f0 prior    : 1.0 ± 0.2
  θ prior     : 0.0 ± 10°

References
----------
NGC 55  HI kinematics: Puche et al. 1991; inclination: Westmeier et al. 2013
NGC 300 HI kinematics: Puche et al. 1990; inclination: Westmeier et al. 2011
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Galaxy parameter table ─────────────────────────────────────────────────────

_DEG2RAD = np.pi / 180.0
_ARCSEC2KPC = 4.84814e-6   # 1 arcsec at 1 kpc = 4.84814e-6 kpc

# Tilted-ring models from HI kinematic studies.
# Columns: projected ring radius (arcsec), V_rot (km/s), PA of receding major axis (deg), i (deg).

# NGC 55 — Westmeier et al. 2013, Table 4
_NGC55_TRING = np.array([
    [150,   18.7, 110.3, 84.5],
    [250,   37.0, 109.9, 81.0],
    [350,   47.5, 109.8, 78.1],
    [450,   59.2, 110.4, 76.4],
    [550,   73.1, 109.7, 75.4],
    [650,   80.6, 109.5, 75.1],
    [750,   87.4, 109.4, 73.9],
    [850,   90.6, 109.3, 73.1],
    [950,   89.1, 108.7, 71.6],
    [1050,  84.2, 107.5, 69.7],
    [1150,  80.1, 104.9, 68.1],
    [1250,  80.5, 102.0, 67.1],
    [1350,  82.5,  99.5, 67.0],
    [1450,  79.7,  97.4, 67.5],
    [1550,  73.1,  97.5, 67.3],
    [1650,  70.5,  95.8, 67.1],
    [1750,  67.9,  95.6, 66.5],
    [1850,  69.5,  93.6, 66.3],
    [1950,  69.7,  93.4, 66.5],
])

# NGC 300 — Westmeier et al. 2011, Table 2
_NGC300_TRING = np.array([
    [100,   43.3, 290.6, 39.9],
    [200,   66.5, 289.3, 40.5],
    [300,   75.4, 289.5, 42.6],
    [400,   80.3, 290.2, 44.6],
    [500,   83.5, 289.9, 45.8],
    [600,   88.4, 290.5, 46.5],
    [700,   92.2, 293.2, 49.0],
    [800,   95.7, 297.8, 51.0],
    [900,   96.8, 304.4, 51.3],
    [1000,  98.8, 311.1, 49.9],
    [1100,  98.3, 316.4, 49.3],
    [1200,  95.4, 319.4, 49.7],
    [1300,  92.8, 321.6, 49.9],
    [1400,  90.7, 324.2, 48.9],
    [1500,  89.1, 327.1, 47.0],
    [1600,  87.8, 329.6, 46.6],
    [1700,  88.4, 331.4, 45.3],
    [1800,  89.1, 331.9, 44.3],
    [1900,  87.3, 332.0, 42.7],
    [2000,  82.7, 331.7, 43.3],
])

GALAXY_PARAMS: dict[str, dict] = {
    'NGC_55': dict(
        ra_cen=3.7246, dec_cen=-39.1964,   # Westmeier+2013 Table 1 kinematic centre
        d_kpc=1932.0,
        plx_pop=5.176e-4,                        # mas
        sigma_plx_tot=2.86e-5,                   # mas
        # Representative inner-disk values (for display / mu_pop.json only)
        pa_deg=110.3,                            # PA of receding major axis at innermost ring
        inc_deg=84.5,
        sigma_pm_disp=0.001,                     # residual PM dispersion (mas/yr, after rotation removed)
        f0=1.0, sigma_f=0.20,                    # prior on f_star_mult
        sigma_theta_deg=10.0,                    # prior on theta_offset (deg)
        mu_pop_init=(-0.0044, -0.0023),
        tilted_ring=_NGC55_TRING,                # full tilted-ring model; overrides pa_deg/inc_deg
    ),
    'NGC_300': dict(
        ra_cen=13.7229, dec_cen=-37.6844,  # Westmeier+2011 Table 1 kinematic centre
        d_kpc=2089.0,
        plx_pop=4.786e-4,
        sigma_plx_tot=1.323e-5,
        pa_deg=290.6,
        inc_deg=39.9,
        sigma_pm_disp=0.001,
        f0=1.0, sigma_f=0.20,
        sigma_theta_deg=10.0,
        mu_pop_init=(-0.0042, -0.0027),
        tilted_ring=_NGC300_TRING,
    ),
}


# ── Rotation model geometry ────────────────────────────────────────────────────
#
# References
# ----------
# NGC 55 tilted-ring model:
#   Westmeier, T., Brüns, C., & Kerp, J. 2013, MNRAS, 432, 3047 (Table 4)
# NGC 300 tilted-ring model:
#   Westmeier, T., Koribalski, B. S., & Braun, R. 2011, MNRAS, 410, 2217 (Table 2)
#
# Kinematic model geometry:
#   van der Hulst, J. M., et al. 1992, AJ, 103, 1457
#   Begeman, K. G. 1989, A&A, 223, 47
#
# The projected angular radius of each star is used to interpolate V_rot(ϑ),
# PA(ϑ), and i(ϑ) from the HI tilted-ring table.  Each star is then deprojected
# using its local ring parameters to obtain the azimuthal angle φ, and the
# expected transverse velocity is projected back onto the sky:
#
#   v_ξ  = −V_rot · sin φ            [along deprojected major axis, km/s]
#   v_η  = +V_rot · cos φ · cos i    [along minor axis on sky, km/s]
#   Δμ_ra*  = (v_ξ sin PA + v_η cos PA) / (d_kpc · κ)
#   Δμ_dec  = (v_ξ cos PA − v_η sin PA) / (d_kpc · κ)
#   κ = 4.74047  km/s per (mas/yr · kpc)


def compute_rotation_offsets(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    gp: dict,
    f: float = 1.0,
    theta_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-star rotation PM offsets (Δμ_ra*, Δμ_dec) in mas/yr.

    Uses the tilted-ring model stored in gp['tilted_ring'] (Westmeier et al.):
    V_rot, PA, and inclination are linearly interpolated at each star's projected
    angular radius.  theta_offset (radians) is added to every ring's PA as a
    global kinematic offset free parameter.

    Parameters
    ----------
    ra_deg, dec_deg : star sky positions (degrees)
    gp              : galaxy parameter dict from GALAXY_PARAMS
    f               : stellar/HI rotation speed ratio (asymmetric-drift factor)
    theta_offset    : global kinematic PA offset added to every ring's PA (radians)

    Returns
    -------
    dmu_ra, dmu_dec : (n_stars,) arrays in mas/yr
    """
    d_kpc   = gp['d_kpc']
    kappa   = 4.74047   # km/s per (mas/yr · kpc)

    # Sky offsets in arcsec (East positive, North positive)
    cos_dec = np.cos(gp['dec_cen'] * _DEG2RAD)
    x_as    = (ra_deg  - gp['ra_cen']) * cos_dec * 3600.0
    y_as    = (dec_deg - gp['dec_cen'])            * 3600.0

    tring = gp.get('tilted_ring')
    if tring is not None:
        # Projected angular radius for each star
        r_as = np.sqrt(x_as**2 + y_as**2)

        # Interpolate ring parameters at each star's projected radius.
        # np.interp clamps to first/last table value outside the range.
        r_tbl    = tring[:, 0]
        vrot_r   = np.interp(r_as, r_tbl, tring[:, 1])   # km/s
        pa_r_rad = np.interp(r_as, r_tbl, tring[:, 2]) * _DEG2RAD + theta_offset
        inc_r    = np.interp(r_as, r_tbl, tring[:, 3])   # degrees
        cos_i    = np.cos(inc_r * _DEG2RAD)               # per-star array

        V_rot = f * vrot_r

    else:
        # Fallback: single flat values from gp (no tilted-ring table)
        pa_r_rad = gp['pa_deg'] * _DEG2RAD + theta_offset
        cos_i    = np.cos(gp['inc_deg'] * _DEG2RAD)
        # Arctangent rotation curve (Courteau 1997) requires deprojected R
        # — computed below after xi/eta
        V_rot    = None  # filled after deprojection

    # Project sky offsets onto local major / minor axis
    xi  =  x_as * np.sin(pa_r_rad) + y_as * np.cos(pa_r_rad)
    eta =  x_as * np.cos(pa_r_rad) - y_as * np.sin(pa_r_rad)

    # Deproject η → disk Y (guard against edge-on singularity)
    cos_i_safe = np.where(np.abs(cos_i) > 0.05, cos_i,
                          0.05 * np.sign(np.where(cos_i >= 0, 1.0, -1.0)))
    scale = d_kpc * _ARCSEC2KPC
    X     = xi  * scale
    Y     = eta / cos_i_safe * scale

    phi = np.arctan2(Y, X)

    if V_rot is None:
        R     = np.sqrt(X**2 + Y**2)
        v_flat = gp['v_rot_flat']
        r_turn = gp['r_turn_kpc']
        V_rot  = f * v_flat * (2.0 / np.pi) * np.arctan(R / max(r_turn, 1e-6))

    # Transverse sky velocity components
    v_xi  = -V_rot * np.sin(phi)
    v_eta =  V_rot * np.cos(phi) * cos_i

    v_east  = v_xi * np.sin(pa_r_rad) + v_eta * np.cos(pa_r_rad)
    v_north = v_xi * np.cos(pa_r_rad) - v_eta * np.sin(pa_r_rad)

    dmu_ra  = v_east  / (d_kpc * kappa)
    dmu_dec = v_north / (d_kpc * kappa)

    return dmu_ra, dmu_dec


def compute_rotation_offsets_jacobian(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    gp: dict,
    f: float,
    theta_offset: float,
    eps_f: float = 1e-4,
    eps_theta: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerical Jacobian of (dmu_ra, dmu_dec) w.r.t. (f, theta_offset).

    Returns
    -------
    ddmu_ra_df, ddmu_dec_df, ddmu_ra_dtheta, ddmu_dec_dtheta
    Each shape (n_stars,).
    """
    r0, d0 = compute_rotation_offsets(ra_deg, dec_deg, gp, f, theta_offset)
    rf, df = compute_rotation_offsets(ra_deg, dec_deg, gp, f + eps_f, theta_offset)
    rt, dt = compute_rotation_offsets(ra_deg, dec_deg, gp, f, theta_offset + eps_theta)

    return (rf - r0) / eps_f, (df - d0) / eps_f, (rt - r0) / eps_theta, (dt - d0) / eps_theta


# ── Modified _joint_solve_pop with rotation model ─────────────────────────────
#
# Copy of run_pop_fit._joint_solve_pop with two targeted changes:
#
#  1. Per-star member prior RHS:
#       h_align/h_all[s, 2/3] += σ^{-2} · (μ_pop[0/1] + rot_ra/dec[s])
#     (base code has σ^{-2} · μ_pop[0/1] only).
#
#  2. Schur RHS correction for μ_pop:
#       rhs_mu += σ^{-2} · Σ a_s[2:4]        (same as base)
#       rhs_mu -= σ^{-2} · Σ rot_ra/dec[s]   (NEW: rotation contribution)
#
# This is exact for fixed rotation (fit_f=False, fit_theta=False).
# The 4D extension for free f/θ is deferred.

def _joint_solve_pop_rot(
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
    rot_ra: "np.ndarray | None" = None,
    rot_dec: "np.ndarray | None" = None,
    fit_f: bool = False,
    fit_theta: bool = False,
    f_current: float = 1.0,
    theta_current: float = 0.0,
    gp: "dict | None" = None,
    gaia_ra: "np.ndarray | None" = None,
    gaia_dec: "np.ndarray | None" = None,
) -> tuple:
    """
    One Newton step for (Δr, Δμ_pop) with per-star rotation prior.

    Per-star member prior centre: (μ_pop[0] + rot_ra[s], μ_pop[1] + rot_dec[s]).
    When fit_f or fit_theta, jointly solves for f/θ alongside μ (fix_r=True only).

    Returns 9-tuple: (r_hat, mu_pop_hat, f_new, theta_new, C_shared, C_vT, a, a_align, K_img).
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(x, **kw):
            return x

    _rot_ra  = rot_ra  if rot_ra  is not None else np.zeros(solver.n_stars)
    _rot_dec = rot_dec if rot_dec is not None else np.zeros(solver.n_stars)

    n_free = int(fit_f) + int(fit_theta)
    assert fix_r or n_free == 0, "f/θ fitting requires fix_r=True"

    # Jacobians for f/θ (only needed if n_free > 0)
    _jfr = _jfd = _jtr = _jtd = None
    if n_free > 0 and gp is not None and gaia_ra is not None:
        _jfr, _jfd, _jtr, _jtd = compute_rotation_offsets_jacobian(
            gaia_ra, gaia_dec, gp, f_current, theta_current)

    # Indices for extended params
    _idx_f = 2 if fit_f else None
    _idx_t = (3 if fit_f else 2) if fit_theta else None

    N_V   = 5
    nr    = solver.N_R
    n_r   = len(image_names) * nr
    n_stars = solver.n_stars

    sigma_pm_inv_sq  = sigma_pm ** -2
    sigma_plx_inv_sq = sigma_plx_tot ** -2

    if fix_r:
        n_shared = 2 + n_free
    else:
        n_shared = n_r + 2
        idx_r  = slice(0, n_r)
        idx_mu = slice(n_r, n_r + 2)

    # ── H_vv: start from Gaia prior ───────────────────────────────────────────
    H_vv = solver.C_survey_inv.copy()

    _nonmem = np.ones(n_stars, dtype=bool)
    _nonmem[member_sidx] = False
    _nonmem_2p = _nonmem & (solver._C_VG_inv_per_star[:, 2] > 0)
    if _nonmem_2p.any():
        for _k in range(N_V):
            H_vv[_nonmem_2p, _k, _k] += solver._C_VG_inv_per_star[_nonmem_2p, _k]

    H_vv[member_sidx, 2, 2] += sigma_pm_inv_sq
    H_vv[member_sidx, 3, 3] += sigma_pm_inv_sq
    H_vv[member_sidx, 4, 4] += sigma_plx_inv_sq

    h_align = solver.C_survey_inv_dot_v.copy()
    h_all   = solver.C_survey_inv_dot_v.copy()

    # ── Population prior RHS — rotation-corrected ────────────────────────────
    h_align[member_sidx, 2] += sigma_pm_inv_sq * (mu_pop_current[0] + _rot_ra[member_sidx])
    h_align[member_sidx, 3] += sigma_pm_inv_sq * (mu_pop_current[1] + _rot_dec[member_sidx])
    h_all  [member_sidx, 2] += sigma_pm_inv_sq * (mu_pop_current[0] + _rot_ra[member_sidx])
    h_all  [member_sidx, 3] += sigma_pm_inv_sq * (mu_pop_current[1] + _rot_dec[member_sidx])
    h_align[member_sidx, 4] += sigma_plx_inv_sq * plx_pop
    h_all  [member_sidx, 4] += sigma_plx_inv_sq * plx_pop

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

    # ── Per-image accumulation ─────────────────────────────────────────────────
    K_img       = {}
    XCs_xresid  = {}
    H_rr_block  = np.zeros((n_r, n_r))
    active_glob = np.zeros(n_stars, dtype=bool)

    for j_idx, img in enumerate(_tqdm(image_names, desc='  pop_rot_solve',
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
        xys = d['xys']

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
        Lambda[:2, :2] = H_mu

    # ── Schur correction for μ block ──────────────────────────────────────────
    if n_mem > 0:
        Cv_m = C_vT[member_sidx]
        mu_mu_schur = sigma_pm_inv_sq ** 2 * Cv_m[:, 2:4, 2:4].sum(axis=0)
        if fix_r:
            Lambda[:2, :2] -= mu_mu_schur
        else:
            Lambda[idx_mu, idx_mu] -= mu_mu_schur
        # Schur RHS: posterior star means contribute, with rotation offset removed
        rhs_mu += sigma_pm_inv_sq * a[member_sidx, 2:4].sum(axis=0)
        rhs_mu -= sigma_pm_inv_sq * np.array([
            _rot_ra[member_sidx].sum(),
            _rot_dec[member_sidx].sum(),
        ])

    if fix_r:
        rhs[:2] = rhs_mu
    else:
        rhs[idx_mu] = rhs_mu

    # ── Extended block (f, θ) — fix_r only ───────────────────────────────────
    if n_free > 0 and n_mem > 0 and _jfr is not None:
        jacs = []
        ext_indices = []
        prior_inv_sqs = []
        prior_centers = []
        param_vals = []

        if fit_f:
            jacs.append(np.column_stack([_jfr[member_sidx], _jfd[member_sidx]]))
            ext_indices.append(_idx_f)
            _sf   = gp.get('sigma_f', 0.2) if gp else 0.2
            _f0   = gp.get('f0', 1.0)      if gp else 1.0
            prior_inv_sqs.append(_sf ** -2)
            prior_centers.append(_f0)
            param_vals.append(f_current)

        if fit_theta:
            jacs.append(np.column_stack([_jtr[member_sidx], _jtd[member_sidx]]))
            ext_indices.append(_idx_t)
            _st = (gp['sigma_theta_deg'] * _DEG2RAD) if gp else (10.0 * _DEG2RAD)
            prior_inv_sqs.append(_st ** -2)
            prior_centers.append(0.0)
            param_vals.append(theta_current)

        Cv_m_pm = C_vT[member_sidx, 2:4, 2:4]
        a_pm_m  = a[member_sidx, 2:4]
        rot_m   = np.column_stack([_rot_ra[member_sidx], _rot_dec[member_sidx]])
        resid_m = a_pm_m - mu_pop_current[None, :] - rot_m

        for ii, (i_idx, jac_i, p_inv_sq, p_ctr, p_val) in enumerate(
                zip(ext_indices, jacs, prior_inv_sqs, prior_centers, param_vals)):
            data_ii  = sigma_pm_inv_sq    * np.einsum('ni,ni->',  jac_i, jac_i)
            schur_ii = sigma_pm_inv_sq**2 * np.einsum('ni,nij,nj->', jac_i, Cv_m_pm, jac_i)
            Lambda[i_idx, i_idx] += p_inv_sq + data_ii - schur_ii

            data_mu_i  = sigma_pm_inv_sq    * jac_i.sum(axis=0)
            schur_mu_i = sigma_pm_inv_sq**2 * np.einsum('nij,nj->i', Cv_m_pm, jac_i)
            Lambda[0, i_idx] += data_mu_i[0] - schur_mu_i[0]
            Lambda[1, i_idx] += data_mu_i[1] - schur_mu_i[1]
            Lambda[i_idx, 0]  = Lambda[0, i_idx]
            Lambda[i_idx, 1]  = Lambda[1, i_idx]

            rhs[i_idx] = (sigma_pm_inv_sq * np.einsum('ni,ni->', jac_i, resid_m)
                          - p_inv_sq * (p_val - p_ctr))

        if n_free == 2:
            jac_0, jac_1 = jacs
            i0, i1 = ext_indices
            data_01  = sigma_pm_inv_sq    * np.einsum('ni,ni->',  jac_0, jac_1)
            schur_01 = sigma_pm_inv_sq**2 * np.einsum('ni,nij,nj->', jac_0, Cv_m_pm, jac_1)
            Lambda[i0, i1] += data_01 - schur_01
            Lambda[i1, i0]  = Lambda[i0, i1]

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

            if use_fmem.any():
                sidx_fm  = sidx[use_fmem]
                K_fm     = K_img[img][use_fmem]
                CvT_M_fm = C_vT[sidx_fm, :, 2:4]
                KT_CvT_M = np.einsum('nji,njk->ik', K_fm, CvT_M_fm)
                Lambda[cs:cs + nr, idx_mu] -= sigma_pm_inv_sq * KT_CvT_M
                Lambda[idx_mu, cs:cs + nr] -= sigma_pm_inv_sq * KT_CvT_M.T

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
        f_new     = f_current     + delta[_idx_f] if (fit_f     and _idx_f is not None) else f_current
        theta_new = theta_current + delta[_idx_t] if (fit_theta and _idx_t is not None) else theta_current
        return r_current.copy(), mu_pop_current + delta[:2], f_new, theta_new, C_shared, C_vT, a, a_align, K_img
    else:
        return (r_current + delta[idx_r],
                mu_pop_current + delta[idx_mu],
                f_current, theta_current,
                C_shared, C_vT, a, a_align, K_img)


# ── Modified _select_members_from_a with rotation-corrected distance ──────────

def _select_members_from_a_rot(
    a_arr: np.ndarray,
    mu_pop: np.ndarray,
    n_hst: np.ndarray,
    C_vT: np.ndarray,
    sigma_pm: float,
    rot_ra: np.ndarray,
    rot_dec: np.ndarray,
    sigma_clip: float = 3.0,
    min_members: int = 5,
    max_sigma_free_pm: float = 1.0,
    pm_sys_floor: float = 0.0,
) -> np.ndarray:
    """
    Select members by Mahalanobis distance from (μ_pop + rot_offset[star]).

    Identical to _select_members_from_a except delta_pm uses the
    per-star rotation-corrected expected PM.
    """
    eidx = np.where(n_hst >= 1)[0]
    if len(eidx) < min_members:
        return eidx

    _has_valid = (C_vT[eidx, 2, 2] > 0) | (C_vT[eidx, 3, 3] > 0)
    eidx = eidx[_has_valid]
    if len(eidx) < min_members:
        return eidx

    pmra  = a_arr[eidx, 2]
    pmdec = a_arr[eidx, 3]

    mu_exp_ra  = mu_pop[0] + rot_ra[eidx]
    mu_exp_dec = mu_pop[1] + rot_dec[eidx]

    C_pm_sub = C_vT[eidx, 2:4, 2:4].copy()
    C_pm_sub[:, 0, 0] += sigma_pm ** 2 + pm_sys_floor ** 2
    C_pm_sub[:, 1, 1] += sigma_pm ** 2 + pm_sys_floor ** 2
    delta_pm = np.column_stack([pmra - mu_exp_ra, pmdec - mu_exp_dec])
    chi2 = np.einsum('ni,nij,nj->n', delta_pm, np.linalg.inv(C_pm_sub), delta_pm)

    sig_free = np.sqrt(np.maximum(
        (C_vT[eidx, 2, 2] + C_vT[eidx, 3, 3]) / 2, 0))
    well_constrained = sig_free < max_sigma_free_pm

    keep = np.isfinite(chi2) & (chi2 < sigma_clip ** 2) & well_constrained
    if keep.sum() < min_members:
        keep = np.isfinite(chi2) & well_constrained
    return eidx[keep]


# ── Modified _compute_free_stellar_posterior ──────────────────────────────────

def _compute_free_stellar_posterior_rot(
    a: np.ndarray,
    C_vT: np.ndarray,
    member_sidx: np.ndarray,
    sigma_pm: float,
    sigma_plx_tot: float,
    mu_pop: np.ndarray,
    plx_pop: float,
    C_VG_inv_per_star: np.ndarray,
    rot_ra: np.ndarray,
    rot_dec: np.ndarray,
) -> tuple:
    """
    Remove the rotation-aware population prior to get the free stellar posterior.

    Identical to _compute_free_stellar_posterior except the stripped prior
    centre is (μ_pop[0] + rot_ra[star], μ_pop[1] + rot_dec[star]).
    """
    if len(member_sidx) == 0:
        return a, C_vT

    _has_cvT = np.diagonal(C_vT[member_sidx], axis1=1, axis2=2).any(axis=1)
    member_sidx = member_sidx[_has_cvT]
    if len(member_sidx) == 0:
        return a, C_vT

    sigma_pm_inv_sq  = sigma_pm ** -2
    sigma_plx_inv_sq = sigma_plx_tot ** -2

    a_free    = a.copy()
    C_vT_free = C_vT.copy()

    H_mem = np.linalg.inv(C_vT[member_sidx])
    h_mem = np.einsum('nij,nj->ni', H_mem, a[member_sidx])

    H_mem[:, 2, 2] -= sigma_pm_inv_sq
    H_mem[:, 3, 3] -= sigma_pm_inv_sq
    H_mem[:, 4, 4] -= sigma_plx_inv_sq

    h_mem[:, 2] -= sigma_pm_inv_sq * (mu_pop[0] + rot_ra[member_sidx])
    h_mem[:, 3] -= sigma_pm_inv_sq * (mu_pop[1] + rot_dec[member_sidx])
    h_mem[:, 4] -= sigma_plx_inv_sq * plx_pop

    needs_diffuse = C_VG_inv_per_star[member_sidx, 2] > 0
    if needs_diffuse.any():
        ndx = member_sidx[needs_diffuse]
        for _k in range(5):
            H_mem[needs_diffuse, _k, _k] += C_VG_inv_per_star[ndx, _k]

    C_vT_free[member_sidx] = np.linalg.inv(H_mem)
    a_free[member_sidx]    = np.einsum('nij,nj->ni', C_vT_free[member_sidx], h_mem)

    return a_free, C_vT_free


# ── Sky PM plot ────────────────────────────────────────────────────────────────

def _plot_sky_pm_members(output_dir, stellar_csv_path: "Path", mu_pop: np.ndarray,
                         field_name: str, gp: dict) -> None:
    """
    3×2 figure: member stars' sky positions coloured by μ_ra* (left) and
    μ_dec (right) for three rows:
      row 0 — pop-prior posteriors  (pmra_bp3m / pmdec_bp3m)
      row 1 — diffuse-prior posteriors (pmra_bp3m_free / pmdec_bp3m_free)
      row 2 — gas-predicted PMs from tilted-ring model on a dense sky grid
               (μ_pop + rot_offset evaluated at every grid point)

    Colour limits are computed from the data rows (rows 0 & 1) and reused
    for the gas row so all three share the same scale.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(stellar_csv_path)
    mem = df[df['is_member'].astype(bool)].copy()
    if len(mem) == 0:
        return

    ra_m  = mem['ra'].to_numpy(float)
    dec_m = mem['dec'].to_numpy(float)

    pmra_pop   = mem['pmra_bp3m'].to_numpy(float)
    pmdec_pop  = mem['pmdec_bp3m'].to_numpy(float)
    pmra_free  = mem['pmra_bp3m_free'].to_numpy(float)
    pmdec_free = mem['pmdec_bp3m_free'].to_numpy(float)

    # Rows 0 & 2 share limits from pop-prior data (narrow, comparable to gas model).
    # Row 1 (diffuse prior) gets its own wider limits.
    spread_ra_pop  = max(np.nanpercentile(np.abs(pmra_pop  - mu_pop[0]), 95), 0.001)
    spread_dec_pop = max(np.nanpercentile(np.abs(pmdec_pop - mu_pop[1]), 95), 0.001)
    spread_ra_free  = max(np.nanpercentile(np.abs(pmra_free  - mu_pop[0]), 95), 0.001)
    spread_dec_free = max(np.nanpercentile(np.abs(pmdec_free - mu_pop[1]), 95), 0.001)
    limits_pop  = [(mu_pop[0] - spread_ra_pop,  mu_pop[0] + spread_ra_pop),
                   (mu_pop[1] - spread_dec_pop, mu_pop[1] + spread_dec_pop)]
    limits_free = [(mu_pop[0] - spread_ra_free,  mu_pop[0] + spread_ra_free),
                   (mu_pop[1] - spread_dec_free, mu_pop[1] + spread_dec_free)]

    # Dense sky grid covering the field footprint
    pad = 0.02  # deg
    ra_lo,  ra_hi  = ra_m.min()  - pad, ra_m.max()  + pad
    dec_lo, dec_hi = dec_m.min() - pad, dec_m.max() + pad
    N_grid = 300
    ra_vec  = np.linspace(ra_lo,  ra_hi,  N_grid)
    dec_vec = np.linspace(dec_lo, dec_hi, N_grid)
    ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)
    dmu_ra_grid, dmu_dec_grid = compute_rotation_offsets(
        ra_grid.ravel(), dec_grid.ravel(), gp)
    pmra_gas_grid  = (mu_pop[0] + dmu_ra_grid ).reshape(N_grid, N_grid)
    pmdec_gas_grid = (mu_pop[1] + dmu_dec_grid).reshape(N_grid, N_grid)

    # row_idx, label, pmra, pmdec, limits
    rows = [
        (0, 'pop prior',         pmra_pop,       pmdec_pop,       limits_pop),
        (1, 'diffuse prior',     pmra_free,       pmdec_free,      limits_free),
        (2, 'gas (tilted-ring)', pmra_gas_grid,  pmdec_gas_grid,  limits_pop),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    fig.suptitle(f'{field_name} — member PMs on sky (N={len(mem)})', fontsize=13)

    for row_idx, row_label, pmra_vals, pmdec_vals, lims in rows:
        for col, (pm_vals, pm_label, (vmin, vmax)) in enumerate([
            (pmra_vals,  r'$\mu_{\alpha^*}$ (mas/yr)', lims[0]),
            (pmdec_vals, r'$\mu_\delta$ (mas/yr)',     lims[1]),
        ]):
            ax = axes[row_idx, col]
            if row_idx == 2:
                sc = ax.pcolormesh(ra_vec, dec_vec, pm_vals, cmap='RdBu_r',
                                   vmin=vmin, vmax=vmax, rasterized=True)
            else:
                sc = ax.scatter(ra_m, dec_m, c=pm_vals, s=6, cmap='RdBu_r',
                                vmin=vmin, vmax=vmax, rasterized=True)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label=pm_label)
            ax.set_xlabel('RA (deg)')
            ax.set_ylabel('Dec (deg)')
            ax.set_title(f'{row_label}  —  {pm_label}')
            ax.invert_xaxis()

    out_path = output_dir / 'sky_pm_members.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: sky_pm_members.png")


# ── Main entry point ───────────────────────────────────────────────────────────

def run_pop_fit_rotation(
    output_dir: Path,
    field_name: str,
    sigma_pm: float | None = None,
    mu_pop_prior_sigma: float = 0.5,
    n_iter_mu: int = 20,
    n_iter_joint: int = 20,
    n_iter_alpha: int = 20,
    alpha_damp: float = 0.5,
    member_sigma_clip: float = 3.0,
    pm_sys_floor: float = 0.25,
    max_sigma_free_pm: float = 3.0,
    fit_f: bool = True,
    fit_theta: bool = True,
    f_init: float = 1.0,
    n_iter_ft: int = 20,
    mu_pop_init: tuple[float, float] | None = None,
    no_plots: bool = False,
    qso_anchors_csv: "Path | str | list | None" = None,
) -> Path:
    """
    Run rotation-model pop-fit for NGC_55 or NGC_300.

    Uses hardcoded galaxy parameters from GALAXY_PARAMS.
    For the first tests use fit_f=False, fit_theta=False (fixed rotation at
    f=1.0, theta=0.0).  Full f/θ fitting is a deferred extension.
    """
    from bp3m.data_loader_flc import load_image_data_flc
    from bp3m.data_loader_flc import build_index_maps, split_images_by_ccd
    from bp3m.solver import BP3MSolver
    from bp3m.pipeline.run_pop_fit import (
        _load_bp3m_outputs, _apply_bp3m_flags,
        _select_initial_members, _estimate_mu_pop_v1, _estimate_mu_pop,
        _compute_alpha_updates,
    )
    from bp3m.pipeline.qso_vetting import find_qso_anchors

    gp = GALAXY_PARAMS.get(field_name)
    if gp is None:
        raise ValueError(
            f"Unknown field '{field_name}'. "
            f"run_pop_fit_rotation only supports: {list(GALAXY_PARAMS)}"
        )

    if sigma_pm is None:
        sigma_pm = gp['sigma_pm_disp']
    if mu_pop_init is None:
        mu_pop_init = gp['mu_pop_init']

    plx_pop       = gp['plx_pop']
    sigma_plx_tot = gp['sigma_plx_tot']

    t_start    = time.time()
    data_root  = Path(output_dir)
    bp3m_dir   = data_root / field_name / 'BP3M_results'
    output_pfr = data_root / field_name / 'BP3M_pop_fit_rotation_results'
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
    poly_order          = int(v1_cfg.get('poly_order', 1))

    print("\n" + "─" * 60)
    print("BP3M pop-fit (rotation model)")
    print("─" * 60)
    print(f"  field={field_name}")
    _tring = gp.get('tilted_ring')
    _rot_desc = (f"tilted-ring ({len(_tring)} rings, "
                 f"Vrot={_tring[0,1]:.0f}–{_tring[-1,1]:.0f} km/s)"
                 if _tring is not None
                 else f"Vrot={gp.get('v_rot_flat','?')} km/s")
    print(f"  PA={gp['pa_deg']}°  i={gp['inc_deg']}°  "
          f"{_rot_desc}  d={gp['d_kpc']} kpc")
    print(f"  fit_f={fit_f}  fit_theta={fit_theta}")
    print(f"  σ_pm={sigma_pm} mas/yr  plx_pop={plx_pop} mas  "
          f"σ_plx_tot={sigma_plx_tot} mas")
    print(f"  μ_pop prior σ={mu_pop_prior_sigma} mas/yr  "
          f"member_sigma_clip={member_sigma_clip}  pm_sys_floor={pm_sys_floor} mas/yr")
    print(f"  n_iter: μ={n_iter_mu}  joint={n_iter_joint}  alpha={n_iter_alpha}  ft={n_iter_ft}")
    print(f"  poly_order={poly_order}  split_ccd={v1_split_ccd}  "
          f"v1 images={len(v1_image_names)}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\n  Loading bp3m input data for '{field_name}'...")
    imgs, stars_per_image, gaia_catalog = load_image_data_flc(data_root, field_name)
    if imgs is None or len(imgs) == 0:
        raise RuntimeError(f"No usable images found for '{field_name}'.")

    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)

    if v1_image_names:
        v1_bases = set()
        for n in v1_image_names:
            base = n[:-3] if n.endswith(('_hi', '_lo')) else n
            v1_bases.add(base)
        image_names = [n for n in image_names if n in v1_bases]
    if not image_names:
        raise RuntimeError("No images remain after filtering to v1 image set.")

    filtered_spi = {n: stars_per_image[n] for n in image_names}

    observed_ids: set = set()
    for spi in filtered_spi.values():
        observed_ids.update(spi['Gaia_id'].values)
    gaia_catalog = (gaia_catalog[gaia_catalog['Gaia_id'].isin(observed_ids)]
                    .reset_index(drop=True))
    star_id_to_idx = {int(gid): i for i, gid in enumerate(gaia_catalog['Gaia_id'])}

    imgs = {n: imgs[n] for n in image_names}

    if v1_split_ccd:
        imgs, filtered_spi = split_images_by_ccd(
            imgs, filtered_spi, min_stars_per_ccd=min_stars_split_ccd)
        image_names = sorted(filtered_spi.keys())
        star_id_to_idx, image_names, star_in_image = build_index_maps(
            filtered_spi, gaia_catalog)

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

    _gaia_dir = data_root / field_name / 'Gaia'
    if qso_anchors_csv is not None:
        _anchor_paths = ([Path(qso_anchors_csv)]
                         if not isinstance(qso_anchors_csv, list)
                         else [Path(p) for p in qso_anchors_csv])
    else:
        _p = find_qso_anchors(_gaia_dir, field_name)
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
            _nq_anch = len(_qdf_anchors)
            print(f"  QSO vetted anchors: {_nq_anch}")

            _qso_idx_list, _qso_pmra_list, _qso_pmdec_list = [], [], []
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
            print(f"    In HST field: {_n_qso_anchors}"
                  + (" ← prior applied" if _n_qso_anchors > 0 else " (none in FOV)"))
        except Exception as _qexc:
            print(f"  WARNING: could not load QSO anchors — {_qexc}")
    else:
        print("  QSO anchor file not found")

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

    # ── Load v1 bp3m posteriors for initial membership ────────────────────────
    v1_astrom_path = bp3m_dir / 'stellar_astrometry.csv'
    v_bp3m = solver.v_survey.copy()

    _pmra_init       = gaia_catalog['pmra'].to_numpy(float).copy()
    _pmdec_init      = gaia_catalog['pmdec'].to_numpy(float).copy()
    _sig_pmra_init   = gaia_catalog['pmra_error'].to_numpy(float).copy()
    _sig_pmdec_init  = gaia_catalog['pmdec_error'].to_numpy(float).copy()
    _corr_pm_init    = (gaia_catalog['pmra_pmdec_corr'].to_numpy(float).copy()
                        if 'pmra_pmdec_corr' in gaia_catalog.columns
                        else np.zeros(solver.n_stars))
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
                _sig_pmra_init  = np.where(np.isfinite(_sig_pmra_init),
                                           _sig_pmra_init,  _gaia_sig_pmra)
                _sig_pmdec_init = np.where(np.isfinite(_sig_pmdec_init),
                                           _sig_pmdec_init, _gaia_sig_pmdec)
                _corr_pm_init   = np.where(np.isfinite(_corr_pm_init),
                                           _corr_pm_init,   _gaia_corr_pm)
        except Exception as _exc:
            print(f"  WARNING: could not load v1 posteriors — {_exc}")

    # ── Apply v1 use_for_fit flags ─────────────────────────────────────────────
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

    # ── Bootstrap μ_pop ────────────────────────────────────────────────────────
    _mu_init_arr = np.array([float(mu_pop_init[0]), float(mu_pop_init[1])])
    if _v1_pm_loaded:
        _pmra_v1_only  = np.where(_v1_matched, _pmra_init,  np.nan)
        _pmdec_v1_only = np.where(_v1_matched, _pmdec_init, np.nan)
    else:
        _pmra_v1_only  = _pmra_init
        _pmdec_v1_only = _pmdec_init

    print(f"\n  Skipping bootstrap — using --mu_pop_init directly: "
          f"({_mu_init_arr[0]:+.4f}, {_mu_init_arr[1]:+.4f}) mas/yr")
    _mu_boot = _mu_init_arr.copy()

    # ── Initial member selection ───────────────────────────────────────────────
    print("\n  Selecting initial members from v1 bp3m PMs...")
    member_sidx = _select_initial_members(
        _pmra_v1_only, _pmdec_v1_only,
        _sig_pmra_init, _sig_pmdec_init, _corr_pm_init,
        _mu_boot, member_sigma_clip, sigma_pm, pm_sys_floor)
    print(f"  Initial members: {len(member_sidx)}")

    # ── μ_pop prior ────────────────────────────────────────────────────────────
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
        print("  WARNING: too few members; using bootstrap center as prior")
        mu_pop_prior = _mu_boot.copy()
        _unc_ra_v1 = _unc_dec_v1 = 0.0
    _n_prior_members = int(_fin_m.sum())
    print(f"  μ_pop prior from v1 members (N={_n_prior_members}): "
          f"({mu_pop_prior[0]:+.4f} ± {_unc_ra_v1:.4f}, "
          f"{mu_pop_prior[1]:+.4f} ± {_unc_dec_v1:.4f}) mas/yr  "
          f"[prior σ = ±{mu_pop_prior_sigma:.2f} mas/yr]")

    C_pop_prior_inv = np.eye(2) / mu_pop_prior_sigma ** 2
    mu_pop_current  = mu_pop_prior.copy()

    solver.gaia_n_hst_used[:] = 0
    for _img in image_names:
        _d = solver._img_data.get(_img)
        if _d is None:
            continue
        _use_any = _d['use_for_fit'] | _d.get('use_for_astrom', _d['use_for_fit'])
        np.add.at(solver.gaia_n_hst_used, _d['sidx'][_use_any], 1)

    # ── Precompute rotation offsets ────────────────────────────────────────────
    f_current     = float(f_init)
    theta_current = 0.0
    gaia_ra  = gaia_catalog['ra'].to_numpy(float)
    gaia_dec = gaia_catalog['dec'].to_numpy(float)
    rot_ra, rot_dec = compute_rotation_offsets(
        gaia_ra, gaia_dec, gp, f=f_current, theta_offset=theta_current)
    print(f"\n  Rotation offsets (f={f_current}, θ={np.degrees(theta_current):.1f}°): "
          f"rms(Δμ_ra*)={np.std(rot_ra):.4f}  "
          f"rms(Δμ_dec)={np.std(rot_dec):.4f} mas/yr")

    # ── Convenience wrapper ────────────────────────────────────────────────────
    def _solve(member_sidx_arg, mu_pop_arg, r_arg,
               fix_r_arg=False, z_weights_arg=None,
               fit_f_arg=False, fit_theta_arg=False,
               f_arg=1.0, theta_arg=0.0):
        return _joint_solve_pop_rot(
            solver, image_names,
            member_sidx_arg, mu_pop_arg,
            sigma_pm, plx_pop, sigma_plx_tot,
            C_pop_prior_inv, mu_pop_prior,
            r_arg, fix_r=fix_r_arg, z_weights=z_weights_arg,
            qso_sidx=_qso_sidx,
            qso_pmra=_qso_pmra_mas,
            qso_pmdec=_qso_pmdec_mas,
            rot_ra=rot_ra,
            rot_dec=rot_dec,
            fit_f=fit_f_arg, fit_theta=fit_theta_arg,
            f_current=f_arg, theta_current=theta_arg,
            gp=gp, gaia_ra=gaia_ra, gaia_dec=gaia_dec,
        )

    def _free_posterior(a_arg, C_vT_arg, msidx_arg, mu_pop_arg):
        return _compute_free_stellar_posterior_rot(
            a_arg, C_vT_arg, msidx_arg,
            sigma_pm, sigma_plx_tot, mu_pop_arg, plx_pop,
            solver._C_VG_inv_per_star, rot_ra, rot_dec)

    def _select_members(a_free_arg, mu_pop_arg, C_free_arg):
        return _select_members_from_a_rot(
            a_free_arg, mu_pop_arg, _n_hst_det, C_free_arg,
            sigma_pm, rot_ra, rot_dec,
            sigma_clip=member_sigma_clip,
            max_sigma_free_pm=max_sigma_free_pm,
            pm_sys_floor=pm_sys_floor)

    # ── Phase 1: μ-only solve ─────────────────────────────────────────────────
    print(f"\n  Phase 1: μ-only solve ({n_iter_mu} iterations, r fixed)...")
    r_current = r_bp3m.copy()
    C_shared_mu = None
    for mu_iter in range(n_iter_mu):
        _, mu_pop_new, _, _, C_shared_mu, C_vT, a_arr, _, _ = _solve(
            member_sidx, mu_pop_current, r_current, fix_r_arg=True)
        delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
        _mu_pop_used = mu_pop_current.copy()
        mu_pop_current = mu_pop_new
        _a_free, _C_free = _free_posterior(a_arr, C_vT, member_sidx, _mu_pop_used)
        member_sidx = _select_members(_a_free, mu_pop_current, _C_free)
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

    # ── Phase 2: joint solve ─────────────────────────────────────────────────
    print(f"\n  Phase 2: joint solve ({n_iter_joint} iterations)...")
    C_shared_joint = None
    for jt_iter in range(n_iter_joint):
        r_new, mu_pop_new, _, _, C_shared_joint, C_vT, a_arr, _, _ = _solve(
            member_sidx, mu_pop_current, r_current)
        delta_r  = float(np.max(np.abs(r_new - r_current)))
        delta_mu = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
        _mu_pop_used = mu_pop_current.copy()
        r_current      = r_new
        mu_pop_current = mu_pop_new
        _a_free, _C_free = _free_posterior(a_arr, C_vT, member_sidx, _mu_pop_used)
        member_sidx = _select_members(_a_free, mu_pop_current, _C_free)
        print(f"    iter {jt_iter + 1}/{n_iter_joint}: "
              f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f})  "
              f"Δr={delta_r:.3e}  Δμ={delta_mu:.3e}  members={len(member_sidx)}")
        solver._update_R(r_current)
        solver._update_geometry(r_current, a_arr)
        if delta_r < 1e-6 and delta_mu < 1e-6:
            print(f"    Converged.")
            break

    # ── Phase 3: joint solve + alpha update ───────────────────────────────────
    if n_iter_alpha > 0:
        print(f"\n  Phase 3: joint solve + alpha update ({n_iter_alpha} iterations)...")
        C_shared_joint_p3 = C_shared_joint
        for al_iter in range(n_iter_alpha):
            r_new, mu_pop_new, _, _, C_shared_joint_p3, C_vT, a_arr, _, _ = _solve(
                member_sidx, mu_pop_current, r_current)
            solver._update_R(r_new)
            solver._update_geometry(r_new, a_arr)

            alpha_info = _compute_alpha_updates(solver, image_names, r_new, a_arr,
                                                alpha_damp=alpha_damp)

            delta_r     = float(np.max(np.abs(r_new - r_current)))
            delta_mu    = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
            _mu_pop_used = mu_pop_current.copy()
            r_current      = r_new
            mu_pop_current = mu_pop_new
            _a_free, _C_free = _free_posterior(a_arr, C_vT, member_sidx, _mu_pop_used)
            member_sidx = _select_members(_a_free, mu_pop_current, _C_free)

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

    # ── Phase 4: free f/θ fitting ─────────────────────────────────────────────
    C_shared_ft = None
    sigma_f_final = sigma_theta_final = np.nan
    if (fit_f or fit_theta) and n_iter_ft > 0:
        print(f"\n  Phase 4: f/θ fitting ({n_iter_ft} iterations, r fixed, members frozen)...")
        # Member set is frozen during f/θ inner optimization.
        # Hard chi² thresholding makes membership discontinuous in f/θ space, causing
        # oscillation if we re-select each iteration.  One re-selection runs at the end.
        _member_sidx_ft = member_sidx.copy()
        for ft_iter in range(n_iter_ft):
            _, mu_pop_new, f_new, theta_new, C_shared_ft, C_vT, a_arr, _, _ = _solve(
                _member_sidx_ft, mu_pop_current, r_current,
                fix_r_arg=True,
                fit_f_arg=fit_f, fit_theta_arg=fit_theta,
                f_arg=f_current, theta_arg=theta_current)
            delta_mu    = float(np.max(np.abs(mu_pop_new - mu_pop_current)))
            delta_f     = abs(f_new     - f_current)     if fit_f     else 0.0
            delta_theta = abs(theta_new - theta_current) if fit_theta else 0.0
            _mu_pop_used = mu_pop_current.copy()
            mu_pop_current = mu_pop_new
            f_current      = f_new
            theta_current  = theta_new
            rot_ra, rot_dec = compute_rotation_offsets(
                gaia_ra, gaia_dec, gp, f=f_current, theta_offset=theta_current)
            print(f"    iter {ft_iter + 1}/{n_iter_ft}: "
                  f"μ_pop=({mu_pop_current[0]:+.4f}, {mu_pop_current[1]:+.4f})  "
                  f"f={f_current:.4f}  θ={np.degrees(theta_current):+.3f}°  "
                  f"Δμ={delta_mu:.3e}  Δf={delta_f:.3e}")
            if delta_mu < 1e-6 and delta_f < 1e-6 and delta_theta < 1e-8:
                print(f"    Converged.")
                break
        # Final member re-selection using converged (f, θ)
        _a_free, _C_free = _free_posterior(a_arr, C_vT, _member_sidx_ft, mu_pop_current)
        member_sidx = _select_members(_a_free, mu_pop_current, _C_free)
        print(f"  Phase 4 final member re-selection: {len(member_sidx)}")

        if C_shared_ft is not None:
            n_ext = int(fit_f) + int(fit_theta)
            # C_shared_ft is (2+n_ext, 2+n_ext); f is [2], θ is [2 or 3]
            if fit_f:
                sigma_f_final = float(np.sqrt(max(C_shared_ft[2, 2], 0.0)))
            if fit_theta:
                _it = 3 if fit_f else 2
                sigma_theta_final = float(np.sqrt(max(C_shared_ft[_it, _it], 0.0)))

    n_r = len(image_names) * solver.N_R
    sigma_mu_joint = (np.sqrt(np.diag(C_shared_joint[n_r:, n_r:]))
                      if C_shared_joint is not None else np.array([np.nan, np.nan]))
    print(f"\n  Final: μ_pop=({mu_pop_current[0]:+.4f} ± {sigma_mu_joint[0]:.4f}, "
          f"{mu_pop_current[1]:+.4f} ± {sigma_mu_joint[1]:.4f}) mas/yr")
    print(f"  f={f_current:.4f} ± {sigma_f_final:.4f}  "
          f"θ={np.degrees(theta_current):+.3f}° ± {np.degrees(sigma_theta_final):.3f}°")
    print(f"  Final members: {len(member_sidx)}")

    # ── Final posterior pass ───────────────────────────────────────────────────
    print("\n  Final posterior pass...")
    _, _, _, _, C_shared_final, C_vT_final, v_mean, _, K_img_final = _solve(
        member_sidx, mu_pop_current, r_current)

    # ── Analytic marginalised posteriors ──────────────────────────────────────
    print("\n  Computing analytic marginalised posteriors...")
    C_r    = C_shared_final[:n_r, :n_r]
    C_mu   = C_shared_final[n_r:, n_r:]
    C_r_mu = C_shared_final[:n_r, n_r:]

    v_mean_marg, v_cov_r = solver.compute_analytic_posteriors(
        r_current, C_r, v_mean, K_img_final, C_vT_final)

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

    CvT_K = np.einsum('nij,njk->nik', C_vT_final, K_all)

    B_all = np.zeros((solver.n_stars, 5, 2))
    B_all[member_sidx] = (sigma_pm ** -2) * C_vT_final[member_sidx, :, 2:4]

    C_extra_mu    = np.einsum('nik,kl,njl->nij', B_all, C_mu, B_all)
    C_extra_cross = (np.einsum('nik,kl,njl->nij', CvT_K, C_r_mu,   B_all) +
                     np.einsum('nik,kl,njl->nij', B_all,  C_r_mu.T, CvT_K))
    v_cov      = v_cov_r + C_extra_cross + C_extra_mu
    v_cov_full = v_cov + C_vT_final

    # ── Diffuse-prior reference posteriors ─────────────────────────────────────
    print("\n  Computing diffuse-prior (free) stellar posteriors...")
    _, _, _, _, _, C_vT_free_sol, v_mean_free_cond, _, K_img_free = _solve(
        np.array([], dtype=int), mu_pop_current, r_current, fix_r_arg=True)
    v_mean_free_marg, v_cov_free_sol = solver.compute_analytic_posteriors(
        r_current, C_r, v_mean_free_cond, K_img_free, C_vT_free_sol)

    # ── Save results ───────────────────────────────────────────────────────────
    print("\n  Saving results...")

    from bp3m.pipeline.run_alignment import compute_chi2_per_star

    # image_transformations.csv
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

    # stellar_astrometry.csv
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

    _is_member_arr = np.zeros(solver.n_stars, dtype=bool)
    if member_sidx is not None and len(member_sidx) > 0:
        _is_member_arr[member_sidx] = True
    g['is_member'] = _is_member_arr

    # Rotation model columns
    g['rot_pm_ra_masyr']  = rot_ra
    g['rot_pm_dec_masyr'] = rot_dec

    g.to_csv(output_pfr / 'stellar_astrometry.csv', index=False)
    print(f"  Saved: stellar_astrometry.csv  "
          f"({len(g)} stars, {solver.gaia_n_hst_used.sum()} HST detections)")

    # Covariance arrays
    np.save(output_pfr / 'v_cov_marginalised.npy', v_cov)
    np.save(output_pfr / 'C_vT.npy',              C_vT_final)
    np.save(output_pfr / 'C_r.npy',               C_r)
    np.save(output_pfr / 'C_joint.npy',            C_shared_final)
    print(f"  Saved: v_cov_marginalised.npy, C_vT.npy, C_r.npy, C_joint.npy")

    # Detection flags
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

    # Per-detection GDC residuals
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

    # mu_pop.json (with rotation model fields)
    _C_mu_out = C_shared_final[n_r:, n_r:]
    _corr_mu = (float(_C_mu_out[0, 1] / (sigma_mu_joint[0] * sigma_mu_joint[1]))
                if (sigma_mu_joint[0] > 0 and sigma_mu_joint[1] > 0) else 0.0)

    _g_pmra  = solver.v_survey[member_sidx, 2]
    _g_pmdec = solver.v_survey[member_sidx, 3]
    _g_C_pm  = solver.C_survey[member_sidx][:, 2:4, 2:4]
    _g_ok    = (np.isfinite(_g_pmra) & np.isfinite(_g_pmdec) &
                np.isfinite(_g_C_pm[:, 0, 0]) & (_g_C_pm[:, 0, 0] > 0) &
                np.isfinite(_g_C_pm[:, 1, 1]) & (_g_C_pm[:, 1, 1] > 0))
    if _g_ok.sum() >= 2:
        _Lambda_g = np.zeros((2, 2))
        _h_g      = np.zeros(2)
        for _k in np.where(_g_ok)[0]:
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
        # Rotation model fields
        'f_star_mult':           float(f_current),
        'sigma_f_star_mult':     None if np.isnan(sigma_f_final) else float(sigma_f_final),
        'theta_offset_deg':      float(np.degrees(theta_current)),
        'sigma_theta_offset_deg': None if np.isnan(sigma_theta_final) else float(np.degrees(sigma_theta_final)),
        'pa_deg':                float(gp['pa_deg']),
        'inc_deg':               float(gp['inc_deg']),
        'rotation_model':        'tilted_ring' if gp.get('tilted_ring') is not None else 'arctangent',
        'n_tilted_rings':        int(len(gp['tilted_ring'])) if gp.get('tilted_ring') is not None else None,
        'd_kpc':                 float(gp['d_kpc']),
        'fit_f':                 fit_f,
        'fit_theta':             fit_theta,
    }
    with open(output_pfr / 'mu_pop.json', 'w') as _f:
        json.dump(mu_result, _f, indent=2)

    # run_config.json
    with open(output_pfr / 'run_config.json', 'w') as _f:
        json.dump({
            'poly_order': poly_order, 'n_r_per_image': solver.N_R,
            'n_images': len(image_names),
            'n_stars': solver.n_stars, 'image_names': image_names,
            'sigma_pm': sigma_pm, 'plx_pop': plx_pop,
            'sigma_plx_tot': sigma_plx_tot,
            'mu_pop_prior_sigma': mu_pop_prior_sigma,
            'n_iter_mu': n_iter_mu, 'n_iter_joint': n_iter_joint,
            'n_iter_ft': n_iter_ft,
            'member_sigma_clip': member_sigma_clip,
            'mu_pop_ra': float(mu_pop_current[0]),
            'mu_pop_dec': float(mu_pop_current[1]),
            'n_members': int(len(member_sidx)),
            'split_ccd': v1_split_ccd,
            'f_star_mult': float(f_current),
            'sigma_f_star_mult': None if np.isnan(sigma_f_final) else float(sigma_f_final),
            'theta_offset_deg': float(np.degrees(theta_current)),
            'sigma_theta_offset_deg': None if np.isnan(sigma_theta_final) else float(np.degrees(sigma_theta_final)),
            'fit_f': fit_f,
            'fit_theta': fit_theta,
        }, _f, indent=2)
    print(f"  Saved: mu_pop.json, run_config.json")

    if not no_plots:
        # ── make_plots (VPD, CMD, chi², etc.) ─────────────────────────────────
        try:
            from bp3m.pipeline.plot_results import make_plots
            print("\n  Generating diagnostic plots...")
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

        # ── PM vs G-mag / HST x / HST y ───────────────────────────────────────
        # Subtract per-star rotation offsets so the plot shows residuals from
        # the expected rotation pattern rather than from mu_pop alone.
        try:
            from bp3m.pipeline.run_pop_fit import _plot_pm_vs_properties
            _plot_dir_plots = output_pfr / 'plots'
            _plot_dir_plots.mkdir(parents=True, exist_ok=True)
            print("\n  Plotting PM vs properties...")
            _dmu_ra, _dmu_dec = compute_rotation_offsets(
                gaia_ra, gaia_dec, gp, f=f_current, theta_offset=theta_current)
            _v_free_rotcorr = v_mean_free_marg.copy()
            _v_free_rotcorr[:, 2] -= _dmu_ra
            _v_free_rotcorr[:, 3] -= _dmu_dec
            _plot_pm_vs_properties(
                _plot_dir_plots, solver, image_names, gaia_catalog,
                _v_free_rotcorr, C_vT_free_sol,
                member_sidx, mu_pop_current, sigma_pm, field_name,
            )
        except Exception as _exc:
            print(f"  WARNING: _plot_pm_vs_properties failed — {_exc}")

        # ── Sky position coloured by PM ────────────────────────────────────────
        try:
            print("\n  Plotting sky PM map for members...")
            _plot_sky_pm_members(
                output_pfr / 'plots',
                output_pfr / 'stellar_astrometry.csv',
                mu_pop_current, field_name, gp,
            )
        except Exception as _exc:
            print(f"  WARNING: sky_pm_members plot failed — {_exc}")

        # ── Residual maps (before = bp3m, after = pop-fit-rotation) — last ────
        _plot_dir_res = output_pfr / 'plots' / 'residuals'
        print(f"\n  Plotting before/after residual maps ({len(image_names)} images)...")
        try:
            from bp3m.pipeline.run_pop_fit import _plot_pop_residual_maps
            _plot_pop_residual_maps(
                _plot_dir_res, image_names, solver,
                r_before=r_bp3m,   v_before=v_bp3m,
                r_after=r_current, v_after=v_mean,
                C_vT_after=C_vT_final,
                prefix='final',
            )
        except Exception as _exc:
            print(f"  WARNING: residual maps failed — {_exc}")

    t_elapsed = time.time() - t_start
    print(f"\n  Done in {t_elapsed:.1f}s")
    return output_pfr


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog='bp3m-pop-fit-rotation',
        description='Rotation-model pop-fit for NGC 55 / NGC 300 (run after bp3m).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', required=True,
                        choices=list(GALAXY_PARAMS),
                        help='Target name')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--sigma_pm', type=float, default=None,
                        help='Residual PM dispersion (mas/yr) after rotation removed. '
                             'Defaults to galaxy-specific value.')
    parser.add_argument('--mu_pop_prior_sigma', type=float, default=0.5)
    parser.add_argument('--n_iter_mu',    type=int, default=20)
    parser.add_argument('--n_iter_joint', type=int, default=20)
    parser.add_argument('--n_iter_alpha', type=int, default=20)
    parser.add_argument('--alpha_damp',   type=float, default=0.5)
    parser.add_argument('--member_sigma_clip', type=float, default=3.0)
    parser.add_argument('--pm_sys_floor', type=float, default=0.25)
    parser.add_argument('--max_sigma_free_pm', type=float, default=3.0)
    parser.add_argument('--f_init',       type=float, default=1.0,
                        help='Initial (and fixed, if --no_fit_f) stellar/HI rotation speed ratio (default 1.0)')
    parser.add_argument('--no_fit_f',     action='store_true',
                        help='Hold f_star_mult fixed at f_init throughout')
    parser.add_argument('--no_fit_theta', action='store_true',
                        help='Hold theta_offset fixed at 0.0')
    parser.add_argument('--n_iter_ft',    type=int, default=20,
                        help='Phase 4 iterations for f/θ fitting (ignored if both --no_fit_f and --no_fit_theta)')
    parser.add_argument('--no_plots',     action='store_true')
    parser.add_argument('--qso_anchors_csv', type=str, default=None, nargs='+')

    args = parser.parse_args()

    run_pop_fit_rotation(
        output_dir=Path(args.output_dir).resolve(),
        field_name=args.name,
        sigma_pm=args.sigma_pm,
        mu_pop_prior_sigma=args.mu_pop_prior_sigma,
        n_iter_mu=args.n_iter_mu,
        n_iter_joint=args.n_iter_joint,
        n_iter_alpha=args.n_iter_alpha,
        alpha_damp=args.alpha_damp,
        member_sigma_clip=args.member_sigma_clip,
        pm_sys_floor=args.pm_sys_floor,
        max_sigma_free_pm=args.max_sigma_free_pm,
        fit_f=not args.no_fit_f,
        fit_theta=not args.no_fit_theta,
        f_init=args.f_init,
        n_iter_ft=args.n_iter_ft,
        no_plots=args.no_plots,
        qso_anchors_csv=[Path(p) for p in args.qso_anchors_csv] if args.qso_anchors_csv else None,
    )
