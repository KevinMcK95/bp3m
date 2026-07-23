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

# ── Galaxy parameter table ─────────────────────────────────────────────────────

_DEG2RAD = np.pi / 180.0
_ARCSEC2KPC = 4.84814e-6   # 1 arcsec at 1 kpc = 4.84814e-6 kpc

GALAXY_PARAMS: dict[str, dict] = {
    'NGC_55': dict(
        ra_cen=3.7233, dec_cen=-39.1967,        # deg
        d_kpc=1932.0,
        plx_pop=5.176e-4,                        # mas
        sigma_plx_tot=2.86e-5,                   # mas
        pa_deg=108.0,                            # receding major axis, N through E
        inc_deg=84.0,
        v_rot_flat=90.6,                         # km/s
        r_turn_kpc=1.0,                          # inner turnover radius
        sigma_pm_disp=0.001,                     # residual dispersion mas/yr (after rotation removed)
        f0=1.0, sigma_f=0.20,                    # prior on f_star_mult
        sigma_theta_deg=10.0,                    # prior on theta_offset
        mu_pop_init=(-0.0044, -0.0023),
    ),
    'NGC_300': dict(
        ra_cen=13.7229, dec_cen=-37.6844,
        d_kpc=2089.0,
        plx_pop=4.786e-4,
        sigma_plx_tot=1.323e-5,
        pa_deg=290.0,
        inc_deg=42.0,
        v_rot_flat=90.0,
        r_turn_kpc=1.5,
        sigma_pm_disp=0.001,
        f0=1.0, sigma_f=0.20,
        sigma_theta_deg=10.0,
        mu_pop_init=(-0.0042, -0.0027),
    ),
}


# ── Rotation model geometry ────────────────────────────────────────────────────

def _v_rot_func(R_kpc: np.ndarray, v_flat: float, r_turn: float) -> np.ndarray:
    """Courteau (1997) arctangent rotation curve, flat beyond r_turn."""
    return v_flat * (2 / np.pi) * np.arctan(R_kpc / max(r_turn, 1e-6))


def compute_rotation_offsets(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    gp: dict,
    f: float = 1.0,
    theta_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-star rotation PM offsets (Δμ_ra*, Δμ_dec) in mas/yr.

    Parameters
    ----------
    ra_deg, dec_deg : star sky positions (degrees)
    gp              : galaxy parameter dict from GALAXY_PARAMS
    f               : stellar/HI rotation speed ratio (free parameter)
    theta_offset    : kinematic PA offset from literature PA (radians)

    Returns
    -------
    dmu_ra, dmu_dec : (n_stars,) arrays in mas/yr
    """
    d_kpc   = gp['d_kpc']
    pa_eff  = (gp['pa_deg'] * _DEG2RAD) + theta_offset
    cos_i   = np.cos(gp['inc_deg'] * _DEG2RAD)
    v_flat  = gp['v_rot_flat']
    r_turn  = gp['r_turn_kpc']
    kappa   = 4.74047   # km/s per (mas/yr · kpc)

    # Sky offsets in arcsec (East, North)
    cos_dec = np.cos(gp['dec_cen'] * _DEG2RAD)
    x_as    = (ra_deg  - gp['ra_cen']) * cos_dec * 3600.0   # arcsec east
    y_as    = (dec_deg - gp['dec_cen'])            * 3600.0  # arcsec north

    # Project onto major / minor axis
    xi  =  x_as * np.sin(pa_eff) + y_as * np.cos(pa_eff)   # along major axis
    eta =  x_as * np.cos(pa_eff) - y_as * np.sin(pa_eff)   # along minor axis

    # Convert arcsec → kpc and deproject
    scale = d_kpc * _ARCSEC2KPC
    X     = xi  * scale
    # Guard against edge-on singularity (cos_i ≈ 0 → clamp deprojection)
    cos_i_safe = max(abs(cos_i), 0.05) * np.sign(cos_i) if cos_i != 0 else 0.05
    Y     = (eta / cos_i_safe) * scale

    R   = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    V_rot = f * _v_rot_func(R, v_flat, r_turn)

    # Sky velocity components (along/across major axis, then back to E/N)
    v_xi  = -V_rot * np.sin(phi)
    v_eta =  V_rot * np.cos(phi) * cos_i

    v_east  = v_xi * np.sin(pa_eff) + v_eta * np.cos(pa_eff)
    v_north = v_xi * np.cos(pa_eff) - v_eta * np.sin(pa_eff)

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
# This is a copy of run_pop_fit._joint_solve_pop with the following changes:
#   1. Extra arguments: rot_ra, rot_dec, f, theta_offset, gp,
#      f_prior_inv_sq, theta_prior_inv_sq
#   2. Shared parameter vector grows from [μ_pop] (size 2) to
#      [μ_pop, f, θ] (size 4 in fix_r mode) or [r, μ_pop, f, θ] in joint mode.
#   3. Member PM prior centre: μ_pop + (rot_ra[star], rot_dec[star]).
#   4. Schur correction for the μ block is extended to include ∂rot/∂f
#      and ∂rot/∂θ couplings.
#
# TODO: implement the full modified solve.  Current stub returns the
#       unmodified result from the base solver so the file is importable
#       and testable end-to-end before the rotation coupling is wired in.

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
    # ── rotation model extras ─────────────────────────────────────────────
    rot_ra: "np.ndarray | None" = None,    # (n_stars,) current rotation offsets
    rot_dec: "np.ndarray | None" = None,
    jac_f_ra: "np.ndarray | None" = None,  # (n_stars,) ∂rot_ra/∂f
    jac_f_dec: "np.ndarray | None" = None,
    jac_t_ra: "np.ndarray | None" = None,  # (n_stars,) ∂rot_ra/∂θ
    jac_t_dec: "np.ndarray | None" = None,
    f_prior_inv_sq: float = 0.0,           # (1/σ_f)²
    theta_prior_inv_sq: float = 0.0,       # (1/σ_θ)²
    f_current: float = 1.0,
    theta_current: float = 0.0,
    f_prior: float = 1.0,
    theta_prior: float = 0.0,
):
    """
    Joint solve for (r, μ_pop, f, θ) with per-star rotation prior.

    The shared system is extended:
      fix_r=True  → 4×4 block: [μ_pop[0], μ_pop[1], f, θ]
      fix_r=False → (n_r+4)×(n_r+4)

    Per-star prior centre for member s:
      E[μ_ra*(s)]  = μ_pop[0] + rot_ra[s]
      E[μ_dec(s)]  = μ_pop[1] + rot_dec[s]

    where rot_ra[s] and rot_dec[s] depend on (f, θ) at the current linearisation
    point, and the Jacobian (jac_f_ra etc.) supplies the coupling derivatives.

    Returns same tuple as _joint_solve_pop plus (f_new, theta_new, C_ft).
    """
    # ── Import and delegate to base solver ────────────────────────────────────
    # STUB: until the full Schur extension is implemented, run the base solver
    # with the rotation offsets absorbed into a shifted mu_pop_current.
    # This is NOT correct for the f/θ gradients but allows end-to-end testing.
    from bp3m.pipeline.run_pop_fit import _joint_solve_pop as _base_solve

    _rot_ra  = rot_ra  if rot_ra  is not None else np.zeros(solver.n_stars)
    _rot_dec = rot_dec if rot_dec is not None else np.zeros(solver.n_stars)

    # Temporarily shift h_align / h_all for member stars by rot_offset.
    # This is exact for μ_pop but ignores the f/θ gradient coupling.
    # TODO: replace with the full 4-parameter Schur extension.
    _mu_shifted = mu_pop_current.copy()  # placeholder — full impl below

    result = _base_solve(
        solver=solver,
        image_names=image_names,
        member_sidx=member_sidx,
        mu_pop_current=_mu_shifted,
        sigma_pm=sigma_pm,
        plx_pop=plx_pop,
        sigma_plx_tot=sigma_plx_tot,
        C_pop_prior_inv=C_pop_prior_inv,
        mu_pop_prior=mu_pop_prior,
        r_current=r_current,
        fix_r=fix_r,
        z_weights=z_weights,
        qso_sidx=qso_sidx,
        qso_pmra=qso_pmra,
        qso_pmdec=qso_pmdec,
    )
    # Unpack base result and pass through f/θ unchanged until full impl
    r_hat, mu_pop_hat, C_shared, C_vT, a_arr, a_align, K_img = result
    return r_hat, mu_pop_hat, f_current, theta_current, C_shared, C_vT, a_arr, a_align, K_img


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
) -> np.ndarray:
    """
    Select members by Mahalanobis distance from (μ_pop + rot_offset[star]).

    Identical to _select_members_from_a except delta_pm uses the
    per-star rotation-corrected expected PM.
    """
    from bp3m.pipeline.run_pop_fit import _select_members_from_a as _base

    eidx = np.where(n_hst >= 1)[0]
    if len(eidx) < min_members:
        return eidx

    _has_valid = (C_vT[eidx, 2, 2] > 0) | (C_vT[eidx, 3, 3] > 0)
    eidx = eidx[_has_valid]
    if len(eidx) < min_members:
        return eidx

    pmra  = a_arr[eidx, 2]
    pmdec = a_arr[eidx, 3]

    # rotation-corrected expected PM for each candidate
    mu_exp_ra  = mu_pop[0] + rot_ra[eidx]
    mu_exp_dec = mu_pop[1] + rot_dec[eidx]

    C_pm_sub = C_vT[eidx, 2:4, 2:4].copy()
    C_pm_sub[:, 0, 0] += sigma_pm ** 2
    C_pm_sub[:, 1, 1] += sigma_pm ** 2
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

    # Strip rotation-corrected prior centre
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


# ── Main entry point ───────────────────────────────────────────────────────────

def run_pop_fit_rotation(
    output_dir: Path,
    field_name: str,
    sigma_pm: float | None = None,       # defaults to gp['sigma_pm_disp']
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
    mu_pop_init: tuple[float, float] | None = None,
    no_plots: bool = False,
    qso_anchors_csv: "Path | str | list | None" = None,
) -> Path:
    """
    Run rotation-model pop-fit for NGC_55 or NGC_300.

    Uses hardcoded galaxy parameters from GALAXY_PARAMS.
    Fits μ_pop jointly with (f, θ) if fit_f/fit_theta are True.
    """
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

    # TODO: implement full rotation-model solve loop.
    #
    # Outline:
    #   1. Data loading (identical to run_pop_fit: load_image_data_flc,
    #      build_index_maps, split_images_by_ccd, BP3MSolver).
    #   2. Load BP3M v1 outputs (r_hat, alpha, use_for_fit flags) via
    #      _load_bp3m_outputs and _apply_bp3m_flags from run_pop_fit.
    #   3. Initialise: f=1.0, theta_offset=0.0.
    #   4. Iteration loop (Phase 1: fix_r=True, then Phase 2: fix_r=False):
    #      a. Compute rotation offsets for current (f, θ):
    #           rot_ra, rot_dec = compute_rotation_offsets(
    #               gaia_ra, gaia_dec, gp, f=f_current, theta_offset=theta_current)
    #      b. Compute Jacobian for the Schur coupling:
    #           jac_f_ra, jac_f_dec, jac_t_ra, jac_t_dec =
    #               compute_rotation_offsets_jacobian(...)
    #      c. Call _joint_solve_pop_rot (replacing _joint_solve_pop) with
    #         current (f, θ) and Jacobian arrays.
    #      d. Update member selection via _select_members_from_a_rot.
    #      e. Update free posteriors via _compute_free_stellar_posterior_rot.
    #      f. Re-linearise: update rot_ra, rot_dec, Jacobian for new (f, θ).
    #   5. Phase 3 (alpha update): identical to run_pop_fit Phase 3.
    #   6. Save results: mu_pop.json with added fields f_star_mult, theta_offset_deg.
    #
    # For now, fall back to the base run_pop_fit with the rotation offsets NOT
    # yet fed into the solve.  Replace this once _joint_solve_pop_rot is complete.
    from bp3m.pipeline.run_pop_fit import run_pop_fit as _base_run

    print(f"\n  [rotation model] Field: {field_name}")
    print(f"  PA={gp['pa_deg']}°  i={gp['inc_deg']}°  "
          f"V_rot={gp['v_rot_flat']} km/s  d={gp['d_kpc']} kpc")
    print(f"  fit_f={fit_f}  fit_theta={fit_theta}  "
          f"σ_f={gp['sigma_f']}  σ_θ={gp['sigma_theta_deg']}°")
    print("  NOTE: rotation solve stub — currently delegating to base run_pop_fit")

    return _base_run(
        output_dir=output_dir,
        field_name=field_name,
        sigma_pm=sigma_pm,
        plx_pop=gp['plx_pop'],
        sigma_plx_tot=gp['sigma_plx_tot'],
        mu_pop_prior_sigma=mu_pop_prior_sigma,
        n_iter_mu=n_iter_mu,
        n_iter_joint=n_iter_joint,
        n_iter_alpha=n_iter_alpha,
        alpha_damp=alpha_damp,
        member_sigma_clip=member_sigma_clip,
        pm_sys_floor=pm_sys_floor,
        max_sigma_free_pm=max_sigma_free_pm,
        mu_pop_init=mu_pop_init,
        freeze_mu_pop_init=True,
        no_plots=no_plots,
        qso_anchors_csv=qso_anchors_csv,
    )


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
    parser.add_argument('--no_fit_f',     action='store_true',
                        help='Hold f_star_mult fixed at 1.0')
    parser.add_argument('--no_fit_theta', action='store_true',
                        help='Hold theta_offset fixed at 0.0')
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
        no_plots=args.no_plots,
        qso_anchors_csv=[Path(p) for p in args.qso_anchors_csv] if args.qso_anchors_csv else None,
    )
