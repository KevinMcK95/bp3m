"""
run_alignment_cte.py  —  Joint CTE + astrometry model for HST ACS/WFC.

Extends BP3M v2 alignment to simultaneously fit per-image transformations (r_j),
stellar astrometry (v_i), and a parametric CTE model θ_CTE.

See docs/cte_joint_model.md for the full mathematical derivation and design decisions.

CTE model summary
-----------------
For chip c, detection k in image j (epoch t_j):

  δCTE_x_k = (t_j − t_0) · φ(mag_k; δ_c) · b_x(X_k, Y_k) · γ_x_c
  δCTE_y_k = (t_j − t_0) · φ(mag_k; δ_c) · b_y(X_k, Y_k) · γ_y_c

  φ(mag; δ) = 10^{0.4·δ·(mag − mag_ref)} − 1   (flux power-law; mag_ref = 20)
  b_y = [Y', X·Y', Y'²]   where Y' = Y_c − Y_readout_c   (boundary: CTE_y = 0 at Y')
  b_x = [X_c, X_c·Y_c, X_c²]                              (boundary: CTE_x = 0 at X=0)

  γ_x_c, γ_y_c: (3,) composite polynomial coefficients (absorb temporal α)
  δ_c: flux power-law exponent (shared between CTE_x and CTE_y for chip c)

Parameters: 14 total (7 per chip: δ, γ_x(3), γ_y(3)).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── ACS/WFC chip geometry constants ──────────────────────────────────────────
# Centered GDC pixel coordinates: X_c = x_gdc − 2048, Y_c = y_gdc − 2048.
# _hi chip: readout register at top of chip (+Y side).
# _lo chip: readout register at bottom of chip (−Y side).
_HI_Y_READOUT = +2047.0   # Y_c at readout register for _hi chip
_LO_Y_READOUT = -2048.0   # Y_c at readout register for _lo chip
_MAG_REF      = 20.0      # reference magnitude for flux power-law normalization


# ── CTE parameter dataclass ───────────────────────────────────────────────────

@dataclass
class CTEChipParams:
    """CTE model parameters for one ACS/WFC chip."""
    chip: str                                              # 'hi' or 'lo'
    y_readout: float                                       # Y_c at readout register
    delta: float = 1.0                                     # flux power-law exponent
    gamma_x: np.ndarray = field(default_factory=lambda: np.zeros(3))  # serial CTE
    gamma_y: np.ndarray = field(default_factory=lambda: np.zeros(3))  # parallel CTE

    def copy(self) -> 'CTEChipParams':
        return CTEChipParams(
            chip=self.chip,
            y_readout=self.y_readout,
            delta=float(self.delta),
            gamma_x=self.gamma_x.copy(),
            gamma_y=self.gamma_y.copy(),
        )

    def as_dict(self) -> dict:
        return {
            'chip': self.chip,
            'y_readout': self.y_readout,
            'delta': float(self.delta),
            'gamma_x': self.gamma_x.tolist(),
            'gamma_y': self.gamma_y.tolist(),
        }


def default_cte_params() -> dict[str, CTEChipParams]:
    """Return default (zero) CTE parameters for both chips."""
    return {
        'hi': CTEChipParams(chip='hi', y_readout=_HI_Y_READOUT),
        'lo': CTEChipParams(chip='lo', y_readout=_LO_Y_READOUT),
    }


# ── Flux power-law model ──────────────────────────────────────────────────────

def phi_flux(mag: np.ndarray, delta: float, mag_ref: float = _MAG_REF) -> np.ndarray:
    """
    φ(mag; δ) = 10^{0.4·δ·(mag − mag_ref)} − 1.

    Physical interpretation: CTE trailing ∝ (flux/flux_ref)^δ − 1.
    At mag = mag_ref: φ = 0.
    For faint stars (mag > mag_ref), δ > 0 → φ < 0 (less flux → less trailing).
    """
    return np.power(10.0, 0.4 * delta * (np.asarray(mag, dtype=float) - mag_ref)) - 1.0


def dphi_ddelta(mag: np.ndarray, delta: float, mag_ref: float = _MAG_REF) -> np.ndarray:
    """
    dφ/dδ = 0.4·ln(10)·(mag − mag_ref)·10^{0.4·δ·(mag − mag_ref)}.

    Analytic gradient of φ w.r.t. δ for Gauss-Newton δ update.
    """
    dm = np.asarray(mag, dtype=float) - mag_ref
    return 0.4 * np.log(10.0) * dm * np.power(10.0, 0.4 * delta * dm)


# ── CTE basis functions ───────────────────────────────────────────────────────

def cte_y_basis(X_c: np.ndarray, Y_c: np.ndarray, y_readout: float) -> np.ndarray:
    """
    Parallel (Y-direction) CTE basis: (n, 3).

    b_y = [Y', X_c·Y', Y'²]  where Y' = Y_c − y_readout.
    Boundary condition: CTE_y = 0 at Y' = 0 (readout register).
    """
    Yp = Y_c - y_readout
    return np.stack([Yp, X_c * Yp, Yp ** 2], axis=1)


def cte_x_basis(X_c: np.ndarray, Y_c: np.ndarray) -> np.ndarray:
    """
    Serial (X-direction) CTE basis: (n, 3).

    b_x = [X_c, X_c·Y_c, X_c²].
    Boundary condition: CTE_x = 0 at X_c = 0 (chip center / amplifier line).
    """
    return np.stack([X_c, X_c * Y_c, X_c ** 2], axis=1)


# ── CTE displacement computation ──────────────────────────────────────────────

def compute_cte_displacement(
    X_c: np.ndarray,
    Y_c: np.ndarray,
    mag: np.ndarray,
    dt: np.ndarray,
    chip_params: CTEChipParams,
) -> np.ndarray:
    """
    Compute CTE displacement in raw chip-centered pixel frame.

    Parameters
    ----------
    X_c, Y_c : (n,) centered GDC pixel positions
    mag       : (n,) instrumental magnitudes
    dt        : (n,) time since t_epoch0 in years (t_j − t_0)
    chip_params : CTEChipParams for this chip

    Returns
    -------
    delta_cte : (n, 2) array of (δCTE_x, δCTE_y) in pixels
    """
    phi  = phi_flux(mag, chip_params.delta)           # (n,)
    Phi  = dt * phi                                    # (n,) temporal amplitude × flux

    Bx = cte_x_basis(X_c, Y_c)                        # (n, 3)
    By = cte_y_basis(X_c, Y_c, chip_params.y_readout) # (n, 3)

    dcte_x = Phi * (Bx @ chip_params.gamma_x)         # (n,)
    dcte_y = Phi * (By @ chip_params.gamma_y)         # (n,)

    return np.stack([dcte_x, dcte_y], axis=1)          # (n, 2)


# ── Solver integration ────────────────────────────────────────────────────────

def _chip_from_image(img: str) -> str:
    """Return 'hi' or 'lo' from image name suffix."""
    return 'hi' if img.endswith('_hi') else 'lo'


def _epoch_from_image(img: str) -> int:
    """Return approximate epoch year from rootname prefix."""
    return 2006 if img.startswith('j9gz') else 2011


def apply_cte_to_solver(
    solver,
    image_names: list[str],
    cte_params: dict[str, CTEChipParams],
    t_epoch0_yr: float,
) -> None:
    """
    Apply CTE correction to solver._img_data[img]['xys'] for all images.

    On first call, stores 'xys_orig' for each image.  Subsequent calls update
    'xys' from 'xys_orig' + R_j @ δCTE, ensuring the correction is always
    applied to the original positions (not accumulated).

    Parameters
    ----------
    solver      : BP3MSolver instance
    image_names : list of sub_name strings
    cte_params  : {'hi': CTEChipParams, 'lo': CTEChipParams}
    t_epoch0_yr : reference epoch (Julian year) for temporal model
    """
    from astropy.time import Time

    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue

        # Store original xys on first call
        if 'xys_orig' not in d:
            d['xys_orig'] = d['xys'].copy()

        chip = _chip_from_image(img)
        if chip not in cte_params:
            continue

        # Image epoch
        hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                             format='mjd').jyear)
        dt = hst_yr - t_epoch0_yr   # scalar

        # Per-detection CTE displacement in chip-centered pixel frame
        X_c = d['X_c']   # (n,)
        Y_c = d['Y_c']   # (n,)
        mag = d.get('mag_inst')
        if mag is None or not np.isfinite(mag).any():
            continue

        delta_cte_raw = compute_cte_displacement(
            X_c, Y_c, mag, np.full(len(X_c), dt), cte_params[chip])  # (n, 2)

        # Rotate chip-frame correction to pseudo-image frame using R_j = solver.R[img]
        R_j = solver.R[img]                                            # (2, 2)
        delta_cte_pseudo = delta_cte_raw @ R_j.T                      # (n, 2)

        d['xys'] = d['xys_orig'] + delta_cte_pseudo


def remove_cte_from_solver(solver, image_names: list[str]) -> None:
    """Restore original xys for all images (set xys = xys_orig)."""
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None or 'xys_orig' not in d:
            continue
        d['xys'] = d['xys_orig'].copy()


# ── Residual collection ───────────────────────────────────────────────────────

def collect_cte_residuals(
    solver,
    image_names: list[str],
    r_hat: np.ndarray,
    t_epoch0_yr: float,
    data_root: Path,
    field_name: str,
    use_catalog_npz: bool = True,
) -> dict[str, dict]:
    """
    Collect per-chip GDC-frame residuals needed to update CTE parameters.

    Reads detections_catalog.npz (all ~127k stars) rather than the solver
    BP3M stars only, to maximize signal on faint stars where CTE is strongest.

    Returns
    -------
    residuals : {'hi': {...}, 'lo': {...}} each containing:
        'dx'    : (n,) x residual in GDC pixels
        'dy'    : (n,) y residual in GDC pixels
        'X_c'   : (n,) centered X position
        'Y_c'   : (n,) centered Y position
        'mag'   : (n,) instrumental magnitude
        'dt'    : (n,) time since t_epoch0_yr [years]
        'z'     : (n,) soft weights (1.0 if not available)
    """
    import pandas as pd
    from astropy.time import Time
    from bp3m.astro_utils import (
        plane_project, plane_project_jacobian, plane_project_tangent_derivs,
        get_tele_position, get_parallax_factors,
    )

    n_r = solver.N_R
    poly_order = solver.poly_order

    xmatch_dir = data_root / field_name / "hst_xmatch"
    cat_npz_path = (data_root / field_name / "BP3M_v2_results"
                    / "detections_catalog.npz")

    residuals = {chip: {'dx': [], 'dy': [], 'X_c': [], 'Y_c': [],
                         'mag': [], 'dt': [], 'z': []}
                 for chip in ('hi', 'lo')}

    if use_catalog_npz and cat_npz_path.exists():
        # Fast path: use precomputed GDC residuals from detections_catalog.npz.
        # These already include the full _save_full_catalog_residuals geometry.
        # We recompute the residuals relative to the CURRENT xys (CTE-corrected),
        # so we load the raw catalog positions and recompute.
        # For now fall through to the geometry path (catalog npz gives post-v2
        # residuals which are a reasonable starting point for the warm start).
        cat = np.load(cat_npz_path, allow_pickle=True)
        for j_idx, img in enumerate(image_names):
            if f'{img}_X_c' not in cat:
                continue
            chip = _chip_from_image(img)
            hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                                 format='mjd').jyear)
            dt = hst_yr - t_epoch0_yr

            X_c = cat[f'{img}_X_c'].astype(float)
            Y_c = cat[f'{img}_Y_c'].astype(float)
            dx  = cat[f'{img}_dx_gdc'].astype(float)
            dy  = cat[f'{img}_dy_gdc'].astype(float)
            mag = cat[f'{img}_mag_inst'].astype(float)

            ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)
            residuals[chip]['dx'].append(dx[ok])
            residuals[chip]['dy'].append(dy[ok])
            residuals[chip]['X_c'].append(X_c[ok])
            residuals[chip]['Y_c'].append(Y_c[ok])
            residuals[chip]['mag'].append(mag[ok])
            residuals[chip]['dt'].append(np.full(ok.sum(), dt))
            residuals[chip]['z'].append(np.ones(ok.sum()))

    else:
        # Fallback: recompute from solver's current _img_data
        print("    collect_cte_residuals: detections_catalog.npz not found, "
              "using solver stars only")
        for j_idx, img in enumerate(image_names):
            d = solver._img_data.get(img)
            if d is None:
                continue
            chip = _chip_from_image(img)
            hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                                 format='mjd').jyear)
            dt = hst_yr - t_epoch0_yr

            r_j   = r_hat[j_idx * n_r:(j_idx + 1) * n_r]
            X_c   = d['X_c']
            Y_c   = d['Y_c']
            mag   = d.get('mag_inst')
            if mag is None:
                continue

            # Current residuals in pseudo-image frame
            xys_orig = d.get('xys_orig', d['xys'])
            X_mat = d['X_mat']
            JU    = d['JU']
            sidx  = d['sidx']
            v_approx = np.zeros((len(sidx), 5))
            v_approx[:, 2] = solver.v_survey[sidx, 2]   # pmra
            v_approx[:, 3] = solver.v_survey[sidx, 3]   # pmdec
            v_approx[:, 4] = solver.v_survey[sidx, 4]   # plx
            pred = (np.einsum('nij,j->ni', X_mat, r_j)
                    - np.einsum('nij,nj->ni', JU, v_approx))
            resid_pseudo = xys_orig - pred   # (n, 2)

            if poly_order == 1:
                J_inv = np.linalg.inv(solver.R[img])
                dxy = resid_pseudo @ J_inv.T
            else:
                from bp3m.astro_utils import compute_poly_jacobian
                J_loc = compute_poly_jacobian(r_j, X_c, Y_c, poly_order)
                J_inv = np.linalg.inv(J_loc)
                dxy = np.einsum('nij,nj->ni', J_inv, resid_pseudo)

            ok = np.isfinite(dxy[:, 0]) & np.isfinite(mag)
            residuals[chip]['dx'].append(dxy[ok, 0])
            residuals[chip]['dy'].append(dxy[ok, 1])
            residuals[chip]['X_c'].append(X_c[ok])
            residuals[chip]['Y_c'].append(Y_c[ok])
            residuals[chip]['mag'].append(mag[ok])
            residuals[chip]['dt'].append(np.full(ok.sum(), dt))
            residuals[chip]['z'].append(np.ones(ok.sum()))

    # Concatenate per-chip lists
    for chip in ('hi', 'lo'):
        for key in residuals[chip]:
            arr = residuals[chip][key]
            residuals[chip][key] = np.concatenate(arr) if arr else np.array([])

    for chip in ('hi', 'lo'):
        n = len(residuals[chip]['dx'])
        print(f"    collect_cte_residuals: {chip} — {n:,} detections")

    return residuals


# ── CTE parameter update ──────────────────────────────────────────────────────

def update_cte_params(
    residuals_by_chip: dict[str, dict],
    cte_params: dict[str, CTEChipParams],
    n_inner: int = 5,
    delta_tol: float = 1e-4,
    regularize: float = 1e-6,
) -> tuple[dict[str, CTEChipParams], dict]:
    """
    Update CTE parameters (γ, δ) from current GDC-frame residuals.

    Uses Gauss-Newton inner iterations to solve for δ jointly with γ.
    The update for γ is linear (least squares); δ update is from the
    linearized (analytic gradient) column appended to the design matrix.

    Parameters
    ----------
    residuals_by_chip : output of collect_cte_residuals
    cte_params        : current parameters (modified in-place copy returned)
    n_inner           : Gauss-Newton iterations for δ
    delta_tol         : convergence threshold for |Δδ|
    regularize        : Tikhonov regularization coefficient (applied to γ)

    Returns
    -------
    new_params : updated CTEChipParams for each chip
    info       : convergence info dict
    """
    new_params = {c: cte_params[c].copy() for c in ('hi', 'lo')}
    info = {}

    for chip in ('hi', 'lo'):
        res = residuals_by_chip[chip]
        if len(res['dx']) == 0:
            info[chip] = {'converged': True, 'n_inner': 0}
            continue

        dx   = res['dx'].astype(float)
        dy   = res['dy'].astype(float)
        X_c  = res['X_c'].astype(float)
        Y_c  = res['Y_c'].astype(float)
        mag  = res['mag'].astype(float)
        dt   = res['dt'].astype(float)
        z    = res['z'].astype(float)

        # Remove invalid rows
        ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag) & np.isfinite(dt)
        ok &= (np.abs(dt) > 0)    # no residual from zero-time images
        dx, dy, X_c, Y_c, mag, dt, z = (arr[ok] for arr in (dx, dy, X_c, Y_c, mag, dt, z))
        n = len(dx)

        if n < 10:
            info[chip] = {'converged': True, 'n_inner': 0, 'n_det': n}
            continue

        p = new_params[chip]
        delta_n = float(p.delta)
        y_readout = p.y_readout

        delta_history = [delta_n]
        for it in range(n_inner):
            phi   = phi_flux(mag, delta_n)        # (n,)
            dphi  = dphi_ddelta(mag, delta_n)     # (n,) analytic gradient

            Phi   = dt * phi                      # (n,)
            dPhi  = dt * dphi                     # (n,) for δ gradient column

            Bx = cte_x_basis(X_c, Y_c)           # (n, 3)
            By = cte_y_basis(X_c, Y_c, y_readout) # (n, 3)

            # Design matrices for x and y (3 γ columns + 1 δ column)
            # dx = Φ·Bx·γ_x + dΦ·Bx·γ_x·Δδ  (linearized around current δ)
            # Collect: A_x = [Φ·Bx, dΦ·Bx·γ_x_current]  shape (n, 4)
            Bx_gamma_curr = Bx @ p.gamma_x   # (n,) current CTE_x prediction (γ factor)
            By_gamma_curr = By @ p.gamma_y   # (n,)

            Ax = np.column_stack([Phi[:, None] * Bx,           # (n, 3) γ_x columns
                                  dPhi * Bx_gamma_curr])        # (n, 1) Δδ column
            Ay = np.column_stack([Phi[:, None] * By,
                                  dPhi * By_gamma_curr])

            # Weighted normal equations: A^T W A θ = A^T W r
            # x and y share the δ column but solve independently per component,
            # then average the Δδ estimates.
            W = z                                               # (n,)

            def solve_wls(A, r):
                """Solve weighted least-squares with Tikhonov regularization."""
                AtW  = A.T * W[None, :]             # (4, n)
                AtWA = AtW @ A                       # (4, 4)
                AtWr = AtW @ r                       # (4,)
                # Regularize γ columns only (first 3)
                reg = np.zeros(4)
                reg[:3] = regularize
                AtWA += np.diag(reg)
                try:
                    theta = np.linalg.solve(AtWA, AtWr)
                except np.linalg.LinAlgError:
                    return None
                return theta

            theta_x = solve_wls(Ax, dx)
            theta_y = solve_wls(Ay, dy)

            if theta_x is None or theta_y is None:
                break

            # Update γ
            p.gamma_x = theta_x[:3]
            p.gamma_y = theta_y[:3]

            # Average Δδ from x and y updates
            delta_delta = 0.5 * (theta_x[3] + theta_y[3])
            delta_n = delta_n + delta_delta
            p.delta = delta_n
            delta_history.append(delta_n)

            if abs(delta_delta) < delta_tol:
                break

        rms_x = float(np.sqrt(np.mean((dx - dt * phi_flux(mag, p.delta) *
                                        (cte_x_basis(X_c, Y_c) @ p.gamma_x)) ** 2)))
        rms_y = float(np.sqrt(np.mean((dy - dt * phi_flux(mag, p.delta) *
                                        (cte_y_basis(X_c, Y_c, y_readout) @ p.gamma_y)) ** 2)))

        info[chip] = {
            'n_det': n,
            'delta_history': delta_history,
            'rms_x': rms_x,
            'rms_y': rms_y,
            'n_inner': len(delta_history) - 1,
        }
        print(f"    CTE update {chip}: δ={p.delta:.4f}  "
              f"|γ_y|={np.linalg.norm(p.gamma_y):.4e}  "
              f"|γ_x|={np.linalg.norm(p.gamma_x):.4e}  "
              f"rms_y={rms_y:.4f}px  rms_x={rms_x:.4f}px  "
              f"n={n:,}")

    return new_params, info


# ── Warm start ────────────────────────────────────────────────────────────────

def warm_start_cte(
    cat_npz_path: Path,
    image_names: list[str],
    solver,
    t_epoch0_yr: float,
) -> dict[str, CTEChipParams]:
    """
    Estimate initial γ from BP3M v2 post-fit residuals in detections_catalog.npz.

    Assumes δ = 1.0 (linear flux model) and regresses the existing GDC residuals
    onto the CTE basis to provide a starting point for the outer loop.

    Returns initial CTEChipParams for each chip.
    """
    from astropy.time import Time

    params = default_cte_params()

    if not cat_npz_path.exists():
        print("  warm_start_cte: detections_catalog.npz not found — using δ=1, γ=0")
        return params

    print("  Warm-starting CTE model from detections_catalog.npz residuals...")
    cat = np.load(cat_npz_path, allow_pickle=True)

    chip_data = {c: {'dx': [], 'dy': [], 'X_c': [], 'Y_c': [],
                      'mag': [], 'dt': []} for c in ('hi', 'lo')}

    for img in image_names:
        if f'{img}_X_c' not in cat:
            continue
        chip = _chip_from_image(img)
        hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                             format='mjd').jyear)
        dt = hst_yr - t_epoch0_yr
        if abs(dt) < 1e-3:
            continue   # skip reference epoch (no differential CTE signal)

        X_c = cat[f'{img}_X_c'].astype(float)
        Y_c = cat[f'{img}_Y_c'].astype(float)
        dx  = cat[f'{img}_dx_gdc'].astype(float)
        dy  = cat[f'{img}_dy_gdc'].astype(float)
        mag = cat[f'{img}_mag_inst'].astype(float)

        ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)
        for key, arr in zip(('dx','dy','X_c','Y_c','mag','dt'),
                             (dx[ok], dy[ok], X_c[ok], Y_c[ok], mag[ok],
                              np.full(ok.sum(), dt))):
            chip_data[chip][key].append(arr)

    for chip in ('hi', 'lo'):
        for key in chip_data[chip]:
            arr = chip_data[chip][key]
            chip_data[chip][key] = np.concatenate(arr) if arr else np.array([])

        cd = chip_data[chip]
        n = len(cd['dx'])
        if n < 10:
            continue

        # Linear regression with δ = 1.0 fixed
        delta_fixed = 1.0
        phi = phi_flux(cd['mag'], delta_fixed)
        dt  = cd['dt']
        Phi = dt * phi
        Bx = cte_x_basis(cd['X_c'], cd['Y_c'])
        By = cte_y_basis(cd['X_c'], cd['Y_c'], params[chip].y_readout)

        Ax = Phi[:, None] * Bx   # (n, 3)
        Ay = Phi[:, None] * By

        try:
            gamma_x, _, _, _ = np.linalg.lstsq(Ax, cd['dx'], rcond=None)
            gamma_y, _, _, _ = np.linalg.lstsq(Ay, cd['dy'], rcond=None)
        except np.linalg.LinAlgError:
            continue

        params[chip].delta   = delta_fixed
        params[chip].gamma_x = gamma_x
        params[chip].gamma_y = gamma_y

        rms_y = float(np.sqrt(np.mean((cd['dy'] - Ay @ gamma_y) ** 2)))
        print(f"  Warm start {chip}: δ=1.0  "
              f"|γ_y|={np.linalg.norm(gamma_y):.4e}  "
              f"rms_y={rms_y:.4f}px  n={n:,}")

    return params


# ── Diagnostic save ───────────────────────────────────────────────────────────

def _save_cte_residuals(
    output_dir: Path,
    solver,
    image_names: list[str],
    r_hat: np.ndarray,
    data_root: Path,
    field_name: str,
    suffix: str = '_cte',
) -> None:
    """Save full-catalog residuals after CTE correction (same format as detections_catalog.npz)."""
    from .run_alignment_v2 import _save_full_catalog_residuals
    try:
        _save_full_catalog_residuals(
            output_dir, solver, image_names, r_hat, data_root, field_name)
        # Rename to detections_catalog{suffix}.npz
        src = output_dir / 'detections_catalog.npz'
        dst = output_dir / f'detections_catalog{suffix}.npz'
        if src.exists():
            src.rename(dst)
            print(f"  Renamed → {dst.name}")
    except Exception as exc:
        import traceback
        print(f"  WARNING: _save_cte_residuals failed — {exc}")
        traceback.print_exc()


def _save_cte_convergence(
    output_dir: Path,
    outer_iter: int,
    cte_params: dict[str, CTEChipParams],
    info: dict,
) -> None:
    """Append one row to cte_convergence.csv."""
    import csv
    csv_path = output_dir / 'cte_convergence.csv'
    fieldnames = ['iter', 'chip', 'delta', 'gamma_y0', 'gamma_y1', 'gamma_y2',
                  'gamma_x0', 'gamma_x1', 'gamma_x2', 'rms_y', 'rms_x', 'n_det']
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for chip in ('hi', 'lo'):
            p = cte_params[chip]
            ci = info.get(chip, {})
            writer.writerow({
                'iter':     outer_iter,
                'chip':     chip,
                'delta':    f'{p.delta:.6f}',
                'gamma_y0': f'{p.gamma_y[0]:.6e}',
                'gamma_y1': f'{p.gamma_y[1]:.6e}',
                'gamma_y2': f'{p.gamma_y[2]:.6e}',
                'gamma_x0': f'{p.gamma_x[0]:.6e}',
                'gamma_x1': f'{p.gamma_x[1]:.6e}',
                'gamma_x2': f'{p.gamma_x[2]:.6e}',
                'rms_y':    f'{ci.get("rms_y", np.nan):.6f}',
                'rms_x':    f'{ci.get("rms_x", np.nan):.6f}',
                'n_det':    ci.get('n_det', 0),
            })


# ── Main function ──────────────────────────────────────────────────────────────

def run_alignment_cte(
    output_dir: Path,
    field_name: str,
    n_iter_bp3m: int = 10,
    n_iter_cte: int = 8,
    n_samples: int = 1000,
    clip_sigma: float = 4.5,
    poly_order: int = 1,
    use_sparse: bool = False,
    no_plots: bool = False,
    hst_enable_iter: int = 5,
    hst_max_pm_unc: float = 5.0,
    hst_max_per_image: int = 1000,
    outlier_sigma: float = 5.0,
    use_influence_clip: bool = True,
    influence_d_thresh: float = 1.0,
    influence_sigma_min: float = 2.0,
    hst_pm_sigma_diffuse: float = 100.0,
    pos_err_floor: float = 5e-3,
    det_chi2_threshold: float | None = None,
    use_soft_weights: bool = False,
    student_t_nu: float = 50.0,
    cte_delta_tol: float = 1e-3,
    cte_gamma_rtol: float = 1e-3,
    n_inner_delta: int = 5,
    bp3m_dir: Path | None = None,
) -> Path:
    """
    Run joint CTE + astrometry alignment using the master_combined_v2.csv catalog.

    This is a wrapper around BP3M v2 that adds an outer Gauss-Newton loop for the
    CTE parameters.  In each outer iteration:
      1. Apply current CTE model to solver xys (modify solver._img_data[img]['xys']).
      2. Run BP3M v2 solve for (r_j, v_i) with CTE-corrected positions.
      3. Collect full-catalog residuals (dx_gdc, dy_gdc for all ~127k stars).
      4. Update CTE parameters (γ_c, δ_c) from residuals via Gauss-Newton.
      5. Check convergence; break early if ‖Δγ‖/‖γ‖ < cte_gamma_rtol.

    Parameters
    ----------
    output_dir      : pipeline root directory
    field_name      : field subdirectory name
    n_iter_bp3m     : BP3M EM iterations per CTE outer iteration
    n_iter_cte      : maximum CTE outer Gauss-Newton iterations
    n_samples       : posterior samples for final marginalisation
    (remaining params same as run_alignment_v2)

    Returns
    -------
    Path to output directory ({output_dir}/{field}/BP3M_cte_results/)
    """
    from bp3m.data_loader import build_index_maps
    from bp3m.solver import BP3MSolver
    from bp3m.solver_sparse import BP3MSolverSparse
    from astropy.time import Time
    import pandas as pd

    from .data_loader_master import load_master_v2
    from .run_alignment_v2 import (
        V2AlignmentCallback,
        _save_full_catalog_residuals,
        _plot_soft_weights,
    )
    from .run_alignment import _save_results

    data_root   = Path(output_dir)
    output_cte  = data_root / field_name / "BP3M_cte_results"
    output_cte.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 60)
    print("BP3M CTE: joint CTE + astrometry alignment")
    print("─" * 60)
    print(f"  n_iter_bp3m={n_iter_bp3m}  n_iter_cte={n_iter_cte}  "
          f"poly_order={poly_order}")

    # ── Load data (same as v2) ─────────────────────────────────────────────────
    print(f"\n  Loading v2 master catalog for '{field_name}'...")
    images, stars_per_image, gaia_catalog, hst_only_mask = load_master_v2(
        data_root, field_name,
        hst_max_pm_unc=hst_max_pm_unc,
        hst_max_per_image=hst_max_per_image,
        pos_err_floor=pos_err_floor,
        det_chi2_threshold=det_chi2_threshold,
    )

    if not images:
        raise RuntimeError(f"No usable images for '{field_name}'.")

    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)
    imgs = {n: images[n] for n in image_names if n in images}
    filtered_spi = {n: stars_per_image[n] for n in image_names}

    print(f"  Stars: {len(gaia_catalog)} "
          f"({int((~hst_only_mask).sum())} Gaia + {int(hst_only_mask.sum())} HST-only)  "
          f"Images: {len(image_names)}")

    # ── Reference epoch: first exposure ───────────────────────────────────────
    # t_epoch0 is the absolute reference for the temporal CTE model.
    # Using the first exposure avoids unidentifiability of the absolute CTE level.
    all_mjds = [float(images[img]['hst_time_mjd']) for img in image_names
                if img in images]
    t_epoch0_mjd = float(min(all_mjds))
    t_epoch0_yr  = float(Time(t_epoch0_mjd, format='mjd').jyear)
    print(f"  t_epoch0 = {t_epoch0_yr:.4f} yr  "
          f"(MJD {t_epoch0_mjd:.2f}, {Time(t_epoch0_mjd, format='mjd').isot[:10]})")

    # ── Inject v1 BP3M transformation (warm start) ────────────────────────────
    v1_bp3m_dir   = data_root / field_name / "BP3M_results"
    v1_xform_path = v1_bp3m_dir / "image_transformations.csv"
    v1_abcdwz: dict[str, np.ndarray] = {}
    if v1_xform_path.exists():
        v1_df = pd.read_csv(v1_xform_path)
        for _, row in v1_df.iterrows():
            img_key = str(row["image_name"])
            v1_abcdwz[img_key] = np.array([
                float(row["a"]), float(row["b"]),
                float(row["c"]), float(row["d"]),
                float(row["w"]), float(row["z"]),
            ])
        imgs = {sub: dict(meta) for sub, meta in imgs.items()}
        for sub, meta in imgs.items():
            if sub in v1_abcdwz:
                meta["fcm_abcdwz"] = v1_abcdwz[sub]
        print(f"  Loaded v1 BP3M: {len(v1_abcdwz)} images as initialization")

    # ── Initialise solver ──────────────────────────────────────────────────────
    SolverClass = BP3MSolverSparse if use_sparse else BP3MSolver
    solver = SolverClass(
        imgs, filtered_spi, gaia_catalog,
        star_id_to_idx, image_names, star_in_image,
        poly_order=poly_order,
    )

    # Load mag_inst into _img_data (needed for CTE calculation)
    # The mag_inst arrays are available in stars_per_image DataFrames.
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        spi_df = filtered_spi.get(img)
        if spi_df is None or 'mag_gdc' not in spi_df.columns:
            continue
        gid_col = spi_df['Gaia_id'].to_numpy(dtype=np.int64)
        # Reorder to match sidx order in d
        mag_by_gid = {int(gid_col[k]): float(spi_df['mag_gdc'].iloc[k])
                      for k in range(len(spi_df))}
        sidx = d['sidx']
        gc_ids = gaia_catalog['Gaia_id'].to_numpy(dtype=np.int64)
        mag_arr = np.full(len(sidx), np.nan)
        for k, s in enumerate(sidx):
            gid = int(gc_ids[s])
            if gid in mag_by_gid:
                mag_arr[k] = mag_by_gid[gid]
        d['mag_inst'] = mag_arr

    # ── PM seed and HST-only diffuse prior ────────────────────────────────────
    if hst_pm_sigma_diffuse != 100.0:
        hst_star_indices = np.where(hst_only_mask)[0]
        if len(hst_star_indices) > 0:
            sigma_pm_inv2 = float(hst_pm_sigma_diffuse) ** -2
            solver._C_VG_inv_per_star[hst_star_indices, 2] = sigma_pm_inv2
            solver._C_VG_inv_per_star[hst_star_indices, 3] = sigma_pm_inv2

    _n_stars = len(gaia_catalog)
    pm_init = np.full((_n_stars, 2), np.nan)
    if "pmra_xmatch" in gaia_catalog.columns:
        pm_init[:, 0] = pd.to_numeric(
            gaia_catalog["pmra_xmatch"], errors='coerce').fillna(np.nan).values
        pm_init[:, 1] = pd.to_numeric(
            gaia_catalog["pmdec_xmatch"], errors='coerce').fillna(np.nan).values

    callback = V2AlignmentCallback(
        hst_star_mask=hst_only_mask,
        hst_enable_iter=hst_enable_iter,
        outlier_sigma=outlier_sigma,
        pm_init=pm_init,
    )

    # ── Phase 0: fixed-transform pre-filter (same as v2) ─────────────────────
    if v1_abcdwz:
        r_init_hat = np.concatenate([solver._img_data[img]["r_init"]
                                      for img in image_names])
        solver._update_R(r_init_hat)
        solver._update_geometry(r_init_hat, solver.v_survey)
        print("\n  Phase 0: fixed-transform pre-filter (v1 BP3M posterior)")
        n_flag0 = 0
        for img in image_names:
            d = solver._img_data.get(img)
            if d is None:
                continue
            j_idx = image_names.index(img)
            r_j   = r_init_hat[j_idx * solver.N_R:(j_idx + 1) * solver.N_R]
            use   = d["use_for_fit"].copy()
            _v_pm = np.zeros_like(solver.v_survey[d["sidx"]])
            _v_pm[:, 2:] = solver.v_survey[d["sidx"], 2:]
            motion    = np.einsum("nij,nj->ni", d["JU"], _v_pm)
            x_pred    = np.einsum("nkl,l->nk", d["X_mat"], r_j) - motion
            resid_mag = np.hypot(*(d["xys"] - x_pred).T)
            if use.any():
                r_align   = resid_mag[use]
                mad_sigma = np.median(np.abs(r_align - np.median(r_align))) / 0.6745
                thresh    = max(5.0 * mad_sigma, 0.3)
                bad       = use & (resid_mag > thresh)
                n_flag0  += int(bad.sum())
                if bad.any():
                    d["use_for_fit"][bad]     = False
                    d["use_for_fit_max"][bad] = False
                    if "use_for_astrom" in d:
                        d["use_for_astrom"][bad] = False
        print(f"  Phase 0: {n_flag0} detections flagged")

    # ── CTE warm start ────────────────────────────────────────────────────────
    cat_npz_path = data_root / field_name / "BP3M_v2_results" / "detections_catalog.npz"
    cte_params = warm_start_cte(cat_npz_path, image_names, solver, t_epoch0_yr)

    # ── Outer CTE + BP3M Gauss-Newton loop ────────────────────────────────────
    clip = clip_sigma if clip_sigma > 0 else None
    _min_outer = max(hst_enable_iter + 3, 4) if n_iter_bp3m >= hst_enable_iter else 4

    r_hat = C_r = v_hat = C_vT = a_arr = K_img = z_weights_out = None
    conv_history = []

    print(f"\n  Starting CTE outer loop ({n_iter_cte} iterations)...")
    for cte_iter in range(n_iter_cte):
        print(f"\n  ─── CTE iteration {cte_iter + 1}/{n_iter_cte} ───")
        t_iter = time.time()

        # Step 1: Apply CTE to solver xys
        if cte_iter > 0:
            # Only apply non-trivial CTE after first iteration (warm start may be zero)
            apply_cte_to_solver(solver, image_names, cte_params, t_epoch0_yr)
        else:
            # Store xys_orig without applying (zero CTE at iteration 0)
            for img in image_names:
                d = solver._img_data.get(img)
                if d is not None and 'xys_orig' not in d:
                    d['xys_orig'] = d['xys'].copy()

        # Step 2: BP3M solve
        print(f"  BP3M solve ({n_iter_bp3m} EM iterations)...")
        (r_hat, C_r, v_hat, C_vT,
         a_arr, K_img, z_weights_out) = solver.fit(
            n_iter=n_iter_bp3m,
            clip_sigma=clip,
            inflate_hst_errors=True,
            inflate_from_iter=0,
            min_outer_iters=_min_outer,
            prefilter=(cte_iter == 0),   # prefilter only on first BP3M solve
            use_influence_clip=use_influence_clip,
            influence_d_thresh=influence_d_thresh,
            influence_sigma_min=influence_sigma_min,
            use_two_tier=True,
            per_iter_callback=callback if cte_iter == 0 else None,
            use_soft_weights=use_soft_weights,
            student_t_nu=student_t_nu,
        )
        print(f"  BP3M done ({time.time() - t_iter:.1f}s)")

        # Step 3: Collect full-catalog residuals
        residuals = collect_cte_residuals(
            solver, image_names, r_hat, t_epoch0_yr,
            data_root, field_name,
            use_catalog_npz=(cte_iter == 0),  # use precomputed npz for warm start
        )

        # Step 4: Update CTE parameters
        gamma_before = {c: cte_params[c].gamma_y.copy() for c in ('hi', 'lo')}
        cte_params, info = update_cte_params(
            residuals, cte_params,
            n_inner=n_inner_delta,
            delta_tol=cte_delta_tol,
        )
        _save_cte_convergence(output_cte, cte_iter + 1, cte_params, info)

        # Step 5: Convergence check
        gamma_rchg = max(
            np.linalg.norm(cte_params[c].gamma_y - gamma_before[c])
            / max(np.linalg.norm(gamma_before[c]), 1e-10)
            for c in ('hi', 'lo')
        )
        conv_history.append({'iter': cte_iter + 1, 'gamma_rchg': gamma_rchg,
                              **{f'delta_{c}': cte_params[c].delta
                                  for c in ('hi', 'lo')}})
        print(f"  γ relative change = {gamma_rchg:.4e}  "
              f"δ_hi={cte_params['hi'].delta:.4f}  δ_lo={cte_params['lo'].delta:.4f}")

        if gamma_rchg < cte_gamma_rtol and cte_iter >= 2:
            print(f"  CTE converged at iteration {cte_iter + 1}")
            break

    # ── Final BP3M solve with converged CTE ───────────────────────────────────
    print("\n  Final BP3M solve with converged CTE parameters...")
    apply_cte_to_solver(solver, image_names, cte_params, t_epoch0_yr)
    (r_hat, C_r, v_hat, C_vT,
     a_arr, K_img, z_weights_out) = solver.fit(
        n_iter=n_iter_bp3m,
        clip_sigma=clip,
        inflate_hst_errors=True,
        inflate_from_iter=0,
        min_outer_iters=_min_outer,
        prefilter=False,
        use_influence_clip=use_influence_clip,
        influence_d_thresh=influence_d_thresh,
        influence_sigma_min=influence_sigma_min,
        use_two_tier=True,
        per_iter_callback=None,
        use_soft_weights=use_soft_weights,
        student_t_nu=student_t_nu,
    )

    # ── Save CTE parameters ────────────────────────────────────────────────────
    cte_params_out = {}
    for chip in ('hi', 'lo'):
        p = cte_params[chip]
        cte_params_out[f'{chip}_delta']   = np.array([p.delta])
        cte_params_out[f'{chip}_gamma_x'] = p.gamma_x
        cte_params_out[f'{chip}_gamma_y'] = p.gamma_y
        cte_params_out[f'{chip}_y_readout'] = np.array([p.y_readout])
    cte_params_out['t_epoch0_yr'] = np.array([t_epoch0_yr])
    np.savez(output_cte / 'cte_params.npz', **cte_params_out)
    print(f"  Saved: cte_params.npz")

    # ── Sample posteriors ──────────────────────────────────────────────────────
    print(f"  Drawing {n_samples} posterior samples...")
    r_samp, v_mean, v_cov = solver.sample_posteriors(
        r_hat, C_r, a_arr, K_img, C_vT, n_samples=n_samples)

    # ── Save results (same format as v2) ──────────────────────────────────────
    _save_results(
        output_cte, solver, imgs, gaia_catalog, image_names,
        r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
        run_config={
            "n_iter_bp3m":    n_iter_bp3m,
            "n_iter_cte":     n_iter_cte,
            "n_samples":      n_samples,
            "clip_sigma":     clip_sigma,
            "poly_order":     poly_order,
            "t_epoch0_yr":    t_epoch0_yr,
            **{f'delta_{c}':  float(cte_params[c].delta) for c in ('hi', 'lo')},
        },
    )

    # ── Save post-CTE full-catalog residuals ──────────────────────────────────
    _save_cte_residuals(output_cte, solver, image_names, r_hat,
                        data_root, field_name, suffix='_cte')

    # ── Soft-weight output ─────────────────────────────────────────────────────
    if use_soft_weights and z_weights_out is not None:
        rows = []
        for img, z in z_weights_out.items():
            if z is None:
                continue
            d = solver._img_data[img]
            for k in range(len(z)):
                rows.append({'image': img, 'star_idx': int(d['sidx'][k]),
                             'z_det': float(z[k])})
        import csv as _csv
        zcsv = output_cte / 'soft_weights.csv'
        with open(zcsv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=['image', 'star_idx', 'z_det'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: soft_weights.csv  ({len(rows)} detection weights)")

    # ── Diagnostic plots ───────────────────────────────────────────────────────
    if not no_plots:
        try:
            from bp3m.pipeline.plot_results import make_plots
            print("  Generating diagnostic plots...")
            make_plots(solver, imgs, gaia_catalog,
                       r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
                       output_dir=output_cte)
        except Exception as exc:
            print(f"  WARNING: plots failed — {exc}")

    print(f"\n  CTE results written to: {output_cte}")
    return output_cte
