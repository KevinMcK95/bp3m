"""
run_alignment_cte.py  —  Joint CTE + astrometry model for HST ACS/WFC.

Extends BP3M v2 alignment to simultaneously fit per-image transformations (r_j),
stellar astrometry (v_i), and a parametric CTE model θ_CTE.

See docs/cte_joint_model.md for full mathematical derivation and design decisions.

CTE model summary
-----------------
For chip c (hi/lo), detection k in image j (epoch t_j):

  δCTE_x_k = (t_j − t_0) · φ(mag_k; δ_c) · b_x(X_k, Y_k) · γ_x_c
  δCTE_y_k = (t_j − t_0) · φ(mag_k; δ_c) · b_y(X_k, Y_k) · γ_y_c

  φ(mag; δ) = 10^{0.4·δ·(mag − mag_ref)} − 1
  b_y = [Y', Xc·Y', Xc²·Y', Y'², Xc·Y'², Xc²·Y'²]  (6 terms, CTE_y=0 at Y'=0)
  b_x = [Xc, Xc·Y', Xc·Y'², Xc², Xc²·Y', Xc²·Y'²]  (6 terms, CTE_x=0 at Xc=0)
  Y' = y_raw − y_readout_raw  (distance from readout register in raw pixels)
  Xc = x_gdc − 2048           (x-position centred on detector)
  γ_x_c: (6,) coefficients;  γ_y_c: (6,) coefficients
  δ_c: shared flux exponent for chip c

Parameters: 13 total per chip: δ(1) + γ_x(6) + γ_y(6) (two chips → 26 params).

Note on y_raw coordinate system: py1pass stores pixel y in a unified global frame
(0..~4096). lo chip occupies y_raw ∈ [8, 2039]; hi chip occupies y_raw ∈ [2056, 4087].
Both chips read AWAY from the gap (CTE trails toward high y_raw), so dy_gdc > 0 and
increases with Y' for both chips. 6-term basis spans the full 2D (Xc, Y') space while
preserving the boundary condition CTE=0 at the readout register.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── ACS/WFC chip geometry constants ──────────────────────────────────────────
# GDC-frame: Y_c = y_gdc − 2048 (centred on detector)
_HI_Y_READOUT = +2047.0   # readout register for _hi chip in GDC-centred frame
_LO_Y_READOUT = -2048.0   # readout register for _lo chip in GDC-centred frame
# Raw-frame: py1pass stores y in a unified global frame (0..~4096).
# lo chip: y_raw ∈ [8, 2039]; readout at the gap edge → y_readout_raw ≈ 0
# hi chip: y_raw ∈ [2056, 4087]; readout at the gap edge → y_readout_raw ≈ 2048
# Both chips read AWAY from the gap (CTE trails toward high y_raw).
# Empirically confirmed: dy_gdc > 0 and increasing with y_raw for both chips.
_HI_Y_READOUT_RAW = 2048.0
_LO_Y_READOUT_RAW = 0.0
_MAG_REF      = -15.0   # just below the brightest instrumental mag (~-14.5); ensures phi>0 for all stars


# ── CTE parameter dataclass ───────────────────────────────────────────────────

@dataclass
class CTEChipParams:
    chip: str
    y_readout: float          # GDC-centred readout Y (kept for diagnostics/legacy)
    y_readout_raw: float      # raw chip-local readout Y (used for CTE basis)
    delta: float = 1.0
    gamma_x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gamma_y: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def copy(self) -> 'CTEChipParams':
        return CTEChipParams(chip=self.chip,
                             y_readout=self.y_readout,
                             y_readout_raw=self.y_readout_raw,
                             delta=float(self.delta),
                             gamma_x=self.gamma_x.copy(),
                             gamma_y=self.gamma_y.copy())


def default_cte_params() -> dict[str, CTEChipParams]:
    return {
        'hi': CTEChipParams(chip='hi',
                            y_readout=_HI_Y_READOUT,
                            y_readout_raw=_HI_Y_READOUT_RAW),
        'lo': CTEChipParams(chip='lo',
                            y_readout=_LO_Y_READOUT,
                            y_readout_raw=_LO_Y_READOUT_RAW),
    }


# ── Chip classification ────────────────────────────────────────────────────────

def _chip_from_image(img: str) -> str | None:
    """Return 'hi', 'lo', or None for merged (unsplit) images."""
    if img.endswith('_hi'):
        return 'hi'
    if img.endswith('_lo'):
        return 'lo'
    return None   # merged image — no CTE model applies


# ── Flux power-law model ──────────────────────────────────────────────────────

def phi_flux(mag: np.ndarray, delta: float,
             mag_ref: float = _MAG_REF) -> np.ndarray:
    """φ(mag; δ) = 10^{0.4·δ·(mag − mag_ref)} − 1."""
    return np.power(10.0, 0.4 * delta * (np.asarray(mag, dtype=float) - mag_ref)) - 1.0


def dphi_ddelta(mag: np.ndarray, delta: float,
                mag_ref: float = _MAG_REF) -> np.ndarray:
    """dφ/dδ = 0.4·ln(10)·(mag − mag_ref)·10^{0.4·δ·(mag − mag_ref)}."""
    dm = np.asarray(mag, dtype=float) - mag_ref
    return 0.4 * np.log(10.0) * dm * np.power(10.0, 0.4 * delta * dm)


# ── CTE basis functions ───────────────────────────────────────────────────────

def cte_y_basis(X_c: np.ndarray, y_raw: np.ndarray,
                y_readout_raw: float) -> np.ndarray:
    """b_y = [Y', Xc·Y', Xc²·Y', Y'², Xc·Y'², Xc²·Y'²]  (6 terms).

    Y' = y_raw − y_readout_raw (distance from readout register in raw pixels).
    Boundary condition: all terms vanish at Y'=0 (readout register).
    Column normalization in callers is essential — the 6 terms span ~10 decades
    in magnitude (Y' vs Xc²·Y'²), so without scaling the lstsq condition number
    exceeds 1e10 and kills the higher-order signal.
    """
    Yp = y_raw - y_readout_raw
    return np.stack([Yp, X_c*Yp, X_c**2*Yp, Yp**2, X_c*Yp**2, X_c**2*Yp**2], axis=1)


def cte_x_basis(X_c: np.ndarray, y_raw: np.ndarray,
                y_readout_raw: float) -> np.ndarray:
    """b_x = [Xc, Xc·Y', Xc·Y'², Xc², Xc²·Y', Xc²·Y'²]  (6 terms).

    Y' = y_raw − y_readout_raw (same raw-frame distance from readout as y-basis).
    Boundary condition: all terms vanish at Xc=0 (centre between serial amplifiers).
    """
    Yp = y_raw - y_readout_raw
    return np.stack([X_c, X_c*Yp, X_c*Yp**2, X_c**2, X_c**2*Yp, X_c**2*Yp**2], axis=1)


# ── CTE displacement computation ──────────────────────────────────────────────

def compute_cte_displacement(
    X_c: np.ndarray, y_raw: np.ndarray,
    mag: np.ndarray, dt: np.ndarray,
    chip_params: CTEChipParams,
) -> np.ndarray:
    """
    CTE displacement in GDC pixel frame (after applying raw→GDC ≈ identity).

    X_c : GDC-centred x coordinate [px]
    y_raw : raw chip-local y coordinate [px] (pre-GDC, 0..2047)

    Returns (n, 2) array of (δCTE_x, δCTE_y) in pixels.
    """
    phi  = phi_flux(mag, chip_params.delta)
    Phi  = dt * phi
    Bx   = cte_x_basis(X_c, y_raw, chip_params.y_readout_raw)
    By   = cte_y_basis(X_c, y_raw, chip_params.y_readout_raw)
    return np.stack([Phi * (Bx @ chip_params.gamma_x),
                     Phi * (By @ chip_params.gamma_y)], axis=1)


# ── mag_inst injection ────────────────────────────────────────────────────────

def _inject_mag_inst(solver, image_names: list, filtered_spi: dict,
                     gaia_catalog) -> None:
    """
    Inject per-detection instrumental mag into solver._img_data[img]['mag_inst'].

    Uses filtered_spi[img]['mag'] (from FITS catalog, already loaded by
    data_loader_master).  Maps via Gaia_id lookup — works for both positive
    Gaia-matched stars and negative HST-only stars.
    """
    gc_ids = gaia_catalog['Gaia_id'].to_numpy(dtype=np.int64)

    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        spi_df = filtered_spi.get(img)
        if spi_df is None or 'mag_gdc' not in spi_df.columns:
            continue

        # Build Gaia_id → mag_gdc lookup (vectorized, no iterrows)
        _gids = spi_df['Gaia_id'].to_numpy(dtype=np.int64)
        _mags = spi_df['mag_gdc'].to_numpy(dtype=float)
        gid_to_mag = {int(_gids[k]): float(_mags[k]) for k in range(len(spi_df))}

        sidx = d['sidx']
        mag_arr = np.array([gid_to_mag.get(int(gc_ids[s]), np.nan) for s in sidx])
        d['mag_inst'] = mag_arr


# ── Solver CTE correction ─────────────────────────────────────────────────────

def apply_cte_to_solver(
    solver,
    image_names: list[str],
    cte_params: dict[str, CTEChipParams],
    t_epoch0_yr: float,
    filtered_spi: dict | None = None,
) -> None:
    """
    Apply CTE correction to solver._img_data[img]['xys'] for all images.

    Stores 'xys_orig' on first call.  Each subsequent call recomputes from
    xys_orig so corrections don't accumulate.

    filtered_spi : stars_per_image dict (same object passed to the solver).
        If provided, raw chip-local y ('Y_orig') is used for the CTE y-basis.
        If None, falls back to an approximation from GDC Y_c.

    Skips images where chip cannot be determined (merged/unsplit images) or
    where mag_inst is missing / entirely NaN.
    """
    from astropy.time import Time

    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue

        if 'xys_orig' not in d:
            d['xys_orig'] = d['xys'].copy()

        chip = _chip_from_image(img)
        if chip is None or chip not in cte_params:
            continue

        mag = d.get('mag_inst')
        if mag is None:
            continue
        ok = np.isfinite(mag)
        if not ok.any():
            continue

        hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                             format='mjd').jyear)
        dt_scalar = hst_yr - t_epoch0_yr
        dt = np.full(len(mag), dt_scalar)

        X_c = d['X_c']   # GDC-centred x

        # Raw chip-local y for CTE y-basis
        if filtered_spi is not None and img in filtered_spi:
            spi_df = filtered_spi[img]
            if 'Y_orig' in spi_df.columns:
                y_raw = spi_df['Y_orig'].to_numpy(float)
            else:
                # Fallback: approximate from GDC
                Y_c = d['Y_c']
                y_raw = (Y_c + 1.0) if chip == 'hi' else (Y_c + 2048.0)
        else:
            Y_c = d['Y_c']
            y_raw = (Y_c + 1.0) if chip == 'hi' else (Y_c + 2048.0)

        # Guard against length mismatch (shouldn't happen but be safe)
        if len(y_raw) != len(mag):
            Y_c = d['Y_c']
            y_raw = (Y_c + 1.0) if chip == 'hi' else (Y_c + 2048.0)

        delta_cte_raw = np.zeros((len(mag), 2))
        delta_cte_raw[ok] = compute_cte_displacement(
            X_c[ok], y_raw[ok], mag[ok], dt[ok], cte_params[chip])

        R_j = solver.R[img]                             # (2, 2)
        delta_cte_pseudo = delta_cte_raw @ R_j.T        # (n, 2)
        d['xys'] = d['xys_orig'] + delta_cte_pseudo


# ── Residual collection ───────────────────────────────────────────────────────

def collect_cte_residuals(
    img_to_df: dict,
    solver,
    image_names: list[str],
    r_hat: np.ndarray,
    t_epoch0_yr: float,
    field_mean_pm: tuple[float, float] | None = None,
) -> dict[str, dict]:
    """
    Collect per-chip GDC-frame residuals for updating CTE parameters.

    Uses the full master-catalog detection set (all ~127k stars via img_to_df),
    not just the solver's BP3M alignment stars.  img_to_df is the output of
    _load_full_catalog_df and should be cached across CTE iterations.

    Parameters
    ----------
    field_mean_pm : (pmra_mean, pmdec_mean) in mas/yr, or None.
        If provided, ALL stars use this common PM for residual computation.
        This removes the CTE absorbed into individual HST-derived PMs for
        faint/HST-only stars, revealing the true CTE signal.
        If None, the per-star pmra_xmatch/pmdec_xmatch from the master catalog
        are used (accurate for Gaia-matched stars; CTE-contaminated for HST-only).

    Returns
    -------
    residuals : {'hi': {...}, 'lo': {...}} each with arrays:
        dx, dy : (n,) GDC residuals [pixels]
        X_c : (n,) GDC-centred x [px]  (for X-direction basis and cross-terms)
        y_raw : (n,) raw chip-local y [px]  (for Y-direction CTE basis)
        mag, dt, z : per-detection geometry/weighting
    """
    from astropy.time import Time
    from .run_alignment_v2 import _compute_full_catalog_residuals_from_df

    bp3m_gaia_ids = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)
    out_arrays = _compute_full_catalog_residuals_from_df(
        img_to_df, bp3m_gaia_ids, solver, image_names, r_hat,
        global_pm_override=field_mean_pm)

    residuals = {c: {'dx': [], 'dy': [], 'X_c': [], 'y_raw': [],
                     'mag': [], 'dt': [], 'z': []}
                 for c in ('hi', 'lo')}

    for img in image_names:
        if f'{img}_X_c' not in out_arrays:
            continue
        chip = _chip_from_image(img)
        if chip is None:
            continue

        hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                             format='mjd').jyear)
        dt = hst_yr - t_epoch0_yr

        X_c  = out_arrays[f'{img}_X_c'].astype(float)
        dx   = out_arrays[f'{img}_dx_gdc'].astype(float)
        dy   = out_arrays[f'{img}_dy_gdc'].astype(float)
        mag  = out_arrays[f'{img}_mag_inst'].astype(float)

        # Use raw chip-local y for CTE y-basis (physically correct readout direction)
        if f'{img}_y_raw' in out_arrays:
            y_raw = out_arrays[f'{img}_y_raw'].astype(float)
        else:
            # Fallback: approximate raw y from GDC
            Y_c = out_arrays[f'{img}_Y_c'].astype(float)
            y_raw = (Y_c + 1.0) if chip == 'hi' else (Y_c + 2048.0)

        ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag) & np.isfinite(y_raw)

        # PM quality filter: only use stars with converged v2 PM
        # (sigma_pmra < 1 mas/yr, sigma_pmdec < 1 mas/yr, |pmra| < 3 mas/yr).
        # This excludes HST-only stars with no Gaia anchor, faint stars with
        # noisy cross-match PMs, and non-members with large PMs.
        # Gaia-matched stars (in_bp3m=True) are always included — they have
        # Gaia priors and their residuals are CTE-free for alignment purposes.
        in_bp3m_raw = out_arrays.get(f'{img}_in_bp3m')
        in_bp3m_flag = (in_bp3m_raw.astype(bool)
                        if in_bp3m_raw is not None
                        else np.zeros(len(dx), dtype=bool))

        sig_pmra  = out_arrays.get(f'{img}_sigma_pmra')
        sig_pmdec = out_arrays.get(f'{img}_sigma_pmdec')
        pmra_arr  = out_arrays.get(f'{img}_pmra_xmatch')
        pmdec_arr = out_arrays.get(f'{img}_pmdec_xmatch')
        if sig_pmra is not None:
            sig_pmra  = sig_pmra.astype(float)
            sig_pmdec = sig_pmdec.astype(float)
            pmra_arr  = pmra_arr.astype(float)
            pmdec_arr = pmdec_arr.astype(float)
            pm_quality = (np.isfinite(sig_pmra) & (sig_pmra < 1.0) &
                          np.isfinite(sig_pmdec) & (sig_pmdec < 1.0) &
                          np.isfinite(pmra_arr)  & (np.abs(pmra_arr)  < 3.0) &
                          np.isfinite(pmdec_arr) & (np.abs(pmdec_arr) < 3.0))
            ok = ok & (in_bp3m_flag | pm_quality)
        else:
            ok = ok & in_bp3m_flag
        n_ok = int(ok.sum())
        if n_ok == 0:
            continue

        residuals[chip]['dx'].append(dx[ok])
        residuals[chip]['dy'].append(dy[ok])
        residuals[chip]['X_c'].append(X_c[ok])
        residuals[chip]['y_raw'].append(y_raw[ok])
        residuals[chip]['mag'].append(mag[ok])
        residuals[chip]['dt'].append(np.full(n_ok, dt))
        residuals[chip]['z'].append(np.ones(n_ok))

    for chip in ('hi', 'lo'):
        for key in residuals[chip]:
            arr = residuals[chip][key]
            residuals[chip][key] = np.concatenate(arr) if arr else np.array([])
        n = len(residuals[chip]['dx'])
        print(f"    {chip}: {n:,} detections")

    return residuals


# ── CTE parameter update ──────────────────────────────────────────────────────

def update_cte_params(
    residuals_by_chip: dict[str, dict],
    cte_params: dict[str, CTEChipParams],
    n_inner: int = 5,
    delta_tol: float = 1e-4,
    regularize: float = 1e-8,
    lambda_delta: float = 1e-6,
) -> tuple[dict[str, CTEChipParams], dict]:
    """
    Update CTE parameters (γ, δ) from GDC-frame residuals via Gauss-Newton.

    Alternating updates — avoids joint [γ, Δδ] degeneracy when γ ≈ 0:
      Step A: Linear WLS for γ_x, γ_y (3 coefficients each) with δ fixed.
      Step B: 1D Newton step for δ with γ fixed.
                Δδ = (Σ z·r_x·j_x + Σ z·r_y·j_y) / (Σ z·j_x² + Σ z·j_y² + λ_δ)
              where j = dt·(dφ/dδ)·(B@γ) is the Jacobian and r is the residual.
              Δδ is clipped to [-0.3, 0.3] per step; δ is clipped to [0.05, 3.0].

    Returns updated copy of cte_params and convergence info dict.
    """
    new_params = {c: cte_params[c].copy() for c in ('hi', 'lo')}
    info = {}

    for chip in ('hi', 'lo'):
        res = residuals_by_chip[chip]
        if len(res['dx']) == 0:
            info[chip] = {'converged': True, 'n_inner': 0, 'n_det': 0}
            continue

        dx    = res['dx'].astype(float)
        dy    = res['dy'].astype(float)
        X_c   = res['X_c'].astype(float)
        y_raw = res['y_raw'].astype(float)
        mag   = res['mag'].astype(float)
        dt    = res['dt'].astype(float)
        z     = res['z'].astype(float)

        ok = (np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)
              & np.isfinite(dt) & np.isfinite(y_raw) & (np.abs(dt) > 0))
        dx, dy, X_c, y_raw, mag, dt, z = (arr[ok] for arr in
                                            (dx, dy, X_c, y_raw, mag, dt, z))
        n = len(dx)

        if n < 10:
            info[chip] = {'converged': True, 'n_inner': 0, 'n_det': n}
            continue

        p              = new_params[chip]
        delta_n        = float(p.delta)
        y_readout_raw  = p.y_readout_raw

        Bx = cte_x_basis(X_c, y_raw, y_readout_raw)
        By = cte_y_basis(X_c, y_raw, y_readout_raw)

        delta_history = [delta_n]
        for it in range(n_inner):
            phi = phi_flux(mag, delta_n)
            Phi = dt * phi              # (n,)

            # ── Step A: linear solve for γ with δ fixed ───────────────────────
            # Sign convention: dy_gdc = y_obs − y_pred > 0 when CTE shifts in +y.
            # Fit Φ·B@γ = −dx/dy so that γ < 0 and apply_cte_to_solver (which adds
            # Φ·B@γ to xys) SUBTRACTS the CTE offset.  Dynamic column normalization
            # is essential — the 6-term basis spans ~10 decades without it.
            raw_Ax = Phi[:, None] * Bx    # (n, 6) before scaling
            raw_Ay = Phi[:, None] * By    # (n, 6) before scaling
            col_scale_x = np.std(raw_Ax, axis=0).clip(min=1e-30)
            col_scale_y = np.std(raw_Ay, axis=0).clip(min=1e-30)
            Ax_s = raw_Ax / col_scale_x   # O(1) columns
            Ay_s = raw_Ay / col_scale_y   # O(1) columns

            AtWA_x = (Ax_s * z[:, None]).T @ Ax_s + regularize * np.eye(6)
            AtWr_x = (Ax_s * z[:, None]).T @ (-dx)
            AtWA_y = (Ay_s * z[:, None]).T @ Ay_s + regularize * np.eye(6)
            AtWr_y = (Ay_s * z[:, None]).T @ (-dy)

            try:
                p.gamma_x = np.linalg.solve(AtWA_x, AtWr_x) / col_scale_x
                p.gamma_y  = np.linalg.solve(AtWA_y, AtWr_y) / col_scale_y
            except np.linalg.LinAlgError:
                break

            # ── Step B: 1D Newton step for δ with γ fixed ────────────────────
            dphi  = dphi_ddelta(mag, delta_n)   # (n,)
            Bx_g  = Bx @ p.gamma_x              # (n,)
            By_g  = By @ p.gamma_y              # (n,)
            jac_x = dt * dphi * Bx_g            # (n,)
            jac_y = dt * dphi * By_g            # (n,)

            rx = -dx - Phi * Bx_g
            ry = -dy - Phi * By_g

            numer = float(np.dot(z * rx, jac_x) + np.dot(z * ry, jac_y))
            denom = float(np.dot(z * jac_x, jac_x) + np.dot(z * jac_y, jac_y)
                          + lambda_delta)

            delta_delta = np.clip(numer / denom if denom > 0 else 0.0, -0.3, 0.3)
            delta_n     = float(np.clip(delta_n + delta_delta, 0.05, 3.0))
            p.delta     = delta_n
            delta_history.append(delta_n)

            if abs(delta_delta) < delta_tol:
                break

        # Residual RMS: how well model Φ·B@γ explains target −dx/dy
        phi_f = phi_flux(mag, p.delta)
        rms_x = float(np.sqrt(np.mean(
            (-dx - dt * phi_f * (Bx @ p.gamma_x)) ** 2)))
        rms_y = float(np.sqrt(np.mean(
            (-dy - dt * phi_f * (By @ p.gamma_y)) ** 2)))

        info[chip] = {
            'n_det': n,
            'delta_history': delta_history,
            'rms_x': rms_x,
            'rms_y': rms_y,
            'n_inner': len(delta_history) - 1,
        }
        print(f"    {chip}: δ={p.delta:.4f}  "
              f"|γ_y|={np.linalg.norm(p.gamma_y):.4e}  "
              f"|γ_x|={np.linalg.norm(p.gamma_x):.4e}  "
              f"rms_y={rms_y:.4f}px  n={n:,}")

    return new_params, info


# ── Field mean PM via iterative sigma-clipping ────────────────────────────────

def _compute_warmstart_field_pm(
    data_root,
    field_name: str,
    n_sigma: float = 3.0,
    n_iter: int = 10,
) -> tuple[float, float] | None:
    """
    Estimate field bulk PM from Gaia-matched stars' pmra_xmatch in master_combined_v2.csv.

    Reads directly from master_combined_v2.csv (where pmra_xmatch is populated for
    all stars including Gaia-matched ones) rather than gaia_catalog (where the
    Gaia-matched rows have NaN pmra_xmatch in memory).  Excludes HST-only stars
    (gaia_source_id == 0) so their CTE-biased within-HST PMs don't inflate γ₀.

    Returns (pmra_mean, pmdec_mean) in mas/yr, or None if insufficient data.
    """
    import pandas as _pd
    mcat_path = (Path(data_root) / field_name / 'hst_xmatch'
                 / 'master_combined_v2.csv')
    if not mcat_path.exists():
        return None

    mcat = _pd.read_csv(mcat_path,
                        usecols=['gaia_source_id', 'pmra_xmatch', 'pmdec_xmatch'],
                        dtype={'gaia_source_id': np.int64},
                        low_memory=False)
    # Gaia-matched rows have gaia_source_id > 0; HST-only rows have 0
    gaia_rows = mcat[mcat['gaia_source_id'] > 0]
    pmra  = gaia_rows['pmra_xmatch'].to_numpy(float)
    pmdec = gaia_rows['pmdec_xmatch'].to_numpy(float)

    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    pmra, pmdec = pmra[finite], pmdec[finite]

    if len(pmra) < 5:
        return None

    keep = np.ones(len(pmra), dtype=bool)
    for _ in range(n_iter):
        if keep.sum() < 5:
            break
        med_ra   = float(np.median(pmra[keep]))
        med_dec  = float(np.median(pmdec[keep]))
        dra, ddec = pmra[keep] - med_ra, pmdec[keep] - med_dec
        sigma    = max(float(np.median(np.hypot(dra, ddec))) / 0.6745, 0.01)
        new_keep = np.hypot(pmra - med_ra, pmdec - med_dec) < n_sigma * sigma
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    pmra_mean  = float(np.mean(pmra[keep]))
    pmdec_mean = float(np.mean(pmdec[keep]))
    print(f"  Warm-start field mean PM (Gaia xmatch, n={keep.sum()}/{len(pmra)}): "
          f"({pmra_mean:+.3f}, {pmdec_mean:+.3f}) mas/yr")
    return (pmra_mean, pmdec_mean)


def _compute_field_mean_pm(
    solver,
    gaia_catalog,
    hst_only_mask: np.ndarray,
    v_hat: np.ndarray,
    n_sigma: float = 3.0,
    n_iter: int = 10,
) -> tuple[float, float]:
    """
    Estimate the field bulk PM from Gaia-matched member stars.

    Uses the BP3M-fitted stellar PMs (v_hat) for Gaia-matched stars that have
    at least one detection contributing to the astrometry solution (use_for_astrom
    or use_for_fit).  Stars with no HST detections default to their prior (v_survey),
    which is 0 for 2p Gaia stars and would bias the mean.

    Applies iterative 2D sigma-clipping to find the member locus.  No prior on
    the mean PM is assumed — the locus is inferred from the data.

    Returns (pmra_mean, pmdec_mean) in mas/yr.
    """
    # v_hat shape: (n_stars, 5) — [Δα, Δδ, μ_α, μ_δ, plx]
    n_stars = v_hat.shape[0]

    # Build per-star "has_hst" mask: True if any detection with use_for_astrom or
    # use_for_fit=True.  Stars excluded from all detections have v_hat = v_survey
    # (0 for 2p) and must NOT be included in the field mean.
    n_hst = np.zeros(n_stars, dtype=int)
    for img in solver.image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        sidx = d['sidx']
        use_any = d.get('use_for_astrom', d['use_for_fit'])
        np.add.at(n_hst, sidx[use_any], 1)

    # Gaia-matched stars with at least one contributing detection
    gaia_mask = (~hst_only_mask) & (n_hst >= 1)

    pmra  = v_hat[gaia_mask, 2].copy()   # mas/yr
    pmdec = v_hat[gaia_mask, 3].copy()   # mas/yr

    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    pmra  = pmra[finite]
    pmdec = pmdec[finite]

    if len(pmra) < 5:
        return (0.0, 0.0)

    # Iterative sigma-clipping in 2D PM space
    keep = np.ones(len(pmra), dtype=bool)
    for _ in range(n_iter):
        if keep.sum() < 5:
            break
        med_ra  = float(np.median(pmra[keep]))
        med_dec = float(np.median(pmdec[keep]))
        # Median absolute deviation → robust sigma
        dra   = pmra[keep]  - med_ra
        ddec  = pmdec[keep] - med_dec
        sigma = float(np.median(np.hypot(dra, ddec))) / 0.6745
        sigma = max(sigma, 0.01)   # avoid zero
        dist  = np.hypot(pmra - med_ra, pmdec - med_dec)
        new_keep = dist < n_sigma * sigma
        if new_keep.sum() == keep.sum():
            break   # converged
        keep = new_keep

    n_members = int(keep.sum())

    # Precision-weighted mean: use Gaia PM uncertainty (pmra_error) as weight proxy.
    # For Gaia 5p/6p stars: pmra_error is O(0.05-1) mas/yr → high weight.
    # For Gaia 2p stars: pmra_error is NaN → use diffuse prior σ=100 mas/yr → low weight.
    # Note: solver._C_VG_inv_per_star[:, 2] is 0 for 5p/6p stars (Gaia covariance is
    # stored separately), so we use the catalog pmra_error directly.
    try:
        if 'pmra_error' not in gaia_catalog.columns:
            raise KeyError('no pmra_error column')
        pmra_err = gaia_catalog['pmra_error'].to_numpy(float)[gaia_mask][finite]
        # Fill NaN (Gaia 2p / no Gaia PM) with diffuse prior σ = 100 mas/yr
        pmra_err = np.where(np.isfinite(pmra_err) & (pmra_err > 0), pmra_err, 100.0)
        prec = 1.0 / np.maximum(pmra_err, 0.01) ** 2
        w = prec[keep]
        w_sum = float(w.sum())
        if w_sum > 0 and np.ptp(w) > 0:
            pmra_mean  = float(np.sum(w * pmra[keep]) / w_sum)
            pmdec_mean = float(np.sum(w * pmdec[keep]) / w_sum)
        else:
            pmra_mean  = float(np.mean(pmra[keep]))
            pmdec_mean = float(np.mean(pmdec[keep]))
        weight_label = 'Gaia-precision-weighted'
    except Exception:
        pmra_mean  = float(np.mean(pmra[keep]))
        pmdec_mean = float(np.mean(pmdec[keep]))
        weight_label = 'unweighted'

    sigma_final = float(np.median(np.hypot(
        pmra[keep] - pmra_mean, pmdec[keep] - pmdec_mean)) / 0.6745)

    print(f"  Field mean PM ({weight_label}): ({pmra_mean:+.3f}, {pmdec_mean:+.3f}) mas/yr  "
          f"σ={sigma_final:.3f}  n_members={n_members}/{len(pmra)}")
    return (pmra_mean, pmdec_mean)


# ── Warm start ────────────────────────────────────────────────────────────────

def warm_start_cte(
    img_to_df: dict,
    solver,
    image_names: list[str],
    r_hat_init: np.ndarray,
    t_epoch0_yr: float,
    field_mean_pm: tuple[float, float] | None = None,
    data_root: Path | None = None,
    field_name: str | None = None,
) -> dict[str, CTEChipParams]:
    """
    Estimate initial γ_y by fitting per-star PM residuals to the CTE model.

    Since v2 BP3M absorbed CTE into individual star PMs, the PM residual
    (pm_i − field_mean) directly encodes the CTE-induced proper motion bias:

        Δpm_dec_i  ≈  pscale · d · φ(mag_i) · B_y(y_raw_i, X_c_i) @ γ_y

    WLS with 1/σ²_pm weights recovers γ_y from the per-star PM pattern.
    An intercept column is included to absorb the mean CTE level (which was
    also absorbed into the v2 field-mean PM and thus cancels out of residuals
    only up to sampling noise).

    Uses sigma_pmdec from master_combined_v2.csv (BP3M v2 fit uncertainty) for
    ALL stars including HST-only, which gives proper weights to faint stars
    that carry the strongest CTE signal.  The per-star key is the rounded
    (pmra_xmatch, pmdec_xmatch) pair, which has near-zero collision rate.

    field_mean_pm : (pmra_mean, pmdec_mean) in mas/yr.  Should be the true
        field mean, e.g. from Gaia-bright member cross-match.
    data_root, field_name : if provided, master_combined_v2.csv is read to
        build the sigma_pmdec lookup.  Required for proper HST-only weights.
    """
    import pandas as _pd

    print("  Warm-starting CTE from v2 PM residuals...")

    mean_pmra  = float(field_mean_pm[0]) if field_mean_pm else 0.0
    mean_pmdec = float(field_mean_pm[1]) if field_mean_pm else 0.0

    params = default_cte_params()
    n_r    = solver.N_R

    # ── Build sigma_pmdec lookup from master catalog ──────────────────────────
    # master_combined_v2.csv has sigma_pmdec (v2 fit uncertainty) for ALL stars,
    # including HST-only stars that lack Gaia priors.  Key: rounded PM pair.
    sigma_lookup: dict[str, float] = {}
    if data_root is not None and field_name is not None:
        mcat_path = (Path(data_root) / field_name / 'hst_xmatch'
                     / 'master_combined_v2.csv')
        if mcat_path.exists():
            _mc = _pd.read_csv(mcat_path,
                               usecols=['pmra_xmatch', 'pmdec_xmatch', 'sigma_pmdec'],
                               low_memory=False)
            fin = _mc['pmra_xmatch'].notna() & _mc['sigma_pmdec'].notna()
            _mc = _mc[fin]
            _keys = (_mc['pmra_xmatch'].round(6).astype(str) + '_'
                     + _mc['pmdec_xmatch'].round(6).astype(str))
            sigma_lookup = dict(zip(_keys, _mc['sigma_pmdec']))
            print(f"    Loaded sigma_pmdec lookup: {len(sigma_lookup):,} stars")

    # ── Mean transformation coefficients and pixel scale per chip ────────────
    # r_j = [a, b, c, d, w, z, ...] — R = [[a,b],[c,d]], so:
    #   a = R[0,0]: x-pixel → RA sky (x-CTE warm start)
    #   d = R[1,1]: y-pixel → Dec sky (y-CTE warm start)
    chip_a      = {'hi': [], 'lo': []}
    chip_d      = {'hi': [], 'lo': []}
    chip_pscale = {'hi': [], 'lo': []}
    for j_idx, img in enumerate(image_names):
        chip = _chip_from_image(img)
        if chip is None:
            continue
        r_j    = r_hat_init[j_idx * n_r : (j_idx + 1) * n_r]
        chip_a[chip].append(float(r_j[0]))
        chip_d[chip].append(float(r_j[3]))
        chip_pscale[chip].append(float(solver.images[img]['orig_pixel_scale']))

    mean_a      = {c: float(np.mean(v)) if v else 1.0  for c, v in chip_a.items()}
    mean_d      = {c: float(np.mean(v)) if v else 1.0  for c, v in chip_d.items()}
    mean_pscale = {c: float(np.mean(v)) if v else 50.0 for c, v in chip_pscale.items()}

    # ── Collect per-detection data from img_to_df ─────────────────────────────
    needed = {'y_raw', 'x_gdc', 'mag_gdc', 'pmra_xmatch', 'pmdec_xmatch'}
    _pm_window = 2.0   # mas/yr half-width membership pre-selection

    rows = []
    for img in image_names:
        chip = _chip_from_image(img)
        if chip is None or img not in img_to_df:
            continue
        df = img_to_df[img]
        if not needed.issubset(df.columns):
            continue

        pmra_arr  = df['pmra_xmatch'].to_numpy(float)
        pmdec_arr = df['pmdec_xmatch'].to_numpy(float)
        member_mask = (
            np.isfinite(pmra_arr)  & (np.abs(pmra_arr  - mean_pmra)  < _pm_window)
            & np.isfinite(pmdec_arr) & (np.abs(pmdec_arr - mean_pmdec) < _pm_window)
            & df['y_raw'].notna().to_numpy()
            & df['mag_gdc'].notna().to_numpy()
        )
        if not member_mask.any():
            continue

        idx = df.index[member_mask]
        tmp = df.loc[idx, ['y_raw', 'x_gdc', 'mag_gdc',
                            'pmra_xmatch', 'pmdec_xmatch']].copy()
        tmp['chip']    = chip

        # Build sigma_pmdec: prefer master-catalog v2 fit uncertainty,
        # fall back to Gaia xmatch sigma, then use 5.0 as last resort.
        pm_keys = (tmp['pmra_xmatch'].round(6).astype(str) + '_'
                   + tmp['pmdec_xmatch'].round(6).astype(str))
        if sigma_lookup:
            sigma_from_master = pm_keys.map(sigma_lookup)
        else:
            sigma_from_master = _pd.Series(np.nan, index=tmp.index)

        if 'sigma_pmdec_xmatch' in df.columns:
            sigma_xmatch = df.loc[idx, 'sigma_pmdec_xmatch'].values
        else:
            sigma_xmatch = np.full(len(tmp), np.nan)

        # Prefer master-catalog sigma; fall back to xmatch sigma; then 5.0
        sigma_vals = np.where(
            np.isfinite(sigma_from_master.values), sigma_from_master.values,
            np.where(np.isfinite(sigma_xmatch), sigma_xmatch, 5.0))
        tmp['sigma_pmdec'] = sigma_vals

        # Per-star key: rounded (pmra, pmdec) pair — near-zero collision rate
        tmp['star_key'] = pm_keys.values
        rows.append(tmp)

    if not rows:
        print("  WARNING: no member detections found — returning zero γ_y")
        return params

    all_dets = _pd.concat(rows, ignore_index=True)

    # Aggregate per (star_key, chip): median position/mag, first PM/sigma
    agg = (all_dets
           .groupby(['star_key', 'chip'], sort=False)
           .agg(y_raw        = ('y_raw',        'median'),
                x_gdc        = ('x_gdc',         'median'),
                mag_gdc      = ('mag_gdc',        'median'),
                pmra_xmatch  = ('pmra_xmatch',    'first'),
                pmdec_xmatch = ('pmdec_xmatch',   'first'),
                sigma_pmdec  = ('sigma_pmdec',    'first'),
                n_det        = ('y_raw',           'count'))
           .reset_index())

    n_hi = int((agg['chip'] == 'hi').sum())
    n_lo = int((agg['chip'] == 'lo').sum())
    print(f"    Member stars (window ±{_pm_window} mas/yr): hi={n_hi:,}  lo={n_lo:,}")

    # ── Per-chip weighted least squares for y-CTE and x-CTE ──────────────────
    for chip in ('hi', 'lo'):
        p       = params[chip]
        pscale  = mean_pscale[chip]   # mas/px
        d_coef  = mean_d[chip]        # ≈ R[1,1], maps y-px → sky-Dec
        a_coef  = mean_a[chip]        # ≈ R[0,0], maps x-px → sky-RA

        sub = agg[agg['chip'] == chip].reset_index(drop=True)

        ok = (sub['pmdec_xmatch'].notna()
              & sub['pmra_xmatch'].notna()
              & sub['y_raw'].notna()
              & sub['mag_gdc'].notna()
              & sub['sigma_pmdec'].notna()
              & (sub['sigma_pmdec'] > 0)
              & (sub['sigma_pmdec'] < 50.0))   # exclude completely unconstrained (2p prior ~100)
        sub = sub[ok].reset_index(drop=True)

        if len(sub) < 20:
            print(f"    {chip}: only {len(sub)} stars — skipping")
            continue

        y_raw  = sub['y_raw'].to_numpy(float)
        X_c    = sub['x_gdc'].to_numpy(float) - 2048.0
        mag    = sub['mag_gdc'].to_numpy(float)
        dpm_dec = sub['pmdec_xmatch'].to_numpy(float) - mean_pmdec   # mas/yr
        dpm_ra  = sub['pmra_xmatch'].to_numpy(float)  - mean_pmra    # mas/yr
        sigma   = sub['sigma_pmdec'].to_numpy(float).clip(0.01)

        phi = phi_flux(mag, 1.0)   # phi > 0, larger for fainter stars
        By  = cte_y_basis(X_c, y_raw, p.y_readout_raw)
        Bx  = cte_x_basis(X_c, y_raw, p.y_readout_raw)

        w   = 1.0 / sigma**2
        sqw = np.sqrt(w)

        # y-CTE: dpm_dec ≈ pscale·d·phi·By @ gamma_y (+ intercept)
        Ay_phys = pscale * d_coef * phi[:, None] * By   # physical units (mas/yr)
        col_scale_y = np.std(Ay_phys, axis=0).clip(min=1e-30)
        A_y = np.column_stack([np.ones(len(sub)), Ay_phys / col_scale_y])
        try:
            coeffs_y, _, _, _ = np.linalg.lstsq(
                A_y * sqw[:, None], dpm_dec * sqw, rcond=None)
            params[chip].gamma_y = coeffs_y[1:] / col_scale_y
        except np.linalg.LinAlgError:
            print(f"    {chip}: y-CTE WLS failed")
            coeffs_y = np.zeros(len(col_scale_y) + 1)

        pred_y = A_y @ coeffs_y
        rms_y  = float(np.sqrt(np.average((dpm_dec - pred_y)**2, weights=w)))
        med_phi = float(np.median(phi))
        med_yp  = float(np.median(np.abs(y_raw - p.y_readout_raw)))
        cte_mas_yr = abs(pscale * d_coef * med_phi * med_yp * params[chip].gamma_y[0])
        print(f"    {chip} y-CTE: γ_y[0]={params[chip].gamma_y[0]:.3e}  "
              f"CTE@med={cte_mas_yr:.3f} mas/yr  "
              f"intercept={coeffs_y[0]:+.3f} mas/yr  "
              f"rms={rms_y:.3f} mas/yr  n={len(sub):,}")

        # x-CTE: dpm_ra ≈ pscale·a·phi·Bx @ gamma_x (+ intercept)
        Ax_phys = pscale * a_coef * phi[:, None] * Bx   # physical units (mas/yr)
        col_scale_x = np.std(Ax_phys, axis=0).clip(min=1e-30)
        A_x = np.column_stack([np.ones(len(sub)), Ax_phys / col_scale_x])
        try:
            coeffs_x, _, _, _ = np.linalg.lstsq(
                A_x * sqw[:, None], dpm_ra * sqw, rcond=None)
            params[chip].gamma_x = coeffs_x[1:] / col_scale_x
        except np.linalg.LinAlgError:
            print(f"    {chip}: x-CTE WLS failed")
            coeffs_x = np.zeros(len(col_scale_x) + 1)

        pred_x = A_x @ coeffs_x
        rms_x  = float(np.sqrt(np.average((dpm_ra - pred_x)**2, weights=w)))
        print(f"    {chip} x-CTE: γ_x[0]={params[chip].gamma_x[0]:.3e}  "
              f"intercept={coeffs_x[0]:+.3f} mas/yr  "
              f"rms={rms_x:.3f} mas/yr")

    # ── Cross-chip fallback for wrong-sign γ_y0 ──────────────────────────────
    # apply_cte_to_solver adds Φ·B@γ to observed positions.  For y-CTE:
    # CTE shifts stars away from readout (+Y' direction), so correction needs
    # δCTE_y < 0 → γ_y[0] < 0.  If v2 over-absorbed the CTE for one chip,
    # the PM residuals can give the wrong sign.  Cross-seed from the other chip.
    other = {'hi': 'lo', 'lo': 'hi'}
    for chip in ('hi', 'lo'):
        gam = params[chip].gamma_y
        if gam[0] > 0:
            fb = params[other[chip]].gamma_y
            if fb[0] < 0:
                params[chip].gamma_y = fb.copy()
                print(f"    {chip}: γ_y[0]={gam[0]:.3e} (wrong sign) "
                      f"→ seeded from {other[chip]}: {fb[0]:.3e}")
            else:
                params[chip].gamma_y = -np.abs(gam)
                print(f"    {chip}: γ_y[0]={gam[0]:.3e} (both wrong sign) → negated")

    return params


# ── Convergence CSV ────────────────────────────────────────────────────────────

def _save_cte_convergence(output_dir: Path, outer_iter: int,
                          cte_params: dict, info: dict) -> None:
    import csv
    csv_path = output_dir / 'cte_convergence.csv'
    _gy_names = [f'gamma_y{k}' for k in range(6)]
    _gx_names = [f'gamma_x{k}' for k in range(6)]
    fieldnames = ['iter', 'chip', 'delta'] + _gy_names + _gx_names + ['rms_y', 'rms_x', 'n_det']
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for chip in ('hi', 'lo'):
            p  = cte_params[chip]
            ci = info.get(chip, {})
            row = {
                'iter':  outer_iter,
                'chip':  chip,
                'delta': f'{p.delta:.6f}',
                'rms_y': f'{ci.get("rms_y", float("nan")):.6f}',
                'rms_x': f'{ci.get("rms_x", float("nan")):.6f}',
                'n_det': ci.get('n_det', 0),
            }
            for k in range(6):
                row[f'gamma_y{k}'] = f'{p.gamma_y[k]:.6e}'
                row[f'gamma_x{k}'] = f'{p.gamma_x[k]:.6e}'
            writer.writerow(row)


# ── Diagnostic plots ───────────────────────────────────────────────────────────

def _plot_cte_diagnostics(output_dir: Path, cte_params: dict,
                          before_npz: Path, after_npz: Path,
                          image_names: list[str], solver) -> None:
    """
    Create CTE diagnostic figures:
      1. cte_amplitude.png   — δCTE_y amplitude vs Y_c for mag bins, per chip
      2. cte_before_after.png — dy_gdc vs Y_c before/after correction, per chip × epoch
      3. cte_slope_vs_mag.png — slope d(dy_gdc)/d(Y_c) vs mag, before/after, both chips
      4. cte_convergence.png  — δ and |γ_y| vs outer iteration (from convergence CSV)
      5. cte_2d_map.png       — 2D detector maps of dy_gdc before/after
    """
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import pandas as pd
    from astropy.time import Time

    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load residual arrays ──────────────────────────────────────────────────
    def _load_npz(path):
        if path is None or not path.exists():
            return None
        return np.load(path, allow_pickle=True)

    npz_before = _load_npz(before_npz)
    npz_after  = _load_npz(after_npz)

    if npz_before is None and npz_after is None:
        print("  _plot_cte_diagnostics: no residual arrays found — skipping plots")
        return

    # Build combined per-chip arrays from npz, annotated with chip/epoch/dt
    def _collect_from_npz(npz, image_names, solver):
        if npz is None:
            return None
        recs = {c: {'X_c': [], 'Y_c': [], 'y_raw': [], 'dx': [], 'dy': [],
                    'mag': [], 'dt': [], 'epoch_yr': []} for c in ('hi', 'lo')}
        for img in image_names:
            if f'{img}_X_c' not in npz:
                continue
            chip = _chip_from_image(img)
            if chip is None:
                continue
            hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                                 format='mjd').jyear)
            X_c = npz[f'{img}_X_c'].astype(float)
            Y_c = npz[f'{img}_Y_c'].astype(float)
            dx  = npz[f'{img}_dx_gdc'].astype(float)
            dy  = npz[f'{img}_dy_gdc'].astype(float)
            mag = npz[f'{img}_mag_inst'].astype(float)
            # Raw chip-local y — prefer stored value, fall back to GDC approximation
            if f'{img}_y_raw' in npz:
                y_raw = npz[f'{img}_y_raw'].astype(float)
            else:
                y_raw = (Y_c + 1.0) if chip == 'hi' else (Y_c + 2048.0)
            ok  = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag) & np.isfinite(y_raw)
            n   = ok.sum()
            if n == 0:
                continue
            recs[chip]['X_c'].append(X_c[ok])
            recs[chip]['Y_c'].append(Y_c[ok])
            recs[chip]['y_raw'].append(y_raw[ok])
            recs[chip]['dx'].append(dx[ok])
            recs[chip]['dy'].append(dy[ok])
            recs[chip]['mag'].append(mag[ok])
            recs[chip]['dt'].append(np.full(n, hst_yr))
            recs[chip]['epoch_yr'].append(np.full(n, round(hst_yr)))
        for chip in ('hi', 'lo'):
            for k in recs[chip]:
                arr = recs[chip][k]
                recs[chip][k] = np.concatenate(arr) if arr else np.array([])
        return recs

    data_before = _collect_from_npz(npz_before, image_names, solver)
    data_after  = _collect_from_npz(npz_after,  image_names, solver)

    chip_colors = {'hi': 'steelblue', 'lo': 'darkorange'}
    mag_ref = _MAG_REF

    # ── Figure 1: CTE amplitude vs raw y for mag bins ────────────────────────
    # Use raw chip-local y (0..2047) for each chip — physically correct range.
    # hi chip: readout at raw row 2047 (top), so Y' = y_raw - 2047 ∈ [-2047, 0]
    # lo chip: readout at raw row 0     (bottom), so Y' = y_raw ∈ [0, 2047]
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        all_mjds  = [float(solver.images[img]['hst_time_mjd'])
                     for img in image_names if img in solver.images]
        t0_mjd    = min(all_mjds)
        t1_mjd    = max(all_mjds)
        dt_max    = (t1_mjd - t0_mjd) / 365.25

        for col_i, chip in enumerate(('hi', 'lo')):
            ax  = axes[col_i]
            p   = cte_params[chip]
            # y_raw grids in the py1pass global frame
            # lo chip: y_raw ∈ [0, 2039]; hi chip: y_raw ∈ [2048, 4087]
            if chip == 'hi':
                y_raw_grid = np.linspace(p.y_readout_raw, p.y_readout_raw + 2039, 500)
            else:
                y_raw_grid = np.linspace(p.y_readout_raw, 2039, 500)
            X_c_grid   = np.zeros_like(y_raw_grid)   # at X centre of chip
            By = cte_y_basis(X_c_grid, y_raw_grid, p.y_readout_raw)

            for mag_v, col, lbl in [(18.0, 'steelblue', 'mag=18'),
                                    (20.0, 'green',     'mag=20'),
                                    (22.0, 'firebrick', 'mag=22')]:
                phi  = float(phi_flux(np.array([mag_v]), p.delta)[0])
                dcte = dt_max * phi * (By @ p.gamma_y)
                ax.plot(y_raw_grid, dcte, color=col, lw=1.8, label=lbl)

            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
            ax.axvline(p.y_readout_raw, color='k', lw=0.8, ls=':', alpha=0.5,
                       label=f'readout (y={p.y_readout_raw:.0f})')
            ax.set_xlabel('Raw global y (px, py1pass frame)')
            ax.set_ylabel('δCTE_y (px)' if col_i == 0 else '')
            ax.set_title(f'_{chip} chip — CTE_y amplitude (Δt={dt_max:.1f} yr, X_c=0)')
            ax.legend(fontsize=9)
        fig.suptitle('CTE correction amplitude (converged parameters, raw detector frame)',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / 'cte_amplitude.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: plots/cte_amplitude.png")
    except Exception as exc:
        print(f"  WARNING: cte_amplitude.png failed — {exc}")

    # ── Helper: binned mean ───────────────────────────────────────────────────
    def _binned(x, y, n_bins=10):
        order = np.argsort(x)
        xo, yo = x[order], y[order]
        edges = np.array_split(np.arange(len(xo)), n_bins)
        xm, ym, ye = [], [], []
        for idx in edges:
            if len(idx) < 3:
                continue
            xm.append(xo[idx].mean())
            ym.append(yo[idx].mean())
            ye.append(yo[idx].std() / np.sqrt(len(idx)))
        return np.array(xm), np.array(ym), np.array(ye)

    def _slope(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            return np.nan, np.nan
        x, y = x[ok], y[ok]
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        b  = np.dot(dx, y - ym) / max(np.dot(dx, dx), 1e-30)
        r  = y - (ym + b * dx)
        var_b = r.var() / max(np.dot(dx, dx), 1e-30)
        return b, np.sqrt(max(var_b, 0))

    # ── Figure 2: dy_gdc vs y_raw before/after — CTE fingerprint ─────────────
    # Layout: 2 rows (hi/lo chip) × 2 cols (dy/dx), before (solid) and after
    # (dashed) overlaid on each axis.  Legend uses proxy artists so the same
    # 5-entry key (3 mag bins + 2 line styles) works for every subplot.
    if data_before is not None or data_after is not None:
        try:
            from matplotlib.lines import Line2D

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            all_mags = []
            for src in [data_before, data_after]:
                if src is not None:
                    for chip in ('hi', 'lo'):
                        m = src[chip]['mag']
                        if len(m):
                            all_mags.append(m)
            if all_mags:
                all_mags = np.concatenate(all_mags)
                pcts = np.nanpercentile(all_mags, [0, 33, 67, 100])
            else:
                pcts = [17, 19, 21, 23]

            bin_cols   = ['steelblue', 'darkorange', 'firebrick']
            bin_labels = [f'bright  {pcts[0]:.1f}–{pcts[1]:.1f}',
                          f'mid  {pcts[1]:.1f}–{pcts[2]:.1f}',
                          f'faint  {pcts[2]:.1f}–{pcts[3]:.1f}']

            # Proxy-artist legend: 3 mag-bin colors + 2 line styles (before/after)
            proxy_handles = (
                [Line2D([0], [0], color=c, lw=2.5, label=lbl)
                 for c, lbl in zip(bin_cols, bin_labels)]
                + [Line2D([0], [0], color='gray', lw=2, ls='-',  label='before CTE'),
                   Line2D([0], [0], color='gray', lw=2, ls='--', label='after CTE')]
            )

            for row_i, chip in enumerate(('hi', 'lo')):
                for col_i, comp in enumerate(('dy', 'dx')):
                    ax = axes[row_i, col_i]

                    for src, ls, alpha_fill in [
                            (data_before, '-',  0.18),
                            (data_after,  '--', 0.10)]:
                        if src is None:
                            continue
                        cd = src[chip]
                        if len(cd.get(comp, [])) == 0 or len(cd.get('y_raw', [])) == 0:
                            continue
                        for bi, col in enumerate(bin_cols):
                            m = cd['mag']
                            mask = (np.isfinite(m) & (m >= pcts[bi]) & (m < pcts[bi+1])
                                    & np.isfinite(cd[comp]) & np.isfinite(cd['y_raw']))
                            if mask.sum() < 10:
                                continue
                            xm, ym, ye = _binned(cd['y_raw'][mask], cd[comp][mask])
                            ax.plot(xm, ym, ls=ls, color=col, lw=1.8, zorder=4)
                            ax.fill_between(xm, ym-ye, ym+ye, color=col,
                                            alpha=alpha_fill, zorder=3)

                    p = cte_params[chip]
                    ax.axvline(p.y_readout_raw, color='k', lw=0.9, ls=':',
                               label=f'readout y={p.y_readout_raw:.0f}')
                    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
                    comp_lbl = (r'$\delta y_\mathrm{GDC}$ (px)' if comp == 'dy'
                                else r'$\delta x_\mathrm{GDC}$ (px)')
                    ax.set_xlabel('Raw chip-local y (px)', fontsize=10)
                    ax.set_ylabel(comp_lbl, fontsize=10)
                    ax.set_title(f'chip {chip} — {comp} residual vs y_raw', fontsize=10)
                    ax.legend(handles=proxy_handles, fontsize=9, loc='best',
                              framealpha=0.85)

            fig.suptitle('CTE fingerprint: GDC residual vs raw y by magnitude\n'
                         'solid = before CTE correction   dashed = after CTE correction',
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(plot_dir / 'cte_before_after.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/cte_before_after.png")
        except Exception as exc:
            print(f"  WARNING: cte_before_after.png failed — {exc}")

    # ── Figure 3: CTE slope vs magnitude before/after (raw y) ────────────────
    # Slope d(residual)/d(y_raw) vs magnitude — the key CTE fingerprint:
    # CTE signal gets steeper (larger |slope|) for fainter stars.
    if data_before is not None or data_after is not None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            N_MAG_BINS = 8

            all_mags = []
            for src in [data_before, data_after]:
                if src is None:
                    continue
                for chip in ('hi', 'lo'):
                    m = src[chip]['mag']
                    if len(m):
                        all_mags.append(m)
            if not all_mags:
                plt.close(fig)
                raise ValueError("no data for slope plot")
            all_mags   = np.concatenate(all_mags)
            mag_edges  = np.nanpercentile(all_mags, np.linspace(0, 100, N_MAG_BINS+1))
            mag_mids   = 0.5 * (mag_edges[:-1] + mag_edges[1:])

            # (chip, when) → (fmt, color, markerfacecolor, label)
            styles = [
                (('hi', 'before'), 'o-',  'steelblue',  'steelblue',  'hi before'),
                (('hi', 'after'),  'o--', 'steelblue',  'none',       'hi after'),
                (('lo', 'before'), 's-',  'darkorange', 'darkorange', 'lo before'),
                (('lo', 'after'),  's--', 'darkorange', 'none',       'lo after'),
            ]

            for col_i, comp in enumerate(('dy', 'dx')):
                ax = axes[col_i]
                ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
                for (chip, when), fmt, color, mfc, label in styles:
                    src = data_before if when == 'before' else data_after
                    if src is None:
                        continue
                    cd = src[chip]
                    if len(cd.get(comp, [])) == 0 or len(cd.get('y_raw', [])) == 0:
                        continue
                    slopes, errs = [], []
                    for bi in range(N_MAG_BINS):
                        m = cd['mag']
                        mask = (np.isfinite(m) & (m >= mag_edges[bi])
                                & (m < mag_edges[bi+1])
                                & np.isfinite(cd[comp]) & np.isfinite(cd['y_raw']))
                        if mask.sum() < 8:
                            slopes.append(np.nan); errs.append(np.nan)
                            continue
                        sl, sl_e = _slope(cd['y_raw'][mask], cd[comp][mask])
                        slopes.append(sl * 1e3); errs.append(sl_e * 1e3)
                    slopes = np.array(slopes); errs = np.array(errs)
                    ok = np.isfinite(slopes)
                    if ok.any():
                        ax.errorbar(mag_mids[ok], slopes[ok], yerr=errs[ok],
                                    fmt=fmt, color=color, markerfacecolor=mfc,
                                    ms=6, capsize=3, lw=1.5, label=label)

                comp_lbl = (r'slope $d(\delta y_\mathrm{GDC})/d(y_\mathrm{raw})$ (mpx/px)'
                            if comp == 'dy'
                            else r'slope $d(\delta x_\mathrm{GDC})/d(y_\mathrm{raw})$ (mpx/px)')
                ax.set_xlabel('Instrumental magnitude', fontsize=10)
                ax.set_ylabel(comp_lbl, fontsize=10)
                ax.set_title(f'CTE slope vs magnitude ({comp})', fontsize=11)
                ax.legend(fontsize=9, framealpha=0.85)

            fig.suptitle(r'CTE fingerprint: slope $d(\delta\mathrm{GDC})/d(y_\mathrm{raw})$'
                         ' before vs after correction\n'
                         r'CTE signal: |slope| increases with magnitude, sign set by readout direction',
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(plot_dir / 'cte_slope_vs_mag.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/cte_slope_vs_mag.png")
        except Exception as exc:
            print(f"  WARNING: cte_slope_vs_mag.png failed — {exc}")

    # ── Figure 4: convergence ─────────────────────────────────────────────────
    try:
        conv_path = output_dir / 'cte_convergence.csv'
        if conv_path.exists():
            import pandas as _pd
            cdf = _pd.read_csv(conv_path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            for chip, col in [('hi', 'steelblue'), ('lo', 'darkorange')]:
                sub = cdf[cdf['chip'] == chip]
                if len(sub) == 0:
                    continue
                axes[0].plot(sub['iter'], sub['delta'], 'o-', color=col,
                             label=f'δ_{chip}', lw=1.8)
                gy_cols = [c for c in sub.columns if c.startswith('gamma_y')]
                gy_norm = np.sqrt((sub[gy_cols] ** 2).sum(axis=1))
                axes[1].plot(sub['iter'], gy_norm, 'o-', color=col,
                             label=f'|γ_y|_{chip}', lw=1.8)
                # Overplot rms_y on secondary axis
            axes[0].set_xlabel('CTE outer iteration')
            axes[0].set_ylabel('δ (flux power-law exponent)')
            axes[0].set_title('CTE δ convergence')
            axes[0].legend(fontsize=9)
            axes[1].set_xlabel('CTE outer iteration')
            axes[1].set_ylabel('|γ_y|')
            axes[1].set_title('CTE |γ_y| convergence')
            axes[1].legend(fontsize=9)
            fig.suptitle('CTE parameter convergence', fontsize=11)
            fig.tight_layout()
            fig.savefig(plot_dir / 'cte_convergence.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/cte_convergence.png")
    except Exception as exc:
        print(f"  WARNING: cte_convergence.png failed — {exc}")

    # ── Figure 5: 2D detector map before/after ─────────────────────────────────
    for label, npz_data in [('before', data_before), ('after', data_after)]:
        if npz_data is None:
            continue
        try:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            clip = 0.05
            norm = TwoSlopeNorm(vcenter=0, vmin=-clip, vmax=clip)
            for col_i, chip in enumerate(('hi', 'lo')):
                ax   = axes[col_i]
                cd   = npz_data[chip]
                if len(cd['dy']) == 0:
                    continue
                ok   = np.isfinite(cd['dy'])
                sc   = ax.scatter(cd['X_c'][ok], cd['Y_c'][ok], c=cd['dy'][ok],
                                  cmap='RdBu_r', norm=norm, s=2, alpha=0.5,
                                  linewidths=0, rasterized=True)
                cb   = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label(r'$\delta y_\mathrm{GDC}$ (px)', fontsize=9)
                ax.set_xlabel('X_c (px)')
                ax.set_ylabel('Y_c (px)')
                ax.set_title(f'_{chip} chip — {label} CTE correction')
            fig.suptitle(f'2D detector map of dy_gdc ({label} CTE)', fontsize=11)
            fig.tight_layout()
            out_path = plot_dir / f'cte_2d_map_{label}.png'
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/cte_2d_map_{label}.png")
        except Exception as exc:
            print(f"  WARNING: cte_2d_map_{label}.png failed — {exc}")

    # ── Figure 6: PM residuals of member stars vs raw detector position ────────
    # Faithfully replicates cte_diagnostic_leo_i.py with before/after overlay.
    # Member selection uses master_combined_v2.csv + Mahalanobis distance,
    # identical to cte_diagnostic_leo_i.py.
    # "Before" = pmra_xmatch from master_combined_v2.csv (all members)
    # "After"  = pmra_bp3m  from stellar_astrometry.csv (matched subset)
    try:
        import pandas as _pd
        from astropy.io import fits as _afits
        from matplotlib.lines import Line2D as _Line2D
        from matplotlib.colors import TwoSlopeNorm as _TSN
        from collections import defaultdict as _dd
        from scipy.spatial import cKDTree as _KDT

        field_dir  = output_dir.parent
        hst_root   = field_dir / 'HST' / 'mastDownload' / 'HST'
        master_csv = field_dir / 'hst_xmatch' / 'master_combined_v2.csv'
        astrom_csv = output_dir / 'stellar_astrometry.csv'
        if not master_csv.exists():
            raise FileNotFoundError(f'master_combined_v2.csv not found: {master_csv}')

        # ── Load master catalog ───────────────────────────────────────────────
        master = _pd.read_csv(master_csv, low_memory=False)

        # ── ACS/WFC filter: keep only rows observed in ACS images (j-prefix) ─
        _sub_cols = [c for c in master.columns if c.startswith('sub_names_')]
        _acs = _pd.Series(False, index=master.index)
        for _c in _sub_cols:
            _acs |= master[_c].fillna('').str.contains(r'\bj', regex=True)
        master = master[_acs].copy().reset_index(drop=True)

        # ── Keep only stars with valid xmatch PMs and uncertainties ──────────
        _ok = (master['pmra_xmatch'].notna() & master['pmdec_xmatch'].notna() &
               master['sigma_pmra_xmatch'].notna() & master['sigma_pmdec_xmatch'].notna() &
               master['corr_pmra_pmdec_xmatch'].notna())
        master = master[_ok].copy().reset_index(drop=True)

        pmra_b  = master['pmra_xmatch'].to_numpy(float)
        pmdec_b = master['pmdec_xmatch'].to_numpy(float)
        s_ra    = master['sigma_pmra_xmatch'].to_numpy(float)
        s_dec   = master['sigma_pmdec_xmatch'].to_numpy(float)
        rho     = master['corr_pmra_pmdec_xmatch'].to_numpy(float)

        # ── Mahalanobis membership (identical to cte_diagnostic_leo_i.py) ────
        _LIT_PMRA  = -0.063
        _LIT_PMDEC = -0.111
        dpmra_lit  = pmra_b  - _LIT_PMRA
        dpmdec_lit = pmdec_b - _LIT_PMDEC
        _floor   = 0.05 ** 2
        t_ra2    = s_ra**2  + _floor
        t_dec2   = s_dec**2 + _floor
        t_cov    = s_ra * s_dec * rho
        test_det = t_ra2 * t_dec2 - t_cov**2
        mahal2   = np.where(test_det > 0,
            (t_dec2*dpmra_lit**2 - 2*t_cov*dpmra_lit*dpmdec_lit + t_ra2*dpmdec_lit**2)
            / test_det, np.nan)
        cov_det  = s_ra**2 * s_dec**2 * (1.0 - rho**2)
        geom_unc = cov_det**0.25
        member   = (geom_unc < 1.0) & (mahal2 < 4.0)
        master   = master[member].copy().reset_index(drop=True)
        pmra_b   = pmra_b[member];  pmdec_b = pmdec_b[member]
        s_ra     = s_ra[member];    s_dec   = s_dec[member]
        rho      = rho[member];     cov_det = cov_det[member]
        print(f"  Figure 6: {len(master):,} member stars (Mahalanobis)")

        # ── Precision-weighted field mean for "before" (full covariance) ─────
        _W00  = s_dec**2 / cov_det;  _W11 = s_ra**2 / cov_det
        _W01  = -rho * s_ra * s_dec / cov_det
        _S00  = _W00.sum();  _S11 = _W11.sum();  _S01 = _W01.sum()
        _detS = _S00*_S11 - _S01**2
        _rhs0 = (_W00*pmra_b + _W01*pmdec_b).sum()
        _rhs1 = (_W01*pmra_b + _W11*pmdec_b).sum()
        mean_pmra_b  = float((_S11*_rhs0 - _S01*_rhs1) / _detS)
        mean_pmdec_b = float((-_S01*_rhs0 + _S00*_rhs1) / _detS)
        res_ra_b  = pmra_b  - mean_pmra_b
        res_dec_b = pmdec_b - mean_pmdec_b
        w_ra_b  = _W11   # per-star weight for pmra
        w_dec_b = _W00   # per-star weight for pmdec
        print(f"  Before field mean: μ_α*={mean_pmra_b:+.4f}  μ_δ={mean_pmdec_b:+.4f} mas/yr")

        # ── Load stellar_astrometry.csv and match to members for "after" ─────
        # Gaia-matched members: match by gaia_source_id (int64) == Gaia_id.
        # HST-only members: positional cross-match on (ra_xmatch, dec_xmatch).
        has_after = np.zeros(len(master), dtype=bool)
        res_ra_a  = np.full(len(master), np.nan)
        res_dec_a = np.full(len(master), np.nan)
        w_ra_a    = np.zeros(len(master))
        w_dec_a   = np.zeros(len(master))
        mean_pmra_a = mean_pmra_b   # fallback
        mean_pmdec_a = mean_pmdec_b

        if astrom_csv.exists():
            astrom = _pd.read_csv(astrom_csv)
            _has_bp3m = (astrom['pmra_bp3m'].notna() & astrom['pmdec_bp3m'].notna() &
                         astrom['sigma_pmra_bp3m'].notna())
            astrom = astrom[_has_bp3m].copy().reset_index(drop=True)

            # Pre-extract Gaia_id as int64 (never use iterrows for Gaia IDs)
            _astrom_gids = astrom['Gaia_id'].to_numpy(np.int64)
            _pmra_a  = astrom['pmra_bp3m'].to_numpy(float)
            _pmdec_a = astrom['pmdec_bp3m'].to_numpy(float)
            _sig_ra_a  = np.maximum(astrom['sigma_pmra_bp3m'].to_numpy(float),  0.001)
            _sig_dec_a = np.maximum(astrom['sigma_pmdec_bp3m'].to_numpy(float), 0.001)

            # Step 1: match by gaia_source_id for Gaia-matched master rows
            _master_gids = master['gaia_source_id'].to_numpy(np.int64) \
                if 'gaia_source_id' in master.columns \
                else np.zeros(len(master), dtype=np.int64)
            _astrom_gid_to_row = {int(_astrom_gids[i]): i
                                  for i in range(len(_astrom_gids))
                                  if int(_astrom_gids[i]) > 0}
            for mi in range(len(master)):
                gid = int(_master_gids[mi])
                if gid > 0 and gid in _astrom_gid_to_row:
                    ai = _astrom_gid_to_row[gid]
                    has_after[mi] = True
                    res_ra_a[mi]  = _pmra_a[ai]
                    res_dec_a[mi] = _pmdec_a[ai]
                    w_ra_a[mi]    = 1.0 / _sig_ra_a[ai]**2
                    w_dec_a[mi]   = 1.0 / _sig_dec_a[ai]**2

            # Step 2: positional cross-match for HST-only master rows
            _unmatched = ~has_after
            if _unmatched.any() and 'ra' in astrom.columns:
                _a_ra  = astrom['ra'].to_numpy(float)
                _a_dec = astrom['dec'].to_numpy(float)
                _a_ok  = np.isfinite(_a_ra) & np.isfinite(_a_dec)
                if _a_ok.any():
                    _tree = _KDT(np.column_stack([_a_ra[_a_ok], _a_dec[_a_ok]]))
                    _m_ra  = master['ra_xmatch'].to_numpy(float) \
                        if 'ra_xmatch' in master.columns \
                        else np.full(len(master), np.nan)
                    _m_dec = master['dec_xmatch'].to_numpy(float) \
                        if 'dec_xmatch' in master.columns \
                        else np.full(len(master), np.nan)
                    _q_ok  = _unmatched & np.isfinite(_m_ra) & np.isfinite(_m_dec)
                    if _q_ok.any():
                        _dists, _near = _tree.query(
                            np.column_stack([_m_ra[_q_ok], _m_dec[_q_ok]]))
                        _a_ok_idx = np.where(_a_ok)[0]
                        _tol = 0.5 / 3600.0
                        for mi, dist, ni in zip(np.where(_q_ok)[0], _dists, _near):
                            if dist < _tol:
                                ai = int(_a_ok_idx[ni])
                                has_after[mi] = True
                                res_ra_a[mi]  = _pmra_a[ai]
                                res_dec_a[mi] = _pmdec_a[ai]
                                w_ra_a[mi]    = 1.0 / _sig_ra_a[ai]**2
                                w_dec_a[mi]   = 1.0 / _sig_dec_a[ai]**2

            n_after = int(has_after.sum())
            print(f"  Matched {n_after:,}/{len(master):,} members to stellar_astrometry")

            # Precision-weighted field mean for "after" (matched subset)
            if n_after >= 5:
                _wa = w_ra_a[has_after]
                _pmra_sub  = res_ra_a[has_after]
                _pmdec_sub = res_dec_a[has_after]
                mean_pmra_a  = float(np.sum(_wa * _pmra_sub)  / np.sum(_wa))
                mean_pmdec_a = float(np.sum(w_dec_a[has_after] * _pmdec_sub)
                                     / np.sum(w_dec_a[has_after]))
                res_ra_a[has_after]  -= mean_pmra_a
                res_dec_a[has_after] -= mean_pmdec_a
                print(f"  After  field mean: μ_α*={mean_pmra_a:+.4f}  "
                      f"μ_δ={mean_pmdec_a:+.4f} mas/yr")
        else:
            n_after = 0
            print("  stellar_astrometry.csv not found — showing before only")

        # ── Parse hst_indices to find best image ──────────────────────────────
        _hst_idx_cols = [c for c in master.columns if c.startswith('hst_indices_')]
        img_member_count = _dd(int)
        star_img_idx = [{} for _ in range(len(master))]
        for _col in _hst_idx_cols:
            for _mi, _val in enumerate(master[_col]):
                if _pd.isna(_val):
                    continue
                for _entry in str(_val).split(','):
                    _entry = _entry.strip()
                    if ':' not in _entry:
                        continue
                    _img, _idx = _entry.rsplit(':', 1)
                    _img = _img.strip()
                    img_member_count[_img] += 1
                    if _img not in star_img_idx[_mi]:
                        star_img_idx[_mi][_img] = int(_idx.strip())

        if not img_member_count:
            raise ValueError('no hst_indices found for member stars')

        rootname_count = _dd(int)
        for _img_name, _cnt in img_member_count.items():
            _root = _img_name.replace('_hi', '').replace('_lo', '')
            rootname_count[_root] += _cnt
        best_root = max(rootname_count, key=rootname_count.get)
        best_variants = [v for v in [best_root+'_hi', best_root+'_lo']
                         if v in img_member_count]
        print(f"  Best image: {best_root}  ({rootname_count[best_root]:,} member detections)")

        # ── Load catalog.fits ─────────────────────────────────────────────────
        cat_path = hst_root / best_root / f'{best_root}_flc_catalog.fits'
        if not cat_path.exists():
            raise FileNotFoundError(f'catalog.fits not found: {cat_path}')
        with _afits.open(str(cat_path)) as hdul:
            _cat   = hdul[1].data
            cat_x  = _cat['x'].copy().astype(float)
            cat_y  = _cat['y'].copy().astype(float)
            cat_chip = _cat['chip_ext'].copy()
            cat_mag  = _cat['mag_st'].copy().astype(float)

        # ── Collect member stars found in best_root ────────────────────────
        rows_with_pos, cat_indices = [], []
        seen_rows: set = set()
        for mi in range(len(master)):
            for variant in best_variants:
                if variant in star_img_idx[mi] and mi not in seen_rows:
                    seen_rows.add(mi)
                    rows_with_pos.append(mi)
                    cat_indices.append(star_img_idx[mi][variant])
                    break
        rows_with_pos = np.array(rows_with_pos, dtype=int)
        cat_indices   = np.array(cat_indices,   dtype=int)
        print(f"  Stars with detector positions: {len(rows_with_pos):,}")

        x_raw  = cat_x[cat_indices]
        y_raw  = cat_y[cat_indices]
        chip   = cat_chip[cat_indices]
        mag    = cat_mag[cat_indices]

        # Per-star arrays aligned to rows_with_pos
        res_ra_b_s   = res_ra_b[rows_with_pos]
        res_dec_b_s  = res_dec_b[rows_with_pos]
        w_ra_b_s     = w_ra_b[rows_with_pos]
        w_dec_b_s    = w_dec_b[rows_with_pos]
        res_ra_a_s   = res_ra_a[rows_with_pos]
        res_dec_a_s  = res_dec_a[rows_with_pos]
        w_ra_a_s     = w_ra_a[rows_with_pos]
        w_dec_a_s    = w_dec_a[rows_with_pos]
        has_after_s  = has_after[rows_with_pos]

        # ── Helpers (same as cte_diagnostic_leo_i.py) ────────────────────────
        def _wmean_binned(x, y, w, n_bins=14):
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
            if ok.sum() < 3:
                return np.array([]), np.array([]), np.array([])
            x, y, w = x[ok], y[ok], w[ok]
            order = np.argsort(x)
            x, y, w = x[order], y[order], w[order]
            xm, ym, ye = [], [], []
            for _idx in np.array_split(np.arange(len(x)), n_bins):
                if len(_idx) < 3:
                    continue
                wi = w[_idx]; wt = wi.sum()
                if wt <= 0:
                    continue
                xm.append(x[_idx].mean())
                ym.append(float(np.sum(wi * y[_idx]) / wt))
                ye.append(float(np.sqrt(1.0 / wt)))
            return np.array(xm), np.array(ym), np.array(ye)

        def _wls_slope(x, y, w):
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
            if ok.sum() < 5:
                return np.nan, np.nan
            x, y, w = x[ok], y[ok], w[ok]
            xm = np.sum(w*x)/np.sum(w);  ym = np.sum(w*y)/np.sum(w)
            dx = x - xm
            denom = np.sum(w * dx**2)
            if denom <= 0:
                return np.nan, np.nan
            b = np.sum(w * dx * (y - ym)) / denom
            resid = y - (ym + b*dx)
            var_b = max(np.sum(w*resid**2) / max(len(x)-2, 1) / denom, 0.0)
            return float(b), float(np.sqrt(var_b))

        # ── Figure layout ─────────────────────────────────────────────────────
        chip_vals   = np.unique(chip)
        chip_masks  = [chip == v for v in chip_vals]
        chip_titles = [f'chip_ext={v}' for v in chip_vals]
        n_chips     = len(chip_vals)

        mag_pcts   = np.nanpercentile(mag, [0, 33, 67, 100])
        bin_colors = ['steelblue', 'darkorange', 'firebrick']
        bin_labels = [
            f'bright  ({mag_pcts[0]:.1f}–{mag_pcts[1]:.1f})',
            f'mid     ({mag_pcts[1]:.1f}–{mag_pcts[2]:.1f})',
            f'faint   ({mag_pcts[2]:.1f}–{mag_pcts[3]:.1f})',
        ]

        fig = plt.figure(figsize=(14, 22))
        gs  = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.32,
                               left=0.08, right=0.96, top=0.94, bottom=0.05)
        clip_pm = 0.35  # mas/yr

        # Proxy artist legend for rows 1–2
        proxy_handles = (
            [_Line2D([0], [0], color=c, lw=2.5, label=lbl)
             for c, lbl in zip(bin_colors, bin_labels)]
            + [_Line2D([0], [0], color='gray', lw=2, ls='-',  label=f'before CTE  N={len(rows_with_pos):,}'),
               _Line2D([0], [0], color='gray', lw=2, ls='--', label=f'after CTE   N={has_after_s.sum():,}')]
        )

        # ── Row 0: 2D detector map, before (left) and after (right) ──────────
        for col_i, (lbl, res_dec_s, n_lbl) in enumerate([
                ('before CTE', res_dec_b_s, len(rows_with_pos)),
                ('after CTE',  np.where(has_after_s, res_dec_a_s, np.nan),
                 int(has_after_s.sum())),
        ]):
            ax = fig.add_subplot(gs[0, col_i])
            norm = _TSN(vcenter=0, vmin=-clip_pm, vmax=clip_pm)
            for cmask, marker, clbl in zip(chip_masks, ['o', 's'], chip_titles):
                sc = ax.scatter(x_raw[cmask], y_raw[cmask], c=res_dec_s[cmask],
                                cmap='RdBu_r', norm=norm, s=4, alpha=0.6,
                                linewidths=0, rasterized=True, marker=marker, label=clbl)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(r'$\Delta\mu_\delta$ (mas yr$^{-1}$)', fontsize=9)
            ax.set_xlabel('Raw detector X (px)', fontsize=11)
            ax.set_ylabel('Raw detector Y (px)', fontsize=11)
            ax.set_title(f'{lbl}  —  {best_root}  (N={n_lbl:,})', fontsize=11)
            ax.legend(fontsize=8, loc='upper right')
            ax.axhline(0, color='k', lw=0.8, ls=':', alpha=0.5)

        # ── Rows 1–n_chips: PM residuals vs y_raw per chip ───────────────────
        def _plot_pm_vs_y(ax, cmask, res_b, w_b, res_a, w_a, has_a, lbl, chip_lbl):
            slopes_b = []
            for bi, (color, blbl) in enumerate(zip(bin_colors, bin_labels)):
                m_b = (np.isfinite(mag) & (mag >= mag_pcts[bi])
                       & (mag < mag_pcts[bi+1]) & cmask)
                m_a = m_b & has_a

                for res, w, m, ls in [(res_b, w_b, m_b, '-'), (res_a, w_a, m_a, '--')]:
                    if m.sum() < 5:
                        continue
                    ax.scatter(y_raw[m], res[m], s=2, alpha=0.15,
                               color=color, linewidths=0, rasterized=True, zorder=2)
                    xm, ym, ye = _wmean_binned(y_raw[m], res[m], w[m])
                    if len(xm):
                        ax.plot(xm, ym, ls=ls, color=color, lw=2.0, zorder=4)
                        ax.fill_between(xm, ym-ye, ym+ye, color=color,
                                        alpha=0.20, zorder=3)
                        sl, sl_e = _wls_slope(y_raw[m], res[m], w[m])
                        if np.isfinite(sl):
                            yfit = np.linspace(y_raw[m].min(), y_raw[m].max(), 200)
                            ymn = np.average(res[m], weights=w[m])
                            xmn = np.average(y_raw[m], weights=w[m])
                            ax.plot(yfit, ymn + sl*(yfit-xmn),
                                    ls=ls, color=color, lw=1.2, zorder=5,
                                    label=f'  slope={sl*1e3:+.2f} μas/yr/px' if ls=='-' else None)
                    if ls == '-' and m.sum() >= 5:
                        sl, _ = _wls_slope(y_raw[m], res[m], w[m])
                        if np.isfinite(sl):
                            slopes_b.append(sl)

            ax.axhline(0, color='k', lw=0.8, alpha=0.5)
            ax.set_xlabel('Raw detector Y (px)', fontsize=11)
            ax.set_ylabel(lbl, fontsize=11)
            ax.set_title(f'{chip_lbl} — {lbl} vs raw Y', fontsize=11)
            ax.set_ylim(-clip_pm*1.5, clip_pm*1.5)
            ax.legend(handles=proxy_handles, fontsize=7, loc='upper left', ncol=2)

            # Annotate readout direction
            if slopes_b:
                mean_sl = float(np.mean(slopes_b))
                ylo, yhi = y_raw[cmask].min(), y_raw[cmask].max()
                side, ha = (ylo, 'left') if mean_sl < 0 else (yhi, 'right')
                ax.annotate('readout', xy=(side, ax.get_ylim()[0]*0.85),
                            fontsize=7, color='0.4', style='italic', ha=ha)

        for chip_row, (cmask, chip_lbl) in enumerate(zip(chip_masks, chip_titles)):
            row = 1 + chip_row
            for col_i, (res_b, w_b, res_a, w_a, pm_lbl) in enumerate([
                    (res_ra_b_s,  w_ra_b_s,  res_ra_a_s,  w_ra_a_s,
                     r'$\Delta\mu_{\alpha*}$ (mas yr$^{-1}$)'),
                    (res_dec_b_s, w_dec_b_s, res_dec_a_s, w_dec_a_s,
                     r'$\Delta\mu_\delta$ (mas yr$^{-1}$)'),
            ]):
                ax = fig.add_subplot(gs[row, col_i])
                _plot_pm_vs_y(ax, cmask, res_b, w_b, res_a, w_a, has_after_s, pm_lbl, chip_lbl)

        # ── Row 3: slope vs magnitude, before and after, both chips ──────────
        N_MAG_BINS = 8
        mag_edges  = np.nanpercentile(mag, np.linspace(0, 100, N_MAG_BINS+1))
        mag_mids   = 0.5 * (mag_edges[:-1] + mag_edges[1:])

        chip_styles = [
            dict(fmt='o-',  fmt_a='o--', color='steelblue',  mfc_b='steelblue',  mfc_a='none', label=chip_titles[0]),
            dict(fmt='s-',  fmt_a='s--', color='darkorange', mfc_b='darkorange', mfc_a='none',
                 label=chip_titles[1] if len(chip_titles) > 1 else ''),
        ]

        for col_i, (res_b, w_b, res_a, w_a, pm_lbl) in enumerate([
                (res_ra_b_s,  w_ra_b_s,  res_ra_a_s,  w_ra_a_s,
                 r'slope $\Delta\mu_{\alpha*}$ (μas yr$^{-1}$ px$^{-1}$)'),
                (res_dec_b_s, w_dec_b_s, res_dec_a_s, w_dec_a_s,
                 r'slope $\Delta\mu_\delta$ (μas yr$^{-1}$ px$^{-1}$)'),
        ]):
            ax = fig.add_subplot(gs[3, col_i])
            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)

            for cmask, style in zip(chip_masks, chip_styles):
                if cmask.sum() < 5:
                    continue
                for (res, w, use_mask, fmt, mfc, lbl_sfx) in [
                        (res_b, w_b, np.ones(len(chip), dtype=bool),
                         style['fmt'],   style['mfc_b'], ' before'),
                        (res_a, w_a, has_after_s,
                         style['fmt_a'], style['mfc_a'], ' after'),
                ]:
                    slopes, errs, ns = [], [], []
                    for k in range(N_MAG_BINS):
                        m = (np.isfinite(mag) & (mag >= mag_edges[k])
                             & (mag < mag_edges[k+1]) & cmask & use_mask)
                        if m.sum() < 10:
                            slopes.append(np.nan); errs.append(np.nan); ns.append(0)
                            continue
                        sl, sl_e = _wls_slope(y_raw[m], res[m], w[m])
                        slopes.append(sl * 1e3); errs.append(sl_e * 1e3); ns.append(m.sum())
                    slopes = np.array(slopes); errs = np.array(errs)
                    ok_sl  = np.isfinite(slopes) & (np.array(ns) >= 10)
                    if ok_sl.any():
                        ax.errorbar(mag_mids[ok_sl], slopes[ok_sl], yerr=errs[ok_sl],
                                    fmt=fmt, color=style['color'], markerfacecolor=mfc,
                                    ms=6, capsize=4, lw=1.5, zorder=4,
                                    label=style['label'] + lbl_sfx)
                    for k in np.where(ok_sl)[0]:
                        ax.text(mag_mids[k], slopes[k] + errs[k] + 0.005,
                                str(ns[k]), ha='center', va='bottom',
                                fontsize=6, color='0.5')

            ax.set_xlabel('HST ST magnitude', fontsize=11)
            ax.set_ylabel(pm_lbl, fontsize=11)
            ax.set_title('CTE fingerprint: gradient slope vs magnitude (per chip)', fontsize=11)
            ax.legend(fontsize=9)
            ax.text(0.97, 0.05,
                    'CTE: |slope| grows with magnitude\n'
                    'Sign set by readout direction\n'
                    '(opposite between chips — expected)',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, color='0.45', style='italic')

        fig.suptitle(
            f'Leo I — CTE diagnostic  ({best_root},  N_before={len(rows_with_pos):,}'
            f'  N_after={int(has_after_s.sum()):,} members)',
            fontsize=13, y=0.97)
        fig.savefig(plot_dir / 'cte_pm_vs_detector.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: plots/cte_pm_vs_detector.png")
    except Exception as exc:
        import traceback
        print(f"  WARNING: cte_pm_vs_detector.png failed — {exc}")
        traceback.print_exc()


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
    hst_max_per_image: int = 100_000,
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
    Joint CTE + astrometry alignment using master_combined_v2.csv catalog.

    Outer Gauss-Newton loop:
      1. Apply current CTE model → modify solver._img_data[img]['xys'].
      2. Run BP3M v2 solve (r_j, v_i) on CTE-corrected positions.
      3. Collect full-catalog GDC residuals (~127k stars) using current r_hat.
      4. Update CTE parameters (γ_c, δ_c) via Gauss-Newton.
      5. Check convergence; stop early if ‖Δγ‖/‖γ‖ < cte_gamma_rtol.

    Parameters
    ----------
    n_iter_bp3m : BP3M EM iterations per CTE outer step
    n_iter_cte  : max CTE Gauss-Newton outer iterations
    (remaining params identical to run_alignment_v2)

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
        V2AlignmentCallback, _load_full_catalog_df,
        _compute_full_catalog_residuals_from_df,
        _plot_soft_weights,
    )
    from .run_alignment import _save_results

    data_root  = Path(output_dir)
    output_cte = data_root / field_name / "BP3M_cte_results"
    output_cte.mkdir(parents=True, exist_ok=True)

    # Remove stale convergence CSV so each run starts fresh
    _conv_csv = output_cte / 'cte_convergence.csv'
    if _conv_csv.exists():
        _conv_csv.unlink()

    print("\n" + "─" * 60)
    print("BP3M CTE: joint CTE + astrometry alignment")
    print("─" * 60)
    print(f"  n_iter_bp3m={n_iter_bp3m}  n_iter_cte={n_iter_cte}  "
          f"poly_order={poly_order}  clip_sigma={clip_sigma}")

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
    imgs         = {n: images[n] for n in image_names if n in images}
    filtered_spi = {n: stars_per_image[n] for n in image_names}

    print(f"  Stars: {len(gaia_catalog)} "
          f"({int((~hst_only_mask).sum())} Gaia + {int(hst_only_mask.sum())} HST-only)  "
          f"Images: {len(image_names)}")

    # ── Load full-catalog detection DataFrame (cached for all CTE iterations) ──
    print("\n  Loading full-catalog detection data (cached for CTE iterations)...")
    img_to_df = _load_full_catalog_df(data_root, field_name)
    if img_to_df is None:
        raise RuntimeError(
            "detections_F814W.csv or master_combined_v2.csv not found. "
            "Run hst_catalog_crossmatch first.")

    # ── Reference epoch: first exposure ───────────────────────────────────────
    all_mjds     = [float(images[img]['hst_time_mjd']) for img in image_names
                    if img in images]
    t_epoch0_mjd = float(min(all_mjds))
    t_epoch0_yr  = float(Time(t_epoch0_mjd, format='mjd').jyear)
    print(f"  t_epoch0 = {t_epoch0_yr:.4f} yr  "
          f"({Time(t_epoch0_mjd, format='mjd').isot[:10]})")

    # ── Inject v1/v2 BP3M transformation as warm start ────────────────────────
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

    # ── Inject mag_inst into solver._img_data ─────────────────────────────────
    # Must be done before apply_cte_to_solver is called.
    _inject_mag_inst(solver, image_names, filtered_spi, gaia_catalog)

    # ── HST-only diffuse PM prior ──────────────────────────────────────────────
    if hst_pm_sigma_diffuse != 100.0:
        hst_star_indices = np.where(hst_only_mask)[0]
        if len(hst_star_indices) > 0:
            sigma_pm_inv2 = float(hst_pm_sigma_diffuse) ** -2
            solver._C_VG_inv_per_star[hst_star_indices, 2] = sigma_pm_inv2
            solver._C_VG_inv_per_star[hst_star_indices, 3] = sigma_pm_inv2

    # ── PM seeds for HST-only (callback) ──────────────────────────────────────
    _n_stars = len(gaia_catalog)
    pm_init  = np.full((_n_stars, 2), np.nan)
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

    # ── Phase 0: fixed-transformation pre-filter ───────────────────────────────
    r_init_hat = None
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
                r_align    = resid_mag[use]
                mad_sigma  = np.median(np.abs(r_align - np.median(r_align))) / 0.6745
                thresh     = max(5.0 * mad_sigma, 0.3)
                bad        = use & (resid_mag > thresh)
                n_flag0   += int(bad.sum())
                if bad.any():
                    d["use_for_fit"][bad]     = False
                    d["use_for_fit_max"][bad] = False
                    if "use_for_astrom" in d:
                        d["use_for_astrom"][bad] = False
        print(f"  Phase 0: {n_flag0} detections flagged")
    else:
        # No v1 results: use the fast_cross_match initialisation
        r_init_hat = np.concatenate([solver._img_data[img]["r_init"]
                                      for img in image_names])
        solver._update_R(r_init_hat)

    # ── Store xys_orig for all images ──────────────────────────────────────────
    for img in image_names:
        d = solver._img_data.get(img)
        if d is not None and 'xys_orig' not in d:
            d['xys_orig'] = d['xys'].copy()

    # ── CTE warm start ─────────────────────────────────────────────────────────
    # Compute field mean PM from Gaia-matched (CTE-free) xmatch PMs before the
    # warm start. Without this, HST-only stars' CTE-contaminated pmra_xmatch
    # inflate γ₀, and subsequent iterations converge to this biased warm start.
    _ws_field_pm = _compute_warmstart_field_pm(data_root, field_name)
    print(f"\n  CTE warm start (field_mean_pm={_ws_field_pm})...")
    cte_params = warm_start_cte(
        img_to_df, solver, image_names, r_init_hat, t_epoch0_yr,
        field_mean_pm=_ws_field_pm)

    # ── Outer CTE + BP3M Gauss-Newton loop ────────────────────────────────────
    clip      = clip_sigma if clip_sigma > 0 else None
    _min_outer = max(hst_enable_iter + 3, 4) if n_iter_bp3m >= hst_enable_iter else 4

    r_hat = C_r = v_hat = C_vT = a_arr = K_img = z_weights_out = None
    print(f"\n  Starting CTE outer loop ({n_iter_cte} iterations)...")

    for cte_iter in range(n_iter_cte):
        print(f"\n  ─── CTE iteration {cte_iter + 1}/{n_iter_cte} ───")
        t_iter = time.time()

        # Step 1: Apply CTE correction to solver xys
        apply_cte_to_solver(solver, image_names, cte_params, t_epoch0_yr,
                            filtered_spi=filtered_spi)

        # Step 2: BP3M solve
        print(f"  BP3M solve ({n_iter_bp3m} EM iterations)...")
        (r_hat, C_r, v_hat, C_vT,
         a_arr, K_img, z_weights_out) = solver.fit(
            n_iter=n_iter_bp3m,
            clip_sigma=clip,
            inflate_hst_errors=True,
            inflate_from_iter=0,
            min_outer_iters=_min_outer,
            # Only run solver prefilter on first iteration (Phase 0 already did it)
            prefilter=False,
            use_influence_clip=use_influence_clip,
            influence_d_thresh=influence_d_thresh,
            influence_sigma_min=influence_sigma_min,
            use_two_tier=True,
            # Fire callback only on iter 0 (enables HST-only sources)
            per_iter_callback=callback if cte_iter == 0 else None,
            use_soft_weights=use_soft_weights,
            student_t_nu=student_t_nu,
        )
        print(f"  BP3M done ({time.time() - t_iter:.1f}s)")

        # Update solver.R for the new r_hat so subsequent CTE computation is correct
        solver._update_R(r_hat)

        # Step 3a: Compute field mean PM from Gaia-matched member stars
        field_mean_pm = _compute_field_mean_pm(
            solver, gaia_catalog, hst_only_mask, v_hat)

        # Step 3b: Collect full-catalog residuals with current r_hat
        print(f"  Collecting residuals ({len(img_to_df)} images)  "
              f"[field PM: {field_mean_pm[0]:.2f}, {field_mean_pm[1]:.2f} mas/yr]...")
        residuals = collect_cte_residuals(
            img_to_df, solver, image_names, r_hat, t_epoch0_yr,
            field_mean_pm=field_mean_pm)

        # Step 4: Update CTE parameters
        gamma_before = {c: cte_params[c].gamma_y.copy() for c in ('hi', 'lo')}
        cte_params, info = update_cte_params(
            residuals, cte_params,
            n_inner=n_inner_delta,
            delta_tol=cte_delta_tol,
        )
        _save_cte_convergence(output_cte, cte_iter + 1, cte_params, info)

        # Step 5: Convergence check (require at least 2 iterations)
        gamma_norms  = [np.linalg.norm(cte_params[c].gamma_y) for c in ('hi', 'lo')]
        gamma_deltas = [np.linalg.norm(cte_params[c].gamma_y - gamma_before[c])
                        for c in ('hi', 'lo')]
        gamma_rchg = max(
            d / max(n, 1e-10) for d, n in zip(gamma_deltas, gamma_norms))

        print(f"  γ_y relative change = {gamma_rchg:.4e}  "
              f"δ_hi={cte_params['hi'].delta:.4f}  δ_lo={cte_params['lo'].delta:.4f}")

        if gamma_rchg < cte_gamma_rtol and cte_iter >= 1:
            print(f"  CTE converged at iteration {cte_iter + 1}")
            break

    # ── Final BP3M solve with converged CTE ───────────────────────────────────
    print("\n  Final BP3M solve with converged CTE parameters...")
    apply_cte_to_solver(solver, image_names, cte_params, t_epoch0_yr,
                        filtered_spi=filtered_spi)
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
    solver._update_R(r_hat)

    # ── Save converged CTE parameters ─────────────────────────────────────────
    cte_out = {}
    for chip in ('hi', 'lo'):
        p = cte_params[chip]
        cte_out[f'{chip}_delta']        = np.array([p.delta])
        cte_out[f'{chip}_gamma_x']      = p.gamma_x
        cte_out[f'{chip}_gamma_y']      = p.gamma_y
        cte_out[f'{chip}_y_readout']    = np.array([p.y_readout])
        cte_out[f'{chip}_y_readout_raw'] = np.array([p.y_readout_raw])
    cte_out['t_epoch0_yr'] = np.array([t_epoch0_yr])
    np.savez(output_cte / 'cte_params.npz', **cte_out)
    print(f"  Saved: cte_params.npz  "
          f"(δ_hi={cte_params['hi'].delta:.4f}, δ_lo={cte_params['lo'].delta:.4f})")

    # ── Sample posteriors ──────────────────────────────────────────────────────
    print(f"  Drawing {n_samples} posterior samples...")
    r_samp, v_mean, v_cov = solver.sample_posteriors(
        r_hat, C_r, a_arr, K_img, C_vT, n_samples=n_samples)

    # ── Save results (same format as v2) ──────────────────────────────────────
    _save_results(
        output_cte, solver, imgs, gaia_catalog, image_names,
        r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
        run_config={
            "n_iter_bp3m":   n_iter_bp3m,
            "n_iter_cte":    n_iter_cte,
            "n_samples":     n_samples,
            "clip_sigma":    clip_sigma,
            "poly_order":    poly_order,
            "t_epoch0_yr":   t_epoch0_yr,
            **{f'delta_{c}': float(cte_params[c].delta) for c in ('hi', 'lo')},
        },
    )

    # ── Save post-CTE full-catalog residuals ──────────────────────────────────
    print("\n  Saving post-CTE full-catalog residuals...")
    bp3m_gaia_ids = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)
    out_arrays_cte = _compute_full_catalog_residuals_from_df(
        img_to_df, bp3m_gaia_ids, solver, image_names, r_hat)
    if out_arrays_cte:
        np.savez(output_cte / 'detections_catalog_cte.npz', **out_arrays_cte)
        n_imgs  = sum(1 for k in out_arrays_cte if k.endswith('_X_c'))
        n_total = sum(len(v) for k, v in out_arrays_cte.items() if k.endswith('_X_c'))
        print(f"  Saved detections_catalog_cte.npz: {n_imgs} images, "
              f"{n_total:,} detections")

    # ── Soft-weight output ─────────────────────────────────────────────────────
    if use_soft_weights and z_weights_out is not None:
        import csv as _csv
        rows = []
        for img, z in z_weights_out.items():
            if z is None:
                continue
            d = solver._img_data[img]
            for k in range(len(z)):
                rows.append({'image': img, 'star_idx': int(d['sidx'][k]),
                             'z_det': float(z[k])})
        zcsv = output_cte / 'soft_weights.csv'
        with open(zcsv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=['image', 'star_idx', 'z_det'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: soft_weights.csv  ({len(rows)} detection weights)")
        _plot_dir = output_cte / 'plots'
        _plot_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_soft_weights(z_weights_out, solver, _plot_dir)
        except Exception as exc:
            print(f"  WARNING: soft_weights_diagnostic plot failed — {exc}")

    # ── Standard BP3M diagnostic plots ────────────────────────────────────────
    if not no_plots:
        try:
            from bp3m.pipeline.plot_results import make_plots
            print("  Generating standard BP3M diagnostic plots...")
            make_plots(solver, imgs, gaia_catalog,
                       r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
                       output_dir=output_cte)
        except Exception as exc:
            print(f"  WARNING: standard plots failed — {exc}")

        # ── CTE-specific diagnostic plots ─────────────────────────────────────
        before_npz = data_root / field_name / "BP3M_v2_results" / "detections_catalog.npz"
        after_npz  = output_cte / "detections_catalog_cte.npz"
        try:
            _plot_cte_diagnostics(
                output_cte, cte_params,
                before_npz=before_npz,
                after_npz=after_npz,
                image_names=image_names,
                solver=solver,
            )
        except Exception as exc:
            import traceback
            print(f"  WARNING: CTE diagnostic plots failed — {exc}")
            traceback.print_exc()

    print(f"\n  CTE results written to: {output_cte}")
    return output_cte
