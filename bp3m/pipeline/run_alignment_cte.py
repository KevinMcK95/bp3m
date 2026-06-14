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
  b_y = [Y', X·Y', Y'²]  where Y' = Y_c − Y_readout_c  (CTE_y=0 at readout)
  b_x = [X_c, X_c·Y_c, X_c²]                           (CTE_x=0 at X_c=0)
  γ_x_c, γ_y_c: (3,) composite coefficients (absorb temporal α)
  δ_c: shared flux exponent for chip c

Parameters: 14 total (7 per chip: δ, γ_x(3), γ_y(3)).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── ACS/WFC chip geometry constants ──────────────────────────────────────────
_HI_Y_READOUT = +2047.0
_LO_Y_READOUT = -2048.0
_MAG_REF      = 20.0


# ── CTE parameter dataclass ───────────────────────────────────────────────────

@dataclass
class CTEChipParams:
    chip: str
    y_readout: float
    delta: float = 1.0
    gamma_x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gamma_y: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> 'CTEChipParams':
        return CTEChipParams(chip=self.chip, y_readout=self.y_readout,
                             delta=float(self.delta),
                             gamma_x=self.gamma_x.copy(),
                             gamma_y=self.gamma_y.copy())


def default_cte_params() -> dict[str, CTEChipParams]:
    return {
        'hi': CTEChipParams(chip='hi', y_readout=_HI_Y_READOUT),
        'lo': CTEChipParams(chip='lo', y_readout=_LO_Y_READOUT),
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

def cte_y_basis(X_c: np.ndarray, Y_c: np.ndarray,
                y_readout: float) -> np.ndarray:
    """b_y = [Y', X_c·Y', Y'²]  (boundary: CTE_y = 0 at Y' = 0)."""
    Yp = Y_c - y_readout
    return np.stack([Yp, X_c * Yp, Yp ** 2], axis=1)


def cte_x_basis(X_c: np.ndarray, Y_c: np.ndarray) -> np.ndarray:
    """b_x = [X_c, X_c·Y_c, X_c²]  (boundary: CTE_x = 0 at X_c = 0)."""
    return np.stack([X_c, X_c * Y_c, X_c ** 2], axis=1)


# ── CTE displacement computation ──────────────────────────────────────────────

def compute_cte_displacement(
    X_c: np.ndarray, Y_c: np.ndarray,
    mag: np.ndarray, dt: np.ndarray,
    chip_params: CTEChipParams,
) -> np.ndarray:
    """
    CTE displacement in raw chip-centered pixel frame.

    Returns (n, 2) array of (δCTE_x, δCTE_y) in pixels.
    """
    phi  = phi_flux(mag, chip_params.delta)
    Phi  = dt * phi
    Bx   = cte_x_basis(X_c, Y_c)
    By   = cte_y_basis(X_c, Y_c, chip_params.y_readout)
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
        if spi_df is None or 'mag' not in spi_df.columns:
            continue

        # Build Gaia_id → mag lookup (vectorized, no iterrows)
        _gids = spi_df['Gaia_id'].to_numpy(dtype=np.int64)
        _mags = spi_df['mag'].to_numpy(dtype=float)
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
) -> None:
    """
    Apply CTE correction to solver._img_data[img]['xys'] for all images.

    Stores 'xys_orig' on first call.  Each subsequent call recomputes from
    xys_orig so corrections don't accumulate.

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

        delta_cte_raw = np.zeros((len(mag), 2))
        if ok.any():
            X_c = d['X_c']
            Y_c = d['Y_c']
            delta_cte_raw[ok] = compute_cte_displacement(
                X_c[ok], Y_c[ok], mag[ok], dt[ok], cte_params[chip])

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
) -> dict[str, dict]:
    """
    Collect per-chip GDC-frame residuals for updating CTE parameters.

    Uses the full master-catalog detection set (all ~127k stars via img_to_df),
    not just the solver's BP3M alignment stars.  img_to_df is the output of
    _load_full_catalog_df and should be cached across CTE iterations.

    Returns
    -------
    residuals : {'hi': {...}, 'lo': {...}} each with arrays:
        dx, dy : (n,) GDC residuals [pixels]
        X_c, Y_c, mag, dt, z : per-detection geometry/weighting
    """
    from astropy.time import Time
    from .run_alignment_v2 import _compute_full_catalog_residuals_from_df

    bp3m_gaia_ids = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)
    out_arrays = _compute_full_catalog_residuals_from_df(
        img_to_df, bp3m_gaia_ids, solver, image_names, r_hat)

    residuals = {c: {'dx': [], 'dy': [], 'X_c': [], 'Y_c': [],
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

        X_c = out_arrays[f'{img}_X_c'].astype(float)
        Y_c = out_arrays[f'{img}_Y_c'].astype(float)
        dx  = out_arrays[f'{img}_dx_gdc'].astype(float)
        dy  = out_arrays[f'{img}_dy_gdc'].astype(float)
        mag = out_arrays[f'{img}_mag_inst'].astype(float)

        ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)
        n_ok = int(ok.sum())
        if n_ok == 0:
            continue

        residuals[chip]['dx'].append(dx[ok])
        residuals[chip]['dy'].append(dy[ok])
        residuals[chip]['X_c'].append(X_c[ok])
        residuals[chip]['Y_c'].append(Y_c[ok])
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
) -> tuple[dict[str, CTEChipParams], dict]:
    """
    Update CTE parameters (γ, δ) from GDC-frame residuals via Gauss-Newton.

    Linear step: solve for γ_x, γ_y (3 coefficients each).
    Nonlinear step: augment design matrix with analytic dφ/dδ column,
    solve for Δδ jointly, update δ ← δ + Δδ.

    Returns updated copy of cte_params and convergence info dict.
    """
    new_params = {c: cte_params[c].copy() for c in ('hi', 'lo')}
    info = {}

    for chip in ('hi', 'lo'):
        res = residuals_by_chip[chip]
        if len(res['dx']) == 0:
            info[chip] = {'converged': True, 'n_inner': 0, 'n_det': 0}
            continue

        dx  = res['dx'].astype(float)
        dy  = res['dy'].astype(float)
        X_c = res['X_c'].astype(float)
        Y_c = res['Y_c'].astype(float)
        mag = res['mag'].astype(float)
        dt  = res['dt'].astype(float)
        z   = res['z'].astype(float)

        ok = (np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)
              & np.isfinite(dt) & (np.abs(dt) > 0))
        dx, dy, X_c, Y_c, mag, dt, z = (arr[ok] for arr in
                                          (dx, dy, X_c, Y_c, mag, dt, z))
        n = len(dx)

        if n < 10:
            info[chip] = {'converged': True, 'n_inner': 0, 'n_det': n}
            continue

        p         = new_params[chip]
        delta_n   = float(p.delta)
        y_readout = p.y_readout

        delta_history = [delta_n]
        for it in range(n_inner):
            phi  = phi_flux(mag, delta_n)
            dphi = dphi_ddelta(mag, delta_n)
            Phi  = dt * phi
            dPhi = dt * dphi

            Bx = cte_x_basis(X_c, Y_c)
            By = cte_y_basis(X_c, Y_c, y_readout)

            # Linearized δ column: d(Φ·B·γ)/dδ = dΦ·B·γ_current
            Bx_g = Bx @ p.gamma_x
            By_g = By @ p.gamma_y

            # Design matrices (n, 4): [γ(3) | Δδ(1)]
            Ax = np.column_stack([Phi[:, None] * Bx, dPhi * Bx_g])
            Ay = np.column_stack([Phi[:, None] * By, dPhi * By_g])

            def _wls(A, r):
                AtW  = A.T * z[None, :]
                AtWA = AtW @ A
                AtWr = AtW @ r
                reg        = np.zeros(4)
                reg[:3]    = regularize
                AtWA      += np.diag(reg)
                try:
                    return np.linalg.solve(AtWA, AtWr)
                except np.linalg.LinAlgError:
                    return None

            tx = _wls(Ax, dx)
            ty = _wls(Ay, dy)
            if tx is None or ty is None:
                break

            p.gamma_x   = tx[:3]
            p.gamma_y   = ty[:3]
            delta_delta = 0.5 * (tx[3] + ty[3])
            delta_n    += delta_delta
            p.delta     = delta_n
            delta_history.append(delta_n)

            if abs(delta_delta) < delta_tol:
                break

        # Residual RMS after update
        phi_f = phi_flux(mag, p.delta)
        rms_x = float(np.sqrt(np.mean(
            (dx - dt * phi_f * (cte_x_basis(X_c, Y_c) @ p.gamma_x)) ** 2)))
        rms_y = float(np.sqrt(np.mean(
            (dy - dt * phi_f * (cte_y_basis(X_c, Y_c, y_readout) @ p.gamma_y)) ** 2)))

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


# ── Warm start ────────────────────────────────────────────────────────────────

def warm_start_cte(
    img_to_df: dict,
    solver,
    image_names: list[str],
    r_hat_init: np.ndarray,
    t_epoch0_yr: float,
) -> dict[str, CTEChipParams]:
    """
    Estimate initial γ from BP3M v2 (or v1) post-fit residuals.

    Collects full-catalog residuals using r_hat_init (the pre-CTE transformation),
    then regresses onto the CTE basis with δ=1.0 fixed.  This gives a starting
    point that reduces the first-iteration residual.
    """
    print("  Warm-starting CTE model from initial transformation residuals...")
    residuals = collect_cte_residuals(
        img_to_df, solver, image_names, r_hat_init, t_epoch0_yr)

    params = default_cte_params()

    for chip in ('hi', 'lo'):
        cd = residuals[chip]
        if len(cd['dx']) == 0:
            continue

        dt  = cd['dt'].astype(float)
        mag = cd['mag'].astype(float)
        ok  = np.isfinite(cd['dx']) & np.isfinite(cd['dy']) & np.isfinite(mag) & (np.abs(dt) > 0)
        if ok.sum() < 10:
            continue

        dx  = cd['dx'][ok]
        dy  = cd['dy'][ok]
        X_c = cd['X_c'][ok]
        Y_c = cd['Y_c'][ok]
        mag = mag[ok]
        dt  = dt[ok]

        delta_fixed = 1.0
        phi = phi_flux(mag, delta_fixed)
        Phi = dt * phi

        Ax = Phi[:, None] * cte_x_basis(X_c, Y_c)
        Ay = Phi[:, None] * cte_y_basis(X_c, Y_c, params[chip].y_readout)

        try:
            gamma_x, _, _, _ = np.linalg.lstsq(Ax, dx, rcond=None)
            gamma_y, _, _, _ = np.linalg.lstsq(Ay, dy, rcond=None)
        except np.linalg.LinAlgError:
            continue

        params[chip].delta   = delta_fixed
        params[chip].gamma_x = gamma_x
        params[chip].gamma_y = gamma_y

        rms_y = float(np.sqrt(np.mean((dy - Ay @ gamma_y) ** 2)))
        print(f"    {chip}: δ=1.0 (fixed)  "
              f"|γ_y|={np.linalg.norm(gamma_y):.4e}  rms_y={rms_y:.4f}px  "
              f"n={ok.sum():,}")

    return params


# ── Convergence CSV ────────────────────────────────────────────────────────────

def _save_cte_convergence(output_dir: Path, outer_iter: int,
                          cte_params: dict, info: dict) -> None:
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
            p  = cte_params[chip]
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
                'rms_y':    f'{ci.get("rms_y", float("nan")):.6f}',
                'rms_x':    f'{ci.get("rms_x", float("nan")):.6f}',
                'n_det':    ci.get('n_det', 0),
            })


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
        recs = {c: {'X_c': [], 'Y_c': [], 'dx': [], 'dy': [], 'mag': [],
                    'dt': [], 'epoch_yr': []} for c in ('hi', 'lo')}
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
            ok  = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)
            n   = ok.sum()
            if n == 0:
                continue
            recs[chip]['X_c'].append(X_c[ok])
            recs[chip]['Y_c'].append(Y_c[ok])
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

    # ── Figure 1: CTE amplitude vs Y_c for mag bins ───────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        Y_grid = np.linspace(-2100, 2100, 500)
        X_grid = np.zeros_like(Y_grid)

        # Use epoch with largest |dt| for amplitude display
        all_mjds  = [float(solver.images[img]['hst_time_mjd'])
                     for img in image_names if img in solver.images]
        t0_mjd    = min(all_mjds)
        t1_mjd    = max(all_mjds)
        dt_max    = (t1_mjd - t0_mjd) / 365.25

        for col_i, chip in enumerate(('hi', 'lo')):
            ax = axes[col_i]
            p  = cte_params[chip]
            for mag_v, col, lbl in [(18.0, 'steelblue', 'mag=18'),
                                    (20.0, 'green',     'mag=20'),
                                    (22.0, 'firebrick', 'mag=22')]:
                phi  = float(phi_flux(np.array([mag_v]), p.delta)[0])
                By   = cte_y_basis(X_grid, Y_grid, p.y_readout)
                dcte = dt_max * phi * (By @ p.gamma_y)
                ax.plot(Y_grid, dcte, color=col, lw=1.8, label=lbl)

            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
            ax.axvline(p.y_readout, color='k', lw=0.8, ls=':', alpha=0.5)
            ax.set_xlabel('Y_c (centered GDC px)')
            ax.set_ylabel('δCTE_y (px)' if col_i == 0 else '')
            ax.set_title(f'_{chip} chip — CTE_y amplitude (at Δt={dt_max:.1f} yr)')
            ax.legend(fontsize=9)
        fig.suptitle('CTE correction amplitude (X_c=0, converged parameters)',
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

    # ── Figure 2: dy_gdc vs Y_c before/after by chip × mag tertile ────────────
    if data_before is not None or data_after is not None:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')

            # Determine global mag tertiles
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
            bin_labels = [f'bright ({pcts[0]:.1f}–{pcts[1]:.1f})',
                          f'mid ({pcts[1]:.1f}–{pcts[2]:.1f})',
                          f'faint ({pcts[2]:.1f}–{pcts[3]:.1f})']

            for row_i, chip in enumerate(('hi', 'lo')):
                for col_i, (src, label, ls) in enumerate(
                        [(data_before, 'before CTE', '-'),
                         (data_after,  'after CTE',  '--')]):
                    ax = axes[row_i, col_i]
                    if src is None:
                        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                                ha='center', va='center')
                        continue
                    cd = src[chip]
                    if len(cd['dy']) == 0:
                        continue
                    for bi, (col, blbl) in enumerate(zip(bin_cols, bin_labels)):
                        m = cd['mag']
                        mask = (np.isfinite(m) & (m >= pcts[bi]) & (m < pcts[bi+1])
                                & np.isfinite(cd['dy']))
                        if mask.sum() < 10:
                            continue
                        xm, ym, ye = _binned(cd['Y_c'][mask], cd['dy'][mask])
                        ax.plot(xm, ym, ls=ls, color=col, lw=1.8,
                                label=blbl, zorder=4)
                        ax.fill_between(xm, ym-ye, ym+ye, color=col,
                                        alpha=0.18, zorder=3)
                    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
                    ax.set_xlabel('Y_c (px)')
                    ax.set_ylabel(r'$\delta y_\mathrm{GDC}$ (px)')
                    ax.set_title(f'_{chip} chip — {label}')
                    ax.set_ylim(-0.12, 0.12)
                    ax.legend(fontsize=7, ncol=1, loc='upper left')

            fig.suptitle('dy_gdc vs Y_c by magnitude: before vs after CTE correction',
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(plot_dir / 'cte_before_after.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/cte_before_after.png")
        except Exception as exc:
            print(f"  WARNING: cte_before_after.png failed — {exc}")

    # ── Figure 3: CTE slope vs magnitude before/after ─────────────────────────
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

            styles = {
                ('hi', 'before'): dict(fmt='o-',  color='steelblue',  label='_hi before'),
                ('hi', 'after'):  dict(fmt='o--', color='steelblue',  label='_hi after',
                                       mfc='none'),
                ('lo', 'before'): dict(fmt='s-',  color='darkorange', label='_lo before'),
                ('lo', 'after'):  dict(fmt='s--', color='darkorange', label='_lo after',
                                       mfc='none'),
            }

            for col_i, comp in enumerate(('dy', 'dx')):
                ax = axes[col_i]
                ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
                for (chip, when), sty in styles.items():
                    src = data_before if when == 'before' else data_after
                    if src is None:
                        continue
                    cd = src[chip]
                    if len(cd[comp]) == 0:
                        continue
                    slopes, errs, ns = [], [], []
                    for bi in range(N_MAG_BINS):
                        m = cd['mag']
                        mask = (np.isfinite(m) & (m >= mag_edges[bi])
                                & (m < mag_edges[bi+1])
                                & np.isfinite(cd[comp]))
                        if mask.sum() < 8:
                            slopes.append(np.nan); errs.append(np.nan); ns.append(0)
                            continue
                        sl, sl_e = _slope(cd['Y_c'][mask], cd[comp][mask])
                        slopes.append(sl); errs.append(sl_e); ns.append(mask.sum())
                    slopes = np.array(slopes); errs = np.array(errs)
                    ok = np.isfinite(slopes)
                    if ok.any():
                        fmt = sty.pop('fmt')
                        ax.errorbar(mag_mids[ok], slopes[ok], yerr=errs[ok],
                                    fmt=fmt, ms=6, capsize=3, lw=1.5, **sty)
                        sty['fmt'] = fmt   # put back for next iteration

                comp_lbl = (r'slope $\delta y_\mathrm{GDC}$ (px/px)'
                            if comp == 'dy'
                            else r'slope $\delta x_\mathrm{GDC}$ (px/px)')
                ax.set_xlabel('Instrumental magnitude', fontsize=10)
                ax.set_ylabel(comp_lbl, fontsize=10)
                ax.set_title(f'CTE slope vs magnitude ({comp})', fontsize=11)
                ax.legend(fontsize=8)

            fig.suptitle('CTE slope d(residual)/d(Y_c): before vs after correction',
                         fontsize=11)
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
                gy_norm = np.sqrt(sub['gamma_y0']**2 + sub['gamma_y1']**2
                                  + sub['gamma_y2']**2)
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
    print("\n  CTE warm start...")
    cte_params = warm_start_cte(
        img_to_df, solver, image_names, r_init_hat, t_epoch0_yr)

    # ── Outer CTE + BP3M Gauss-Newton loop ────────────────────────────────────
    clip      = clip_sigma if clip_sigma > 0 else None
    _min_outer = max(hst_enable_iter + 3, 4) if n_iter_bp3m >= hst_enable_iter else 4

    r_hat = C_r = v_hat = C_vT = a_arr = K_img = z_weights_out = None
    print(f"\n  Starting CTE outer loop ({n_iter_cte} iterations)...")

    for cte_iter in range(n_iter_cte):
        print(f"\n  ─── CTE iteration {cte_iter + 1}/{n_iter_cte} ───")
        t_iter = time.time()

        # Step 1: Apply CTE correction to solver xys
        apply_cte_to_solver(solver, image_names, cte_params, t_epoch0_yr)

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

        # Step 3: Collect full-catalog residuals with current r_hat
        print(f"  Collecting residuals ({len(img_to_df)} images)...")
        residuals = collect_cte_residuals(
            img_to_df, solver, image_names, r_hat, t_epoch0_yr)

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
    solver._update_R(r_hat)

    # ── Save converged CTE parameters ─────────────────────────────────────────
    cte_out = {}
    for chip in ('hi', 'lo'):
        p = cte_params[chip]
        cte_out[f'{chip}_delta']    = np.array([p.delta])
        cte_out[f'{chip}_gamma_x']  = p.gamma_x
        cte_out[f'{chip}_gamma_y']  = p.gamma_y
        cte_out[f'{chip}_y_readout'] = np.array([p.y_readout])
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
