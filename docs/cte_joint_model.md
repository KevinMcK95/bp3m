# Joint CTE + Astrometry Model for HST ACS/WFC

## Overview

This document describes the design and implementation plan for jointly fitting HST image
transformations (r_j), stellar astrometry (v_i), and a parametric CTE (Charge Transfer
Efficiency) model within the BP3M framework. The code lives in
`bp3m/pipeline/run_alignment_cte.py`.

## Motivation

The current BP3M v2 alignment absorbs CTE-induced position errors into the per-image
transformation r_j for the brightest Gaia-matched alignment stars. This is inconsistent
because:

1. CTE residuals are magnitude-dependent but r_j fits a magnitude-independent
   transformation. The absorbed CTE therefore depends on the magnitude distribution
   of whichever stars happen to be in a given image.
2. Faint stars (which dominate the full catalog) experience systematically larger CTE
   trailing than the bright Gaia alignment stars. The absorbed correction is wrong for them.
3. The 1.66M detections in `detections_catalog.npz` contain clear magnitude- and
   Y-position-dependent residuals consistent with CTE, motivating a direct physical model.

The joint model solves all three problems simultaneously.

---

## Physical Model

### ACS/WFC Chip Geometry

- Two chips: `_hi` (chip 1, extension 1) and `_lo` (chip 2, extension 4).
- Parallel-register readout runs along +Y for `_hi` and along -Y for `_lo`.
- Readout registers at outer chip edges:
  - `_hi`: Y_readout ≈ +2047 (top, Y increasing toward register)
  - `_lo`: Y_readout ≈ -2048 (bottom, Y decreasing toward register)
- Serial amplifiers at X_c ≈ ±2048 (chip edges), so CTE_x boundary: CTE_x = 0 at
  |X_c| = 2048.

### CTE Displacement (Raw Chip Frame)

For chip c, detection in image j (epoch t_j), with centered pixel position (X_c, Y_c):

```
δCTE_x(X_c, Y_c, mag, t_j; θ_c) = (t_j - t_0) · φ(mag; δ_c) · b_x(X_c, Y_c) · γ_x_c
δCTE_y(X_c, Y_c, mag, t_j; θ_c) = (t_j - t_0) · φ(mag; δ_c) · b_y(X_c, Y_c) · γ_y_c
```

where:
- `t_0 = t_epoch0` = first exposure MJD (NOT ACS launch date 2002-03-01).
  Using t_epoch0 avoids the unidentifiability of the absolute CTE level at a single
  epoch; only the differential CTE between epochs is learnable.
- `φ(mag; δ) = 10^{0.4·δ·(mag − mag_ref)} − 1` is the flux power-law function.
  This is physically motivated: CTE trailing depends on electron count (flux), not
  log(flux). `mag_ref = 20.0` (typical star in the sample). At δ=0, φ=0 (no CTE).
  The sign convention is that increasing δ makes φ more negative for faint stars
  (mag > mag_ref), meaning brighter stars are more impacted.
- `b_y(X_c, Y_c) = [Y', X_c·Y', Y'²]`  where  `Y' = Y_c − Y_readout_c`.
  **Boundary condition**: CTE_y = 0 at the readout register (Y' = 0). The 3
  polynomial basis terms all contain Y' as a factor, guaranteeing this.
- `b_x(X_c, Y_c) = [X_c, X_c·Y_c, X_c²]`.
  **Boundary condition**: CTE_x = 0 at X_c = 0 (chip center). All basis terms
  contain X_c as a factor. Serial CTE is typically much smaller than parallel CTE
  and may be consistent with zero.
- `γ_x_c` (3,) and `γ_y_c` (3,) are **composite** polynomial coefficients that absorb
  the temporal amplitude α. Since h(t) = α·(t_j − t_0) is linear in t, α is
  degenerate with the magnitude of γ and is not a free parameter.
- `δ_c` is shared between CTE_x and CTE_y for chip c (same electron-count physics).
  Each chip has its own δ.

### Parameter Summary (14 total per run)

| Parameter    | Shape | Description                                    |
|--------------|-------|------------------------------------------------|
| `δ_hi`       | (1,)  | Flux power-law exponent, _hi chip              |
| `γ_x_hi`     | (3,)  | X composite polynomial coefficients, _hi       |
| `γ_y_hi`     | (3,)  | Y composite polynomial coefficients, _hi       |
| `δ_lo`       | (1,)  | Flux power-law exponent, _lo chip              |
| `γ_x_lo`     | (3,)  | X composite polynomial coefficients, _lo       |
| `γ_y_lo`     | (3,)  | Y composite polynomial coefficients, _lo       |

---

## Forward Model Integration with BP3M

### Solver Integration (no solver.py changes)

The CTE correction is applied by modifying `solver._img_data[img]['xys']` before each
BP3M solve pass. The workflow:

1. **Before the outer loop**: store `d['xys_orig'] = d['xys'].copy()` for all images.
2. **Each CTE iteration**:
   - Compute `δCTE_raw` = (δCTE_x, δCTE_y) in raw chip-centered pixel frame.
   - Map to pseudo-image frame: `Δxys = R_j @ δCTE_raw` (2×2 rotation from r_j).
   - Set `d['xys'] = d['xys_orig'] + Δxys`.
3. **Run BP3M solve** (r_j, v_i) with CTE-corrected xys.
4. **Collect full-catalog residuals** from detections_catalog.npz (all ~127k stars).
5. **Update CTE parameters** (γ, δ) from residuals.

The rotation matrix `R_j = solver.R[img]` (2×2, updated by `solver._update_R`) converts
chip-frame pixel shifts to pseudo-image-frame shifts for poly_order=1. This is the same
matrix used in `compute_gdc_residuals` in solver.py.

### Soft Weights Compatibility

The CTE correction modifies `xys` before the IRLS computation. Since z_{ij} weights in
soft-weight IRLS are applied to `(xys - X_mat @ r_j - JU @ v_i)^2`, and `xys` now
includes the CTE correction, the z weights automatically down-weight CTE-corrected
detections with large residuals. No special handling is needed.

---

## Parameter Update Algorithm

### Linear update for γ (holding δ fixed)

Given current residuals in GDC frame `(dx_k, dy_k)` for detection k in image j (chip c),
the CTE design matrix columns are:

```
Φ_k = (t_j − t_0) · φ(mag_k; δ_c)
```

For CTE_y: design row = `Φ_k · [Y'_k, X_k·Y'_k, Y'_k²]` (3 columns)
For CTE_x: design row = `Φ_k · [X_k, X_k·Y_k, X_k²]` (3 columns)

Note: the GDC-frame residuals already include R_j^{-1}, so this directly solves for
the chip-frame γ coefficients.

Stack all detections for chip c across all images → solve a (2N, 6) weighted least
squares system: `min_γ Σ_k z_k · ||residual_k − A_k @ γ_c||²`.

The x and y components are independent (A_x only depends on CTE_x basis, A_y only on
CTE_y basis), so they can be solved separately as two (N, 3) systems.

### Gauss-Newton update for δ (nonlinear)

`φ(mag; δ) = 10^{0.4·δ·(mag − mag_ref)} − 1`
`dφ/dδ = 0.4·ln(10)·(mag − mag_ref)·10^{0.4·δ·(mag − mag_ref)}`

Augment the design matrix with a column for Δδ:
`extra_col_k = (t_j − t_0) · dφ/dδ(mag_k; δ_c^{(n)}) · b(X_k, Y_k)`

This extra column has length 1 (scalar Δδ per chip). Solve for [γ_c, Δδ_c] jointly in
the augmented (N, 7) system, then update `δ_c ← δ_c + Δδ_c`.

Repeat until |Δδ_c| < tolerance (typically 3-5 inner iterations suffice).

---

## Warm-Start Strategy

Before the outer Gauss-Newton loop, estimate initial γ from the existing BP3M v2
residuals in `detections_catalog.npz`:

1. Load `detections_catalog.npz` (post-BP3M-v2 residuals).
2. For each chip, select faint stars (mag > mag_ref) where CTE is largest.
3. Build CTE design matrix assuming δ=1.0 (pure linear flux model) and solve for γ_c.
4. This provides a first estimate for γ that reduces the initial CTE residual.

The warm start uses `dx_gdc`/`dy_gdc` from the catalog residuals, which already have
the BP3M v2 transformation absorbed — they represent what the CTE model needs to explain.

---

## Implementation Steps

### Step 1: CTEModel dataclass
```python
@dataclass
class CTEChipParams:
    delta: float = 1.0       # flux power-law exponent
    gamma_x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gamma_y: np.ndarray = field(default_factory=lambda: np.zeros(3))
    y_readout: float = 0.0   # Y_c at readout register
    chip: str = 'hi'
```

### Step 2: Flux model functions
- `phi_flux(mag, delta, mag_ref=20.0)` → array, analytic
- `dphi_ddelta(mag, delta, mag_ref=20.0)` → array, analytic gradient

### Step 3: CTE basis functions
- `cte_y_basis(X_c, Y_c, y_readout)` → (n, 3), boundary-satisfying
- `cte_x_basis(X_c, Y_c)` → (n, 3), boundary-satisfying

### Step 4: `compute_cte_displacement(X_c, Y_c, mag, dt, chip_params)`
- Returns (n, 2) array of (δCTE_x, δCTE_y) in raw chip-centered pixels.

### Step 5: `apply_cte_to_solver(solver, image_names, cte_params)`
- Stores `xys_orig` on first call.
- Updates `d['xys']` for all images using current CTE parameters.
- Uses `solver.R[img]` to rotate chip-frame correction to pseudo-image frame.

### Step 6: `warm_start_cte(detections_catalog_path, image_names, solver)`
- Loads `detections_catalog.npz`.
- Builds per-chip design matrices, solves for γ with δ=1.0.
- Returns initial `CTEChipParams` for _hi and _lo.

### Step 7: `update_cte_params(residuals_by_chip, cte_params, z_weights, n_inner=5)`
- Runs Gauss-Newton inner loop to solve for [γ_c, Δδ_c].
- Returns updated `CTEChipParams`.

### Step 8: `collect_residuals(solver, image_names, r_hat, cte_params, data_root, field_name)`
- Uses same geometry as `_save_full_catalog_residuals` but returns arrays (not saves).
- Returns per-chip (dx, dy, X_c, Y_c, mag, dt, z_weight) arrays for all detections.

### Step 9: `run_alignment_cte(...)` main function
- Outer loop: apply_cte → BP3M fit → collect_residuals → update_cte → repeat.
- Convergence: ‖Δγ‖/‖γ‖ < tol and |Δδ| < delta_tol.

---

## Convergence Behavior and Diagnostics

Expected behavior:
- After 2-3 outer iterations, γ_y should converge to nonzero values for both chips.
- δ_hi and δ_lo should converge to values near 0.3–0.6 (empirically from ACS/WFC
  literature). Values near 0 indicate CTE is purely a linear function of flux (standard
  assumption); values near 1 indicate quadratic flux dependence.
- CTE_x coefficients are expected to be small; if they converge to ~0 within noise, the
  serial CTE is negligible for this dataset.

Diagnostic outputs saved to `BP3M_cte_results/`:
- `cte_params.npz`: converged CTEChipParams for each chip.
- `detections_catalog_cte.npz`: post-CTE residuals (same format as detections_catalog.npz).
- `cte_convergence.csv`: per-outer-iteration γ_c, δ_c, RMS residuals.
- `cte_diagnostic.png`: 4-panel plot of CTE correction amplitude vs Y_c and magnitude.

---

## Tests with Leo I (2 Epochs: 2006, 2011)

With only 2 epochs, the temporal model is fully linear (h(t) = t − t_0). Key tests:

1. **Do both chips agree on δ?** Physical expectation: yes (same silicon, same radiation
   environment), but the readout geometry differs. A large discrepancy suggests a modeling
   error or that chip-specific PSF effects are confounded with CTE.

2. **Do χ² vs magnitude improve?** Compare `cte_diagnostic_bright_v1.png` slope
   (dy_gdc vs Y_c as function of mag) before and after CTE correction. The slope should
   approach zero, especially for faint stars.

3. **PM sensitivity test**: Compare Leo I bulk proper motion (field mean μ_α*, μ_δ)
   from BP3M v2 vs BP3M-CTE. CTE effects project partly onto the bulk PM; any shift
   exceeding ~0.1 mas/yr is significant.

4. **Comparison with pixel-level correction**: The FLC files apply a pixel-level CTE
   correction (Anderson & Bedin 2010). The residual CTE captured by our model is
   whatever the pixel correction missed. Its magnitude indicates how well the standard
   correction performs at 15+ years post-launch.

5. **Individual star PM uncertainties**: After CTE correction, faint stars should have
   reduced scatter in their per-image residuals, leading to smaller formal PM errors.

---

## Future Work: Draco dSph and Multi-Epoch Fields

For fields with N > 2 HST epochs (e.g., Draco_dSph from the HST archive), the linear
temporal model h(t) = t − t_0 may be insufficient.

**Planned extension**:
- Replace the linear model with a piecewise-linear or low-order polynomial in t.
- The identifiability constraint changes: with ≥3 epochs, the absolute CTE level at
  t_0 may be partially constrained (vs purely differential with 2 epochs).
- Allow α(t) to be a free polynomial in t: `h_k(t) = (t − t_0)^k` for k=1,2,...
  Each power becomes an independent set of composite coefficients γ_ck.
- This introduces a more complex bordered block structure but the same Gauss-Newton
  approach applies.

**Other future improvements**:
- Fit CTE separately per sub-image (jitter in readout efficiency).
- Cross-validate δ_hi vs δ_lo — if they agree well across multiple fields, fix δ
  as a global ACS/WFC constant and solve only for γ per field.
- Use the magnitude-dependence of CTE to improve stellar mass estimates (confusion
  between intrinsic color and CTE-induced photometric error).
- Apply the learned CTE model as a correction to the FLC positions before running
  the standard pipeline (feedback loop into py1pass).
- Test whether CTE_x (serial register) is detectable with deep, high-stellar-density
  fields (e.g., ω Cen, 47 Tuc).

---

## References

- Anderson & Bedin (2010): pixel-level ACS/WFC CTE correction (FLC pipeline)
- Massey (2010): power-law flux model for CTE, Equation 1 motivates φ(mag; δ)
- ACS Instrument Science Report ACS 2012-03: time-dependent CTE model for ACS/WFC
- McKinnon et al. (2024): BP3M algorithm (bp3m paper; see memory/reference_key_papers.md)
