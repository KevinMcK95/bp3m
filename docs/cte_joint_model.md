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
  - `_hi`: readout at raw row ≈ 2048 (gap edge); CTE trails toward increasing raw y
  - `_lo`: readout at raw row ≈ 0 (gap edge); CTE trails toward increasing raw y
- In py1pass's unified global frame (0..~4096): lo chip occupies y_raw ∈ [8, 2039],
  hi chip occupies y_raw ∈ [2056, 4087]. Both chips read away from the gap.

### CTE Displacement (Raw Chip Frame)

For chip c (hi/lo), detection k in image j at epoch t_j:

```
δCTE_x_k = (t_j − t_launch) · func1(mag_k) · b(xt_k, yt_k) · γ_x_c
δCTE_y_k = (t_j − t_launch) · func1(mag_k) · b(xt_k, yt_k) · γ_y_c
```

where:

- `t_launch` = ACS launch date (2002-03-01, JYear ≈ 2002.165). Using the absolute launch
  date means both epochs receive nonzero corrections weighted by their time since launch,
  allowing the model to constrain the absolute CTE level (not just differential CTE
  between epochs). For Leo I: epoch 1 (2006) dt ≈ 4.0 yr, epoch 2 (2011) dt ≈ 9.0 yr.

- `func1(mag_raw)` is a fixed 3rd or 4th order polynomial in the raw instrumental
  magnitude. This encodes the physical flux dependence of CTE: fainter stars (higher mag,
  fewer electrons) experience stronger parallel CTE trailing. The polynomial is evaluated
  per star and produces a single scalar — there are no free parameters in func1 itself.
  All amplitude and spatial information is absorbed into the free γ coefficients.
  A natural choice approximates the empirical flux-dependence over the data's magnitude
  range (e.g., a normalised polynomial fit from pilot residuals, or func1(mag) ∝ mag^3).

- `b(xt, yt) = [yt, yt², xt·yt, xt²·yt, xt·yt²]` (5 terms; yt appears in every term).
  **Boundary condition**: all terms vanish at yt = 0 (readout register), so δCTE = 0
  at the readout by construction. The 5 terms capture:
  - `yt`: linear CTE growth with distance from readout
  - `yt²`: quadratic growth (second-order effect in parallel distance)
  - `xt·yt`: first-order variation of CTE amplitude across chip width
  - `xt²·yt`: second-order variation across chip width
  - `xt·yt²`: coupling between chip-width position and quadratic CTE growth

- `(xt, yt)` are normalised chip-local coordinates:
  ```
  xt = (x_raw − x0) / 2048
  yt = (y_raw − y_readout) / 2048
  ```
  where `x0 = 2048` (chip centre between serial amplifiers) and `y_readout` is the raw
  readout row for each chip (lo ≈ 0, hi ≈ 2048 in the py1pass global frame). Division
  by 2048 normalises both to approximately [−1, 1] over the detector, keeping the
  5-term polynomial basis well-conditioned.

- `γ_x_c` (5,) and `γ_y_c` (5,) are the free CTE coefficients per chip c.

Note: Both δCTE_x and δCTE_y use the same 5-term spatial basis b(xt, yt). The
yt-in-all-terms boundary condition is physically motivated for parallel CTE (y-direction).
For serial CTE (x-direction) the boundary condition is less well-defined physically, but
if serial CTE is small γ_x will converge to ≈ 0.

### Parameter Summary (10 per chip, 20 total)

| Parameter    | Shape | Description                           |
|--------------|-------|---------------------------------------|
| `γ_x_hi`     | (5,)  | X CTE coefficients, _hi chip          |
| `γ_y_hi`     | (5,)  | Y CTE coefficients, _hi chip          |
| `γ_x_lo`     | (5,)  | X CTE coefficients, _lo chip          |
| `γ_y_lo`     | (5,)  | Y CTE coefficients, _lo chip          |

With func1 as a fixed scalar per star (no free parameters), the 5-term spatial basis gives
5 free coefficients per CTE direction. With 2 directions (x, y) and 2 chips: **10 per chip,
20 total**. If γ were shared between chips (assuming identical CTE physics for hi and lo),
the total would reduce to 10.

The previous `δ` (flux power-law exponent) parameter has been removed. The magnitude
dependence is now captured by the fixed polynomial func1, leaving γ as the only free
CTE parameters.

---

## Current GitHub Model vs Proposed Model

| Aspect                    | Current (on GitHub)                                 | Proposed                                        |
|---------------------------|-----------------------------------------------------|-------------------------------------------------|
| **Time reference**        | `t − t_epoch0` (first exposure MJD)                 | `t − t_launch` (ACS launch 2002-03-01)         |
| **Magnitude function**    | `φ(mag; δ) = 10^{0.4δ(mag−mag_ref)} − 1` (free δ)  | `func1(mag)` = fixed polynomial in mag_raw     |
| **y-CTE basis**           | `[Y', Xc·Y']` (2 terms, un-normalised)              | `[yt, yt², xt·yt, xt²·yt, xt·yt²]` (5 terms)  |
| **x-CTE basis**           | `[Xc, Xc·Y']` (2 terms, un-normalised)              | `[yt, yt², xt·yt, xt²·yt, xt·yt²]` (5 terms, same as y) |
| **Coordinates**           | `Y' = y_raw − y_readout_raw`, `Xc = x_gdc − 2048`  | `yt = (y_raw − y_readout)/2048`, `xt = (x_raw − 2048)/2048` |
| **Free params per chip**  | δ(1) + γ_x(2) + γ_y(2) = **5**                     | γ_x(5) + γ_y(5) = **10**                       |
| **Total CTE params**      | 5×2 chips = **10**                                  | 10×2 chips = **20**                             |
| **δ update**              | Gauss-Newton inner loop (nonlinear)                 | None (func1 is fixed — fully linear solve)      |

Key changes:
- **t_launch vs t_epoch0**: switching to the ACS launch date gives each epoch its own
  nonzero CTE amplitude, allowing the model to constrain absolute (not just differential)
  CTE level.
- **Polynomial vs power-law**: removing the free δ parameter simplifies the fit to a
  pure linear WLS — no Gauss-Newton inner loop needed. The magnitude dependence is
  absorbed into a fixed polynomial shape chosen from physical/empirical considerations.
- **5-term vs 2-term basis**: the new basis has 5 spatial terms (instead of 2 separate
  2-term bases for x and y), providing more flexibility to capture the 2D position
  dependence of CTE. Crucially, both x and y CTE now use the same basis, which may
  over-constrain x-CTE (y-direction boundary condition imposed on x-CTE), but serial
  CTE is expected to be small.
- **Normalised coordinates**: dividing by 2048 keeps all 5 basis terms at similar scales
  (previously Y' ranged 0–2047 and Xc ranged −2048 to +2048, causing conditioning issues).

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
5. **Update CTE parameters** (γ_x, γ_y) from residuals via linear WLS.

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

### Linear update for γ (fully linear solve)

Given current residuals in GDC frame `(dx_k, dy_k)` for detection k in image j (chip c),
define the time-magnitude weight:

```
Ψ_k = (t_j − t_launch) · func1(mag_k)
```

The CTE design matrix row for each detection is:

```
A_k = Ψ_k · [yt_k, yt_k², xt_k·yt_k, xt_k²·yt_k, xt_k·yt_k²]    (5 columns)
```

Stack all detections for chip c across all images → solve two independent (N, 5) weighted
least squares systems:

```
γ_y_c = argmin Σ_k z_k · (−dy_k − A_k @ γ)²
γ_x_c = argmin Σ_k z_k · (−dx_k − A_k @ γ)²
```

Dynamic column normalisation (dividing each column by its standard deviation) keeps the
5-term basis well-conditioned.

There is no nonlinear δ update. With func1 fixed, the entire parameter estimation is a
single linear WLS solve per chip per direction — simpler and faster than the previous
Gauss-Newton inner loop.

---

## Warm-Start Strategy

Before the outer iteration loop, estimate initial γ from existing BP3M v2 PM residuals:

1. Load per-star PM residuals (pmdec_xmatch − field_mean_pmdec) from master_combined_v2.csv.
2. For each chip, select member stars within ±2 mas/yr of the field mean PM.
3. Build the CTE design matrix Ψ·b(xt, yt) with func1 evaluated at each star's mag.
4. Solve the (N, 5) WLS system (per direction) for initial γ_x, γ_y.
5. Cross-seed: if γ_y[0] > 0 for one chip (wrong sign), borrow from the other chip.

The warm start uses 1/σ²_pmdec weighting so faint stars with large CTE signal have
more influence on the initial estimate.

---

## Implementation Steps

### Step 1: CTEChipParams dataclass (proposed revision)
```python
@dataclass
class CTEChipParams:
    chip: str
    y_readout_raw: float      # raw readout row (lo ≈ 0, hi ≈ 2048)
    x0: float = 2048.0        # chip centre x for normalisation
    gamma_x: np.ndarray = field(default_factory=lambda: np.zeros(5))
    gamma_y: np.ndarray = field(default_factory=lambda: np.zeros(5))
```
Note: `delta` has been removed; all magnitude dependence is in the fixed func1.

### Step 2: Magnitude weighting function
- `func1_mag(mag, order=3, mag_ref=mag_ref)` → scalar per star  
  Evaluates a fixed polynomial at each star's mag. No free parameters.

### Step 3: Unified CTE basis function (same for x and y)
- `cte_basis(xt, yt)` → (n, 5): `[yt, yt², xt·yt, xt²·yt, xt·yt²]`

### Step 4: `compute_cte_displacement(xt, yt, mag, dt, chip_params)`
- `dt = t_j − t_launch` (years since ACS launch, not t_epoch0).
- Returns (n, 2) array of (δCTE_x, δCTE_y) in normalised chip coordinates.

### Step 5: `apply_cte_to_solver(solver, image_names, cte_params, t_launch_yr)`
- Stores `xys_orig` on first call.
- Updates `d['xys']` for all images using current CTE parameters.
- Uses `solver.R[img]` to rotate chip-frame correction to pseudo-image frame.

### Step 6: `warm_start_cte(img_to_df, solver, image_names, r_hat_init, t_launch_yr, field_mean_pm)`
- Estimates initial γ from PM residuals (see Warm-Start Strategy above).

### Step 7: `update_cte_params(residuals_by_chip, cte_params)`
- Linear WLS for γ_x, γ_y (5 coefficients each, solved separately).
- No δ update needed — fully linear solve.

### Step 8: `collect_cte_residuals(img_to_df, solver, image_names, r_hat, t_launch_yr, field_mean_pm)`
- Returns per-chip (dx, dy, xt, yt, mag, dt, z) arrays from the full master catalog.

### Step 9: `run_alignment_cte(...)` main function
- Outer loop: apply_cte → BP3M fit → collect_residuals → update_cte → repeat.
- Convergence: ‖Δγ‖/‖γ‖ < tol.

---

## Convergence Behavior and Diagnostics

Expected behavior:
- After 2-3 outer iterations, γ_y should converge to nonzero values for both chips.
- γ_y[0] (the [yt] coefficient) captures the dominant parallel CTE signal.
- CTE_x coefficients are expected to be small; if they converge to ~0 within noise, the
  serial CTE is negligible for this dataset.

Diagnostic outputs saved to `BP3M_cte_results/`:
- `cte_params.npz`: converged CTEChipParams for each chip (γ_x, γ_y per chip).
- `detections_catalog_cte.npz`: post-CTE residuals (same format as detections_catalog.npz).
- `cte_convergence.csv`: per-outer-iteration γ_c, RMS residuals.
- `cte_diagnostic.png`: 4-panel plot of CTE correction amplitude vs yt and magnitude.

---

## Tests with Leo I (2 Epochs: 2006, 2011)

Leo I is the current development target. With t_launch = 2002, the two epochs have
dt ≈ 4.0 yr and ≈ 9.0 yr respectively, giving a 4/9 amplitude ratio. Key tests:

1. **Do both chips agree on γ_y[0]?** Physical expectation: yes (same silicon, same
   radiation environment). A large discrepancy suggests a modeling error or chip-specific
   PSF effects confounded with CTE.

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

For fields with N > 2 HST epochs (e.g., Draco_dSph), the linear temporal model
h(t) = t − t_launch is well-motivated (CTE accumulates linearly with radiation dose).
With more epochs, the absolute CTE level is better constrained.

**Other future improvements**:
- Fit CTE separately per sub-image (jitter in readout efficiency).
- Cross-validate γ_hi vs γ_lo — if they agree across multiple fields, share γ between
  chips and reduce to 10 total CTE parameters.
- Use the magnitude-dependence of CTE to improve stellar mass estimates.
- Apply the learned CTE model as a correction to the FLC positions before running the
  standard pipeline (feedback loop into py1pass).
- Test whether CTE_x (serial register) is detectable with deep, high-stellar-density
  fields (e.g., ω Cen, 47 Tuc).
- Allow func1 to have free polynomial coefficients in a future extension (would require
  an alternating or joint nonlinear solve coupling the magnitude and spatial parameters).

---

## References

- Anderson & Bedin (2010): pixel-level ACS/WFC CTE correction (FLC pipeline)
- Massey (2010): power-law flux model for CTE, Equation 1 motivates the flux dependence
- ACS Instrument Science Report ACS 2012-03: time-dependent CTE model for ACS/WFC
- McKinnon et al. (2024): BP3M algorithm (bp3m paper; see memory/reference_key_papers.md)
