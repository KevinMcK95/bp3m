# Joint CTE + Astrometry Model for HST ACS/WFC

## Overview

This document describes the design and implementation plan for jointly fitting HST image
transformations (r_j), stellar astrometry (v_i), a parametric CTE (Charge Transfer
Efficiency) model, and a population-level PM prior within the BP3M framework. The code
lives in `bp3m/pipeline/run_alignment_cte.py`.

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

- `func1(mag_raw)` is a fixed polynomial in the raw instrumental magnitude. This encodes
  the physical flux dependence of CTE: fainter stars (higher mag, fewer electrons)
  experience stronger parallel CTE trailing. The polynomial is evaluated per star and
  produces a single scalar — there are no free parameters in func1 itself. All amplitude
  and spatial information is absorbed into the free γ coefficients.

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

## Population-Level Priors and Membership Selection

### Membership Selection

Only likely member stars contribute to constraining the CTE parameters and the population
mean PM. A star is a likely member if its PM is consistent with the field mean to within
a few times the intrinsic PM dispersion σ_pm (see below). This applies equally to all
stellar types:

- **Gaia 5p/6p stars**: membership assessed from their Gaia-measured PMs. Members receive
  a population prior on their PMs alongside their Gaia prior; non-members are excluded
  from the CTE and μ_pop constraints. (Their detections still receive the CTE position
  correction, but their residuals do not inform γ or μ_pop.)
- **HST-only / Gaia 2p stars**: membership assessed from their BP3M-fitted PMs, which
  evolve each outer iteration. On the first iteration (warm start), a ±2 mas/yr window
  around the empirical field mean is used as a preliminary membership selection.

Non-member stars still get the CTE correction applied to their observed positions (they
experience CTE like any other star on the detector), but they are excluded from the linear
system that determines γ and μ_pop.

### Population Mean PM (Free Parameter)

The population mean PM μ_pop = (μ_α*, μ_δ*) is estimated jointly with all other shared
parameters. Its prior is:

```
μ_pop  ~  N(μ_empirical, C_pop_prior)
C_pop_prior  =  (0.5 mas/yr)² × I₂
```

where μ_empirical is the field mean PM from the warm-start empirical estimate (derived
from Gaia-matched member stars via iterative sigma-clipping of their pmra_xmatch /
pmdec_xmatch values in master_combined_v2.csv). The 0.5 mas/yr prior width is wide
enough to be nearly uninformative relative to the constraint from hundreds to thousands
of member stars, but provides regularisation on the first outer iteration before the
star PMs have converged.

### Intrinsic PM Dispersion from LVD

The Local Volume Database (LVD; Pace et al. 2024) provides for each system:
- Line-of-sight velocity dispersion: σ_LOS (km/s)
- Distance: d (kpc)

Under the assumption of spherical symmetry, the 1D proper motion dispersion equals the
1D LOS velocity dispersion in physical units:

```
σ_pm  =  σ_LOS / (4.74047 × d)    [mas/yr]
```

(4.74047 converts 1 kpc·mas/yr to km/s). This σ_pm sets the population prior width on
individual member star PMs. For each member star i, the prior on the PM components of
v_i is:

```
(μ_α_i, μ_δ_i) | μ_pop  ~  N(μ_pop, σ_pm² × I₂)
```

For Leo I: σ_LOS ≈ 9.2 km/s, d ≈ 254 kpc → σ_pm ≈ 0.008 mas/yr. Member PMs are
essentially at a single point in PM space; the population prior is very tight. For more
nearby or dynamically hotter systems (Draco: σ_LOS ≈ 9.1 km/s, d ≈ 76 kpc →
σ_pm ≈ 0.025 mas/yr), σ_pm is still small compared to HST measurement uncertainties
for faint stars, so the constraint is significant.

### Parallax Prior from LVD (Fixed)

LVD provides:
- Mean distance: d ± σ_d (kpc)
- Physical depth estimate: δ_d (kpc, e.g., from half-light radius or reported uncertainty)

These map to a parallax prior applied to every member star:

```
plx_pop       =  1 / d                          [mas]
σ_plx_dist    =  σ_d / d²                       [mas]   (from distance uncertainty)
σ_plx_depth   =  δ_d / (√3 × d²)               [mas]   (uniform depth → 1σ half-width)
σ_plx_tot²    =  σ_plx_dist² + σ_plx_depth²

plx_i  ~  N(plx_pop, σ_plx_tot²)
```

Unlike μ_pop, the parallax population mean plx_pop is treated as a **fixed input** from
LVD, not a free parameter. Individual star parallax deviations from plx_pop reflect either
true line-of-sight depth differences (negligible for dSph members at HST precision) or
HST systematic parallax effects. The tight prior substantially reduces the PM–parallax
degeneracy for stars with non-uniform epoch parallax factors, and is especially important
for HST-only stars whose parallax was previously unconstrained (σ ~ 100 mas/yr diffuse
prior in v2).

---

## Current GitHub Model vs Proposed Model

| Aspect                    | Current (on GitHub)                                 | Proposed                                         |
|---------------------------|-----------------------------------------------------|--------------------------------------------------|
| **Time reference**        | `t − t_epoch0` (first exposure MJD)                 | `t − t_launch` (ACS launch 2002-03-01)          |
| **Magnitude function**    | `φ(mag; δ) = 10^{0.4δ(mag−mag_ref)} − 1` (free δ)  | `func1(mag)` = fixed polynomial in mag_raw      |
| **y-CTE basis**           | `[Y', Xc·Y']` (2 terms, un-normalised)              | `[yt, yt², xt·yt, xt²·yt, xt·yt²]` (5 terms)   |
| **x-CTE basis**           | `[Xc, Xc·Y']` (2 terms, un-normalised)              | same 5-term basis as y                           |
| **Coordinates**           | `Y' = y_raw − y_readout_raw`, `Xc = x_gdc − 2048`  | `yt = (y_raw − y_readout)/2048`, `xt = X_c/2048` |
| **Free params per chip**  | δ(1) + γ_x(2) + γ_y(2) = **5**                     | γ_x(5) + γ_y(5) = **10**                        |
| **Total CTE params**      | 5 × 2 chips = **10**                                | 10 × 2 chips = **20**                            |
| **δ update**              | Gauss-Newton inner loop (nonlinear)                 | None (func1 is fixed — fully linear solve)       |
| **Population mean PM**    | Fixed empirical estimate, not a model parameter     | Free parameter μ_pop jointly fit with γ and θ_dist |
| **Pop PM prior**          | —                                                   | N(μ_empirical, (0.5 mas/yr)² I₂)                |
| **Intrinsic PM dispersion** | Not modelled                                      | σ_pm = σ_LOS / (4.74 × d) from LVD              |
| **Parallax prior**        | Diffuse (HST-only: σ ≈ 100 mas/yr)                  | N(plx_LVD, σ_plx_tot²) from LVD distance+depth  |
| **Member selection**      | PM quality cut + \|pm\| < 3 mas/yr (absolute)       | Population prior; member window ±N·σ_pm          |
| **Inference approach**    | Alternating BP3M + CTE updates (EM-like)            | Single joint marginalisation over {v_i} per iteration |
| **Shared param count**    | N_images × N_dist                                   | N_images × N_dist + 20 (CTE) + 2 (μ_pop)        |

Key changes beyond the CTE model itself:
- **Joint marginalisation** eliminates the EM-like alternation between BP3M, CTE, and
  population mean updates (see Joint Marginalisation Framework section).
- **Population prior** tightens PM and parallax constraints for HST-only faint stars,
  breaking the degeneracy between per-star PM absorption and CTE trailing.
- **μ_pop as free parameter** allows the bulk field PM to respond to CTE corrections
  and propagates the CTE ↔ field-mean covariance into the posterior.

---

## Joint Marginalisation Framework

### Why Not Alternating Updates (EM)?

An appealing but suboptimal approach is to alternate:
1. Fix (γ, μ_pop) → BP3M solve for ({v_i}, θ_dist)
2. Fix ({v_i}, μ_pop) → WLS update for γ from residuals
3. Fix {v_i} → precision-weighted mean update for μ_pop
... and repeat.

This is expectation-maximisation (EM). For linear Gaussian models EM converges to the
correct MAP, but has two important drawbacks:

1. **Convergence rate**: EM converges at most linearly. The joint solve below converges
   in one pass per outer CTE iteration.
2. **Missing cross-covariances**: EM's separate steps do not propagate correlations
   between θ_dist, γ, and μ_pop. The posterior is overconfident, and physically important
   covariances are missed — in particular, misestimated CTE appears partly as a bulk PM
   shift, so γ and μ_pop are correlated; EM treats them as independent.

### The Joint Solve

Define the shared parameter vector:

```
θ_shared  =  (θ_dist_1, …, θ_dist_N,   γ_CTE,   μ_pop)
                    ↑                     ↑          ↑
              per-image distortion    20 params   2 params
              (N_images × N_dist)
```

For each **member** star i with J_i detections, reparametrise the stellar astrometry:

```
v_i  =  δv_i  +  M_i μ_pop  +  b_i
```

where:
- `δv_i ~ N(0, C_prior_i)` is the deviation from the population-mean prior.
  C_prior_i is diagonal with σ_pm² in the PM rows and σ_plx_tot² in the parallax row;
  for Gaia 5p/6p member stars, C_prior_i also includes the Gaia measurement covariance.
- `M_i` is the 5×2 matrix that places μ_pop into the (μ_α, μ_δ) rows of v_i.
- `b_i` is the fixed prior mean: (0, 0, 0, 0, plx_pop)ᵀ for HST-only stars; for Gaia
  stars it also shifts the position and PM components by their Gaia-measured values.

After marginalising δv_i analytically, the stacked detections of star i satisfy:

```
ỹ_i  ~  N(D_i θ_shared,  Ω_i)

ỹ_i  =  y_i − B_i b_i                        (data adjusted for fixed prior mean)
D_i  =  [A_i  |  F_i  |  B_i M_i]           (shared-parameter design matrix)
Ω_i  =  Σ_obs_i + B_i C_prior_i Bᵢᵀ         (obs covariance inflated by marginalisation)
```

Columns of D_i:
- **A_i**: distortion polynomial rows for the images where star i is detected — identical
  to the current K_img computation.
- **F_i**: CTE rows, one per detection: `dt_j · func1(mag_i) · b(xt_ij, yt_ij)` → 5 columns.
- **B_i M_i**: population-mean coupling. B_ij is the pointing matrix (maps v_i to the
  observable frame); M_i selects the PM components; the product maps μ_pop into the
  observable. This column block is nonzero for all member stars.

Each member star contributes to the joint precision and information vector:

```
Λ_shared  +=  D_iᵀ Ω_i⁻¹ D_i
r_shared  +=  D_iᵀ Ω_i⁻¹ ỹ_i
```

Adding the μ_pop prior and solving:

```
Λ_total   =  Λ_shared  +  diag(0, …, 0,  C_pop_prior⁻¹)
θ_shared  =  Λ_total⁻¹ r_shared

Posterior covariance:  C_shared  =  Λ_total⁻¹
```

This is the exact joint marginal posterior p(θ_dist, γ, μ_pop | data), computed in a
single pass over the member stars — no alternating inner loop.

### System Dimensions

| Block | Typical size |
|---|---|
| Per-image distortion (N_images × N_dist) | ~150 (15 images × 10 params) |
| CTE γ (2 chips × 2 directions × 5 terms) | 20 |
| Population mean PM | 2 |
| **Total** | **~172** |

A 172 × 172 linear system. The matrix has an arrowhead structure (per-image distortion
blocks are nearly independent; CTE and μ_pop rows couple all images), but at this scale
a direct dense solve is trivially fast. Inversion is O(172³) ≈ 5 million flops.

### The Remaining Outer Iteration

The outer CTE loop persists because the CTE design matrix F_i depends on raw pixel
positions (xt_ij, yt_ij), which are the result of the distortion correction θ_dist.
This mild coupling means the design matrix must be rebuilt each outer iteration around
the current solution. Within each outer iteration, (θ_dist, γ, μ_pop) are solved jointly
in one pass. Convergence is monitored via ‖Δγ‖/‖γ‖ and ‖Δμ_pop‖ / (0.5 mas/yr);
3–5 outer iterations typically suffice.

---

## Forward Model Integration with BP3M

### Design Principle: No Core BP3M Modifications

The CTE pipeline is intentionally standalone. **No files in the core BP3M package
(solver.py, run_alignment.py, etc.) are modified.** Where the joint model requires
logic beyond what the existing solver exposes (e.g., the joint K_img-style accumulation
over member stars), new helper functions are written in `run_alignment_cte.py` or
adjacent CTE-specific modules, reusing BP3M internals as black-box calls where possible.

### Solver Integration

The CTE correction is applied by modifying `solver._img_data[img]['xys']` before each
BP3M solve pass — the only interface to the core solver. The workflow:

1. **Before the outer loop**: store `d['xys_orig'] = d['xys'].copy()` for all images.
2. **Each CTE iteration**:
   - Compute `δCTE_raw` = (δCTE_x, δCTE_y) in raw chip-centered pixel frame.
   - Map to pseudo-image frame: `Δxys = R_j @ δCTE_raw` (2×2 rotation from r_j).
   - Set `d['xys'] = d['xys_orig'] + Δxys`.
3. **Run joint solve** over member stars for (θ_dist, γ, μ_pop) — single pass, see above.
   This step is implemented entirely in new helper code within run_alignment_cte.py;
   it does not call solver.solve() or modify any solver internals.
4. **Collect full-catalog residuals** from detections_catalog.npz (all ~127k stars, for
   CTE diagnostic purposes and to feed back into next iteration's design matrices).
5. **Update membership** based on current μ_pop and σ_pm.

### Soft Weights Compatibility

The CTE correction modifies `xys` before the IRLS computation. Since z_{ij} weights in
soft-weight IRLS are applied to `(xys - X_mat @ r_j - JU @ v_i)²`, and `xys` now
includes the CTE correction, the z weights automatically down-weight CTE-corrected
detections with large residuals. No special handling is needed.

---

## Warm-Start Strategy

Before the outer iteration loop, estimate initial (γ, μ_pop) from existing BP3M v2 PM
residuals:

1. Compute the empirical field mean PM μ_empirical from Gaia-matched member stars in
   master_combined_v2.csv via iterative 2D sigma-clipping. This initialises μ_pop.
2. For each chip, select likely member stars within ±2 mas/yr of μ_empirical.
3. Build the CTE design matrix Ψ·b(xt, yt) with func1 evaluated at each star's mag.
4. Solve the (N, 5) WLS system (per direction) for initial γ_x, γ_y.
5. Cross-seed: if γ_y[0] > 0 for one chip (wrong sign), borrow from the other chip.

The warm start uses 1/σ²_pmdec weighting so faint stars with large CTE signal have
more influence on the initial estimate. The initial μ_pop is then refined jointly with
γ in the first outer iteration.

---

## Implementation Steps

### Step 1: CTEChipParams dataclass
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
- `func1_mag(mag, mag_ref=_MAG_REF)` → scalar per star.
  Fixed function, no free parameters. Current implementation: `10^{0.4*(mag − mag_ref)}`.

### Step 3: Unified CTE basis function (same for x and y)
- `cte_basis(xt, yt)` → (n, 5): `[yt, yt², xt·yt, xt²·yt, xt·yt²]`

### Step 4: `compute_cte_displacement(X_c, y_raw, mag, dt, chip_params)`
- `dt = t_j − t_launch` (years since ACS launch, not t_epoch0).
- `xt = X_c / 2048`, `yt = (y_raw − y_readout_raw) / 2048`.
- Returns (n, 2) array of (δCTE_x, δCTE_y) in pixels.

### Step 5: `apply_cte_to_solver(solver, image_names, cte_params, t_launch_yr)`
- Stores `xys_orig` on first call.
- Updates `d['xys']` for all images using current CTE parameters.
- Uses `solver.R[img]` to rotate chip-frame correction to pseudo-image frame.

### Step 6: `warm_start_cte(img_to_df, solver, image_names, r_hat_init, t_epoch0_yr, field_mean_pm)`
- Estimates initial γ from PM residuals and initialises μ_pop (see Warm-Start above).

### Step 7: Joint solve step (target implementation)
- New function `_joint_solve_cte(solver, image_names, member_mask, cte_params, mu_pop,
  sigma_pm, plx_pop, sigma_plx, C_pop_prior, t_launch_yr)` in run_alignment_cte.py.
- Iterates over member stars, builds D_i = [A_i | F_i | B_i M_i] and Ω_i for each,
  accumulates Λ_shared and r_shared, then solves Λ_total θ_shared = r_shared.
- No changes to solver.py; the function reads solver._img_data and solver internals
  (R, JU, X_mat, etc.) as read-only inputs.
- Current stepping stone: alternating BP3M `solver.solve()` + `update_cte_params` WLS.

### Step 8: `collect_cte_residuals(img_to_df, solver, image_names, r_hat, t_launch_yr, field_mean_pm)`
- Returns per-chip (dx, dy, X_c, y_raw, mag, dt, z) arrays from member stars only.

### Step 9: `run_alignment_cte(...)` main function
- Outer loop: apply_cte → joint solve (or alternating) → collect_residuals → repeat.
- Convergence: ‖Δγ‖/‖γ‖ < tol and ‖Δμ_pop‖ < 0.01 mas/yr.

---

## Convergence Behavior and Diagnostics

Expected behavior:
- After 2–3 outer iterations, γ_y should converge to nonzero values for both chips.
- γ_y[0] (the [yt] coefficient) captures the dominant parallel CTE signal.
- CTE_x coefficients are expected to be small; if they converge to ~0 within noise, the
  serial CTE is negligible for this dataset.
- μ_pop should shift from the empirical warm-start estimate by at most ~0.1–0.2 mas/yr
  once CTE is properly accounted for; a larger shift indicates significant CTE absorption
  in the v2 field mean.

Diagnostic outputs saved to `BP3M_cte_results/`:
- `cte_params.npz`: converged CTEChipParams for each chip (γ_x, γ_y per chip, μ_pop).
- `detections_catalog_cte.npz`: post-CTE residuals (same format as detections_catalog.npz).
- `cte_convergence.csv`: per-outer-iteration γ_c, μ_pop, RMS residuals.
- `cte_diagnostic.png`: diagnostic plots of CTE correction amplitude, PM residuals, and
  μ_pop convergence.

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
   exceeding ~0.1 mas/yr is significant. With the joint model, μ_pop is a model output
   rather than a fixed input, so this shift is measured directly.

4. **Comparison with pixel-level correction**: The FLC files apply a pixel-level CTE
   correction (Anderson & Bedin 2010). The residual CTE captured by our model is
   whatever the pixel correction missed. Its magnitude indicates how well the standard
   correction performs at 15+ years post-launch.

5. **Individual star PM uncertainties**: After CTE correction, faint stars should have
   reduced scatter in their per-image residuals, leading to smaller formal PM errors.

6. **μ_pop shift**: The difference between μ_pop (jointly estimated) and μ_empirical
   (pre-CTE field mean from Gaia xmatch) quantifies how much CTE leaked into the v2
   field mean estimate.

---

## Future Work

- **Implement full joint solve** (Step 7 above): currently the code uses alternating
  BP3M + CTE updates as a stepping stone. Replacing this with the single-pass joint
  marginalisation will improve convergence speed and give correct cross-covariances.
- **Draco dSph and multi-epoch fields**: with N > 2 epochs, the absolute CTE level is
  better constrained. The linear temporal model h(t) = t − t_launch is well-motivated
  (CTE accumulates linearly with radiation dose); more epochs directly test this.
- **Cross-validate γ_hi vs γ_lo**: if they agree across multiple fields, share γ between
  chips to reduce to 10 total CTE parameters.
- **Fit func1 jointly** (future extension): allowing the magnitude weighting to have free
  polynomial coefficients would require extending the joint design matrix D_i. With the
  marginalisation framework already in place, this is a natural extension.
- **Sub-image CTE variation**: fit CTE separately per sub-image (jitter in readout
  efficiency) by splitting the 20 CTE parameters into per-image groups.
- **Apply γ to FLC positions**: feed the learned CTE model back into py1pass as a
  correction to raw positions before running the standard pipeline.
- **Test serial CTE (γ_x)** in high-stellar-density fields (ω Cen, 47 Tuc) where serial
  CTE may be detectable.

---

## References

- Anderson & Bedin (2010): pixel-level ACS/WFC CTE correction (FLC pipeline)
- Massey (2010): power-law flux model for CTE, Equation 1 motivates the flux dependence
- ACS Instrument Science Report ACS 2012-03: time-dependent CTE model for ACS/WFC
- McKinnon et al. (2024): BP3M algorithm (bp3m paper; see memory/reference_key_papers.md)
- Pace et al. (2024): Local Volume Database (LVD) — σ_LOS, distances, and structural
  parameters for Local Group dwarf galaxies
