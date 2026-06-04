# V2 Alignment Debugging Notes

## Status: RESOLVED for E3 and Pal5

All known root causes identified and fixed. Both E3 and Pal5 give correct PMs.

**Test commands**:
```bash
# E3 (original test field)
python /Users/kevinm/Documents/Claude/Projects/gaiahub_improved/run_iterate_v2.py \
  --data_root ~/Documents/UCSC/HST_Gaia_PMs/GaiaHub_results \
  --field E3 \
  --n_refine 1 \
  --n_iter 20 --clip_sigma 4.5 --hst_enable_iter 5

# Pal5 (current test field — use this to validate HST-only fixes)
python /Users/kevinm/Documents/Claude/Projects/gaiahub_improved/run_iterate_v2.py \
  --data_root ~/Documents/UCSC/HST_Gaia_PMs/GaiaHub_results \
  --field Pal5 \
  --n_refine 1 \
  --n_iter 20 --clip_sigma 4.5 --hst_enable_iter 5
```

**Current results (n_iter=20)**:
- E3: v2 pmra offset = **+0.003 mas/yr** vs Gaia ✓
- Pal5: v2 pmra offset = **+0.000 mas/yr**, pmdec = **+0.030 mas/yr** vs Gaia ✓

---

## Root Causes Found and Fixed (all fields)

### Bug 1: Gaia CSV loaded without int64 dtype (`data_loader_master.py`)

**File**: `ghi/data_loader_master.py`

Gaia source IDs are 19-digit integers. Without `dtype={'source_id': np.int64}`, pandas reads them
as float64, causing precision loss (e.g., `5203318003970743680` → `5203318003970743296`). Multiple
distinct Gaia IDs can round to the same float64 value — the "PM=0 bug" from 57 collisions in E3.

**Fix**: Added `dtype={'source_id': np.int64, 'SOURCE_ID': np.int64}` to the `pd.read_csv` call
in `load_master_v2()`.

---

### Bug 2: `iterrows()` corrupts int64 Gaia IDs in pandas 3.x (`run_alignment_v2.py`)

**File**: `ghi/run_alignment_v2.py` (Phase 0 validation section)

In pandas 3.x, `iterrows()` converts int64 Series elements to Python scalars **via float64**,
silently corrupting 19-digit Gaia IDs even when the column dtype is `int64`. This caused the
Phase 0 v1 BP3M vs Gaia prior diagnostic to match only 39 stars instead of the expected ~300+.

**Fix**: Replaced all `iterrows()` usage for Gaia ID lookup with vectorised `.values` array
access (`_v1_ids = df['Gaia_id'].values; int(_v1_ids[k])`).

---

### Bug 3: Phase 0 flagging not updating `use_for_astrom` (`run_alignment_v2.py`)

Phase 0 pixel residual screening set `use_for_fit=False` for bad detections but did not update
`use_for_astrom`. In two-tier mode, outlier detections still in `use_for_astrom` inflated `H_vv`,
broke the Schur complement, and caused ~0.5 px epoch-dependent transformation drift.

**Fix**: In Phase 0 flagging loops, also set `use_for_astrom[bad] = False`.

---

### Bug 4: `_update_use_for_fit` used a 3× lenient astrometry threshold (`solver.py`)

In two-tier mode, `use_for_astrom` was set to a 3× more lenient residual threshold than
`use_for_fit`, recreating the Schur complement corruption after every Phase 2 outer iteration.

**Fix**: Set `use_for_astrom = use_for_fit` for initially-aligned stars in `_update_use_for_fit`.
HST-only stars (managed by `V2AlignmentCallback`) are left untouched.

```python
align_init = np.asarray(self._img_data[img]["use_for_align_init"], dtype=bool)
new_use_astrom[align_init] = new_use[align_init]  # keep astrom = fit for Gaia stars
```

---

### Bug 5: `xys` not updated after injecting v1 transformation (`run_alignment_v2.py`)

`solver.__init__` computed `xys` at the fast_cross_match transformation. Phase 0 injected the v1
transformation via `_update_R()` without rebuilding `xys`, causing Phase 1's `_solve_one_pass`
to see residuals that included the v1−fcm difference (~0.5 px epoch-dependent), which it then
re-absorbed as a transformation shift.

**Fix**: After `solver._update_R(r_init_hat)`, call `solver._update_geometry(r_init_hat,
solver.v_survey)` to rebuild `xys`, `JU`, and `X_mat` at the v1 transformation.

---

### Bug 6: HST-only stars flooding alignment tier via test-3 re-admission (`solver.py`)

**File**: `bp3m/solver.py`, `_update_use_for_fit()`; `ghi/run_alignment_v2.py`, `V2AlignmentCallback`

After HST-only stars were added to the **astrometry tier** (`use_for_astrom=True`, `use_for_fit=False`)
by the callback and got good PM estimates, their `sigma_resid` became small. Test-3 then automatically
admitted them to the **alignment tier** (`use_for_fit=True`) because `use_for_fit_max=True` for
those with good initial residuals. At outer iter 7 (first iteration after enabling), 6792 HST-only
detections flooded the alignment tier simultaneously. This corrupted the transformation, which then
caused Gaia-matched stars to fail the chi2 test (test-1/2) in subsequent iterations.

**Fix (two parts)**:

1. **`V2AlignmentCallback`** — changed to set `use_for_fit=True` **AND** `use_for_astrom=True`
   for qualifying HST-only sources at the transition (instead of astrometry-only). This lets
   them contribute to the transformation fit as intended, while v_survey is seeded simultaneously
   to prevent PM=0 bias.

2. **`_update_use_for_fit`** — added re-admission guard:
   ```python
   current_fit = np.asarray(self._img_data[img]["use_for_fit"], dtype=bool)
   can_enter_fit = align_init | current_fit   # only if initially aligned OR currently in
   new_use = new_use & can_enter_fit
   ```
   Stars not initially in alignment can only enter `use_for_fit` if they are **currently** in it
   (i.e., explicitly admitted by the callback). Once removed by test-1/2 or test-3, they cannot
   re-enter automatically.

---

### Bug 7: Alpha update locked out for small n_iter (`solver.py`, `run_alignment_v2.py`)

The alpha inflation update condition `iteration >= 3` meant alpha never updated with n_iter ≤ 3.
For images where v1 alpha > 1.0, the decrease toward 1.0 was blocked.

**Fix**: Added `inflate_from_iter` parameter to `solver.fit()` (default 3). `run_alignment_v2.py`
passes `inflate_from_iter=0` since v1 alpha is pre-validated and can be updated immediately.
Also: alpha is now displayed as the **actual applied value** (not alpha_raw) in early iterations.

---

### Bug 8: No minimum convergence after HST-only introduction (`run_alignment_v2.py`)

Early stopping could fire before the EM had time to converge after HST-only stars were introduced.

**Fix**: `run_alignment_v2.py` computes `min_outer_iters = max(hst_enable_iter + 3, 4)` and
passes it to `solver.fit()`. This ensures at least `hst_enable_iter + 3` outer iterations run
when HST-only are enabled during the run.

---

## Bug 9 (resolved): Pal5 PM pulled toward origin after HST-only enabled

**Fields affected**: Any field with significant foreground/background contamination in HST-only
sources (stream fields like Pal5, sparse clusters, pure field-star pointings).

**Symptom**: After `hst_enable_iter`, Gaia-matched stars were thrown out of alignment
(test-3 cascade: 6792–9420 changes at the transition), the 5p/6p chi2 distribution inflated
(p50 from 2.3 → 5–7), adaptive thresholds grew from 15 → 35, and final PMs were pulled
toward origin.

**Root cause**: `V2AlignmentCallback` was setting `use_for_fit=True` (alignment tier) for all
qualifying HST-only sources. In sparse fields like Pal5, HST-only include many foreground/
background stars with diverse PMs. Adding 2803 HST-only to alignment vs 140 Gaia stars (20:1
ratio) overwhelmed the Gaia constraint. Once in alignment, their diverse PMs biased the
transformation, Gaia stars' MAP PMs drifted from their priors, the adaptive Gaia chi2 threshold
inflated to accommodate this drift, and eventually Gaia stars were rejected in bulk.

**Fix**: `V2AlignmentCallback` now sets `use_for_astrom=True` only (NOT `use_for_fit`). HST-only
sources get stellar PM estimates from the Gaia-constrained transformation but do not influence
the transformation itself. This design is field-agnostic:
- **Dense GC (E3)**: Gaia alone constrains the transformation; HST-only get cluster PM estimates.
- **Stream (Pal5)**: Same — Gaia constrains, HST-only get individual field + cluster PM estimates.
- **Pure field**: Works identically — HST-only get estimates without biasing the alignment.

**Result**:
- E3: Gaia pmra offset = +0.003 mas/yr ✓
- Pal5: Gaia pmra offset = +0.000 mas/yr ✓ (perfect), pmdec +0.030 mas/yr ✓

**Why use_for_fit=True is fundamentally unstable for HST-only (investigated June 2026)**:

Testing confirmed that changing `sigma_pm_diffuse` from 100→1000 mas/yr makes **no difference**
— the diffuse prior is already dominated by detection data for any well-measured star.

The root cause is in the **Schur complement structure**. Per-image cancellation requires
`x_resid + JU @ a_align ≈ 0` for each detection. For Gaia-matched stars: the tight Gaia PM
prior forces `a_align ≈ v_gaia_pm` regardless of per-image noise, guaranteeing cancellation.
For HST-only: `a_align` is purely data-driven (averaged over all images), so individual-image
contributions are non-zero (signal from that image minus global average). With 2800 HST-only
vs 100 Gaia stars (20:1 in Pal5), these small per-star, per-image biases accumulate across
thousands of detections over many EM iterations, eventually destabilizing the solution.

The cascade: 1) HST-only create small transformation drift; 2) some Gaia detections become
high-leverage (Cook's D) relative to the shifted transformation; 3) test-4 removes them;
4) Gaia stars begin failing test-1/2; 5) runaway exclusion.

**Future work**: A two-stage approach — stage 1 astrometry-only for accurate HST-only PM
estimates, stage 2 uses those as tight priors to enable use_for_fit safely.

---

## Bug 10 (resolved): Master catalogue missing V1 Gaia-matched stars (Pal5)

**Symptom**: V2 stellar_astrometry.csv had only 104 Gaia-matched stars vs V1's 119 (−15). Five
missing stars were confirmed Pal5 cluster members; PM precision was slightly degraded.

**Root cause (two parts)**:
1. Phase 1 `min_detections=2` drops stars with only 1 detection per filter — V2's multi-epoch
   approach loses stars that V1's per-image matching included.
2. Even stars with F814W=2 detections could be dropped: Phase 1 magnitude consistency check
   fails for faint stars (large scatter between epochs → exceeds mag_n_sigma).

**Fixes in `hst_catalog_crossmatch.py`**:
1. **Phase 0b** (new step between Phase 0 and Phase 1): For each V1 BP3M Gaia star not yet
   found in det_df, predicts its sky position in each sub-image using V1 PM + transformation,
   searches 50-px radius / k=5 candidates, applies per-image colour/ZP filter, marks the
   best candidate as a Gaia match. Uses `anchor_bp3m_dir` (always set to V1 BP3M path in
   `run_iterate_v2.py`).
2. **Phase 1 keeps Gaia-anchored sources**: In `_within_filter_match`, sources with
   `has_gaia_match=True` are kept even with n=1 detection per filter.

**Result for Pal5**: master_combined_v2.csv: 161 Gaia-matched (was 104). stellar_astrometry:
117 matched to V1 (was 104). Only 2 non-member stars (n_hst_v1 ≤ 1) remain missing.

**Pipeline restructuring** (`hst_catalog_crossmatch.py`): The crossmatch now follows the new
phase order to maximise Gaia star completeness before the within-filter min_detections cut:
- **Phase 0**: Load all HST detections → RA/Dec (unchanged)
- **Phase 1** (was Phase 0b): V1 BP3M Gaia star anchoring — all V1 stars in all images
- **Phase 2** (new): Gaia catalog anchoring for non-V1 stars — catches new Gaia stars
- **Phase 3** (was Phase 1): Within-filter crossmatch — single-image detections now survive
- **Phase 4** (was Phase 2): Cross-filter matching
- **Phase 5** (was Phase 3): Gaia recovery post-crossmatch
- **Phase 6** (was Phase 4): Proper astrometry
- **Phase 7** (was Phase 5): PM-guided second-pass re-detection

---

## Additional improvements made during debugging

### Phase 0 chi2 outlier flagging
- Phase 0 now computes a full **df=5 chi2** for 5p/6p Gaia stars against the Gaia prior
- For 2p stars: df=2 chi2 against the diffuse PM prior (100 mas/yr)
- Stars with chi2 > 20.5 (5p/6p) or chi2 > 13.8 (2p) are permanently removed before Phase 1

### Phase 0 MAP vs catalogue diagnostic
- Condensed to compare only 5p/6p stars; shows median PM offset vs `pmra_xmatch`

### v1 BP3M posterior vs Gaia prior diagnostic
- Full df=5 chi2 including position offsets (`delta_racosdec_bp3m`)

### n_iter=0 HST-only astrometry fix
- When `n_iter=0`, `V2AlignmentCallback` transition never fires → PM = 0 for HST-only
- **Fix**: Manually trigger callback before `solver.fit()` when `n_iter=0`

### Phase 1 skip for n_iter=0
- `solver.fit()` now skips `_inner_converge` when `n_iter=0` (single `_solve_one_pass` only)

---

---

## Bug 11 (resolved): Test-4 fires on fully-converged EM, destabilizing sparse fields

**Fields affected**: Pal5 (116 Gaia alignment stars); sparse fields generally.

**Symptom**: EM stable for 6+ consecutive iterations, then test-4 (Cook's D) fires 2 removals at
iter 9, causing chi2 cascade: 5p/6p chi2 inflates from [0.5, 2.0, 6.0] → [0.8, 2.6, 10.6], then
8 test-1/2 changes + 63 test-3 changes, then 9 more test-4 removals. Never stabilizes.

**Root cause**: V2 starts from V1's tight transformation, so EM converges in 2-3 iterations
(vs V1's 10-15). Test-4 fires at `it_outer >= min_outer=4`, long after the EM is stable.
For sparse fields (116 Gaia stars), even 2 Cook's D removals perturb the transformation
enough to cascade. Scaling `influence_d_thresh` by the V1/V2 C_r ratio (σ_w_V2/σ_w_V1 = 1.14)
is insufficient — the timing is the primary issue.

**Fix**: Track `_n_consec_stable` (consecutive outer iterations with zero tests-1/2/3 changes).
Suppress test-4 when `_n_consec_stable >= 2`. This prevents Cook's D from destabilizing a
solution that has already converged. For V1 (10-15 iterations to converge), test-4 still
fires freely during convergence. For V2 (2-3 iterations), test-4 is suppressed from iter 5+.

**Also fixed**: Phase 0 astrometry validation NameError: `_diffuse_pm_inv` → `_diffuse_pm_inv_gaia`.

**Results after fix**:
- Pal5: pmra offset = -0.006 mas/yr, pmdec = -0.011 mas/yr (N=167) ✓
- E3:   pmra offset = 0.000 mas/yr, pmdec = 0.000 mas/yr (N=412) ✓

---

---

## Soft-weight IRLS implementation (`--soft_weights`)

**Purpose**: Experimental alternative to the hard EM (tests 1-4). Instead of binary
include/exclude decisions, each detection gets a continuous weight z ∈ (0, 1] based on
its chi² residual under the Student-t model. Enabled via `--soft_weights` in
`run_iterate_v2.py`; hard EM remains the default.

**Model**: z_k ~ Gamma(ν/2, ν/2), so the marginal likelihood is Student-t with ν dof.
EM weight update: z_k = min(1, (ν+2)/(ν+χ²_k)) where χ²_k = res_k^T Cs_k^{-1} res_k.
Larger ν → weights closer to 1 (nearly Gaussian); smaller ν → heavier tail (more aggressive
downweighting of outliers). Default ν=50.

**Sign convention** (critical): The BP3M model is `xys = X r - JU a + noise`, so the
full residual for χ² is `res = xys - X r + JU a` (positive JU term). This matches
`sample_posteriors` line: `resid = xys - X_mat @ r_hat_j + JU @ v_hat_i`. Getting this
wrong (using - JU a) inflates χ² ~4× for well-fitted detections → z ≈ 0 for all
detections → HST-only PMs collapse to prior mean (0 mas/yr). Gaia-matched survive via
their tight 5p/6p prior but HST-only do not.

**Two-tier structure in soft-weight mode**:
- Transformation (H_rr): `use_for_fit` only — same post-Phase-0 Gaia population as hard EM
- Stellar astrometry (H_vv): `use_for_fit | use_for_astrom` — Gaia + callback-enabled HST-only
- Phase-0-rejected Gaia detections: z=0 in both tiers (excluded, same as hard EM)

**Critical bug (population mask)**: Early implementation used `use_for_align_init` for
the transformation tier and `use_for_fit_max` for the χ² mask. Both are set BEFORE Phase 0.
Phase 0 removes some Gaia detections from `use_for_fit`, but they remain in
`use_for_align_init` and `use_for_fit_max`. With ν=50, these Phase-0-rejected detections
got z≈0.8-0.9 (χ²≈5-10) and were included in H_rr, shifting the transformation ~0.5 px
and causing PM errors of 1.4/-3.5 mas/yr for E3. Fix: use `use_for_fit` for the alignment
tier and `use_for_fit | use_for_astrom` for the χ² mask — exactly mirroring the hard EM.

**PM seeding for HST-only**: The IRLS seeds PM estimates before the first weight
computation by calling `per_iter_callback(solver, hst_enable_iter)`. This triggers the
V2AlignmentCallback transition, which sets `v_survey` for HST-only from xmatch PMs and
re-computes `C_survey_inv_dot_v`. Without seeding, HST-only PMs start at 0 mas/yr →
large χ² → z ≈ 0 → PMs never improve (chicken-and-egg).

**ν selection**: Default ν=50 is recommended. With correct population masking, all ν
values converge correctly (E3 ν=5: converges in 8 iters, pmra=0.0000). Smaller ν gives
more aggressive downweighting and is appropriate when true outliers have χ² >> ν.
For clean data (E3 cluster), χ² ~ χ²(2) with mean 2, so ν=5 downweights ~37% of
legitimate detections — valid but more conservative than ν=50.

**Results (ν=50 default, both fields, with Phase-6 warm start)**:
- E3:   N_eff=10238/10359 (98.8%), converges in 9 iters, pmra=0.0000, pmdec=0.0000 ✓
- Pal5: N_eff=15436/15677 (98.5%), converges in 9 iters, pmra=-0.009, pmdec=0.000 ✓

**Diagnostic outputs**: `soft_weights.csv` (per-detection z values) and
`soft_weights_diagnostic.png` (weight histogram by star type + per-image N_eff bar chart)
saved to `BP3M_v2_results/` when `--soft_weights` is active.

---

## Phase-6 chi² warm start for soft-weight IRLS

**Purpose**: Pre-populate initial z weights from Phase-6 per-detection chi² values
(computed during `hst_catalog_crossmatch.py` and stored in `master_combined_v2.csv`
as `det_chi2`), rather than computing z from the seed-solve residuals.

**Coordinate system**: Phase-6 already works in the same pseudo-pixel space as BP3M.
The Phase-6 `Big_C` includes `X C_r X^T` (transformation uncertainty) plus
`J α² C_hst J^T` (HST noise), while BP3M `Cs = J C_hst J^T` only. This makes
Phase-6 chi² slightly smaller than BP3M chi² for the same detections (larger
denominator), but they are directly comparable.

**Implementation**:
- `data_loader_master.py`: stores `det_chi2_by_img = {sname: chi2}` per source record;
  threads `det_chi2_val` float through `img_records` into a `det_chi2` column in the
  per-image DataFrames passed to the solver.
- `run_alignment_v2.py`: after `solver.setup_images()`, builds `_z_init` dict by
  computing `z = min(1, (ν+2)/(ν+chi2))` from the `det_chi2` column. Uses
  `use_for_fit_max` as the mask (includes all Phase-0-surviving detections). Sets
  z=1.0 for detections without Phase-6 chi² (HST-only, catalogue gaps).
- `solver.fit()`: new `z_init` parameter. When provided, uses it as the initial
  z_weights instead of calling `_update_soft_weights` on the seed-solve residuals.

**Key bug fixed — timing of mask**: Original implementation used `use_for_fit |
use_for_astrom` as the z_init mask, but `use_for_astrom` is False for HST-only
*before* the callback fires inside `solver.fit()`. This zeroed out all ~6500 HST-only
detections in z_init, while `_update_soft_weights` (post-callback) correctly gives
z≈0.98 for them. The spurious difference caused Δz≈6419 at iter 1. Fix: use
`use_for_fit_max` as the mask, which includes all Phase-0-surviving detections
(Gaia and HST-only alike) regardless of callback state.

**Residual Δz at iter 1**: ~213 (E3) and ~157 (Pal5) after the fix. Two sources:
1. ~70 Phase-0-rejected detections are in `use_for_fit_max` (so z_init > 0) but
   `_update_soft_weights` sets them to z=0. Their contribution is small.
2. Phase-6 WLS chi² (per-source independent fit) differs slightly from BP3M
   joint-solve chi² (all stars simultaneously), because the WLS stellar astrometry
   `u` differs from the BP3M MAP `a_arr`.

**Future improvement**: Store the 5-component Phase-6 WLS astrometry estimate
(`u = Δα*, Δδ, pmra, pmdec, plx`) per source in `master_combined_v2.csv` and use
it to initialise `v_survey` before `solver.fit()`. This would allow skipping the
seed `_inner_converge` entirely, saving one full linear solve per run.

---

## Files modified

| File | Change |
|---|---|
| `ghi/data_loader_master.py` | Read Gaia source_id as int64; `det_chi2_by_img` per source; `det_chi2` float column in per-image DataFrames |
| `ghi/run_alignment_v2.py` | iterrows() fix; Phase 0 chi2; use_for_astrom sync; _update_geometry; n_iter=0 callback; inflate_from_iter=0; min_outer_iters; C_r ratio scaling; `_diffuse_pm_inv_gaia` fix; soft-weight params; `_plot_soft_weights()`; Phase-6 z_init builder with `use_for_fit_max` mask |
| `bp3m/solver.py` | n_iter=0 Phase 1 skip; use_for_astrom=use_for_fit for Gaia stars; can_enter_fit guard; inflate_from_iter/min_outer_iters params; `_n_consec_stable` test-4 timing guard; soft-weight IRLS branch; `_update_soft_weights()`; 7-value return from `fit()`; `z_init` parameter |
| `run_iterate_v2.py` | `--soft_weights`, `--student_t_nu`, `--phase4_outlier_sigma`, `--det_chi2_threshold` CLI args |
| `ghi/hst_catalog_crossmatch.py` | `phase4_outlier_sigma` param; `det_chi2` column |
| `bp3m/plot_results.py` | HST-only removed from upper 1:1 panels |
