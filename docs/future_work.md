# Future Work and Planned Improvements

## HST Cross-match Catalog Refinement

### 1. PM-guided re-detection pass

After `run_hst_crossmatch()` produces `master_combined.csv` with measured PMs, a second pass over the raw per-image catalogs (`{img}_flc_catalog.fits`) could recover detections missed in the initial within-filter matching:

- Propagate each source's position to each image epoch using its measured PM (+ Gaia parallax where available)
- Search within ~2–3 px of the predicted position (much tighter than the initial blind match)
- Require candidate magnitude within ±0.5 mag of the master weighted-mean for that filter
- Apply a chi2 / residual quality cut on any recovered detection to reduce false positives in crowded fields

**Gain**: Sources with only 2–3 detections may gain additional matches, reducing σ_PM roughly as 1/√N. Sources near the magnitude limit or chip edges are the primary targets.

**Risk**: In dense fields (E3), a misestimated PM could lock onto a nearby blended source. A PM quality gate (σ_PM < ~5 mas/yr) should be applied before using a source's PM as a prior. One pass is likely sufficient; iterating risks over-fitting faint/crowded detections.

**Variant**: Use BP3M posteriors (if available) rather than xmatch PMs as the position prior — these are significantly better constrained and include the full covariance.

### 2. Duplicate identification and cleaning — IMPLEMENTED

Implemented as `_deduplicate_merged` in `hst_catalog_crossmatch.py`, called at two points in the pipeline:

- **After Phase 2** (cross-filter merge): uses Phase-1 `ra0`/`dec0` positions. Catches Gaia-ID duplicates (unambiguous) and most cross-filter close pairs.
- **Phase 4b — after Phase 4 astrometry**: re-runs `_deduplicate_merged` using accurate `ra_xmatch`/`dec_xmatch` positions. Catches cross-filter pairs that Phase 2 missed because Phase-1 inter-filter positional offsets were too large (e.g. 595 mas in E3).

Design: Pass 1 merges rows sharing a Gaia source ID (primary = most detections). Pass 2 merges close positional pairs within 50 mas that have disjoint or overlapping-but-not-identical filter sets. Same-single-filter pairs within 50 mas are left untouched — these are genuine blends or real close pairs in dense fields (e.g. Leo_I has ~6600 such pairs). After any merge, `n_detect`, `n_filters`, and `filter_list` are recomputed from the merged detection set. Sources that gained detections in Phase 4b are re-fitted with `_measure_astrometry_proper`.

---

## BP3M v2 — Re-alignment Using Phase 5 Cross-matches

After `run_hst_crossmatch()` Phase 5 produces `master_combined_v2.csv`, a significant fraction of sources will have gained detections from cross-filter or cross-epoch matches that were invisible to the initial Gaia-only BP3M run. Feeding these back into BP3M (as a "v2" run) tightens the inter-image transformation constraints and may improve astrometry for faint HST-only sources.

### Motivation

The current BP3M run uses only Gaia-matched stars for alignment. Phase 5 may add 10–100× more sources with multi-epoch positions. Even without Gaia priors, these sources act as "grid stars" constraining the transformation model. Dense chip-wide coverage exposes spatial structure in residual GDC corrections not visible with sparse Gaia samples.

## BP3M v2 — Re-alignment Using Phase 5 Cross-matches

After `run_hst_crossmatch()` Phase 5 produces `master_combined_v2.csv`, a significant fraction of sources will have gained detections from cross-filter or cross-epoch matches invisible to the original Gaia-only BP3M run. Feeding these back into BP3M (as a "v2" run) tightens the inter-image transformation constraints and improves astrometry for faint HST-only sources.

### `ghi/data_loader_master.py`

Reads `master_combined_v2.csv` and builds BP3M input arrays:

1. **Load and classify sources**:
   - Gaia 5p/6p: full Gaia prior (inflated covariance via `_build_gaia_cov5`), `ra`/`dec` from Gaia catalogue at J2016 (barycentric). No diffuse prior.
   - Gaia 2p: position-only Gaia covariance, Michalik parallax prior + 100 mas/yr PM prior.
   - HST-only: zero 5×5 Gaia inv_cov; `ra_xmatch`/`dec_xmatch` from the master catalogue as J2016 reference position (see below); Michalik parallax prior + 100 mas/yr PM prior.

2. **J2016 barycentric reference position for HST-only sources**: `ra_xmatch`/`dec_xmatch` in the master catalogue is the output of the 5-parameter fit in `_measure_astrometry_proper`, which models parallax explicitly — the solved (Δα*, Δδ) offset corresponds to the barycentric J2016.0 position (i.e., with the parallax contribution at each observation epoch removed). This is exactly equivalent to how Gaia reports positions: in the parallax-free solar system barycentric frame. No additional propagation is needed; `ra_xmatch`/`dec_xmatch` is used directly.

3. **Strip outlier detections**: For each source, parse the `outlier_images` column from the master catalogue and exclude `(sub_name, catalog_index)` pairs where `sub_name` is in the per-source outlier list. These were rejected by the astrometric fit as likely bad matches.

4. **Detection uniqueness check**: Verify no `(sub_name, catalog_index)` pair appears in more than one source row. Abort with a descriptive error if violated.

5. **Initial `use_for_fit` / `use_for_astrom` assignments**:
   - Gaia sources: `use_for_fit=True`, `use_for_astrom=True` (subject to normal quality checks).
   - HST-only: `use_for_fit=False` initially, `use_for_astrom=True`. Flipped by the callback in `run_alignment_v2.py` at iteration ≥ `hst_enable_iter`.

6. **Per-image top-N pre-selection of HST-only sources**: For each image, among HST-only sources with detections in that image, retain at most `hst_max_per_image` (default 1000) sources ranked by `sigma_pmra_xmatch`. Sources eliminated by this cap are never enabled (`use_for_fit` stays False throughout). This prevents any single image from being dominated by noisy HST-only grid stars.

7. **Global quality cuts for HST-only eligibility**:
   - `n_detect_fit >= 2` after outlier stripping
   - `sigma_pmra_xmatch < hst_max_pm_unc` (default 5 mas/yr — deliberately conservative to start)

### `ghi/run_alignment_v2.py` — Phased solve

**Solver callback hook**: A new optional `per_iter_callback(solver, iter_num)` argument is added to `BP3MSolver.run()`. The callback is invoked at the end of each outer EM iteration with full access to the solver instance (`r_hat`, `v_hat`, `use_for_fit` arrays, geometry methods). The callback in `run_alignment_v2.py` is a `V2AlignmentCallback` instance that manages the phased inclusion of HST-only sources.

**Phase structure:**

*Pre-inclusion phase (iterations 1 – `hst_enable_iter − 1`, default first 4 iterations):*
- HST-only sources: `use_for_fit=False`, `use_for_astrom=True`.
  - Their detection residuals are computed each iteration against the current `r_hat`.
  - Per-detection chi2 residuals above `outlier_sigma` (default 5σ) are **soft-flagged** — recorded but not yet permanently removed. Philosophy: avoid aggressive early pruning when the Gaia-only `r_hat` may still be rough.
- Gaia sources proceed normally through their own chi2 tests and inflation learning.
- Goal: allow Gaia sources to reject outliers, converge on a stable transformation posterior, and for BP3M to learn the per-image HST uncertainty inflation (alpha factors) before HST-only sources enter the fit.

*Reference position update (once, at the end of iteration `hst_enable_iter − 1`):*
- For each HST-only source, `v_hat[0:2]` (the BP3M-estimated correction to the J2016 barycentric position in mas) is added to the initial `ra_xmatch`/`dec_xmatch`:
  ```
  new_ra  = ra_xmatch  + v_hat[0] / (cos(dec) × MAS_PER_DEG)
  new_dec = dec_xmatch + v_hat[1] / MAS_PER_DEG
  ```
  This sharpens the parallax factor vectors (`f_α`, `f_δ`) and tangent-plane Jacobians at each image epoch. `solver._precompute_geometry(r_hat)` is called once after all updates.
  Whether this correction is large enough to matter will be checked empirically after the first v2 run; if offsets are consistently < 1 mas, the update can be dropped.

*Inclusion transition (iteration `hst_enable_iter`, default 5):*
- Soft-flagged detections from the pre-inclusion phase are permanently removed (any detection soft-flagged in ≥ 1 iteration is dropped; the source's `n_detect_fit` is updated).
- Sources still meeting quality cuts (`n_detect_fit ≥ 2`, `sigma_pm < 5 mas/yr`, within per-image top-1000) have `use_for_fit` flipped to True.
- No special uncertainty inflation is applied — HST position uncertainties are inflated per-image by exactly the same alpha factors BP3M has already learned from the Gaia sources. These apply identically to HST-only detections since they use the same pixel-level covariances.

*Post-inclusion phase (iterations `hst_enable_iter` onwards):*
- HST-only and Gaia sources are treated identically: same chi2 tests (Gaia prior test for Gaia sources, diffuse prior test for HST-only), same per-image residual clipping, same hysteresis logic.
- HST-only sources contribute to `H_rr` (the transformation normal equations) — this is the primary scientific gain: denser spatial coverage constrains higher-order polynomial distortion terms.

**Warm start**: `r_hat` is initialised from the Phase 4/5 posterior (the previous `r_hat` used to build the master catalogue, stored in `_p4` stash). Prior widths on `r_hat` are unchanged.

### Open question: parallax prior for HST-only sources

With zero Gaia inverse covariance, the Michalik prior (σ_plx ≈ 4–11 mas at G=20 depending on field direction) is now implemented and substantially reduces the PM inflation compared to the old 20 mas flat prior. Whether this is tight enough for fields with only two HST epochs (e.g. E3's ~5× inflation at the old σ=20 mas drops to ~2–3× with σ≈5 mas) is a question to revisit after the first v2 run.

### Key benefit for GDC understanding

A large uniform sample of faint stars across the full chip area, with BP3M-fitted residuals, reveals spatial systematics in GDC corrections at a level not possible with sparse Gaia-only samples. This is especially valuable for per-chip or per-quadrant distortion residuals in ACS/WFC and WFC3/UVIS fields.

### Planned future iteration

After the v2 BP3M run completes, Phase 5 of `run_hst_crossmatch` should be re-run using the improved v2 transformation parameters, then the v2 master catalogue rebuilt, then BP3M v2 re-run. One such outer loop is likely sufficient.

---

## Performance Scaling of `hst_catalog_crossmatch.py`

Runtime scales poorly beyond ~10,000 sources per image and becomes prohibitive at >100,000 (fields with many deep exposures can reach >1,000,000 total detections). The main bottlenecks and potential fixes are below.

### Phase 1 (within-filter matching) — dominant cost for large fields

**Repeated KDTree builds**: The master-source KDTree is rebuilt from scratch before each new image is matched. For N_master sources and N_images images, this is O(N_images × N_master × log N_master). For 100 images with 100K master sources, ~100 tree builds dominate runtime.

- **Fix**: Build the tree once per filter at the start, or maintain it incrementally. Since master positions are propagated to the current image epoch (PM × dt), a static tree won't work directly, but at short baselines the shift is small. Alternative: query the tree using bounding-box pre-filters to limit the candidate set before the PM propagation.

**Hungarian assignment (`linear_sum_assignment`)**: Used to resolve ambiguous matches within the search radius. Scales as O(k³) where k is the number of candidates in a local neighbourhood. For crowded fields, k can be large.

- **Fix**: Cap the candidate set to the N_cand nearest neighbours (e.g. N_cand=5) before running the assignment. This already mostly holds in practice; the key is ensuring the search radius is tight.

**Phase 1 is embarrassingly parallel across filters**: Each filter's matching loop is completely independent.

- **Fix (implemented)**: The per-filter loop body is wrapped in `_process_one_filter(filt, fdf_raw)` and dispatched via `ThreadPoolExecutor`. scipy KDTree queries and `linear_sum_assignment` release the GIL, so threads achieve real multi-core speedup. Expected gain: 2–4× for typical 2–4 filter fields.

### Phase 4 (astrometry fitting) — dominant cost for sources with many epochs

`_measure_astrometry_proper` fits a Bayesian linear model per source in a Python loop. For 1M sources this is ~1M Python function calls.

- **Fix (implemented)**: The per-source fitting loop is extracted into `_fit_one_source(row_i)` (a closure over read-only shared lookups) and dispatched in chunks via `ThreadPoolExecutor(max_workers=min(8, cpu_count))`. `np.linalg.inv/solve` (LAPACK) releases the GIL, giving real multi-core parallelism. Expected gain: up to 8× on an 8-core machine.
- **Alternative (not implemented)**: Vectorise by grouping sources with identical epoch counts and solving in a single batched LAPACK call. This would give 50–200× speedup but requires restructuring the full Bayesian normal-equations formulation.

### I/O — significant for very large catalogs

CSV read/write is slow and large for >100K sources. `master_combined.csv` at 1M rows × 30 columns is several hundred MB and takes 10–30s to read.

- **Fix**: Write `master_combined.parquet` alongside (or instead of) the CSV. Parquet is columnar, compressed, and 5–10× faster to read/write. Downstream code and notebooks can use `pd.read_parquet`. The CSV can still be written for human inspection if needed.

### Memory — for fields with millions of detections

Loading all per-image catalogs into memory at once in `_load_all_detections` can exceed RAM for very large fields.

- **Fix**: Process filters one at a time (they are independent), loading and releasing each filter's detections after Phase 1. Only the master catalog (much smaller than raw detections) needs to persist across filters. Currently the code already iterates by filter, but all detections are loaded upfront into `det_df` — this could be deferred.

### Spatial chunking — for extremely large fields (>1M sources)

For fields with very high source density, all operations scale with the total N. Processing in spatial tiles (e.g. CCD quadrants or sky position bins) and merging at the edge would cap per-tile N at a manageable level.

- **Concern**: Stars near tile boundaries need to be matched across tiles. A 10% overlap region with post-hoc deduplication handles this, at the cost of implementation complexity.

### Priority order

1. ~~**Vectorize Phase 4 astrometry**~~ → **Implemented** (ThreadPoolExecutor, up to 8× speedup).
2. ~~**Parallelize Phase 1 across filters**~~ → **Implemented** (ThreadPoolExecutor, 2–4× speedup).
3. ~~**Vectorize Phase 0 (dict-append → DataFrame)**~~ → **Implemented** (per-image DataFrame + pd.concat; eliminates O(N_detections) Python loop).
4. ~~**Vectorize `_project_to_radec` inner loops**~~ → **Implemented** (vectorised design-matrix build + batched einsum for C_hst propagation; removes two per-source Python loops).
5. ~~**Parallelize Phase 0 outer per-image loop**~~ → **Implemented** (ThreadPoolExecutor up to 16 workers; FITS I/O and projection release the GIL).
6. **Parquet I/O** — low-effort, large benefit for repeated runs.
7. **Defer per-filter detection loading** — reduces peak memory, moderate effort.
8. **KDTree / Hungarian improvements** — most complex, only needed if items 6–7 are still insufficient.
9. **Vectorized Phase 4 (batched LAPACK)** — much larger gain than threading but higher implementation complexity; worth considering if Phase 4 is still a bottleneck at >100K sources.

---

## CMD Axis Convention

G always on the y-axis; x-axis always blue − red (shorter wavelength minus longer wavelength). For G vs F814W: x = G − F814W (positive for red stars). For F475W vs G: x = F475W − G (positive for red stars). The wavelength ordering of the pair determines the colour sign, independently of which band is assigned to the y-axis.
