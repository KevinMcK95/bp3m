# Implementation Notes

## Smart caching — all steps default to using cached results

- **Step 1 (Gaia)**: Filename encodes query geometry — `{field}_ra{ra:.4f}_dec{dec:+.4f}_w{w:.4f}_h{h:.4f}_G{min}[_{max}]_gaia.csv`. A JSON sidecar `{stem}.query.json` stores all query params (including ADQL). On re-run the sidecar is compared to current params; a mismatch prints a diff and re-downloads. Individual magnitude-bin CSVs are cached during download then deleted on success. Override: `--force_redownload_gaia`.
- **Step 2 (HST)**: MAST search results cached as `{field}_obs.csv` / `{field}_data_products.csv`. Each FITS file is validated on disk (size check vs MAST-reported size, `fits.open()` integrity check, and `EXPTIME==0` failed-observation check) before being skipped. Failed observations (e.g. `EXPFLAG='EXCESSIVE DOWNTIME'`) are kept on disk but written to `{field}_failed_obsids.json` and excluded from `{field}_selected_obsids.json` so all downstream steps skip them automatically. Override: `--force_redownload_hst`.
- **Step 3 (PSF)**: Each image skipped if `{img}_flc_catalog.fits` + `psf_params.json` both exist and params match. The sidecar is deleted before writing the catalog and re-written only after a successful write, so an interrupted run leaves no stale sidecar. Override: `--force_refit_psf`.
- **Step 4 (cross-match)**: Each image skipped if `matched_gaia.csv` + `xmatch_params.json` both exist and params match. Same delete-before-write sidecar safety. Override: `--force_rematch`.

**Sidecar safety pattern**: All JSON sidecars are deleted *before* writing the corresponding data file and only written *after* the data file is fully flushed. This ensures an interrupted run always leaves no sidecar, so the next run re-runs rather than loading a corrupt or partial output.

---

## PSF fitting (`ghi/psf_fitting.py`)

**Defaults**: `n_passes=2`, `max_iter_fit=100`, `fmin=100`, `hmin=4`, `half_width=3`, `sat_threshold=60000`, `conc_limit=0.9`. Each fitted image produces five diagnostic figures: `psf_catalog_stats.png`, `psf_diagnostics.png`, `psf_residual_map.png`, `psf_concentration.png`, `psf_perturbation.png` — all prefixed `psf_` to avoid collision with cross-match outputs. A binary `psf_delta.npy` is also written (cumulative PSF perturbation).

**Progress display**: `run_psf_fitting` and `remeasure_psf_perturbation` both print `[i/N] field_name image_name` for each image so progress is visible without scrolling.

**PSF perturbation measurement**: After each fitting pass, `measure_psf_perturbation()` drizzles normalised leave-one-out residuals `(res_loo/flux − P)` from all star-candidate detections into the oversampled PSF grid with flux²-weighted bilinear drizzling. Two constraints are enforced: (1) zero sum — preserves flux normalisation; (2) zero first moments — preserves PSF centroid so GDC corrections remain valid. `psf_delta.npy` stores the **cumulative** correction relative to the bare stdpsf; each iteration adds its incremental δP to the running total.

**PSF perturbation outlier rejection (two-pass sigma-clip)**: Pass 1 drizzles all qualifying stars. A scoring pass evaluates each star's weighted RMS against the consensus δP — stars whose score exceeds `median + 4σ × 1.4826 × MAD` are excluded. Pass 2 re-drizzles surviving stars. Clipping only activates when ≥ 20 qualifying stars are present.

**PSF perturbation coverage floor**: `coverage_min_frac=0.05` (5% of peak accumulated weight) zeros `delta_raw` at under-sampled PSF edge pixels before applying zero-sum and zero-moment constraints. This eliminates boundary spikes. Set `coverage_min_frac=0` to disable.

**Iterative PSF refinement logic per image**:
- `psf_delta.npy` exists and `--clean_psf` not set → **1 iteration**, using stored cumulative δP.
- No `psf_delta.npy` (or `--clean_psf`) → **2 iterations** — iter 1 builds δP from bare stdpsf, iter 2 applies it.
- `--n_psf_iter N`: explicit override.
- `--force_refit_psf`: re-fits even if catalog is cached; uses any existing `psf_delta.npy` (same rules).
- `--clean_psf`: ignores `psf_delta.npy`, starts from bare stdpsf.

**eps_psf_star noise inflation**: `fit_star()` accepts `eps_psf_star` (float, default 0). When > 0, adds `(eps_psf_star × max(flux × P, 0))²` to per-pixel variance. Pass 1 uses `eps_psf_star=0`; pass 2+ passes `rec.eps_psf` from the previous pass.

**Reclassification pipeline** (`--reclassify_stars`): Re-runs the full post-fit calibration chain on existing catalogs without re-fitting. Sequence: (1) load catalog, un-inflate chi2 → raw photon-noise covariance; (2) `classify_stars()` with new `conc_limit`; (3) `inflate_chi2()` with new star population; (4) re-apply GDC Jacobian (requires `--lib_dir`; approximation used if unavailable); (5) `estimate_systematic_floor()`; (6) patch all dependent columns. Invalidates the cross-match cache.

**Remeasure PSF perturbation** (`--remeasure_psf_perturbation`): Standalone re-measurement of `psf_delta.npy` on already-fitted images. Loads catalog, applies the existing cumulative `psf_delta.npy` to the stdpsf cube, reconstructs the per-chip residual by subtracting all star models, then calls `measure_psf_perturbation`.

**GDC covariance correctness**: The stored `cov_xx/yy_gdc` columns are `chi2_scale² × J @ raw_cov @ J.T + floor²`. During reclassification, `_records_from_fits_table` strips both the floor and the chi2 inflation before returning records, so `r.cov` holds the truly raw photon-noise covariance. Off-diagonal covariances are also chi2-scaled (no floor).

**Star/non-star classification**: py1pass computes three concentration metrics per detection — `concentration` (1×1 peak pixel), `concentration_2x2` (2×2 sum), `concentration_3x3` (3×3 sum) — each normalised to ≈1.0 for a perfect point source. An iterative `classify_stars()` routine fits per-magnitude-bin adaptive bounds (±4σ) and sets `is_star_candidate` (bool) in the output catalog.

---

## HST cross-matching (`ghi/cross_match.py` + `ghi/hst_catalog_crossmatch.py`)

### Per-image Gaia matching (`ghi/cross_match.py`)

- Calls `cross_match_cli.process_single_image` per image
- **Requires `is_star_candidate`** in HST catalog; images without it are skipped with a warning
- 4P discovery uses star candidates only; 6P affine refinement and final pass use all sources, each tagged with `hst_is_star`
- `cross_match_validator.validate_target` writes `cross_match_catalog.csv` with per-source cross-image star statistics (`is_star_all_images`, `is_star_any_image`, `non_star_images`)
- Saves `xmatch_params.json` sidecar; skips images where `matched_gaia.csv` + sidecar exist with matching params

### Within-field PM measurement (`ghi/hst_catalog_crossmatch.py`)

`run_hst_crossmatch()` aggregates all per-image HST catalogs into a field-wide master catalog and measures proper motions from the multi-epoch positions.

**`_within_filter_match`**: groups all HST detections by filter, builds a running master source list, then cross-matches detections image-by-image (ordered by epoch) to build multi-epoch trajectories. Key design decisions:

- **Two-pass star/non-star**: Stars (is_star_candidate=True) are matched first with standard `max_mag_diff=0.75` mag. Non-stars are matched second with `max_mag_diff * nonstar_mag_relax=1.5`, and cannot claim master entries already taken by stars. This lets non-stars be tracked without polluting the star master list.
- **Epoch tracking**: Each master source carries a running weighted-mean epoch `master_epoch`. The PM search radius at each new image is computed using `dt = |master_epoch - cur_epoch_mjd| / 365.25`, so the tolerance correctly reflects actual time gaps between the source's mean epoch and the current image. Critical for fields with multi-year baselines (e.g. Pal5: 2011 + 2014 observations).
- **`mag_st_gdc` required**: `_load_all_detections` enforces that every catalog has the `mag_st_gdc` column (py1pass STMAG calibrated via PHOTFLAM/EXPTIME + GDC pixel-area correction). If it is absent the catalog is stale (old py1pass version); the catalog file and its `psf_params.json` sidecar are deleted so py1pass will re-run on the next pipeline invocation, and the image is skipped for the current run.
- **Sigma-based magnitude matching**: `_match_two_sets` uses `|Δmag| < max(mag_n_sigma × √(σ_a² + σ_b²), mag_floor)` instead of a fixed threshold. Defaults: `mag_n_sigma=3.0`, `mag_floor=0.01` mag. Since `mag_st_gdc` is on an absolute STMAG scale within a filter, no ZP correction is needed and the combined photometric uncertainty is the only relevant scale — bright stars get a tighter cut, faint stars get an appropriate looser one. Non-star matching uses `mag_n_sigma × nonstar_mag_relax` (default relaxation factor 1.5). The master magnitude is updated with inverse-variance weighting and `master_mag_err` tracks the propagated uncertainty.
- **Inter-image ZP consistency check**: After within-filter grouping, `_process_one_filter` computes the per-image median `mag_st_gdc` of star candidates and checks against the filter-wide median. A deviation >0.1 mag prints a `WARNING`; >0.5 mag prints an `ERROR`, listing each offending image with its offset. Since `mag_st_gdc` is an absolute STMAG calibration, inter-image ZP scatter should be effectively 0; any substantial offset indicates a stale catalog, wrong filter assignment, or failed cross-match.
- **Post-grouping magnitude outlier rejection**: After grouping, sources with ≥ 3 detections have their per-detection magnitudes compared to the weighted mean. Detections deviating by more than `max(5σ, 0.5 mag)` (MAD-based) are rejected if at least `min_detections=2` remain. This catches cross-match misidentifications that slipped through the magnitude search radius.
- **Filter-level parallelism**: The per-filter loop body is wrapped in `_process_one_filter(filt, fdf_raw)` — a closure capturing `ra0`, `dec0`, and all parameters. When >1 filter is present, all filters are dispatched concurrently via `ThreadPoolExecutor`. The `_update_master`/`_extend_master` helpers (which use `nonlocal` to update per-filter master arrays) remain correctly scoped inside the closure. scipy KDTree queries and `linear_sum_assignment` release the GIL, so threads run in true parallel for the matching-intensive inner loop.

**Source deduplication** (`_deduplicate_merged`): After Phase 2 cross-filter merging, `_deduplicate_merged` enforces the invariant that (a) each Gaia source ID appears in at most one row, and (b) no HST detection `(sub_name, catalog_index)` pair appears in more than one row. Two passes:

- **Pass 1 — Gaia-ID duplicates**: Groups all rows sharing the same `gaia_source_id`. The row with the most detections becomes the primary; all secondary rows' `hst_indices_*` columns are merged in (concatenating detection lists after checking for shared `sub_name` conflicts). After merging, `n_detect`, `n_filters`, and `filter_list` are recomputed from the merged detection set.
- **Pass 2 — Cross-filter positional close pairs**: Builds a KD-tree on `ra0`/`dec0`. Pairs within `pos_threshold_mas` (default 50 mas) where the two rows have disjoint or overlapping-but-not-identical filter sets and no shared `sub_name` references are merged. The row with more detections is primary. Same-single-filter pairs (both have exactly one filter and they match) are left untouched — these are genuine blends or real close pairs in a dense field, not cross-matching failures. Index preservation: the returned DataFrame uses the same integer index as the input, so subsequent row comparisons (e.g. in Phase 4b) are safe.

**Phase 4 astrometry** (`_measure_astrometry_proper`): 5-parameter Bayesian fit per source in tangent-plane pixel space, marginalising over the BP3M transformation uncertainty `C_r`. The 5 parameters are (Δα*, Δδ, μα*, μδ, ϖ) in units of (mas, mas, mas/yr, mas/yr, mas). For each detection `j` in image with epoch `dt_j = epoch_j − 2016.0` (decimal years), the design matrix row is:

```
H_j = J_j @ U_j
U_j = [[1, 0, dt_j, 0,    f_ra_j ],
       [0, 1, 0,    dt_j, f_dec_j]]
```

where `J_j` is the 2×2 tangent-plane Jacobian from BP3M and `f_ra_j`, `f_dec_j` are the Earth parallax factors at epoch `j`, computed via `get_tele_position(AstropyTime(mjd_j, 'mjd'), curr_id='earth')` and `get_parallax_factors(ra, dec, xyz)` — identical to BP3M `solver.py`. Earth positions are looked up once per unique image (`_tele_xyz_cache`) before the parallel fitting loop, not once per detection. Priors: `σ_pos = 1e6 mas` (diffuse), `σ_pm = 100 mas/yr` (diffuse), `σ_plx = 20 mas` (diffuse). Gaia-matched sources additionally receive the full 5×5 Gaia posterior covariance as a prior. PMs are set to NaN if either PM uncertainty exceeds `_SIGMA_PM_DIFFUSE / 10 = 10 mas/yr` (effectively unconstrained).

**PM uncertainty inflation for HST-only sources**: Because the parallax prior for HST-only sources is diffuse (σ=20 mas), the geometric coupling between the parallax column and the PM column in the normal equations can inflate PM uncertainties substantially when only two epochs are available and they do not span the full annual parallax ellipse. For E3 (F606W 2006.3 + F814W 2021.9, ~7.5 months apart in the year), the simulation gives ~5× inflation for 2-epoch sources. This is correct behaviour — it honestly reflects the astrometric information content. Gaia-matched sources are unaffected because their tight Gaia parallax prior pins the parallax term. Adding more HST epochs spread across the year would reduce the inflation.

**Phase 4 parallelism**: The per-source fitting loop is extracted into `_fit_one_source(row_i)` — a closure over the pre-built `det_lookup`, `gaia_lookup`, `src_detections`, `r_hat_arr`, and `C_r`. All shared data is read-only (no mutable state between sources). Sources are chunked and dispatched to a `ThreadPoolExecutor(max_workers=min(8, cpu_count))`. `np.linalg.inv` and `np.linalg.solve` (LAPACK) release the GIL, giving real multi-core parallelism. Falls back to a serial list comprehension for small catalogs (<200 sources).

**Phase 4b — post-astrometry deduplication**: After Phase 4 computes accurate `ra_xmatch`/`dec_xmatch` positions, `_deduplicate_merged` is run a second time using those positions instead of the Phase-1 `ra0`/`dec0` estimates. This catches cross-filter close pairs that Phase 2 missed because Phase-1 positional offsets between filters were too large (e.g. 595 mas for an E3 F606W/F814W pair). The procedure: (1) copy `ra_xmatch`/`dec_xmatch` into `ra0`/`dec0` for sources where they exist; (2) call `_deduplicate_merged`; (3) restore original `ra0`/`dec0` and `pmra`/`pmdec` for surviving rows; (4) re-run `_measure_astrometry_proper` only for the subset of sources whose `n_detect` increased (those that absorbed additional detections). Typically O(1)–O(10) sources are re-fitted in Phase 4b for well-processed fields.

**Phase 5 — PM-guided second-pass cross-match** (`_second_pass_match`): After Phase 4 PMs are measured, a second pass over all images can recover detections missed by the initial within-filter matching. The key idea is that the first-pass PM estimate (plus Gaia parallax where available) predicts each source's position at each image epoch much more precisely than a blind position-only search.

- **Image ordering** (`_order_images_greedy`): Images are ordered greedily so the most informative images are searched first. Score = `(overlap_weight + 1e-3) × exp(-|Δt| / time_scale)` where `overlap_weight` counts how many first-pass detections a given image would contribute to a current template source. A centre-based overlap estimate runs in O(N_images²); `time_scale = 3 years` so temporal proximity is mildly preferred but not required.
- **Per-image search**: For each image, the search radius is `max(r_pm_sigma, r_pm_min)` where `r_pm_sigma` is derived from the PM uncertainty propagated to the image epoch. Templates with `σ_PM > second_pass_max_pm_unc` (default 10 mas/yr) are searched with a fixed fallback radius only.
- **Cross-filter magnitude matching**: If the template has photometry in the image's filter (column `mag_wmean_{filt}`), a magnitude check `|Δmag| < max(mag_n_sigma × σ_mag, mag_floor)` gates candidate acceptance. Templates lacking photometry in that filter (true cross-filter matches) skip the magnitude check and match on position only. The ZP between instrument/filter is estimated from the median offset of first-pass matches in that image.
- **Detection set update**: If a source gains or loses detections relative to its first-pass set, `pass2_hst_indices` is set in the output DataFrame. This column deliberately does **not** start with `hst_indices_` so `_parse_hst_indices_columns` does not double-count it alongside the old per-filter columns.
- **Phase 4 re-run for changed sources**: Any source whose `pass2_hst_indices` is set is re-fitted with the full `_measure_astrometry_proper` machinery (Gaia prior, C_r marginalisation, 5-parameter Bayesian fit). The re-fit uses a temporary `hst_indices_pass2` column (which *does* start with `hst_indices_` so it is picked up by `_parse_hst_indices_columns`) after all old `hst_indices_*` columns are dropped from the per-source subset. The resulting astrometry overwrites the corresponding rows in `combined_v2_df`, giving the v2 catalogue identical quality to Phase 4 — no WLS shortcuts.
- **Summary statistics**: After Phase 5, the following are printed: total templates; templates that gained / lost / were unchanged; templates dropped below `min_detections`; median and maximum detection gain; new cross-filter assignments; images searched.
- **Output**: `master_combined_v2.csv` alongside `master_combined.csv`. The v2 file has identical columns; only changed sources differ in astrometry or detection count.

**`_p4` stash**: After a successful Phase 4 run, the key BP3M outputs (`r_hat`, `C_r`, `image_names`, `n_r`, `poly_order`, `sub_img_meta`, `ra0_field`, `dec0_field`, `pscale`) are stashed in a local dict `_p4`. Phase 5 reads directly from this dict to invoke `_measure_astrometry_proper` without re-deriving these quantities.

**Phase 0 (`_load_all_detections`) vectorisation and parallelism**: Each sub-image is processed by `_process_one_sub_image` — a closure capturing `r_vecs`, `alpha_lookup`, `zp_legacy`, `hst_root`, `n_r`, `poly_order` (all read-only). For ≥4 images, up to 16 threads run concurrently (FITS I/O and numpy projection release the GIL). Each call returns a per-image DataFrame built directly from numpy arrays — the old per-source Python dict-append loop (O(N_total_detections)) is replaced by `pd.concat` over per-image DataFrames. For a field with 100 images × 10K sources this reduces from ~1M Python dict appends to ~100 DataFrame constructions.

**`_project_to_radec` inner loops vectorised**: Both per-source Python loops replaced with numpy batch operations: (1) the `build_X_matrix` loop is inlined as direct array assignment + polynomial column fills (columns 6–7 stay 0 in this context, tangent-point derivatives are always 0); (2) the `hst_position_cov` loop is replaced with batched `(n, 2, 2)` covariance construction + einsum for `J_trans @ C_hst @ J_trans.T`.

**Magnitude calibration**: `mag_st_gdc` (py1pass output) is preferred — it is EXPTIME-corrected to STMAG. `mag_gdc` is uncalibrated instrumental mag (no EXPTIME). The `magnitude_zp_offsets.csv` cross-image ZP offsets were measured on `mag_gdc` and must **not** be applied to `mag_st_gdc` (which is already on an absolute scale). The `_apply_legacy_zp` flag in `_load_all_detections` prevents double-correction.

**PSF+GDC availability filtering**: `get_available_psf_gdc_combos(lib_dir)` scans `lib_dir/STDPSFs/` and `lib_dir/STDGDCs/` to find instrument+filter combinations with both files present. Only those combos are queried from MAST and downloaded.

---

## Diagnostic figures (`ghi/hst_catalog_crossmatch.py`)

All plots use grey (`#e8e8e8`) panel backgrounds for visibility of yellow (high-σ) points.
PM plots are filtered to `|pmra| ≤ 50` and `|pmdec| ≤ 50` mas/yr for display; measurements outside this range are retained in `master_combined.csv`.

| File | Description |
|---|---|
| `sky_distribution.png` | RA/Dec scatter coloured by log(|PM|) with `plasma` colormap |
| `vpd.png` | Row 0: VPD with four zoom levels (Full, 95%, 64%, 50% of ±50 mas/yr subsample), points coloured by log(σ_PM_geom). Row 1: σ_PM_geom vs magnitude per filter, same coloring with running median overlay |
| `cmds.png` | All pairwise CMDs from available HST filters + Gaia G (case-insensitive `SOURCE_ID` lookup; Gaia G shown only for matched sources). Points coloured by log(|PM|) with `plasma` colormap. Colorbar via `make_axes_locatable` on last panel. G always on y-axis; x-axis always blue−red (wavelength order, independent of G involvement). Scatter row + hist2d row (log-scaled count, `viridis`). G-involved panels grouped first. |
| `gaia_comparison.png` | 6-panel comparison of xmatch PMs vs Gaia PMs for Gaia-matched sources: (0,0) pmra 1:1 scatter; (0,1) pmdec 1:1 scatter (coloured by G mag); (0,2) σ_PM_geom vs G for both Gaia and xmatch with running medians; (1,0) Gaia VPD; (1,1) xmatch VPD; (1,2) pull histogram (pm_xmatch − pm_gaia) / σ_Gaia with N(0,1) reference |

VPD colorbar: log-scale `plasma_r`, attached to the rightmost VPD panel via `make_axes_locatable`.
CMD colorbar: log-scale `plasma`, attached to the last CMD panel via `make_axes_locatable`.

**Phase 5 v2 diagnostic plots**: When `run_second_pass=True` and Phase 5 changes at least one source, a second set of the same four figures is written with `_v2` suffix (`sky_distribution_v2.png`, `vpd_v2.png`, `cmds_v2.png`, `gaia_comparison_v2.png`). The originals are **not** overwritten. V2 plots remap only the sky positions (`ra_xmatch` / `dec_xmatch`) from the updated `ra0_v2` / `dec0_v2` columns for changed sources; all PM columns in v2 plots come from the full Phase 4 re-fit (stored in the standard `pmra_xmatch` etc. columns after overwriting), not from any WLS intermediates.

---

## BP3M defaults

`split_ccd=True` and `inflate_hst_errors=True`. Disable with `--no_split_ccd` / `--no_inflate_hst_errors`. Always invoked in `--flc-pipeline` mode via `data_loader_flc.py`.

`data_loader_flc.py` automatically reads `cross_match_catalog.csv` (if present) to set initial `use_for_alignment` per star×image pair — sources that are non-stars in all images or inconsistent across images start masked; the EM loop can re-admit them via chi² tests.

**Command echoing**: Steps 3–5 print the equivalent CLI command for the underlying tool (py1pass, fast_cross_match, run_bp3m) before running, for reproducibility and debugging.

**Additional CLI parameters** (added in recent sessions):

| Flag | Default | Description |
|---|---|---|
| `--cross_match_pix_floor` | 0.05 px | Minimum HST pixel error floor applied during per-image Gaia cross-matching. Passed as `hst_pix_floor` to `cross_match.py`. |
| `--bp3m_pos_err_floor` | 5e-3 mas | Minimum positional error floor passed to BP3M via `data_loader_flc.py`; prevents BP3M from treating any detection as perfectly placed. |
| `--max_mag_diff` | 3.0 mag | Maximum allowed Gaia–HST magnitude difference during per-image cross-matching. |
| `--plot_residuals` | False | Pass `--plot_residuals` to BP3M's `run_alignment.py` to write per-iteration residual maps. |
| `--plot_influence` | False | Pass `--plot_influence` to BP3M to write per-source influence diagnostic figures. |
| `--run_second_pass` / `--no_second_pass` | enabled | Enable/disable the Phase 5 PM-guided second-pass cross-match in `hst_catalog_crossmatch.py`. |
| `--second_pass_max_pm_unc` | 10.0 mas/yr | Templates with σ_PM larger than this use only a positional fallback radius in Phase 5. |
