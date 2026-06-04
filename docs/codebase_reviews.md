# Codebase Reviews

## 1. GaiaHub (original) — `~/GaiaHub-master/python_codes/`

Two files: `GaiaHub.py` (CLI entry point) and `GaiaHubmod.py` (all functions, ~2700 lines).

**CLI inputs** (`GaiaHub.py`):
- `--name`, `--ra`, `--dec`, `--search_radius/width/height`
- `--min_gmag` (16.0), `--max_gmag` (21.5)
- `--hst_filters` (default `any`), `--hst_im_type` (`_flc`)
- `--time_baseline` (2190 days), `--hst_integration_time_min/max`
- `--source_table` (default `gaiadr3.gaia_source`)

**Gaia download** (`incremental_query`, `gaia_query`):
- Uses `astroquery.gaia` TAP interface, ADQL queries
- Downloads in parallel magnitude bins to avoid query row limits
- ADQL columns: `source_id`, `ra`, `dec`, `ra_error`, `dec_error`, all correlations, `parallax`, `pmra`, `pmdec` + errors, `gmag`, `bpmag`, `rpmag`, `bp_rp`, `ruwe`, `pseudocolour`, `ref_epoch`, quality flags
- Saves to `./{name}/Gaia/{name}_clean_table.csv`

**HST download** (`search_mast`, `download_HST_images`):
- Uses `astroquery.mast.Observations.query_criteria` and `download_products`
- Filters: `ACS/WFC` and `WFC3/UVIS` only, by RA/Dec box, filters, exposure time, time baseline
- Downloads FLC images (or FLT) to `./{name}/HST/mastDownload/HST/{obs_id}/`

**Legacy steps** (replaced in this project):
- PSF fitting: calls Fortran `hst1pass` binary
- Cross-matching: calls Fortran `xym2pm` binary

**Output structure**: `{name}/Gaia/`, `{name}/HST/`, `{name}/GaiaHub_output/`

**Useful utilities**:
- `get_object_properties` — resolves target name to RA/Dec
- `correct_flux_excess_factor`, `pre_clean_data` — Gaia quality flags
- `get_real_error` — inflation of Gaia positional errors per EDR3 papers
- `columns_n_conditions` — ADQL query builder

---

## 2. hst1pass_improved (py1pass) — `../hst1pass_improved/py1pass/`

Python PSF-fitting package replacing GaiaHub's Fortran `hst1pass`.

**Package structure**:
- `py1pass/core.py` — PSF evaluation with Numba/NumPy bicubic B-spline kernel, source finding, linear fitter
- `py1pass/io.py` — STDPSF/STDGDC loaders, FITS image reader, GDC correction, `catalog_to_table`
- `py1pass/multipass.py` — multi-pass fitting loop
- `py1pass/diagnostics.py` — `estimate_systematic_floor`, `plot_diagnostics`, `plot_catalog_stats`, `plot_psf_residual_map`
- `scripts/py1pass_run.py` — CLI entry point

**Input**:
- HST FLC or FLT FITS file (ACS/WFC, WFC3/UVIS, WFC3/IR supported)
- `--lib_dir` pointing to `lib/STDPSFs/{detector}/` and `lib/STDGDCs/{detector}/` (Anderson STDPSFs library)
- Key parameters: `--fmin` (min flux above sky), `--hmin` (4 px NMS radius), `--n_passes` (1), `--half_width` (3)

**Output catalog** (FITS/ECSV/CSV, one row per detected star):
- Raw frame: `x`, `y`, `flux`, `flux_err`, `sky`, `mag`, `mag_err`, `qfit`, `chi2`, `dist_nearest`
- Full covariance: `cov_xx`, `cov_yy`, `cov_xy`, `cov_ff`, `cov_fx`, `cov_fy`, etc.
- GDC-corrected (distortion-free Anderson J-frame): `x_gdc`, `y_gdc`, `mag_gdc`, `mag_err_gdc`, `cov_xx_gdc`, `cov_yy_gdc`, `cov_xy_gdc`
- STMAG-calibrated, EXPTIME-corrected: `mag_st_gdc`, `mag_err_st_gdc` — **preferred for cross-matching**
- WCS sky coords: `ra`, `dec`, `ra_err`, `dec_err`, `cov_ra_ra`, `cov_dec_dec`, `cov_ra_dec`
- Header keywords: `CHIP{n}_CRPIX1_GDC`, `CHIP{n}_CRPIX2_GDC`, `CHIP{n}_CRVAL1/2` — used by fast_cross_match

**Magnitude columns**:
- `mag_gdc`: uncalibrated instrumental mag + GDC pixel-area correction. No EXPTIME correction.
- `mag_st_gdc`: STMAG zero-point + GDC area correction + `+2.5*log10(EXPTIME)` correction. Preferred for photometric comparison across images. py1pass reads EXPTIME from the primary FITS header.
- `magnitude_zp_offsets.csv`: cross-image ZP offsets measured relative to `mag_gdc`. Do **not** apply these to `mag_st_gdc` (which is already on an absolute scale).

**Bulk mode**: `--image_bulk_dir` processes all `{dir}/{name}/{name}_flc.fits` in one call

---

## 3. fast_cross_match_claude — `../fast_cross_match_claude/`

Python cross-matcher replacing GaiaHub's Fortran `xym2pm`.

**Key files**:
- `cross_match_cli.py` — main CLI (`process_single_image`, `main`)
- `catalog_matcher.py` — `fit_affine_weighted`, `fit_4p_weighted`, `apply_affine`, `compute_mahalanobis`, `find_scale_and_offset`
- `miracle_match.py` — triangle-hash geometric matching (`miracle_match`, `rd2x`, `rd2y`)
- `diagnostic_plotter.py` — diagnostic plot helpers

**Input**:
- `--target` (field name) + `--data-dir` (root)
- Gaia CSV: `{data_dir}/{target}/Gaia/*.csv` — needs columns: `source_id`, `ra`, `dec`, `gmag`, `pmra`, `pmdec`, `parallax`, full 5×5 covariance columns, `pseudocolour`, `ref_epoch`, `bp_rp`
- HST catalog FITS: `{data_dir}/{target}/HST/{img}/{img}_flc_catalog.fits` — needs: `x_gdc`, `y_gdc`, `mag_gdc`, `mag_err_gdc`, `cov_xx_gdc`, `cov_yy_gdc`, `cov_xy_gdc`, `qfit`, `chi2`; header with `CHIP{n}_CRPIX1_GDC` etc.

**Algorithm**:
1. Propagate Gaia positions to HST epoch (including parallax, PM)
2. 4-parameter (similarity) discovery: tier-walks qfit × magnitude limits
3. 6-parameter (affine) refinement: iterative matched-filter fitting
4. Final match pass with converged transform

**Output per image** (written to `{data_dir}/{target}/HST/{img}/`):
- `matched_gaia.csv`: `hst_index`, `hst_x_gdc`, `hst_y_gdc`, `hst_mag_gdc`, `gaia_source_id`, `gaia_ra_prop`, `gaia_dec_prop`, `gaia_gmag`, `residual_x`, `residual_y`, `residual_sigma`
- `transformation.csv`: `A`, `B`, `C`, `D`, `xs_o`, `ys_o`, `xt_o`, `yt_o`, `ra_cen`, `dec_cen`, `x_cen`, `y_cen`, `pixel_scale`, `orientat`
- Diagnostic plots: `diagnostic_plots.png`, `offset_histogram.png`

---

## 4. bp3m_improved — `../bp3m_improved/`

Bayesian astrometric alignment and proper motion measurement, replacing GaiaHub's alignment step.

**Package structure** (`bp3m/`):
- `data_loader_flc.py` — loads from new FLC pipeline layout (used by this project)
- `data_loader.py` — loads from GaiaHub legacy layout (Bayesian_PMs dir)
- `solver.py` — `BP3MSolver` (dense)
- `solver_sparse.py` — `BP3MSolverSparse` (faster for large fields)
- `checkpointing.py` — save/load inputs and results
- `coords.py`, `astro_utils.py` — coordinate utilities

**Entry point**: `run_bp3m.py`

**FLC pipeline input** (mode `--flc-pipeline`):
```
{data_root}/{field}/HST/mastDownload/HST/{img}/{img}_flc.fits
{data_root}/{field}/HST/mastDownload/HST/{img}/{img}_flc_catalog.fits
{data_root}/{field}/HST/mastDownload/HST/{img}/transformation.csv
{data_root}/{field}/HST/mastDownload/HST/{img}/matched_gaia.csv
{data_root}/{field}/Gaia/*_gaia.csv
```

**Key parameters**: `--n-iter` (20), `--n-samples` (1000), `--clip-sigma` (4.5), `--poly-order` (1=linear), `--split-ccd`, `--sparse`

**Outputs** → `{data_root}/{field}/BP3M_results/`:
- `v_mean` (n_stars × 5): posterior mean [ra, dec, parallax, pmra, pmdec]
- `v_cov` (n_stars × 5 × 5): posterior covariance
- `r_hat` (n_r,): image transformation posteriors
- `C_r` (n_r × n_r): transformation covariance
- Diagnostic plots via `plot_results.py`
