# Exploration Utilities and Notebooks

## Shared utility module: `ghi/explore_utils.py`

Functions callable from both the pipeline and notebooks:

| Function | Description |
|---|---|
| `load_gaia_catalog(path)` | Load Gaia CSV, compute `bp_rp` if missing, parse column types |
| `load_bp3m_results(results_dir)` | Load `proper_motions.csv` + `v_cov_marginalised.npy` |
| `build_gaia_cov(df)` | Build inflated 5×5 covariance per star (matches bp3m exactly) |
| `cov2_geom_sigma(C_2x2)` | Geometric mean: `det(C)^(1/4)`, vectorised over N×2×2 |
| `extract_subcov(C_5x5, i, j)` | Extract 2×2 sub-covariance from N×5×5 for axes (i,j) |
| `pm_uncertainty(C_5x5)` | `cov2_geom_sigma(C[:, 2:4, 2:4])` |
| `pos_uncertainty(C_5x5)` | `cov2_geom_sigma(C[:, 0:2, 0:2])` |
| `propagate_gaia(df, mjd)` | Propagate Gaia positions to target MJD (parallax + PM) |
| `load_cross_match(hst_dir)` | Load all `matched_gaia.csv` files for a field |
| `load_transformations(hst_dir)` | Load all `transformation.csv` files for a field |

---

## Notebooks: `notebooks/`

**01_field_overview.ipynb**
- Sky positions of Gaia sources (RA/Dec scatter, coloured by magnitude)
- Gaia CMD: G vs BP-RP
- HST CMDs: instrumental magnitude (from `hst_mag_gdc`) vs Gaia G for each filter
- Per-image footprints overlaid on the sky

**02_proper_motions.ipynb**
- VPD (μ_α* vs μ_δ): Gaia PMs and BP3M PMs side-by-side
- Error ellipses on VPD
- PM uncertainty histogram: Gaia vs BP3M comparison
- Parallax distribution
- PM vs sky position (check for spatial trends)

**03_astrometric_quality.ipynb**
- Position σ_geom vs G magnitude (Gaia inflated cov and BP3M)
- PM σ_geom vs G magnitude (Gaia inflated cov and BP3M)
- Improvement ratio: σ_PM(Gaia) / σ_PM(BP3M) vs magnitude and number of images
- Correlation matrix visualisation (median per field)

**04_cross_match_diagnostics.ipynb**
- Residual (dx, dy) maps per image
- Residual vs magnitude per image
- Match count and transformation parameters (scale, rotation) per image
- Sigma distribution histogram

**05_synthetic_test.ipynb** *(planned — see [synthetic_test_plan.md](synthetic_test_plan.md))*
- Comparison of BP3M recovered parameters against synthetic ground truth
- Pull distributions, uncertainty calibration, VPD comparison
