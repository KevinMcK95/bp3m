# Interface Constraints and Conventions

## Key Interface Constraints

1. **Gaia column naming**: bp3m and fast_cross_match both expect `gmag` (not `phot_g_mean_mag`). GaiaHub uses `phot_g_mean_mag AS gmag` in its ADQL query — we must do the same.
2. **py1pass → fast_cross_match**: The catalog FITS must contain `x_gdc`, `y_gdc`, `mag_gdc`, `cov_xx_gdc`, `cov_yy_gdc`, `cov_xy_gdc`, `qfit`, `chi2` plus header keywords `CHIP{n}_CRPIX1_GDC` — all produced by py1pass when given a GDC library.
3. **fast_cross_match → bp3m**: `matched_gaia.csv` and `transformation.csv` columns must match what `data_loader_flc.py` expects.
4. **PSF library**: py1pass requires external STDPSFs and STDGDCs library files. The tool must accept a `--lib_dir` argument pointing to these.
5. **HST instruments**: Currently supporting ACS/WFC and WFC3/UVIS (as in GaiaHub). WFC3/IR is partially supported in py1pass but not in fast_cross_match's pixel-scale table. JWST instruments (`NIRCAM`, `NIRISS`) are stubbed and will raise `NotImplementedError` until py1pass and fast_cross_match are updated.
6. **Gaia CSV column casing**: The downloaded Gaia CSV has `SOURCE_ID` (uppercase). Any code that joins on this column must normalise to lowercase (`gdf.rename(columns=str.lower)`) before indexing. The `gaia_source_id=0` value in the combined catalog means no Gaia match and must be excluded from lookups.

---

## Gaia Covariance Inflation (bp3m convention)

Applied identically in `bp3m/solver.py` and `ghi/catalog_utils.py`, and reproduced in notebooks via `ghi/explore_utils.build_gaia_cov()`:

```python
from bp3m.astro_utils import GAIA_SYS_DICT
# GAIA_SYS_DICT = {'mult_6p': 1.22, 'mult_5p': 1.05, 'mult_2p': 1.00,
#                  'parallax_sys_err': 0.011,  # mas
#                  'pm_sys_err': 0.026}          # mas/yr

gaia_6p = np.isfinite(df['pseudocolour'])
gaia_5p = np.isfinite(df['pmra']) & ~gaia_6p
gaia_2p = ~gaia_5p & ~gaia_6p

C[gaia_6p] *= GAIA_SYS_DICT['mult_6p']   # multiply covariance by factor (not factor²)
C[gaia_5p] *= GAIA_SYS_DICT['mult_5p']
C += np.diag([0, 0,
              GAIA_SYS_DICT['parallax_sys_err'],
              GAIA_SYS_DICT['pm_sys_err'],
              GAIA_SYS_DICT['pm_sys_err']])**2   # systematic floor on plx, pm
```

**NOTE**: Both codepaths multiply by `mult_6p` (not `mult_6p²`). This is intentional and internally consistent.

**Geometric mean uncertainty** from a 2×2 sub-covariance (e.g. position or PM):
`σ_geom = det(C_2x2)^(1/4)` = `sqrt(σ_x * σ_y)` in the uncorrelated limit.

---

## BP3M Output Columns (in `proper_motions.csv`)

All Gaia catalog columns, plus:
- `pmra_bp3m`, `pmdec_bp3m`, `parallax_bp3m` — marginalised posterior means (mas/yr, mas)
- `delta_racosdec_bp3m`, `delta_dec_bp3m` — position update from Gaia prior (mas)
- `sigma_pmra_bp3m`, `sigma_pmdec_bp3m`, `sigma_parallax_bp3m` — marginalised 1σ
- `pmra_bp3m_cond`, `pmdec_bp3m_cond` — conditional (image transforms fixed) means
- `sigma_pmra_bp3m_cond`, `sigma_pmdec_bp3m_cond` — conditional 1σ
- Correlation terms: `corr_pmra_pmdec`, `corr_pmra_plx`, `corr_pmdec_plx`, `corr_dra_pmra`, etc.
- Full 5×5 marginalised covariance saved separately as `v_cov_marginalised.npy` (n_stars × 5 × 5)
