# Synthetic Test Plan (Phase 8)

## Goal

Validate the BP3M solver end-to-end by generating synthetic observations from a real processed field. The "true" stellar parameters and image transformation parameters are known exactly, so the BP3M posteriors can be evaluated against ground truth.

Invoked with `--test_synthetic` on a field that has already completed Step 4 (cross-match). Requires no new network downloads or PSF fitting.

---

## Inputs (from real processed field)

| File | Used for |
|---|---|
| `{field}/Gaia/{field}_gaia.csv` | Gaia prior means and covariances; defines which stars exist |
| `{field}/{tel}/mastDownload/{tel}/{img}/matched_gaia.csv` | Which stars are observed in which image (the observation graph) |
| `{field}/{tel}/mastDownload/{tel}/{img}/transformation.csv` | Initial transformation parameters to use as the true values |
| `{field}/{tel}/mastDownload/{tel}/{img}/{img}_flc_catalog.fits` | Per-star HST positional covariances (noise model) |
| `{field}/BP3M_results/` | (Optional) Use BP3M MAP posteriors as truth instead of Gaia values |

---

## Step-by-step plan

### Step 1: Draw true stellar parameters

For each Gaia star with at least one HST detection, draw true 5-vector `v_true = (Δα*, Δδ, μα*, μδ, ϖ)`:

- **Option A (default)**: Use the real Gaia maximum-likelihood values directly (no random draw). Gives a deterministic, reproducible truth. Position offset `(Δα*, Δδ) = (0, 0)` relative to Gaia prior.
- **Option B** (`--synthetic_draw_from_prior`): Draw `v_true ~ N(v_gaia, C_gaia)`. Tests bias averaged over many realisations.

True parallax is optionally zeroed (`--synthetic_zero_parallax`).

**2-param star parallax**: For 2-param Gaia stars (no measured PM/parallax), true parallax is drawn from `N(0, 1)` (allowing negative values). Using `|N(0,1)|` would create a systematic negative residual bias because BP3M recovers ≈0 for unconstrained parallax.

### Step 2: Draw true transformation parameters

```
r_true[img] = r_from_transformation_csv[img] + N(0, sigma_jitter)
```

`--synthetic_jitter_sigma` defaults to 0. Perturbed values are used as both truth AND initial point for BP3M.

### Step 3: Forward-model synthetic HST positions

For each (star `i`, image `j`) pair in `matched_gaia.csv`:

1. Propagate true sky position to HST epoch using `v_true[i]` (PM + parallax).
2. Apply true transformation `r_true[j]` to get noise-free predicted pixel position `(x_pred, y_pred)`.
3. Read actual per-star covariance `C_hst[i,j]` from `{img}_flc_catalog.fits` (`cov_xx_gdc, cov_yy_gdc, cov_xy_gdc`).
4. Draw: `(x_syn, y_syn) = (x_pred, y_pred) + N(0, C_hst[i,j])`.
5. Draw: `mag_syn = mag_true + N(0, mag_err)` using real `mag_err_gdc`.

### Step 4: Draw synthetic Gaia observations

```
v_gaia_syn = v_true + N(0, C_gaia_inflated)
```

`--synthetic_true_gaia`: skip this step and feed true values directly as Gaia prior mean.

### Step 5: Write synthetic data to disk

```
{output_dir}/{field}/synthetic/
    Gaia/{field}_gaia.csv
    {tel}/mastDownload/{tel}/{img}/
        matched_gaia.csv
        transformation.csv
        {img}_flc_catalog.fits   ← symlink to real file
    truth/
        stellar_truth.csv
        image_truth.csv
```

### Step 6: Run BP3M on synthetic data

`run_alignment()` called on the synthetic directory. Output → `{field}/synthetic/BP3M_results/`.

### Step 7: Compare recovered parameters to truth

**Stellar**: For each `p ∈ {pmra, pmdec, parallax, delta_racosdec, delta_dec}`:
- Residual: `Δp = p_recovered - p_true`
- Pull: `(p_recovered - p_true) / sigma_recovered` (should be ≈N(0,1))

**Image transformations**: For each `p ∈ {a, b, c, d, w, z}`. Note: BP3M has 8 params per image but `_SIGMA_POINTING = 1e-6` arcsec pins `Δα0 ≈ Δδ0 ≈ 0`. Truth is computed with a 6-param fit `_fit_abcdwz`.

**split_ccd truth**: `split_ccd=True` splits each image into `_hi`/`_lo` halves initialized from the same prior, so truth is computed once per full image and applied to both halves.

---

## Diagnostic plots

| Filename | Content |
|---|---|
| `plots_syn_pm_residuals.png` | Δpmra, Δpmdec vs G magnitude; colour by N_HST |
| `plots_syn_pm_pulls.png` | Pull histograms for pmra, pmdec, parallax |
| `plots_syn_pos_residuals.png` | Δ(Δα*), Δ(Δδ) positional offset residuals |
| `plots_syn_image_params.png` | Recovered vs true transformation parameters per image |
| `plots_syn_image_pulls.png` | Pull distributions for image transformation parameters |
| `plots_syn_vpd_comparison.png` | VPD of true PMs vs recovered PMs |

---

## New files to create

| File | Purpose |
|---|---|
| `ghi/synthetic.py` | `generate_synthetic_data()`, `compare_synthetic_results()`, `write_synthetic_truth()` |
| `notebooks/05_synthetic_test.ipynb` | Interactive exploration of synthetic test results |

## CLI additions to `gaiahub_improved.py`

```
--test_synthetic            Run synthetic data test (requires completed cross-match)
--synthetic_draw_from_prior Draw true stellar params from Gaia posterior (default: use MAP values)
--synthetic_zero_parallax   Set true parallax = 0 for all stars
--synthetic_true_gaia       Skip Gaia noise (feed true values directly as prior mean)
--synthetic_jitter_sigma N  Std dev of perturbation added to true transformation parameters (default 0)
--synthetic_seed N          Random seed for reproducibility (default 42)
```

When `--test_synthetic` is set, the pipeline skips Steps 1–4 and runs Steps 5–7. Results go to `{output_dir}/{field}/synthetic/`.
