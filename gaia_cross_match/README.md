# gaia_cross_match

> **Note:** gaia_cross_match is bundled as part of [bp3m](https://github.com/KevinMcK95/bp3m) and is no longer maintained as a standalone package.

Fast astrometric cross-matching between HST PSF-fitting catalogs and Gaia DR3.
Uses an affine transformation model with robust outlier rejection.

Inspired by the Fortran cross-matching routine `xym2pm_GH.F` from GaiaHub
(del Pino et al. 2022, ApJ 933 76; https://github.com/AndresdPM/GaiaHub).

## Installation

```bash
pip install git+https://github.com/kevinmckinnon/gaia_cross_match
```

Or in development mode after cloning:

```bash
git clone https://github.com/kevinmckinnon/gaia_cross_match
cd gaia_cross_match
pip install -e ".[dev]"
```

## Quick usage

### Command-line

```bash
python -m gaia_cross_match.cross_match \
    --target Fornax_dSph \
    --data-dir ./data \
    --threads 4
```

### Python API

```python
import numpy as np
from gaia_cross_match import (
    process_single_image,
    load_gaia_data,
    find_hst_image_folders,
    fit_affine_weighted,
    apply_affine,
)

# Load Gaia catalog for a field
gaia_df = load_gaia_data("Fornax_dSph", "./data")

# Discover HST image folders (each must contain an _flc_catalog.fits)
hst_folders = find_hst_image_folders("Fornax_dSph", "./data")

# Cross-match a single image
process_single_image(hst_folders[0], gaia_df)

# Low-level: fit a weighted affine transform
n = 50
cov = np.zeros((n, 2, 2))
cov[:, 0, 0] = 0.05**2
cov[:, 1, 1] = 0.05**2
result, p_err, inv_lhs, chi2 = fit_affine_weighted(
    x_src, y_src, x_tgt, y_tgt, cov, cov
)
A, B, C, D, xs_o, ys_o, xt_o, yt_o = result
x_proj, y_proj = apply_affine(x_src, y_src, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
```

## Cross-image validation

After all images are processed, run validation to produce per-image
`source_quality.csv` files and a target-level `cross_match_catalog.csv`:

```python
from gaia_cross_match import validate_target

validate_target("Fornax_dSph", "./data")
```

## Status and feedback

gaia_cross_match has been tested on a range of real HST–Gaia datasets, but as with any research software there may be edge cases and bugs that haven't been caught yet. If you run into unexpected behaviour or incorrect results, please open a GitHub issue — all feedback is welcome.

## Development notes

The Python translation from the original Fortran routine and subsequent optimizations were developed with assistance from [Claude Code](https://claude.ai/code) (Anthropic).

## Attribution

This package is a Python reimplementation of algorithms from the GaiaHub
Fortran routine `xym2pm_GH.F`:

> del Pino, A., et al. (2022). *GaiaHub: A Method for Combining HST and Gaia
> to Obtain Improved Proper Motions for HST Observations.* ApJ, 933, 76.
> https://doi.org/10.3847/1538-4357/ac71ae

Fortran source: https://github.com/AndresdPM/GaiaHub/blob/master/fortran_codes/xym2pm_GH.F
