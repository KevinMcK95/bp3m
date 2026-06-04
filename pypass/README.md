# pypass

> **Note:** pypass is bundled as part of [bp3m](https://github.com/KevinMcK95/bp3m) and is no longer maintained as a standalone package.

Python PSF-fitting photometry for HST images. Implements iterative multi-pass source detection with a JAX-accelerated backend. Inspired by hst1pass (Anderson 2022, WFC ISR 2022-05; https://ui.adsabs.harvard.edu/abs/2022wfc..rept....5A/abstract).

## Installation

```bash
pip install git+https://github.com/kevinmckinnon/pypass
```

For JAX acceleration (GPU/XLA):

```bash
pip install "pypass[jax] @ git+https://github.com/kevinmckinnon/pypass"
# Then install jaxlib with GPU support per https://jax.readthedocs.io/en/latest/installation.html
```

## Quick Start

### Command-line interface

```bash
pypass --image j9gz04tsq_flc.fits \
       --lib_dir ./lib \
       --n_passes 2 \
       --verbose
```

### Python API

```python
from pypass.io import run_photometry_fits

records, psf_path, gdc_path = run_photometry_fits(
    image_path='j9gz04tsq_flc.fits',
    psf_path=None,          # auto-detected from header
    lib_dir='./lib',        # contains STDPSFs/ and STDGDCs/
    n_passes=2,
    half_width=3,
    fmin_thresh=70.0,
    mag_st_max=28.0,
    verbose=True,
)

from pypass.io import catalog_to_table
table = catalog_to_table(records)
table.write('catalog.fits', overwrite=True)
print(f'{len(records)} stars measured')
```

## Key Features

- **JAX backend**: GPU/XLA-accelerated Newton fitting via `jax.vmap` + `jax.jit`. Falls back automatically to a pure NumPy path when JAX is unavailable or for small catalogues. Controlled by the `PYPASS_BACKEND` environment variable (`auto`/`numpy`/`jax`) or `--backend` CLI flag.
- **Iterative multi-pass detection**: Each pass subtracts well-fit stars from the residual image before searching for new sources, recovering faint neighbours that were blended with brighter stars on the first pass.
- **Per-chip chi2 scaling**: Magnitude-dependent chi2 inflation corrects covariances to account for systematic PSF model errors, giving realistic position and flux uncertainties at all brightness levels.
- **Magnitude-dependent uncertainty calibration**: A three-component noise model (floor + Poisson + background) is fit to the catalogue and used to report per-star systematic floor estimates.

## Requirements

- Python >= 3.10
- numpy
- scipy
- astropy

**Optional** (for GPU/XLA acceleration):
- jax
- jaxlib (with GPU support if desired; see https://jax.readthedocs.io/en/latest/installation.html)

## Status and feedback

pypass has been tested on a range of real HST ACS/WFC datasets, but as with any research software there may be edge cases and bugs that haven't been caught yet. If you run into unexpected behaviour or incorrect results, please open a GitHub issue — all feedback is welcome.

## Development notes

The Python implementation and performance optimizations were developed with assistance from [Claude Code](https://claude.ai/code) (Anthropic).

## Attribution

This package is a Python re-implementation of the `hst1pass` Fortran photometry routine:

> Anderson, J. 2022, "One-Pass HST Photometry with hst1pass", Space Telescope WFC Instrument Science Report 2022-05.
> https://ui.adsabs.harvard.edu/abs/2022wfc..rept....5A/abstract
