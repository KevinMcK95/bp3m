# bp3m

bp3m is a Python pipeline for measuring proper motions of stars using HST imaging and Gaia astrometry. It implements and extends the Bayesian proper motion method of McKinnon et al. (2024, ApJ 972 150), replacing the original MCMC posterior with a closed-form Gaussian solution that is analytically exact and fast enough to simultaneously fit thousands of stars across >100 HST images. The pipeline follows the science workflow of GaiaHub (del Pino et al. 2022, ApJ 933 76) and uses pypass, a Python implementation of the hst1pass photometry algorithm (Anderson 2022, WFC ISR 2022-05).

## Installation

```bash
pip install git+https://github.com/kevinmckinnon/bp3m
```

For the full environment including PyMC (required for the Bayesian solver):

```bash
conda env create -f environment.yml
conda activate bp3m
pip install -e .
```

## Setup

After installation, run the setup command to download the required HST PSF and geometric distortion correction (GDC) library files from STScI:

```bash
bp3m-setup
```

## Quick start

```bash
bp3m --name "Leo I" --search_radius 0.1 --output_dir ./outputs
```

## Key features

- Closed-form Gaussian posterior (not MCMC) — exact and scales to thousands of stars across >100 images
- Full Python pipeline from HST download through proper motion measurement
- Iterative multi-pass PSF photometry with JAX acceleration (via pypass)
- Robust Gaia cross-matching with affine transformation (via gaia_cross_match)
- Magnitude-dependent chi2 uncertainty calibration
- Diagnostic plots at every pipeline stage

## Pipeline steps

1. **Download Gaia** — query Gaia DR3 via TAP and cache the result
2. **Download HST** — search MAST and download FLC/FLT images
3. **PSF fitting** — run iterative PSF photometry on each image (pypass)
4. **Cross-match** — match each HST catalog to Gaia with an affine transformation (gaia_cross_match)
5. **Bayesian alignment** — simultaneously solve for image transformations and stellar proper motions/parallaxes using the closed-form BP3M algorithm

## Status and feedback

bp3m has been tested on a range of stellar fields across multiple HST instruments and epochs, but as with any research software there may be edge cases and bugs that haven't been caught yet. If you run into unexpected behaviour or incorrect results, please open a GitHub issue — all feedback is welcome.

## Development notes

Code optimization, the Python translation of supporting routines, and pipeline development were assisted by [Claude Code](https://claude.ai/code) (Anthropic).

## References

- McKinnon et al. 2024, ApJ 972 150 — https://ui.adsabs.harvard.edu/abs/2024ApJ...972..150M/abstract
- del Pino et al. 2022, ApJ 933 76 (GaiaHub) — https://ui.adsabs.harvard.edu/abs/2022ApJ...933...76D/abstract
- Anderson 2022, WFC ISR 2022-05 (hst1pass) — https://ui.adsabs.harvard.edu/abs/2022wfc..rept....5A/abstract
