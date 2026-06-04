"""Shared fixtures for py1pass tests."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pypass.core import eval_psf_and_grad


PSF_SCALE = 4
PSF_SIZE  = 101


def make_gaussian_psf(fwhm_pix=1.8, psf_scale=PSF_SCALE, size=PSF_SIZE):
    """Supersampled Gaussian PSF normalised so sum/psf_scale² ≈ 1."""
    sigma_ss = fwhm_pix * psf_scale / 2.3548
    c = size // 2
    y, x = np.mgrid[:size, :size]
    r2 = (x - c) ** 2 + (y - c) ** 2
    psf = np.exp(-0.5 * r2 / sigma_ss**2)
    psf /= psf.sum() / psf_scale**2
    return psf


def inject_stars(stars, psf, psf_scale, sky=200.0, image_size=(256, 256),
                 gain=4.0, read_noise=5.0, seed=42):
    """Return noisy synthetic image with stars injected at known (x, y, flux).

    stars : list of (x, y, flux)
    """
    ny, nx = image_size
    image = np.full((ny, nx), sky, dtype=np.float64)
    hw = 12  # large injection window

    for x0, y0, flux in stars:
        xi = int(round(x0))
        yi = int(round(y0))
        dx = x0 - xi
        dy = y0 - yi
        y_lo = max(0, yi - hw);  y_hi = min(ny, yi + hw + 1)
        x_lo = max(0, xi - hw);  x_hi = min(nx, xi + hw + 1)
        diy = (np.arange(y_lo, y_hi) - yi)[:, np.newaxis]
        dix = (np.arange(x_lo, x_hi) - xi)[np.newaxis, :]
        P, _, _ = eval_psf_and_grad(psf, dx, dy, dix, diy, psf_scale)
        image[y_lo:y_hi, x_lo:x_hi] += flux * P

    rng = np.random.default_rng(seed)
    # Poisson noise
    image_pos = np.maximum(image, 0.0)
    noisy = rng.poisson(image_pos).astype(np.float64)
    # Read noise
    noisy += rng.normal(0.0, read_noise, size=(ny, nx))
    return noisy


@pytest.fixture
def gauss_psf():
    return make_gaussian_psf()


@pytest.fixture
def psf_cube(gauss_psf):
    return gauss_psf[np.newaxis]  # single PSF, shape (1, 101, 101)


@pytest.fixture
def psf_positions():
    return (np.array([0.0]), np.array([0.0]))
