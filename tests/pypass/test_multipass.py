"""Tests for multi-pass photometry and neighbour subtraction."""

import numpy as np
import pytest
from helpers import make_gaussian_psf, inject_stars, PSF_SCALE

from pypass.core import run_photometry, StarRecord
from pypass.multipass import subtract_stars


@pytest.fixture
def gauss_psf():
    return make_gaussian_psf()


@pytest.fixture
def psf_cube(gauss_psf):
    return gauss_psf[np.newaxis]


@pytest.fixture
def psf_positions():
    return (np.array([0.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# 7. Multi-pass completeness
# ---------------------------------------------------------------------------

def test_multipass_completeness(gauss_psf, psf_cube, psf_positions):
    """Pass 2 should detect a faint neighbour missed or poorly measured in pass 1."""
    sky = 300.0
    bright = (80.0, 80.0, 8000.0)
    faint  = (85.0, 80.0, 2000.0)   # 5 pixels away, 4× fainter
    image = inject_stars([bright, faint], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(160, 160), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    records_2pass = run_photometry(
        data=image, psf_models=psf_cube, psf_positions=(xs, ys),
        psf_scale=PSF_SCALE, half_width=2,
        sky_inner=4, sky_outer=8, hmin=3, fmin=0.0,
        max_iter_fit=10, tol=1e-5, n_passes=2, n_discovery_passes=2,
        gain=4.0, read_noise=5.0, zero_point=0.0,
    )

    # After 2 passes we should have at least both stars
    assert len(records_2pass) >= 2, \
        f"Only {len(records_2pass)} stars found in 2-pass run, expected ≥ 2"

    # At least one record should be close to each injected star
    xs_fit = np.array([r.x for r in records_2pass])
    ys_fit = np.array([r.y for r in records_2pass])

    for x0, y0, _ in [bright, faint]:
        dists = np.hypot(xs_fit - x0, ys_fit - y0)
        assert dists.min() < 1.5, \
            f"No star within 1.5 px of injected position ({x0}, {y0})"


# ---------------------------------------------------------------------------
# 8. Neighbour subtraction
# ---------------------------------------------------------------------------

def test_neighbour_subtraction(gauss_psf, psf_cube, psf_positions):
    """After subtracting a bright star, residual at its position should be < 3σ sky."""
    sky = 300.0
    x0, y0, flux = 64.0, 64.0, 20000.0
    image = inject_stars([(x0, y0, flux)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    records = run_photometry(
        data=image, psf_models=psf_cube, psf_positions=(xs, ys),
        psf_scale=PSF_SCALE, half_width=2, sky_inner=4, sky_outer=8,
        hmin=3, fmin=0.0, n_passes=1, gain=4.0, read_noise=5.0,
    )

    residual = image.copy()
    subtract_stars(residual, records, psf_cube, xs, ys,
                   PSF_SCALE, hw=2, x_offset=0.0, y_offset=0.0)

    # Sky sigma at the subtracted position
    from pypass.core import estimate_sky
    sky_est, sky_sig = estimate_sky(residual, int(round(x0)), int(round(y0)),
                                    sky_inner=4, sky_outer=8)
    peak_residual = abs(residual[int(round(y0)), int(round(x0))] - sky_est)
    assert peak_residual < 3.0 * sky_sig, \
        f"Residual peak {peak_residual:.2f} > 3σ ({3*sky_sig:.2f})"
