"""pytest fixtures shared across py1pass tests."""

import numpy as np
import pytest
from helpers import make_gaussian_psf, inject_stars, PSF_SCALE, PSF_SIZE


@pytest.fixture
def gauss_psf():
    return make_gaussian_psf()


@pytest.fixture
def psf_cube(gauss_psf):
    return gauss_psf[np.newaxis]


@pytest.fixture
def psf_positions():
    return (np.array([0.0]), np.array([0.0]))
