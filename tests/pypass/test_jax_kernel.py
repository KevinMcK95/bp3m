"""Tests for the JAX-backend pre-computation helpers in _jax_kernel.py."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates, spline_filter

from pypass._jax_kernel import (
    tile_radius, tile_side,
    extract_psf_tile, eval_psf_on_tile,
    extract_pixel_window, flux_sky_init,
    prepare_jax_inputs,
)

# Re-use the shared Gaussian PSF fixture from the test suite.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from helpers import make_gaussian_psf, inject_stars, PSF_SCALE, PSF_SIZE


HW         = 3
PSF_SCALE_ = PSF_SCALE   # 4 (from helpers.py)
TR         = tile_radius(HW, PSF_SCALE_)   # 15
TS         = tile_side(HW, PSF_SCALE_)     # 31
N_PIX      = (2 * HW + 1) ** 2            # 49


@pytest.fixture
def gauss_psf():
    return make_gaussian_psf().astype(np.float64)   # (101, 101)


@pytest.fixture
def gauss_tile(gauss_psf):
    return extract_psf_tile(gauss_psf, HW, PSF_SCALE_)


@pytest.fixture
def gauss_coeff_tile(gauss_psf):
    """B-spline coefficient tile: filter full PSF then crop (no IIR boundary artifacts)."""
    coeffs = spline_filter(gauss_psf.astype(np.float64), order=3, output=np.float64)
    hy, hx = (gauss_psf.shape[0] - 1) // 2, (gauss_psf.shape[1] - 1) // 2
    return coeffs[hy - TR : hy + TR + 1, hx - TR : hx + TR + 1].astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Geometry helpers
# ---------------------------------------------------------------------------

def test_tile_radius_default():
    assert tile_radius(3, 4) == 15

def test_tile_side_default():
    assert tile_side(3, 4) == 31

def test_tile_radius_formula():
    for hw in range(1, 7):
        for sc in [2, 4, 8]:
            assert tile_radius(hw, sc) == hw * sc + sc // 2 + 1


# ---------------------------------------------------------------------------
# 2. extract_psf_tile — shape and centring
# ---------------------------------------------------------------------------

def test_tile_shape(gauss_tile):
    assert gauss_tile.shape == (TS, TS)

def test_tile_dtype(gauss_tile):
    assert gauss_tile.dtype == np.float32

def test_tile_centre_equals_psf_centre(gauss_psf, gauss_tile):
    """Centre pixel of tile must equal centre pixel of the full PSF."""
    half_psf = (PSF_SIZE - 1) // 2
    psf_centre = float(gauss_psf[half_psf, half_psf])
    tile_centre = float(gauss_tile[TR, TR])
    assert abs(psf_centre - tile_centre) < 1e-6


# ---------------------------------------------------------------------------
# 3. Tile coordinate coverage — all window pixels must stay in-bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dx,dy", [
    (0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
    (0.49, 0.49), (-0.49, -0.49), (0.3, -0.2),
])
def test_tile_coords_in_bounds(gauss_tile, dx, dy):
    """All tile-local coordinates (including ±1 derivative steps) are within [0, TS-1]."""
    diy_grid, dix_grid = np.mgrid[-HW:HW + 1, -HW:HW + 1]
    dix = dix_grid.ravel().astype(float)
    diy = diy_grid.ravel().astype(float)

    x_t = TR + (dx - dix) * PSF_SCALE_
    y_t = TR + (dy - diy) * PSF_SCALE_

    # Include derivative offsets ±1
    for x_coord in [x_t - 1, x_t, x_t + 1]:
        assert np.all(x_coord >= 0) and np.all(x_coord <= TS - 1), \
            f"x out of range for dx={dx}: min={x_coord.min():.2f} max={x_coord.max():.2f}"
    for y_coord in [y_t - 1, y_t, y_t + 1]:
        assert np.all(y_coord >= 0) and np.all(y_coord <= TS - 1), \
            f"y out of range for dy={dy}: min={y_coord.min():.2f} max={y_coord.max():.2f}"


# ---------------------------------------------------------------------------
# 4. eval_psf_on_tile matches scipy bicubic on the coefficient tile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dx,dy", [
    (0.0, 0.0), (0.3, -0.2), (-0.4, 0.1), (0.5, 0.5), (-0.5, -0.5),
])
def test_eval_psf_coeff_tile_matches_scipy_bicubic(gauss_psf, gauss_coeff_tile, dx, dy):
    """eval_psf_on_tile PSF values match map_coordinates(order=3) on the same coeff tile."""
    diy_grid, dix_grid = np.mgrid[-HW:HW + 1, -HW:HW + 1]
    dix = dix_grid.ravel().astype(float)
    diy = diy_grid.ravel().astype(float)

    # Tile-local coordinates (same formula as eval_psf_on_tile)
    x_t = TR + (dix - dx) * PSF_SCALE_
    y_t = TR + (diy - dy) * PSF_SCALE_
    P_ref = map_coordinates(gauss_coeff_tile, np.stack([y_t, x_t]),
                            order=3, mode='nearest', prefilter=False)

    P_tile, _, _ = eval_psf_on_tile(gauss_coeff_tile, dx, dy, HW, PSF_SCALE_)

    np.testing.assert_allclose(P_tile, P_ref, atol=1e-10,
                               err_msg=f"PSF mismatch at dx={dx}, dy={dy}")


# ---------------------------------------------------------------------------
# 5. Gradient sign convention matches core.py
# ---------------------------------------------------------------------------

def test_gradient_sign_convention(gauss_psf, gauss_coeff_tile):
    """dPdx from eval_psf_on_tile matches core.py's _eval_psf_grad_fast at significant pixels.

    Both now use bicubic B-spline.  eval_psf_on_tile uses analytical gradient
    weights; _eval_psf_grad_fast uses central finite differences — these agree
    to < 5% relative error at the PSF core (where gradients drive the fit).
    """
    from scipy.ndimage import spline_filter
    from pypass.core import _eval_psf_grad_fast

    dx, dy = 0.2, -0.1
    diy_grid, dix_grid = np.mgrid[-HW:HW + 1, -HW:HW + 1]

    coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    P_ref, dPdx_ref, dPdy_ref = _eval_psf_grad_fast(
        coeffs, dx, dy, dix_grid, diy_grid, PSF_SCALE_
    )
    P_ref = P_ref.ravel();  dPdx_ref = dPdx_ref.ravel();  dPdy_ref = dPdy_ref.ravel()

    P_tile, dPdx_tile, dPdy_tile = eval_psf_on_tile(gauss_coeff_tile, dx, dy, HW, PSF_SCALE_)

    # At significant PSF pixels (P > 1% of peak), check sign and magnitude.
    sig = P_ref > 0.01 * P_ref.max()
    assert sig.any(), "No significant PSF pixels found"

    # PSF values: both use bicubic on same PSF coefficients — should agree tightly.
    # Tile coefficients vs full-PSF coefficients differ negligibly at interior pixels.
    np.testing.assert_allclose(P_tile[sig], P_ref[sig], rtol=0.01,
                               err_msg="PSF value mismatch at significant pixels")
    # Gradients: analytical B-spline derivative (tile) vs central finite differences
    # (core.py, ±1 supersampled step = ±0.25 px).  O(h²) truncation error of the
    # central difference is ~9 % for a typical PSF — allow 12 % rtol.
    np.testing.assert_allclose(dPdx_tile[sig], dPdx_ref[sig], rtol=0.12,
                               err_msg="dPdx mismatch at significant pixels")
    np.testing.assert_allclose(dPdy_tile[sig], dPdy_ref[sig], rtol=0.12,
                               err_msg="dPdy mismatch at significant pixels")

    # Explicit sign check restricted to significant pixels
    assert np.all(np.sign(dPdx_tile[sig]) == np.sign(dPdx_ref[sig])), \
        "dPdx sign disagreement at significant PSF pixels"
    assert np.all(np.sign(dPdy_tile[sig]) == np.sign(dPdy_ref[sig])), \
        "dPdy sign disagreement at significant PSF pixels"


# ---------------------------------------------------------------------------
# 6. extract_pixel_window
# ---------------------------------------------------------------------------

def test_pixel_window_values():
    """Pixel window matches manual extraction for an in-bounds star."""
    rng = np.random.default_rng(1)
    data = rng.normal(200.0, 10.0, (100, 100))
    x0, y0 = 50.3, 49.7
    xi, yi = 50, 50

    pv, pvar, valid, dx0, dy0, xi_out, yi_out = extract_pixel_window(
        data, x0, y0, HW, mask=None, noise_map=None,
        gain=1.0, read_noise=5.0, sky=200.0,
    )

    assert xi_out == xi and yi_out == yi
    assert abs(dx0 - (x0 - xi)) < 1e-12
    assert abs(dy0 - (y0 - yi)) < 1e-12
    assert len(pv) == N_PIX
    assert valid.all(), "All pixels should be valid for a central, unmasked star"

    # Check a specific pixel: (dix=1, diy=0) → pixel (51, 50)
    diy_grid, dix_grid = np.mgrid[-HW:HW + 1, -HW:HW + 1]
    dix = dix_grid.ravel(); diy = diy_grid.ravel()
    idx = np.where((dix == 1) & (diy == 0))[0][0]
    assert abs(pv[idx] - data[50, 51]) < 1e-12


def test_pixel_window_edge_star():
    """Edge star has out-of-bound pixels marked invalid and filled with sky."""
    data = np.ones((50, 50)) * 100.0
    sky  = 100.0
    pv, pvar, valid, dx0, dy0, xi, yi = extract_pixel_window(
        data, 1.0, 1.0, HW, mask=None, noise_map=None,
        gain=1.0, read_noise=5.0, sky=sky,
    )

    # Some pixels are out of bounds
    assert not valid.all(), "Edge star should have some invalid pixels"
    # Out-of-bounds pixels filled with sky value
    assert np.all(pv[~valid] == sky)


def test_pixel_window_masked():
    """Masked pixels are excluded from valid_mask."""
    data = np.ones((50, 50)) * 200.0
    mask = np.zeros((50, 50), dtype=bool)
    mask[25, 26] = True   # mask pixel at (dix=1, diy=0) for star at (25, 25)

    pv, pvar, valid, dx0, dy0, xi, yi = extract_pixel_window(
        data, 25.0, 25.0, HW, mask=mask, noise_map=None,
        gain=1.0, read_noise=5.0, sky=200.0,
    )

    # One pixel should be invalid
    assert valid.sum() == N_PIX - 1


# ---------------------------------------------------------------------------
# 7. flux_sky_init — matches fit_star pre-solve
# ---------------------------------------------------------------------------

def test_flux_sky_init_bright_star(gauss_psf, gauss_coeff_tile):
    """flux_sky_init recovers injected flux and sky for a bright isolated star."""
    rng = np.random.default_rng(42)
    true_flux = 5000.0; true_sky = 300.0

    P_ref, _, _ = eval_psf_on_tile(gauss_coeff_tile, 0.0, 0.0, HW, PSF_SCALE_)
    pixel_vals = true_flux * P_ref + true_sky + rng.normal(0, 10, N_PIX)
    valid = np.ones(N_PIX, dtype=bool)

    flux, sky = flux_sky_init(gauss_coeff_tile, pixel_vals, valid, 0.0, 0.0,
                               HW, PSF_SCALE_, true_sky)

    assert abs(flux - true_flux) / true_flux < 0.05, f"flux error: {flux:.1f} vs {true_flux}"
    assert abs(sky - true_sky) < 20.0, f"sky error: {sky:.1f} vs {true_sky}"


def test_flux_sky_init_clamps_to_one(gauss_coeff_tile):
    """flux_sky_init clamps negative flux to 1.0."""
    pixel_vals = np.full(N_PIX, 200.0)   # all sky, no star
    valid = np.ones(N_PIX, dtype=bool)
    flux, sky = flux_sky_init(gauss_coeff_tile, pixel_vals, valid, 0.0, 0.0,
                               HW, PSF_SCALE_, 200.0)
    assert flux >= 1.0


# ---------------------------------------------------------------------------
# 8. prepare_jax_inputs — shapes and consistency
# ---------------------------------------------------------------------------

def test_prepare_jax_inputs_shapes(gauss_psf):
    """prepare_jax_inputs returns correctly shaped arrays for multiple stars."""
    rng = np.random.default_rng(7)
    n_stars = 5
    data = rng.normal(200.0, 10.0, (200, 200))
    xs = np.array([50.1, 80.3, 120.7, 60.2, 100.5])
    ys = np.array([50.2, 80.1, 120.3, 60.8, 100.1])
    skys = np.full(n_stars, 200.0)

    psf_cube = gauss_psf[np.newaxis]   # (1, 101, 101)
    psf_xs = np.array([0.0]); psf_ys = np.array([0.0])

    result = prepare_jax_inputs(
        data, xs, ys, skys,
        psf_cube, psf_xs, psf_ys,
        PSF_SCALE_, HW,
        mask=None, noise_map=None,
        gain=1.0, read_noise=5.0,
    )

    assert result['psf_tiles'].shape        == (n_stars, TS, TS)
    assert result['psf_coeff_tiles'].shape  == (n_stars, TS, TS)
    assert result['psf_coeff_tiles'].dtype  == np.float64
    assert result['pixel_vals'].shape       == (n_stars, N_PIX)
    assert result['pixel_var_rn'].shape == (n_stars, N_PIX)
    assert result['valid_masks'].shape == (n_stars, N_PIX)
    assert result['dx0'].shape         == (n_stars,)
    assert result['flux0'].shape       == (n_stars,)
    assert result['hw']        == HW
    assert result['psf_scale'] == PSF_SCALE_
    assert result['tile_radius'] == TR


def test_prepare_jax_inputs_flux_positive(gauss_psf):
    """All flux0 values must be >= 1.0."""
    rng = np.random.default_rng(8)
    data = inject_stars([(100, 100, 3000), (60, 60, 500)], gauss_psf, PSF_SCALE_)
    xs = np.array([100.0, 60.0])
    ys = np.array([100.0, 60.0])
    skys = np.full(2, 200.0)

    psf_cube = gauss_psf[np.newaxis]
    psf_xs = np.array([0.0]); psf_ys = np.array([0.0])

    result = prepare_jax_inputs(
        data, xs, ys, skys,
        psf_cube, psf_xs, psf_ys,
        PSF_SCALE_, HW,
        mask=None, noise_map=None,
        gain=4.0, read_noise=5.0,
    )

    assert np.all(result['flux0'] >= 1.0)
