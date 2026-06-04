"""
Tests for gaia_cross_match.catalog_matcher.

NOTE: The original test_catalog_matcher.py referenced functions
(rd2x, rd2y, find_offset with old return signature, fit_affine,
match_triangles, match_catalogs) that no longer exist in the
current catalog_matcher API.  These tests have been rewritten
to cover the actual public API.
"""

import numpy as np
import pytest
from gaia_cross_match.catalog_matcher import (
    get_inv_2x2,
    fit_affine_weighted,
    fit_4p_weighted,
    apply_affine,
    find_offset,
    find_scale_and_offset,
    compute_mahalanobis,
    compute_logprob_cost,
)
from gaia_cross_match.miracle_match import rd2x, rd2y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_covs(n, sigma=0.05):
    C = np.zeros((n, 2, 2))
    C[:, 0, 0] = sigma ** 2
    C[:, 1, 1] = sigma ** 2
    return C


# ---------------------------------------------------------------------------
# get_inv_2x2
# ---------------------------------------------------------------------------

def test_get_inv_2x2_identity():
    C = np.array([np.eye(2)])
    inv, det = get_inv_2x2(C)
    np.testing.assert_allclose(inv[0], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(det, [1.0], atol=1e-12)


def test_get_inv_2x2_diagonal():
    C = np.zeros((1, 2, 2))
    C[0, 0, 0] = 4.0
    C[0, 1, 1] = 9.0
    inv, det = get_inv_2x2(C)
    np.testing.assert_allclose(inv[0, 0, 0], 1.0 / 4.0, atol=1e-12)
    np.testing.assert_allclose(inv[0, 1, 1], 1.0 / 9.0, atol=1e-12)
    np.testing.assert_allclose(det, [36.0], atol=1e-12)


# ---------------------------------------------------------------------------
# apply_affine / fit_affine_weighted round-trip
# ---------------------------------------------------------------------------

def test_apply_affine_identity():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    xp, yp = apply_affine(x, y, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(xp, x)
    np.testing.assert_allclose(yp, y)


def test_fit_affine_weighted_roundtrip():
    """Fit should recover a known affine transform on noiseless data."""
    rng = np.random.default_rng(42)
    n = 50
    x_src = rng.uniform(0, 1000, n)
    y_src = rng.uniform(0, 1000, n)

    A_true, B_true, C_true, D_true = 1.01, 0.005, -0.005, 0.99
    xs_o, ys_o, xt_o, yt_o = 500.0, 500.0, 510.0, 490.0
    x_tgt, y_tgt = apply_affine(x_src, y_src, A_true, B_true, C_true, D_true,
                                  xs_o, ys_o, xt_o, yt_o)
    # Add tiny noise
    x_tgt += rng.normal(0, 0.001, n)
    y_tgt += rng.normal(0, 0.001, n)

    cov_src = _identity_covs(n, sigma=0.05)
    cov_tgt = _identity_covs(n, sigma=0.05)

    res, p_err, inv_lhs, chi2 = fit_affine_weighted(
        x_src, y_src, x_tgt, y_tgt, cov_src, cov_tgt)
    A, B, C, D, xs_o_f, ys_o_f, xt_o_f, yt_o_f = res

    np.testing.assert_allclose(A, A_true, atol=0.005)
    np.testing.assert_allclose(B, B_true, atol=0.005)
    np.testing.assert_allclose(C, C_true, atol=0.005)
    np.testing.assert_allclose(D, D_true, atol=0.005)


def test_fit_affine_weighted_insufficient_points():
    """With < 3 points, fit_affine_weighted should return the identity fallback."""
    n = 2
    cov = _identity_covs(n)
    res, _, _, chi2 = fit_affine_weighted(
        np.ones(n), np.ones(n), np.ones(n), np.ones(n), cov, cov)
    assert chi2 == pytest.approx(1e10)


# ---------------------------------------------------------------------------
# fit_4p_weighted
# ---------------------------------------------------------------------------

def test_fit_4p_weighted_pure_translation():
    rng = np.random.default_rng(7)
    n = 40
    x_src = rng.uniform(0, 1000, n)
    y_src = rng.uniform(0, 1000, n)

    dx, dy = 15.3, -22.7
    x_tgt = x_src + dx + rng.normal(0, 0.001, n)
    y_tgt = y_src + dy + rng.normal(0, 0.001, n)

    cov_src = _identity_covs(n, sigma=0.05)
    cov_tgt = _identity_covs(n, sigma=0.05)

    res, _, _, chi2 = fit_4p_weighted(x_src, y_src, x_tgt, y_tgt, cov_src, cov_tgt)
    A, B, C, D, xs_o, ys_o, xt_o, yt_o = res

    # For pure translation: scale ~ 1, rotation ~ 0
    scale = np.sqrt(A * D - B * C)
    np.testing.assert_allclose(scale, 1.0, atol=0.01)


def test_fit_4p_weighted_insufficient_points():
    n = 1
    cov = _identity_covs(n)
    res, _, _, chi2 = fit_4p_weighted(
        np.ones(n), np.ones(n), np.ones(n), np.ones(n), cov, cov)
    assert chi2 == pytest.approx(1e10)


# ---------------------------------------------------------------------------
# find_offset
# ---------------------------------------------------------------------------

def test_find_offset_recovers_translation():
    rng = np.random.default_rng(99)
    n = 200
    x1 = rng.uniform(0, 2000, n)
    y1 = rng.uniform(0, 2000, n)
    m1 = rng.uniform(15, 22, n)

    dx_true, dy_true = 47.0, -31.0
    x2 = x1 + dx_true + rng.normal(0, 0.3, n)
    y2 = y1 + dy_true + rng.normal(0, 0.3, n)
    m2 = m1 + rng.normal(0, 0.1, n)

    peaks = find_offset(x1, y1, m1, x2, y2, m2, max_offset=100, bin_size=2, top_n=1)
    dx_est, dy_est, score = peaks[0]

    assert abs(dx_est - dx_true) < 3.0, f"dx error too large: {dx_est} vs {dx_true}"
    assert abs(dy_est - dy_true) < 3.0, f"dy error too large: {dy_est} vs {dy_true}"


# ---------------------------------------------------------------------------
# compute_mahalanobis / compute_logprob_cost
# ---------------------------------------------------------------------------

def test_compute_mahalanobis_zero():
    n = 5
    C = _identity_covs(n, sigma=1.0)
    m = compute_mahalanobis(np.zeros(n), np.zeros(n), C)
    np.testing.assert_allclose(m, np.zeros(n), atol=1e-12)


def test_compute_mahalanobis_unit():
    """1-sigma displacement along x with identity covariance → mahal = 1."""
    n = 3
    C = _identity_covs(n, sigma=1.0)
    dx = np.ones(n)
    dy = np.zeros(n)
    m = compute_mahalanobis(dx, dy, C)
    np.testing.assert_allclose(m, np.ones(n), atol=1e-10)


def test_compute_logprob_cost_finite():
    n = 4
    C = _identity_covs(n, sigma=0.5)
    dx = np.array([0.1, 0.2, 0.0, 0.3])
    dy = np.array([0.0, 0.1, 0.2, 0.3])
    cost = compute_logprob_cost(dx, dy, C)
    assert np.all(np.isfinite(cost))
    # All costs should be positive (log(det > 0) + mahal >= 0)
    assert np.all(cost > 0)


# ---------------------------------------------------------------------------
# rd2x / rd2y (from miracle_match)
# ---------------------------------------------------------------------------

def test_rd2x_rd2y_small_angle():
    """For small separations the gnomonic projection should be close to linear."""
    r0, d0 = 0.0, 0.0
    r, d = 0.1, 0.1
    x = rd2x(r, d, r0, d0)
    y = rd2y(r, d, r0, d0)
    assert np.isclose(x, 0.1, atol=1e-3), f"rd2x: {x}"
    assert np.isclose(y, 0.1, atol=1e-3), f"rd2y: {y}"


def test_rd2x_zero_at_center():
    r0, d0 = 83.8, -5.4
    assert np.isclose(rd2x(r0, d0, r0, d0), 0.0, atol=1e-12)
    assert np.isclose(rd2y(r0, d0, r0, d0), 0.0, atol=1e-12)
