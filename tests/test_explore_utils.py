"""
Unit tests for ghi.explore_utils (no file I/O or network access).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from bp3m.pipeline.explore_utils import vpd_limits, sky_extent, bp3m_full_cov


# ── vpd_limits ────────────────────────────────────────────────────────────────

def test_vpd_limits_centered_on_median():
    pmra  = np.array([-5., 0., 5., 0., 0.])
    pmdec = np.array([0., 0., 0., -3., 3.])
    (xlo, xhi), (ylo, yhi) = vpd_limits(pmra, pmdec)
    med_ra  = np.median(pmra)
    med_dec = np.median(pmdec)
    # Centre of xlim and ylim should be the median
    assert (xlo + xhi) / 2 == pytest.approx(med_ra,  abs=1e-10)
    assert (ylo + yhi) / 2 == pytest.approx(med_dec, abs=1e-10)


def test_vpd_limits_symmetric():
    pmra  = np.array([0., 1., -1., 0.5, -0.5])
    pmdec = np.array([0., 0., 0., 0.,   0.])
    (xlo, xhi), (ylo, yhi) = vpd_limits(pmra, pmdec)
    med = np.median(pmra)
    assert abs((xhi - med) - (med - xlo)) < 1e-10


def test_vpd_limits_minimum_spread():
    """Even for identical values, the half-width should be at least n_sigma * 0.1."""
    pmra  = np.zeros(10)
    pmdec = np.zeros(10)
    n_sigma = 4.0
    (xlo, xhi), _ = vpd_limits(pmra, pmdec, n_sigma=n_sigma)
    assert (xhi - xlo) >= 2 * n_sigma * 0.1


def test_vpd_limits_custom_n_sigma():
    pmra  = np.array([0., 2., -2.])
    pmdec = np.array([0., 0.,  0.])
    (xlo4, xhi4), _ = vpd_limits(pmra, pmdec, n_sigma=4.0)
    (xlo8, xhi8), _ = vpd_limits(pmra, pmdec, n_sigma=8.0)
    assert (xhi8 - xlo8) == pytest.approx(2 * (xhi4 - xlo4), rel=1e-6)


# ── sky_extent ────────────────────────────────────────────────────────────────

def test_sky_extent_padding():
    ra  = np.array([10., 12.])
    dec = np.array([-5., -3.])
    (ralo, rahi), (declo, dechi) = sky_extent(ra, dec, pad_frac=0.1)
    # RA is displayed right-to-left so ralo > rahi
    raw_span_ra  = 2.0
    raw_span_dec = 2.0
    pad_ra  = raw_span_ra  * 0.1
    pad_dec = raw_span_dec * 0.1
    assert rahi  == pytest.approx(10. - pad_ra,  abs=1e-10)
    assert ralo  == pytest.approx(12. + pad_ra,  abs=1e-10)
    assert declo == pytest.approx(-5. - pad_dec, abs=1e-10)
    assert dechi == pytest.approx(-3. + pad_dec, abs=1e-10)


def test_sky_extent_ra_decreases():
    """RA axis: xlim[0] > xlim[1] (right-to-left convention)."""
    ra  = np.linspace(10., 12., 100)
    dec = np.zeros(100)
    (ralo, rahi), _ = sky_extent(ra, dec)
    assert ralo > rahi


def test_sky_extent_dec_increases():
    ra  = np.zeros(10)
    dec = np.linspace(-5., 5., 10)
    _, (declo, dechi) = sky_extent(ra, dec)
    assert dechi > declo


# ── bp3m_full_cov ────────────────────────────────────────────────────────────

def test_bp3m_full_cov_sum():
    n = 4
    v_cov = np.ones((n, 5, 5)) * 2.0
    C_vT  = np.ones((n, 5, 5)) * 3.0
    bp3m  = {'v_cov': v_cov, 'C_vT': C_vT}
    result = bp3m_full_cov(bp3m)
    assert result.shape == (n, 5, 5)
    assert np.allclose(result, 5.0)


def test_bp3m_full_cov_identity():
    n = 3
    v_cov = np.zeros((n, 5, 5))
    C_vT  = np.eye(5)[np.newaxis].repeat(n, axis=0)
    bp3m  = {'v_cov': v_cov, 'C_vT': C_vT}
    result = bp3m_full_cov(bp3m)
    assert np.allclose(result, C_vT)
