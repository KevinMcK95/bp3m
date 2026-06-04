"""
Unit tests for ghi.catalog_utils.

These tests exercise covariance construction, inflation factors, and uncertainty
helper functions without any file I/O or network access.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from bp3m.pipeline.catalog_utils import (
    GAIA_SYS, build_gaia_cov, cov2_geom_sigma,
    pm_uncertainty, pos_uncertainty,
)


def _fake_gaia_df(n=3, solution_type='6p'):
    """
    Build a minimal Gaia DataFrame with all columns required by build_gaia_cov.

    build_gaia_cov detects solution type via:
      6p: pseudocolour is finite
      5p: pmra is finite and pseudocolour is NaN
      2p: both NaN
    """
    rng = np.random.default_rng(42)

    data = {
        'ra_error':              rng.uniform(0.01, 0.5, n),
        'dec_error':             rng.uniform(0.01, 0.5, n),
        'pmra_error':            rng.uniform(0.02, 1.0, n),
        'pmdec_error':           rng.uniform(0.02, 1.0, n),
        'parallax_error':        rng.uniform(0.02, 0.5, n),
        'ra_dec_corr':           rng.uniform(-0.5, 0.5, n),
        'ra_pmra_corr':          rng.uniform(-0.5, 0.5, n),
        'ra_pmdec_corr':         rng.uniform(-0.5, 0.5, n),
        'ra_parallax_corr':      rng.uniform(-0.5, 0.5, n),
        'dec_pmra_corr':         rng.uniform(-0.5, 0.5, n),
        'dec_pmdec_corr':        rng.uniform(-0.5, 0.5, n),
        'dec_parallax_corr':     rng.uniform(-0.5, 0.5, n),
        'pmra_pmdec_corr':       rng.uniform(-0.5, 0.5, n),
        'parallax_pmra_corr':    rng.uniform(-0.5, 0.5, n),
        'parallax_pmdec_corr':   rng.uniform(-0.5, 0.5, n),
        'pmra':                  rng.uniform(-5.0, 5.0, n),
    }

    if solution_type == '6p':
        data['pseudocolour'] = rng.uniform(1.5, 2.5, n)  # finite → 6p
    elif solution_type == '5p':
        data['pseudocolour'] = np.full(n, np.nan)         # NaN + pmra finite → 5p
    else:
        data['pseudocolour'] = np.full(n, np.nan)
        data['pmra']         = np.full(n, np.nan)         # both NaN → 2p

    return pd.DataFrame(data)


# ── GAIA_SYS ─────────────────────────────────────────────────────────────────

def test_gaia_sys_values():
    assert GAIA_SYS['mult_6p'] == pytest.approx(1.22)
    assert GAIA_SYS['mult_5p'] == pytest.approx(1.05)
    assert GAIA_SYS['mult_2p'] == pytest.approx(1.00)
    assert GAIA_SYS['parallax_sys_err'] == pytest.approx(0.011)
    assert GAIA_SYS['pm_sys_err'] == pytest.approx(0.026)


# ── build_gaia_cov ───────────────────────────────────────────────────────────

def test_build_gaia_cov_shape():
    df = _fake_gaia_df(n=5)
    C = build_gaia_cov(df)
    assert C.shape == (5, 5, 5)


def test_build_gaia_cov_symmetric():
    df = _fake_gaia_df(n=4)
    C = build_gaia_cov(df)
    for i in range(4):
        assert np.allclose(C[i], C[i].T, atol=1e-12)


def test_inflation_6p():
    """
    For a 6p star with pmra_error=2.0, the inflated PM variance should be
    sigma² * mult_6p + pm_sys_err² = 4.0 * 1.22 + 0.026² = 4.880676.
    (The floor adds pm_sys_err² as a systematic component, not a minimum.)
    """
    df = _fake_gaia_df(n=1, solution_type='6p')
    df['pmra_error'] = 2.0
    C = build_gaia_cov(df)
    pm_floor = GAIA_SYS['pm_sys_err'] ** 2
    expected = 4.0 * GAIA_SYS['mult_6p'] + pm_floor
    assert C[0, 2, 2] == pytest.approx(expected, rel=1e-6)


def test_inflation_5p():
    df = _fake_gaia_df(n=1, solution_type='5p')
    df['pmra_error'] = 2.0
    C = build_gaia_cov(df)
    pm_floor = GAIA_SYS['pm_sys_err'] ** 2
    expected = 4.0 * GAIA_SYS['mult_5p'] + pm_floor
    assert C[0, 2, 2] == pytest.approx(expected, rel=1e-6)


def test_parallax_floor():
    """With near-zero parallax_error the floor dominates: C[4,4] ≈ parallax_sys_err²."""
    df = _fake_gaia_df(n=1, solution_type='6p')
    df['parallax_error'] = 1e-10
    C = build_gaia_cov(df)
    plx_floor = GAIA_SYS['parallax_sys_err'] ** 2
    assert C[0, 4, 4] == pytest.approx(plx_floor, rel=1e-4)


def test_pm_floor():
    """With near-zero pmra_error the floor dominates: C[2,2] ≈ pm_sys_err²."""
    df = _fake_gaia_df(n=1, solution_type='6p')
    df['pmra_error'] = 1e-10
    df['pmdec_error'] = 1e-10
    C = build_gaia_cov(df)
    pm_floor = GAIA_SYS['pm_sys_err'] ** 2
    assert C[0, 2, 2] == pytest.approx(pm_floor, rel=1e-4)
    assert C[0, 3, 3] == pytest.approx(pm_floor, rel=1e-4)


# ── cov2_geom_sigma ──────────────────────────────────────────────────────────

def test_cov2_geom_sigma_isotropic():
    """2×2 identity → det^(1/4) = 1.0."""
    C = np.eye(2)[np.newaxis]   # (1, 2, 2)
    sig = cov2_geom_sigma(C)
    assert sig == pytest.approx(1.0)


def test_cov2_geom_sigma_scaled():
    """Scaling the identity by s² → det^(1/4) = s."""
    s = 3.0
    C = (s**2 * np.eye(2))[np.newaxis]
    sig = cov2_geom_sigma(C)
    assert sig == pytest.approx(s, rel=1e-6)


def test_cov2_geom_sigma_batch():
    n = 7
    C = np.eye(2)[np.newaxis].repeat(n, axis=0)
    sig = cov2_geom_sigma(C)
    assert sig.shape == (n,)
    assert np.allclose(sig, 1.0)


# ── pm_uncertainty / pos_uncertainty ────────────────────────────────────────

def test_pm_uncertainty_shape():
    C = np.eye(5)[np.newaxis].repeat(6, axis=0)
    sig = pm_uncertainty(C)
    assert sig.shape == (6,)


def test_pos_uncertainty_shape():
    C = np.eye(5)[np.newaxis].repeat(6, axis=0)
    sig = pos_uncertainty(C)
    assert sig.shape == (6,)


def test_pm_uncertainty_isotropic():
    """For a 5×5 identity the PM block is [[1,0],[0,1]] → det^(1/4) = 1."""
    C = np.eye(5)[np.newaxis]
    assert pm_uncertainty(C) == pytest.approx(1.0)


def test_pos_uncertainty_isotropic():
    C = np.eye(5)[np.newaxis]
    assert pos_uncertainty(C) == pytest.approx(1.0)
