"""Tests for STDPSF loading and catalog output."""

import numpy as np
import pytest
import os

from pypass.io import load_stdpsf, catalog_to_table
from pypass.core import StarRecord


PSF_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../lib/STDPSFs/ACSWFC/STDPSF_ACSWFC_F814W_SM4.fits'
)


# ---------------------------------------------------------------------------
# 9. Catalogue columns and dtypes
# ---------------------------------------------------------------------------

def test_catalog_columns():
    """catalog_to_table produces all expected columns with correct dtypes."""
    cov = np.eye(4) * 0.01
    records = [
        StarRecord(x=10.1, y=20.2, flux=1000.0, flux_err=10.0,
                   sky=200.0, sky_err=2.0, mag=15.0, mag_err=0.01,
                   qfit=0.05, chi2=1.1, central_res=0.02,
                   n_sat=0, psf_frac=0.27, psf_peak=0.28,
                   peak=500.0, cov=cov, pass_number=1,
                   n_neighbors=1, dist_nearest=40.2, dist_nearest_brighter=np.inf),
        StarRecord(x=50.3, y=30.4, flux=500.0, flux_err=8.0,
                   sky=200.0, sky_err=2.0, mag=15.75, mag_err=0.016,
                   qfit=0.07, chi2=1.2, central_res=-0.01,
                   n_sat=0, psf_frac=0.26, psf_peak=0.28,
                   peak=250.0, cov=cov, pass_number=2,
                   n_neighbors=1, dist_nearest=40.2, dist_nearest_brighter=40.2),
    ]

    table = catalog_to_table(records, zero_point=25.0)

    required = ['x', 'y', 'flux', 'flux_err', 'sky', 'sky_err', 'mag', 'mag_err',
                'qfit', 'chi2', 'central_res', 'n_sat', 'psf_frac', 'psf_peak',
                'peak', 'pass_number',
                'n_neighbors', 'dist_nearest', 'dist_nearest_brighter',
                'cov_ff', 'cov_xx', 'cov_yy', 'cov_ss',
                'cov_fx', 'cov_fy', 'cov_fs', 'cov_xy', 'cov_xs', 'cov_ys',
                'n_iter', 'converged', 'delta_max', 'chi2_scale', 'eps_psf',
                'concentration', 'is_star_candidate',
                'sigma_x_model', 'sigma_y_model', 'sigma_f_model',
                'chip_ext', 'x_gdc', 'y_gdc', 'mag_gdc', 'mag_err_gdc',
                'mag_st', 'mag_ab', 'mag_st_gdc']
    for col in required:
        assert col in table.colnames, f"Missing column: {col}"

    assert table['pass_number'].dtype.kind in ('i', 'u'), "pass_number should be integer"
    assert len(table) == 2
    assert table.meta.get('ZP') == 25.0


def test_catalog_empty():
    """catalog_to_table handles an empty record list."""
    table = catalog_to_table([], zero_point=0.0)
    assert len(table) == 0
    for col in ['x', 'y', 'flux', 'pass_number']:
        assert col in table.colnames


# ---------------------------------------------------------------------------
# STDPSF loading
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(PSF_PATH), reason="PSF file not available")
def test_load_stdpsf_shape():
    """STDPSF loader returns correct cube shape and grid dimensions."""
    psf_cube, xs, ys, psf_scale, grid_shape = load_stdpsf(PSF_PATH)
    ny_g, nx_g = grid_shape

    assert psf_cube.ndim == 3
    assert psf_cube.shape[0] == ny_g * nx_g
    assert len(xs) == nx_g
    assert len(ys) == ny_g
    assert psf_scale == 4


@pytest.mark.skipif(not os.path.exists(PSF_PATH), reason="PSF file not available")
def test_load_stdpsf_normalisation():
    """Each PSF in the cube should be normalised: sum ≈ psf_scale²."""
    psf_cube, xs, ys, psf_scale, grid_shape = load_stdpsf(PSF_PATH)
    for i in range(psf_cube.shape[0]):
        s = psf_cube[i].sum() / psf_scale**2
        assert abs(s - 1.0) < 0.05, f"PSF[{i}] norm = {s:.4f}"
