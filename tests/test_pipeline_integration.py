"""
Integration tests for CLI argument parsing in bp3m_run.py.
These tests exercise _parse_args() without running the pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

# Patch sys.argv and import the parser
import importlib


def _parse(argv):
    """Call _parse_args() with a custom argv list."""
    old = sys.argv
    sys.argv = ['bp3m_run'] + argv
    try:
        import bp3m_run as ghi
        # Re-import to pick up fresh parse each call
        importlib.reload(ghi)
        return ghi._parse_args()
    finally:
        sys.argv = old


REQUIRED = ['--name', 'TestField']


def test_parse_args_defaults():
    args = _parse(REQUIRED)
    assert args.name == 'TestField'
    from pathlib import Path
    assert args.lib_dir == str(Path.home() / 'GaiaHub-master' / 'lib')
    assert args.min_gmag == 16.0
    assert args.max_gmag is None        # no faint limit by default
    assert args.time_baseline is None   # no baseline limit by default
    assert args.hst_im_type == '_flc'
    assert args.telescope == 'HST'
    assert args.n_bp3m_iter == 20
    assert args.poly_order == 1


def test_parse_args_lib_dir_override():
    args = _parse(REQUIRED + ['--lib_dir', '/custom/lib'])
    assert args.lib_dir == '/custom/lib'


def test_parse_args_name_target():
    args = _parse(REQUIRED)
    assert args.name == 'TestField'
    assert args.ra is None
    assert args.dec is None


def test_parse_args_ra_dec():
    args = _parse(['--ra', '15.0', '--dec', '-33.7', '--search_radius', '0.3'])
    assert args.ra == pytest.approx(15.0)
    assert args.dec == pytest.approx(-33.7)
    assert args.search_radius == pytest.approx(0.3)


def test_parse_args_skip_flags():
    args = _parse(REQUIRED + [
        '--skip_download', '--skip_psf', '--skip_crossmatch', '--skip_alignment'])
    assert args.skip_download is True
    assert args.skip_psf is True
    assert args.skip_crossmatch is True
    assert args.skip_alignment is True


def test_parse_args_force_redownload():
    args = _parse(REQUIRED + ['--force_redownload_gaia'])
    assert args.force_redownload_gaia is True
    assert args.force_redownload_hst is False

    args2 = _parse(REQUIRED + ['--force_redownload_hst'])
    assert args2.force_redownload_hst is True
    assert args2.force_redownload_gaia is False

    args3 = _parse(REQUIRED + ['--force_redownload_gaia', '--force_redownload_hst'])
    assert args3.force_redownload_gaia is True
    assert args3.force_redownload_hst is True


def test_parse_args_obs_date():
    args = _parse(REQUIRED + ['--obs_date_min', '2005-01-01',
                               '--obs_date_max', '2020-12-31'])
    assert args.obs_date_min == '2005-01-01'
    assert args.obs_date_max == '2020-12-31'


def test_parse_args_obs_date_defaults():
    args = _parse(REQUIRED)
    assert args.obs_date_min is None
    assert args.obs_date_max is None


def test_parse_args_instruments():
    args = _parse(REQUIRED + ['--instruments', 'ACS/WFC', 'WFC3/UVIS'])
    assert 'ACS/WFC' in args.instruments
    assert 'WFC3/UVIS' in args.instruments


def test_parse_args_instruments_default():
    args = _parse(REQUIRED)
    assert args.instruments is None


def test_parse_args_bp3m_options():
    args = _parse(REQUIRED + [
        '--n_bp3m_iter', '30',
        '--bp3m_clip_sigma', '3.5',
        '--poly_order', '2',
        '--sparse',
    ])
    assert args.n_bp3m_iter == 30
    assert args.bp3m_clip_sigma == pytest.approx(3.5)
    assert args.poly_order == 2
    assert args.sparse is True
    # split_ccd and inflate_hst_errors are on by default; opt-out flags default False
    assert args.no_split_ccd is False
    assert args.no_inflate_hst_errors is False


def test_parse_args_bp3m_opt_out_flags():
    args = _parse(REQUIRED + ['--no_split_ccd', '--no_inflate_hst_errors'])
    assert args.no_split_ccd is True
    assert args.no_inflate_hst_errors is True


def test_parse_args_gaia_options():
    args = _parse(REQUIRED + [
        '--min_gmag', '17.0',
        '--max_gmag', '22.0',
        '--only_5p',
    ])
    assert args.min_gmag == pytest.approx(17.0)
    assert args.max_gmag == pytest.approx(22.0)
    assert args.only_5p is True
