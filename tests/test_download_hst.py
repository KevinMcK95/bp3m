"""
Unit tests for ghi.download_hst (no network calls).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import pytest

from bp3m.pipeline.download_hst import (
    get_available_psf_gdc_combos,
    find_flc_images,
    _INSTRUMENTS,
    _INST_TO_LIBDIR,
)


# ── get_available_psf_gdc_combos ─────────────────────────────────────────────

def test_get_available_psf_gdc_combos_missing_dir(tmp_path):
    result = get_available_psf_gdc_combos(tmp_path / "nonexistent")
    assert result == {}


def test_get_available_psf_gdc_combos_no_psf_dir(tmp_path):
    (tmp_path / "STDGDCs").mkdir()
    result = get_available_psf_gdc_combos(tmp_path)
    assert result == {}


def test_get_available_psf_gdc_combos_empty_dirs(tmp_path):
    (tmp_path / "STDPSFs").mkdir()
    (tmp_path / "STDGDCs").mkdir()
    result = get_available_psf_gdc_combos(tmp_path)
    assert result == {}


def test_get_available_psf_gdc_combos_synthetic(tmp_path):
    """Create fake PSF and GDC files and verify the function finds the overlap."""
    psf_dir = tmp_path / "STDPSFs" / "ACSWFC"
    gdc_dir = tmp_path / "STDGDCs" / "ACSWFC"
    psf_dir.mkdir(parents=True)
    gdc_dir.mkdir(parents=True)

    # F606W has both; F814W has only PSF; F475W has only GDC
    (psf_dir / "STDPSF_ACSWFC_F606W.fits").touch()
    (psf_dir / "STDPSF_ACSWFC_F814W.fits").touch()
    (psf_dir / "STDPSF_ACSWFC_F814W_SM4.fits").touch()   # variant — not counted separately
    (gdc_dir / "STDGDC_ACSWFC_F606W.fits").touch()
    (gdc_dir / "STDGDC_ACSWFC_F475W.fits").touch()

    result = get_available_psf_gdc_combos(tmp_path)
    assert 'ACS/WFC' in result
    assert 'F606W' in result['ACS/WFC']
    assert 'F814W' not in result['ACS/WFC']
    assert 'F475W' not in result['ACS/WFC']


@pytest.mark.skipif(
    not Path("~/GaiaHub-master/lib").expanduser().exists(),
    reason="GaiaHub-master/lib not present on this machine",
)
def test_get_available_psf_gdc_combos_real():
    lib_dir = Path("~/GaiaHub-master/lib").expanduser()
    result = get_available_psf_gdc_combos(lib_dir)
    assert 'ACS/WFC' in result
    assert 'F606W' in result['ACS/WFC']
    assert 'F814W' in result['ACS/WFC']


# ── _INST_TO_LIBDIR coverage ─────────────────────────────────────────────────

def test_inst_to_libdir_covers_hst_instruments():
    for inst in _INSTRUMENTS['HST']:
        assert inst in _INST_TO_LIBDIR, f"'{inst}' not in _INST_TO_LIBDIR"


# ── find_flc_images ──────────────────────────────────────────────────────────

def test_find_flc_images_empty(tmp_path):
    result = find_flc_images(tmp_path, "no_field")
    assert result == []


def test_find_flc_images_finds_files(tmp_path):
    obs_dir = tmp_path / "MY_FIELD" / "HST" / "mastDownload" / "HST" / "j8abc1"
    obs_dir.mkdir(parents=True)
    fake = obs_dir / "j8abc1_flc.fits"
    fake.touch()

    result = find_flc_images(tmp_path, "MY_FIELD")
    assert len(result) == 1
    assert result[0] == fake


def test_find_flc_images_multiple(tmp_path):
    base = tmp_path / "FIELD" / "HST" / "mastDownload" / "HST"
    for obs_id in ["j001", "j002", "j003"]:
        d = base / obs_id
        d.mkdir(parents=True)
        (d / f"{obs_id}_flc.fits").touch()

    result = find_flc_images(tmp_path, "FIELD")
    assert len(result) == 3
    assert all(p.suffix == '.fits' for p in result)
