"""
Canonical on-disk layout, shared by every instrument and every pipeline stage.

Single source of truth for output paths.  The cross-match and the alignment both
import from here, so their directory names cannot drift apart — which is exactly
what happened before (`Gaia_xmatch/cfht_X/xmatch/` and bare `bp3m_results/cfht_X/05/`
for CFHT, `xmatch_results/visit_X/detector_147_i/` for LSST).

Layout
------
    <field_root>/
        Gaia/                             Gaia catalogue(s) for the field
            *_gaia.csv
        xmatch/
            <exposure_id>/                cfht_2159742  |  lsst_2025121400828
                <det_token>/              det_05        |  det_147_i
                    matched_gaia.csv
                    transformation.csv
                    processing_log.txt
                    plots/
                all_matched_gaia.csv      exposure-level concatenation
                all_transformations.csv
            all_exposures_summary.csv
        align/
            <exposure_id>/
                <det_token>/
                    stellar_astrometry.csv
                    image_transformations.csv
                    plots/
            joint_<label>/                solves spanning multiple exposures
                stellar_astrometry.csv
                image_transformations.csv
                plots/

Naming rules
------------
    exposure_id : '<instrument>_<native exposure id>', lowercase.
    det_token   : 'det_<num>' with an optional '_<band>' suffix.  Always the
                  literal prefix 'det_' — never 'detector_', never a bare number.
"""

from __future__ import annotations

from pathlib import Path

GAIA_DIR = 'Gaia'
XMATCH_DIR = 'xmatch'
ALIGN_DIR = 'align'
PLOTS_DIR = 'plots'

MATCHED_CSV = 'matched_gaia.csv'
TRANSFORM_CSV = 'transformation.csv'
LOG_TXT = 'processing_log.txt'

ALL_MATCHED_CSV = 'all_matched_gaia.csv'
ALL_TRANSFORM_CSV = 'all_transformations.csv'
SUMMARY_CSV = 'all_exposures_summary.csv'

STELLAR_CSV = 'stellar_astrometry.csv'
IMAGE_TRANSFORM_CSV = 'image_transformations.csv'


def exposure_id(instrument: str, exposure: object) -> str:
    """
    Canonical exposure directory name.

    >>> exposure_id('cfht', 2159742)
    'cfht_2159742'
    >>> exposure_id('lsst', 2025121400828)
    'lsst_2025121400828'
    """
    return f'{instrument.lower()}_{exposure}'


def det_token(detector: object, band: str | None = None, width: int = 0) -> str:
    """
    Canonical detector directory name.

    Always prefixed 'det_'.  `width` zero-pads the number so directories sort
    naturally (CFHT uses 2, LSST uses 3).

    >>> det_token(5, width=2)
    'det_05'
    >>> det_token(147, 'i', width=3)
    'det_147_i'
    """
    if isinstance(detector, (int, float)) and width:
        core = f'det_{int(detector):0{width}d}'
    else:
        core = f'det_{detector}'
    return f'{core}_{band}' if band else core


def image_id(instrument: str, exposure: object, detector: object,
             band: str | None = None, width: int = 0) -> str:
    """Flat, unique, filesystem-safe identifier for one image."""
    return f'{exposure_id(instrument, exposure)}_{det_token(detector, band, width)}'


# ── Roots ────────────────────────────────────────────────────────────────────

def gaia_dir(field_root: Path) -> Path:
    return Path(field_root) / GAIA_DIR


def xmatch_root(field_root: Path) -> Path:
    return Path(field_root) / XMATCH_DIR


def align_root(field_root: Path) -> Path:
    return Path(field_root) / ALIGN_DIR


# ── Per-image directories ────────────────────────────────────────────────────

def xmatch_dir(field_root: Path, exp_id: str, det: str) -> Path:
    return xmatch_root(field_root) / exp_id / det


def align_dir(field_root: Path, exp_id: str, det: str) -> Path:
    return align_root(field_root) / exp_id / det


def joint_align_dir(field_root: Path, label: str = 'all') -> Path:
    """Output directory for a solve spanning multiple exposures/detectors."""
    return align_root(field_root) / f'joint_{label}'


def plots_dir(base: Path) -> Path:
    return Path(base) / PLOTS_DIR


__all__ = [
    'exposure_id', 'det_token', 'image_id',
    'gaia_dir', 'xmatch_root', 'align_root',
    'xmatch_dir', 'align_dir', 'joint_align_dir', 'plots_dir',
    'GAIA_DIR', 'XMATCH_DIR', 'ALIGN_DIR', 'PLOTS_DIR',
    'MATCHED_CSV', 'TRANSFORM_CSV', 'LOG_TXT',
    'ALL_MATCHED_CSV', 'ALL_TRANSFORM_CSV', 'SUMMARY_CSV',
    'STELLAR_CSV', 'IMAGE_TRANSFORM_CSV',
]
