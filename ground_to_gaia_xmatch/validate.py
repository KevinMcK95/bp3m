"""
Cross-image validation of ground-based cross-matches.

Runs the SAME validator bp3m runs after its cross-match —
``gaia_cross_match.validator.validate_target`` — rather than reimplementing it.
That matters because the validator is what produces the ``is_trustworthy`` flag
and ``cross_match_catalog.csv`` that everything downstream keys on; a parallel
implementation would drift.

What it does (bp3m's logic, unchanged): group images by filter, compute pairwise
zero-points from per-star magnitude differences between overlapping images,
split each filter into connected components via the ZP graph, propagate ZPs from
each component's reference image by BFS, then write per-image
``source_quality.csv`` with ``is_trustworthy``, a field-level
``magnitude_zp_offsets.csv``, ``cross_match_catalog.csv``, and the CMD /
colour-colour plots.

How it is adapted
-----------------
Only image DISCOVERY is replaced.  ``validator.find_processed_images`` walks
``<data_dir>/<target>/HST`` looking for ``<name>_flc_catalog.fits``, and
``load_image_data`` opens ``*_flc.fits`` to read EXPTIME / filter / INSTRUME /
DETECTOR / CRVAL from the headers — none of which exists for LSST or CFHT.  So
``find_processed_images`` is monkeypatched to build the identical dict from our
layout, and ``validate_target`` is then called untouched.

Everything else lines up already: ``build_global_catalog`` and
``plot_photometry_catalog`` take ``(target, data_dir)`` and write to
``data_dir/target/``, and our ``field_root`` *is* ``data_dir/target``.

Column and unit mapping, all verified against the validator's own code rather
than assumed:

===========================  ==========================  ====================
validator expects            ours                        note
===========================  ==========================  ====================
``hst_mag_st_gdc``           ``src_mag``                 ZPs are median per-star
                                                         differences, so any
                                                         consistent per-image
                                                         magnitude scale works
``hst_mag_err_gdc``          ``src_magerr``              optional; the validator
                                                         checks for it
``hst_index``                ``src_index``               join key for DELVE
``hst_is_star``              ``src_is_star``
``filter``                   ``band``
``crval1`` / ``crval2``      ``ra0`` / ``dec0``          tangent point
``dec_cen``                  ``dec0``
``pixel_scale``              ``pixel_scale`` / 1000      OURS IS mas/px, the
                                                         validator's is
                                                         arcsec/px (HST writes
                                                         0.05).  wcs_offset_px
                                                         divides arcsec by it,
                                                         so an unconverted
                                                         199.7 would under-report
                                                         the offset 1000x.
``exptime``                  1.0                         stored but never used in
                                                         the ZP maths, so our
                                                         already-calibrated
                                                         magnitudes need no
                                                         un-normalising
===========================  ==========================  ====================

``camera`` is the instrument, not instrument/detector.  In HST it is
INSTRUME/DETECTOR — ACS/WFC, a whole camera, not one chip — and it is used only
to widen the magnitude tolerance when comparing images from different
photometric systems.  Every LSST detector in a band shares one calibrated
system, so keying on the detector would mark nearly every pair "cross-camera"
and loosen the tolerance for no reason.  Spatial separation is already handled:
``validate_target`` groups by filter alone and derives overlap from the pairwise
ZP graph, so images that share no sky fall into separate components by
construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import layout

# validator column  <-  our column
_RENAME = {
    'src_mag': 'hst_mag_st_gdc',
    'src_magerr': 'hst_mag_err_gdc',
    'src_index': 'hst_index',
    'src_is_star': 'hst_is_star',
}


def _load_one(image_dir: Path, image_id: str) -> dict | None:
    """One image in the shape validator.load_image_data returns."""
    mpath = image_dir / layout.MATCHED_CSV
    tpath = image_dir / layout.TRANSFORM_CSV
    if not (mpath.exists() and tpath.exists()):
        return None

    matched = pd.read_csv(mpath)
    for src, dst in _RENAME.items():
        if src in matched.columns and dst not in matched.columns:
            matched[dst] = matched[src]
    if 'hst_mag_st_gdc' not in matched.columns:
        return None

    tr = pd.read_csv(tpath, index_col='parameter')['value']

    def _f(key, default=np.nan):
        try:
            return float(tr[key])
        except Exception:
            return default

    band = str(tr.get('band', '')) if 'band' in tr.index else ''
    instrument = str(tr.get('instrument', '')) if 'instrument' in tr.index else ''
    # See the module docstring: camera is the instrument, not the detector.
    camera = instrument or 'ground'
    ra0, dec0 = _f('ra0'), _f('dec0')

    return {
        'image_name':    image_id,
        'image_dir':     str(image_dir),
        'matched':       matched,
        'transform':     tr,
        'exptime':       1.0,
        'filter':        band,
        'instrume':      instrument,
        'detector':      str(tr.get('detector', '')) if 'detector' in tr.index else '',
        'camera':        camera,
        'filter_camera': f'{band}/{camera}',
        'has_stmag':     bool(matched['hst_mag_st_gdc'].notna().any()),
        'zp':            _f('zp', 0.0),
        'crval1':        ra0,
        'crval2':        dec0,
        # mas/px -> arcsec/px
        'pixel_scale':   _f('pixel_scale', 200.0) / 1000.0,
        'dec_cen':       dec0,
    }


def collect_images(field_root, metas=None) -> dict:
    """Every cross-matched image in a field, keyed by image_id."""
    field_root = Path(field_root)
    root = layout.xmatch_root(field_root)
    images = {}
    if metas is not None:
        for meta in metas:
            d = _load_one(root / meta.rel_dir(), meta.image_id)
            if d is not None:
                images[meta.image_id] = d
        return images
    # No instrument handy: walk the layout instead.
    for mpath in sorted(root.rglob(layout.MATCHED_CSV)):
        image_dir = mpath.parent
        if image_dir.name == layout.PLOTS_DIR:
            continue
        tr_path = image_dir / layout.TRANSFORM_CSV
        image_id = image_dir.name
        if tr_path.exists():
            try:
                tr = pd.read_csv(tr_path, index_col='parameter')['value']
                if 'image_id' in tr.index:
                    image_id = str(tr['image_id'])
            except Exception:
                pass
        d = _load_one(image_dir, image_id)
        if d is not None:
            images[image_id] = d
    return images


def validate_field(field_root, metas=None, mag_scatter_thr: float = 0.1,
                   offset_tol_mag: float = 0.05, offset_tol_px: float = 10.0,
                   quiet: bool = False):
    """
    Run bp3m's validate_target over a ground field.

    Returns the path to cross_match_catalog.csv, or None when there was nothing
    to validate.
    """
    field_root = Path(field_root).resolve()
    images = collect_images(field_root, metas)
    if not images:
        if not quiet:
            print('  validate: no cross-matched images found — skipping')
        return None
    if not quiet:
        print(f'  validate: {len(images)} cross-matched image(s)')

    import gaia_cross_match.validator as V

    original = V.find_processed_images
    V.find_processed_images = lambda target, data_dir: images
    try:
        V.validate_target(field_root.name, str(field_root.parent),
                          mag_scatter_thr=mag_scatter_thr,
                          offset_tol_mag=offset_tol_mag,
                          offset_tol_px=offset_tol_px)
    finally:
        V.find_processed_images = original

    cat = field_root / 'cross_match_catalog.csv'
    return cat if cat.exists() else None


__all__ = ['collect_images', 'validate_field']
