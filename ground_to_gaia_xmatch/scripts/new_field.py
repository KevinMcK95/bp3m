#!/usr/bin/env python
"""
Set up a new LSST field end to end: download, lay out, and optionally analyse.

    python -m ground_to_gaia_xmatch.scripts.new_field \
        --name Sculptor_dSph --ra 15.021 --dec -33.6815 --radius 0.75 --run

Does, in order:
  1. dp2.Source in the cone            -> table_dp2.<name>-data.tbl
  2. dp2.VisitDetector over 2x the cone -> table_dp2.<name>-VisitDetector.tbl
  3. Gaia over the same footprint      -> Gaia/<name>_ra..._gaia.csv
  4. with --run: cross-match, per-image alignment, and diagnostics

Needs an RSP token for steps 1-2 (see download_lsst).  Step 3 needs no
credentials.

The Gaia box is sized from the footprint the LSST data actually covers, not from
the requested radius, so it always contains the detectors that came back.  Note
download_gaia nests its output under a <field_name>/ subdirectory, which the
adapter's non-recursive glob would miss — this script flattens it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path('/home/jupyter-kmckinnon/data_bootes/bp3m/Rubin/LSST')

# Padding beyond the LSST footprint for the Gaia box, in degrees.  One LSST
# detector is ~0.23 deg across, so half a detector plus slack.
GAIA_PAD_DEG = 0.4


def footprint(field_root: Path, name: str):
    """(ra, dec, width, height) covering the downloaded visit-detector centres."""
    from astropy.io import ascii as ascii_io
    vd = ascii_io.read(field_root / f'table_dp2.{name}-VisitDetector.tbl',
                       format='ipac').to_pandas()
    ra0, dec0 = float(vd.ra.mean()), float(vd.dec.mean())
    # Width in RA degrees: divide the great-circle span by cos(dec), since a
    # search box in RA covers less sky at higher |dec|.
    cd = max(np.cos(np.radians(dec0)), 1e-6)
    w = (float(vd.ra.max() - vd.ra.min()) + 2 * GAIA_PAD_DEG / cd)
    h = (float(vd.dec.max() - vd.dec.min()) + 2 * GAIA_PAD_DEG)
    return ra0, dec0, w, h


def fetch_gaia(field_root: Path, name: str, quiet=False, query_timeout=600,
               force=False):
    """Gaia over the LSST footprint, flattened into <field_root>/Gaia/."""
    from bp3m.pipeline.download_gaia import download_gaia
    ra0, dec0, w, h = footprint(field_root, name)
    gdir = field_root / 'Gaia'
    gdir.mkdir(parents=True, exist_ok=True)
    if list(gdir.glob('*_gaia.csv')) and not force:
        if not quiet:
            print('  Gaia catalogue already present (--force-download to refetch)')
        return
    if not quiet:
        print(f'  Gaia: centre ({ra0:.4f}, {dec0:.4f}) box {w:.4f} x {h:.4f} deg')
    download_gaia(ra=ra0, dec=dec0, search_width=w, search_height=h,
                  output_dir=str(gdir), field_name=name,
                  min_gmag=0.0, max_gmag=None, n_processes=4,
                  query_timeout=query_timeout, quiet=quiet)
    # download_gaia nests its output, and by more than one level: the observed
    # layout is <output_dir>/<field_name>/Gaia/<files>.  Flatten whatever depth
    # it used, because the adapter's glob is non-recursive.
    moved = 0
    for f in sorted(gdir.rglob('*')):
        if f.is_file() and f.parent != gdir:
            dest = gdir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                moved += 1
    # Remove the now-empty nest, deepest first.
    for d in sorted((d for d in gdir.rglob('*') if d.is_dir()),
                    key=lambda x: -len(x.parts)):
        try:
            d.rmdir()
        except OSError:
            pass
    if moved and not quiet:
        print(f'  flattened {moved} file(s) into Gaia/')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True, help='Field name, e.g. Sculptor_dSph')
    p.add_argument('--ra', type=float, required=True)
    p.add_argument('--dec', type=float, required=True)
    p.add_argument('--radius', '--search-radius', '--search_radius',
                   type=float, required=True, dest='radius',
                   help='Source cone radius [deg]')
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                   help=f'Parent directory for fields (default {DEFAULT_ROOT})')
    p.add_argument('--bands', nargs='+', default=None)
    p.add_argument('--max-rows', type=int, default=None)
    p.add_argument('--token-file', default=None)
    p.add_argument('--use-delve', '--use_delve', dest='use_delve',
                   action='store_true',
                   help='Also cross-match each image against DELVE and write '
                        'matched_delve.csv, so DELVE PM priors and DELVE-only '
                        'stars are available to the fit (bp3m --use_delve).')
    p.add_argument('--delve-dir', '--delve_dir', dest='delve_dir', default=None,
                   help='Directory of DELVE PM_hp*.fits tiles '
                        '(default: the bp3m DELVE_ProperMotion/PMCatalog path)')
    p.add_argument('--delve-use-for-align', '--delve_use_for_align',
                   dest='delve_use_for_align', action='store_true',
                   help='Let DELVE-only sources help calibrate image transforms')
    p.add_argument('--force-download', '--force_download',
                   dest='force_download', action='store_true',
                   help='Re-query the LSST and Gaia catalogues even when tables '
                        'are already present.  Needed when EXTENDING a field to a '
                        'larger --search_radius, since the download cache checks '
                        'only that files exist, not that they cover the request. '
                        'Deliberately separate from --force so the cross-match '
                        'can still resume: images already done are skipped and '
                        'only the newly-covered ones are processed.')
    p.add_argument('--gaia-timeout', type=int, default=600,
                   help='Per-magnitude-bin Gaia TAP timeout in seconds '
                        '(default 600).  Raise it for crowded low-latitude '
                        'fields: Sagittarius dSph at l=5.6, b=-14 timed out on '
                        'the G 17-19 bin at 600 s.')
    p.add_argument('--vd-radius-factor', type=float, default=2.0)
    p.add_argument('--skip-lsst', action='store_true')
    p.add_argument('--skip-gaia', action='store_true')
    p.add_argument('--run', action='store_true',
                   help='Also run cross-match, alignment and diagnostics')
    p.add_argument('--align-mode', '--align_mode', dest='align_mode',
                   default='per-image',
                   choices=['per-image', 'joint', 'both', 'none'],
                   help="With --run, which alignment to do (default per-image). "
                        "'joint' skips the per-image solves entirely and fits all "
                        "images together -- much faster on large fields, since "
                        "the joint solve reads the cross-match output directly "
                        "and never needs the per-image results.  'none' stops "
                        "after the cross-match.")
    p.add_argument('--joint-label', default='all',
                   help='Output label for the joint fit (align/joint_<label>/)')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='Reprocess images that already have complete output. '
                        'By default a directory carrying a .complete sentinel is '
                        'skipped, so reruns and radius changes resume instead of '
                        'redoing finished work.')
    args = p.parse_args(argv)

    field_root = args.root / args.name
    field_root.mkdir(parents=True, exist_ok=True)
    print(f'field root: {field_root}')

    if not args.skip_lsst:
        from ..download_lsst import download_lsst
        print('[1/3] LSST DP2 source + visit-detector tables')
        download_lsst(args.ra, args.dec, args.radius,
                      field_root=field_root, field_name=args.name,
                      bands=args.bands, max_rows=args.max_rows,
                      token_file=args.token_file,
                      vd_radius_factor=args.vd_radius_factor,
                      force=args.force_download)

    if not args.skip_gaia:
        print('[2/3] Gaia catalogue over the LSST footprint')
        fetch_gaia(field_root, args.name, query_timeout=args.gaia_timeout,
                   force=args.force_download)

    if not args.run:
        print('\nready. analyse with:')
        print(f'  python -m ground_to_gaia_xmatch.scripts.run_xmatch '
              f'--instrument lsst --field-root {field_root}')
        return

    print('[3/3] cross-match + alignment + diagnostics')
    from .. import xmatch
    from ..align import driver
    from ..discovery import magnitude_tiers
    from ..instruments.lsst import LSSTInstrument
    inst = LSSTInstrument(field_root)
    delve_df = None
    if args.use_delve:
        from ..delve import DEFAULT_DELVE_DIR, fetch_delve, load_delve
        print('  DELVE: extracting catalogue over the field footprint')
        fetch_delve(field_root, args.name,
                    delve_dir=args.delve_dir or DEFAULT_DELVE_DIR,
                    force=args.force_download)
        delve_df = load_delve(field_root)
        if delve_df is None:
            print('  no DELVE coverage for this field — continuing Gaia-only')
    xmatch.run(inst, field_root, source_tiers=magnitude_tiers,
               make_plots=not args.no_plots, force=args.force,
               delve_df=delve_df)

    mode = args.align_mode
    if mode in ('per-image', 'both'):
        driver.run_per_image(inst, field_root, make_plots=not args.no_plots,
                             verbose=False, force=args.force,
                             use_delve=args.use_delve,
                             delve_use_for_align=args.delve_use_for_align)
    if mode in ('joint', 'both'):
        driver.run_joint(inst, field_root, label=args.joint_label,
                         make_plots=not args.no_plots, force=args.force,
                         use_delve=args.use_delve,
                         delve_use_for_align=args.delve_use_for_align)

    # diagnose_transforms reads the PER-IMAGE align output, so it only has
    # something to say when those solves ran.
    if mode in ('per-image', 'both'):
        from .diagnose_transforms import main as diag
        diag(['--instrument', 'lsst', '--field-root', str(field_root)])
    elif mode == 'joint':
        print('  (skipping transform diagnostics: they read the per-image '
              'output, which --align-mode joint does not produce)')


if __name__ == '__main__':
    main()
