#!/usr/bin/env python
"""
Cross-match a ground-based dataset against Gaia.

    python -m ground_to_gaia_xmatch.scripts.run_xmatch --instrument lsst \
        --field-root /path/to/Fornax_dSph
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import xmatch
from ..discovery import magnitude_tiers


def build_instrument(name: str, field_root: Path,
                     exposures=None, detectors=None):
    """Instantiate the adapter for `name`, along with its source tier-walk."""
    if name == 'lsst':
        from ..instruments.lsst import LSSTInstrument
        return (LSSTInstrument(field_root, exposures=exposures,
                               detectors=detectors), magnitude_tiers)
    if name == 'cfht':
        from ..instruments.cfht import CFHTInstrument, magerr_mag_tiers
        return (CFHTInstrument(field_root, exposures=exposures,
                               detectors=detectors), magerr_mag_tiers)
    raise SystemExit(f'unknown instrument: {name!r} (expected lsst or cfht)')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument', required=True, choices=['lsst', 'cfht'])
    p.add_argument('--field-root', required=True, type=Path,
                   help='Field directory (contains Gaia/ and the source tables)')
    p.add_argument('--exposure', type=int, nargs='+', default=None,
                   help='Restrict to these exposure ids (visit / expnum)')
    p.add_argument('--detector', type=int, nargs='+', default=None,
                   help='Restrict to these detector ids (detector / ext)')
    p.add_argument('--use-delve', '--use_delve', dest='use_delve',
                   action='store_true',
                   help='Also cross-match each image against DELVE and write '
                        'matched_delve.csv, so DELVE PM priors and DELVE-only '
                        'stars are available to the fit (bp3m --use_delve).')
    p.add_argument('--delve-dir', '--delve_dir', dest='delve_dir', default=None,
                   help='Directory of DELVE PM_hp*.fits tiles '
                        '(default: the bp3m DELVE_ProperMotion/PMCatalog path)')
    p.add_argument('--no-validate', dest='validate', action='store_false',
                   help='Skip the cross-image validation that normally follows '
                        'the cross-match (bp3m runs it via '
                        '_validate_catalog_if_needed)')
    p.add_argument('--force', action='store_true',
                   help='Reprocess images that already have complete output. '
                        'By default a directory carrying a .complete sentinel is '
                        'skipped, so reruns and radius changes resume instead of '
                        'redoing finished work.')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip diagnostic figures (use for bulk reduction)')
    args = p.parse_args(argv)

    inst, tiers = build_instrument(args.instrument, args.field_root,
                                   exposures=args.exposure,
                                   detectors=args.detector)
    delve_df = None
    if args.use_delve:
        from ..delve import DEFAULT_DELVE_DIR, fetch_delve, load_delve
        fetch_delve(args.field_root, args.field_root.name,
                    delve_dir=args.delve_dir or DEFAULT_DELVE_DIR)
        delve_df = load_delve(args.field_root)
        if delve_df is None:
            print('  no DELVE coverage for this field — continuing Gaia-only')
    xmatch.run(inst, args.field_root, source_tiers=tiers, force=args.force,
               delve_df=delve_df,
               make_plots=not args.no_plots)

    # bp3m runs its validator immediately after cross-matching
    # (cross_match._validate_catalog_if_needed), because the is_trustworthy flag
    # and cross_match_catalog.csv it produces are what downstream steps key on.
    if args.validate:
        from ..validate import validate_field
        validate_field(args.field_root)


if __name__ == '__main__':
    main()
