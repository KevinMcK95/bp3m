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
    p.add_argument('--no-plots', action='store_true',
                   help='Skip diagnostic figures (use for bulk reduction)')
    args = p.parse_args(argv)

    inst, tiers = build_instrument(args.instrument, args.field_root,
                                   exposures=args.exposure,
                                   detectors=args.detector)
    xmatch.run(inst, args.field_root, source_tiers=tiers,
               make_plots=not args.no_plots)


if __name__ == '__main__':
    main()
