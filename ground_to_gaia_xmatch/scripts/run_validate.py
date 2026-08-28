#!/usr/bin/env python
"""
Cross-image validation of a field's cross-matches (bp3m's validator).

    python -m ground_to_gaia_xmatch.scripts.run_validate --field-root <dir>

Runs gaia_cross_match.validator.validate_target over the ground layout, writing
per-image source_quality.csv plus field-level magnitude_zp_offsets.csv,
cross_match_catalog.csv and the CMD / colour-colour plots.  See
ground_to_gaia_xmatch.validate for how the HST-specific image discovery is
adapted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..validate import validate_field


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--field-root', required=True, type=Path, nargs='+')
    p.add_argument('--mag-scatter-threshold', type=float, default=0.1,
                   help='MAD threshold for the magnitude-consistency flag [mag]')
    p.add_argument('--offset-tol-mag', type=float, default=0.05)
    p.add_argument('--offset-tol-px', type=float, default=10.0)
    args = p.parse_args(argv)

    for fr in args.field_root:
        print(f'\n=== {fr} ===')
        validate_field(fr, mag_scatter_thr=args.mag_scatter_threshold,
                       offset_tol_mag=args.offset_tol_mag,
                       offset_tol_px=args.offset_tol_px)


if __name__ == '__main__':
    main()
