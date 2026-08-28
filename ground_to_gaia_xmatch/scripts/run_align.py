#!/usr/bin/env python
"""
Align a ground-based dataset against Gaia, using the cross-match outputs.

    # one independent solve per image
    python -m ground_to_gaia_xmatch.scripts.run_align --instrument lsst \
        --field-root /path/to/Fornax_dSph --mode per-image

    # one joint solve across every image, sharing stars
    python -m ground_to_gaia_xmatch.scripts.run_align --instrument lsst \
        --field-root /path/to/Fornax_dSph --mode joint
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..align import driver


def build_instrument(name: str, field_root: Path,
                     exposures=None, detectors=None):
    if name == 'lsst':
        from ..instruments.lsst import LSSTInstrument
        return LSSTInstrument(field_root, exposures=exposures, detectors=detectors)
    if name == 'cfht':
        from ..instruments.cfht import CFHTInstrument
        return CFHTInstrument(field_root, exposures=exposures, detectors=detectors)
    raise SystemExit(f'unknown instrument: {name!r} (expected lsst or cfht)')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument', required=True, choices=['lsst', 'cfht'])
    p.add_argument('--field-root', required=True, type=Path)
    p.add_argument('--mode', default='per-image', choices=['per-image', 'joint'])
    p.add_argument('--label', default='all',
                   help="Output label for joint mode -> align/joint_<label>/")
    p.add_argument('--n-iter', type=int, default=20)
    p.add_argument('--clip-sigma', type=float, default=4.5)
    p.add_argument('--exposure', type=int, nargs='+', default=None,
                   help='Restrict to these exposure ids (visit / expnum)')
    p.add_argument('--detector', type=int, nargs='+', default=None,
                   help='Restrict to these detector ids (detector / ext)')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip diagnostic figures (use for bulk reduction)')
    p.add_argument('--alpha-scale-chi2', action='store_true',
                   help='Rescale the Test-3 chi2 by alpha before thresholding, '
                        'so the cut is uniform across images in units of image '
                        'noise (bp3m alpha_scale_chi2; off by default there too)')
    p.add_argument('--exclude-2p-from-alignment', action='store_true',
                   help='Gaia 2p stars get astrometry but do not constrain the '
                        'image transformation (bp3m exclude_2p_from_alignment)')
    p.add_argument('--select-radius', '--select_radius', dest='select_radius',
                   type=float, default=None,
                   help='Restrict the fit to images whose detector centre lies '
                        'within this many degrees of the field centre.  Operates '
                        'on the EXISTING cross-match, so shrinking the analysis '
                        'region costs nothing and needs no new download.')
    p.add_argument('--select-center', '--select_center', dest='select_center',
                   type=float, nargs=2, default=None, metavar=('RA', 'DEC'),
                   help='Centre for --select-radius (default: the cone centre '
                        'recorded in table_dp2.*-query.json)')
    p.add_argument('--use-delve', '--use_delve', dest='use_delve',
                   action='store_true',
                   help='Fold DELVE proper-motion priors into the fit and admit '
                        'DELVE-only stars, as bp3m --use_delve does.  Requires '
                        'matched_delve.csv from a cross-match run with '
                        '--use-delve.')
    p.add_argument('--delve-use-for-align', '--delve_use_for_align',
                   dest='delve_use_for_align', action='store_true',
                   help='Let DELVE-only sources help calibrate image transforms '
                        '(off by default; they inform their own astrometry only).')
    p.add_argument('--force', action='store_true',
                   help='Re-solve images that already have complete output. '
                        'By default a directory carrying a .complete sentinel is '
                        'reused, so an interrupted or extended run resumes '
                        'instead of redoing finished work.')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)

    inst = build_instrument(args.instrument, args.field_root,
                            exposures=args.exposure,
                            detectors=args.detector)
    kw = dict(n_iter=args.n_iter, clip_sigma=args.clip_sigma,
              alpha_scale_chi2=args.alpha_scale_chi2,
              exclude_2p_from_alignment=args.exclude_2p_from_alignment,
              make_plots=not args.no_plots, verbose=not args.quiet)

    if args.mode == 'joint':
        driver.run_joint(inst, args.field_root, force=args.force, label=args.label,
                         select_radius=args.select_radius,
                         select_center=tuple(args.select_center)
                         if args.select_center else None,
                         use_delve=args.use_delve,
                         delve_use_for_align=args.delve_use_for_align, **kw)
    else:
        driver.run_per_image(inst, args.field_root, force=args.force,
                             use_delve=args.use_delve,
                             delve_use_for_align=args.delve_use_for_align, **kw)


if __name__ == '__main__':
    main()
