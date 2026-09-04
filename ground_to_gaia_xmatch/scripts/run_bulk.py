#!/usr/bin/env python
"""
Bulk cross-match + alignment over a whole dataset, one worker's share.

Launch N workers, each with a distinct --worker-id.  Partitioning is
deterministic (index %% n_workers), so no two workers ever touch the same
exposure and no coordination is needed.

    for i in $(seq 0 3); do
        python -m ground_to_gaia_xmatch.scripts.run_bulk \
            --instrument cfht --field-root /path/to/CFHT/UNIONS \
            --n-workers 4 --worker-id $i --no-plots &
    done

Resume is automatic: exposures with a completion sentinel are skipped, so a
killed run is restarted with the identical command.  Use --force to redo them.

Progress and failures:
    python -m ground_to_gaia_xmatch.scripts.run_bulk ... --status
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import bulk
from ..discovery import magnitude_tiers


def make_factory(instrument: str, field_root: Path):
    """Return instrument_factory(exposure) -> (instrument, source_tiers)."""
    if instrument == 'cfht':
        from ..instruments.cfht import CFHTInstrument, magerr_mag_tiers

        def factory(exp):
            return CFHTInstrument(field_root, exposures=(int(exp),)), magerr_mag_tiers
        return factory
    if instrument == 'lsst':
        from ..instruments.lsst import LSSTInstrument

        def factory(exp):
            return LSSTInstrument(field_root, exposures=(int(exp),)), magnitude_tiers
        return factory
    raise SystemExit(f'unknown instrument: {instrument!r}')


def all_exposures(instrument: str, field_root: Path):
    if instrument == 'cfht':
        from ..instruments.cfht import CFHTInstrument
        return CFHTInstrument(field_root).available_exposures()
    from ..instruments.lsst import LSSTInstrument
    inst = LSSTInstrument(field_root)
    return sorted({m.exposure for m in inst.iter_images()})


def show_status(field_root: Path, instrument: str):
    """Aggregate progress and failures across every worker's manifest."""
    import pandas as pd
    df = bulk.read_manifests(field_root)
    if df.empty:
        print('no manifest entries yet')
        return
    ev = df[df.event.isin(['ok', 'fail'])] if 'event' in df else df
    total = len(all_exposures(instrument, field_root))
    n_ok = int((ev.event == 'ok').sum())
    n_fail = int((ev.event == 'fail').sum())
    print(f'exposures: {total} total  |  {n_ok} ok  |  {n_fail} failed  '
          f'|  {total - n_ok - n_fail} remaining  ({100*n_ok/max(total,1):.1f}% done)')
    if 'seconds' in ev.columns and n_ok:
        s = pd.to_numeric(ev.loc[ev.event == 'ok', 'seconds'], errors='coerce')
        print(f'  median {s.median():.1f}s/exposure')
        if 'worker' in ev.columns:
            per = ev[ev.event == 'ok'].groupby('worker').size()
            print(f'  per worker: {per.to_dict()}')
        rem = total - n_ok - n_fail
        nw = ev.worker.nunique() if 'worker' in ev.columns else 1
        if rem > 0 and nw:
            print(f'  ETA at current rate: {bulk._fmt_eta(s.median()*rem/nw)}')
    if n_fail:
        print(f'\n{n_fail} FAILURES:')
        f = ev[ev.event == 'fail']
        if 'error' in f.columns:
            for err, k in f.error.value_counts().head(10).items():
                print(f'  {k:5d}  {err[:100]}')
            print(f'\n  re-run failures with:  --exposure '
                  f'{" ".join(map(str, f.exposure.head(20).tolist()))}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument', required=True, choices=['lsst', 'cfht'])
    p.add_argument('--field-root', required=True, type=Path)
    p.add_argument('--n-workers', type=int, default=1)
    p.add_argument('--worker-id', type=int, default=0)
    p.add_argument('--exposure', nargs='+', default=None,
                   help='Restrict to these exposures (e.g. to retry failures)')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip figures — strongly recommended for bulk')
    p.add_argument('--skip-xmatch', action='store_true')
    p.add_argument('--skip-align', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='Redo exposures that already have a completion sentinel')
    p.add_argument('--redo-before', default=None,
                   help='Redo exposures whose completion sentinel is older '
                        'than this (unix mtime or YYYY-MM-DD). Resumable '
                        'stale-results overwrite, e.g. for the 2026-09 '
                        'parallax-factor fix.')
    p.add_argument('--dry-run', action='store_true',
                   help='Show this worker\'s slice without running')
    p.add_argument('--status', action='store_true',
                   help='Report aggregate progress and failures, then exit')
    args = p.parse_args(argv)

    if args.status:
        show_status(args.field_root, args.instrument)
        return

    exps = ([int(e) for e in args.exposure] if args.exposure
            else all_exposures(args.instrument, args.field_root))
    _redo_before = None
    if args.redo_before is not None:
        try:
            _redo_before = float(args.redo_before)
        except ValueError:
            import datetime as _dt
            _redo_before = _dt.datetime.strptime(
                args.redo_before, '%Y-%m-%d').timestamp()
    bulk.run_bulk(make_factory(args.instrument, args.field_root),
                  args.field_root, exps,
                  n_workers=args.n_workers, worker_id=args.worker_id,
                  do_xmatch=not args.skip_xmatch,
                  do_align=not args.skip_align,
                  make_plots=not args.no_plots,
                  force=args.force, dry_run=args.dry_run,
                  redo_before=_redo_before)


if __name__ == '__main__':
    main()
