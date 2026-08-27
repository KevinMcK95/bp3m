#!/usr/bin/env python
"""
Write .complete sentinels for output directories produced before resume support.

    python -m ground_to_gaia_xmatch.scripts.backfill_sentinels --field-root <dir>
    python -m ground_to_gaia_xmatch.scripts.backfill_sentinels --field-root <dir> --dry-run

Best-effort by construction: a sentinel is normally written last, after every
output, so its presence proves completion.  Backfilling can only check that the
expected final files exist, which cannot distinguish "finished" from "died after
writing these but before the rest".  It therefore requires ALL of a stage's
files, and skips anything incomplete rather than guessing.  Run it only on output
you believe finished cleanly; otherwise just let --force redo the work.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import layout

# Every file a stage must have before it counts as complete.
XMATCH_REQUIRED = (layout.MATCHED_CSV, layout.TRANSFORM_CSV)
ALIGN_REQUIRED = (layout.STELLAR_CSV, layout.IMAGE_TRANSFORM_CSV,
                  'C_vT.npy', 'C_r.npy', 'detections.npz')


def _complete_dirs(root: Path, required) -> list[Path]:
    """Directories under root holding every required file."""
    if not root.is_dir():
        return []
    hits = []
    for d in sorted(p for p in root.rglob('*') if p.is_dir()):
        if d.name == layout.PLOTS_DIR or layout.PLOTS_DIR in d.parts[-2:-1]:
            continue
        if all((d / f).is_file() for f in required):
            hits.append(d)
    return hits


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--field-root', required=True, type=Path, nargs='+')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    total = 0
    for fr in args.field_root:
        for stage, root, req in (
                ('xmatch', layout.xmatch_root(fr), XMATCH_REQUIRED),
                ('align', layout.align_root(fr), ALIGN_REQUIRED)):
            dirs = _complete_dirs(root, req)
            todo = [d for d in dirs if not layout.is_complete(d)]
            print(f'{fr.name:20s} {stage:7s} {len(dirs):5d} complete-looking, '
                  f'{len(todo):5d} need a sentinel')
            if not args.dry_run:
                for d in todo:
                    layout.mark_complete(d, {'backfilled': True, 'stage': stage})
                total += len(todo)
    print(f'\n{"would write" if args.dry_run else "wrote"} {total} sentinel(s)')


if __name__ == '__main__':
    main()
