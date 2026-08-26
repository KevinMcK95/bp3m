"""
Bulk processing: deterministic work partitioning, resume, and a failure manifest.

Designed for many workers over a large dataset (CFHT/UNIONS: 21,200 exposures,
~848,000 images), where three things matter that do not matter for a smoke test:

  partitioning  Two workers must never touch the same exposure.  Concurrent
                writes to the same output directory corrupt it silently — no
                exception, just interleaved files.  Partitioning is by
                `index % n_workers == worker_id` on a sorted exposure list:
                deterministic, needs no coordination, and has none of the
                stale-lock failure modes of a lock file.

  resume        A crash 60 hours in must not mean starting over.  Each finished
                exposure gets a sentinel written AFTER its outputs, so a
                half-written exposure is never treated as done.  Checking one
                sentinel also avoids stat-ing 40 directories per exposure on a
                slow filesystem.

  manifest      One JSONL line per exposure per worker, appended immediately.
                Failures in a run this size are invisible in stdout; the
                manifest makes them greppable, countable, and re-runnable.

Each worker writes only to its own manifest, so there is no write contention.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import layout

SENTINEL = '.complete'
MANIFEST_DIR = 'bulk_logs'


# ── partitioning ─────────────────────────────────────────────────────────────

def partition(items, n_workers: int, worker_id: int) -> list:
    """
    Deterministic slice of `items` for this worker.

    Round-robin on a SORTED list, so the split depends only on (n_workers,
    worker_id) and never on timing, filesystem order, or what other workers
    have done.  Re-running a worker reproduces exactly its own slice.
    """
    if not (0 <= worker_id < n_workers):
        raise ValueError(f'worker_id {worker_id} out of range for {n_workers} workers')
    ordered = sorted(items, key=str)
    return [x for i, x in enumerate(ordered) if i % n_workers == worker_id]


# ── resume ───────────────────────────────────────────────────────────────────

def sentinel_path(field_root: Path, exp_id: str) -> Path:
    return layout.align_root(field_root) / exp_id / SENTINEL


def is_complete(field_root: Path, exp_id: str) -> bool:
    return sentinel_path(field_root, exp_id).exists()


def mark_complete(field_root: Path, exp_id: str, summary: dict) -> None:
    """
    Write the sentinel LAST, once outputs are safely on disk.

    Written to a temp file and renamed, so an interrupted write cannot leave a
    truncated sentinel that would make a partial exposure look finished.
    """
    p = sentinel_path(field_root, exp_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2, default=str))
    os.replace(tmp, p)


# ── manifest ─────────────────────────────────────────────────────────────────

class Manifest:
    """Append-only JSONL record, one file per worker."""

    def __init__(self, field_root: Path, worker_id: int):
        d = Path(field_root) / MANIFEST_DIR
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f'worker_{worker_id:03d}.jsonl'

    def record(self, **entry) -> None:
        entry.setdefault('time', datetime.now().isoformat(timespec='seconds'))
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')
            f.flush()


def read_manifests(field_root: Path):
    """Every manifest entry across all workers, for progress and triage."""
    import pandas as pd
    d = Path(field_root) / MANIFEST_DIR
    if not d.is_dir():
        return pd.DataFrame()
    rows = []
    for f in sorted(d.glob('worker_*.jsonl')):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass   # a torn final line from a killed worker
    return pd.DataFrame(rows)


# ── driver ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime('%H:%M:%S')


def _n_images(inst) -> int:
    """Detector count for this exposure, for the success-rate column."""
    try:
        return len(list(inst.iter_images()))
    except Exception:
        return 0


def _fmt_eta(seconds: float) -> str:
    if not seconds or seconds < 0 or seconds != seconds:
        return '?'
    h, m = divmod(int(seconds) // 60, 60)
    d, h = divmod(h, 24)
    return f'{d}d {h}h {m}m' if d else (f'{h}h {m}m' if h else f'{m}m')


def run_bulk(instrument_factory, field_root: Path, exposures,
             n_workers: int, worker_id: int,
             do_xmatch=True, do_align=True, make_plots=False,
             force=False, dry_run=False, **align_kw):
    """
    Process this worker's slice of `exposures`.

    instrument_factory(exposure) -> (instrument, source_tiers)

    Every exposure is independent: a failure is recorded and the run continues,
    rather than losing the remaining hours of work.
    """
    from . import xmatch as xmatch_mod
    from .align import driver as align_driver

    field_root = Path(field_root)
    mine = partition(exposures, n_workers, worker_id)
    todo = mine if force else [e for e in mine
                               if not is_complete(field_root, f'cfht_{e}')]

    print(f'[{_ts()}] [w{worker_id}/{n_workers}] {len(mine)} assigned, '
          f'{len(todo)} to do, {len(mine) - len(todo)} already complete',
          flush=True)
    if dry_run:
        print(f'  dry run — first 10: {todo[:10]}')
        return todo

    man = Manifest(field_root, worker_id)
    man.record(event='start', n_assigned=len(mine), n_todo=len(todo),
               n_workers=n_workers)

    t_start = time.time()
    n_ok = n_fail = 0
    for i, exp in enumerate(todo, 1):
        exp_id = f'cfht_{exp}'
        t0 = time.time()
        try:
            inst, tiers = instrument_factory(exp)
            n_matched = 0
            if do_xmatch:
                # Per-image detail is captured in each detector's
                # processing_log.txt; at this scale it would bury the
                # per-exposure summary line, so keep stdout to one line each.
                import contextlib, io as _io
                _buf = _io.StringIO()
                with contextlib.redirect_stdout(_buf):
                    df = xmatch_mod.run(inst, field_root, source_tiers=tiers,
                                        make_plots=make_plots)
                if len(df) and 'n_matched' in df.columns:
                    n_matched = int(df['n_matched'].fillna(0).sum())
            n_solved = 0
            if do_align:
                import contextlib, io as _io
                _buf2 = _io.StringIO()
                with contextlib.redirect_stdout(_buf2):
                    outs = align_driver.run_per_image(inst, field_root,
                                                      make_plots=make_plots,
                                                      verbose=False, **align_kw)
                n_solved = len(outs or [])

            dt = time.time() - t0
            n_det = _n_images(inst)
            summary = dict(exposure=exp, n_matched=n_matched,
                           n_solved=n_solved, n_detectors=n_det,
                           seconds=round(dt, 1), worker=worker_id)
            mark_complete(field_root, exp_id, summary)
            man.record(event='ok', **summary)
            n_ok += 1
            print(f'[{_ts()}] [w{worker_id}] [{i:5d}/{len(todo)}] {exp_id}: '
                  f'{n_matched:6d} matches  {n_solved:3d}/{n_det:<3d} detectors '
                  f'({100*n_solved/max(n_det,1):5.1f}%)  {dt:5.1f}s', flush=True)
        except Exception as e:
            dt = time.time() - t0
            n_fail += 1
            man.record(event='fail', exposure=exp, seconds=round(dt, 1),
                       worker=worker_id, error=f'{type(e).__name__}: {e}',
                       traceback=traceback.format_exc()[-2000:])
            print(f'[{_ts()}] [w{worker_id}] [{i:5d}/{len(todo)}] {exp_id}: '
                  f'FAILED after {dt:.1f}s — {type(e).__name__}: {e}', flush=True)

        if i % 10 == 0 or i == len(todo):
            elapsed = time.time() - t_start
            rate = elapsed / i
            print(f'[{_ts()}] [w{worker_id}] --- {i}/{len(todo)}  ok={n_ok} '
                  f'fail={n_fail}  {rate:.1f}s/exp  '
                  f'ETA {_fmt_eta(rate * (len(todo) - i))} ---', flush=True)

    man.record(event='done', n_ok=n_ok, n_fail=n_fail,
               seconds=round(time.time() - t_start, 1))
    print(f'[{_ts()}] [w{worker_id}] FINISHED: {n_ok} ok, {n_fail} failed, '
          f'{_fmt_eta(time.time() - t_start)} elapsed', flush=True)
    return n_ok, n_fail


__all__ = ['partition', 'is_complete', 'mark_complete', 'sentinel_path',
           'Manifest', 'read_manifests', 'run_bulk', 'SENTINEL', 'MANIFEST_DIR']
