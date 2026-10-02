"""Detach hard-linked output files before a stage rewrites them (2026-10-01).

Field copies (<FIELD>_fs_*, A/B arms) are built with hard links to save space.  BP3M writers
(to_csv, np.savez, write_text, copy2, savefig) write IN PLACE, so rewriting a shared file in the
original field silently rewrites it in every copy (Leo_I BP3M_results lost 2026-09-18).  Replacing
each shared file by a private copy (copy + os.replace) first leaves the copies untouched.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def detach_file(path) -> bool:
    p = Path(path)
    try:
        if p.is_symlink() or not p.is_file() or p.stat().st_nlink <= 1:
            return False
        tmp = p.with_name(f'.{p.name}.detach_{os.getpid()}')
        shutil.copy2(p, tmp)
        os.replace(tmp, p)
        return True
    except OSError:
        return False


def detach_tree(root, recursive: bool = True, keep_shared=lambda name: False) -> int:
    """Detach every hard-linked file under root (except names for which keep_shared is True)."""
    n = 0
    root = Path(root)
    if not root.is_dir():
        return 0
    it = os.walk(root) if recursive else [(str(root), [], [e.name for e in os.scandir(root) if e.is_file()])]
    for d, _, files in it:
        for f in files:
            if not keep_shared(f) and detach_file(Path(d) / f):
                n += 1
    return n
