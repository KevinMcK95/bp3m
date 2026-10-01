"""Per-output-directory record of the command that produced it (user 2026-09-30).

Every BP3M tool writes ``bp3m_command.txt`` (timestamp, tool name, the full command line) into the
directory it just filled -- BP3M_results[_suffix], BP3M_indv_results, BP3M_v2_results,
BP3M_pop_fit*_results, notebooks/ -- on successful completion, and appends the same line to
``bp3m_command_history.txt`` there.  The field-level ``bp3m_command.txt`` keeps meaning "the most
recent bp3m run of this field"; the per-directory copies say what produced *that* result, which
survives the directory being renamed or the field-level file being overwritten by a later run.
"""
from __future__ import annotations
import shlex, sys
from datetime import datetime
from pathlib import Path


def replace_text(path, text: str) -> None:
    """Write `text` to `path` via a temp file + os.replace, so a hard-linked copy of the
    file (field copies made with cp -al) is detached instead of overwritten through the link
    (Leo_I/bp3m_command.txt ended up holding a Leo_I_fs_gdcB command that way)."""
    import os, tempfile
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f'.{path.name}.', suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def _append_text(path, text: str) -> None:
    path = Path(path)
    old = path.read_text() if path.exists() else ''
    if path.exists() and path.stat().st_nlink > 1:
        replace_text(path, old + text)      # detach from the shared inode first
    else:
        with open(path, 'a') as f:
            f.write(text)


def write_command_record(out_dir, tool: str = 'bp3m', argv=None, note: str | None = None) -> None:
    try:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        argv = list(sys.argv if argv is None else argv)
        line = (f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  tool={tool}"
                + (f"  {note}" if note else '') + "\n" + ' '.join(shlex.quote(str(a)) for a in argv) + "\n")
        replace_text(out_dir / 'bp3m_command.txt', line)
        _append_text(out_dir / 'bp3m_command_history.txt', line)
    except Exception as e:      # a bookkeeping failure must never fail the run
        print(f"  WARNING: could not write bp3m_command.txt in {out_dir}: {e}")
