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


def write_command_record(out_dir, tool: str = 'bp3m', argv=None, note: str | None = None) -> None:
    try:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        argv = list(sys.argv if argv is None else argv)
        line = (f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  tool={tool}"
                + (f"  {note}" if note else '') + "\n" + ' '.join(shlex.quote(str(a)) for a in argv) + "\n")
        (out_dir / 'bp3m_command.txt').write_text(line)
        with open(out_dir / 'bp3m_command_history.txt', 'a') as f:
            f.write(line)
    except Exception as e:      # a bookkeeping failure must never fail the run
        print(f"  WARNING: could not write bp3m_command.txt in {out_dir}: {e}")
