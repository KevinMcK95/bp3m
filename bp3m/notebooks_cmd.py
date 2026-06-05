"""bp3m-notebooks — generate and execute analysis notebooks for a bp3m field."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _find_templates() -> list[Path]:
    """Return sorted list of bundled notebook templates."""
    import bp3m as _pkg
    nb_dir = Path(_pkg.__file__).parent / 'notebooks'
    return sorted(nb_dir.glob('*.ipynb'))


def _configure_notebook(nb: dict, output_dir: Path, field: str) -> dict:
    """Replace OUTPUT_DIR and FIELD_NAME config values in all cells."""
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', [])
        lines = source if isinstance(source, list) else [source]
        if not any('OUTPUT_DIR' in l or 'FIELD_NAME' in l for l in lines):
            continue
        new_lines = []
        for line in lines:
            line = re.sub(
                r"OUTPUT_DIR\s*=\s*['\"][^'\"]*['\"]",
                f"OUTPUT_DIR = '{output_dir}'",
                line)
            line = re.sub(
                r"FIELD_NAME\s*=\s*['\"][^'\"]*['\"]",
                f"FIELD_NAME = '{field}'",
                line)
            new_lines.append(line)
        cell['source'] = new_lines
    return nb


def _current_kernel_name() -> str:
    """Return the Jupyter kernel name for the running Python, registering it if needed."""
    import subprocess
    kernel_name = Path(sys.executable).parent.parent.name  # e.g. 'bp3m_env'
    # Check if already registered
    result = subprocess.run(
        [sys.executable, '-m', 'jupyter', 'kernelspec', 'list', '--json'],
        capture_output=True, text=True)
    try:
        import json as _json
        specs = _json.loads(result.stdout).get('kernelspecs', {})
    except Exception:
        specs = {}
    if kernel_name not in specs:
        # Register the current environment as a kernel
        subprocess.run(
            [sys.executable, '-m', 'ipykernel', 'install',
             '--user', '--name', kernel_name, '--display-name', kernel_name],
            check=True, capture_output=True)
        print(f"    Registered kernel '{kernel_name}' for this environment.")
    return kernel_name


def _execute_notebook(nb_path: Path) -> bool:
    """Execute a notebook in-place. Returns True on success."""
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    except ImportError:
        print("    nbconvert not installed — skipping execution.")
        print("    Install with: pip install nbconvert")
        return False

    try:
        kernel_name = _current_kernel_name()
        nb = nbformat.read(nb_path, as_version=4)
        ep = ExecutePreprocessor(timeout=1800, kernel_name=kernel_name)
        ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent)}})
        nbformat.write(nb, nb_path)
        return True
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        msg  = getattr(e, 'evalue', str(e)) or str(e)
        print(f"    {name}: {msg}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='bp3m-notebooks',
        description=(
            'Generate pre-configured analysis notebooks for a bp3m field '
            'and optionally execute them to pre-populate all figures.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', required=True,
                        help='Target name (same as used for bp3m)')
    parser.add_argument('--output_dir', default='.',
                        help='Root output directory (default: current directory)')
    parser.add_argument('--no-execute', action='store_true',
                        help='Copy and configure notebooks but do not execute them')
    parser.add_argument('--notebooks', nargs='+', default=None,
                        metavar='NB',
                        help='Restrict to specific notebooks by prefix '
                             '(e.g. --notebooks 01 02 05). Default: all.')
    args = parser.parse_args()

    field      = args.name.replace(' ', '_')
    output_dir = Path(args.output_dir).expanduser().resolve()
    field_dir  = output_dir / field
    nb_out_dir = field_dir / 'notebooks'

    if not field_dir.exists():
        print(f"Error: field directory not found: {field_dir}")
        print(f"  (looked for '{field}' — run bp3m first, or check --output_dir)")
        sys.exit(1)

    templates = _find_templates()
    if args.notebooks:
        templates = [t for t in templates
                     if any(t.name.startswith(n) for n in args.notebooks)]
    if not templates:
        print("No matching notebook templates found.")
        sys.exit(1)

    nb_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Field:   {field}")
    print(f"Results: {field_dir}")
    print(f"Notebooks → {nb_out_dir}/\n")

    # Copy and configure
    configured: list[Path] = []
    for template in templates:
        try:
            nb = json.loads(template.read_text())
        except Exception as e:
            print(f"  WARNING: could not read {template.name}: {e}")
            continue
        nb = _configure_notebook(nb, output_dir, field)
        out_path = nb_out_dir / template.name
        out_path.write_text(json.dumps(nb, indent=1))
        configured.append(out_path)
        print(f"  Configured: {template.name}")

    if not configured:
        print("No notebooks were written.")
        sys.exit(1)

    if args.no_execute:
        print(f"\nDone. Open with:\n  jupyter notebook {nb_out_dir}/")
        return

    # Execute
    print(f"\nExecuting {len(configured)} notebook(s)...")
    n_ok = n_fail = 0
    for nb_path in configured:
        print(f"  {nb_path.name} ...", end=' ', flush=True)
        if _execute_notebook(nb_path):
            print("done")
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{n_ok} succeeded, {n_fail} failed.")
    print(f"Open with:\n  jupyter notebook {nb_out_dir}/")


if __name__ == '__main__':
    main()
