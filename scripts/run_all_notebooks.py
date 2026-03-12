#!/usr/bin/env python
"""Run ALL notebooks in examples/ and report comprehensive results.

Usage:
    python scripts/run_all_notebooks.py [timeout] [skip_pattern ...]

Examples:
    python scripts/run_all_notebooks.py                    # All notebooks, 600s timeout
    python scripts/run_all_notebooks.py 1200               # All notebooks, 1200s timeout
    python scripts/run_all_notebooks.py 600 sticky_rouse   # Skip notebooks matching pattern
"""

import os
import sys
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["FAST_MODE"] = "1"

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

_SETUP_CODE = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "plt.ioff()\n"
)


def _is_setup_cell(cell):
    """Check if a cell is an injected matplotlib setup cell."""
    return cell.get("metadata", {}).get("_runner_setup") or (
        cell.cell_type == "code"
        and "matplotlib.use('Agg')" in cell.source
        and "plt.ioff()" in cell.source
        and len(cell.source.strip().splitlines()) <= 5
    )


def run_notebook(nb_path, timeout=600):
    """Execute a single notebook. Returns (success, error_message, duration)."""
    nb_path = Path(nb_path).resolve()
    start = time.time()

    try:
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        setup_cell = nbformat.v4.new_code_cell(source=_SETUP_CODE)
        setup_cell.metadata["_runner_setup"] = True
        nb.cells.insert(0, setup_cell)

        client = NotebookClient(
            nb, timeout=timeout, kernel_name="python3", allow_errors=False,
        )

        original_cwd = os.getcwd()
        os.chdir(nb_path.parent)
        try:
            client.execute()
        finally:
            os.chdir(original_cwd)

        nb.cells = [c for c in nb.cells if not _is_setup_cell(c)]

        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        return True, None, time.time() - start

    except CellExecutionError as e:
        lines = str(e).split("\n")
        error_lines = [l for l in lines if l.strip()][-5:]
        return False, "\n".join(error_lines), time.time() - start
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", time.time() - start


def _print_result(index, total, name, success, error, duration):
    """Print a single notebook result."""
    print(f"\n[{index}/{total}] {name}...")
    if success:
        print(f"  PASS ({duration:.1f}s)")
    else:
        print(f"  FAIL ({duration:.1f}s)")
        for line in (error or "").split("\n")[:3]:
            print(f"  | {line}")


def discover_notebooks(base_dir):
    """Find all runnable notebooks, excluding archive/checkpoint/hidden files."""
    return sorted(
        nb for nb in Path(base_dir).rglob("*.ipynb")
        if "archive" not in str(nb)
        and not nb.name.startswith(".")
        and "checkpoint" not in str(nb)
    )


def main():
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    skip = set(sys.argv[2:]) if len(sys.argv) > 2 else set()

    notebooks = discover_notebooks("examples")

    print(f"Found {len(notebooks)} notebooks")
    print(f"Timeout: {timeout}s per notebook")
    if skip:
        print(f"Skipping: {skip}")
    print("=" * 70)

    results = []
    for i, nb_path in enumerate(notebooks, 1):
        rel = str(nb_path)
        if any(s in rel for s in skip):
            print(f"\n[{i}/{len(notebooks)}] SKIP {rel}")
            continue

        success, error, duration = run_notebook(nb_path, timeout=timeout)
        _print_result(i, len(notebooks), rel, success, error, duration)
        results.append((rel, success, error, duration))

    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for _, s, _, _ in results if s)
    failed = sum(1 for _, s, _, _ in results if not s)
    total_time = sum(d for _, _, _, d in results)
    print(f"TOTAL: {passed}/{len(results)} passed, {failed} failed, {total_time:.0f}s")

    if failed:
        print("\nFAILED:")
        for name, success, error, _ in results:
            if not success:
                err_line = (error or "").split("\n")[0][:100]
                print(f"  {name}: {err_line}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
