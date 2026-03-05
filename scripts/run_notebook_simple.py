#!/usr/bin/env python
"""Simple notebook runner that executes cells sequentially."""

import os
import sys

# Set up environment
os.environ["MPLBACKEND"] = "Agg"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["FAST_MODE"] = "1"

import nbformat
from nbclient import NotebookClient


def run_notebook(nb_path, timeout=600):
    """Execute notebook cells sequentially."""
    from pathlib import Path

    nb_path = Path(nb_path).resolve()
    print(f"Loading notebook: {nb_path}")

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Add setup cell for headless matplotlib
    setup_code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.ioff()\n"
    )
    setup_cell = nbformat.v4.new_code_cell(source=setup_code)
    # Tag it so we can reliably remove it before saving
    setup_cell.metadata["_runner_setup"] = True
    nb.cells.insert(0, setup_cell)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
    )

    # Change CWD to notebook directory so relative paths work
    original_cwd = os.getcwd()
    os.chdir(nb_path.parent)
    print(f"Working directory: {nb_path.parent}")
    try:
        print("Executing notebook...")
        client.execute()
    finally:
        os.chdir(original_cwd)

    # Remove all injected setup cells before saving (tagged + legacy untagged)
    nb.cells = [
        c for c in nb.cells
        if not (
            c.get("metadata", {}).get("_runner_setup")
            or (
                c.cell_type == "code"
                and "matplotlib.use('Agg')" in c.source
                and "plt.ioff()" in c.source
                and len(c.source.strip().splitlines()) <= 5
            )
        )
    ]

    print("Saving notebook...")
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook.ipynb>")
        sys.exit(1)

    run_notebook(sys.argv[1])
