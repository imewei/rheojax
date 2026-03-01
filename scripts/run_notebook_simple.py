#!/usr/bin/env python
"""Simple notebook runner that executes cells sequentially."""

import sys
import os

# Set up environment
os.environ["MPLBACKEND"] = "Agg"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["FAST_MODE"] = "1"

import nbformat
from nbclient import NotebookClient


def run_notebook(nb_path, timeout=300):
    """Execute notebook cells sequentially."""
    print(f"Loading notebook: {nb_path}")

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Add setup cell
    setup_code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.ioff()\n"
    )
    setup_cell = nbformat.v4.new_code_cell(source=setup_code)
    nb.cells.insert(0, setup_cell)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
    )

    print("Executing notebook...")
    client.execute()

    print("Saving notebook...")
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook.ipynb>")
        sys.exit(1)

    run_notebook(sys.argv[1])
