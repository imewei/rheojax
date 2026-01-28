#!/usr/bin/env python
"""Test FIKH notebooks 02-06 to identify errors."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

NOTEBOOKS = [
    "02_fikh_startup_shear.ipynb",
    "03_fikh_stress_relaxation.ipynb",
    "04_fikh_creep.ipynb",
    "05_fikh_saos.ipynb",
    "06_fikh_laos.ipynb",
]

def test_notebook(nb_path: Path) -> tuple[bool, str]:
    """Test a notebook and return (success, error_message)."""
    os.chdir(nb_path.parent)

    try:
        nb = nbformat.read(nb_path.name, as_version=4)
        client = NotebookClient(nb, timeout=180, kernel_name='python3')
        client.execute()
        return True, ""
    except CellExecutionError as e:
        # Extract cell number and error
        error_msg = str(e)
        lines = error_msg.split('\n')
        # Get first few lines for context
        context = '\n'.join(lines[:20])
        return False, context
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {str(e)[:500]}"

if __name__ == "__main__":
    nb_dir = Path(__file__).parent

    for nb_name in NOTEBOOKS:
        print(f"\n{'='*70}")
        print(f"Testing: {nb_name}")
        print('='*70)

        nb_path = nb_dir / nb_name
        success, error = test_notebook(nb_path)

        if success:
            print(f"✓ SUCCESS")
        else:
            print(f"✗ FAILED")
            print(f"\nError details:\n{error}")
