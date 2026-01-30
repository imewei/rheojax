#!/usr/bin/env python
"""Test notebooks and capture errors."""
import sys
import nbformat
from nbclient import NotebookClient
from pathlib import Path
import traceback

def test_notebook(nb_path: str, timeout: int = 120) -> dict:
    """Test a single notebook and return result."""
    import os
    # Change to notebook's directory for relative paths to work
    nb_abs_path = Path(nb_path).resolve()
    original_cwd = os.getcwd()
    try:
        os.chdir(nb_abs_path.parent)
        nb = nbformat.read(nb_abs_path, as_version=4)
        client = NotebookClient(nb, timeout=timeout, kernel_name='python3')
        client.execute()
        return {"status": "PASS", "error": None}
    except Exception as e:
        # Extract error details
        error_msg = str(e)
        # Try to get the cell that failed
        if hasattr(e, 'traceback'):
            tb = ''.join(e.traceback[-5:]) if len(e.traceback) > 5 else ''.join(e.traceback)
        else:
            tb = error_msg[:500]
        return {"status": "FAIL", "error": tb}
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    notebooks = sys.argv[1:]
    results = {}

    for nb_path in notebooks:
        print(f"Testing: {nb_path}...", end=" ", flush=True)
        result = test_notebook(nb_path)
        results[nb_path] = result
        print(result["status"])
        if result["status"] == "FAIL":
            print(f"  ERROR: {result['error'][:200]}...")

    # Summary
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    failed = sum(1 for r in results.values() if r["status"] == "FAIL")
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print(f"\nFailed notebooks:")
        for nb, r in results.items():
            if r["status"] == "FAIL":
                print(f"  - {nb}")
