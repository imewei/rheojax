#!/usr/bin/env python
"""Script to run all notebooks and report errors."""
import sys
import traceback
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def run_notebook(notebook_path: Path) -> tuple[bool, str]:
    """Run a single notebook and return success status and error message."""
    import os
    # Save current directory
    original_cwd = os.getcwd()
    try:
        # Change to notebook directory for relative paths to work
        os.chdir(notebook_path.parent.absolute())

        nb = nbformat.read(notebook_path.name, as_version=4)
        client = NotebookClient(
            nb,
            timeout=600,  # 10 minutes per cell
            kernel_name='python3',
            allow_errors=False
        )
        client.execute()
        nbformat.write(nb, notebook_path.name)
        return True, ""
    except CellExecutionError as e:
        return False, f"Cell execution error:\n{e}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        # Restore original directory
        os.chdir(original_cwd)


def main():
    examples_dir = Path("examples")
    notebooks = sorted(examples_dir.rglob("*.ipynb"))

    # Skip hidden files and archive directories
    notebooks = [
        nb for nb in notebooks
        if not any(part.startswith('.') for part in nb.parts)
        and 'archive' not in str(nb).lower()
    ]

    print(f"Found {len(notebooks)} notebooks to run\n")

    results = {"passed": [], "failed": []}

    for i, notebook in enumerate(notebooks, 1):
        rel_path = notebook.relative_to(examples_dir)
        print(f"[{i}/{len(notebooks)}] Running {rel_path}...", end=" ", flush=True)

        success, error = run_notebook(notebook)

        if success:
            print("✓")
            results["passed"].append(str(rel_path))
        else:
            print("✗")
            print(f"  Error: {error[:500]}...")
            results["failed"].append((str(rel_path), error))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {len(results['passed'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["failed"]:
        print("\nFailed notebooks:")
        for path, error in results["failed"]:
            print(f"\n  {path}")
            # Show first few lines of error
            error_lines = error.split('\n')[:10]
            for line in error_lines:
                print(f"    {line[:100]}")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
