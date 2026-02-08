#!/usr/bin/env python
"""
Dedicated runner for HVNM notebook suite.

Executes all notebooks under examples/hvnm/ with:
- 48-hour per-cell timeout (for Bayesian/NUTS completion)
- Per-notebook stdout/stderr capture
- Warning capture per notebook
- Deterministic working-directory handling
- Timestamped master log + per-notebook logs
- Clear PASS/FAIL/TIMEOUT summary
"""
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError


# Configuration
CELL_TIMEOUT = 172800  # 48 hours in seconds
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HVNM_DIR = PROJECT_ROOT / "examples" / "hvnm"
LOGS_DIR = HVNM_DIR / "_run_logs"


def get_notebooks() -> list[Path]:
    """Find all HVNM notebooks, sorted by name."""
    notebooks = sorted(HVNM_DIR.glob("*.ipynb"))
    # Skip checkpoint files and hidden files
    notebooks = [
        nb for nb in notebooks
        if not nb.name.startswith(".")
        and ".ipynb_checkpoints" not in str(nb)
    ]
    return notebooks


def extract_warnings_from_outputs(nb: nbformat.NotebookNode) -> list[str]:
    """Extract warning messages from notebook cell outputs (stderr streams)."""
    warnings_found = []
    warning_pattern = re.compile(
        r"((?:\S+Warning|DeprecationWarning|FutureWarning|RuntimeWarning"
        r"|UserWarning|PendingDeprecationWarning|ImportWarning"
        r"|ResourceWarning|NumbaWarning)[^\n]*)",
        re.IGNORECASE,
    )
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            text = ""
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                text = output.get("text", "")
            elif output.get("output_type") == "error":
                text = "\n".join(output.get("traceback", []))
            if text:
                for match in warning_pattern.finditer(text):
                    warnings_found.append(match.group(0).strip())
    return warnings_found


def extract_all_stderr(nb: nbformat.NotebookNode) -> str:
    """Extract all stderr output from notebook cells."""
    stderr_parts = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                stderr_parts.append(output.get("text", ""))
    return "\n".join(stderr_parts)


def run_notebook(notebook_path: Path, log_path: Path) -> dict:
    """Run a single notebook and return detailed result dict."""
    result = {
        "notebook": str(notebook_path.relative_to(PROJECT_ROOT)),
        "status": "UNKNOWN",
        "runtime_seconds": 0.0,
        "warnings": [],
        "warning_count": 0,
        "error_traceback": "",
        "stderr": "",
    }

    original_cwd = os.getcwd()
    start_time = time.time()

    try:
        # Change to notebook directory for relative path resolution
        os.chdir(notebook_path.parent.absolute())

        nb = nbformat.read(notebook_path.name, as_version=4)

        # Set headless matplotlib backend via environment variable
        # This is more reliable than cell injection and affects all imports
        os.environ["MPLBACKEND"] = "Agg"

        client = NotebookClient(
            nb,
            timeout=CELL_TIMEOUT,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(notebook_path.parent.absolute())}},
        )
        client.execute()

        # Extract warnings from outputs
        warnings_found = extract_warnings_from_outputs(nb)
        stderr_text = extract_all_stderr(nb)

        result["status"] = "PASS"
        result["warnings"] = warnings_found
        result["warning_count"] = len(warnings_found)
        result["stderr"] = stderr_text

        # Write executed notebook back (preserves outputs for inspection)
        nbformat.write(nb, notebook_path.name)

    except CellTimeoutError as e:
        result["status"] = "TIMEOUT"
        result["error_traceback"] = str(e)
    except CellExecutionError as e:
        result["status"] = "FAIL"
        result["error_traceback"] = str(e)
    except Exception as e:
        result["status"] = "FAIL"
        result["error_traceback"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        os.chdir(original_cwd)
        result["runtime_seconds"] = time.time() - start_time

    # Write per-notebook log
    with open(log_path, "w") as f:
        f.write(f"Notebook: {result['notebook']}\n")
        f.write(f"Status: {result['status']}\n")
        f.write(f"Runtime: {result['runtime_seconds']:.1f}s\n")
        f.write(f"Warnings: {result['warning_count']}\n")
        if result["warnings"]:
            f.write("\nWarnings:\n")
            for w in result["warnings"]:
                f.write(f"  - {w}\n")
        if result["error_traceback"]:
            f.write(f"\nError:\n{result['error_traceback']}\n")
        if result["stderr"]:
            f.write(f"\nStderr:\n{result['stderr']}\n")

    return result


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = get_notebooks()
    if not notebooks:
        print("No HVNM notebooks found!")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    master_log_path = LOGS_DIR / f"master_{timestamp}.log"

    print(f"HVNM Notebook Runner")
    print(f"{'=' * 60}")
    print(f"Found {len(notebooks)} notebooks")
    print(f"Cell timeout: {CELL_TIMEOUT}s ({CELL_TIMEOUT/3600:.0f}h)")
    print(f"Logs: {LOGS_DIR}")
    print(f"{'=' * 60}\n")

    all_results = []
    suite_start = time.time()

    for i, notebook in enumerate(notebooks, 1):
        nb_name = notebook.stem
        print(f"[{i}/{len(notebooks)}] {nb_name}...", end=" ", flush=True)

        log_path = LOGS_DIR / f"{nb_name}.log"
        result = run_notebook(notebook, log_path)
        all_results.append(result)

        status_icon = {"PASS": "PASS", "FAIL": "FAIL", "TIMEOUT": "TIMEOUT"}[result["status"]]
        runtime = format_duration(result["runtime_seconds"])
        warn_str = f" [{result['warning_count']} warnings]" if result["warning_count"] else ""
        print(f"{status_icon} ({runtime}){warn_str}")

        if result["status"] != "PASS":
            # Show first few lines of error
            error_lines = result["error_traceback"].split("\n")[:5]
            for line in error_lines:
                print(f"    {line[:120]}")

    suite_runtime = time.time() - suite_start

    # Summary
    passed = [r for r in all_results if r["status"] == "PASS"]
    failed = [r for r in all_results if r["status"] == "FAIL"]
    timed_out = [r for r in all_results if r["status"] == "TIMEOUT"]
    total_warnings = sum(r["warning_count"] for r in all_results)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total:    {len(all_results)}")
    print(f"Passed:   {len(passed)}")
    print(f"Failed:   {len(failed)}")
    print(f"Timeout:  {len(timed_out)}")
    print(f"Warnings: {total_warnings}")
    print(f"Runtime:  {format_duration(suite_runtime)}")

    # Write master log
    with open(master_log_path, "w") as f:
        f.write(f"HVNM Notebook Suite Run — {timestamp}\n")
        f.write(f"{'=' * 60}\n\n")
        for r in all_results:
            f.write(f"{r['notebook']}: {r['status']} ({format_duration(r['runtime_seconds'])})")
            if r["warning_count"]:
                f.write(f" [{r['warning_count']} warnings]")
            f.write("\n")
        f.write(f"\nTotal runtime: {format_duration(suite_runtime)}\n")

    # Write results as JSON for programmatic access
    json_path = LOGS_DIR / f"results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nMaster log: {master_log_path}")
    print(f"Results JSON: {json_path}")

    return 0 if (not failed and not timed_out and total_warnings == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
