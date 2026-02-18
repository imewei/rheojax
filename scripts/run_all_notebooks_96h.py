#!/usr/bin/env python
"""
Run ALL notebooks in examples/ with a 96-hour timeout.

Goal: Ensure complete execution of slow Bayesian analysis notebooks
and catch deep root cause errors (divergences, numerical instabilities).

Usage:
    uv run python scripts/run_all_notebooks_96h.py
    uv run python scripts/run_all_notebooks_96h.py --dry-run
    uv run python scripts/run_all_notebooks_96h.py --subdir examples/fluidity
"""

import argparse
import json
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

# 96 hours in seconds
TIMEOUT_96H = 96 * 60 * 60  # 345,600 seconds


def setup_environment() -> dict[str, str]:
    """Set up deterministic environment variables."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["JAX_ENABLE_X64"] = "True"
    env["PYTHONHASHSEED"] = "42"
    # Avoid preallocating all GPU memory to allow other processes (or long runs) to coexist better
    # and avoid OOMs when creating many kernels sequentially
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    # Disable FAST_MODE to force full/slow execution (e.g. NLSQ+Bayesian)
    env["FAST_MODE"] = "0"

    return env


def run_notebook(
    notebook_path: Path,
    timeout: int,
    log_dir: Path,
) -> dict[str, Any]:
    """
    Run a single notebook with comprehensive logging.

    Returns dict with status, runtime, errors, warnings.
    """
    start_time = datetime.now()
    result = {
        "notebook": notebook_path.name,
        "path": str(notebook_path),
        "suite": notebook_path.parent.name,
        "status": "UNKNOWN",
        "runtime_seconds": 0,
        "runtime_human": "",
        "error": None,
        "traceback": None,
        "warnings": [],
        "cell_count": 0,
        "executed_cells": 0,
        "start_time": start_time.isoformat(),
        "end_time": None,
    }

    original_cwd = os.getcwd()
    notebook_dir = notebook_path.parent.absolute()
    log_file = log_dir / f"{notebook_path.stem}.log"

    try:
        os.chdir(notebook_dir)

        nb = nbformat.read(notebook_path.name, as_version=4)
        result["cell_count"] = len([c for c in nb.cells if c.cell_type == "code"])

        captured_warnings = []

        def warning_handler(message, category, filename, lineno, file=None, line=None):
            warning_str = f"{category.__name__}: {message} ({filename}:{lineno})"
            captured_warnings.append(warning_str)

        old_showwarning = warnings.showwarning
        warnings.showwarning = warning_handler

        try:
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name="python3",
                allow_errors=False,
                resources={"metadata": {"path": str(notebook_dir)}},
            )

            # print(f"    Executing... (timeout: {timeout/3600:.1f} hours)")
            client.execute()

            # Save executed notebook
            nbformat.write(nb, notebook_path.name)

            result["status"] = "PASS"
            result["executed_cells"] = result["cell_count"]

        finally:
            warnings.showwarning = old_showwarning

        # Capture warnings from outputs
        for cell in nb.cells:
            if cell.cell_type == "code" and hasattr(cell, "outputs"):
                for output in cell.outputs:
                    if (
                        output.get("output_type") == "stream"
                        and output.get("name") == "stderr"
                    ):
                        text = output.get("text", "")
                        if text.strip():
                            for line in text.split("\n"):
                                line = line.strip()
                                if line and ("Warning" in line or "warning" in line):
                                    captured_warnings.append(line)

        result["warnings"] = captured_warnings

    except CellTimeoutError as e:
        result["status"] = "TIMEOUT"
        result["error"] = f"Cell timeout after {timeout/3600:.1f} hours"
        result["traceback"] = str(e)[:4000]

    except CellExecutionError as e:
        result["status"] = "FAIL"
        result["error"] = str(e)[:4000]
        result["traceback"] = traceback.format_exc()

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        result["traceback"] = traceback.format_exc()

    finally:
        os.chdir(original_cwd)
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["runtime_seconds"] = (end_time - start_time).total_seconds()

        # Human-readable runtime
        runtime_sec = result["runtime_seconds"]
        if runtime_sec >= 3600:  # type: ignore[operator]
            hours = runtime_sec / 3600  # type: ignore[operator]
            result["runtime_human"] = f"{hours:.2f} hours"
        elif runtime_sec >= 60:  # type: ignore[operator]
            minutes = runtime_sec / 60  # type: ignore[operator]
            result["runtime_human"] = f"{minutes:.1f} minutes"
        else:
            result["runtime_human"] = f"{runtime_sec:.1f} seconds"

        # Write detailed log
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"NOTEBOOK: {notebook_path.name}\n")
            f.write(f"SUITE: {result['suite']}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Started: {start_time.isoformat()}\n")
            f.write(f"Finished: {end_time.isoformat()}\n")
            f.write(f"Runtime: {result['runtime_human']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(
                f"Cells executed: {result['executed_cells']}/{result['cell_count']}\n"
            )
            f.write(f"Warnings: {len(result['warnings'])}\n\n")  # type: ignore[arg-type]

            if result["warnings"]:
                f.write("=== WARNINGS ===\n")
                for w in result["warnings"][:100]:  # type: ignore[index]
                    f.write(f"  {w}\n")
                f.write("\n")

            if result["error"]:
                f.write("=== ERROR ===\n")
                f.write(f"{result['error']}\n\n")

            if result["traceback"]:
                f.write("=== TRACEBACK ===\n")
                f.write(f"{result['traceback']}\n")

    return result


def categorize_issue(result: dict[str, Any]) -> str:
    """Categorize the issue type."""
    if result["status"] == "PASS":
        if result["warnings"]:
            warnings_text = " ".join(result["warnings"][:20])
            if "divergence" in warnings_text.lower():
                return "numpyro_divergence"
            if (
                "DeprecationWarning" in warnings_text
                or "FutureWarning" in warnings_text
            ):
                return "deprecation_warning"
            return "pass_with_warnings"
        return "clean_pass"

    if result["status"] == "TIMEOUT":
        return "performance_timeout"

    error_text = (result.get("error") or "") + (result.get("traceback") or "")
    error_lower = error_text.lower()

    if "import" in error_lower or "module" in error_lower:
        return "import_error"
    if "jax" in error_lower or "xla" in error_lower:
        return "jax_error"
    if "nan" in error_lower or "inf" in error_lower:
        return "numerical_error"
    if "numpyro" in error_lower or "diverge" in error_lower:
        return "numpyro_error"
    if "stz" in error_lower or "dynamcis" in error_lower:
        return "physics_error"

    return "other_error"


def main():
    parser = argparse.ArgumentParser(
        description="Run ALL notebooks with 96-hour timeout for deep debugging"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_96H,
        help=f"Timeout per notebook in seconds (default: {TIMEOUT_96H} = 96 hours)",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Run notebooks only in this subdirectory (e.g. examples/fluidity)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List notebooks without executing",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Determine search root
    if args.subdir:
        search_root = project_root / args.subdir
        if not search_root.exists():
            print(f"ERROR: subdirectory {search_root} does not exist", file=sys.stderr)
            return 1
    else:
        search_root = project_root / "examples"

    # Log directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "examples" / "_run_logs" / f"96h_run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Find notebooks recursively
    print(f"Scanning for notebooks in {search_root}...")
    notebooks = sorted(search_root.rglob("*.ipynb"))

    # Filter out checkpoints, logs, archives, and templates
    notebooks_to_run = [
        nb
        for nb in notebooks
        if not any(part.startswith(".") for part in nb.parts)
        and "archive" not in str(nb).lower()
        and "_run_logs" not in str(nb)
        and ".ipynb_checkpoints" not in str(nb)
        and "template" not in nb.name.lower()
        and "golden_data"
        not in str(nb)  # often contains support files, not runnable nbs? check later
    ]

    # Special check: verify golden_data doesn't contain notebooks we *should* run.
    # Usually strictly data, but safety first.
    # Actually, let's keep it consistent with run_notebooks.py exclusions if possible.

    if not notebooks_to_run:
        print(f"No notebooks found under {search_root}")
        return 0

    print("=" * 70)
    print("96-HOUR TIMEOUT NOTEBOOK RUNNER")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Log directory: {log_dir}")
    print(f"Timeout: {args.timeout/3600:.1f} hours per notebook")
    print(f"Notebooks to run: {len(notebooks_to_run)}")
    print()

    if args.dry_run:
        print("[DRY RUN] Notebooks identified:")
        for nb in notebooks_to_run:
            print(f"  - {nb.relative_to(project_root)}")
        return 0

    print("=" * 70)
    print("EXECUTION START")
    print("=" * 70)

    # Set up environment
    env = setup_environment()
    for k, v in env.items():
        os.environ[k] = v

    results = []
    start_time_total = datetime.now()

    for i, nb_path in enumerate(notebooks_to_run, 1):
        nb_rel = nb_path.relative_to(project_root)
        print(f"[{i}/{len(notebooks_to_run)}] {nb_rel} ...", end=" ", flush=True)

        # We need to print start separately because it might hang for 96h

        result = run_notebook(nb_path, args.timeout, log_dir)
        result["category"] = categorize_issue(result)
        results.append(result)

        status_emoji = {
            "PASS": "✓",
            "FAIL": "✗",
            "TIMEOUT": "⏰",
        }.get(result["status"], "?")

        # Overwrite previous line or print new line?
        # Since we used end=" ", we are on the same line.
        # But if the notebook printed to stdout extensively (it shouldn't, capturing is mostly silent in run_notebook),
        # it might be messy. run_notebook does print "Executing..." if uncommented.

        print(f"{status_emoji} ({result['runtime_human']})")

        # Log to stdout immediately if fail
        if result["status"] != "PASS":
            print(f"    -> {result['category']}")
            if result["error"]:
                first_line = result["error"].split("\n")[0][:80]
                print(f"    -> Error: {first_line}...")

    end_time_total = datetime.now()
    total_runtime = (end_time_total - start_time_total).total_seconds()

    # Save master results JSON
    master_log = log_dir / "master_run_stats.json"
    with open(master_log, "w") as f:
        json.dump(
            {
                "run_type": "96h_full_sweep",
                "start_time": start_time_total.isoformat(),
                "end_time": end_time_total.isoformat(),
                "total_runtime_seconds": total_runtime,
                "timeout_per_notebook": args.timeout,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    # Generate summary report
    summary_path = log_dir / "summary.md"
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")

    with open(summary_path, "w") as f:
        f.write("# 96-Hour Full Run Summary\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Total Runtime:** {total_runtime/3600:.2f} hours\n")
        f.write(f"**Stats:** PASS={passed} | FAIL={failed} | TIMEOUT={timeouts}\n\n")

        f.write("## Results\n\n")
        f.write("| Notebook | Status | Runtime | Category |\n")
        f.write("|----------|--------|---------|----------|\n")
        for r in results:
            # Shorten name for table
            name = r["notebook"]
            f.write(
                f"| {name} | {r['status']} | {r['runtime_human']} | {r['category']} |\n"
            )

        if failed or timeouts:
            f.write("\n## Failures & Timeouts\n\n")
            for r in results:
                if r["status"] != "PASS":
                    f.write(f"### {r['notebook']}\n")
                    f.write(f"- Status: **{r['status']}**\n")
                    f.write(f"- Category: `{r['category']}`\n")
                    f.write(
                        f"- Log: `{log_dir}/{r['notebook'].replace('.ipynb', '.log')}`\n"
                    )
                    if r["error"]:
                        f.write(f"```\n{r['error'][:500]}\n```\n")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {total_runtime/3600:.2f} hours")
    print(f"PASS: {passed}, FAIL: {failed}, TIMEOUT: {timeouts}")
    print(f"Results saved to: {master_log}")
    print(f"Summary: {summary_path}")

    return 0 if failed == 0 and timeouts == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
