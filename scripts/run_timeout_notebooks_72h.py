#!/usr/bin/env python
"""
Run the 6 timeout notebooks with 72-hour timeout.

Target notebooks (from issue inventory):
- HL: 01_hl_flow_curve, 02_hl_relaxation, 03_hl_creep, 05_hl_startup, 06_hl_laos
- STZ: 02_stz_startup_shear

These notebooks timeout on Bayesian inference due to:
- HL: forward-mode AD + Volterra solver (~30-60 mins per NUTS sample)
- STZ: ODE-based NUTS (~15-30 mins for startup)

Usage:
    uv run python scripts/run_timeout_notebooks_72h.py
    uv run python scripts/run_timeout_notebooks_72h.py --single 01_hl_flow_curve.ipynb
    uv run python scripts/run_timeout_notebooks_72h.py --dry-run
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

# Notebooks that timeout in normal runs
TIMEOUT_NOTEBOOKS = [
    ("hl", "01_hl_flow_curve.ipynb"),
    ("hl", "02_hl_relaxation.ipynb"),
    ("hl", "03_hl_creep.ipynb"),
    ("hl", "05_hl_startup.ipynb"),
    ("hl", "06_hl_laos.ipynb"),
    ("stz", "02_stz_startup_shear.ipynb"),
]

# 72 hours in seconds
TIMEOUT_72H = 72 * 60 * 60  # 259200 seconds


def setup_environment() -> dict[str, str]:
    """Set up deterministic environment variables."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["JAX_ENABLE_X64"] = "True"
    env["PYTHONHASHSEED"] = "42"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
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
    log_file = log_dir / f"{notebook_path.stem}_72h.log"

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

            print(f"    Executing... (timeout: {timeout/3600:.1f} hours)")
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
        result["traceback"] = str(e)[:2000]

    except CellExecutionError as e:
        result["status"] = "FAIL"
        result["error"] = str(e)[:2000]
        result["traceback"] = traceback.format_exc()

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:1000]}"
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

    return "other_error"


def main():
    parser = argparse.ArgumentParser(
        description="Run timeout notebooks with 72-hour timeout"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_72H,
        help=f"Timeout per notebook in seconds (default: {TIMEOUT_72H} = 72 hours)",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Run a single notebook by name (e.g., 01_hl_flow_curve.ipynb)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List notebooks without executing",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    log_dir = project_root / "examples" / "_run_logs" / "72h_run"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filter notebooks if --single specified
    notebooks_to_run = TIMEOUT_NOTEBOOKS
    if args.single:
        notebooks_to_run = [
            (suite, nb)
            for suite, nb in TIMEOUT_NOTEBOOKS
            if nb == args.single or nb.replace(".ipynb", "") == args.single
        ]
        if not notebooks_to_run:
            print(f"ERROR: Notebook '{args.single}' not in timeout list")
            print(f"Available: {[nb for _, nb in TIMEOUT_NOTEBOOKS]}")
            return 1

    print("=" * 70)
    print("72-HOUR TIMEOUT NOTEBOOK RUNNER")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Log directory: {log_dir}")
    print(f"Timeout: {args.timeout/3600:.1f} hours per notebook")
    print(f"Notebooks to run: {len(notebooks_to_run)}")
    print()

    for suite, nb in notebooks_to_run:
        print(f"  [{suite.upper()}] {nb}")

    if args.dry_run:
        print("\n[DRY RUN] No notebooks executed.")
        return 0

    print()
    print("=" * 70)
    print("EXECUTION")
    print("=" * 70)

    # Set up environment
    env = setup_environment()
    for k, v in env.items():
        os.environ[k] = v

    results = []
    start_time = datetime.now()

    for i, (suite, nb_name) in enumerate(notebooks_to_run, 1):
        nb_path = project_root / "examples" / suite / nb_name

        if not nb_path.exists():
            print(f"\n[{i}/{len(notebooks_to_run)}] SKIP: {nb_path} not found")
            continue

        print(f"\n[{i}/{len(notebooks_to_run)}] {suite.upper()}/{nb_name}")
        print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        result = run_notebook(nb_path, args.timeout, log_dir)
        result["category"] = categorize_issue(result)
        results.append(result)

        status_emoji = {
            "PASS": "✓",
            "FAIL": "✗",
            "TIMEOUT": "⏰",
        }.get(result["status"], "?")

        print(
            f"    Status: {status_emoji} {result['status']} ({result['runtime_human']})"
        )
        print(f"    Category: {result['category']}")
        if result["warnings"]:
            print(f"    Warnings: {len(result['warnings'])}")
        if result["error"]:
            error_preview = result["error"][:100].replace("\n", " ")
            print(f"    Error: {error_preview}...")

    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()

    # Save master results JSON
    master_log = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(master_log, "w") as f:
        json.dump(
            {
                "run_type": "72h_timeout_notebooks",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
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
    with open(summary_path, "w") as f:
        f.write("# 72-Hour Timeout Run Summary\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Total Runtime:** {total_runtime/3600:.2f} hours\n\n")

        f.write("## Results\n\n")
        f.write("| Notebook | Status | Runtime | Category |\n")
        f.write("|----------|--------|---------|----------|\n")
        for r in results:
            f.write(
                f"| {r['notebook']} | {r['status']} | {r['runtime_human']} | {r['category']} |\n"
            )

        f.write("\n## Statistics\n\n")
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")
        f.write(f"- **PASS:** {passed}\n")
        f.write(f"- **FAIL:** {failed}\n")
        f.write(f"- **TIMEOUT:** {timeouts}\n")

        if any(r["status"] != "PASS" for r in results):
            f.write("\n## Issues\n\n")
            for r in results:
                if r["status"] != "PASS" or r["warnings"]:
                    f.write(f"### {r['notebook']}\n\n")
                    f.write(f"- **Status:** {r['status']}\n")
                    f.write(f"- **Category:** {r['category']}\n")
                    f.write(f"- **Runtime:** {r['runtime_human']}\n")
                    if r["error"]:
                        f.write(f"\n```\n{r['error'][:1000]}\n```\n")
                    f.write("\n")

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {total_runtime/3600:.2f} hours")
    print(f"PASS: {passed}, FAIL: {failed}, TIMEOUT: {timeouts}")
    print(f"Results saved to: {master_log}")
    print(f"Summary: {summary_path}")

    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
