#!/usr/bin/env python
"""
Dedicated runner for IKH and FIKH notebooks with long timeout support.

Features:
- Per-notebook timeout (default 86400s = 24 hours)
- Per-notebook logging with warnings capture
- Issue categorization
- JSON summary output
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, DeadKernelError

# Default timeout: 24 hours per notebook
DEFAULT_TIMEOUT = 86400

# IKH and FIKH notebook directories
IKH_DIR = Path("examples/ikh")
FIKH_DIR = Path("examples/fikh")


def categorize_issue(error: str, warnings_list: list[str]) -> str:
    """Categorize the issue based on error/warning patterns."""
    error_lower = error.lower() if error else ""

    # Check errors
    if "modulenotfound" in error_lower or "importerror" in error_lower:
        return "import_error"
    if "filenotfound" in error_lower or "no such file" in error_lower:
        return "paths_data"
    if "shape" in error_lower or "dtype" in error_lower or "broadcast" in error_lower:
        return "shape_dtype"
    if "jax" in error_lower and ("compile" in error_lower or "device" in error_lower):
        return "jax_compilation"
    if "numpyro" in error_lower or "divergence" in error_lower or "nuts" in error_lower:
        return "numpyro_sampling"
    if "nan" in error_lower or "inf" in error_lower or "overflow" in error_lower:
        return "numerical_instability"
    if "memory" in error_lower or "oom" in error_lower:
        return "resource_oom"
    if "deadkernel" in error_lower or "kernel died" in error_lower:
        return "kernel_death"
    if "timeout" in error_lower:
        return "timeout"
    if "attribute" in error_lower:
        return "attribute_error"
    if "syntax" in error_lower:
        return "syntax_error"

    # Check warnings
    for w in warnings_list:
        w_lower = w.lower()
        if "deprecat" in w_lower:
            return "deprecation_warning"
        if "future" in w_lower:
            return "future_warning"
        if "runtimewarning" in w_lower:
            return "numerical_warning"
        if "complexwarning" in w_lower:
            return "numerical_warning"
        if "userwarning" in w_lower:
            return "other_warning"

    if error:
        return "other_error"
    if warnings_list:
        return "other_warning"

    return "clean"


def run_notebook(
    notebook_path: Path,
    log_dir: Path,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Run a single notebook and capture results."""
    result = {
        "notebook": notebook_path.name,
        "status": "PASS",
        "runtime_seconds": 0.0,
        "error": None,
        "traceback": None,
        "warnings": [],
        "warnings_count": 0,
        "cell_count": 0,
        "executed_cells": 0,
        "category": "clean",
    }

    # Capture warnings
    captured_warnings = []

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        warning_str = f"{filename}:{lineno}: {category.__name__}: {message}"
        captured_warnings.append(warning_str)

    old_showwarning = warnings.showwarning
    warnings.showwarning = warning_handler

    start_time = datetime.now()
    log_file = log_dir / f"{notebook_path.stem}.log"

    original_cwd = os.getcwd()

    try:
        # Change to notebook directory
        os.chdir(notebook_path.parent.absolute())

        # Read notebook
        nb = nbformat.read(notebook_path.name, as_version=4)
        result["cell_count"] = len([c for c in nb.cells if c.cell_type == "code"])

        # Execute
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=False,
        )
        client.execute()

        # Count executed cells
        result["executed_cells"] = result["cell_count"]

        # Save executed notebook
        nbformat.write(nb, notebook_path.name)

    except CellExecutionError as e:
        result["status"] = "FAIL"
        result["error"] = str(e)[:2000]
        result["traceback"] = str(e)
        # Count cells executed before failure
        result["executed_cells"] = sum(
            1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count")
        )

    except DeadKernelError as e:
        result["status"] = "FAIL"
        result["error"] = f"Kernel died: {e}"
        result["traceback"] = str(e)

    except TimeoutError as e:
        result["status"] = "TIMEOUT"
        result["error"] = f"Timeout after {timeout}s"
        result["traceback"] = str(e)

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = str(e)

    finally:
        os.chdir(original_cwd)
        warnings.showwarning = old_showwarning

        end_time = datetime.now()
        result["runtime_seconds"] = (end_time - start_time).total_seconds()
        result["warnings"] = captured_warnings
        result["warnings_count"] = len(captured_warnings)
        result["category"] = categorize_issue(result["error"], captured_warnings)  # type: ignore[arg-type]

        # Write per-notebook log
        with open(log_file, "w") as f:
            f.write(f"Notebook: {notebook_path.absolute()}\n")
            f.write(f"Started: {start_time.isoformat()}\n")
            f.write(f"Finished: {end_time.isoformat()}\n")
            f.write(f"Runtime: {result['runtime_seconds']:.2f}s\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Cells: {result['executed_cells']}/{result['cell_count']}\n")
            f.write(f"Warnings: {result['warnings_count']}\n")

            if result["warnings"]:
                f.write("\n=== WARNINGS ===\n")
                for w in result["warnings"][:20]:  # type: ignore[index]
                    f.write(f"{w}\n")

            if result["error"]:
                f.write("\n=== ERROR ===\n")
                f.write(result["error"][:5000])  # type: ignore[index]
                f.write("\n")

            if result["traceback"] and result["status"] == "FAIL":
                f.write("\n=== TRACEBACK ===\n")
                f.write(result["traceback"][:10000])  # type: ignore[index]
                f.write("\n")

    return result


def run_suite(
    suite_name: str,
    suite_dir: Path,
    notebooks: list[Path],
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    """Run all notebooks in a suite."""
    log_dir = suite_dir / "_run_logs"
    log_dir.mkdir(exist_ok=True)

    results = []
    total = len(notebooks)

    print(f"\n{'='*60}")
    print(f"Running {suite_name} suite: {total} notebooks")
    print(f"Timeout per notebook: {timeout}s ({timeout/3600:.1f} hours)")
    print(f"{'='*60}\n")

    for i, nb_path in enumerate(notebooks, 1):
        print(f"[{i}/{total}] {nb_path.name}...", end=" ", flush=True)

        result = run_notebook(nb_path, log_dir, timeout)
        results.append(result)

        status_icon = "✓" if result["status"] == "PASS" else "✗"
        warn_str = (
            f" ({result['warnings_count']} warnings)"
            if result["warnings_count"] > 0
            else ""
        )
        print(f"{status_icon} {result['runtime_seconds']:.1f}s{warn_str}")

        if result["status"] != "PASS":
            print(f"    Category: {result['category']}")
            if result["error"]:
                error_preview = result["error"][:200].replace("\n", " ")
                print(f"    Error: {error_preview}...")

    return results


def write_issue_inventory(
    ikh_results: list[dict],
    fikh_results: list[dict],
    output_path: Path,
):
    """Write combined issue inventory in markdown format."""
    all_results = [("ikh", r) for r in ikh_results] + [
        ("fikh", r) for r in fikh_results
    ]

    # Count issues
    issues = [
        (suite, r)
        for suite, r in all_results
        if r["status"] != "PASS" or r["warnings_count"] > 0
    ]

    with open(output_path, "w") as f:
        f.write("# IKH + FIKH Notebook Issue Inventory\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Suite | Notebook | Status | Runtime | Warnings | Category |\n")
        f.write("|-------|----------|--------|---------|----------|----------|\n")

        for suite, r in all_results:
            f.write(
                f"| {suite} | {r['notebook']} | {r['status']} | {r['runtime_seconds']:.1f}s | {r['warnings_count']} | {r['category']} |\n"
            )

        # Detailed issues
        if issues:
            f.write("\n## Detailed Issues\n\n")

            for suite, r in issues:
                f.write(f"### {r['notebook']}\n\n")
                f.write(f"- **Suite**: {suite}\n")
                f.write(f"- **Status**: {r['status']}\n")
                f.write(f"- **Runtime**: {r['runtime_seconds']:.1f}s\n")
                f.write(f"- **Warnings**: {r['warnings_count']}\n")
                f.write(f"- **Category**: {r['category']}\n")

                if r["warnings"]:
                    f.write("\n**Top Warnings**:\n")
                    for w in r["warnings"][:5]:
                        f.write(f"- `{w[:150]}`\n")

                if r["error"]:
                    f.write("\n**Error**:\n```\n")
                    f.write(r["error"][:1000])
                    f.write("\n```\n")

                if r["traceback"] and r["status"] == "FAIL":
                    # Extract relevant frames
                    tb_lines = r["traceback"].split("\n")
                    relevant = [
                        line
                        for line in tb_lines
                        if "rheojax" in line.lower()
                        or "Error" in line
                        or "Exception" in line
                    ][:10]
                    if relevant:
                        f.write("\n**Traceback (relevant)**:\n```\n")
                        f.write("\n".join(relevant))
                        f.write("\n```\n")

                # Reproduction command
                f.write("\n**Reproduce**:\n```bash\n")
                f.write(
                    f"cd /Users/b80985/Projects/rheojax && uv run python scripts/run_ikh_fikh_notebooks.py --suite {suite} --single {r['notebook']}\n"
                )
                f.write("```\n\n")

    print(f"\nIssue inventory written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run IKH and FIKH notebooks")
    parser.add_argument(
        "--suite",
        choices=["ikh", "fikh", "both"],
        default="both",
        help="Which suite to run",
    )
    parser.add_argument(
        "--single", type=str, default=None, help="Run only a single notebook (by name)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per notebook in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only generate inventory from existing logs",
    )

    args = parser.parse_args()

    # Collect notebooks
    ikh_notebooks = (
        sorted(IKH_DIR.glob("*.ipynb")) if args.suite in ["ikh", "both"] else []
    )
    fikh_notebooks = (
        sorted(FIKH_DIR.glob("*.ipynb")) if args.suite in ["fikh", "both"] else []
    )

    # Filter for single notebook
    if args.single:
        ikh_notebooks = [nb for nb in ikh_notebooks if nb.name == args.single]
        fikh_notebooks = [nb for nb in fikh_notebooks if nb.name == args.single]

        if not ikh_notebooks and not fikh_notebooks:
            print(f"Error: Notebook '{args.single}' not found in selected suite(s)")
            return 1

    # Run suites
    ikh_results = []
    fikh_results = []

    if ikh_notebooks:
        ikh_results = run_suite("IKH", IKH_DIR, ikh_notebooks, args.timeout)

    if fikh_notebooks:
        fikh_results = run_suite("FIKH", FIKH_DIR, fikh_notebooks, args.timeout)

    # Write summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = ikh_results + fikh_results
    summary = {
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_runtime_seconds": sum(r["runtime_seconds"] for r in all_results),
        "summary": {
            "clean_pass": sum(
                1
                for r in all_results
                if r["status"] == "PASS" and r["warnings_count"] == 0
            ),
            "pass_with_warnings": sum(
                1
                for r in all_results
                if r["status"] == "PASS" and r["warnings_count"] > 0
            ),
            "fail": sum(1 for r in all_results if r["status"] == "FAIL"),
            "timeout": sum(1 for r in all_results if r["status"] == "TIMEOUT"),
            "total": len(all_results),
        },
        "ikh_results": ikh_results,
        "fikh_results": fikh_results,
    }

    # Write JSON summary
    if ikh_results:
        json_path = IKH_DIR / "_run_logs" / f"run_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {"suite": "ikh", "results": ikh_results, "summary": summary["summary"]},
                f,
                indent=2,
            )

    if fikh_results:
        json_path = FIKH_DIR / "_run_logs" / f"run_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "suite": "fikh",
                    "results": fikh_results,
                    "summary": summary["summary"],
                },
                f,
                indent=2,
            )

    # Combined summary
    combined_json = Path("examples/_run_logs") / f"ikh_fikh_run_{timestamp}.json"
    combined_json.parent.mkdir(exist_ok=True)
    with open(combined_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Write issue inventory
    inventory_path = Path("examples/_run_logs/ikh_fikh_issue_inventory.md")
    write_issue_inventory(ikh_results, fikh_results, inventory_path)

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Clean pass: {summary['summary']['clean_pass']}")
    print(f"Pass with warnings: {summary['summary']['pass_with_warnings']}")
    print(f"Fail: {summary['summary']['fail']}")
    print(f"Timeout: {summary['summary']['timeout']}")
    print(f"Total: {summary['summary']['total']}")
    print(
        f"\nTotal runtime: {summary['total_runtime_seconds']:.1f}s ({summary['total_runtime_seconds']/60:.1f} min)"
    )

    # Return non-zero if any failures
    if summary["summary"]["fail"] > 0 or summary["summary"]["timeout"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
