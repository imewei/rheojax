#!/usr/bin/env python
"""
Dedicated ITT-MCT notebook runner with long timeout support.

Features:
- 24-hour per-notebook timeout (configurable)
- Per-notebook stdout/stderr capture
- Master timestamped log
- Warning capture
- Deterministic seeding setup
- Robust working-directory handling
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


def setup_environment() -> dict[str, str]:
    """Set up deterministic environment variables."""
    env = os.environ.copy()
    # Ensure headless matplotlib
    env["MPLBACKEND"] = "Agg"
    # Ensure float64 for JAX
    env["JAX_ENABLE_X64"] = "True"
    # Deterministic seeds
    env["PYTHONHASHSEED"] = "42"
    # Disable JAX compilation caching issues
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    return env


def run_notebook(
    notebook_path: Path,
    timeout: int = 86400,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run a single notebook with comprehensive logging.

    Args:
        notebook_path: Path to the notebook
        timeout: Timeout per cell in seconds (default 24 hours)
        log_dir: Directory for log files

    Returns:
        Dictionary with execution results
    """
    start_time = datetime.now()
    result = {
        "notebook": notebook_path.name,
        "status": "UNKNOWN",
        "runtime_seconds": 0,
        "error": None,
        "traceback": None,
        "warnings": [],
        "warnings_count": 0,
        "cell_count": 0,
        "executed_cells": 0,
    }

    original_cwd = os.getcwd()
    notebook_dir = notebook_path.parent.absolute()
    log_file = None

    if log_dir:
        log_file = log_dir / f"{notebook_path.stem}.log"

    try:
        # Change to notebook directory for relative paths
        os.chdir(notebook_dir)

        # Read notebook
        nb = nbformat.read(notebook_path.name, as_version=4)
        result["cell_count"] = len([c for c in nb.cells if c.cell_type == "code"])

        # Configure warning capture
        captured_warnings = []

        def warning_handler(message, category, filename, lineno, file=None, line=None):
            warning_str = f"{category.__name__}: {message} ({filename}:{lineno})"
            captured_warnings.append(warning_str)

        old_showwarning = warnings.showwarning
        warnings.showwarning = warning_handler

        try:
            # Create notebook client with long timeout
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name="python3",
                allow_errors=False,
                resources={"metadata": {"path": str(notebook_dir)}},
            )

            # Execute notebook
            client.execute()

            # Write executed notebook
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
                            # Parse warnings from stderr
                            for line in text.split("\n"):
                                line = line.strip()
                                if line and ("Warning" in line or "warning" in line):
                                    captured_warnings.append(line)

        result["warnings"] = captured_warnings
        result["warnings_count"] = len(captured_warnings)

    except CellTimeoutError as e:
        result["status"] = "TIMEOUT"
        result["error"] = f"Cell timeout after {timeout} seconds"
        result["traceback"] = str(e)

    except CellExecutionError as e:
        result["status"] = "FAIL"
        result["error"] = str(e)[:1000]
        result["traceback"] = traceback.format_exc()

        # Extract cell info if available
        if hasattr(e, "cell_index"):
            result["failed_cell_index"] = e.cell_index

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        result["traceback"] = traceback.format_exc()

    finally:
        os.chdir(original_cwd)
        end_time = datetime.now()
        result["runtime_seconds"] = (end_time - start_time).total_seconds()

        # Write per-notebook log
        if log_file:
            with open(log_file, "w") as f:
                f.write(f"Notebook: {notebook_path}\n")
                f.write(f"Started: {start_time.isoformat()}\n")
                f.write(f"Finished: {end_time.isoformat()}\n")
                f.write(f"Runtime: {result['runtime_seconds']:.2f}s\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Cells: {result['executed_cells']}/{result['cell_count']}\n")
                f.write(f"Warnings: {result['warnings_count']}\n")
                f.write("\n")

                if result["warnings"]:
                    f.write("=== WARNINGS ===\n")
                    for w in result["warnings"][:50]:  # type: ignore[index]
                        f.write(f"  {w}\n")
                    f.write("\n")

                if result["error"]:
                    f.write("=== ERROR ===\n")
                    f.write(f"{result['error']}\n")
                    f.write("\n")

                if result["traceback"]:
                    f.write("=== TRACEBACK ===\n")
                    f.write(f"{result['traceback']}\n")

    return result


def categorize_error(result: dict[str, Any]) -> str:
    """Categorize the root cause of an error."""
    if result["status"] == "PASS":
        if result["warnings_count"] > 0:
            warnings_text = " ".join(result["warnings"])
            if (
                "DeprecationWarning" in warnings_text
                or "FutureWarning" in warnings_text
            ):
                return "deprecation_warning"
            if "RuntimeWarning" in warnings_text or "invalid value" in warnings_text:
                return "numerical_warning"
            if "divergence" in warnings_text.lower():
                return "numpyro_divergence"
            return "other_warning"
        return "clean"

    if result["status"] == "TIMEOUT":
        return "resource_timeout"

    error_text = (result.get("error") or "") + (result.get("traceback") or "")
    error_lower = error_text.lower()

    # Categorize by error patterns
    if "import" in error_lower or "module" in error_lower:
        return "import_api_drift"
    if "filenotfound" in error_lower or "no such file" in error_lower:
        return "paths_data"
    if "jax" in error_lower or "xla" in error_lower or "jit" in error_lower:
        return "jax_compilation"
    if "dtype" in error_lower or "float64" in error_lower or "float32" in error_lower:
        return "dtype_error"
    if "shape" in error_lower or "dimension" in error_lower:
        return "shape_error"
    if "nan" in error_lower or "inf" in error_lower or "overflow" in error_lower:
        return "numerical_instability"
    if "numpyro" in error_lower or "mcmc" in error_lower or "nuts" in error_lower:
        return "numpyro_init_divergence"
    if "diverge" in error_lower:
        return "numpyro_init_divergence"
    if "memory" in error_lower or "oom" in error_lower:
        return "resource_oom"
    if "matplotlib" in error_lower or "plot" in error_lower or "figure" in error_lower:
        return "plotting_backend"

    return "other"


def run_suite(
    notebook_dir: Path,
    log_dir: Path,
    timeout: int = 86400,
    single_notebook: str | None = None,
) -> list[dict[str, Any]]:
    """Run the full ITT-MCT notebook suite."""
    # Find notebooks
    if single_notebook:
        notebooks = [notebook_dir / single_notebook]
        if not notebooks[0].exists():
            print(f"ERROR: Notebook not found: {notebooks[0]}")
            return []
    else:
        notebooks = sorted(notebook_dir.glob("*.ipynb"))
        # Skip hidden and checkpoint files
        notebooks = [
            nb
            for nb in notebooks
            if not nb.name.startswith(".") and ".ipynb_checkpoints" not in str(nb)
        ]

    print("\nITT-MCT Notebook Suite Runner")
    print("=" * 60)
    print(f"Directory: {notebook_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Timeout per notebook: {timeout}s ({timeout/3600:.1f} hours)")
    print(f"Notebooks found: {len(notebooks)}")
    print("=" * 60)

    results = []
    start_time = datetime.now()

    for i, notebook in enumerate(notebooks, 1):
        print(f"\n[{i}/{len(notebooks)}] Running: {notebook.name}")
        print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        result = run_notebook(notebook, timeout=timeout, log_dir=log_dir)
        result["category"] = categorize_error(result)
        results.append(result)

        status_symbol = {"PASS": "\u2713", "FAIL": "\u2717", "TIMEOUT": "\u23f0"}.get(
            result["status"], "?"
        )
        runtime_str = f"{result['runtime_seconds']:.1f}s"

        print(f"    Status: {status_symbol} {result['status']} ({runtime_str})")
        if result["warnings_count"] > 0:
            print(f"    Warnings: {result['warnings_count']}")
        if result["error"]:
            error_preview = result["error"][:200].replace("\n", " ")
            print(f"    Error: {error_preview}...")
        print(f"    Category: {result['category']}")

    # Summary
    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(
        1 for r in results if r["status"] == "PASS" and r["warnings_count"] == 0
    )
    passed_with_warnings = sum(
        1 for r in results if r["status"] == "PASS" and r["warnings_count"] > 0
    )
    failed = sum(1 for r in results if r["status"] == "FAIL")
    timeout = sum(1 for r in results if r["status"] == "TIMEOUT")

    print(f"Clean PASS: {passed}")
    print(f"PASS with warnings: {passed_with_warnings}")
    print(f"FAIL: {failed}")
    print(f"TIMEOUT: {timeout}")
    print(f"Total runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} min)")

    # Write master log
    master_log = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(master_log, "w") as f:
        json.dump(
            {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_runtime_seconds": total_runtime,
                "summary": {
                    "clean_pass": passed,
                    "pass_with_warnings": passed_with_warnings,
                    "fail": failed,
                    "timeout": timeout,
                    "total": len(results),
                },
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nMaster log written to: {master_log}")

    return results


def generate_issue_inventory(results: list[dict[str, Any]], output_path: Path) -> None:
    """Generate the issue inventory markdown report."""
    with open(output_path, "w") as f:
        f.write("# ITT-MCT Notebook Issue Inventory\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Notebook | Status | Runtime | Warnings | Category |\n")
        f.write("|----------|--------|---------|----------|----------|\n")

        for r in results:
            runtime = f"{r['runtime_seconds']:.1f}s"
            f.write(
                f"| {r['notebook']} | {r['status']} | {runtime} | {r['warnings_count']} | {r['category']} |\n"
            )

        f.write("\n## Detailed Issues\n\n")

        for r in results:
            if r["status"] != "PASS" or r["warnings_count"] > 0:
                f.write(f"### {r['notebook']}\n\n")
                f.write(f"- **Status**: {r['status']}\n")
                f.write(f"- **Runtime**: {r['runtime_seconds']:.1f}s\n")
                f.write(f"- **Warnings**: {r['warnings_count']}\n")
                f.write(f"- **Category**: {r['category']}\n")

                if r["warnings"]:
                    f.write("\n**Top Warnings**:\n")
                    for w in r["warnings"][:10]:
                        f.write(f"- `{w[:150]}`\n")

                if r["error"]:
                    f.write(f"\n**Error**:\n```\n{r['error'][:1000]}\n```\n")

                if r["traceback"]:
                    # Extract most relevant frames
                    tb_lines = r["traceback"].split("\n")
                    relevant = [
                        line
                        for line in tb_lines
                        if "rheojax" in line or "Error" in line
                    ][:10]
                    if relevant:
                        f.write("\n**Traceback (relevant)**:\n```\n")
                        f.write("\n".join(relevant))
                        f.write("\n```\n")

                # Reproduction command
                f.write("\n**Reproduce**:\n")
                f.write("```bash\n")
                f.write(
                    f"cd /Users/b80985/Projects/rheojax && uv run python scripts/run_itt_mct_notebooks.py --single {r['notebook']}\n"
                )
                f.write("```\n\n")

    print(f"Issue inventory written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ITT-MCT notebook suite")
    parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="Timeout per notebook in seconds (default: 86400 = 24 hours)",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Run a single notebook by name",
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Generate issue inventory after run",
    )
    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent
    notebook_dir = project_root / "examples" / "itt_mct"
    log_dir = notebook_dir / "_run_logs"
    log_dir.mkdir(exist_ok=True)

    # Set up environment
    env = setup_environment()
    for k, v in env.items():
        os.environ[k] = v

    # Run suite
    results = run_suite(
        notebook_dir=notebook_dir,
        log_dir=log_dir,
        timeout=args.timeout,
        single_notebook=args.single,
    )

    # Generate inventory
    if args.inventory or not args.single:
        inventory_path = log_dir / "issue_inventory.md"
        generate_issue_inventory(results, inventory_path)

    # Return exit code
    has_issues = any(r["status"] != "PASS" or r["warnings_count"] > 0 for r in results)
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
