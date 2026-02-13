#!/usr/bin/env python
"""
Dedicated HL + STZ notebook runner with long timeout support.

Features:
- 24-hour per-notebook timeout (configurable)
- Per-notebook stdout/stderr capture
- Master timestamped log
- Warning capture
- Deterministic seeding setup
- Robust working-directory handling
- Combined suite support (HL + STZ)
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
        "path": str(notebook_path),
        "suite": notebook_path.parent.name,  # hl or stz
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
                    if output.get("output_type") == "stream" and output.get("name") == "stderr":
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
            # Ensure log directory exists (may have been deleted during long run)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                f.write(f"Notebook: {notebook_path}\n")
                f.write(f"Suite: {result['suite']}\n")
                f.write(f"Started: {start_time.isoformat()}\n")
                f.write(f"Finished: {end_time.isoformat()}\n")
                f.write(f"Runtime: {result['runtime_seconds']:.2f}s\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Cells: {result['executed_cells']}/{result['cell_count']}\n")
                f.write(f"Warnings: {result['warnings_count']}\n")
                f.write("\n")

                if result["warnings"]:
                    f.write("=== WARNINGS ===\n")
                    for w in result["warnings"][:50]:  # Limit to 50 warnings
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
            if "DeprecationWarning" in warnings_text or "FutureWarning" in warnings_text:
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
    suite_name: str,
    notebook_dir: Path,
    log_dir: Path,
    timeout: int = 86400,
    single_notebook: str | None = None,
) -> list[dict[str, Any]]:
    """Run a single notebook suite (HL or STZ)."""
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
            if not nb.name.startswith(".")
            and ".ipynb_checkpoints" not in str(nb)
        ]

    print(f"\n{suite_name.upper()} Notebook Suite")
    print("-" * 60)
    print(f"Directory: {notebook_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Notebooks found: {len(notebooks)}")

    results = []
    start_time = datetime.now()

    for i, notebook in enumerate(notebooks, 1):
        print(f"\n  [{i}/{len(notebooks)}] Running: {notebook.name}")
        print(f"      Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        result = run_notebook(notebook, timeout=timeout, log_dir=log_dir)
        result["category"] = categorize_error(result)
        results.append(result)

        status_symbol = {"PASS": "\u2713", "FAIL": "\u2717", "TIMEOUT": "\u23f0"}.get(
            result["status"], "?"
        )
        runtime_str = f"{result['runtime_seconds']:.1f}s"

        print(f"      Status: {status_symbol} {result['status']} ({runtime_str})")
        if result["warnings_count"] > 0:
            print(f"      Warnings: {result['warnings_count']}")
        if result["error"]:
            error_preview = result["error"][:200].replace("\n", " ")
            print(f"      Error: {error_preview}...")
        print(f"      Category: {result['category']}")

    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()

    # Write suite master log
    master_log = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    passed = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] == 0)
    passed_with_warnings = sum(
        1 for r in results if r["status"] == "PASS" and r["warnings_count"] > 0
    )
    failed = sum(1 for r in results if r["status"] == "FAIL")
    timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")

    with open(master_log, "w") as f:
        json.dump(
            {
                "suite": suite_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_runtime_seconds": total_runtime,
                "summary": {
                    "clean_pass": passed,
                    "pass_with_warnings": passed_with_warnings,
                    "fail": failed,
                    "timeout": timeouts,
                    "total": len(results),
                },
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    return results


def run_combined_suites(
    project_root: Path,
    timeout: int = 86400,
    single_suite: str | None = None,
    single_notebook: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run both HL and STZ notebook suites."""
    hl_results = []
    stz_results = []

    print("=" * 60)
    print("HL + STZ NOTEBOOK SUITE RUNNER")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Timeout per notebook: {timeout}s ({timeout/3600:.1f} hours)")
    print("=" * 60)

    # HL suite
    if single_suite is None or single_suite == "hl":
        hl_dir = project_root / "examples" / "hl"
        hl_log_dir = hl_dir / "_run_logs"
        hl_log_dir.mkdir(exist_ok=True)
        hl_results = run_suite("hl", hl_dir, hl_log_dir, timeout, single_notebook)

    # STZ suite
    if single_suite is None or single_suite == "stz":
        stz_dir = project_root / "examples" / "stz"
        stz_log_dir = stz_dir / "_run_logs"
        stz_log_dir.mkdir(exist_ok=True)
        stz_results = run_suite("stz", stz_dir, stz_log_dir, timeout, single_notebook)

    return hl_results, stz_results


def generate_combined_inventory(
    hl_results: list[dict[str, Any]],
    stz_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate the combined issue inventory markdown report."""
    all_results = hl_results + stz_results

    with open(output_path, "w") as f:
        f.write("# HL + STZ Notebook Issue Inventory\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary statistics
        total = len(all_results)
        clean_pass = sum(1 for r in all_results if r["status"] == "PASS" and r["warnings_count"] == 0)
        pass_with_warnings = sum(1 for r in all_results if r["status"] == "PASS" and r["warnings_count"] > 0)
        failed = sum(1 for r in all_results if r["status"] == "FAIL")
        timeouts = sum(1 for r in all_results if r["status"] == "TIMEOUT")

        f.write("## Overall Summary\n\n")
        f.write(f"- **Total notebooks**: {total}\n")
        f.write(f"- **Clean PASS**: {clean_pass}\n")
        f.write(f"- **PASS with warnings**: {pass_with_warnings}\n")
        f.write(f"- **FAIL**: {failed}\n")
        f.write(f"- **TIMEOUT**: {timeouts}\n\n")

        # Summary tables by suite
        for suite_name, results in [("HL", hl_results), ("STZ", stz_results)]:
            if not results:
                continue

            f.write(f"## {suite_name} Suite\n\n")
            f.write("| Notebook | Status | Runtime | Warnings | Category |\n")
            f.write("|----------|--------|---------|----------|----------|\n")

            for r in results:
                runtime = f"{r['runtime_seconds']:.1f}s"
                f.write(
                    f"| {r['notebook']} | {r['status']} | {runtime} | {r['warnings_count']} | {r['category']} |\n"
                )

            f.write("\n")

        # Detailed issues
        f.write("## Detailed Issues\n\n")

        for r in all_results:
            if r["status"] != "PASS" or r["warnings_count"] > 0:
                f.write(f"### [{r['suite'].upper()}] {r['notebook']}\n\n")
                f.write(f"- **Path**: `{r['path']}`\n")
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
                    relevant = [l for l in tb_lines if "rheojax" in l or "Error" in l][:10]
                    if relevant:
                        f.write("\n**Traceback (relevant)**:\n```\n")
                        f.write("\n".join(relevant))
                        f.write("\n```\n")

                # Reproduction command
                f.write("\n**Reproduce**:\n")
                f.write("```bash\n")
                f.write(
                    f"cd /Users/b80985/Projects/rheojax && uv run python scripts/run_hl_stz_notebooks.py --suite {r['suite']} --single {r['notebook']}\n"
                )
                f.write("```\n\n")

    print(f"\nCombined issue inventory written to: {output_path}")


def print_summary(hl_results: list[dict[str, Any]], stz_results: list[dict[str, Any]]) -> None:
    """Print final summary."""
    all_results = hl_results + stz_results

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")

    for suite_name, results in [("HL", hl_results), ("STZ", stz_results)]:
        if not results:
            continue

        passed = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] == 0)
        passed_with_warnings = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] > 0)
        failed = sum(1 for r in results if r["status"] == "FAIL")
        timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")
        total_runtime = sum(r["runtime_seconds"] for r in results)

        print(f"\n{suite_name} Suite:")
        print(f"  Clean PASS: {passed}")
        print(f"  PASS with warnings: {passed_with_warnings}")
        print(f"  FAIL: {failed}")
        print(f"  TIMEOUT: {timeouts}")
        print(f"  Total runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} min)")

    # Combined
    total = len(all_results)
    clean_pass = sum(1 for r in all_results if r["status"] == "PASS" and r["warnings_count"] == 0)
    total_runtime = sum(r["runtime_seconds"] for r in all_results)

    print("\nCombined:")
    print(f"  Total notebooks: {total}")
    print(f"  Fully clean: {clean_pass}/{total}")
    print(f"  Total runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="Run HL + STZ notebook suites")
    parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="Timeout per notebook in seconds (default: 86400 = 24 hours)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["hl", "stz"],
        default=None,
        help="Run only one suite (default: both)",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Run a single notebook by name",
    )
    parser.add_argument(
        "--no-inventory",
        action="store_true",
        help="Skip generating issue inventory",
    )
    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent
    combined_log_dir = project_root / "examples" / "_run_logs"
    combined_log_dir.mkdir(exist_ok=True)

    # Set up environment
    env = setup_environment()
    for k, v in env.items():
        os.environ[k] = v

    # Run suites
    hl_results, stz_results = run_combined_suites(
        project_root=project_root,
        timeout=args.timeout,
        single_suite=args.suite,
        single_notebook=args.single,
    )

    # Print summary
    print_summary(hl_results, stz_results)

    # Generate inventory
    if not args.no_inventory and not args.single:
        inventory_path = combined_log_dir / "hl_stz_issue_inventory.md"
        generate_combined_inventory(hl_results, stz_results, inventory_path)

    # Return exit code
    all_results = hl_results + stz_results
    has_issues = any(
        r["status"] != "PASS" or r["warnings_count"] > 0 for r in all_results
    )
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
