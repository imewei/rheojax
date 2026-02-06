#!/usr/bin/env python
"""
Full examples notebook runner with 24-hour timeout support.

Features:
- 24-hour per-notebook timeout (configurable)
- Per-notebook stdout/stderr capture
- Master timestamped log
- Warning capture from both Python warnings module and stderr
- Deterministic seeding setup
- Robust working-directory handling
- Issue inventory with root cause categorization
- Environment snapshot

Usage:
    # Run full suite
    uv run python scripts/run_examples_notebooks_24h.py

    # Run specific subdirectory
    uv run python scripts/run_examples_notebooks_24h.py --subdir hl

    # Run single notebook
    uv run python scripts/run_examples_notebooks_24h.py --single examples/hl/01_hl_flow_curve.ipynb

    # Custom timeout (seconds)
    uv run python scripts/run_examples_notebooks_24h.py --timeout 3600
"""
import argparse
import json
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError


def get_environment_snapshot() -> dict[str, Any]:
    """Capture environment information for reproducibility."""
    import importlib.metadata

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "packages": {},
    }

    # Key packages
    key_packages = [
        "jax", "jaxlib", "nlsq", "numpyro", "arviz", "numpy", "scipy",
        "matplotlib", "pandas", "nbformat", "nbclient", "interpax",
        "optimistix", "optax", "equinox", "diffrax",
    ]

    for pkg in key_packages:
        try:
            version = importlib.metadata.version(pkg)
            snapshot["packages"][pkg] = version
        except importlib.metadata.PackageNotFoundError:
            snapshot["packages"][pkg] = "not installed"

    # JAX device info
    try:
        import jax
        snapshot["jax_devices"] = [str(d) for d in jax.devices()]
        snapshot["jax_default_backend"] = str(jax.default_backend())
    except Exception as e:
        snapshot["jax_devices"] = f"error: {e}"

    return snapshot


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

    # Determine suite from path (e.g., hl, stz, bayesian, etc.)
    relative_to_examples = notebook_path.relative_to(
        notebook_path.parent.parent if notebook_path.parent.name != "examples" else notebook_path.parent
    )
    suite = relative_to_examples.parts[0] if len(relative_to_examples.parts) > 1 else "root"

    result = {
        "notebook": notebook_path.name,
        "path": str(notebook_path),
        "suite": suite,
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
        # Include suite in log filename to avoid collisions
        log_file = log_dir / f"{suite}_{notebook_path.stem}.log"

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

        # Capture warnings from cell outputs (stderr)
        for cell in nb.cells:
            if cell.cell_type == "code" and hasattr(cell, "outputs"):
                for output in cell.outputs:
                    if output.get("output_type") == "stream" and output.get("name") == "stderr":
                        text = output.get("text", "")
                        if text.strip():
                            # Parse warnings from stderr
                            for line in text.split("\n"):
                                line = line.strip()
                                if line and ("Warning" in line or "warning" in line or
                                             "Deprecat" in line or "Future" in line):
                                    captured_warnings.append(line)

        result["warnings"] = list(set(captured_warnings))  # Deduplicate
        result["warnings_count"] = len(result["warnings"])

    except CellTimeoutError as e:
        result["status"] = "TIMEOUT"
        result["error"] = f"Cell timeout after {timeout} seconds"
        result["traceback"] = str(e)

    except CellExecutionError as e:
        result["status"] = "FAIL"
        result["error"] = str(e)[:2000]
        result["traceback"] = traceback.format_exc()

        # Extract cell info if available
        if hasattr(e, "cell_index"):
            result["failed_cell_index"] = e.cell_index

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:1000]}"
        result["traceback"] = traceback.format_exc()

    finally:
        os.chdir(original_cwd)
        end_time = datetime.now()
        result["runtime_seconds"] = (end_time - start_time).total_seconds()

        # Write per-notebook log
        if log_file:
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
                    for w in result["warnings"][:100]:  # Limit to 100 warnings
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
            if "divergence" in warnings_text.lower() or "divergent" in warnings_text.lower():
                return "numpyro_divergence"
            if "max_tree_depth" in warnings_text.lower():
                return "numpyro_max_tree_depth"
            return "other_warning"
        return "clean"

    if result["status"] == "TIMEOUT":
        return "resource_timeout"

    error_text = (result.get("error") or "") + (result.get("traceback") or "")
    error_lower = error_text.lower()

    # Categorize by error patterns (ordered by specificity)
    if "importerror" in error_lower or "modulenotfounderror" in error_lower:
        return "import_error"
    if "attributeerror" in error_lower:
        return "api_drift"
    if "filenotfound" in error_lower or "no such file" in error_lower:
        return "paths_data"
    if "float64" in error_lower or "x64" in error_lower:
        return "jax_float64"
    if "dtype" in error_lower and ("jax" in error_lower or "array" in error_lower):
        return "dtype_error"
    if "tracer" in error_lower or "concretization" in error_lower:
        return "jax_tracing"
    if "jit" in error_lower and "recompil" in error_lower:
        return "jax_recompilation"
    if "xla" in error_lower or "jaxlib" in error_lower:
        return "jax_xla"
    if "shape" in error_lower or "dimension" in error_lower or "broadcast" in error_lower:
        return "shape_error"
    if "nan" in error_lower or "inf" in error_lower or "overflow" in error_lower:
        return "numerical_instability"
    if "diverge" in error_lower or "divergent" in error_lower:
        return "numpyro_divergence"
    if "numpyro" in error_lower or "mcmc" in error_lower or "nuts" in error_lower:
        return "numpyro_init"
    if "memory" in error_lower or "oom" in error_lower or "resourceexhausted" in error_lower:
        return "resource_oom"
    if "matplotlib" in error_lower or "figure" in error_lower:
        return "plotting_backend"
    if "keyerror" in error_lower or "indexerror" in error_lower:
        return "data_access"
    if "typeerror" in error_lower:
        return "type_error"
    if "valueerror" in error_lower:
        return "value_error"

    return "other"


def find_notebooks(
    examples_dir: Path,
    subdir: str | None = None,
    single: str | None = None,
) -> list[Path]:
    """Find notebooks to run."""
    if single:
        single_path = Path(single)
        if not single_path.is_absolute():
            single_path = Path.cwd() / single_path
        if not single_path.exists():
            print(f"ERROR: Notebook not found: {single_path}")
            return []
        return [single_path]

    search_dir = examples_dir / subdir if subdir else examples_dir
    if not search_dir.exists():
        print(f"ERROR: Directory not found: {search_dir}")
        return []

    notebooks = sorted(search_dir.rglob("*.ipynb"))

    # Filter out hidden files, checkpoints, and archive directories
    notebooks = [
        nb for nb in notebooks
        if not nb.name.startswith(".")
        and ".ipynb_checkpoints" not in str(nb)
        and "/archive/" not in str(nb)
        and "/_run_logs/" not in str(nb)
    ]

    return notebooks


def run_all_notebooks(
    examples_dir: Path,
    log_dir: Path,
    timeout: int = 86400,
    subdir: str | None = None,
    single: str | None = None,
) -> list[dict[str, Any]]:
    """Run all notebooks in the examples directory."""
    notebooks = find_notebooks(examples_dir, subdir, single)

    if not notebooks:
        print("No notebooks found!")
        return []

    print(f"\n{'=' * 70}")
    print("RHEOJAX EXAMPLES NOTEBOOK RUNNER")
    print(f"{'=' * 70}")
    print(f"Examples directory: {examples_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Timeout per notebook: {timeout}s ({timeout/3600:.1f} hours)")
    print(f"Notebooks to run: {len(notebooks)}")
    print(f"{'=' * 70}\n")

    # Write environment snapshot
    env_snapshot = get_environment_snapshot()
    snapshot_file = log_dir / "environment_snapshot.json"
    with open(snapshot_file, "w") as f:
        json.dump(env_snapshot, f, indent=2)
    print(f"Environment snapshot: {snapshot_file}\n")

    results = []
    start_time = datetime.now()

    for i, notebook in enumerate(notebooks, 1):
        rel_path = notebook.relative_to(examples_dir.parent)
        print(f"[{i:3d}/{len(notebooks)}] {rel_path}")
        print(f"         Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        result = run_notebook(notebook, timeout=timeout, log_dir=log_dir)
        result["category"] = categorize_error(result)
        results.append(result)

        status_symbol = {"PASS": "✓", "FAIL": "✗", "TIMEOUT": "⏰"}.get(
            result["status"], "?"
        )
        runtime_str = f"{result['runtime_seconds']:.1f}s"

        status_line = f"         Status: {status_symbol} {result['status']} ({runtime_str})"
        if result["warnings_count"] > 0:
            status_line += f" | Warnings: {result['warnings_count']}"
        status_line += f" | Category: {result['category']}"
        print(status_line)

        if result["error"]:
            error_preview = result["error"][:150].replace("\n", " ")
            print(f"         Error: {error_preview}...")

    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()

    # Write master log as JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_log = log_dir / f"run_{timestamp}.json"

    clean_pass = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] == 0)
    pass_with_warnings = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] > 0)
    failed = sum(1 for r in results if r["status"] == "FAIL")
    timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")

    with open(master_log, "w") as f:
        json.dump({
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_runtime_seconds": total_runtime,
            "timeout_per_notebook": timeout,
            "summary": {
                "clean_pass": clean_pass,
                "pass_with_warnings": pass_with_warnings,
                "fail": failed,
                "timeout": timeouts,
                "total": len(results),
            },
            "environment": env_snapshot,
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nMaster log: {master_log}")

    return results


def generate_issue_inventory(
    results: list[dict[str, Any]],
    output_path: Path,
    examples_dir: Path,
) -> None:
    """Generate the issue inventory markdown report."""
    with open(output_path, "w") as f:
        f.write("# RheoJAX Examples Notebook Issue Inventory\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary statistics
        total = len(results)
        clean_pass = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] == 0)
        pass_with_warnings = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] > 0)
        failed = sum(1 for r in results if r["status"] == "FAIL")
        timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")
        total_runtime = sum(r["runtime_seconds"] for r in results)

        f.write("## Overall Summary\n\n")
        f.write(f"- **Total notebooks**: {total}\n")
        f.write(f"- **Clean PASS**: {clean_pass}\n")
        f.write(f"- **PASS with warnings**: {pass_with_warnings}\n")
        f.write(f"- **FAIL**: {failed}\n")
        f.write(f"- **TIMEOUT**: {timeouts}\n")
        f.write(f"- **Total runtime**: {total_runtime:.1f}s ({total_runtime/3600:.2f} hours)\n\n")

        # Category breakdown
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        f.write("## Issues by Category\n\n")
        f.write("| Category | Count | Notebooks |\n")
        f.write("|----------|-------|----------|\n")
        for cat in sorted(categories.keys()):
            items = categories[cat]
            if cat != "clean":
                notebooks = ", ".join(r["notebook"][:20] + "..." if len(r["notebook"]) > 20 else r["notebook"] for r in items[:3])
                if len(items) > 3:
                    notebooks += f" +{len(items)-3} more"
                f.write(f"| {cat} | {len(items)} | {notebooks} |\n")
        f.write("\n")

        # Summary by suite
        suites = {}
        for r in results:
            suite = r["suite"]
            if suite not in suites:
                suites[suite] = {"clean": 0, "warn": 0, "fail": 0, "timeout": 0}
            if r["status"] == "PASS" and r["warnings_count"] == 0:
                suites[suite]["clean"] += 1
            elif r["status"] == "PASS":
                suites[suite]["warn"] += 1
            elif r["status"] == "TIMEOUT":
                suites[suite]["timeout"] += 1
            else:
                suites[suite]["fail"] += 1

        f.write("## Summary by Suite\n\n")
        f.write("| Suite | Clean | Warnings | Failed | Timeout |\n")
        f.write("|-------|-------|----------|--------|--------|\n")
        for suite in sorted(suites.keys()):
            s = suites[suite]
            f.write(f"| {suite} | {s['clean']} | {s['warn']} | {s['fail']} | {s['timeout']} |\n")
        f.write("\n")

        # Full status table
        f.write("## Full Status Table\n\n")
        f.write("| Notebook | Suite | Status | Runtime | Warnings | Category |\n")
        f.write("|----------|-------|--------|---------|----------|----------|\n")

        for r in results:
            runtime = f"{r['runtime_seconds']:.1f}s"
            status_emoji = {"PASS": "✓", "FAIL": "✗", "TIMEOUT": "⏰"}.get(r["status"], "?")
            f.write(
                f"| {r['notebook']} | {r['suite']} | {status_emoji} {r['status']} | {runtime} | {r['warnings_count']} | {r['category']} |\n"
            )

        f.write("\n")

        # Detailed issues (only non-clean)
        f.write("## Detailed Issues\n\n")

        for r in results:
            if r["status"] != "PASS" or r["warnings_count"] > 0:
                f.write(f"### [{r['suite'].upper()}] {r['notebook']}\n\n")
                f.write(f"- **Path**: `{r['path']}`\n")
                f.write(f"- **Status**: {r['status']}\n")
                f.write(f"- **Runtime**: {r['runtime_seconds']:.1f}s\n")
                f.write(f"- **Warnings**: {r['warnings_count']}\n")
                f.write(f"- **Category**: {r['category']}\n")

                if r["warnings"]:
                    f.write(f"\n**Top Warnings**:\n")
                    for w in r["warnings"][:15]:
                        # Truncate long warnings
                        w_display = w[:200] + "..." if len(w) > 200 else w
                        f.write(f"- `{w_display}`\n")

                if r["error"]:
                    f.write(f"\n**Error**:\n```\n{r['error'][:2000]}\n```\n")

                if r["traceback"]:
                    # Extract most relevant frames
                    tb_lines = r["traceback"].split("\n")
                    relevant = [l for l in tb_lines if "rheojax" in l.lower() or "Error" in l or "Exception" in l][:15]
                    if relevant:
                        f.write(f"\n**Traceback (relevant)**:\n```\n")
                        f.write("\n".join(relevant))
                        f.write("\n```\n")

                # Reproduction command
                rel_path = Path(r["path"]).relative_to(examples_dir.parent)
                f.write(f"\n**Reproduce**:\n")
                f.write(f"```bash\n")
                f.write(
                    f"cd /Users/b80985/Projects/rheojax && uv run python scripts/run_examples_notebooks_24h.py --single {rel_path}\n"
                )
                f.write(f"```\n\n")
                f.write("---\n\n")

    print(f"Issue inventory written to: {output_path}")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print final summary."""
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    total = len(results)
    clean_pass = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] == 0)
    pass_with_warnings = sum(1 for r in results if r["status"] == "PASS" and r["warnings_count"] > 0)
    failed = sum(1 for r in results if r["status"] == "FAIL")
    timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")
    total_runtime = sum(r["runtime_seconds"] for r in results)

    print(f"\nResults:")
    print(f"  Clean PASS:          {clean_pass:4d}")
    print(f"  PASS with warnings:  {pass_with_warnings:4d}")
    print(f"  FAIL:                {failed:4d}")
    print(f"  TIMEOUT:             {timeouts:4d}")
    print(f"  ─────────────────────────")
    print(f"  Total:               {total:4d}")
    print(f"\n  Total runtime: {total_runtime:.1f}s ({total_runtime/3600:.2f} hours)")

    # Check if fully clean
    if clean_pass == total:
        print("\n✓ ALL NOTEBOOKS CLEAN - SUITE PASSED")
    else:
        issues = total - clean_pass
        print(f"\n✗ {issues} NOTEBOOKS WITH ISSUES - SUITE NEEDS FIXES")


def main():
    parser = argparse.ArgumentParser(
        description="Run all RheoJAX examples notebooks with 24h timeout"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="Timeout per notebook in seconds (default: 86400 = 24 hours)",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Run only notebooks in a specific subdirectory (e.g., 'hl', 'bayesian')",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Run a single notebook by path",
    )
    parser.add_argument(
        "--no-inventory",
        action="store_true",
        help="Skip generating issue inventory",
    )
    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"
    log_dir = examples_dir / "_run_logs"
    log_dir.mkdir(exist_ok=True)

    # Set up environment
    env = setup_environment()
    for k, v in env.items():
        os.environ[k] = v

    # Run notebooks
    results = run_all_notebooks(
        examples_dir=examples_dir,
        log_dir=log_dir,
        timeout=args.timeout,
        subdir=args.subdir,
        single=args.single,
    )

    if not results:
        return 1

    # Print summary
    print_summary(results)

    # Generate inventory
    if not args.no_inventory and not args.single:
        inventory_path = log_dir / "issue_inventory.md"
        generate_issue_inventory(results, inventory_path, examples_dir)

    # Return exit code
    has_issues = any(
        r["status"] != "PASS" or r["warnings_count"] > 0 for r in results
    )
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
