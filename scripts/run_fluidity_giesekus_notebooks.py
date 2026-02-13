#!/usr/bin/env python
"""
Fluidity + Giesekus Notebook Runner - Specialized runner for EVP/viscoelastic tutorials.

Features:
- Per-notebook timeout (default 24 hours for Bayesian inference)
- Warning capture and reporting
- Detailed per-notebook logging
- Headless matplotlib handling
- Single-notebook and batch execution modes
- Issue inventory generation
- Suite selection (fluidity, giesekus, or both)
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

# Categorize warnings by their source/type
WARNING_CATEGORIES = {
    "deprecation": ["DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"],
    "numerical": ["RuntimeWarning", "overflow", "invalid value", "divide by zero"],
    "jax": ["jax", "jaxlib", "XLA"],
    "numpyro": ["numpyro", "divergence", "r_hat", "ess"],
    "matplotlib": ["matplotlib", "Axes", "Figure"],
    "pandas": ["pandas", "DataFrame", "Series"],
}


def categorize_warning(warning_msg: str) -> str:
    """Categorize a warning message."""
    msg_lower = warning_msg.lower()
    for category, keywords in WARNING_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in msg_lower:
                return category
    return "other"


class WarningCapture:
    """Context manager to capture warnings."""

    def __init__(self):
        self.warnings = []
        self._old_showwarning = None

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        self.warnings.append({
            "message": str(message),
            "category": category.__name__,
            "filename": str(filename),
            "lineno": lineno,
            "line": line,
            "classified_as": categorize_warning(str(message)),
        })
        # Still show the warning in stderr
        if self._old_showwarning:
            self._old_showwarning(message, category, filename, lineno, file, line)

    def __enter__(self):
        self._old_showwarning = warnings.showwarning
        warnings.showwarning = self._showwarning
        return self

    def __exit__(self, *args):
        warnings.showwarning = self._old_showwarning


def setup_headless_matplotlib():
    """Configure matplotlib for headless execution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()


def run_notebook(
    notebook_path: Path,
    timeout: int = 86400,
    log_dir: Path = None,
    suite: str = "unknown",
    cell_timeout: int = None,
) -> dict:
    """
    Run a single notebook and return detailed results.

    Args:
        notebook_path: Path to the notebook
        timeout: Per-notebook timeout in seconds (default 24 hours)
        log_dir: Directory for per-notebook logs
        suite: Suite name (fluidity or giesekus)
        cell_timeout: Per-cell timeout (defaults to notebook timeout)

    Returns:
        Dictionary with execution results
    """
    if cell_timeout is None:
        cell_timeout = timeout

    result = {
        "notebook": str(notebook_path),
        "suite": suite,
        "status": "UNKNOWN",
        "runtime_seconds": 0,
        "warnings": [],
        "warnings_count": 0,
        "error": None,
        "traceback": None,
        "stdout": "",
        "stderr": "",
        "cell_errors": [],
        "suspected_cause": None,
        "reproduction_command": f"uv run python scripts/run_fluidity_giesekus_notebooks.py --single {notebook_path}",
    }

    # Set up headless matplotlib before running
    setup_headless_matplotlib()

    original_cwd = os.getcwd()
    start_time = time.time()

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Change to notebook directory for relative paths
        os.chdir(notebook_path.parent.absolute())

        # Read notebook
        nb = nbformat.read(notebook_path.name, as_version=4)

        # Create client with long timeout
        client = NotebookClient(
            nb,
            timeout=cell_timeout,
            kernel_name="python3",
            allow_errors=False,
            force_raise_errors=True,
        )

        # Execute with warning capture
        with WarningCapture() as wc:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                client.execute()

        result["status"] = "PASS"
        result["warnings"] = wc.warnings
        result["warnings_count"] = len(wc.warnings)

        # Write executed notebook back
        nbformat.write(nb, notebook_path.name)

    except CellTimeoutError as e:
        result["status"] = "TIMEOUT"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["suspected_cause"] = "timeout/resource"

    except CellExecutionError as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

        # Categorize the error
        error_str = str(e).lower()
        if "import" in error_str or "module" in error_str:
            result["suspected_cause"] = "import/API"
        elif "shape" in error_str or "dtype" in error_str or "broadcast" in error_str:
            result["suspected_cause"] = "shape/dtype"
        elif "file" in error_str or "path" in error_str or "not found" in error_str:
            result["suspected_cause"] = "paths/data"
        elif "jax" in error_str or "jit" in error_str or "xla" in error_str:
            result["suspected_cause"] = "jax/compilation"
        elif "numpyro" in error_str or "mcmc" in error_str or "nuts" in error_str:
            result["suspected_cause"] = "numpyro/inference"
        elif "nan" in error_str or "inf" in error_str or "overflow" in error_str:
            result["suspected_cause"] = "numerical"
        elif "memory" in error_str or "oom" in error_str:
            result["suspected_cause"] = "resource/OOM"
        elif "matplotlib" in error_str or "plot" in error_str or "figure" in error_str:
            result["suspected_cause"] = "plotting"
        elif "attribute" in error_str:
            result["suspected_cause"] = "attribute/API"
        else:
            result["suspected_cause"] = "unknown"

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["suspected_cause"] = "unknown"

    finally:
        os.chdir(original_cwd)
        result["runtime_seconds"] = time.time() - start_time
        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue()

    # If passed but has warnings, note it
    if result["status"] == "PASS" and result["warnings_count"] > 0:
        result["status"] = "PASS_WITH_WARNINGS"

    # Write per-notebook log
    if log_dir:
        log_file = log_dir / f"{notebook_path.stem}.json"
        with open(log_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


def generate_issue_inventory(results: list[dict], output_path: Path, title: str = "Issue Inventory"):
    """Generate the issue inventory markdown file."""
    with open(output_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "PASS")
        passed_warnings = sum(1 for r in results if r["status"] == "PASS_WITH_WARNINGS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        timeout = sum(1 for r in results if r["status"] == "TIMEOUT")

        f.write("## Summary\n\n")
        f.write(f"- **Total notebooks**: {total}\n")
        f.write(f"- **PASS**: {passed}\n")
        f.write(f"- **PASS_WITH_WARNINGS**: {passed_warnings}\n")
        f.write(f"- **FAIL**: {failed}\n")
        f.write(f"- **TIMEOUT**: {timeout}\n\n")

        # Per-suite summary
        suites = set(r.get("suite", "unknown") for r in results)
        if len(suites) > 1:
            f.write("### Per-Suite Summary\n\n")
            for suite in sorted(suites):
                suite_results = [r for r in results if r.get("suite") == suite]
                s_passed = sum(1 for r in suite_results if r["status"] == "PASS")
                s_warnings = sum(1 for r in suite_results if r["status"] == "PASS_WITH_WARNINGS")
                s_failed = sum(1 for r in suite_results if r["status"] == "FAIL")
                s_timeout = sum(1 for r in suite_results if r["status"] == "TIMEOUT")
                f.write(f"**{suite}**: {len(suite_results)} notebooks ")
                f.write(f"(PASS: {s_passed}, WARN: {s_warnings}, FAIL: {s_failed}, TIMEOUT: {s_timeout})\n\n")

        # Detailed per-notebook results
        f.write("## Detailed Results\n\n")

        for r in results:
            nb_name = Path(r["notebook"]).name
            f.write(f"### {nb_name}\n\n")
            f.write(f"- **Suite**: {r.get('suite', 'unknown')}\n")
            f.write(f"- **Status**: {r['status']}\n")
            f.write(f"- **Runtime**: {r['runtime_seconds']:.1f}s\n")
            f.write(f"- **Warnings**: {r['warnings_count']}\n")
            f.write(f"- **Suspected cause**: {r['suspected_cause'] or 'N/A'}\n")
            f.write(f"- **Reproduction**: `{r['reproduction_command']}`\n")

            if r["status"] in ("FAIL", "TIMEOUT"):
                f.write("\n**Error**:\n```\n")
                if r["error"]:
                    f.write(r["error"][:2000])
                f.write("\n```\n")

                if r["traceback"]:
                    f.write("\n**Traceback** (truncated):\n```\n")
                    # Show last 30 lines of traceback
                    tb_lines = r["traceback"].split("\n")[-30:]
                    f.write("\n".join(tb_lines))
                    f.write("\n```\n")

            if r["warnings_count"] > 0:
                f.write("\n**Top Warnings**:\n")
                # Group warnings by category
                by_cat = {}
                for w in r["warnings"][:20]:  # Limit to first 20
                    cat = w["classified_as"]
                    if cat not in by_cat:
                        by_cat[cat] = []
                    by_cat[cat].append(w)

                for cat, warns in by_cat.items():
                    f.write(f"\n*{cat}* ({len(warns)}):\n")
                    for w in warns[:5]:  # Show first 5 per category
                        msg = w["message"][:200].replace("\n", " ")
                        f.write(f"- `{w['category']}`: {msg}\n")

            f.write("\n---\n\n")

        # Root cause summary
        f.write("## Root Cause Categories\n\n")
        causes = {}
        for r in results:
            if r["suspected_cause"]:
                causes[r["suspected_cause"]] = causes.get(r["suspected_cause"], 0) + 1

        for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
            f.write(f"- **{cause}**: {count} notebooks\n")


def generate_final_report(results: list[dict], output_path: Path, suite_name: str):
    """Generate the final clean report."""
    with open(output_path, "w") as f:
        f.write(f"# {suite_name} Notebooks - Final Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        total = len(results)
        passed = sum(1 for r in results if r["status"] in ("PASS", "PASS_WITH_WARNINGS"))
        total_warnings = sum(r["warnings_count"] for r in results)
        total_runtime = sum(r["runtime_seconds"] for r in results)

        f.write("## Summary\n\n")
        f.write(f"- **Total notebooks**: {total}\n")
        f.write(f"- **All passed**: {passed == total}\n")
        f.write(f"- **Total warnings**: {total_warnings}\n")
        f.write(f"- **Total runtime**: {total_runtime:.1f}s ({total_runtime/60:.1f}m)\n\n")

        f.write("## Per-Notebook Results\n\n")
        f.write("| Notebook | Status | Runtime | Warnings |\n")
        f.write("|----------|--------|---------|----------|\n")

        for r in results:
            nb_name = Path(r["notebook"]).name
            status = r["status"]
            runtime = f"{r['runtime_seconds']:.1f}s"
            warnings_count = r["warnings_count"]
            f.write(f"| {nb_name} | {status} | {runtime} | {warnings_count} |\n")

        f.write("\n## Reproduction Commands\n\n")
        f.write("```bash\n")
        f.write(f"# Run all {suite_name} notebooks\n")
        f.write(f"uv run python scripts/run_fluidity_giesekus_notebooks.py --suite {suite_name.lower()}\n\n")
        f.write("# Run single notebook\n")
        if results:
            f.write(f"# {results[0]['reproduction_command']}\n")
        f.write("```\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Fluidity/Giesekus notebooks with extended timeout and detailed logging"
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Run a single notebook instead of all",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["fluidity", "giesekus", "both"],
        default="both",
        help="Which suite to run (default: both)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="Per-notebook timeout in seconds (default: 86400 = 24 hours)",
    )
    parser.add_argument(
        "--cell-timeout",
        type=int,
        default=None,
        help="Per-cell timeout in seconds (defaults to --timeout)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for logs (auto-selected per suite if not specified)",
    )
    parser.add_argument(
        "--no-inventory",
        action="store_true",
        help="Skip generating issue inventory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 5 min per-cell timeout for fast baseline",
    )

    args = parser.parse_args()

    # Quick mode override
    if args.quick:
        args.cell_timeout = 300  # 5 minutes per cell
        if args.timeout == 86400:  # Only override if default
            args.timeout = 600  # 10 min per notebook

    # Ensure we're in project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.single:
        # Single notebook mode
        notebook_path = Path(args.single)
        if not notebook_path.exists():
            print(f"Error: Notebook not found: {notebook_path}")
            sys.exit(1)

        # Determine suite from path
        if "fluidity" in str(notebook_path):
            suite = "fluidity"
            log_dir = Path("examples/fluidity/_run_logs")
        elif "giesekus" in str(notebook_path):
            suite = "giesekus"
            log_dir = Path("examples/giesekus/_run_logs")
        else:
            suite = "unknown"
            log_dir = Path("examples/_run_logs")

        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running single notebook: {notebook_path}")
        print(f"Suite: {suite}")
        print(f"Timeout: {args.timeout}s")
        print(f"Cell timeout: {args.cell_timeout or args.timeout}s")
        print("-" * 60)

        result = run_notebook(
            notebook_path,
            timeout=args.timeout,
            log_dir=log_dir,
            suite=suite,
            cell_timeout=args.cell_timeout,
        )

        print(f"\nStatus: {result['status']}")
        print(f"Runtime: {result['runtime_seconds']:.1f}s")
        print(f"Warnings: {result['warnings_count']}")

        if result["error"]:
            print(f"\nError: {result['error'][:1000]}")

        sys.exit(0 if result["status"] in ("PASS", "PASS_WITH_WARNINGS") else 1)

    else:
        # Batch mode
        all_results = []

        # Define suites to run
        suites_to_run = []
        if args.suite in ("fluidity", "both"):
            suites_to_run.append(("fluidity", Path("examples/fluidity"), Path("examples/fluidity/_run_logs")))
        if args.suite in ("giesekus", "both"):
            suites_to_run.append(("giesekus", Path("examples/giesekus"), Path("examples/giesekus/_run_logs")))

        print("Fluidity + Giesekus Notebook Runner")
        print("=" * 60)
        print(f"Suites: {', '.join(s[0] for s in suites_to_run)}")
        print(f"Timeout per notebook: {args.timeout}s ({args.timeout/3600:.1f}h)")
        print(f"Cell timeout: {args.cell_timeout or args.timeout}s")
        if args.quick:
            print("Mode: QUICK (5 min cell timeout)")
        print("=" * 60)
        print()

        for suite_name, suite_dir, log_dir in suites_to_run:
            log_dir.mkdir(parents=True, exist_ok=True)

            # Get notebooks
            notebook_list_file = log_dir / "notebook_list.txt"
            notebooks = sorted(suite_dir.glob("*.ipynb"))

            # Write notebook list
            with open(notebook_list_file, "w") as f:
                for nb in notebooks:
                    f.write(f"{nb}\n")

            print(f"\n[{suite_name.upper()}] Running {len(notebooks)} notebooks")
            print("-" * 60)

            suite_results = []

            for i, nb_path in enumerate(notebooks, 1):
                print(f"[{i}/{len(notebooks)}] {nb_path.name}", end=" ", flush=True)

                result = run_notebook(
                    nb_path,
                    timeout=args.timeout,
                    log_dir=log_dir,
                    suite=suite_name,
                    cell_timeout=args.cell_timeout,
                )
                suite_results.append(result)
                all_results.append(result)

                status_symbol = {
                    "PASS": "\u2713",
                    "PASS_WITH_WARNINGS": "\u26a0",
                    "FAIL": "\u2717",
                    "TIMEOUT": "\u23f1",
                }.get(result["status"], "?")

                print(f"{status_symbol} ({result['runtime_seconds']:.1f}s, {result['warnings_count']} warnings)")

                if result["status"] in ("FAIL", "TIMEOUT"):
                    print(f"    Cause: {result['suspected_cause']}")
                    if result["error"]:
                        error_preview = result["error"][:200].replace("\n", " ")
                        print(f"    Error: {error_preview}...")

            # Generate per-suite inventory
            if not args.no_inventory:
                inventory_path = log_dir / "issue_inventory.md"
                generate_issue_inventory(suite_results, inventory_path, f"{suite_name.capitalize()} Issue Inventory")
                print(f"\n{suite_name} inventory: {inventory_path}")

            # Write per-suite master log
            master_log = log_dir / f"run_{timestamp}.json"
            with open(master_log, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "suite": suite_name,
                    "timeout": args.timeout,
                    "cell_timeout": args.cell_timeout,
                    "total": len(suite_results),
                    "passed": sum(1 for r in suite_results if r["status"] == "PASS"),
                    "passed_warnings": sum(1 for r in suite_results if r["status"] == "PASS_WITH_WARNINGS"),
                    "failed": sum(1 for r in suite_results if r["status"] == "FAIL"),
                    "timeout_count": sum(1 for r in suite_results if r["status"] == "TIMEOUT"),
                    "results": suite_results,
                }, f, indent=2, default=str)
            print(f"{suite_name} master log: {master_log}")

        # Generate combined inventory if running both suites
        if args.suite == "both" and not args.no_inventory:
            combined_log_dir = Path("examples/_run_logs")
            combined_log_dir.mkdir(parents=True, exist_ok=True)

            combined_inventory = combined_log_dir / "fluidity_giesekus_issue_inventory.md"
            generate_issue_inventory(all_results, combined_inventory, "Fluidity + Giesekus Issue Inventory")
            print(f"\nCombined inventory: {combined_inventory}")

            # Combined master log
            combined_master = combined_log_dir / f"fluidity_giesekus_run_{timestamp}.json"
            with open(combined_master, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "timeout": args.timeout,
                    "cell_timeout": args.cell_timeout,
                    "total": len(all_results),
                    "passed": sum(1 for r in all_results if r["status"] == "PASS"),
                    "passed_warnings": sum(1 for r in all_results if r["status"] == "PASS_WITH_WARNINGS"),
                    "failed": sum(1 for r in all_results if r["status"] == "FAIL"),
                    "timeout_count": sum(1 for r in all_results if r["status"] == "TIMEOUT"),
                    "results": all_results,
                }, f, indent=2, default=str)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in all_results if r["status"] == "PASS")
        passed_warnings = sum(1 for r in all_results if r["status"] == "PASS_WITH_WARNINGS")
        failed = sum(1 for r in all_results if r["status"] == "FAIL")
        timeout_count = sum(1 for r in all_results if r["status"] == "TIMEOUT")

        print(f"PASS: {passed}")
        print(f"PASS_WITH_WARNINGS: {passed_warnings}")
        print(f"FAIL: {failed}")
        print(f"TIMEOUT: {timeout_count}")

        sys.exit(0 if (failed + timeout_count) == 0 else 1)


if __name__ == "__main__":
    main()
