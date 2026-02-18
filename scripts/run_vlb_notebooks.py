#!/usr/bin/env python
"""Run VLB notebook suite with long timeouts and detailed logging.

Usage:
    python scripts/run_vlb_notebooks.py                     # Run all VLB notebooks
    python scripts/run_vlb_notebooks.py 01 07               # Run specific notebooks (prefix match)
    python scripts/run_vlb_notebooks.py --timeout 3600      # Custom timeout per notebook (seconds)
    python scripts/run_vlb_notebooks.py --single 07         # Run single notebook with verbose output
"""

import argparse
import datetime
import io
import os
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VLB_DIR = Path(__file__).resolve().parent.parent / "examples" / "vlb"
LOG_DIR = VLB_DIR / "_run_logs"


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _elapsed(start: float) -> str:
    dt = time.time() - start
    if dt < 60:
        return f"{dt:.1f}s"
    if dt < 3600:
        return f"{dt / 60:.1f}min"
    return f"{dt / 3600:.2f}h"


def discover_notebooks(prefixes: list[str] | None = None) -> list[Path]:
    """Return sorted list of VLB notebooks, optionally filtered by prefix."""
    nbs = sorted(VLB_DIR.glob("*.ipynb"))
    nbs = [p for p in nbs if not p.name.startswith(".")]
    if prefixes:
        nbs = [p for p in nbs if any(p.name.startswith(pfx) for pfx in prefixes)]
    return nbs


# ---------------------------------------------------------------------------
# Single-notebook execution
# ---------------------------------------------------------------------------


class NotebookResult:
    """Outcome of executing one notebook."""

    def __init__(self, path: Path):
        self.path = path
        self.status: str = "PENDING"  # PASS / FAIL / TIMEOUT
        self.runtime: float = 0.0
        self.stdout: str = ""
        self.stderr: str = ""
        self.warnings: list[str] = []
        self.error_tb: str = ""
        self.error_summary: str = ""

    def as_dict(self) -> dict:
        return {
            "notebook": str(self.path.relative_to(VLB_DIR.parent.parent)),
            "status": self.status,
            "runtime_s": round(self.runtime, 2),
            "warnings_count": len(self.warnings),
            "top_warnings": self.warnings[:10],
            "error_summary": self.error_summary[:2000] if self.error_summary else "",
        }


def run_notebook(nb_path: Path, timeout: int = 172800) -> NotebookResult:
    """Execute a single notebook, capturing all output and warnings."""
    result = NotebookResult(nb_path)
    original_cwd = os.getcwd()
    t0 = time.time()

    try:
        os.chdir(nb_path.parent.resolve())

        nb = nbformat.read(nb_path.name, as_version=4)

        # Set up headless matplotlib before execution
        # Inject a setup cell at the beginning
        setup_code = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "plt.ioff()\n"
        )
        setup_cell = nbformat.v4.new_code_cell(source=setup_code)
        setup_cell.metadata["tags"] = ["injected-setup"]
        nb.cells.insert(0, setup_cell)

        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=False,
        )

        # Capture warnings from the parent process (not the kernel)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                client.execute()

        result.stdout = stdout_buf.getvalue()
        result.stderr = stderr_buf.getvalue()
        result.warnings = [str(w.message) for w in caught_warnings]

        # Also scan cell outputs for warnings/errors from the kernel
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                text = ""
                if output.output_type == "stream":
                    text = output.get("text", "")
                elif output.output_type == "error":
                    tb_lines = output.get("traceback", [])
                    text = "\n".join(tb_lines)
                # Capture kernel-side warnings
                if text:
                    for line in text.split("\n"):
                        stripped = line.strip()
                        if any(
                            kw in stripped
                            for kw in [
                                "Warning",
                                "WARNING",
                                "DeprecationWarning",
                                "FutureWarning",
                                "RuntimeWarning",
                                "UserWarning",
                                "warn(",
                            ]
                        ):
                            result.warnings.append(stripped[:300])

        # Remove injected setup cell before saving
        nb.cells = [c for c in nb.cells if c.metadata.get("tags") != ["injected-setup"]]

        result.status = "PASS"

    except CellTimeoutError as e:
        result.status = "TIMEOUT"
        result.error_summary = f"Cell timed out after {timeout}s"
        result.error_tb = str(e)[:3000]

    except CellExecutionError as e:
        result.status = "FAIL"
        result.error_summary = str(e)[:500]
        result.error_tb = str(e)[:5000]

    except Exception as e:
        result.status = "FAIL"
        result.error_summary = f"{type(e).__name__}: {e}"
        result.error_tb = traceback.format_exc()[:5000]

    finally:
        result.runtime = time.time() - t0
        os.chdir(original_cwd)

    return result


# ---------------------------------------------------------------------------
# Suite execution + reporting
# ---------------------------------------------------------------------------


def write_per_notebook_log(result: NotebookResult, log_dir: Path) -> None:
    """Write a per-notebook log file."""
    stem = result.path.stem
    log_path = log_dir / f"{stem}.log"
    with open(log_path, "w") as f:
        f.write(f"Notebook: {result.path.name}\n")
        f.write(f"Status:   {result.status}\n")
        f.write(f"Runtime:  {result.runtime:.1f}s\n")
        f.write(f"Warnings: {len(result.warnings)}\n")
        f.write("=" * 70 + "\n\n")

        if result.warnings:
            f.write("--- WARNINGS ---\n")
            for w in result.warnings:
                f.write(f"  {w}\n")
            f.write("\n")

        if result.error_tb:
            f.write("--- ERROR TRACEBACK ---\n")
            f.write(result.error_tb)
            f.write("\n\n")

        if result.stdout:
            f.write("--- STDOUT ---\n")
            f.write(result.stdout[:10000])
            f.write("\n")

        if result.stderr:
            f.write("--- STDERR ---\n")
            f.write(result.stderr[:10000])
            f.write("\n")


def write_master_log(results: list[NotebookResult], log_dir: Path) -> None:
    """Write timestamped master log."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_path = log_dir / f"master_{ts}.log"

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    timeout = sum(1 for r in results if r.status == "TIMEOUT")
    total_time = sum(r.runtime for r in results)

    with open(master_path, "w") as f:
        f.write(f"VLB Notebook Suite — {_ts()}\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"Total: {len(results)} | PASS: {passed} | FAIL: {failed} | TIMEOUT: {timeout}\n"
        )
        f.write(f"Total runtime: {total_time:.1f}s ({total_time / 3600:.2f}h)\n\n")

        for r in results:
            status_icon = {"PASS": "✓", "FAIL": "✗", "TIMEOUT": "⏱"}.get(r.status, "?")
            f.write(
                f"  {status_icon} {r.path.name:<50s} {r.status:<8s} {_elapsed(time.time() - r.runtime):<10s} warnings={len(r.warnings)}\n"
            )
            if r.error_summary:
                f.write(f"      Error: {r.error_summary[:200]}\n")

    print(f"\nMaster log: {master_path}")


def write_issue_inventory(results: list[NotebookResult], log_dir: Path) -> None:
    """Write issue_inventory.md for Phase 1."""
    inv_path = log_dir / "issue_inventory.md"
    ts = _ts()

    with open(inv_path, "w") as f:
        f.write("# VLB Notebook Issue Inventory\n\n")
        f.write(f"Generated: {ts}\n\n")

        passed = sum(1 for r in results if r.status == "PASS")
        failed = sum(1 for r in results if r.status == "FAIL")
        timeout = sum(1 for r in results if r.status == "TIMEOUT")
        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total notebooks | {len(results)} |\n")
        f.write(f"| PASS | {passed} |\n")
        f.write(f"| FAIL | {failed} |\n")
        f.write(f"| TIMEOUT | {timeout} |\n\n")

        for r in results:
            f.write(f"## {r.path.name}\n\n")
            f.write(f"- **Status**: {r.status}\n")
            f.write(
                f"- **Runtime**: {r.runtime:.1f}s ({_elapsed(time.time() - r.runtime + r.runtime)})\n"
            )
            f.write(f"- **Warnings**: {len(r.warnings)}\n")

            if r.warnings:
                f.write("- **Top warnings**:\n")
                seen = set()
                for w in r.warnings[:15]:
                    if w not in seen:
                        seen.add(w)
                        f.write(f"  - `{w[:200]}`\n")

            if r.error_summary:
                f.write(f"- **Error**: `{r.error_summary[:300]}`\n")
                f.write(
                    f"- **Traceback** (key frames):\n```\n{r.error_tb[:3000]}\n```\n"
                )

            f.write("- **Reproduction**:\n")
            f.write("  ```bash\n")
            f.write(
                f"  python scripts/run_vlb_notebooks.py --single {r.path.stem[:2]}\n"
            )
            f.write("  ```\n\n")

    print(f"Issue inventory: {inv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run VLB notebook suite")
    parser.add_argument(
        "prefixes", nargs="*", help="Notebook prefixes to filter (e.g., 01 07)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=172800,
        help="Per-notebook timeout in seconds (default: 172800 = 48h)",
    )
    parser.add_argument(
        "--single", type=str, help="Run a single notebook by prefix (verbose)"
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.single:
        nbs = discover_notebooks([args.single])
        if not nbs:
            print(f"No notebook found with prefix '{args.single}'")
            return 1
        nb = nbs[0]
        print(f"Running single notebook: {nb.name} (timeout={args.timeout}s)")
        result = run_notebook(nb, timeout=args.timeout)
        write_per_notebook_log(result, LOG_DIR)
        print(
            f"\nStatus: {result.status} | Runtime: {_elapsed(time.time() - result.runtime + result.runtime)} | Warnings: {len(result.warnings)}"
        )
        if result.error_tb:
            print(f"\nError:\n{result.error_tb[:3000]}")
        return 0 if result.status == "PASS" else 1

    nbs = discover_notebooks(args.prefixes or None)
    if not nbs:
        print("No notebooks found.")
        return 1

    print(f"VLB Notebook Suite — {_ts()}")
    print(f"Notebooks: {len(nbs)} | Timeout: {args.timeout}s per notebook")
    print("=" * 70)

    results: list[NotebookResult] = []

    for i, nb in enumerate(nbs, 1):
        print(f"\n[{i}/{len(nbs)}] {nb.name} ...", flush=True)
        t0 = time.time()
        result = run_notebook(nb, timeout=args.timeout)
        results.append(result)
        write_per_notebook_log(result, LOG_DIR)

        icon = {"PASS": "✓", "FAIL": "✗", "TIMEOUT": "⏱"}.get(result.status, "?")
        print(
            f"  {icon} {result.status} in {_elapsed(t0)} | warnings={len(result.warnings)}"
        )
        if result.error_summary:
            print(f"  Error: {result.error_summary[:200]}")

    # Write summary reports
    write_master_log(results, LOG_DIR)
    write_issue_inventory(results, LOG_DIR)

    # Print final summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    timeout = sum(1 for r in results if r.status == "TIMEOUT")
    print(f"\n{'=' * 70}")
    print(
        f"FINAL: {passed} PASS | {failed} FAIL | {timeout} TIMEOUT out of {len(results)}"
    )
    print(f"{'=' * 70}")

    return 0 if failed == 0 and timeout == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
