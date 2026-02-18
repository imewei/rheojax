#!/usr/bin/env python
"""Run non-model-family notebook suite with long timeouts and detailed logging.

Covers: basic/, bayesian/, advanced/, transforms/, io/, verification/
(Excludes model-family dirs: dmt, epm, fikh, fluidity, giesekus, hl,
 hvm, hvnm, ikh, itt_mct, sgr, stz, tnt, vlb)

Usage:
    python scripts/run_classical_notebooks.py                     # Run all
    python scripts/run_classical_notebooks.py 01 07               # Run prefix matches
    python scripts/run_classical_notebooks.py --timeout 3600      # Custom timeout
    python scripts/run_classical_notebooks.py --single basic/01   # Run single notebook
    python scripts/run_classical_notebooks.py --subdir basic      # Run one subdir only
"""

import argparse
import datetime
import io
import os
import re
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
# Constants
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
LOG_DIR = EXAMPLES_DIR / "classical" / "_run_logs"

# Model-family dirs to exclude
MODEL_FAMILY_DIRS = {
    "dmt",
    "epm",
    "fikh",
    "fluidity",
    "giesekus",
    "hl",
    "hvm",
    "hvnm",
    "ikh",
    "itt_mct",
    "sgr",
    "stz",
    "tnt",
    "vlb",
}

# Other dirs to exclude
EXCLUDE_DIRS = {"outputs", "data", "_run_logs", "archive", "utils", "classical"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def discover_notebooks(
    prefixes: list[str] | None = None,
    subdir: str | None = None,
) -> list[Path]:
    """Return sorted list of target notebooks."""
    if subdir:
        search_dir = EXAMPLES_DIR / subdir
        if not search_dir.exists():
            return []
        nbs = sorted(search_dir.rglob("*.ipynb"))
    else:
        nbs = sorted(EXAMPLES_DIR.rglob("*.ipynb"))

    # Filter out model-family dirs, hidden files, excluded dirs
    filtered = []
    for p in nbs:
        rel_parts = p.relative_to(EXAMPLES_DIR).parts
        # Skip hidden files
        if any(part.startswith(".") for part in rel_parts):
            continue
        # Skip model-family and excluded dirs
        if rel_parts[0] in MODEL_FAMILY_DIRS or rel_parts[0] in EXCLUDE_DIRS:
            continue
        # Skip archive subdirs
        if "archive" in rel_parts:
            continue
        filtered.append(p)

    if prefixes:
        filtered = [
            p
            for p in filtered
            if any(
                p.stem.startswith(pfx)
                or str(p.relative_to(EXAMPLES_DIR)).startswith(pfx)
                for pfx in prefixes
            )
        ]

    return filtered


# ---------------------------------------------------------------------------
# Single-notebook execution
# ---------------------------------------------------------------------------


class NotebookResult:
    """Outcome of executing one notebook."""

    def __init__(self, path: Path):
        self.path = path
        self.status: str = "PENDING"
        self.runtime: float = 0.0
        self.stdout: str = ""
        self.stderr: str = ""
        self.warnings: list[str] = []
        self.error_tb: str = ""
        self.error_summary: str = ""

    @property
    def rel_path(self) -> str:
        return str(self.path.relative_to(EXAMPLES_DIR))

    def as_dict(self) -> dict:
        return {
            "notebook": self.rel_path,
            "status": self.status,
            "runtime_s": round(self.runtime, 2),
            "warnings_count": len(self.warnings),
            "top_warnings": self.warnings[:10],
            "error_summary": self.error_summary[:2000] if self.error_summary else "",
        }


def run_notebook(nb_path: Path, timeout: int = 345600) -> NotebookResult:
    """Execute a single notebook, capturing all output and warnings."""
    result = NotebookResult(nb_path)
    original_cwd = os.getcwd()
    t0 = time.time()

    try:
        os.chdir(nb_path.parent.resolve())

        nb = nbformat.read(nb_path.name, as_version=4)

        # Inject headless matplotlib setup cell
        setup_code = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "plt.ioff()\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore', message='.*non-interactive.*')\n"
            "warnings.filterwarnings('ignore', message='.*Tight layout not applied.*')\n"
            "warnings.filterwarnings('ignore', message='.*No artists with labels.*')\n"
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

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                client.execute()

        result.stdout = stdout_buf.getvalue()
        result.stderr = stderr_buf.getvalue()
        result.warnings = [str(w.message) for w in caught_warnings]

        # Scan cell outputs for kernel-side warnings using Python warning format:
        # /path/file.py:LINE: WarningCategory: message
        _WARN_RE = re.compile(
            r"^\s*(?:/\S+\.py:\d+:|<\S+>:\d+:)\s*"
            r"(?:UserWarning|RuntimeWarning|DeprecationWarning|FutureWarning"
            r"|PendingDeprecationWarning|SyntaxWarning|ResourceWarning"
            r"|ImportWarning|UnicodeWarning|BytesWarning)\b"
        )
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                text = ""
                if output.output_type == "stream" and output.get("name") == "stderr":
                    text = output.get("text", "")
                elif output.output_type == "error":
                    tb_lines = output.get("traceback", [])
                    text = "\n".join(tb_lines)
                if text:
                    for line in text.split("\n"):
                        stripped = line.strip()
                        if _WARN_RE.match(stripped):
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
# Reporting
# ---------------------------------------------------------------------------


def write_per_notebook_log(result: NotebookResult, log_dir: Path) -> None:
    """Write a per-notebook log file."""
    # Use subdir__stem naming to avoid collisions
    rel = result.path.relative_to(EXAMPLES_DIR)
    safe_name = (
        "__".join(rel.parts[:-1]) + "__" + rel.stem if len(rel.parts) > 1 else rel.stem
    )
    log_path = log_dir / f"{safe_name}.log"

    with open(log_path, "w") as f:
        f.write(f"Notebook: {result.rel_path}\n")
        f.write(f"Status:   {result.status}\n")
        f.write(f"Runtime:  {result.runtime:.1f}s ({_elapsed(result.runtime)})\n")
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
        f.write(f"Classical Notebook Suite — {_ts()}\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"Total: {len(results)} | PASS: {passed} | FAIL: {failed} | TIMEOUT: {timeout}\n"
        )
        f.write(f"Total runtime: {total_time:.1f}s ({total_time / 3600:.2f}h)\n\n")

        for r in results:
            icon = {"PASS": "\u2713", "FAIL": "\u2717", "TIMEOUT": "\u23f1"}.get(
                r.status, "?"
            )
            f.write(
                f"  {icon} {r.rel_path:<65s} {r.status:<8s} "
                f"{_elapsed(r.runtime):<10s} warnings={len(r.warnings)}\n"
            )
            if r.error_summary:
                f.write(f"      Error: {r.error_summary[:200]}\n")

    print(f"\nMaster log: {master_path}")


def write_issue_inventory(results: list[NotebookResult], log_dir: Path) -> None:
    """Write issue_inventory.md."""
    inv_path = log_dir / "issue_inventory.md"
    ts = _ts()

    with open(inv_path, "w") as f:
        f.write("# Classical Notebook Suite \u2014 Issue Inventory\n\n")
        f.write(f"Generated: {ts}\n\n")

        passed = sum(1 for r in results if r.status == "PASS")
        failed = sum(1 for r in results if r.status == "FAIL")
        timeout = sum(1 for r in results if r.status == "TIMEOUT")
        total_warnings = sum(len(r.warnings) for r in results)
        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total notebooks | {len(results)} |\n")
        f.write(f"| PASS | {passed} |\n")
        f.write(f"| FAIL | {failed} |\n")
        f.write(f"| TIMEOUT | {timeout} |\n")
        f.write(f"| Total warnings | {total_warnings} |\n\n")

        for r in results:
            f.write(f"## {r.rel_path}\n\n")
            f.write(f"- **Status**: {r.status}\n")
            f.write(f"- **Runtime**: {r.runtime:.1f}s ({_elapsed(r.runtime)})\n")
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
                    f"- **Traceback** (key frames):\n"
                    f"```\n{r.error_tb[:3000]}\n```\n"
                )

            f.write("- **Reproduction**:\n")
            f.write("  ```bash\n")
            f.write(
                f"  python scripts/run_classical_notebooks.py "
                f"--single {r.rel_path}\n"
            )
            f.write("  ```\n\n")

    print(f"Issue inventory: {inv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run classical (non-model-family) notebook suite"
    )
    parser.add_argument(
        "prefixes",
        nargs="*",
        help="Notebook prefixes or subdir/prefix to filter",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=345600,
        help="Per-notebook timeout in seconds (default: 345600 = 96h)",
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Run single notebook by subdir/prefix (verbose)",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        help="Run only notebooks in a specific subdir (e.g., basic, bayesian)",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.single:
        nbs = discover_notebooks([args.single])
        if not nbs:
            print(f"No notebook found matching '{args.single}'")
            return 1
        nb = nbs[0]
        print(
            f"Running single notebook: {nb.relative_to(EXAMPLES_DIR)} (timeout={args.timeout}s)"
        )
        result = run_notebook(nb, timeout=args.timeout)
        write_per_notebook_log(result, LOG_DIR)
        print(
            f"\nStatus: {result.status} | "
            f"Runtime: {_elapsed(result.runtime)} | "
            f"Warnings: {len(result.warnings)}"
        )
        if result.error_tb:
            print(f"\nError:\n{result.error_tb[:3000]}")
        return 0 if result.status == "PASS" else 1

    nbs = discover_notebooks(args.prefixes or None, subdir=args.subdir)
    if not nbs:
        print("No notebooks found.")
        return 1

    print(f"Classical Notebook Suite — {_ts()}")
    print(f"Notebooks: {len(nbs)} | Timeout: {args.timeout}s per notebook")
    print("=" * 70)

    results: list[NotebookResult] = []

    for i, nb in enumerate(nbs, 1):
        rel = nb.relative_to(EXAMPLES_DIR)
        print(f"\n[{i}/{len(nbs)}] {rel} ...", flush=True)
        t0 = time.time()
        result = run_notebook(nb, timeout=args.timeout)
        results.append(result)
        write_per_notebook_log(result, LOG_DIR)

        icon = {"PASS": "\u2713", "FAIL": "\u2717", "TIMEOUT": "\u23f1"}.get(
            result.status, "?"
        )
        print(
            f"  {icon} {result.status} in {_elapsed(time.time() - t0)} | warnings={len(result.warnings)}"
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
