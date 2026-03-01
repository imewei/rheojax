#!/usr/bin/env python
"""
Unified Notebook Runner with long timeouts and detailed logging.

Replaces 12 legacy run_* scripts with a single robust runner.
Supports running individual notebooks or recursively scanning directories.

Usage:
    python scripts/run_single_notebook_96h.py examples/vlb/01_model_behavior.ipynb
    python scripts/run_single_notebook_96h.py examples/vlb examples/hl
"""

import argparse
import datetime
import io
import json
import os
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _elapsed(start: float) -> str:
    dt = time.time() - start
    if dt < 60:
        return f"{dt:.1f}s"
    if dt < 3600:
        return f"{dt / 60:.1f}min"
    return f"{dt / 3600:.2f}h"


def setup_environment() -> dict[str, str]:
    """Set up deterministic environment variables."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["JAX_ENABLE_X64"] = "True"
    env["PYTHONHASHSEED"] = "42"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    env["FAST_MODE"] = os.environ.get(
        "FAST_MODE", "1"
    )  # Default to fast mode unless explicitly set to 0

    # Isolate Jupyter environment to avoid PermissionError/Config issues
    import tempfile

    temp_jupyter_dir = tempfile.mkdtemp(prefix="jupyter_runtime_")
    env["JUPYTER_CONFIG_DIR"] = os.path.join(temp_jupyter_dir, "config")
    env["JUPYTER_DATA_DIR"] = os.path.join(temp_jupyter_dir, "data")
    env["JUPYTER_RUNTIME_DIR"] = os.path.join(temp_jupyter_dir, "runtime")
    env["IPYTHONDIR"] = os.path.join(temp_jupyter_dir, "ipython")

    return env


class NotebookResult:
    """Outcome of executing one notebook."""

    def __init__(self, path: Path):
        self.path = path.absolute()
        self.name = path.name
        self.status: str = "PENDING"  # PASS / FAIL / TIMEOUT
        self.runtime: float = 0.0
        self.stdout: str = ""
        self.stderr: str = ""
        self.warnings: list[str] = []
        self.error_tb: str = ""
        self.error_summary: str = ""
        self.category: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "notebook": self.name,
            "path": str(self.path),
            "status": self.status,
            "runtime_s": round(self.runtime, 2),
            "warnings_count": len(self.warnings),
            "top_warnings": self.warnings[:10],
            "error_summary": self.error_summary[:2000] if self.error_summary else "",
            "category": self.category,
        }


def categorize_error(result: NotebookResult) -> str:
    """Categorize the root cause of an error or warning."""
    if result.status == "PASS":
        if result.warnings:
            warnings_text = " ".join(result.warnings)
            if (
                "DeprecationWarning" in warnings_text
                or "FutureWarning" in warnings_text
            ):
                return "deprecation_warning"
            if "RuntimeWarning" in warnings_text or "invalid value" in warnings_text:
                return "numerical_warning"
            if "divergence" in warnings_text.lower():
                return "numpyro_divergence"
            return "pass_with_warnings"
        return "clean_pass"

    if result.status == "TIMEOUT":
        return "resource_timeout"

    error_text = (result.error_summary or "") + (result.error_tb or "")
    error_lower = error_text.lower()

    if "import" in error_lower or "module" in error_lower:
        return "import_error"
    if "jax" in error_lower or "xla" in error_lower or "jit" in error_lower:
        return "jax_compilation"
    if "dtype" in error_lower or "float64" in error_lower or "float32" in error_lower:
        return "dtype_error"
    if "shape" in error_lower or "dimension" in error_lower:
        return "shape_error"
    if "nan" in error_lower or "inf" in error_lower or "overflow" in error_lower:
        return "numerical_instability"
    if "numpyro" in error_lower or "diverge" in error_lower or "mcmc" in error_lower:
        return "numpyro_error"
    if "memory" in error_lower or "oom" in error_lower:
        return "resource_oom"
    if "matplotlib" in error_lower or "plot" in error_lower or "figure" in error_lower:
        return "plotting_backend"

    return "other_error"


def run_notebook(nb_path: Path, timeout: int) -> NotebookResult:
    """Execute a single notebook, capturing all output and warnings."""
    result = NotebookResult(nb_path)
    original_cwd = os.getcwd()
    t0 = time.time()

    try:
        os.chdir(nb_path.parent.resolve())

        nb = nbformat.read(nb_path.name, as_version=4)

        # Inject a setup cell at the beginning to force headless rendering
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
            resources={"metadata": {"path": str(nb_path.parent)}},
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

        # Scan cell outputs for kernel-side warnings/errors
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

        # Save the successfully executed notebook back to disk
        nbformat.write(nb, nb_path.name)
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

        # Partially save notebook on failure for debugging if possible
        if result.status != "PASS":
            try:
                # Still try to remove the injected setup cell before saving partial state
                nb.cells = [
                    c
                    for c in nb.cells
                    if c.metadata.get("tags", []) != ["injected-setup"]
                ]
                nbformat.write(nb, nb_path)
            except Exception:
                pass

        result.category = categorize_error(result)

    return result


def write_per_notebook_log(result: NotebookResult, log_dir: Path) -> None:
    """Write a per-notebook detailed log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{result.path.stem}.log"
    with open(log_path, "w") as f:
        f.write(f"Notebook: {result.path}\n")
        f.write(f"Status:   {result.status}\n")
        f.write(f"Category: {result.category}\n")
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


def write_master_log(
    results: list[NotebookResult], log_dir: Path, timeout: int
) -> None:
    """Write timestamped master JSON log."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    master_path = log_dir / f"master_{ts}.json"

    with open(master_path, "w") as f:
        json.dump(
            {
                "run_type": "unified_runner",
                "timestamp": ts,
                "timeout_per_notebook_s": timeout,
                "total_notebooks": len(results),
                "total_runtime_s": sum(r.runtime for r in results),
                "results": [r.as_dict() for r in results],
            },
            f,
            indent=2,
            default=str,
        )


def write_issue_inventory(results: list[NotebookResult], log_dir: Path) -> None:
    """Write an aggregated issue inventory markdown file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    inv_path = log_dir / "issue_inventory.md"

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    timeout = sum(1 for r in results if r.status == "TIMEOUT")

    with open(inv_path, "w") as f:
        f.write("# Notebook Runner Issue Inventory\n\n")
        f.write(f"Generated: {_ts()}\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total notebooks | {len(results)} |\n")
        f.write(f"| PASS | {passed} |\n")
        f.write(f"| FAIL | {failed} |\n")
        f.write(f"| TIMEOUT | {timeout} |\n\n")

        for r in results:
            if r.status == "PASS" and not r.warnings:
                continue

            f.write(f"## {r.name}\n\n")
            f.write(f"- **Path**: `{r.path}`\n")
            f.write(f"- **Status**: {r.status}\n")
            f.write(f"- **Category**: {r.category}\n")
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
            f.write(f"  uv run python scripts/run_single_notebook_96h.py {r.path}\n")
            f.write("  ```\n\n")


def discover_notebooks(targets: list[str]) -> list[Path]:
    """Find all .ipynb files from the given targets (directories or files)."""
    notebooks = set()
    for target in targets:
        p = Path(target)
        if p.is_file() and p.suffix == ".ipynb":
            notebooks.add(p.absolute())
        elif p.is_dir():
            for nb in p.rglob("*.ipynb"):
                if (
                    not nb.name.startswith(".")
                    and ".ipynb_checkpoints" not in nb.parts
                    and "_run_logs" not in nb.parts
                    and "archive" not in str(nb).lower()
                    and "template" not in nb.name.lower()
                ):
                    notebooks.add(nb.absolute())
        else:
            print(f"Warning: Target {target} is neither a file nor a directory.")

    return sorted(notebooks)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Notebook Runner (replaces all specialized scripts)."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="Notebook files or directories to execute.",
    )
    args = parser.parse_args()

    # Fixed timeout: 96 hours per notebook
    TIMEOUT_PER_NOTEBOOK = 96 * 3600

    nbs = discover_notebooks(args.targets)
    if not nbs:
        print("No valid notebooks found.")
        sys.exit(1)

    print("======================================================================")
    print("UNIFIED NOTEBOOK RUNNER")
    print("======================================================================")
    print(f"Notebooks    : {len(nbs)}")
    print(
        f"Timeout/nb   : {TIMEOUT_PER_NOTEBOOK}s ({TIMEOUT_PER_NOTEBOOK / 3600:.1f}h)"
    )
    print(f"Start Time   : {_ts()}")
    print("======================================================================")

    env_vars = setup_environment()
    os.environ.update(env_vars)

    project_root = Path(__file__).resolve().parent.parent
    base_log_dir = (
        project_root
        / "examples"
        / "_run_logs"
        / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    results = []

    for i, nb in enumerate(nbs, 1):
        print(f"\n[{i}/{len(nbs)}] {nb.name} ... ", end="", flush=True)
        t0 = time.time()

        result = run_notebook(nb, timeout=TIMEOUT_PER_NOTEBOOK)
        results.append(result)

        # Log to a directory adjacent to the notebook if possible, else to a central log dir
        nb_log_dir = nb.parent / "_run_logs"
        write_per_notebook_log(result, nb_log_dir)

        icon = {"PASS": "✓", "FAIL": "✗", "TIMEOUT": "⏱"}.get(result.status, "?")
        print(
            f"{icon} {result.status} in {_elapsed(t0)} | warn={len(result.warnings)} | cat={result.category}"
        )
        if result.error_summary:
            print(f"    Error: {result.error_summary[:150]}")

    write_master_log(results, base_log_dir, TIMEOUT_PER_NOTEBOOK)
    write_issue_inventory(results, base_log_dir)

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    timeouts = sum(1 for r in results if r.status == "TIMEOUT")

    print(f"\n{'=' * 70}")
    print(
        f"FINAL: {passed} PASS | {failed} FAIL | {timeouts} TIMEOUT (out of {len(results)})"
    )
    print(f"Master Log: {base_log_dir}")
    print(f"{'=' * 70}")

    return 0 if failed == 0 and timeouts == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
