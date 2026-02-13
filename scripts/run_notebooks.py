#!/usr/bin/env python
"""Script to run notebooks and report errors, warnings, and timing.

Usage:
    # Run all notebooks
    uv run python scripts/run_notebooks.py

    # Run only a specific subdirectory
    uv run python scripts/run_notebooks.py --subdir examples/fluidity

    # Set per-notebook timeout (seconds)
    uv run python scripts/run_notebooks.py --subdir examples/fluidity --timeout 345600

    # Custom log directory
    uv run python scripts/run_notebooks.py --subdir examples/fluidity --log-dir examples/fluidity/_run_logs
"""
import argparse
import gc
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def _get_kernel_pids() -> set[int]:
    """Get PIDs of all running ipykernel processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ipykernel_launcher"],
            capture_output=True, text=True, timeout=5,
        )
        return {int(pid.strip()) for pid in result.stdout.splitlines() if pid.strip()}
    except Exception:
        return set()


def _kill_leaked_kernels(before_pids: set[int]) -> int:
    """Kill kernel processes that appeared after 'before_pids' snapshot."""
    after_pids = _get_kernel_pids()
    leaked = after_pids - before_pids
    for pid in leaked:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(leaked)


def extract_warnings_from_outputs(nb: nbformat.NotebookNode) -> list[str]:
    """Extract warning messages from notebook cell outputs (stderr streams)."""
    warnings_found = []
    warning_pattern = re.compile(
        r"(?:^|\n)\s*(?:/[^\n]*?:\d+:\s*)?(UserWarning|DeprecationWarning|FutureWarning|"
        r"RuntimeWarning|PendingDeprecationWarning|NumbaWarning|"
        r"ImportWarning|SyntaxWarning)[:\s]",
        re.MULTILINE,
    )
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            text = ""
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                text = output.get("text", "")
            elif output.get("output_type") == "error":
                text = "\n".join(output.get("traceback", []))
            if text and warning_pattern.search(text):
                # Capture the warning lines
                for line in text.splitlines():
                    if warning_pattern.search(line) or "Warning" in line:
                        clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                        if clean:
                            warnings_found.append(clean)
    return warnings_found


def run_notebook(
    notebook_path: Path,
    timeout: int = 600,
    log_dir: Path | None = None,
) -> dict:
    """Run a single notebook and return detailed result dict."""
    result = {
        "path": str(notebook_path),
        "status": "UNKNOWN",
        "runtime_sec": 0.0,
        "error": "",
        "warnings": [],
        "warnings_count": 0,
    }
    original_cwd = os.getcwd()
    t0 = time.monotonic()

    # Prepare per-notebook log capture
    nb_log_path = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        nb_log_path = log_dir / f"{notebook_path.stem}.log"

    client = None
    try:
        os.chdir(notebook_path.parent.absolute())

        nb = nbformat.read(notebook_path.name, as_version=4)

        # Append a temporary cleanup cell to free JAX JIT caches and memory
        # before the kernel exits. This prevents memory pressure cascades
        # when running many JAX-heavy notebooks sequentially.
        cleanup_source = (
            "import gc as _gc; _gc.collect()\n"
            "try:\n"
            "    import jax as _jax; _jax.clear_caches()\n"
            "except Exception:\n"
            "    pass\n"
        )
        cleanup_cell = nbformat.v4.new_code_cell(source=cleanup_source)
        cleanup_cell.metadata["tags"] = ["cleanup"]
        nb.cells.append(cleanup_cell)

        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=False,
        )
        client.execute()

        # Remove the temporary cleanup cell before writing back
        if nb.cells and nb.cells[-1].metadata.get("tags") == ["cleanup"]:
            nb.cells.pop()

        # Extract warnings from executed notebook outputs
        result["warnings"] = extract_warnings_from_outputs(nb)
        result["warnings_count"] = len(result["warnings"])

        # Write back executed notebook (preserves outputs for inspection)
        nbformat.write(nb, notebook_path.name)
        result["status"] = "PASS"

    except CellExecutionError as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        # Still try to extract warnings from partial execution
        try:
            result["warnings"] = extract_warnings_from_outputs(nb)
            result["warnings_count"] = len(result["warnings"])
        except Exception:
            pass
    except TimeoutError:
        result["status"] = "TIMEOUT"
        result["error"] = f"Notebook exceeded {timeout}s timeout"
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        # Explicitly shut down the kernel to prevent leaked processes
        if client is not None:
            try:
                client._cleanup_kernel()
            except Exception:
                pass
            del client
            client = None
        # Free the executed notebook (can be large with base64 image outputs)
        try:
            del nb
        except UnboundLocalError:
            pass
        gc.collect()
        os.chdir(original_cwd)
        result["runtime_sec"] = round(time.monotonic() - t0, 2)

    # Write per-notebook log
    if nb_log_path:
        with open(nb_log_path, "w") as f:
            f.write(f"Notebook: {notebook_path}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Runtime: {result['runtime_sec']}s\n")
            f.write(f"Warnings: {result['warnings_count']}\n")
            if result["warnings"]:
                f.write("\n--- Warnings ---\n")
                for w in result["warnings"]:
                    f.write(f"  {w}\n")
            if result["error"]:
                f.write("\n--- Error ---\n")
                f.write(result["error"])
                f.write("\n")

    return result


def run_notebook_isolated(
    notebook_path: Path,
    timeout: int = 600,
    log_dir: Path | None = None,
) -> dict:
    """Run a single notebook in a subprocess for complete memory isolation.

    This avoids cumulative memory pressure when running many notebooks
    sequentially, since each subprocess's memory is fully reclaimed by the OS.
    """
    import tempfile

    result_file = tempfile.mktemp(suffix=".json")
    _log_dir_str = str(log_dir) if log_dir else ""
    script = f"""
import json, sys
sys.path.insert(0, '.')
from pathlib import Path
from scripts.run_notebooks import run_notebook
result = run_notebook(Path({str(notebook_path)!r}), timeout={timeout}, log_dir=Path({str(log_dir)!r}) if {bool(log_dir)} else None)
with open({result_file!r}, 'w') as f:
    json.dump(result, f)
"""
    t0 = time.monotonic()
    # Set JAX to avoid preallocating all GPU/CPU memory, reducing footprint
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "MPLBACKEND": "Agg",  # headless matplotlib
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        timeout=timeout + 60,  # extra margin for kernel startup
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = round(time.monotonic() - t0, 2)

    try:
        with open(result_file) as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        result = {
            "path": str(notebook_path),
            "status": "FAIL",
            "runtime_sec": elapsed,
            "error": f"Subprocess failed (exit={proc.returncode})\nstderr: {proc.stderr[:1000]}",
            "warnings": [],
            "warnings_count": 0,
        }
    finally:
        try:
            os.unlink(result_file)
        except OSError:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Jupyter notebooks and report results.")
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Only run notebooks under this subdirectory (e.g. examples/fluidity)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-notebook cell timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for per-notebook log files",
    )
    args = parser.parse_args()

    # Determine search root
    if args.subdir:
        search_root = Path(args.subdir)
        if not search_root.exists():
            print(f"ERROR: subdirectory {search_root} does not exist", file=sys.stderr)
            return 1
    else:
        search_root = Path("examples")

    log_dir = Path(args.log_dir) if args.log_dir else None

    # Find notebooks
    notebooks = sorted(search_root.rglob("*.ipynb"))
    notebooks = [
        nb
        for nb in notebooks
        if not any(part.startswith(".") for part in nb.parts)
        and "archive" not in str(nb).lower()
        and "_run_logs" not in str(nb)
        and ".ipynb_checkpoints" not in str(nb)
    ]

    if not notebooks:
        print(f"No notebooks found under {search_root}")
        return 0

    print(f"Found {len(notebooks)} notebooks under {search_root}")
    print(f"Timeout: {args.timeout}s per cell")
    if log_dir:
        print(f"Logs: {log_dir}")
    print()

    # Master log
    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    master_log_path = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        master_log_path = log_dir / f"run_{run_timestamp}.log"

    all_results = []
    pre_run_pids = _get_kernel_pids()
    max_retries = 3  # Retry kernel-death failures (transient OOM)
    for i, notebook in enumerate(notebooks, 1):
        print(f"[{i}/{len(notebooks)}] {notebook.name} ...", end=" ", flush=True)
        result = run_notebook_isolated(notebook, timeout=args.timeout, log_dir=log_dir)

        # Retry on kernel death (transient OOM from sequential memory pressure)
        retry = 0
        while (
            result["status"] == "FAIL"
            and "Kernel died" in result.get("error", "")
            and retry < max_retries
        ):
            retry += 1
            _kill_leaked_kernels(pre_run_pids)
            gc.collect()
            print(f"RETRY {retry} ...", end=" ", flush=True)
            time.sleep(10 * retry)  # Exponential backoff: 10s, 20s, 30s
            result = run_notebook_isolated(notebook, timeout=args.timeout, log_dir=log_dir)

        all_results.append(result)
        # Kill any leaked kernel processes to free system memory
        _kill_leaked_kernels(pre_run_pids)
        gc.collect()
        # Pause to let OS reclaim subprocess memory before next kernel starts
        time.sleep(5)

        status_icon = {"PASS": "PASS", "FAIL": "FAIL", "TIMEOUT": "TIMEOUT"}.get(
            result["status"], "????"
        )
        runtime_str = f"{result['runtime_sec']:.1f}s"
        warn_str = f" [{result['warnings_count']} warnings]" if result["warnings_count"] else ""
        print(f"{status_icon} ({runtime_str}){warn_str}")

        if result["status"] != "PASS":
            # Print first 5 lines of error for quick visibility
            for line in result["error"].splitlines()[:5]:
                print(f"    {line[:120]}")

    # Summary
    passed = [r for r in all_results if r["status"] == "PASS"]
    failed = [r for r in all_results if r["status"] == "FAIL"]
    timed_out = [r for r in all_results if r["status"] == "TIMEOUT"]
    total_warnings = sum(r["warnings_count"] for r in all_results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  PASS:     {len(passed)}/{len(all_results)}")
    print(f"  FAIL:     {len(failed)}/{len(all_results)}")
    print(f"  TIMEOUT:  {len(timed_out)}/{len(all_results)}")
    print(f"  Warnings: {total_warnings} total")
    total_runtime = sum(r["runtime_sec"] for r in all_results)
    print(f"  Runtime:  {total_runtime:.1f}s total")

    if failed:
        print("\nFailed notebooks:")
        for r in failed:
            print(f"  - {r['path']}")
    if timed_out:
        print("\nTimed-out notebooks:")
        for r in timed_out:
            print(f"  - {r['path']}")

    # Write master log
    if master_log_path:
        with open(master_log_path, "w") as f:
            f.write(f"Run timestamp: {run_timestamp}\n")
            f.write(f"Search root: {search_root}\n")
            f.write(f"Timeout: {args.timeout}s\n")
            f.write(f"Total notebooks: {len(all_results)}\n")
            f.write(f"Passed: {len(passed)}\n")
            f.write(f"Failed: {len(failed)}\n")
            f.write(f"Timed out: {len(timed_out)}\n")
            f.write(f"Total warnings: {total_warnings}\n")
            f.write(f"Total runtime: {total_runtime:.1f}s\n\n")
            for r in all_results:
                f.write(f"--- {r['path']} ---\n")
                f.write(f"Status: {r['status']}\n")
                f.write(f"Runtime: {r['runtime_sec']}s\n")
                f.write(f"Warnings: {r['warnings_count']}\n")
                if r["error"]:
                    f.write(f"Error:\n{r['error'][:2000]}\n")
                f.write("\n")
        print(f"\nMaster log: {master_log_path}")

    return 0 if (not failed and not timed_out) else 1


if __name__ == "__main__":
    sys.exit(main())
