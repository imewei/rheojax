"""CLI subcommand for batch processing multiple rheological data files.

Globs files matching a pattern, fits each with the specified model, and
collects a summary. Supports JSON output and saving individual result files.

Usage:
    rheojax batch "data/*.csv" --model maxwell --test-mode relaxation
    rheojax batch "*.trios" --model springpot --output-dir results/
    rheojax batch "data/*.csv" --model maxwell --test-mode oscillation --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rheojax.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for batch subcommand."""
    from rheojax.cli._globals import create_global_parser

    parser = argparse.ArgumentParser(
        prog="rheojax batch",
        description="Batch fit a model across multiple data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[create_global_parser()],
        epilog="""
Examples:
  # Fit Maxwell to all CSV files in current dir
  rheojax batch "*.csv" --model maxwell --test-mode relaxation

  # Fit with auto test-mode detection (TRIOS files)
  rheojax batch "data/*.trios" --model fractional_maxwell_gel

  # Save per-file results and a JSON summary
  rheojax batch "data/*.csv" --model maxwell --test-mode relaxation \\
      --output-dir results/ --json

  # Process in parallel (future; currently sequential)
  rheojax batch "data/*.csv" --model maxwell --parallel
        """,
    )

    parser.add_argument(
        "pattern",
        type=str,
        help="Glob pattern for input files (e.g. 'data/*.csv', '*.trios')",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        type=str,
        help="Model name to fit (e.g., maxwell, springpot)",
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        type=str,
        default=None,
        help="Test mode override (oscillation, relaxation, creep, flow_curve, startup, laos)",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=None,
        help="X column name (for CSV/Excel)",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="Y column name (for CSV/Excel, single-column data)",
    )
    parser.add_argument(
        "--y-cols",
        type=str,
        default=None,
        help="Comma-separated Y column names (for oscillation data)",
    )
    parser.add_argument(
        "--deformation-mode",
        type=str,
        default=None,
        help="Deformation mode for DMTA data (tension, compression, bending, shear)",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=None,
        help="Poisson ratio for DMTA modulus conversion (default: 0.5)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max NLSQ iterations per file (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save per-file result JSON files",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing (currently sequential; reserved for future use)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4; no-op until --parallel is implemented)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output a JSON array of all results to stdout",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining files after a failure (default: True)",
    )

    return parser


def _fit_single(
    file_path: Path,
    model_name: str,
    test_mode_override: str | None,
    load_kwargs: dict,
    fit_kwargs: dict,
) -> dict:
    """Load, fit, and return a result dict for a single file.

    Returns a dict with keys: file, status, parameters, test_mode,
    fit_time_seconds, and optionally error.
    """
    from rheojax.io import auto_load

    result: dict = {"file": str(file_path), "status": "error"}

    # Load
    data = auto_load(str(file_path), **load_kwargs)
    if isinstance(data, list):
        if not data:
            raise ValueError("File contained no data segments")
        if len(data) > 1:
            logger.warning("Multi-segment file, using first segment", file=str(file_path))
        data = data[0]

    # Resolve test_mode
    test_mode = test_mode_override
    if test_mode is None:
        test_mode = getattr(data, "test_mode", None)
        if test_mode is None and hasattr(data, "metadata"):
            test_mode = data.metadata.get("test_mode")
    if test_mode is None:
        raise ValueError(
            "Could not auto-detect test mode. Use --test-mode to specify."
        )

    # Validate
    x_arr = np.asarray(data.x)
    y_arr = np.asarray(data.y)
    if np.any(~np.isfinite(x_arr)):
        raise ValueError("Data contains NaN/Inf in x column")
    if np.iscomplexobj(y_arr):
        if np.any(~np.isfinite(np.abs(y_arr))):
            raise ValueError("Data contains NaN/Inf in complex y column")
    elif np.any(~np.isfinite(y_arr)):
        raise ValueError("Data contains NaN/Inf in y column")

    # Create model (registration happens at import time in main)
    from rheojax.core.registry import ModelRegistry

    model = ModelRegistry.create(model_name)

    # Fit
    effective_fit_kwargs = {**fit_kwargs, "test_mode": test_mode}
    start = time.perf_counter()
    model.fit(data.x, data.y, **effective_fit_kwargs)
    fit_time = time.perf_counter() - start

    params = {
        name: float(model.parameters[name].value)
        for name in model.parameters.keys()
    }

    result.update(
        {
            "status": "success",
            "test_mode": test_mode,
            "parameters": params,
            "fit_time_seconds": round(fit_time, 3),
        }
    )
    return result


def _print_summary_table(results: list[dict]) -> None:
    """Print an aligned summary table of batch results."""
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] != "success"]

    col_file = max((len(Path(r["file"]).name) for r in results), default=10)
    col_file = max(col_file, 8)

    header = f"{'File':<{col_file}}  {'Status':<8}  {'Test mode':<14}  {'Time (s)'}"
    print(header)
    print("-" * len(header))

    for r in results:
        fname = Path(r["file"]).name
        status = r["status"]
        test_mode = r.get("test_mode", "-")
        fit_time = r.get("fit_time_seconds", "-")
        time_str = f"{fit_time:.3f}" if isinstance(fit_time, float) else str(fit_time)
        print(f"{fname:<{col_file}}  {status:<8}  {test_mode:<14}  {time_str}")

    print()
    print(f"Processed: {len(results)} files")
    print(f"Success:   {len(successes)}")
    if failures:
        print(f"Failed:    {len(failures)}")
        for r in failures:
            print(f"  {r['file']}: {r.get('error', 'unknown error')}")


def main(args: list[str] | None = None) -> int:
    """Batch fit a model across multiple files."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    from rheojax.cli._globals import apply_globals

    apply_globals(parsed)

    logger.info(
        "CLI batch command",
        pattern=parsed.pattern,
        model=parsed.model,
        test_mode=parsed.test_mode,
    )

    if parsed.max_iter < 1:
        print(f"Error: --max-iter must be >= 1, got {parsed.max_iter}", file=sys.stderr)
        return 1

    if parsed.parallel:
        print(
            "Note: --parallel is reserved for future use; running sequentially.",
            file=sys.stderr,
        )

    # Resolve glob
    import glob as _glob

    matched = sorted(_glob.glob(parsed.pattern, recursive=True))
    if not matched:
        print(f"Error: No files matched pattern '{parsed.pattern}'", file=sys.stderr)
        return 1

    files = [Path(f) for f in matched if Path(f).is_file()]
    if not files:
        print(f"Error: Pattern matched entries but none are files: '{parsed.pattern}'", file=sys.stderr)
        return 1

    print(f"Found {len(files)} file(s) matching '{parsed.pattern}'")

    # Trigger model registration
    import rheojax.models  # noqa: F401 — trigger registration

    # Build shared kwargs
    load_kwargs: dict = {}
    if parsed.x_col is not None:
        load_kwargs["x_col"] = parsed.x_col
    if parsed.y_col is not None:
        load_kwargs["y_col"] = parsed.y_col
    if parsed.y_cols is not None:
        load_kwargs["y_cols"] = [c.strip() for c in parsed.y_cols.split(",")]

    fit_kwargs: dict = {"max_iter": parsed.max_iter}
    if parsed.deformation_mode is not None:
        fit_kwargs["deformation_mode"] = parsed.deformation_mode
    if parsed.poisson_ratio is not None:
        fit_kwargs["poisson_ratio"] = parsed.poisson_ratio

    # Create output directory if requested
    if parsed.output_dir is not None:
        parsed.output_dir.mkdir(parents=True, exist_ok=True)

    # Process files sequentially
    all_results: list[dict] = []

    for i, file_path in enumerate(files, 1):
        prefix = f"[{i}/{len(files)}] {file_path.name}"
        print(f"{prefix} ...", end=" ", flush=True)

        try:
            result = _fit_single(
                file_path,
                parsed.model,
                parsed.test_mode,
                load_kwargs,
                fit_kwargs,
            )
            t = result.get("fit_time_seconds", "?")
            print(f"OK ({t}s)")
            logger.info("Batch file fitted", file=str(file_path), time=t)
        except Exception as e:
            result = {
                "file": str(file_path),
                "status": "error",
                "error": str(e),
            }
            print(f"FAILED: {e}")
            logger.error("Batch file failed", file=str(file_path), error=str(e))

        all_results.append(result)

        # Save individual result file if output-dir was given
        if parsed.output_dir is not None and result["status"] == "success":
            out_file = parsed.output_dir / f"{file_path.stem}_result.json"
            try:
                from rheojax.io.json_encoder import NumpyJSONEncoder

                out_file.write_text(json.dumps(result, indent=2, cls=NumpyJSONEncoder))
            except Exception as e:
                logger.warning("Could not save result file", file=str(out_file), error=str(e))

    print()

    # Output
    if parsed.json_output:
        from rheojax.io.json_encoder import NumpyJSONEncoder

        print(json.dumps(all_results, indent=2, cls=NumpyJSONEncoder))
    else:
        _print_summary_table(all_results)

    n_failed = sum(1 for r in all_results if r["status"] != "success")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
