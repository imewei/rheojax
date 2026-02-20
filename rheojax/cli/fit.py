"""CLI subcommand for NLSQ model fitting.

Usage:
    rheojax fit data.csv --model maxwell --x-col time --y-col modulus
    rheojax fit data.csv --model maxwell --test-mode oscillation
    rheojax fit data.trios --model springpot --max-iter 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

if TYPE_CHECKING:
    pass

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for fit subcommand."""
    parser = argparse.ArgumentParser(
        prog="rheojax fit",
        description="NLSQ model fitting for rheological data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit a Maxwell model to CSV data
  rheojax fit data.csv --model maxwell --x-col time --y-col G_t

  # Fit with specific test mode
  rheojax fit data.csv --model springpot --test-mode oscillation --x-col freq --y-cols "G_prime,G_double_prime"

  # Fit TRIOS file (auto-detected columns/test mode)
  rheojax fit data.trios --model fractional_maxwell_gel

  # Fit with custom options
  rheojax fit data.csv --model maxwell --x-col time --y-col G_t --max-iter 5000

  # Output results as JSON
  rheojax fit data.csv --model maxwell --x-col time --y-col G_t --json
        """,
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Input data file (CSV, Excel, or TRIOS format)",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name (e.g., maxwell, springpot, fractional_zener_solid_solid)",
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        type=str,
        default=None,
        help="Test mode (oscillation, relaxation, creep, flow_curve, startup, laos)",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=None,
        help="X column name or index (for CSV/Excel)",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="Y column name or index (for CSV/Excel, single-column data)",
    )
    parser.add_argument(
        "--y-cols",
        type=str,
        default=None,
        help="Comma-separated Y column names (for complex data like G',G'')",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum optimization iterations (default: 100)",
    )
    parser.add_argument(
        "--deformation-mode",
        type=str,
        default=None,
        help="Deformation mode for DMTA (tension, shear, bending, compression)",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=None,
        help="Poisson ratio for E*-G* conversion (default: 0.5 for rubber)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file for results (default: stdout)",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """Run NLSQ fit from CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    logger.info(
        "CLI fit command",
        model=parsed.model,
        input_file=str(parsed.input_file),
    )

    # Validate input file
    if not parsed.input_file.exists():
        print(f"Error: File not found: {parsed.input_file}", file=sys.stderr)
        return 1

    # Validate max_iter
    if parsed.max_iter is not None and parsed.max_iter < 1:
        print(f"Error: --max-iter must be >= 1, got {parsed.max_iter}", file=sys.stderr)
        return 1

    # Load data
    try:
        from rheojax.io import auto_load

        load_kwargs: dict = {}
        if parsed.x_col is not None:
            load_kwargs["x_col"] = parsed.x_col
        if parsed.y_col is not None:
            load_kwargs["y_col"] = parsed.y_col
        if parsed.y_cols is not None:
            load_kwargs["y_cols"] = [c.strip() for c in parsed.y_cols.split(",")]

        data = auto_load(str(parsed.input_file), **load_kwargs)

        # Handle multi-segment data (use first segment)
        if isinstance(data, list):
            if not data:
                print("Error: No data segments found in file", file=sys.stderr)
                return 1
            logger.warning("File contains %d segments, using first segment", len(data))
            data = data[0]

        logger.debug("Data loaded", shape=str(data.x.shape))

    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    # Create model
    try:
        import rheojax.models  # noqa: F401 — trigger registration
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create(parsed.model)
        logger.debug("Model created", model=parsed.model)

    except KeyError:
        print(f"Error: Unknown model '{parsed.model}'", file=sys.stderr)
        print("Use 'rheojax inventory' to see available models", file=sys.stderr)
        return 1

    # Determine test mode
    test_mode = parsed.test_mode
    if test_mode is None:
        test_mode = getattr(data, "test_mode", None)
        if test_mode is None and hasattr(data, "metadata"):
            test_mode = data.metadata.get("test_mode")
    if test_mode is None:
        test_mode = "oscillation"
        logger.warning(
            "No test mode detected, defaulting to '%s'", test_mode
        )

    # Run fit
    try:
        start_time = time.perf_counter()

        fit_kwargs: dict = {
            "test_mode": test_mode,
            "max_iter": parsed.max_iter,
        }
        if parsed.deformation_mode is not None:
            fit_kwargs["deformation_mode"] = parsed.deformation_mode
        if parsed.poisson_ratio is not None:
            fit_kwargs["poisson_ratio"] = parsed.poisson_ratio

        model.fit(data.x, data.y, **fit_kwargs)
        fit_time = time.perf_counter() - start_time

        logger.info("Fit complete", model=parsed.model, time=f"{fit_time:.2f}s")

    except Exception as e:
        print(f"Error during fitting: {e}", file=sys.stderr)
        logger.error("Fit failed", model=parsed.model, error=str(e), exc_info=True)
        return 1

    # Extract results
    params = {}
    for name in model.parameters.keys():
        params[name] = float(model.parameters[name].value)

    result = {
        "model": parsed.model,
        "test_mode": test_mode,
        "parameters": params,
        "fit_time_seconds": round(fit_time, 3),
    }

    # Format output
    if parsed.json_output:
        output_text = json.dumps(result, indent=2)
    else:
        lines = [
            f"Model: {parsed.model}",
            f"Test mode: {test_mode}",
            f"Fit time: {fit_time:.3f}s",
            "",
            "Fitted Parameters:",
            "-" * 40,
        ]
        for name, value in params.items():
            lines.append(f"  {name:20s} = {value:.6g}")
        lines.append("-" * 40)
        output_text = "\n".join(lines)

    # Write output
    if parsed.output:
        parsed.output.write_text(output_text)
        print(f"Results written to {parsed.output}")
    else:
        print(output_text)

    return 0
