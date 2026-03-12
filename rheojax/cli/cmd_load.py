"""CLI subcommand for loading rheological data files.

Wraps auto_load() to load data and output as a JSON envelope for piping
to other rheojax subcommands, or as a human-readable summary.

Usage:
    rheojax load data.csv
    rheojax load data.trios --test-mode oscillation
    rheojax load data.csv --x-col time --y-col G_t --json
    rheojax load data.csv --y-cols "G_prime,G_double_prime" | rheojax fit --model maxwell
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rheojax.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for load subcommand."""
    from rheojax.cli._globals import create_global_parser

    parser = argparse.ArgumentParser(
        prog="rheojax load",
        description="Load rheological data and output summary or JSON envelope",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[create_global_parser()],
        epilog="""
Examples:
  # Print a human-readable data summary
  rheojax load data.csv

  # Load with explicit column names
  rheojax load data.csv --x-col time --y-col G_t

  # Load complex oscillation data (G', G'')
  rheojax load data.csv --y-cols "G_prime,G_double_prime" --test-mode oscillation

  # Output JSON envelope for piping
  rheojax load data.trios --json | rheojax fit --model maxwell --json

  # Auto-detect from TRIOS file
  rheojax load sweep.trios
        """,
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Input data file (CSV, Excel, or TRIOS format)",
    )
    parser.add_argument(
        "--format",
        "-f",
        dest="file_format",
        type=str,
        default=None,
        help="File format override (csv, excel, trios, anton_paar)",
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
        help="Comma-separated Y column names (e.g., 'G_prime,G_double_prime')",
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        type=str,
        default=None,
        help="Test mode override (oscillation, relaxation, creep, flow_curve, startup, laos)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output a JSON envelope to stdout (for piping to other commands)",
    )

    return parser


def _validate_data(x_arr: np.ndarray, y_arr: np.ndarray) -> str | None:
    """Validate loaded arrays for NaN/Inf. Returns error message or None."""
    if np.any(~np.isfinite(x_arr)):
        return "Data contains NaN/Inf values in x column"
    if np.iscomplexobj(y_arr):
        if np.any(~np.isfinite(np.abs(y_arr))):
            return "Data contains NaN/Inf values in complex y column"
    elif np.any(~np.isfinite(y_arr)):
        return "Data contains NaN/Inf values in y column"
    return None


def main(args: list[str] | None = None) -> int:
    """Load data from file and print summary or JSON envelope."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    from rheojax.cli._globals import apply_globals

    apply_globals(parsed)

    logger.info("CLI load command", input_file=str(parsed.input_file))

    if not parsed.input_file.exists():
        print(f"Error: File not found: {parsed.input_file}", file=sys.stderr)
        return 1

    # Build load kwargs
    load_kwargs: dict = {}
    if parsed.x_col is not None:
        load_kwargs["x_col"] = parsed.x_col
    if parsed.y_col is not None:
        load_kwargs["y_col"] = parsed.y_col
    if parsed.y_cols is not None:
        load_kwargs["y_cols"] = [c.strip() for c in parsed.y_cols.split(",")]

    # Load data
    try:
        from rheojax.io import auto_load

        data = auto_load(str(parsed.input_file), **load_kwargs)

        # Handle multi-segment data — use first segment with a warning
        if isinstance(data, list):
            if not data:
                print("Error: No data segments found in file", file=sys.stderr)
                return 1
            if len(data) > 1:
                print(
                    f"Warning: File contains {len(data)} segments; using first segment.",
                    file=sys.stderr,
                )
                logger.warning(
                    "Multi-segment file: using first segment",
                    n_segments=len(data),
                )
            data = data[0]

    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        logger.error("Data load failed", input_file=str(parsed.input_file))
        logger.debug("Data load traceback", exc_info=True)
        return 1

    # Apply test_mode override
    if parsed.test_mode is not None:
        data.metadata["test_mode"] = parsed.test_mode

    # Resolve effective test_mode for display
    test_mode = getattr(data, "test_mode", None)
    if test_mode is None and hasattr(data, "metadata"):
        test_mode = data.metadata.get("test_mode")

    # Validate
    x_arr = np.asarray(data.x)
    y_arr = np.asarray(data.y)
    err = _validate_data(x_arr, y_arr)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    logger.debug(
        "Data loaded and validated", shape=str(x_arr.shape), test_mode=test_mode
    )

    if parsed.json_output:
        # Build JSON envelope for piping
        from rheojax.cli._envelope import create_data_envelope

        metadata = dict(data.metadata) if hasattr(data, "metadata") else {}
        metadata["source"] = str(parsed.input_file)
        if test_mode:
            metadata["test_mode"] = test_mode
        envelope = create_data_envelope(x_arr, y_arr, metadata=metadata)
        envelope.write_stdout()
    else:
        # Human-readable summary
        x_min, x_max = float(x_arr.min()), float(x_arr.max())
        if np.iscomplexobj(y_arr):
            y_desc = (
                f"complex (G'+iG''), |G*| in [{float(np.abs(y_arr).min()):.4g}, "
                f"{float(np.abs(y_arr).max()):.4g}]"
            )
        elif y_arr.ndim == 2:
            y_desc = f"shape {y_arr.shape}, cols range [{float(y_arr.min()):.4g}, {float(y_arr.max()):.4g}]"
        else:
            y_desc = f"[{float(y_arr.min()):.4g}, {float(y_arr.max()):.4g}]"

        lines = [
            f"File:      {parsed.input_file}",
            f"Points:    {len(x_arr)}",
            f"Test mode: {test_mode or '(unknown)'}",
            f"x range:   [{x_min:.4g}, {x_max:.4g}]",
            f"y:         {y_desc}",
        ]
        if hasattr(data, "metadata") and data.metadata:
            extra = {k: v for k, v in data.metadata.items() if k != "test_mode"}
            if extra:
                lines.append(f"Metadata:  {extra}")
        print("\n".join(lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())
