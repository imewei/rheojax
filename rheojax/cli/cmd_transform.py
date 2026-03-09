"""CLI subcommand for applying rheological transforms.

Wraps TransformRegistry to apply any registered transform to data loaded from
a file or piped in as a JSON envelope from another command.

Usage:
    rheojax transform fft_analysis --input data.csv --param n_harmonics=39
    rheojax load data.csv --json | rheojax transform mastercurve --json
    rheojax transform owchirp --input sweep.trios --output result.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rheojax.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for transform subcommand."""
    from rheojax.cli._globals import create_global_parser

    parser = argparse.ArgumentParser(
        prog="rheojax transform",
        description="Apply a registered transform to rheological data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[create_global_parser()],
        epilog="""
Examples:
  # Apply FFT transform to a CSV file
  rheojax transform fft_analysis --input data.csv

  # Pipe data from load command
  rheojax load data.trios --json | rheojax transform mastercurve --json

  # Pass transform parameters
  rheojax transform owchirp --input sweep.csv --param n_harmonics=39 --param step_size=8

  # Write output to a file
  rheojax transform fft_analysis --input data.csv --output fft_result.csv

  # List available transforms
  rheojax inventory
        """,
    )

    parser.add_argument(
        "transform_name",
        type=str,
        help="Name of the registered transform (e.g., fft_analysis, mastercurve)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        metavar="FILE|-",
        help="Input data file or '-' to read JSON envelope from stdin",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="FILE|-",
        help="Output file or '-' for stdout JSON envelope (default: stdout summary)",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Transform parameter as key=value pair (repeatable)",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=None,
        help="X column name (when loading from file)",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="Y column name (when loading from file, single-column data)",
    )
    parser.add_argument(
        "--y-cols",
        type=str,
        default=None,
        help="Comma-separated Y column names (when loading from file)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as JSON envelope to stdout",
    )

    return parser


def _parse_params(param_list: list[str]) -> dict:
    """Parse a list of 'key=value' strings into a dict with numeric conversion."""
    params: dict = {}
    for item in param_list:
        if "=" not in item:
            raise ValueError(f"Invalid parameter format '{item}': expected key=value")
        key, _, raw_val = item.partition("=")
        key = key.strip()
        raw_val = raw_val.strip()
        # Attempt numeric coercion: int → float → str
        try:
            params[key] = int(raw_val)
        except ValueError:
            try:
                params[key] = float(raw_val)
            except ValueError:
                params[key] = raw_val
    return params


def _load_from_envelope(text: str):
    """Reconstruct a minimal RheoData-like object from a JSON envelope."""
    envelope = json.loads(text)

    if not isinstance(envelope, dict):
        raise ValueError(
            f"Expected a JSON object from stdin, got {type(envelope).__name__}"
        )
    if "x" not in envelope:
        raise ValueError(
            "JSON envelope is missing required 'x' key. "
            "Pipe from 'rheojax fit --json' or 'rheojax load --json'."
        )

    from rheojax.core.data import RheoData

    x = np.array(envelope["x"])
    y_payload = envelope.get("y", {})
    if isinstance(y_payload, dict) and y_payload.get("complex"):
        y = np.array(y_payload["real"]) + 1j * np.array(y_payload["imag"])
    elif isinstance(y_payload, dict):
        if "values" not in y_payload:
            raise ValueError(
                "JSON envelope 'y' dict has neither 'complex' nor 'values' key. "
                "Pipe from 'rheojax load --json' or 'rheojax fit --json'."
            )
        y = np.array(y_payload["values"])
    else:
        y = np.array(y_payload)

    metadata = envelope.get("metadata", {})
    test_mode = envelope.get("test_mode") or metadata.get("test_mode")
    if test_mode:
        metadata["test_mode"] = test_mode

    return RheoData(x=x, y=y, metadata=metadata)


def _load_from_file(input_path: str, x_col, y_col, y_cols):
    """Load data from a file using auto_load."""
    from rheojax.io import auto_load

    load_kwargs: dict = {}
    if x_col is not None:
        load_kwargs["x_col"] = x_col
    if y_col is not None:
        load_kwargs["y_col"] = y_col
    if y_cols is not None:
        load_kwargs["y_cols"] = [c.strip() for c in y_cols.split(",")]

    data = auto_load(input_path, **load_kwargs)

    if isinstance(data, list):
        if not data:
            raise ValueError("File contained no data segments")
        if len(data) > 1:
            print(
                f"Warning: File contains {len(data)} segments; using first segment.",
                file=sys.stderr,
            )
        data = data[0]

    return data


def main(args: list[str] | None = None) -> int:
    """Apply a transform to data and print results or a JSON envelope."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    from rheojax.cli._globals import apply_globals

    apply_globals(parsed)

    logger.info("CLI transform command", transform=parsed.transform_name)

    # Parse transform parameters
    try:
        transform_params = _parse_params(parsed.param)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Determine data source: stdin envelope vs. file
    reading_stdin = parsed.input == "-" or (
        parsed.input is None and not sys.stdin.isatty()
    )

    if reading_stdin:
        try:
            from rheojax.cli._envelope import MAX_STDIN_BYTES

            raw = sys.stdin.read(MAX_STDIN_BYTES)
            data = _load_from_envelope(raw)
            logger.debug("Loaded data from stdin envelope")
        except Exception as e:
            print(f"Error reading envelope from stdin: {e}", file=sys.stderr)
            return 1
    elif parsed.input is not None:
        try:
            data = _load_from_file(
                parsed.input, parsed.x_col, parsed.y_col, parsed.y_cols
            )
            logger.debug("Loaded data from file", input=parsed.input)
        except Exception as e:
            print(f"Error loading data from '{parsed.input}': {e}", file=sys.stderr)
            return 1
    else:
        print(
            "Error: No input specified. Use --input FILE or pipe a JSON envelope via stdin.",
            file=sys.stderr,
        )
        return 1

    # Create and apply the transform
    try:
        import rheojax.transforms  # noqa: F401 — trigger registration
        from rheojax.core.registry import TransformRegistry

        transform = TransformRegistry.create(parsed.transform_name, **transform_params)
        logger.debug("Transform created", name=parsed.transform_name, params=transform_params)
    except (KeyError, ValueError, TypeError) as e:
        print(
            f"Error: Could not create transform '{parsed.transform_name}': {e}",
            file=sys.stderr,
        )
        print("Use 'rheojax inventory' to see available transforms", file=sys.stderr)
        return 1

    in_shape = np.asarray(data.x).shape
    try:
        result = transform.transform(data)
        logger.info("Transform applied", name=parsed.transform_name)
    except Exception as e:
        print(f"Error applying transform '{parsed.transform_name}': {e}", file=sys.stderr)
        logger.error("Transform failed", name=parsed.transform_name)
        logger.debug("Transform traceback", exc_info=True)
        return 1

    # Decide output mode
    writing_json = parsed.json_output or parsed.output == "-"

    if writing_json:
        from rheojax.cli._envelope import create_data_envelope

        result_x = np.asarray(result.x) if hasattr(result, "x") else np.array([])
        result_y = np.asarray(result.y) if hasattr(result, "y") else np.array([])
        metadata = dict(result.metadata) if hasattr(result, "metadata") else {}
        metadata["transform"] = parsed.transform_name
        metadata["source"] = parsed.input or "stdin"
        envelope = create_data_envelope(result_x, result_y, metadata=metadata)

        if parsed.output and parsed.output != "-":
            try:
                Path(parsed.output).write_text(envelope.to_json())
                print(f"JSON envelope written to: {parsed.output}")
            except OSError as e:
                print(f"Error writing output: {e}", file=sys.stderr)
                return 1
        else:
            envelope.write_stdout()
    else:
        # Human-readable summary
        out_shape = np.asarray(result.x).shape if hasattr(result, "x") else "(unknown)"
        lines = [
            f"Transform: {parsed.transform_name}",
            f"Input shape:  x{in_shape}",
            f"Output shape: x{out_shape}",
        ]
        if transform_params:
            lines.append(f"Parameters: {transform_params}")
        print("\n".join(lines))

        if parsed.output and parsed.output != "-":
            # Attempt to save via pandas if result has x/y
            try:
                import pandas as pd

                res_x = np.asarray(result.x)
                res_y = np.asarray(result.y)
                if np.iscomplexobj(res_y):
                    df = pd.DataFrame({"x": res_x, "y_real": res_y.real, "y_imag": res_y.imag})
                elif res_y.ndim == 2:
                    df = pd.DataFrame(np.column_stack([res_x, res_y]))
                else:
                    df = pd.DataFrame({"x": res_x, "y": res_y})
                df.to_csv(parsed.output, index=False)
                print(f"Results written to: {parsed.output}")
            except Exception as e:
                print(f"Error writing output: {e}", file=sys.stderr)
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
