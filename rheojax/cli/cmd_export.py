"""CLI subcommand for exporting pipeline analysis results.

Wraps AnalysisExporter to write structured output (directory, Excel, or HDF5)
from a previous pipeline run's results directory or HDF5 archive, or from a
JSON envelope piped via stdin.

Usage:
    rheojax export results/ --output export/ --format directory
    rheojax export analysis.h5 --output bundle.xlsx --format excel
    rheojax fit data.csv --model maxwell --json | rheojax export - --output ./out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rheojax.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_VALID_FORMATS = ("directory", "excel", "hdf5")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for export subcommand."""
    from rheojax.cli._globals import create_global_parser

    parser = argparse.ArgumentParser(
        prog="rheojax export",
        description="Export pipeline analysis results to a structured output format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[create_global_parser()],
        epilog="""
Examples:
  # Export a previous results directory to HDF5
  rheojax export results/ --output bundle.h5 --format hdf5

  # Export to Excel workbook
  rheojax export results/ --output summary.xlsx --format excel

  # Export in directory layout (default)
  rheojax export analysis.h5 --output ./export_dir --format directory

  # Pipe fit results directly into export
  rheojax fit data.csv --model maxwell --json | rheojax export - --output ./out
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help=(
            "Results directory, HDF5 file from a previous run, "
            "or '-' to read a JSON envelope from stdin"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path (directory, .xlsx, or .h5 depending on --format)",
    )
    parser.add_argument(
        "--format",
        "-f",
        dest="export_format",
        type=str,
        default="directory",
        choices=_VALID_FORMATS,
        help="Export format: directory (default), excel, or hdf5",
    )
    parser.add_argument(
        "--figure-formats",
        type=str,
        default="pdf,png",
        help="Comma-separated figure formats for directory export (default: pdf,png)",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="Figure DPI for raster formats (default: 300)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print a JSON summary of the export to stdout on success",
    )

    return parser


def _build_pipeline_from_envelope(envelope: dict):
    """Construct a minimal Pipeline populated with envelope data for export."""

    import numpy as np

    from rheojax.core.data import RheoData

    x = np.array(envelope.get("x", []))
    y_payload = envelope.get("y", {})
    if isinstance(y_payload, dict) and y_payload.get("complex"):
        y = np.array(y_payload["real"]) + 1j * np.array(y_payload["imag"])
    elif isinstance(y_payload, dict):
        y = np.array(y_payload.get("values", []))
    else:
        y = np.array(y_payload)

    metadata = envelope.get("metadata", {})
    test_mode = envelope.get("test_mode") or metadata.get("test_mode")
    if test_mode:
        metadata["test_mode"] = test_mode

    data = RheoData(x=x, y=y, metadata=metadata)

    try:
        from rheojax.pipeline import Pipeline

        pipe = Pipeline()
        pipe._data = data  # type: ignore[attr-defined]
        pipe._source = envelope.get("source", "stdin")  # type: ignore[attr-defined]
        return pipe
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Could not construct a Pipeline from the stdin envelope. "
            "Pipe from 'rheojax fit --json' or provide a results directory."
        ) from exc


def main(args: list[str] | None = None) -> int:
    """Export pipeline results to the chosen format."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    from rheojax.cli._globals import apply_globals

    apply_globals(parsed)

    logger.info(
        "CLI export command",
        input=parsed.input,
        output=str(parsed.output),
        format=parsed.export_format,
    )

    figure_formats = tuple(
        f.strip() for f in parsed.figure_formats.split(",") if f.strip()
    )

    try:
        from rheojax.io.analysis_exporter import AnalysisExporter

        exporter = AnalysisExporter(
            figure_formats=figure_formats,
            figure_dpi=parsed.figure_dpi,
        )
    except Exception as e:
        print(f"Error initialising AnalysisExporter: {e}", file=sys.stderr)
        return 1

    reading_stdin = parsed.input == "-"

    if reading_stdin:
        # Build a minimal Pipeline from the piped envelope
        try:
            import json as _json

            from rheojax.cli._envelope import MAX_STDIN_BYTES

            raw = sys.stdin.read(MAX_STDIN_BYTES)
            envelope = _json.loads(raw)
            pipeline = _build_pipeline_from_envelope(envelope)
            logger.debug("Pipeline built from stdin envelope")
        except Exception as e:
            print(f"Error reading envelope from stdin: {e}", file=sys.stderr)
            return 1
    else:
        input_path = Path(parsed.input)
        if not input_path.exists():
            print(f"Error: Input path not found: {input_path}", file=sys.stderr)
            return 1

        if input_path.is_dir():
            # Treat as a results directory — load the pipeline state from it
            try:
                from rheojax.pipeline import Pipeline

                pipeline = Pipeline.load(str(input_path))  # type: ignore[attr-defined]
                logger.debug("Pipeline loaded from results directory", path=str(input_path))
            except (ImportError, AttributeError):
                # Pipeline.load() may not exist; fall back to passing path to exporter
                pipeline = None
            except Exception as e:
                print(f"Error loading pipeline from directory: {e}", file=sys.stderr)
                return 1
        elif input_path.suffix in (".h5", ".hdf5"):
            try:
                from rheojax.pipeline import Pipeline

                pipeline = Pipeline.load_hdf5(str(input_path))  # type: ignore[attr-defined]
                logger.debug("Pipeline loaded from HDF5", path=str(input_path))
            except (ImportError, AttributeError):
                pipeline = None
            except Exception as e:
                print(f"Error loading pipeline from HDF5: {e}", file=sys.stderr)
                return 1
        else:
            print(
                f"Error: Unsupported input type '{input_path}'. "
                "Expected a results directory, HDF5 file, or '-' for stdin.",
                file=sys.stderr,
            )
            return 1

    # Validate output path safety — reject '..' traversal and absolute paths
    output_path = parsed.output
    if ".." in output_path.parts:
        print(
            f"Error: Output path must not contain '..' segments, got '{output_path}'",
            file=sys.stderr,
        )
        return 1
    if output_path.is_absolute():
        print(
            f"Error: Output path must be relative, got absolute '{output_path}'",
            file=sys.stderr,
        )
        return 1

    # Run the export
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if parsed.export_format == "directory":
            exporter.export_directory(pipeline, output_dir=str(output_path))
        elif parsed.export_format == "excel":
            exporter.export_excel(pipeline, output_path=str(output_path))
        elif parsed.export_format == "hdf5":
            exporter.export_hdf5(pipeline, output_path=str(output_path))

        logger.info("Export complete", output=str(output_path), format=parsed.export_format)

    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        logger.error("Export failed")
        logger.debug("Export traceback", exc_info=True)
        return 1

    if parsed.json_output:
        import json as _json

        from rheojax.io.json_encoder import NumpyJSONEncoder

        summary = {
            "status": "success",
            "output": str(output_path),
            "format": parsed.export_format,
        }
        print(_json.dumps(summary, cls=NumpyJSONEncoder))
    else:
        print(f"Export complete: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
