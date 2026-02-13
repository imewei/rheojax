"""TA Instruments TRIOS multi-format file readers.

This package provides readers for TRIOS rheometer data in multiple formats:
- TXT: Plain text export (LIMS format)
- CSV: Comma/tab-separated values export
- Excel: .xlsx/.xls export
- JSON: JSON export (schema-validated)

Usage:
    >>> from rheojax.io.readers import load_trios
    >>> data = load_trios('frequency_sweep.csv')  # Auto-detects format

    >>> from rheojax.io.readers.trios import load_trios_csv
    >>> data = load_trios_csv('frequency_sweep.csv')  # Explicit format

All loaders return RheoData objects compatible with RheoJAX analysis pipelines.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rheojax.core.data import RheoData
from rheojax.io.readers.trios.common import (
    STEP_COLUMN_CANDIDATES,
    TRIOS_COLUMN_MAPPINGS,
    TRIOS_UNIT_CONVERSIONS,
    ColumnMapping,
    DataSegment,
    TRIOSFile,
    TRIOSTable,
    construct_complex_modulus,
    convert_unit,
    detect_step_column,
    detect_test_type,
    map_columns_to_canonical,
    segment_to_rheodata,
    select_xy_columns,
    split_by_step,
)

# CSV reader
from rheojax.io.readers.trios.csv import load_trios_csv, parse_trios_csv

# Excel reader
from rheojax.io.readers.trios.excel import load_trios_excel, parse_trios_excel

# JSON reader
from rheojax.io.readers.trios.json import load_trios_json
from rheojax.io.readers.trios.schema import TRIOSExperiment
from rheojax.io.readers.trios.txt import convert_units, load_trios_chunked

# TXT reader (original functionality)
from rheojax.io.readers.trios.txt import load_trios as load_trios_txt
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Supported formats and their extensions
TRIOS_FORMAT_EXTENSIONS: dict[str, str] = {
    ".txt": "txt",
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
}


def detect_trios_format(filepath: str | Path) -> str:
    """Detect TRIOS file format from extension.

    Args:
        filepath: Path to TRIOS file

    Returns:
        Format string: "txt", "csv", "excel", or "json"

    Raises:
        ValueError: If extension is not recognized
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in TRIOS_FORMAT_EXTENSIONS:
        return TRIOS_FORMAT_EXTENSIONS[ext]

    raise ValueError(
        f"Unknown TRIOS file format: {ext}. "
        f"Supported extensions: {list(TRIOS_FORMAT_EXTENSIONS.keys())}"
    )


def load_trios(
    filepath: str | Path,
    *,
    return_all_segments: bool = False,
    test_mode: str | None = None,
    encoding: str | None = None,
    decimal_separator: str = ".",
    validate: bool = True,
    auto_chunk: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    **kwargs: Any,
) -> RheoData | list[RheoData]:
    """Load TRIOS file with automatic format detection.

    Detects format from file extension and delegates to appropriate parser:
    - .txt → load_trios_txt (existing TXT behavior)
    - .csv → load_trios_csv
    - .xlsx, .xls → load_trios_excel
    - .json → load_trios_json

    Args:
        filepath: Path to TRIOS file
        return_all_segments: Return list for multi-step files (default: False)
        test_mode: Override auto-detection ("creep", "relaxation", "oscillation", "rotation")
        encoding: File encoding override (auto-detected if None)
        decimal_separator: "." for US (default), "," for European
        validate: Validate RheoData on creation (default: True)
        auto_chunk: Auto-chunk large files > 5MB (default: True, TXT only)
        progress_callback: Progress callback(current, total) for large files
        **kwargs: Format-specific options passed to underlying parser

    Returns:
        Single RheoData for single-segment files
        List of RheoData when return_all_segments=True or multiple segments

    Raises:
        FileNotFoundError: File does not exist
        ValueError: Unsupported format or no data found
        UnicodeDecodeError: Encoding detection failed

    Example:
        >>> data = load_trios("frequency_sweep.csv")
        >>> print(data.test_mode)  # 'oscillation'
        >>> print(data.is_complex)  # True (G* = G' + iG'')

        >>> data = load_trios("creep_recovery.xlsx", sheet_name="all")
        >>> print(len(data))  # Multiple sheets
    """
    filepath = Path(filepath)
    logger.info("Opening file", filepath=str(filepath))

    if not filepath.exists():
        logger.error("File not found", filepath=str(filepath))
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect format from extension
    file_format = detect_trios_format(filepath)

    logger.debug("Detected TRIOS format", format=file_format, filepath=str(filepath))

    # Dispatch to appropriate loader
    if file_format == "txt":
        result = load_trios_txt(
            filepath,
            return_all_segments=return_all_segments,
            auto_chunk=auto_chunk,
            progress_callback=progress_callback,
            **kwargs,
        )
        _log_parse_result(filepath, result)
        return result

    elif file_format == "csv":
        result = load_trios_csv(
            filepath,
            return_all_segments=return_all_segments,
            test_mode=test_mode,
            encoding=encoding,
            decimal_separator=decimal_separator,
            validate=validate,
            progress_callback=progress_callback,
            **kwargs,
        )
        _log_parse_result(filepath, result)
        return result

    elif file_format == "excel":
        # Excel-specific kwargs
        sheet_name = kwargs.pop("sheet_name", None)
        result = load_trios_excel(
            filepath,
            sheet_name=sheet_name,
            return_all_segments=return_all_segments,
            test_mode=test_mode,
            validate=validate,
            **kwargs,
        )
        _log_parse_result(filepath, result)
        return result

    elif file_format == "json":
        # JSON-specific kwargs
        result_index = kwargs.pop("result_index", 0)
        validate_schema = kwargs.pop("validate_schema", True)
        result = load_trios_json(
            filepath,
            return_all_segments=return_all_segments,
            test_mode=test_mode,
            result_index=result_index,
            validate_schema=validate_schema,
            validate=validate,
            **kwargs,
        )
        _log_parse_result(filepath, result)
        return result

    else:
        logger.error("Unsupported TRIOS format", format=file_format)
        raise ValueError(f"Unsupported TRIOS format: {file_format}")


def _log_parse_result(filepath: Path, result: RheoData | list[RheoData]) -> None:
    """Log parse completion with record count."""
    if isinstance(result, list):
        total_records = sum(len(r.x) for r in result)  # type: ignore[arg-type, misc]
        logger.info(
            "File parsed",
            filepath=str(filepath),
            n_segments=len(result),
            n_records=total_records,
        )
    else:
        logger.info(
            "File parsed",
            filepath=str(filepath),
            n_records=len(result.x),  # type: ignore[arg-type]
            test_mode=result.test_mode,
        )


__all__ = [
    # Public loaders
    "load_trios",
    "load_trios_chunked",
    "load_trios_txt",
    "load_trios_csv",
    "parse_trios_csv",
    "load_trios_excel",
    "parse_trios_excel",
    "load_trios_json",
    "TRIOSExperiment",
    # Format detection
    "detect_trios_format",
    "TRIOS_FORMAT_EXTENSIONS",
    # Data structures
    "TRIOSFile",
    "TRIOSTable",
    "DataSegment",
    "ColumnMapping",
    # Constants
    "TRIOS_COLUMN_MAPPINGS",
    "TRIOS_UNIT_CONVERSIONS",
    "STEP_COLUMN_CANDIDATES",
    # Utility functions
    "detect_test_type",
    "map_columns_to_canonical",
    "select_xy_columns",
    "convert_unit",
    "convert_units",
    "detect_step_column",
    "split_by_step",
    "construct_complex_modulus",
    "segment_to_rheodata",
]
