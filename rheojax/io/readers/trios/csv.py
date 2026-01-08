"""TA Instruments TRIOS CSV file reader.

This module provides a reader for TRIOS CSV exports with support for:
- Tab or comma delimiters (auto-detected)
- Metadata header rows
- Units in parentheses or separate row
- Step/Segment columns for multi-step experiments
- Complex modulus construction (G' + iG'')
- Automatic encoding detection (UTF-8, Latin-1, CP1252)

Usage:
    >>> from rheojax.io.readers.trios import load_trios_csv
    >>> data = load_trios_csv('frequency_sweep.csv')
    >>> print(data.test_mode)  # 'oscillation'
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData
from rheojax.io.readers.trios.common import (
    DataSegment,
    TRIOSFile,
    TRIOSTable,
    construct_complex_modulus,
    convert_unit,
    detect_step_column,
    detect_test_type,
    segment_to_rheodata,
    select_xy_columns,
    split_by_step,
)

logger = logging.getLogger(__name__)

# Encoding cascade for auto-detection
ENCODING_CASCADE = ["utf-8", "latin-1", "cp1252"]

# Auto-chunking threshold (5 MB)
AUTO_CHUNK_THRESHOLD_MB = 5.0


def detect_encoding(filepath: Path) -> str:
    """Detect file encoding using cascade approach.

    Tries UTF-8, Latin-1, and CP1252 in order.

    Args:
        filepath: Path to file

    Returns:
        Detected encoding string

    Raises:
        UnicodeDecodeError: If none of the encodings work
    """
    for encoding in ENCODING_CASCADE:
        try:
            with open(filepath, encoding=encoding) as f:
                # Read first 1KB to check encoding
                f.read(1024)
            return encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Could not decode file with any of: {ENCODING_CASCADE}",
    )


def detect_delimiter(content: str) -> str:
    """Detect delimiter (tab vs comma) from file content.

    TRIOS CSV files typically use tabs, but may use commas.

    Args:
        content: First few lines of file content

    Returns:
        Delimiter character ('\t' or ',')
    """
    # Count occurrences in first few lines
    lines = content.split("\n")[:10]
    tab_count = sum(line.count("\t") for line in lines)
    comma_count = sum(line.count(",") for line in lines)

    # Prefer tabs for TRIOS files (typical format)
    if tab_count >= comma_count:
        return "\t"
    return ","


def parse_metadata_header(
    lines: list[str],
    delimiter: str,
) -> tuple[dict[str, Any], int]:
    """Extract metadata from file header.

    TRIOS CSV files have metadata key-value pairs at the top,
    followed by the column headers.

    Args:
        lines: File lines
        delimiter: Field delimiter

    Returns:
        Tuple of (metadata dict, header row index)
    """
    metadata: dict[str, Any] = {}
    header_row = 0

    # Known metadata patterns
    metadata_patterns = {
        "filename": r"^Filename",
        "instrument_serial_number": r"^Instrument serial number",
        "instrument_name": r"^Instrument name",
        "operator": r"^[Oo]perator",
        "run_date": r"^[Rr]undate",
        "sample_name": r"^Sample name",
        "geometry": r"^Geometry name",
        "geometry_type": r"^Geometry type",
        "gap": r"^Gap",
        "temperature": r"^Temperature",
    }

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        # Check if this is a metadata line
        is_metadata = False
        for key, pattern in metadata_patterns.items():
            if re.match(pattern, line, re.IGNORECASE):
                parts = line.split(delimiter)
                if len(parts) >= 2:
                    value = parts[1].strip()
                    metadata[key] = value
                is_metadata = True
                break

        # Check if this looks like a header row (multiple text columns)
        if not is_metadata:
            parts = line.split(delimiter)
            # Header rows typically have "Variables" or multiple column names
            if (
                parts[0].strip().lower() == "variables"
                or len([p for p in parts if p.strip() and not p.strip().isdigit()]) >= 3
            ):
                header_row = i
                break
            # Or if it starts with "Number of points" we're close to data
            if parts[0].strip().lower() == "number of points":
                if len(parts) >= 2:
                    try:
                        metadata["number_of_points"] = int(parts[1].strip())
                    except ValueError:
                        pass
                # Header is next line
                header_row = i + 1
                break

    return metadata, header_row


def detect_header_row(
    lines: list[str],
    delimiter: str,
    start_index: int = 0,
) -> int:
    """Find first row with column headers (data table start).

    Args:
        lines: File lines
        delimiter: Field delimiter
        start_index: Index to start searching from

    Returns:
        Header row index
    """
    for i in range(start_index, len(lines)):
        line = lines[i].strip()
        if not line:
            continue

        parts = line.split(delimiter)

        # Check for "Variables" row (TRIOS format)
        if parts[0].strip().lower() == "variables":
            return i

        # Check for "Number of points" - header is next
        if parts[0].strip().lower() == "number of points":
            return i + 1

        # Check for multiple non-numeric columns (likely headers)
        non_numeric = 0
        for p in parts[1:]:  # Skip first column (often a label)
            if p.strip() and not _is_numeric(p.strip()):
                non_numeric += 1

        if non_numeric >= 2:
            return i

    return start_index


def _is_numeric(s: str) -> bool:
    """Check if string represents a numeric value."""
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        return False


def _default_x_units(test_mode: str) -> str:
    """Get default x-axis units for a test mode."""
    if test_mode == "oscillation":
        return "rad/s"
    elif test_mode == "rotation":
        return "1/s"
    return "s"


def extract_units_from_header(
    header: list[str],
    unit_row: list[str] | None = None,
) -> dict[str, str]:
    """Parse units from column headers or separate unit row.

    TRIOS exports may have units in parentheses: "Angular Frequency (rad/s)"
    Or in a separate row below headers.

    Args:
        header: Column header names
        unit_row: Optional separate unit row

    Returns:
        Dict mapping column names to units
    """
    units: dict[str, str] = {}

    for i, col in enumerate(header):
        col_clean = col.strip()

        # Check for units in parentheses
        match = re.search(r"\(([^)]+)\)$", col_clean)
        if match:
            units[col_clean] = match.group(1)
            # Also store under name without units
            name_without_units = re.sub(r"\s*\([^)]+\)$", "", col_clean).strip()
            units[name_without_units] = match.group(1)
        elif unit_row and i < len(unit_row):
            # Use separate unit row
            unit = unit_row[i].strip()
            if unit:
                units[col_clean] = unit

    return units


def detect_repeated_headers(
    lines: list[str],
    delimiter: str,
    first_header: list[str],
    start_index: int,
) -> list[int]:
    """Find multi-table boundaries (repeated header rows).

    Args:
        lines: File lines
        delimiter: Field delimiter
        first_header: Column headers from first table
        start_index: Index after first table header

    Returns:
        List of line indices where new tables begin
    """
    table_starts = []
    header_pattern = [h.lower().strip() for h in first_header[:3] if h.strip()]

    for i in range(start_index, len(lines)):
        line = lines[i].strip()
        if not line:
            continue

        parts = line.split(delimiter)
        if len(parts) >= len(header_pattern):
            current = [p.lower().strip() for p in parts[:3] if p.strip()]
            # Check if this looks like a repeated header
            if current == header_pattern:
                table_starts.append(i)

    return table_starts


def parse_trios_csv(
    filepath: str | Path,
    *,
    encoding: str | None = None,
    decimal_separator: str = ".",
    delimiter: str | None = None,
) -> TRIOSFile:
    """Low-level CSV parser returning raw TRIOSFile structure.

    For advanced users who need access to raw tables and metadata
    before RheoData conversion.

    Args:
        filepath: Path to TRIOS CSV file
        encoding: File encoding (auto-detected if None)
        decimal_separator: Decimal separator ("." or ",")
        delimiter: Delimiter override (None = auto)

    Returns:
        TRIOSFile with parsed tables and metadata

    Raises:
        FileNotFoundError: File does not exist
        ValueError: No data tables found
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect or use provided encoding
    if encoding is None:
        encoding = detect_encoding(filepath)

    # Read file content
    with open(filepath, encoding=encoding, errors="replace") as f:
        content = f.read()

    lines = content.split("\n")

    # Detect delimiter
    if delimiter is None:
        delimiter = detect_delimiter(content)

    # Parse metadata and find header row
    metadata, header_start = parse_metadata_header(lines, delimiter)

    # Find actual header row
    header_row = detect_header_row(lines, delimiter, header_start)

    if header_row >= len(lines):
        raise ValueError("No data tables found in TRIOS CSV file")

    # Parse header
    header_line = lines[header_row]
    header = [h.strip() for h in header_line.split(delimiter)]

    # Check for unit row (next line may contain units)
    unit_row = None
    data_start = header_row + 1

    if data_start < len(lines):
        next_line = lines[data_start]
        parts = next_line.split(delimiter)
        # Unit row typically has no leading label and contains unit strings
        if parts and not parts[0].strip().lower().startswith("data"):
            # Check if it looks like units (not numeric values)
            is_unit_row = True
            for p in parts[1:5]:  # Check first few columns
                if p.strip() and _is_numeric(p.strip()):
                    is_unit_row = False
                    break
            if is_unit_row:
                unit_row = parts
                data_start += 1

    # Extract units
    units = extract_units_from_header(header, unit_row)

    # Parse data rows
    data_rows = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("["):
            break

        parts = line.split(delimiter)
        if len(parts) == len(header):
            row = []
            for j, val in enumerate(parts):
                val_clean = val.strip()
                if j == 0:
                    # First column is often "Data point" label - skip
                    if val_clean.lower().startswith("data"):
                        continue
                    row.append(np.nan)
                elif not val_clean:
                    row.append(np.nan)
                else:
                    try:
                        if decimal_separator == ",":
                            val_clean = val_clean.replace(",", ".")
                        row.append(float(val_clean))
                    except ValueError:
                        row.append(np.nan)
            if row:
                data_rows.append(row)

    if not data_rows:
        raise ValueError("No data rows found in TRIOS CSV file")

    # Adjust header to match data (skip first column if it's a label)
    if header[0].lower() == "variables" or header[0].lower().startswith("data"):
        header = header[1:]
        units = {
            k: v
            for k, v in units.items()
            if k.lower() != "variables" and not k.lower().startswith("data")
        }

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=header[: len(data_rows[0])])

    # Detect step column
    step_col = detect_step_column(df)
    step_values = None
    if step_col:
        step_values = df[step_col].unique().tolist()

    # Create TRIOSTable
    table = TRIOSTable(
        table_index=0,
        header=list(df.columns),
        units=units,
        df=df,
        step_values=step_values,
    )

    return TRIOSFile(
        filepath=str(filepath),
        format="csv",
        metadata=metadata,
        tables=[table],
        encoding=encoding,
        decimal_separator=decimal_separator,
    )


def load_trios_csv(
    filepath: str | Path,
    *,
    return_all_segments: bool = False,
    test_mode: str | None = None,
    encoding: str | None = None,
    decimal_separator: str = ".",
    delimiter: str | None = None,
    validate: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RheoData | list[RheoData]:
    """Load TRIOS CSV export file.

    Handles TRIOS-specific CSV format with:
    - Metadata header rows before data
    - Tab or comma delimiters (auto-detected)
    - Units in parentheses or separate row
    - Step/Segment columns for multi-step experiments
    - Repeated headers for multi-table files

    Args:
        filepath: Path to TRIOS CSV file
        return_all_segments: Return list for multi-step files
        test_mode: Override auto-detection ("creep", "relaxation", "oscillation", "rotation")
        encoding: File encoding (auto-detected: UTF-8, Latin-1, CP1252)
        decimal_separator: "." for US, "," for European
        delimiter: Override delimiter detection (None = auto)
        validate: Validate RheoData on creation
        progress_callback: Progress callback(current, total)

    Returns:
        Single RheoData or list of RheoData

    Raises:
        FileNotFoundError: File does not exist
        ValueError: No data found or invalid format

    Example:
        >>> data = load_trios_csv('frequency_sweep.csv')
        >>> print(data.test_mode)  # 'oscillation'
        >>> print(np.iscomplexobj(data.y))  # True for G* = G' + iG''
    """
    # Parse CSV file
    trios_file = parse_trios_csv(
        filepath,
        encoding=encoding,
        decimal_separator=decimal_separator,
        delimiter=delimiter,
    )

    # Convert tables to RheoData
    rheo_data_list: list[RheoData] = []

    for table in trios_file.tables:
        df = table.df
        units = table.units

        # Detect or use provided test mode
        detected_mode = test_mode or detect_test_type(df)

        # Check for step column and split if needed
        step_col = detect_step_column(df)
        segments = (
            [df]
            if not step_col or not return_all_segments
            else split_by_step(df, step_col)
        )

        for seg_idx, seg_df in enumerate(segments):
            # Select x/y columns
            x_col, y_col, y2_col = select_xy_columns(seg_df, detected_mode)

            if x_col is None or y_col is None:
                logger.warning(
                    f"Could not determine x/y columns for segment {seg_idx}. "
                    f"Available columns: {list(seg_df.columns)}"
                )
                continue

            # Extract data
            x_data = seg_df[x_col].values.astype(float)

            # Get units
            x_units = units.get(x_col, "")
            y_units = units.get(y_col, "Pa")

            # Handle complex modulus case
            if y2_col is not None:
                y_real = seg_df[y_col].values.astype(float)
                y_imag = seg_df[y2_col].values.astype(float)

                # Convert units if needed
                y_units_orig = units.get(y_col, "Pa")
                y2_units_orig = units.get(y2_col, "Pa")
                y_real, _ = convert_unit(y_real, y_units_orig, "Pa")
                y_imag, _ = convert_unit(y_imag, y2_units_orig, "Pa")

                # Construct complex modulus
                y_data = construct_complex_modulus(y_real, y_imag)
                y_units = "Pa"
                is_complex = True
            else:
                y_data = seg_df[y_col].values.astype(float)
                is_complex = False

            # Convert x units (e.g., Hz to rad/s)
            if detected_mode == "oscillation":
                x_data, x_units = convert_unit(x_data, x_units, "rad/s")

            # Remove NaN values
            if is_complex:
                valid_mask = ~(
                    np.isnan(x_data)
                    | np.isnan(np.real(y_data))
                    | np.isnan(np.imag(y_data))
                )
            else:
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))

            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]

            if len(x_data) == 0:
                continue

            # Build metadata
            seg_metadata = trios_file.metadata.copy()
            seg_metadata["test_mode"] = detected_mode
            seg_metadata["source_format"] = "csv"
            seg_metadata["x_column"] = x_col
            seg_metadata["y_column"] = y_col
            if y2_col:
                seg_metadata["y2_column"] = y2_col
            seg_metadata["is_complex"] = is_complex

            # Create DataSegment and convert to RheoData
            segment = DataSegment(
                segment_index=seg_idx,
                test_mode=detected_mode,
                x_data=x_data,
                y_data=y_data,
                x_column=x_col,
                y_column=y_col,
                x_units=x_units or _default_x_units(detected_mode),
                y_units=y_units,
                is_complex=is_complex,
                metadata=seg_metadata,
            )

            rheo_data = segment_to_rheodata(segment, validate=validate)
            rheo_data_list.append(rheo_data)

    if not rheo_data_list:
        raise ValueError(f"No valid data segments could be parsed from {filepath}")

    # Return single or list
    if len(rheo_data_list) == 1 and not return_all_segments:
        return rheo_data_list[0]
    return rheo_data_list
