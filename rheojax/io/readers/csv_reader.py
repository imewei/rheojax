"""CSV file reader for rheological data."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData

# Exported for lightweight preview/loading helpers
__all__ = ["load_csv", "detect_csv_delimiter"]


def load_csv(
    filepath: str | Path,
    x_col: str | int,
    y_col: str | int,
    x_units: str | None = None,
    y_units: str | None = None,
    domain: str = "time",
    delimiter: str | None = None,
    header: int | None = 0,
    **kwargs,
) -> RheoData:
    """Load data from CSV file.

    Args:
        filepath: Path to CSV file
        x_col: Column name or index for x-axis data
        y_col: Column name or index for y-axis data
        x_units: Units for x-axis (optional)
        y_units: Units for y-axis (optional)
        domain: Data domain ('time' or 'frequency')
        delimiter: Column delimiter (auto-detected if None)
        header: Row number for column headers (None if no header)
        **kwargs: Additional arguments passed to pandas.read_csv

    Returns:
        RheoData object

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If specified columns don't exist
        ValueError: If data cannot be parsed
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Auto-detect delimiter if not specified
    if delimiter is None:
        delimiter = detect_csv_delimiter(filepath)

    # Choose encoding based on BOM/byte sniffing
    default_encoding = "utf-8-sig"
    try:
        head_bytes = filepath.read_bytes()[:4]
        if b"\xff\xfe" in head_bytes or b"\xfe\xff" in head_bytes or b"\x00" in head_bytes:
            default_encoding = "utf-16"
    except FileNotFoundError:
        raise

    # Read CSV file with tolerant encoding/dialect handling
    read_kwargs = dict(
        sep=delimiter,
        header=header,
        encoding=default_encoding,
        encoding_errors="replace",
        engine="python",
        **kwargs,
    )
    tried_utf16 = False
    try:
        df = pd.read_csv(filepath, **read_kwargs)
    except UnicodeDecodeError:
        read_kwargs["encoding"] = "utf-16le"
        tried_utf16 = True
        df = pd.read_csv(filepath, **read_kwargs)
    except Exception as e:
        # If UTF-8 path failed and we haven't tried utf-16, attempt before giving up
        if not tried_utf16:
            try:
                read_kwargs["encoding"] = "utf-16le"
                df = pd.read_csv(filepath, **read_kwargs)
            except Exception:
                raise ValueError(f"Failed to parse CSV file: {e}") from e
        else:
            raise ValueError(f"Failed to parse CSV file: {e}") from e

    # Extract x and y columns
    try:
        x_data = (
            df[x_col].values if isinstance(x_col, str) else df.iloc[:, x_col].values
        )
        y_data = (
            df[y_col].values if isinstance(y_col, str) else df.iloc[:, y_col].values
        )
    except (KeyError, IndexError) as e:
        raise KeyError(f"Column not found: {e}") from e

    # Convert to numpy arrays and handle NaN
    def _to_float(arr):
        arr = np.array(arr)
        if arr.dtype.kind in {"U", "S", "O"}:
            # Handle European decimal comma by replacement if needed
            arr = np.char.replace(arr.astype(str), ",", ".")
        return arr.astype(float)

    x_data = _to_float(x_data)
    y_data = _to_float(y_data)

    # Remove NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]

    if len(x_data) == 0:
        raise ValueError("No valid data points after removing NaN values")

    # Create metadata
    metadata = {
        "source_file": str(filepath),
        "file_type": "csv",
        "x_column": x_col,
        "y_column": y_col,
    }

    return RheoData(
        x=x_data,
        y=y_data,
        x_units=x_units,
        y_units=y_units,
        domain=domain,
        metadata=metadata,
        validate=True,
    )


def _detect_delimiter(filepath: Path) -> str:
    """Auto-detect CSV delimiter using csv.Sniffer with fallbacks."""
    try:
        with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                return dialect.delimiter
            except Exception:
                pass
    except FileNotFoundError:
        raise

    # Fallback heuristic
    delimiters = [",", "\t", ";", "|"]
    counts = {d: sample.count(d) for d in delimiters}
    best = max(counts, key=counts.get)
    return best or ","


def detect_csv_delimiter(filepath: str | Path) -> str:
    """Public helper to auto-detect CSV/TSV delimiter.

    Wrapper around the internal detection so that GUI helpers and previews
    can share the same logic as the main CSV reader.

    Args:
        filepath: Path to the text-based data file

    Returns:
        Detected delimiter character
    """

    return _detect_delimiter(Path(filepath))
