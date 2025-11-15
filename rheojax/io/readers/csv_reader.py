"""CSV file reader for rheological data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData


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
        delimiter = _detect_delimiter(filepath)

    # Read CSV file
    try:
        df = pd.read_csv(filepath, sep=delimiter, header=header, **kwargs)
    except Exception as e:
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
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

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
    """Auto-detect CSV delimiter.

    Args:
        filepath: Path to file

    Returns:
        Detected delimiter character
    """
    with open(filepath, encoding='utf-8') as f:
        # Read first few lines
        lines = [f.readline() for _ in range(5)]

    # Count occurrences of common delimiters
    delimiters = [",", "\t", ";", "|"]
    delimiter_counts = dict.fromkeys(delimiters, 0)

    for line in lines:
        for delimiter in delimiters:
            delimiter_counts[delimiter] += line.count(delimiter)

    # Return most common delimiter
    max_count = max(delimiter_counts.values())
    if max_count == 0:
        return ","  # Default to comma

    return max(delimiter_counts, key=delimiter_counts.get)
