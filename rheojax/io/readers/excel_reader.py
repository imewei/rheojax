"""Excel file reader for rheological data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rheojax.core.data import RheoData


def load_excel(
    filepath: str,
    x_col: str | int,
    y_col: str | int,
    sheet: str | int = 0,
    x_units: str | None = None,
    y_units: str | None = None,
    domain: str = "time",
    header: int | None = 0,
    **kwargs,
) -> RheoData:
    """Load data from Excel file.

    Args:
        filepath: Path to Excel file (.xlsx or .xls)
        x_col: Column name or index for x-axis data
        y_col: Column name or index for y-axis data
        sheet: Sheet name or index (default: 0 - first sheet)
        x_units: Units for x-axis (optional)
        y_units: Units for y-axis (optional)
        domain: Data domain ('time' or 'frequency')
        header: Row number for column headers (None if no header)
        **kwargs: Additional arguments passed to pandas.read_excel

    Returns:
        RheoData object

    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If pandas or openpyxl not installed
        KeyError: If specified columns or sheet don't exist
        ValueError: If data cannot be parsed
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for Excel reading. Install with: pip install pandas openpyxl"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read Excel file
    try:
        df = pd.read_excel(filepath, sheet_name=sheet, header=header, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to parse Excel file: {e}")

    # Extract x and y columns
    try:
        x_data = (
            df[x_col].values if isinstance(x_col, str) else df.iloc[:, x_col].values
        )
        y_data = (
            df[y_col].values if isinstance(y_col, str) else df.iloc[:, y_col].values
        )
    except (KeyError, IndexError) as e:
        raise KeyError(f"Column not found: {e}")

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
        "file_type": "excel",
        "sheet": sheet,
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
