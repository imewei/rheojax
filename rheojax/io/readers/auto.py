"""Auto-detection wrapper for file readers."""

from __future__ import annotations

import warnings
from pathlib import Path

from rheo.core.data import RheoData
from rheo.io.readers.csv_reader import load_csv
from rheo.io.readers.excel_reader import load_excel
from rheo.io.readers.trios import load_trios


def auto_load(filepath: str, **kwargs) -> RheoData | list[RheoData]:
    """Automatically detect file format and load data.

    This function attempts to determine the file format based on:
    1. File extension
    2. File content inspection
    3. Sequential reader attempts

    Args:
        filepath: Path to data file
        **kwargs: Additional arguments passed to specific readers
            - x_col, y_col: Required for CSV/Excel if auto-detection fails
            - return_all_segments: For TRIOS files with multiple segments

    Returns:
        RheoData object or list of RheoData objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no reader can parse the file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    extension = filepath.suffix.lower()

    # Try based on file extension first
    if extension == ".txt":
        return _try_trios_then_csv(filepath, **kwargs)

    elif extension == ".csv":
        return _try_csv(filepath, **kwargs)

    elif extension in [".xlsx", ".xls"]:
        return _try_excel(filepath, **kwargs)

    elif extension == ".tsv":
        kwargs["delimiter"] = "\t"
        return _try_csv(filepath, **kwargs)

    else:
        # Unknown extension - try readers in sequence
        return _try_all_readers(filepath, **kwargs)


def _try_trios_then_csv(filepath: Path, **kwargs) -> RheoData | list[RheoData]:
    """Try TRIOS reader first, then CSV.

    Args:
        filepath: File path
        **kwargs: Additional arguments

    Returns:
        RheoData object(s)
    """
    # Try TRIOS first
    try:
        return load_trios(str(filepath), **kwargs)
    except Exception as e:
        warnings.warn(f"TRIOS reader failed: {e}. Trying CSV reader.", stacklevel=2)

    # Try CSV as fallback
    try:
        return _try_csv(filepath, **kwargs)
    except Exception as e:
        raise ValueError(f"Could not parse file as TRIOS or CSV: {e}")


def _try_csv(filepath: Path, **kwargs) -> RheoData:
    """Try CSV reader with auto-detection.

    Args:
        filepath: File path
        **kwargs: Additional arguments

    Returns:
        RheoData object
    """
    # Check if x_col and y_col are specified
    if "x_col" not in kwargs or "y_col" not in kwargs:
        # Try to auto-detect common column names
        import pandas as pd

        try:
            df = pd.read_csv(filepath)
            columns_lower = [c.lower() for c in df.columns]

            # Try to find time/frequency column
            x_col = None
            for col_name in [
                "time",
                "frequency",
                "angular frequency",
                "t",
                "f",
                "omega",
            ]:
                if col_name in columns_lower:
                    x_col = df.columns[columns_lower.index(col_name)]
                    break

            # Try to find stress/modulus column
            y_col = None
            for col_name in [
                "stress",
                "strain",
                "modulus",
                "storage modulus",
                "viscosity",
            ]:
                if col_name in columns_lower:
                    y_col = df.columns[columns_lower.index(col_name)]
                    break

            if x_col is None or y_col is None:
                raise ValueError(
                    "Could not auto-detect x and y columns. Please specify x_col and y_col."
                )

            kwargs["x_col"] = x_col
            kwargs["y_col"] = y_col

        except Exception as e:
            raise ValueError(
                f"Could not auto-detect columns: {e}. Please specify x_col and y_col."
            )

    return load_csv(str(filepath), **kwargs)


def _try_excel(filepath: Path, **kwargs) -> RheoData:
    """Try Excel reader with auto-detection.

    Args:
        filepath: File path
        **kwargs: Additional arguments

    Returns:
        RheoData object
    """
    # Check if x_col and y_col are specified
    if "x_col" not in kwargs or "y_col" not in kwargs:
        raise ValueError("For Excel files, please specify x_col and y_col parameters")

    return load_excel(str(filepath), **kwargs)


def _try_all_readers(filepath: Path, **kwargs) -> RheoData | list[RheoData]:
    """Try all available readers in sequence.

    Args:
        filepath: File path
        **kwargs: Additional arguments

    Returns:
        RheoData object(s)

    Raises:
        ValueError: If no reader can parse the file
    """
    readers = [
        ("TRIOS", lambda: load_trios(str(filepath), **kwargs)),
        ("CSV", lambda: _try_csv(filepath, **kwargs)),
    ]

    errors = []
    for reader_name, reader_func in readers:
        try:
            return reader_func()
        except Exception as e:
            errors.append(f"{reader_name}: {e}")

    # All readers failed
    error_msg = "Could not parse file with any available reader:\n" + "\n".join(errors)
    raise ValueError(error_msg)
