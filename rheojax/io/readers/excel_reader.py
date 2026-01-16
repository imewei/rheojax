"""Excel file reader for rheological data."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from rheojax.core.data import RheoData
from rheojax.io.readers._utils import (
    VALID_TEST_MODES,
    VALID_TRANSFORMS,
    construct_complex_modulus,
    detect_domain,
    detect_test_mode_from_columns,
    extract_unit_from_header,
    validate_transform,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


def load_excel(
    filepath: str | Path,
    x_col: str | int,
    y_col: str | int | None = None,
    *,
    y_cols: list[str | int] | None = None,
    sheet: str | int = 0,
    x_units: str | None = None,
    y_units: str | None = None,
    domain: str | None = None,
    test_mode: str | None = None,
    temperature: float | None = None,
    metadata: dict | None = None,
    intended_transform: str | None = None,
    header: int | None = 0,
    **kwargs,
) -> RheoData:
    """Load data from Excel file into RheoData.

    Args:
        filepath: Path to Excel file (.xlsx or .xls).
        x_col: Column name or index for x-axis data.
        y_col: Column name or index for y-axis data (single column).
            Mutually exclusive with y_cols.
        y_cols: List of two column names/indices for complex modulus [G', G''].
            First column is storage modulus (G'), second is loss modulus (G'').
            Mutually exclusive with y_col.
        sheet: Sheet name or index (default: 0 - first sheet).
        x_units: Units for x-axis (auto-detected from header if None).
        y_units: Units for y-axis (auto-detected from header if None).
        domain: Data domain ('time' or 'frequency', auto-detected if None).
        test_mode: Test mode ('relaxation', 'creep', 'oscillation', 'rotation').
            Auto-detected if None.
        temperature: Temperature in Kelvin for TTS workflows.
        metadata: Additional metadata dict to merge.
        intended_transform: Transform type for metadata validation. One of
            'mastercurve', 'srfs', 'owchirp', 'spp', 'fft', 'mutation', 'derivative'.
        header: Row number for column headers (None if no header).
        **kwargs: Additional arguments passed to pandas.read_excel.

    Returns:
        RheoData object with populated fields.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ImportError: If pandas or openpyxl not installed.
        KeyError: If specified columns or sheet don't exist.
        ValueError: If data cannot be parsed, y_cols has wrong length,
            or both y_col and y_cols are provided.

    Warnings:
        UserWarning: If intended_transform metadata is missing.
        UserWarning: If domain incompatible with intended_transform.
        UserWarning: If test_mode conflicts with intended_transform.

    Example:
        >>> # Simple creep data from specific sheet
        >>> data = load_excel(
        ...     "data.xlsx",
        ...     x_col="time (s)",
        ...     y_col="J(t) (1/Pa)",
        ...     sheet="Creep Test",
        ... )
        >>> # Flow curve with explicit test mode
        >>> data = load_excel(
        ...     "flow_curve.xlsx",
        ...     x_col=0,
        ...     y_col=1,
        ...     test_mode='rotation',
        ...     x_units='1/s',
        ...     y_units='Pa·s',
        ... )
        >>> # Complex modulus from Excel
        >>> data = load_excel(
        ...     "frequency_sweep.xlsx",
        ...     x_col="omega (rad/s)",
        ...     y_cols=["G' (Pa)", "G'' (Pa)"],
        ...     intended_transform='mastercurve',
        ...     temperature=298.15,
        ... )
    """
    try:
        import pandas as pd
    except ImportError as exc:
        logger.error("pandas not installed for Excel reading", exc_info=True)
        raise ImportError(
            "pandas is required for Excel reading. Install with: pip install pandas openpyxl"
        ) from exc

    filepath = Path(filepath)
    logger.info("Opening file", filepath=str(filepath))

    if not filepath.exists():
        logger.error("File not found", filepath=str(filepath))
        raise FileNotFoundError(f"File not found: {filepath}")

    # Validate y_col / y_cols mutual exclusivity
    if y_col is not None and y_cols is not None:
        raise ValueError("Cannot specify both y_col and y_cols. Use one or the other.")
    if y_col is None and y_cols is None:
        raise ValueError("Must specify either y_col or y_cols.")
    if y_cols is not None and len(y_cols) != 2:
        raise ValueError(
            f"y_cols must contain exactly 2 columns [G', G'']. Got {len(y_cols)} columns."
        )

    # Validate test_mode if provided
    if test_mode is not None and test_mode.lower() not in VALID_TEST_MODES:
        raise ValueError(
            f"Invalid test_mode '{test_mode}'. "
            f"Valid options: {sorted(VALID_TEST_MODES)}"
        )

    # Validate intended_transform if provided
    if (
        intended_transform is not None
        and intended_transform.lower() not in VALID_TRANSFORMS
    ):
        raise ValueError(
            f"Invalid intended_transform '{intended_transform}'. "
            f"Valid options: {sorted(VALID_TRANSFORMS)}"
        )

    # Build list of columns to load (memory optimization for wide files)
    # Only use usecols when all column specifiers are strings (not indices)
    usecols = None
    if isinstance(x_col, str):
        cols_needed = [x_col]
        if y_col is not None and isinstance(y_col, str):
            cols_needed.append(y_col)
        elif y_cols is not None:
            cols_needed.extend([c for c in y_cols if isinstance(c, str)])
        # Only set usecols if all columns are strings
        if len(cols_needed) == (1 + (1 if y_col is not None else len(y_cols or []))):
            usecols = cols_needed

    # Read Excel file
    try:
        logger.debug("Reading Excel file", sheet=sheet)
        df = pd.read_excel(
            filepath, sheet_name=sheet, header=header, usecols=usecols, **kwargs
        )
    except Exception as e:
        logger.error(
            "Failed to parse Excel file", filepath=str(filepath), exc_info=True
        )
        raise ValueError(f"Failed to parse Excel file: {e}") from e

    logger.debug("Excel file read successfully", n_rows=len(df), n_cols=len(df.columns))

    # Get column headers for detection
    x_header = _get_column_header(df, x_col)

    # Extract x data
    try:
        x_data = _get_column_data(df, x_col)
    except (KeyError, IndexError) as e:
        logger.error("X column not found", x_col=x_col, exc_info=True)
        raise KeyError(f"X column not found: {e}") from e

    # Extract y data (single column or complex modulus)
    is_complex = y_cols is not None
    if is_complex:
        y_headers = [_get_column_header(df, col) for col in y_cols]
        try:
            g_prime_data = _get_column_data(df, y_cols[0])
            g_double_prime_data = _get_column_data(df, y_cols[1])
        except (KeyError, IndexError) as e:
            logger.error("Y column not found", y_cols=y_cols, exc_info=True)
            raise KeyError(f"Y column not found: {e}") from e

        # Convert to float arrays before constructing complex modulus
        g_prime_data = np.array(g_prime_data, dtype=float)
        g_double_prime_data = np.array(g_double_prime_data, dtype=float)
        y_data = construct_complex_modulus(g_prime_data, g_double_prime_data)
        logger.debug("Constructed complex modulus from G' and G''")
    else:
        y_headers = [_get_column_header(df, y_col)]
        try:
            y_data = _get_column_data(df, y_col)
        except (KeyError, IndexError) as e:
            logger.error("Y column not found", y_col=y_col, exc_info=True)
            raise KeyError(f"Y column not found: {e}") from e

    # Convert to numpy arrays and handle NaN
    x_data = np.array(x_data, dtype=float)
    if not is_complex:
        y_data = np.array(y_data, dtype=float)

    # Remove NaN values in single pass (avoid intermediate copies)
    if is_complex:
        valid_idx = np.flatnonzero(
            ~(np.isnan(x_data) | np.isnan(y_data.real) | np.isnan(y_data.imag))
        )
    else:
        valid_idx = np.flatnonzero(~(np.isnan(x_data) | np.isnan(y_data)))
    x_data = np.take(x_data, valid_idx)
    y_data = np.take(y_data, valid_idx)

    if len(x_data) == 0:
        logger.error(
            "No valid data points after removing NaN values", filepath=str(filepath)
        )
        raise ValueError("No valid data points after removing NaN values")

    logger.debug("Data points after NaN removal", n_points=len(x_data))

    # Auto-extract units from headers if not provided
    if x_units is None:
        _, x_units = extract_unit_from_header(x_header)
    if y_units is None:
        # Use first y column header for units
        _, y_units = extract_unit_from_header(y_headers[0])

    # Auto-detect domain if not provided
    if domain is None:
        domain = detect_domain(x_header, x_units, y_headers)
        logger.debug("Auto-detected domain", domain=domain)

    # Auto-detect test mode if not provided
    detected_test_mode = None
    if test_mode is None:
        detected_test_mode = detect_test_mode_from_columns(
            x_header, y_headers, x_units, y_units
        )
        # If y_cols provided, default to oscillation
        if detected_test_mode is None and is_complex:
            detected_test_mode = "oscillation"
        logger.debug("Auto-detected test mode", test_mode=detected_test_mode)
    else:
        detected_test_mode = test_mode.lower()

    # Build source metadata
    source_metadata = {
        "source_file": str(filepath.absolute()),
        "file_type": "excel",
        "sheet": sheet,
        "x_column": x_col,
        "y_column": y_cols if is_complex else y_col,
    }

    # Merge with user metadata
    final_metadata = {**source_metadata}
    if metadata:
        final_metadata.update(metadata)

    # Add temperature if provided
    if temperature is not None:
        final_metadata["temperature"] = temperature

    # Add intended_transform if provided
    if intended_transform is not None:
        final_metadata["intended_transform"] = intended_transform.lower()

        # Validate transform requirements and emit warnings
        warning_messages = validate_transform(
            intended_transform,
            domain,
            final_metadata,
            detected_test_mode,
        )
        for msg in warning_messages:
            warnings.warn(msg, UserWarning, stacklevel=2)

    logger.info(
        "File parsed",
        filepath=str(filepath),
        n_records=len(x_data),
        test_mode=detected_test_mode,
        domain=domain,
    )

    return RheoData(
        x=x_data,
        y=y_data,
        x_units=x_units,
        y_units=y_units,
        domain=domain,
        initial_test_mode=detected_test_mode,
        metadata=final_metadata,
        validate=True,
    )


def _get_column_header(df, col: str | int) -> str:
    """Get column header string from DataFrame."""
    if isinstance(col, str):
        return col
    return str(df.columns[col])


def _get_column_data(df, col: str | int):
    """Get column data from DataFrame."""
    if isinstance(col, str):
        return df[col].values
    return df.iloc[:, col].values
