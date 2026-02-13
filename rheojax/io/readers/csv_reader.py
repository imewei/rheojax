"""CSV file reader for rheological data."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData
from rheojax.io.readers._utils import (
    VALID_TEST_MODES,
    VALID_TRANSFORMS,
    construct_complex_modulus,
    detect_deformation_mode_from_columns,
    detect_domain,
    detect_test_mode_from_columns,
    extract_unit_from_header,
    validate_transform,
)
from rheojax.logging import get_logger, log_io

logger = get_logger(__name__)

# Exported for lightweight preview/loading helpers
__all__ = ["load_csv", "detect_csv_delimiter"]


def load_csv(
    filepath: str | Path,
    x_col: str | int,
    y_col: str | int | None = None,
    *,
    y_cols: list[str | int] | None = None,
    x_units: str | None = None,
    y_units: str | None = None,
    domain: str | None = None,
    test_mode: str | None = None,
    deformation_mode: str | None = None,
    temperature: float | None = None,
    metadata: dict | None = None,
    intended_transform: str | None = None,
    delimiter: str | None = None,
    encoding: str | None = None,
    header: int | None = 0,
    **kwargs,
) -> RheoData:
    """Load data from CSV or ASCII text file into RheoData.

    Args:
        filepath: Path to CSV or text file.
        x_col: Column name or index for x-axis data.
        y_col: Column name or index for y-axis data (single column).
            Mutually exclusive with y_cols.
        y_cols: List of two column names/indices for complex modulus [G', G'']
            or [E', E''].
            First column is storage modulus, second is loss modulus.
            Mutually exclusive with y_col.
        x_units: Units for x-axis (auto-detected from header if None).
        y_units: Units for y-axis (auto-detected from header if None).
        domain: Data domain ('time' or 'frequency', auto-detected if None).
        test_mode: Test mode ('relaxation', 'creep', 'oscillation', 'rotation').
            Auto-detected if None.
        deformation_mode: Deformation mode ('shear', 'tension', 'bending',
            'compression'). Auto-detected from column names if None.
            If 'tension'/'bending'/'compression', sets metadata for DMTA support.
        temperature: Temperature in Kelvin for TTS workflows.
        metadata: Additional metadata dict to merge.
        intended_transform: Transform type for metadata validation. One of
            'mastercurve', 'srfs', 'owchirp', 'spp', 'fft', 'mutation', 'derivative'.
        delimiter: Column delimiter (auto-detected if None).
        encoding: File encoding (e.g. 'utf-8', 'latin-1', 'cp1252').
            Auto-detected if None. Use this to override detection for files
            with known encoding.
        header: Row number for column headers (None if no header).
        **kwargs: Additional arguments passed to pandas.read_csv.

    Returns:
        RheoData object with populated fields.

    Raises:
        FileNotFoundError: If file doesn't exist.
        KeyError: If specified columns don't exist.
        ValueError: If data cannot be parsed, y_cols has wrong length,
            or both y_col and y_cols are provided.

    Warnings:
        UserWarning: If intended_transform metadata is missing.
        UserWarning: If domain incompatible with intended_transform.
        UserWarning: If test_mode conflicts with intended_transform.

    Example:
        >>> # Simple relaxation data
        >>> data = load_csv("relaxation.csv", x_col="time (s)", y_col="G(t) (Pa)")
        >>> # Complex modulus oscillation data
        >>> data = load_csv(
        ...     "frequency_sweep.csv",
        ...     x_col="omega (rad/s)",
        ...     y_cols=["G' (Pa)", "G'' (Pa)"],
        ...     intended_transform='mastercurve',
        ...     temperature=298.15,
        ... )
    """
    filepath = Path(filepath)

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

    # Auto-detect delimiter if not specified
    if delimiter is None:
        delimiter = detect_csv_delimiter(filepath)
        logger.debug("Auto-detected delimiter", delimiter=repr(delimiter))

    # Choose encoding: explicit parameter > BOM/byte sniffing > default
    if encoding is not None:
        default_encoding = encoding
        logger.debug("Using explicit encoding", encoding=encoding)
    else:
        default_encoding = "utf-8-sig"
        try:
            head_bytes = filepath.read_bytes()[:4]
            if (
                b"\xff\xfe" in head_bytes
                or b"\xfe\xff" in head_bytes
                or b"\x00" in head_bytes
            ):
                default_encoding = "utf-16"
            logger.debug("Auto-detected encoding", encoding=default_encoding)
        except FileNotFoundError:
            raise

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

    # Read CSV file with tolerant encoding/dialect handling
    read_kwargs = dict(
        sep=delimiter,
        header=header,
        encoding=default_encoding,
        encoding_errors="replace",
        engine="python",
        usecols=usecols,
        **kwargs,
    )
    tried_utf16 = False

    used_encoding = default_encoding

    with log_io(logger, "read", filepath=str(filepath)) as io_ctx:
        try:
            logger.debug("Reading CSV file", encoding=default_encoding)
            df = pd.read_csv(filepath, **read_kwargs)
        except UnicodeDecodeError:
            read_kwargs["encoding"] = "utf-16le"
            tried_utf16 = True
            logger.info(
                "Encoding fallback triggered",
                filepath=str(filepath),
                from_encoding=default_encoding,
                to_encoding="utf-16le",
            )
            df = pd.read_csv(filepath, **read_kwargs)
            used_encoding = "utf-16le"
        except Exception as e:
            # If UTF-8 path failed and we haven't tried utf-16, attempt before giving up
            if not tried_utf16:
                try:
                    read_kwargs["encoding"] = "utf-16le"
                    logger.info(
                        "Encoding fallback triggered",
                        filepath=str(filepath),
                        from_encoding=default_encoding,
                        to_encoding="utf-16le",
                    )
                    df = pd.read_csv(filepath, **read_kwargs)
                    used_encoding = "utf-16le"
                except Exception:
                    logger.error(
                        "Failed to parse CSV file",
                        filepath=str(filepath),
                        tried_encodings=[default_encoding, "utf-16le"],
                        exc_info=True,
                    )
                    raise ValueError(f"Failed to parse CSV file: {e}") from e
            else:
                logger.error(
                    "Failed to parse CSV file",
                    filepath=str(filepath),
                    tried_encodings=[default_encoding, "utf-16le"],
                    exc_info=True,
                )
                raise ValueError(f"Failed to parse CSV file: {e}") from e

        io_ctx["rows"] = len(df)
        io_ctx["columns"] = len(df.columns)
        io_ctx["encoding"] = used_encoding
        logger.debug(
            "CSV file read successfully",
            n_rows=len(df),
            n_cols=len(df.columns),
            encoding=used_encoding,
        )

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
        y_headers = [_get_column_header(df, col) for col in y_cols]  # type: ignore[union-attr]
        try:
            g_prime_data = _get_column_data(df, y_cols[0])  # type: ignore[index]
            g_double_prime_data = _get_column_data(df, y_cols[1])  # type: ignore[index]
        except (KeyError, IndexError) as e:
            logger.error("Y column not found", y_cols=y_cols, exc_info=True)
            raise KeyError(f"Y column not found: {e}") from e
        y_data = construct_complex_modulus(g_prime_data, g_double_prime_data)
        logger.debug("Constructed complex modulus from G' and G''")
    else:
        y_headers = [_get_column_header(df, y_col)]  # type: ignore[arg-type]
        try:
            y_data = _get_column_data(df, y_col)  # type: ignore[arg-type]
        except (KeyError, IndexError) as e:
            logger.error("Y column not found", y_col=y_col, exc_info=True)
            raise KeyError(f"Y column not found: {e}") from e

    # Convert to numpy arrays and handle NaN
    x_data = _to_float(x_data)
    if not is_complex:
        y_data = _to_float(y_data)

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

    # Build source metadata (includes encoding provenance for debugging)
    source_metadata = {
        "source_file": str(filepath.absolute()),
        "file_type": "csv" if filepath.suffix.lower() in {".csv", ""} else "txt",
        "x_column": x_col,
        "y_column": y_cols if is_complex else y_col,
        "encoding": used_encoding,
    }

    # Merge with user metadata
    final_metadata = {**source_metadata}
    if metadata:
        final_metadata.update(metadata)

    # Add temperature if provided
    if temperature is not None:
        final_metadata["temperature"] = temperature  # type: ignore[assignment]

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

    # Auto-detect deformation mode from y column names if not provided
    if deformation_mode is None:
        detected_deformation = detect_deformation_mode_from_columns(
            y_headers, y_units
        )
        if detected_deformation is not None:
            deformation_mode = detected_deformation
            logger.debug(
                "Auto-detected deformation mode", deformation_mode=deformation_mode
            )

    # Store deformation mode in metadata for BaseModel.fit() auto-detection
    if deformation_mode is not None:
        final_metadata["deformation_mode"] = deformation_mode

    logger.info(
        "File parsed",
        filepath=str(filepath),
        n_records=len(x_data),
        test_mode=detected_test_mode,
        domain=domain,
        deformation_mode=deformation_mode,
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


def _get_column_header(df: pd.DataFrame, col: str | int) -> str:
    """Get column header string from DataFrame."""
    if isinstance(col, str):
        return col
    return str(df.columns[col])


def _get_column_data(df: pd.DataFrame, col: str | int) -> np.ndarray:
    """Get column data from DataFrame."""
    if isinstance(col, str):
        return df[col].values
    return df.iloc[:, col].values


def _to_float(arr: np.ndarray) -> np.ndarray:
    """Convert array to float, handling European decimal comma."""
    arr = np.array(arr)
    if arr.dtype.kind in {"U", "S", "O"}:
        # Handle European decimal comma by replacement if needed
        arr = np.char.replace(arr.astype(str), ",", ".")
    return arr.astype(float)


def _detect_delimiter(filepath: Path) -> str:
    """Auto-detect CSV delimiter using csv.Sniffer with fallbacks."""
    sample = ""
    try:
        with open(filepath, encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(
                    sample, delimiters=[",", "\t", ";", "|", " "]  # type: ignore[arg-type]
                )
                return dialect.delimiter
            except Exception:
                pass
    except FileNotFoundError:
        raise

    # Fallback heuristic - check for common delimiters
    delimiters = [",", "\t", ";", "|"]
    counts = {d: sample.count(d) for d in delimiters}
    best = max(counts, key=counts.get)  # type: ignore[arg-type]

    # If no common delimiter found, check for space-delimited
    if counts[best] == 0:
        # Check if multiple spaces separate columns
        lines = sample.strip().split("\n")
        if len(lines) > 0:
            # Check if lines have multiple whitespace-separated tokens
            tokens = lines[0].split()
            if len(tokens) > 1:
                return r"\s+"  # Regex for whitespace

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
