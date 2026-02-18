"""Auto-detection wrapper for file readers."""

from __future__ import annotations

import warnings
from pathlib import Path

from rheojax.core.data import RheoData
from rheojax.io.readers.anton_paar import load_anton_paar
from rheojax.io.readers.csv_reader import detect_csv_delimiter, load_csv
from rheojax.io.readers.excel_reader import load_excel
from rheojax.io.readers.trios import load_trios
from rheojax.logging import get_logger, log_io

logger = get_logger(__name__)

# Exceptions that indicate a real system-level failure, NOT a format mismatch.
# These must never be caught by the reader cascade.
_FATAL_EXCEPTIONS = (
    KeyboardInterrupt,
    SystemExit,
    MemoryError,
    PermissionError,
    OSError,
)

# File size threshold for warning (100 MB)
_FILE_SIZE_WARNING_BYTES = 100 * 1024 * 1024


def auto_load(filepath: str | Path, **kwargs) -> RheoData | list[RheoData]:
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
        logger.error("File not found", filepath=str(filepath))
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {filepath}")

    # Warn about large files that may consume significant memory
    file_size = filepath.stat().st_size
    if file_size > _FILE_SIZE_WARNING_BYTES:
        size_mb = file_size / (1024 * 1024)
        logger.warning(
            "Large file detected — loading may consume significant memory",
            filepath=str(filepath),
            size_mb=f"{size_mb:.1f}",
        )
        warnings.warn(
            f"File is {size_mb:.0f} MB. Loading may consume significant memory. "
            f"Consider using chunked reading if available.",
            ResourceWarning,
            stacklevel=2,
        )

    extension = filepath.suffix.lower()

    with log_io(logger, "read", filepath=str(filepath)) as io_ctx:
        io_ctx["extension"] = extension
        logger.debug("Detecting format from extension", extension=extension)

        # Try based on file extension first
        if extension == ".txt":
            result = _try_trios_then_anton_then_csv(filepath, **kwargs)
        elif extension == ".csv":
            result = _try_csv(filepath, **kwargs)
        elif extension in [".xlsx", ".xls"]:
            result = _try_excel(filepath, **kwargs)
        elif extension == ".tsv":
            kwargs["delimiter"] = "\t"
            result = _try_csv(filepath, **kwargs)
        else:
            # Unknown extension - try readers in sequence (CSV then Excel)
            logger.debug("Unknown extension, trying all readers")
            result = _try_all_readers(filepath, **kwargs)

        # Add record count to context
        if isinstance(result, list):
            io_ctx["records"] = sum(len(r.x) for r in result)
            io_ctx["segments"] = len(result)
        else:
            io_ctx["records"] = len(result.x)

        return result


def _try_trios_then_anton_then_csv(
    filepath: Path, **kwargs
) -> RheoData | list[RheoData]:
    """Try TRIOS first, then Anton Paar, then CSV.

    Args:
        filepath: File path
        **kwargs: Additional arguments

    Returns:
        RheoData object(s)
    """
    # Try TRIOS first
    try:
        logger.debug("Trying TRIOS reader", filepath=str(filepath))
        result = load_trios(filepath, **kwargs)
        logger.debug("TRIOS reader succeeded", filepath=str(filepath))
        return result
    except _FATAL_EXCEPTIONS:
        raise
    except Exception as e:
        logger.debug("TRIOS reader failed", filepath=str(filepath), error=str(e))
        warnings.warn(
            f"TRIOS reader failed: {e}. Trying Anton Paar reader.", stacklevel=2
        )

    try:
        logger.debug("Trying Anton Paar reader", filepath=str(filepath))
        result = load_anton_paar(filepath, **kwargs)
        logger.debug("Anton Paar reader succeeded", filepath=str(filepath))
        return result
    except _FATAL_EXCEPTIONS:
        raise
    except Exception as e:
        logger.debug("Anton Paar reader failed", filepath=str(filepath), error=str(e))
        warnings.warn(
            f"Anton Paar reader failed: {e}. Trying CSV reader.", stacklevel=2
        )

    # Try CSV as fallback
    try:
        logger.debug("Trying CSV reader", filepath=str(filepath))
        result = _try_csv(filepath, **kwargs)
        logger.debug("CSV reader succeeded", filepath=str(filepath))
        return result
    except _FATAL_EXCEPTIONS:
        raise
    except Exception as e:
        logger.error(
            "Could not parse file with any reader",
            filepath=str(filepath),
            exc_info=True,
        )
        raise ValueError(
            f"Could not parse file as TRIOS, Anton Paar, or CSV: {e}"
        ) from e


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
            logger.debug("Auto-detecting columns for CSV", filepath=str(filepath))
            delimiter = detect_csv_delimiter(filepath)
            df = pd.read_csv(filepath, sep=delimiter)
            columns_lower = [c.lower() for c in df.columns]
            logger.debug(
                "CSV columns detected",
                filepath=str(filepath),
                columns=list(df.columns),
            )

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

            # Try to find complex modulus pair (E'/E'' or G'/G'')
            y_cols_pair = _detect_modulus_pair(df.columns, columns_lower)
            if y_cols_pair is not None:
                kwargs["y_cols"] = y_cols_pair
                kwargs.pop("y_col", None)
                y_col = "FOUND_PAIR"
            else:
                y_col = None

            # Try to find stress/modulus column (single y)
            if y_cols_pair is None:
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
                logger.error(
                    "Could not auto-detect x and y columns",
                    filepath=str(filepath),
                    available_columns=list(df.columns),
                )
                raise ValueError(
                    "Could not auto-detect x and y columns. "
                    "Please specify x_col and y_col."
                )

            kwargs["x_col"] = x_col
            if y_col != "FOUND_PAIR":
                kwargs["y_col"] = y_col
            logger.debug(
                "Auto-detected columns",
                filepath=str(filepath),
                x_col=x_col,
                y_col=y_col if y_col != "FOUND_PAIR" else kwargs.get("y_cols"),
            )

        except Exception as e:
            logger.error(
                "Could not auto-detect columns",
                filepath=str(filepath),
                exc_info=True,
            )
            raise ValueError(
                f"Could not auto-detect columns: {e}. Please specify x_col and y_col."
            ) from e

    return load_csv(filepath, **kwargs)


def _try_excel(filepath: Path, **kwargs) -> RheoData:
    """Try Excel reader with auto-detection.

    Args:
        filepath: File path
        **kwargs: Additional arguments

    Returns:
        RheoData object
    """
    logger.debug("Trying Excel reader", filepath=str(filepath))

    # Check if x_col and y_col are specified
    if "x_col" not in kwargs or "y_col" not in kwargs:
        logger.error("x_col and y_col required for Excel files", filepath=str(filepath))
        raise ValueError("For Excel files, please specify x_col and y_col parameters")

    return load_excel(filepath, **kwargs)


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
        ("TRIOS", lambda: load_trios(filepath, **kwargs)),
        ("ANTON_PAAR", lambda: load_anton_paar(filepath, **kwargs)),
        ("CSV", lambda: _try_csv(filepath, **kwargs)),
        ("EXCEL", lambda: _try_excel(filepath, **kwargs)),
    ]

    errors = []
    for reader_name, reader_func in readers:
        try:
            logger.debug("Trying reader", filepath=str(filepath), reader=reader_name)
            result = reader_func()
            logger.debug(
                "Reader succeeded", filepath=str(filepath), reader=reader_name.lower()
            )
            return result
        except _FATAL_EXCEPTIONS:
            raise
        except Exception as e:
            logger.debug(
                "Reader failed",
                filepath=str(filepath),
                reader=reader_name,
                error=str(e),
            )
            errors.append(f"{reader_name}: {e}")

    # All readers failed — chain the last error for traceback context
    error_msg = "Could not parse file with any available reader:\n" + "\n".join(errors)
    logger.error(
        "All readers failed",
        filepath=str(filepath),
        tried_readers=[r[0] for r in readers],
    )
    raise ValueError(error_msg)


def _detect_modulus_pair(
    columns: list[str], columns_lower: list[str]
) -> list[str] | None:
    """Detect E'/E'' or G'/G'' column pairs for complex modulus construction.

    Searches for common DMTA (E'/E'') and shear (G'/G'') column patterns.

    Args:
        columns: Original column names
        columns_lower: Lowercased column names

    Returns:
        List of [storage, loss] column names, or None if no pair found
    """
    import re

    # Patterns for storage/loss modulus pairs: (storage_pattern, loss_pattern)
    pair_patterns = [
        # E'/E'' (DMTA)
        (
            re.compile(r"^e['\u2032]", re.IGNORECASE),
            re.compile(r'^e[""\u2033]', re.IGNORECASE),
        ),
        # E_stor/E_loss (pyvisco style)
        (
            re.compile(r"^e[-_]?stor", re.IGNORECASE),
            re.compile(r"^e[-_]?loss", re.IGNORECASE),
        ),
        # G'/G'' (shear)
        (
            re.compile(r"^g['\u2032]", re.IGNORECASE),
            re.compile(r'^g[""\u2033]', re.IGNORECASE),
        ),
        # G_stor/G_loss
        (
            re.compile(r"^g[-_]?stor", re.IGNORECASE),
            re.compile(r"^g[-_]?loss", re.IGNORECASE),
        ),
        # Storage Modulus / Loss Modulus (generic)
        (
            re.compile(r"storage\s+modulus", re.IGNORECASE),
            re.compile(r"loss\s+modulus", re.IGNORECASE),
        ),
    ]

    for stor_pat, loss_pat in pair_patterns:
        stor_col = None
        loss_col = None
        for _i, col in enumerate(columns):
            if stor_pat.search(col):
                stor_col = col
            if loss_pat.search(col):
                loss_col = col
        if stor_col is not None and loss_col is not None:
            logger.debug(
                "Detected modulus pair",
                storage=stor_col,
                loss=loss_col,
            )
            return [stor_col, loss_col]

    return None
