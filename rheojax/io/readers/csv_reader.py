"""CSV file reader for rheological data."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheojax.core._validation import reject_removed_options
from rheojax.core.data import RheoData
from rheojax.io.readers._utils import (
    VALID_TEST_MODES,
    VALID_TRANSFORMS,
    check_tensile_guard,
    construct_complex_modulus,
    detect_domain,
    detect_test_mode_from_columns,
    extract_unit_from_header,
    infer_y_unit_from_name,
    normalize_units,
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
    temperature: float | None = None,
    metadata: dict | None = None,
    intended_transform: str | None = None,
    delimiter: str | None = None,
    encoding: str | None = None,
    column_mapping: dict[str, str] | None = None,
    strain_amplitude: float | None = None,
    angular_frequency: float | None = None,
    applied_stress: float | None = None,
    shear_rate: float | None = None,
    reference_gamma_dot: float | None = None,
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
        temperature: Temperature in Kelvin for TTS workflows.
        metadata: Additional metadata dict to merge.
        intended_transform: Transform type for metadata validation. One of
            'mastercurve', 'srfs', 'owchirp', 'spp', 'fft', 'mutation', 'derivative'.
        delimiter: Column delimiter (auto-detected if None).
        encoding: File encoding (e.g. 'utf-8', 'latin-1', 'cp1252').
            Auto-detected if None. Use this to override detection for files
            with known encoding.
        column_mapping: Optional dict mapping original column names to new names.
            Applied immediately after reading, before any column lookup.
            Example: {"t": "time", "sigma": "stress"}.
        strain_amplitude: Strain amplitude (gamma_0) stored in metadata as
            ``gamma_0``. Used for LAOS/oscillation protocols.
        angular_frequency: Angular frequency (omega) stored in metadata as
            ``omega``. Used for oscillation protocols.
        applied_stress: Applied stress stored in metadata as ``sigma_applied``.
            Used for creep protocols.
        shear_rate: Shear rate stored in metadata as ``gamma_dot``.
            Used for flow/startup protocols.
        reference_gamma_dot: Reference shear rate stored in metadata as
            ``reference_gamma_dot``. Used for dimensionless flow analysis.
        header: Row number for column headers (None if no header).
        **kwargs: Additional arguments passed to pandas.read_csv. Pass
            ``thousands=","`` if numeric columns use US-style thousands
            grouping without a decimal point (e.g. "1,234" meaning 1234) —
            otherwise the locale-detection heuristic assumes EU decimal-comma
            (see :func:`_to_float`).

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

    reject_removed_options(kwargs)

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
            with open(filepath, "rb") as f:
                head_bytes = f.read(4)
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
    # Skip when column_mapping is provided — file columns differ from target names
    usecols = None
    if column_mapping is not None:
        pass  # Cannot use usecols with column_mapping (file has pre-rename names)
    elif isinstance(x_col, str):
        cols_needed = [x_col]
        if y_col is not None and isinstance(y_col, str):
            cols_needed.append(y_col)
        elif y_cols is not None:
            cols_needed.extend([c for c in y_cols if isinstance(c, str)])
        # Only set usecols if all columns are strings
        if len(cols_needed) == (1 + (1 if y_col is not None else len(y_cols or []))):
            usecols = cols_needed

    # Auto-detect comment preamble: if file starts with '#' lines and user
    # hasn't explicitly set a comment character, pass comment='#' to pandas.
    # Skip this when the requested columns already carry a literal '#'
    # prefix (e.g. auto-detected from a raw "#time,stress" header): that
    # means the '#' is part of the header itself, not a preamble marker,
    # and stripping it here would make usecols/header disagree.
    comment_char = kwargs.pop("comment", None)
    has_hash_column = any(
        isinstance(c, str) and c.startswith("#") for c in (usecols or [])
    )
    if comment_char is None and header == 0 and not has_hash_column:
        try:
            with open(filepath, encoding=default_encoding, errors="replace") as _f:
                first_line = _f.readline()
            if first_line.startswith("#"):
                comment_char = "#"
                logger.debug("Auto-detected '#' comment preamble")
        except (OSError, UnicodeDecodeError):
            logger.debug("Could not peek at file for comment detection")

    # Read CSV file with tolerant encoding/dialect handling.
    # "replace" kwarg is kept as the tolerant fallback; "strict" is tried first
    # so that silent corruption is caught and logged before falling back.
    read_kwargs = dict(
        sep=delimiter,
        header=header,
        encoding=default_encoding,
        encoding_errors="replace",
        engine="python",
        usecols=usecols,
        comment=comment_char,
        **kwargs,
    )
    tried_utf16 = False

    used_encoding = default_encoding

    with log_io(logger, "read", filepath=str(filepath)) as io_ctx:
        try:
            # Try strict encoding first to detect corruption early
            logger.debug(
                "Reading CSV file (strict encoding)", encoding=default_encoding
            )
            df = pd.read_csv(filepath, **{**read_kwargs, "encoding_errors": "strict"})
        except UnicodeDecodeError:
            # Strict failed — fall back to replacement characters with a warning
            logger.warning(
                "Encoding errors in CSV file — using replacement characters",
                filepath=str(filepath),
                encoding=default_encoding,
            )
            try:
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

        # VIS-CSV-001: Check for encoding replacement artifacts without
        # materialising .astype(str) twice per column. Cache col_str and reuse
        # it for both the detection pass and the numeric-corruption check.
        affected_cols: list[str] = []
        for col in df.columns:
            col_str = df[col].astype(str)  # single materialisation per column
            if col_str.str.contains("\ufffd", na=False).any():
                affected_cols.append(col)
                # Reuse col_str — no second astype(str) needed
                sample = col_str.str.replace("\ufffd", "", regex=False)
                if sample.str.match(r"^[\d.eE+\-,]*\d[\d.eE+\-,]*$", na=False).any():
                    raise ValueError(
                        f"Encoding corruption detected in numeric column '{col}'. "
                        f"The file may need to be re-exported with UTF-8 encoding. "
                        f"Affected file: {filepath}"
                    )
        if affected_cols:
            logger.warning(
                "Encoding replacement characters (\\ufffd) detected in CSV file — "
                "some values may be corrupted",
                filepath=str(filepath),
                affected_columns=affected_cols,
            )

        io_ctx["rows"] = len(df)
        io_ctx["columns"] = len(df.columns)
        io_ctx["encoding"] = used_encoding
        logger.debug(
            "CSV file read successfully",
            n_rows=len(df),
            n_cols=len(df.columns),
            encoding=used_encoding,
        )

    # Guard: check for tensile/E* columns/units/mode. Must run on the
    # original file headers (before column_mapping renaming) — otherwise
    # renaming a tensile column (e.g. "E'" -> "modulus") would silently
    # bypass the safety check.
    check_tensile_guard(df.columns, units=y_units)

    # Apply column renaming if provided
    if column_mapping is not None:
        df = df.rename(columns=column_mapping)
        logger.debug("Applied column_mapping", mapping=column_mapping)

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
    y_data: np.ndarray
    if is_complex:
        if y_cols is None:  # pragma: no cover — guarded by is_complex
            raise ValueError("y_cols must not be None for complex data")
        y_headers = [_get_column_header(df, col) for col in y_cols]
        try:
            g_prime_data = _get_column_data(df, y_cols[0])
            g_double_prime_data = _get_column_data(df, y_cols[1])
        except (KeyError, IndexError) as e:
            logger.error("Y column not found", y_cols=y_cols, exc_info=True)
            raise KeyError(f"Y column not found: {e}") from e
        # Run both columns through the same locale-aware decimal detection as
        # the single y_col path (_to_float) before casting to complex, so
        # European decimal-comma files (e.g. "1000,5") don't crash here.
        g_prime_data = _to_float(g_prime_data)
        g_double_prime_data = _to_float(g_double_prime_data)
        y_data = construct_complex_modulus(g_prime_data, g_double_prime_data)
        logger.debug("Constructed complex modulus from G' and G''")
    else:
        if y_col is None:  # pragma: no cover — guarded by is_complex
            raise ValueError("y_col must not be None for real data")
        y_headers = [_get_column_header(df, y_col)]
        try:
            y_data = _get_column_data(df, y_col)
        except (KeyError, IndexError) as e:
            logger.error("Y column not found", y_col=y_col, exc_info=True)
            raise KeyError(f"Y column not found: {e}") from e

    # Convert to numpy arrays and handle NaN
    x_data = _to_float(x_data)
    if not is_complex:
        y_data = _to_float(y_data)

    # Remove non-finite values (NaN and ±inf) in single pass.
    # np.isfinite covers both, preventing RheoData's isfinite check from
    # raising a confusing ValueError on instrument artefacts.
    if is_complex:
        valid_idx = np.flatnonzero(
            np.isfinite(x_data) & np.isfinite(y_data.real) & np.isfinite(y_data.imag)
        )
    else:
        valid_idx = np.flatnonzero(np.isfinite(x_data) & np.isfinite(y_data))
    n_dropped = len(x_data) - len(valid_idx)
    if n_dropped > 0:
        logger.warning(
            "Dropped non-finite (NaN/Inf) rows during loading",
            n_dropped=n_dropped,
            n_total=len(x_data),
        )
    x_data = np.take(x_data, valid_idx)
    if y_data.ndim > 1:
        y_data = y_data[valid_idx]
    else:
        y_data = np.take(y_data, valid_idx)

    if len(x_data) == 0:
        logger.error(
            "No valid data points after removing NaN values", filepath=str(filepath)
        )
        raise ValueError("No valid data points after removing NaN values")

    logger.debug("Data points after NaN removal", n_points=len(x_data))

    # Auto-extract units from headers if not provided, normalizing the
    # extracted unit (and the numeric data) to SI via UNIFIED_UNIT_CONVERSIONS —
    # matching the anton_paar.py/trios readers' behavior. Units passed
    # explicitly by the caller are trusted as-is and left unconverted.
    if x_units is None:
        _, extracted_x_units = extract_unit_from_header(x_header)
        if extracted_x_units is not None:
            x_data, x_units = normalize_units(x_data, extracted_x_units)
        else:
            x_units = extracted_x_units
    if y_units is None:
        # Use first y column header for units
        _, extracted_y_units = extract_unit_from_header(y_headers[0])
        if extracted_y_units is None:
            # No bracketed unit (e.g. header is just "Viscosity" or "Stress")
            # -- infer from the name itself so a flow-curve viscosity column
            # never gets silently treated as stress (or vice versa) downstream.
            extracted_y_units = infer_y_unit_from_name(y_headers[0])
        if extracted_y_units is not None:
            if is_complex:
                real_part, y_units = normalize_units(y_data.real, extracted_y_units)
                imag_part, _ = normalize_units(y_data.imag, extracted_y_units)
                y_data = real_part + 1j * imag_part
            else:
                y_data, y_units = normalize_units(
                    np.asarray(y_data, dtype=np.float64), extracted_y_units
                )
        else:
            y_units = extracted_y_units

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
        "source_file": filepath.name,
        "file_type": "csv" if filepath.suffix.lower() in {".csv", ""} else "txt",
        "x_column": x_col,
        "y_column": y_cols if is_complex else y_col,
        "encoding": used_encoding,
    }
    if used_encoding != default_encoding:
        source_metadata["encoding_fallback"] = True

    # Merge with user metadata
    final_metadata: dict[str, Any] = {**source_metadata}
    if metadata:
        reject_removed_options(metadata)
        final_metadata.update(metadata)

    # Add temperature if provided
    if temperature is not None:
        final_metadata["temperature"] = temperature

    # Store protocol metadata
    if strain_amplitude is not None:
        final_metadata["gamma_0"] = strain_amplitude
    if angular_frequency is not None:
        final_metadata["omega"] = angular_frequency
    if applied_stress is not None:
        final_metadata["sigma_applied"] = applied_stress
    if shear_rate is not None:
        final_metadata["gamma_dot"] = shear_rate
    if reference_gamma_dot is not None:
        final_metadata["reference_gamma_dot"] = reference_gamma_dot

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


def _looks_like_eu_thousands(val: str) -> bool:
    """Whether a dot-only numeric string has the shape of EU thousands
    grouping (e.g. "1.234", "12.345.678") rather than a plain decimal.

    Real EU thousands grouping always groups digits in 3s: every group after
    the first has exactly 3 digits, and the leading group has 1-3. A value
    that doesn't fit that shape (e.g. "5.5", or sci-notation "5.5e3") is a
    plain decimal, not a thousands-grouped integer.
    """
    groups = val.split(".")
    if len(groups) < 2 or not all(g.isdigit() for g in groups):
        return False
    return len(groups[0]) in (1, 2, 3) and all(len(g) == 3 for g in groups[1:])


def _to_float(arr: np.ndarray) -> np.ndarray:
    """Convert array to float, handling European decimal comma and US thousands.

    Samples up to 20 non-empty values with separators to determine locale format,
    avoiding misdetection when the first value is a plain integer:
    - "1,234.56" (US thousands): remove commas
    - "1.234,56" (EU thousands+decimal): remove dots, comma→dot
    - "1,56" (EU decimal only): comma→dot
    - "1.56" (standard): no change

    Note:
        A comma-only value with no decimal point (e.g. "1,234") is ambiguous
        between EU decimal-comma (1.234) and US thousands-grouping (1234) —
        there is no way to distinguish the two from the string alone, so this
        heuristic always assumes EU decimal-comma. Callers whose data uses
        thousands-grouping without a decimal point should pass
        ``thousands=","`` to :func:`load_csv` (forwarded to ``pandas.read_csv``
        via ``**kwargs``); pandas will then parse the column as numeric before
        it ever reaches this function, bypassing the heuristic entirely.
    """
    arr = np.array(arr)
    if arr.dtype.kind in {"U", "S", "O"}:
        str_arr = arr.astype(str)
        # Sample up to 20 non-empty values with a separator for locale detection
        samples = []
        for s in str_arr.flat:
            s_stripped = s.strip()
            if s_stripped and ("," in s_stripped or "." in s_stripped):
                samples.append(s_stripped)
                if len(samples) >= 20:
                    break

        # Determine format from samples
        has_both = any("," in s and "." in s for s in samples)
        has_comma_only = any("," in s and "." not in s for s in samples)

        if has_both:
            # Pick format from first sample with both separators
            sample = next(s for s in samples if "," in s and "." in s)
            last_comma = sample.rfind(",")
            last_dot = sample.rfind(".")
            if last_comma > last_dot:
                # EU: 1.234,56 — dot=thousands, comma=decimal. Convert
                # element-wise since not every value in the column need be
                # EU-formatted:
                # - a value with a comma is unambiguously EU (strip dots,
                #   comma -> decimal point)
                # - a dot-only value is ambiguous ("1.234" could be EU
                #   thousands-grouped 1234, or a plain decimal 1.234) —
                #   resolve it the same way genuine EU thousands grouping
                #   would look: every dot-separated group must be all-digit,
                #   the leading group 1-3 digits, and every following group
                #   exactly 3 digits (e.g. "1.234", "12.345.678"). A value
                #   like "5.5" or sci-notation like "5.5e3" fails that shape
                #   (a 1-digit or non-digit trailing group) and is left as a
                #   plain decimal — blanket-stripping its dot would silently
                #   corrupt it (e.g. "5.5" becoming 55.0).
                result = np.empty(str_arr.shape, dtype=float)
                for idx in np.ndindex(str_arr.shape):
                    val = str(str_arr[idx]).strip()
                    if "," in val:
                        val = val.replace(".", "").replace(",", ".")
                    elif _looks_like_eu_thousands(val):
                        val = val.replace(".", "")
                    try:
                        result[idx] = float(val)
                    except ValueError:
                        result[idx] = np.nan
                nan_ratio = np.isnan(result).sum() / max(len(result), 1)
                if nan_ratio > 0.5:
                    logger.warning(
                        "More than 50% of values could not be converted to float — "
                        "decimal separator detection may be incorrect. "
                        "Consider specifying the decimal separator explicitly.",
                        nan_ratio=f"{nan_ratio:.1%}",
                        n_total=len(result),
                    )
                return result
            else:
                # US: 1,234.56 — comma=thousands, dot=decimal
                str_arr = np.char.replace(str_arr, ",", "")
        elif has_comma_only:
            # EU decimal only: 1,56 → 1.56
            str_arr = np.char.replace(str_arr, ",", ".")
        arr = str_arr
    try:
        result = arr.astype(float)
    except (ValueError, TypeError):
        result = pd.to_numeric(pd.Series(arr.ravel()), errors="coerce").values.astype(
            float
        )
    nan_ratio = np.isnan(result).sum() / max(len(result), 1)
    if nan_ratio > 0.5:
        logger.warning(
            "More than 50% of values could not be converted to float — "
            "decimal separator detection may be incorrect. "
            "Consider specifying the decimal separator explicitly.",
            nan_ratio=f"{nan_ratio:.1%}",
            n_total=len(result),
        )
    return result


def _detect_delimiter(filepath: Path) -> str:
    """Auto-detect CSV delimiter using csv.Sniffer with fallbacks."""
    sample = ""
    try:
        with open(filepath, encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(8192)
            try:
                dialect = csv.Sniffer().sniff(
                    sample,
                    delimiters=[",", "\t", ";", "|", " "],  # type: ignore[arg-type]
                )
                return dialect.delimiter
            except csv.Error:
                pass
    except FileNotFoundError:
        raise

    # Fallback heuristic - check for common delimiters
    delimiters = [",", "\t", ";", "|"]
    counts = {d: sample.count(d) for d in delimiters}
    best = max(counts, key=lambda d: counts[d])

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
