"""RheoCompass CSV parser for Anton Paar rheometer exports.

This module provides a complete parser for RheoCompass CSV exports with:
- Interval-based data block parsing
- Automatic encoding detection (UTF-16, UTF-8, Latin-1)
- Test type auto-detection (creep, relaxation, oscillation, rotation)
- Metadata extraction (geometry, gap, temperature)
- Unit normalization to SI
- Derived quantity computation (J(t), G(t), G*)

The parser handles RheoCompass-specific format features including tab-separated
values, "Interval and data points:" markers, and locale-aware decimal separators.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData
from rheojax.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Structures (T007)
# =============================================================================


@dataclass
class IntervalBlock:
    """Container for a single interval's data and metadata.

    Attributes:
        interval_index: 1-based interval number from file
        n_points: Number of data points (if specified in header)
        units: Column name to unit string mapping
        df: Parsed data as DataFrame
    """

    interval_index: int
    n_points: int | None
    units: dict[str, str]
    df: pd.DataFrame


# =============================================================================
# Column Mappings (T008)
# =============================================================================

# Maps RheoCompass column name patterns to canonical RheoJAX names
# Each entry: canonical_name -> (regex patterns, SI unit, applicable test types)
COLUMN_MAPPINGS: dict[str, tuple[list[str], str, list[str]]] = {
    "time": (
        [r"^time$", r"^t$", r"^zeit$"],
        "s",
        ["creep", "relaxation", "oscillation", "rotation"],
    ),
    "angular_frequency": (
        [r"^angular[\s_]?frequency$", r"^frequency$", r"^omega$", r"^ω$"],
        "rad/s",
        ["oscillation"],
    ),
    "shear_stress": (
        [r"^shear[\s_]?stress$", r"^stress$", r"^τ$", r"^tau$"],
        "Pa",
        ["creep", "relaxation", "rotation"],
    ),
    "shear_strain": (
        [r"^shear[\s_]?strain$", r"^strain$", r"^γ$", r"^gamma$"],
        "dimensionless",
        ["creep", "relaxation"],
    ),
    "shear_rate": (
        [r"^shear[\s_]?rate$", r"^γ̇$", r"^gamma[\s_]?dot$"],
        "1/s",
        ["rotation"],
    ),
    "compliance": (
        [r"^compliance$", r"^j\(?t\)?$"],
        "1/Pa",
        ["creep"],
    ),
    "relaxation_modulus": (
        [r"^relaxation[\s_]?modulus$", r"^g\(?t\)?$"],
        "Pa",
        ["relaxation"],
    ),
    "storage_modulus": (
        [r"^storage[\s_]?modulus$", r"^g'$", r"^g_prime$"],
        "Pa",
        ["oscillation"],
    ),
    "loss_modulus": (
        [r"^loss[\s_]?modulus$", r"^g''$", r'^g"$', r"^g_double_prime$"],
        "Pa",
        ["oscillation"],
    ),
    "complex_modulus": (
        [r"^complex[\s_]?modulus$", r"^g\*$", r"^\|g\*\|$"],
        "Pa",
        ["oscillation"],
    ),
    "tensile_storage_modulus": (
        [
            r"^e'$", r"^e_prime$", r"^e_stor$",
            r"^tensile[\s_]?storage[\s_]?modulus$",
            r"^young'?s?[\s_]?storage[\s_]?modulus$",
        ],
        "Pa",
        ["oscillation"],
    ),
    "tensile_loss_modulus": (
        [
            r"^e''$", r'^e"$', r"^e_double_prime$", r"^e_loss$",
            r"^tensile[\s_]?loss[\s_]?modulus$",
            r"^young'?s?[\s_]?loss[\s_]?modulus$",
        ],
        "Pa",
        ["oscillation"],
    ),
    "viscosity": (
        [r"^viscosity$", r"^η$", r"^eta$"],
        "Pa.s",
        ["rotation"],
    ),
    "complex_viscosity": (
        [r"^complex[\s_]?viscosity$", r"^η\*$", r"^eta\*$"],
        "Pa.s",
        ["oscillation"],
    ),
    "phase_angle": (
        [r"^phase[\s_]?angle$", r"^δ$", r"^delta$"],
        "deg",
        ["oscillation"],
    ),
    "temperature": (
        [r"^temperature$", r"^temp$"],
        "°C",
        ["creep", "relaxation", "oscillation", "rotation"],
    ),
    "normal_force": (
        [r"^normal[\s_]?force$"],
        "N",
        ["creep", "relaxation", "oscillation", "rotation"],
    ),
    "torque": (
        [r"^torque$"],
        "N.m",
        ["rotation"],
    ),
    "strain_amplitude": (
        [r"^strain[\s_]?amplitude$"],
        "dimensionless",
        ["oscillation"],
    ),
    "stress_amplitude": (
        [r"^stress[\s_]?amplitude$"],
        "Pa",
        ["oscillation"],
    ),
}

# Pre-compiled patterns for column mapping (performance optimization)
_COLUMN_PATTERNS_COMPILED: dict[str, list[re.Pattern]] = {
    canonical: [re.compile(p, re.IGNORECASE) for p in patterns]
    for canonical, (patterns, _, _) in COLUMN_MAPPINGS.items()
}

# Pre-compiled pattern for unit extraction
_UNIT_EXTRACTION_PATTERN = re.compile(r"^(.*?)[\[(](.*?)[\])]")


# =============================================================================
# Unit Conversions (T009)
# =============================================================================

# Maps source units to (target_unit, conversion_factor)
UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    "hz": ("rad/s", 2 * math.pi),
    "1/hz": ("rad/s", 2 * math.pi),
    "ms": ("s", 0.001),
    "min": ("s", 60.0),
    "mins": ("s", 60.0),
    "minutes": ("s", 60.0),
    "kpa": ("Pa", 1000.0),
    "mpa": ("Pa", 1e6),
    "mpa·s": ("Pa.s", 0.001),
    "mpa.s": ("Pa.s", 0.001),
    "%": ("dimensionless", 0.01),
}


# =============================================================================
# Encoding Detection (T010)
# =============================================================================


def _detect_encoding(filepath: Path) -> str:
    """Detect file encoding using cascade approach.

    RheoCompass exports are typically UTF-16 with BOM. Falls back through
    common encodings.

    Args:
        filepath: Path to file

    Returns:
        Detected encoding string

    Raises:
        UnicodeDecodeError: If no encoding works
    """
    encodings = ["utf-16", "utf-8-sig", "utf-8", "latin-1"]

    for encoding in encodings:
        try:
            with open(filepath, encoding=encoding) as f:
                # Read a sample to verify encoding works
                f.read(4096)
            logger.debug("Detected encoding", encoding=encoding)
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Last resort: latin-1 with error replacement
    logger.warning("Could not detect encoding, using latin-1 with error replacement")
    return "latin-1"


@lru_cache(maxsize=128)
def _detect_encoding_cached(filepath_str: str) -> str:
    """Cached encoding detection by file path string.

    This wrapper enables caching for repeated file access during batch operations.

    Args:
        filepath_str: File path as string (for hashability)

    Returns:
        Detected encoding string
    """
    return _detect_encoding(Path(filepath_str))


# =============================================================================
# Decimal Separator Detection (T015)
# =============================================================================


def _detect_decimal_separator(text_sample: str) -> str:
    """Detect decimal separator from text sample.

    European locales may use comma as decimal separator and period as
    thousands separator.

    Args:
        text_sample: Sample text containing numeric values

    Returns:
        Detected decimal separator ('.' or ',')
    """
    # Count patterns like "digit.digit" and "digit,digit"
    dot_pattern = re.findall(r"\d\.\d", text_sample)
    comma_pattern = re.findall(r"\d,\d", text_sample)

    if len(comma_pattern) > len(dot_pattern) * 2:
        return ","
    return "."


def _normalize_numeric_text(text: str, decimal_sep: str) -> str:
    """Normalize numeric text to use dot as decimal separator.

    Args:
        text: Text with numeric values
        decimal_sep: Current decimal separator

    Returns:
        Text with normalized decimal separators
    """
    if decimal_sep == ",":
        # Remove thousands separator (period), then convert comma to period
        text = text.replace(".", "")
        text = text.replace(",", ".")
    else:
        # Remove thousands separator (comma) — repeat to handle "1,234,567"
        while True:
            new_text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)
            if new_text == text:
                break
            text = new_text
    return text


# =============================================================================
# Global Metadata Extraction (T012)
# =============================================================================


def _extract_global_metadata(lines: list[str]) -> dict[str, Any]:
    """Extract key:value metadata pairs before first interval marker.

    Args:
        lines: All lines from file

    Returns:
        Dictionary of metadata key-value pairs
    """
    metadata: dict[str, Any] = {}

    for line in lines:
        # Stop at first interval marker
        if line.strip().startswith("Interval and data points:"):
            break

        # Parse key:\tvalue or key:\tvalue format
        if "\t" in line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                key = parts[0].strip().rstrip(":")
                value = parts[1].strip()
                if key and value:
                    metadata[key] = value
        elif ":" in line and not line.strip().startswith("Interval"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    metadata[key] = value

    return metadata


# =============================================================================
# Interval Boundary Detection (T013)
# =============================================================================


def _find_interval_boundaries(lines: list[str]) -> list[tuple[int, int, int | None]]:
    """Find all interval markers and their boundaries.

    Args:
        lines: All lines from file

    Returns:
        List of (start_line_idx, interval_index, n_points) tuples
    """
    boundaries = []
    marker_pattern = re.compile(r"Interval and data points:\s*(\d+)(?:\s+(\d+))?")

    for i, line in enumerate(lines):
        match = marker_pattern.search(line)
        if match:
            interval_idx = int(match.group(1))
            n_points = int(match.group(2)) if match.group(2) else None
            boundaries.append((i, interval_idx, n_points))

    return boundaries


# =============================================================================
# Single Interval Parsing (T014)
# =============================================================================


def _extract_unit(column_name: str) -> tuple[str, str | None]:
    """Extract base name and unit from column header.

    Handles both bracket [unit] and parentheses (unit) notation.

    Args:
        column_name: Column header like "Time [s]" or "Stress (Pa)"

    Returns:
        Tuple of (base_name, unit) where unit may be None
    """
    # Match [unit] or (unit) using pre-compiled pattern
    match = _UNIT_EXTRACTION_PATTERN.search(column_name)
    if match:
        base = match.group(1).strip()
        unit = match.group(2).strip()
        return base, unit
    return column_name.strip(), None


def _parse_single_interval(
    lines: list[str], start_idx: int, end_idx: int | None, decimal_sep: str
) -> IntervalBlock:
    """Parse a single interval block into an IntervalBlock.

    Args:
        lines: All lines from file
        start_idx: Start line index (at interval marker)
        end_idx: End line index (exclusive), None for end of file
        decimal_sep: Decimal separator to use

    Returns:
        Parsed IntervalBlock
    """
    interval_lines = lines[start_idx : end_idx if end_idx else len(lines)]

    # Parse interval header
    header_match = re.search(
        r"Interval and data points:\s*(\d+)(?:\s+(\d+))?", interval_lines[0]
    )
    interval_idx = int(header_match.group(1)) if header_match else 1
    n_points = (
        int(header_match.group(2)) if header_match and header_match.group(2) else None
    )

    logger.debug("Parsing interval", interval_index=interval_idx, n_points=n_points)

    # Find "Interval data:" line with column headers
    data_start_idx = None
    column_headers = []
    units_dict: dict[str, str] = {}

    for i, line in enumerate(interval_lines[1:], 1):
        if line.strip().startswith("Interval data:"):
            # Column headers follow "Interval data:" prefix
            parts = line.split("\t")
            # Skip "Interval data:" prefix
            column_headers = [p.strip() for p in parts[1:] if p.strip()]
            data_start_idx = i + 1
            break

    if data_start_idx is None or not column_headers:
        raise ValueError(
            f"Could not find 'Interval data:' header in interval {interval_idx}"
        )

    # Check for units line (starts with tab and contains [unit])
    if data_start_idx < len(interval_lines):
        potential_units_line = interval_lines[data_start_idx]
        if potential_units_line.strip().startswith("[") or (
            "\t[" in potential_units_line
            and not potential_units_line.strip()[0].isdigit()
        ):
            # Parse units - skip empty first part if line starts with tab
            unit_parts = potential_units_line.split("\t")
            # Filter out empty parts and align with columns
            unit_parts = [p.strip() for p in unit_parts if p.strip()]
            for col, unit_str in zip(column_headers, unit_parts, strict=False):
                if unit_str.startswith("[") and unit_str.endswith("]"):
                    units_dict[col] = unit_str[1:-1]
                elif unit_str.startswith("(") and unit_str.endswith(")"):
                    units_dict[col] = unit_str[1:-1]
                elif unit_str:
                    units_dict[col] = unit_str
            data_start_idx += 1

    # Extract column units from headers if not in separate line
    for col in column_headers:
        base_name, unit = _extract_unit(col)
        if unit and col not in units_dict:
            units_dict[col] = unit

    # Collect data rows
    data_rows = []
    for line in interval_lines[data_start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Stop at next interval marker or metadata-like lines
        if stripped.startswith("Interval and data points:"):
            break

        # Parse numeric values
        parts = line.split("\t")
        row_values = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Normalize decimal separator
            if decimal_sep == ",":
                p = p.replace(".", "").replace(",", ".")
            try:
                row_values.append(float(p))
            except ValueError:
                # Non-numeric value - could be end of data
                break

        if row_values and len(row_values) == len(column_headers):
            data_rows.append(row_values)
        elif row_values and len(row_values) > 0:
            # Partial row - pad with NaN
            while len(row_values) < len(column_headers):
                row_values.append(float("nan"))
            data_rows.append(row_values)

    if not data_rows:
        raise ValueError(f"No valid data rows found in interval {interval_idx}")

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=column_headers)

    logger.debug(
        "Interval parsed",
        interval_index=interval_idx,
        n_rows=len(df),
        n_cols=len(df.columns),
    )

    return IntervalBlock(
        interval_index=interval_idx,
        n_points=n_points,
        units=units_dict,
        df=df,
    )


# =============================================================================
# Main Interval Parser (T016)
# =============================================================================


def parse_rheocompass_intervals(
    filepath: str | Path,
    *,
    encoding: str | None = None,
    marker: str = "Interval and data points:",
) -> tuple[dict[str, Any], list[IntervalBlock]]:
    """Parse RheoCompass file returning raw interval blocks.

    Low-level parser for advanced users who need full access to all
    columns and metadata without RheoData mapping.

    Args:
        filepath: Path to RheoCompass CSV export file
        encoding: File encoding override (auto-detected if None)
        marker: Interval start marker string

    Returns:
        Tuple of (global_metadata, interval_blocks)

    Raises:
        FileNotFoundError: File does not exist
        ValueError: No interval blocks found
        UnicodeDecodeError: Encoding detection failed
    """
    filepath = Path(filepath)
    logger.info("Opening file", filepath=str(filepath))

    if not filepath.exists():
        logger.error("File not found", filepath=str(filepath))
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect encoding (using cached version for repeated file access)
    if encoding is None:
        encoding = _detect_encoding_cached(str(filepath))

    # Read entire file
    with open(filepath, encoding=encoding, errors="replace") as f:
        content = f.read()

    lines = content.splitlines()

    # Detect decimal separator from content sample
    decimal_sep = _detect_decimal_separator(content[:4096])
    logger.debug("Detected decimal separator", decimal_sep=decimal_sep)

    # Extract global metadata
    global_metadata = _extract_global_metadata(lines)

    # Find interval boundaries
    boundaries = _find_interval_boundaries(lines)
    if not boundaries:
        logger.error("No interval blocks found", filepath=str(filepath))
        raise ValueError(
            f"No interval blocks found in file. "
            f"Expected '{marker}' markers in RheoCompass format."
        )

    logger.debug("Found interval boundaries", n_intervals=len(boundaries))

    # Parse each interval, tracking skipped intervals for data integrity
    blocks = []
    skipped_intervals = []
    for i, (start_idx, interval_idx, _n_points) in enumerate(boundaries):
        end_idx = boundaries[i + 1][0] if i + 1 < len(boundaries) else None
        try:
            block = _parse_single_interval(lines, start_idx, end_idx, decimal_sep)
            blocks.append(block)
        except ValueError as e:
            logger.warning(
                "Skipping unparseable interval — data will be incomplete",
                filepath=str(filepath),
                interval=interval_idx,
                error=str(e),
            )
            skipped_intervals.append((interval_idx, str(e)))
            continue

    if not blocks:
        logger.error("Failed to parse any interval blocks", filepath=str(filepath))
        raise ValueError("Failed to parse any interval blocks from file")

    # Warn loudly if a significant fraction of intervals was lost
    n_total = len(boundaries)
    n_skipped = len(skipped_intervals)
    if n_skipped > 0:
        skip_pct = 100.0 * n_skipped / n_total
        logger.warning(
            "Some intervals could not be parsed",
            filepath=str(filepath),
            skipped=n_skipped,
            total=n_total,
            skip_percent=f"{skip_pct:.0f}%",
            skipped_ids=[s[0] for s in skipped_intervals],
        )
        if n_skipped > n_total / 2:
            raise ValueError(
                f"More than half of the intervals ({n_skipped}/{n_total}) "
                f"failed to parse. The file may be corrupt or in an "
                f"unsupported format. Skipped intervals: "
                f"{[s[0] for s in skipped_intervals]}"
            )

    logger.info(
        "File parsed",
        filepath=str(filepath),
        n_intervals=len(blocks),
        n_skipped=n_skipped,
    )

    return global_metadata, blocks


# =============================================================================
# Column Mapping (T022)
# =============================================================================


def _map_column_to_canonical(column_name: str) -> str | None:
    """Map a RheoCompass column name to canonical name.

    Args:
        column_name: Original column name (may include unit)

    Returns:
        Canonical name or None if no match
    """
    # Extract base name without unit
    base_name, _ = _extract_unit(column_name)
    base_lower = base_name.lower().strip()

    # Use pre-compiled patterns for performance
    for canonical, patterns in _COLUMN_PATTERNS_COMPILED.items():
        for pattern in patterns:
            if pattern.match(base_lower):
                return canonical
    return None


def _convert_unit(
    values: np.ndarray, source_unit: str | None, target_unit: str
) -> tuple[np.ndarray, str]:
    """Convert values from source unit to target SI unit.

    Args:
        values: Array of values
        source_unit: Source unit string (may be None)
        target_unit: Target SI unit

    Returns:
        Tuple of (converted_values, actual_unit)
    """
    if source_unit is None:
        return values, target_unit

    source_lower = source_unit.lower().strip()
    if source_lower in UNIT_CONVERSIONS:
        target, factor = UNIT_CONVERSIONS[source_lower]
        return values * factor, target

    return values, source_unit


def _map_columns_to_canonical(
    df: pd.DataFrame, units_dict: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map DataFrame columns to canonical names with SI units.

    Args:
        df: Original DataFrame
        units_dict: Column name to unit mapping

    Returns:
        Tuple of (mapped DataFrame, canonical units dict)
    """
    mapped_df = pd.DataFrame()
    mapped_units: dict[str, str] = {}

    for col in df.columns:
        canonical = _map_column_to_canonical(col)
        source_unit = units_dict.get(col)

        if canonical:
            # Get target SI unit
            _, target_unit, _ = COLUMN_MAPPINGS[canonical]
            values = df[col].values
            converted, actual_unit = _convert_unit(values, source_unit, target_unit)
            mapped_df[canonical] = converted
            mapped_units[canonical] = actual_unit
        else:
            # Keep original column name (for auxiliary data)
            base_name, _ = _extract_unit(col)
            mapped_df[base_name] = df[col].values
            if source_unit:
                mapped_units[base_name] = source_unit

    return mapped_df, mapped_units


# =============================================================================
# Derived Quantity Computation (T023, T024, T031)
# =============================================================================


def _compute_compliance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate compliance J(t) = strain/stress when absent.

    Args:
        df: DataFrame with canonical column names

    Returns:
        DataFrame with compliance column added if computed
    """
    if "compliance" in df.columns:
        return df

    if "shear_strain" in df.columns and "shear_stress" in df.columns:
        strain = df["shear_strain"].values
        stress = df["shear_stress"].values
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            compliance = np.where(stress != 0, strain / stress, 0.0)
        df = df.copy()
        df["compliance"] = compliance
        logger.debug("Computed compliance J(t) = strain/stress")

    return df


def _compute_relaxation_modulus(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate relaxation modulus G(t) = stress/strain when absent.

    Args:
        df: DataFrame with canonical column names

    Returns:
        DataFrame with relaxation_modulus column added if computed
    """
    if "relaxation_modulus" in df.columns:
        return df

    if "shear_stress" in df.columns and "shear_strain" in df.columns:
        stress = df["shear_stress"].values
        strain = df["shear_strain"].values
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            modulus = np.where(strain != 0, stress / strain, 0.0)
        df = df.copy()
        df["relaxation_modulus"] = modulus
        logger.debug("Computed relaxation modulus G(t) = stress/strain")

    return df


def _compute_complex_modulus(
    df: pd.DataFrame,
) -> tuple[np.ndarray | None, str | None]:
    """Calculate complex modulus G* = G' + i*G'' or E* = E' + i*E''.

    Checks shear modulus columns first, then tensile modulus columns.

    Args:
        df: DataFrame with canonical column names

    Returns:
        Tuple of (complex array, deformation_mode) where deformation_mode
        is "shear", "tension", or None if cannot compute.
    """
    if "storage_modulus" in df.columns and "loss_modulus" in df.columns:
        g_prime = df["storage_modulus"].values
        g_double_prime = df["loss_modulus"].values
        return g_prime + 1j * g_double_prime, "shear"
    if (
        "tensile_storage_modulus" in df.columns
        and "tensile_loss_modulus" in df.columns
    ):
        e_prime = df["tensile_storage_modulus"].values
        e_double_prime = df["tensile_loss_modulus"].values
        return e_prime + 1j * e_double_prime, "tension"
    return None, None


# =============================================================================
# Test Type Detection (T041, T042)
# =============================================================================


def _is_column_constant(series: pd.Series, threshold: float = 0.01) -> bool:
    """Check if a column has constant values (low variance).

    Args:
        series: Pandas series to check
        threshold: Relative variance threshold (default 1%)

    Returns:
        True if column appears constant
    """
    values = series.dropna().values
    if len(values) < 2:
        return True

    mean_val = np.mean(np.abs(values))
    if mean_val == 0:
        return True

    std_val = np.std(values)
    return (std_val / mean_val) < threshold


def _detect_test_type(df: pd.DataFrame) -> str | None:
    """Detect test type from column presence and data characteristics.

    Detection rules (evaluated in priority order):
    1. Oscillatory: Has G'/G'' and frequency
    2. Creep: Has compliance/strain with constant stress
    3. Relaxation: Has G(t)/stress with constant strain
    4. Rotation: Has shear rate and viscosity/stress

    Args:
        df: DataFrame with canonical column names

    Returns:
        Test mode string or None if ambiguous
    """
    columns = set(df.columns)

    # Priority 1: Oscillatory (frequency domain) — includes tensile moduli (DMTA)
    has_frequency = "angular_frequency" in columns
    has_moduli = "storage_modulus" in columns or "loss_modulus" in columns
    has_tensile_moduli = (
        "tensile_storage_modulus" in columns or "tensile_loss_modulus" in columns
    )

    if has_frequency and (has_moduli or has_tensile_moduli):
        return "oscillation"

    # Priority 2: Creep (time domain, constant stress)
    has_time = "time" in columns
    has_compliance_data = "compliance" in columns or "shear_strain" in columns

    if has_time and has_compliance_data:
        if "shear_stress" in columns:
            if _is_column_constant(df["shear_stress"]):
                return "creep"
        elif "compliance" in columns:
            # Has explicit compliance column - likely creep
            return "creep"

    # Priority 3: Relaxation (time domain, constant strain)
    has_relaxation_data = "relaxation_modulus" in columns or "shear_stress" in columns

    if has_time and has_relaxation_data:
        if "shear_strain" in columns:
            if _is_column_constant(df["shear_strain"]):
                return "relaxation"
        elif "relaxation_modulus" in columns:
            # Has explicit G(t) column - likely relaxation
            return "relaxation"

    # Priority 4: Rotation (flow test)
    has_shear_rate = "shear_rate" in columns
    has_flow_data = "viscosity" in columns or "shear_stress" in columns

    if has_shear_rate and has_flow_data:
        # Make sure it's not oscillatory
        if not has_moduli:
            return "rotation"

    return None


# =============================================================================
# Metadata Extraction (T050, T051, T052)
# =============================================================================


def _extract_geometry_metadata(global_meta: dict[str, Any]) -> dict[str, Any]:
    """Extract geometry information from global metadata.

    Args:
        global_meta: Global metadata dictionary

    Returns:
        Dictionary with geometry, gap, diameter keys
    """
    geometry_meta: dict[str, Any] = {}

    # Common geometry keys
    for key in ["Geometry", "geometry", "Measuring System"]:
        if key in global_meta:
            geometry_meta["geometry"] = global_meta[key]
            break

    for key in ["Gap", "gap", "Measuring Gap"]:
        if key in global_meta:
            geometry_meta["gap"] = global_meta[key]
            break

    for key in ["Diameter", "diameter"]:
        if key in global_meta:
            geometry_meta["diameter"] = global_meta[key]
            break

    return geometry_meta


def _extract_temperature_metadata(
    global_meta: dict[str, Any], df: pd.DataFrame
) -> dict[str, Any]:
    """Extract temperature from header and per-point data.

    Args:
        global_meta: Global metadata dictionary
        df: DataFrame with data columns

    Returns:
        Dictionary with temperature info
    """
    temp_meta: dict[str, Any] = {}

    # Header temperature
    for key in ["Temperature", "temperature", "Temp"]:
        if key in global_meta:
            temp_meta["temperature"] = global_meta[key]
            break

    # Per-point temperature
    if "temperature" in df.columns:
        temp_meta["temperature_data"] = df["temperature"].values

    return temp_meta


def _extract_auxiliary_columns(
    df: pd.DataFrame, units_dict: dict[str, str]
) -> dict[str, Any]:
    """Extract auxiliary columns (normal force, torque) into metadata.

    Args:
        df: DataFrame with canonical column names
        units_dict: Column units

    Returns:
        Dictionary with auxiliary data
    """
    aux_meta: dict[str, Any] = {}

    for col in ["normal_force", "torque", "phase_angle", "complex_viscosity"]:
        if col in df.columns:
            aux_meta[col] = df[col].values
            if col in units_dict:
                aux_meta[f"{col}_units"] = units_dict[col]

    return aux_meta


# =============================================================================
# RheoData Converters (T025, T026, T032, T058)
# =============================================================================


def _interval_to_rheodata_creep(
    block: IntervalBlock,
    global_meta: dict[str, Any],
    mapped_df: pd.DataFrame,
    mapped_units: dict[str, str],
) -> RheoData:
    """Convert interval block to RheoData for creep test.

    Args:
        block: Parsed interval block
        global_meta: Global file metadata
        mapped_df: DataFrame with canonical columns
        mapped_units: Units for canonical columns

    Returns:
        RheoData configured for creep analysis
    """
    # Compute compliance if needed
    mapped_df = _compute_compliance(mapped_df)

    # Extract x (time) and y (compliance)
    x = (
        mapped_df["time"].values
        if "time" in mapped_df.columns
        else np.arange(len(mapped_df))
    )

    # Prefer compliance over raw strain
    if "compliance" in mapped_df.columns:
        y = mapped_df["compliance"].values
        y_units = mapped_units.get("compliance", "1/Pa")
    else:
        y = mapped_df["shear_strain"].values
        y_units = mapped_units.get("shear_strain", "dimensionless")

    x_units = mapped_units.get("time", "s")

    # Build metadata
    metadata = {
        "source": "rheocompass",
        "interval_index": block.interval_index,
        "test_mode": "creep",
        **_extract_geometry_metadata(global_meta),
        **_extract_temperature_metadata(global_meta, mapped_df),
        **_extract_auxiliary_columns(mapped_df, mapped_units),
        "columns": list(mapped_df.columns),
        "global_metadata": global_meta,
    }

    return RheoData(
        x=x,
        y=y,
        x_units=x_units,
        y_units=y_units,
        domain="time",
        initial_test_mode="creep",
        metadata=metadata,
        validate=False,
    )


def _interval_to_rheodata_relaxation(
    block: IntervalBlock,
    global_meta: dict[str, Any],
    mapped_df: pd.DataFrame,
    mapped_units: dict[str, str],
) -> RheoData:
    """Convert interval block to RheoData for relaxation test.

    Args:
        block: Parsed interval block
        global_meta: Global file metadata
        mapped_df: DataFrame with canonical columns
        mapped_units: Units for canonical columns

    Returns:
        RheoData configured for relaxation analysis
    """
    # Compute relaxation modulus if needed
    mapped_df = _compute_relaxation_modulus(mapped_df)

    # Extract x (time) and y (G(t))
    x = (
        mapped_df["time"].values
        if "time" in mapped_df.columns
        else np.arange(len(mapped_df))
    )

    # Prefer relaxation_modulus, then shear_stress, fallback to first y-like column
    if "relaxation_modulus" in mapped_df.columns:
        y = mapped_df["relaxation_modulus"].values
        y_units = mapped_units.get("relaxation_modulus", "Pa")
    elif "shear_stress" in mapped_df.columns:
        y = mapped_df["shear_stress"].values
        y_units = mapped_units.get("shear_stress", "Pa")
    else:
        # Fallback: use second column if available
        cols = [c for c in mapped_df.columns if c != "time"]
        if cols:
            y = mapped_df[cols[0]].values
            y_units = mapped_units.get(cols[0], "Pa")
        else:
            y = np.zeros(len(x))
            y_units = "Pa"

    x_units = mapped_units.get("time", "s")

    # Build metadata
    metadata = {
        "source": "rheocompass",
        "interval_index": block.interval_index,
        "test_mode": "relaxation",
        **_extract_geometry_metadata(global_meta),
        **_extract_temperature_metadata(global_meta, mapped_df),
        **_extract_auxiliary_columns(mapped_df, mapped_units),
        "columns": list(mapped_df.columns),
        "global_metadata": global_meta,
    }

    return RheoData(
        x=x,
        y=y,
        x_units=x_units,
        y_units=y_units,
        domain="time",
        initial_test_mode="relaxation",
        metadata=metadata,
        validate=False,
    )


def _interval_to_rheodata_oscillation(
    block: IntervalBlock,
    global_meta: dict[str, Any],
    mapped_df: pd.DataFrame,
    mapped_units: dict[str, str],
) -> RheoData:
    """Convert interval block to RheoData for oscillatory test.

    Args:
        block: Parsed interval block
        global_meta: Global file metadata
        mapped_df: DataFrame with canonical columns
        mapped_units: Units for canonical columns

    Returns:
        RheoData configured for oscillatory analysis with complex G*
    """
    # Extract x (frequency)
    x = (
        mapped_df["angular_frequency"].values
        if "angular_frequency" in mapped_df.columns
        else np.arange(len(mapped_df))
    )

    # Compute complex modulus G* = G' + i*G'' or E* = E' + i*E''
    modulus_star, deformation_mode = _compute_complex_modulus(mapped_df)
    if modulus_star is not None:
        y = modulus_star
    elif "complex_modulus" in mapped_df.columns:
        y = mapped_df["complex_modulus"].values
        deformation_mode = "shear"
    else:
        # Fallback to storage modulus only (shear or tensile)
        if "storage_modulus" in mapped_df.columns:
            y = mapped_df["storage_modulus"].values
            deformation_mode = "shear"
        elif "tensile_storage_modulus" in mapped_df.columns:
            y = mapped_df["tensile_storage_modulus"].values
            deformation_mode = "tension"
        else:
            y = np.zeros(len(mapped_df))

    x_units = mapped_units.get("angular_frequency", "rad/s")
    y_units = "Pa"  # Complex modulus in Pa

    # Build metadata with G' and G'' accessible
    metadata = {
        "source": "rheocompass",
        "interval_index": block.interval_index,
        "test_mode": "oscillation",
        **_extract_geometry_metadata(global_meta),
        **_extract_temperature_metadata(global_meta, mapped_df),
        **_extract_auxiliary_columns(mapped_df, mapped_units),
        "columns": list(mapped_df.columns),
        "global_metadata": global_meta,
    }

    # Set deformation_mode if detected from column names
    if deformation_mode is not None:
        metadata["deformation_mode"] = deformation_mode

    return RheoData(
        x=x,
        y=y,
        x_units=x_units,
        y_units=y_units,
        domain="frequency",
        initial_test_mode="oscillation",
        metadata=metadata,
        validate=False,
    )


def _interval_to_rheodata_rotation(
    block: IntervalBlock,
    global_meta: dict[str, Any],
    mapped_df: pd.DataFrame,
    mapped_units: dict[str, str],
) -> RheoData:
    """Convert interval block to RheoData for rotational/flow test.

    Args:
        block: Parsed interval block
        global_meta: Global file metadata
        mapped_df: DataFrame with canonical columns
        mapped_units: Units for canonical columns

    Returns:
        RheoData configured for flow analysis
    """
    # Extract x (shear rate) and y (viscosity)
    x = (
        mapped_df["shear_rate"].values
        if "shear_rate" in mapped_df.columns
        else np.arange(len(mapped_df))
    )

    if "viscosity" in mapped_df.columns:
        y = mapped_df["viscosity"].values
        y_units = mapped_units.get("viscosity", "Pa.s")
    elif "shear_stress" in mapped_df.columns:
        y = mapped_df["shear_stress"].values
        y_units = mapped_units.get("shear_stress", "Pa")
    else:
        y = np.zeros(len(mapped_df))
        y_units = "Pa.s"

    x_units = mapped_units.get("shear_rate", "1/s")

    # Build metadata
    metadata = {
        "source": "rheocompass",
        "interval_index": block.interval_index,
        "test_mode": "rotation",
        **_extract_geometry_metadata(global_meta),
        **_extract_temperature_metadata(global_meta, mapped_df),
        **_extract_auxiliary_columns(mapped_df, mapped_units),
        "columns": list(mapped_df.columns),
        "global_metadata": global_meta,
    }

    return RheoData(
        x=x,
        y=y,
        x_units=x_units,
        y_units=y_units,
        domain="time",  # Flow curves are rate-domain but use time paradigm
        initial_test_mode="rotation",
        metadata=metadata,
        validate=False,
    )


# =============================================================================
# Main API (T065)
# =============================================================================


def load_anton_paar(
    filepath: str | Path,
    *,
    test_mode: str | None = None,
    interval: int | None = None,
    return_all: bool = False,
    encoding: str | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RheoData | list[RheoData]:
    """Load RheoCompass CSV export file and return RheoData object(s).

    Handles interval-based file structure, auto-detects test type, extracts
    metadata, and normalizes units to SI.

    Args:
        filepath: Path to RheoCompass CSV export file
        test_mode: Explicit test mode override ("creep", "relaxation",
            "oscillation", "rotation"). If None, auto-detected from columns.
        interval: Specific interval index to load (1-based). If None with
            return_all=False, returns first interval.
        return_all: If True, always return list of RheoData.
        encoding: File encoding override (auto-detected if None).
        x_col: Override for x-axis column selection.
        y_col: Override for y-axis column selection.
        progress_callback: Callback receiving (current, total) for progress.

    Returns:
        Single RheoData for single-interval files (unless return_all=True).
        List of RheoData for multi-interval files or when return_all=True.

    Raises:
        FileNotFoundError: File does not exist
        ValueError: No interval blocks, cannot detect test type, or interval
            index out of range
    """
    # Parse raw intervals
    global_meta, blocks = parse_rheocompass_intervals(filepath, encoding=encoding)

    if not blocks:
        raise ValueError("No interval blocks found in file")

    # Handle interval selection
    if interval is not None:
        # Find block with matching interval index
        matching = [b for b in blocks if b.interval_index == interval]
        if not matching:
            valid_indices = [b.interval_index for b in blocks]
            logger.error(
                "Interval not found", interval=interval, valid_indices=valid_indices
            )
            raise ValueError(
                f"Interval {interval} not found. Valid intervals: {valid_indices}"
            )
        blocks = matching

    total_blocks = len(blocks)
    results: list[RheoData] = []

    for i, block in enumerate(blocks):
        if progress_callback:
            progress_callback(i + 1, total_blocks)

        # Map columns to canonical names
        mapped_df, mapped_units = _map_columns_to_canonical(block.df, block.units)

        # Handle custom x/y column selection
        if x_col is not None and x_col in mapped_df.columns:
            pass  # Will be used in converter
        if y_col is not None and y_col in mapped_df.columns:
            pass  # Will be used in converter

        # Detect or use specified test mode
        detected_mode = test_mode
        if detected_mode is None:
            detected_mode = _detect_test_type(mapped_df)
            logger.debug(
                "Auto-detected test mode",
                test_mode=detected_mode,
                interval=block.interval_index,
            )

        if detected_mode is None:
            warnings.warn(
                f"Could not auto-detect test type for interval {block.interval_index}. "
                "Specify test_mode parameter explicitly.",
                UserWarning,
                stacklevel=2,
            )
            # Default to relaxation as safest assumption for time-domain data
            detected_mode = "relaxation"

        # Convert to RheoData using appropriate converter
        if detected_mode == "creep":
            rheo_data = _interval_to_rheodata_creep(
                block, global_meta, mapped_df, mapped_units
            )
        elif detected_mode == "relaxation":
            rheo_data = _interval_to_rheodata_relaxation(
                block, global_meta, mapped_df, mapped_units
            )
        elif detected_mode == "oscillation":
            rheo_data = _interval_to_rheodata_oscillation(
                block, global_meta, mapped_df, mapped_units
            )
        elif detected_mode == "rotation":
            rheo_data = _interval_to_rheodata_rotation(
                block, global_meta, mapped_df, mapped_units
            )
        else:
            logger.error("Unknown test mode", test_mode=detected_mode)
            raise ValueError(f"Unknown test mode: {detected_mode}")

        # Handle custom column overrides
        if x_col is not None and x_col in mapped_df.columns:
            rheo_data = RheoData(
                x=mapped_df[x_col].values,
                y=rheo_data.y,
                x_units=mapped_units.get(x_col),
                y_units=rheo_data.y_units,
                domain=rheo_data.domain,
                initial_test_mode=detected_mode,
                metadata=rheo_data.metadata,
                validate=False,
            )

        if y_col is not None and y_col in mapped_df.columns:
            rheo_data = RheoData(
                x=rheo_data.x,
                y=mapped_df[y_col].values,
                x_units=rheo_data.x_units,
                y_units=mapped_units.get(y_col),
                domain=rheo_data.domain,
                initial_test_mode=detected_mode,
                metadata=rheo_data.metadata,
                validate=False,
            )

        results.append(rheo_data)

    # Return single or list based on parameters
    if return_all or len(results) > 1:
        return results
    return results[0]


# =============================================================================
# Excel Export (save_intervals_to_excel)
# =============================================================================


def save_intervals_to_excel(
    rheo_data_list: list[RheoData] | RheoData,
    filepath: str | Path,
    *,
    include_metadata_sheet: bool = True,
    sheet_prefix: str = "Interval",
) -> None:
    """Export multi-interval RheoData to Excel with one sheet per interval.

    Creates an Excel workbook where each interval becomes its own sheet
    (Interval_1, Interval_2, ...) plus an optional Metadata sheet containing
    global metadata and per-interval summary.

    Args:
        rheo_data_list: Single RheoData or list of RheoData objects
            (typically from load_anton_paar with return_all=True)
        filepath: Output Excel file path (.xlsx)
        include_metadata_sheet: Add a Metadata sheet with global info (default True)
        sheet_prefix: Prefix for interval sheet names (default "Interval")

    Raises:
        ImportError: If pandas or openpyxl not installed
        ValueError: If rheo_data_list is empty

    Example:
        >>> data_list = load_anton_paar("temp_sweep.csv", return_all=True)
        >>> save_intervals_to_excel(data_list, "output.xlsx")
        # Creates: Metadata, Interval_1, Interval_2, Interval_3 sheets
    """
    try:
        import pandas as pd
    except ImportError as exc:
        logger.error("pandas not installed for Excel export", exc_info=True)
        raise ImportError(
            "pandas is required for Excel export. Install with: pip install pandas openpyxl"
        ) from exc

    # Normalize input to list
    if isinstance(rheo_data_list, RheoData):
        rheo_data_list = [rheo_data_list]

    if not rheo_data_list:
        raise ValueError("rheo_data_list cannot be empty")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Write Metadata sheet first
        if include_metadata_sheet:
            metadata_df = _create_metadata_sheet(rheo_data_list)
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

        # Write each interval as its own sheet
        for i, rheo_data in enumerate(rheo_data_list, start=1):
            # Get interval index from metadata if available
            interval_idx = rheo_data.metadata.get("interval_index", i)
            sheet_name = f"{sheet_prefix}_{interval_idx}"

            # Create DataFrame for this interval
            interval_df = _create_interval_dataframe(rheo_data)
            interval_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(
        "Exported intervals to Excel",
        filepath=str(filepath),
        n_intervals=len(rheo_data_list),
    )


def _create_metadata_sheet(rheo_data_list: list[RheoData]) -> pd.DataFrame:
    """Create metadata DataFrame summarizing all intervals.

    Args:
        rheo_data_list: List of RheoData objects

    Returns:
        DataFrame with global metadata and per-interval summary
    """
    import pandas as pd

    rows = []

    # Extract global metadata from first interval
    first_data = rheo_data_list[0]
    global_meta = first_data.metadata.get("global_metadata", {})

    # Add global metadata rows
    for key, value in global_meta.items():
        rows.append({"Property": key, "Value": str(value), "Interval": "Global"})

    # Add per-interval summary
    for i, rheo_data in enumerate(rheo_data_list, start=1):
        interval_idx = rheo_data.metadata.get("interval_index", i)
        rows.append(
            {
                "Property": f"Interval {interval_idx} - Test Mode",
                "Value": rheo_data.test_mode,
                "Interval": str(interval_idx),
            }
        )
        rows.append(
            {
                "Property": f"Interval {interval_idx} - Points",
                "Value": str(len(rheo_data.x)),
                "Interval": str(interval_idx),
            }
        )
        rows.append(
            {
                "Property": f"Interval {interval_idx} - X Units",
                "Value": rheo_data.x_units or "",
                "Interval": str(interval_idx),
            }
        )
        rows.append(
            {
                "Property": f"Interval {interval_idx} - Y Units",
                "Value": rheo_data.y_units or "",
                "Interval": str(interval_idx),
            }
        )

        # Add temperature if available
        temp = rheo_data.metadata.get("temperature")
        if temp:
            rows.append(
                {
                    "Property": f"Interval {interval_idx} - Temperature",
                    "Value": str(temp),
                    "Interval": str(interval_idx),
                }
            )

    return pd.DataFrame(rows)


def _create_interval_dataframe(rheo_data: RheoData) -> pd.DataFrame:
    """Create DataFrame for a single interval's data.

    Args:
        rheo_data: RheoData object for one interval

    Returns:
        DataFrame with x, y (and y_real/y_imag for complex) columns
    """
    import pandas as pd

    # Determine column names based on test mode
    test_mode = rheo_data.test_mode
    x_name = _get_x_column_name(test_mode, rheo_data.x_units)
    y_name = _get_y_column_name(test_mode, rheo_data.y_units)

    data: dict[str, np.ndarray] = {}

    # Add x column
    data[x_name] = np.asarray(rheo_data.x)

    # Add y column(s) - handle complex data
    if rheo_data.is_complex:
        # For complex data, add separate G' and G'' columns
        data["G' (Storage Modulus) [Pa]"] = np.asarray(rheo_data.y_real)
        data["G'' (Loss Modulus) [Pa]"] = np.asarray(rheo_data.y_imag)
        data["|G*| (Complex Modulus) [Pa]"] = np.abs(np.asarray(rheo_data.y))
    else:
        data[y_name] = np.asarray(rheo_data.y)

    # Add auxiliary columns from metadata
    for aux_col in ["temperature_data", "normal_force", "torque", "phase_angle"]:
        if aux_col in rheo_data.metadata:
            aux_data = rheo_data.metadata[aux_col]
            if len(aux_data) == len(rheo_data.x):
                col_name = _format_aux_column_name(aux_col, rheo_data.metadata)
                data[col_name] = np.asarray(aux_data)

    return pd.DataFrame(data)


def _get_x_column_name(test_mode: str, units: str | None) -> str:
    """Get descriptive x-axis column name based on test mode."""
    unit_str = f" [{units}]" if units else ""

    names = {
        "creep": f"Time{unit_str}",
        "relaxation": f"Time{unit_str}",
        "oscillation": f"Angular Frequency{unit_str}",
        "rotation": f"Shear Rate{unit_str}",
    }
    return names.get(test_mode, f"X{unit_str}")


def _get_y_column_name(test_mode: str, units: str | None) -> str:
    """Get descriptive y-axis column name based on test mode."""
    unit_str = f" [{units}]" if units else ""

    names = {
        "creep": f"Compliance J(t){unit_str}",
        "relaxation": f"Relaxation Modulus G(t){unit_str}",
        "oscillation": f"Complex Modulus G*{unit_str}",
        "rotation": f"Viscosity η{unit_str}",
    }
    return names.get(test_mode, f"Y{unit_str}")


def _format_aux_column_name(col_name: str, metadata: dict) -> str:
    """Format auxiliary column name with units."""
    units_key = f"{col_name}_units"
    units = metadata.get(units_key, "")
    unit_str = f" [{units}]" if units else ""

    names = {
        "temperature_data": f"Temperature{unit_str}",
        "normal_force": f"Normal Force{unit_str}",
        "torque": f"Torque{unit_str}",
        "phase_angle": f"Phase Angle{unit_str}",
    }
    return names.get(col_name, col_name)
