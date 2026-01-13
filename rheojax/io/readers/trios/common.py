"""Common utilities for TRIOS file parsers.

This module provides shared data structures, column mappings, unit conversions,
and utility functions used by all TRIOS format parsers (TXT, CSV, Excel, JSON).

The design follows patterns established in anton_paar.py (IntervalBlock) and
trios.py (chunked reading), adapted for the multi-format package structure.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Type aliases
FilePath = str | Path
TestMode = Literal["creep", "relaxation", "oscillation", "rotation", "unknown"]
TRIOSFormat = Literal["csv", "excel", "json", "txt"]
DecimalSeparator = Literal[".", ","]


# =============================================================================
# Data Classes (T006-T009)
# =============================================================================


@dataclass
class TRIOSTable:
    """A single data table from a TRIOS file.

    TRIOS CSV files may contain multiple tables when:
    - Headers repeat (multi-step experiments exported separately)
    - Step column indicates different test phases

    Attributes:
        table_index: 0-based index of this table in the file
        header: Column names as they appear in file
        units: Column name to unit string mapping
        df: Parsed data as DataFrame with original column names
        step_values: Unique step/segment values if step column detected
    """

    table_index: int
    header: list[str]
    units: dict[str, str]
    df: pd.DataFrame
    step_values: list[int] | None = None


@dataclass
class TRIOSFile:
    """Complete parsed TRIOS file.

    Contains global metadata extracted from file header and one or more
    data tables. For single-table files, tables will have exactly one entry.

    Attributes:
        filepath: Original file path
        format: Detected format ("csv", "excel", "json", "txt")
        metadata: Global metadata from file header
        tables: List of parsed data tables
        encoding: Detected or specified encoding
        decimal_separator: Detected decimal separator ("." or ",")
    """

    filepath: str
    format: str
    metadata: dict[str, Any]
    tables: list[TRIOSTable]
    encoding: str = "utf-8"
    decimal_separator: str = "."

    @property
    def has_multiple_tables(self) -> bool:
        """Check if file contains multiple data tables."""
        return len(self.tables) > 1

    @property
    def primary_table(self) -> TRIOSTable:
        """Get the primary (first) data table."""
        return self.tables[0]


@dataclass
class DataSegment:
    """Processed data segment ready for RheoData conversion.

    Represents data after column mapping and unit normalization.
    Multiple segments may exist for multi-step experiments.

    Attributes:
        segment_index: 0-based segment index
        test_mode: Detected or specified test mode
        x_data: Independent variable data (time, frequency, shear rate)
        y_data: Dependent variable data (modulus, compliance, viscosity)
        x_column: Canonical x column name
        y_column: Canonical y column name
        x_units: SI units for x data
        y_units: SI units for y data
        is_complex: Whether y data is complex (G* = G' + iG'')
        metadata: Segment-specific metadata
        auxiliary_columns: Additional columns preserved in metadata
    """

    segment_index: int
    test_mode: str
    x_data: np.ndarray
    y_data: np.ndarray  # May be complex for oscillation data
    x_column: str
    y_column: str
    x_units: str
    y_units: str
    is_complex: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary_columns: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ColumnMapping:
    """Column mapping configuration.

    Defines how a TRIOS column name maps to a canonical RheoJAX name
    with unit conversion rules.

    Attributes:
        canonical_name: RheoJAX canonical column name
        patterns: Regex patterns matching TRIOS column names
        si_unit: Target SI unit after conversion
        applicable_modes: Test modes where this column is relevant
        is_x_candidate: Whether this column can be x-axis
        is_y_candidate: Whether this column can be y-axis
        priority: Selection priority (lower = higher priority)
    """

    canonical_name: str
    patterns: list[str]
    si_unit: str
    applicable_modes: list[str]
    is_x_candidate: bool = False
    is_y_candidate: bool = False
    priority: int = 100


# =============================================================================
# Constants (T010-T012)
# =============================================================================

TRIOS_COLUMN_MAPPINGS: dict[str, ColumnMapping] = {
    # Time/Frequency (x-axis)
    "time": ColumnMapping(
        canonical_name="time",
        patterns=[r"^time$", r"^t$", r"^step\s*time$"],
        si_unit="s",
        applicable_modes=["creep", "relaxation", "rotation"],
        is_x_candidate=True,
        priority=10,
    ),
    "angular_frequency": ColumnMapping(
        canonical_name="angular_frequency",
        patterns=[r"^angular[\s_]?frequency$", r"^frequency$", r"^omega$", r"^ω$"],
        si_unit="rad/s",
        applicable_modes=["oscillation"],
        is_x_candidate=True,
        priority=5,  # Prefer over time for oscillation
    ),
    "shear_rate": ColumnMapping(
        canonical_name="shear_rate",
        patterns=[r"^shear[\s_]?rate$", r"^γ̇$", r"^gamma[\s_]?dot$"],
        si_unit="1/s",
        applicable_modes=["rotation"],
        is_x_candidate=True,
        priority=5,
    ),
    # Moduli (y-axis for oscillation)
    "storage_modulus": ColumnMapping(
        canonical_name="storage_modulus",
        patterns=[r"^storage[\s_]?modulus$", r"^g'$", r"^g_prime$"],
        si_unit="Pa",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=5,
    ),
    "loss_modulus": ColumnMapping(
        canonical_name="loss_modulus",
        patterns=[r"^loss[\s_]?modulus$", r"^g''$", r'^g"$'],
        si_unit="Pa",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=5,
    ),
    # Stress/Strain (y-axis for creep/relaxation)
    "compliance": ColumnMapping(
        canonical_name="compliance",
        patterns=[r"^compliance$", r"^j\(?t\)?$"],
        si_unit="1/Pa",
        applicable_modes=["creep"],
        is_y_candidate=True,
        priority=5,
    ),
    "relaxation_modulus": ColumnMapping(
        canonical_name="relaxation_modulus",
        patterns=[r"^relaxation[\s_]?modulus$", r"^g\(?t\)?$"],
        si_unit="Pa",
        applicable_modes=["relaxation"],
        is_y_candidate=True,
        priority=5,
    ),
    # Flow (y-axis for rotation)
    "viscosity": ColumnMapping(
        canonical_name="viscosity",
        patterns=[r"^viscosity$", r"^η$", r"^eta$"],
        si_unit="Pa.s",
        applicable_modes=["rotation"],
        is_y_candidate=True,
        priority=5,
    ),
    # Auxiliary columns
    "shear_stress": ColumnMapping(
        canonical_name="shear_stress",
        patterns=[r"^shear[\s_]?stress$", r"^stress$", r"^τ$"],
        si_unit="Pa",
        applicable_modes=["creep", "relaxation", "rotation"],
        is_y_candidate=True,
        priority=20,  # Fallback y-axis
    ),
    "shear_strain": ColumnMapping(
        canonical_name="shear_strain",
        patterns=[r"^shear[\s_]?strain$", r"^strain$", r"^γ$"],
        si_unit="dimensionless",
        applicable_modes=["creep", "relaxation"],
        is_y_candidate=True,
        priority=20,
    ),
    "temperature": ColumnMapping(
        canonical_name="temperature",
        patterns=[r"^temperature$", r"^temp$"],
        si_unit="°C",
        applicable_modes=["creep", "relaxation", "oscillation", "rotation"],
        is_x_candidate=False,
        is_y_candidate=False,
        priority=100,
    ),
    "torque": ColumnMapping(
        canonical_name="torque",
        patterns=[r"^torque$"],
        si_unit="N.m",
        applicable_modes=["rotation"],
        is_x_candidate=False,
        is_y_candidate=False,
        priority=100,
    ),
}

TRIOS_UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    # Frequency
    "hz": ("rad/s", 2 * math.pi),
    "Hz": ("rad/s", 2 * math.pi),
    "1/hz": ("rad/s", 2 * math.pi),
    # Time
    "ms": ("s", 0.001),
    "min": ("s", 60.0),
    "mins": ("s", 60.0),
    # Pressure/Modulus
    "kPa": ("Pa", 1000.0),
    "kpa": ("Pa", 1000.0),
    "MPa": ("Pa", 1e6),
    "mPa": ("Pa", 1e6),
    # Viscosity
    "mPa·s": ("Pa.s", 0.001),
    "mPa.s": ("Pa.s", 0.001),
    # Dimensionless
    "%": ("dimensionless", 0.01),
}

STEP_COLUMN_CANDIDATES: list[str] = [
    "step",
    "segment",
    "step_number",
    "segment_number",
    "procedure step",
    "step index",
]


# =============================================================================
# Utility Functions (T013-T020)
# =============================================================================


def detect_test_type(
    df: pd.DataFrame,
    column_mappings: dict[str, ColumnMapping] | None = None,
) -> str:
    """Detect test type from DataFrame columns and data.

    Detection priority:
    1. Oscillation: angular_frequency + (storage_modulus OR loss_modulus)
    2. Creep: time + (compliance OR constant stress with strain)
    3. Relaxation: time + (relaxation_modulus OR constant strain with stress)
    4. Rotation: shear_rate + (viscosity OR shear_stress)
    5. Unknown: Fallback

    Args:
        df: DataFrame with data columns
        column_mappings: Override default column mappings

    Returns:
        Test mode string: "oscillation", "creep", "relaxation", "rotation", "unknown"
    """
    logger.debug(
        "Detecting test type from columns",
        columns=list(df.columns),
        n_rows=len(df),
    )

    if column_mappings is None:
        column_mappings = TRIOS_COLUMN_MAPPINGS

    columns_lower = [c.lower().strip() for c in df.columns]

    def has_column(canonical_name: str) -> bool:
        """Check if a canonical column exists in the DataFrame."""
        if canonical_name not in column_mappings:
            return False
        mapping = column_mappings[canonical_name]
        for pattern in mapping.patterns:
            for col in columns_lower:
                if re.match(pattern, col, re.IGNORECASE):
                    return True
        return False

    # Check for oscillation (highest priority)
    if has_column("angular_frequency") and (
        has_column("storage_modulus") or has_column("loss_modulus")
    ):
        logger.debug("Detected test type: oscillation")
        return "oscillation"

    # Check for creep
    if has_column("time") and has_column("compliance"):
        logger.debug("Detected test type: creep")
        return "creep"

    # Check for relaxation
    if has_column("time") and has_column("relaxation_modulus"):
        logger.debug("Detected test type: relaxation")
        return "relaxation"

    # Check for rotation/flow
    if has_column("shear_rate") and (
        has_column("viscosity") or has_column("shear_stress")
    ):
        logger.debug("Detected test type: rotation")
        return "rotation"

    # Fallback: check for modulus with time (relaxation) or stress/strain
    if has_column("time"):
        if has_column("shear_stress") and has_column("shear_strain"):
            # Could be creep or relaxation - check for constant values
            # For now, default to creep if strain is the dependent variable
            logger.debug("Detected test type: creep (fallback from stress/strain)")
            return "creep"

    logger.debug("Could not detect test type, returning unknown")
    return "unknown"


def map_columns_to_canonical(
    df: pd.DataFrame,
    units: dict[str, str],
    column_mappings: dict[str, ColumnMapping] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map DataFrame columns to canonical names with SI units.

    Args:
        df: Original DataFrame
        units: Column name to unit mapping
        column_mappings: Override default column mappings

    Returns:
        Tuple of (mapped DataFrame, canonical units dict)
    """
    if column_mappings is None:
        column_mappings = TRIOS_COLUMN_MAPPINGS

    new_columns = {}
    new_units = {}

    for col in df.columns:
        col_lower = col.lower().strip()
        matched = False

        for canonical_name, mapping in column_mappings.items():
            for pattern in mapping.patterns:
                if re.match(pattern, col_lower, re.IGNORECASE):
                    new_columns[col] = canonical_name
                    new_units[canonical_name] = units.get(col, mapping.si_unit)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            # Keep original column name if no mapping found
            new_columns[col] = col
            new_units[col] = units.get(col, "")

    # Rename columns
    mapped_df = df.rename(columns=new_columns)

    return mapped_df, new_units


def select_xy_columns(
    df: pd.DataFrame,
    test_mode: str,
    column_mappings: dict[str, ColumnMapping] | None = None,
) -> tuple[str, str, str | None]:
    """Select x, y, and optional y2 columns for given test mode.

    For oscillation data with both G' and G'', returns both y columns
    for complex modulus construction.

    Args:
        df: DataFrame with canonical column names
        test_mode: Test mode for column selection
        column_mappings: Override default column mappings

    Returns:
        Tuple of (x_col, y_col, y2_col) where y2_col is None for non-complex data
    """
    logger.debug(
        "Selecting x/y columns",
        test_mode=test_mode,
        available_columns=list(df.columns),
    )

    if column_mappings is None:
        column_mappings = TRIOS_COLUMN_MAPPINGS

    columns = list(df.columns)
    columns_lower = [c.lower() for c in columns]

    # Get x candidates for this test mode
    x_candidates = []
    for _name, mapping in column_mappings.items():
        if mapping.is_x_candidate and test_mode in mapping.applicable_modes:
            for pattern in mapping.patterns:
                for i, col in enumerate(columns_lower):
                    if re.match(pattern, col, re.IGNORECASE):
                        x_candidates.append((mapping.priority, columns[i]))

    # Get y candidates for this test mode
    y_candidates = []
    for name, mapping in column_mappings.items():
        if mapping.is_y_candidate and test_mode in mapping.applicable_modes:
            for pattern in mapping.patterns:
                for i, col in enumerate(columns_lower):
                    if re.match(pattern, col, re.IGNORECASE):
                        y_candidates.append((mapping.priority, columns[i], name))

    # Sort by priority
    x_candidates.sort(key=lambda x: x[0])
    y_candidates.sort(key=lambda x: x[0])

    x_col = x_candidates[0][1] if x_candidates else None
    y_col = y_candidates[0][1] if y_candidates else None

    # Check for complex modulus case (oscillation with both G' and G'')
    y2_col = None
    if test_mode == "oscillation":
        storage_col = None
        loss_col = None
        for _, col, name in y_candidates:
            if name == "storage_modulus":
                storage_col = col
            elif name == "loss_modulus":
                loss_col = col

        if storage_col and loss_col:
            y_col = storage_col
            y2_col = loss_col
            logger.debug(
                "Selected complex modulus columns",
                x_col=x_col,
                y_col=y_col,
                y2_col=y2_col,
            )

    if x_col is None or y_col is None:
        logger.warning(
            f"Could not determine x/y columns for test mode '{test_mode}'. "
            f"Available columns: {columns}"
        )
    else:
        logger.debug(
            "Selected columns",
            x_col=x_col,
            y_col=y_col,
            y2_col=y2_col,
        )

    return x_col, y_col, y2_col


def convert_unit(
    values: np.ndarray,
    source_unit: str | None,
    target_unit: str,
    conversions: dict[str, tuple[str, float]] | None = None,
) -> tuple[np.ndarray, str]:
    """Convert values from source unit to target SI unit.

    Args:
        values: Array of values
        source_unit: Source unit string (may be None)
        target_unit: Target SI unit
        conversions: Override default unit conversions

    Returns:
        Tuple of (converted_values, actual_unit)
    """
    if conversions is None:
        conversions = TRIOS_UNIT_CONVERSIONS

    if source_unit is None or source_unit == target_unit:
        return values, target_unit

    # Look up conversion
    source_key = source_unit.strip()
    if source_key in conversions:
        converted_target, factor = conversions[source_key]
        if converted_target == target_unit or target_unit in converted_target:
            logger.debug(
                "Converting units",
                source_unit=source_unit,
                target_unit=target_unit,
                factor=factor,
            )
            return values * factor, target_unit

    # No conversion found, return original
    logger.debug(
        "No unit conversion found",
        source_unit=source_unit,
        target_unit=target_unit,
    )
    return values, source_unit or target_unit


def detect_step_column(
    df: pd.DataFrame,
    candidates: list[str] | None = None,
) -> str | None:
    """Find step/segment column in DataFrame.

    Args:
        df: DataFrame to search
        candidates: Override default step column candidates

    Returns:
        Column name if found, None otherwise
    """
    if candidates is None:
        candidates = STEP_COLUMN_CANDIDATES

    for col in df.columns:
        col_lower = col.lower().strip()
        for candidate in candidates:
            if candidate in col_lower:
                # Verify it contains integer-like values
                if df[col].dtype in (int, "int64") or df[col].nunique() < 20:
                    return col
    return None


def split_by_step(
    df: pd.DataFrame,
    step_col: str,
) -> list[pd.DataFrame]:
    """Split DataFrame into per-step DataFrames.

    Args:
        df: DataFrame with step column
        step_col: Name of step column

    Returns:
        List of DataFrames, one per step value
    """
    return [group.copy() for _, group in df.groupby(step_col, sort=False)]


def construct_complex_modulus(
    g_prime: np.ndarray,
    g_double_prime: np.ndarray,
) -> np.ndarray:
    """Construct complex modulus G* = G' + i*G''.

    Args:
        g_prime: Storage modulus array
        g_double_prime: Loss modulus array

    Returns:
        Complex array representing G*
    """
    return g_prime + 1j * g_double_prime


def segment_to_rheodata(
    segment: DataSegment,
    validate: bool = True,
) -> RheoData:
    """Convert DataSegment to RheoData.

    Args:
        segment: Processed data segment
        validate: Validate RheoData on creation

    Returns:
        RheoData object ready for analysis
    """
    # Determine domain from test mode
    domain_map = {
        "oscillation": "frequency",
        "creep": "time",
        "relaxation": "time",
        "rotation": "time",
        "unknown": "time",
    }
    domain = domain_map.get(segment.test_mode, "time")

    # Build metadata
    metadata = segment.metadata.copy()
    metadata["test_mode"] = segment.test_mode
    metadata["x_column"] = segment.x_column
    metadata["y_column"] = segment.y_column
    metadata["is_complex"] = segment.is_complex

    if segment.auxiliary_columns:
        metadata["auxiliary_columns"] = {
            name: arr.tolist() for name, arr in segment.auxiliary_columns.items()
        }

    return RheoData(
        x=segment.x_data,
        y=segment.y_data,
        x_units=segment.x_units,
        y_units=segment.y_units,
        domain=domain,
        metadata=metadata,
        validate=validate,
    )
