"""TA Instruments TRIOS JSON file reader.

This module provides a reader for TRIOS JSON exports with support for:
- Schema validation against official TRIOS JSON Export Schema
- Structured parsing using TRIOSExperiment dataclasses
- Multiple results and datasets
- Step/Segment columns for multi-step experiments
- Complex modulus construction (G' + iG'')

Usage:
    >>> from rheojax.io.readers.trios import load_trios_json
    >>> data = load_trios_json('relaxation.json')
    >>> print(data.test_mode)  # 'relaxation'
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from rheojax.core.data import RheoData
from rheojax.io.readers.trios.common import (
    DataSegment,
    construct_complex_modulus,
    convert_unit,
    detect_step_column,
    detect_test_type,
    segment_to_rheodata,
    select_xy_columns,
    split_by_step,
)
from rheojax.io.readers.trios.schema import TRIOSExperiment

logger = logging.getLogger(__name__)

# Path to bundled schema
SCHEMA_PATH = Path(__file__).parent / "schema" / "TRIOSJSONExportSchema.json"


def _load_schema() -> dict[str, Any] | None:
    """Load the bundled TRIOS JSON schema.

    Returns:
        Schema dictionary or None if not found
    """
    if not SCHEMA_PATH.exists():
        return None

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_schema(
    data: dict[str, Any],
    *,
    raise_on_error: bool = False,
) -> tuple[bool, list[str]]:
    """Validate JSON data against bundled TRIOS schema.

    Args:
        data: Parsed JSON dictionary
        raise_on_error: Raise ValueError if validation fails

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    if not HAS_JSONSCHEMA:
        logger.debug("jsonschema not installed, skipping validation")
        return True, []

    schema = _load_schema()
    if schema is None:
        logger.warning("TRIOS JSON schema not found, skipping validation")
        return True, []

    errors: list[str] = []

    try:
        jsonschema.validate(data, schema)
        return True, []
    except jsonschema.ValidationError as e:
        error_msg = f"Schema validation error: {e.message}"
        errors.append(error_msg)
        logger.warning(error_msg)

        if raise_on_error:
            raise ValueError(error_msg) from e

        return False, errors
    except jsonschema.SchemaError as e:
        error_msg = f"Schema error: {e.message}"
        errors.append(error_msg)
        logger.error(error_msg)
        return False, errors


def parse_trios_json(
    filepath: str | Path,
    *,
    validate: bool = True,
) -> tuple[TRIOSExperiment, dict[str, Any]]:
    """Low-level JSON parser returning TRIOSExperiment and metadata.

    Args:
        filepath: Path to TRIOS JSON file
        validate: Validate against bundled schema

    Returns:
        Tuple of (TRIOSExperiment, metadata dict)

    Raises:
        FileNotFoundError: File does not exist
        json.JSONDecodeError: Invalid JSON syntax
        ValueError: Invalid structure or schema validation failed
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read and parse JSON
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    # Validate schema if requested
    if validate:
        is_valid, errors = validate_schema(data)
        if not is_valid:
            logger.warning(
                f"Schema validation failed with {len(errors)} error(s). "
                "Attempting best-effort parsing."
            )

    # Check for schema version mismatch
    data_schema = data.get("$schema") or data.get("schemaVersion")
    if data_schema:
        logger.debug(f"JSON schema version: {data_schema}")

    # Parse into TRIOSExperiment
    try:
        experiment = TRIOSExperiment.from_json(data)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid TRIOS JSON structure: {e}") from e

    # Extract metadata
    metadata = experiment.get_metadata()
    metadata["source_file"] = str(filepath)
    metadata["source_format"] = "json"

    return experiment, metadata


def load_trios_json(
    filepath: str | Path,
    *,
    return_all_segments: bool = False,
    test_mode: str | None = None,
    result_index: int = 0,
    validate_schema: bool = True,
    validate: bool = True,
) -> RheoData | list[RheoData]:
    """Load TRIOS JSON export file.

    Uses adapted tadatakit code to parse TRIOS JSON format with
    schema validation against official TRIOS JSON Export Schema.

    Args:
        filepath: Path to TRIOS JSON file
        return_all_segments: Return list for multi-step files
        test_mode: Override auto-detection ("creep", "relaxation", "oscillation", "rotation")
        result_index: Result set index to load (default: 0, or -1 for all)
        validate_schema: Validate against TRIOS schema (default: True)
        validate: Validate RheoData on creation

    Returns:
        Single RheoData or list of RheoData

    Raises:
        FileNotFoundError: File does not exist
        ValueError: Invalid JSON structure or schema mismatch
        json.JSONDecodeError: Invalid JSON syntax

    Notes:
        Schema version mismatch logs warning but attempts parsing.

    Example:
        >>> data = load_trios_json('relaxation.json')
        >>> print(data.test_mode)  # 'relaxation'
        >>> print(data.x_units)  # 's' (time)
        >>> print(data.y_units)  # 'Pa' (relaxation modulus)
    """
    # Parse JSON file
    experiment, base_metadata = parse_trios_json(filepath, validate=validate_schema)

    if experiment.n_results == 0:
        raise ValueError(f"No results found in {filepath}")

    # Determine which results to process
    if result_index == -1:
        result_indices = list(range(experiment.n_results))
    else:
        if result_index >= experiment.n_results:
            raise ValueError(
                f"Result index {result_index} out of range. "
                f"File contains {experiment.n_results} result(s)."
            )
        result_indices = [result_index]

    rheo_data_list: list[RheoData] = []

    for res_idx in result_indices:
        result = experiment.results[res_idx]
        df = result.get_dataframe()
        units = result.get_units()

        if df.empty:
            logger.warning(f"Result {res_idx} has no data, skipping")
            continue

        # Detect or use provided test mode
        detected_mode = test_mode or detect_test_type(df)

        # Check for step column and split if needed
        step_col = detect_step_column(df)
        segments = (
            [df]
            if not step_col or not return_all_segments
            else split_by_step(df, step_col)
        )

        for seg_idx, seg_df in enumerate(segments):
            # Select x/y columns
            x_col, y_col, y2_col = select_xy_columns(seg_df, detected_mode)

            if x_col is None or y_col is None:
                logger.warning(
                    f"Could not determine x/y columns for result {res_idx}, "
                    f"segment {seg_idx}. Available columns: {list(seg_df.columns)}"
                )
                continue

            # Extract data
            x_data = seg_df[x_col].values.astype(float)

            # Get units
            x_units = units.get(x_col, "")
            y_units = units.get(y_col, "Pa")

            # Handle complex modulus case
            if y2_col is not None:
                y_real = seg_df[y_col].values.astype(float)
                y_imag = seg_df[y2_col].values.astype(float)

                # Convert units if needed
                y_units_orig = units.get(y_col, "Pa")
                y2_units_orig = units.get(y2_col, "Pa")
                y_real, _ = convert_unit(y_real, y_units_orig, "Pa")
                y_imag, _ = convert_unit(y_imag, y2_units_orig, "Pa")

                # Construct complex modulus
                y_data = construct_complex_modulus(y_real, y_imag)
                y_units = "Pa"
                is_complex = True
            else:
                y_data = seg_df[y_col].values.astype(float)
                is_complex = False

            # Convert x units (e.g., Hz to rad/s)
            if detected_mode == "oscillation":
                x_data, x_units = convert_unit(x_data, x_units, "rad/s")

            # Remove NaN values
            if is_complex:
                valid_mask = ~(
                    np.isnan(x_data)
                    | np.isnan(np.real(y_data))
                    | np.isnan(np.imag(y_data))
                )
            else:
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))

            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]

            if len(x_data) == 0:
                continue

            # Determine default x_units based on test mode
            if not x_units:
                if detected_mode == "oscillation":
                    x_units = "rad/s"
                elif detected_mode == "rotation":
                    x_units = "1/s"
                else:
                    x_units = "s"

            # Build metadata
            seg_metadata = base_metadata.copy()
            seg_metadata["test_mode"] = detected_mode
            seg_metadata["result_index"] = res_idx
            seg_metadata["x_column"] = x_col
            seg_metadata["y_column"] = y_col
            if y2_col:
                seg_metadata["y2_column"] = y2_col
            seg_metadata["is_complex"] = is_complex

            # Add result-level properties
            if result.properties:
                for key, value in result.properties.items():
                    seg_metadata[f"result_{_snake_case(key)}"] = value

            # Create DataSegment and convert to RheoData
            segment = DataSegment(
                segment_index=seg_idx,
                test_mode=detected_mode,
                x_data=x_data,
                y_data=y_data,
                x_column=x_col,
                y_column=y_col,
                x_units=x_units,
                y_units=y_units,
                is_complex=is_complex,
                metadata=seg_metadata,
            )

            rheo_data = segment_to_rheodata(segment, validate=validate)
            rheo_data_list.append(rheo_data)

    if not rheo_data_list:
        raise ValueError(f"No valid data segments could be parsed from {filepath}")

    # Return single or list
    if len(rheo_data_list) == 1 and not return_all_segments and result_index != -1:
        return rheo_data_list[0]
    return rheo_data_list


def _snake_case(s: str) -> str:
    """Convert CamelCase to snake_case."""
    result = []
    for i, char in enumerate(s):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)
