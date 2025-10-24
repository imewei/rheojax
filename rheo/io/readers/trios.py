"""TA Instruments TRIOS file reader.

This module provides a reader for TA Instruments rheometer files exported
as .txt format using the TRIOS "Export to LIMS" functionality.

Reference: Ported from hermes-rheo TriosRheoReader
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np

from rheo.core.data import RheoData


# Unit conversion factors
UNIT_CONVERSIONS = {
    'MPa': ('Pa', 1e6),
    'kPa': ('Pa', 1e3),
    '%': ('unitless', 0.01),
}


def convert_units(value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert values between units.

    Args:
        value: Value or array to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value(s)
    """
    if from_unit == to_unit:
        return value

    if from_unit in UNIT_CONVERSIONS:
        target, factor = UNIT_CONVERSIONS[from_unit]
        if target == to_unit or to_unit == 'Pa':
            return value * factor

    return value


def load_trios(filepath: str, **kwargs) -> Union[RheoData, List[RheoData]]:
    """Load TA Instruments TRIOS .txt file.

    Reads rheological data from TRIOS exported .txt files. Supports multiple
    measurement types including:
    - Frequency sweep (SAOS)
    - Amplitude sweep
    - Flow ramp (steady shear)
    - Stress relaxation
    - Creep
    - Temperature sweep
    - Arbitrary wave

    Args:
        filepath: Path to TRIOS .txt file
        **kwargs: Additional options
            - return_all_segments: If True, return list of RheoData for each segment

    Returns:
        RheoData object or list of RheoData objects (if multiple segments)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not recognized
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read file contents
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Split into lines
    lines = content.split('\n')

    # Extract metadata
    metadata = _extract_metadata(lines)

    # Find all data segments
    segments = _find_data_segments(lines)

    if not segments:
        raise ValueError("No data segments found in TRIOS file")

    # Parse each segment
    rheo_data_list = []
    for seg_start, seg_end in segments:
        try:
            data = _parse_segment(lines, seg_start, seg_end, metadata)
            if data is not None:
                rheo_data_list.append(data)
        except Exception as e:
            warnings.warn(f"Failed to parse segment starting at line {seg_start}: {e}")

    if not rheo_data_list:
        raise ValueError("No valid data segments could be parsed")

    # Return single RheoData or list
    return_all = kwargs.get('return_all_segments', False)
    if len(rheo_data_list) == 1 and not return_all:
        return rheo_data_list[0]
    else:
        return rheo_data_list


def _extract_metadata(lines: List[str]) -> Dict:
    """Extract metadata from file header.

    Args:
        lines: File lines

    Returns:
        Dictionary of metadata
    """
    metadata = {}

    # Regular expressions for metadata
    patterns = {
        'filename': r'Filename\s+(.*)',
        'instrument_serial_number': r'Instrument serial number\s+(.*)',
        'instrument_name': r'Instrument name\s+(.*)',
        'operator': r'operator\s+(.*)',
        'run_date': r'rundate\s+(.*)',
        'sample_name': r'Sample name\s+(.*)',
        'geometry': r'Geometry name\s+(.*)',
        'geometry_type': r'Geometry type\s+(.*)',
    }

    for line in lines[:100]:  # Check first 100 lines for metadata
        for key, pattern in patterns.items():
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()

    return metadata


def _find_data_segments(lines: List[str]) -> List[tuple]:
    """Find all [step] data segments in file.

    Args:
        lines: File lines

    Returns:
        List of (start_index, end_index) tuples
    """
    segments = []
    step_pattern = r'\[step\]'

    for i, line in enumerate(lines):
        if re.match(step_pattern, line, re.IGNORECASE):
            segments.append(i)

    # Convert to (start, end) pairs
    segment_pairs = []
    for i in range(len(segments)):
        start = segments[i]
        end = segments[i + 1] if i + 1 < len(segments) else len(lines)
        segment_pairs.append((start, end))

    return segment_pairs


def _parse_segment(lines: List[str], start: int, end: int, metadata: Dict) -> Optional[RheoData]:
    """Parse a single data segment.

    Args:
        lines: File lines
        start: Segment start index
        end: Segment end index
        metadata: File metadata

    Returns:
        RheoData object or None if segment can't be parsed
    """
    # Find header and data lines
    segment_lines = lines[start:end]

    # Look for "Number of points" line
    num_points_line = None
    for i, line in enumerate(segment_lines):
        if line.startswith('Number of points'):
            num_points_line = i
            break

    if num_points_line is not None:
        header_offset = num_points_line + 1
    else:
        # Try to find column headers
        header_offset = 1

    # Extract column headers and units
    if header_offset >= len(segment_lines):
        return None

    header_line = segment_lines[header_offset].strip()
    unit_line = segment_lines[header_offset + 1].strip() if header_offset + 1 < len(segment_lines) else ""

    if not header_line:
        return None

    # Parse column names
    columns = [col.strip() for col in header_line.split('\t')]
    units = [u.strip() for u in unit_line.split('\t')] if unit_line else [''] * len(columns)

    # Ensure we have same number of units as columns
    while len(units) < len(columns):
        units.append('')

    # Parse data rows
    data_start = header_offset + 2
    data_rows = []

    for line in segment_lines[data_start:]:
        if not line.strip() or line.startswith('['):
            break

        values = line.split('\t')
        if len(values) == len(columns):
            try:
                row = [float(v) if v.strip() else np.nan for v in values]
                data_rows.append(row)
            except ValueError:
                # Skip rows that can't be converted
                continue

    if not data_rows:
        return None

    # Convert to numpy array
    data_array = np.array(data_rows)

    # Determine x and y columns based on common column names
    x_col, x_units, y_col, y_units = _determine_xy_columns(columns, units, data_array)

    if x_col is None or y_col is None:
        warnings.warn(f"Could not determine x/y columns from: {columns}")
        return None

    # Extract x and y data
    x_data = data_array[:, x_col]
    y_data = data_array[:, y_col]

    # Remove NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]

    if len(x_data) == 0:
        return None

    # Determine domain and test mode
    domain, test_mode = _infer_domain_and_mode(columns[x_col], columns[y_col], x_units, y_units)

    # Update metadata
    segment_metadata = metadata.copy()
    segment_metadata['test_mode'] = test_mode
    segment_metadata['columns'] = columns
    segment_metadata['units'] = units

    return RheoData(
        x=x_data,
        y=y_data,
        x_units=x_units,
        y_units=y_units,
        domain=domain,
        metadata=segment_metadata,
        validate=True
    )


def _determine_xy_columns(columns: List[str], units: List[str], data: np.ndarray) -> tuple:
    """Determine which columns to use for x and y.

    Args:
        columns: Column names
        units: Column units
        data: Data array

    Returns:
        Tuple of (x_col_index, x_units, y_col_index, y_units)
    """
    columns_lower = [c.lower() for c in columns]

    # Priority lists for x and y columns
    x_priorities = [
        'time', 'angular frequency', 'frequency', 'shear rate',
        'temperature', 'strain', 'step time'
    ]

    y_priorities = [
        'storage modulus', 'loss modulus', 'stress', 'strain',
        'viscosity', 'complex modulus', 'complex viscosity',
        'torque', 'normal stress'
    ]

    # Find x column
    x_col = None
    for priority in x_priorities:
        for i, col in enumerate(columns_lower):
            if priority in col:
                x_col = i
                break
        if x_col is not None:
            break

    # Find y column (prefer storage/loss modulus for SAOS)
    y_col = None
    for priority in y_priorities:
        for i, col in enumerate(columns_lower):
            if priority in col and i != x_col:
                y_col = i
                break
        if y_col is not None:
            break

    # Fallback: use first two numeric columns
    if x_col is None or y_col is None:
        numeric_cols = []
        for i in range(min(data.shape[1], len(columns))):
            if not np.all(np.isnan(data[:, i])):
                numeric_cols.append(i)

        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0] if x_col is None else x_col
            y_col = numeric_cols[1] if y_col is None else y_col

    if x_col is None or y_col is None:
        return None, None, None, None

    x_units = units[x_col] if x_col < len(units) else ''
    y_units = units[y_col] if y_col < len(units) else ''

    return x_col, x_units, y_col, y_units


def _infer_domain_and_mode(x_name: str, y_name: str, x_units: str, y_units: str) -> tuple:
    """Infer domain and test mode from column names and units.

    Args:
        x_name: X column name
        y_name: Y column name
        x_units: X units
        y_units: Y units

    Returns:
        Tuple of (domain, test_mode)
    """
    x_lower = x_name.lower()
    y_lower = y_name.lower()

    # Frequency domain (SAOS)
    if 'frequency' in x_lower or 'rad/s' in x_units.lower() or 'hz' in x_units.lower():
        if 'modulus' in y_lower:
            return 'frequency', 'oscillation'

    # Time domain
    if 'time' in x_lower or 's' == x_units.lower():
        if 'stress' in y_lower:
            # Check if strain or stress in name
            if 'relax' in y_lower:
                return 'time', 'relaxation'
            else:
                return 'time', 'creep'
        elif 'modulus' in y_lower:
            return 'time', 'relaxation'

    # Shear rate (steady shear / flow)
    if 'shear rate' in x_lower or '1/s' in x_units:
        return 'time', 'rotation'

    # Temperature sweep
    if 'temperature' in x_lower:
        if 'modulus' in y_lower:
            return 'frequency', 'oscillation'  # Temperature sweep at constant frequency
        else:
            return 'time', 'temperature_sweep'

    # Default
    return 'time', 'unknown'
