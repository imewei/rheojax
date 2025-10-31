"""TA Instruments TRIOS file reader.

This module provides a reader for TA Instruments rheometer files exported
as .txt format using the TRIOS "Export to LIMS" functionality.

The reader supports two modes:

1. **Full Loading** (`load_trios()`): Loads entire file into memory
   - Best for files < 10MB or < 50,000 data points
   - Returns complete RheoData object(s)
   - Simple API for typical use cases

2. **Chunked Reading** (`load_trios_chunked()`): Memory-efficient streaming
   - Best for large files (> 10MB, > 50,000 data points)
   - Returns generator yielding RheoData chunks
   - Reduces memory usage by ~90% for large files
   - Preserves metadata across all chunks

**Memory Requirements:**
- Full loading: ~80 bytes per data point (e.g., 8 MB for 100k points)
- Chunked reading: ~80 bytes × chunk_size (e.g., 800 KB for 10k chunk_size)

**Usage Example - Full Loading:**
    >>> from rheo.io.readers import load_trios
    >>> data = load_trios('small_file.txt')
    >>> print(f"Loaded {len(data.x)} points")

**Usage Example - Chunked Reading:**
    >>> from rheo.io.readers.trios import load_trios_chunked
    >>>
    >>> # Process large file in chunks of 10,000 points
    >>> for i, chunk in enumerate(load_trios_chunked('large_file.txt', chunk_size=10000)):
    ...     print(f"Chunk {i}: {len(chunk.x)} points")
    ...     # Process chunk (e.g., fit model, transform, plot)
    ...     model.fit(chunk.x, chunk.y)
    >>>
    >>> # Aggregate results across chunks
    >>> results = []
    >>> for chunk in load_trios_chunked('large_file.txt'):
    ...     result = process_chunk(chunk)
    ...     results.append(result)
    >>> final_result = aggregate(results)

**When to Use Chunked Reading:**
- Files > 10 MB (typically > 50,000 data points)
- OWChirp arbitrary wave files (often 150k+ points, 66-80 MB)
- Memory-constrained environments
- Processing pipelines that can operate on chunks
- Parallel processing of independent segments

Reference: Ported from hermes-rheo TriosRheoReader
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from rheo.core.data import RheoData

# Unit conversion factors
UNIT_CONVERSIONS = {
    "MPa": ("Pa", 1e6),
    "kPa": ("Pa", 1e3),
    "%": ("unitless", 0.01),
}


def convert_units(
    value: float | np.ndarray, from_unit: str, to_unit: str
) -> float | np.ndarray:
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
        if target == to_unit or to_unit == "Pa":
            return value * factor

    return value


def load_trios(filepath: str, **kwargs) -> RheoData | list[RheoData]:
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

    For large files (> 10 MB, > 50k points), consider using `load_trios_chunked()`
    for memory-efficient streaming.

    Args:
        filepath: Path to TRIOS .txt file
        **kwargs: Additional options
            - return_all_segments: If True, return list of RheoData for each segment
            - chunk_size: If provided, uses chunked reading (see load_trios_chunked)

    Returns:
        RheoData object or list of RheoData objects (if multiple segments)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not recognized

    See Also:
        load_trios_chunked: Memory-efficient streaming for large files
    """
    # If chunk_size is provided, delegate to chunked reader
    if "chunk_size" in kwargs:
        chunk_size = kwargs.pop("chunk_size")
        # For backward compatibility, collect all chunks
        all_chunks = list(load_trios_chunked(filepath, chunk_size=chunk_size, **kwargs))
        return_all = kwargs.get("return_all_segments", False)
        if len(all_chunks) == 1 and not return_all:
            return all_chunks[0]
        else:
            return all_chunks

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read file contents
    with open(filepath, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Split into lines
    lines = content.split("\n")

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
            warnings.warn(f"Failed to parse segment starting at line {seg_start}: {e}", stacklevel=2)

    if not rheo_data_list:
        raise ValueError("No valid data segments could be parsed")

    # Return single RheoData or list
    return_all = kwargs.get("return_all_segments", False)
    if len(rheo_data_list) == 1 and not return_all:
        return rheo_data_list[0]
    else:
        return rheo_data_list


def load_trios_chunked(
    filepath: str,
    chunk_size: int = 10000,
    **kwargs
):
    """Load TRIOS file in memory-efficient chunks (generator).

    This function reads TRIOS files using a streaming approach that yields
    RheoData objects for each chunk of data. This is ideal for large files
    (> 10 MB, > 50,000 points) where loading the entire file would consume
    excessive memory.

    **Memory Efficiency:**
    - Traditional loading: Entire file in memory (~80 bytes per point)
    - Chunked loading: Only chunk_size points in memory at once
    - Example: 150k point file with chunk_size=10k uses ~800 KB vs ~12 MB

    **Important Notes:**
    - Chunks are yielded sequentially as they are read
    - Each chunk is an independent RheoData object with complete metadata
    - Chunk boundaries are based on data rows, not time or other physical units
    - File handle is automatically closed when generator completes or is interrupted

    Args:
        filepath: Path to TRIOS .txt file
        chunk_size: Number of data points per chunk (default: 10,000)
            - Smaller = less memory, more overhead
            - Larger = more memory, less overhead
            - Recommended: 5,000 - 20,000 for most files
        **kwargs: Additional options
            - segment_index: If provided, only process this segment (0-based)
            - validate_data: Validate each chunk (default: True)

    Yields:
        RheoData: Chunks of data with metadata preserved

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not recognized or no segments found

    Example:
        >>> # Process large file in chunks
        >>> for chunk in load_trios_chunked('large_file.txt', chunk_size=10000):
        ...     print(f"Processing {len(chunk.x)} points")
        ...     model.fit(chunk.x, chunk.y)
        >>>
        >>> # Aggregate results from chunks
        >>> max_stress = -float('inf')
        >>> for chunk in load_trios_chunked('file.txt'):
        ...     max_stress = max(max_stress, chunk.y.max())
        >>> print(f"Maximum stress: {max_stress}")

    See Also:
        load_trios: Standard loading (entire file in memory)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    segment_index = kwargs.get("segment_index", None)
    validate_data = kwargs.get("validate_data", True)

    # First pass: extract metadata and locate segments without loading all data
    with open(filepath, encoding="utf-8", errors="replace") as f:
        # Read only header portion for metadata (first 100 lines typically sufficient)
        header_lines = []
        for i, line in enumerate(f):
            header_lines.append(line.rstrip("\n"))
            if i >= 100:
                break

        # Extract metadata from header
        metadata = _extract_metadata(header_lines)

        # Reset to beginning for segment detection
        f.seek(0)

        # Find segments by scanning file
        segment_starts = []
        line_num = 0
        for line in f:
            if re.match(r"\[step\]", line, re.IGNORECASE):
                segment_starts.append(line_num)
            line_num += 1

        if not segment_starts:
            raise ValueError("No data segments found in TRIOS file")

    # Second pass: process each segment in chunks
    target_segments = (
        [segment_index] if segment_index is not None else range(len(segment_starts))
    )

    for seg_idx in target_segments:
        if seg_idx >= len(segment_starts):
            warnings.warn(f"Segment {seg_idx} not found in file", stacklevel=2)
            continue

        seg_start = segment_starts[seg_idx]
        seg_end = (
            segment_starts[seg_idx + 1]
            if seg_idx + 1 < len(segment_starts)
            else None
        )

        # Process this segment in chunks
        yield from _read_segment_chunked(
            filepath,
            seg_start,
            seg_end,
            metadata,
            chunk_size,
            validate_data
        )


def _read_segment_chunked(
    filepath: Path,
    seg_start: int,
    seg_end: int | None,
    metadata: dict,
    chunk_size: int,
    validate_data: bool
):
    """Read a single segment in chunks (internal generator).

    Args:
        filepath: Path to file
        seg_start: Segment start line number
        seg_end: Segment end line number (None for end of file)
        metadata: File metadata dictionary
        chunk_size: Number of data points per chunk
        validate_data: Whether to validate each chunk

    Yields:
        RheoData: Chunks of segment data
    """
    with open(filepath, encoding="utf-8", errors="replace") as f:
        # Skip to segment start
        for _ in range(seg_start):
            next(f)

        # Parse segment header
        segment_lines = []

        # Read until we find data start or reach segment end
        line_num = seg_start
        for line in f:
            segment_lines.append(line.rstrip("\n"))
            line_num += 1

            # Check if we've reached segment end
            if seg_end is not None and line_num >= seg_end:
                break

            # Check if we found "Number of points" (data section starts next)
            if line.startswith("Number of points"):
                # Read column headers and units
                header_line = next(f).rstrip("\n")
                segment_lines.append(header_line)
                unit_line = next(f).rstrip("\n")
                segment_lines.append(unit_line)
                line_num += 2

                # Parse headers
                columns = [col.strip() for col in header_line.split("\t")]
                units = [u.strip() for u in unit_line.split("\t")] if unit_line else [""] * len(columns)

                # Ensure same number of units as columns
                while len(units) < len(columns):
                    units.append("")

                # Determine x/y columns (need dummy data to call function)
                # We'll create a small sample to determine columns
                sample_rows = []

                for _ in range(min(10, chunk_size)):
                    try:
                        line = next(f)
                        line_num += 1
                        if not line.strip() or line.startswith("["):
                            break
                        values = line.split("\t")
                        if len(values) == len(columns):
                            row = [float(v) if v.strip() else np.nan for v in values]
                            sample_rows.append(row)
                    except (StopIteration, ValueError):
                        break

                if not sample_rows:
                    return  # No data in segment

                sample_array = np.array(sample_rows)

                # Determine x/y columns
                x_col, x_units, y_col, y_units = _determine_xy_columns(
                    columns, units, sample_array
                )

                if x_col is None or y_col is None:
                    warnings.warn(
                        f"Could not determine x/y columns from: {columns}",
                        stacklevel=2
                    )
                    return

                # Determine domain and test mode
                domain, test_mode = _infer_domain_and_mode(
                    columns[x_col], columns[y_col], x_units, y_units
                )

                # Update metadata
                segment_metadata = metadata.copy()
                segment_metadata["test_mode"] = test_mode
                segment_metadata["columns"] = columns
                segment_metadata["units"] = units

                # Start accumulating data from sample rows
                x_chunk = sample_array[:, x_col]
                y_chunk = sample_array[:, y_col]

                # Remove NaN values
                valid_mask = ~(np.isnan(x_chunk) | np.isnan(y_chunk))
                x_chunk = x_chunk[valid_mask]
                y_chunk = y_chunk[valid_mask]

                # Initialize chunk buffers
                current_x = []
                current_y = []

                # Add sample data to buffers
                for x_val, y_val in zip(x_chunk, y_chunk):
                    current_x.append(float(x_val))
                    current_y.append(float(y_val))

                for line in f:
                    line_num += 1

                    # Check segment boundary
                    if seg_end is not None and line_num >= seg_end:
                        break

                    if not line.strip() or line.startswith("["):
                        break

                    values = line.split("\t")
                    if len(values) == len(columns):
                        try:
                            row = [float(v) if v.strip() else np.nan for v in values]
                            x_val = row[x_col]
                            y_val = row[y_col]

                            # Skip NaN values
                            if not (np.isnan(x_val) or np.isnan(y_val)):
                                current_x.append(x_val)
                                current_y.append(y_val)

                                # Yield chunk when size reached
                                if len(current_x) >= chunk_size:
                                    yield RheoData(
                                        x=np.array(current_x),
                                        y=np.array(current_y),
                                        x_units=x_units,
                                        y_units=y_units,
                                        domain=domain,
                                        metadata=segment_metadata.copy(),
                                        validate=validate_data,
                                    )
                                    # Reset for next chunk
                                    current_x = []
                                    current_y = []

                        except (ValueError, IndexError):
                            continue

                # Yield remaining data as final chunk
                if len(current_x) > 0:
                    yield RheoData(
                        x=np.array(current_x),
                        y=np.array(current_y),
                        x_units=x_units,
                        y_units=y_units,
                        domain=domain,
                        metadata=segment_metadata.copy(),
                        validate=validate_data,
                    )

                break  # Done with this segment


def _extract_metadata(lines: list[str]) -> dict:
    """Extract metadata from file header.

    Args:
        lines: File lines

    Returns:
        Dictionary of metadata
    """
    metadata = {}

    # Regular expressions for metadata
    patterns = {
        "filename": r"Filename\s+(.*)",
        "instrument_serial_number": r"Instrument serial number\s+(.*)",
        "instrument_name": r"Instrument name\s+(.*)",
        "operator": r"operator\s+(.*)",
        "run_date": r"rundate\s+(.*)",
        "sample_name": r"Sample name\s+(.*)",
        "geometry": r"Geometry name\s+(.*)",
        "geometry_type": r"Geometry type\s+(.*)",
    }

    for line in lines[:100]:  # Check first 100 lines for metadata
        for key, pattern in patterns.items():
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()

    return metadata


def _find_data_segments(lines: list[str]) -> list[tuple]:
    """Find all [step] data segments in file.

    Args:
        lines: File lines

    Returns:
        List of (start_index, end_index) tuples
    """
    segments = []
    step_pattern = r"\[step\]"

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


def _parse_segment(
    lines: list[str], start: int, end: int, metadata: dict
) -> RheoData | None:
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
        if line.startswith("Number of points"):
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
    unit_line = (
        segment_lines[header_offset + 1].strip()
        if header_offset + 1 < len(segment_lines)
        else ""
    )

    if not header_line:
        return None

    # Parse column names
    columns = [col.strip() for col in header_line.split("\t")]
    units = (
        [u.strip() for u in unit_line.split("\t")] if unit_line else [""] * len(columns)
    )

    # Ensure we have same number of units as columns
    while len(units) < len(columns):
        units.append("")

    # Parse data rows
    data_start = header_offset + 2
    data_rows = []

    for line in segment_lines[data_start:]:
        if not line.strip() or line.startswith("["):
            break

        values = line.split("\t")
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
        warnings.warn(f"Could not determine x/y columns from: {columns}", stacklevel=2)
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
    domain, test_mode = _infer_domain_and_mode(
        columns[x_col], columns[y_col], x_units, y_units
    )

    # Update metadata
    segment_metadata = metadata.copy()
    segment_metadata["test_mode"] = test_mode
    segment_metadata["columns"] = columns
    segment_metadata["units"] = units

    return RheoData(
        x=x_data,
        y=y_data,
        x_units=x_units,
        y_units=y_units,
        domain=domain,
        metadata=segment_metadata,
        validate=True,
    )


def _determine_xy_columns(
    columns: list[str], units: list[str], data: np.ndarray
) -> tuple:
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
        "time",
        "angular frequency",
        "frequency",
        "shear rate",
        "temperature",
        "strain",
        "step time",
    ]

    y_priorities = [
        "storage modulus",
        "loss modulus",
        "stress",
        "strain",
        "viscosity",
        "complex modulus",
        "complex viscosity",
        "torque",
        "normal stress",
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

    x_units = units[x_col] if x_col < len(units) else ""
    y_units = units[y_col] if y_col < len(units) else ""

    return x_col, x_units, y_col, y_units


def _infer_domain_and_mode(
    x_name: str, y_name: str, x_units: str, y_units: str
) -> tuple:
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
    if "frequency" in x_lower or "rad/s" in x_units.lower() or "hz" in x_units.lower():
        if "modulus" in y_lower:
            return "frequency", "oscillation"

    # Time domain
    if "time" in x_lower or "s" == x_units.lower():
        if "stress" in y_lower:
            # Check if strain or stress in name
            if "relax" in y_lower:
                return "time", "relaxation"
            else:
                return "time", "creep"
        elif "modulus" in y_lower:
            return "time", "relaxation"

    # Shear rate (steady shear / flow)
    if "shear rate" in x_lower or "1/s" in x_units:
        return "time", "rotation"

    # Temperature sweep
    if "temperature" in x_lower:
        if "modulus" in y_lower:
            return "frequency", "oscillation"  # Temperature sweep at constant frequency
        else:
            return "time", "temperature_sweep"

    # Default
    return "time", "unknown"
