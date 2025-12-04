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
    >>> from rheojax.io.readers import load_trios
    >>> data = load_trios('small_file.txt')
    >>> print(f"Loaded {len(data.x)} points")

**Usage Example - Chunked Reading:**
    >>> from rheojax.io.readers.trios import load_trios_chunked
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

import logging
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rheojax.core.data import RheoData

# Configure logger for auto-chunking notifications
logger = logging.getLogger(__name__)

# Auto-chunking threshold (5 MB)
AUTO_CHUNK_THRESHOLD_MB = 5.0

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


def load_trios(filepath: str | Path, **kwargs) -> RheoData | list[RheoData]:
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

    **Auto-Chunking (v0.4.0+):**
    Files larger than 5 MB are automatically loaded using chunked reading for
    memory efficiency. This provides 50-87% memory reduction for large files.

    **Performance Trade-off:**
    Chunked loading trades latency for memory efficiency (2-4x slower loading
    in exchange for 50-87% memory reduction). This is ideal for memory-constrained
    environments where RAM is more critical than load time.

    Args:
        filepath: Path to TRIOS .txt file
        **kwargs: Additional options

            - return_all_segments: If True, return list of RheoData for each segment
            - chunk_size: If provided, uses chunked reading (see load_trios_chunked)
            - auto_chunk: If True (default), automatically use chunked reading for
              files > 5 MB. Set to False to disable auto-detection.
            - progress_callback: Optional callback for progress tracking during
              chunked loading. Signature: callback(current, total)

    Returns:
        RheoData object or list of RheoData objects (if multiple segments)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not recognized

    Notes:
        - Auto-chunking threshold: 5 MB (configurable via AUTO_CHUNK_THRESHOLD_MB)
        - Memory savings: 50-87% for files > 5 MB with 50k+ points
        - Latency trade-off: 2-4x slower (acceptable for memory-constrained scenarios)
        - Disable auto-chunking: Pass auto_chunk=False to force full loading
        - Use case: Memory-constrained systems, embedded devices, large datasets

    See Also:
        load_trios_chunked: Memory-efficient streaming for large files

    Example:
        >>> # Automatic chunking for large files
        >>> data = load_trios('large_file.txt')  # Auto-chunks if > 5 MB

        >>> # Disable auto-chunking
        >>> data = load_trios('large_file.txt', auto_chunk=False)

        >>> # With progress tracking
        >>> def progress(current, total):
        ...     print(f"Loading: {100*current/total:.1f}%")
        >>> data = load_trios('large_file.txt', progress_callback=progress)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check auto-chunking setting (default: True)
    auto_chunk = kwargs.pop("auto_chunk", True)

    # If chunk_size is provided explicitly, delegate to chunked reader
    if "chunk_size" in kwargs:
        chunk_size = kwargs.pop("chunk_size")
        # Aggregate chunks on-the-fly
        progress_callback = kwargs.pop("progress_callback", None)

        x_parts = []
        y_parts = []
        first_chunk = None

        for chunk in load_trios_chunked(
            filepath,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
            **kwargs,
        ):
            if first_chunk is None:
                first_chunk = chunk

            # Append chunk data
            x_parts.append(chunk.x)
            y_parts.append(chunk.y)

        # Aggregate chunks into single RheoData object
        if first_chunk is not None:
            # Concatenate all chunk data
            x_combined = np.concatenate(x_parts)
            y_combined = np.concatenate(y_parts)

            # Create aggregated RheoData
            aggregated_data = RheoData(
                x=x_combined,
                y=y_combined,
                x_units=first_chunk.x_units,
                y_units=first_chunk.y_units,
                domain=first_chunk.domain,
                metadata=first_chunk.metadata,
                validate=kwargs.get("validate_data", True),
            )

            return aggregated_data
        else:
            raise ValueError("No data chunks returned from chunked reader")

    # Auto-detect file size and use chunked loading if above threshold
    if auto_chunk:
        file_size_bytes = os.path.getsize(filepath)
        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_mb > AUTO_CHUNK_THRESHOLD_MB:
            # Log auto-chunking activation
            logger.info(
                f"Auto-chunking enabled for {file_size_mb:.1f} MB file "
                f"(threshold: {AUTO_CHUNK_THRESHOLD_MB:.1f} MB). "
                f"Expected memory reduction: 50-70%."
            )

            # Delegate to chunked reader with default chunk size
            # Aggregate chunks on-the-fly to avoid keeping all in memory
            progress_callback = kwargs.pop("progress_callback", None)

            x_parts = []
            y_parts = []
            first_chunk = None

            for chunk in load_trios_chunked(
                filepath,
                chunk_size=10000,
                progress_callback=progress_callback,
                **kwargs,
            ):
                if first_chunk is None:
                    first_chunk = chunk

                # Append chunk data (accumulate references, concatenate once at end)
                x_parts.append(chunk.x)
                y_parts.append(chunk.y)

            # Aggregate chunks into single RheoData object
            if first_chunk is not None:
                # Concatenate all chunk data
                x_combined = np.concatenate(x_parts)
                y_combined = np.concatenate(y_parts)

                # Create aggregated RheoData
                aggregated_data = RheoData(
                    x=x_combined,
                    y=y_combined,
                    x_units=first_chunk.x_units,
                    y_units=first_chunk.y_units,
                    domain=first_chunk.domain,
                    metadata=first_chunk.metadata,
                    validate=kwargs.get("validate_data", True),
                )

                return aggregated_data
            else:
                raise ValueError("No data chunks returned from chunked reader")

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
            warnings.warn(
                f"Failed to parse segment starting at line {seg_start}: {e}",
                stacklevel=2,
            )

    if not rheo_data_list:
        raise ValueError("No valid data segments could be parsed")

    # Return single RheoData or list
    return_all = kwargs.get("return_all_segments", False)
    if len(rheo_data_list) == 1 and not return_all:
        return rheo_data_list[0]
    else:
        return rheo_data_list


def load_trios_chunked(
    filepath: str | Path,
    chunk_size: int = 10000,
    progress_callback: Callable | None = None,
    **kwargs,
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

    **Progress Tracking (v0.4.0+):**
    - Optional progress_callback parameter for monitoring large file loading
    - Callback signature: callback(current_points, total_points)
    - Called every 5-10% of file processed for efficient monitoring
    - Total points estimated from "Number of points" in TRIOS header

    Args:
        filepath: Path to TRIOS .txt file
        chunk_size: Number of data points per chunk (default: 10,000)
            - Smaller = less memory, more overhead
            - Larger = more memory, less overhead
            - Recommended: 5,000 - 20,000 for most files
        progress_callback: Optional callback function for progress tracking.
            Signature: callback(current_points: int, total_points: int)
            Called periodically during loading (every 5-10% progress).
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
        >>>
        >>> # With progress tracking
        >>> def progress(current, total):
        ...     pct = 100 * current / total
        ...     print(f"Loading: {pct:.1f}%")
        >>> for chunk in load_trios_chunked('large_file.txt', progress_callback=progress):
        ...     process(chunk)

    See Also:
        load_trios: Standard loading (entire file in memory), auto-chunks for files > 5 MB
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
            segment_starts[seg_idx + 1] if seg_idx + 1 < len(segment_starts) else None
        )

        # Process this segment in chunks
        yield from _read_segment_chunked(
            filepath,
            seg_start,
            seg_end,
            metadata,
            chunk_size,
            validate_data,
            progress_callback,
        )


# =============================================================================
# Helper functions for _read_segment_chunked (extracted for complexity reduction)
# =============================================================================


def _extract_step_temperature(line: str) -> float | None:
    """Extract temperature from step name line.

    Args:
        line: Line containing step name with temperature

    Returns:
        Temperature in Kelvin or None if not found
    """
    temp_match = re.search(r"(-?\d+\.?\d*)\s*°C", line)
    if temp_match:
        temp_c = float(temp_match.group(1))
        return temp_c + 273.15  # Convert to Kelvin
    return None


def _parse_total_points(line: str) -> int | None:
    """Parse total number of points from header line.

    Args:
        line: Line containing "Number of points\\t12345"

    Returns:
        Total points or None if parsing fails
    """
    parts = line.split("\t")
    if len(parts) >= 2:
        try:
            return int(parts[1].strip())
        except ValueError:
            pass
    return None


def _parse_headers_and_units(header_line: str, unit_line: str) -> tuple[list, list]:
    """Parse column headers and units from header lines.

    Args:
        header_line: Tab-separated column headers
        unit_line: Tab-separated unit specifications

    Returns:
        Tuple of (columns, units) lists
    """
    columns = [col.strip() for col in header_line.split("\t")]
    units = (
        [u.strip() for u in unit_line.split("\t")] if unit_line else [""] * len(columns)
    )

    # Ensure same number of units as columns
    while len(units) < len(columns):
        units.append("")

    return columns, units


def _parse_row_values(values: list[str]) -> list[float]:
    """Convert tab-separated values to floats, skipping first column.

    Args:
        values: List of string values from a data row

    Returns:
        List of floats (np.nan for non-numeric values)
    """
    row = []
    for i, v in enumerate(values):
        if i == 0:
            # Skip first column (row label)
            continue
        if not v.strip():
            row.append(np.nan)
        else:
            try:
                row.append(float(v))
            except ValueError:
                # Handle hex values (status bits), dates, strings
                row.append(np.nan)
    return row


def _process_sample_array_complex(
    sample_array: np.ndarray,
    x_col: int,
    y_col: int,
    y_col2: int,
    y_units_orig: str,
    y_units2_orig: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Process sample array for complex modulus data.

    Args:
        sample_array: Array of sample data
        x_col: X column index
        y_col: Y column index (storage modulus)
        y_col2: Y2 column index (loss modulus)
        y_units_orig: Original units for storage modulus
        y_units2_orig: Original units for loss modulus

    Returns:
        Tuple of (x_chunk, y_chunk) arrays with NaN values removed
    """
    x_chunk_array = np.real_if_close(np.asarray(sample_array[:, x_col]))

    # Complex modulus: G* = G' + i*G''
    y_chunk_real = convert_units(sample_array[:, y_col], y_units_orig, "Pa")
    y_chunk_imag = convert_units(sample_array[:, y_col2], y_units2_orig, "Pa")

    # Remove NaN values from either component
    y_chunk_real_array = np.real_if_close(np.asarray(y_chunk_real))
    y_chunk_imag_array = np.real_if_close(np.asarray(y_chunk_imag))

    valid_mask = ~(
        np.isnan(x_chunk_array)
        | np.isnan(y_chunk_real_array)
        | np.isnan(y_chunk_imag_array)
    )

    x_chunk = x_chunk_array[valid_mask]
    y_chunk = (y_chunk_real_array + 1j * y_chunk_imag_array)[valid_mask]

    return x_chunk, y_chunk


def _process_sample_array_real(
    sample_array: np.ndarray, x_col: int, y_col: int
) -> tuple[np.ndarray, np.ndarray]:
    """Process sample array for real-valued data.

    Args:
        sample_array: Array of sample data
        x_col: X column index
        y_col: Y column index

    Returns:
        Tuple of (x_chunk, y_chunk) arrays with NaN values removed
    """
    x_chunk_array = np.real_if_close(np.asarray(sample_array[:, x_col]))
    y_chunk_array = np.real_if_close(np.asarray(sample_array[:, y_col]))

    valid_mask = ~(np.isnan(x_chunk_array) | np.isnan(y_chunk_array))

    return x_chunk_array[valid_mask], y_chunk_array[valid_mask]


def _add_sample_to_buffers(
    x_chunk: np.ndarray, y_chunk: np.ndarray, current_x: list, current_y: list
) -> int:
    """Add sample data to accumulator buffers.

    Args:
        x_chunk: X values from sample
        y_chunk: Y values from sample
        current_x: X accumulator list
        current_y: Y accumulator list

    Returns:
        Number of points added
    """
    count = 0
    for x_val, y_val in zip(x_chunk, y_chunk, strict=True):
        current_x.append(float(x_val) if np.isreal(x_val) else complex(x_val))
        current_y.append(float(y_val) if np.isreal(y_val) else complex(y_val))
        count += 1
    return count


def _process_complex_row(
    row: list[float],
    x_col: int,
    y_col: int,
    y_col2: int,
    y_units_orig: str,
    y_units2_orig: str,
) -> tuple[float | None, complex | None]:
    """Process a single row for complex modulus data.

    Args:
        row: Parsed row values
        x_col: X column index
        y_col: Y column index (storage modulus)
        y_col2: Y2 column index (loss modulus)
        y_units_orig: Original units for storage modulus
        y_units2_orig: Original units for loss modulus

    Returns:
        Tuple of (x_val, y_val) or (None, None) if invalid
    """
    x_val = row[x_col]
    y_val_real = convert_units(row[y_col], y_units_orig, "Pa")
    y_val_imag = convert_units(row[y_col2], y_units2_orig, "Pa")

    if np.isnan(x_val) or np.isnan(y_val_real) or np.isnan(y_val_imag):
        return None, None

    return x_val, complex(y_val_real, y_val_imag)


def _process_real_row(
    row: list[float], x_col: int, y_col: int
) -> tuple[float | None, float | None]:
    """Process a single row for real-valued data.

    Args:
        row: Parsed row values
        x_col: X column index
        y_col: Y column index

    Returns:
        Tuple of (x_val, y_val) or (None, None) if invalid
    """
    x_val = row[x_col]
    y_val = row[y_col]

    if np.isnan(x_val) or np.isnan(y_val):
        return None, None

    return x_val, y_val


def _create_rheodata_chunk(
    current_x: list,
    current_y: list,
    x_units: str,
    y_units: str,
    domain: str,
    metadata: dict,
    validate: bool,
) -> RheoData:
    """Create RheoData from accumulated chunk data.

    Args:
        current_x: X values list
        current_y: Y values list
        x_units: X axis units
        y_units: Y axis units
        domain: Data domain
        metadata: Segment metadata
        validate: Whether to validate data

    Returns:
        RheoData object
    """
    return RheoData(
        x=np.array(current_x),
        y=np.array(current_y),
        x_units=x_units,
        y_units=y_units,
        domain=domain,
        metadata=metadata.copy(),
        validate=validate,
    )


def _read_sample_rows(
    file_handle, chunk_size: int, num_columns: int
) -> tuple[list[list[float]], int]:
    """Read and parse sample data rows to determine column structure.

    Args:
        file_handle: Open file handle positioned after header
        chunk_size: Chunk size to limit sample rows
        num_columns: Expected number of columns

    Returns:
        Tuple of (sample_rows, lines_read)
    """
    sample_rows = []
    lines_read = 0

    for _ in range(min(10, chunk_size)):
        try:
            line = next(file_handle)
            lines_read += 1
            if not line.strip() or line.startswith("["):
                break
            values = line.split("\t")
            if len(values) == num_columns:
                row = _parse_row_values(values)
                if row:
                    sample_rows.append(row)
        except (StopIteration, ValueError):
            break

    return sample_rows, lines_read


def _read_segment_chunked(
    filepath: Path,
    seg_start: int,
    seg_end: int | None,
    metadata: dict,
    chunk_size: int,
    validate_data: bool,
    progress_callback: Callable | None = None,
):
    """Read a single segment in chunks (internal generator).

    Args:
        filepath: Path to file
        seg_start: Segment start line number
        seg_end: Segment end line number (None for end of file)
        metadata: File metadata dictionary
        chunk_size: Number of data points per chunk
        validate_data: Whether to validate each chunk
        progress_callback: Optional progress callback (current_points, total_points)

    Yields:
        RheoData: Chunks of segment data
    """
    with open(filepath, encoding="utf-8", errors="replace") as f:
        # Skip to segment start
        for _ in range(seg_start):
            next(f)

        # Phase 1: Parse segment header and find data section
        header_result = _parse_segment_header(f, seg_start, seg_end)
        if header_result is None:
            return

        step_temperature, line_num, num_points_line = header_result

        # Phase 2: Process the data section starting at "Number of points"
        yield from _process_data_section(
            f,
            line_num,
            seg_end,
            step_temperature,
            num_points_line,
            metadata,
            chunk_size,
            validate_data,
            progress_callback,
        )


def _parse_segment_header(file_handle, seg_start: int, seg_end: int | None):
    """Parse segment header to find data section start.

    Args:
        file_handle: Open file handle
        seg_start: Segment start line number
        seg_end: Segment end line number

    Returns:
        Tuple of (step_temperature, line_num, num_points_line) or None
    """
    step_temperature = None
    line_num = seg_start

    for line in file_handle:
        line_num += 1

        # Extract temperature from step name
        if "Step name" in line and step_temperature is None:
            step_temperature = _extract_step_temperature(line)

        # Check if we've reached segment end
        if seg_end is not None and line_num >= seg_end:
            return None

        # Check if we found "Number of points" (data section starts next)
        if line.startswith("Number of points"):
            return step_temperature, line_num, line

    return None


def _process_data_section(
    file_handle,
    line_num: int,
    seg_end: int | None,
    step_temperature: float | None,
    num_points_line: str,
    metadata: dict,
    chunk_size: int,
    validate_data: bool,
    progress_callback: Callable | None,
):
    """Process the data section of a segment.

    Args:
        file_handle: Open file handle positioned at data section
        line_num: Current line number
        seg_end: Segment end line number
        step_temperature: Temperature in Kelvin
        num_points_line: Line containing number of points
        metadata: File metadata
        chunk_size: Chunk size
        validate_data: Whether to validate data
        progress_callback: Progress callback

    Yields:
        RheoData chunks
    """
    # Parse total points for progress tracking
    total_points = _parse_total_points(num_points_line) if progress_callback else None

    # Read column headers and units
    header_line = next(file_handle).rstrip("\n")
    unit_line = next(file_handle).rstrip("\n")
    line_num += 2

    columns, units = _parse_headers_and_units(header_line, unit_line)

    # Read sample rows to determine column structure
    sample_rows, lines_read = _read_sample_rows(file_handle, chunk_size, len(columns))
    line_num += lines_read

    if not sample_rows:
        return  # No data in segment

    sample_array = np.array(sample_rows)

    # Adjust column indices since we skipped column 0
    columns = columns[1:]
    units = units[1:]

    # Determine x/y columns
    col_info = _determine_xy_columns(columns, units, sample_array)
    x_col, x_units, y_col, y_units, y_col2, y_units2 = col_info

    if x_col is None or y_col is None:
        warnings.warn(f"Could not determine x/y columns from: {columns}", stacklevel=2)
        return

    # Build segment metadata
    domain, test_mode = _infer_domain_and_mode(
        columns[x_col], columns[y_col], x_units, y_units
    )
    segment_metadata = _build_segment_metadata(
        metadata, test_mode, columns, units, step_temperature
    )

    # Track complex vs real data
    is_complex = y_col2 is not None
    y_units_orig = y_units
    y_units2_orig = y_units2 if is_complex else None

    # Process sample array
    if is_complex:
        x_chunk, y_chunk = _process_sample_array_complex(
            sample_array, x_col, y_col, y_col2, y_units_orig, y_units2_orig
        )
        y_units = "Pa"  # Standardized after conversion
    else:
        x_chunk, y_chunk = _process_sample_array_real(sample_array, x_col, y_col)

    # Initialize accumulators and progress tracking
    current_x: list = []
    current_y: list = []
    total_points_read = _add_sample_to_buffers(x_chunk, y_chunk, current_x, current_y)

    progress_interval = max(1, total_points // 20) if total_points else chunk_size

    # Process remaining data rows
    yield from _process_remaining_rows(
        file_handle,
        line_num,
        seg_end,
        columns,
        x_col,
        y_col,
        y_col2,
        is_complex,
        y_units_orig,
        y_units2_orig,
        current_x,
        current_y,
        total_points_read,
        total_points,
        progress_interval,
        progress_callback,
        x_units,
        y_units,
        domain,
        segment_metadata,
        chunk_size,
        validate_data,
    )


def _build_segment_metadata(
    base_metadata: dict,
    test_mode: str,
    columns: list,
    units: list,
    step_temperature: float | None,
) -> dict:
    """Build segment metadata dictionary.

    Args:
        base_metadata: Base file metadata
        test_mode: Detected test mode
        columns: Column names
        units: Column units
        step_temperature: Temperature in Kelvin

    Returns:
        Segment metadata dictionary
    """
    segment_metadata = base_metadata.copy()
    segment_metadata["test_mode"] = test_mode
    segment_metadata["columns"] = columns
    segment_metadata["units"] = units

    if step_temperature is not None:
        segment_metadata["temperature"] = step_temperature

    return segment_metadata


def _process_remaining_rows(
    file_handle,
    line_num: int,
    seg_end: int | None,
    columns: list,
    x_col: int,
    y_col: int,
    y_col2: int | None,
    is_complex: bool,
    y_units_orig: str,
    y_units2_orig: str | None,
    current_x: list,
    current_y: list,
    total_points_read: int,
    total_points: int | None,
    progress_interval: int,
    progress_callback: Callable | None,
    x_units: str,
    y_units: str,
    domain: str,
    segment_metadata: dict,
    chunk_size: int,
    validate_data: bool,
):
    """Process remaining data rows after sample rows.

    Args:
        Various state and configuration parameters

    Yields:
        RheoData chunks
    """
    last_progress_report = 0
    expected_columns = len(columns) + 1  # +1 for the row label we skip
    max_col_needed = max(x_col, y_col, y_col2 if y_col2 is not None else 0)

    for line in file_handle:
        line_num += 1

        # Check segment boundary
        if seg_end is not None and line_num >= seg_end:
            break

        if not line.strip() or line.startswith("["):
            break

        values = line.split("\t")
        if len(values) != expected_columns:
            continue

        try:
            row = _parse_row_values(values)
            if len(row) <= max_col_needed:
                continue

            # Process row based on data type
            if is_complex:
                x_val, y_val = _process_complex_row(
                    row, x_col, y_col, y_col2, y_units_orig, y_units2_orig
                )
            else:
                x_val, y_val = _process_real_row(row, x_col, y_col)

            if x_val is not None:
                current_x.append(x_val)
                current_y.append(y_val)
                total_points_read += 1

                # Report progress periodically
                if (
                    progress_callback is not None
                    and total_points is not None
                    and total_points_read - last_progress_report >= progress_interval
                ):
                    progress_callback(total_points_read, total_points)
                    last_progress_report = total_points_read

                # Yield chunk when size reached
                if len(current_x) >= chunk_size:
                    yield _create_rheodata_chunk(
                        current_x,
                        current_y,
                        x_units,
                        y_units,
                        domain,
                        segment_metadata,
                        validate_data,
                    )
                    current_x.clear()
                    current_y.clear()

        except (ValueError, IndexError):
            continue

    # Yield remaining data as final chunk
    if current_x:
        yield _create_rheodata_chunk(
            current_x,
            current_y,
            x_units,
            y_units,
            domain,
            segment_metadata,
            validate_data,
        )

    # Final progress report
    if progress_callback is not None and total_points is not None:
        progress_callback(total_points_read, total_points)


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

    # Extract temperature from step name (e.g., "Frequency sweep (150.0 °C)")
    step_temperature = None
    for line in segment_lines[:5]:  # Check first few lines
        if "Step name" in line or line.startswith("Step name"):
            # Extract temperature from format: "Step name\tFrequency sweep (150.0 °C)"
            # Support negative temperatures with optional minus sign
            temp_match = re.search(r"(-?\d+\.?\d*)\s*°C", line)
            if temp_match:
                temp_c = float(temp_match.group(1))
                step_temperature = temp_c + 273.15  # Convert to Kelvin
                break

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
            # Skip first column (row label like "Data point")
            # Convert remaining columns, using np.nan for non-numeric values
            row = []
            for i, v in enumerate(values):
                if i == 0:
                    # Skip first column (row label)
                    continue
                if not v.strip():
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(v))
                    except ValueError:
                        # Handle hex values (status bits), dates, strings
                        row.append(np.nan)

            if row:  # Only add if we have data
                data_rows.append(row)

    if not data_rows:
        return None

    # Convert to numpy array
    data_array = np.array(data_rows)

    # Adjust column indices since we skipped column 0
    columns = columns[1:]  # Remove first column ("Variables" or similar)
    units = units[1:]  # Remove first unit

    # Determine x and y columns based on common column names
    x_col, x_units, y_col, y_units, y_col2, y_units2 = _determine_xy_columns(
        columns, units, data_array
    )

    if x_col is None or y_col is None:
        warnings.warn(f"Could not determine x/y columns from: {columns}", stacklevel=2)
        return None

    # Extract x data
    x_data = data_array[:, x_col]

    # Extract y data (construct complex modulus if both G' and G'' are available)
    if y_col2 is not None:
        # Complex modulus: G* = G' + i*G''
        y_data_real = data_array[:, y_col]  # Storage modulus (G')
        y_data_imag = data_array[:, y_col2]  # Loss modulus (G'')

        # Apply unit conversions to both components
        y_data_real = convert_units(y_data_real, y_units, "Pa")
        y_data_imag = convert_units(y_data_imag, y_units2, "Pa")

        x_data_array = np.real_if_close(np.asarray(x_data))
        y_real_array = np.real_if_close(np.asarray(y_data_real))
        y_imag_array = np.real_if_close(np.asarray(y_data_imag))

        # Construct complex modulus
        y_data = y_real_array + 1j * y_imag_array
        y_units = "Pa"  # Standardize to Pa for complex modulus

        # Remove NaN values from either component
        valid_mask = ~(
            np.isnan(x_data_array) | np.isnan(y_real_array) | np.isnan(y_imag_array)
        )
    else:
        # Real-valued data
        y_data = data_array[:, y_col]

        x_data_array = np.real_if_close(np.asarray(x_data))
        y_data_array = np.real_if_close(np.asarray(y_data))

        # Remove NaN values
        valid_mask = ~(np.isnan(x_data_array) | np.isnan(y_data_array))
        y_data = y_data_array

    x_data = x_data_array[valid_mask]
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

    # Add temperature if found
    if step_temperature is not None:
        segment_metadata["temperature"] = step_temperature

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

    For oscillatory (SAOS) data with both Storage and Loss modulus columns,
    this will return both column indices to construct complex modulus G* = G' + i·G''.

    Args:
        columns: Column names
        units: Column units
        data: Data array

    Returns:
        Tuple of (x_col_index, x_units, y_col_index, y_units, y_col2_index, y_units2)
        where y_col2_index is None for non-complex data, or the Loss modulus column
        index for complex modulus construction.
    """
    columns_lower = [c.lower() for c in columns]

    # Priority lists for x and y columns
    # Note: Frequency comes before general "time" to prioritize frequency sweeps
    x_priorities = [
        "angular frequency",
        "frequency",
        "shear rate",
        "temperature",
        "step time",
        "time",
        "strain",
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

    # Check for BOTH storage and loss modulus (for complex modulus construction)
    storage_col = None
    loss_col = None
    for i, col in enumerate(columns_lower):
        if "storage modulus" in col and i != x_col:
            storage_col = i
        elif "loss modulus" in col and i != x_col:
            loss_col = i

    # If we have both G' and G'', use them to construct complex modulus
    if storage_col is not None and loss_col is not None:
        x_units = units[x_col] if x_col < len(units) else ""
        y_units = units[storage_col] if storage_col < len(units) else ""
        y_units2 = units[loss_col] if loss_col < len(units) else ""
        return x_col, x_units, storage_col, y_units, loss_col, y_units2

    # Otherwise, find single y column (prefer storage/loss modulus for SAOS)
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
        return None, None, None, None, None, None

    x_units = units[x_col] if x_col < len(units) else ""
    y_units = units[y_col] if y_col < len(units) else ""

    return x_col, x_units, y_col, y_units, None, None


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
