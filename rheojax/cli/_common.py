"""Shared helpers for CLI subcommands.

load_and_flatten() was independently reimplemented (with small wording and
behavior drift) in fit.py, bayesian.py, cmd_load.py, cmd_transform.py,
spp.py, and cmd_batch.py. This module is the single copy the single-file
subcommands (including spp.py's two auto_load call sites) route through;
cmd_batch.py keeps its own variant since batch processing needs per-file
logging rather than a printed stderr note.
"""

from __future__ import annotations

import sys

from rheojax.core.data import RheoData
from rheojax.logging import get_logger

logger = get_logger(__name__)


def load_and_flatten(
    input_path: str,
    x_col: str | None = None,
    y_col: str | None = None,
    y_cols: str | None = None,
    file_format: str | None = None,
    warn_on_multi_segment: bool = True,
) -> RheoData:
    """Load a file via auto_load() and flatten multi-segment results.

    Args:
        input_path: Path to the data file.
        x_col: X column name (for CSV/Excel).
        y_col: Y column name (for CSV/Excel, single-column data).
        y_cols: Comma-separated Y column names (for oscillation data).
        file_format: Explicit format override for auto_load (e.g. "csv",
            "trios"). None lets auto_load detect it.
        warn_on_multi_segment: Print a stderr note when a multi-segment file
            is flattened to its first segment.

    Returns:
        The loaded RheoData (first segment, if the file contained several).

    Raises:
        ValueError: If the file contains zero data segments.
    """
    from rheojax.io import auto_load

    load_kwargs: dict = {}
    if x_col is not None:
        load_kwargs["x_col"] = x_col
    if y_col is not None:
        load_kwargs["y_col"] = y_col
    if y_cols is not None:
        load_kwargs["y_cols"] = [c.strip() for c in y_cols.split(",")]
    if file_format is not None:
        load_kwargs["format"] = file_format

    data = auto_load(input_path, **load_kwargs)

    if isinstance(data, list):
        if not data:
            raise ValueError("File contained no data segments")
        if len(data) > 1:
            if warn_on_multi_segment:
                print(
                    f"Note: File contains {len(data)} segments, using first segment",
                    file=sys.stderr,
                )
            logger.warning(
                "Multi-segment file: using first segment",
                input_file=input_path,
                n_segments=len(data),
            )
        data = data[0]

    return data
