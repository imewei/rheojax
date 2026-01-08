"""File readers for rheological data formats.

This module provides readers for various instrument data formats:
- TA Instruments TRIOS (.txt, .csv, .xlsx, .json)
- CSV/TSV files
- Excel files (.xlsx, .xls)
- Anton Paar RheoCompass files
- Auto-detection wrapper
"""

from rheojax.io.readers.anton_paar import (
    IntervalBlock,
    load_anton_paar,
    parse_rheocompass_intervals,
    save_intervals_to_excel,
)
from rheojax.io.readers.auto import auto_load
from rheojax.io.readers.csv_reader import load_csv
from rheojax.io.readers.excel_reader import load_excel

# TRIOS multi-format support
from rheojax.io.readers.trios import (
    TRIOSFile,
    TRIOSTable,
    load_trios,
    load_trios_chunked,
    load_trios_csv,
    load_trios_excel,
    load_trios_json,
    parse_trios_csv,
)

__all__ = [
    # TRIOS loaders
    "load_trios",
    "load_trios_chunked",
    "load_trios_csv",
    "load_trios_excel",
    "load_trios_json",
    "parse_trios_csv",
    "TRIOSFile",
    "TRIOSTable",
    # Generic loaders
    "load_csv",
    "load_excel",
    # Anton Paar
    "load_anton_paar",
    "parse_rheocompass_intervals",
    "save_intervals_to_excel",
    "IntervalBlock",
    # Auto-detection
    "auto_load",
]
