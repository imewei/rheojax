"""File readers for rheological data formats.

This module provides readers for various instrument data formats:
- TA Instruments TRIOS (.txt)
- CSV/TSV files
- Excel files (.xlsx, .xls)
- Anton Paar files
- Auto-detection wrapper
"""

from rheo.io.readers.trios import load_trios
from rheo.io.readers.csv_reader import load_csv
from rheo.io.readers.excel_reader import load_excel
from rheo.io.readers.anton_paar import load_anton_paar
from rheo.io.readers.auto import auto_load

__all__ = [
    'load_trios',
    'load_csv',
    'load_excel',
    'load_anton_paar',
    'auto_load',
]