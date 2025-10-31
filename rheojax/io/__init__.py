"""Unified file I/O for rheological data.

This module provides readers and writers for various data formats:
- TA Instruments TRIOS
- Anton Paar
- CSV, Excel
- HDF5 for large datasets
- Automatic format detection
"""

from rheo.io import readers, writers

# Import commonly used functions for convenience
from rheo.io.readers import (
    auto_load,
    load_anton_paar,
    load_csv,
    load_excel,
    load_trios,
)
from rheo.io.writers import (
    load_hdf5,
    save_excel,
    save_hdf5,
)

__all__ = [
    "readers",
    "writers",
    # Readers
    "load_trios",
    "load_csv",
    "load_excel",
    "load_anton_paar",
    "auto_load",
    # Writers
    "save_hdf5",
    "load_hdf5",
    "save_excel",
]
