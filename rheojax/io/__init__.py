"""Unified file I/O for rheological data.

This module provides readers and writers for various data formats:
- TA Instruments TRIOS
- Anton Paar
- CSV, Excel
- HDF5 for large datasets
- Automatic format detection
- SPP analysis export (MATLAB-compatible)
"""

from rheojax.io import readers, writers

# Import commonly used functions for convenience
from rheojax.io.readers import (
    auto_load,
    load_anton_paar,
    load_csv,
    load_excel,
    load_trios,
)
from rheojax.io.writers import (
    load_hdf5,
    save_excel,
    save_hdf5,
)

# SPP export functions
from rheojax.io.spp_export import (
    export_spp_csv,
    export_spp_hdf5,
    export_spp_txt,
    to_matlab_dict,
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
    # SPP Export
    "export_spp_txt",
    "export_spp_hdf5",
    "export_spp_csv",
    "to_matlab_dict",
]
