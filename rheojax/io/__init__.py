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
from rheojax.io._exceptions import RheoJaxFormatError, RheoJaxValidationWarning

# Import commonly used functions for convenience
from rheojax.io.readers import (
    auto_load,
    load_anton_paar,
    load_csv,
    load_excel,
    load_series,
    load_srfs,
    load_trios,
    load_tts,
)
from rheojax.io.readers._validation import validate_protocol

# SPP export functions
from rheojax.io.spp_export import (
    export_spp_csv,
    export_spp_hdf5,
    export_spp_txt,
    to_matlab_dict,
)
from rheojax.io.writers import (
    load_hdf5,
    load_npz,
    save_excel,
    save_hdf5,
    save_npz,
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
    # Multi-file loaders
    "load_tts",
    "load_srfs",
    "load_series",
    # Writers
    "save_hdf5",
    "load_hdf5",
    "save_excel",
    "save_npz",
    "load_npz",
    # SPP Export
    "export_spp_txt",
    "export_spp_hdf5",
    "export_spp_csv",
    "to_matlab_dict",
    # Exceptions and warnings
    "RheoJaxFormatError",
    "RheoJaxValidationWarning",
    # Validation
    "validate_protocol",
]
