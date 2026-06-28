# tests/io/test_dmta_rejection.py
import h5py
import numpy as np
import pandas as pd
import pytest

from rheojax.io._exceptions import UnsupportedDataError
from rheojax.io.readers.auto import auto_load


def test_tensile_columns_rejected(tmp_path):
    f = tmp_path / "dma.csv"
    pd.DataFrame(
        {
            "frequency": [1.0, 2, 3],  # 'frequency' (auto-detected), not 'freq'
            "Tensile Storage Modulus": [1e6, 2e6, 3e6],
            "Tensile Loss Modulus": [1e5, 2e5, 3e5],
        }
    ).to_csv(f, index=False)
    with pytest.raises(UnsupportedDataError):
        auto_load(str(f))


def test_hdf5_tensile_read_rejected(tmp_path):
    f = tmp_path / "t.h5"
    with h5py.File(f, "w") as h:
        h.create_dataset("x", data=np.array([1.0, 2, 3]))
        h.create_dataset("y", data=np.array([1.0, 2, 3]))
        h.attrs["measurement_geometry"] = "tension"
    from rheojax.io.writers.hdf5_writer import load_hdf5

    with pytest.raises(UnsupportedDataError):
        load_hdf5(str(f))
