# tests/io/test_dmta_rejection.py
import importlib
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest

from rheojax.io._exceptions import UnsupportedDataError
from rheojax.io.readers._utils import check_tensile_guard
from rheojax.io.readers.auto import auto_load
from rheojax.io.writers.hdf5_writer import load_hdf5


def _write_hdf5_with_geometry_marker(path, location, name, value):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.array([1.0, 2.0, 3.0]))
        handle.create_dataset("y", data=np.array([1.0, 2.0, 3.0]))
        attrs = handle.attrs
        if location == "metadata":
            attrs = handle.create_group("metadata").attrs
        attrs[name] = value


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
    _write_hdf5_with_geometry_marker(f, "top", "measurement_geometry", "tension")

    with pytest.raises(UnsupportedDataError):
        load_hdf5(str(f))


@pytest.mark.parametrize(
    ("location", "name", "value"),
    [
        ("top", "measurement_geometry", "  TeNsIoN  "),
        ("top", "deformation_mode", np.array([b"BENDING"])),
        ("metadata", "measurement_geometry", b" tensile "),
        ("metadata", "deformation_mode", np.array([b"CoMpReSsIoN"])),
    ],
)
def test_hdf5_encoded_tensile_markers_rejected(tmp_path, location, name, value):
    f = tmp_path / f"{location}_{name}.h5"
    _write_hdf5_with_geometry_marker(f, location, name, value)

    with pytest.raises(UnsupportedDataError, match="Unsupported measurement geometry"):
        load_hdf5(f)


@pytest.mark.parametrize("reader_name", ("csv_reader", "excel_reader"))
@pytest.mark.parametrize("removed_key", ("deformation_mode", "poisson_ratio"))
def test_reader_metadata_rejects_removed_keys_before_data_construction(
    tmp_path, reader_name, removed_key
):
    module = importlib.import_module(f"rheojax.io.readers.{reader_name}")
    if reader_name == "csv_reader":
        path = tmp_path / "shear.csv"
        pd.DataFrame({"time": [1.0, 2.0], "stress": [3.0, 4.0]}).to_csv(
            path, index=False
        )
        load = module.load_csv
    else:
        pytest.importorskip("openpyxl")
        path = tmp_path / "shear.xlsx"
        pd.DataFrame({"time": [1.0, 2.0], "stress": [3.0, 4.0]}).to_excel(
            path, index=False
        )
        load = module.load_excel

    with patch.object(module, "RheoData") as rheodata_constructor:
        with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
            load(
                path,
                x_col="time",
                y_col="stress",
                metadata={removed_key: "shear"},
            )

    rheodata_constructor.assert_not_called()


@pytest.mark.parametrize("location", ("top", "metadata"))
@pytest.mark.parametrize(
    ("removed_key", "value"),
    (("deformation_mode", "shear"), ("poisson_ratio", 0.5)),
)
def test_hdf5_removed_metadata_is_not_propagated(
    tmp_path, location, removed_key, value
):
    f = tmp_path / f"{location}_{removed_key}.h5"
    _write_hdf5_with_geometry_marker(f, location, removed_key, value)

    loaded = load_hdf5(f)

    assert removed_key not in loaded.metadata


def test_hdf5_irrelevant_numeric_and_array_geometry_metadata_loads(tmp_path):
    f = tmp_path / "numeric_geometry.h5"
    with h5py.File(f, "w") as handle:
        handle.create_dataset("x", data=np.array([1.0, 2.0, 3.0]))
        handle.create_dataset("y", data=np.array([1.0, 2.0, 3.0]))
        handle.attrs["measurement_geometry"] = 42
        metadata = handle.create_group("metadata")
        metadata.attrs["deformation_mode"] = np.array([1, 2])

    loaded = load_hdf5(f)
    assert len(loaded.x) == 3


def test_compression_guard_distinguishes_active_compression_from_e_comp():
    check_tensile_guard(["Mode ActiveCompression is selected"])

    with pytest.raises(UnsupportedDataError):
        check_tensile_guard(["E_comp"])
