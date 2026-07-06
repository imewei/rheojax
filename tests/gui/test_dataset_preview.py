from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from rheojax.gui.dialogs.dataset_preview import RheoDataTableModel


def test_empty_model_reports_zero_rows_and_columns():
    model = RheoDataTableModel()
    assert model.rowCount() == 0
    assert model.columnCount() == 0


def test_real_valued_data_has_two_columns():
    model = RheoDataTableModel()
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    model.set_data(x, y, "x", ["y"])
    assert model.rowCount() == 3
    assert model.columnCount() == 2
    assert model.headerData(0, Qt.Orientation.Horizontal) == "x"
    assert model.headerData(1, Qt.Orientation.Horizontal) == "y"


def test_complex_valued_data_has_three_columns():
    model = RheoDataTableModel()
    x = np.array([1.0, 2.0])
    y = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    model.set_data(x, y, "x", ["G'", "G''"])
    assert model.columnCount() == 3
    assert model.headerData(1, Qt.Orientation.Horizontal) == "G'"
    assert model.headerData(2, Qt.Orientation.Horizontal) == "G''"


def test_cell_values_are_formatted_with_6g():
    model = RheoDataTableModel()
    x = np.array([0.00012345])
    y = np.array([98765.4321])
    model.set_data(x, y, "x", ["y"])
    assert model.data(model.index(0, 0)) == f"{0.00012345:.6g}"
    assert model.data(model.index(0, 1)) == f"{98765.4321:.6g}"


def test_complex_cell_splits_into_real_and_imag_columns():
    model = RheoDataTableModel()
    x = np.array([1.0])
    y = np.array([3.0 + 4.0j])
    model.set_data(x, y, "x", ["Re(y)", "Im(y)"])
    assert model.data(model.index(0, 1)) == f"{3.0:.6g}"
    assert model.data(model.index(0, 2)) == f"{4.0:.6g}"


def test_set_data_resets_model_shape_on_reuse():
    model = RheoDataTableModel()
    model.set_data(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j]),
        "x",
        ["G'", "G''"],
    )
    assert model.columnCount() == 3
    model.set_data(np.array([1.0]), np.array([5.0]), "x", ["y"])
    assert model.columnCount() == 2
    assert model.rowCount() == 1
    assert model.data(model.index(0, 1)) == f"{5.0:.6g}"


def test_set_data_none_resets_to_empty():
    model = RheoDataTableModel()
    model.set_data(np.array([1.0]), np.array([5.0]), "x", ["y"])
    model.set_data(None, None, "", [])
    assert model.rowCount() == 0
    assert model.columnCount() == 0


def test_data_falls_back_to_str_for_non_numeric_values():
    # Non-numeric (e.g. object-dtype/string) values can pass shape-only
    # validation upstream (Task 4's handler only checks ndim/length, not
    # dtype) -- the model must not crash Qt's paint/scroll machinery when
    # it eventually reads such a cell.
    model = RheoDataTableModel()
    x = np.array(["a", "b"], dtype=object)
    y = np.array(["c", "d"], dtype=object)
    model.set_data(x, y, "x", ["y"])
    assert model.data(model.index(0, 0)) == "a"
    assert model.data(model.index(0, 1)) == "c"
