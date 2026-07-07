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


from rheojax.core.data import RheoData
from rheojax.gui.dialogs.dataset_preview import DatasetPreviewDialog
from rheojax.gui.foundation.library import DatasetRef


def _ref(protocol_type="oscillation", units=None, row_count=0):
    return DatasetRef(
        id="d1",
        name="d1",
        protocol_type=protocol_type,
        origin="imported",
        units=units or {},
        row_count=row_count,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_dialog_is_non_modal(qtbot):
    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    assert dialog.isModal() is False


def test_table_view_resizes_columns_to_contents(qtbot):
    from PySide6.QtWidgets import QHeaderView

    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    mode = dialog._table_view.horizontalHeader().sectionResizeMode(0)
    assert mode == QHeaderView.ResizeMode.ResizeToContents


def test_set_dataset_shows_table_and_hides_no_data_label(qtbot):
    # isVisible() reflects the whole ancestor chain, not just this widget's
    # own setVisible() flag -- the dialog itself must actually be shown for
    # these assertions to mean anything (qtbot.addWidget() alone doesn't).
    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    data = RheoData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), validate=False)

    dialog.set_dataset(_ref(), data, [])

    assert dialog._no_data_label.isVisible() is False
    assert dialog._table_view.isVisible() is True
    assert dialog._model.rowCount() == 2


def test_set_dataset_with_none_shows_no_data_label(qtbot):
    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.set_dataset(_ref(row_count=7), None, [])

    assert dialog._no_data_label.isVisible() is True
    assert dialog._table_view.isVisible() is False
    assert dialog._model.rowCount() == 0
    # Summary header must stay populated/visible in the no-data state too --
    # only the warnings/table region should change.
    assert dialog._name_label.text() == "d1"
    assert dialog._protocol_label.text() == "oscillation"
    assert dialog._origin_label.text() == "imported"
    assert dialog._domain_label.text() == "unknown"
    assert dialog._row_count_label.text() == "7"


def test_set_dataset_uses_len_x_for_row_count_not_ref_row_count(qtbot):
    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    data = RheoData(
        x=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        validate=False,
    )

    dialog.set_dataset(_ref(row_count=0), data, [])

    assert dialog._row_count_label.text() == "5"


def test_set_dataset_with_simplenamespace_derived_payload(qtbot):
    from types import SimpleNamespace

    from rheojax.gui.utils.rheodata import rheodata_from_any

    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    payload = SimpleNamespace(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]))
    data = rheodata_from_any(payload).to_numpy()

    dialog.set_dataset(_ref(row_count=0), data, [])

    assert dialog._domain_label.text() == "time"
    assert dialog._row_count_label.text() == "2"


def test_warnings_banner_replaces_labels_on_reuse(qtbot):
    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    data = RheoData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), validate=False)

    dialog.set_dataset(_ref(), data, ["one", "two", "three"])
    assert dialog._warnings_layout.count() == 3

    dialog.set_dataset(_ref(), data, ["only one now"])
    assert dialog._warnings_layout.count() == 1
    assert dialog._warnings_layout.itemAt(0).widget().text() == "only one now"


def test_warnings_container_hidden_when_no_warnings(qtbot):
    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    data = RheoData(x=np.array([1.0]), y=np.array([2.0]), validate=False)

    dialog.set_dataset(_ref(), data, [])

    assert dialog._warnings_container.isVisible() is False


def test_set_dataset_reuse_updates_columns_and_values(qtbot):
    # Spec-level reuse test through the dialog's own set_dataset(), which
    # also recomputes headers/visibility -- a superset of Task 1's raw
    # model-level reuse test.
    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    complex_data = RheoData(
        x=np.array([1.0, 2.0]), y=np.array([1.0 + 1.0j, 2.0 + 2.0j]), validate=False
    )
    dialog.set_dataset(_ref(protocol_type="oscillation"), complex_data, [])
    assert dialog._model.columnCount() == 3

    real_data = RheoData(x=np.array([9.0]), y=np.array([42.0]), validate=False)
    dialog.set_dataset(_ref(protocol_type="creep"), real_data, [])

    assert dialog._model.columnCount() == 2
    assert dialog._model.rowCount() == 1
    assert dialog._model.data(dialog._model.index(0, 1)) == f"{42.0:.6g}"


def test_column_headers_include_units(qtbot):
    from PySide6.QtCore import Qt as QtQt

    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    data = RheoData(
        x=np.array([1.0, 2.0]),
        y=np.array([3.0, 4.0]),
        x_units="s",
        y_units="Pa",
        validate=False,
    )

    dialog.set_dataset(_ref(units={"x": "s", "y": "Pa"}), data, [])

    assert dialog._model.headerData(0, QtQt.Orientation.Horizontal) == "x [s]"
    assert dialog._model.headerData(1, QtQt.Orientation.Horizontal) == "y [Pa]"


def test_complex_columns_labeled_gprime_for_oscillation_protocol(qtbot):
    from PySide6.QtCore import Qt as QtQt

    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    data = RheoData(x=np.array([1.0]), y=np.array([1.0 + 2.0j]), validate=False)

    dialog.set_dataset(_ref(protocol_type="oscillation"), data, [])

    assert dialog._model.headerData(1, QtQt.Orientation.Horizontal) == "G'"
    assert dialog._model.headerData(2, QtQt.Orientation.Horizontal) == "G''"


def test_complex_columns_labeled_re_im_for_non_oscillation_protocol(qtbot):
    from PySide6.QtCore import Qt as QtQt

    dialog = DatasetPreviewDialog()
    qtbot.addWidget(dialog)
    data = RheoData(x=np.array([1.0]), y=np.array([1.0 + 2.0j]), validate=False)

    dialog.set_dataset(_ref(protocol_type="creep"), data, [])

    assert dialog._model.headerData(1, QtQt.Orientation.Horizontal) == "Re(y)"
    assert dialog._model.headerData(2, QtQt.Orientation.Horizontal) == "Im(y)"
