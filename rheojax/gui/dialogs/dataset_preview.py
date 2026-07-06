"""
Dataset Preview Dialog
=======================

Non-modal dialog for inspecting a dataset's raw table, summary, and
data-quality warnings from the workspace LibraryRail.
"""

from __future__ import annotations

import numpy as np

from rheojax.core.data import RheoData
from rheojax.gui.compat import (
    QDialog,
    QFormLayout,
    QHeaderView,
    QLabel,
    Qt,
    QtCore,
    QtWidgets,
    QVBoxLayout,
    QWidget,
)
from rheojax.gui.foundation.library import DatasetRef


class RheoDataTableModel(QtCore.QAbstractTableModel):
    """Read-only table model over a dataset's x/y arrays.

    `y` may be real-valued (one data column) or complex-valued (two
    columns: real then imaginary). Which is which, and what the columns
    are labeled, is entirely the caller's decision (see `set_data`) --
    this model only decides *how many cells* to read off `y`, based on
    `np.iscomplexobj(y)`.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._headers: list[str] = []

    def set_data(
        self,
        x: np.ndarray | None,
        y: np.ndarray | None,
        x_header: str,
        y_headers: list[str],
    ) -> None:
        """Replace the model's data, safely notifying attached views.

        Pass `x=None, y=None, x_header="", y_headers=[]` to reset the
        model to an empty (0-row, 0-column) state.
        """
        self.beginResetModel()
        self._x = x
        self._y = y
        self._headers = [x_header, *y_headers] if x is not None else []
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return 0 if self._x is None else len(self._x)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._x is None:
            return None
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        col = index.column()
        row = index.row()
        if col == 0:
            value = self._x[row]
        elif np.iscomplexobj(self._y):
            value = self._y[row].real if col == 1 else self._y[row].imag
        else:
            value = self._y[row]
        try:
            return f"{value:.6g}"
        except (TypeError, ValueError):
            # Non-numeric values can slip past shape-only validation
            # upstream -- render as plain text instead of crashing the view.
            return str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation != Qt.Orientation.Horizontal:
            return None
        if section < 0 or section >= len(self._headers):
            return None
        return self._headers[section]


class DatasetPreviewDialog(QDialog):
    """Non-modal preview of a single dataset's table, summary, and warnings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset Preview")
        self.setModal(False)

        self._name_label = QLabel(self)
        self._protocol_label = QLabel(self)
        self._origin_label = QLabel(self)
        self._domain_label = QLabel(self)
        self._row_count_label = QLabel(self)
        summary_form = QFormLayout()
        summary_form.addRow("Name:", self._name_label)
        summary_form.addRow("Protocol:", self._protocol_label)
        summary_form.addRow("Origin:", self._origin_label)
        summary_form.addRow("Domain:", self._domain_label)
        summary_form.addRow("Rows:", self._row_count_label)

        self._no_data_label = QLabel("No data available for this dataset", self)

        self._warnings_layout = QVBoxLayout()
        self._warnings_container = QWidget(self)
        self._warnings_container.setLayout(self._warnings_layout)

        self._model = RheoDataTableModel(self)
        self._table_view = QtWidgets.QTableView(self)
        self._table_view.setModel(self._model)
        self._table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        # ponytail: PySide6 6.11's QHeaderView.sectionResizeMode(0) reports
        # Fixed instead of the mode set above when the header has zero
        # sections (i.e. before the first set_dataset() call). Force one
        # placeholder section into existence so the mode sticks immediately;
        # set_dataset()/the no-data path reset it back to empty as normal.
        # Until then, columnCount()==1 with headerData(0)=="" (not the true
        # 0-column x=None state) — required for the workaround, not a bug.
        self._model.set_data(np.array([]), np.array([]), "", [])

        layout = QVBoxLayout(self)
        layout.addLayout(summary_form)
        layout.addWidget(self._no_data_label)
        layout.addWidget(self._warnings_container)
        layout.addWidget(self._table_view)

    def set_dataset(
        self, ref: DatasetRef, data: RheoData | None, warnings: list[str]
    ) -> None:
        """Populate the dialog for ref/data (data is None for the no-data state)."""
        self._name_label.setText(ref.name)
        self._protocol_label.setText(ref.protocol_type or "(none)")
        self._origin_label.setText(ref.origin)

        has_data = data is not None
        self._domain_label.setText(data.domain if has_data else "unknown")
        self._row_count_label.setText(str(len(data.x) if has_data else ref.row_count))

        self._no_data_label.setVisible(not has_data)
        self._warnings_container.setVisible(has_data and bool(warnings))
        self._table_view.setVisible(has_data)

        self._set_warnings(warnings if has_data else [])

        if not has_data:
            self._model.set_data(None, None, "", [])
            return

        is_complex = np.iscomplexobj(data.y)
        x_unit = ref.units.get("x") or data.x_units
        y_unit = ref.units.get("y") or data.y_units
        x_header = f"x [{x_unit}]" if x_unit else "x"
        if is_complex:
            if ref.protocol_type == "oscillation":
                y_names = ["G'", "G''"]
            else:
                y_names = ["Re(y)", "Im(y)"]
        else:
            y_names = ["y"]
        y_headers = [f"{name} [{y_unit}]" if y_unit else name for name in y_names]
        self._model.set_data(data.x, data.y, x_header, y_headers)

    def _set_warnings(self, warnings: list[str]) -> None:
        while self._warnings_layout.count():
            item = self._warnings_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for message in warnings:
            self._warnings_layout.addWidget(QLabel(message, self._warnings_container))
