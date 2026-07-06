"""
Dataset Preview Dialog
=======================

Non-modal dialog for inspecting a dataset's raw table, summary, and
data-quality warnings from the workspace LibraryRail.
"""

from __future__ import annotations

import numpy as np

from rheojax.gui.compat import Qt, QtCore


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
