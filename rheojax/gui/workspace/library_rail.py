from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.utils.layout_helpers import set_panel_margins


class LibraryRail(QWidget):
    dataset_selected = Signal(str)
    import_requested = Signal()

    def __init__(self, library: DatasetLibrary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = library
        self._list = QListWidget(self)
        self._import = QPushButton("+ Import data…", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        lay.addWidget(self._list)
        lay.addWidget(self._import)
        self._list.itemClicked.connect(
            lambda it: self.dataset_selected.emit(it.data(Qt.ItemDataRole.UserRole))
        )
        self._import.clicked.connect(self.import_requested.emit)
        self.refresh()

    def refresh(self) -> None:
        self._list.clear()
        for ref in self._library.all():
            it = QListWidgetItem(f"{ref.name}   [{ref.protocol_type}]")
            it.setData(Qt.ItemDataRole.UserRole, ref.id)
            self._list.addItem(it)

    def count(self) -> int:
        return self._list.count()

    def select_row(self, row: int) -> None:
        self._list.setCurrentRow(row)
        item = self._list.item(row)
        if item is None:
            return
        self.dataset_selected.emit(item.data(Qt.ItemDataRole.UserRole))
