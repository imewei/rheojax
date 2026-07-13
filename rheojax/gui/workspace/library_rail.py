from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.utils.layout_helpers import set_panel_margins


class LibraryRail(QWidget):
    dataset_selected = Signal(str)
    dataset_preview_requested = Signal(str)
    dataset_delete_requested = Signal(str)
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
        self._list.itemDoubleClicked.connect(
            lambda it: self.dataset_preview_requested.emit(
                it.data(Qt.ItemDataRole.UserRole)
            )
        )
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu_requested)
        self._import.clicked.connect(self.import_requested.emit)
        self.refresh()

    def refresh(self) -> None:
        self._list.clear()
        for ref in self._library.all():
            it = QListWidgetItem(f"{ref.name}   [{ref.protocol_type}]")
            it.setData(Qt.ItemDataRole.UserRole, ref.id)
            self._list.addItem(it)

    def _on_context_menu_requested(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        preview_action = menu.addAction("Preview…")
        delete_action = menu.addAction("Delete…")
        # ponytail: call via QMenu.exec(menu, ...) rather than menu.exec(...) --
        # functionally identical, but PySide6 6.11's instance attribute lookup
        # bypasses Python-level monkeypatching of QMenu.exec (class-level
        # override only takes effect through the unbound call). This keeps
        # QMenu.exec mockable in tests without changing runtime behavior.
        chosen = QMenu.exec(menu, self._list.mapToGlobal(pos))
        if chosen is preview_action:
            self.dataset_preview_requested.emit(item.data(Qt.ItemDataRole.UserRole))
        elif chosen is delete_action:
            self.dataset_delete_requested.emit(item.data(Qt.ItemDataRole.UserRole))

    def count(self) -> int:
        return self._list.count()

    def select_row(self, row: int) -> None:
        self._list.setCurrentRow(row)
        item = self._list.item(row)
        if item is None:
            return
        self.dataset_selected.emit(item.data(Qt.ItemDataRole.UserRole))
