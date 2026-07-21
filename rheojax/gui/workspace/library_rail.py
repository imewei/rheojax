from __future__ import annotations

from PySide6.QtCore import Qt, Signal, SignalInstance
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.resources.styles.tokens import section_header_style, themed
from rheojax.gui.utils.layout_helpers import set_panel_margins


class LibraryRail(QWidget):
    dataset_selected = Signal(str)
    dataset_preview_requested = Signal(str)
    dataset_delete_requested = Signal(str)
    import_requested = Signal()

    def __init__(self, library: DatasetLibrary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = library
        self._header = QLabel("Datasets", self)
        self._header.setStyleSheet(section_header_style())
        self._list = QListWidget(self)
        self._import = QPushButton("+ Import data…", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        lay.addWidget(self._header)
        lay.addWidget(self._list)
        lay.addWidget(self._import)
        self._list.itemClicked.connect(
            lambda it: self._emit_if_dataset(self.dataset_selected, it)
        )
        self._list.itemDoubleClicked.connect(
            lambda it: self._emit_if_dataset(self.dataset_preview_requested, it)
        )
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu_requested)
        self._import.clicked.connect(self.import_requested.emit)
        self.refresh()

    def refresh(self) -> None:
        self._list.clear()
        refs = self._library.all()
        if not refs:
            placeholder = QListWidgetItem("No datasets yet — import a file to begin")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            # Visually distinguish the empty-state row from real dataset rows
            # (muted color + italic) -- previously identical styling made it
            # look like a selectable/real entry rather than a status message.
            placeholder.setForeground(QColor(themed("TEXT_MUTED")))
            font = placeholder.font()
            font.setItalic(True)
            placeholder.setFont(font)
            self._list.addItem(placeholder)
            return
        for ref in refs:
            label = f"{ref.name}   [{ref.protocol_type}]"
            it = QListWidgetItem(label)
            # QListWidget does not elide item text; a long dataset name (common
            # for TRIOS exports) is otherwise clipped with no way to recover it.
            it.setToolTip(label)
            it.setData(Qt.ItemDataRole.UserRole, ref.id)
            self._list.addItem(it)

    def _emit_if_dataset(self, signal: SignalInstance, item: QListWidgetItem) -> None:
        # The empty-state placeholder item has no UserRole data; skip it so
        # clicking/double-clicking the "No datasets yet" row is a no-op.
        dataset_id = item.data(Qt.ItemDataRole.UserRole)
        if dataset_id is not None:
            signal.emit(dataset_id)

    def _on_context_menu_requested(self, pos) -> None:
        item = self._list.itemAt(pos)
        # itemAt() is a pure geometry hit-test and ignores item flags, so the
        # empty-state placeholder (NoItemFlags, no UserRole data) still gets
        # returned here; treat it the same as "no item under the cursor" so
        # we don't open a context menu with actions that have no target.
        if item is None or item.data(Qt.ItemDataRole.UserRole) is None:
            return
        menu = QMenu(self)
        preview_action = menu.addAction("Preview…")
        delete_action = menu.addAction("Delete…")
        chosen = self._exec_context_menu(menu, self._list.mapToGlobal(pos))
        if chosen is preview_action:
            self.dataset_preview_requested.emit(item.data(Qt.ItemDataRole.UserRole))
        elif chosen is delete_action:
            self.dataset_delete_requested.emit(item.data(Qt.ItemDataRole.UserRole))

    def _exec_context_menu(self, menu: QMenu, global_pos):
        # ponytail: seam for tests to monkeypatch instead of QMenu.exec directly --
        # PySide6 6.11's bound-method lookup for QMenu.exec bypasses class-level
        # Python monkeypatching, so patching Qt's own class silently falls through
        # to the real (blocking) exec(). Patching this plain-Python method instead
        # behaves like normal Python attribute lookup.
        return menu.exec(global_pos)

    def count(self) -> int:
        # self._list.count() is the raw QListWidget row count, which includes
        # the "No datasets yet" placeholder refresh() adds when the library is
        # empty -- that placeholder made count() return 1, never 0, for an
        # empty library. Count real datasets instead.
        return len(self._library.all())

    def select_row(self, row: int) -> None:
        self._list.setCurrentRow(row)
        item = self._list.item(row)
        if item is None:
            return
        self.dataset_selected.emit(item.data(Qt.ItemDataRole.UserRole))
