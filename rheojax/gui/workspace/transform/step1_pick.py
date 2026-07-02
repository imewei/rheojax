from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

import rheojax.transforms  # noqa: F401
from rheojax.core.registry import TransformRegistry
from rheojax.gui.foundation.state import TransformState


class TransformPickStep(QWidget):
    edited = Signal()

    def __init__(self, state: TransformState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._list = QListWidget(self)
        for cat, keys in self.groups().items():
            for k in keys:
                it = QListWidgetItem(f"{k}   [{cat}]")
                it.setData(Qt.ItemDataRole.UserRole, k)
                self._list.addItem(it)
        QVBoxLayout(self).addWidget(self._list)
        self._list.itemClicked.connect(lambda it: self._select(it.data(Qt.ItemDataRole.UserRole)))

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key in sorted(TransformRegistry.list_transforms()):
            info = TransformRegistry.get_info(key)
            if info is None:
                continue  # registry entry pending — skip safely
            cat = str(info.transform_type).split(".")[-1].lower()
            out.setdefault(cat, []).append(key)
        return out

    def set_transform(self, key: str) -> None:
        self._select(key)

    def _select(self, key: str) -> None:
        self._state.transform_key = key
        self.edited.emit()

    def is_ready(self) -> bool:
        return self._state.transform_key is not None
