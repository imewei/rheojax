from __future__ import annotations

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget


class InspectorPanel(QWidget):
    _TABS = ["params", "priors", "log"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tabs = QTabWidget(self)
        self._index: dict[str, int] = {}
        for name in self._TABS:
            self._index[name] = self._tabs.addTab(QWidget(self), name)
        QVBoxLayout(self).addWidget(self._tabs)

    def tab_names(self) -> list[str]:
        return list(self._TABS)

    def set_tab_widget(self, name: str, widget: QWidget) -> None:
        idx = self._index[name]
        self._tabs.removeTab(idx)
        self._tabs.insertTab(idx, widget, name)

    def show_tab(self, name: str) -> None:
        self._tabs.setCurrentIndex(self._index[name])

    def current_tab(self) -> str:
        return self._TABS[self._tabs.currentIndex()]
