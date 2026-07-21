from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from rheojax.gui.resources.styles.tokens import themed
from rheojax.gui.utils.layout_helpers import set_zero_margins


class InspectorPanel(QWidget):
    # ponytail: set_tab_widget() has no caller anywhere in the app, so all
    # three tabs stay empty placeholders. Populating them for real requires
    # designing what each tab shows per active mode/step (a params view
    # synced to the current FitState/TransformState, a priors editor, a log
    # viewer) -- a feature to design deliberately, not a one-line wiring fix.
    # Each placeholder is an explicit "not connected yet" label (muted,
    # italic) rather than a bare blank QWidget, so an empty tab reads as
    # unfinished rather than broken.
    _TABS = ["params", "priors", "log"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tabs = QTabWidget(self)
        self._index: dict[str, int] = {}
        for name in self._TABS:
            self._index[name] = self._tabs.addTab(
                self._placeholder(f"{name.capitalize()} view not yet connected"), name
            )
        lay = QVBoxLayout(self)
        set_zero_margins(lay)
        lay.addWidget(self._tabs)
        # The parent splitter has no explicit setSizes(), so it squeezes this
        # panel to whatever space is left over -- without a floor, the tab
        # bar's own labels (e.g. "log") get pushed behind the overflow
        # scroll button. Bound to the tab bar's current sizeHint rather than
        # a fixed pixel value so it stays correct at any font size/DPI.
        self.setMinimumWidth(self._tabs.tabBar().sizeHint().width())

    def _placeholder(self, text: str) -> QWidget:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"QLabel {{ color: {themed('TEXT_MUTED')}; font-style: italic; }}")
        wrapper = QWidget(self)
        lay = QVBoxLayout(wrapper)
        lay.addWidget(label)
        return wrapper

    def tab_names(self) -> list[str]:
        return list(self._TABS)

    def set_tab_widget(self, name: str, widget: QWidget) -> None:
        idx = self._index[name]
        old = self._tabs.widget(idx)
        was_current = self._tabs.currentIndex() == idx
        self._tabs.removeTab(idx)
        self._tabs.insertTab(idx, widget, name)
        if was_current:
            self._tabs.setCurrentIndex(idx)
        if old is not None:
            old.deleteLater()

    def show_tab(self, name: str) -> None:
        self._tabs.setCurrentIndex(self._index[name])

    def current_tab(self) -> str:
        idx = self._tabs.currentIndex()
        return self._TABS[idx] if 0 <= idx < len(self._TABS) else ""
