from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.utils.layout_helpers import set_toolbar_margins, set_zero_margins
from rheojax.gui.workspace.controller import WorkflowController


class StepperCanvas(QWidget):
    step_clicked = Signal(int)

    def __init__(
        self, controller: WorkflowController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._ctl = controller
        self._rail = QHBoxLayout()
        self._buttons: list[QPushButton] = []
        self._stack = QStackedWidget(self)
        for i, step in enumerate(controller.steps):
            b = QPushButton(f"{i + 1} {step.title}", self)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, idx=i: self.step_clicked.emit(idx))
            self._buttons.append(b)
            self._rail.addWidget(b)
            self._stack.addWidget(QWidget(self))  # placeholder body
        set_toolbar_margins(self._rail)
        lay = QVBoxLayout(self)
        set_zero_margins(lay)
        lay.addLayout(self._rail)
        lay.addWidget(self._stack)
        self.refresh()

    def set_body(self, index: int, widget: QWidget) -> None:
        old = self._stack.widget(index)
        self._stack.insertWidget(index, widget)
        self._stack.removeWidget(old)
        if old is not None:
            old.deleteLater()

    def is_enabled(self, index: int) -> bool:
        return self._buttons[index].isEnabled()

    def is_active(self, index: int) -> bool:
        return self._buttons[index].isChecked()

    def current_index(self) -> int:
        return self._stack.currentIndex()

    def click_step(self, index: int) -> None:
        self._buttons[index].click()

    def refresh(self) -> None:
        for i, b in enumerate(self._buttons):
            b.setEnabled(i in self._ctl.reached)
            b.setChecked(i == self._ctl.current)
        self._stack.setCurrentIndex(self._ctl.current)
