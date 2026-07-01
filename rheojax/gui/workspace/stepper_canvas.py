from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.workspace.controller import WorkflowController


class StepperCanvas(QWidget):
    step_clicked = Signal(int)

    def __init__(self, controller: WorkflowController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctl = controller
        self._rail = QHBoxLayout()
        self._buttons: list[QPushButton] = []
        self._stack = QStackedWidget(self)
        for i, step in enumerate(controller.steps):
            b = QPushButton(f"{i + 1} {step.title}", self)
            b.clicked.connect(lambda _=False, idx=i: self.step_clicked.emit(idx))
            self._buttons.append(b)
            self._rail.addWidget(b)
            self._stack.addWidget(QWidget(self))  # placeholder body
        lay = QVBoxLayout(self)
        lay.addLayout(self._rail)
        lay.addWidget(self._stack)
        self.refresh()

    def set_body(self, index: int, widget: QWidget) -> None:
        old = self._stack.widget(index)
        self._stack.insertWidget(index, widget)
        self._stack.removeWidget(old)

    def is_enabled(self, index: int) -> bool:
        return self._buttons[index].isEnabled()

    def click_step(self, index: int) -> None:
        self._buttons[index].click()

    def refresh(self) -> None:
        for i, b in enumerate(self._buttons):
            b.setEnabled(i in self._ctl.reached)
        self._stack.setCurrentIndex(self._ctl.current)
