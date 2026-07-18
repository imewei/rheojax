from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.utils.layout_helpers import set_toolbar_margins, set_zero_margins
from rheojax.gui.workspace.controller import WorkflowController


class _CurrentPageStack(QStackedWidget):
    """QStackedWidget whose size hints reflect only the current page.

    Qt's default QStackedWidget sizes itself to the largest of ALL pages
    it holds, even hidden ones (documented behavior), so the tallest step
    (e.g. NUTS diagnostics) permanently forces every other step's minimum
    window size -- on a real display this can exceed the screen's usable
    area and make the window impossible to maximize.
    """

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        current = self.currentWidget()
        if current is None:
            return super().sizeHint()
        # Match Qt's own QWidgetItem::sizeHint() formula (expandedTo minimumSize)
        # so this never reports a preferred size smaller than the minimum below.
        return current.sizeHint().expandedTo(current.minimumSize())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        current = self.currentWidget()
        if current is None:
            return super().minimumSizeHint()
        # Match Qt's own QWidgetItem::minimumSize() formula so a page that
        # calls setMinimumSize() directly (not just via child layout
        # constraints) is still honored.
        return current.minimumSizeHint().expandedTo(current.minimumSize())


class StepperCanvas(QWidget):
    step_clicked = Signal(int)

    def __init__(
        self, controller: WorkflowController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._ctl = controller
        self._rail = QHBoxLayout()
        self._buttons: list[QToolButton] = []
        self._stack = _CurrentPageStack(self)
        self._stack.currentChanged.connect(lambda _: self._stack.updateGeometry())
        # QToolButton (not QPushButton) for the step rail, same rationale as
        # WorkspaceWindow's mode switcher: base.qss's checked-pill toggle
        # style (light highlight, not a filled gradient) reads as "you are
        # here" progress-track navigation, matching what a wizard step rail
        # should look like rather than a row of primary-action buttons.
        for i, step in enumerate(controller.steps):
            b = QToolButton(self)
            b.setText(f"{i + 1} {step.title}")
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
        if index == self._stack.currentIndex():
            self._stack.updateGeometry()

    def is_enabled(self, index: int) -> bool:
        return self._buttons[index].isEnabled()

    def is_active(self, index: int) -> bool:
        return self._buttons[index].isChecked()

    def current_index(self) -> int:
        return self._stack.currentIndex()

    def click_step(self, index: int) -> None:
        self._buttons[index].click()

    def refresh(self) -> None:
        # Enabled/checked state alone doesn't tell a screen reader (or a
        # sighted user hovering before clicking) *why* a step is locked --
        # the per-button tooltip below keeps that in sync with actual state.
        for i, b in enumerate(self._buttons):
            reached = i in self._ctl.reached
            b.setEnabled(reached)
            b.setChecked(i == self._ctl.current)
            if i == self._ctl.current:
                b.setToolTip(f"Step {i + 1} of {len(self._buttons)} (current)")
            elif reached:
                b.setToolTip(f"Step {i + 1} of {len(self._buttons)} -- click to revisit")
            else:
                b.setToolTip("Complete the earlier steps to unlock this one")
        self._stack.setCurrentIndex(self._ctl.current)
