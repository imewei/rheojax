import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget

from rheojax.gui.workspace.controller import Step, WorkflowController
from rheojax.gui.workspace.stepper_canvas import StepperCanvas


def _ctl():
    steps = [
        Step(id=str(i), title=str(i), is_ready=(lambda: True), validate=(lambda: True))
        for i in range(2)
    ]
    return WorkflowController(steps)


def test_size_hint_tracks_current_page_only(qapp):
    # A QStackedWidget sizes itself to the largest of ALL pages by default
    # (Qt's documented behavior), so a single oversized step would forever
    # inflate the window's minimum size -- even while a small step is shown.
    canvas = StepperCanvas(_ctl(), None)

    small = QWidget()
    small.setMinimumSize(100, 100)
    tall = QWidget()
    tall.setMinimumSize(100, 900)

    canvas.set_body(0, small)
    canvas.set_body(1, tall)
    canvas._stack.setCurrentIndex(0)

    assert canvas._stack.minimumSizeHint().height() == 100

    canvas._stack.setCurrentIndex(1)
    assert canvas._stack.minimumSizeHint().height() == 900

    canvas._stack.setCurrentIndex(0)
    assert canvas._stack.minimumSizeHint().height() == 100
