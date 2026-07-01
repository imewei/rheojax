from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QSplitter,
    QToolBar,
    QWidget,
)

from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.controller import FitController, Step, TransformController
from rheojax.gui.workspace.inspector import InspectorPanel
from rheojax.gui.workspace.library_rail import LibraryRail
from rheojax.gui.workspace.stepper_canvas import StepperCanvas


def _skeleton_steps(ids: list[str]) -> list[Step]:
    # ponytail: trivially-ready stubs; Plans 3/4 replace bodies + real is_ready
    return [
        Step(id=i, title=i.replace("_", " ").title(), is_ready=lambda: True, validate=lambda: True)
        for i in ids
    ]


class WorkspaceWindow(QMainWindow):
    mode_changed = Signal(str)

    def __init__(self, app_state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = app_state
        self._mode = "fit"
        self._controllers = {
            "fit": FitController(_skeleton_steps(FitController.STEP_IDS)),
            "transform": TransformController(_skeleton_steps(TransformController.STEP_IDS)),
        }

        bar = QToolBar(self)
        self.addToolBar(bar)
        self._fit_btn = QPushButton("Fit", self)
        self._tx_btn = QPushButton("Transform", self)
        self._fit_btn.clicked.connect(lambda: self.set_mode("fit"))
        self._tx_btn.clicked.connect(lambda: self.set_mode("transform"))
        bar.addWidget(self._fit_btn)
        bar.addWidget(self._tx_btn)

        self._rail = LibraryRail(app_state.library, self)
        self._inspector = InspectorPanel(self)
        self._canvas = StepperCanvas(self._controllers["fit"], self)
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(self._rail)
        self._splitter.addWidget(self._canvas)
        self._splitter.addWidget(self._inspector)
        self.setCentralWidget(self._splitter)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        new_canvas = StepperCanvas(self._controllers[mode], self)
        self._splitter.replaceWidget(1, new_canvas)
        self._canvas = new_canvas
        self._state.ui["mode"] = mode
        self.mode_changed.emit(mode)

    def active_step_count(self) -> int:
        return len(self._controllers[self._mode].steps)
