from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QToolBar,
    QWidget,
)

from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.fit.fit_controller import build_fit_controller
from rheojax.gui.workspace.inspector import InspectorPanel
from rheojax.gui.workspace.library_rail import LibraryRail
from rheojax.gui.workspace.stepper_canvas import StepperCanvas
from rheojax.gui.workspace.transform.transform_controller import (
    build_transform_controller,
)


class WorkspaceWindow(QMainWindow):
    mode_changed = Signal(str)
    MODES = ("fit", "transform")

    def __init__(self, app_state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = app_state
        initial_mode = app_state.ui.get("mode", "fit")
        self._mode = initial_mode if initial_mode in self.MODES else "fit"
        fit_ctl, self._fit_bodies = build_fit_controller(app_state)
        transform_ctl, self._transform_bodies = build_transform_controller(app_state)
        self._controllers = {
            "fit": fit_ctl,
            "transform": transform_ctl,
        }
        # Auto-advance each workflow whenever a step's edits make it ready.
        # Connected after build_*_controller so each body's own edited ->
        # invalidation wiring (inside build_*_controller) has already run by
        # the time these handlers inspect state.
        for body in self._fit_bodies:
            if hasattr(body, "edited"):
                body.edited.connect(self._on_fit_body_edited)
        for body in self._transform_bodies:
            if hasattr(body, "edited"):
                body.edited.connect(self._on_transform_body_edited)

        self.setWindowTitle("RheoJAX Workspace")
        self.resize(1200, 800)

        bar = QToolBar(self)
        self.addToolBar(bar)
        self._fit_btn = QPushButton("Fit", self)
        self._tx_btn = QPushButton("Transform", self)
        self._fit_btn.setCheckable(True)
        self._tx_btn.setCheckable(True)
        self._fit_btn.clicked.connect(lambda: self.set_mode("fit"))
        self._tx_btn.clicked.connect(lambda: self.set_mode("transform"))
        bar.addWidget(self._fit_btn)
        bar.addWidget(self._tx_btn)
        self._status_label = QLabel(self)
        bar.addWidget(self._status_label)
        self._sync_mode_buttons()
        self._refresh_status_label()

        self._rail = LibraryRail(app_state.library, self)
        self._inspector = InspectorPanel(self)
        self._canvas = StepperCanvas(self._controllers[self._mode], self)
        self._wire_canvas(self._canvas)
        self._install_bodies(self._mode, self._canvas)
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(self._rail)
        self._splitter.addWidget(self._canvas)
        self._splitter.addWidget(self._inspector)
        self.setCentralWidget(self._splitter)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {self.MODES}")
        if mode == self._mode:
            return
        old_bodies = self._fit_bodies if self._mode == "fit" else self._transform_bodies
        self._mode = mode
        new_canvas = StepperCanvas(self._controllers[mode], self)
        self._wire_canvas(new_canvas)
        self._install_bodies(mode, new_canvas)
        old_canvas = self._splitter.replaceWidget(1, new_canvas)
        # Both fit and transform bodies are persistent, stateful widgets
        # owned by their controller (not disposable skeleton stubs) — pull
        # them out of the outgoing canvas before it's deleted, or
        # deleteLater() would cascade-delete them along with their
        # QStackedWidget parent.
        for body in old_bodies:
            body.setParent(self)
            body.hide()
        if old_canvas is not None:
            old_canvas.deleteLater()
        self._canvas = new_canvas
        self._state.ui["mode"] = mode
        self._sync_mode_buttons()
        self.mode_changed.emit(mode)

    def _sync_mode_buttons(self) -> None:
        self._fit_btn.setChecked(self._mode == "fit")
        self._tx_btn.setChecked(self._mode == "transform")

    def _refresh_status_label(self) -> None:
        # ponytail: "active project" status deferred — needs the Plan 3/4
        # AppState.project slice, which doesn't exist yet (spec §4.3).
        try:
            from rheojax.gui.utils.jax_utils import get_jax_info

            info = get_jax_info()
            fp = "float64 ✓" if info.get("float64_enabled") else "float64 ✗"
            device = info.get("default_device", "?")
            self._status_label.setText(f"{fp}  |  {device}")
        except Exception:
            self._status_label.setText("")

    def fit_bodies(self) -> list[QWidget]:
        return self._fit_bodies

    def transform_bodies(self) -> list[QWidget]:
        return self._transform_bodies

    def _install_bodies(self, mode: str, canvas: StepperCanvas) -> None:
        bodies = self._fit_bodies if mode == "fit" else self._transform_bodies
        for i, body in enumerate(bodies):
            canvas.set_body(i, body)

    def active_step_count(self) -> int:
        return len(self._controllers[self._mode].steps)

    def current_step(self) -> int:
        return self._controllers[self._mode].current

    def _wire_canvas(self, canvas: StepperCanvas) -> None:
        canvas.step_clicked.connect(self._on_step_clicked)

    def _on_step_clicked(self, index: int) -> None:
        if self._controllers[self._mode].goto(index):
            self._canvas.refresh()

    def _advance_and_unlock(self, mode: str) -> None:
        # Shared by both workflows: auto-advance the controller once its
        # current step becomes ready, then walk forward unlocking (but not
        # navigating past) any trailing steps whose predecessor is trivially
        # ready. Read-only tail steps (Visualize/Export-only bodies) have no
        # `edited` signal, so nothing else ever re-checks whether the final
        # step has become reachable once the user is already sitting on the
        # step just before it -- arriving there doesn't itself fire this
        # handler again. See test_export_step_unlocked_once_visualize_reached
        # (fit) and test_window_transform_export_step_reachable_without_
        # forced_navigation (transform) for the regression this guards.
        ctl = self._controllers[mode]
        if ctl.can_advance():
            ctl.advance()
        i = ctl.current
        while i + 1 < len(ctl.steps) and ctl.steps[i].is_ready() and (i + 1) not in ctl.reached:
            ctl.reached.add(i + 1)
            i += 1
        if self._mode == mode:
            self._canvas.refresh()

    def _on_fit_body_edited(self) -> None:
        self._advance_and_unlock("fit")

    def _on_transform_body_edited(self) -> None:
        self._advance_and_unlock("transform")
