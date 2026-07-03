from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QToolBar,
    QWidget,
)

from rheojax.gui.foundation.notifier import DatasetLibraryNotifier
from rheojax.gui.foundation.state import AppState
from rheojax.gui.services.pipeline_execution_service import PipelineExecutionService
from rheojax.gui.workspace.fit.fit_controller import build_fit_controller
from rheojax.gui.workspace.inspector import InspectorPanel
from rheojax.gui.workspace.library_rail import LibraryRail
from rheojax.gui.workspace.stepper_canvas import StepperCanvas
from rheojax.gui.workspace.transform.transform_controller import (
    build_transform_controller,
)


class WorkspaceWindow(QMainWindow):
    mode_changed = Signal(str)
    MODES = ("fit", "transform", "pipeline")

    def __init__(self, app_state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._epoch = 0
        self._notifier = DatasetLibraryNotifier()
        self._notifier.changed.connect(self._mark_dirty)
        self._pipeline_service = PipelineExecutionService()
        self._pipeline_stop_event: threading.Event | None = None
        self._active_jobs_action_pending = False
        self._pipeline_service.phase_worker_ready.connect(
            self._on_phase_worker_ready, Qt.ConnectionType.QueuedConnection
        )

        self.setWindowTitle("RheoJAX Workspace")
        self.resize(1200, 800)

        bar = QToolBar(self)
        self.addToolBar(bar)
        self._fit_btn = QPushButton("Fit", self)
        self._tx_btn = QPushButton("Transform", self)
        self._pipeline_btn = QPushButton("Pipeline", self)
        self._fit_btn.setCheckable(True)
        self._tx_btn.setCheckable(True)
        self._pipeline_btn.setCheckable(True)
        self._fit_btn.clicked.connect(lambda: self.set_mode("fit"))
        self._tx_btn.clicked.connect(lambda: self.set_mode("transform"))
        self._pipeline_btn.clicked.connect(lambda: self.set_mode("pipeline"))
        bar.addWidget(self._fit_btn)
        bar.addWidget(self._tx_btn)
        bar.addWidget(self._pipeline_btn)
        self._status_label = QLabel(self)
        bar.addWidget(self._status_label)
        self._build_file_menu()

        self._build_workspace(app_state)

    def _build_file_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        menu.addAction("&New", self._on_new, "Ctrl+N")
        menu.addAction("&Open...", self._on_open, "Ctrl+O")
        menu.addAction("&Save", self._on_save, "Ctrl+S")
        menu.addAction("Save &As...", self._on_save_as, "Ctrl+Shift+S")
        menu.addAction("&Close", self._on_close)

    def _build_workspace(self, state: AppState) -> None:
        self._state = state
        initial_mode = state.ui.mode
        self._mode = initial_mode if initial_mode in self.MODES else "fit"
        from rheojax.gui.workspace.pipeline.controller import build_pipeline_controller

        fit_ctl, self._fit_bodies = build_fit_controller(state)
        transform_ctl, self._transform_bodies = build_transform_controller(state)
        pipeline_ctl, self._pipeline_bodies = build_pipeline_controller(
            state, self._pipeline_service, epoch=self._epoch, guard=self._guard
        )
        self._controllers = {
            "fit": fit_ctl,
            "transform": transform_ctl,
            "pipeline": pipeline_ctl,
        }
        self._body_lists = {
            "fit": self._fit_bodies,
            "transform": self._transform_bodies,
            "pipeline": self._pipeline_bodies,
        }
        # Auto-advance each workflow whenever a step's edits make it ready.
        # Connected after build_*_controller so each body's own edited ->
        # invalidation wiring (inside build_*_controller) has already run by
        # the time these handlers inspect state.
        for body in self._fit_bodies:
            if hasattr(body, "edited"):
                body.edited.connect(self._on_fit_body_edited)
            if hasattr(body, "config_edited"):
                body.config_edited.connect(self._on_fit_body_edited)
            if hasattr(body, "dataset_commit_requested"):
                body.dataset_commit_requested.connect(self._commit_dataset)
        for body in self._transform_bodies:
            if hasattr(body, "edited"):
                body.edited.connect(self._on_transform_body_edited)
            if hasattr(body, "dataset_commit_requested"):
                body.dataset_commit_requested.connect(self._commit_dataset)
        for body in self._pipeline_bodies:
            if hasattr(body, "edited"):
                body.edited.connect(self._on_pipeline_body_edited)
            if hasattr(body, "run_requested"):
                body.run_requested.connect(self._on_pipeline_run_requested)

        self._sync_mode_buttons()
        self._refresh_status_label()

        self._rail = LibraryRail(state.library, self)
        self._inspector = InspectorPanel(self)
        self._canvas = StepperCanvas(self._controllers[self._mode], self)
        self._wire_canvas(self._canvas)
        self._install_bodies(self._mode, self._canvas)
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(self._rail)
        self._splitter.addWidget(self._canvas)
        self._splitter.addWidget(self._inspector)
        self.setCentralWidget(self._splitter)

    def _dispose_workspace(self) -> None:
        for body in list(self._fit_bodies) + list(self._transform_bodies) + list(self._pipeline_bodies):
            if hasattr(body, "edited"):
                try:
                    body.edited.disconnect(self._on_fit_body_edited)
                except (RuntimeError, TypeError):
                    pass
                try:
                    body.edited.disconnect(self._on_transform_body_edited)
                except (RuntimeError, TypeError):
                    pass
                try:
                    body.edited.disconnect(self._on_pipeline_body_edited)
                except (RuntimeError, TypeError):
                    pass
            if hasattr(body, "config_edited"):
                try:
                    body.config_edited.disconnect(self._on_fit_body_edited)
                except (RuntimeError, TypeError):
                    pass
            if hasattr(body, "dataset_commit_requested"):
                try:
                    body.dataset_commit_requested.disconnect(self._commit_dataset)
                except (RuntimeError, TypeError):
                    pass
            if hasattr(body, "run_requested"):
                try:
                    body.run_requested.disconnect(self._on_pipeline_run_requested)
                except (RuntimeError, TypeError):
                    pass
            body.deleteLater()
        for widget in (self._canvas, self._rail, self._inspector, self._splitter):
            widget.deleteLater()
        self._controllers = {}
        self._body_lists = {}
        self._fit_bodies = []
        self._transform_bodies = []
        self._pipeline_bodies = []

    def _rebuild(self, new_state: AppState) -> None:
        self._epoch += 1
        self._dispose_workspace()
        self._build_workspace(new_state)

    def _guard(self, epoch_at_connect: int, fn):
        def wrapped(*args, **kwargs):
            if epoch_at_connect == self._epoch:
                return fn(*args, **kwargs)
            return None

        return wrapped

    def _on_new(self) -> None:
        self._maybe_confirm_active_jobs(
            lambda: self._maybe_confirm_unsaved_changes(lambda: self._rebuild(AppState()))
        )

    def _on_open(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        def _open():
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Project", "", "RheoJAX Project (*.rheojax)"
            )
            if path:
                from zipfile import BadZipFile

                from rheojax.gui.foundation.project_codec import load_project_v2

                try:
                    new_state = load_project_v2(path)
                except (ValueError, FileNotFoundError, OSError, BadZipFile) as exc:
                    QMessageBox.critical(self, "Open Failed", str(exc))
                    return
                self._rebuild(new_state)
                self._state.project.dirty = False
                self._state.project.path = path

        self._maybe_confirm_active_jobs(
            lambda: self._maybe_confirm_unsaved_changes(_open)
        )

    def _on_save(self) -> None:
        if self._state.project.path is None:
            self._on_save_as()
            return
        from PySide6.QtWidgets import QMessageBox

        from rheojax.gui.foundation.project_codec import save_project_v2

        try:
            save_project_v2(self._state, self._state.project.path)
        except (ValueError, FileNotFoundError, OSError) as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return
        self._state.project.dirty = False

    def _on_save_as(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "RheoJAX Project (*.rheojax)"
        )
        if path:
            from rheojax.gui.foundation.project_codec import save_project_v2

            try:
                save_project_v2(self._state, path)
            except (ValueError, FileNotFoundError, OSError) as exc:
                QMessageBox.critical(self, "Save Failed", str(exc))
                return
            self._state.project.path = path
            self._state.project.name = Path(path).stem
            self._state.project.dirty = False

    def _on_close(self) -> None:
        self._maybe_confirm_active_jobs(
            lambda: self._maybe_confirm_unsaved_changes(lambda: self._rebuild(AppState()))
        )

    def _maybe_confirm_active_jobs(self, proceed) -> None:
        if not self._state.active_jobs.by_id:
            proceed()
            return
        if self._active_jobs_action_pending:
            # A Close/New/Open request is already being handled (dialog shown or poll in
            # flight) -- ignore this second trigger rather than showing a second dialog or
            # starting a second, independent polling chain that could both call proceed().
            return
        self._active_jobs_action_pending = True

        from PySide6.QtWidgets import QMessageBox

        choice = QMessageBox.question(
            self, "Jobs Running",
            f"{len(self._state.active_jobs.by_id)} job(s) still running. Cancel them and continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice != QMessageBox.StandardButton.Yes:
            self._active_jobs_action_pending = False
            return

        from PySide6.QtCore import QThreadPool

        from rheojax.gui.workspace.pipeline.cancel_runnable import CancelWorkerRunnable

        if self._pipeline_stop_event is not None:
            self._pipeline_stop_event.set()  # stops PipelineBatchRunner from advancing to
                                               # further queued datasets (Task 7's run() loop)
        for job in self._state.active_jobs.by_id.values():
            worker = job.get("worker")
            if worker is not None:
                QThreadPool.globalInstance().start(CancelWorkerRunnable(worker))

        self._poll_active_jobs_then(proceed, remaining_polls=120)  # 120 * 250ms = 30s

    def _poll_active_jobs_then(self, proceed, remaining_polls: int) -> None:
        if not self._state.active_jobs.by_id or remaining_polls <= 0:
            self._active_jobs_action_pending = False
            proceed()
            return
        from PySide6.QtCore import QTimer

        QTimer.singleShot(250, lambda: self._poll_active_jobs_then(proceed, remaining_polls - 1))

    def _on_phase_worker_ready(self, dataset_id: str, step_id: str, phase: str, worker) -> None:
        job = self._state.active_jobs.by_id.get(dataset_id)
        if job is not None:
            job["worker"] = worker

    def _maybe_confirm_unsaved_changes(self, proceed) -> None:
        if not self._state.project.dirty:
            proceed()
            return
        from PySide6.QtWidgets import QMessageBox

        choice = QMessageBox.question(
            self,
            "Unsaved Changes",
            "Save changes before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Save:
            self._on_save()
            if not self._state.project.dirty:
                proceed()
        elif choice == QMessageBox.StandardButton.Discard:
            proceed()

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {self.MODES}")
        if mode == self._mode:
            return
        old_bodies = self._body_lists[self._mode]
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
        self._state.ui.mode = mode
        self._sync_mode_buttons()
        self.mode_changed.emit(mode)

    def _mark_dirty(self) -> None:
        self._state.project.dirty = True

    def _commit_dataset(self, ref, payload=None, overwrite: bool = False) -> None:
        self._state.library.add(ref, overwrite=overwrite)
        if payload is not None:
            self._state.library.store_payload(ref.id, payload)
        self._notifier.changed.emit()

    def _sync_mode_buttons(self) -> None:
        self._fit_btn.setChecked(self._mode == "fit")
        self._tx_btn.setChecked(self._mode == "transform")
        self._pipeline_btn.setChecked(self._mode == "pipeline")

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
        bodies = self._body_lists[mode]
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
        while (
            i + 1 < len(ctl.steps)
            and ctl.steps[i].is_ready()
            and ctl.steps[i].validate()
            and (i + 1) not in ctl.reached
        ):
            ctl.reached.add(i + 1)
            i += 1
        if self._mode == mode:
            self._canvas.refresh()

    def _on_fit_body_edited(self) -> None:
        self._advance_and_unlock("fit")

    def _on_transform_body_edited(self) -> None:
        self._advance_and_unlock("transform")

    def _on_pipeline_body_edited(self) -> None:
        pass  # mirrors _on_fit_body_edited/_on_transform_body_edited's advance-eligibility recheck

    def _on_pipeline_run_requested(self) -> None:
        if self._state.active_jobs.by_id:
            # A batch is already running -- ignore a repeat "Run All" trigger rather than
            # starting a second PipelineBatchRunner, which would orphan the first runner's
            # stop event and race active_jobs/DatasetLibrary from two worker threads.
            return

        from PySide6.QtCore import QThreadPool

        from rheojax.gui.workspace.pipeline.batch_runner import PipelineBatchRunner

        pipeline_state = self._state.pipeline
        self._pipeline_stop_event = threading.Event()
        runner = PipelineBatchRunner(
            service=self._pipeline_service, steps=pipeline_state.steps,
            selected_dataset_ids=pipeline_state.selected_dataset_ids, library=self._state.library,
            stop_requested=self._pipeline_stop_event,
        )
        QThreadPool.globalInstance().start(runner)
