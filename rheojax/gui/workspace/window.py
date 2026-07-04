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
            state, self._pipeline_service, epoch=self._epoch, guard=self._guard,
            notify=self._notifier.changed.emit,
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
        # LibraryRail otherwise only ever renders the snapshot present at construction
        # time -- nothing else calls .refresh() -- so without this, a dataset added by
        # any commit path (interactive export or a Pipeline job) never appears until the
        # next full _rebuild() (Open/New/Close).
        self._notifier.changed.connect(self._rail.refresh)
        self._rail.import_requested.connect(self._on_import_requested)
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
        try:
            self._notifier.changed.disconnect(self._rail.refresh)
        except (RuntimeError, TypeError):
            pass
        try:
            self._rail.import_requested.disconnect(self._on_import_requested)
        except (RuntimeError, TypeError):
            pass
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

    def _blocked_by_active_jobs(self, action: str) -> bool:
        # Spec §3.3: "Save snapshots job_history only; blocked while active_jobs is
        # non-empty" -- a running Pipeline batch mutates DatasetLibrary/job_history from
        # a worker thread, so a concurrent Save could serialize a torn, inconsistent
        # snapshot. Unlike Close/New/Open (which offer to cancel jobs and proceed), Save
        # is a hard block: there is no safe "discard the running job" option here.
        if not self._state.active_jobs.by_id:
            return False
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(
            self, "Jobs Running",
            f"Cannot {action} while {len(self._state.active_jobs.by_id)} job(s) are still "
            "running. Wait for them to finish, or cancel them via Close/New/Open first.",
        )
        return True

    def _on_save(self) -> None:
        if self._blocked_by_active_jobs("save"):
            return
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
        if self._blocked_by_active_jobs("save"):
            return
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
            if self._state.active_jobs.by_id:
                # Reachable when _maybe_confirm_active_jobs gave up waiting for jobs to
                # finish cancelling (its 30s poll timeout) with active_jobs still
                # non-empty -- Save is unconditionally blocked in that state
                # (_blocked_by_active_jobs), so calling it here would silently leave
                # `proceed()` never called with no indication that the Close/New/Open
                # the user already confirmed was itself aborted. Tell them directly
                # instead of showing only the generic "Cannot save" dialog.
                QMessageBox.warning(
                    self, "Cannot Save",
                    "Some jobs are still finishing cancellation, so the project can't "
                    "be saved yet. This action was cancelled -- try again shortly, or "
                    "choose Discard to proceed without saving.",
                )
                return
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

    # detect_test_mode() reports "flow" for flow-curve data (F-IO-R... naming
    # predates Protocol.FLOW_CURVE), which never equality-matches
    # DatasetLibrary.datasets_of_type("flow_curve") -- normalize it here so
    # imported flow-curve datasets actually show up in the Fit/Transform Data
    # step instead of silently landing in no protocol bucket at all.
    _IMPORT_TEST_MODE_ALIASES = {"flow": "flow_curve"}

    def _on_import_requested(self) -> None:
        """Handle LibraryRail's "+ Import data..." button."""
        from PySide6.QtWidgets import QFileDialog

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Data",
            "",
            "All Supported (*.csv *.txt *.xlsx *.xls *.tri *.dat *.json);;All Files (*.*)",
        )
        if not paths:
            return

        self._launch_import([Path(p) for p in paths])

    def _launch_import(
        self,
        file_paths: list[Path],
        x_col: str | None = None,
        y_col: str | None = None,
        y2_col: str | None = None,
        temp_col: str | None = None,
    ) -> None:
        from PySide6.QtCore import QThreadPool

        from rheojax.gui.jobs.import_worker import ImportWorker
        from rheojax.gui.services.data_service import DataService

        worker = ImportWorker(
            data_service=DataService(),
            file_path=file_paths[0],
            file_paths=file_paths,
            x_col=x_col,
            y_col=y_col,
            y2_col=y2_col,
            temp_col=temp_col,
        )
        # QueuedConnection guarantees the callback runs on the main thread --
        # ImportWorker emits its signals from a QThreadPool worker thread.
        worker.signals.completed.connect(
            self._on_import_completed, Qt.ConnectionType.QueuedConnection
        )
        # file_paths is bound per-connection (not read from shared instance
        # state) so that a second import launched before this worker's
        # "failed" signal is delivered can never make the failure handler
        # act on the wrong file's paths.
        worker.signals.failed.connect(
            lambda msg, fp=file_paths: self._on_import_failed(msg, fp),
            Qt.ConnectionType.QueuedConnection,
        )
        # Keep a reference alive for the worker's lifetime -- nothing else
        # holds the ImportWorker/ImportWorkerSignals QObject, so it would
        # otherwise be garbage-collected mid-run and drop the signal.
        self._active_import_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_import_completed(self, datasets: list) -> None:
        import hashlib
        import uuid

        from rheojax.gui.foundation.library import DatasetRef
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        hash_cache: dict[str, str] = {}
        # ponytail: multiple segments from one source file all get that
        # file's stem as their name -- add "_segment_N" suffixing if that
        # proves confusing in practice.
        for rheo_data in datasets:
            meta = rheo_data.metadata
            source_file = meta.pop("_source_file", None)
            test_mode = meta.get("test_mode") or service.detect_test_mode(rheo_data)
            test_mode = self._IMPORT_TEST_MODE_ALIASES.get(test_mode, test_mode)
            meta["test_mode"] = test_mode

            if source_file and source_file not in hash_cache:
                hash_cache[source_file] = hashlib.sha256(
                    Path(source_file).read_bytes()
                ).hexdigest()

            ref = DatasetRef(
                id=uuid.uuid4().hex,
                name=Path(source_file).stem if source_file else "imported",
                protocol_type=test_mode,
                origin="imported",
                units={
                    k: v
                    for k, v in (("x", rheo_data.x_units), ("y", rheo_data.y_units))
                    if v
                },
                row_count=len(rheo_data.x),
                hash=hash_cache.get(source_file, ""),
                provenance={"source": "gui_import", "path": source_file},
                lineage=[],
            )
            self._commit_dataset(ref, rheo_data, overwrite=False)

    # Extensions ColumnMapperDialog can actually parse (see its _load_data) --
    # anything else (.tri, .json, .dat, ...) falls back to a raw pd.read_csv
    # attempt there too, so offering the dialog for those would just stack a
    # second, more confusing failure on top of the original one.
    _COLUMN_MAPPABLE_SUFFIXES = {".csv", ".txt", ".xlsx", ".xls"}

    def _on_import_failed(
        self, error_msg: str, file_paths: list[Path] | None = None
    ) -> None:
        from PySide6.QtWidgets import QDialog, QMessageBox

        # Root cause: this import path has no column-mapping step, so a CSV
        # whose headers don't match auto_load's heuristic name list (see
        # rheojax/io/readers/auto.py::_try_csv) fails with no way for the
        # user to specify columns. Give them one via the existing (until now
        # unwired) ColumnMapperDialog, then retry with the chosen mapping.
        # ponytail: matched on message text rather than a typed exception --
        # auto_load raises plain ValueError for every failure mode. Upgrade
        # to a dedicated exception type if more branches need to key off it.
        paths = file_paths or []
        offer_mapper = (
            len(paths) == 1
            and paths[0].suffix.lower() in self._COLUMN_MAPPABLE_SUFFIXES
            and "auto-detect" in error_msg.lower()
        )
        if offer_mapper:
            from rheojax.gui.dialogs.column_mapper import ColumnMapperDialog

            dialog = ColumnMapperDialog(str(paths[0]), parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                mapping = dialog.get_mapping()
                if mapping.get("x") and mapping.get("y"):
                    self._launch_import(
                        paths,
                        x_col=mapping["x"],
                        y_col=mapping["y"],
                        y2_col=mapping.get("y2"),
                        temp_col=mapping.get("temperature"),
                    )
                    return

        QMessageBox.critical(self, "Import Failed", error_msg)

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
        # `edited`/`config_edited` fire on real content changes (model/protocol/prior/
        # NLSQ-option edits) that persist into fit.json -- distinct from navigation,
        # which spec §5.2 explicitly excludes from dirty tracking.
        self._mark_dirty()
        self._advance_and_unlock("fit")

    def _on_transform_body_edited(self) -> None:
        self._mark_dirty()
        self._advance_and_unlock("transform")

    def _on_pipeline_body_edited(self) -> None:
        # Pipeline step-list edits persist into pipeline.json, same rationale as above.
        self._mark_dirty()

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
