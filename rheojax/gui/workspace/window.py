from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rheojax.gui.compat import (
    QAction,
    QApplication,
    QButtonGroup,
    QCloseEvent,
    QDialog,
    QInputDialog,
    QKeySequence,
    QMainWindow,
    QShortcut,
    QSplitter,
    Qt,
    QToolBar,
    QToolButton,
    QWidget,
    Signal,
)
from rheojax.gui.dialogs.preferences import PreferencesDialog
from rheojax.gui.foundation.notifier import DatasetLibraryNotifier
from rheojax.gui.foundation.state import AppState
from rheojax.gui.resources import load_stylesheet
from rheojax.gui.resources.styles.tokens import ThemeManager
from rheojax.gui.services.pipeline_execution_service import PipelineExecutionService
from rheojax.gui.widgets.log_dock import LogDockWidget
from rheojax.gui.workspace.fit.fit_controller import build_fit_controller
from rheojax.gui.workspace.inspector import InspectorPanel
from rheojax.gui.workspace.library_rail import LibraryRail
from rheojax.gui.workspace.status_bar import StatusBar
from rheojax.gui.workspace.stepper_canvas import StepperCanvas
from rheojax.gui.workspace.transform.transform_controller import (
    build_transform_controller,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


class _WindowChrome:
    """Menu, theme, help, and command-palette collaborator for WorkspaceWindow.

    Split out of the former monolithic WorkspaceWindow class (ASSESSMENT.md
    Technical Debt #6) -- window chrome (menus, theme application, OS color
    scheme watching, help dialogs, the command palette, log dock append)
    doesn't touch the stepper/dataset state machine and is the most
    self-contained concern to extract, per that finding's own suggestion.
    Composed into WorkspaceWindow; methods operate on attributes/methods
    owned by WorkspaceWindow.__init__ and its other methods (self._state,
    self.log_dock, self._on_new/_on_open/_on_save/_on_save_as/_on_close,
    self.set_mode). rheojax.gui.* is mypy-ignored (PySide6 stub issues,
    see pyproject.toml), so no typing-only attribute stub is needed here
    (contrast rheojax/pipeline/base.py's _PipelineState, which types-checks).
    """

    def _build_file_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        menu.addAction("&New", self._on_new, "Ctrl+N")
        menu.addAction("&Open...", self._on_open, "Ctrl+O")
        menu.addAction("&Save", self._on_save, "Ctrl+S")
        menu.addAction("Save &As...", self._on_save_as, "Ctrl+Shift+S")
        menu.addAction("&Close", self._on_close)
        menu.addSeparator()
        menu.addAction("&Preferences...", self._on_preferences)

    def _on_preferences(self) -> None:
        dialog = PreferencesDialog(
            current_preferences={"theme": self._state.ui.theme}, parent=self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_preferences(dialog.get_preferences())

    def apply_preferences(self, prefs: dict[str, Any]) -> None:
        """Apply preferences from PreferencesDialog.get_preferences().

        Only the "theme" key is acted on today -- the dialog exposes 15
        other settings (autosave, JAX device, visualization style, ...)
        this shell has no existing mechanism to honor yet; they are
        accepted here and ignored, not silently dropped by omission.
        """
        theme = prefs.get("theme")
        if theme is not None:
            self._apply_theme(theme)

    def _command_palette_actions(self) -> dict[str, Callable[[], None]]:
        """Build the command palette's action dict fresh on every open.

        Rebuilt (not cached) so "Cycle Theme" always closes over the
        *current* self._state.ui.theme rather than a stale value captured
        the first time the palette was opened.
        """
        theme_order = ["light", "dark", "system"]

        def _cycle_theme() -> None:
            current = self._state.ui.theme
            idx = theme_order.index(current) if current in theme_order else 0
            self._apply_theme(theme_order[(idx + 1) % len(theme_order)])

        return {
            "New Project": self._on_new,
            "Open Project...": self._on_open,
            "Save Project": self._on_save,
            "Save Project As...": self._on_save_as,
            "Switch to Fit Mode": lambda: self.set_mode("fit"),
            "Switch to Transform Mode": lambda: self.set_mode("transform"),
            "Switch to Pipeline Mode": lambda: self.set_mode("pipeline"),
            "Toggle Log Panel": self.view_log_dock_action.trigger,
            "Preferences...": self._on_preferences,
            "Cycle Theme": _cycle_theme,
        }

    def _open_command_palette(self) -> None:
        actions = self._command_palette_actions()
        labels = sorted(actions.keys())
        label, ok = QInputDialog.getItem(
            self, "Command Palette", "Action:", labels, 0, False
        )
        if ok and label:
            actions[label]()

    def _build_view_menu(self) -> None:
        menu = self.menuBar().addMenu("&View")
        action = QAction("&Log Panel", self)
        action.setCheckable(True)
        action.setChecked(False)
        action.setStatusTip("Toggle log panel visibility")
        action.triggered.connect(lambda: self.log_dock.setVisible(action.isChecked()))
        menu.addAction(action)
        self.view_log_dock_action = action

        theme_menu = menu.addMenu("&Theme")
        self._theme_light_action = QAction("&Light", self)
        self._theme_dark_action = QAction("&Dark", self)
        self._theme_system_action = QAction("&System", self)
        for theme_action in (
            self._theme_light_action,
            self._theme_dark_action,
            self._theme_system_action,
        ):
            theme_action.setCheckable(True)
        self._theme_light_action.triggered.connect(lambda: self._apply_theme("light"))
        self._theme_dark_action.triggered.connect(lambda: self._apply_theme("dark"))
        self._theme_system_action.triggered.connect(
            lambda: self._apply_theme("system")
        )
        theme_menu.addAction(self._theme_light_action)
        theme_menu.addAction(self._theme_dark_action)
        theme_menu.addAction(self._theme_system_action)

    def _build_help_menu(self) -> None:
        menu = self.menuBar().addMenu("&Help")
        menu.addAction("&Documentation", self._on_open_docs)
        menu.addAction("&Tutorials", self._on_open_tutorials)
        menu.addAction("&Keyboard Shortcuts", self._on_show_shortcuts)
        menu.addSeparator()
        menu.addAction("&About RheoJAX", self._on_about)

    def _on_open_docs(self) -> None:
        import webbrowser

        webbrowser.open("https://rheojax.readthedocs.io")

    def _on_open_tutorials(self) -> None:
        import webbrowser

        webbrowser.open("https://github.com/imewei/rheojax/tree/main/examples")

    def _on_show_shortcuts(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        shortcuts = """
<h3>Keyboard Shortcuts</h3>
<table>
<tr><td><b>Ctrl+N</b></td><td>New Project</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Open Project</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save Project</td></tr>
<tr><td><b>Ctrl+Shift+S</b></td><td>Save Project As</td></tr>
<tr><td><b>Ctrl+K</b></td><td>Command Palette</td></tr>
<tr><td><b>Ctrl+1..9</b></td><td>Jump to Step 1..9 (current mode)</td></tr>
</table>
        """
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)

    def _on_about(self) -> None:
        from rheojax.gui.dialogs.about import AboutDialog

        AboutDialog(self).exec()

    def _apply_theme(self, theme: str) -> None:
        """Apply a QSS theme to the QApplication and sync UI/state.

        ``theme`` is normalized to lowercase on entry -- ``PreferencesDialog``
        (Task 3) round-trips theme as "Light"/"Dark"/"System" (its combo
        box's exact item text via ``currentText()``), while every other
        caller in this file already passes lowercase. Normalizing here, once,
        in the function every caller routes through, means no caller needs
        to know or care which convention the *other* callers use --
        ``load_stylesheet()`` only ever accepts lowercase "light"/"dark" and
        raises ``ValueError`` on anything else, so this must happen before
        ``chosen`` is computed. The stored preference is the lowercased
        *requested* value (not the resolved one), so a saved "system"
        preference round-trips as "system", not as whatever the OS scheme
        happened to be at the time it was applied.
        """
        theme = theme.lower()
        app = QApplication.instance()
        if app is None:
            return
        chosen = self._detect_os_color_scheme() if theme == "system" else theme
        try:
            stylesheet = load_stylesheet(chosen)
            # main.py sets an adaptive base font size at startup and appends
            # a matching "* { font-size: ...pt; }" rule to the stylesheet
            # (see main.py's app.setFont(base_font) + stylesheet +=
            # f"\n* {{ font-size: {base_font_size:.1f}pt; }}\n"). A bare
            # app.setStyleSheet(load_stylesheet(chosen)) here would replace
            # the whole stylesheet and silently drop that rule on every
            # theme switch, including the first one (_build_workspace calls
            # _apply_theme almost immediately after main.py's initial
            # setStyleSheet). Recover the already-applied base font size
            # from the app's current QFont and re-append the same rule.
            font_size = app.font().pointSizeF()
            stylesheet += f"\n* {{ font-size: {font_size:.1f}pt; }}\n"
            app.setStyleSheet(stylesheet)
            ThemeManager.set_theme(chosen)
        except Exception as exc:
            logger.error(
                "Failed to apply theme", theme=theme, error=str(exc), exc_info=True
            )
            return
        self._state.ui.theme = theme
        self._theme_light_action.setChecked(theme == "light")
        self._theme_dark_action.setChecked(theme == "dark")
        self._theme_system_action.setChecked(theme == "system")
        self.statusBar().show_message(f"Theme: {theme.capitalize()}", 2000)

    @staticmethod
    def _detect_os_color_scheme() -> str:
        """Detect the current OS color scheme.

        Tries ``QStyleHints.colorScheme()`` (Qt 6.5+) first, then falls back
        to measuring the ``QPalette.Window`` luminance.

        Returns
        -------
        str
            ``"dark"`` or ``"light"``.
        """
        app = QApplication.instance()
        if app is None:
            return "light"
        try:
            hints = app.styleHints()
            if hasattr(hints, "colorScheme"):
                from rheojax.gui.compat import Qt as _Qt

                scheme = hints.colorScheme()
                if scheme == _Qt.ColorScheme.Dark:
                    return "dark"
                if scheme == _Qt.ColorScheme.Light:
                    return "light"
        except Exception:
            pass
        palette = app.palette()
        window_color = palette.color(palette.ColorRole.Window)
        luminance = (
            0.299 * window_color.red()
            + 0.587 * window_color.green()
            + 0.114 * window_color.blue()
        )
        return "dark" if luminance < 128 else "light"

    def _setup_os_theme_watcher(self) -> None:
        """Hook ``QStyleHints.colorSchemeChanged`` (Qt 6.5+) to re-apply "system" theme.

        Call once during window init. Silently no-ops on Qt < 6.5.
        """
        try:
            app = QApplication.instance()
            if app is None:
                return
            hints = app.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                hints.colorSchemeChanged.connect(self._on_os_color_scheme_changed)
        except Exception:
            logger.debug("QStyleHints.colorSchemeChanged not available")

    def _on_os_color_scheme_changed(self) -> None:
        """Re-apply the theme only when the user preference is "system"."""
        if self._state.ui.theme == "system":
            self._apply_theme("system")

    def log(self, message: str) -> None:
        """Append message to the log dock at INFO level."""
        self.log_dock.append_record(logging.INFO, message)


class WorkspaceWindow(QMainWindow, _WindowChrome):
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
        self._close_confirmed = False
        self._preview_dialog = None
        # Keyed by job_id, not a single scalar -- a second concurrent import
        # launched before the first one finishes must not drop the only
        # Python reference to the first ImportWorker/ImportWorkerSignals
        # QObject (which would otherwise risk it being garbage-collected
        # mid-run and losing its queued signal).
        self._active_import_workers: dict[str, object] = {}
        self._pipeline_service.phase_worker_ready.connect(
            self._on_phase_worker_ready, Qt.ConnectionType.QueuedConnection
        )

        self.setWindowTitle("RheoJAX Workspace")
        self.resize(1200, 800)

        bar = QToolBar(self)
        self.addToolBar(bar)
        # QToolButton (not QPushButton) for the mode switcher: base.qss already
        # styles QToolButton as a flat toggle pill (checked = @primary_light
        # highlight), whereas QPushButton carries the heavy gradient "primary
        # action" look meant for Save/Open-type CTAs -- wrong affordance for a
        # segmented mode switch that should read as navigation, not a command.
        self._fit_btn = QToolButton(self)
        self._fit_btn.setText("Fit")
        self._fit_btn.setToolTip("Switch to the model-fitting workflow")
        self._tx_btn = QToolButton(self)
        self._tx_btn.setText("Transform")
        self._tx_btn.setToolTip("Switch to the data-transform workflow")
        self._pipeline_btn = QToolButton(self)
        self._pipeline_btn.setText("Pipeline")
        self._pipeline_btn.setToolTip("Switch to the batch-pipeline workflow")
        self._fit_btn.setCheckable(True)
        self._tx_btn.setCheckable(True)
        self._pipeline_btn.setCheckable(True)
        # Exclusive group so Qt itself refuses to uncheck the sole checked
        # button on a re-click. Without this, clicking the already-active
        # mode pill toggles it unchecked before set_mode()'s no-op early
        # return (mode == self._mode) ever reaches _sync_mode_buttons() --
        # the pill goes visually unselected while the app stays in that mode.
        self._mode_btn_group = QButtonGroup(self)
        self._mode_btn_group.setExclusive(True)
        for btn in (self._fit_btn, self._tx_btn, self._pipeline_btn):
            self._mode_btn_group.addButton(btn)
        self._fit_btn.clicked.connect(lambda: self.set_mode("fit"))
        self._tx_btn.clicked.connect(lambda: self.set_mode("transform"))
        self._pipeline_btn.clicked.connect(lambda: self.set_mode("pipeline"))
        bar.addWidget(self._fit_btn)
        bar.addWidget(self._tx_btn)
        bar.addWidget(self._pipeline_btn)
        self.setStatusBar(StatusBar(self))
        self._build_file_menu()

        self.log_dock = LogDockWidget(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.setVisible(False)
        self._build_view_menu()
        self._build_help_menu()

        # These fire ~20 times through a real pipeline run but had no
        # listener anywhere -- real per-step/per-phase progress was computed
        # and silently discarded, with no user-visible feedback beyond the
        # coarse per-dataset start/finish. Route them to the log dock (the
        # existing mechanism for this kind of event, see log_dock.append_record
        # elsewhere in this file) rather than building new progress UI.
        # QueuedConnection: PipelineExecutionService documents that it runs
        # off the GUI thread, same as phase_worker_ready above.
        self._pipeline_service.pipeline_started.connect(
            self._on_pipeline_started, Qt.ConnectionType.QueuedConnection
        )
        self._pipeline_service.pipeline_completed.connect(
            self._on_pipeline_completed, Qt.ConnectionType.QueuedConnection
        )
        self._pipeline_service.pipeline_failed.connect(
            self._on_pipeline_failed, Qt.ConnectionType.QueuedConnection
        )
        self._pipeline_service.step_started.connect(
            self._on_pipeline_step_started, Qt.ConnectionType.QueuedConnection
        )
        self._pipeline_service.step_completed.connect(
            self._on_pipeline_step_completed, Qt.ConnectionType.QueuedConnection
        )
        self._pipeline_service.step_failed.connect(
            self._on_pipeline_step_failed, Qt.ConnectionType.QueuedConnection
        )
        self._pipeline_service.step_phase_started.connect(
            self._on_pipeline_step_phase_started, Qt.ConnectionType.QueuedConnection
        )
        self._pipeline_service.step_phase_completed.connect(
            self._on_pipeline_step_phase_completed, Qt.ConnectionType.QueuedConnection
        )
        self._pipeline_service.step_phase_failed.connect(
            self._on_pipeline_step_phase_failed, Qt.ConnectionType.QueuedConnection
        )

        self._build_workspace(app_state)
        self._setup_os_theme_watcher()
        QShortcut(QKeySequence("Ctrl+K"), self, self._open_command_palette)
        # Ctrl+1..Ctrl+9 jump to a step in the *current* mode's wizard.
        # self._canvas always points at the mode's current StepperCanvas even
        # though set_mode() builds a brand-new instance on every switch --
        # binding these once here (dereferencing self._canvas dynamically at
        # call time, not capturing today's instance) keeps them live across
        # mode switches without re-binding per StepperCanvas. Out-of-range
        # indices are rejected by _jump_to_step's own bounds check below;
        # a not-yet-reached-but-in-range index reaches click_step(), which
        # is a no-op there because that button is disabled.
        for step_idx in range(9):
            QShortcut(
                QKeySequence(f"Ctrl+{step_idx + 1}"),
                self,
                lambda i=step_idx: self._jump_to_step(i),
            )

    def _build_workspace(self, state: AppState) -> None:
        self._state = state
        self._apply_theme(state.ui.theme)
        initial_mode = state.ui.mode
        self._mode = initial_mode if initial_mode in self.MODES else "fit"
        from rheojax.gui.workspace.pipeline.controller import build_pipeline_controller

        fit_ctl, self._fit_bodies = build_fit_controller(state, self.statusBar())
        transform_ctl, self._transform_bodies = build_transform_controller(state)
        pipeline_ctl, self._pipeline_bodies = build_pipeline_controller(
            state,
            self._pipeline_service,
            epoch=self._epoch,
            guard=self._guard,
            notify=self._notifier.changed.emit,
        )
        # Named references for the step bodies this window reaches into by
        # semantic role (dataset refresh/selection), not by wizard position --
        # derived once here so a future step insertion can't silently shift
        # what index 1/0 means at each of the reach-in call sites below.
        self._fit_data_step = self._fit_bodies[1]  # build_fit_controller: DataStep
        self._transform_slots_step = self._transform_bodies[1]  # SlotsStep
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
        # Deferred (not called inline): get_jax_info() queries live JAX
        # devices/memory, which is needless synchronous work on the path to
        # first paint. Parented to self, like the same pattern used by
        # _poll_active_jobs_then's timer elsewhere in this file, so a
        # destroyed-mid-construction window (e.g. test teardown) takes the
        # timer down with it instead of firing into a dangling self.
        from PySide6.QtCore import QTimer

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._refresh_status_bar)
        timer.start(0)

        self._rail = LibraryRail(state.library, self)
        # LibraryRail otherwise only ever renders the snapshot present at construction
        # time -- nothing else calls .refresh() -- so without this, a dataset added by
        # any commit path (interactive export or a Pipeline job) never appears until the
        # next full _rebuild() (Open/New/Close).
        self._notifier.changed.connect(self._rail.refresh)
        # LibraryRail isn't the only widget snapshotting the library at
        # construction/last-edit time -- DataStep, SlotsStep, and
        # PipelineConfigureRunStep each render their own dataset list/combo
        # and were never told to rebuild when a new dataset is committed
        # elsewhere (import, export-to-library, a Pipeline job).
        self._notifier.changed.connect(self._fit_data_step.refresh)
        self._notifier.changed.connect(self._transform_slots_step.refresh)
        self._notifier.changed.connect(self._pipeline_bodies[0].refresh)
        self._rail.import_requested.connect(self._on_import_requested)
        self._rail.dataset_preview_requested.connect(
            self._on_dataset_preview_requested
        )
        self._rail.dataset_selected.connect(self._on_rail_dataset_selected)
        self._rail.dataset_delete_requested.connect(self._on_dataset_delete_requested)
        self._inspector = InspectorPanel(self)
        self._canvas = StepperCanvas(self._controllers[self._mode], self)
        self._wire_canvas(self._canvas)
        self._install_bodies(self._mode, self._canvas)
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(self._rail)
        self._splitter.addWidget(self._canvas)
        self._splitter.addWidget(self._inspector)
        self.setCentralWidget(self._splitter)

        # A rebuild (fresh New, or Open loading a saved project) always starts
        # every controller locked at reached={0}, regardless of how much
        # progress its Fit/Transform/PipelineState already reflects -- e.g.
        # reopening a project with a completed fit rendered a blank, locked
        # wizard instead of the restored results. Reuse the same forward-unlock
        # walk that live edits trigger (_advance_and_unlock), looped until each
        # controller's `current` stops advancing, so already-completed steps
        # come back unlocked instead of only the first one.
        for m in self._controllers:
            ctl = self._controllers[m]
            prev_current = -1
            while ctl.current != prev_current:
                prev_current = ctl.current
                self._advance_and_unlock(m)

    def _dispose_workspace(self) -> None:
        # The pipeline controller connects to self._pipeline_service (created
        # once in __init__ and never recreated) rather than anything owned by
        # this workspace instance, so unlike the body-list signals below it
        # survives a rebuild unless explicitly disconnected here -- otherwise
        # every New/Open leaks the old PipelineController (and its AppState)
        # for the life of the window.
        pipeline_ctl = self._controllers.get("pipeline")
        if pipeline_ctl is not None:
            try:
                self._pipeline_service.dataset_run_started.disconnect(
                    pipeline_ctl._started_slot
                )
            except (RuntimeError, TypeError):
                pass
            try:
                self._pipeline_service.dataset_run_finished.disconnect(
                    pipeline_ctl._finished_slot
                )
            except (RuntimeError, TypeError):
                pass
        # Disconnect only the handler each body list actually wired in
        # __init__ -- attempting to disconnect a slot that was never
        # connected to a given body (e.g. _on_transform_body_edited from a
        # fit body's `edited`) makes PySide6 emit a "Failed to disconnect"
        # RuntimeWarning at the libpyside level, which isn't a Python
        # exception and so isn't caught by the try/except below.
        for body in self._fit_bodies:
            if hasattr(body, "edited"):
                try:
                    body.edited.disconnect(self._on_fit_body_edited)
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
            body.deleteLater()
        for body in self._transform_bodies:
            if hasattr(body, "edited"):
                try:
                    body.edited.disconnect(self._on_transform_body_edited)
                except (RuntimeError, TypeError):
                    pass
            if hasattr(body, "dataset_commit_requested"):
                try:
                    body.dataset_commit_requested.disconnect(self._commit_dataset)
                except (RuntimeError, TypeError):
                    pass
            body.deleteLater()
        for body in self._pipeline_bodies:
            if hasattr(body, "edited"):
                try:
                    body.edited.disconnect(self._on_pipeline_body_edited)
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
            self._notifier.changed.disconnect(self._fit_data_step.refresh)
        except (RuntimeError, TypeError):
            pass
        try:
            self._notifier.changed.disconnect(self._transform_slots_step.refresh)
        except (RuntimeError, TypeError):
            pass
        try:
            self._notifier.changed.disconnect(self._pipeline_bodies[0].refresh)
        except (RuntimeError, TypeError):
            pass
        try:
            self._rail.import_requested.disconnect(self._on_import_requested)
        except (RuntimeError, TypeError):
            pass
        try:
            self._rail.dataset_preview_requested.disconnect(
                self._on_dataset_preview_requested
            )
        except (RuntimeError, TypeError):
            pass
        try:
            self._rail.dataset_selected.disconnect(self._on_rail_dataset_selected)
        except (RuntimeError, TypeError):
            pass
        try:
            self._rail.dataset_delete_requested.disconnect(
                self._on_dataset_delete_requested
            )
        except (RuntimeError, TypeError):
            pass
        from rheojax.gui.compat import _is_qobject_alive

        if self._preview_dialog is not None:
            if _is_qobject_alive(self._preview_dialog):
                self._preview_dialog.close()
                self._preview_dialog.deleteLater()
            self._preview_dialog = None
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
        def _new():
            self._rebuild(AppState())
            self.statusBar().show_message("New project created", 2000)

        self._maybe_confirm_active_jobs(
            lambda: self._maybe_confirm_unsaved_changes(_new)
        )

    def _on_open(self) -> None:
        from rheojax.gui.compat import QFileDialog, QMessageBox

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
                self.statusBar().show_message("Project opened", 2000)

        self._maybe_confirm_active_jobs(
            lambda: self._maybe_confirm_unsaved_changes(_open)
        )

    def _blocked_by_active_jobs(
        self, action: str, on_unblocked: Callable[[], None] | None = None
    ) -> bool:
        # Spec §3.3: "Save snapshots job_history only; blocked while active_jobs is
        # non-empty" -- a running Pipeline batch mutates DatasetLibrary/job_history from
        # a worker thread, so a concurrent Save could serialize a torn, inconsistent
        # snapshot. Unlike Close/New/Open (which offer to cancel jobs and proceed), Save
        # is a hard block: there is no safe "discard the running job" option here.
        #
        # on_unblocked, if given, runs while active_jobs.lock is still held --
        # closes the gap between "is anything running?" and the caller's own
        # save I/O that a worker thread could otherwise slip a new job
        # registration into (the exact race Save is meant to prevent). Any
        # exception on_unblocked raises propagates to the caller after the
        # lock is released (the `with` block's __exit__ still runs).
        with self._state.active_jobs.lock:
            job_count = len(self._state.active_jobs.by_id)
            if job_count == 0:
                if on_unblocked is not None:
                    on_unblocked()
                return False
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "Jobs Running",
            f"Cannot {action} while {job_count} job(s) are still "
            "running. Wait for them to finish, or cancel them via Close/New/Open first.",
        )
        return True

    def _on_save(self) -> None:
        if self._state.project.path is None:
            self._on_save_as()
            return
        from rheojax.gui.compat import QMessageBox
        from rheojax.gui.foundation.project_codec import save_project_v2

        try:
            if self._blocked_by_active_jobs(
                "save",
                on_unblocked=lambda: save_project_v2(
                    self._state, self._state.project.path
                ),
            ):
                return
        except (ValueError, FileNotFoundError, OSError) as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return
        self._state.project.dirty = False
        self.statusBar().show_message("Project saved", 2000)

    def _on_save_as(self) -> None:
        from rheojax.gui.compat import QFileDialog, QMessageBox

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "RheoJAX Project (*.rheojax)"
        )
        if path:
            from rheojax.gui.foundation.project_codec import save_project_v2

            try:
                if self._blocked_by_active_jobs(
                    "save", on_unblocked=lambda: save_project_v2(self._state, path)
                ):
                    return
            except (ValueError, FileNotFoundError, OSError) as exc:
                QMessageBox.critical(self, "Save Failed", str(exc))
                return
            self._state.project.path = path
            self._state.project.name = Path(path).stem
            self._state.project.dirty = False
            self.statusBar().show_message("Project saved", 2000)

    def _on_close(self) -> None:
        # Actually close the window (same chain closeEvent uses for the OS ✕
        # button), not a workspace reset -- this used to call
        # self._rebuild(AppState()), copy-pasted from _on_new, which just
        # blanked the project instead of closing anything.
        self._maybe_confirm_active_jobs(
            lambda: self._maybe_confirm_unsaved_changes(self._confirmed_close)
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        # Closing via the OS window controls (X button, Alt+F4, Cmd+Q) hits
        # this instead of the File>Close menu action, which is wired to
        # _on_close. Without this override Qt's default closeEvent just
        # accepts and closes immediately, skipping the active-jobs
        # cancel/confirm chain and orphaning running QThreadPool workers.
        # Reuse that same chain here; only accept once it has actually run
        # to completion (job cancellation may finish asynchronously via
        # _poll_active_jobs_then), then re-issue close() to get a second,
        # now-trivial closeEvent that accepts.
        if self._close_confirmed:
            event.accept()
            return
        event.ignore()
        self._maybe_confirm_active_jobs(
            lambda: self._maybe_confirm_unsaved_changes(self._confirmed_close)
        )

    def _confirmed_close(self) -> None:
        # ponytail: assumes WorkspaceWindow is top-level (its only construction site is
        # _create_workspace_window() in rheojax/gui/main.py); self.close() semantics
        # would need reconsidering if it's ever embedded as a child widget instead.
        self._close_confirmed = True
        self.close()

    def _maybe_confirm_active_jobs(self, proceed) -> None:
        with self._state.active_jobs.lock:
            job_count = len(self._state.active_jobs.by_id)
        if job_count == 0:
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
            self,
            "Jobs Running",
            f"{job_count} job(s) still running. Cancel them and continue?",
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
        # Snapshot under the lock: PipelineBatchRunner writes directly into
        # this same plain dict from a QThreadPool worker thread (see its
        # comment), so iterating the live dict here risks
        # "RuntimeError: dictionary changed size during iteration" if a new
        # job is registered mid-loop.
        with self._state.active_jobs.lock:
            jobs_snapshot = list(self._state.active_jobs.by_id.values())
        for job in jobs_snapshot:
            worker = job.get("worker")
            if worker is not None:
                QThreadPool.globalInstance().start(CancelWorkerRunnable(worker))

        self._poll_active_jobs_then(proceed, remaining_polls=120)  # 120 * 250ms = 30s

    def _poll_active_jobs_then(self, proceed, remaining_polls: int) -> None:
        from rheojax.gui.compat import _is_qobject_alive

        if not _is_qobject_alive(self):
            # Window was destroyed (e.g. test teardown) while this poll chain
            # was still in flight -- the QTimer.singleShot below already fired
            # into a dangling `self`. Bail out before touching any Qt API.
            return
        if not self._state.active_jobs.by_id:
            self._active_jobs_action_pending = False
            proceed()
            return
        if remaining_polls <= 0:
            # Jobs are still running after the 30s cancellation budget --
            # do NOT proceed(): that would rebuild/replace self._state while a
            # worker is still mutating the project it belonged to. Tell the
            # user and let them retry once cancellation actually finishes.
            #
            # Only show that warning if this window is actually visible.
            # _is_qobject_alive above only catches the case where self's C++
            # object has already been destroyed -- it does NOT catch a
            # window that is still fully alive but orphaned (e.g. a stale
            # instance from a previous test that was never shown, still
            # ticking down this same 250ms/30s chain). A real user always
            # has this window visible for the entire duration of a close/new
            # confirmation, so gating on isVisible() costs nothing for real
            # usage while preventing this modal from firing into whatever
            # unrelated context happens to be running when a stale chain's
            # timer finally elapses -- that reentrant QMessageBox.warning()
            # call is what caused a reproducible Fatal Python error: Aborted
            # crash under the full GUI test suite.
            if self.isVisible():
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "Jobs Still Running",
                    "Some jobs did not finish cancelling in time. This "
                    "action was cancelled -- try again once they've "
                    "stopped.",
                )
            self._active_jobs_action_pending = False
            return
        from PySide6.QtCore import QTimer

        # Parented to self (not the static QTimer.singleShot(), which is a
        # free-floating timer nothing owns): when self is destroyed mid-chain
        # (e.g. test teardown), Qt's parent-child ownership stops and deletes
        # this timer automatically, instead of it firing 250ms-1120x30s later
        # into whatever unrelated widget/test happens to be running then.
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda: self._poll_active_jobs_then(proceed, remaining_polls - 1)
        )
        timer.start(250)

    def _on_phase_worker_ready(
        self, dataset_id: str, step_id: str, phase: str, worker
    ) -> None:
        # Every other active_jobs.by_id access in this file takes this lock --
        # PipelineBatchRunner mutates the same dict from a worker thread, so
        # skipping it here left a TOCTOU gap where a job could be evicted
        # between the .get() and the assignment, silently dropping the worker
        # reference the cancel-on-close path (_maybe_confirm_active_jobs)
        # needs to actually cancel a still-running job.
        with self._state.active_jobs.lock:
            job = self._state.active_jobs.by_id.get(dataset_id)
            if job is not None:
                job["worker"] = worker

    def _on_pipeline_started(self) -> None:
        self.log_dock.append_record(logging.INFO, "Pipeline run started")

    def _on_pipeline_completed(self) -> None:
        self.log_dock.append_record(logging.INFO, "Pipeline run completed")

    def _on_pipeline_failed(self, error: str) -> None:
        self.log_dock.append_record(logging.ERROR, f"Pipeline run failed: {error}")

    def _on_pipeline_step_started(self, step_id: str) -> None:
        self.log_dock.append_record(logging.INFO, f"Step {step_id} started")

    def _on_pipeline_step_completed(self, step_id: str) -> None:
        self.log_dock.append_record(logging.INFO, f"Step {step_id} completed")

    def _on_pipeline_step_failed(self, step_id: str, error: str) -> None:
        self.log_dock.append_record(
            logging.ERROR, f"Step {step_id} failed: {error}"
        )

    def _on_pipeline_step_phase_started(self, step_id: str, phase: str) -> None:
        self.log_dock.append_record(
            logging.INFO, f"Step {step_id}: {phase} phase started"
        )

    def _on_pipeline_step_phase_completed(self, step_id: str, phase: str) -> None:
        self.log_dock.append_record(
            logging.INFO, f"Step {step_id}: {phase} phase completed"
        )

    def _on_pipeline_step_phase_failed(
        self, step_id: str, phase: str, error: str
    ) -> None:
        self.log_dock.append_record(
            logging.ERROR, f"Step {step_id}: {phase} phase failed: {error}"
        )

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
                    self,
                    "Cannot Save",
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
        # _install_bodies() swaps every placeholder page for its real body via
        # a sequence of QStackedWidget insertWidget()/removeWidget() calls;
        # Qt reassigns currentIndex during that churn to keep pointing at
        # whatever widget was "current" at each step, which does not reliably
        # land back on index 0. refresh() re-syncs the displayed page (and
        # button checked-state) to the controller's actual current step.
        new_canvas.refresh()
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

    def _on_dataset_preview_requested(self, dataset_id: str) -> None:
        from rheojax.gui.compat import _is_qobject_alive
        from rheojax.gui.dialogs.dataset_preview import DatasetPreviewDialog
        from rheojax.gui.services.data_service import DataService
        from rheojax.gui.utils.rheodata import rheodata_from_any

        with self._state.library.lock:
            try:
                ref = self._state.library.get(dataset_id)
            except KeyError:
                self.log_dock.append_record(
                    logging.WARNING,
                    f"Preview requested for unknown dataset id: {dataset_id}",
                )
                return
            try:
                payload = self._state.library.load_payload(dataset_id)
            except KeyError:
                payload = None

        data = None
        if payload is not None:
            try:
                candidate = rheodata_from_any(payload).to_numpy()
                if candidate.x.ndim == 1 and candidate.y.ndim == 1:
                    data = candidate
            except (TypeError, ValueError, IndexError):
                data = None

        warnings: list[str] = []
        if data is not None:
            if len(data.x) == 0:
                warnings = ["Dataset contains no rows"]
            else:
                try:
                    warnings = DataService().validate_data(data)
                except Exception as exc:
                    # Object-dtype/non-numeric values can pass the shape checks
                    # above but still trip validate_data()'s numeric operations
                    # (np.isfinite, np.ptp, percentiles) -- surface that as a
                    # warning instead of crashing the preview.
                    warnings = [f"Validation check failed: {exc}"]

        if self._preview_dialog is None or not _is_qobject_alive(self._preview_dialog):
            self._preview_dialog = DatasetPreviewDialog(self)
        self._preview_dialog.set_dataset(ref, data, warnings)
        self._preview_dialog.show()
        self._preview_dialog.raise_()
        self._preview_dialog.activateWindow()

    def _on_dataset_delete_requested(self, dataset_id: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        with self._state.library.lock:
            try:
                ref = self._state.library.get(dataset_id)
            except KeyError:
                self.log_dock.append_record(
                    logging.WARNING,
                    f"Delete requested for unknown dataset id: {dataset_id}",
                )
                return

        # Mirrors _blocked_by_active_jobs (used for Save/Close): a running fit
        # or Pipeline batch job reads a dataset's payload via
        # library.load_payload() from a worker thread. Deleting the dataset
        # out from under that read would pop the payload mid-job instead of
        # failing cleanly, so check ActiveJobsState before offering the
        # confirm-delete prompt at all.
        # fit_controller.py's NLSQ/NUTS jobs key by "{dataset_id}:nlsq" /
        # "{dataset_id}:nuts" (not the bare dataset id, so two concurrent
        # jobs on the same dataset don't clobber each other's active_jobs
        # entry) -- match both the bare key (other producers, e.g. Pipeline
        # batch jobs) and the ":"-suffixed form.
        with self._state.active_jobs.lock:
            has_active_job = any(
                key == dataset_id or key.startswith(f"{dataset_id}:")
                for key in self._state.active_jobs.by_id
            )
        if has_active_job:
            QMessageBox.warning(
                self,
                "Delete Dataset",
                "A job is currently running on this dataset. Wait for it to "
                "finish (or cancel it) before deleting.",
            )
            return

        choice = QMessageBox.question(
            self,
            "Delete Dataset",
            f'Delete dataset "{ref.name}"? This cannot be undone.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return

        with self._state.library.lock:
            self._state.library.remove(dataset_id)
        self._notifier.changed.emit()
        self.log_dock.append_record(logging.INFO, f'Deleted dataset "{ref.name}"')

    def _commit_dataset(self, ref, payload=None, overwrite: bool = False) -> None:
        # Hold the library's lock across both calls so a concurrent reader never
        # observes a DatasetRef registered by add() before store_payload() has
        # written its payload (see DatasetLibrary.lock's docstring in library.py).
        with self._state.library.lock:
            self._state.library.add(ref, overwrite=overwrite)
            if payload is not None:
                self._state.library.store_payload(ref.id, payload)
        self._notifier.changed.emit()

    _IMPORT_TEST_MODE_ALIASES = {"flow": "flow_curve"}

    @classmethod
    def _normalize_import_test_mode(cls, test_mode: str | None) -> str | None:
        if test_mode in cls._IMPORT_TEST_MODE_ALIASES:
            return cls._IMPORT_TEST_MODE_ALIASES[test_mode]
        try:
            from rheojax.core.test_modes import TestModeEnum

            protocol = TestModeEnum(str(test_mode).lower()).to_protocol()
        except ValueError:
            return test_mode
        return protocol.value if protocol is not None else test_mode

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

    def _on_rail_dataset_selected(self, dataset_id: str) -> None:
        """Forward a LibraryRail click to the Fit workflow's dataset selector.

        Transform's SlotsStep manages multiple per-slot combos and Pipeline's
        PipelineConfigureRunStep selects datasets for batch execution --
        neither has a single "make this the active dataset" concept to
        forward to, so only Fit mode's DataStep (which has exactly that) is
        wired here.
        """
        if self._mode == "fit":
            self._fit_data_step.select_dataset(dataset_id)

    def _launch_import(
        self,
        file_paths: list[Path],
        x_col: str | None = None,
        y_col: str | None = None,
        y2_col: str | None = None,
        temp_col: str | None = None,
    ) -> None:
        import uuid

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
        # Register in active_jobs so New/Open/Close (_maybe_confirm_active_jobs)
        # see this import as in-flight and wait/warn instead of racing _rebuild
        # -- previously an unregistered import could still be running when the
        # user opened a different project, and would then write its results
        # into that new project's library once it completed.  No "worker" key:
        # ImportWorker has no cancel() path, so CancelWorkerRunnable is skipped
        # for this entry; the poll loop just waits for it to finish naturally.
        job_id = f"import:{uuid.uuid4().hex}"
        self._state.active_jobs.by_id[job_id] = {}
        # QueuedConnection guarantees the callback runs on the main thread --
        # ImportWorker emits its signals from a QThreadPool worker thread.
        # job_id is bound per-connection so each launch's completion/failure
        # pops its own registration, not whichever import happens to be
        # current AppState by the time the signal is delivered.
        worker.signals.completed.connect(
            lambda datasets, jid=job_id: self._on_import_completed(
                datasets, job_id=jid
            ),
            Qt.ConnectionType.QueuedConnection,
        )
        # file_paths is bound per-connection (not read from shared instance
        # state) so that a second import launched before this worker's
        # "failed" signal is delivered can never make the failure handler
        # act on the wrong file's paths.
        worker.signals.failed.connect(
            lambda msg, fp=file_paths, jid=job_id: self._on_import_failed(
                msg, fp, job_id=jid
            ),
            Qt.ConnectionType.QueuedConnection,
        )
        # Keep a reference alive for the worker's lifetime -- nothing else
        # holds the ImportWorker/ImportWorkerSignals QObject, so it would
        # otherwise be garbage-collected mid-run and drop the signal. Keyed
        # by job_id so a second concurrent import can't evict the first.
        self._active_import_workers[job_id] = worker
        QThreadPool.globalInstance().start(worker)

    def _on_import_completed(self, datasets: list, job_id: str | None = None) -> None:
        self._state.active_jobs.by_id.pop(job_id, None)
        self._active_import_workers.pop(job_id, None)
        import hashlib
        import uuid
        from collections import Counter

        from rheojax.gui.foundation.library import DatasetRef
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        hash_cache: dict[str, str] = {}
        # Multiple segments from one source file get "<stem>_segment_N"
        # suffixing (N starting at 1) instead of all sharing the bare stem --
        # otherwise the rail/combo shows indistinguishable duplicate entries
        # for e.g. a multi-frequency-segment TRIOS file. Counted up front
        # (non-destructive .get, before the pop below) so a file that turns
        # out to have exactly one segment keeps its plain stem name.
        source_file_counts = Counter(
            sf
            for rheo_data in datasets
            if (sf := rheo_data.metadata.get("_source_file"))
        )
        segment_index: dict[str, int] = {}
        for rheo_data in datasets:
            meta = rheo_data.metadata
            source_file = meta.pop("_source_file", None)
            test_mode = meta.get("test_mode") or service.detect_test_mode(rheo_data)
            test_mode = self._normalize_import_test_mode(test_mode)
            meta["test_mode"] = test_mode

            if source_file and source_file not in hash_cache:
                hash_cache[source_file] = hashlib.sha256(
                    Path(source_file).read_bytes()
                ).hexdigest()

            if source_file:
                stem = Path(source_file).stem
                if source_file_counts[source_file] > 1:
                    segment_index[source_file] = segment_index.get(source_file, 0) + 1
                    name = f"{stem}_segment_{segment_index[source_file]}"
                else:
                    name = stem
            else:
                name = "imported"

            ref = DatasetRef(
                id=uuid.uuid4().hex,
                name=name,
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

    _COLUMN_MAPPABLE_SUFFIXES = {".csv", ".txt", ".xlsx", ".xls"}

    def _on_import_failed(
        self,
        error_msg: str,
        file_paths: list[Path] | None = None,
        job_id: str | None = None,
    ) -> None:
        self._state.active_jobs.by_id.pop(job_id, None)
        self._active_import_workers.pop(job_id, None)
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
            # ColumnMapperDialog re-reads the file itself and can fail even
            # for a "mappable" extension (e.g. malformed/mixed-delimiter
            # content) -- its own failure path only shows a warning and
            # leaves the dialog open with empty column lists. Detect that
            # here (self.columns stays [] on load failure) and skip straight
            # to the original error instead of showing a broken, unusable
            # dialog on top of it.
            if dialog.columns and dialog.exec() == QDialog.DialogCode.Accepted:
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

    def _refresh_status_bar(self) -> None:
        # ponytail: "active project" status deferred — needs the Plan 3/4
        # AppState.project slice, which doesn't exist yet (spec §4.3).
        try:
            from rheojax.gui.utils.jax_utils import get_jax_info

            info = get_jax_info()
            self.statusBar().update_jax_status(
                device=info.get("default_device", "?"),
                memory_used=info.get("memory_used_mb", 0),
                memory_total=info.get("memory_total_mb", 0),
                float64_enabled=info.get("float64_enabled", False),
            )
        except Exception:
            pass

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

    def _jump_to_step(self, index: int) -> None:
        """Ctrl+N handler: jump to step `index` of the current mode's wizard.

        No-op for an index past the current mode's step count (e.g. Ctrl+9
        while Transform, which only has 5 steps) or a step not yet reached.
        """
        if index < self.active_step_count():
            self._canvas.click_step(index)

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

        import uuid

        from PySide6.QtCore import QThreadPool

        from rheojax.gui.workspace.pipeline.batch_runner import PipelineBatchRunner

        pipeline_state = self._state.pipeline
        self._pipeline_stop_event = threading.Event()
        # Register a batch sentinel synchronously on the GUI thread, under the same
        # lock the above guard reads, before QThreadPool.start() can even schedule the
        # runner -- PipelineBatchRunner.run() only writes its first per-dataset
        # active_jobs entry once it actually starts executing on a worker thread,
        # which QThreadPool.start() does not guarantee happens before it returns. Without
        # this, a second "Run All" click landing in that gap would see an empty
        # active_jobs and start a second runner. PipelineBatchRunner pops this key
        # itself once the batch reaches a terminal state (batch_runner.py's run()).
        batch_job_id = f"pipeline_batch:{uuid.uuid4().hex}"
        with self._state.active_jobs.lock:
            self._state.active_jobs.by_id[batch_job_id] = {"status": "starting"}
        runner = PipelineBatchRunner(
            service=self._pipeline_service,
            # Shallow-copy both lists: PipelineBatchRunner.run() iterates
            # them on a worker thread across the whole (potentially long)
            # batch. Add/Remove Step in step1_configure_run.py mutate
            # `pipeline_state.steps` in place (append/del) while staying
            # clickable on the GUI thread -- without a copy, an in-flight
            # batch can silently skip, duplicate, or misconfigure steps for
            # whichever dataset happens to be substituting when the edit
            # lands. `selected_dataset_ids` is only ever wholesale-reassigned
            # today (never mutated in place), so this copy is precautionary
            # for it rather than closing a currently-reachable race.
            steps=list(pipeline_state.steps),
            selected_dataset_ids=list(pipeline_state.selected_dataset_ids),
            library=self._state.library,
            stop_requested=self._pipeline_stop_event,
            app_state=self._state,
            batch_job_id=batch_job_id,
        )
        QThreadPool.globalInstance().start(runner)
