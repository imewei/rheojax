"""
Main Application Window
=======================

Central window coordinating pages, state, and services with dock-based layout.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.gui.app.menu_bar import MenuBar
from rheojax.gui.app.status_bar import StatusBar
from rheojax.gui.compat import (
    QApplication,
    QCloseEvent,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QKeySequence,
    QMainWindow,
    QMessageBox,
    QShortcut,
    QSplitter,
    Qt,
    QTextEdit,
    QToolBar,
    QWidget,
    Signal,
    Slot,
)
from rheojax.gui.dialogs.about import AboutDialog
from rheojax.gui.dialogs.import_wizard import ImportWizard
from rheojax.gui.dialogs.preferences import PreferencesDialog
from rheojax.gui.jobs.transform_worker import TransformWorker
from rheojax.gui.jobs.worker_pool import WorkerPool

if TYPE_CHECKING:
    from rheojax.gui.pages.bayesian_page import BayesianPage  # noqa: F401
    from rheojax.gui.pages.data_page import DataPage  # noqa: F401
    from rheojax.gui.pages.diagnostics_page import DiagnosticsPage  # noqa: F401
    from rheojax.gui.pages.export_page import ExportPage  # noqa: F401
    from rheojax.gui.pages.fit_page import FitPage  # noqa: F401
    from rheojax.gui.pages.home_page import HomePage  # noqa: F401
    from rheojax.gui.pages.transform_page import TransformPage  # noqa: F401
from rheojax.gui.resources import load_stylesheet
from rheojax.gui.resources.styles import ThemeManager
from rheojax.gui.state import actions as pipeline_actions
from rheojax.gui.state.signals import StateSignals
from rheojax.gui.state.store import (
    AppState,
    DatasetState,
    FitResult,
    PipelineStep,
    StateStore,
    StepStatus,
    WorkflowMode,
)
from rheojax.gui.widgets.dataset_tree import DatasetTree
from rheojax.gui.widgets.pipeline_chips import PipelineChips
from rheojax.gui.widgets.pipeline_sidebar import PipelineSidebar
from rheojax.gui.widgets.workspace_container import WorkspaceContainer
from rheojax.logging import get_logger

logger = get_logger(__name__)


class RheoJAXMainWindow(QMainWindow):
    """Main application window for RheoJAX GUI.

    Architecture:
        - QSplitter central layout (PipelineSidebar left, WorkspaceContainer right)
        - Bottom dock: log panel
        - Central state store with Redux-like actions/reducers
        - Service layer for RheoJAX API integration
        - Background worker pool for long-running tasks

    Attributes
    ----------
    store : Store
        Central state management
    menu_bar : MenuBar
        Application menu bar
    status_bar : StatusBar
        Status bar with progress and system indicators
    sidebar : PipelineSidebar
        Left pipeline step navigation sidebar
    workspace : WorkspaceContainer
        Right stacked page container

    Example
    -------
    >>> from rheojax.gui.app import RheoJAXMainWindow  # doctest: +SKIP
    >>> window = RheoJAXMainWindow()  # doctest: +SKIP
    >>> window.show()  # doctest: +SKIP
    """

    # Custom signals
    state_dirty = Signal(bool)  # Emitted when unsaved changes exist

    def __init__(
        self, parent: QWidget | None = None, start_maximized: bool = False
    ) -> None:
        """Initialize main window with pages and services.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        start_maximized : bool, optional
            Skip initial resize when launching maximized to avoid clobbering
            the requested window state.
        """
        super().__init__(parent)

        logger.debug("Initializing", class_name=self.__class__.__name__)

        self._start_maximized = start_maximized

        # Initialize state store
        self.store = StateStore()
        # Wire Qt signals for state changes
        self.store.set_signals(self.store.signals or StateSignals())
        self._has_unsaved_changes = False
        self.worker_pool: WorkerPool | None = None
        self._job_types: dict[str, str] = {}
        self._job_metadata: dict[str, dict] = {}  # Capture context at submission time
        self._active_relays: set = (
            set()
        )  # Keep QObject relays alive until signals delivered
        self._navigating: bool = False  # GUI-013: re-entry guard for navigate_to
        self._plot_style: str = "default"
        self._current_workflow_mode: WorkflowMode = WorkflowMode.FITTING

        # Setup UI components
        logger.debug("Setting up UI components")
        self.setup_ui()
        self.setup_docks()
        self.setup_workspace()

        # Connect signals
        logger.debug("Connecting signals")
        self.connect_signals()
        self._connect_state_signals()
        self._init_worker_pool()

        # Hook OS color-scheme changes for auto-theme
        self._setup_os_theme_watcher()

        # Initial status
        self.status_bar.show_message("Ready", 3000)
        logger.info("Application initialized", window_title=self.windowTitle())

    def setup_ui(self) -> None:
        """Create all UI elements."""
        logger.debug("Setting up main UI elements")
        # Window properties
        self.setWindowTitle("RheoJAX - Rheological Analysis")
        if not getattr(self, "_start_maximized", False):
            self.resize(1400, 900)

        # Menu bar
        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Pipeline chips (kept for status wiring but not shown as a toolbar)
        self.pipeline_chips = PipelineChips(self)

        # (Quick fit strip removed)

        # Status bar
        self.status_bar = StatusBar(self)
        self.setStatusBar(self.status_bar)
        # Show pipeline chips in status bar for quick navigation/status
        self.status_bar.addPermanentWidget(self.pipeline_chips)

        # Update JAX status on startup
        self._update_jax_status()
        self._setup_shortcuts()

        logger.debug("UI setup complete")

    def setup_docks(self) -> None:
        """Create dock widgets for log panel.

        The left DatasetTree dock has been replaced by PipelineSidebar inside
        the central QSplitter.  The DatasetTree widget is still constructed here
        as ``self.data_tree`` so that existing signal connections remain valid;
        it is not added to a dock widget.
        """
        logger.debug("Setting up dock widgets")

        # Keep DatasetTree widget for signal compatibility (not shown in a dock).
        self.data_tree = DatasetTree(self)
        # Expose a None sentinel so any code that checks self.data_dock doesn't crash.
        self.data_dock = None

        # Bottom dock: Log panel (collapsible)
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("LogDock")
        self.log_panel = QTextEdit(self)
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumHeight(200)
        self.log_panel.setPlaceholderText("Application logs will appear here...")
        self.log_dock.setWidget(self.log_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        # Start with log panel hidden
        self.log_dock.setVisible(False)

    def setup_workspace(self) -> None:
        """Create sidebar + workspace splitter as central widget.

        The WorkspaceContainer owns the page instances.  Convenience references
        (self.home_page, self.data_page, …) are exposed here so that existing
        signal connection code in connect_signals() continues to work without
        modification.
        """
        logger.debug("Setting up workspace layout")

        # Left sidebar and right workspace container
        self.sidebar = PipelineSidebar(self)
        self.workspace = WorkspaceContainer(self)

        # Expose page references for backward-compatibility with connect_signals()
        self.home_page = self.workspace._home_page
        self.data_page = self.workspace._data_page
        self.transform_page = self.workspace._transform_page
        self.fit_page = self.workspace._fit_page
        self.bayesian_page = self.workspace._bayesian_page
        self.diagnostics_page = self.workspace._diagnostics_page
        self.export_page = self.workspace._export_page

        # Horizontal splitter: sidebar (fixed left) | workspace (expanding right)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.workspace)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([260, 1140])
        self.sidebar.setMinimumWidth(260)

        self.setCentralWidget(self.splitter)

        logger.debug("Workspace layout setup complete")

    def _update_tabs_visibility(self, mode: WorkflowMode) -> None:
        """Update visible page on mode change.

        In the new splitter layout the sidebar always shows all pipeline steps;
        workflow-mode filtering is informational only and logged for diagnostics.

        Parameters
        ----------
        mode : WorkflowMode
            Active workflow mode
        """
        logger.debug("Workflow mode updated", mode=mode.name)

    def _wrap_widget_in_toolbar(self, widget: QWidget) -> QToolBar:
        """Place an arbitrary widget inside a non-movable toolbar."""
        toolbar = QToolBar(self)
        toolbar.setMovable(False)
        toolbar.addWidget(widget)
        return toolbar

    def connect_signals(self) -> None:
        """Connect state signals to UI updates."""
        # Subscribe to store updates
        self.store.subscribe(self._on_state_changed)

        # Connect all menu bar actions
        self._connect_file_menu()
        self._connect_edit_menu()
        self._connect_view_menu()
        self._connect_data_menu()
        self._connect_models_menu()
        self._connect_transforms_menu()
        self._connect_analysis_menu()
        self._connect_pipeline_menu()
        self._connect_tools_menu()
        self._connect_help_menu()

        # Pipeline chips navigation
        self.pipeline_chips.step_clicked.connect(
            lambda step: self.navigate_to(step.name.lower())
        )

        # WorkspaceContainer page-change logging
        self.workspace.page_changed.connect(
            lambda page: logger.debug("Page changed", to_page=page)
        )

        # Sidebar navigation signals
        self.sidebar.step_selected.connect(self._on_sidebar_step_selected)
        self.sidebar.run_all_requested.connect(self._on_run_all)
        self.sidebar.run_step_requested.connect(self._on_run_step)
        # GUI-009: auto-navigate when a new step is added so the workspace
        # immediately shows the correct page for the newly created step.
        self.sidebar.step_added.connect(self._on_sidebar_step_selected)
        # GUI-016: auto-populate step config from current page selections
        # when a new step is added (model name, transform name, etc.)
        self.sidebar.step_added.connect(self._auto_populate_step_config)

        # Dataset tree selection
        self.data_tree.dataset_selected.connect(self._on_dataset_selected)
        # R13-GUI-CTX-001: connect context_action_triggered so tree context menu
        # actions (Remove, Rename, Export Data, etc.) are actually handled.
        self.data_tree.context_action_triggered.connect(
            self._on_context_action_triggered
        )

        # Transform page callbacks
        self.transform_page.transform_applied.connect(
            self._on_transform_applied_from_page
        )
        # GUI-016: sync pipeline step config when user browses transforms
        self.transform_page.transform_selected.connect(
            lambda _name: self._sync_transform_step_config()
        )

        # Fit page requests (run through centralized job pipeline)
        self.fit_page.fit_requested.connect(self._on_fit_requested_from_page)

        # Connect home page shortcuts/links
        self.home_page.open_project_requested.connect(self._on_open_file)
        self.home_page.import_data_requested.connect(self._on_import)
        self.home_page.new_project_requested.connect(self._on_new_file)
        self.home_page.recent_project_opened.connect(self._on_open_recent_project)

        # Diagnostics page signals
        self.diagnostics_page.show_requested.connect(self._on_show_diagnostics)
        self.diagnostics_page.plot_requested.connect(
            self._on_diagnostics_plot_requested
        )

        # Export page signals
        self.export_page.export_requested.connect(self._on_export_requested)
        self.export_page.export_completed.connect(self._on_export_completed)
        self.export_page.export_failed.connect(self._on_export_failed)

    def _connect_state_signals(self) -> None:
        """Connect store signals to UI updates (theme, pipeline, datasets)."""

        signals = self.store.signals
        if signals is None:
            return

        signals.theme_changed.connect(self._apply_theme)
        signals.dataset_added.connect(
            lambda dataset_id: self.status_bar.show_message(
                f"Dataset added: {dataset_id}", 2000
            )
        )
        signals.dataset_added.connect(self._on_dataset_added)
        signals.pipeline_step_changed.connect(self._on_pipeline_step_changed)

        # GUI-016: keep selected pipeline step config in sync with page changes
        signals.model_selected.connect(self._sync_fit_step_config)
        signals.model_params_changed.connect(self._sync_fit_step_config)

    def _init_worker_pool(self) -> None:
        """Create and connect WorkerPool if PySide6 is available."""
        logger.debug("Initializing worker pool")

        try:
            self.worker_pool = WorkerPool.instance()
            logger.debug("Worker pool initialized successfully")
        except Exception as exc:
            logger.error(
                "Worker pool initialization failed", error=str(exc), exc_info=True
            )
            self.log(f"Worker pool unavailable: {exc}")
            return

        # R12-C-008: Use explicit QueuedConnection so that WorkerPool signals
        # emitted from background threads are always delivered on the main Qt
        # event-loop thread, preventing cross-thread slot invocation violations.
        self.worker_pool.job_started.connect(
            self._on_job_started, Qt.ConnectionType.QueuedConnection
        )
        self.worker_pool.job_progress.connect(
            self._on_job_progress, Qt.ConnectionType.QueuedConnection
        )
        self.worker_pool.job_completed.connect(
            self._on_job_completed, Qt.ConnectionType.QueuedConnection
        )
        self.worker_pool.job_failed.connect(
            self._on_job_failed, Qt.ConnectionType.QueuedConnection
        )
        self.worker_pool.job_cancelled.connect(
            self._on_job_cancelled, Qt.ConnectionType.QueuedConnection
        )

    def _connect_file_menu(self) -> None:
        """Connect File menu actions."""
        self.menu_bar.new_file_action.triggered.connect(self._on_new_file)
        self.menu_bar.open_file_action.triggered.connect(self._on_open_file)
        self.menu_bar.save_file_action.triggered.connect(self._on_save_file)
        self.menu_bar.save_as_action.triggered.connect(self._on_save_as)
        self.menu_bar.import_action.triggered.connect(self._on_import)
        self.menu_bar.export_action.triggered.connect(self._on_export)
        self.menu_bar.exit_action.triggered.connect(self.close)

    def _connect_edit_menu(self) -> None:
        """Connect Edit menu actions."""
        self.menu_bar.undo_action.triggered.connect(self._on_undo)
        self.menu_bar.redo_action.triggered.connect(self._on_redo)
        self.menu_bar.cut_action.triggered.connect(self._on_cut)
        self.menu_bar.copy_action.triggered.connect(self._on_copy)
        self.menu_bar.paste_action.triggered.connect(self._on_paste)
        self.menu_bar.preferences_action.triggered.connect(self._on_preferences)

    def _connect_view_menu(self) -> None:
        """Connect View menu actions."""
        self.menu_bar.zoom_in_action.triggered.connect(self._on_zoom_in)
        self.menu_bar.zoom_out_action.triggered.connect(self._on_zoom_out)
        self.menu_bar.reset_zoom_action.triggered.connect(self._on_reset_zoom)
        self.menu_bar.view_data_dock_action.triggered.connect(
            lambda: self.sidebar.setVisible(
                self.menu_bar.view_data_dock_action.isChecked()
            )
        )
        self.menu_bar.view_log_dock_action.triggered.connect(
            lambda: self.log_dock.setVisible(
                self.menu_bar.view_log_dock_action.isChecked()
            )
        )
        self.menu_bar.theme_light_action.triggered.connect(
            lambda: self._on_theme_changed("light")
        )
        self.menu_bar.theme_dark_action.triggered.connect(
            lambda: self._on_theme_changed("dark")
        )
        self.menu_bar.theme_auto_action.triggered.connect(
            lambda: self._on_theme_changed("auto")
        )

    def _connect_data_menu(self) -> None:
        """Connect Data menu actions."""
        self.menu_bar.new_dataset_action.triggered.connect(self._on_new_dataset)
        self.menu_bar.delete_dataset_action.triggered.connect(self._on_delete_dataset)
        self.menu_bar.test_mode_oscillation.triggered.connect(
            lambda: self._on_set_test_mode("oscillation")
        )
        self.menu_bar.test_mode_relaxation.triggered.connect(
            lambda: self._on_set_test_mode("relaxation")
        )
        self.menu_bar.test_mode_creep.triggered.connect(
            lambda: self._on_set_test_mode("creep")
        )
        self.menu_bar.test_mode_rotation.triggered.connect(
            lambda: self._on_set_test_mode("rotation")
        )
        self.menu_bar.test_mode_flow_curve.triggered.connect(
            lambda: self._on_set_test_mode("flow_curve")
        )
        self.menu_bar.test_mode_startup.triggered.connect(
            lambda: self._on_set_test_mode("startup")
        )
        self.menu_bar.test_mode_laos.triggered.connect(
            lambda: self._on_set_test_mode("laos")
        )
        self.menu_bar.auto_detect_mode_action.triggered.connect(
            self._on_auto_detect_mode
        )

    def _connect_models_menu(self) -> None:
        """Connect Models menu actions."""
        # Classical models
        self.menu_bar.model_maxwell.triggered.connect(
            lambda: self._on_select_model("maxwell")
        )
        self.menu_bar.model_zener.triggered.connect(
            lambda: self._on_select_model("zener")
        )
        self.menu_bar.model_springpot.triggered.connect(
            lambda: self._on_select_model("springpot")
        )

        # Flow models
        self.menu_bar.model_power_law.triggered.connect(
            lambda: self._on_select_model("power_law")
        )
        self.menu_bar.model_carreau.triggered.connect(
            lambda: self._on_select_model("carreau")
        )
        self.menu_bar.model_carreau_yasuda.triggered.connect(
            lambda: self._on_select_model("carreau_yasuda")
        )
        self.menu_bar.model_cross.triggered.connect(
            lambda: self._on_select_model("cross")
        )
        self.menu_bar.model_herschel_bulkley.triggered.connect(
            lambda: self._on_select_model("herschel_bulkley")
        )
        self.menu_bar.model_bingham.triggered.connect(
            lambda: self._on_select_model("bingham")
        )

        # Fractional Maxwell family
        self.menu_bar.model_fmaxwell_gel.triggered.connect(
            lambda: self._on_select_model("fractional_maxwell_gel")
        )
        self.menu_bar.model_fmaxwell_liquid.triggered.connect(
            lambda: self._on_select_model("fractional_maxwell_liquid")
        )
        self.menu_bar.model_fmaxwell_model.triggered.connect(
            lambda: self._on_select_model("fractional_maxwell_model")
        )
        self.menu_bar.model_fkelvin_voigt.triggered.connect(
            lambda: self._on_select_model("fractional_kelvin_voigt")
        )

        # Fractional Zener family
        self.menu_bar.model_fzener_sl.triggered.connect(
            lambda: self._on_select_model("fractional_zener_sl")
        )
        self.menu_bar.model_fzener_ss.triggered.connect(
            lambda: self._on_select_model("fractional_zener_ss")
        )
        self.menu_bar.model_fzener_ll.triggered.connect(
            lambda: self._on_select_model("fractional_zener_ll")
        )
        self.menu_bar.model_fkv_zener.triggered.connect(
            lambda: self._on_select_model("fractional_kv_zener")
        )

        # Advanced fractional models
        self.menu_bar.model_fburgers.triggered.connect(
            lambda: self._on_select_model("fractional_burgers")
        )
        self.menu_bar.model_fpoynting.triggered.connect(
            lambda: self._on_select_model("fractional_poynting_thomson")
        )
        self.menu_bar.model_fjeffreys.triggered.connect(
            lambda: self._on_select_model("fractional_jeffreys")
        )

        # Multi-mode models
        self.menu_bar.model_gmaxwell.triggered.connect(
            lambda: self._on_select_model("generalized_maxwell")
        )

        # SGR models
        self.menu_bar.model_sgr_conventional.triggered.connect(
            lambda: self._on_select_model("sgr_conventional")
        )
        self.menu_bar.model_sgr_generic.triggered.connect(
            lambda: self._on_select_model("sgr_generic")
        )

        # SPP LAOS models
        self.menu_bar.model_spp_yield_stress.triggered.connect(
            lambda: self._on_select_model("spp_yield_stress")
        )

        # STZ models
        self.menu_bar.model_stz_conventional.triggered.connect(
            lambda: self._on_select_model("stz_conventional")
        )

        # EPM models
        self.menu_bar.model_lattice_epm.triggered.connect(
            lambda: self._on_select_model("lattice_epm")
        )
        self.menu_bar.model_tensorial_epm.triggered.connect(
            lambda: self._on_select_model("tensorial_epm")
        )

        # Fluidity models
        self.menu_bar.model_fluidity_local.triggered.connect(
            lambda: self._on_select_model("fluidity_local")
        )
        self.menu_bar.model_fluidity_nonlocal.triggered.connect(
            lambda: self._on_select_model("fluidity_nonlocal")
        )

        # Fluidity-Saramito EVP models
        self.menu_bar.model_saramito_local.triggered.connect(
            lambda: self._on_select_model("fluidity_saramito_local")
        )
        self.menu_bar.model_saramito_nonlocal.triggered.connect(
            lambda: self._on_select_model("fluidity_saramito_nonlocal")
        )

        # IKH models
        self.menu_bar.model_mikh.triggered.connect(
            lambda: self._on_select_model("mikh")
        )
        self.menu_bar.model_mlikh.triggered.connect(
            lambda: self._on_select_model("ml_ikh")
        )

        # FIKH models
        self.menu_bar.model_fikh.triggered.connect(
            lambda: self._on_select_model("fikh")
        )
        self.menu_bar.model_fmlikh.triggered.connect(
            lambda: self._on_select_model("fmlikh")
        )

        # Hébraud-Lequeux
        self.menu_bar.model_hebraud_lequeux.triggered.connect(
            lambda: self._on_select_model("hebraud_lequeux")
        )

        # ITT-MCT models
        self.menu_bar.model_itt_mct_schematic.triggered.connect(
            lambda: self._on_select_model("itt_mct_schematic")
        )
        self.menu_bar.model_itt_mct_isotropic.triggered.connect(
            lambda: self._on_select_model("itt_mct_isotropic")
        )

        # DMT models
        self.menu_bar.model_dmt_local.triggered.connect(
            lambda: self._on_select_model("dmt_local")
        )
        self.menu_bar.model_dmt_nonlocal.triggered.connect(
            lambda: self._on_select_model("dmt_nonlocal")
        )

        # Giesekus models
        self.menu_bar.model_giesekus_single.triggered.connect(
            lambda: self._on_select_model("giesekus_single")
        )
        self.menu_bar.model_giesekus_multi.triggered.connect(
            lambda: self._on_select_model("giesekus_multi")
        )

        # TNT models
        self.menu_bar.model_tnt_single_mode.triggered.connect(
            lambda: self._on_select_model("tnt_single_mode")
        )
        self.menu_bar.model_tnt_cates.triggered.connect(
            lambda: self._on_select_model("tnt_cates")
        )
        self.menu_bar.model_tnt_loop_bridge.triggered.connect(
            lambda: self._on_select_model("tnt_loop_bridge")
        )
        self.menu_bar.model_tnt_multi_species.triggered.connect(
            lambda: self._on_select_model("tnt_multi_species")
        )
        self.menu_bar.model_tnt_sticky_rouse.triggered.connect(
            lambda: self._on_select_model("tnt_sticky_rouse")
        )

        # VLB models
        self.menu_bar.model_vlb_local.triggered.connect(
            lambda: self._on_select_model("vlb_local")
        )
        self.menu_bar.model_vlb_multi_network.triggered.connect(
            lambda: self._on_select_model("vlb_multi_network")
        )
        self.menu_bar.model_vlb_variant.triggered.connect(
            lambda: self._on_select_model("vlb_variant")
        )
        self.menu_bar.model_vlb_nonlocal.triggered.connect(
            lambda: self._on_select_model("vlb_nonlocal")
        )

        # HVM models
        self.menu_bar.model_hvm_local.triggered.connect(
            lambda: self._on_select_model("hvm_local")
        )

        # HVNM models
        self.menu_bar.model_hvnm_local.triggered.connect(
            lambda: self._on_select_model("hvnm_local")
        )

    def _connect_transforms_menu(self) -> None:
        """Connect Transforms menu actions."""
        self.menu_bar.transform_fft.triggered.connect(
            lambda: self._on_apply_transform("fft")
        )
        self.menu_bar.transform_mastercurve.triggered.connect(
            lambda: self._on_apply_transform("mastercurve")
        )
        self.menu_bar.transform_srfs.triggered.connect(
            lambda: self._on_apply_transform("srfs")
        )
        self.menu_bar.transform_mutation.triggered.connect(
            lambda: self._on_apply_transform("mutation_number")
        )
        self.menu_bar.transform_owchirp.triggered.connect(
            lambda: self._on_apply_transform("owchirp")
        )
        self.menu_bar.transform_derivatives.triggered.connect(
            lambda: self._on_apply_transform("derivative")
        )
        self.menu_bar.transform_spp.triggered.connect(
            lambda: self._on_apply_transform("spp")
        )
        self.menu_bar.transform_cox_merz.triggered.connect(
            lambda: self._on_apply_transform("cox_merz")
        )
        self.menu_bar.transform_lve_envelope.triggered.connect(
            lambda: self._on_apply_transform("lve_envelope")
        )
        self.menu_bar.transform_prony.triggered.connect(
            lambda: self._on_apply_transform("prony_conversion")
        )
        self.menu_bar.transform_spectrum.triggered.connect(
            lambda: self._on_apply_transform("spectrum_inversion")
        )

    def _connect_analysis_menu(self) -> None:
        """Connect Analysis menu actions."""
        self.menu_bar.analysis_fit_nlsq.triggered.connect(self._on_fit)
        self.menu_bar.analysis_fit_bayesian.triggered.connect(self._on_bayesian)
        self.menu_bar.analysis_batch_fit.triggered.connect(self._on_batch_fit)
        self.menu_bar.analysis_compare.triggered.connect(self._on_compare_models)
        self.menu_bar.analysis_compatibility.triggered.connect(
            self._on_check_compatibility
        )

    def _connect_pipeline_menu(self) -> None:
        """Connect Pipeline menu actions."""
        self.menu_bar.pipeline_new_action.triggered.connect(self._on_new_pipeline)
        self.menu_bar.pipeline_open_action.triggered.connect(self._on_open_pipeline)
        self.menu_bar.pipeline_save_action.triggered.connect(self._on_save_pipeline)
        for key, action in self.menu_bar.pipeline_template_actions.items():
            action.triggered.connect(
                lambda _checked=False, k=key: self._on_load_pipeline_template(k)
            )

    def _on_load_pipeline_template(self, template_key: str) -> None:
        """Load a pipeline from a built-in template.

        Parameters
        ----------
        template_key : str
            Key identifying the template (e.g. ``"basic"``).
        """
        logger.debug("Pipeline template requested", template=template_key)
        try:
            from rheojax.cli._templates import get_template
            from rheojax.gui.state.actions import load_pipeline
            from rheojax.gui.state.store import VisualPipelineState
            from rheojax.gui.utils.pipeline_serializer import from_yaml

            yaml_str = get_template(template_key)
            steps, name = from_yaml(yaml_str)
            vp = VisualPipelineState(steps=steps, pipeline_name=name)
            load_pipeline(vp)
            self.log(f"Loaded pipeline template: {template_key}")
            self.status_bar.show_message(f"Template loaded: {template_key}", 2000)
        except KeyError:
            logger.warning("Unknown pipeline template", template=template_key)
            self.status_bar.show_message(f"Unknown template: {template_key}", 3000)
        except Exception as exc:
            logger.error(
                "Failed to load template",
                template=template_key,
                error=str(exc),
                exc_info=True,
            )
            self.status_bar.show_message(f"Template load failed: {exc}", 5000)

    def _connect_tools_menu(self) -> None:
        """Connect Tools menu actions."""
        self.menu_bar.tools_console.triggered.connect(self._on_python_console)
        self.menu_bar.tools_jax_profiler.triggered.connect(self._on_jax_profiler)
        self.menu_bar.tools_memory_monitor.triggered.connect(self._on_memory_monitor)
        self.menu_bar.tools_preferences.triggered.connect(self._on_preferences)

    def _connect_help_menu(self) -> None:
        """Connect Help menu actions."""
        self.menu_bar.help_docs.triggered.connect(self._on_open_docs)
        self.menu_bar.help_tutorials.triggered.connect(self._on_open_tutorials)
        self.menu_bar.help_shortcuts.triggered.connect(self._on_show_shortcuts)
        self.menu_bar.help_about.triggered.connect(self._on_about)

    def navigate_to(self, page_name: str) -> None:
        """Navigate to specified page.

        Parameters
        ----------
        page_name : str
            Target page identifier (home, data, transform, fit, bayesian, diagnostics, export)
        """
        # GUI-013: prevent re-entrant calls (e.g. sidebar selection emitting
        # step_selected which calls navigate_to again).
        # NOTE: This guard assumes single-threaded (GUI main thread) access.
        # navigate_to must never be called from a worker thread.
        assert (
            QApplication.instance() is None
            or QApplication.instance().thread() is None
            # Only enforce the GUI-thread check when a QApplication exists.
            # In test environments the application thread may differ.
            or threading.current_thread() is threading.main_thread()
        ), "navigate_to() called from a non-GUI thread — this is unsafe"
        if self._navigating:
            return
        self._navigating = True
        try:
            # Map public page names to WorkspaceContainer step_type keys.
            # WorkspaceContainer uses "load" for DataPage and None for HomePage.
            _page_to_step: dict[str, str | None] = {
                "home": None,
                "data": "load",
                "transform": "transform",
                "fit": "fit",
                "bayesian": "bayesian",
                "diagnostics": "diagnostics",
                "export": "export",
            }
            name = page_name.lower()
            if name in _page_to_step:
                step_type = _page_to_step[name]
                to_page = name.capitalize()
                self.workspace.show_step(step_type)
                # GUI-008: sidebar items are UUID-keyed; select_step_by_type
                # iterates ROLE_STEP_TYPE data to find the correct item.
                self.sidebar.select_step_by_type(step_type or "")
                # Navigation is handled by workspace.show_step() above;
                # no dispatch needed (SET_TAB had no reducer).
                logger.info("Page navigated", to_page=to_page)
                self.log(f"Navigated to {to_page} page")
        finally:
            self._navigating = False

    def log(self, message: str) -> None:
        """Append message to log panel.

        Parameters
        ----------
        message : str
            Log message
        """
        self.log_panel.append(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event with cleanup.

        Parameters
        ----------
        event : QCloseEvent
            Close event
        """
        logger.debug("Application shutting down")
        # Check for unsaved changes
        if self._has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )

            if reply == QMessageBox.StandardButton.Save:
                # GUI-P1-003 fix: only accept if save succeeds.  _on_save_file
                # returns True when the save completes (or no changes needed)
                # and False when the save is not implemented / failed.
                saved = self._on_save_file()
                if saved:
                    event.accept()
                else:
                    # Save failed or not implemented — give the user a second
                    # chance to explicitly discard or cancel, so data is never
                    # silently lost.
                    discard_reply = QMessageBox.question(
                        self,
                        "Save Failed",
                        "Save could not be completed.\n\n"
                        "Discard unsaved changes and close anyway?",
                        QMessageBox.StandardButton.Discard
                        | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Cancel,
                    )
                    if discard_reply == QMessageBox.StandardButton.Discard:
                        event.accept()
                    else:
                        logger.debug("Shutdown cancelled after failed save")
                        event.ignore()
                        return
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                logger.debug("Shutdown cancelled by user")
                event.ignore()
                return

        # Cleanup: stop workers and disconnect state signals to prevent
        # callbacks during teardown
        state = self.store.get_state()
        dataset_count = len(state.datasets) if state.datasets else 0
        fit_count = len(state.fit_results) if state.fit_results else 0
        logger.info(
            "Application closing",
            datasets=dataset_count,
            fit_results=fit_count,
        )
        self.log("Shutting down...")

        # Unsubscribe state store callback to prevent post-destroy invocation
        try:
            self.store.unsubscribe(self._on_state_changed)
        except Exception:
            logger.debug("State unsubscribe failed during shutdown", exc_info=True)

        # Notify child pages to disconnect worker signals BEFORE shutdown
        # kills subprocesses.  This prevents queued signals from being
        # delivered to destroyed child widgets (segfault root cause).
        try:
            if hasattr(self, "bayesian_page") and hasattr(
                self.bayesian_page, "prepare_for_close"
            ):
                self.bayesian_page.prepare_for_close()
        except Exception:
            logger.debug("BayesianPage close prep failed", exc_info=True)

        # Attempt graceful worker shutdown with a generous timeout.
        # If workers are still running after the timeout (e.g. NUTS sampling
        # stuck in JIT-compiled XLA code that cannot be interrupted), schedule
        # a hard os._exit() to prevent the process from hanging indefinitely
        # when QThreadPool's destructor tries an infinite waitForDone().
        _workers_stopped = True
        try:
            if hasattr(self, "worker_pool"):
                self.worker_pool.shutdown(wait=True, timeout_ms=10000)
                if self.worker_pool.is_busy():
                    _workers_stopped = False
        except Exception:
            logger.debug("Worker pool shutdown failed during cleanup", exc_info=True)

        if not _workers_stopped:
            logger.warning(
                "Workers still running after shutdown timeout — "
                "scheduling forced exit to prevent zombie process"
            )

            # Give Qt a brief moment to drain signals, then force-exit.
            # os._exit() bypasses QThreadPool destructor's infinite wait.
            def _force_exit():
                logger.warning("Force-exiting process (stuck workers)")
                os._exit(0)

            _force_timer = threading.Timer(2.0, _force_exit)
            _force_timer.daemon = True
            _force_timer.start()

        # R10-MW-003: Disconnect only THIS window's signal connections to prevent
        # delivery to destroyed widgets during Qt object teardown.  We disconnect
        # specific slots rather than calling .disconnect() with no args, which
        # would strip ALL connections on the singleton StateSignals — breaking
        # any other objects (BayesianPage, DataPage) still connected.
        try:
            signals = self.store._signals
            if signals is not None:
                _slot_map = [
                    ("theme_changed", self._apply_theme),
                    ("dataset_added", self._on_dataset_added),
                    ("pipeline_step_changed", self._on_pipeline_step_changed),
                ]
                for signal_name, slot in _slot_map:
                    if hasattr(signals, signal_name):
                        try:
                            getattr(signals, signal_name).disconnect(slot)
                        except (TypeError, RuntimeError):
                            pass
        except Exception:
            logger.debug("Signal disconnect failed during shutdown", exc_info=True)

        # Drain pending queued events with a timed loop to ensure all
        # already-posted cross-thread signals are delivered (and dropped
        # by the _closing guard) before widget destruction begins.
        from rheojax.gui.compat import QApplication

        _drain_deadline = time.monotonic() + 0.2  # 200 ms max
        while time.monotonic() < _drain_deadline:
            QApplication.processEvents()
            time.sleep(0.005)  # yield CPU between drain iterations

        logger.info("Application shutdown complete")
        event.accept()

    @Slot(object)
    def _on_state_changed(self, state) -> None:
        """Handle state changes from store.

        Parameters
        ----------
        state : AppState
            Updated application state
        """
        # Update UI based on state changes
        # This is called whenever the store state updates

        # Update window title if project name changes
        if state.project_name:
            self.setWindowTitle(f"RheoJAX - {state.project_name}")

        # Update dirty state
        if state.is_modified != self._has_unsaved_changes:
            self._has_unsaved_changes = state.is_modified
            self.state_dirty.emit(self._has_unsaved_changes)

        # Sync pipeline chips
        self._update_pipeline_chips_from_state(state)

        # Update workflow mode
        if state.workflow_mode != self._current_workflow_mode:
            previous_mode = self._current_workflow_mode
            self._current_workflow_mode = state.workflow_mode
            self._update_tabs_visibility(self._current_workflow_mode)
            logger.info(
                "Workflow mode changed",
                from_mode=previous_mode.name,
                to_mode=self._current_workflow_mode.name,
            )
            self.log(f"Switched to {self._current_workflow_mode.name.title()} Workflow")

    # -------------------------------------------------------------------------
    # File Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_new_file(self) -> None:
        """Handle new file action."""
        logger.debug("New file action triggered")
        if self._has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before creating a new project?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_file()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self.store.dispatch("NEW_PROJECT")
        self.log("Created new project")
        logger.info("New project created")
        self.status_bar.show_message("New project created", 3000)

    @Slot()
    def _on_open_file(self) -> None:
        """Handle open file action."""
        logger.debug("Open file action triggered")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "RheoJAX Project (*.rheojax);;HDF5 Files (*.h5 *.hdf5);;All Files (*.*)",
        )
        if file_path:
            self.log(f"Opening project: {file_path}")
            logger.info("Opening project", file_path=file_path)
            self.store.dispatch("LOAD_PROJECT", {"file_path": file_path})
            self.status_bar.show_message(f"Opened: {file_path}", 3000)

    @Slot()
    def _on_save_file(self) -> bool:
        """Handle save file action.

        Returns
        -------
        bool
            True if the save succeeded (or there was nothing to save),
            False if the save could not be completed.  Callers should check
            this return value before accepting a close event to prevent
            silent data loss (GUI-P1-003).
        """
        logger.debug("Save file action triggered")

        if not self._has_unsaved_changes:
            self.status_bar.show_message("No unsaved changes", 2000)
            return True  # Nothing to save — treat as success

        # TODO: Implement project serialization (state → JSON/HDF5).
        # For now, inform the user that save is not yet implemented and
        # return False so that closeEvent can offer an explicit Discard option
        # rather than silently accepting the close (GUI-P1-003 fix).
        from rheojax.gui.compat import QMessageBox

        QMessageBox.information(
            self,
            "Save Not Yet Implemented",
            "Project save is not yet implemented.\n"
            "Use 'Export Results' from the Bayesian or Fit tabs to save individual results.",
        )
        logger.warning("Save file action: not yet implemented")
        return False

    @Slot()
    def _on_save_as(self) -> None:
        """Handle save as action."""
        logger.debug("Save as action triggered")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "RheoJAX Project (*.rheojax);;HDF5 Files (*.h5 *.hdf5);;All Files (*.*)",
        )
        if file_path:
            self.log(f"Saving project as: {file_path}")
            logger.info("Project saved as", file_path=file_path)
            self.store.dispatch("SAVE_PROJECT", {"file_path": file_path})
            self.status_bar.show_message(f"Saved as: {file_path}", 3000)
            self._has_unsaved_changes = False

    @Slot(Path)
    def _on_open_recent_project(self, project_path: Path) -> None:
        """Handle opening a recent project from the home page."""
        path = Path(project_path)
        if not path.exists():
            logger.warning("Recent project not found", path=str(path))
            QMessageBox.warning(
                self,
                "Project Not Found",
                f"Recent project is missing:\n{path}",
            )
            return

        self.log(f"Opening recent project: {path}")
        logger.info("Opening recent project", file_path=str(path))
        self.store.dispatch("LOAD_PROJECT", {"file_path": str(path)})
        self.navigate_to("data")
        self.status_bar.show_message(f"Opened: {path.name}", 3000)

    @Slot()
    def _on_import(self) -> None:
        """Handle import data action."""
        logger.debug("Import action triggered")
        wizard = ImportWizard(self)
        if wizard.exec():
            config = wizard.get_result()
            # Pre-generate dataset_id for consistent state tracking
            config["dataset_id"] = str(uuid.uuid4())
            self.log(f"Importing data from: {config['file_path']}")
            logger.info(
                "Importing data",
                file_path=config["file_path"],
                dataset_id=config["dataset_id"],
            )
            self.store.dispatch("IMPORT_DATA", config)

            # Defer the blocking I/O to the next event loop iteration so the
            # dialog close animation is not held up.
            # Guard prevents overlapping imports from rapid double-clicks.
            if getattr(self, "_import_in_progress", False):
                logger.warning("Import already in progress, ignoring duplicate")
                return
            self._import_in_progress = True

            from rheojax.gui.compat import QTimer

            QTimer.singleShot(0, lambda cfg=config: self._do_import(cfg))

    def _do_import(self, config: dict) -> None:
        """Perform the deferred file import (blocking I/O)."""
        # R8-NEW-003: TODO — move blocking I/O into ImportWorker to avoid
        # stalling the main thread on large files.
        # Interim: yield to event loop at key checkpoints below.
        try:
            from rheojax.gui.services.data_service import DataService

            service = DataService()
            QApplication.processEvents()  # Keep UI responsive during service init
            rheo_data = service.load_file(
                file_path=config["file_path"],
                x_col=config.get("x_column"),
                y_col=config.get("y_column"),
                y2_col=config.get("y2_column"),
                test_mode=config.get("test_mode"),
            )

            # Auto-detect test mode if requested
            test_mode = config.get("test_mode")
            if config.get("auto_detect_mode"):
                test_mode = service.detect_test_mode(rheo_data)

            self.store.dispatch(
                "IMPORT_DATA_SUCCESS",
                {
                    "dataset_id": config["dataset_id"],
                    "file_path": str(config["file_path"]),
                    "name": Path(config["file_path"]).stem,
                    "test_mode": test_mode if test_mode is not None else "unknown",
                    "x_data": rheo_data.x,
                    "y_data": rheo_data.y,
                    "y2_data": getattr(rheo_data, "y2", None),
                    "metadata": getattr(rheo_data, "metadata", {}),
                },
            )
            self.navigate_to("data")
            logger.info(
                "Data import successful",
                dataset_id=config["dataset_id"],
                test_mode=test_mode,
            )
            self.status_bar.show_message("Data imported successfully", 3000)
        except Exception as exc:
            logger.error("Import failed", error=str(exc), exc_info=True)
            self.log(f"Import failed: {exc}")
            self.store.dispatch("IMPORT_DATA_FAILED", {"error": str(exc), **config})
            self.status_bar.show_message(f"Import failed: {exc}", 5000)
        finally:
            self._import_in_progress = False

    @Slot()
    def _on_export(self) -> None:
        """Handle export action."""
        logger.debug("Export action triggered")
        self.log("Opening export page...")
        self.navigate_to("export")
        self.status_bar.show_message("Configure export options", 2000)

        # Quick export of current fit parameters if available
        try:
            from rheojax.gui.services.export_service import ExportService

            state = self.store.get_state()
            fit_result = (
                next(iter(state.fit_results.values()), None)
                if state.fit_results
                else None
            )
            if fit_result:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Export Fit Parameters", "", "CSV (*.csv);;JSON (*.json)"
                )
                if path:
                    ExportService().export_parameters(fit_result, path)
                    logger.info("Fit parameters exported", path=path)
                    self.status_bar.show_message(
                        f"Exported fit parameters to {Path(path).name}", 3000
                    )
        except Exception as exc:
            logger.error("Export not available", error=str(exc), exc_info=True)
            self.log(f"Export not available: {exc}")

    # -------------------------------------------------------------------------
    # Edit Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_undo(self) -> None:
        """Handle undo action."""
        logger.debug("Undo action triggered")
        self.store.dispatch("UNDO")
        self.log("Undo")
        self.status_bar.show_message("Undo", 2000)

    @Slot()
    def _on_redo(self) -> None:
        """Handle redo action."""
        logger.debug("Redo action triggered")
        self.store.dispatch("REDO")
        self.log("Redo")
        self.status_bar.show_message("Redo", 2000)

    @Slot()
    def _on_cut(self) -> None:
        """Handle cut action."""
        focused = QApplication.focusWidget()
        if focused and hasattr(focused, "cut"):
            focused.cut()
            self.log("Cut to clipboard")

    @Slot()
    def _on_copy(self) -> None:
        """Handle copy action."""
        focused = QApplication.focusWidget()
        if focused and hasattr(focused, "copy"):
            focused.copy()
            self.log("Copied to clipboard")

    @Slot()
    def _on_paste(self) -> None:
        """Handle paste action."""
        focused = QApplication.focusWidget()
        if focused and hasattr(focused, "paste"):
            focused.paste()
            self.log("Pasted from clipboard")

    @Slot()
    def _on_preferences(self) -> None:
        """Handle preferences action."""
        logger.debug("Preferences action triggered")
        dialog = PreferencesDialog(parent=self)
        if dialog.exec():
            prefs = dialog.get_preferences()
            self.store.dispatch("UPDATE_PREFERENCES", prefs)
            self.log("Preferences updated")
            logger.info("Preferences updated")
            self.status_bar.show_message("Preferences saved", 3000)

    # -------------------------------------------------------------------------
    # View Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_zoom_in(self) -> None:
        """Handle zoom in action."""
        current_page = self.workspace.get_current_page()
        if hasattr(current_page, "zoom_in"):
            current_page.zoom_in()
        self.log("Zoom in")
        self.status_bar.show_message("Zoom in", 1000)

    @Slot()
    def _on_zoom_out(self) -> None:
        """Handle zoom out action."""
        current_page = self.workspace.get_current_page()
        if hasattr(current_page, "zoom_out"):
            current_page.zoom_out()
        self.log("Zoom out")
        self.status_bar.show_message("Zoom out", 1000)

    @Slot()
    def _on_reset_zoom(self) -> None:
        """Handle reset zoom action."""
        current_page = self.workspace.get_current_page()
        if hasattr(current_page, "reset_zoom"):
            current_page.reset_zoom()
        self.log("Zoom reset")
        self.status_bar.show_message("Zoom reset", 1000)

    # -------------------------------------------------------------------------
    # Sidebar and Pipeline Handlers
    # -------------------------------------------------------------------------

    @Slot(str)
    def _on_sidebar_step_selected(self, step_id: str) -> None:
        """Navigate to the workspace page matching the sidebar step selection.

        The sidebar emits a step_id corresponding to a VisualPipelineStep UUID.
        We retrieve the step's step_type from state and map it to a page name.

        Parameters
        ----------
        step_id : str
            Step identifier from PipelineSidebar (UUID or step_type key).
        """
        logger.debug("Sidebar step selected", step_id=step_id)
        # step_id may be a UUID for a VisualPipelineStep or a raw step_type.
        # Try to look up by UUID first, then fall back to treating it as step_type.
        _step_type_to_page = {
            "load": "data",
            "transform": "transform",
            "fit": "fit",
            "bayesian": "bayesian",
            "export": "export",
        }
        try:
            from rheojax.gui.state import selectors

            step = selectors.get_pipeline_step_by_id(step_id)
            if step is not None:
                step_type = step.step_type
            else:
                # GUI-015: step not found — fall back to home rather than
                # using the raw UUID as a page name (which would never match).
                logger.warning(
                    "Sidebar step_id not found in pipeline state, falling back to home",
                    step_id=step_id,
                )
                self.status_bar.show_message(
                    "Pipeline step not found — navigating to Home", 3000
                )
                step_type = "home"
        except Exception:
            step_type = step_id

        page_name = _step_type_to_page.get(step_type, step_type)
        self.navigate_to(page_name)

    @Slot(str)
    def _auto_populate_step_config(self, step_id: str) -> None:
        """Auto-populate a newly-added pipeline step's config from current UI state.

        Called when a step is added via the sidebar.  Reads the current model
        selection (FitPage), transform selection (TransformPage), active dataset
        file path (DataPage), and Bayesian settings (BayesianPage) to fill the
        step's config dict so that pipeline execution has the required keys.

        Parameters
        ----------
        step_id : str
            UUID of the newly-added step.
        """
        try:
            from rheojax.gui.state import selectors
            from rheojax.gui.state.actions import update_step_config

            step = selectors.get_pipeline_step_by_id(step_id)
            if step is None:
                return

            config: dict[str, Any] = {}

            if step.step_type == "load":
                dataset = self.store.get_active_dataset()
                if dataset and dataset.file_path:
                    config["file"] = str(dataset.file_path)
                    if dataset.test_mode:
                        config["test_mode"] = dataset.test_mode

            elif step.step_type == "transform":
                tp = self.transform_page
                if tp._selected_key:
                    config["name"] = tp._selected_key
                    params = tp.get_selected_params()
                    config.update(params)

            elif step.step_type == "fit":
                state = self.store.get_state()
                if state.active_model_name:
                    config["model"] = state.active_model_name
                if state.model_params:
                    config["params"] = {
                        name: p.value for name, p in state.model_params.items()
                    }
                # Include test mode and deformation mode
                test_mode = getattr(state, "test_mode", None)
                if test_mode:
                    config["test_mode"] = test_mode
                dm = getattr(state, "deformation_mode", None)
                if dm:
                    config["deformation_mode"] = dm
                pr = getattr(state, "poisson_ratio", None)
                if pr is not None:
                    config["poisson_ratio"] = pr

            elif step.step_type == "bayesian":
                state = self.store.get_state()
                if state.active_model_name:
                    config["model"] = state.active_model_name
                # Pull Bayesian settings from the BayesianPage if available
                try:
                    bp = self.bayesian_page
                    if hasattr(bp, "_sampler_combo"):
                        config["num_warmup"] = 1000
                        config["num_samples"] = 2000
                        config["num_chains"] = 4
                except Exception:
                    pass

            elif step.step_type == "export":
                config["format"] = "directory"
                config["output"] = "results/"

            if config:
                update_step_config(step_id, config)
                logger.debug(
                    "Auto-populated step config",
                    step_id=step_id,
                    step_type=step.step_type,
                    config_keys=list(config.keys()),
                )

        except Exception:
            logger.debug(
                "Could not auto-populate step config",
                step_id=step_id,
                exc_info=True,
            )

    @Slot()
    def _on_new_pipeline(self) -> None:
        """Clear the current visual pipeline state."""
        logger.debug("New pipeline action triggered")
        from rheojax.gui.state.actions import clear_pipeline

        clear_pipeline()
        self.pipeline_chips.reset()
        self.log("Pipeline cleared")
        self.status_bar.show_message("New pipeline created", 2000)

    @Slot(str)
    def _sync_fit_step_config(self, _model_name: str = "") -> None:
        """Update the selected pipeline step's config when model selection changes.

        Called when model_selected or model_params_changed signals fire.
        Only acts if the currently-selected pipeline step is a 'fit' or
        'bayesian' step.
        """
        try:
            from rheojax.gui.state import selectors
            from rheojax.gui.state.actions import update_step_config

            step = selectors.get_selected_pipeline_step()
            if step is None or step.step_type not in ("fit", "bayesian"):
                return

            state = self.store.get_state()
            config: dict[str, Any] = {}
            if state.active_model_name:
                config["model"] = state.active_model_name
            if state.model_params:
                config["params"] = {
                    name: p.value for name, p in state.model_params.items()
                }

            if config:
                update_step_config(step.id, config)
        except Exception:
            logger.debug("Could not sync fit step config", exc_info=True)

    def _sync_transform_step_config(self) -> None:
        """Update the selected pipeline step's config when transform selection changes.

        Called from TransformPage transform_selected signal.
        Only acts if the currently-selected pipeline step is a 'transform' step.
        """
        try:
            from rheojax.gui.state import selectors
            from rheojax.gui.state.actions import update_step_config

            step = selectors.get_selected_pipeline_step()
            if step is None or step.step_type != "transform":
                return

            tp = self.transform_page
            if not tp._selected_key:
                return

            config: dict[str, Any] = {"name": tp._selected_key}
            config.update(tp.get_selected_params())
            update_step_config(step.id, config)
        except Exception:
            logger.debug("Could not sync transform step config", exc_info=True)

    @Slot()
    def _on_open_pipeline(self) -> None:
        """Open a pipeline from a YAML file."""
        logger.debug("Open pipeline action triggered")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pipeline", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            try:
                from rheojax.gui.state.actions import load_pipeline
                from rheojax.gui.state.store import VisualPipelineState
                from rheojax.gui.utils.pipeline_serializer import from_yaml

                with open(path) as f:
                    steps, name = from_yaml(f.read())
                vp = VisualPipelineState(steps=steps, pipeline_name=name)
                load_pipeline(vp)
                self.log(f"Pipeline loaded from: {path}")
                logger.info("Pipeline loaded", path=path)
                self.status_bar.show_message(
                    f"Pipeline loaded: {Path(path).name}", 3000
                )
            except Exception as exc:
                logger.error("Failed to open pipeline", error=str(exc), exc_info=True)
                self.log(f"Failed to open pipeline: {exc}")
                self.status_bar.show_message(f"Open pipeline failed: {exc}", 5000)

    @Slot()
    def _on_save_pipeline(self) -> None:
        """Save the current pipeline to a YAML file."""
        logger.debug("Save pipeline action triggered")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "pipeline.yaml", "YAML Files (*.yaml *.yml)"
        )
        if path:
            try:
                from rheojax.gui.state.selectors import (
                    get_pipeline_name,
                    get_visual_pipeline_steps,
                )
                from rheojax.gui.utils.pipeline_serializer import to_yaml

                yaml_str = to_yaml(get_visual_pipeline_steps(), get_pipeline_name())
                with open(path, "w") as f:
                    f.write(yaml_str)
                self.log(f"Pipeline saved to: {path}")
                logger.info("Pipeline saved", path=path)
                self.status_bar.show_message(f"Pipeline saved: {Path(path).name}", 3000)
            except Exception as exc:
                logger.error("Failed to save pipeline", error=str(exc), exc_info=True)
                self.log(f"Failed to save pipeline: {exc}")
                self.status_bar.show_message(f"Save pipeline failed: {exc}", 5000)

    @Slot()
    def _on_run_all(self) -> None:
        """Execute all pipeline steps in sequence on a background thread.

        PipelineExecutionService runs fit and Bayesian steps synchronously —
        calling it on the main thread would freeze the Qt event loop.  We
        submit a QRunnable to the global thread pool and route the result
        back to the main thread via typed Qt signals on a QObject relay.
        """
        logger.debug("Run all pipeline steps requested")
        try:
            from rheojax.gui.state.selectors import get_visual_pipeline_steps

            steps = get_visual_pipeline_steps()
        except Exception as exc:
            logger.error(
                "Could not retrieve pipeline steps", error=str(exc), exc_info=True
            )
            steps = []

        if not steps:
            self.status_bar.show_message("No pipeline steps to run", 2000)
            return

        from rheojax.gui.compat import QObject, QRunnable, QThreadPool, Signal

        class _Relay(QObject):
            finished = Signal()
            error = Signal(str)

        relay = _Relay()
        relay.finished.connect(
            lambda: self.status_bar.show_message("Pipeline completed", 3000),
            Qt.ConnectionType.QueuedConnection,
        )
        relay.error.connect(
            self._on_pipeline_run_error, Qt.ConnectionType.QueuedConnection
        )

        # Keep relay alive until signals are delivered — prevent premature GC
        # when the QRunnable completes and releases its closure references.
        # Use a set so concurrent runs don't clobber each other's relay.
        self._active_relays.add(relay)

        _steps = steps
        _relay = relay
        _self = self

        class _RunAllWorker(QRunnable):
            def run(self) -> None:  # type: ignore[override]
                try:
                    from rheojax.gui.services.pipeline_execution_service import (
                        PipelineExecutionService,
                    )

                    # parent=None: QObject parent must live in the same thread.
                    # The window lives on the main thread; this runs on a pool
                    # thread, so cross-thread parenting is forbidden by Qt.
                    service = PipelineExecutionService(None)
                    service.execute_all(_steps)
                    _relay.finished.emit()
                except Exception as exc:
                    logger.error(
                        "Pipeline execution failed", error=str(exc), exc_info=True
                    )
                    _relay.error.emit(str(exc))
                finally:
                    _self._active_relays.discard(_relay)

        worker = _RunAllWorker()
        worker.setAutoDelete(True)
        QThreadPool.globalInstance().start(worker)
        self.status_bar.show_message("Running pipeline…", 0)

    @Slot(str)
    def _on_run_step(self, step_id: str) -> None:
        """Execute a single pipeline step with accumulated context.

        Builds context from all preceding steps' cached results, then
        executes only the target step via PipelineExecutionService on a
        background thread.

        Parameters
        ----------
        step_id : str
            Identifier of the step to execute.
        """
        logger.debug("Run single step requested", step_id=step_id)

        try:
            from rheojax.gui.services.pipeline_execution_service import (
                PipelineExecutionService,
            )
            from rheojax.gui.state.selectors import get_visual_pipeline_steps

            steps = get_visual_pipeline_steps()
        except Exception as exc:
            logger.error(
                "Could not retrieve pipeline steps", error=str(exc), exc_info=True
            )
            self.status_bar.show_message(f"Run step failed: {exc}", 5000)
            return

        # Find the target step and build context from prior completed steps.
        target_step = None
        prior_steps = []
        for s in steps:
            if s.id == step_id:
                target_step = s
                break
            prior_steps.append(s)

        if target_step is None:
            self.status_bar.show_message(f"Step '{step_id}' not found", 3000)
            return

        # Reconstruct context by replaying prior steps' cached results.
        context: dict = {}
        store = self.store
        vp = store.get_state().visual_pipeline
        for prior in prior_steps:
            cached = vp.step_results.get(prior.id)
            if cached is None:
                continue
            # Mirror PipelineExecutionService context accumulation.
            if prior.step_type == "load":
                context["data"] = cached
            elif prior.step_type == "transform":
                context["data"] = cached
            elif prior.step_type == "fit":
                context["fit_result"] = cached
                context["model_name"] = prior.config.get("model", "")
            elif prior.step_type == "bayesian":
                context["bayesian_result"] = cached
                # Recover model_name from prior Bayesian step config.
                if not context.get("model_name"):
                    context["model_name"] = prior.config.get("model", "")
            elif prior.step_type == "export":
                context["export_path"] = cached

        # GUI-007: If running a standalone Bayesian step after a cached fit,
        # model_name may be absent. Recover it from the fit result object.
        if not context.get("model_name") and context.get("fit_result"):
            context["model_name"] = getattr(
                context["fit_result"], "model_name", ""
            ) or getattr(context["fit_result"], "_model_name", "")

        # Offload to a background thread — same reason as _on_run_all.
        from rheojax.gui.compat import QObject, QRunnable, QThreadPool, Signal

        class _Relay(QObject):
            finished = Signal(str)
            error = Signal(str)

        relay = _Relay()
        relay.finished.connect(
            self._on_pipeline_step_done, Qt.ConnectionType.QueuedConnection
        )
        relay.error.connect(
            self._on_pipeline_run_error, Qt.ConnectionType.QueuedConnection
        )

        # Keep relay alive until signals are delivered — prevent premature GC.
        # Use a set so concurrent step runs don't clobber each other's relay.
        self._active_relays.add(relay)

        _target = target_step
        _context = context
        _step_id = step_id
        _relay = relay
        _self = self

        class _RunStepWorker(QRunnable):
            def run(self) -> None:  # type: ignore[override]
                try:
                    # parent=None: QObject parent must live in the same thread.
                    service = PipelineExecutionService(None)
                    service.execute_single_step(_target, _context)
                    _relay.finished.emit(_target.name)
                except Exception as exc:
                    logger.error(
                        "Single step execution failed",
                        step_id=_step_id,
                        error=str(exc),
                        exc_info=True,
                    )
                    _relay.error.emit(str(exc))
                finally:
                    _self._active_relays.discard(_relay)

        worker = _RunStepWorker()
        worker.setAutoDelete(True)
        QThreadPool.globalInstance().start(worker)
        self.status_bar.show_message(f"Running step '{target_step.name}'…", 0)

    @Slot(str)
    def _on_pipeline_step_done(self, step_name: str) -> None:
        """Called on the main thread when a single pipeline step completes."""
        self.status_bar.show_message(f"Step '{step_name}' completed", 3000)
        self.log(f"Step '{step_name}' completed")

    @Slot(str)
    def _on_pipeline_run_error(self, error_msg: str) -> None:
        """Called on the main thread when a pipeline run or step fails."""
        logger.error("Pipeline worker error", error=error_msg)
        self.log(f"Pipeline execution failed: {error_msg}")
        self.status_bar.show_message(f"Pipeline run failed: {error_msg}", 5000)
        # GUI-010: belt-and-suspenders — re-enable run buttons even when the
        # execution service failed before dispatching pipeline_execution_completed.
        pipeline_actions.set_pipeline_running(False)

    def _on_theme_changed(self, theme: str) -> None:
        """Handle theme change."""
        logger.debug("Theme change requested", theme=theme)
        self.menu_bar.theme_light_action.setChecked(theme == "light")
        self.menu_bar.theme_dark_action.setChecked(theme == "dark")
        self.menu_bar.theme_auto_action.setChecked(theme == "auto")
        self._apply_theme(theme)
        self.log(f"Theme changed to: {theme}")
        logger.info("Theme changed", theme=theme)
        self.status_bar.show_message(f"Theme: {theme.capitalize()}", 2000)

    def _apply_theme(self, theme: str) -> None:
        """Apply QSS theme to the QApplication.

        When *theme* is ``"auto"``, the resolved OS color scheme is detected
        via ``QStyleHints.colorScheme()`` (Qt 6.5+) with a palette-luminance
        fallback for older Qt versions.
        """
        app = QApplication.instance()
        if app is None:
            return

        if theme == "auto":
            chosen = self._detect_os_color_scheme()
        else:
            chosen = theme

        try:
            app.setStyleSheet(load_stylesheet(chosen))
            ThemeManager.set_theme(chosen)
        except Exception as exc:  # pragma: no cover - GUI runtime
            logger.error(
                "Failed to apply theme", theme=theme, error=str(exc), exc_info=True
            )
            self.log(f"Failed to apply theme {theme}: {exc}")
        else:
            # Persist resolved theme into state
            self.store.dispatch("SET_THEME", {"theme": chosen})
            self.store.dispatch("SET_OS_THEME", {"os_theme": chosen})
            self._plot_style = self._select_plot_style(chosen)

    def _select_plot_style(self, theme: str) -> str:
        """Map UI theme to default plot style."""
        theme_lower = (theme or "light").lower()
        if theme_lower == "dark":
            return "dark"
        return "default"

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
        try:
            app = QApplication.instance()
            if app is None:
                return "light"

            # Qt 6.5+ exposes colorScheme() directly on QStyleHints
            hints = app.styleHints()
            if hasattr(hints, "colorScheme"):
                from PySide6.QtCore import Qt as QtCore_Qt

                scheme = hints.colorScheme()
                if scheme == QtCore_Qt.ColorScheme.Dark:
                    return "dark"
                if scheme == QtCore_Qt.ColorScheme.Light:
                    return "light"
                # Qt.ColorScheme.Unknown — fall through to luminance check
        except Exception:
            pass

        # Palette luminance fallback for Qt < 6.5 or Unknown scheme
        try:
            app = QApplication.instance()
            if app is not None:
                window_color = app.palette().color(app.palette().Window)
                # ITU-R BT.601 luma: dark if luminance < 128
                luma = (
                    0.299 * window_color.red()
                    + 0.587 * window_color.green()
                    + 0.114 * window_color.blue()
                )
                return "dark" if luma < 128 else "light"
        except Exception:
            pass

        return "light"

    def _setup_os_theme_watcher(self) -> None:
        """Hook ``QStyleHints.colorSchemeChanged`` (Qt 6.5+) to re-apply auto theme.

        If the user's theme preference is ``"auto"``, changes to the OS color
        scheme are automatically reflected.  Call this once during window init.
        """
        try:
            app = QApplication.instance()
            if app is None:
                return
            hints = app.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                hints.colorSchemeChanged.connect(self._on_os_color_scheme_changed)
                logger.debug("Connected to QStyleHints.colorSchemeChanged")
        except Exception:
            logger.debug("QStyleHints.colorSchemeChanged not available")

    def _on_os_color_scheme_changed(self) -> None:
        """Handle OS color scheme change.

        Re-applies the theme only when the user preference is ``"auto"``.
        """
        state = self.store.get_state()
        if state.theme == "auto":
            self._apply_theme("auto")

    def _on_pipeline_step_changed(self, step: str, status: str) -> None:
        """Update pipeline chips when state changes."""

        step_enum = step.upper()
        status_enum = status.upper()
        try:
            step_val = PipelineStep[step_enum]
            status_val = StepStatus[status_enum]
        except Exception:
            return
        self.pipeline_chips.set_step_status(step_val, status_val)

    def _update_pipeline_chips_from_state(self, state: AppState) -> None:
        """Sync chip statuses from the current pipeline state."""

        for step, status in state.pipeline_state.steps.items():
            self.pipeline_chips.set_step_status(step, status)

    def _on_dataset_added(self, dataset_id: str) -> None:
        """Add dataset to the tree when state signals dataset addition."""

        dataset = self.store.get_dataset(dataset_id)
        if isinstance(dataset, DatasetState):
            self.data_tree.add_dataset(dataset)
            self.store.dispatch("SET_ACTIVE_DATASET", {"dataset_id": dataset_id})
            # Reflect the loaded dataset in the Data page preview
            try:
                self.data_page.show_dataset(dataset_id)
            except Exception as exc:
                logger.error(
                    "Could not update Data page preview", error=str(exc), exc_info=True
                )
                self.log(f"Could not update Data page preview: {exc}")

    @Slot(str)
    def _on_dataset_selected(self, dataset_id: str) -> None:
        """Update active dataset when selected from tree."""
        logger.debug("Dataset selected", dataset_id=dataset_id)

        self.store.dispatch("SET_ACTIVE_DATASET", {"dataset_id": dataset_id})
        try:
            self.data_page.show_dataset(dataset_id)
        except Exception as exc:
            logger.error(
                "Could not update Data page preview", error=str(exc), exc_info=True
            )
            self.log(f"Could not update Data page preview: {exc}")

    @Slot(str)
    def _on_context_action_triggered(self, action_text: str) -> None:
        """Route DatasetTree context menu actions to the appropriate handler.

        R13-GUI-CTX-001: context_action_triggered was added in Round 2 but was
        never connected.  All context menu items silently did nothing.

        Parameters
        ----------
        action_text : str
            Menu action label, e.g. 'Remove', 'Rename...', 'Export Data...'
        """
        logger.debug("Context action triggered", action=action_text)
        if action_text == "Remove":
            self._on_delete_dataset()
        elif action_text == "Rename...":
            # R13-GUI-CTX-M4: RENAME_DATASET reducer not yet implemented in
            # the store — log and notify the user via status bar.
            logger.info("Rename dataset not yet implemented in store reducer")
            self.status_bar.show_message("Rename: not yet implemented", 3000)
        elif action_text in ("Export Data...", "Export Parameters..."):
            self.navigate_to("export")
        elif action_text == "Duplicate":
            # R13-GUI-CTX-M4: DUPLICATE_DATASET reducer not yet implemented.
            logger.info("Duplicate dataset not yet implemented in store reducer")
            self.status_bar.show_message("Duplicate: not yet implemented", 3000)
        elif action_text == "Expand All":
            self.data_tree.expandAll()
        elif action_text == "Collapse All":
            self.data_tree.collapseAll()
        elif action_text == "Add Dataset...":
            self._on_import()
        elif action_text == "Import Folder...":
            self._on_import()
        elif action_text in ("Open in External Editor", "Show in Folder"):
            file_path = self.data_tree.get_selected_file_path()
            if file_path and file_path.exists():
                try:
                    from PySide6.QtCore import QUrl
                    from PySide6.QtGui import QDesktopServices

                    if action_text == "Show in Folder":
                        # Open the containing directory
                        QDesktopServices.openUrl(
                            QUrl.fromLocalFile(str(file_path.parent))
                        )
                    else:
                        QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))
                except Exception as exc:
                    logger.error(
                        "Could not open file with system handler",
                        file=str(file_path),
                        error=str(exc),
                    )
        elif action_text == "Remove from Dataset":
            logger.info("Remove file from dataset not yet implemented")
            self.status_bar.show_message("Remove file: not yet implemented", 3000)
        else:
            logger.debug("Unhandled context action", action=action_text)

    # ------------------------------------------------------------------
    # Worker pool callbacks
    # ------------------------------------------------------------------

    def _on_job_started(self, job_id: str) -> None:
        logger.debug("Job started", job_id=job_id)
        job_type = self._job_types.get(job_id, "")
        # R6-MW-001: Map job_type directly to pipeline step name.
        # The old ternary "fit" if … else "bayesian" incorrectly mapped
        # transform jobs to the "bayesian" pipeline step.
        _type_to_step = {"fit": "fit", "bayesian": "bayesian", "transform": "transform"}
        step = _type_to_step.get(job_type, "")
        if step:
            self.store.dispatch("SET_PIPELINE_STEP", {"step": step, "status": "ACTIVE"})
            pipeline_step = getattr(PipelineStep, step.upper(), None)
            if pipeline_step is not None:
                self.pipeline_chips.set_step_status(pipeline_step, StepStatus.ACTIVE)
        self.status_bar.show_message(f"Job started: {job_id}", 1000)

    def _on_job_progress(
        self, job_id: str, current: int, total: int, message: str
    ) -> None:
        job_type = self._job_types.get(job_id, "")
        max_value = (
            int(total) if isinstance(total, (int, float)) and 0 < total <= 100 else 0
        )
        msg_text = str(message) if message is not None else f"{job_type} running..."
        self.status_bar.show_progress(int(current), max_value, msg_text)
        if job_type == "fit":
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": "fit", "status": "ACTIVE"}
            )
        elif job_type == "bayesian":
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": "bayesian", "status": "ACTIVE"}
            )

    def _on_job_completed(self, job_id: str, result: object) -> None:
        job_type = self._job_types.pop(job_id, "")
        meta = self._job_metadata.pop(job_id, {})
        logger.info("Job completed", job_id=job_id, job_type=job_type)
        self.status_bar.hide_progress()
        # Use metadata captured at submission time to avoid race on dataset switch
        state = self.store.get_state()
        dataset_id = meta.get("dataset_id") or state.active_dataset_id
        model_name = meta.get("model_name") or state.active_model_name
        if job_type == "fit":
            # Subprocess workers return a plain dict — reconstruct FitResult
            if isinstance(result, dict) and not isinstance(result, FitResult):
                from rheojax.gui.jobs.process_adapter import fit_result_from_dict

                result = fit_result_from_dict(result)
            fit_success = getattr(result, "success", False)
            if fit_success:
                # Persist result in state and refresh UI
                self.store.dispatch(
                    "STORE_FIT_RESULT",
                    {
                        "result": result,
                        "dataset_id": dataset_id,
                        "model_name": model_name,
                    },
                )
                self.fit_page.apply_fit_result(result)
                self._update_fit_plot(result)
                # R8-NEW-001: include identifiers so subscribers can match the completion
                self.store.dispatch(
                    "FITTING_COMPLETED",
                    {
                        "result": result,
                        "model_name": model_name,
                        "dataset_id": dataset_id,
                    },
                )
                # R10-MW-001: Removed redundant SET_PIPELINE_STEP dispatch.
                # STORE_FIT_RESULT reducer already sets PipelineStep.FIT →
                # StepStatus.COMPLETE, and FITTING_COMPLETED reducer does the same.
                # A third dispatch is unnecessary and causes triple state churn.
                self.status_bar.show_message("Fit complete", 3000)
                self._auto_save_if_enabled()
            else:
                error_msg = getattr(result, "message", "Fit failed")
                logger.warning("Fit unsuccessful", error=error_msg)
                # MW-FAIL-003: Include identifiers so fit_failed signal carries
                # model_name/dataset_id (model_name and dataset_id already
                # resolved from meta above in this same _on_job_completed call).
                self.store.dispatch(
                    "FITTING_FAILED",
                    {
                        "error": error_msg,
                        "model_name": model_name,
                        "dataset_id": dataset_id,
                    },
                )
                # R11-MW-003: FITTING_FAILED reducer already sets pipeline
                # step to ERROR; no separate SET_PIPELINE_STEP needed.
                self.status_bar.show_message(f"Fit failed: {error_msg}", 5000)
        elif job_type == "bayesian":
            self.store.dispatch(
                "STORE_BAYESIAN_RESULT",
                {
                    "result": result,
                    "dataset_id": dataset_id,
                    "model_name": model_name,
                },
            )
            # R11-BP-001: STORE_BAYESIAN_RESULT reducer already sets pipeline to COMPLETE
            self.status_bar.show_message("Bayesian inference complete", 3000)
            self._auto_save_if_enabled()
        elif job_type == "transform":
            transform_result = result  # TransformResult dataclass
            success = getattr(transform_result, "success", False)
            if success:
                transformed = getattr(transform_result, "data", None)
                extras = getattr(transform_result, "extras", {}) or {}
                transform_id = meta.get("transform_id", "unknown")
                source_dataset = type(
                    "DS",
                    (),
                    {
                        "id": meta.get("source_dataset_id"),
                        "name": meta.get("source_dataset_name", "data"),
                        "test_mode": None,
                    },
                )()
                params = meta.get("parameters", {})
                try:
                    self._handle_transform_result(
                        transform_id, transformed, extras, source_dataset, params
                    )
                    self.store.dispatch(
                        "SET_PIPELINE_STEP",
                        {"step": "transform", "status": "COMPLETE"},
                    )
                except Exception as exc:
                    logger.error(
                        "Transform result handling failed",
                        error=str(exc),
                        exc_info=True,
                    )
                    self.store.dispatch(
                        "SET_PIPELINE_STEP",
                        {"step": "transform", "status": "ERROR"},
                    )
                    self.status_bar.show_message(f"Transform failed: {exc}", 8000)
            else:
                error_msg = getattr(transform_result, "message", "Transform failed")
                logger.warning("Transform unsuccessful", error=error_msg)
                self.store.dispatch(
                    "SET_PIPELINE_STEP",
                    {"step": "transform", "status": "ERROR"},
                )
                self.status_bar.show_message(f"Transform failed: {error_msg}", 5000)
        self.log(f"Job {job_id} completed ({job_type})")

    def _on_job_failed(self, job_id: str, error: str) -> None:
        job_type = self._job_types.pop(job_id, "")
        # MW-FAIL-001: Pop metadata so identifiers accompany FITTING_FAILED /
        # BAYESIAN_FAILED.  Without model_name/dataset_id the store emits
        # fit_failed("","",error) — BayesianPage._on_bayesian_failed() rejects
        # anonymous signals leaving the UI stuck in a "running" state.
        meta = self._job_metadata.pop(job_id, {})
        logger.error("Job failed", job_id=job_id, job_type=job_type, error=error)
        self.status_bar.hide_progress()
        state = self.store.get_state()
        model_name = meta.get("model_name") or state.active_model_name or ""
        dataset_id = meta.get("dataset_id") or state.active_dataset_id or ""
        # R11-MW-002: FITTING_FAILED/BAYESIAN_FAILED reducers already handle
        # the pipeline step transition to ERROR, so no standalone
        # SET_PIPELINE_STEP dispatch is needed for fit/bayesian.
        if job_type == "fit":
            self.store.dispatch(
                "FITTING_FAILED",
                {
                    "error": str(error),
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                },
            )
        elif job_type == "bayesian":
            self.store.dispatch(
                "BAYESIAN_FAILED",
                {
                    "error": str(error),
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                },
            )
        elif job_type == "transform":
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": job_type, "status": "ERROR"}
            )
            self.status_bar.show_message(f"Transform failed: {error}", 8000)
        self.log(f"Job {job_id} failed: {error}")
        self.status_bar.show_message(f"Job failed: {error}", 5000)

    def _on_job_cancelled(self, job_id: str) -> None:
        job_type = self._job_types.pop(job_id, "")
        # MW-FAIL-002: Pop metadata so identifiers accompany FITTING_FAILED /
        # BAYESIAN_FAILED on cancellation (mirrors the fix in _on_job_failed).
        meta = self._job_metadata.pop(job_id, {})
        logger.info("Job cancelled", job_id=job_id, job_type=job_type)
        self.status_bar.hide_progress()
        state = self.store.get_state()
        model_name = meta.get("model_name") or state.active_model_name or ""
        dataset_id = meta.get("dataset_id") or state.active_dataset_id or ""
        if job_type and job_type not in ("fit", "bayesian"):
            # fit/bayesian have their own FAILED reducers that set ERROR status;
            # dispatching WARNING first would cause double state churn.
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": job_type, "status": "WARNING"}
            )
        if job_type == "fit":
            self.store.dispatch(
                "FITTING_FAILED",
                {
                    "error": "Cancelled",
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                },
            )
        elif job_type == "bayesian":
            self.store.dispatch(
                "BAYESIAN_FAILED",
                {
                    "error": "Cancelled",
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                },
            )
        elif job_type == "transform":
            self.log("Transform cancelled")
        self.log(f"Job {job_id} cancelled")
        self.status_bar.show_message("Job cancelled", 2000)

    def _update_fit_plot(self, fit_result: FitResult) -> None:
        """Render fit plot and residuals on fit page using PlotService."""

        dataset = self.store.get_active_dataset()
        if dataset is None:
            return

        try:
            from rheojax.gui.services.plot_service import PlotService
            from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

            rheo_data = rheodata_from_dataset_state(dataset)
            plot_service = PlotService()
            fig = plot_service.create_fit_plot(
                rheo_data,
                fit_result,
                style=self._plot_style,
                test_mode=dataset.test_mode,
            )
            self.fit_page.set_plot_figure(fig)

            if fit_result.residuals is not None and fit_result.x_fit is not None:
                self.fit_page.plot_residuals(
                    x=np.asarray(fit_result.x_fit),
                    residuals=np.asarray(fit_result.residuals),
                    y_pred=(
                        np.asarray(fit_result.y_fit)
                        if fit_result.y_fit is not None
                        else None
                    ),
                )
        except Exception as exc:
            logger.error("Failed to update fit plot", error=str(exc), exc_info=True)
            self.log(f"Failed to update fit plot: {exc}")

    # -------------------------------------------------------------------------
    # Data Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_new_dataset(self) -> None:
        """Handle new dataset action."""
        logger.debug("New dataset action triggered")
        self._on_import()

    @Slot()
    def _on_delete_dataset(self) -> None:
        """Handle delete dataset action."""
        logger.debug("Delete dataset action triggered")
        reply = QMessageBox.question(
            self,
            "Delete Dataset",
            "Are you sure you want to delete the selected dataset?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.store.dispatch("DELETE_SELECTED_DATASET")
            self.log("Dataset deleted")
            logger.info("Dataset deleted")
            self.status_bar.show_message("Dataset deleted", 3000)

    def _on_set_test_mode(self, mode: str) -> None:
        """Handle set test mode action."""
        logger.debug("Test mode set", mode=mode)
        self.store.dispatch("SET_TEST_MODE", {"test_mode": mode})
        self.log(f"Test mode set to: {mode}")
        self.status_bar.show_message(f"Test mode: {mode}", 2000)

    @Slot()
    def _on_auto_detect_mode(self) -> None:
        """Handle auto-detect test mode action."""
        logger.debug("Auto-detect test mode action triggered")
        state = self.store.get_state()
        dataset_id = state.active_dataset_id
        if dataset_id and dataset_id in state.datasets:
            ds = state.datasets[dataset_id]
            if ds.x_data is not None and ds.y_data is not None:
                # R6-GUI-006: Defer detect_test_mode to avoid blocking the main
                # thread with statistical analysis on large datasets.
                def _do_detect(ds_ref=ds, did=dataset_id):
                    try:
                        from rheojax.core.data import RheoData
                        from rheojax.gui.services.data_service import DataService

                        svc = DataService()
                        mode = svc.detect_test_mode(
                            RheoData(
                                x=ds_ref.x_data,
                                y=ds_ref.y_data,
                                y_units=None,
                                x_units=None,
                                domain=ds_ref.metadata.get("domain", "time"),
                                metadata=ds_ref.metadata,
                                validate=False,
                            )
                        )
                    except Exception as exc:
                        logger.error(
                            "Auto-detect test mode failed",
                            error=str(exc),
                            exc_info=True,
                        )
                        self.log(f"Auto-detect test mode failed: {exc}")
                        self.status_bar.show_message(
                            "Auto-detect test mode failed", 2500
                        )
                        return
                    self.store.dispatch(
                        "AUTO_DETECT_TEST_MODE",
                        {"dataset_id": did, "inferred_mode": mode},
                    )
                    if did:
                        try:
                            self.data_page.show_dataset(did)
                        except Exception:
                            logger.debug(
                                "Data page refresh after auto-detect failed",
                                dataset_id=did,
                                exc_info=True,
                            )

                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, _do_detect)
                return
        if dataset_id:
            try:
                self.data_page.show_dataset(dataset_id)
                resolved_mode = (
                    self.store.get_state().datasets.get(dataset_id).test_mode
                )
                logger.info("Auto-detected test mode", mode=resolved_mode)
                self.status_bar.show_message(
                    f"Auto-detected test mode: {resolved_mode}", 2500
                )
            except Exception as exc:
                logger.error(
                    "Auto-detect test mode failed", error=str(exc), exc_info=True
                )
                self.log(f"Auto-detect test mode failed: {exc}")
                self.status_bar.show_message("Auto-detect test mode failed", 2500)
        else:
            self.status_bar.show_message("No active dataset to auto-detect", 2000)

    # -------------------------------------------------------------------------
    # Models Menu Handlers
    # -------------------------------------------------------------------------

    def _on_select_model(self, model_id: str) -> None:
        """Handle model selection."""
        logger.debug("Model selected", model_id=model_id)
        from rheojax.gui.services.model_service import normalize_model_name

        normalized = normalize_model_name(model_id)
        self.store.dispatch("SET_ACTIVE_MODEL", {"model_name": normalized})
        self.navigate_to("fit")
        self.log(f"Selected model: {normalized}")
        logger.info("Model selection changed", model_name=normalized)
        self.status_bar.show_message(f"Model: {normalized}", 2000)

    @Slot(str)
    def _on_toolbar_model_selected(self, model_id: str) -> None:
        """Handle model change from the quick-fit toolbar."""
        if model_id:
            self._on_select_model(model_id)

    @Slot(str)
    def _setup_shortcuts(self) -> None:
        """Register application-wide shortcuts and command palette."""
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_open_file)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save_file)
        QShortcut(QKeySequence("Ctrl+I"), self, self._on_import)
        QShortcut(QKeySequence("Ctrl+F"), self, self._on_fit)
        QShortcut(QKeySequence("Ctrl+B"), self, self._on_bayesian)
        QShortcut(QKeySequence("Ctrl+D"), self, lambda: self.navigate_to("diagnostics"))
        QShortcut(QKeySequence("Ctrl+E"), self, self._on_export)
        QShortcut(QKeySequence("Ctrl+K"), self, self._open_command_palette)

    def _open_command_palette(self) -> None:
        """Simple command palette to trigger common actions."""
        logger.debug("Command palette opened")
        actions: dict[str, callable] = {
            "Open Project": self._on_open_file,
            "Import Data": self._on_import,
            "Run Fit": self._on_fit,
            "Run Bayesian": self._on_bayesian,
            "Show Diagnostics": lambda: self.navigate_to("diagnostics"),
            "Go to Data": lambda: self.navigate_to("data"),
            "Go to Transform": lambda: self.navigate_to("transform"),
            "Go to Fit": lambda: self.navigate_to("fit"),
            "Go to Bayesian": lambda: self.navigate_to("bayesian"),
            "Go to Export": lambda: self.navigate_to("export"),
            "Export Results": self._on_export,
        }
        labels = sorted(actions.keys())
        selected, ok = QInputDialog.getItem(
            self, "Command Palette", "Action:", labels, 0, False
        )
        if ok and selected in actions:
            logger.debug("Command palette action selected", action=selected)
            actions[selected]()

    # -------------------------------------------------------------------------
    # Transforms Menu Handlers
    # -------------------------------------------------------------------------

    # Transforms that operate on multiple datasets (list[RheoData]).
    # cox_merz requires exactly 2 datasets (oscillation + flow curve).
    _MULTI_DATASET_TRANSFORMS = frozenset({"mastercurve", "srfs", "cox_merz"})

    def _on_apply_transform(
        self, transform_id: str, params: dict | None = None
    ) -> None:
        """Handle transform application via background worker (T-009)."""
        logger.debug("Transform requested", transform_id=transform_id)
        self.store.dispatch("APPLY_TRANSFORM", {"transform_id": transform_id})
        dataset = self.store.get_active_dataset()
        if dataset is None:
            self.status_bar.show_message("Load data before applying transform", 3000)
            return

        try:
            from rheojax.gui.services.transform_service import TransformService
            from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

            # Mastercurve / SRFS need all loaded datasets, not just the active one.
            if transform_id in self._MULTI_DATASET_TRANSFORMS:
                state = self.store.get_state()
                all_datasets = state.datasets or {}
                if len(all_datasets) < 2:
                    self.status_bar.show_message(
                        f"{transform_id} requires at least 2 datasets "
                        f"(loaded {len(all_datasets)})",
                        5000,
                    )
                    return
                rheo_data = [
                    rheodata_from_dataset_state(ds)
                    for ds in all_datasets.values()
                    if ds.x_data is not None and ds.y_data is not None
                ]
            else:
                rheo_data = rheodata_from_dataset_state(dataset)

            # Resolve default params if none provided
            if params is None:
                transform_service = TransformService()
                param_specs = transform_service.get_transform_params(transform_id)
                params = {
                    name: spec.get("default")
                    for name, spec in param_specs.items()
                    if isinstance(spec, dict) and "default" in spec
                }
        except Exception as exc:
            logger.error(
                "Transform setup failed",
                transform_id=transform_id,
                error=str(exc),
                exc_info=True,
            )
            self.status_bar.show_message(f"Transform failed: {exc}", 8000)
            return

        if not self.worker_pool:
            # Fallback: run synchronously if WorkerPool unavailable
            self._run_transform_sync(transform_id, rheo_data, params, dataset)
            return

        # Submit to WorkerPool for async execution (T-009)
        self.store.dispatch(
            "SET_PIPELINE_STEP", {"step": "transform", "status": "ACTIVE"}
        )
        self.status_bar.show_progress(0, 0, f"Applying {transform_id}...")

        worker = TransformWorker(
            transform_id=transform_id,
            data=rheo_data,
            params=params or {},
        )
        try:
            job_id = self.worker_pool.submit(
                worker,
                on_job_registered=lambda jid: self._job_types.__setitem__(
                    jid, "transform"
                ),
            )
            self._job_metadata[job_id] = {
                "transform_id": transform_id,
                "source_dataset_id": dataset.id,
                "source_dataset_name": dataset.name,
                "parameters": params or {},
            }
            logger.info(
                "Transform job submitted",
                job_id=job_id,
                transform_id=transform_id,
            )
        except Exception as exc:
            logger.error(
                "Transform job submission failed",
                error=str(exc),
                exc_info=True,
            )
            self.status_bar.hide_progress()
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": "transform", "status": "ERROR"}
            )
            self.status_bar.show_message(f"Transform failed: {exc}", 8000)
            self.log(f"Transform failed: {exc}")

    def _run_transform_sync(
        self,
        transform_id: str,
        rheo_data: object,
        params: dict | None,
        dataset: object,
    ) -> None:
        """Synchronous fallback when WorkerPool is unavailable."""
        try:
            from rheojax.gui.services.transform_service import TransformService

            result = TransformService().apply_transform(
                transform_id, rheo_data, params=params or {}
            )
            extras = {}
            if isinstance(result, tuple):
                transformed, extras = result
            else:
                transformed = result

            self._handle_transform_result(
                transform_id, transformed, extras, dataset, params
            )
        except Exception as exc:
            logger.error(
                "Transform failed",
                transform_id=transform_id,
                error=str(exc),
                exc_info=True,
            )
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": "transform", "status": "ERROR"}
            )
            self.status_bar.show_message(f"Transform failed: {exc}", 8000)
            self.log(f"Transform failed: {exc}")

    def _handle_transform_result(
        self,
        transform_id: str,
        transformed: object,
        extras: dict,
        dataset: object,
        params: dict | None,
    ) -> None:
        """Process transform output into state store (shared by async and sync paths)."""
        # Validate transform output shape (T-008)
        # Convert JAX arrays to NumPy at the GUI boundary (state store requires NumPy)
        x_data = getattr(transformed, "x", None)
        y_data = getattr(transformed, "y", None)
        y2_data = getattr(transformed, "y2", None)
        if x_data is not None and hasattr(x_data, "device"):
            x_data = np.asarray(x_data)
        if y_data is not None and hasattr(y_data, "device"):
            y_data = np.asarray(y_data)
        if y2_data is not None and hasattr(y2_data, "device"):
            y2_data = np.asarray(y2_data)
        if x_data is not None and y_data is not None:
            if len(x_data) != len(y_data):
                raise ValueError(
                    f"Transform output shape mismatch: x={len(x_data)}, y={len(y_data)}"
                )

        # Merge transform extras into metadata (T-004)
        metadata = {**getattr(transformed, "metadata", {})}
        if extras:
            metadata["_transform_extras"] = extras

        # Use test_mode from transform metadata if updated (T-003)
        new_test_mode = metadata.get("test_mode", getattr(dataset, "test_mode", None))

        new_id = str(uuid.uuid4())
        dataset_name = getattr(dataset, "name", "data")
        self.store.dispatch(
            "IMPORT_DATA_SUCCESS",
            {
                "dataset_id": new_id,
                "file_path": None,
                "name": f"{dataset_name}-{transform_id}",
                "test_mode": new_test_mode,
                "x_data": x_data,
                "y_data": y_data,
                "y2_data": y2_data,
                "metadata": metadata,
            },
        )
        # Pass provenance IDs for TransformRecord (T-005)
        source_id = getattr(dataset, "id", None)
        self.store.dispatch(
            "TRANSFORM_COMPLETED",
            {
                "transform_id": transform_id,
                "source_dataset_id": source_id,
                "target_dataset_id": new_id,
                "parameters": params or {},
            },
        )
        logger.info(
            "Transform applied", transform_id=transform_id, new_dataset_id=new_id
        )
        self.status_bar.show_message(f"Transform applied: {transform_id}", 2000)
        self.log(f"Applied transform {transform_id} -> dataset {new_id}")
        self.navigate_to("transform")

    def _on_transform_applied_from_page(
        self, transform_name: str, dataset_id: str
    ) -> None:
        """Handle transform requests originating from TransformPage.

        Uses TransformPage._selected_key (internal key like "owchirp") when
        available, falling back to a display-name → key map for robustness.
        """
        # Prefer the internal key stored on the TransformPage — it is set
        # when the user clicks a sidebar item and is always a valid
        # TransformService key (e.g. "owchirp", "spp", "derivative").
        transform_id: str = getattr(self.transform_page, "_selected_key", "") or ""
        if not transform_id:
            # Fallback: build map from TransformService metadata so it stays
            # in sync with TransformService.get_transform_metadata() rather
            # than a hardcoded dict that drifts when transforms are added.
            _display_to_key: dict[str, str] = {}
            try:
                for meta in self.transform_page.get_available_transforms():
                    _display_to_key[meta["name"].lower()] = meta["key"]
            except Exception:
                pass
            transform_id = _display_to_key.get(
                transform_name.lower(), transform_name.lower()
            )

        params = self.transform_page.get_selected_params()
        # Activate selected dataset
        self.store.dispatch("SET_ACTIVE_DATASET", {"dataset_id": dataset_id})
        self._on_apply_transform(transform_id, params=params)

    @Slot(object)
    def _on_fit_requested_from_page(self, payload: object) -> None:
        """Handle fit requests originating from FitPage.

        FitPage emits a payload instead of starting a worker directly so we can
        use the centralized WorkerPool/job pipeline and always run
        `_update_fit_plot` on completion.
        """
        logger.debug("Fit requested from page")

        if not isinstance(payload, dict):
            return

        model_name = (
            payload.get("model_name") or self.store.get_state().active_model_name
        )
        dataset_id = (
            payload.get("dataset_id") or self.store.get_state().active_dataset_id
        )
        initial_params = payload.get("initial_params")
        options = payload.get("options") or {}

        if dataset_id:
            self.store.dispatch("SET_ACTIVE_DATASET", {"dataset_id": str(dataset_id)})
        if model_name:
            self.store.dispatch("SET_ACTIVE_MODEL", {"model_name": str(model_name)})

        dataset = self.store.get_active_dataset()
        model_name = self.store.get_state().active_model_name
        if dataset is None or model_name is None:
            self.status_bar.show_message("Select data and model before fitting", 4000)
            return

        # Build RheoData from state (handles y2_data → complex G* for oscillation)
        try:
            from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

            rheo_data = rheodata_from_dataset_state(dataset)
        except Exception as exc:
            logger.error(
                "Cannot start fit: failed to build RheoData",
                error=str(exc),
                exc_info=True,
            )
            self.log(f"Cannot start fit: {exc}")
            self.status_bar.show_message("Unable to build dataset for fit", 4000)
            return

        if not self.worker_pool:
            self.status_bar.show_message("Worker pool unavailable", 3000)
            # MW-FAIL-003: Include identifiers in pre-submission failure so
            # fit_failed signal is not anonymous.
            self.store.dispatch(
                "FITTING_FAILED",
                {
                    "error": "Worker pool unavailable",
                    "model_name": model_name or "",
                    "dataset_id": dataset_id or "",
                },
            )
            self.store.dispatch("SET_PIPELINE_STEP", {"step": "fit", "status": "ERROR"})
            return

        # Normalize initial parameters to a plain float mapping.
        init_params_dict: dict[str, float] | None = None
        if isinstance(initial_params, dict):
            try:
                init_params_dict = {}
                for key, value in initial_params.items():
                    if hasattr(value, "value"):
                        init_params_dict[str(key)] = float(value.value)
                    else:
                        init_params_dict[str(key)] = float(value)
            except Exception:
                logger.debug(
                    "Failed to normalize initial_params from payload; will try state",
                    exc_info=True,
                )
                init_params_dict = None
        if init_params_dict is None:
            # Use current state values if present.
            state_params = self.store.get_state().model_params
            if state_params:
                try:
                    init_params_dict = {
                        name: float(p.value) for name, p in state_params.items()
                    }
                    logger.debug(
                        "Initial params loaded from state",
                        param_count=len(init_params_dict),
                    )
                except Exception:
                    logger.debug(
                        "Failed to load initial params from state; will use defaults",
                        exc_info=True,
                    )
                    init_params_dict = None
        if init_params_dict is None:
            # Fall back to defaults.
            try:
                from rheojax.gui.services.model_service import ModelService

                defaults = ModelService().get_parameter_defaults(model_name)
                init_params_dict = {
                    name: float(p.value) for name, p in defaults.items()
                }
                logger.debug(
                    "Initial params loaded from model defaults",
                    model_name=model_name,
                    param_count=len(init_params_dict),
                )
            except Exception:
                logger.debug(
                    "Failed to load model defaults; fitting without initial params",
                    model_name=model_name,
                    exc_info=True,
                )
                init_params_dict = None

        if not isinstance(options, dict):
            options = {}

        # Emit state signals for FitPage button/UI reactions
        self.store.dispatch(
            "START_FITTING", {"model_name": model_name, "dataset_id": dataset.id}
        )

        # Dispatch pipeline state
        self.store.dispatch("SET_PIPELINE_STEP", {"step": "fit", "status": "ACTIVE"})
        self.status_bar.show_progress(0, 0, "Fitting model...")

        from rheojax.gui.jobs.process_adapter import make_fit_worker

        worker = make_fit_worker(
            model_name=model_name,
            data=rheo_data,
            initial_params=init_params_dict,
            options=options,
            dataset_id=dataset.id,
        )
        try:
            job_id = self.worker_pool.submit(
                worker,
                on_job_registered=lambda jid: self._job_types.__setitem__(jid, "fit"),
            )
            self._job_metadata[job_id] = {
                "model_name": model_name,
                "dataset_id": dataset.id,
            }
            logger.info("Fit job submitted", job_id=job_id, model_name=model_name)
        except Exception as exc:
            logger.error("Fit job submission failed", error=str(exc), exc_info=True)
            self.status_bar.hide_progress()
            # MW-FAIL-003: Include identifiers so fit_failed signal is not
            # anonymous (model_name and dataset.id are in local scope here).
            self.store.dispatch(
                "FITTING_FAILED",
                {
                    "error": str(exc),
                    "model_name": model_name or "",
                    "dataset_id": dataset.id if dataset is not None else "",
                },
            )
            self.store.dispatch("SET_PIPELINE_STEP", {"step": "fit", "status": "ERROR"})
            self.status_bar.show_message(f"Fit failed: {exc}", 5000)
            return

        # R13-GUI-DBL-001: Removed redundant direct call to _on_job_started().
        # WorkerPool.submit() already emits job_started (connected via
        # QueuedConnection in _init_worker_pool), which delivers _on_job_started
        # on the next event loop tick.  The direct call caused double pipeline
        # status updates and duplicate log entries for every fit job.
        # The QueuedConnection is sufficient; QThreadPool queues the actual
        # run() call so the job_started signal is always emitted first.

    # -------------------------------------------------------------------------
    # Analysis Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_fit(self) -> None:
        """Handle fit action."""
        logger.debug("Fit action triggered")
        self.log("Starting NLSQ fit...")
        self.navigate_to("fit")
        self._on_fit_requested_from_page({})

    @Slot()
    def _on_bayesian(self) -> None:
        """Handle bayesian fit action from menu/shortcut.

        Delegates to the BayesianPage's run mechanism which correctly
        reads all GUI settings (warmup, samples, chains, warm-start,
        deformation mode, Poisson ratio).
        """
        logger.debug("Bayesian action triggered (delegating to BayesianPage)")
        self.navigate_to("bayesian")

        # Validate prerequisites before delegating
        dataset = self.store.get_active_dataset()
        model_name = self.store.get_state().active_model_name
        if dataset is None or model_name is None:
            self.status_bar.show_message(
                "Select data and model before Bayesian run", 4000
            )
            return

        # Delegate to BayesianPage which handles warm-start, NUTS settings,
        # deformation mode, Poisson ratio, and proper worker construction.
        self.bayesian_page._on_run_clicked()

    @Slot()
    def _on_show_diagnostics(self) -> None:
        """Navigate to diagnostics and refresh the current model plots."""
        logger.debug("Show diagnostics action triggered")
        self.navigate_to("diagnostics")
        state = self.store.get_state()
        model_name = state.active_model_name
        dataset_id = state.active_dataset_id
        if model_name:
            self.diagnostics_page.show_diagnostics(
                model_name=model_name, dataset_id=dataset_id
            )
        else:
            self.status_bar.show_message("Select a model to view diagnostics", 3000)

    @Slot(str, str)
    def _on_diagnostics_plot_requested(self, plot_type: str, model_id: str) -> None:
        """Handle diagnostics plot request.

        Parameters
        ----------
        plot_type : str
            Type of plot (Trace, Forest, Pair, etc.)
        model_id : str
            Model name/ID
        """
        logger.debug(
            "Diagnostics plot requested", plot_type=plot_type, model_id=model_id
        )
        self.log(f"Diagnostics: generating {plot_type} plot for {model_id}")
        self.status_bar.show_message(f"Generating {plot_type} plot...", 2000)

    @Slot(dict)
    def _on_export_requested(self, config: dict) -> None:
        """Handle export request from export page.

        Parameters
        ----------
        config : dict
            Export configuration
        """
        output_dir = config.get("output_dir", "")
        logger.debug("Export requested", output_dir=output_dir)
        self.log(f"Export: starting export to {output_dir}")
        self.status_bar.show_message("Export in progress...", 0)
        # R6-EXP-003: Mark the pipeline EXPORT step as ACTIVE so the
        # pipeline chips reflect the in-progress state.
        self.store.dispatch("SET_PIPELINE_STEP", {"step": "export", "status": "ACTIVE"})

    @Slot(str)
    def _on_export_completed(self, output_path: str) -> None:
        """Handle export completion.

        Parameters
        ----------
        output_path : str
            Path where files were exported
        """
        logger.info("Export completed", output_path=output_path)
        self.log(f"Export: completed to {output_path}")
        self.status_bar.show_message(f"Export completed: {output_path}", 5000)
        # R6-EXP-001: Update pipeline step so the EXPORT chip reflects
        # completion instead of staying stuck on ACTIVE/PENDING.
        self.store.dispatch("EXPORT_RESULTS", {"file_path": output_path})
        # R6-REV-001: Also dispatch SET_PIPELINE_STEP so pipeline_step_changed
        # signal fires (EXPORT_RESULTS reducer doesn't emit domain signals).
        self.store.dispatch(
            "SET_PIPELINE_STEP", {"step": "export", "status": "COMPLETE"}
        )

    @Slot(str)
    def _on_export_failed(self, error_msg: str) -> None:
        """Handle export failure.

        Parameters
        ----------
        error_msg : str
            Error message
        """
        logger.error("Export failed", error=error_msg)
        self.log(f"Export: failed - {error_msg}")
        self.status_bar.show_message(f"Export failed: {error_msg}", 5000)
        # R6-EXP-002: Update pipeline step so the EXPORT chip reflects
        # the failure instead of staying stuck on ACTIVE/PENDING.
        self.store.dispatch("SET_PIPELINE_STEP", {"step": "export", "status": "ERROR"})

    @Slot()
    def _on_batch_fit(self) -> None:
        """Handle batch fit action — opens the BatchPanel dialog."""
        logger.debug("Batch fit action triggered")
        self.log("Opening batch processing panel...")

        from rheojax.gui.compat import QDialog, QVBoxLayout
        from rheojax.gui.widgets.batch_panel import BatchPanel

        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Processing")
        dialog.resize(700, 500)
        layout = QVBoxLayout(dialog)
        batch_panel = BatchPanel(dialog)
        layout.addWidget(batch_panel)

        # P2-2: Connect batch_requested → _on_batch_requested handler.
        batch_panel.batch_requested.connect(
            lambda d, p, files: self._on_batch_requested(d, p, files, batch_panel)
        )
        dialog.exec()

    def _on_batch_requested(
        self,
        directory: str,
        pattern: str,
        file_paths: list,
        batch_panel: Any,
    ) -> None:
        """Execute the current pipeline for each file in the batch.

        For each file, modifies the load step's config to point at the file,
        executes the full pipeline, and updates the BatchPanel status.

        The heavy JAX fitting work runs on a thread-pool thread so the Qt
        event loop stays responsive.  A relay QObject with typed signals
        routes per-file progress updates back to the main thread where the
        BatchPanel widgets live.
        """
        logger.info(
            "Batch execution started",
            directory=directory,
            pattern=pattern,
            file_count=len(file_paths),
        )
        self.status_bar.show_message(f"Batch: processing {len(file_paths)} files...", 0)

        try:
            from rheojax.gui.state.selectors import get_visual_pipeline_steps

            template_steps = get_visual_pipeline_steps()
        except Exception as exc:
            logger.error("Could not retrieve pipeline steps", error=str(exc))
            batch_panel.finish_batch(0, len(file_paths))
            return

        if not template_steps:
            self.log("Batch: no pipeline steps configured")
            batch_panel.finish_batch(0, len(file_paths))
            return

        # --- Relay for cross-thread progress updates --------------------------
        from rheojax.gui.compat import QObject, QRunnable, QThreadPool

        class _BatchRelay(QObject):
            # file_path, status, elapsed (negative means no elapsed)
            file_status = Signal(str, str, float)
            # current, total, message
            progress = Signal(int, int, str)
            # success_count, total_count
            finished = Signal(int, int)

        relay = _BatchRelay()
        relay.file_status.connect(
            lambda fp, st, el: batch_panel.set_file_status(
                fp, st, el if el >= 0 else None
            ),
            Qt.ConnectionType.QueuedConnection,
        )
        relay.progress.connect(
            batch_panel.set_progress,
            Qt.ConnectionType.QueuedConnection,
        )

        def _on_batch_finished(success: int, total: int) -> None:
            batch_panel.finish_batch(success, total)
            self.status_bar.show_message(
                f"Batch complete: {success}/{total} succeeded",
                5000,
            )
            self.log(f"Batch complete: {success}/{total} files succeeded")
            self._active_relays.discard(relay)

        relay.finished.connect(
            _on_batch_finished,
            Qt.ConnectionType.QueuedConnection,
        )

        # Prevent premature GC — same pattern as _on_run_all.
        self._active_relays.add(relay)

        # Capture closures for the worker thread.
        _steps = template_steps
        _files = file_paths
        _relay = relay

        class _BatchWorker(QRunnable):
            def run(self) -> None:  # type: ignore[override]
                import time
                from dataclasses import replace as dc_replace

                from rheojax.gui.services.pipeline_execution_service import (
                    PipelineExecutionService,
                )

                success_count = 0
                total = len(_files)

                for i, fpath in enumerate(_files):
                    _relay.file_status.emit(fpath, "RUNNING", -1.0)
                    _relay.progress.emit(i, total, f"Processing {Path(fpath).name}...")
                    start = time.perf_counter()

                    try:
                        run_steps = []
                        for step in _steps:
                            if step.step_type == "load":
                                patched = {**step.config, "file": fpath}
                                run_steps.append(dc_replace(step, config=patched))
                            else:
                                run_steps.append(step)

                        # parent=None: worker thread cannot parent to main-thread widget.
                        service = PipelineExecutionService(None)
                        service.execute_all(run_steps)
                        elapsed = time.perf_counter() - start
                        _relay.file_status.emit(fpath, "DONE", elapsed)
                        success_count += 1
                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        _relay.file_status.emit(fpath, "FAILED", elapsed)
                        logger.warning(
                            "Batch file failed",
                            file=fpath,
                            error=str(exc),
                        )

                _relay.finished.emit(success_count, total)

        worker = _BatchWorker()
        worker.setAutoDelete(True)
        QThreadPool.globalInstance().start(worker)

    @Slot()
    def _on_compare_models(self) -> None:
        """Handle compare models action."""
        logger.debug("Compare models action triggered")
        self.log("Opening model comparison...")
        self.navigate_to("diagnostics")
        self.status_bar.show_message("Model comparison", 2000)

    @Slot()
    def _on_check_compatibility(self) -> None:
        """Handle compatibility check action."""
        logger.debug("Compatibility check action triggered")
        self.store.dispatch("CHECK_COMPATIBILITY")
        self.log("Checking model-data compatibility...")
        self.status_bar.show_message("Checking compatibility...", 2000)

    @Slot()
    def _on_stop(self) -> None:
        """Handle stop action."""
        logger.debug("Stop action triggered")
        self.store.dispatch("CANCEL_JOBS")
        self.log("Stopping current operation...")
        if self.worker_pool:
            self.worker_pool.cancel_all()
        self.status_bar.show_message("Operation stopped", 3000)

    # -------------------------------------------------------------------------
    # Tools Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_python_console(self) -> None:
        """Handle Python console action."""
        logger.debug("Python console action triggered")
        self.log("Python console not yet implemented")
        self.status_bar.show_message("Python console (coming soon)", 3000)

    @Slot()
    def _on_jax_profiler(self) -> None:
        """Handle JAX profiler action."""
        logger.debug("JAX profiler action triggered")
        self.log("JAX profiler not yet implemented")
        self.status_bar.show_message("JAX profiler (coming soon)", 3000)

    @Slot()
    def _on_memory_monitor(self) -> None:
        """Handle memory monitor action."""
        logger.debug("Memory monitor action triggered")
        self.log("Memory monitor not yet implemented")
        self.status_bar.show_message("Memory monitor (coming soon)", 3000)

    # -------------------------------------------------------------------------
    # Help Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_open_docs(self) -> None:
        """Handle open documentation action."""
        logger.debug("Open docs action triggered")
        webbrowser.open("https://rheojax.readthedocs.io")
        self.log("Opening documentation in browser")
        self.status_bar.show_message("Opening documentation...", 2000)

    @Slot()
    def _on_open_tutorials(self) -> None:
        """Handle open tutorials action."""
        logger.debug("Open tutorials action triggered")
        webbrowser.open("https://rheojax.readthedocs.io/en/latest/tutorials/")
        self.log("Opening tutorials in browser")
        self.status_bar.show_message("Opening tutorials...", 2000)

    @Slot()
    def _on_show_shortcuts(self) -> None:
        """Handle show shortcuts action."""
        logger.debug("Show shortcuts action triggered")
        shortcuts = """
<h3>Keyboard Shortcuts</h3>
<table>
<tr><td><b>Ctrl+N</b></td><td>New Project</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Open Project</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save Project</td></tr>
<tr><td><b>Ctrl+I</b></td><td>Import Data</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export Results</td></tr>
<tr><td><b>Ctrl+F</b></td><td>Fit Model (NLSQ)</td></tr>
<tr><td><b>Ctrl+B</b></td><td>Bayesian Inference</td></tr>
<tr><td><b>Ctrl+D</b></td><td>Open Diagnostics</td></tr>
<tr><td><b>Ctrl+K</b></td><td>Command Palette</td></tr>
<tr><td><b>Ctrl++</b></td><td>Zoom In</td></tr>
<tr><td><b>Ctrl+-</b></td><td>Zoom Out</td></tr>
<tr><td><b>Ctrl+0</b></td><td>Reset Zoom</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>
<tr><td><b>Ctrl+Y</b></td><td>Redo</td></tr>
<tr><td><b>Ctrl+Q</b></td><td>Exit</td></tr>
</table>
        """
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)
        self.log("Displayed keyboard shortcuts")

    @Slot()
    def _on_about(self) -> None:
        """Handle about action."""
        logger.debug("About action triggered")
        dialog = AboutDialog(self)
        dialog.exec()
        self.log("Displayed About dialog")

    def _update_jax_status(self) -> None:
        """Update JAX device and memory status in status bar."""
        try:
            from rheojax.core.jax_config import safe_import_jax

            jax, _ = safe_import_jax()

            # Get default device
            device = jax.devices()[0]
            device_name = str(device.device_kind)

            # Check if float64 is enabled (use jax already imported via safe_import_jax)
            float64_enabled = jax.config.jax_enable_x64

            # Get memory info (simplified - actual implementation would query device)
            memory_used = 0
            memory_total = 8192  # Placeholder

            self.status_bar.update_jax_status(
                device_name, memory_used, memory_total, float64_enabled
            )
            logger.debug(
                "JAX status updated", device=device_name, float64=float64_enabled
            )
        except Exception as e:
            logger.error("Failed to update JAX status", error=str(e), exc_info=True)
            self.log(f"Failed to update JAX status: {e}")

    def _auto_save_if_enabled(self) -> None:
        """Auto-save project if a path is set and auto-save is enabled.

        Defers the actual work to the next event loop iteration via
        QTimer.singleShot so the caller is never blocked.
        """
        from rheojax.gui.compat import QTimer

        QTimer.singleShot(0, self._do_auto_save)

    def _do_auto_save(self) -> None:
        """Perform the deferred auto-save."""
        try:
            state = self.store.get_state()
            if state.auto_save_enabled and state.project_path:
                project_path = state.project_path
                logger.debug("Auto-saving project", path=str(project_path))
                self.store.dispatch("SAVE_PROJECT", {"file_path": str(project_path)})

                # Export artifacts: parameters, plot, and Bayesian posterior if available
                fit_result = self.store.get_active_fit_result()
                bayes_result = self.store.get_active_bayesian_result()
                if fit_result or bayes_result:
                    from rheojax.gui.services.export_service import ExportService
                    from rheojax.gui.services.plot_service import PlotService

                    export_service = ExportService()
                    plot_service = PlotService()

                    if fit_result:
                        params_path = project_path.with_suffix(".fit_params.json")
                        fig_path = project_path.with_suffix(".fit_plot.png")

                        try:
                            export_service.export_parameters(fit_result, params_path)
                        except Exception as exc:
                            logger.error(
                                "Auto-export params failed",
                                error=str(exc),
                                exc_info=True,
                            )
                            self.log(f"Auto-export params failed: {exc}")

                        try:
                            dataset = self.store.get_active_dataset()
                            if dataset:
                                from rheojax.gui.utils.rheodata import (
                                    rheodata_from_dataset_state,
                                )

                                rheo_data = rheodata_from_dataset_state(dataset)
                                fig = plot_service.create_fit_plot(
                                    rheo_data,
                                    fit_result,
                                    style=self._plot_style,
                                    test_mode=dataset.test_mode,
                                )
                                export_service.export_figure(fig, fig_path)
                        except Exception as exc:
                            logger.error(
                                "Auto-export plot failed", error=str(exc), exc_info=True
                            )
                            self.log(f"Auto-export plot failed: {exc}")

                    if bayes_result:
                        post_path = project_path.with_suffix(".bayes_posterior.h5")
                        diag_path = project_path.with_suffix(".bayes_diagnostics.png")
                        forest_path = project_path.with_suffix(".bayes_forest.png")
                        energy_path = project_path.with_suffix(".bayes_energy.png")
                        try:
                            export_service.export_posterior(bayes_result, post_path)
                        except Exception as exc:
                            logger.error(
                                "Auto-export posterior failed",
                                error=str(exc),
                                exc_info=True,
                            )
                            self.log(f"Auto-export posterior failed: {exc}")
                        try:
                            fig = plot_service.create_arviz_plot(
                                bayes_result, plot_type="trace", style=self._plot_style
                            )
                            export_service.export_figure(fig, diag_path)
                        except Exception as exc:
                            logger.error(
                                "Auto-export diagnostics failed",
                                error=str(exc),
                                exc_info=True,
                            )
                            self.log(f"Auto-export diagnostics failed: {exc}")
                        try:
                            fig = plot_service.create_arviz_plot(
                                bayes_result, plot_type="forest", style=self._plot_style
                            )
                            export_service.export_figure(fig, forest_path)
                        except Exception as exc:
                            logger.error(
                                "Auto-export forest failed",
                                error=str(exc),
                                exc_info=True,
                            )
                            self.log(f"Auto-export forest failed: {exc}")
                        try:
                            fig = plot_service.create_arviz_plot(
                                bayes_result, plot_type="energy", style=self._plot_style
                            )
                            export_service.export_figure(fig, energy_path)
                        except Exception as exc:
                            logger.error(
                                "Auto-export energy failed",
                                error=str(exc),
                                exc_info=True,
                            )
                            self.log(f"Auto-export energy failed: {exc}")

                logger.info("Auto-save completed", path=str(project_path))
                self.status_bar.show_message(f"Auto-saved to {project_path.name}", 2000)
        except Exception as exc:
            logger.error("Auto-save skipped", error=str(exc), exc_info=True)
            self.log(f"Auto-save skipped: {exc}")
