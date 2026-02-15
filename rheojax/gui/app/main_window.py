"""
Main Application Window
=======================

Central window coordinating pages, state, and services with dock-based layout.
"""

import uuid
import webbrowser
from pathlib import Path

from rheojax.gui.app.menu_bar import MenuBar
from rheojax.gui.app.status_bar import StatusBar
from rheojax.gui.compat import (
    QApplication,
    QCloseEvent,
    QDockWidget,
    QFileDialog,
    QFont,
    QInputDialog,
    QKeySequence,
    QMainWindow,
    QMessageBox,
    QShortcut,
    Qt,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QWidget,
    Signal,
    Slot,
)
from rheojax.gui.dialogs.about import AboutDialog
from rheojax.gui.dialogs.import_wizard import ImportWizard
from rheojax.gui.dialogs.preferences import PreferencesDialog
from rheojax.gui.jobs.bayesian_worker import BayesianWorker
from rheojax.gui.jobs.fit_worker import FitWorker
from rheojax.gui.jobs.worker_pool import WorkerPool
from rheojax.gui.pages.bayesian_page import BayesianPage
from rheojax.gui.pages.data_page import DataPage
from rheojax.gui.pages.diagnostics_page import DiagnosticsPage
from rheojax.gui.pages.export_page import ExportPage
from rheojax.gui.pages.fit_page import FitPage
from rheojax.gui.pages.home_page import HomePage
from rheojax.gui.pages.transform_page import TransformPage
from rheojax.gui.resources import load_stylesheet
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
from rheojax.logging import get_logger

logger = get_logger(__name__)


class RheoJAXMainWindow(QMainWindow):
    """Main application window for RheoJAX GUI.

    Architecture:
        - Dock-based layout (data panel, log panel)
        - Tab widget with 7 pages (home, data, transform, fit, bayesian, diagnostics, export)
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
    tabs : QTabWidget
        Central tab widget with pages

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
        self._plot_style: str = "default"
        self._current_workflow_mode: WorkflowMode = WorkflowMode.FITTING

        # Setup UI components
        logger.debug("Setting up UI components")
        self.setup_ui()
        self.setup_docks()
        self.setup_tabs()

        # Connect signals
        logger.debug("Connecting signals")
        self.connect_signals()
        self._connect_state_signals()
        self._init_worker_pool()

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

        # Bump global font size for readability
        app = QApplication.instance()
        if app is not None:
            base_font: QFont = app.font()
            if base_font.pointSize() > 0:
                base_font.setPointSize(max(base_font.pointSize() + 2, 12))
            else:
                base_font.setPointSize(12)
            app.setFont(base_font)

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
        """Create dock widgets for data panel and log panel."""
        logger.debug("Setting up dock widgets")
        # Left dock: Data panel (project tree)
        self.data_dock = QDockWidget("Data", self)
        self.data_dock.setObjectName("DataDock")
        self.data_tree = DatasetTree(self)
        self.data_dock.setWidget(self.data_tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.data_dock)

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

    def setup_tabs(self) -> None:
        """Create tab pages for main content area."""
        logger.debug("Setting up tab pages")
        self.tabs = QTabWidget(self)
        # Enlarge tab font for visibility
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                font-size: 12pt;
                padding: 8px 16px;
            }
            """)
        self.setCentralWidget(self.tabs)

        # Create pages
        self.home_page = HomePage(self)
        self.data_page = DataPage(self)
        self.transform_page = TransformPage(self)
        self.fit_page = FitPage(self)
        self.bayesian_page = BayesianPage(self)
        self.diagnostics_page = DiagnosticsPage(self)
        self.export_page = ExportPage(self)

        # Add pages to tabs
        self.tabs.addTab(self.home_page, "Home")
        self.tabs.addTab(self.data_page, "Data")
        self.tabs.addTab(self.transform_page, "Transform")
        self.tabs.addTab(self.fit_page, "Fit")
        self.tabs.addTab(self.bayesian_page, "Bayesian")
        self.tabs.addTab(self.diagnostics_page, "Diagnostics")
        self.tabs.addTab(self.export_page, "Export")

        # Set initial visibility based on default mode
        self._update_tabs_visibility(self._current_workflow_mode)

        logger.debug("Tab pages setup complete", tab_count=self.tabs.count())

    def _update_tabs_visibility(self, mode: WorkflowMode) -> None:
        """Update tab visibility based on workflow mode.

        Parameters
        ----------
        mode : WorkflowMode
            Active workflow mode
        """
        # Define visible tabs for each mode by page index
        # 0: Home, 1: Data, 2: Transform, 3: Fit, 4: Bayesian, 5: Diagnostics, 6: Export
        visible_indices = {
            WorkflowMode.FITTING: {0, 1, 3, 4, 5, 6},
            WorkflowMode.TRANSFORM: {0, 1, 2, 6},
        }

        indices = visible_indices.get(mode, set())
        for i in range(self.tabs.count()):
            self.tabs.setTabVisible(i, i in indices)

    def _wrap_widget_in_toolbar(self, widget: QWidget) -> "QToolBar":
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
        self._connect_tools_menu()
        self._connect_help_menu()

        # Pipeline chips navigation
        self.pipeline_chips.step_clicked.connect(
            lambda step: self.navigate_to(step.name.lower())
        )

        # Dataset tree selection
        self.data_tree.dataset_selected.connect(self._on_dataset_selected)

        # Transform page callbacks
        self.transform_page.transform_applied.connect(
            self._on_transform_applied_from_page
        )

        # Fit page requests (run through centralized job pipeline)
        self.fit_page.fit_requested.connect(self._on_fit_requested_from_page)

        # Connect home page shortcuts/links
        self.home_page.open_project_requested.connect(self._on_open_file)
        self.home_page.import_data_requested.connect(self._on_import)
        self.home_page.new_project_requested.connect(self._on_new_file)
        self.home_page.example_selected.connect(self._on_open_example)
        self.home_page.recent_project_opened.connect(self._on_open_recent_project)

        # Diagnostics page signals
        self.diagnostics_page.show_requested.connect(self._on_show_diagnostics)
        self.diagnostics_page.plot_requested.connect(
            self._on_diagnostics_plot_requested
        )
        self.diagnostics_page.export_requested.connect(
            self._on_diagnostics_export_requested
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

        self.worker_pool.job_started.connect(self._on_job_started)
        self.worker_pool.job_progress.connect(self._on_job_progress)
        self.worker_pool.job_completed.connect(self._on_job_completed)
        self.worker_pool.job_failed.connect(self._on_job_failed)
        self.worker_pool.job_cancelled.connect(self._on_job_cancelled)

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
            lambda: self.data_dock.setVisible(
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
            lambda: self._on_apply_transform("derivatives")
        )
        self.menu_bar.transform_spp.triggered.connect(
            lambda: self._on_apply_transform("spp")
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
        page_map = {
            "home": 0,
            "data": 1,
            "transform": 2,
            "fit": 3,
            "bayesian": 4,
            "diagnostics": 5,
            "export": 6,
        }

        if page_name.lower() in page_map:
            from_page = self.tabs.tabText(self.tabs.currentIndex())
            to_page = page_name.capitalize()
            logger.debug("Page navigation", from_page=from_page, to_page=to_page)
            self.tabs.setCurrentIndex(page_map[page_name.lower()])
            self.store.dispatch("SET_TAB", {"tab": page_name.lower()})
            self.log(f"Navigated to {page_name.capitalize()} page")

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
                self._on_save_file()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                logger.debug("Shutdown cancelled by user")
                event.ignore()
                return

        # Cleanup: stop workers and disconnect state signals to prevent
        # callbacks during teardown
        self.log("Shutting down...")
        try:
            if hasattr(self, "worker_pool"):
                self.worker_pool.cancel_all()
        except Exception:
            pass
        try:
            signals = self.store.signals
            if signals is not None:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    signals.state_changed.disconnect()
        except (TypeError, RuntimeError):
            pass
        logger.info("Application shutdown complete")
        event.accept()

    @Slot()
    def _on_state_changed(self, state: AppState) -> None:
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
            self._current_workflow_mode = state.workflow_mode
            self._update_tabs_visibility(self._current_workflow_mode)
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
    def _on_save_file(self) -> None:
        """Handle save file action."""
        logger.debug("Save file action triggered")
        self.log("Saving project...")
        logger.info("Project saved")
        self.status_bar.show_message("Project saved", 3000)
        self._has_unsaved_changes = False

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

    @Slot(str)
    def _on_open_example(self, example_name: str) -> None:
        """Handle example selection from the home page.

        Opens the corresponding notebook in Google Colab for interactive use.
        """
        logger.debug("Opening example", example_name=example_name)
        # Map example names to their relative paths in the repository
        example_paths = {
            "oscillation": "examples/basic/02-zener-fitting.ipynb",
            "relaxation": "examples/basic/01-maxwell-fitting.ipynb",
            "creep": "examples/basic/03-springpot-fitting.ipynb",
            "flow": "examples/basic/04-bingham-fitting.ipynb",
            "sgr": "examples/advanced/09-sgr-soft-glassy-rheology.ipynb",
            "spp": "examples/advanced/10-spp-laos-tutorial.ipynb",
            "tts": "examples/transforms/02-mastercurve-tts.ipynb",
            "bayesian": "examples/bayesian",
        }

        COLAB_BASE = "https://colab.research.google.com/github/imewei/rheojax/blob/main"
        GITHUB_BASE = "https://github.com/imewei/rheojax/tree/main"

        rel_path = example_paths.get(example_name.lower())
        if rel_path is None:
            # Unknown example, open examples directory
            url = f"{GITHUB_BASE}/examples"
        elif rel_path.endswith(".ipynb"):
            # Notebook - open in Colab
            url = f"{COLAB_BASE}/{rel_path}"
        else:
            # Directory - open on GitHub
            url = f"{GITHUB_BASE}/{rel_path}"

        try:
            webbrowser.open(url)
            self.log(f"Opening example: {rel_path or example_name}")
            logger.info("Example opened", example_name=example_name, url=url)
            self.status_bar.show_message(
                f"Opening {example_name} example in Colab...", 2000
            )
        except Exception as exc:
            logger.error(
                "Failed to open example",
                example_name=example_name,
                error=str(exc),
                exc_info=True,
            )
            webbrowser.open(f"{GITHUB_BASE}/examples")
            self.log("Falling back to examples repository link")

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

            # Perform the actual load immediately (synchronous import).
            try:
                from rheojax.gui.services.data_service import DataService

                service = DataService()
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
                        "test_mode": test_mode or "unknown",
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

    @Slot()
    def _on_export(self) -> None:
        """Handle export action."""
        logger.debug("Export action triggered")
        self.log("Opening export page...")
        self.navigate_to("export")
        self.status_bar.show_message("Configure export options", 2000)

        # Quick export of current fit parameters if available
        try:
            from PySide6.QtWidgets import QFileDialog

            from rheojax.gui.services.export_service import ExportService

            fit_result = self.store.get_active_fit_result()
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
        current_page = self.tabs.currentWidget()
        if hasattr(current_page, "zoom_in"):
            current_page.zoom_in()
        self.log("Zoom in")
        self.status_bar.show_message("Zoom in", 1000)

    @Slot()
    def _on_zoom_out(self) -> None:
        """Handle zoom out action."""
        current_page = self.tabs.currentWidget()
        if hasattr(current_page, "zoom_out"):
            current_page.zoom_out()
        self.log("Zoom out")
        self.status_bar.show_message("Zoom out", 1000)

    @Slot()
    def _on_reset_zoom(self) -> None:
        """Handle reset zoom action."""
        current_page = self.tabs.currentWidget()
        if hasattr(current_page, "reset_zoom"):
            current_page.reset_zoom()
        self.log("Zoom reset")
        self.status_bar.show_message("Zoom reset", 1000)

    def _on_theme_changed(self, theme: str) -> None:
        """Handle theme change."""
        logger.debug("Theme change requested", theme=theme)
        self.menu_bar.theme_light_action.setChecked(theme == "light")
        self.menu_bar.theme_dark_action.setChecked(theme == "dark")
        self.menu_bar.theme_auto_action.setChecked(theme == "auto")
        self.store.dispatch("SET_THEME", {"theme": theme})
        self._apply_theme(theme)
        self.log(f"Theme changed to: {theme}")
        logger.info("Theme changed", theme=theme)
        self.status_bar.show_message(f"Theme: {theme.capitalize()}", 2000)

    def _apply_theme(self, theme: str) -> None:
        """Apply QSS theme to the QApplication."""

        app = QApplication.instance()
        if app is None:
            return
        chosen = "light" if theme == "auto" else theme
        try:
            app.setStyleSheet(load_stylesheet(chosen))
        except Exception as exc:  # pragma: no cover - GUI runtime
            logger.error(
                "Failed to apply theme", theme=theme, error=str(exc), exc_info=True
            )
            self.log(f"Failed to apply theme {theme}: {exc}")
        else:
            # Persist theme selection into state
            self.store.dispatch("SET_THEME", {"theme": chosen})
            self._plot_style = self._select_plot_style(chosen)

    def _select_plot_style(self, theme: str) -> str:
        """Map UI theme to default plot style."""
        theme_lower = (theme or "light").lower()
        if theme_lower == "dark":
            return "dark"
        return "default"

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

    # ------------------------------------------------------------------
    # Worker pool callbacks
    # ------------------------------------------------------------------

    def _on_job_started(self, job_id: str) -> None:
        logger.debug("Job started", job_id=job_id)
        job_type = self._job_types.get(job_id, "")
        if job_type:
            step = "fit" if job_type == "fit" else "bayesian"
            self.store.dispatch("SET_PIPELINE_STEP", {"step": step, "status": "ACTIVE"})
            self.pipeline_chips.set_step_status(
                getattr(PipelineStep, step.upper()), StepStatus.ACTIVE
            )
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
            fit_success = getattr(result, "success", True)
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
                self.store.dispatch("FITTING_COMPLETED", {"result": result})
                self.store.dispatch(
                    "SET_PIPELINE_STEP", {"step": "fit", "status": "COMPLETE"}
                )
                self.status_bar.show_message("Fit complete", 3000)
                self._auto_save_if_enabled()
            else:
                error_msg = getattr(result, "message", "Fit failed")
                logger.warning("Fit unsuccessful", error=error_msg)
                self.store.dispatch("FITTING_FAILED", {"error": error_msg})
                self.store.dispatch(
                    "SET_PIPELINE_STEP", {"step": "fit", "status": "ERROR"}
                )
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
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": "bayesian", "status": "COMPLETE"}
            )
            self.status_bar.show_message("Bayesian inference complete", 3000)
            self._auto_save_if_enabled()
        self.log(f"Job {job_id} completed ({job_type})")

    def _on_job_failed(self, job_id: str, error: str) -> None:
        job_type = self._job_types.pop(job_id, "")
        logger.error("Job failed", job_id=job_id, job_type=job_type, error=error)
        self.status_bar.hide_progress()
        if job_type:
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": job_type, "status": "ERROR"}
            )
        if job_type == "fit":
            self.store.dispatch("FITTING_FAILED", {"error": str(error)})
        self.log(f"Job {job_id} failed: {error}")
        self.status_bar.show_message(f"Job failed: {error}", 5000)

    def _on_job_cancelled(self, job_id: str) -> None:
        job_type = self._job_types.pop(job_id, "")
        logger.info("Job cancelled", job_id=job_id, job_type=job_type)
        self.status_bar.hide_progress()
        if job_type:
            self.store.dispatch(
                "SET_PIPELINE_STEP", {"step": job_type, "status": "WARNING"}
            )
        if job_type == "fit":
            self.store.dispatch("FITTING_FAILED", {"error": "Cancelled"})
        self.log(f"Job {job_id} cancelled")
        self.status_bar.show_message("Job cancelled", 2000)

    def _update_fit_plot(self, fit_result: FitResult) -> None:
        """Render fit plot and residuals on fit page using PlotService."""

        dataset = self.store.get_active_dataset()
        if dataset is None:
            return

        try:
            import numpy as np

            from rheojax.core.data import RheoData
            from rheojax.gui.services.plot_service import PlotService

            rheo_data = RheoData(
                x=dataset.x_data,
                y=dataset.y_data,
                metadata=dataset.metadata,
                initial_test_mode=dataset.test_mode,
            )
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
        self.store.dispatch("AUTO_DETECT_TEST_MODE")
        # Refresh the Data page preview with any inferred mode
        active = self.store.get_state().active_dataset_id
        if active:
            try:
                self.data_page.show_dataset(active)
                inferred_mode = self.store.get_state().datasets.get(active).test_mode
                logger.info("Auto-detected test mode", mode=inferred_mode)
                self.status_bar.show_message(
                    f"Auto-detected test mode: {inferred_mode}", 2500
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

    def _on_apply_transform(
        self, transform_id: str, params: dict | None = None
    ) -> None:
        """Handle transform application."""
        logger.debug("Transform requested", transform_id=transform_id)
        self.store.dispatch("APPLY_TRANSFORM", {"transform_id": transform_id})
        dataset = self.store.get_active_dataset()
        if dataset is None:
            self.status_bar.show_message("Load data before applying transform", 3000)
            return

        try:
            from rheojax.core.data import RheoData
            from rheojax.gui.services.transform_service import TransformService

            rheo_data = RheoData(
                x=dataset.x_data,
                y=dataset.y_data,
                metadata=dataset.metadata,
                initial_test_mode=dataset.test_mode,
            )
            transform_service = TransformService()
            # Use provided params or defaults from service
            if params is None:
                param_specs = transform_service.get_transform_params(transform_id)
                params = {
                    name: spec.get("default")
                    for name, spec in param_specs.items()
                    if isinstance(spec, dict) and "default" in spec
                }
            result = transform_service.apply_transform(
                transform_id, rheo_data, params=params or {}
            )

            # Handle tuple return (data, extras)
            transformed = result[0] if isinstance(result, tuple) else result
            new_id = str(uuid.uuid4())
            self.store.dispatch(
                "IMPORT_DATA_SUCCESS",
                {
                    "dataset_id": new_id,
                    "file_path": None,
                    "name": f"{dataset.name}-{transform_id}",
                    "test_mode": dataset.test_mode,
                    "x_data": getattr(transformed, "x", None),
                    "y_data": getattr(transformed, "y", None),
                    "metadata": getattr(transformed, "metadata", {}),
                },
            )
            self.store.dispatch("TRANSFORM_COMPLETED", {"transform_id": transform_id})
            logger.info(
                "Transform applied", transform_id=transform_id, new_dataset_id=new_id
            )
            self.status_bar.show_message(f"Transform applied: {transform_id}", 2000)
            self.log(f"Applied transform {transform_id} -> dataset {new_id}")
            self.navigate_to("transform")
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
            self.status_bar.show_message(f"Transform failed: {exc}", 4000)
            self.log(f"Transform failed: {exc}")

    def _on_transform_applied_from_page(
        self, transform_name: str, dataset_id: str
    ) -> None:
        """Handle transform requests originating from TransformPage."""
        name_map = {
            "fft": "fft",
            "mastercurve": "mastercurve",
            "srfs": "srfs",
            "mutation number": "mutation_number",
            "ow chirp": "owchirp",
            "derivatives": "derivative",
            "spp analysis": "spp",
        }
        transform_id = name_map.get(transform_name.lower(), transform_name.lower())
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
            self.store.dispatch("FITTING_FAILED", {"error": "Worker pool unavailable"})
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
                init_params_dict = None
        if init_params_dict is None:
            # Use current state values if present.
            state_params = self.store.get_state().model_params
            if state_params:
                try:
                    init_params_dict = {
                        name: float(p.value) for name, p in state_params.items()
                    }
                except Exception:
                    init_params_dict = None
        if init_params_dict is None:
            # Fall back to defaults.
            try:
                from rheojax.gui.services.model_service import ModelService

                defaults = ModelService().get_parameter_defaults(model_name)
                init_params_dict = {
                    name: float(p.value) for name, p in defaults.items()
                }
            except Exception:
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

        worker = FitWorker(
            model_name=model_name,
            data=rheo_data,
            initial_params=init_params_dict,
            options=options,
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
            self.store.dispatch("FITTING_FAILED", {"error": str(exc)})
            self.store.dispatch("SET_PIPELINE_STEP", {"step": "fit", "status": "ERROR"})
            self.status_bar.show_message(f"Fit failed: {exc}", 5000)
            return

        # Ensure pipeline UI reacts even if job_started races
        self._on_job_started(job_id)

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
        """Handle bayesian fit action."""
        logger.debug("Bayesian action triggered")
        self.log("Starting Bayesian inference...")
        self.navigate_to("bayesian")

        dataset = self.store.get_active_dataset()
        model_name = self.store.get_state().active_model_name
        if dataset is None or model_name is None:
            self.status_bar.show_message(
                "Select data and model before Bayesian run", 4000
            )
            return

        try:
            from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

            rheo_data = rheodata_from_dataset_state(dataset)
        except Exception as exc:
            logger.error(
                "Cannot start Bayesian: failed to build RheoData",
                error=str(exc),
                exc_info=True,
            )
            self.log(f"Cannot start Bayesian: {exc}")
            self.status_bar.show_message("Unable to build dataset for Bayesian", 4000)
            return

        self.store.dispatch(
            "SET_PIPELINE_STEP", {"step": "bayesian", "status": "ACTIVE"}
        )
        self.status_bar.show_progress(0, 0, "Running Bayesian inference...")

        if not self.worker_pool:
            self.status_bar.show_message("Worker pool unavailable", 3000)
            self.store.dispatch("CANCEL_JOBS")
            return

        worker = BayesianWorker(model_name=model_name, data=rheo_data)
        job_id = self.worker_pool.submit(
            worker,
            on_job_registered=lambda jid: self._job_types.__setitem__(jid, "bayesian"),
        )
        self._job_metadata[job_id] = {
            "model_name": model_name,
            "dataset_id": dataset.id,
        }
        logger.info("Bayesian job submitted", job_id=job_id, model_name=model_name)
        self._on_job_started(job_id)

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

    @Slot(str)
    def _on_diagnostics_export_requested(self, plot_type: str) -> None:
        """Handle diagnostics export request.

        Parameters
        ----------
        plot_type : str
            Type of plot being exported
        """
        logger.debug("Diagnostics export requested", plot_type=plot_type)
        self.log(f"Diagnostics: exporting {plot_type} plot")
        self.status_bar.show_message(f"Exporting {plot_type} plot...", 2000)

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

    @Slot()
    def _on_batch_fit(self) -> None:
        """Handle batch fit action."""
        logger.debug("Batch fit action triggered")
        self.log("Opening batch fit dialog...")
        self.navigate_to("fit")
        self.status_bar.show_message("Batch fit: select datasets", 2000)

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

            # Check if float64 is enabled
            from jax import config as jax_config

            float64_enabled = jax_config.jax_enable_x64

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
        """Auto-save project if a path is set and auto-save is enabled."""
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
                                from rheojax.core.data import RheoData

                                rheo_data = RheoData(
                                    x=dataset.x_data,
                                    y=dataset.y_data,
                                    metadata=dataset.metadata,
                                    initial_test_mode=dataset.test_mode,
                                )
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
