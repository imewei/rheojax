"""
Main Application Window
=======================

Central window coordinating pages, state, and services with dock-based layout.
"""


from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from rheojax.gui.app.menu_bar import MenuBar
from rheojax.gui.app.status_bar import StatusBar
from rheojax.gui.app.toolbar import MainToolBar, QuickFitStrip
from rheojax.gui.pages.bayesian_page import BayesianPage
from rheojax.gui.pages.data_page import DataPage
from rheojax.gui.pages.diagnostics_page import DiagnosticsPage
from rheojax.gui.pages.export_page import ExportPage
from rheojax.gui.pages.fit_page import FitPage
from rheojax.gui.pages.home_page import HomePage
from rheojax.gui.pages.transform_page import TransformPage
from rheojax.gui.state.store import AppState, StateStore
from rheojax.gui.widgets.dataset_tree import DatasetTree
from rheojax.gui.widgets.parameter_table import ParameterTable


class RheoJAXMainWindow(QMainWindow):
    """Main application window for RheoJAX GUI.

    Architecture:
        - Dock-based layout (data panel, parameter/results panel, log panel)
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
    toolbar : MainToolBar
        Main toolbar with common actions
    quick_fit_strip : QuickFitStrip
        Quick fit workflow toolbar
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

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize main window with pages and services.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        # Initialize state store
        self.store = StateStore()
        self._has_unsaved_changes = False

        # Setup UI components
        self.setup_ui()
        self.setup_docks()
        self.setup_tabs()

        # Connect signals
        self.connect_signals()

        # Initial status
        self.status_bar.show_message("Ready", 3000)

    def setup_ui(self) -> None:
        """Create all UI elements."""
        # Window properties
        self.setWindowTitle("RheoJAX - Rheological Analysis")
        self.resize(1400, 900)

        # Menu bar
        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Main toolbar
        self.toolbar = MainToolBar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Quick fit strip
        self.quick_fit_strip = QuickFitStrip(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.quick_fit_strip)

        # Status bar
        self.status_bar = StatusBar(self)
        self.setStatusBar(self.status_bar)

        # Update JAX status on startup
        self._update_jax_status()

    def setup_docks(self) -> None:
        """Create dock widgets for data panel, parameter panel, and log panel."""
        # Left dock: Data panel (project tree)
        self.data_dock = QDockWidget("Data", self)
        self.data_dock.setObjectName("DataDock")
        self.data_tree = DatasetTree(self)
        self.data_dock.setWidget(self.data_tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.data_dock)

        # Right dock: Parameter/Results panel
        self.param_dock = QDockWidget("Parameters & Results", self)
        self.param_dock.setObjectName("ParamDock")
        self.param_table = ParameterTable(self)
        self.param_dock.setWidget(self.param_table)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.param_dock)

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
        self.tabs = QTabWidget(self)
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

    def connect_signals(self) -> None:
        """Connect state signals to UI updates."""
        # Subscribe to store updates
        self.store.subscribe(self._on_state_changed)

        # Connect menu bar actions
        self.menu_bar.new_file_action.triggered.connect(self._on_new_file)
        self.menu_bar.open_file_action.triggered.connect(self._on_open_file)
        self.menu_bar.save_file_action.triggered.connect(self._on_save_file)
        self.menu_bar.save_as_action.triggered.connect(self._on_save_as)
        self.menu_bar.exit_action.triggered.connect(self.close)

        # Connect toolbar actions
        self.toolbar.open_action.triggered.connect(self._on_open_file)
        self.toolbar.save_action.triggered.connect(self._on_save_file)
        self.toolbar.fit_action.triggered.connect(self._on_fit)
        self.toolbar.bayesian_action.triggered.connect(self._on_bayesian)
        self.toolbar.stop_action.triggered.connect(self._on_stop)

        # Connect quick fit strip
        self.quick_fit_strip.load_clicked.connect(self._on_open_file)
        self.quick_fit_strip.fit_clicked.connect(self._on_quick_fit)
        self.quick_fit_strip.plot_clicked.connect(self._on_plot)
        self.quick_fit_strip.export_clicked.connect(self._on_export)

        # Connect view menu actions to dock visibility
        self.menu_bar.view_data_dock_action.triggered.connect(
            lambda: self.data_dock.setVisible(self.menu_bar.view_data_dock_action.isChecked())
        )
        self.menu_bar.view_param_dock_action.triggered.connect(
            lambda: self.param_dock.setVisible(self.menu_bar.view_param_dock_action.isChecked())
        )
        self.menu_bar.view_log_dock_action.triggered.connect(
            lambda: self.log_dock.setVisible(self.menu_bar.view_log_dock_action.isChecked())
        )

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
            self.tabs.setCurrentIndex(page_map[page_name.lower()])
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
                event.ignore()
                return

        # Cleanup
        self.log("Shutting down...")
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

    @Slot()
    def _on_new_file(self) -> None:
        """Handle new file action."""
        # Check for unsaved changes
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

        # Dispatch new project action
        self.store.dispatch("NEW_PROJECT")
        self.log("Created new project")
        self.status_bar.show_message("New project created", 3000)

    @Slot()
    def _on_open_file(self) -> None:
        """Handle open file action."""
        # This will be implemented with file dialog
        self.log("Open file dialog (not yet implemented)")
        self.status_bar.show_message("Open file...", 2000)

    @Slot()
    def _on_save_file(self) -> None:
        """Handle save file action."""
        self.log("Saving project...")
        self.status_bar.show_message("Project saved", 3000)
        self._has_unsaved_changes = False

    @Slot()
    def _on_save_as(self) -> None:
        """Handle save as action."""
        self.log("Save as dialog (not yet implemented)")
        self.status_bar.show_message("Save as...", 2000)

    @Slot()
    def _on_fit(self) -> None:
        """Handle fit action."""
        self.log("Starting NLSQ fit...")
        self.navigate_to("fit")
        self.status_bar.show_message("Fitting model...", 0)

    @Slot()
    def _on_bayesian(self) -> None:
        """Handle bayesian fit action."""
        self.log("Starting Bayesian inference...")
        self.navigate_to("bayesian")
        self.status_bar.show_message("Running Bayesian inference...", 0)

    @Slot()
    def _on_stop(self) -> None:
        """Handle stop action."""
        self.log("Stopping current operation...")
        self.status_bar.show_message("Operation stopped", 3000)

    @Slot()
    def _on_quick_fit(self) -> None:
        """Handle quick fit from quick fit strip."""
        mode = self.quick_fit_strip.get_mode()
        model = self.quick_fit_strip.get_model()
        self.log(f"Quick fit: {model} in {mode} mode")
        self.navigate_to("fit")

    @Slot()
    def _on_plot(self) -> None:
        """Handle plot action."""
        self.log("Generating plot...")
        self.navigate_to("fit")

    @Slot()
    def _on_export(self) -> None:
        """Handle export action."""
        self.log("Exporting results...")
        self.navigate_to("export")

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

            self.status_bar.update_jax_status(device_name, memory_used, memory_total, float64_enabled)
        except Exception as e:
            self.log(f"Failed to update JAX status: {e}")
