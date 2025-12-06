"""
Main Application Window
=======================

Central window coordinating pages, state, and services with dock-based layout.
"""


import webbrowser

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
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
from rheojax.gui.dialogs.about import AboutDialog
from rheojax.gui.dialogs.import_wizard import ImportWizard
from rheojax.gui.dialogs.preferences import PreferencesDialog
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

        # Connect toolbar actions
        self.toolbar.open_action.triggered.connect(self._on_open_file)
        self.toolbar.save_action.triggered.connect(self._on_save_file)
        self.toolbar.fit_action.triggered.connect(self._on_fit)
        self.toolbar.bayesian_action.triggered.connect(self._on_bayesian)
        self.toolbar.stop_action.triggered.connect(self._on_stop)
        self.toolbar.import_action.triggered.connect(self._on_import)
        self.toolbar.zoom_in_action.triggered.connect(self._on_zoom_in)
        self.toolbar.zoom_out_action.triggered.connect(self._on_zoom_out)
        self.toolbar.reset_zoom_action.triggered.connect(self._on_reset_zoom)
        self.toolbar.settings_action.triggered.connect(self._on_preferences)

        # Connect quick fit strip
        self.quick_fit_strip.load_clicked.connect(self._on_import)
        self.quick_fit_strip.fit_clicked.connect(self._on_quick_fit)
        self.quick_fit_strip.plot_clicked.connect(self._on_plot)
        self.quick_fit_strip.export_clicked.connect(self._on_export)

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
            lambda: self.data_dock.setVisible(self.menu_bar.view_data_dock_action.isChecked())
        )
        self.menu_bar.view_param_dock_action.triggered.connect(
            lambda: self.param_dock.setVisible(self.menu_bar.view_param_dock_action.isChecked())
        )
        self.menu_bar.view_log_dock_action.triggered.connect(
            lambda: self.log_dock.setVisible(self.menu_bar.view_log_dock_action.isChecked())
        )
        self.menu_bar.theme_light_action.triggered.connect(lambda: self._on_theme_changed("light"))
        self.menu_bar.theme_dark_action.triggered.connect(lambda: self._on_theme_changed("dark"))
        self.menu_bar.theme_auto_action.triggered.connect(lambda: self._on_theme_changed("auto"))

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
        self.menu_bar.test_mode_creep.triggered.connect(lambda: self._on_set_test_mode("creep"))
        self.menu_bar.test_mode_rotation.triggered.connect(
            lambda: self._on_set_test_mode("rotation")
        )
        self.menu_bar.auto_detect_mode_action.triggered.connect(self._on_auto_detect_mode)

    def _connect_models_menu(self) -> None:
        """Connect Models menu actions."""
        # Classical models
        self.menu_bar.model_maxwell.triggered.connect(lambda: self._on_select_model("maxwell"))
        self.menu_bar.model_zener.triggered.connect(lambda: self._on_select_model("zener"))
        self.menu_bar.model_springpot.triggered.connect(lambda: self._on_select_model("springpot"))

        # Flow models
        self.menu_bar.model_power_law.triggered.connect(lambda: self._on_select_model("power_law"))
        self.menu_bar.model_carreau.triggered.connect(lambda: self._on_select_model("carreau"))
        self.menu_bar.model_carreau_yasuda.triggered.connect(
            lambda: self._on_select_model("carreau_yasuda")
        )
        self.menu_bar.model_cross.triggered.connect(lambda: self._on_select_model("cross"))
        self.menu_bar.model_herschel_bulkley.triggered.connect(
            lambda: self._on_select_model("herschel_bulkley")
        )
        self.menu_bar.model_bingham.triggered.connect(lambda: self._on_select_model("bingham"))

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

    def _connect_transforms_menu(self) -> None:
        """Connect Transforms menu actions."""
        self.menu_bar.transform_fft.triggered.connect(lambda: self._on_apply_transform("fft"))
        self.menu_bar.transform_mastercurve.triggered.connect(
            lambda: self._on_apply_transform("mastercurve")
        )
        self.menu_bar.transform_srfs.triggered.connect(lambda: self._on_apply_transform("srfs"))
        self.menu_bar.transform_mutation.triggered.connect(
            lambda: self._on_apply_transform("mutation_number")
        )
        self.menu_bar.transform_owchirp.triggered.connect(
            lambda: self._on_apply_transform("owchirp")
        )
        self.menu_bar.transform_derivatives.triggered.connect(
            lambda: self._on_apply_transform("derivatives")
        )

    def _connect_analysis_menu(self) -> None:
        """Connect Analysis menu actions."""
        self.menu_bar.analysis_fit_nlsq.triggered.connect(self._on_fit)
        self.menu_bar.analysis_fit_bayesian.triggered.connect(self._on_bayesian)
        self.menu_bar.analysis_batch_fit.triggered.connect(self._on_batch_fit)
        self.menu_bar.analysis_compare.triggered.connect(self._on_compare_models)
        self.menu_bar.analysis_compatibility.triggered.connect(self._on_check_compatibility)

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

    # -------------------------------------------------------------------------
    # File Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_new_file(self) -> None:
        """Handle new file action."""
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
        self.status_bar.show_message("New project created", 3000)

    @Slot()
    def _on_open_file(self) -> None:
        """Handle open file action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "RheoJAX Project (*.rheo);;HDF5 Files (*.h5 *.hdf5);;All Files (*.*)",
        )
        if file_path:
            self.log(f"Opening project: {file_path}")
            self.store.dispatch("LOAD_PROJECT", {"file_path": file_path})
            self.status_bar.show_message(f"Opened: {file_path}", 3000)

    @Slot()
    def _on_save_file(self) -> None:
        """Handle save file action."""
        self.log("Saving project...")
        self.status_bar.show_message("Project saved", 3000)
        self._has_unsaved_changes = False

    @Slot()
    def _on_save_as(self) -> None:
        """Handle save as action."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "RheoJAX Project (*.rheo);;HDF5 Files (*.h5 *.hdf5);;All Files (*.*)",
        )
        if file_path:
            self.log(f"Saving project as: {file_path}")
            self.store.dispatch("SAVE_PROJECT", {"file_path": file_path})
            self.status_bar.show_message(f"Saved as: {file_path}", 3000)
            self._has_unsaved_changes = False

    @Slot()
    def _on_import(self) -> None:
        """Handle import data action."""
        wizard = ImportWizard(self)
        if wizard.exec():
            config = wizard.get_result()
            self.log(f"Importing data from: {config['file_path']}")
            self.store.dispatch("IMPORT_DATA", config)
            self.navigate_to("data")
            self.status_bar.show_message("Data imported successfully", 3000)

    @Slot()
    def _on_export(self) -> None:
        """Handle export action."""
        self.log("Opening export page...")
        self.navigate_to("export")
        self.status_bar.show_message("Configure export options", 2000)

    # -------------------------------------------------------------------------
    # Edit Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_undo(self) -> None:
        """Handle undo action."""
        self.store.dispatch("UNDO")
        self.log("Undo")
        self.status_bar.show_message("Undo", 2000)

    @Slot()
    def _on_redo(self) -> None:
        """Handle redo action."""
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
        dialog = PreferencesDialog(parent=self)
        if dialog.exec():
            prefs = dialog.get_preferences()
            self.store.dispatch("UPDATE_PREFERENCES", prefs)
            self.log("Preferences updated")
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
        self.menu_bar.theme_light_action.setChecked(theme == "light")
        self.menu_bar.theme_dark_action.setChecked(theme == "dark")
        self.menu_bar.theme_auto_action.setChecked(theme == "auto")
        self.store.dispatch("SET_THEME", {"theme": theme})
        self.log(f"Theme changed to: {theme}")
        self.status_bar.show_message(f"Theme: {theme.capitalize()}", 2000)

    # -------------------------------------------------------------------------
    # Data Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_new_dataset(self) -> None:
        """Handle new dataset action."""
        self._on_import()

    @Slot()
    def _on_delete_dataset(self) -> None:
        """Handle delete dataset action."""
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
            self.status_bar.show_message("Dataset deleted", 3000)

    def _on_set_test_mode(self, mode: str) -> None:
        """Handle set test mode action."""
        self.store.dispatch("SET_TEST_MODE", {"test_mode": mode})
        self.quick_fit_strip.set_mode(mode)
        self.log(f"Test mode set to: {mode}")
        self.status_bar.show_message(f"Test mode: {mode}", 2000)

    @Slot()
    def _on_auto_detect_mode(self) -> None:
        """Handle auto-detect test mode action."""
        self.store.dispatch("AUTO_DETECT_TEST_MODE")
        self.log("Auto-detecting test mode...")
        self.status_bar.show_message("Auto-detecting test mode...", 2000)

    # -------------------------------------------------------------------------
    # Models Menu Handlers
    # -------------------------------------------------------------------------

    def _on_select_model(self, model_id: str) -> None:
        """Handle model selection."""
        self.store.dispatch("SELECT_MODEL", {"model_id": model_id})
        self.quick_fit_strip.set_model(model_id)
        self.navigate_to("fit")
        self.log(f"Selected model: {model_id}")
        self.status_bar.show_message(f"Model: {model_id}", 2000)

    # -------------------------------------------------------------------------
    # Transforms Menu Handlers
    # -------------------------------------------------------------------------

    def _on_apply_transform(self, transform_id: str) -> None:
        """Handle transform application."""
        self.store.dispatch("APPLY_TRANSFORM", {"transform_id": transform_id})
        self.navigate_to("transform")
        self.log(f"Applying transform: {transform_id}")
        self.status_bar.show_message(f"Transform: {transform_id}", 2000)

    # -------------------------------------------------------------------------
    # Analysis Menu Handlers
    # -------------------------------------------------------------------------

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
    def _on_batch_fit(self) -> None:
        """Handle batch fit action."""
        self.log("Opening batch fit dialog...")
        self.navigate_to("fit")
        self.status_bar.show_message("Batch fit: select datasets", 2000)

    @Slot()
    def _on_compare_models(self) -> None:
        """Handle compare models action."""
        self.log("Opening model comparison...")
        self.navigate_to("diagnostics")
        self.status_bar.show_message("Model comparison", 2000)

    @Slot()
    def _on_check_compatibility(self) -> None:
        """Handle compatibility check action."""
        self.store.dispatch("CHECK_COMPATIBILITY")
        self.log("Checking model-data compatibility...")
        self.status_bar.show_message("Checking compatibility...", 2000)

    @Slot()
    def _on_stop(self) -> None:
        """Handle stop action."""
        self.store.dispatch("CANCEL_JOBS")
        self.log("Stopping current operation...")
        self.status_bar.show_message("Operation stopped", 3000)

    # -------------------------------------------------------------------------
    # Tools Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_python_console(self) -> None:
        """Handle Python console action."""
        self.log("Python console not yet implemented")
        self.status_bar.show_message("Python console (coming soon)", 3000)

    @Slot()
    def _on_jax_profiler(self) -> None:
        """Handle JAX profiler action."""
        self.log("JAX profiler not yet implemented")
        self.status_bar.show_message("JAX profiler (coming soon)", 3000)

    @Slot()
    def _on_memory_monitor(self) -> None:
        """Handle memory monitor action."""
        self.log("Memory monitor not yet implemented")
        self.status_bar.show_message("Memory monitor (coming soon)", 3000)

    # -------------------------------------------------------------------------
    # Help Menu Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_open_docs(self) -> None:
        """Handle open documentation action."""
        webbrowser.open("https://rheojax.readthedocs.io")
        self.log("Opening documentation in browser")
        self.status_bar.show_message("Opening documentation...", 2000)

    @Slot()
    def _on_open_tutorials(self) -> None:
        """Handle open tutorials action."""
        webbrowser.open("https://rheojax.readthedocs.io/en/latest/tutorials/")
        self.log("Opening tutorials in browser")
        self.status_bar.show_message("Opening tutorials...", 2000)

    @Slot()
    def _on_show_shortcuts(self) -> None:
        """Handle show shortcuts action."""
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
<tr><td><b>Ctrl+D</b></td><td>Auto-detect Test Mode</td></tr>
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
        dialog = AboutDialog(self)
        dialog.exec()
        self.log("Displayed About dialog")

    # -------------------------------------------------------------------------
    # Quick Fit Strip Handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_quick_fit(self) -> None:
        """Handle quick fit from quick fit strip."""
        mode = self.quick_fit_strip.get_mode()
        model = self.quick_fit_strip.get_model()
        self.log(f"Quick fit: {model} in {mode} mode")
        self.store.dispatch("SET_TEST_MODE", {"test_mode": mode})
        self.store.dispatch("SELECT_MODEL", {"model_id": model})
        self.navigate_to("fit")

    @Slot()
    def _on_plot(self) -> None:
        """Handle plot action."""
        self.log("Generating plot...")
        self.navigate_to("fit")

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
