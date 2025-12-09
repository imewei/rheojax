"""
Menu Bar
========

Application menu bar with File, Edit, View, Data, Models, Transforms, Analysis, Tools, and Help menus.
"""


from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenuBar, QWidget


class MenuBar(QMenuBar):
    """Application menu bar for RheoJAX GUI.

    Menu Structure:
        - File: New, Open, Save, Save As, Import, Export, Recent, Exit
        - Edit: Undo, Redo, Cut, Copy, Paste, Preferences
        - View: Zoom In/Out, Reset, Layout options, Dock toggles, Theme
        - Data: New Dataset, Delete, Set Test Mode, Auto-detect
        - Models: Submenus for Classical, Flow, Fractional, Multi-Mode, SGR
        - Transforms: FFT, Mastercurve, SRFS, Mutation Number, OWChirp, Derivatives
        - Analysis: Fit NLSQ, Fit Bayesian, Batch Fit, Compare Models, Compatibility
        - Tools: Python Console, JAX Profiler, Memory Monitor, Preferences
        - Help: Documentation, Tutorials, Shortcuts, About

    Example
    -------
    >>> menu_bar = MenuBar()  # doctest: +SKIP
    >>> menu_bar.new_file_action.triggered.connect(on_new_file)  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize menu bar.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        # Create menus
        self._create_file_menu()
        self._create_edit_menu()
        self._create_view_menu()
        self._create_data_menu()
        self._create_models_menu()
        self._create_transforms_menu()
        self._create_analysis_menu()
        self._create_tools_menu()
        self._create_help_menu()

    def _create_file_menu(self) -> None:
        """Create File menu."""
        file_menu = self.addMenu("&File")

        # New
        self.new_file_action = QAction("&New", self)
        self.new_file_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_file_action.setStatusTip("Create a new project")
        file_menu.addAction(self.new_file_action)

        # Open
        self.open_file_action = QAction("&Open...", self)
        self.open_file_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_file_action.setStatusTip("Open an existing project")
        file_menu.addAction(self.open_file_action)

        file_menu.addSeparator()

        # Save
        self.save_file_action = QAction("&Save", self)
        self.save_file_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_file_action.setStatusTip("Save the current project")
        file_menu.addAction(self.save_file_action)

        # Save As
        self.save_as_action = QAction("Save &As...", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.setStatusTip("Save the current project with a new name")
        file_menu.addAction(self.save_as_action)

        file_menu.addSeparator()

        # Import
        self.import_action = QAction("&Import Data...", self)
        self.import_action.setShortcut(QKeySequence("Ctrl+I"))
        self.import_action.setStatusTip("Import rheological data")
        file_menu.addAction(self.import_action)

        # Export
        self.export_action = QAction("&Export Results...", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.setStatusTip("Export analysis results")
        file_menu.addAction(self.export_action)

        file_menu.addSeparator()

        # Recent Files
        self.recent_menu = file_menu.addMenu("Recent Files")
        self._populate_recent_files()

        file_menu.addSeparator()

        # Exit
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.setStatusTip("Exit the application")
        file_menu.addAction(self.exit_action)

    def _create_edit_menu(self) -> None:
        """Create Edit menu."""
        edit_menu = self.addMenu("&Edit")

        # Undo
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setStatusTip("Undo last action")
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)

        # Redo
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setStatusTip("Redo last undone action")
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        # Cut
        self.cut_action = QAction("Cu&t", self)
        self.cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        self.cut_action.setStatusTip("Cut selection to clipboard")
        edit_menu.addAction(self.cut_action)

        # Copy
        self.copy_action = QAction("&Copy", self)
        self.copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.copy_action.setStatusTip("Copy selection to clipboard")
        edit_menu.addAction(self.copy_action)

        # Paste
        self.paste_action = QAction("&Paste", self)
        self.paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        self.paste_action.setStatusTip("Paste from clipboard")
        edit_menu.addAction(self.paste_action)

        edit_menu.addSeparator()

        # Preferences
        self.preferences_action = QAction("&Preferences...", self)
        self.preferences_action.setShortcut(QKeySequence.StandardKey.Preferences)
        self.preferences_action.setStatusTip("Open preferences dialog")
        edit_menu.addAction(self.preferences_action)

    def _create_view_menu(self) -> None:
        """Create View menu."""
        view_menu = self.addMenu("&View")

        # Zoom In
        self.zoom_in_action = QAction("Zoom &In", self)
        self.zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.zoom_in_action.setStatusTip("Zoom in on plot")
        view_menu.addAction(self.zoom_in_action)

        # Zoom Out
        self.zoom_out_action = QAction("Zoom &Out", self)
        self.zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.zoom_out_action.setStatusTip("Zoom out on plot")
        view_menu.addAction(self.zoom_out_action)

        # Reset Zoom
        self.reset_zoom_action = QAction("&Reset Zoom", self)
        self.reset_zoom_action.setShortcut(QKeySequence("Ctrl+0"))
        self.reset_zoom_action.setStatusTip("Reset plot zoom to default")
        view_menu.addAction(self.reset_zoom_action)

        view_menu.addSeparator()

        # Dock visibility toggles
        self.view_data_dock_action = QAction("&Data Panel", self)
        self.view_data_dock_action.setCheckable(True)
        self.view_data_dock_action.setChecked(True)
        self.view_data_dock_action.setStatusTip("Toggle data panel visibility")
        view_menu.addAction(self.view_data_dock_action)

        self.view_param_dock_action = QAction("&Parameters Panel", self)
        self.view_param_dock_action.setCheckable(True)
        self.view_param_dock_action.setChecked(True)
        self.view_param_dock_action.setStatusTip("Toggle parameters panel visibility")
        view_menu.addAction(self.view_param_dock_action)

        self.view_log_dock_action = QAction("&Log Panel", self)
        self.view_log_dock_action.setCheckable(True)
        self.view_log_dock_action.setChecked(False)
        self.view_log_dock_action.setStatusTip("Toggle log panel visibility")
        view_menu.addAction(self.view_log_dock_action)

        view_menu.addSeparator()

        # Theme
        self.theme_menu = view_menu.addMenu("&Theme")
        self.theme_light_action = QAction("&Light", self)
        self.theme_light_action.setCheckable(True)
        self.theme_menu.addAction(self.theme_light_action)

        self.theme_dark_action = QAction("&Dark", self)
        self.theme_dark_action.setCheckable(True)
        self.theme_menu.addAction(self.theme_dark_action)

        self.theme_auto_action = QAction("&Auto (System)", self)
        self.theme_auto_action.setCheckable(True)
        self.theme_auto_action.setChecked(True)
        self.theme_menu.addAction(self.theme_auto_action)

    def _create_data_menu(self) -> None:
        """Create Data menu."""
        data_menu = self.addMenu("&Data")

        # New Dataset
        self.new_dataset_action = QAction("&New Dataset...", self)
        self.new_dataset_action.setShortcut(QKeySequence("Ctrl+Shift+N"))
        self.new_dataset_action.setStatusTip("Create a new dataset")
        data_menu.addAction(self.new_dataset_action)

        # Delete Dataset
        self.delete_dataset_action = QAction("&Delete Dataset", self)
        self.delete_dataset_action.setShortcut(QKeySequence.StandardKey.Delete)
        self.delete_dataset_action.setStatusTip("Delete selected dataset")
        data_menu.addAction(self.delete_dataset_action)

        data_menu.addSeparator()

        # Set Test Mode
        self.test_mode_menu = data_menu.addMenu("Set Test &Mode")
        self.test_mode_oscillation = QAction("&Oscillation", self)
        self.test_mode_menu.addAction(self.test_mode_oscillation)

        self.test_mode_relaxation = QAction("&Relaxation", self)
        self.test_mode_menu.addAction(self.test_mode_relaxation)

        self.test_mode_creep = QAction("&Creep", self)
        self.test_mode_menu.addAction(self.test_mode_creep)

        self.test_mode_rotation = QAction("R&otation", self)
        self.test_mode_menu.addAction(self.test_mode_rotation)

        # Auto-detect Test Mode
        self.auto_detect_mode_action = QAction("&Auto-detect Test Mode", self)
        self.auto_detect_mode_action.setShortcut(QKeySequence("Ctrl+D"))
        self.auto_detect_mode_action.setStatusTip("Automatically detect test mode from data")
        data_menu.addAction(self.auto_detect_mode_action)

    def _create_models_menu(self) -> None:
        """Create Models menu with submenus."""
        models_menu = self.addMenu("&Models")

        # Classical Models submenu
        classical_menu = models_menu.addMenu("&Classical")
        self.model_maxwell = QAction("Maxwell", self)
        classical_menu.addAction(self.model_maxwell)
        self.model_zener = QAction("Zener (SLS)", self)
        classical_menu.addAction(self.model_zener)
        self.model_springpot = QAction("SpringPot", self)
        classical_menu.addAction(self.model_springpot)

        # Flow Models submenu
        flow_menu = models_menu.addMenu("&Flow (Non-Newtonian)")
        self.model_power_law = QAction("Power Law", self)
        flow_menu.addAction(self.model_power_law)
        self.model_carreau = QAction("Carreau", self)
        flow_menu.addAction(self.model_carreau)
        self.model_carreau_yasuda = QAction("Carreau-Yasuda", self)
        flow_menu.addAction(self.model_carreau_yasuda)
        self.model_cross = QAction("Cross", self)
        flow_menu.addAction(self.model_cross)
        self.model_herschel_bulkley = QAction("Herschel-Bulkley", self)
        flow_menu.addAction(self.model_herschel_bulkley)
        self.model_bingham = QAction("Bingham", self)
        flow_menu.addAction(self.model_bingham)

        # Fractional Models submenu
        fractional_menu = models_menu.addMenu("F&ractional")

        # Fractional Maxwell family
        fmaxwell_menu = fractional_menu.addMenu("Maxwell Family")
        self.model_fmaxwell_gel = QAction("Maxwell Gel", self)
        fmaxwell_menu.addAction(self.model_fmaxwell_gel)
        self.model_fmaxwell_liquid = QAction("Maxwell Liquid", self)
        fmaxwell_menu.addAction(self.model_fmaxwell_liquid)
        self.model_fmaxwell_model = QAction("Maxwell Model", self)
        fmaxwell_menu.addAction(self.model_fmaxwell_model)
        self.model_fkelvin_voigt = QAction("Kelvin-Voigt", self)
        fmaxwell_menu.addAction(self.model_fkelvin_voigt)

        # Fractional Zener family
        fzener_menu = fractional_menu.addMenu("Zener Family")
        self.model_fzener_sl = QAction("Solid-Liquid (FZSL)", self)
        fzener_menu.addAction(self.model_fzener_sl)
        self.model_fzener_ss = QAction("Solid-Solid (FZSS)", self)
        fzener_menu.addAction(self.model_fzener_ss)
        self.model_fzener_ll = QAction("Liquid-Liquid (FZLL)", self)
        fzener_menu.addAction(self.model_fzener_ll)
        self.model_fkv_zener = QAction("KV-Zener (FKVZ)", self)
        fzener_menu.addAction(self.model_fkv_zener)

        # Advanced fractional models
        fractional_menu.addSeparator()
        self.model_fburgers = QAction("Burgers (FBM)", self)
        fractional_menu.addAction(self.model_fburgers)
        self.model_fpoynting = QAction("Poynting-Thomson (FPT)", self)
        fractional_menu.addAction(self.model_fpoynting)
        self.model_fjeffreys = QAction("Jeffreys (FJM)", self)
        fractional_menu.addAction(self.model_fjeffreys)

        # Multi-Mode Models submenu
        multimode_menu = models_menu.addMenu("&Multi-Mode")
        self.model_gmaxwell = QAction("Generalized Maxwell", self)
        multimode_menu.addAction(self.model_gmaxwell)

        # SGR Models submenu
        sgr_menu = models_menu.addMenu("&Soft Glassy Rheology")
        self.model_sgr_conventional = QAction("SGR Conventional", self)
        sgr_menu.addAction(self.model_sgr_conventional)
        self.model_sgr_generic = QAction("SGR GENERIC", self)
        sgr_menu.addAction(self.model_sgr_generic)

        # SPP LAOS Models submenu
        spp_menu = models_menu.addMenu("S&PP (LAOS)")
        self.model_spp_yield_stress = QAction("SPP Yield Stress", self)
        spp_menu.addAction(self.model_spp_yield_stress)

    def _create_transforms_menu(self) -> None:
        """Create Transforms menu."""
        transforms_menu = self.addMenu("&Transforms")

        # FFT
        self.transform_fft = QAction("&FFT (Fourier Transform)", self)
        self.transform_fft.setStatusTip("Apply Fast Fourier Transform")
        transforms_menu.addAction(self.transform_fft)

        # Mastercurve
        self.transform_mastercurve = QAction("&Mastercurve (TTS)", self)
        self.transform_mastercurve.setStatusTip("Time-Temperature Superposition")
        transforms_menu.addAction(self.transform_mastercurve)

        # SRFS
        self.transform_srfs = QAction("&SRFS (Strain-Rate Frequency Superposition)", self)
        self.transform_srfs.setStatusTip("Strain-Rate Frequency Superposition")
        transforms_menu.addAction(self.transform_srfs)

        transforms_menu.addSeparator()

        # Mutation Number
        self.transform_mutation = QAction("Mutation &Number", self)
        self.transform_mutation.setStatusTip("Calculate mutation number")
        transforms_menu.addAction(self.transform_mutation)

        # OWChirp
        self.transform_owchirp = QAction("&OWChirp", self)
        self.transform_owchirp.setStatusTip("Optimally Windowed Chirp transform")
        transforms_menu.addAction(self.transform_owchirp)

        # Derivatives
        self.transform_derivatives = QAction("&Derivatives", self)
        self.transform_derivatives.setStatusTip("Calculate numerical derivatives")
        transforms_menu.addAction(self.transform_derivatives)

        transforms_menu.addSeparator()

        # SPP Analysis
        self.transform_spp = QAction("S&PP (LAOS Analysis)", self)
        self.transform_spp.setStatusTip("Sequence of Physical Processes yield stress extraction")
        transforms_menu.addAction(self.transform_spp)

    def _create_analysis_menu(self) -> None:
        """Create Analysis menu."""
        analysis_menu = self.addMenu("&Analysis")

        # Fit NLSQ
        self.analysis_fit_nlsq = QAction("&Fit (NLSQ)", self)
        self.analysis_fit_nlsq.setShortcut(QKeySequence("Ctrl+F"))
        self.analysis_fit_nlsq.setStatusTip("Fit model using non-linear least squares")
        analysis_menu.addAction(self.analysis_fit_nlsq)

        # Fit Bayesian
        self.analysis_fit_bayesian = QAction("Fit &Bayesian (NUTS)", self)
        self.analysis_fit_bayesian.setShortcut(QKeySequence("Ctrl+B"))
        self.analysis_fit_bayesian.setStatusTip("Fit model using Bayesian inference with NUTS")
        analysis_menu.addAction(self.analysis_fit_bayesian)

        analysis_menu.addSeparator()

        # Batch Fit
        self.analysis_batch_fit = QAction("&Batch Fit...", self)
        self.analysis_batch_fit.setStatusTip("Fit multiple datasets in parallel")
        analysis_menu.addAction(self.analysis_batch_fit)

        # Compare Models
        self.analysis_compare = QAction("&Compare Models...", self)
        self.analysis_compare.setStatusTip("Compare multiple model fits")
        analysis_menu.addAction(self.analysis_compare)

        analysis_menu.addSeparator()

        # Compatibility Check
        self.analysis_compatibility = QAction("Check &Compatibility", self)
        self.analysis_compatibility.setStatusTip("Check model-data compatibility")
        analysis_menu.addAction(self.analysis_compatibility)

    def _create_tools_menu(self) -> None:
        """Create Tools menu."""
        tools_menu = self.addMenu("&Tools")

        # Python Console
        self.tools_console = QAction("&Python Console", self)
        self.tools_console.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self.tools_console.setStatusTip("Open Python console")
        tools_menu.addAction(self.tools_console)

        tools_menu.addSeparator()

        # JAX Profiler
        self.tools_jax_profiler = QAction("&JAX Profiler", self)
        self.tools_jax_profiler.setStatusTip("Profile JAX performance")
        tools_menu.addAction(self.tools_jax_profiler)

        # Memory Monitor
        self.tools_memory_monitor = QAction("&Memory Monitor", self)
        self.tools_memory_monitor.setStatusTip("Monitor memory usage")
        tools_menu.addAction(self.tools_memory_monitor)

        tools_menu.addSeparator()

        # Preferences (duplicate for convenience)
        self.tools_preferences = QAction("&Preferences...", self)
        self.tools_preferences.setShortcut(QKeySequence.StandardKey.Preferences)
        self.tools_preferences.setStatusTip("Open preferences dialog")
        tools_menu.addAction(self.tools_preferences)

    def _create_help_menu(self) -> None:
        """Create Help menu."""
        help_menu = self.addMenu("&Help")

        # Documentation
        self.help_docs = QAction("&Documentation", self)
        self.help_docs.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.help_docs.setStatusTip("Open online documentation")
        help_menu.addAction(self.help_docs)

        # Tutorials
        self.help_tutorials = QAction("&Tutorials", self)
        self.help_tutorials.setStatusTip("View tutorials")
        help_menu.addAction(self.help_tutorials)

        # Keyboard Shortcuts
        self.help_shortcuts = QAction("&Keyboard Shortcuts", self)
        self.help_shortcuts.setStatusTip("View keyboard shortcuts")
        help_menu.addAction(self.help_shortcuts)

        help_menu.addSeparator()

        # About
        self.help_about = QAction("&About RheoJAX", self)
        self.help_about.setStatusTip("About RheoJAX")
        help_menu.addAction(self.help_about)

    def _populate_recent_files(self) -> None:
        """Populate recent files menu (placeholder)."""
        # This will be implemented to load recent files from config
        no_recent = QAction("No recent files", self)
        no_recent.setEnabled(False)
        self.recent_menu.addAction(no_recent)
