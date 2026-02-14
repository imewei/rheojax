"""
Menu Bar
========

Application menu bar with File, Edit, View, Data, Models, Transforms, Analysis, Tools, and Help menus.
"""

from rheojax.gui.compat import QAction, QKeySequence, QMenuBar, QWidget
from rheojax.logging import get_logger

logger = get_logger(__name__)


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
        logger.debug("Initializing", class_name=self.__class__.__name__)

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

        logger.debug(
            "Menu bar initialized",
            class_name=self.__class__.__name__,
            menu_count=len(self.actions()),
        )

    def _create_file_menu(self) -> None:
        """Create File menu."""
        logger.debug("Creating menu", menu="File")
        file_menu = self.addMenu("&File")

        # New
        self.new_file_action = QAction("&New", self)
        self.new_file_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_file_action.setStatusTip("Create a new project")
        self.new_file_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="new_file", menu="File")
        )
        file_menu.addAction(self.new_file_action)

        # Open
        self.open_file_action = QAction("&Open...", self)
        self.open_file_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_file_action.setStatusTip("Open an existing project")
        self.open_file_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="open_file", menu="File")
        )
        file_menu.addAction(self.open_file_action)

        file_menu.addSeparator()

        # Save
        self.save_file_action = QAction("&Save", self)
        self.save_file_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_file_action.setStatusTip("Save the current project")
        self.save_file_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="save_file", menu="File")
        )
        file_menu.addAction(self.save_file_action)

        # Save As
        self.save_as_action = QAction("Save &As...", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.setStatusTip("Save the current project with a new name")
        self.save_as_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="save_as", menu="File")
        )
        file_menu.addAction(self.save_as_action)

        file_menu.addSeparator()

        # Import
        self.import_action = QAction("&Import Data...", self)
        self.import_action.setShortcut(QKeySequence("Ctrl+I"))
        self.import_action.setStatusTip("Import rheological data")
        self.import_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="import_data", menu="File")
        )
        file_menu.addAction(self.import_action)

        # Export
        self.export_action = QAction("&Export Results...", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.setStatusTip("Export analysis results")
        self.export_action.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="export_results", menu="File"
            )
        )
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
        self.exit_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="exit", menu="File")
        )
        file_menu.addAction(self.exit_action)

        logger.debug(
            "Menu created", menu="File", action_count=file_menu.actions().__len__()
        )

    def _create_edit_menu(self) -> None:
        """Create Edit menu."""
        logger.debug("Creating menu", menu="Edit")
        edit_menu = self.addMenu("&Edit")

        # Undo
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setStatusTip("Undo last action")
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="undo", menu="Edit")
        )
        edit_menu.addAction(self.undo_action)

        # Redo
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setStatusTip("Redo last undone action")
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="redo", menu="Edit")
        )
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        # Cut
        self.cut_action = QAction("Cu&t", self)
        self.cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        self.cut_action.setStatusTip("Cut selection to clipboard")
        self.cut_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="cut", menu="Edit")
        )
        edit_menu.addAction(self.cut_action)

        # Copy
        self.copy_action = QAction("&Copy", self)
        self.copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.copy_action.setStatusTip("Copy selection to clipboard")
        self.copy_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="copy", menu="Edit")
        )
        edit_menu.addAction(self.copy_action)

        # Paste
        self.paste_action = QAction("&Paste", self)
        self.paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        self.paste_action.setStatusTip("Paste from clipboard")
        self.paste_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="paste", menu="Edit")
        )
        edit_menu.addAction(self.paste_action)

        edit_menu.addSeparator()

        # Preferences
        self.preferences_action = QAction("&Preferences...", self)
        self.preferences_action.setShortcut(QKeySequence.StandardKey.Preferences)
        self.preferences_action.setStatusTip("Open preferences dialog")
        self.preferences_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="preferences", menu="Edit")
        )
        edit_menu.addAction(self.preferences_action)

        logger.debug(
            "Menu created", menu="Edit", action_count=edit_menu.actions().__len__()
        )

    def _create_view_menu(self) -> None:
        """Create View menu."""
        logger.debug("Creating menu", menu="View")
        view_menu = self.addMenu("&View")

        # Zoom In
        self.zoom_in_action = QAction("Zoom &In", self)
        self.zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.zoom_in_action.setStatusTip("Zoom in on plot")
        self.zoom_in_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="zoom_in", menu="View")
        )
        view_menu.addAction(self.zoom_in_action)

        # Zoom Out
        self.zoom_out_action = QAction("Zoom &Out", self)
        self.zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.zoom_out_action.setStatusTip("Zoom out on plot")
        self.zoom_out_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="zoom_out", menu="View")
        )
        view_menu.addAction(self.zoom_out_action)

        # Reset Zoom
        self.reset_zoom_action = QAction("&Reset Zoom", self)
        self.reset_zoom_action.setShortcut(QKeySequence("Ctrl+0"))
        self.reset_zoom_action.setStatusTip("Reset plot zoom to default")
        self.reset_zoom_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="reset_zoom", menu="View")
        )
        view_menu.addAction(self.reset_zoom_action)

        view_menu.addSeparator()

        # Dock visibility toggles
        self.view_data_dock_action = QAction("&Data Panel", self)
        self.view_data_dock_action.setCheckable(True)
        self.view_data_dock_action.setChecked(True)
        self.view_data_dock_action.setStatusTip("Toggle data panel visibility")
        self.view_data_dock_action.triggered.connect(
            lambda checked: logger.debug(
                "Action triggered",
                action="toggle_data_panel",
                menu="View",
                checked=checked,
            )
        )
        view_menu.addAction(self.view_data_dock_action)

        self.view_log_dock_action = QAction("&Log Panel", self)
        self.view_log_dock_action.setCheckable(True)
        self.view_log_dock_action.setChecked(False)
        self.view_log_dock_action.setStatusTip("Toggle log panel visibility")
        self.view_log_dock_action.triggered.connect(
            lambda checked: logger.debug(
                "Action triggered",
                action="toggle_log_panel",
                menu="View",
                checked=checked,
            )
        )
        view_menu.addAction(self.view_log_dock_action)

        view_menu.addSeparator()

        # Theme
        self.theme_menu = view_menu.addMenu("&Theme")
        self.theme_light_action = QAction("&Light", self)
        self.theme_light_action.setCheckable(True)
        self.theme_light_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="theme_light", menu="View")
        )
        self.theme_menu.addAction(self.theme_light_action)

        self.theme_dark_action = QAction("&Dark", self)
        self.theme_dark_action.setCheckable(True)
        self.theme_dark_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="theme_dark", menu="View")
        )
        self.theme_menu.addAction(self.theme_dark_action)

        self.theme_auto_action = QAction("&Auto (System)", self)
        self.theme_auto_action.setCheckable(True)
        self.theme_auto_action.setChecked(True)
        self.theme_auto_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="theme_auto", menu="View")
        )
        self.theme_menu.addAction(self.theme_auto_action)

        logger.debug(
            "Menu created", menu="View", action_count=view_menu.actions().__len__()
        )

    def _create_data_menu(self) -> None:
        """Create Data menu."""
        logger.debug("Creating menu", menu="Data")
        data_menu = self.addMenu("&Data")

        # New Dataset
        self.new_dataset_action = QAction("&New Dataset...", self)
        self.new_dataset_action.setShortcut(QKeySequence("Ctrl+Shift+N"))
        self.new_dataset_action.setStatusTip("Create a new dataset")
        self.new_dataset_action.triggered.connect(
            lambda: logger.debug("Action triggered", action="new_dataset", menu="Data")
        )
        data_menu.addAction(self.new_dataset_action)

        # Delete Dataset
        self.delete_dataset_action = QAction("&Delete Dataset", self)
        self.delete_dataset_action.setShortcut(QKeySequence.StandardKey.Delete)
        self.delete_dataset_action.setStatusTip("Delete selected dataset")
        self.delete_dataset_action.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="delete_dataset", menu="Data"
            )
        )
        data_menu.addAction(self.delete_dataset_action)

        data_menu.addSeparator()

        # Set Test Mode
        self.test_mode_menu = data_menu.addMenu("Set Test &Mode")
        self.test_mode_oscillation = QAction("&Oscillation", self)
        self.test_mode_oscillation.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Data",
                action="set_test_mode",
                mode="oscillation",
            )
        )
        self.test_mode_menu.addAction(self.test_mode_oscillation)

        self.test_mode_relaxation = QAction("&Relaxation", self)
        self.test_mode_relaxation.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Data",
                action="set_test_mode",
                mode="relaxation",
            )
        )
        self.test_mode_menu.addAction(self.test_mode_relaxation)

        self.test_mode_creep = QAction("&Creep", self)
        self.test_mode_creep.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Data",
                action="set_test_mode",
                mode="creep",
            )
        )
        self.test_mode_menu.addAction(self.test_mode_creep)

        self.test_mode_rotation = QAction("R&otation", self)
        self.test_mode_rotation.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Data",
                action="set_test_mode",
                mode="rotation",
            )
        )
        self.test_mode_menu.addAction(self.test_mode_rotation)

        self.test_mode_flow_curve = QAction("&Flow Curve", self)
        self.test_mode_flow_curve.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Data",
                action="set_test_mode",
                mode="flow_curve",
            )
        )
        self.test_mode_menu.addAction(self.test_mode_flow_curve)

        self.test_mode_startup = QAction("&Startup", self)
        self.test_mode_startup.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Data",
                action="set_test_mode",
                mode="startup",
            )
        )
        self.test_mode_menu.addAction(self.test_mode_startup)

        self.test_mode_laos = QAction("&LAOS", self)
        self.test_mode_laos.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Data",
                action="set_test_mode",
                mode="laos",
            )
        )
        self.test_mode_menu.addAction(self.test_mode_laos)

        # Auto-detect Test Mode
        self.auto_detect_mode_action = QAction("&Auto-detect Test Mode", self)
        self.auto_detect_mode_action.setShortcut(QKeySequence("Ctrl+D"))
        self.auto_detect_mode_action.setStatusTip(
            "Automatically detect test mode from data"
        )
        self.auto_detect_mode_action.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="auto_detect_mode", menu="Data"
            )
        )
        data_menu.addAction(self.auto_detect_mode_action)

        logger.debug(
            "Menu created", menu="Data", action_count=data_menu.actions().__len__()
        )

    def _create_models_menu(self) -> None:
        """Create Models menu with submenus."""
        logger.debug("Creating menu", menu="Models")
        models_menu = self.addMenu("&Models")

        # Classical Models submenu
        classical_menu = models_menu.addMenu("&Classical")
        self.model_maxwell = QAction("Maxwell", self)
        self.model_maxwell.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Classical",
                model="Maxwell",
            )
        )
        classical_menu.addAction(self.model_maxwell)
        self.model_zener = QAction("Zener (SLS)", self)
        self.model_zener.triggered.connect(
            lambda: logger.debug(
                "Menu item selected", menu="Models", submenu="Classical", model="Zener"
            )
        )
        classical_menu.addAction(self.model_zener)
        self.model_springpot = QAction("SpringPot", self)
        self.model_springpot.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Classical",
                model="SpringPot",
            )
        )
        classical_menu.addAction(self.model_springpot)

        # Flow Models submenu
        flow_menu = models_menu.addMenu("&Flow (Non-Newtonian)")
        self.model_power_law = QAction("Power Law", self)
        self.model_power_law.triggered.connect(
            lambda: logger.debug(
                "Menu item selected", menu="Models", submenu="Flow", model="PowerLaw"
            )
        )
        flow_menu.addAction(self.model_power_law)
        self.model_carreau = QAction("Carreau", self)
        self.model_carreau.triggered.connect(
            lambda: logger.debug(
                "Menu item selected", menu="Models", submenu="Flow", model="Carreau"
            )
        )
        flow_menu.addAction(self.model_carreau)
        self.model_carreau_yasuda = QAction("Carreau-Yasuda", self)
        self.model_carreau_yasuda.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Flow",
                model="CarreauYasuda",
            )
        )
        flow_menu.addAction(self.model_carreau_yasuda)
        self.model_cross = QAction("Cross", self)
        self.model_cross.triggered.connect(
            lambda: logger.debug(
                "Menu item selected", menu="Models", submenu="Flow", model="Cross"
            )
        )
        flow_menu.addAction(self.model_cross)
        self.model_herschel_bulkley = QAction("Herschel-Bulkley", self)
        self.model_herschel_bulkley.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Flow",
                model="HerschelBulkley",
            )
        )
        flow_menu.addAction(self.model_herschel_bulkley)
        self.model_bingham = QAction("Bingham", self)
        self.model_bingham.triggered.connect(
            lambda: logger.debug(
                "Menu item selected", menu="Models", submenu="Flow", model="Bingham"
            )
        )
        flow_menu.addAction(self.model_bingham)

        # Fractional Models submenu
        fractional_menu = models_menu.addMenu("F&ractional")

        # Fractional Maxwell family
        fmaxwell_menu = fractional_menu.addMenu("Maxwell Family")
        self.model_fmaxwell_gel = QAction("Maxwell Gel", self)
        self.model_fmaxwell_gel.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional/Maxwell",
                model="MaxwellGel",
            )
        )
        fmaxwell_menu.addAction(self.model_fmaxwell_gel)
        self.model_fmaxwell_liquid = QAction("Maxwell Liquid", self)
        self.model_fmaxwell_liquid.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional/Maxwell",
                model="MaxwellLiquid",
            )
        )
        fmaxwell_menu.addAction(self.model_fmaxwell_liquid)
        self.model_fmaxwell_model = QAction("Maxwell Model", self)
        self.model_fmaxwell_model.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional/Maxwell",
                model="MaxwellModel",
            )
        )
        fmaxwell_menu.addAction(self.model_fmaxwell_model)
        self.model_fkelvin_voigt = QAction("Kelvin-Voigt", self)
        self.model_fkelvin_voigt.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional/Maxwell",
                model="KelvinVoigt",
            )
        )
        fmaxwell_menu.addAction(self.model_fkelvin_voigt)

        # Fractional Zener family
        fzener_menu = fractional_menu.addMenu("Zener Family")
        self.model_fzener_sl = QAction("Solid-Liquid (FZSL)", self)
        self.model_fzener_sl.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional/Zener",
                model="FZSL",
            )
        )
        fzener_menu.addAction(self.model_fzener_sl)
        self.model_fzener_ss = QAction("Solid-Solid (FZSS)", self)
        self.model_fzener_ss.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional/Zener",
                model="FZSS",
            )
        )
        fzener_menu.addAction(self.model_fzener_ss)
        self.model_fzener_ll = QAction("Liquid-Liquid (FZLL)", self)
        self.model_fzener_ll.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional/Zener",
                model="FZLL",
            )
        )
        fzener_menu.addAction(self.model_fzener_ll)
        self.model_fkv_zener = QAction("KV-Zener (FKVZ)", self)
        self.model_fkv_zener.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional/Zener",
                model="FKVZ",
            )
        )
        fzener_menu.addAction(self.model_fkv_zener)

        # Advanced fractional models
        fractional_menu.addSeparator()
        self.model_fburgers = QAction("Burgers (FBM)", self)
        self.model_fburgers.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional",
                model="FBM",
            )
        )
        fractional_menu.addAction(self.model_fburgers)
        self.model_fpoynting = QAction("Poynting-Thomson (FPT)", self)
        self.model_fpoynting.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional",
                model="FPT",
            )
        )
        fractional_menu.addAction(self.model_fpoynting)
        self.model_fjeffreys = QAction("Jeffreys (FJM)", self)
        self.model_fjeffreys.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fractional",
                model="FJM",
            )
        )
        fractional_menu.addAction(self.model_fjeffreys)

        # Multi-Mode Models submenu
        multimode_menu = models_menu.addMenu("&Multi-Mode")
        self.model_gmaxwell = QAction("Generalized Maxwell", self)
        self.model_gmaxwell.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Multi-Mode",
                model="GeneralizedMaxwell",
            )
        )
        multimode_menu.addAction(self.model_gmaxwell)

        # SGR Models submenu
        sgr_menu = models_menu.addMenu("&Soft Glassy Rheology")
        self.model_sgr_conventional = QAction("SGR Conventional", self)
        self.model_sgr_conventional.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="SGR",
                model="SGRConventional",
            )
        )
        sgr_menu.addAction(self.model_sgr_conventional)
        self.model_sgr_generic = QAction("SGR GENERIC", self)
        self.model_sgr_generic.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="SGR",
                model="SGRGeneric",
            )
        )
        sgr_menu.addAction(self.model_sgr_generic)

        # SPP LAOS Models submenu
        spp_menu = models_menu.addMenu("S&PP (LAOS)")
        self.model_spp_yield_stress = QAction("SPP Yield Stress", self)
        self.model_spp_yield_stress.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="SPP",
                model="SPPYieldStress",
            )
        )
        spp_menu.addAction(self.model_spp_yield_stress)

        models_menu.addSeparator()

        # STZ Models submenu
        stz_menu = models_menu.addMenu("ST&Z (Shear Transformation Zone)")
        self.model_stz_conventional = QAction("STZ Conventional", self)
        self.model_stz_conventional.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="STZ",
                model="STZConventional",
            )
        )
        stz_menu.addAction(self.model_stz_conventional)

        # EPM Models submenu
        epm_menu = models_menu.addMenu("&EPM (Elasto-Plastic)")
        self.model_lattice_epm = QAction("Lattice EPM", self)
        self.model_lattice_epm.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="EPM",
                model="LatticeEPM",
            )
        )
        epm_menu.addAction(self.model_lattice_epm)
        self.model_tensorial_epm = QAction("Tensorial EPM", self)
        self.model_tensorial_epm.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="EPM",
                model="TensorialEPM",
            )
        )
        epm_menu.addAction(self.model_tensorial_epm)

        # Fluidity Models submenu
        fluidity_menu = models_menu.addMenu("F&luidity")
        self.model_fluidity_local = QAction("Fluidity Local", self)
        self.model_fluidity_local.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fluidity",
                model="FluidityLocal",
            )
        )
        fluidity_menu.addAction(self.model_fluidity_local)
        self.model_fluidity_nonlocal = QAction("Fluidity Nonlocal", self)
        self.model_fluidity_nonlocal.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Fluidity",
                model="FluidityNonlocal",
            )
        )
        fluidity_menu.addAction(self.model_fluidity_nonlocal)

        # Fluidity-Saramito EVP Models submenu
        saramito_menu = models_menu.addMenu("Saramito (&EVP)")
        self.model_saramito_local = QAction("Saramito Local", self)
        self.model_saramito_local.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Saramito",
                model="SaramitoLocal",
            )
        )
        saramito_menu.addAction(self.model_saramito_local)
        self.model_saramito_nonlocal = QAction("Saramito Nonlocal", self)
        self.model_saramito_nonlocal.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Saramito",
                model="SaramitoNonlocal",
            )
        )
        saramito_menu.addAction(self.model_saramito_nonlocal)

        # IKH Models submenu
        ikh_menu = models_menu.addMenu("&IKH (Isotropic Kinematic)")
        self.model_mikh = QAction("MIKH", self)
        self.model_mikh.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="IKH",
                model="MIKH",
            )
        )
        ikh_menu.addAction(self.model_mikh)
        self.model_mlikh = QAction("MLIKH", self)
        self.model_mlikh.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="IKH",
                model="MLIKH",
            )
        )
        ikh_menu.addAction(self.model_mlikh)

        # FIKH Models submenu
        fikh_menu = models_menu.addMenu("FI&KH (Fractional IKH)")
        self.model_fikh = QAction("FIKH", self)
        self.model_fikh.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="FIKH",
                model="FIKH",
            )
        )
        fikh_menu.addAction(self.model_fikh)
        self.model_fmlikh = QAction("FMLIKH", self)
        self.model_fmlikh.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="FIKH",
                model="FMLIKH",
            )
        )
        fikh_menu.addAction(self.model_fmlikh)

        # Hébraud-Lequeux Models submenu
        hl_menu = models_menu.addMenu("Hébraud-&Lequeux")
        self.model_hebraud_lequeux = QAction("Hébraud-Lequeux", self)
        self.model_hebraud_lequeux.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="HL",
                model="HebraudLequeux",
            )
        )
        hl_menu.addAction(self.model_hebraud_lequeux)

        # ITT-MCT Models submenu
        itt_mct_menu = models_menu.addMenu("ITT-&MCT")
        self.model_itt_mct_schematic = QAction("Schematic (F₁₂)", self)
        self.model_itt_mct_schematic.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="ITT-MCT",
                model="Schematic",
            )
        )
        itt_mct_menu.addAction(self.model_itt_mct_schematic)
        self.model_itt_mct_isotropic = QAction("Isotropic (ISM)", self)
        self.model_itt_mct_isotropic.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="ITT-MCT",
                model="Isotropic",
            )
        )
        itt_mct_menu.addAction(self.model_itt_mct_isotropic)

        # DMT Thixotropic Models submenu
        dmt_menu = models_menu.addMenu("&DMT (Thixotropic)")
        self.model_dmt_local = QAction("DMT Local", self)
        self.model_dmt_local.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="DMT",
                model="DMTLocal",
            )
        )
        dmt_menu.addAction(self.model_dmt_local)
        self.model_dmt_nonlocal = QAction("DMT Nonlocal", self)
        self.model_dmt_nonlocal.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="DMT",
                model="DMTNonlocal",
            )
        )
        dmt_menu.addAction(self.model_dmt_nonlocal)

        # Giesekus Models submenu
        giesekus_menu = models_menu.addMenu("&Giesekus")
        self.model_giesekus_single = QAction("Single Mode", self)
        self.model_giesekus_single.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Giesekus",
                model="SingleMode",
            )
        )
        giesekus_menu.addAction(self.model_giesekus_single)
        self.model_giesekus_multi = QAction("Multi Mode", self)
        self.model_giesekus_multi.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="Giesekus",
                model="MultiMode",
            )
        )
        giesekus_menu.addAction(self.model_giesekus_multi)

        # TNT Transient Network Models submenu
        tnt_menu = models_menu.addMenu("&TNT (Transient Network)")
        self.model_tnt_single_mode = QAction("Single Mode", self)
        self.model_tnt_single_mode.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="TNT",
                model="SingleMode",
            )
        )
        tnt_menu.addAction(self.model_tnt_single_mode)
        self.model_tnt_cates = QAction("Cates", self)
        self.model_tnt_cates.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="TNT",
                model="Cates",
            )
        )
        tnt_menu.addAction(self.model_tnt_cates)
        self.model_tnt_loop_bridge = QAction("Loop-Bridge", self)
        self.model_tnt_loop_bridge.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="TNT",
                model="LoopBridge",
            )
        )
        tnt_menu.addAction(self.model_tnt_loop_bridge)
        self.model_tnt_multi_species = QAction("Multi-Species", self)
        self.model_tnt_multi_species.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="TNT",
                model="MultiSpecies",
            )
        )
        tnt_menu.addAction(self.model_tnt_multi_species)
        self.model_tnt_sticky_rouse = QAction("Sticky Rouse", self)
        self.model_tnt_sticky_rouse.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="TNT",
                model="StickyRouse",
            )
        )
        tnt_menu.addAction(self.model_tnt_sticky_rouse)

        # VLB Models submenu
        vlb_menu = models_menu.addMenu("V&LB (Viscoelastic Liquid-Bridge)")
        self.model_vlb_local = QAction("Local", self)
        self.model_vlb_local.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="VLB",
                model="Local",
            )
        )
        vlb_menu.addAction(self.model_vlb_local)
        self.model_vlb_multi_network = QAction("Multi-Network", self)
        self.model_vlb_multi_network.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="VLB",
                model="MultiNetwork",
            )
        )
        vlb_menu.addAction(self.model_vlb_multi_network)
        self.model_vlb_variant = QAction("Variant (Bell/FENE)", self)
        self.model_vlb_variant.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="VLB",
                model="Variant",
            )
        )
        vlb_menu.addAction(self.model_vlb_variant)
        self.model_vlb_nonlocal = QAction("Nonlocal", self)
        self.model_vlb_nonlocal.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="VLB",
                model="Nonlocal",
            )
        )
        vlb_menu.addAction(self.model_vlb_nonlocal)

        # HVM Hybrid Vitrimer Models submenu
        hvm_menu = models_menu.addMenu("H&VM (Hybrid Vitrimer)")
        self.model_hvm_local = QAction("HVM Local", self)
        self.model_hvm_local.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="HVM",
                model="HVMLocal",
            )
        )
        hvm_menu.addAction(self.model_hvm_local)

        # HVNM Vitrimer Nanocomposite Models submenu
        hvnm_menu = models_menu.addMenu("HVN&M (Vitrimer Nanocomposite)")
        self.model_hvnm_local = QAction("HVNM Local", self)
        self.model_hvnm_local.triggered.connect(
            lambda: logger.debug(
                "Menu item selected",
                menu="Models",
                submenu="HVNM",
                model="HVNMLocal",
            )
        )
        hvnm_menu.addAction(self.model_hvnm_local)

        logger.debug(
            "Menu created", menu="Models", action_count=models_menu.actions().__len__()
        )

    def _create_transforms_menu(self) -> None:
        """Create Transforms menu."""
        logger.debug("Creating menu", menu="Transforms")
        transforms_menu = self.addMenu("&Transforms")

        # FFT
        self.transform_fft = QAction("&FFT (Fourier Transform)", self)
        self.transform_fft.setStatusTip("Apply Fast Fourier Transform")
        self.transform_fft.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="transform_fft", menu="Transforms"
            )
        )
        transforms_menu.addAction(self.transform_fft)

        # Mastercurve
        self.transform_mastercurve = QAction("&Mastercurve (TTS)", self)
        self.transform_mastercurve.setStatusTip("Time-Temperature Superposition")
        self.transform_mastercurve.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="transform_mastercurve", menu="Transforms"
            )
        )
        transforms_menu.addAction(self.transform_mastercurve)

        # SRFS
        self.transform_srfs = QAction(
            "&SRFS (Strain-Rate Frequency Superposition)", self
        )
        self.transform_srfs.setStatusTip("Strain-Rate Frequency Superposition")
        self.transform_srfs.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="transform_srfs", menu="Transforms"
            )
        )
        transforms_menu.addAction(self.transform_srfs)

        transforms_menu.addSeparator()

        # Mutation Number
        self.transform_mutation = QAction("Mutation &Number", self)
        self.transform_mutation.setStatusTip("Calculate mutation number")
        self.transform_mutation.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="transform_mutation", menu="Transforms"
            )
        )
        transforms_menu.addAction(self.transform_mutation)

        # OWChirp
        self.transform_owchirp = QAction("&OWChirp", self)
        self.transform_owchirp.setStatusTip("Optimally Windowed Chirp transform")
        self.transform_owchirp.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="transform_owchirp", menu="Transforms"
            )
        )
        transforms_menu.addAction(self.transform_owchirp)

        # Derivatives
        self.transform_derivatives = QAction("&Derivatives", self)
        self.transform_derivatives.setStatusTip("Calculate numerical derivatives")
        self.transform_derivatives.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="transform_derivatives", menu="Transforms"
            )
        )
        transforms_menu.addAction(self.transform_derivatives)

        transforms_menu.addSeparator()

        # SPP Analysis
        self.transform_spp = QAction("S&PP (LAOS Analysis)", self)
        self.transform_spp.setStatusTip(
            "Sequence of Physical Processes yield stress extraction"
        )
        self.transform_spp.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="transform_spp", menu="Transforms"
            )
        )
        transforms_menu.addAction(self.transform_spp)

        logger.debug(
            "Menu created",
            menu="Transforms",
            action_count=transforms_menu.actions().__len__(),
        )

    def _create_analysis_menu(self) -> None:
        """Create Analysis menu."""
        logger.debug("Creating menu", menu="Analysis")
        analysis_menu = self.addMenu("&Analysis")

        # Fit NLSQ
        self.analysis_fit_nlsq = QAction("&Fit (NLSQ)", self)
        self.analysis_fit_nlsq.setShortcut(QKeySequence("Ctrl+F"))
        self.analysis_fit_nlsq.setStatusTip("Fit model using non-linear least squares")
        self.analysis_fit_nlsq.triggered.connect(
            lambda: logger.debug("Action triggered", action="fit_nlsq", menu="Analysis")
        )
        analysis_menu.addAction(self.analysis_fit_nlsq)

        # Fit Bayesian
        self.analysis_fit_bayesian = QAction("Fit &Bayesian (NUTS)", self)
        self.analysis_fit_bayesian.setShortcut(QKeySequence("Ctrl+B"))
        self.analysis_fit_bayesian.setStatusTip(
            "Fit model using Bayesian inference with NUTS"
        )
        self.analysis_fit_bayesian.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="fit_bayesian", menu="Analysis"
            )
        )
        analysis_menu.addAction(self.analysis_fit_bayesian)

        analysis_menu.addSeparator()

        # Batch Fit
        self.analysis_batch_fit = QAction("&Batch Fit...", self)
        self.analysis_batch_fit.setStatusTip("Fit multiple datasets in parallel")
        self.analysis_batch_fit.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="batch_fit", menu="Analysis"
            )
        )
        analysis_menu.addAction(self.analysis_batch_fit)

        # Compare Models
        self.analysis_compare = QAction("&Compare Models...", self)
        self.analysis_compare.setStatusTip("Compare multiple model fits")
        self.analysis_compare.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="compare_models", menu="Analysis"
            )
        )
        analysis_menu.addAction(self.analysis_compare)

        analysis_menu.addSeparator()

        # Compatibility Check
        self.analysis_compatibility = QAction("Check &Compatibility", self)
        self.analysis_compatibility.setStatusTip("Check model-data compatibility")
        self.analysis_compatibility.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="check_compatibility", menu="Analysis"
            )
        )
        analysis_menu.addAction(self.analysis_compatibility)

        logger.debug(
            "Menu created",
            menu="Analysis",
            action_count=analysis_menu.actions().__len__(),
        )

    def _create_tools_menu(self) -> None:
        """Create Tools menu."""
        logger.debug("Creating menu", menu="Tools")
        tools_menu = self.addMenu("&Tools")

        # Python Console (not yet implemented)
        self.tools_console = QAction("&Python Console", self)
        self.tools_console.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self.tools_console.setStatusTip("Python console (coming soon)")
        self.tools_console.setEnabled(False)
        tools_menu.addAction(self.tools_console)

        tools_menu.addSeparator()

        # JAX Profiler (not yet implemented)
        self.tools_jax_profiler = QAction("&JAX Profiler", self)
        self.tools_jax_profiler.setStatusTip("JAX profiler (coming soon)")
        self.tools_jax_profiler.setEnabled(False)
        tools_menu.addAction(self.tools_jax_profiler)

        # Memory Monitor
        # Memory Monitor (not yet implemented)
        self.tools_memory_monitor = QAction("&Memory Monitor", self)
        self.tools_memory_monitor.setStatusTip("Memory monitor (coming soon)")
        self.tools_memory_monitor.setEnabled(False)
        tools_menu.addAction(self.tools_memory_monitor)

        tools_menu.addSeparator()

        # Preferences (duplicate for convenience)
        self.tools_preferences = QAction("&Preferences...", self)
        self.tools_preferences.setShortcut(QKeySequence.StandardKey.Preferences)
        self.tools_preferences.setStatusTip("Open preferences dialog")
        self.tools_preferences.triggered.connect(
            lambda: logger.debug("Action triggered", action="preferences", menu="Tools")
        )
        tools_menu.addAction(self.tools_preferences)

        logger.debug(
            "Menu created", menu="Tools", action_count=tools_menu.actions().__len__()
        )

    def _create_help_menu(self) -> None:
        """Create Help menu."""
        logger.debug("Creating menu", menu="Help")
        help_menu = self.addMenu("&Help")

        # Documentation
        self.help_docs = QAction("&Documentation", self)
        self.help_docs.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.help_docs.setStatusTip("Open online documentation")
        self.help_docs.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="documentation", menu="Help"
            )
        )
        help_menu.addAction(self.help_docs)

        # Tutorials
        self.help_tutorials = QAction("&Tutorials", self)
        self.help_tutorials.setStatusTip("View tutorials")
        self.help_tutorials.triggered.connect(
            lambda: logger.debug("Action triggered", action="tutorials", menu="Help")
        )
        help_menu.addAction(self.help_tutorials)

        # Keyboard Shortcuts
        self.help_shortcuts = QAction("&Keyboard Shortcuts", self)
        self.help_shortcuts.setStatusTip("View keyboard shortcuts")
        self.help_shortcuts.triggered.connect(
            lambda: logger.debug(
                "Action triggered", action="keyboard_shortcuts", menu="Help"
            )
        )
        help_menu.addAction(self.help_shortcuts)

        help_menu.addSeparator()

        # About
        self.help_about = QAction("&About RheoJAX", self)
        self.help_about.setStatusTip("About RheoJAX")
        self.help_about.triggered.connect(
            lambda: logger.debug("Action triggered", action="about", menu="Help")
        )
        help_menu.addAction(self.help_about)

        logger.debug(
            "Menu created", menu="Help", action_count=help_menu.actions().__len__()
        )

    def _populate_recent_files(self) -> None:
        """Populate recent files menu (placeholder)."""
        logger.debug("Populating recent files menu")
        # This will be implemented to load recent files from config
        no_recent = QAction("No recent files", self)
        no_recent.setEnabled(False)
        self.recent_menu.addAction(no_recent)
