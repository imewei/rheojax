"""
Toolbars
========

Main toolbar and quick fit strip for common actions and workflow.
"""


from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QLabel, QToolBar, QToolButton, QWidget


class MainToolBar(QToolBar):
    """Main application toolbar with common actions.

    Actions:
        - File operations: Open, Save, Import
        - Fitting: Fit, Bayesian, Stop
        - View: Zoom In, Zoom Out, Reset
        - Settings icon

    Example
    -------
    >>> toolbar = MainToolBar()  # doctest: +SKIP
    >>> toolbar.open_action.triggered.connect(on_open)  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize main toolbar.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__("Main Toolbar", parent)

        self.setObjectName("MainToolBar")
        self.setMovable(False)

        # File operations
        self.open_action = QAction("Open", self)
        self.open_action.setToolTip("Open file (Ctrl+O)")
        self.addAction(self.open_action)

        self.save_action = QAction("Save", self)
        self.save_action.setToolTip("Save file (Ctrl+S)")
        self.addAction(self.save_action)

        self.import_action = QAction("Import", self)
        self.import_action.setToolTip("Import data (Ctrl+I)")
        self.addAction(self.import_action)

        self.addSeparator()

        # Fitting operations
        self.fit_action = QAction("Fit", self)
        self.fit_action.setToolTip("Fit model (Ctrl+F)")
        self.addAction(self.fit_action)

        self.bayesian_action = QAction("Bayesian", self)
        self.bayesian_action.setToolTip("Bayesian inference (Ctrl+B)")
        self.addAction(self.bayesian_action)

        self.stop_action = QAction("Stop", self)
        self.stop_action.setToolTip("Stop current operation")
        self.stop_action.setEnabled(False)
        self.addAction(self.stop_action)

        self.addSeparator()

        # View controls
        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.setToolTip("Zoom in (Ctrl++)")
        self.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.setToolTip("Zoom out (Ctrl+-)")
        self.addAction(self.zoom_out_action)

        self.reset_zoom_action = QAction("Reset", self)
        self.reset_zoom_action.setToolTip("Reset zoom (Ctrl+0)")
        self.addAction(self.reset_zoom_action)

        self.addSeparator()

        # Settings
        self.settings_action = QAction("Settings", self)
        self.settings_action.setToolTip("Settings")
        self.addAction(self.settings_action)


class QuickFitStrip(QToolBar):
    """Quick fit workflow toolbar with visual flow arrows.

    Components:
        - Load button
        - Mode dropdown (oscillation, relaxation, creep, rotation)
        - Model dropdown (populated from ModelRegistry)
        - Fit button (prominent)
        - Plot button
        - Export button
        - Visual flow arrows between elements

    Signals
    -------
    load_clicked : Signal
        Emitted when load button clicked
    fit_clicked : Signal
        Emitted when fit button clicked
    plot_clicked : Signal
        Emitted when plot button clicked
    export_clicked : Signal
        Emitted when export button clicked

    Example
    -------
    >>> strip = QuickFitStrip()  # doctest: +SKIP
    >>> strip.fit_clicked.connect(on_fit)  # doctest: +SKIP
    """

    # Signals
    load_clicked = Signal()
    fit_clicked = Signal()
    plot_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize quick fit strip.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__("Quick Fit Strip", parent)

        self.setObjectName("QuickFitStrip")
        self.setMovable(False)

        # Load button
        self.load_button = QToolButton(self)
        self.load_button.setText("Load")
        self.load_button.setToolTip("Load data file")
        self.load_button.clicked.connect(self.load_clicked.emit)
        self.addWidget(self.load_button)

        # Arrow
        self.addWidget(self._create_arrow_label())

        # Mode dropdown
        self.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["oscillation", "relaxation", "creep", "rotation"])
        self.mode_combo.setToolTip("Select test mode")
        self.mode_combo.setMinimumWidth(120)
        self.addWidget(self.mode_combo)

        # Arrow
        self.addWidget(self._create_arrow_label())

        # Model dropdown
        self.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox(self)
        self._populate_models()
        self.model_combo.setToolTip("Select rheological model")
        self.model_combo.setMinimumWidth(180)
        self.addWidget(self.model_combo)

        # Arrow
        self.addWidget(self._create_arrow_label())

        # Fit button (prominent)
        self.fit_button = QToolButton(self)
        self.fit_button.setText("Fit")
        self.fit_button.setToolTip("Fit model to data (Ctrl+F)")
        self.fit_button.setStyleSheet("QToolButton { font-weight: bold; padding: 5px 15px; }")
        self.fit_button.clicked.connect(self.fit_clicked.emit)
        self.addWidget(self.fit_button)

        # Arrow
        self.addWidget(self._create_arrow_label())

        # Plot button
        self.plot_button = QToolButton(self)
        self.plot_button.setText("Plot")
        self.plot_button.setToolTip("Generate plot")
        self.plot_button.clicked.connect(self.plot_clicked.emit)
        self.addWidget(self.plot_button)

        # Arrow
        self.addWidget(self._create_arrow_label())

        # Export button
        self.export_button = QToolButton(self)
        self.export_button.setText("Export")
        self.export_button.setToolTip("Export results")
        self.export_button.clicked.connect(self.export_clicked.emit)
        self.addWidget(self.export_button)

    def _create_arrow_label(self) -> QLabel:
        """Create arrow label for visual flow.

        Returns
        -------
        QLabel
            Arrow label widget
        """
        arrow = QLabel("→")
        arrow.setStyleSheet("QLabel { font-size: 16px; color: #666; margin: 0 5px; }")
        return arrow

    def _populate_models(self) -> None:
        """Populate model dropdown with available models."""
        # Classical models
        self.model_combo.addItem("Classical Models", None)
        self.model_combo.addItem("  Maxwell", "maxwell")
        self.model_combo.addItem("  Zener (SLS)", "zener")
        self.model_combo.addItem("  SpringPot", "springpot")

        # Flow models
        self.model_combo.addItem("Flow Models (Non-Newtonian)", None)
        self.model_combo.addItem("  Power Law", "power_law")
        self.model_combo.addItem("  Carreau", "carreau")
        self.model_combo.addItem("  Carreau-Yasuda", "carreau_yasuda")
        self.model_combo.addItem("  Cross", "cross")
        self.model_combo.addItem("  Herschel-Bulkley", "herschel_bulkley")
        self.model_combo.addItem("  Bingham", "bingham")

        # Fractional models
        self.model_combo.addItem("Fractional Models", None)
        self.model_combo.addItem("  Fractional Maxwell Gel", "fractional_maxwell_gel")
        self.model_combo.addItem("  Fractional Maxwell Liquid", "fractional_maxwell_liquid")
        self.model_combo.addItem("  Fractional Maxwell Model", "fractional_maxwell_model")
        self.model_combo.addItem("  Fractional Kelvin-Voigt", "fractional_kelvin_voigt")
        self.model_combo.addItem("  Fractional Zener SL (FZSL)", "fractional_zener_sl")
        self.model_combo.addItem("  Fractional Zener SS (FZSS)", "fractional_zener_ss")
        self.model_combo.addItem("  Fractional Zener LL (FZLL)", "fractional_zener_ll")
        self.model_combo.addItem("  Fractional KV-Zener (FKVZ)", "fractional_kv_zener")
        self.model_combo.addItem("  Fractional Burgers (FBM)", "fractional_burgers")
        self.model_combo.addItem("  Fractional Poynting-Thomson (FPT)", "fractional_poynting_thomson")
        self.model_combo.addItem("  Fractional Jeffreys (FJM)", "fractional_jeffreys")

        # Multi-mode models
        self.model_combo.addItem("Multi-Mode Models", None)
        self.model_combo.addItem("  Generalized Maxwell", "generalized_maxwell")

        # SGR models
        self.model_combo.addItem("Soft Glassy Rheology", None)
        self.model_combo.addItem("  SGR Conventional", "sgr_conventional")
        self.model_combo.addItem("  SGR GENERIC", "sgr_generic")

        # Set first actual model as default (skip headers)
        self.model_combo.setCurrentIndex(1)

        # Disable headers
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) is None:
                # Disable header items
                model = self.model_combo.model()
                item = model.item(i)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)

    def get_mode(self) -> str:
        """Get selected test mode.

        Returns
        -------
        str
            Selected test mode (oscillation, relaxation, creep, rotation)
        """
        return self.mode_combo.currentText()

    def get_model(self) -> str:
        """Get selected model identifier.

        Returns
        -------
        str
            Selected model identifier
        """
        return self.model_combo.currentData() or ""

    def set_mode(self, mode: str) -> None:
        """Set test mode.

        Parameters
        ----------
        mode : str
            Test mode to set
        """
        index = self.mode_combo.findText(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)

    def set_model(self, model_id: str) -> None:
        """Set model by identifier.

        Parameters
        ----------
        model_id : str
            Model identifier
        """
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model_id:
                self.model_combo.setCurrentIndex(i)
                break
