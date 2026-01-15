"""
Transform Page
=============

Transform application interface (mastercurve, FFT, SRFS, etc.).
"""

from typing import Any

from rheojax.gui.compat import (
    Qt,
    Signal,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.plot_canvas import PlotCanvas
from rheojax.logging import get_logger

logger = get_logger(__name__)


class TransformPage(QWidget):
    """Transform application page."""

    transform_selected = Signal(str)
    transform_applied = Signal(str, str)  # transform_name, dataset_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self._store = StateStore()
        self._selected_transform: str | None = None
        self._param_controls: dict[str, list] = {}
        self.setup_ui()
        logger.debug(
            "Initialization complete",
            class_name=self.__class__.__name__,
            page="TransformPage",
        )

    def setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Transform cards grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(15)

        transforms = [
            ("FFT", "Fast Fourier Transform for frequency analysis", "#FF5722"),
            ("Mastercurve", "Time-temperature superposition", "#2196F3"),
            ("SRFS", "Strain-rate frequency superposition", "#4CAF50"),
            ("Mutation Number", "Calculate mutation number", "#9C27B0"),
            ("OW Chirp", "Optimally-windowed chirp analysis", "#FF9800"),
            (
                "SPP Analysis",
                "LAOS yield stress and cage modulus extraction",
                "#E91E63",
            ),
            ("Derivatives", "Calculate numerical derivatives", "#607D8B"),
        ]

        for i, (name, desc, color) in enumerate(transforms):
            card = self._create_transform_card(name, desc, color)
            grid_layout.addWidget(card, i // 3, i % 3)

        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, 1)

        # Config and preview
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_config_panel())
        splitter.addWidget(self._create_preview_panel())
        splitter.setSizes([300, 700])
        layout.addWidget(splitter, 2)

    def _create_transform_card(self, name: str, desc: str, color: str) -> QWidget:
        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        # Let height adapt to content but stay compact in the grid.
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        card.setMaximumHeight(200)
        card.setStyleSheet(
            f"""
            QFrame {{
                background-color: {color};
                border-radius: 8px;
                padding: 15px;
            }}
            QFrame:hover {{
                background-color: {self._darken(color)};
            }}
        """
        )
        card.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        name_label = QLabel(name)
        name_label.setStyleSheet("color: white; font-size: 13pt; font-weight: bold;")

        layout.addWidget(name_label)
        layout.addStretch()

        btn = QPushButton("Configure")
        btn.setStyleSheet(
            "background-color: white; color: black; font-weight: bold; padding: 5px;"
        )
        btn.clicked.connect(lambda: self._select_transform(name))
        layout.addWidget(btn)

        return card

    def _create_config_panel(self) -> QWidget:
        panel = QGroupBox("Configuration")
        layout = QVBoxLayout(panel)

        self._config_widget = QWidget()
        self._config_layout = QVBoxLayout(self._config_widget)
        placeholder = QLabel("Select a transform to configure")
        placeholder.setStyleSheet("color: #666;")
        self._config_layout.addWidget(placeholder)
        layout.addWidget(self._config_widget)

        layout.addStretch()

        btn_apply = QPushButton("Apply Transform")
        btn_apply.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        btn_apply.clicked.connect(self._apply_transform)
        layout.addWidget(btn_apply)

        return panel

    def _create_preview_panel(self) -> QWidget:
        panel = QGroupBox("Preview")
        layout = QVBoxLayout(panel)

        # Before/after split
        splitter = QSplitter(Qt.Horizontal)

        before_widget = QWidget()
        before_layout = QVBoxLayout(before_widget)
        before_layout.addWidget(
            QLabel("Before", styleSheet="font-weight: bold; font-size: 11pt;")
        )
        self._before_canvas = PlotCanvas()
        before_layout.addWidget(self._before_canvas)

        after_widget = QWidget()
        after_layout = QVBoxLayout(after_widget)
        after_layout.addWidget(
            QLabel("After", styleSheet="font-weight: bold; font-size: 11pt;")
        )
        self._after_canvas = PlotCanvas()
        after_layout.addWidget(self._after_canvas)

        splitter.addWidget(before_widget)
        splitter.addWidget(after_widget)
        splitter.setSizes([500, 500])

        layout.addWidget(splitter)

        return panel

    def _select_transform(self, name: str) -> None:
        logger.debug("Transform selected", transform=name, page="TransformPage")
        self._selected_transform = name
        self.transform_selected.emit(name)

        # Clear and rebuild config
        while self._config_layout.count():
            child = self._config_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self._config_layout.addWidget(
            QLabel(
                f"{name} Parameters", styleSheet="font-weight: bold; font-size: 11pt;"
            )
        )
        self._param_controls.clear()

        # Add transform-specific parameters
        if name == "Mastercurve":
            self._config_layout.addWidget(QLabel("Reference Temperature (°C):"))
            spin = QDoubleSpinBox()
            spin.setRange(-100, 300)
            spin.setValue(25)
            spin.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="reference_temp",
                    value=v,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin)
            self._param_controls.setdefault("mastercurve", []).append(
                ("reference_temp", spin)
            )

            check = QCheckBox("Auto-detect shift factors")
            check.setChecked(True)
            check.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="auto_shift",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(check)
            self._param_controls.setdefault("mastercurve", []).append(
                ("auto_shift", check)
            )

            self._config_layout.addWidget(QLabel("Shift Method:"))
            combo_shift = QComboBox()
            combo_shift.addItems(["wlf", "arrhenius", "manual"])
            combo_shift.setCurrentText("wlf")
            combo_shift.currentTextChanged.connect(
                lambda t: logger.debug(
                    "Parameter changed",
                    parameter="shift_method",
                    value=t,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(combo_shift)
            self._param_controls.setdefault("mastercurve", []).append(
                ("shift_method", combo_shift)
            )

        elif name == "SRFS":
            self._config_layout.addWidget(QLabel("Reference Shear Rate (1/s):"))
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1000)
            spin.setValue(1.0)
            spin.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="reference_gamma_dot",
                    value=v,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin)
            self._param_controls.setdefault("srfs", []).append(
                ("reference_gamma_dot", spin)
            )

            auto_shift = QCheckBox("Auto-detect shift factors")
            auto_shift.setChecked(True)
            auto_shift.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="auto_shift",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(auto_shift)
            self._param_controls.setdefault("srfs", []).append(
                ("auto_shift", auto_shift)
            )

        elif name == "FFT":
            self._config_layout.addWidget(QLabel("Direction:"))
            combo_dir = QComboBox()
            combo_dir.addItems(["forward", "inverse"])
            combo_dir.currentTextChanged.connect(
                lambda t: logger.debug(
                    "Parameter changed",
                    parameter="direction",
                    value=t,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(combo_dir)
            self._param_controls.setdefault("fft", []).append(("direction", combo_dir))

            self._config_layout.addWidget(QLabel("Window Function:"))
            combo_window = QComboBox()
            combo_window.addItems(["hann", "hamming", "blackman", "rectangular"])
            combo_window.currentTextChanged.connect(
                lambda t: logger.debug(
                    "Parameter changed",
                    parameter="window",
                    value=t,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(combo_window)
            self._param_controls.setdefault("fft", []).append(("window", combo_window))

            check_detrend = QCheckBox("Detrend (remove DC)")
            check_detrend.setChecked(True)
            check_detrend.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="detrend",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(check_detrend)
            self._param_controls.setdefault("fft", []).append(
                ("detrend", check_detrend)
            )

            check_norm = QCheckBox("Normalize amplitude")
            check_norm.setChecked(True)
            check_norm.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="normalize",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(check_norm)
            self._param_controls.setdefault("fft", []).append(("normalize", check_norm))

            check_psd = QCheckBox("Return PSD")
            check_psd.setChecked(False)
            check_psd.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="return_psd",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(check_psd)
            self._param_controls.setdefault("fft", []).append(("return_psd", check_psd))

        elif name == "Mutation Number":
            self._config_layout.addWidget(QLabel("Integration Method:"))
            combo_method = QComboBox()
            combo_method.addItems(["trapz", "simpson", "romberg"])
            combo_method.currentTextChanged.connect(
                lambda t: logger.debug(
                    "Parameter changed",
                    parameter="integration_method",
                    value=t,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(combo_method)
            self._param_controls.setdefault("mutation_number", []).append(
                ("integration_method", combo_method)
            )

            check_extrap = QCheckBox("Extrapolate data")
            check_extrap.setChecked(False)
            check_extrap.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="extrapolate",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(check_extrap)
            self._param_controls.setdefault("mutation_number", []).append(
                ("extrapolate", check_extrap)
            )

            self._config_layout.addWidget(QLabel("Extrapolation Model:"))
            combo_extrap = QComboBox()
            combo_extrap.addItems(["exponential", "power_law", "linear"])
            combo_extrap.currentTextChanged.connect(
                lambda t: logger.debug(
                    "Parameter changed",
                    parameter="extrapolation_model",
                    value=t,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(combo_extrap)
            self._param_controls.setdefault("mutation_number", []).append(
                ("extrapolation_model", combo_extrap)
            )

        elif name == "OW Chirp":
            self._config_layout.addWidget(QLabel("Min Frequency (rad/s):"))
            spin_min = QDoubleSpinBox()
            spin_min.setRange(0.0001, 1e6)
            spin_min.setValue(0.01)
            spin_min.setDecimals(4)
            spin_min.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="min_frequency",
                    value=v,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_min)
            self._param_controls.setdefault("owchirp", []).append(
                ("min_frequency", spin_min)
            )

            self._config_layout.addWidget(QLabel("Max Frequency (rad/s):"))
            spin_max = QDoubleSpinBox()
            spin_max.setRange(0.0001, 1e6)
            spin_max.setValue(100.0)
            spin_max.setDecimals(4)
            spin_max.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="max_frequency",
                    value=v,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_max)
            self._param_controls.setdefault("owchirp", []).append(
                ("max_frequency", spin_max)
            )

            self._config_layout.addWidget(QLabel("# Frequencies:"))
            spin_n = QDoubleSpinBox()
            spin_n.setRange(4, 5000)
            spin_n.setDecimals(0)
            spin_n.setValue(100)
            spin_n.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="n_frequencies",
                    value=int(v),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_n)
            self._param_controls.setdefault("owchirp", []).append(
                ("n_frequencies", spin_n)
            )

            self._config_layout.addWidget(QLabel("Wavelet Width:"))
            spin_w = QDoubleSpinBox()
            spin_w.setRange(1.0, 20.0)
            spin_w.setDecimals(2)
            spin_w.setValue(5.0)
            spin_w.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="wavelet_width",
                    value=v,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_w)
            self._param_controls.setdefault("owchirp", []).append(
                ("wavelet_width", spin_w)
            )

            check_harm = QCheckBox("Extract harmonics")
            check_harm.setChecked(True)
            check_harm.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="extract_harmonics",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(check_harm)
            self._param_controls.setdefault("owchirp", []).append(
                ("extract_harmonics", check_harm)
            )

            self._config_layout.addWidget(QLabel("Max Harmonic:"))
            spin_h = QDoubleSpinBox()
            spin_h.setRange(1, 99)
            spin_h.setDecimals(0)
            spin_h.setValue(7)
            spin_h.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="max_harmonic",
                    value=int(v),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_h)
            self._param_controls.setdefault("owchirp", []).append(
                ("max_harmonic", spin_h)
            )

        elif name == "Derivatives":
            self._config_layout.addWidget(QLabel("Order:"))
            spin_order = QDoubleSpinBox()
            spin_order.setRange(1, 4)
            spin_order.setDecimals(0)
            spin_order.setValue(1)
            spin_order.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="order",
                    value=int(v),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_order)
            self._param_controls.setdefault("derivative", []).append(
                ("order", spin_order)
            )

            self._config_layout.addWidget(QLabel("Window Length:"))
            spin_window = QDoubleSpinBox()
            spin_window.setRange(3, 201)
            spin_window.setDecimals(0)
            spin_window.setValue(11)
            spin_window.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="window_length",
                    value=int(v),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_window)
            self._param_controls.setdefault("derivative", []).append(
                ("window_length", spin_window)
            )

            self._config_layout.addWidget(QLabel("Polynomial Order:"))
            spin_poly = QDoubleSpinBox()
            spin_poly.setRange(1, 10)
            spin_poly.setDecimals(0)
            spin_poly.setValue(3)
            spin_poly.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="poly_order",
                    value=int(v),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_poly)
            self._param_controls.setdefault("derivative", []).append(
                ("poly_order", spin_poly)
            )

            self._config_layout.addWidget(QLabel("Mode (padding):"))
            combo_mode = QComboBox()
            combo_mode.addItems(["mirror", "nearest", "constant", "wrap"])
            combo_mode.currentTextChanged.connect(
                lambda t: logger.debug(
                    "Parameter changed",
                    parameter="mode",
                    value=t,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(combo_mode)
            self._param_controls.setdefault("derivative", []).append(
                ("mode", combo_mode)
            )

            self._config_layout.addWidget(QLabel("Validate Window Length (odd):"))
            check_validate = QCheckBox("Force odd window length")
            check_validate.setChecked(True)
            check_validate.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="validate_window",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(check_validate)
            self._param_controls.setdefault("derivative", []).append(
                ("validate_window", check_validate)
            )

        elif name == "SPP Analysis":
            self._config_layout.addWidget(QLabel("Angular Frequency (rad/s):"))
            spin_omega = QDoubleSpinBox()
            spin_omega.setRange(0.001, 1000)
            spin_omega.setValue(1.0)
            spin_omega.setDecimals(3)
            spin_omega.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="omega",
                    value=v,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_omega)
            self._param_controls.setdefault("spp", []).append(("omega", spin_omega))

            self._config_layout.addWidget(QLabel("Strain Amplitude (gamma_0):"))
            spin_gamma0 = QDoubleSpinBox()
            spin_gamma0.setRange(0.0001, 100.0)
            spin_gamma0.setValue(1.0)
            spin_gamma0.setDecimals(4)
            spin_gamma0.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="gamma_0",
                    value=v,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_gamma0)
            self._param_controls.setdefault("spp", []).append(("gamma_0", spin_gamma0))

            self._config_layout.addWidget(QLabel("Strain Amplitude:"))
            spin_gamma = QDoubleSpinBox()
            spin_gamma.setRange(0.001, 100)
            spin_gamma.setValue(1.0)
            spin_gamma.setDecimals(3)

            self._config_layout.addWidget(spin_gamma)

            self._config_layout.addWidget(QLabel("Number of Harmonics:"))
            spin_harmonics = QDoubleSpinBox()
            spin_harmonics.setRange(1, 99)
            spin_harmonics.setValue(39)
            spin_harmonics.setDecimals(0)
            spin_harmonics.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="n_harmonics",
                    value=int(v),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_harmonics)
            self._param_controls.setdefault("spp", []).append(
                ("n_harmonics", spin_harmonics)
            )

            self._config_layout.addWidget(QLabel("Start Cycle (skip transients):"))
            spin_start = QDoubleSpinBox()
            spin_start.setRange(0, 100)
            spin_start.setValue(0)
            spin_start.setDecimals(0)
            spin_start.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="start_cycle",
                    value=int(v),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_start)
            self._param_controls.setdefault("spp", []).append(
                ("start_cycle", spin_start)
            )

            self._config_layout.addWidget(QLabel("End Cycle (optional):"))
            spin_end = QDoubleSpinBox()
            spin_end.setRange(0, 1000)
            spin_end.setValue(0)
            spin_end.setDecimals(0)
            spin_end.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="end_cycle",
                    value=int(v),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_end)
            self._param_controls.setdefault("spp", []).append(("end_cycle", spin_end))

            self._config_layout.addWidget(QLabel("Yield Tolerance:"))
            spin_tol = QDoubleSpinBox()
            spin_tol.setRange(0.0001, 1.0)
            spin_tol.setDecimals(4)
            spin_tol.setValue(0.02)
            spin_tol.valueChanged.connect(
                lambda v: logger.debug(
                    "Parameter changed",
                    parameter="yield_tolerance",
                    value=v,
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(spin_tol)
            self._param_controls.setdefault("spp", []).append(
                ("yield_tolerance", spin_tol)
            )

            check_numerical = QCheckBox("Use numerical method (MATLAB-compatible)")
            check_numerical.setChecked(False)
            check_numerical.stateChanged.connect(
                lambda s: logger.debug(
                    "Parameter changed",
                    parameter="use_numerical_method",
                    value=bool(s),
                    page="TransformPage",
                )
            )
            self._config_layout.addWidget(check_numerical)
            self._param_controls.setdefault("spp", []).append(
                ("use_numerical_method", check_numerical)
            )

        self._config_layout.addStretch()

    def _apply_transform(self) -> None:
        if not self._selected_transform:
            logger.debug(
                "Apply transform called with no transform selected",
                page="TransformPage",
            )
            return

        logger.debug(
            "Transform triggered",
            transform=self._selected_transform,
            page="TransformPage",
        )

        dataset = self._store.get_active_dataset()
        if dataset:
            self.transform_applied.emit(self._selected_transform, dataset.id)
            logger.info(
                "Transform applied",
                transform=self._selected_transform,
                dataset_id=dataset.id,
                page="TransformPage",
            )
        else:
            logger.debug(
                "No active dataset for transform",
                transform=self._selected_transform,
                page="TransformPage",
            )

    def get_selected_params(self) -> dict[str, Any]:
        """Return current parameter values for the selected transform."""

        if not self._selected_transform:
            return {}

        key = self._selected_transform.lower().replace(" ", "_")
        params: dict[str, Any] = {}
        for name, widget in self._param_controls.get(key, []):
            if isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText().lower()
        # Enforce Savitzky-Golay odd window if requested
        if key == "derivative" and params.get("validate_window", False):
            window_val = int(params.get("window_length", 11))
            if window_val % 2 == 0:
                params["window_length"] = window_val + 1
        return params

    def _darken(self, hex_color: str) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
        return f"#{int(r*0.9):02x}{int(g*0.9):02x}{int(b*0.9):02x}"

    def apply_transform(
        self,
        transform_name: str,
        dataset_ids: list[str],
        params: dict[str, Any] | None = None,
    ) -> None:
        """Apply transform to datasets via signal emission.

        This method triggers the transform_applied signal which is handled
        by MainWindow to delegate to TransformService.

        Parameters
        ----------
        transform_name : str
            Name of the transform to apply
        dataset_ids : list[str]
            IDs of datasets to transform
        params : dict, optional
            Transform parameters (uses current UI values if not provided)
        """
        if not dataset_ids:
            logger.debug(
                "apply_transform called with empty dataset_ids",
                transform=transform_name,
                page="TransformPage",
            )
            return

        logger.debug(
            "Transform triggered",
            transform=transform_name,
            dataset_count=len(dataset_ids),
            page="TransformPage",
        )

        # Use provided params or get from current UI
        if params is None:
            params = self.get_selected_params()

        # Set the selected transform for signal emission
        self._selected_transform = transform_name

        # Emit signal for first dataset (MainWindow handles the rest)
        self.transform_applied.emit(transform_name, dataset_ids[0])
        logger.info(
            "Transform applied",
            transform=transform_name,
            dataset_id=dataset_ids[0],
            page="TransformPage",
        )

    def show_transform_preview(
        self, transform_name: str, params: dict[str, Any]
    ) -> None:
        """Update preview canvases with before/after visualization.

        Parameters
        ----------
        transform_name : str
            Name of the transform to preview
        params : dict
            Transform parameters
        """
        logger.debug(
            "Showing transform preview",
            transform=transform_name,
            params=params,
            page="TransformPage",
        )

        # Get active dataset for preview
        dataset = self._store.get_active_dataset()
        if dataset is None:
            logger.debug(
                "No active dataset for preview",
                transform=transform_name,
                page="TransformPage",
            )
            return

        # Update "Before" canvas with original data
        try:
            self._before_canvas.plot(
                dataset.x,
                dataset.y,
                label="Original",
                xlabel=dataset.x_units or "x",
                ylabel=dataset.y_units or "y",
                title="Before Transform",
            )
            logger.debug(
                "Preview before canvas updated",
                transform=transform_name,
                page="TransformPage",
            )
        except Exception as e:
            # If plot fails, clear the canvas
            logger.error(
                "Failed to update before canvas",
                transform=transform_name,
                error=str(e),
                page="TransformPage",
                exc_info=True,
            )
            self._before_canvas.clear()

        # For "After" canvas, we'd need to actually compute the transform
        # which requires the TransformService. For now, show placeholder.
        # The actual preview is computed when MainWindow calls TransformService.preview_transform()
        self._after_canvas.clear()

    def get_available_transforms(self) -> list[dict[str, Any]]:
        """Return list of available transforms with metadata.

        Returns
        -------
        list[dict]
            Transform definitions with name, description, color, and key
        """
        return [
            {
                "name": "FFT",
                "key": "fft",
                "description": "Fast Fourier Transform for frequency analysis",
                "color": "#FF5722",
                "requires_multiple": False,
            },
            {
                "name": "Mastercurve",
                "key": "mastercurve",
                "description": "Time-Temperature Superposition",
                "color": "#2196F3",
                "requires_multiple": True,
            },
            {
                "name": "SRFS",
                "key": "srfs",
                "description": "Strain-Rate Frequency Superposition",
                "color": "#4CAF50",
                "requires_multiple": True,
            },
            {
                "name": "Mutation Number",
                "key": "mutation_number",
                "description": "Calculate mutation number",
                "color": "#9C27B0",
                "requires_multiple": False,
            },
            {
                "name": "OW Chirp",
                "key": "owchirp",
                "description": "Optimally-windowed chirp analysis",
                "color": "#FF9800",
                "requires_multiple": False,
            },
            {
                "name": "SPP Analysis",
                "key": "spp",
                "description": "LAOS yield stress and cage modulus extraction",
                "color": "#E91E63",
                "requires_multiple": False,
            },
            {
                "name": "Derivatives",
                "key": "derivative",
                "description": "Calculate numerical derivatives",
                "color": "#607D8B",
                "requires_multiple": False,
            },
        ]
