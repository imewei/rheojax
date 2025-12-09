"""
Transform Page
=============

Transform application interface (mastercurve, FFT, SRFS, etc.).
"""

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.plot_canvas import PlotCanvas


class TransformPage(QWidget):
    """Transform application page."""

    transform_selected = Signal(str)
    transform_applied = Signal(str, str)  # transform_name, dataset_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = StateStore()
        self._selected_transform: str | None = None
        self._param_controls: dict[str, list] = {}
        self.setup_ui()

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
            ("Mastercurve", "Time-Temperature Superposition", "#2196F3"),
            ("SRFS", "Strain-Rate Frequency Superposition", "#4CAF50"),
            ("Mutation Number", "Calculate mutation number", "#9C27B0"),
            ("OW Chirp", "Optimally-windowed chirp analysis", "#FF9800"),
            ("SPP Analysis", "LAOS yield stress and cage modulus extraction", "#E91E63"),
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
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 8px;
                padding: 15px;
            }}
            QFrame:hover {{
                background-color: {self._darken(color)};
            }}
        """)
        card.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        name_label = QLabel(name)
        name_label.setStyleSheet("color: white; font-size: 13pt; font-weight: bold;")

        layout.addWidget(name_label)
        layout.addStretch()

        btn = QPushButton("Configure")
        btn.setStyleSheet("background-color: white; color: black; font-weight: bold; padding: 5px;")
        btn.clicked.connect(lambda: self._select_transform(name))
        layout.addWidget(btn)

        return card

    def _create_config_panel(self) -> QWidget:
        panel = QGroupBox("Configuration")
        layout = QVBoxLayout(panel)

        self._config_widget = QWidget()
        self._config_layout = QVBoxLayout(self._config_widget)
        self._config_layout.addWidget(QLabel("Select a transform to configure"))
        layout.addWidget(self._config_widget)

        layout.addStretch()

        btn_apply = QPushButton("Apply Transform")
        btn_apply.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
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
        before_layout.addWidget(QLabel("Before", styleSheet="font-weight: bold; font-size: 11pt;"))
        self._before_canvas = PlotCanvas()
        before_layout.addWidget(self._before_canvas)

        after_widget = QWidget()
        after_layout = QVBoxLayout(after_widget)
        after_layout.addWidget(QLabel("After", styleSheet="font-weight: bold; font-size: 11pt;"))
        self._after_canvas = PlotCanvas()
        after_layout.addWidget(self._after_canvas)

        splitter.addWidget(before_widget)
        splitter.addWidget(after_widget)
        splitter.setSizes([500, 500])

        layout.addWidget(splitter)

        return panel

    def _select_transform(self, name: str) -> None:
        self._selected_transform = name
        self.transform_selected.emit(name)

        # Clear and rebuild config
        while self._config_layout.count():
            child = self._config_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self._config_layout.addWidget(QLabel(f"{name} Parameters", styleSheet="font-weight: bold; font-size: 11pt;"))
        self._param_controls.clear()

        # Add transform-specific parameters
        if name == "Mastercurve":
            self._config_layout.addWidget(QLabel("Reference Temperature (°C):"))
            spin = QDoubleSpinBox()
            spin.setRange(-100, 300)
            spin.setValue(25)
            self._config_layout.addWidget(spin)
            self._param_controls.setdefault("mastercurve", []).append(("reference_temp", spin))

            check = QCheckBox("Auto-detect shift factors")
            check.setChecked(True)
            self._config_layout.addWidget(check)
            self._param_controls.setdefault("mastercurve", []).append(("auto_shift", check))

        elif name == "SRFS":
            self._config_layout.addWidget(QLabel("Reference Shear Rate (1/s):"))
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1000)
            spin.setValue(1.0)
            self._config_layout.addWidget(spin)
            self._param_controls.setdefault("srfs", []).append(("reference_gamma_dot", spin))

        elif name == "FFT":
            self._config_layout.addWidget(QLabel("Direction:"))
            combo_dir = QComboBox()
            combo_dir.addItems(["forward", "inverse"])
            self._config_layout.addWidget(combo_dir)
            self._param_controls.setdefault("fft", []).append(("direction", combo_dir))

            self._config_layout.addWidget(QLabel("Window Function:"))
            combo_window = QComboBox()
            combo_window.addItems(["hann", "hamming", "blackman", "rectangular"])
            self._config_layout.addWidget(combo_window)
            self._param_controls.setdefault("fft", []).append(("window", combo_window))

        elif name == "Mutation Number":
            self._config_layout.addWidget(QLabel("Integration Method:"))
            combo_method = QComboBox()
            combo_method.addItems(["trapz", "simpson", "romberg"])
            self._config_layout.addWidget(combo_method)
            self._param_controls.setdefault("mutation_number", []).append(("integration_method", combo_method))

            check_extrap = QCheckBox("Extrapolate data")
            check_extrap.setChecked(False)
            self._config_layout.addWidget(check_extrap)
            self._param_controls.setdefault("mutation_number", []).append(("extrapolate", check_extrap))

            self._config_layout.addWidget(QLabel("Extrapolation Model:"))
            combo_extrap = QComboBox()
            combo_extrap.addItems(["exponential", "power_law", "linear"])
            self._config_layout.addWidget(combo_extrap)
            self._param_controls.setdefault("mutation_number", []).append(("extrapolation_model", combo_extrap))

        elif name == "OW Chirp":
            self._config_layout.addWidget(QLabel("Min Frequency (rad/s):"))
            spin_min = QDoubleSpinBox()
            spin_min.setRange(0.0001, 1e6)
            spin_min.setValue(0.01)
            spin_min.setDecimals(4)
            self._config_layout.addWidget(spin_min)
            self._param_controls.setdefault("owchirp", []).append(("min_frequency", spin_min))

            self._config_layout.addWidget(QLabel("Max Frequency (rad/s):"))
            spin_max = QDoubleSpinBox()
            spin_max.setRange(0.0001, 1e6)
            spin_max.setValue(100.0)
            spin_max.setDecimals(4)
            self._config_layout.addWidget(spin_max)
            self._param_controls.setdefault("owchirp", []).append(("max_frequency", spin_max))

        elif name == "Derivatives":
            self._config_layout.addWidget(QLabel("Order:"))
            spin_order = QDoubleSpinBox()
            spin_order.setRange(1, 4)
            spin_order.setDecimals(0)
            spin_order.setValue(1)
            self._config_layout.addWidget(spin_order)
            self._param_controls.setdefault("derivative", []).append(("order", spin_order))

            self._config_layout.addWidget(QLabel("Window Length:"))
            spin_window = QDoubleSpinBox()
            spin_window.setRange(3, 201)
            spin_window.setDecimals(0)
            spin_window.setValue(11)
            self._config_layout.addWidget(spin_window)
            self._param_controls.setdefault("derivative", []).append(("window_length", spin_window))

            self._config_layout.addWidget(QLabel("Polynomial Order:"))
            spin_poly = QDoubleSpinBox()
            spin_poly.setRange(1, 10)
            spin_poly.setDecimals(0)
            spin_poly.setValue(3)
            self._config_layout.addWidget(spin_poly)
            self._param_controls.setdefault("derivative", []).append(("poly_order", spin_poly))

            self._config_layout.addWidget(QLabel("Mode (padding):"))
            combo_mode = QComboBox()
            combo_mode.addItems(["mirror", "nearest", "constant", "wrap"])
            self._config_layout.addWidget(combo_mode)
            self._param_controls.setdefault("derivative", []).append(("mode", combo_mode))

            self._config_layout.addWidget(QLabel("Validate Window Length (odd):"))
            check_validate = QCheckBox("Force odd window length")
            check_validate.setChecked(True)
            self._config_layout.addWidget(check_validate)
            self._param_controls.setdefault("derivative", []).append(("validate_window", check_validate))

        elif name == "SPP Analysis":
            self._config_layout.addWidget(QLabel("Angular Frequency (rad/s):"))
            spin_omega = QDoubleSpinBox()
            spin_omega.setRange(0.001, 1000)
            spin_omega.setValue(1.0)
            spin_omega.setDecimals(3)
            self._config_layout.addWidget(spin_omega)
            self._param_controls.setdefault("spp", []).append(("omega", spin_omega))

            self._config_layout.addWidget(QLabel("Strain Amplitude (γ0):"))
            spin_gamma0 = QDoubleSpinBox()
            spin_gamma0.setRange(0.0001, 100.0)
            spin_gamma0.setValue(1.0)
            spin_gamma0.setDecimals(4)
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
            self._config_layout.addWidget(spin_harmonics)
            self._param_controls.setdefault("spp", []).append(("n_harmonics", spin_harmonics))

            self._config_layout.addWidget(QLabel("Start Cycle (skip transients):"))
            spin_start = QDoubleSpinBox()
            spin_start.setRange(0, 100)
            spin_start.setValue(0)
            spin_start.setDecimals(0)
            self._config_layout.addWidget(spin_start)
            self._param_controls.setdefault("spp", []).append(("start_cycle", spin_start))

            self._config_layout.addWidget(QLabel("End Cycle (optional):"))
            spin_end = QDoubleSpinBox()
            spin_end.setRange(0, 1000)
            spin_end.setValue(0)
            spin_end.setDecimals(0)
            self._config_layout.addWidget(spin_end)
            self._param_controls.setdefault("spp", []).append(("end_cycle", spin_end))

            self._config_layout.addWidget(QLabel("Yield Tolerance:"))
            spin_tol = QDoubleSpinBox()
            spin_tol.setRange(0.0001, 1.0)
            spin_tol.setDecimals(4)
            spin_tol.setValue(0.02)
            self._config_layout.addWidget(spin_tol)
            self._param_controls.setdefault("spp", []).append(("yield_tolerance", spin_tol))

            check_numerical = QCheckBox("Use numerical method (MATLAB-compatible)")
            check_numerical.setChecked(False)
            self._config_layout.addWidget(check_numerical)
            self._param_controls.setdefault("spp", []).append(("use_numerical_method", check_numerical))

        self._config_layout.addStretch()

    def _apply_transform(self) -> None:
        if not self._selected_transform:
            return

        dataset = self._store.get_active_dataset()
        if dataset:
            self.transform_applied.emit(self._selected_transform, dataset.id)

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
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"#{int(r*0.9):02x}{int(g*0.9):02x}{int(b*0.9):02x}"

    def apply_transform(self, transform_name: str, dataset_ids: list[str], params: dict[str, Any] | None = None) -> None:
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
            return

        # Use provided params or get from current UI
        if params is None:
            params = self.get_selected_params()

        # Set the selected transform for signal emission
        self._selected_transform = transform_name

        # Emit signal for first dataset (MainWindow handles the rest)
        self.transform_applied.emit(transform_name, dataset_ids[0])

    def show_transform_preview(self, transform_name: str, params: dict[str, Any]) -> None:
        """Update preview canvases with before/after visualization.

        Parameters
        ----------
        transform_name : str
            Name of the transform to preview
        params : dict
            Transform parameters
        """
        import numpy as np

        # Get active dataset for preview
        dataset = self._store.get_active_dataset()
        if dataset is None:
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
        except Exception:
            # If plot fails, clear the canvas
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
