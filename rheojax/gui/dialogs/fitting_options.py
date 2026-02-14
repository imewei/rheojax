"""
Fitting Options Dialog
=====================

Configure NLSQ optimization parameters.
"""

from typing import Any

from rheojax.gui.compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    Qt,
    QVBoxLayout,
    QWidget,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


class FittingOptionsDialog(QDialog):
    """Fitting configuration dialog.

    Options:
        - Optimization algorithm
        - Convergence tolerances
        - Multi-start parameters
        - Bounds usage

    Example
    -------
    >>> dialog = FittingOptionsDialog()  # doctest: +SKIP
    >>> if dialog.exec() == QDialog.DialogCode.Accepted:  # doctest: +SKIP
    ...     options = dialog.get_options()  # doctest: +SKIP
    """

    def __init__(
        self,
        current_options: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize fitting options dialog.

        Parameters
        ----------
        current_options : dict[str, Any], optional
            Current fitting options
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        # Guard against non-dict inputs (e.g., accidental widget objects)
        self.current_options = (
            current_options if isinstance(current_options, dict) else {}
        )

        self.setWindowTitle("Fitting Options")
        self.setMinimumSize(500, 400)

        self._setup_ui()
        self._load_current_options()

    def _setup_ui(self) -> None:
        """Set up user interface."""
        layout = QVBoxLayout()

        # Algorithm selection
        algo_group = QGroupBox("Algorithm")
        algo_layout = QFormLayout()

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(
            ["NLSQ (default)", "NLSQ Global", "L-BFGS-B", "Trust Region"]
        )
        self.algo_combo.currentTextChanged.connect(self._on_algorithm_changed)
        algo_layout.addRow("Optimization Method:", self.algo_combo)

        # Jacobian mode
        self.jacobian_combo = QComboBox()
        self.jacobian_combo.addItems(["Auto", "Forward (fwd)", "Reverse (rev)"])
        self.jacobian_combo.setToolTip(
            "Jacobian computation mode: Auto selects based on problem size. "
            "Forward is better for few parameters, Reverse for many."
        )
        self.jacobian_combo.currentTextChanged.connect(
            lambda text: self._on_option_changed("jacobian_mode", text)
        )
        algo_layout.addRow("Jacobian Mode:", self.jacobian_combo)

        # Stability check
        self.stability_check = QCheckBox("Enable stability checks")
        self.stability_check.setChecked(False)
        self.stability_check.setToolTip(
            "Run stability analysis on the fitted parameters"
        )
        self.stability_check.stateChanged.connect(
            lambda s: self._on_option_changed("stability", "auto" if s != 0 else None)
        )
        algo_layout.addRow("", self.stability_check)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # Convergence parameters
        conv_group = QGroupBox("Convergence Parameters")
        conv_layout = QFormLayout()

        # Max iterations
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 50000)
        self.max_iter_spin.setValue(5000)
        self.max_iter_spin.setSingleStep(500)
        self.max_iter_spin.valueChanged.connect(
            lambda v: self._on_option_changed("max_iter", v)
        )
        conv_layout.addRow("Max Iterations:", self.max_iter_spin)

        # Function tolerance
        self.ftol_spin = QDoubleSpinBox()
        self.ftol_spin.setDecimals(10)
        self.ftol_spin.setRange(1e-15, 1e-3)
        self.ftol_spin.setValue(1e-8)
        self.ftol_spin.setSingleStep(1e-9)
        self.ftol_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self.ftol_spin.valueChanged.connect(
            lambda v: self._on_option_changed("ftol", v)
        )
        conv_layout.addRow("Function Tolerance (ftol):", self.ftol_spin)

        # Parameter tolerance
        self.xtol_spin = QDoubleSpinBox()
        self.xtol_spin.setDecimals(10)
        self.xtol_spin.setRange(1e-15, 1e-3)
        self.xtol_spin.setValue(1e-8)
        self.xtol_spin.setSingleStep(1e-9)
        self.xtol_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self.xtol_spin.valueChanged.connect(
            lambda v: self._on_option_changed("xtol", v)
        )
        conv_layout.addRow("Parameter Tolerance (xtol):", self.xtol_spin)

        conv_group.setLayout(conv_layout)
        layout.addWidget(conv_group)

        # Multi-start optimization
        multistart_group = QGroupBox("Multi-Start Optimization")
        multistart_layout = QVBoxLayout()

        # Enable multi-start
        self.multistart_check = QCheckBox("Enable multi-start optimization")
        self.multistart_check.setChecked(False)
        self.multistart_check.stateChanged.connect(self._on_multistart_changed)
        multistart_layout.addWidget(self.multistart_check)

        # Number of starts
        starts_layout = QHBoxLayout()
        starts_layout.addWidget(QLabel("Number of starts:"))
        self.num_starts_spin = QSpinBox()
        self.num_starts_spin.setRange(2, 20)
        self.num_starts_spin.setValue(5)
        self.num_starts_spin.setEnabled(False)
        self.num_starts_spin.valueChanged.connect(
            lambda v: self._on_option_changed("num_starts", v)
        )
        starts_layout.addWidget(self.num_starts_spin)
        starts_layout.addStretch()
        multistart_layout.addLayout(starts_layout)

        multistart_group.setLayout(multistart_layout)
        layout.addWidget(multistart_group)

        # Other options
        other_group = QGroupBox("Other Options")
        other_layout = QVBoxLayout()

        # Use bounds
        self.use_bounds_check = QCheckBox("Use parameter bounds")
        self.use_bounds_check.setChecked(True)
        self.use_bounds_check.stateChanged.connect(
            lambda s: self._on_option_changed("use_bounds", s != 0)
        )
        other_layout.addWidget(self.use_bounds_check)

        # Verbose output
        self.verbose_check = QCheckBox("Verbose output")
        self.verbose_check.setChecked(False)
        self.verbose_check.stateChanged.connect(
            lambda s: self._on_option_changed("verbose", s != 0)
        )
        other_layout.addWidget(self.verbose_check)

        other_group.setLayout(other_layout)
        layout.addWidget(other_group)

        # Reset to defaults button
        reset_layout = QHBoxLayout()
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        reset_layout.addStretch()
        reset_layout.addWidget(reset_button)
        layout.addLayout(reset_layout)

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self._on_rejected)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _load_current_options(self) -> None:
        """Load current options into UI."""
        if not self.current_options:
            return

        # Algorithm
        if "algorithm" in self.current_options:
            algo = self.current_options["algorithm"]
            # Use MatchStartsWith so "NLSQ" matches "NLSQ (default)"
            idx = self.algo_combo.findText(algo, flags=Qt.MatchFlag.MatchStartsWith)
            if idx >= 0:
                self.algo_combo.setCurrentIndex(idx)

        # Max iterations
        if "max_iter" in self.current_options:
            self.max_iter_spin.setValue(self.current_options["max_iter"])

        # Tolerances
        if "ftol" in self.current_options:
            self.ftol_spin.setValue(self.current_options["ftol"])

        if "xtol" in self.current_options:
            self.xtol_spin.setValue(self.current_options["xtol"])

        # Multi-start
        if "multistart" in self.current_options:
            multistart = self.current_options["multistart"]
            self.multistart_check.setChecked(multistart)

        if "num_starts" in self.current_options:
            self.num_starts_spin.setValue(self.current_options["num_starts"])

        # Use bounds
        if "use_bounds" in self.current_options:
            self.use_bounds_check.setChecked(self.current_options["use_bounds"])

        # Verbose
        if "verbose" in self.current_options:
            self.verbose_check.setChecked(self.current_options["verbose"])

        # Jacobian mode
        if "jacobian_mode" in self.current_options:
            jac = self.current_options["jacobian_mode"]
            jac_map = {"fwd": "Forward (fwd)", "rev": "Reverse (rev)"}
            text = jac_map.get(jac, "Auto")
            idx = self.jacobian_combo.findText(text)
            if idx >= 0:
                self.jacobian_combo.setCurrentIndex(idx)

        # Stability
        if "stability" in self.current_options:
            self.stability_check.setChecked(
                self.current_options["stability"] is not None
            )

    def _on_algorithm_changed(self, text: str) -> None:
        """Handle algorithm combo box change."""
        logger.debug(
            "Option changed",
            dialog=self.__class__.__name__,
            option="algorithm",
            value=text,
        )

    def _on_option_changed(self, option: str, value: Any) -> None:
        """Handle option change."""
        logger.debug(
            "Option changed",
            dialog=self.__class__.__name__,
            option=option,
            value=value,
        )

    def _on_multistart_changed(self, state: int) -> None:
        """Handle multi-start checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        self.num_starts_spin.setEnabled(enabled)
        logger.debug(
            "Option changed",
            dialog=self.__class__.__name__,
            option="multistart",
            value=enabled,
        )

    def _reset_defaults(self) -> None:
        """Reset all options to defaults."""
        logger.debug("Resetting to defaults", dialog=self.__class__.__name__)
        self.algo_combo.setCurrentIndex(0)  # NLSQ
        self.max_iter_spin.setValue(5000)
        self.ftol_spin.setValue(1e-8)
        self.xtol_spin.setValue(1e-8)
        self.multistart_check.setChecked(False)
        self.num_starts_spin.setValue(5)
        self.use_bounds_check.setChecked(True)
        self.verbose_check.setChecked(False)
        self.jacobian_combo.setCurrentIndex(0)  # Auto
        self.stability_check.setChecked(False)

    def _on_accepted(self) -> None:
        """Handle dialog accepted."""
        logger.debug("Options applied", dialog=self.__class__.__name__)
        self.accept()

    def _on_rejected(self) -> None:
        """Handle dialog rejected."""
        logger.debug("Dialog closed", dialog=self.__class__.__name__)
        self.reject()

    def showEvent(self, event) -> None:
        """Handle show event."""
        super().showEvent(event)
        logger.debug("Dialog opened", dialog=self.__class__.__name__)

    def get_options(self) -> dict[str, Any]:
        """Get fitting options.

        Returns
        -------
        dict[str, Any]
            Fitting options with keys:
            - algorithm: Optimization algorithm
            - max_iter: Maximum iterations
            - ftol: Function tolerance
            - xtol: Parameter tolerance
            - multistart: Whether to use multi-start
            - num_starts: Number of starts (if multistart)
            - use_bounds: Whether to use parameter bounds
            - verbose: Whether to print verbose output
        """
        algo_text = self.algo_combo.currentText()
        # Extract algorithm name (remove " (default)" suffix if present)
        algorithm = algo_text.split(" (")[0]

        options = {
            "algorithm": algorithm,
            "max_iter": self.max_iter_spin.value(),
            "ftol": self.ftol_spin.value(),
            "xtol": self.xtol_spin.value(),
            "multistart": self.multistart_check.isChecked(),
            "use_bounds": self.use_bounds_check.isChecked(),
            "verbose": self.verbose_check.isChecked(),
        }

        # Add num_starts only if multistart is enabled
        if options["multistart"]:
            options["num_starts"] = self.num_starts_spin.value()

        # Jacobian mode
        jacobian_text = self.jacobian_combo.currentText()
        if "fwd" in jacobian_text.lower():
            options["jacobian_mode"] = "fwd"
        elif "rev" in jacobian_text.lower():
            options["jacobian_mode"] = "rev"
        # else: Auto (omit to let backend decide)

        # Stability
        if self.stability_check.isChecked():
            options["stability"] = "auto"

        return options


# Alias for backward compatibility
FittingOptions = FittingOptionsDialog
