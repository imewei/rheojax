"""
Priors Editor Widget
===================

Interactive prior distribution editor for Bayesian inference.
"""

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Available prior distributions
DISTRIBUTIONS = [
    ("normal", "Normal", ["loc", "scale"]),
    ("lognormal", "Log-Normal", ["loc", "scale"]),
    ("uniform", "Uniform", ["low", "high"]),
    ("halfnormal", "Half-Normal", ["scale"]),
    ("exponential", "Exponential", ["scale"]),
    ("gamma", "Gamma", ["concentration", "rate"]),
    ("beta", "Beta", ["concentration0", "concentration1"]),
]

# Distribution parameter defaults
DIST_DEFAULTS = {
    "normal": {"loc": 1.0, "scale": 0.5},
    "lognormal": {"loc": 0.0, "scale": 1.0},
    "uniform": {"low": 0.0, "high": 1.0},
    "halfnormal": {"scale": 1.0},
    "exponential": {"scale": 1.0},
    "gamma": {"concentration": 2.0, "rate": 1.0},
    "beta": {"concentration0": 2.0, "concentration1": 2.0},
}


class PriorsEditor(QWidget):
    """Prior specification editor for Bayesian parameters.

    Features:
        - Distribution selector (Normal, Log-Normal, Uniform, etc.)
        - Interactive parameter controls with spinboxes
        - Live preview plot of prior distribution
        - Table view of all parameter priors
        - Import/export prior configurations

    Signals
    -------
    prior_changed : Signal(str, str, dict)
        Emitted when a prior is changed (param_name, dist_name, dist_params)
    priors_reset : Signal()
        Emitted when priors are reset to defaults

    Example
    -------
    >>> editor = PriorsEditor()  # doctest: +SKIP
    >>> editor.set_prior('G', 'normal', loc=1e6, scale=1e5)  # doctest: +SKIP
    >>> priors = editor.get_all_priors()  # doctest: +SKIP
    """

    prior_changed = Signal(str, str, dict)  # param_name, dist_name, dist_params
    priors_reset = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize priors editor.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._parameters: list[str] = []
        self._priors: dict[str, dict[str, Any]] = {}
        self._current_param: str | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Splitter for table and editor
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Parameter table
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_label = QLabel("Parameters")
        table_label.setStyleSheet("font-weight: bold;")
        table_layout.addWidget(table_label)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Parameter", "Distribution", "Values"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table_layout.addWidget(self._table)

        splitter.addWidget(table_frame)

        # Right: Editor panel
        editor_frame = QFrame()
        editor_layout = QVBoxLayout(editor_frame)
        editor_layout.setContentsMargins(8, 0, 0, 0)

        # Distribution selector
        dist_group = QGroupBox("Distribution")
        dist_layout = QFormLayout(dist_group)

        self._dist_combo = QComboBox()
        for dist_id, display_name, _ in DISTRIBUTIONS:
            self._dist_combo.addItem(display_name, dist_id)
        dist_layout.addRow("Type:", self._dist_combo)

        editor_layout.addWidget(dist_group)

        # Parameters group
        params_group = QGroupBox("Parameters")
        self._params_layout = QFormLayout(params_group)

        # Create spinboxes for all possible parameters
        self._param_spinboxes: dict[str, QDoubleSpinBox] = {}
        all_params = [
            "loc",
            "scale",
            "low",
            "high",
            "concentration",
            "rate",
            "concentration0",
            "concentration1",
        ]
        for param in all_params:
            spinbox = QDoubleSpinBox()
            spinbox.setDecimals(6)
            spinbox.setRange(-1e12, 1e12)
            spinbox.setValue(1.0)
            spinbox.setSingleStep(0.1)
            self._param_spinboxes[param] = spinbox
            self._params_layout.addRow(f"{param}:", spinbox)

        editor_layout.addWidget(params_group)

        # Preview plot
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self._preview_figure = Figure(figsize=(4, 3), dpi=80)
        self._preview_canvas = FigureCanvasQTAgg(self._preview_figure)
        self._preview_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview_layout.addWidget(self._preview_canvas)

        editor_layout.addWidget(preview_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip("Apply changes to selected parameter")
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Reset to default priors")
        btn_layout.addWidget(self._apply_btn)
        btn_layout.addWidget(self._reset_btn)
        btn_layout.addStretch()
        editor_layout.addLayout(btn_layout)

        splitter.addWidget(editor_frame)
        splitter.setSizes([200, 300])

        layout.addWidget(splitter)

        # Initially hide all parameter spinboxes
        self._update_param_visibility("normal")

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._dist_combo.currentIndexChanged.connect(self._on_dist_changed)
        self._apply_btn.clicked.connect(self._apply_prior)
        self._reset_btn.clicked.connect(self._reset_priors)

        # Connect all spinboxes to preview update
        for spinbox in self._param_spinboxes.values():
            spinbox.valueChanged.connect(self._update_preview)

    def set_parameters(self, parameters: list[str]) -> None:
        """Set model parameters to configure priors for.

        Parameters
        ----------
        parameters : list[str]
            List of parameter names
        """
        self._parameters = parameters
        self._priors = {}

        # Initialize with default priors
        for param in parameters:
            self._priors[param] = {
                "distribution": "normal",
                "params": {"loc": 1.0, "scale": 0.5},
            }

        self._rebuild_table()

    def _rebuild_table(self) -> None:
        """Rebuild the parameter table."""
        self._table.setRowCount(len(self._parameters))

        for row, param in enumerate(self._parameters):
            prior = self._priors.get(param, {})
            dist = prior.get("distribution", "normal")
            params = prior.get("params", {})

            # Parameter name
            name_item = QTableWidgetItem(param)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            # Distribution
            dist_item = QTableWidgetItem(dist)
            dist_item.setFlags(dist_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 1, dist_item)

            # Values
            values_str = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
            values_item = QTableWidgetItem(values_str)
            values_item.setFlags(values_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 2, values_item)

    def _on_selection_changed(self) -> None:
        """Handle table selection change."""
        rows = self._table.selectedIndexes()
        if not rows:
            return

        row = rows[0].row()
        if row < len(self._parameters):
            param = self._parameters[row]
            self._current_param = param
            self._load_prior_to_editor(param)

    def _load_prior_to_editor(self, param: str) -> None:
        """Load prior settings to editor panel.

        Parameters
        ----------
        param : str
            Parameter name
        """
        prior = self._priors.get(param, {})
        dist = prior.get("distribution", "normal")
        params = prior.get("params", DIST_DEFAULTS.get(dist, {}))

        # Set distribution
        idx = self._dist_combo.findData(dist)
        if idx >= 0:
            self._dist_combo.setCurrentIndex(idx)

        # Set parameter values
        for key, spinbox in self._param_spinboxes.items():
            if key in params:
                spinbox.blockSignals(True)
                spinbox.setValue(params[key])
                spinbox.blockSignals(False)

        self._update_param_visibility(dist)
        self._update_preview()

    def _on_dist_changed(self, index: int) -> None:
        """Handle distribution type change.

        Parameters
        ----------
        index : int
            New combo box index
        """
        dist = self._dist_combo.currentData()
        self._update_param_visibility(dist)

        # Load defaults for new distribution
        defaults = DIST_DEFAULTS.get(dist, {})
        for key, value in defaults.items():
            if key in self._param_spinboxes:
                self._param_spinboxes[key].blockSignals(True)
                self._param_spinboxes[key].setValue(value)
                self._param_spinboxes[key].blockSignals(False)

        self._update_preview()

    def _update_param_visibility(self, dist: str) -> None:
        """Update visibility of parameter spinboxes.

        Parameters
        ----------
        dist : str
            Distribution name
        """
        # Find required parameters for this distribution
        required_params = []
        for dist_id, _, params in DISTRIBUTIONS:
            if dist_id == dist:
                required_params = params
                break

        # Show/hide rows
        for i in range(self._params_layout.rowCount()):
            label = self._params_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
            field = self._params_layout.itemAt(i, QFormLayout.ItemRole.FieldRole)
            if label and field:
                label_widget = label.widget()
                field_widget = field.widget()
                if label_widget and field_widget:
                    param_name = label_widget.text().rstrip(":")
                    visible = param_name in required_params
                    label_widget.setVisible(visible)
                    field_widget.setVisible(visible)

    def _update_preview(self) -> None:
        """Update the preview plot."""
        self._preview_figure.clear()
        ax = self._preview_figure.add_subplot(111)

        dist = self._dist_combo.currentData()

        try:
            x, pdf = self._compute_pdf(dist)
            ax.fill_between(x, pdf, alpha=0.3)
            ax.plot(x, pdf, linewidth=2)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.set_title(f"{dist.title()} Distribution")
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error: {e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        self._preview_figure.tight_layout()
        self._preview_canvas.draw()

    def _compute_pdf(self, dist: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute PDF for preview.

        Parameters
        ----------
        dist : str
            Distribution name

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            x values and corresponding PDF values
        """
        from scipy import stats

        if dist == "normal":
            loc = self._param_spinboxes["loc"].value()
            scale = self._param_spinboxes["scale"].value()
            rv = stats.norm(loc=loc, scale=scale)
            x = np.linspace(loc - 4 * scale, loc + 4 * scale, 200)
        elif dist == "lognormal":
            loc = self._param_spinboxes["loc"].value()
            scale = self._param_spinboxes["scale"].value()
            rv = stats.lognorm(s=scale, loc=0, scale=np.exp(loc))
            x = np.linspace(0.001, rv.ppf(0.999), 200)
        elif dist == "uniform":
            low = self._param_spinboxes["low"].value()
            high = self._param_spinboxes["high"].value()
            rv = stats.uniform(loc=low, scale=high - low)
            margin = (high - low) * 0.1
            x = np.linspace(low - margin, high + margin, 200)
        elif dist == "halfnormal":
            scale = self._param_spinboxes["scale"].value()
            rv = stats.halfnorm(scale=scale)
            x = np.linspace(0, 4 * scale, 200)
        elif dist == "exponential":
            scale = self._param_spinboxes["scale"].value()
            rv = stats.expon(scale=scale)
            x = np.linspace(0, 5 * scale, 200)
        elif dist == "gamma":
            conc = self._param_spinboxes["concentration"].value()
            rate = self._param_spinboxes["rate"].value()
            rv = stats.gamma(a=conc, scale=1 / rate)
            x = np.linspace(0, rv.ppf(0.999), 200)
        elif dist == "beta":
            a = self._param_spinboxes["concentration0"].value()
            b = self._param_spinboxes["concentration1"].value()
            rv = stats.beta(a=a, b=b)
            x = np.linspace(0.001, 0.999, 200)
        else:
            raise ValueError(f"Unknown distribution: {dist}")

        return x, rv.pdf(x)

    def _apply_prior(self) -> None:
        """Apply current editor settings to selected parameter."""
        if self._current_param is None:
            return

        dist = self._dist_combo.currentData()

        # Get relevant parameters
        params = {}
        for dist_id, _, param_names in DISTRIBUTIONS:
            if dist_id == dist:
                for name in param_names:
                    params[name] = self._param_spinboxes[name].value()
                break

        # Update prior
        self._priors[self._current_param] = {"distribution": dist, "params": params}

        # Update table
        self._rebuild_table()

        # Emit signal
        self.prior_changed.emit(self._current_param, dist, params)

    def _reset_priors(self) -> None:
        """Reset all priors to defaults."""
        for param in self._parameters:
            self._priors[param] = {
                "distribution": "normal",
                "params": {"loc": 1.0, "scale": 0.5},
            }

        self._rebuild_table()
        self.priors_reset.emit()

    def set_prior(self, param: str, dist: str, **params: Any) -> None:
        """Set prior distribution for a parameter.

        Parameters
        ----------
        param : str
            Parameter name
        dist : str
            Distribution name
        **params
            Distribution parameters
        """
        if param not in self._parameters:
            self._parameters.append(param)

        self._priors[param] = {"distribution": dist, "params": params}
        self._rebuild_table()

    def get_prior(self, param: str) -> dict[str, Any] | None:
        """Get prior for a parameter.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        dict or None
            Prior specification or None if not found
        """
        return self._priors.get(param)

    def get_all_priors(self) -> dict[str, dict[str, Any]]:
        """Get all prior specifications.

        Returns
        -------
        dict
            All priors keyed by parameter name
        """
        return self._priors.copy()

    def clear(self) -> None:
        """Clear all priors."""
        self._parameters = []
        self._priors = {}
        self._current_param = None
        self._table.setRowCount(0)
        self._preview_figure.clear()
        self._preview_canvas.draw()
