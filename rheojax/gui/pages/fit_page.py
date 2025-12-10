"""
Fit Page
========

Model fitting interface with parameter controls and residual analysis.
"""

from typing import Any

import numpy as np
from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.jobs.fit_worker import FitResult, FitWorker
from rheojax.gui.jobs.worker_pool import WorkerPool
from rheojax.gui.services.model_service import ModelService
from rheojax.gui.state.actions import (
    fitting_completed,
    fitting_failed,
    set_active_model,
    start_fitting,
    update_fit_progress,
)
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.parameter_table import ParameterTable
from rheojax.gui.widgets.plot_canvas import PlotCanvas
from rheojax.gui.widgets.residuals_panel import ResidualsPanel
from matplotlib.figure import Figure


class FitPage(QWidget):
    """Model fitting page with integrated model browser and service calls.

    Features:
        - Model browser for model selection
        - Parameter table with editable values
        - Plot canvas showing data and fit
        - Residuals panel for fit diagnostics
        - Background fitting with progress updates
    """

    fit_requested = Signal(str, str)  # model_name, dataset_id
    fit_completed = Signal(object)  # FitResult
    parameter_changed = Signal(str, float)  # param_name, value

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize fit page.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._store = StateStore()
        self._model_service = ModelService()
        self._worker_pool = WorkerPool()
        self._current_worker: FitWorker | None = None
        # Persist user-selected fitting options; start with dialog defaults
        self._fit_options: dict[str, Any] = {
            "algorithm": "NLSQ",
            "max_iter": 5000,
            "ftol": 1e-8,
            "xtol": 1e-8,
            "multistart": False,
            "num_starts": 5,
            "use_bounds": True,
            "verbose": False,
        }

        self._setup_ui()
        self._connect_signals()
        self._load_models()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)

        # Left: Model browser (removed to avoid duplication; selection via context controls)
        self._model_browser = None

        # Center: Visualization (40%)
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, 2)

        # Right: Parameters and controls (30%)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def _create_center_panel(self) -> QWidget:
        """Create center panel with plot and residuals."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Plot canvas
        self._plot_canvas = PlotCanvas()
        layout.addWidget(self._plot_canvas, 3)
        self._plot_placeholder = QLabel("No fit plot yet. Run a fit to see results.")
        self._plot_placeholder.setAlignment(Qt.AlignCenter)
        self._plot_placeholder.setStyleSheet("color: #94A3B8; padding: 6px;")
        layout.addWidget(self._plot_placeholder)

        # Residuals panel
        self._residuals_panel = ResidualsPanel()
        layout.addWidget(self._residuals_panel, 1)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right panel with parameters and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Context selectors (mode + quick model)
        context_group = QGroupBox("Context")
        context_layout = QHBoxLayout(context_group)
        context_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["oscillation", "relaxation", "creep", "rotation"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        context_layout.addWidget(self._mode_combo)
        context_layout.addWidget(QLabel("Model:"))
        self._quick_model_combo = QComboBox()
        self._quick_model_combo.setEditable(True)
        self._quick_model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._quick_model_combo.currentIndexChanged.connect(self._on_quick_model_changed)
        context_layout.addWidget(self._quick_model_combo, 1)
        layout.addWidget(context_group)

        # Compatibility status
        compat_group = QGroupBox("Compatibility")
        compat_layout = QVBoxLayout(compat_group)
        self._compat_label = QLabel("Select a model and dataset")
        self._compat_label.setWordWrap(True)
        compat_layout.addWidget(self._compat_label)
        layout.addWidget(compat_group)

        # Parameter table
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        self._parameter_table = ParameterTable()
        self._parameter_table.parameter_changed.connect(self._on_parameter_changed)
        param_layout.addWidget(self._parameter_table)
        empty_label = QLabel("No model loaded. Select a model to edit parameters.")
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet("color: #666; padding: 6px;")
        param_layout.addWidget(empty_label)
        self._empty_label = empty_label
        layout.addWidget(param_group)

        # Fit button and options
        btn_layout = QHBoxLayout()
        btn_options = QPushButton("Options...")
        btn_options.setProperty("variant", "secondary")
        btn_options.clicked.connect(self._show_fit_options)
        btn_layout.addWidget(btn_options)

        self._btn_fit = QPushButton("Fit Model")
        self._btn_fit.setProperty("variant", "primary")
        self._btn_fit.clicked.connect(self._on_fit_clicked)
        self._btn_fit.setEnabled(False)
        self._btn_fit.setToolTip("Run the selected model with the current dataset and parameters")
        btn_layout.addWidget(self._btn_fit, 2)

        layout.addLayout(btn_layout)

        # Results display
        results_group = QGroupBox("Fit Results")
        results_layout = QVBoxLayout(results_group)
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(150)
        results_layout.addWidget(self._results_text)

        self._empty_results = QLabel("No fit has been run yet. Configure parameters and click Fit.")
        self._empty_results.setAlignment(Qt.AlignCenter)
        self._empty_results.setStyleSheet("color: #94A3B8; padding: 8px;")
        results_layout.addWidget(self._empty_results)
        layout.addWidget(results_group)

        return panel

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Store signals (only if signals are set)
        if self._store.signals is not None:
            self._store.signals.dataset_selected.connect(self._on_dataset_changed)
            self._store.signals.fit_started.connect(self._on_fitting_started)
            self._store.signals.fit_completed.connect(self._on_fitting_completed)

    def _load_models(self) -> None:
        """Load available models into browser."""
        models = self._model_service.get_available_models()
        self._populate_quick_model_selector(models)

    def _populate_quick_model_selector(self, models: dict[str, list[str]]) -> None:
        """Populate quick model selector with available models."""
        self._quick_model_combo.blockSignals(True)
        self._quick_model_combo.clear()
        self._quick_model_combo.addItem("Select model...", None)
        for category in models.values():
            for model_name in category:
                self._quick_model_combo.addItem(model_name, model_name)
        self._quick_model_combo.blockSignals(False)

    @Slot(str)
    def _on_model_selected(self, model_name: str) -> None:
        """Handle model selection.

        Parameters
        ----------
        model_name : str
            Selected model name
        """
        # Update state
        self._store.dispatch(set_active_model(model_name))

        # Load model parameters
        model_info = self._model_service.get_model_info(model_name)
        params = model_info.get("parameters", {})

        # Import ParameterState for table format
        from rheojax.gui.state.store import ParameterState

        # Convert to dict[str, ParameterState] format for parameter table
        param_states: dict[str, ParameterState] = {}
        for name, details in params.items():
            bounds = details.get("bounds", (0.0, float("inf")))
            param_states[name] = ParameterState(
                name=name,
                value=details.get("default", 1.0),
                min_bound=bounds[0] if bounds[0] is not None else 0.0,
                max_bound=bounds[1] if bounds[1] is not None else float("inf"),
                fixed=False,
                unit=details.get("units", ""),
                description=details.get("description", ""),
            )

        self._parameter_table.set_parameters(param_states)

        # Check compatibility with active dataset
        self._check_compatibility(model_name)

        # Enable fit button if we have data
        dataset = self._store.get_active_dataset()
        self._btn_fit.setEnabled(dataset is not None)

        # Sync quick selector
        idx = self._quick_model_combo.findData(model_name)
        if idx >= 0:
            was_blocked = self._quick_model_combo.blockSignals(True)
            self._quick_model_combo.setCurrentIndex(idx)
            self._quick_model_combo.blockSignals(was_blocked)

    @Slot(str)
    def _on_model_double_clicked(self, model_name: str) -> None:
        """Handle model double-click (quick fit).

        Parameters
        ----------
        model_name : str
            Double-clicked model name
        """
        self._on_model_selected(model_name)
        if self._btn_fit.isEnabled():
            self._on_fit_clicked()

    @Slot()
    def _on_dataset_changed(self) -> None:
        """Handle active dataset change."""
        dataset = self._store.get_active_dataset()
        model_name = None

        if dataset is not None:
            # Plot data
            self._plot_data(dataset)
            # Update mode indicator
            if dataset.test_mode:
                idx = self._mode_combo.findText(dataset.test_mode)
                if idx >= 0:
                    was_blocked = self._mode_combo.blockSignals(True)
                    self._mode_combo.setCurrentIndex(idx)
                    self._mode_combo.blockSignals(was_blocked)

            # Check compatibility
            if model_name:
                self._check_compatibility(model_name)
                self._btn_fit.setEnabled(True)
        else:
            self._btn_fit.setEnabled(False)
            self._compat_label.setText("Select a dataset to enable fitting")
            self._compat_label.setStyleSheet("color: gray;")

    def _on_mode_changed(self, mode: str) -> None:
        """Update store test mode from selector."""
        if mode:
            self._store.dispatch("SET_TEST_MODE", {"test_mode": mode})
            self._compat_label.setText(f"Mode set to {mode}")
            self._compat_label.setStyleSheet("color: #555;")

    def _on_quick_model_changed(self, index: int) -> None:
        """Handle quick model combo changes."""
        model_name = self._quick_model_combo.itemData(index)
        if model_name:
            self._on_model_selected(model_name)

    def _check_compatibility(self, model_name: str) -> None:
        """Check model-data compatibility.

        Parameters
        ----------
        model_name : str
            Model name to check
        """
        dataset = self._store.get_active_dataset()
        if dataset is None:
            self._compat_label.setText("No dataset loaded")
            self._compat_label.setStyleSheet("color: gray;")
            return

        rheo_data = self._to_rheodata(dataset)
        result = self._model_service.check_compatibility(
            model_name, rheo_data, rheo_data.metadata.get("test_mode")
        )

        if result.get("compatible", True):
            text = f"Compatible\nDecay: {result.get('decay_type', 'unknown')}"
            self._compat_label.setText(text)
            self._compat_label.setStyleSheet("color: green;")
        else:
            warnings = result.get("warnings", [])
            text = "Compatibility issues:\n" + "\n".join(f"- {w}" for w in warnings[:3])
            self._compat_label.setText(text)
            self._compat_label.setStyleSheet("color: orange;")

    def _plot_data(self, dataset: Any) -> None:
        """Plot dataset on canvas.

        Accepts either a RheoData object or a DatasetState from the store.
        """
        if dataset is None:
            return

        # Support both RheoData and DatasetState
        x = getattr(dataset, "x", None)
        y = getattr(dataset, "y", None)
        metadata = getattr(dataset, "metadata", {}) or {}

        if x is None or y is None:
            x = getattr(dataset, "x_data", None)
            y = getattr(dataset, "y_data", None)

        if x is None or y is None:
            return

        x = np.asarray(x)
        y = np.asarray(y)

        if x.size == 0 or y.size == 0:
            return

        self._plot_canvas.clear()
        ax = self._plot_canvas.get_axes()

        if np.iscomplexobj(y):
            ax.loglog(x, np.real(y), "o", label="G' (Storage)")
            ax.loglog(x, np.abs(np.imag(y)), "s", label="G'' (Loss)")
            ax.legend()
        else:
            ax.loglog(x, y, "o", label="Data")

        x_label = metadata.get("x_label") or metadata.get("x_column") or "x"
        y_label = metadata.get("y_label") or metadata.get("y_column") or "y"
        title = metadata.get("name") or metadata.get("file")
        if not title and getattr(dataset, "file_path", None):
            title = getattr(dataset, "file_path").name
        if not title:
            title = "Data"

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        self._plot_canvas.refresh()

    def _to_rheodata(self, dataset) -> Any:
        """Convert DatasetState or RheoData into RheoData for services/workers."""
        from rheojax.core.data import RheoData

        # Already RheoData-like
        if hasattr(dataset, "x") and hasattr(dataset, "y"):
            return dataset

        x = getattr(dataset, "x_data", None)
        y = getattr(dataset, "y_data", None)
        if x is None or y is None:
            raise ValueError("Dataset is missing x/y data for fitting")

        metadata = dict(getattr(dataset, "metadata", {}) or {})
        if getattr(dataset, "file_path", None) and "file" not in metadata:
            metadata["file"] = str(dataset.file_path)
        if getattr(dataset, "name", None):
            metadata.setdefault("name", dataset.name)
        metadata.setdefault("test_mode", getattr(dataset, "test_mode", "unknown"))

        return RheoData(
            x=x,
            y=y,
            x_units=metadata.get("x_units"),
            y_units=metadata.get("y_units"),
            domain=metadata.get("domain", "time"),
            metadata=metadata,
            validate=False,
        )

    def _on_parameter_changed(self, param_name: str, value: float) -> None:
        """Handle parameter value change.

        Parameters
        ----------
        param_name : str
            Parameter name
        value : float
            New value
        """
        self.parameter_changed.emit(param_name, value)

    def _on_fit_clicked(self) -> None:
        """Handle fit button click."""
        # Get model from quick selector (model browser was removed)
        model_name = self._quick_model_combo.currentData()
        dataset = self._store.get_active_dataset()

        if not model_name or not dataset:
            return

        # Get current parameter values
        params = self._parameter_table.get_parameters()
        param_dict = {name: state.value for name, state in params.items()}

        # Determine test mode
        test_mode = dataset.metadata.get("test_mode", "oscillation")

        # Update state
        self._store.dispatch(start_fitting(model_name, dataset.id))

        # Create and run fit worker
        self._current_worker = FitWorker(
            model_name=model_name,
            data=self._to_rheodata(dataset),
            initial_params=param_dict,
            options=self._fit_options,
        )
        self._current_worker.signals.progress.connect(self._on_fit_progress)
        self._current_worker.signals.completed.connect(self._on_fit_finished)
        self._current_worker.signals.failed.connect(self._on_fit_error)

        self._worker_pool.submit(self._current_worker)
        self.fit_requested.emit(model_name, dataset.id)

    @Slot(int, float, str)
    def _on_fit_progress(self, iteration: int, loss: float, message: str) -> None:
        """Handle fit progress update.

        Parameters
        ----------
        iteration : int
            Current optimization iteration
        loss : float
            Current loss value
        message : str
            Status message
        """
        self._store.dispatch(update_fit_progress(iteration))
        self._results_text.setText(f"Fitting... Iteration {iteration}\nLoss: {loss:.6e}\n{message}")
        if hasattr(self, "_empty_results"):
            self._empty_results.hide()

    @Slot(object)
    def _on_fit_finished(self, result: FitResult) -> None:
        """Handle fit completion.

        Parameters
        ----------
        result : FitResult
            Fitting result from FitWorker
        """
        self._current_worker = None

        if result.success:
            self._store.dispatch(fitting_completed(result))

            # Update parameter table with fitted values
            from rheojax.gui.state.store import ParameterState

            param_states: dict[str, ParameterState] = {}
            for name, value in result.parameters.items():
                param_states[name] = ParameterState(
                    name=name,
                    value=value,
                    min_bound=0.0,
                    max_bound=float("inf"),
                    fixed=False,
                    unit="",
                    description="",
                )
            self._parameter_table.set_parameters(param_states)

            # Update results text
            chi_sq = result.chi_squared
            text = f"Fit successful!\n\nR²: {result.r_squared:.4f}\n"
            text += f"Chi-squared: {chi_sq:.4g}\n"
            text += f"MPE: {result.mpe:.2f}%\n"
            text += f"Time: {result.fit_time:.2f}s\n\nParameters:\n"
            for name, value in result.parameters.items():
                text += f"  {name}: {value:.4g}\n"
            self._results_text.setText(text)
            if hasattr(self, "_empty_results"):
                self._empty_results.hide()

            self.fit_completed.emit(result)
        else:
            error_msg = f"Fit did not converge (iterations: {result.n_iterations})"
            self._store.dispatch(fitting_failed(error_msg))
            self._results_text.setText(f"Fit failed:\n{error_msg}")
            if hasattr(self, "_empty_results"):
                self._empty_results.hide()

    @Slot(str)
    def _on_fit_error(self, error_msg: str) -> None:
        """Handle fit error.

        Parameters
        ----------
        error_msg : str
            Error message
        """
        self._current_worker = None
        self._store.dispatch(fitting_failed(error_msg))
        self._results_text.setText(f"Fit error:\n{error_msg}")
        if hasattr(self, "_empty_results"):
            self._empty_results.hide()
        QMessageBox.warning(self, "Fit Error", error_msg)

    @Slot()
    def _on_fitting_started(self) -> None:
        """Handle fitting started signal."""
        self._btn_fit.setEnabled(False)
        self._btn_fit.setText("Fitting...")

    @Slot()
    def _on_fitting_completed(self) -> None:
        """Handle fitting completed signal."""
        self._btn_fit.setEnabled(True)
        self._btn_fit.setText("Fit Model")

    def _plot_fit(self, result: FitResult) -> None:
        """Plot fit result on canvas.

        Parameters
        ----------
        result : FitResult
            Fitting result
        """
        ax = self._plot_canvas.get_axes()

        # Plot fit line
        if np.iscomplexobj(result.y_fit):
            ax.loglog(result.x_fit, np.real(result.y_fit), "-",
                     color="C0", linewidth=2, label="G' Fit")
            ax.loglog(result.x_fit, np.abs(np.imag(result.y_fit)), "-",
                     color="C1", linewidth=2, label="G'' Fit")
        else:
            ax.loglog(result.x_fit, result.y_fit, "-",
                     color="C0", linewidth=2, label="Fit")

        ax.legend()
        self._plot_canvas.refresh()

    def _show_fit_options(self) -> None:
        """Show fitting options dialog."""
        from rheojax.gui.dialogs.fitting_options import FittingOptionsDialog
        dialog = FittingOptionsDialog(current_options=self._fit_options, parent=self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self._fit_options = dialog.get_options()

    def fit_model(
        self,
        model_name: str,
        dataset_id: str,
        test_mode: str,
        initial_params: dict[str, float] | None = None,
    ) -> None:
        """Programmatically fit a model.

        Parameters
        ----------
        model_name : str
            Model name
        dataset_id : str
            Dataset ID
        test_mode : str
            Test mode
        initial_params : dict, optional
            Initial parameter values
        """
        # Select model via quick selector (model browser was removed)
        idx = self._quick_model_combo.findData(model_name)
        if idx >= 0:
            self._quick_model_combo.setCurrentIndex(idx)
        else:
            # Model not in combo, trigger selection directly
            self._on_model_selected(model_name)

        # Set initial params if provided
        if initial_params:
            from rheojax.gui.state.store import ParameterState

            param_states: dict[str, ParameterState] = {}
            for name, value in initial_params.items():
                param_states[name] = ParameterState(
                    name=name,
                    value=value,
                    min_bound=0.0,
                    max_bound=float("inf"),
                    fixed=False,
                    unit="",
                    description="",
                )
            self._parameter_table.set_parameters(param_states)

        # Trigger fit
        self._on_fit_clicked()

    def update_parameter(self, model_id: str, param_name: str, value: float) -> None:
        """Update a parameter value.

        Parameters
        ----------
        model_id : str
            Model identifier
        param_name : str
            Parameter name
        value : float
            New value
        """
        self._parameter_table.set_parameter_value(param_name, value)

    def compare_models(self, model_ids: list[str]) -> None:
        """Compare multiple fitted models.

        Parameters
        ----------
        model_ids : list[str]
            List of model IDs to compare
        """
        if not model_ids:
            QMessageBox.warning(self, "No Models", "No models selected for comparison.")
            return

        # Get fit results from state
        state = self._store.get_state()
        results: list[tuple[str, Any]] = []

        for model_id in model_ids:
            if model_id in state.fit_results:
                results.append((model_id, state.fit_results[model_id]))

        if len(results) < 2:
            QMessageBox.warning(
                self,
                "Insufficient Models",
                "At least 2 fitted models are required for comparison.",
            )
            return

        # Build comparison summary
        comparison_text = "Model Comparison:\n" + "=" * 40 + "\n\n"

        for _model_id, result in results:
            comparison_text += f"Model: {result.model_name}\n"
            comparison_text += f"  R²: {result.r_squared:.6f}\n"
            comparison_text += f"  MPE: {result.mpe:.4f}%\n"
            comparison_text += f"  χ²: {result.chi_squared:.6f}\n"
            comparison_text += f"  Iterations: {result.num_iterations}\n"
            comparison_text += f"  Fit Time: {result.fit_time:.3f}s\n\n"

        # Find best model by R²
        best_model = max(results, key=lambda x: x[1].r_squared)
        comparison_text += "-" * 40 + "\n"
        comparison_text += f"Best Model (by R²): {best_model[1].model_name}\n"

        QMessageBox.information(self, "Model Comparison", comparison_text)

    def show_residuals(self, model_id: str) -> None:
        """Show residuals for a fitted model.

        Parameters
        ----------
        model_id : str
            Model identifier
        """
        state = self._store.get_state()
        result = state.fit_results.get(model_id)
        if result is None or getattr(result, "residuals", None) is None:
            QMessageBox.information(
                self,
                "No Residuals",
                "Residuals not available for the selected model.",
            )
            return

        x_vals = getattr(result, "x_fit", None)
        residuals = getattr(result, "residuals", None)
        y_pred = getattr(result, "y_fit", None)

        if x_vals is not None and residuals is not None:
            self.plot_residuals(x_vals, residuals, y_pred)

    # External hooks from MainWindow
    def set_plot_figure(self, fig: Figure) -> None:
        """Replace the plot canvas figure."""
        # Swap the FigureCanvas content
        self._plot_canvas.figure = fig
        self._plot_canvas.canvas.figure = fig
        self._plot_canvas.axes = fig.gca()
        self._plot_canvas.canvas.draw_idle()
        if hasattr(self, "_plot_placeholder"):
            self._plot_placeholder.hide()

    def plot_residuals(self, x: np.ndarray, residuals: np.ndarray, y_pred: np.ndarray | None = None) -> None:
        """Forward residuals to the residuals panel."""
        if y_pred is not None:
            self._residuals_panel.plot_residuals(y_true=y_pred + residuals, y_pred=y_pred, x=x)
        else:
            self._residuals_panel.set_residuals(residuals)
