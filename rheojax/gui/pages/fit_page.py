"""Fit Page.

Fit model controls + visualization (fit plot + residuals).
"""

from dataclasses import replace
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.services.model_service import ModelService, normalize_model_name
from rheojax.gui.state.actions import (
    set_active_model,
    toggle_parameter_fixed,
    update_parameter,
    update_parameter_bounds,
)
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.parameter_table import ParameterTable
from rheojax.gui.widgets.plot_canvas import PlotCanvas
from rheojax.gui.widgets.residuals_panel import ResidualsPanel
from rheojax.logging import get_logger

logger = get_logger(__name__)


class FitPage(QWidget):
    """Model fitting page with integrated model browser and service calls.

    Features:
        - Model browser for model selection
        - Parameter table with editable values
        - Plot canvas showing data and fit
        - Residuals panel for fit diagnostics
        - Background fitting with progress updates
    """

    # Emitted to request a fit from the main window.
    # Payload keys: model_name, dataset_id, options, initial_params
    fit_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize fit page.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self._store = StateStore()
        self._model_service = ModelService()
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
        self._current_model: str | None = None
        self._is_compatible: bool = False
        self._params_model_name: str | None = None
        self._empty_params_default_text = (
            "No model loaded. Select a model to edit parameters."
        )

        self._setup_ui()
        self._connect_signals()
        self._load_models()
        logger.debug(
            "Initialization complete",
            class_name=self.__class__.__name__,
            page="FitPage",
        )

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)

        # Left: Fit model controls (yellow rectangle in reference screenshot)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Center: Visualization (fit plot + residuals)
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, 3)

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

    def _create_left_panel(self) -> QWidget:
        """Create left panel with fit controls and parameters."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Match the reference screenshot: Context -> Compatibility -> buttons -> Fit Results
        context_group = QGroupBox("Context")
        context_layout = QVBoxLayout(context_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["oscillation", "relaxation", "creep", "rotation"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo, 1)
        context_layout.addLayout(mode_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._quick_model_combo = QComboBox()
        self._quick_model_combo.setEditable(True)
        self._quick_model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._quick_model_combo.currentIndexChanged.connect(
            self._on_quick_model_changed
        )
        if self._quick_model_combo.lineEdit() is not None:
            self._quick_model_combo.lineEdit().editingFinished.connect(
                self._on_quick_model_edited
            )
        model_row.addWidget(self._quick_model_combo, 1)
        context_layout.addLayout(model_row)
        layout.addWidget(context_group)

        compat_group = QGroupBox("Compatibility")
        compat_layout = QVBoxLayout(compat_group)
        self._compat_label = QLabel("Select a model and dataset")
        self._compat_label.setWordWrap(True)
        self._compat_label.setStyleSheet("color: gray;")
        compat_layout.addWidget(self._compat_label)
        layout.addWidget(compat_group)

        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        self._parameter_table = ParameterTable()
        self._parameter_table.setEnabled(False)
        self._parameter_table.parameter_changed.connect(
            self._on_parameter_value_changed
        )
        self._parameter_table.bounds_changed.connect(self._on_parameter_bounds_changed)
        self._parameter_table.fixed_toggled.connect(self._on_parameter_fixed_toggled)
        params_layout.addWidget(self._parameter_table)

        self._empty_params = QLabel(self._empty_params_default_text)
        self._empty_params.setAlignment(Qt.AlignCenter)
        self._empty_params.setStyleSheet("color: #666; padding: 6px;")
        params_layout.addWidget(self._empty_params)
        layout.addWidget(params_group, 2)

        btn_layout = QHBoxLayout()
        self._btn_options = QPushButton("Options...")
        self._btn_options.setProperty("variant", "secondary")
        self._btn_options.clicked.connect(self._show_fit_options)
        btn_layout.addWidget(self._btn_options)

        self._btn_fit = QPushButton("Fit Model")
        self._btn_fit.setProperty("variant", "primary")
        self._btn_fit.clicked.connect(self._on_fit_clicked)
        self._btn_fit.setEnabled(False)
        self._btn_fit.setToolTip("Fit the selected model to the active dataset")
        btn_layout.addWidget(self._btn_fit, 2)
        layout.addLayout(btn_layout)

        results_group = QGroupBox("Fit Results")
        results_layout = QVBoxLayout(results_group)
        self._status_text = QTextEdit()
        self._status_text.setReadOnly(True)
        self._status_text.setMaximumHeight(180)
        results_layout.addWidget(self._status_text)

        self._empty_results = QLabel("No fit results yet.")
        self._empty_results.setAlignment(Qt.AlignCenter)
        self._empty_results.setStyleSheet("color: #94A3B8; padding: 8px;")
        results_layout.addWidget(self._empty_results)
        layout.addWidget(results_group)

        layout.addStretch(1)

        return panel

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Store signals (only if signals are set)
        if self._store.signals is not None:
            self._store.signals.dataset_selected.connect(self._on_dataset_changed)
            self._store.signals.model_selected.connect(self._on_external_model_selected)
            self._store.signals.model_params_changed.connect(
                self._on_model_params_changed
            )
            self._store.signals.fit_started.connect(self._on_fitting_started)
            self._store.signals.fit_completed.connect(self._on_fitting_completed)
            self._store.signals.fit_failed.connect(self._on_fitting_failed)

    def _load_models(self) -> None:
        """Load available models into browser."""
        logger.debug("Loading available models", page="FitPage")
        models = self._model_service.get_available_models()
        self._populate_quick_model_selector(models)
        logger.debug(
            "Models loaded",
            page="FitPage",
            category_count=len(models),
        )

    def _populate_quick_model_selector(self, models: dict[str, list[str]]) -> None:
        """Populate quick model selector with available models."""
        self._quick_model_combo.blockSignals(True)
        self._quick_model_combo.clear()
        self._quick_model_combo.addItem("Select model...", None)
        for category in models.values():
            for model_name in category:
                self._quick_model_combo.addItem(model_name, model_name)
        self._quick_model_combo.blockSignals(False)

    def _apply_model_selection(self, model_name: str, dispatch: bool) -> None:
        """Apply model selection and refresh compatibility/controls."""
        model_name = normalize_model_name(model_name)
        if not model_name:
            return

        if not dispatch and model_name == self._current_model:
            return

        logger.debug(
            "Model selection changed",
            model=model_name,
            page="FitPage",
            dispatch=dispatch,
        )

        if dispatch:
            self._store.dispatch(set_active_model(model_name))

        self._current_model = model_name

        # Load/initialize model parameters for the Parameters section
        self._load_parameters_for_model(model_name)

        # Check compatibility with active dataset
        self._check_compatibility(model_name)

        # Enable fit button if we have data
        self._update_fit_enabled()

        self._sync_quick_model_selection(model_name)

    def _load_parameters_for_model(self, model_name: str) -> None:
        """Initialize the parameter table from state or model defaults."""
        model_name = normalize_model_name(model_name)
        if not model_name:
            return

        state = self._store.get_state()
        current = state.model_params
        self._empty_params.setText(self._empty_params_default_text)

        # If we already have parameters for the current model, keep them.
        if self._params_model_name == model_name and current:
            self._parameter_table.set_parameters(current)
            self._parameter_table.setEnabled(True)
            self._empty_params.hide()
            return

        try:
            defaults = self._model_service.get_parameter_defaults(model_name)
        except Exception:
            logger.error(
                "Failed to get parameter defaults",
                model=model_name,
                page="FitPage",
                exc_info=True,
            )
            defaults = {}
            self._empty_params.setText(
                f"Model '{model_name}' is unavailable. Select a different model."
            )
        if not defaults:
            self._parameter_table.setRowCount(0)
            self._parameter_table.setEnabled(False)
            self._empty_params.show()
            self._params_model_name = None
            self._is_compatible = False
            self._update_fit_enabled()
            return

        # If state params already look compatible, prefer them (e.g., switching tabs).
        params_to_show = defaults
        if (
            state.active_model_name == model_name
            and current
            and set(current.keys()) == set(defaults.keys())
        ):
            params_to_show = current

        # Ensure state has params for fitting (used by FitPage and MainWindow).
        if not current or set(current.keys()) != set(defaults.keys()):

            def updater(s):
                return replace(
                    s, model_params={k: v.clone() for k, v in defaults.items()}
                )

            self._store.update_state(updater, track_undo=False, emit_signal=True)

        self._parameter_table.set_parameters(params_to_show)
        self._parameter_table.setEnabled(True)
        self._empty_params.hide()
        self._params_model_name = model_name

    @Slot(str)
    def _on_model_params_changed(self, model_name: str) -> None:
        """Refresh the parameter table when state parameters change."""
        model_name = normalize_model_name(model_name)
        if not model_name or model_name != (self._current_model or ""):
            return
        if not hasattr(self, "_parameter_table"):
            return

        # Avoid resetting the editor while the user is typing.
        try:
            if self._parameter_table.state() == QAbstractItemView.State.EditingState:
                return
        except Exception:
            pass

        params = self._store.get_state().model_params
        if not params:
            self._parameter_table.setRowCount(0)
            self._parameter_table.setEnabled(False)
            self._empty_params.show()
            return

        self._parameter_table.set_parameters(params)
        self._parameter_table.setEnabled(True)
        self._empty_params.hide()
        self._params_model_name = model_name

    def _on_parameter_value_changed(self, param_name: str, value: float) -> None:
        """Update state when the user edits a parameter value."""
        logger.debug(
            "Parameter value changed",
            param_name=param_name,
            value=value,
            page="FitPage",
        )
        update_parameter(param_name, float(value))

    def _on_parameter_bounds_changed(
        self, param_name: str, min_val: float, max_val: float
    ) -> None:
        """Update state when the user edits parameter bounds."""
        logger.debug(
            "Parameter bounds changed",
            param_name=param_name,
            min_val=min_val,
            max_val=max_val,
            page="FitPage",
        )
        update_parameter_bounds(param_name, float(min_val), float(max_val))

    def _on_parameter_fixed_toggled(self, param_name: str, is_fixed: bool) -> None:
        """Update state when the user toggles parameter fixed state."""
        logger.debug(
            "Parameter fixed toggled",
            param_name=param_name,
            is_fixed=is_fixed,
            page="FitPage",
        )
        toggle_parameter_fixed(param_name, bool(is_fixed))

    @Slot(str)
    def _on_model_selected(self, model_name: str) -> None:
        """Handle model selection from within the fit page."""
        self._apply_model_selection(normalize_model_name(model_name), dispatch=True)

    @Slot(str)
    def _on_model_double_clicked(self, model_name: str) -> None:
        """Handle model double-click (quick fit).

        Parameters
        ----------
        model_name : str
            Double-clicked model name
        """
        logger.debug(
            "Model double-clicked",
            model=model_name,
            page="FitPage",
        )
        self._apply_model_selection(normalize_model_name(model_name), dispatch=True)
        if self._btn_fit.isEnabled():
            self._on_fit_clicked()

    @Slot(str)
    def _on_external_model_selected(self, model_name: str) -> None:
        """Apply model selection coming from outside the Fit page."""
        self._apply_model_selection(normalize_model_name(model_name), dispatch=False)

    def _sync_quick_model_selection(self, model_name: str) -> None:
        """Synchronize the quick model combo without emitting signals."""
        idx = self._quick_model_combo.findData(model_name)
        if idx >= 0:
            was_blocked = self._quick_model_combo.blockSignals(True)
            self._quick_model_combo.setCurrentIndex(idx)
            self._quick_model_combo.blockSignals(was_blocked)

    @Slot(str)
    def _on_dataset_changed(self, _dataset_id: str = "") -> None:
        """Handle active dataset change."""
        logger.debug(
            "Dataset changed",
            dataset_id=_dataset_id,
            page="FitPage",
        )
        dataset = self._store.get_active_dataset()
        model_name = self._current_model

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
        else:
            self._compat_label.setText("Select a dataset to enable fitting")
            self._compat_label.setStyleSheet("color: gray;")

        self._update_fit_enabled()

    def _on_mode_changed(self, mode: str) -> None:
        """Update store test mode from selector."""
        logger.debug(
            "Mode changed",
            mode=mode,
            page="FitPage",
        )
        if mode:
            self._store.dispatch("SET_TEST_MODE", {"test_mode": mode})
            self._compat_label.setText(f"Mode set to {mode}")
            self._compat_label.setStyleSheet("color: #555;")

    def _on_quick_model_changed(self, index: int) -> None:
        """Handle quick model combo changes."""
        model_name = self._quick_model_combo.itemData(index)
        if model_name:
            self._apply_model_selection(model_name, dispatch=True)

    def _on_quick_model_edited(self) -> None:
        """Handle aliases typed into the quick model combo."""
        text = self._quick_model_combo.currentText().strip()
        if not text:
            return

        normalized = normalize_model_name(text)
        current_index = self._quick_model_combo.currentIndex()
        current_slug = self._quick_model_combo.itemData(current_index) or ""

        match_index = -1
        for i in range(self._quick_model_combo.count()):
            data = self._quick_model_combo.itemData(i)
            if data == normalized:
                match_index = i
                break
            label = (self._quick_model_combo.itemText(i) or "").strip()
            if label.lower() == text.lower():
                match_index = i
                normalized = data or normalized
                break

        if match_index >= 0:
            if (
                normalized
                and normalized == current_slug
                and match_index == current_index
            ):
                return
            was_blocked = self._quick_model_combo.blockSignals(True)
            self._quick_model_combo.setCurrentIndex(match_index)
            self._quick_model_combo.blockSignals(was_blocked)
            self._apply_model_selection(normalized, dispatch=True)
        else:
            if normalized == current_slug:
                return
            self._apply_model_selection(normalized, dispatch=True)

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
            self._is_compatible = False
            return

        logger.debug(
            "Checking compatibility",
            model=model_name,
            page="FitPage",
        )
        rheo_data = self._to_rheodata(dataset)
        result = self._model_service.check_compatibility(
            model_name, rheo_data, rheo_data.metadata.get("test_mode")
        )

        if result.get("compatible", True):
            text = f"Compatible\nDecay: {result.get('decay_type', 'unknown')}"
            self._compat_label.setText(text)
            self._compat_label.setStyleSheet("color: green;")
            self._is_compatible = True
            logger.debug(
                "Compatibility check passed",
                model=model_name,
                decay_type=result.get("decay_type"),
                page="FitPage",
            )
        else:
            warnings = result.get("warnings", [])
            text = "Compatibility issues:\n" + "\n".join(f"- {w}" for w in warnings[:3])
            self._compat_label.setText(text)
            self._compat_label.setStyleSheet("color: orange;")
            self._is_compatible = False
            logger.debug(
                "Compatibility check failed",
                model=model_name,
                warnings=warnings,
                page="FitPage",
            )

        self._update_fit_enabled()

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
            title = dataset.file_path.name
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

    def _on_fit_clicked(self) -> None:
        """Handle fit button click."""
        logger.debug("Button clicked", button_id="fit_button", page="FitPage")
        model_name = self._quick_model_combo.currentData()
        dataset = self._store.get_active_dataset()

        if not model_name or not dataset:
            logger.debug(
                "Fit aborted - missing model or dataset",
                model=model_name,
                has_dataset=dataset is not None,
                page="FitPage",
            )
            return

        logger.debug(
            "Fit triggered",
            model=model_name,
            dataset_id=str(dataset.id),
            page="FitPage",
        )
        initial_params = self._get_initial_params_for_fit(model_name)
        payload = {
            "model_name": str(model_name),
            "dataset_id": str(dataset.id),
            "options": dict(self._fit_options),
            "initial_params": initial_params,
        }
        logger.info(
            "Requesting fit",
            model=model_name,
            dataset_id=str(dataset.id),
            algorithm=self._fit_options.get("algorithm"),
            page="FitPage",
        )
        self._status_text.setText("Starting fit...")
        if hasattr(self, "_empty_results"):
            self._empty_results.hide()
        self.fit_requested.emit(payload)

    def _get_initial_params_for_fit(self, model_name: str) -> dict[str, float] | None:
        """Return numeric initial params (state values if present else defaults)."""
        state_params = self._store.get_state().model_params
        if state_params:
            try:
                return {
                    name: float(param.value) for name, param in state_params.items()
                }
            except Exception:
                logger.error(
                    "Failed to extract initial params from state",
                    model=model_name,
                    page="FitPage",
                    exc_info=True,
                )
                return None

        try:
            defaults = self._model_service.get_parameter_defaults(model_name)
        except Exception:
            logger.error(
                "Failed to get parameter defaults for initial params",
                model=model_name,
                page="FitPage",
                exc_info=True,
            )
            return None
        if not defaults:
            return None
        return {name: float(param.value) for name, param in defaults.items()}

    @Slot(str, str)
    def _on_fitting_started(self, _model_name: str = "", _dataset_id: str = "") -> None:
        """Handle fitting started signal."""
        logger.debug(
            "Fitting started",
            model=_model_name,
            dataset_id=_dataset_id,
            page="FitPage",
        )
        self._btn_fit.setEnabled(False)
        self._btn_fit.setText("Fitting...")

    @Slot(str, str)
    def _on_fitting_completed(
        self, _model_name: str = "", _dataset_id: str = ""
    ) -> None:
        """Handle fitting completed signal."""
        logger.info(
            "Fitting completed",
            model=_model_name,
            dataset_id=_dataset_id,
            page="FitPage",
        )
        self._btn_fit.setText("Fit Model")
        self._update_fit_enabled()

    @Slot(str, str, str)
    def _on_fitting_failed(
        self, _model_name: str, _dataset_id: str, error: str
    ) -> None:
        """Handle fitting failure signal."""
        logger.error(
            "Fitting failed",
            model=_model_name,
            dataset_id=_dataset_id,
            error=error,
            page="FitPage",
        )
        self._btn_fit.setText("Fit Model")
        self._update_fit_enabled()
        if error:
            self._status_text.setText(f"Fit failed:\n{error}")
            if hasattr(self, "_empty_results"):
                self._empty_results.hide()

    def _show_fit_options(self) -> None:
        """Show fitting options dialog."""
        logger.debug("Button clicked", button_id="options_button", page="FitPage")
        from rheojax.gui.dialogs.fitting_options import FittingOptionsDialog

        dialog = FittingOptionsDialog(current_options=self._fit_options, parent=self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self._fit_options = dialog.get_options()
            logger.debug(
                "Fit options updated",
                algorithm=self._fit_options.get("algorithm"),
                max_iter=self._fit_options.get("max_iter"),
                page="FitPage",
            )

    def apply_fit_result(self, fit_result: Any) -> None:
        """Update the Fit page status from a stored FitResult."""
        try:
            r2 = getattr(fit_result, "r_squared", None)
            mpe = getattr(fit_result, "mpe", None)
            chi2 = getattr(fit_result, "chi_squared", None)
            fit_time = getattr(fit_result, "fit_time", None)
            params = getattr(fit_result, "parameters", {}) or {}
        except Exception:
            logger.error(
                "Failed to extract fit result attributes",
                page="FitPage",
                exc_info=True,
            )
            return

        logger.debug(
            "Applying fit result",
            r_squared=r2,
            mpe=mpe,
            chi_squared=chi2,
            fit_time=fit_time,
            page="FitPage",
        )

        lines: list[str] = ["Fit successful!", ""]
        if r2 is not None:
            lines.append(f"R²: {float(r2):.4f}")
        if chi2 is not None:
            lines.append(f"Chi-squared: {float(chi2):.6g}")
        if mpe is not None:
            lines.append(f"MPE: {float(mpe):.2f}%")
        if fit_time is not None:
            lines.append(f"Time: {float(fit_time):.2f}s")

        if params:
            lines.append("")
            lines.append("Parameters:")
            for name, value in sorted(params.items()):
                try:
                    lines.append(f"  {name}: {float(value):.6g}")
                except Exception:
                    lines.append(f"  {name}: {value}")

            # Also reflect fitted params in the Parameters table (preserve bounds when possible)
            try:
                state_params = self._store.get_state().model_params
                if state_params:
                    updated = {k: v.clone() for k, v in state_params.items()}
                    for name, value in params.items():
                        if name in updated:
                            updated[name] = replace(updated[name], value=float(value))
                    self._parameter_table.set_parameters(updated)

                    def updater(s):
                        return replace(s, model_params=updated)

                    self._store.update_state(
                        updater, track_undo=False, emit_signal=True
                    )
                    self._empty_params.hide()
                    self._parameter_table.setEnabled(True)
                    self._params_model_name = normalize_model_name(
                        getattr(fit_result, "model_name", self._current_model or "")
                    )
            except Exception:
                logger.error(
                    "Failed to update parameter table from fit result",
                    page="FitPage",
                    exc_info=True,
                )

        self._status_text.setText("\n".join(lines))
        if hasattr(self, "_empty_results"):
            self._empty_results.hide()

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

    def plot_residuals(
        self, x: np.ndarray, residuals: np.ndarray, y_pred: np.ndarray | None = None
    ) -> None:
        """Forward residuals to the residuals panel."""
        if y_pred is not None:
            self._residuals_panel.plot_residuals(
                y_true=y_pred + residuals, y_pred=y_pred, x=x
            )
        else:
            self._residuals_panel.set_residuals(residuals)

    def _update_fit_enabled(self) -> None:
        dataset = self._store.get_active_dataset()
        model_name = self._quick_model_combo.currentData()
        compatible = getattr(self, "_is_compatible", False)
        enabled = dataset is not None and bool(model_name) and compatible
        self._btn_fit.setEnabled(enabled)
