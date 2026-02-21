"""
Bayesian Page
============

Bayesian inference interface with prior specification and MCMC monitoring.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.gui.compat import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    Signal,
    Slot,
)
from rheojax.gui.jobs.bayesian_worker import BayesianWorker
from rheojax.gui.jobs.worker_pool import WorkerPool
from rheojax.gui.resources.styles.tokens import (
    ColorPalette,
    Spacing,
    Typography,
    button_style,
)
from rheojax.gui.services.bayesian_service import BayesianService
from rheojax.gui.services.data_service import DataService
from rheojax.gui.services.model_service import ModelService
from rheojax.gui.services.plot_service import PlotService
from rheojax.gui.state.actions import (
    bayesian_failed,
    start_bayesian,
    store_bayesian_result,
    update_bayesian_progress,
)
from rheojax.gui.state.store import BayesianResult, StateStore
from rheojax.gui.utils.rheodata import rheodata_from_dataset_state
from rheojax.gui.widgets.plot_canvas import PlotCanvas
from rheojax.logging import get_logger

logger = get_logger(__name__)


class BayesianPage(QWidget):
    """Bayesian inference page with integrated ArviZ diagnostics.

    Features:
        - MCMC configuration (sampler, warmup, samples, chains)
        - Priors editor for parameter priors
        - Real-time progress monitoring per chain
        - Raw + fitted data plot (posterior representative)
        - Convergence diagnostics (R-hat, ESS)
        - Credible interval display
    """

    run_requested = Signal(str, str, dict)  # model_name, dataset_id, config
    run_completed = Signal(object)  # BayesianResult
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize Bayesian page.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self._store = StateStore()
        self._bayesian_service = BayesianService()
        self._model_service = ModelService()
        self._worker_pool = WorkerPool()
        self._current_worker: BayesianWorker | None = None
        self._is_running = False
        self._current_preset: str = "custom"
        self._preset_priors: dict[str, dict[str, Any]] | None = None
        self._preset_dataset_path: str | None = None
        self._data_service = DataService()

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(
            Spacing.PAGE_MARGIN,
            Spacing.PAGE_MARGIN,
            Spacing.PAGE_MARGIN,
            Spacing.PAGE_MARGIN,
        )
        main_layout.setSpacing(Spacing.LG)

        # Left: Configuration (25%) — scrollable for many controls
        left_panel = self._create_config_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(left_scroll, 1)

        # Center: Progress + Fit Plot (50%)
        center_splitter = QSplitter(Qt.Orientation.Vertical)

        progress_panel = self._create_progress_panel()
        center_splitter.addWidget(progress_panel)

        center_splitter.addWidget(self._create_fit_plot_panel())

        center_splitter.setSizes([200, 400])
        main_layout.addWidget(center_splitter, 2)

        # Right: Results (25%)
        right_panel = self._create_results_panel()
        main_layout.addWidget(right_panel, 1)

    def _create_config_panel(self) -> QWidget:
        """Create configuration panel."""
        panel = QGroupBox("Configuration")
        layout = QVBoxLayout(panel)

        # Model info
        self._model_label = QLabel("Model: (select in Fit tab)")
        self._model_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._model_label)

        # Sampler selection
        sampler_label = QLabel("Sampler:")
        layout.addWidget(sampler_label)
        self._sampler_combo = QComboBox()
        self._sampler_combo.addItems(["NUTS (No U-Turn Sampler)"])
        self._sampler_combo.setToolTip("Select the MCMC sampler (NUTS recommended)")
        layout.addWidget(self._sampler_combo)

        # Warmup samples
        warm_label = QLabel("Warmup Samples:")
        layout.addWidget(warm_label)
        self._warmup_spin = QSpinBox()
        self._warmup_spin.setRange(100, 10000)
        self._warmup_spin.setValue(1000)
        self._warmup_spin.setSingleStep(100)
        self._warmup_spin.setToolTip("Number of burn-in iterations before sampling")
        layout.addWidget(self._warmup_spin)

        # Posterior samples
        samples_label = QLabel("Posterior Samples:")
        layout.addWidget(samples_label)
        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(100, 10000)
        self._samples_spin.setValue(2000)
        self._samples_spin.setSingleStep(100)
        self._samples_spin.setToolTip("Number of draws per chain after warmup")
        layout.addWidget(self._samples_spin)

        # Number of chains
        chains_label = QLabel("Number of Chains:")
        layout.addWidget(chains_label)
        self._chains_spin = QSpinBox()
        self._chains_spin.setRange(1, 8)
        self._chains_spin.setValue(4)
        self._chains_spin.setToolTip("Parallel chains improve convergence diagnostics")
        layout.addWidget(self._chains_spin)

        # HDI probability
        hdi_label = QLabel("HDI Probability:")
        layout.addWidget(hdi_label)
        self._hdi_combo = QComboBox()
        self._hdi_combo.addItems(["0.90", "0.94", "0.95", "0.99"])
        self._hdi_combo.setCurrentText("0.94")
        self._hdi_combo.setToolTip("Credible interval probability for summaries")
        layout.addWidget(self._hdi_combo)

        # Presets from example notebooks
        preset_label = QLabel("Sampler Preset:")
        layout.addWidget(preset_label)
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(
            [
                "Custom",
                "Bayesian Demo (chains=4, 1000/2000)",
                "GMM Quick (chains=1, 500/1000)",
                "SPP LAOS (chains=4, 1000/2000)",
                "SPP Dense (chains=4, 2000/2000)",
            ]
        )
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        self._preset_combo.setToolTip(
            "Quickly apply sampler settings and suggested priors"
        )
        layout.addWidget(self._preset_combo)

        # Warm start
        self._warmstart_check = QCheckBox("Use NLSQ warm-start (recommended)")
        self._warmstart_check.setChecked(True)
        self._warmstart_check.setToolTip(
            "Initialize MCMC from NLSQ point estimates for faster convergence"
        )
        layout.addWidget(self._warmstart_check)

        # Deformation mode (DMTA / DMA support)
        deform_label = QLabel("Deformation Mode:")
        layout.addWidget(deform_label)
        self._deformation_combo = QComboBox()
        self._deformation_combo.addItems(["Shear", "Tension", "Bending", "Compression"])
        self._deformation_combo.setToolTip(
            "Select deformation mode (Tension for DMTA/DMA data)"
        )
        self._deformation_combo.currentTextChanged.connect(
            self._on_deformation_mode_changed
        )
        layout.addWidget(self._deformation_combo)

        poisson_layout = QHBoxLayout()
        poisson_layout.addWidget(QLabel("Poisson:"))
        from rheojax.gui.compat import QDoubleSpinBox

        self._poisson_spin = QDoubleSpinBox()
        self._poisson_spin.setRange(0.0, 0.5)
        self._poisson_spin.setValue(0.5)
        self._poisson_spin.setSingleStep(0.05)
        self._poisson_spin.setDecimals(3)
        self._poisson_spin.setToolTip(
            "Poisson ratio for E*-G* conversion (rubber=0.5, glassy=0.35)"
        )
        self._poisson_spin.valueChanged.connect(self._on_poisson_ratio_changed)
        poisson_layout.addWidget(self._poisson_spin)
        layout.addLayout(poisson_layout)

        # Priors editor button
        btn_priors = QPushButton("Edit Priors...")
        btn_priors.setToolTip("View or edit prior distributions used for this run")
        btn_priors.clicked.connect(self._edit_priors)
        layout.addWidget(btn_priors)

        layout.addStretch()

        # Run button
        self._btn_run = QPushButton("Run Bayesian Inference")
        self._btn_run.setStyleSheet(button_style("accent", "lg"))
        self._btn_run.clicked.connect(self._on_run_clicked)
        layout.addWidget(self._btn_run)

        # Cancel button
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(self._btn_cancel)

        return panel

    def _create_progress_panel(self) -> QWidget:
        """Create progress panel."""
        panel = QGroupBox("Progress")
        layout = QVBoxLayout(panel)

        # Overall progress
        layout.addWidget(QLabel("Overall Progress:"))
        self._overall_progress = QProgressBar()
        self._overall_progress.setTextVisible(True)
        layout.addWidget(self._overall_progress)

        # Status info
        layout.addWidget(QLabel("Status:"))
        self._status_text = QTextEdit()
        self._status_text.setReadOnly(True)
        self._status_text.setMaximumHeight(120)
        layout.addWidget(self._status_text)

        # ETA and divergences
        info_layout = QHBoxLayout()
        self._eta_label = QLabel("ETA: --:--")
        self._divergence_label = QLabel("Divergences: 0")
        self._divergence_label.setStyleSheet(
            f"color: {ColorPalette.ERROR}; font-weight: bold;"
        )
        info_layout.addWidget(self._eta_label)
        info_layout.addStretch()
        info_layout.addWidget(self._divergence_label)
        layout.addLayout(info_layout)

        return panel

    def _create_fit_plot_panel(self) -> QWidget:
        """Create the raw + fitted data plot panel.

        This area should mirror the Fit tab plot: raw data + fitted curve.
        """
        panel = QGroupBox("Raw + Fitted")
        layout = QVBoxLayout(panel)

        self._fit_plot_canvas = PlotCanvas()
        layout.addWidget(self._fit_plot_canvas, 3)

        self._fit_plot_placeholder = QLabel(
            "No Bayesian fit plot yet. Run inference to see raw + fitted data."
        )
        self._fit_plot_placeholder.setAlignment(Qt.AlignCenter)
        self._fit_plot_placeholder.setStyleSheet(
            f"color: {ColorPalette.TEXT_MUTED}; padding: {Spacing.SM}px;"
        )
        layout.addWidget(self._fit_plot_placeholder)

        return panel

    def _create_results_panel(self) -> QWidget:
        """Create results panel."""
        panel = QGroupBox("Results")
        layout = QVBoxLayout(panel)

        # Convergence diagnostics
        layout.addWidget(QLabel("Convergence Diagnostics:"))

        diag_layout = QVBoxLayout()
        self._rhat_label = QLabel("R-hat: --")
        self._ess_label = QLabel("ESS: --")
        warning_label = QLabel("Diagnostics: awaiting run")
        warning_label.setStyleSheet(
            f"color: {ColorPalette.WARNING}; padding: {Spacing.XXS}px;"
        )
        self._diag_warning = warning_label
        # Use design token monospace font
        self._rhat_label.setStyleSheet(f"font-family: {Typography.FONT_FAMILY_MONO};")
        self._ess_label.setStyleSheet(f"font-family: {Typography.FONT_FAMILY_MONO};")
        diag_layout.addWidget(self._rhat_label)
        diag_layout.addWidget(self._ess_label)
        diag_layout.addWidget(warning_label)
        layout.addLayout(diag_layout)

        # Credible intervals table
        layout.addWidget(QLabel("Credible Intervals:"))

        self._intervals_table = QTableWidget()
        self._intervals_table.setColumnCount(4)
        self._intervals_table.setHorizontalHeaderLabels(
            ["Parameter", "Mean", "Lower", "Upper"]
        )
        self._intervals_table.setAlternatingRowColors(True)
        self._intervals_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self._intervals_table)

        # Export button
        btn_export = QPushButton("Export Results...")
        btn_export.clicked.connect(self._export_results)
        layout.addWidget(btn_export)

        layout.addStretch()

        return panel

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Store signals (only if signals are set)
        if self._store.signals is not None:
            self._store.signals.model_selected.connect(self._on_model_changed)
            self._store.signals.bayesian_started.connect(self._on_bayesian_started)
            self._store.signals.bayesian_completed.connect(self._on_bayesian_completed)
            self._store.signals.state_changed.connect(self._sync_deformation_from_store)

    @Slot()
    def _on_model_changed(self) -> None:
        """Handle model change in state."""
        state = self._store.get_state()
        model_name = state.active_model_name

        if model_name:
            self._model_label.setText(f"Model: {model_name}")
            # F-010 fix: Validate DMTA support for selected model
            self._update_deformation_combo(model_name)
        else:
            self._model_label.setText("Model: (select in Fit tab)")

    def _update_deformation_combo(self, model_name: str) -> None:
        """Enable/disable deformation mode combo based on model DMTA support."""
        try:
            from rheojax.gui.services.model_service import ModelService

            svc = ModelService()
            supported = svc.get_supported_deformation_modes(model_name)
        except Exception:
            supported = []
        has_tension = "tension" in supported
        self._deformation_combo.setEnabled(has_tension)
        if not has_tension:
            was_blocked = self._deformation_combo.blockSignals(True)
            self._deformation_combo.setCurrentIndex(0)  # "Shear"
            self._deformation_combo.blockSignals(was_blocked)
            self._deformation_combo.setToolTip(
                f"Model '{model_name}' supports shear deformation only"
            )
        else:
            self._deformation_combo.setToolTip(
                "Select deformation mode (Tension for DMTA/DMA data)"
            )

    @Slot()
    def _sync_deformation_from_store(self) -> None:
        """Sync deformation mode and Poisson ratio from store state."""
        state = self._store.get_state()

        # Sync deformation combo
        mode_text = state.deformation_mode.capitalize()  # "shear" -> "Shear"
        current = self._deformation_combo.currentText()
        if current != mode_text:
            was_blocked = self._deformation_combo.blockSignals(True)
            idx = self._deformation_combo.findText(mode_text)
            if idx >= 0:
                self._deformation_combo.setCurrentIndex(idx)
            self._deformation_combo.blockSignals(was_blocked)

        # Sync Poisson ratio
        if abs(self._poisson_spin.value() - state.poisson_ratio) > 1e-6:
            was_blocked = self._poisson_spin.blockSignals(True)
            self._poisson_spin.setValue(state.poisson_ratio)
            self._poisson_spin.blockSignals(was_blocked)

    def _on_deformation_mode_changed(self, mode: str) -> None:
        """Dispatch deformation mode change to store (bidirectional sync)."""
        if mode:
            self._store.dispatch(
                "SET_DEFORMATION_MODE", {"deformation_mode": mode.lower()}
            )

    def _on_poisson_ratio_changed(self, value: float) -> None:
        """Dispatch Poisson ratio change to store (bidirectional sync)."""
        self._store.dispatch("SET_POISSON_RATIO", {"poisson_ratio": value})

    def _on_run_clicked(self) -> None:
        """Handle run button click."""
        logger.debug("Inference triggered", page="BayesianPage")
        state = self._store.get_state()
        model_name = state.active_model_name
        dataset = self._store.get_active_dataset()

        if not model_name:
            QMessageBox.warning(
                self, "No Model", "Please select a model in the Fit tab first."
            )
            return

        if not dataset:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        # F-HL-019 fix: Warn about HL memory requirements for PDE-based NUTS
        _mn_lower = model_name.lower()
        if "hebraud" in _mn_lower or "hl" == _mn_lower or _mn_lower.startswith("hl_"):
            test_mode_check = dataset.test_mode if hasattr(dataset, "test_mode") else ""
            if test_mode_check in ("creep", "relaxation", "startup", "laos", "oscillation"):
                reply = QMessageBox.question(
                    self,
                    "HL Memory Warning",
                    "The Hébraud-Lequeux model uses a PDE solver (lax.scan) which "
                    "requires forward-mode AD through 500-2000 scan steps per "
                    "NUTS leapfrog step.\n\n"
                    "This can exceed 16 GB RAM with num_chains=4.\n"
                    "Consider using num_chains=1 and fewer warmup samples.\n\n"
                    "Continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.No:
                    return

        # F-006 fix: Warn about EPM memory requirements for large lattices
        if "epm" in _mn_lower:
            try:
                from rheojax.core.registry import ModelRegistry

                model_cls = ModelRegistry.get(model_name)
                if model_cls is not None:
                    tmp = model_cls()
                    lattice_L = getattr(tmp, "L", 64)
                    if lattice_L > 32:
                        reply = QMessageBox.question(
                            self,
                            "EPM Memory Warning",
                            f"EPM model with L={lattice_L} lattice requires significant "
                            f"memory for Bayesian inference (~{lattice_L**2 * 8 // 1024} KB "
                            "per field × forward+backward passes).\n\n"
                            "Consider reducing L to 16-32 for systems with <32 GB RAM.\n\n"
                            "Continue anyway?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes,
                        )
                        if reply == QMessageBox.StandardButton.No:
                            return
            except Exception:
                pass  # Don't block Bayesian if check fails

        # Get configuration
        config = {
            "num_warmup": self._warmup_spin.value(),
            "num_samples": self._samples_spin.value(),
            "num_chains": self._chains_spin.value(),
            "warm_start": self._warmstart_check.isChecked(),
            "hdi_prob": float(self._hdi_combo.currentText()),
            "target_accept_prob": 0.9,
            "max_tree_depth": 10,
        }

        # Preset-specific tweaks
        if getattr(self, "_current_preset", "") == "gmm":
            config["prior_mode"] = "warn"
            config["allow_fallback_priors"] = True
            config["max_tree_depth"] = 8
            config["target_accept_prob"] = 0.9
        elif getattr(self, "_current_preset", "") in ("spp", "spp_dense"):
            config["target_accept_prob"] = 0.99
            config["max_tree_depth"] = 12 if self._current_preset == "spp_dense" else 10

        # Attach preset priors if defined
        if self._preset_priors:
            config["priors"] = self._preset_priors

        # If no dataset loaded, try preset dataset path
        if dataset is None and self._preset_dataset_path:
            try:
                rheo_data = self._data_service.load_file(self._preset_dataset_path)
                dataset_id = str(uuid.uuid4())
                self._store.dispatch(
                    "IMPORT_DATA_SUCCESS",
                    {
                        "dataset_id": dataset_id,
                        "file_path": self._preset_dataset_path,
                        "name": Path(self._preset_dataset_path).stem,
                        "test_mode": rheo_data.metadata.get("test_mode", "oscillation"),
                        "x_data": rheo_data.x,
                        "y_data": rheo_data.y,
                        "y2_data": getattr(rheo_data, "y2", None),
                        "metadata": getattr(rheo_data, "metadata", {}),
                    },
                )
                dataset = self._store.get_dataset(dataset_id)
            except Exception:
                dataset = None

        if dataset is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        # Update state
        self._store.dispatch(start_bayesian(model_name, dataset.id))

        # Handle warm_start: convert bool to dict (get NLSQ params) or None
        warm_start_dict: dict[str, float] | None = None
        if config.get("warm_start", False):
            warm_start_dict = self._select_warm_start_params(model_name, dataset.id)
            if warm_start_dict:
                self._status_text.append(f"Using NLSQ warm-start: {warm_start_dict}")
            else:
                self._status_text.append(
                    "Warning: warm-start requested but no NLSQ fit found for "
                    f"{model_name}/{dataset.id}. Running with default priors."
                )
                logger.warning(
                    "Warm-start requested but no NLSQ result available",
                    model_name=model_name,
                    dataset_id=dataset.id,
                )

        # Capture dataset_id, model_name, and a dataset snapshot at submission
        # time to avoid TOCTOU race if active selection changes before
        # _on_finished runs. The snapshot is used for posterior plotting
        # (GUI-011 fix).
        self._submitted_dataset_id = dataset.id
        self._submitted_model_name = model_name
        self._submitted_dataset_snapshot = dataset.clone()

        # Log inference start
        logger.info(
            "Bayesian inference started",
            model_name=model_name,
            dataset_id=dataset.id,
            num_warmup=config.get("num_warmup", 1000),
            num_samples=config.get("num_samples", 2000),
            num_chains=config.get("num_chains", 4),
            warm_start=warm_start_dict is not None,
            page="BayesianPage",
        )

        # Get DMTA deformation mode and Poisson ratio from UI
        deform_text = self._deformation_combo.currentText().lower()
        poisson_val = self._poisson_spin.value()

        # Snapshot the dataset to prevent TOCTOU mutation if the user
        # changes the active selection while inference is running.
        dataset_snapshot = dataset.clone()

        # F-HL-005 fix: Extract fitted model state from FitResult so
        # stateful models (HL, DMT, ITT-MCT) can restore _last_fit_kwargs,
        # _fit_data_metadata, etc. on the fresh Bayesian model instance.
        fitted_model_state = None
        active_fit = self._store.get_active_fit_result()
        if active_fit and hasattr(active_fit, "metadata") and active_fit.metadata:
            fitted_model_state = active_fit.metadata.get("fitted_model_state")

        self._current_worker = BayesianWorker(
            model_name=model_name,
            data=dataset_snapshot,
            num_warmup=config.get("num_warmup", 1000),
            num_samples=config.get("num_samples", 2000),
            num_chains=config.get("num_chains", 4),
            warm_start=warm_start_dict,
            priors=config.get("priors"),
            seed=config.get("seed", state.current_seed),
            deformation_mode=deform_text if deform_text != "shear" else None,
            poisson_ratio=poisson_val if deform_text != "shear" else None,
            fitted_model_state=fitted_model_state,
        )

        # Connect BayesianWorker-specific signals with QueuedConnection.
        # WorkerPool.submit() separately connects completed/failed for job
        # lifecycle tracking; the connections below are for page-level UI
        # updates and are NOT duplicates.
        self._current_worker.signals.progress.connect(
            self._on_worker_progress, Qt.ConnectionType.QueuedConnection
        )
        self._current_worker.signals.stage_changed.connect(
            self._on_stage_changed, Qt.ConnectionType.QueuedConnection
        )
        self._current_worker.signals.completed.connect(
            self._on_finished, Qt.ConnectionType.QueuedConnection
        )
        self._current_worker.signals.failed.connect(
            self._on_error, Qt.ConnectionType.QueuedConnection
        )
        self._current_worker.signals.divergence_detected.connect(
            self._on_divergence, Qt.ConnectionType.QueuedConnection
        )

        # Use submit() method, not start()
        self._worker_pool.submit(self._current_worker)

        self._is_running = True
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)

        self.run_requested.emit(model_name, dataset.id, config)

    def _select_warm_start_params(
        self, model_name: str, dataset_id: str
    ) -> dict[str, float] | None:
        """Pick warm-start parameters that match the current model/dataset."""
        state = self._store.get_state()
        # Validate that the dataset_id matches the active dataset
        if dataset_id != state.active_dataset_id:
            logger.warning(
                "Warm-start dataset mismatch — skipping",
                requested=dataset_id,
                active=state.active_dataset_id,
            )
            return None
        key = f"{model_name}_{dataset_id}"
        fit_result = state.fit_results.get(key)
        if fit_result and hasattr(fit_result, "parameters"):
            try:
                return {str(k): float(v) for k, v in fit_result.parameters.items()}
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to extract warm-start params", error=str(exc))
                return None
        return None

    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        if self._current_worker:
            self._current_worker.cancel_token.cancel()
            self._status_text.append("Cancelling...")

        self.cancel_requested.emit()

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        """Handle overall progress update (legacy slot for compatibility).

        Parameters
        ----------
        percent : int
            Progress percentage
        message : str
            Status message
        """
        self._overall_progress.setValue(percent)
        self._status_text.append(message)
        self._store.dispatch(update_bayesian_progress(percent))

    @Slot(int, int, str)
    def _on_worker_progress(self, percent: int, total: int, message: str) -> None:
        """Handle progress update from BayesianWorker.

        BayesianWorker emits: progress(percent, total, message)

        Parameters
        ----------
        percent : int
            Current progress value
        total : int
            Total value (usually 100)
        message : str
            Status message
        """
        # Normalize to percentage
        progress_pct = int(percent / max(total, 1) * 100) if total > 0 else percent
        self._overall_progress.setValue(progress_pct)
        self._status_text.append(message)
        self._store.dispatch(update_bayesian_progress(progress_pct))

    @Slot(str)
    def _on_stage_changed(self, stage: str) -> None:
        """Handle MCMC stage change (warmup/sampling).

        Parameters
        ----------
        stage : str
            Current stage ('warmup' or 'sampling')
        """
        self._status_text.append(f"Stage: {stage}")

    @Slot(int)
    def _on_divergence(self, count: int) -> None:
        """Handle divergence detection.

        Parameters
        ----------
        count : int
            Number of divergent transitions detected
        """
        self._divergence_label.setText(f"Divergences: {count}")
        if count > 0:
            self._divergence_label.setStyleSheet(
                f"color: {ColorPalette.ERROR}; font-weight: bold;"
            )
            self._status_text.append(f"WARNING: {count} divergent transitions detected")

    @Slot(object)
    def _on_finished(self, result: Any) -> None:
        """Handle Bayesian inference completion.

        Parameters
        ----------
        result : BayesianResult (from bayesian_worker.py)
            Inference result with posterior_samples, summary, diagnostics, etc.
        """
        self._current_worker = None
        self._is_running = False
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)

        # Use captured values from submission time (TOCTOU-safe)
        dataset_id = getattr(self, "_submitted_dataset_id", None)
        model_name = getattr(result, "model_name", None) or getattr(
            self, "_submitted_model_name", None
        )
        if not dataset_id or not model_name:
            state = self._store.get_state()
            dataset_id = dataset_id or state.active_dataset_id
            model_name = model_name or state.active_model_name

        if model_name and dataset_id:
            try:
                from dataclasses import replace as dc_replace

                stored_result = dc_replace(result, dataset_id=str(dataset_id))
            except TypeError:
                logger.warning(
                    "BayesianResult type mismatch in _on_finished, "
                    "falling back to manual construction",
                    exc_info=True,
                )
                stored_result = BayesianResult(
                    model_name=str(model_name),
                    dataset_id=str(dataset_id),
                    posterior_samples=getattr(result, "posterior_samples", {}),
                    summary=getattr(result, "summary", None),
                    r_hat=getattr(result, "r_hat", {}),
                    ess=getattr(result, "ess", {}),
                    divergences=int(getattr(result, "divergences", 0) or 0),
                    credible_intervals=getattr(result, "credible_intervals", {}),
                    mcmc_time=float(getattr(result, "mcmc_time", 0.0) or 0.0),
                    timestamp=getattr(result, "timestamp", datetime.now()),
                    num_warmup=int(getattr(result, "num_warmup", 0) or 0),
                    num_samples=int(getattr(result, "num_samples", 0) or 0),
                    num_chains=int(getattr(result, "num_chains", 4) or 4),
                    inference_data=getattr(result, "inference_data", None),
                )
            # store_bayesian_result dispatches STORE_BAYESIAN_RESULT, whose
            # reducer already sets pipeline step to COMPLETE.
            store_bayesian_result(stored_result)

        # Update diagnostics display
        self._update_diagnostics(result)

        # Update raw + fitted plot
        try:
            self._update_fit_plot_from_posterior(result)
        except Exception:
            # Plotting is best-effort; keep the Bayesian results UI usable.
            logger.error(
                "Failed to update fit plot from posterior",
                page="BayesianPage",
                exc_info=True,
            )

        # Update credible intervals table
        self._update_intervals_table(result)

        # Log inference completion
        sampling_time = getattr(result, "sampling_time", 0.0)
        logger.info(
            "Bayesian inference completed",
            model_name=model_name,
            dataset_id=dataset_id,
            sampling_time=sampling_time,
            page="BayesianPage",
        )

        self._status_text.append("Bayesian inference completed successfully!")
        self._status_text.append(f"Sampling time: {sampling_time:.2f}s")
        self.run_completed.emit(result)

    @Slot(str)
    def _on_error(self, error_msg: str) -> None:
        """Handle error during inference.

        Parameters
        ----------
        error_msg : str
            Error message
        """
        self._current_worker = None
        self._is_running = False
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)

        self._store.dispatch(bayesian_failed(error_msg))
        self._status_text.append(f"Error: {error_msg}")
        logger.error(
            "Bayesian inference error",
            error_msg=error_msg,
            page="BayesianPage",
            exc_info=True,
        )
        QMessageBox.critical(self, "Inference Error", error_msg)

    @Slot(str, str)
    def _on_bayesian_started(self, model_name: str = "", dataset_id: str = "") -> None:
        """Handle Bayesian started signal from state."""
        self._overall_progress.setValue(0)
        self._status_text.clear()
        self._status_text.append("Starting Bayesian inference...")

    @Slot(str, str)
    def _on_bayesian_completed(self, model_name: str, dataset_id: str) -> None:
        """Handle Bayesian completed signal from state."""
        self._overall_progress.setValue(100)
        self._diag_warning.setText("Diagnostics available below")

    def _update_diagnostics(self, result: BayesianResult) -> None:
        """Update convergence diagnostics display.

        Parameters
        ----------
        result : BayesianResult
            Inference result
        """
        # R-hat
        rhat = result.diagnostics.get("r_hat", {})
        if rhat:
            max_rhat = max(rhat.values()) if rhat else 0
            status = "OK" if max_rhat < 1.01 else "WARNING"
            color = ColorPalette.SUCCESS if max_rhat < 1.01 else ColorPalette.WARNING
            self._rhat_label.setText(f"R-hat (max): {max_rhat:.4f} [{status}]")
            self._rhat_label.setStyleSheet(
                f"color: {color}; font-family: {Typography.FONT_FAMILY_MONO};"
            )
        else:
            self._rhat_label.setText("R-hat: --")

        # ESS
        ess = result.diagnostics.get("ess", {})
        if ess:
            min_ess = min(ess.values()) if ess else 0
            status = "OK" if min_ess > 400 else "LOW"
            color = ColorPalette.SUCCESS if min_ess > 400 else ColorPalette.WARNING
            self._ess_label.setText(f"ESS (min): {min_ess:.0f} [{status}]")
            self._ess_label.setStyleSheet(
                f"color: {color}; font-family: {Typography.FONT_FAMILY_MONO};"
            )
        else:
            self._ess_label.setText("ESS: --")

        # Divergences (-1 = unknown/unavailable)
        divergences = result.diagnostics.get("divergences", 0)
        if divergences == -1:
            divergences = 0  # Treat unknown as zero for display
        if divergences > 0:
            self._divergence_label.setText(f"Divergences: {divergences}")
            self._divergence_label.setStyleSheet(
                f"color: {ColorPalette.ERROR}; font-weight: bold;"
            )
            self._diag_warning.setText(
                "Warning: Divergences detected; consider higher target_accept"
            )
        else:
            self._divergence_label.setText("Divergences: 0")
            self._divergence_label.setStyleSheet(
                f"color: {ColorPalette.SUCCESS}; font-weight: bold;"
            )
            if "OK" in self._rhat_label.text() and "OK" in self._ess_label.text():
                self._diag_warning.setText("Diagnostics look good")

    def _update_intervals_table(self, result: BayesianResult) -> None:
        """Update credible intervals table.

        Parameters
        ----------
        result : BayesianResult
            Inference result
        """
        intervals = result.credible_intervals or {}
        summary = getattr(result, "summary", {}) or {}

        self._intervals_table.setRowCount(len(intervals))

        for row, (param_name, values) in enumerate(intervals.items()):
            # Parameter name
            name_item = QTableWidgetItem(param_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._intervals_table.setItem(row, 0, name_item)

            # Normalize tuple/list intervals to dict form
            if isinstance(values, (tuple, list)):
                if len(values) == 3:
                    mean, lower, upper = values
                elif len(values) == 2:
                    mean = summary.get(param_name, {}).get("mean", 0.0)
                    lower, upper = values
                else:
                    mean = summary.get(param_name, {}).get("mean", 0.0)
                    lower = upper = 0.0
                values = {"mean": mean, "lower": lower, "upper": upper}

            # Mean
            mean = float(
                values.get("mean", summary.get(param_name, {}).get("mean", 0.0))
            )
            mean_item = QTableWidgetItem(f"{mean:.4g}")
            mean_item.setFlags(mean_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._intervals_table.setItem(row, 1, mean_item)

            # Lower bound
            lower = float(values.get("lower", 0.0))
            lower_item = QTableWidgetItem(f"{lower:.4g}")
            lower_item.setFlags(lower_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._intervals_table.setItem(row, 2, lower_item)

            # Upper bound
            upper = float(values.get("upper", 0.0))
            upper_item = QTableWidgetItem(f"{upper:.4g}")
            upper_item.setFlags(upper_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._intervals_table.setItem(row, 3, upper_item)

    def _set_fit_plot_figure(self, fig) -> None:
        """Replace the Fit Plot canvas figure."""
        self._fit_plot_canvas.figure = fig
        self._fit_plot_canvas.canvas.figure = fig
        self._fit_plot_canvas.axes = fig.gca()
        self._fit_plot_canvas.canvas.draw_idle()
        self._fit_plot_placeholder.hide()

        logger.debug(
            "fit_plot_figure_set",
            extra={
                "page": "bayesian",
                "num_axes": len(fig.get_axes()),
            },
        )

    def _posterior_mean_params(
        self, posterior_samples: dict[str, Any]
    ) -> dict[str, float]:
        """Compute a representative parameter set from posterior samples."""
        means: dict[str, float] = {}
        for name, samples in (posterior_samples or {}).items():
            try:
                arr = np.asarray(samples)
                if arr.size == 0:
                    continue
                means[name] = float(np.nanmean(arr.reshape(-1)))
            except Exception:
                continue
        return means

    def _infer_model_kwargs(
        self, model_name: str, param_names: list[str]
    ) -> dict[str, Any]:
        """Infer model initialization kwargs from parameter names.

        For models like GeneralizedMaxwell that require n_modes during init,
        we count the numbered parameters to determine the correct configuration.

        Parameters
        ----------
        model_name : str
            Name of the model
        param_names : list[str]
            List of parameter names from posterior samples

        Returns
        -------
        dict
            Model initialization kwargs (e.g., {"n_modes": 4})
        """
        import re

        model_kwargs: dict[str, Any] = {}

        # Handle GeneralizedMaxwell: count G_i or tau_i parameters
        if "maxwell" in model_name.lower() and "generalized" in model_name.lower():
            # Count G_i parameters (excluding G_inf)
            g_pattern = re.compile(r"^G_(\d+)$")
            g_indices = []
            for name in param_names:
                match = g_pattern.match(name)
                if match:
                    g_indices.append(int(match.group(1)))

            if g_indices:
                n_modes = max(g_indices)
                model_kwargs["n_modes"] = n_modes
                logger.debug(f"Inferred n_modes={n_modes} for {model_name}")

        return model_kwargs

    def _posterior_draw_indices(
        self, posterior_samples: dict[str, Any], max_draws: int
    ) -> np.ndarray:
        """Select draw indices consistently across parameters."""
        lengths: list[int] = []
        for samples in (posterior_samples or {}).values():
            try:
                arr = np.asarray(samples).reshape(-1)
                if arr.size:
                    lengths.append(int(arr.size))
            except Exception:
                continue

        if not lengths:
            return np.array([], dtype=int)

        total = int(min(lengths))
        if total <= 0:
            return np.array([], dtype=int)

        n = int(min(max_draws, total))
        if n <= 1:
            return np.array([0], dtype=int)

        # Use evenly spaced indices for determinism and stability.
        idx = np.linspace(0, total - 1, num=n)
        return np.unique(idx.astype(int))

    def _posterior_params_at_index(
        self, posterior_samples: dict[str, Any], index: int
    ) -> dict[str, float]:
        """Extract a single posterior draw as parameter dict."""
        params: dict[str, float] = {}
        for name, samples in (posterior_samples or {}).items():
            try:
                arr = np.asarray(samples).reshape(-1)
                if arr.size == 0:
                    continue
                if index >= arr.size:
                    continue
                value = float(arr[index])
                if np.isfinite(value):
                    params[name] = value
            except Exception:
                continue
        return params

    def _update_fit_plot_from_posterior(self, result: Any) -> None:
        """Render raw data + posterior-representative fitted curve."""
        # GUI-011 fix: Use the dataset snapshot captured at inference start
        # rather than the current active dataset, to avoid TOCTOU race where
        # the user changes the active dataset while inference is running.
        dataset = getattr(self, "_submitted_dataset_snapshot", None)
        if dataset is None:
            # Fallback to active dataset if no snapshot available
            dataset = self._store.get_active_dataset()
        if dataset is None:
            logger.debug("plot_update skipped: no active dataset")
            return

        model_name = (
            getattr(result, "model_name", None)
            or self._store.get_state().active_model_name
        )
        if not model_name:
            logger.debug("plot_update skipped: no model name")
            return

        rheo_data = rheodata_from_dataset_state(dataset)
        x = np.asarray(rheo_data.x)
        y = np.asarray(rheo_data.y)
        test_mode = rheo_data.metadata.get("test_mode")

        logger.debug(
            "plot_update",
            extra={
                "page": "bayesian",
                "plot_type": "fit",
                "data_shape": x.shape,
                "has_posterior": bool(getattr(result, "posterior_samples", None)),
                "model_name": model_name,
                "test_mode": test_mode,
            },
        )

        posterior_samples = getattr(result, "posterior_samples", None) or {}
        # Posterior predictive band is computed by sampling predictions.
        # Use a small cap for responsiveness in the GUI.
        draw_indices = self._posterior_draw_indices(posterior_samples, max_draws=200)
        if draw_indices.size == 0:
            logger.debug("plot_update skipped: no posterior draw indices")
            return

        # Infer model kwargs from parameter names (e.g., n_modes for GeneralizedMaxwell)
        model_kwargs = self._infer_model_kwargs(
            model_name, list(posterior_samples.keys())
        )

        y_draws: list[np.ndarray] = []
        # F-HL-005 fix: Pass fitted_model_state via model_kwargs for predict()
        active_fit = self._store.get_active_fit_result()
        if active_fit and hasattr(active_fit, "metadata") and active_fit.metadata:
            _fms = active_fit.metadata.get("fitted_model_state")
            if _fms:
                model_kwargs = dict(model_kwargs or {})
                model_kwargs["fitted_model_state"] = _fms

        for idx in draw_indices:
            params = self._posterior_params_at_index(posterior_samples, int(idx))
            if not params:
                continue
            try:
                y_pred = self._model_service.predict(
                    model_name,
                    params,
                    x,
                    test_mode=test_mode,
                    model_kwargs=model_kwargs,
                )
                y_pred_arr = np.asarray(y_pred)
                # F-HL-007 fix: Accept (N, 2) output from oscillation models
                # (HL, GMM, etc.) that return [G', G'']. Convert to complex
                # for the existing plotting pipeline.
                if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == 2 and y_pred_arr.shape[0] == len(x):
                    y_pred_arr = y_pred_arr[:, 0] + 1j * y_pred_arr[:, 1]
                if y_pred_arr.shape == x.shape:
                    y_draws.append(y_pred_arr)
            except Exception:
                continue

        if not y_draws:
            return

        y_stack = np.stack(y_draws, axis=0)
        is_oscillation = (test_mode or "") == "oscillation"
        is_complex = np.iscomplexobj(y_stack)

        # Use the HDI probability selection if possible.
        try:
            hdi_prob = float(self._hdi_combo.currentText())
        except Exception:
            hdi_prob = 0.94
        alpha = (1.0 - float(hdi_prob)) / 2.0
        q_lo = float(alpha)
        q_hi = float(1.0 - alpha)

        if is_oscillation and is_complex:
            y_re = np.real(y_stack)
            y_im = np.imag(y_stack)
            y_center_re = np.nanmean(y_re, axis=0)
            y_center_im = np.nanmean(y_im, axis=0)
            y_lower_re = np.nanquantile(y_re, q_lo, axis=0)
            y_upper_re = np.nanquantile(y_re, q_hi, axis=0)
            y_lower_im = np.nanquantile(y_im, q_lo, axis=0)
            y_upper_im = np.nanquantile(y_im, q_hi, axis=0)
            y_center = y_center_re + 1j * y_center_im
        else:
            # For oscillation magnitude (non-complex) or other modes, compute scalar bands.
            if is_oscillation and np.iscomplexobj(y_stack):
                y_scalar = np.abs(y_stack)
            else:
                y_scalar = y_stack
            y_center = np.nanmean(y_scalar, axis=0)
            y_lower = np.nanquantile(y_scalar, q_lo, axis=0)
            y_upper = np.nanquantile(y_scalar, q_hi, axis=0)

        # Reuse the Fit tab plotting style via PlotService.
        from types import SimpleNamespace

        fit_like = SimpleNamespace(model_name=model_name, y_fit=y_center)

        fig = PlotService().create_fit_plot(
            rheo_data,
            fit_like,
            style="default",
            test_mode=test_mode,
        )

        # Overlay the credible interval band(s) using the same axes.
        try:
            ax = fig.gca()
            x_plot = np.asarray(x)

            positive = np.asarray(y)
            if np.iscomplexobj(positive):
                positive = np.abs(positive)
            positive = positive[np.isfinite(positive) & (positive > 0)]
            eps = float(np.nanmin(positive)) * 1e-3 if positive.size else 1e-12

            pct = int(round(hdi_prob * 100))

            if is_oscillation and is_complex:
                lines = {line.get_label(): line for line in ax.get_lines()}
                gprime_color = (
                    lines.get("G' (fit)").get_color() if "G' (fit)" in lines else None
                )
                gdouble_color = (
                    lines.get('G" (fit)').get_color() if 'G" (fit)' in lines else None
                )

                lower_re = np.asarray(y_lower_re)
                upper_re = np.asarray(y_upper_re)
                # PlotService uses abs(imag) for oscillation; match that convention.
                lower_im = np.abs(np.asarray(y_lower_im))
                upper_im = np.abs(np.asarray(y_upper_im))

                if ax.get_xscale() == "log" or ax.get_yscale() == "log":
                    lower_re = np.maximum(lower_re, eps)
                    upper_re = np.maximum(upper_re, eps)
                    lower_im = np.maximum(lower_im, eps)
                    upper_im = np.maximum(upper_im, eps)

                ax.fill_between(
                    x_plot,
                    lower_re,
                    upper_re,
                    alpha=0.18,
                    color=gprime_color,
                    label=f"G' {pct}% CI",
                )
                ax.fill_between(
                    x_plot,
                    lower_im,
                    upper_im,
                    alpha=0.18,
                    color=gdouble_color,
                    label=f'G" {pct}% CI',
                )
            else:
                line_color = None
                lines = ax.get_lines()
                if lines:
                    line_color = lines[-1].get_color()

                lower_plot = np.asarray(y_lower)
                upper_plot = np.asarray(y_upper)
                if ax.get_xscale() == "log" or ax.get_yscale() == "log":
                    lower_plot = np.maximum(lower_plot, eps)
                    upper_plot = np.maximum(upper_plot, eps)

                ax.fill_between(
                    x_plot,
                    lower_plot,
                    upper_plot,
                    alpha=0.18,
                    color=line_color,
                    label=f"{pct}% CI",
                )

            ax.legend()
        except Exception:
            logger.error(
                "Failed to plot credible intervals",
                page="BayesianPage",
                exc_info=True,
            )

        self._set_fit_plot_figure(fig)

    def _edit_priors(self) -> None:
        """Show priors editor dialog and consume result."""
        from rheojax.gui.dialogs.bayesian_options import BayesianOptionsDialog

        dialog = BayesianOptionsDialog(
            current_options={
                "priors": self._preset_priors,
            },
            parent=self,
        )
        if dialog.exec_() == dialog.DialogCode.Accepted:
            options = dialog.get_options()
            self._preset_priors = options.get("priors", self._preset_priors)
            logger.debug(
                "Priors updated from dialog",
                has_priors=self._preset_priors is not None,
            )

    def _apply_preset(self, name: str) -> None:
        """Apply sampler/prior presets derived from example notebooks."""
        logger.debug(
            "Prior config changed",
            parameter="preset",
            value=name,
            page="BayesianPage",
        )
        self._current_preset = "custom"
        self._preset_priors = None
        self._preset_dataset_path = None
        if name.startswith("Bayesian Demo"):
            self._warmup_spin.setValue(1000)
            self._samples_spin.setValue(2000)
            self._chains_spin.setValue(4)
            self._hdi_combo.setCurrentText("0.94")
            self._warmstart_check.setChecked(True)
            self._current_preset = "demo"
        elif name.startswith("GMM Quick"):
            self._warmup_spin.setValue(500)
            self._samples_spin.setValue(1000)
            self._chains_spin.setValue(1)
            self._hdi_combo.setCurrentText("0.90")
            self._warmstart_check.setChecked(True)
            self._current_preset = "gmm"
            fixture_path = Path("tests/fixtures/bayesian_multi_technique.csv")
            self._preset_dataset_path = (
                str(fixture_path)
                if fixture_path.exists()
                else "examples/data/experimental/multi_technique.txt"
            )
        elif name.startswith("SPP LAOS (chains=4, 1000/2000)"):
            self._warmup_spin.setValue(1000)
            self._samples_spin.setValue(2000)
            self._chains_spin.setValue(4)
            self._hdi_combo.setCurrentText("0.95")
            self._warmstart_check.setChecked(True)
            self._current_preset = "spp"
            fixture_path = Path("tests/fixtures/bayesian_owchirp_tts.csv")
            self._preset_dataset_path = (
                str(fixture_path)
                if fixture_path.exists()
                else "examples/data/experimental/owchirp_tts.txt"
            )
            self._preset_priors = {
                "G_cage": {
                    "dist": "lognormal",
                    "loc": float(np.log(5000)),
                    "scale": 1.0,
                },
                "sigma_y_static": {
                    "dist": "lognormal",
                    "loc": float(np.log(500)),
                    "scale": 1.0,
                },
                "sigma_y_dynamic": {
                    "dist": "lognormal",
                    "loc": float(np.log(500)),
                    "scale": 1.0,
                },
                "sigma_G": {"dist": "halfnormal", "scale": 500.0},
                "sigma_static": {"dist": "halfnormal", "scale": 100.0},
                "sigma_dynamic": {"dist": "halfnormal", "scale": 100.0},
            }
        elif name.startswith("SPP Dense"):
            self._warmup_spin.setValue(2000)
            self._samples_spin.setValue(2000)
            self._chains_spin.setValue(4)
            self._hdi_combo.setCurrentText("0.95")
            self._warmstart_check.setChecked(True)
            self._current_preset = "spp_dense"
            fixture_path = Path("tests/fixtures/bayesian_owchirp_tts.csv")
            self._preset_dataset_path = (
                str(fixture_path)
                if fixture_path.exists()
                else "examples/data/experimental/owchirp_tts.txt"
            )
            self._preset_priors = {
                "G_cage": {
                    "dist": "lognormal",
                    "loc": float(np.log(5000)),
                    "scale": 1.0,
                },
                "sigma_y_static": {
                    "dist": "lognormal",
                    "loc": float(np.log(500)),
                    "scale": 1.0,
                },
                "sigma_y_dynamic": {
                    "dist": "lognormal",
                    "loc": float(np.log(500)),
                    "scale": 1.0,
                },
                "sigma_G": {"dist": "halfnormal", "scale": 500.0},
                "sigma_static": {"dist": "halfnormal", "scale": 100.0},
                "sigma_dynamic": {"dist": "halfnormal", "scale": 100.0},
            }

    def _export_results(self) -> None:
        """Export Bayesian results to JSON or HDF5 format."""
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Bayesian Results",
            "bayesian_results.json",
            "JSON Files (*.json);;HDF5 Files (*.h5)",
        )

        if not filepath:
            return

        # Get current Bayesian result from state
        state = self._store.get_state()
        if not state.bayesian_results:
            QMessageBox.warning(
                self, "No Results", "No Bayesian results available to export."
            )
            return

        # Get most recent result
        result_id = list(state.bayesian_results.keys())[-1]
        result = state.bayesian_results[result_id]

        path = Path(filepath)

        try:
            if path.suffix.lower() == ".json":
                # Export as JSON
                export_data = {
                    "model_name": result.model_name,
                    "dataset_id": getattr(result, "dataset_id", "unknown"),
                    "num_samples": result.num_samples,
                    "num_chains": result.num_chains,
                    "r_hat": result.r_hat,
                    "ess": result.ess,
                    "sampling_time": result.sampling_time,
                    "timestamp": result.timestamp.isoformat(),
                    "credible_intervals": result.credible_intervals,
                    "diagnostics": result.diagnostics,
                }
                with path.open("w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, default=str)

                QMessageBox.information(
                    self, "Export Complete", f"Results exported to: {filepath}"
                )
            else:
                # HDF5 export - requires h5py
                QMessageBox.information(
                    self,
                    "HDF5 Export",
                    f"HDF5 export requires ArviZ InferenceData.\n"
                    f"Results would be exported to: {filepath}",
                )
        except Exception as e:
            logger.error(
                "Failed to export Bayesian results",
                filepath=filepath,
                page="BayesianPage",
                exc_info=True,
            )
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {e}")

    def run_bayesian(
        self,
        model_id: str,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        **kwargs: Any,
    ) -> None:
        """Programmatically run Bayesian inference.

        Parameters
        ----------
        model_id : str
            Model identifier
        num_warmup : int
            Number of warmup samples
        num_samples : int
            Number of posterior samples
        num_chains : int
            Number of MCMC chains
        **kwargs
            Additional options
        """
        self._warmup_spin.setValue(num_warmup)
        self._samples_spin.setValue(num_samples)
        self._chains_spin.setValue(num_chains)

        if "warm_start" in kwargs:
            self._warmstart_check.setChecked(kwargs["warm_start"])

        self._on_run_clicked()

    def update_prior(
        self, model_id: str, param_name: str, prior_spec: dict[str, Any]
    ) -> None:
        """Update prior specification for a parameter.

        Parameters
        ----------
        model_id : str
            Model identifier
        param_name : str
            Parameter name
        prior_spec : dict
            Prior specification (e.g., {"distribution": "normal", "loc": 0, "scale": 1})
        """
        logger.debug(
            "Prior config changed",
            parameter=param_name,
            value=prior_spec,
            page="BayesianPage",
        )
        # Store prior specs in the instance's preset_priors dict.
        # These are forwarded to the BayesianWorker via config["priors"]
        # in _on_run_clicked().
        if self._preset_priors is None:
            self._preset_priors = {}

        self._preset_priors[param_name] = prior_spec

    def show_posterior_summary(self, model_id: str) -> None:
        """Show posterior summary for a model.

        Parameters
        ----------
        model_id : str
            Model identifier
        """
        # Get Bayesian result from state
        state = self._store.get_state()

        # Find result for this model
        target_result = None
        for result in state.bayesian_results.values():
            if result.model_name == model_id or model_id in str(getattr(result, "dataset_id", "")):
                target_result = result
                break

        if target_result is None:
            QMessageBox.warning(
                self,
                "No Results",
                f"No Bayesian results found for model: {model_id}",
            )
            return

        # Build summary text
        summary = f"Posterior Summary for {target_result.model_name}\n"
        summary += "=" * 50 + "\n\n"

        summary += f"Samples: {target_result.num_samples}\n"
        summary += f"Chains: {target_result.num_chains}\n"
        summary += f"Sampling Time: {target_result.sampling_time:.2f}s\n\n"

        summary += "Convergence Diagnostics:\n"
        summary += f"  R-hat: {target_result.r_hat}\n"
        summary += f"  ESS: {target_result.ess}\n\n"

        if target_result.credible_intervals:
            summary += "Credible Intervals (95%):\n"
            for param, interval in target_result.credible_intervals.items():
                if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                    summary += f"  {param}: [{interval[0]:.4f}, {interval[1]:.4f}]\n"

        QMessageBox.information(self, "Posterior Summary", summary)

    def update_sampling_progress(
        self, job_id: str, progress: int, message: str
    ) -> None:
        """Update sampling progress (for external calls).

        Parameters
        ----------
        job_id : str
            Job identifier
        progress : int
            Progress percentage
        message : str
            Status message
        """
        self._overall_progress.setValue(progress)
        self._status_text.append(message)
