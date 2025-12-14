"""
Bayesian Page
============

Bayesian inference interface with prior specification and MCMC monitoring.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid
import numpy as np

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.jobs.bayesian_worker import BayesianWorker
from rheojax.gui.jobs.worker_pool import WorkerPool
from rheojax.gui.services.model_service import ModelService
from rheojax.gui.services.plot_service import PlotService
from rheojax.gui.services.bayesian_service import BayesianResult, BayesianService
from rheojax.gui.state.actions import bayesian_failed, start_bayesian, store_bayesian_result, update_bayesian_progress
from rheojax.gui.state.store import BayesianResult as StoredBayesianResult
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.plot_canvas import PlotCanvas
from rheojax.gui.services.data_service import DataService


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

        # Left: Configuration (25%)
        left_panel = self._create_config_panel()
        main_layout.addWidget(left_panel, 1)

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
        self._chains_spin.valueChanged.connect(self._update_chain_progress_bars)
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
        self._preset_combo.addItems([
            "Custom",
            "Bayesian Demo (chains=4, 1000/2000)",
            "GMM Quick (chains=1, 500/1000)",
            "SPP LAOS (chains=4, 1000/2000)",
            "SPP Dense (chains=4, 2000/2000)",
        ])
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        self._preset_combo.setToolTip("Quickly apply sampler settings and suggested priors")
        layout.addWidget(self._preset_combo)

        # Warm start
        self._warmstart_check = QCheckBox("Use NLSQ warm-start (recommended)")
        self._warmstart_check.setChecked(True)
        self._warmstart_check.setToolTip(
            "Initialize MCMC from NLSQ point estimates for faster convergence"
        )
        layout.addWidget(self._warmstart_check)

        # Priors editor button
        btn_priors = QPushButton("Edit Priors...")
        btn_priors.setToolTip("View or edit prior distributions used for this run")
        btn_priors.clicked.connect(self._edit_priors)
        layout.addWidget(btn_priors)

        layout.addStretch()

        # Run button
        self._btn_run = QPushButton("Run Bayesian Inference")
        self._btn_run.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                padding: 12px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
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

        # Per-chain progress
        layout.addWidget(QLabel("Chain Progress:"))

        self._chain_progress_bars: list[QProgressBar] = []
        self._chain_labels: list[QLabel] = []
        for i in range(4):
            chain_layout = QHBoxLayout()
            label = QLabel(f"Chain {i + 1}:")
            chain_layout.addWidget(label)
            progress_bar = QProgressBar()
            progress_bar.setTextVisible(True)
            chain_layout.addWidget(progress_bar, 1)
            layout.addLayout(chain_layout)
            self._chain_progress_bars.append(progress_bar)
            self._chain_labels.append(label)

        # Status info
        layout.addWidget(QLabel("Status:"))
        self._status_text = QTextEdit()
        self._status_text.setReadOnly(True)
        self._status_text.setMaximumHeight(100)
        layout.addWidget(self._status_text)

        # ETA and divergences
        info_layout = QHBoxLayout()
        self._eta_label = QLabel("ETA: --:--")
        self._divergence_label = QLabel("Divergences: 0")
        self._divergence_label.setStyleSheet("color: #F44336; font-weight: bold;")
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
        self._fit_plot_placeholder.setStyleSheet("color: #94A3B8; padding: 6px;")
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
        warning_label.setStyleSheet("color: #e65100; padding: 2px;")
        self._diag_warning = warning_label
        # Use specific monospace fonts with fallbacks to avoid Qt font lookup warning
        self._rhat_label.setStyleSheet(
            'font-family: "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", monospace;'
        )
        self._ess_label.setStyleSheet(
            'font-family: "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", monospace;'
        )
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

    def _update_chain_progress_bars(self, num_chains: int) -> None:
        """Update visibility of chain progress bars.

        Parameters
        ----------
        num_chains : int
            Number of chains
        """
        for i, (bar, label) in enumerate(
            zip(self._chain_progress_bars, self._chain_labels, strict=False)
        ):
            visible = i < num_chains
            bar.setVisible(visible)
            label.setVisible(visible)

    @Slot()
    def _on_model_changed(self) -> None:
        """Handle model change in state."""
        state = self._store.get_state()
        model_name = state.active_model_name

        if model_name:
            self._model_label.setText(f"Model: {model_name}")
        else:
            self._model_label.setText("Model: (select in Fit tab)")

    def _on_run_clicked(self) -> None:
        """Handle run button click."""
        state = self._store.get_state()
        model_name = state.active_model_name
        dataset = self._store.get_active_dataset()

        if not model_name:
            QMessageBox.warning(
                self, "No Model", "Please select a model in the Fit tab first."
            )
            return

        if not dataset:
            QMessageBox.warning(
                self, "No Data", "Please load a dataset first."
            )
            return

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

        # Get test mode from dataset
        test_mode = dataset.metadata.get("test_mode", "oscillation")

        # Update state
        self._store.dispatch(start_bayesian(model_name, dataset.id))

        # Handle warm_start: convert bool to dict (get NLSQ params) or None
        warm_start_dict: dict[str, float] | None = None
        if config.get("warm_start", False):
            warm_start_dict = self._select_warm_start_params(model_name, dataset.id)
            if warm_start_dict:
                self._status_text.append(f"Using NLSQ warm-start: {warm_start_dict}")

        # Create and run worker (only pass args BayesianWorker accepts)
        self._current_worker = BayesianWorker(
            model_name=model_name,
            data=dataset,
            num_warmup=config.get("num_warmup", 1000),
            num_samples=config.get("num_samples", 2000),
            num_chains=config.get("num_chains", 4),
            warm_start=warm_start_dict,
        )

        # Connect signals properly (worker.signals.X, not worker.X)
        self._current_worker.signals.progress.connect(self._on_worker_progress)
        self._current_worker.signals.stage_changed.connect(self._on_stage_changed)
        self._current_worker.signals.completed.connect(self._on_finished)
        self._current_worker.signals.failed.connect(self._on_error)
        self._current_worker.signals.divergence_detected.connect(self._on_divergence)

        # Use submit() method, not start()
        self._worker_pool.submit(self._current_worker)

        self._is_running = True
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)

        self.run_requested.emit(model_name, dataset.id, config)

    def _select_warm_start_params(self, model_name: str, dataset_id: str) -> dict[str, float] | None:
        """Pick warm-start parameters that match the current model/dataset."""
        state = self._store.get_state()
        key = f"{model_name}_{dataset_id}"
        fit_result = state.fit_results.get(key)
        if fit_result and hasattr(fit_result, "parameters"):
            return dict(getattr(fit_result, "parameters"))
        return None

    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        if self._current_worker:
            self._current_worker.cancel()
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
        # Update chain progress bars based on stage
        if stage == "warmup":
            # During warmup, show warmup progress
            for bar in self._chain_progress_bars:
                bar.setFormat("Warmup: %p%")
        else:
            # During sampling, show sampling progress
            for bar in self._chain_progress_bars:
                bar.setFormat("Sampling: %p%")

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
            self._divergence_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self._status_text.append(f"WARNING: {count} divergent transitions detected")

    @Slot(int, int)
    def _on_chain_progress(self, chain_idx: int, percent: int) -> None:
        """Handle per-chain progress update (legacy slot).

        Parameters
        ----------
        chain_idx : int
            Chain index (0-based)
        percent : int
            Progress percentage
        """
        if 0 <= chain_idx < len(self._chain_progress_bars):
            self._chain_progress_bars[chain_idx].setValue(percent)

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

        # BayesianWorker's BayesianResult has .success bool (default True)
        success = getattr(result, "success", True)

        if success:
            state = self._store.get_state()
            model_name = getattr(result, "model_name", None) or state.active_model_name
            dataset_id = state.active_dataset_id

            if model_name and dataset_id:
                diagnostics = getattr(result, "diagnostics", {}) or {}
                stored_result = StoredBayesianResult(
                    model_name=str(model_name),
                    dataset_id=str(dataset_id),
                    posterior_samples=getattr(result, "posterior_samples", {}),
                    summary=getattr(result, "summary", None),
                    r_hat=diagnostics.get("r_hat", {}) or getattr(result, "r_hat", {}),
                    ess=diagnostics.get("ess", {}) or getattr(result, "ess", {}),
                    divergences=int(diagnostics.get("divergences", getattr(result, "divergences", 0)) or 0),
                    credible_intervals=getattr(result, "credible_intervals", {}) or {},
                    mcmc_time=float(getattr(result, "sampling_time", getattr(result, "mcmc_time", 0.0)) or 0.0),
                    timestamp=getattr(result, "timestamp", datetime.now()),
                    num_warmup=int(getattr(result, "num_warmup", self._warmup_spin.value()) or 0),
                    num_samples=int(getattr(result, "num_samples", self._samples_spin.value()) or 0),
                )
                store_bayesian_result(stored_result)
                self._store.dispatch("SET_PIPELINE_STEP", {"step": "bayesian", "status": "COMPLETE"})
            else:
                # Fall back to updating pipeline status only.
                self._store.dispatch("SET_PIPELINE_STEP", {"step": "bayesian", "status": "COMPLETE"})

            # Update diagnostics display
            self._update_diagnostics(result)

            # Update raw + fitted plot
            try:
                self._update_fit_plot_from_posterior(result)
            except Exception:
                # Plotting is best-effort; keep the Bayesian results UI usable.
                pass

            # Update credible intervals table
            self._update_intervals_table(result)

            self._status_text.append("Bayesian inference completed successfully!")
            self._status_text.append(f"Sampling time: {result.sampling_time:.2f}s")
            self.run_completed.emit(result)
        else:
            # Get message if available, otherwise use generic
            message = getattr(result, "message", "Inference did not converge")
            self._store.dispatch(bayesian_failed(message))
            self._status_text.append(f"Bayesian inference failed: {message}")
            QMessageBox.warning(self, "Inference Failed", message)

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
        QMessageBox.critical(self, "Inference Error", error_msg)

    @Slot()
    def _on_bayesian_started(self) -> None:
        """Handle Bayesian started signal from state."""
        self._overall_progress.setValue(0)
        for bar in self._chain_progress_bars:
            bar.setValue(0)
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
            color = "green" if max_rhat < 1.01 else "orange"
            self._rhat_label.setText(f"R-hat (max): {max_rhat:.4f} [{status}]")
            self._rhat_label.setStyleSheet(
                f'color: {color}; font-family: "SF Mono", "Menlo", "Consolas", '
                '"DejaVu Sans Mono", monospace;'
            )
        else:
            self._rhat_label.setText("R-hat: --")

        # ESS
        ess = result.diagnostics.get("ess", {})
        if ess:
            min_ess = min(ess.values()) if ess else 0
            status = "OK" if min_ess > 400 else "LOW"
            color = "green" if min_ess > 400 else "orange"
            self._ess_label.setText(f"ESS (min): {min_ess:.0f} [{status}]")
            self._ess_label.setStyleSheet(
                f'color: {color}; font-family: "SF Mono", "Menlo", "Consolas", '
                '"DejaVu Sans Mono", monospace;'
            )
        else:
            self._ess_label.setText("ESS: --")

        # Divergences
        divergences = result.diagnostics.get("divergences", 0)
        if divergences > 0:
            self._divergence_label.setText(f"Divergences: {divergences}")
            self._divergence_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self._diag_warning.setText("Warning: Divergences detected; consider higher target_accept")
        else:
            self._divergence_label.setText("Divergences: 0")
            self._divergence_label.setStyleSheet("color: green; font-weight: bold;")
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
            mean = float(values.get("mean", summary.get(param_name, {}).get("mean", 0.0)))
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

    def _posterior_mean_params(self, posterior_samples: dict[str, Any]) -> dict[str, float]:
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

    def _update_fit_plot_from_posterior(self, result: Any) -> None:
        """Render raw data + posterior-representative fitted curve."""
        dataset = self._store.get_active_dataset()
        if dataset is None:
            return

        model_name = getattr(result, "model_name", None) or self._store.get_state().active_model_name
        if not model_name:
            return

        x = np.asarray(dataset.x_data)
        y = np.asarray(dataset.y_data)
        test_mode = dataset.test_mode or getattr(dataset, "metadata", {}).get("test_mode")

        posterior_samples = getattr(result, "posterior_samples", None) or {}
        params = self._posterior_mean_params(posterior_samples)
        if not params:
            return

        y_fit = self._model_service.predict(model_name, params, x, test_mode=test_mode)

        # Reuse the Fit tab plotting style via PlotService.
        from rheojax.core.data import RheoData
        from types import SimpleNamespace

        rheo_data = RheoData(
            x=x,
            y=y,
            metadata=getattr(dataset, "metadata", {}) or {},
            initial_test_mode=test_mode,
            validate=False,
        )
        fit_like = SimpleNamespace(model_name=model_name, y_fit=y_fit)

        fig = PlotService().create_fit_plot(rheo_data, fit_like, style="default", test_mode=test_mode)
        self._set_fit_plot_figure(fig)

    def _edit_priors(self) -> None:
        """Show priors editor dialog."""
        from rheojax.gui.dialogs.bayesian_options import BayesianOptionsDialog

        dialog = BayesianOptionsDialog(
            current_options={
                "priors": self._preset_priors,
            },
            parent=self,
        )
        dialog.exec()

    def _apply_preset(self, name: str) -> None:
        """Apply sampler/prior presets derived from example notebooks."""
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
                "G_cage": {"dist": "lognormal", "loc": float(np.log(5000)), "scale": 1.0},
                "sigma_y_static": {"dist": "lognormal", "loc": float(np.log(500)), "scale": 1.0},
                "sigma_y_dynamic": {"dist": "lognormal", "loc": float(np.log(500)), "scale": 1.0},
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
                "G_cage": {"dist": "lognormal", "loc": float(np.log(5000)), "scale": 1.0},
                "sigma_y_static": {"dist": "lognormal", "loc": float(np.log(500)), "scale": 1.0},
                "sigma_y_dynamic": {"dist": "lognormal", "loc": float(np.log(500)), "scale": 1.0},
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
                    "dataset_id": result.dataset_id,
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
            QMessageBox.critical(
                self, "Export Error", f"Failed to export results: {e}"
            )

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
        # Store prior specs in state using a nested dict structure
        state = self._store.get_state()

        # Initialize prior_specs dict if not present
        if not hasattr(state, "prior_specs"):
            # Store in pipeline_state metadata for now
            state.pipeline_state.metadata["prior_specs"] = {}

        prior_specs = state.pipeline_state.metadata.get("prior_specs", {})

        if model_id not in prior_specs:
            prior_specs[model_id] = {}

        prior_specs[model_id][param_name] = prior_spec
        state.pipeline_state.metadata["prior_specs"] = prior_specs

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
            if result.model_name == model_id or model_id in str(result.dataset_id):
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
