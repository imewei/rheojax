"""
Bayesian Page
============

Bayesian inference interface with prior specification and MCMC monitoring.
"""

import json
from pathlib import Path
from typing import Any

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
from rheojax.gui.services.bayesian_service import BayesianResult, BayesianService
from rheojax.gui.state.actions import (
    bayesian_completed,
    bayesian_failed,
    start_bayesian,
    update_bayesian_progress,
)
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.arviz_canvas import ArvizCanvas


class BayesianPage(QWidget):
    """Bayesian inference page with integrated ArviZ diagnostics.

    Features:
        - MCMC configuration (sampler, warmup, samples, chains)
        - Priors editor for parameter priors
        - Real-time progress monitoring per chain
        - ArviZ diagnostic plots (trace, pair, forest, etc.)
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
        self._worker_pool = WorkerPool()
        self._current_worker: BayesianWorker | None = None
        self._is_running = False

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)

        # Left: Configuration (25%)
        left_panel = self._create_config_panel()
        main_layout.addWidget(left_panel, 1)

        # Center: Progress and Diagnostics (50%)
        center_splitter = QSplitter(Qt.Orientation.Vertical)

        progress_panel = self._create_progress_panel()
        center_splitter.addWidget(progress_panel)

        diagnostics_panel = self._create_diagnostics_panel()
        center_splitter.addWidget(diagnostics_panel)

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
        layout.addWidget(QLabel("Sampler:"))
        self._sampler_combo = QComboBox()
        self._sampler_combo.addItems(["NUTS (No U-Turn Sampler)"])
        layout.addWidget(self._sampler_combo)

        # Warmup samples
        layout.addWidget(QLabel("Warmup Samples:"))
        self._warmup_spin = QSpinBox()
        self._warmup_spin.setRange(100, 10000)
        self._warmup_spin.setValue(1000)
        self._warmup_spin.setSingleStep(100)
        layout.addWidget(self._warmup_spin)

        # Posterior samples
        layout.addWidget(QLabel("Posterior Samples:"))
        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(100, 10000)
        self._samples_spin.setValue(2000)
        self._samples_spin.setSingleStep(100)
        layout.addWidget(self._samples_spin)

        # Number of chains
        layout.addWidget(QLabel("Number of Chains:"))
        self._chains_spin = QSpinBox()
        self._chains_spin.setRange(1, 8)
        self._chains_spin.setValue(4)
        self._chains_spin.valueChanged.connect(self._update_chain_progress_bars)
        layout.addWidget(self._chains_spin)

        # HDI probability
        layout.addWidget(QLabel("HDI Probability:"))
        self._hdi_combo = QComboBox()
        self._hdi_combo.addItems(["0.90", "0.94", "0.95", "0.99"])
        self._hdi_combo.setCurrentText("0.94")
        layout.addWidget(self._hdi_combo)

        # Warm start
        self._warmstart_check = QCheckBox("Use NLSQ warm-start (recommended)")
        self._warmstart_check.setChecked(True)
        self._warmstart_check.setToolTip(
            "Initialize MCMC from NLSQ point estimates for faster convergence"
        )
        layout.addWidget(self._warmstart_check)

        # Priors editor button
        btn_priors = QPushButton("Edit Priors...")
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

    def _create_diagnostics_panel(self) -> QWidget:
        """Create ArviZ diagnostics panel."""
        panel = QGroupBox("Diagnostics")
        layout = QVBoxLayout(panel)

        # ArviZ canvas
        self._arviz_canvas = ArvizCanvas()
        self._arviz_canvas.export_requested.connect(self._export_diagnostic_plot)
        layout.addWidget(self._arviz_canvas)

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
        # Use specific monospace fonts with fallbacks to avoid Qt font lookup warning
        self._rhat_label.setStyleSheet(
            'font-family: "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", monospace;'
        )
        self._ess_label.setStyleSheet(
            'font-family: "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", monospace;'
        )
        diag_layout.addWidget(self._rhat_label)
        diag_layout.addWidget(self._ess_label)
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
        }

        # Get test mode from dataset
        test_mode = dataset.metadata.get("test_mode", "oscillation")

        # Update state
        self._store.dispatch(start_bayesian(model_name, dataset.id))

        # Handle warm_start: convert bool to dict (get NLSQ params) or None
        warm_start_dict: dict[str, float] | None = None
        if config.get("warm_start", False):
            # Try to get NLSQ fitted parameters from state
            state = self._store.get_state()
            if state.fit_results:
                # Get most recent fit result
                latest_key = list(state.fit_results.keys())[-1]
                fit_result = state.fit_results[latest_key]
                if hasattr(fit_result, "parameters"):
                    warm_start_dict = fit_result.parameters
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
            self._store.dispatch(bayesian_completed(result))

            # Update diagnostics display
            self._update_diagnostics(result)

            # Update ArviZ canvas if inference_data is available
            # Note: BayesianWorker result may not have inference_data
            if hasattr(result, "inference_data") and result.inference_data is not None:
                self._arviz_canvas.set_inference_data(result.inference_data)
            elif hasattr(result, "posterior_samples") and result.posterior_samples:
                # Create ArviZ InferenceData from posterior samples
                try:
                    import arviz as az
                    idata_dict = {}
                    for param_name, samples in result.posterior_samples.items():
                        if hasattr(samples, "ndim"):
                            if samples.ndim == 1:
                                idata_dict[param_name] = samples.reshape(1, -1)
                            else:
                                idata_dict[param_name] = samples
                        else:
                            idata_dict[param_name] = samples
                    idata = az.from_dict(idata_dict)
                    self._arviz_canvas.set_inference_data(idata)
                except Exception:
                    pass  # ArviZ visualization optional

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

    @Slot()
    def _on_bayesian_completed(self) -> None:
        """Handle Bayesian completed signal from state."""
        self._overall_progress.setValue(100)

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
        else:
            self._divergence_label.setText("Divergences: 0")
            self._divergence_label.setStyleSheet("color: green; font-weight: bold;")

    def _update_intervals_table(self, result: BayesianResult) -> None:
        """Update credible intervals table.

        Parameters
        ----------
        result : BayesianResult
            Inference result
        """
        intervals = result.credible_intervals
        self._intervals_table.setRowCount(len(intervals))

        for row, (param_name, values) in enumerate(intervals.items()):
            # Parameter name
            name_item = QTableWidgetItem(param_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._intervals_table.setItem(row, 0, name_item)

            # Mean
            mean = values.get("mean", 0)
            mean_item = QTableWidgetItem(f"{mean:.4g}")
            mean_item.setFlags(mean_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._intervals_table.setItem(row, 1, mean_item)

            # Lower bound
            lower = values.get("lower", 0)
            lower_item = QTableWidgetItem(f"{lower:.4g}")
            lower_item.setFlags(lower_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._intervals_table.setItem(row, 2, lower_item)

            # Upper bound
            upper = values.get("upper", 0)
            upper_item = QTableWidgetItem(f"{upper:.4g}")
            upper_item.setFlags(upper_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._intervals_table.setItem(row, 3, upper_item)

    def _edit_priors(self) -> None:
        """Show priors editor dialog."""
        from rheojax.gui.dialogs.bayesian_options import BayesianOptionsDialog

        dialog = BayesianOptionsDialog(self)
        dialog.exec()

    def _export_diagnostic_plot(self) -> None:
        """Export current diagnostic plot."""
        from PySide6.QtWidgets import QFileDialog

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Diagnostic Plot",
            "diagnostic_plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
        )

        if filepath:
            self._arviz_canvas.export_figure(filepath)

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
