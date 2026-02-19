"""
Diagnostics Page
===============

MCMC diagnostics and posterior analysis with ArviZ integration.
"""

from typing import Any

import numpy as np

from rheojax.gui.compat import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    Signal,
    Slot,
)
from rheojax.gui.resources.styles.tokens import ColorPalette, Spacing
from rheojax.gui.services.bayesian_service import BayesianService
from rheojax.gui.state.store import BayesianResult, StateStore
from rheojax.gui.widgets.arviz_canvas import ArviZCanvas
from rheojax.logging import get_logger

logger = get_logger(__name__)


class DiagnosticsPage(QWidget):
    """Bayesian diagnostics page with ArviZ plots."""

    plot_requested = Signal(str, str)  # plot_type, model_id
    export_requested = Signal(str)  # plot_type
    show_requested = Signal()  # ask main window to refresh diagnostics

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self._store = StateStore()
        self._bayesian_service = BayesianService()
        self._current_model_id: str | None = None
        self._current_inference_data: Any | None = None
        self.setup_ui()
        self._connect_state_signals()

    def setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Top CTA
        btn_show = QPushButton("Show Diagnostics")
        btn_show.setProperty("variant", "primary")
        btn_show.setToolTip("Refresh diagnostics from the latest Bayesian run")
        btn_show.clicked.connect(self.show_requested.emit)
        main_layout.addWidget(btn_show, 0, Qt.AlignmentFlag.AlignLeft)

        # Plot type tabs
        self._plot_tabs = QTabWidget()
        self._plot_tabs.currentChanged.connect(self._on_tab_changed)

        # Create tabs for each plot type
        plot_types = ["Trace", "Forest", "Pair", "Energy", "Autocorr", "Rank", "ESS"]
        for plot_type in plot_types:
            tab_widget = self._create_plot_tab(plot_type)
            self._plot_tabs.addTab(tab_widget, plot_type)

        main_layout.addWidget(self._plot_tabs)

        # Bottom panel: Metrics and comparison
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(self._create_metrics_panel())
        bottom_splitter.addWidget(self._create_comparison_panel())
        bottom_splitter.setSizes([500, 500])
        main_layout.addWidget(bottom_splitter)

    def _connect_state_signals(self) -> None:
        """Connect to state change signals."""
        signals = self._store.signals
        if signals:
            # Subscribe to bayesian completion
            if hasattr(signals, "bayesian_completed"):
                signals.bayesian_completed.connect(self._on_bayesian_completed)

    def _create_plot_tab(self, plot_type: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Scroll area for the plot canvas
        scroll_area = QScrollArea()
        # Use True to let canvas resize with container while still allowing scrolling
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        # Always show scrollbars when content exceeds viewport
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # ArviZ canvas for this plot type
        canvas = ArviZCanvas()
        setattr(self, f"_{plot_type.lower()}_canvas", canvas)
        scroll_area.setWidget(canvas)
        layout.addWidget(scroll_area, 1)  # Give scroll area stretch priority

        # Export button
        btn_export = QPushButton(f"Export {plot_type} Plot")
        btn_export.clicked.connect(lambda: self._export_plot(plot_type))
        layout.addWidget(btn_export)

        return widget

    def _create_metrics_panel(self) -> QWidget:
        panel = QGroupBox("Goodness of Fit Metrics")
        layout = QVBoxLayout(panel)

        # GOF metrics table
        self._gof_table = QTableWidget()
        self._gof_table.setColumnCount(2)
        self._gof_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._gof_table.setAlternatingRowColors(True)
        self._gof_table.setToolTip("Key diagnostics from the last Bayesian run")

        # Metric names (values will be updated dynamically)
        self._metric_names = [
            "R-squared",
            "Chi-squared",
            "MPE (%)",
            "WAIC",
            "LOO",
            "Effective Sample Size (min)",
            "R-hat (max)",
            "Divergences",
        ]

        self._gof_table.setRowCount(len(self._metric_names))
        for i, metric in enumerate(self._metric_names):
            self._gof_table.setItem(i, 0, QTableWidgetItem(metric))
            self._gof_table.setItem(i, 1, QTableWidgetItem("--"))

        layout.addWidget(self._gof_table)

        return panel

    def _create_comparison_panel(self) -> QWidget:
        panel = QGroupBox("Model Comparison")
        layout = QVBoxLayout(panel)

        # Model comparison table
        self._comparison_table = QTableWidget()
        self._comparison_table.setColumnCount(5)
        self._comparison_table.setHorizontalHeaderLabels(
            ["Model", "WAIC", "LOO", "ELPD", "Weight"]
        )
        self._comparison_table.setAlternatingRowColors(True)
        self._comparison_table.setToolTip(
            "Model comparison metrics; populate by running multiple models"
        )
        layout.addWidget(self._comparison_table)

        # Empty state
        empty_label = QLabel(
            "No diagnostics yet. Run Bayesian inference to populate metrics."
        )
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_SECONDARY}; padding: {Spacing.SM}px;"
        )
        layout.addWidget(empty_label)
        self._empty_label = empty_label

        # Refresh button
        btn_refresh = QPushButton("Refresh Comparison")
        btn_refresh.clicked.connect(self._refresh_comparison)
        layout.addWidget(btn_refresh)

        return panel

    def _on_tab_changed(self, index: int) -> None:
        """Handle plot tab change - refresh the plot for current model."""
        plot_type = self._plot_tabs.tabText(index)
        logger.debug(
            "Diagnostic selected", diagnostic=plot_type, page="DiagnosticsPage"
        )
        if self._current_model_id:
            self._update_plot(plot_type.lower())
            self.plot_requested.emit(plot_type, self._current_model_id)

    def _export_plot(self, plot_type: str) -> None:
        """Export current plot to file."""
        canvas_attr = f"_{plot_type.lower()}_canvas"
        canvas = getattr(self, canvas_attr, None)

        if canvas is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {plot_type} Plot",
            f"{plot_type.lower()}_plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
        )

        if filepath:
            try:
                canvas.export_figure(filepath)
                logger.info(
                    "Plot exported successfully",
                    plot_type=plot_type,
                    filepath=filepath,
                    page="DiagnosticsPage",
                )
                QMessageBox.information(
                    self, "Export Successful", f"Plot saved to {filepath}"
                )
            except Exception as e:
                logger.error(
                    "Failed to export plot",
                    plot_type=plot_type,
                    filepath=filepath,
                    error=str(e),
                    page="DiagnosticsPage",
                    exc_info=True,
                )
                QMessageBox.warning(
                    self, "Export Failed", f"Failed to export plot: {e}"
                )

        self.export_requested.emit(plot_type)

    def _refresh_comparison(self) -> None:
        """Refresh model comparison table with all bayesian results."""
        state = self._store.get_state()
        bayesian_results = state.bayesian_results

        if not bayesian_results:
            self._comparison_table.setRowCount(0)
            if hasattr(self, "_empty_label"):
                self._empty_label.show()
            return

        # Prepare results for comparison
        results_list = list(bayesian_results.values())
        self._comparison_table.setRowCount(len(results_list))
        if hasattr(self, "_empty_label"):
            self._empty_label.hide()

        for i, result in enumerate(results_list):
            model_name = result.model_name
            self._comparison_table.setItem(i, 0, QTableWidgetItem(model_name))

            # Try to calculate WAIC/LOO if we have posterior samples
            waic_val = "--"
            loo_val = "--"
            elpd_val = "--"
            weight_val = (
                "--"  # Weighting requires multi-model compare; keep placeholder
            )

            try:
                if hasattr(result, "posterior_samples") and result.posterior_samples:
                    import arviz as az

                    idata = self._get_inference_data(result)
                    if idata is not None:
                        try:
                            waic_res = az.waic(idata, scale="deviance")
                            waic_val = f"{float(waic_res.waic):.2f}"
                            # elpd_waic available on ArviZ >=0.17
                            if hasattr(waic_res, "elpd_waic"):
                                elpd_val = f"{float(waic_res.elpd_waic):.2f}"
                            elif hasattr(waic_res, "elpd"):
                                elpd_val = f"{float(waic_res.elpd):.2f}"
                        except Exception:
                            pass

                        try:
                            loo_res = az.loo(idata, scale="deviance")
                            loo_val = f"{float(loo_res.loo):.2f}"
                            if hasattr(loo_res, "elpd_loo"):
                                elpd_val = f"{float(loo_res.elpd_loo):.2f}"
                        except Exception:
                            pass
            except Exception:
                # Leave display as graceful '--'
                pass

            self._comparison_table.setItem(i, 1, QTableWidgetItem(waic_val))
            self._comparison_table.setItem(i, 2, QTableWidgetItem(loo_val))
            self._comparison_table.setItem(i, 3, QTableWidgetItem(elpd_val))
            self._comparison_table.setItem(i, 4, QTableWidgetItem(weight_val))

    @Slot(str, str)
    def _on_bayesian_completed(self, model_name: str, dataset_id: str) -> None:
        """Handle Bayesian inference completion from state.

        Parameters
        ----------
        model_name : str
            Model name
        dataset_id : str
            Dataset identifier
        """
        self.show_diagnostics(model_name=model_name, dataset_id=dataset_id)

    def show_diagnostics(self, model_name: str, dataset_id: str | None = None) -> None:
        """Show diagnostics for specified model.

        Parameters
        ----------
        model_name : str
            Model name/ID to show diagnostics for
        dataset_id : str, optional
            Dataset identifier. If omitted, uses the active dataset.
        """
        self._current_model_id = model_name

        # Get Bayesian result from state
        state = self._store.get_state()
        resolved_dataset_id = dataset_id or state.active_dataset_id
        bayesian_result: BayesianResult | None = None

        # Primary lookup: results are stored as "{model_name}_{dataset_id}".
        if resolved_dataset_id:
            key = f"{model_name}_{resolved_dataset_id}"
            bayesian_result = state.bayesian_results.get(key)

        # Fallback: attempt legacy lookup by model name.
        if bayesian_result is None:
            bayesian_result = state.bayesian_results.get(model_name)

        # Fallback: pick the most recent result for this model.
        if bayesian_result is None:
            prefix = f"{model_name}_"
            candidates = [
                res
                for key, res in state.bayesian_results.items()
                if isinstance(key, str) and key.startswith(prefix)
            ]
            if candidates:
                try:
                    bayesian_result = max(
                        candidates,
                        key=lambda r: getattr(r, "timestamp", None) or 0,
                    )
                except Exception:
                    bayesian_result = candidates[0]

        if bayesian_result is None:
            logger.debug(
                "No Bayesian results found",
                model_name=model_name,
                dataset_id=resolved_dataset_id,
                page="DiagnosticsPage",
            )
            QMessageBox.information(
                self,
                "No Bayesian Results",
                f"No Bayesian inference results found for model '{model_name}'.\n"
                "Run Bayesian inference first from the Bayesian tab.",
            )
            return

        # Build inference data from posterior samples
        self._current_inference_data = self._get_inference_data(bayesian_result)

        # Update metrics table
        self._update_metrics_table(bayesian_result)

        # Update current plot
        current_plot = self._plot_tabs.tabText(self._plot_tabs.currentIndex())
        self._update_plot(current_plot.lower())

        logger.info(
            "Diagnostics displayed",
            model_name=model_name,
            dataset_id=resolved_dataset_id,
            plot_type=current_plot,
            page="DiagnosticsPage",
        )

        self.plot_requested.emit(current_plot, model_name)

    def _get_inference_data(self, result: BayesianResult) -> Any:
        """Convert BayesianResult to ArviZ InferenceData.

        Parameters
        ----------
        result : BayesianResult
            Bayesian result from state

        Returns
        -------
        arviz.InferenceData or None
        """
        try:
            import arviz as az

            # Use stored InferenceData if available (has sample_stats for energy plot)
            inference_data = getattr(result, "inference_data", None)
            if inference_data is not None:
                # Verify it has the expected structure
                if hasattr(inference_data, "posterior"):
                    return inference_data

            posterior_samples = result.posterior_samples
            if posterior_samples is None:
                return None

            # If already InferenceData, return directly
            if hasattr(posterior_samples, "posterior"):
                return posterior_samples

            # Convert dict of samples to InferenceData
            idata_dict = {}
            for param_name, samples in posterior_samples.items():
                if isinstance(samples, np.ndarray):
                    if samples.ndim == 1:
                        # Single chain: reshape to (1, n_samples)
                        idata_dict[param_name] = samples.reshape(1, -1)
                    else:
                        idata_dict[param_name] = samples
                else:
                    # Convert to array
                    arr = np.asarray(samples)
                    if arr.ndim == 1:
                        idata_dict[param_name] = arr.reshape(1, -1)
                    else:
                        idata_dict[param_name] = arr

            return az.from_dict(idata_dict)

        except Exception as e:
            logger.error(
                "Failed to convert BayesianResult to InferenceData",
                model_name=result.model_name,
                error=str(e),
                page="DiagnosticsPage",
                exc_info=True,
            )
            return None

    def _update_metrics_table(self, result: BayesianResult) -> None:
        """Update GOF metrics table with Bayesian result.

        Parameters
        ----------
        result : BayesianResult
            Bayesian result from state
        """
        # Map metric names to values
        values = {
            "R-squared": "--",
            "Chi-squared": "--",
            "MPE (%)": "--",
            "WAIC": "--",
            "LOO": "--",
            "Effective Sample Size (min)": "--",
            "R-hat (max)": "--",
            "Divergences": "--",
        }

        # Get R-hat max
        if result.r_hat:
            max_rhat = max(result.r_hat.values())
            values["R-hat (max)"] = f"{max_rhat:.4f}"
            # Color code: green if <1.01, yellow if <1.1, red otherwise
            if max_rhat > 1.1:
                self._set_table_value_color(6, ColorPalette.ERROR)  # Red
            elif max_rhat > 1.01:
                self._set_table_value_color(6, ColorPalette.WARNING)  # Yellow
            else:
                self._set_table_value_color(6, ColorPalette.SUCCESS)  # Green

        # Get ESS min
        if result.ess:
            min_ess = min(result.ess.values())
            values["Effective Sample Size (min)"] = f"{min_ess:.0f}"
            # Color code: green if >400, yellow if >100, red otherwise
            if min_ess < 100:
                self._set_table_value_color(5, ColorPalette.ERROR)  # Red
            elif min_ess < 400:
                self._set_table_value_color(5, ColorPalette.WARNING)  # Yellow
            else:
                self._set_table_value_color(5, ColorPalette.SUCCESS)  # Green

        # Get divergences
        values["Divergences"] = str(result.divergences)
        if result.divergences > 0:
            self._set_table_value_color(7, ColorPalette.ERROR)  # Red
        else:
            self._set_table_value_color(7, ColorPalette.SUCCESS)  # Green

        # Update table
        for i, metric in enumerate(self._metric_names):
            value_item = self._gof_table.item(i, 1)
            if value_item:
                value_item.setText(values.get(metric, "--"))

    def _set_table_value_color(self, row: int, color: str) -> None:
        """Set background color for a table value cell.

        Parameters
        ----------
        row : int
            Row index
        color : str
            Hex color code
        """
        from rheojax.gui.compat import QColor

        item = self._gof_table.item(row, 1)
        if item:
            item.setBackground(QColor(color))

    def _update_plot(self, plot_type: str) -> None:
        """Update the specified plot with current inference data.

        Parameters
        ----------
        plot_type : str
            Plot type (trace, forest, pair, etc.)
        """
        if self._current_inference_data is None:
            logger.debug(
                "Plot update skipped: no inference data",
                plot_type=plot_type,
                page="DiagnosticsPage",
            )
            return

        canvas_attr = f"_{plot_type}_canvas"
        canvas = getattr(self, canvas_attr, None)

        if canvas is None:
            logger.debug(
                "Plot update skipped: no canvas",
                plot_type=plot_type,
                page="DiagnosticsPage",
            )
            return

        logger.debug(
            "Visualization triggered",
            diagnostic=plot_type,
            page="DiagnosticsPage",
            model_id=self._current_model_id,
        )

        # Set inference data on canvas and let it render
        canvas.set_inference_data(self._current_inference_data)
        canvas.set_plot_type(plot_type)

    def plot_trace(self, model_id: str) -> None:
        """Generate trace plot for model.

        Parameters
        ----------
        model_id : str
            Model name/ID
        """
        logger.debug("Diagnostic selected", diagnostic="trace", page="DiagnosticsPage")
        self._current_model_id = model_id
        self._plot_tabs.setCurrentIndex(0)  # Trace tab
        self.show_diagnostics(model_id)

    def plot_pair(self, model_id: str, show_divergences: bool = True) -> None:
        """Generate pair plot for model.

        Parameters
        ----------
        model_id : str
            Model name/ID
        show_divergences : bool
            Whether to show divergences on plot
        """
        logger.debug("Diagnostic selected", diagnostic="pair", page="DiagnosticsPage")
        self._current_model_id = model_id
        self._plot_tabs.setCurrentIndex(2)  # Pair tab
        self.show_diagnostics(model_id)

    def plot_forest(self, model_id: str, hdi_prob: float = 0.95) -> None:
        """Generate forest plot for model.

        Parameters
        ----------
        model_id : str
            Model name/ID
        hdi_prob : float
            HDI probability for credible intervals
        """
        logger.debug("Diagnostic selected", diagnostic="forest", page="DiagnosticsPage")
        self._current_model_id = model_id
        self._plot_tabs.setCurrentIndex(1)  # Forest tab

        # Set HDI probability on the forest canvas
        if hasattr(self, "_forest_canvas"):
            self._forest_canvas.set_hdi_prob(hdi_prob)

        self.show_diagnostics(model_id)

    def get_diagnostic_summary(self, model_name: str) -> dict[str, Any]:
        """Get diagnostic summary for model.

        Parameters
        ----------
        model_name : str
            Model name/ID

        Returns
        -------
        dict
            Diagnostic summary including R-hat, ESS, divergences
        """
        state = self._store.get_state()
        dataset_id = state.active_dataset_id
        result = None
        if dataset_id:
            result = state.bayesian_results.get(f"{model_name}_{dataset_id}")
        if result is None:
            result = state.bayesian_results.get(model_name)

        if result is None:
            return {}

        return {
            "model_name": result.model_name,
            "r_hat": result.r_hat,
            "ess": result.ess,
            "divergences": result.divergences,
            "credible_intervals": result.credible_intervals,
            "num_warmup": result.num_warmup,
            "num_samples": result.num_samples,
            "mcmc_time": result.mcmc_time,
            "max_r_hat": max(result.r_hat.values()) if result.r_hat else None,
            "min_ess": min(result.ess.values()) if result.ess else None,
            "converged": (max(result.r_hat.values()) < 1.1 if result.r_hat else False)
            and (min(result.ess.values()) > 400 if result.ess else False)
            and result.divergences == 0,
        }
