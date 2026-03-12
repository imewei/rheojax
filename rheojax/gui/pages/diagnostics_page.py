"""
Diagnostics Page
===============

MCMC diagnostics and posterior analysis with ArviZ integration.
"""

from typing import Any

import numpy as np

from rheojax.gui.compat import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Signal,
    Slot,
)
from rheojax.gui.resources.styles.tokens import ColorPalette
from rheojax.gui.services.bayesian_service import BayesianService
from rheojax.gui.state.store import BayesianResult, FitResult, StateStore
from rheojax.gui.utils.layout_helpers import apply_group_box_style, set_compact_margins
from rheojax.gui.widgets.arviz_canvas import ArviZCanvas
from rheojax.logging import get_logger

logger = get_logger(__name__)


class DiagnosticsPage(QWidget):
    """Bayesian diagnostics page with ArviZ plots."""

    plot_requested = Signal(str, str)  # plot_type, model_id
    show_requested = Signal()  # kept for backward compat (main_window connection)

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

        # ArviZCanvas has its own internal QScrollArea that wraps only the
        # figure canvas.  The toolbar (Plot Type, Refresh, Export) stays fixed
        # above the scroll area so buttons are always visible.
        self._canvas = ArviZCanvas()
        main_layout.addWidget(self._canvas, 1)

        # Bottom panel: Metrics and comparison (compact, fixed height)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self._create_metrics_panel())
        bottom_splitter.addWidget(self._create_comparison_panel())
        bottom_splitter.setSizes([500, 500])
        bottom_splitter.setMaximumHeight(200)
        main_layout.addWidget(bottom_splitter)

    def _connect_state_signals(self) -> None:
        """Connect to state change signals."""
        signals = self._store.signals
        if signals:
            # Subscribe to bayesian completion
            if hasattr(signals, "bayesian_completed"):
                signals.bayesian_completed.connect(self._on_bayesian_completed)

    def _create_metrics_panel(self) -> QWidget:
        panel = QGroupBox("Goodness of Fit Metrics")
        layout = QVBoxLayout(panel)
        set_compact_margins(layout)

        # GOF metrics in a compact 4-column layout:
        #   Metric | Value | Metric | Value
        self._metric_names = [
            "R²",
            "Chi²",
            "MPE (%)",
            "WAIC",
            "LOO",
            "ESS (min)",
            "R-hat (max)",
            "Divergences",
        ]

        n_cols_pair = 2  # number of metric-value pairs per row
        n_rows = (len(self._metric_names) + n_cols_pair - 1) // n_cols_pair

        self._gof_table = QTableWidget()
        self._gof_table.setRowCount(n_rows)
        self._gof_table.setColumnCount(n_cols_pair * 2)
        self._gof_table.setHorizontalHeaderLabels(["Metric", "Value"] * n_cols_pair)
        self._gof_table.setAlternatingRowColors(True)
        self._gof_table.setToolTip("Key diagnostics from the last Bayesian run")

        # Compact sizing: hide row numbers, no scrollbars.
        # Do NOT force header height — the QSS QHeaderView::section has
        # padding: 8px 12px which needs ~34px; forcing 22px clips the text.
        self._gof_table.verticalHeader().setVisible(False)
        self._gof_table.verticalHeader().setDefaultSectionSize(24)
        self._gof_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._gof_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._gof_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._gof_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Map each metric name → (row, value_col) for fast lookup
        self._metric_cell_map: dict[str, tuple[int, int]] = {}
        for idx, metric in enumerate(self._metric_names):
            row = idx % n_rows
            pair = idx // n_rows  # 0 = left pair, 1 = right pair
            metric_col = pair * 2
            value_col = metric_col + 1

            self._gof_table.setItem(row, metric_col, QTableWidgetItem(metric))
            self._gof_table.setItem(row, value_col, QTableWidgetItem("--"))
            self._metric_cell_map[metric] = (row, value_col)

        self._gof_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._gof_table)

        apply_group_box_style(panel, "card")
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        return panel

    def _create_comparison_panel(self) -> QWidget:
        panel = QGroupBox("Model Comparison")
        layout = QVBoxLayout(panel)
        set_compact_margins(layout)

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
        # Compact sizing — do NOT force header height (same reason as GOF table)
        self._comparison_table.verticalHeader().setVisible(False)
        self._comparison_table.verticalHeader().setDefaultSectionSize(24)
        self._comparison_table.horizontalHeader().setStretchLastSection(True)
        self._comparison_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._comparison_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        layout.addWidget(self._comparison_table)

        # Empty state (inline with refresh button)
        bottom_row = QHBoxLayout()
        empty_label = QLabel("Run Bayesian inference to populate.")
        empty_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_SECONDARY}; font-size: 11px;"
        )
        bottom_row.addWidget(empty_label)
        self._empty_label = empty_label

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(80)
        btn_refresh.clicked.connect(self._refresh_comparison)
        bottom_row.addWidget(btn_refresh)
        layout.addLayout(bottom_row)

        apply_group_box_style(panel, "card")
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        return panel

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

        # Look up the corresponding NLSQ FitResult for GOF metrics
        fit_result = self._find_fit_result(model_name, resolved_dataset_id)

        # Build inference data from posterior samples
        self._current_inference_data = self._get_inference_data(bayesian_result)

        # Update metrics table with both Bayesian diagnostics and NLSQ GOF
        self._update_metrics_table(bayesian_result, fit_result)

        # Update plot on the single canvas
        if self._current_inference_data is not None:
            self._canvas.set_inference_data(self._current_inference_data)

        current_plot = self._canvas.get_plot_type()

        logger.info(
            "Diagnostics displayed",
            model_name=model_name,
            dataset_id=resolved_dataset_id,
            plot_type=current_plot,
            page="DiagnosticsPage",
        )

        self.plot_requested.emit(current_plot, model_name)

    def _find_fit_result(
        self, model_name: str, dataset_id: str | None
    ) -> FitResult | None:
        """Look up the NLSQ FitResult for the given model/dataset.

        Parameters
        ----------
        model_name : str
            Model name
        dataset_id : str, optional
            Dataset identifier

        Returns
        -------
        FitResult or None
        """
        state = self._store.get_state()

        # Primary lookup: "{model_name}_{dataset_id}"
        if dataset_id:
            key = f"{model_name}_{dataset_id}"
            result = state.fit_results.get(key)
            if result is not None:
                return result

        # Fallback: by model name
        result = state.fit_results.get(model_name)
        if result is not None:
            return result

        # Fallback: most recent for this model
        prefix = f"{model_name}_"
        candidates = [
            res
            for key, res in state.fit_results.items()
            if isinstance(key, str) and key.startswith(prefix)
        ]
        if candidates:
            # Return the last inserted (most recent by dict order in Python 3.7+)
            return candidates[-1]

        return None

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

            # Use num_chains to reshape flat posterior arrays into
            # (num_chains, num_samples_per_chain) so ArviZ gets proper chain
            # structure.  This is critical: NumPyro's get_samples() returns
            # flat (num_chains*num_samples,) arrays, but sample_stats from
            # the subprocess preserve the (num_chains, num_samples) shape.
            # A mismatch causes IndexError in trace/pair plots with divergences.
            num_chains = getattr(result, "num_chains", 1) or 1

            # Convert dict of samples to InferenceData
            idata_dict = {}
            for param_name, samples in posterior_samples.items():
                arr = np.asarray(samples)
                if arr.ndim == 1:
                    # Flat array — reshape to (num_chains, samples_per_chain)
                    if num_chains > 1 and arr.size % num_chains == 0:
                        idata_dict[param_name] = arr.reshape(num_chains, -1)
                    else:
                        idata_dict[param_name] = arr.reshape(1, -1)
                else:
                    idata_dict[param_name] = arr

            # Build sample_stats dict for energy plot if available
            sample_stats_dict = None
            raw_stats = getattr(result, "sample_stats", None)
            if raw_stats and isinstance(raw_stats, dict):
                sample_stats_dict = {}
                for stat_name, stat_arr in raw_stats.items():
                    arr = np.asarray(stat_arr)
                    if arr.ndim == 1:
                        if num_chains > 1 and arr.size % num_chains == 0:
                            sample_stats_dict[stat_name] = arr.reshape(num_chains, -1)
                        else:
                            sample_stats_dict[stat_name] = arr.reshape(1, -1)
                    else:
                        sample_stats_dict[stat_name] = arr

            return az.from_dict(
                posterior=idata_dict,
                sample_stats=sample_stats_dict,
            )

        except Exception as e:
            logger.error(
                "Failed to convert BayesianResult to InferenceData",
                model_name=result.model_name,
                error=str(e),
                page="DiagnosticsPage",
                exc_info=True,
            )
            return None

    def _update_metrics_table(
        self,
        result: BayesianResult,
        fit_result: FitResult | None = None,
    ) -> None:
        """Update GOF metrics table with Bayesian diagnostics and NLSQ fit metrics.

        Parameters
        ----------
        result : BayesianResult
            Bayesian result from state
        fit_result : FitResult, optional
            NLSQ fit result for R-squared, Chi-squared, MPE
        """
        # Map metric names to values
        values: dict[str, str] = dict.fromkeys(self._metric_names, "--")

        # --- GOF metrics from NLSQ FitResult ---
        if fit_result is not None:
            r2 = getattr(fit_result, "r_squared", None)
            if r2 is not None:
                values["R²"] = f"{float(r2):.6f}"
                if r2 >= 0.99:
                    self._color_metric("R²", ColorPalette.SUCCESS)
                elif r2 >= 0.95:
                    self._color_metric("R²", ColorPalette.WARNING)
                else:
                    self._color_metric("R²", ColorPalette.ERROR)

            chi2 = getattr(fit_result, "chi_squared", None)
            if chi2 is not None:
                values["Chi²"] = f"{float(chi2):.4e}"

            mpe = getattr(fit_result, "mpe", None)
            if mpe is not None:
                values["MPE (%)"] = f"{float(mpe):.2f}"
                if abs(mpe) <= 5.0:
                    self._color_metric("MPE (%)", ColorPalette.SUCCESS)
                elif abs(mpe) <= 15.0:
                    self._color_metric("MPE (%)", ColorPalette.WARNING)
                else:
                    self._color_metric("MPE (%)", ColorPalette.ERROR)

        # --- WAIC / LOO from InferenceData ---
        idata = self._current_inference_data
        if idata is not None:
            try:
                import arviz as az

                try:
                    waic_res = az.waic(idata, scale="deviance")
                    values["WAIC"] = f"{float(waic_res.waic):.2f}"
                except Exception:
                    pass

                try:
                    loo_res = az.loo(idata, scale="deviance")
                    values["LOO"] = f"{float(loo_res.loo):.2f}"
                except Exception:
                    pass
            except ImportError:
                pass

        # --- MCMC diagnostics from BayesianResult ---
        if result.r_hat:
            max_rhat = max(result.r_hat.values())
            values["R-hat (max)"] = f"{max_rhat:.4f}"
            if max_rhat > 1.1:
                self._color_metric("R-hat (max)", ColorPalette.ERROR)
            elif max_rhat > 1.01:
                self._color_metric("R-hat (max)", ColorPalette.WARNING)
            else:
                self._color_metric("R-hat (max)", ColorPalette.SUCCESS)

        if result.ess:
            min_ess = min(result.ess.values())
            values["ESS (min)"] = f"{min_ess:.0f}"
            if min_ess < 100:
                self._color_metric("ESS (min)", ColorPalette.ERROR)
            elif min_ess < 400:
                self._color_metric("ESS (min)", ColorPalette.WARNING)
            else:
                self._color_metric("ESS (min)", ColorPalette.SUCCESS)

        display_divergences = max(result.divergences, 0)
        values["Divergences"] = (
            "unknown" if result.divergences == -1 else str(display_divergences)
        )
        if display_divergences > 0:
            self._color_metric("Divergences", ColorPalette.ERROR)
        else:
            self._color_metric("Divergences", ColorPalette.SUCCESS)

        # Update table cells via the name → (row, col) map
        for metric, val in values.items():
            cell = self._metric_cell_map.get(metric)
            if cell is not None:
                row, col = cell
                item = self._gof_table.item(row, col)
                if item:
                    item.setText(val)

    def _color_metric(self, metric_name: str, color: str) -> None:
        """Set background color for a metric's value cell.

        Parameters
        ----------
        metric_name : str
            Metric name (must exist in ``_metric_cell_map``)
        color : str
            Hex color code
        """
        from rheojax.gui.compat import QColor

        cell = self._metric_cell_map.get(metric_name)
        if cell is not None:
            row, col = cell
            item = self._gof_table.item(row, col)
            if item:
                item.setBackground(QColor(color))

    def plot_trace(self, model_id: str) -> None:
        """Generate trace plot for model.

        Parameters
        ----------
        model_id : str
            Model name/ID
        """
        logger.debug("Diagnostic selected", diagnostic="trace", page="DiagnosticsPage")
        self._current_model_id = model_id
        self._canvas.set_plot_type("trace")
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
        self._canvas.set_plot_type("pair")
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
        self._canvas.set_hdi_prob(hdi_prob)
        self._canvas.set_plot_type("forest")
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
            and result.divergences <= 0,  # 0 = no divergences, -1 = unknown
        }
