"""
Diagnostics Page
===============

MCMC diagnostics and posterior analysis with ArviZ integration.
"""

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.arviz_canvas import ArviZCanvas


class DiagnosticsPage(QWidget):
    """Bayesian diagnostics page with ArviZ plots."""

    plot_requested = Signal(str, str)  # plot_type, model_id
    export_requested = Signal(str)  # plot_type
    show_requested = Signal()  # ask main window to refresh diagnostics

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = StateStore()
        self._current_model_id: str | None = None
        self.setup_ui()

    def setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Top CTA
        btn_show = QPushButton("Show Diagnostics")
        btn_show.setStyleSheet("QPushButton { font-weight: bold; padding: 8px 12px; }")
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

    def _create_plot_tab(self, plot_type: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ArviZ canvas for this plot type
        canvas = ArviZCanvas()
        setattr(self, f"_{plot_type.lower()}_canvas", canvas)
        layout.addWidget(canvas)

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

        metrics = [
            ("R-squared", "--"),
            ("Chi-squared", "--"),
            ("MPE (%)", "--"),
            ("WAIC", "--"),
            ("LOO", "--"),
            ("Effective Sample Size (min)", "--"),
            ("R-hat (max)", "--"),
            ("Divergences", "--"),
        ]

        self._gof_table.setRowCount(len(metrics))
        for i, (metric, value) in enumerate(metrics):
            self._gof_table.setItem(i, 0, QTableWidgetItem(metric))
            self._gof_table.setItem(i, 1, QTableWidgetItem(value))

        layout.addWidget(self._gof_table)

        return panel

    def _create_comparison_panel(self) -> QWidget:
        panel = QGroupBox("Model Comparison")
        layout = QVBoxLayout(panel)

        # Model comparison table
        self._comparison_table = QTableWidget()
        self._comparison_table.setColumnCount(5)
        self._comparison_table.setHorizontalHeaderLabels(["Model", "WAIC", "LOO", "ELPD", "Weight"])
        self._comparison_table.setAlternatingRowColors(True)
        layout.addWidget(self._comparison_table)

        # Refresh button
        btn_refresh = QPushButton("Refresh Comparison")
        btn_refresh.clicked.connect(self._refresh_comparison)
        layout.addWidget(btn_refresh)

        return panel

    def _on_tab_changed(self, index: int) -> None:
        plot_type = self._plot_tabs.tabText(index)
        if self._current_model_id:
            self.plot_requested.emit(plot_type, self._current_model_id)

    def _export_plot(self, plot_type: str) -> None:
        self.export_requested.emit(plot_type)

    def _refresh_comparison(self) -> None:
        # Refresh model comparison table
        pass

    def show_diagnostics(self, model_id: str) -> None:
        self._current_model_id = model_id
        current_plot = self._plot_tabs.tabText(self._plot_tabs.currentIndex())
        self.plot_requested.emit(current_plot, model_id)

    def plot_trace(self, model_id: str) -> None:
        pass

    def plot_pair(self, model_id: str, show_divergences: bool = True) -> None:
        pass

    def plot_forest(self, model_id: str, hdi_prob: float = 0.95) -> None:
        pass

    def get_diagnostic_summary(self, model_id: str) -> dict[str, Any]:
        return {}
