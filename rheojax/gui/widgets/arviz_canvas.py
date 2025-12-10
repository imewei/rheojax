"""
ArviZ Canvas Widget
==================

ArviZ diagnostic plot integration for Bayesian inference visualization.
"""

from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Available ArviZ plot types
PLOT_TYPES = [
    ("trace", "Trace Plot", "MCMC trace and posterior distributions"),
    ("pair", "Pair Plot", "Pairwise parameter correlations"),
    ("forest", "Forest Plot", "Credible intervals comparison"),
    ("posterior", "Posterior", "Posterior distributions"),
    ("energy", "Energy", "MCMC energy diagnostics"),
    ("rank", "Rank Plot", "Chain rank statistics"),
    ("ess", "ESS", "Effective sample size"),
    ("autocorr", "Autocorrelation", "Chain autocorrelation"),
]


class ArvizCanvas(QWidget):
    """ArviZ diagnostic canvas for Bayesian visualization.

    Features:
        - All major ArviZ plot types (trace, pair, forest, etc.)
        - Interactive matplotlib toolbar for zoom/pan
        - Plot type selector dropdown
        - Export support (PNG, PDF, SVG)
        - HDI probability control

    Signals
    -------
    plot_changed : Signal(str)
        Emitted when plot type changes
    export_requested : Signal()
        Emitted when export is requested

    Example
    -------
    >>> canvas = ArvizCanvas()  # doctest: +SKIP
    >>> canvas.plot_trace(inference_data)  # doctest: +SKIP
    >>> canvas.set_plot_type("pair")  # doctest: +SKIP
    """

    plot_changed = Signal(str)
    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize ArviZ canvas.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._inference_data: Any | None = None
        self._current_plot_type = "trace"
        self._hdi_prob = 0.94

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(4, 4, 4, 4)

        # Plot type selector
        type_label = QLabel("Plot Type:")
        toolbar_layout.addWidget(type_label)

        self._type_combo = QComboBox()
        self._type_combo.setMinimumWidth(120)
        for plot_id, display_name, tooltip in PLOT_TYPES:
            self._type_combo.addItem(display_name, plot_id)
            idx = self._type_combo.count() - 1
            self._type_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        toolbar_layout.addWidget(self._type_combo)

        toolbar_layout.addStretch()

        # Refresh button
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Regenerate current plot")
        toolbar_layout.addWidget(self._refresh_btn)

        # Export button
        self._export_btn = QPushButton("Export")
        self._export_btn.setToolTip("Export plot to file")
        toolbar_layout.addWidget(self._export_btn)

        layout.addLayout(toolbar_layout)

        # Create matplotlib figure and canvas
        self._figure = Figure(figsize=(8, 6), dpi=100)
        self._figure.set_layout_engine("tight")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Navigation toolbar
        self._nav_toolbar = NavigationToolbar2QT(self._canvas, self)

        layout.addWidget(self._nav_toolbar)
        layout.addWidget(self._canvas, 1)

        # Status / empty label
        self._status_label = QLabel("No diagnostics yet. Run Bayesian inference to view plots.")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: #94A3B8; padding: 6px;")
        layout.addWidget(self._status_label)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._refresh_btn.clicked.connect(self._refresh_plot)
        self._export_btn.clicked.connect(self.export_requested.emit)

    def _on_type_changed(self, index: int) -> None:
        """Handle plot type change.

        Parameters
        ----------
        index : int
            New combo box index
        """
        plot_type = self._type_combo.currentData()
        self._current_plot_type = plot_type
        self.plot_changed.emit(plot_type)

        if self._inference_data is not None:
            self._refresh_plot()

    def _refresh_plot(self) -> None:
        """Refresh the current plot."""
        if self._inference_data is None:
            self._status_label.show()
            return

        self._status_label.hide()

        # Clear figure
        self._figure.clear()

        # Generate plot based on type
        plot_funcs = {
            "trace": self._plot_trace,
            "pair": self._plot_pair,
            "forest": self._plot_forest,
            "posterior": self._plot_posterior,
            "energy": self._plot_energy,
            "rank": self._plot_rank,
            "ess": self._plot_ess,
            "autocorr": self._plot_autocorr,
        }

        func = plot_funcs.get(self._current_plot_type)
        if func:
            try:
                func()
                self._status_label.setText(f"Showing: {self._current_plot_type}")
                self._status_label.hide()
            except Exception as e:
                self._status_label.setText(f"Error: {e}")
                self._status_label.show()
                ax = self._figure.add_subplot(111)
                ax.text(
                    0.5, 0.5, f"Error generating plot:\n{e}",
                    ha="center", va="center", transform=ax.transAxes
                )
        else:
            self._status_label.setText("Unsupported plot type")
            self._status_label.show()

        self._canvas.draw()

    def set_inference_data(self, idata: Any) -> None:
        """Set ArviZ InferenceData object.

        Parameters
        ----------
        idata : arviz.InferenceData
            Inference data from Bayesian fitting
        """
        self._inference_data = idata
        self._refresh_plot()

    def set_hdi_prob(self, prob: float) -> None:
        """Set HDI probability for credible intervals.

        Parameters
        ----------
        prob : float
            HDI probability (0-1), e.g., 0.94 for 94% HDI
        """
        self._hdi_prob = prob
        self._refresh_plot()

    def set_plot_type(self, plot_type: str) -> None:
        """Set current plot type.

        Parameters
        ----------
        plot_type : str
            Plot type identifier
        """
        idx = self._type_combo.findData(plot_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)

    def get_plot_type(self) -> str:
        """Get current plot type.

        Returns
        -------
        str
            Plot type identifier
        """
        return self._current_plot_type

    def _plot_trace(self) -> None:
        """Generate trace plot."""
        try:
            import arviz as az
            az.plot_trace(self._inference_data, figure=self._figure)
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_pair(self) -> None:
        """Generate pair plot."""
        try:
            import arviz as az
            az.plot_pair(
                self._inference_data,
                divergences=True,
                figure=self._figure
            )
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_forest(self) -> None:
        """Generate forest plot."""
        try:
            import arviz as az
            az.plot_forest(
                self._inference_data,
                hdi_prob=self._hdi_prob,
                combined=True,
                figure=self._figure
            )
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_posterior(self) -> None:
        """Generate posterior plot."""
        try:
            import arviz as az
            az.plot_posterior(
                self._inference_data,
                hdi_prob=self._hdi_prob,
                figure=self._figure
            )
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_energy(self) -> None:
        """Generate energy plot."""
        try:
            import arviz as az
            az.plot_energy(self._inference_data, figure=self._figure)
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_rank(self) -> None:
        """Generate rank plot."""
        try:
            import arviz as az
            az.plot_rank(self._inference_data, figure=self._figure)
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_ess(self) -> None:
        """Generate ESS plot."""
        try:
            import arviz as az
            az.plot_ess(self._inference_data, figure=self._figure)
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_autocorr(self) -> None:
        """Generate autocorrelation plot."""
        try:
            import arviz as az
            az.plot_autocorr(self._inference_data, figure=self._figure)
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_fallback(self, message: str) -> None:
        """Display fallback message when plotting fails.

        Parameters
        ----------
        message : str
            Message to display
        """
        ax = self._figure.add_subplot(111)
        ax.text(
            0.5, 0.5, message,
            ha="center", va="center", fontsize=14,
            transform=ax.transAxes
        )
        ax.set_axis_off()

    def plot_trace(self, data: Any) -> None:
        """Generate trace plot (convenience method).

        Parameters
        ----------
        data : arviz.InferenceData
            Inference data
        """
        self.set_inference_data(data)
        self.set_plot_type("trace")

    def plot_pair(self, data: Any) -> None:
        """Generate pair plot (convenience method).

        Parameters
        ----------
        data : arviz.InferenceData
            Inference data
        """
        self.set_inference_data(data)
        self.set_plot_type("pair")

    def export_figure(self, filepath: str, dpi: int = 150) -> None:
        """Export figure to file.

        Parameters
        ----------
        filepath : str
            Output file path
        dpi : int, optional
            Resolution for raster formats
        """
        self._figure.savefig(filepath, dpi=dpi, bbox_inches="tight")

    def clear(self) -> None:
        """Clear the canvas."""
        self._figure.clear()
        self._inference_data = None
        self._status_label.setText("No data loaded")
        self._canvas.draw()

    def get_figure(self) -> Figure:
        """Get matplotlib figure.

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        return self._figure


# Alias for backward compatibility
ArviZCanvas = ArvizCanvas
