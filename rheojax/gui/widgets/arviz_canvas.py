"""
ArviZ Canvas Widget
==================

ArviZ diagnostic plot integration for Bayesian inference visualization.
"""

import logging
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from rheojax.gui.widgets.base_arviz_widget import BaseArviZWidget, PlotMetrics

logger = logging.getLogger(__name__)

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


class ArvizCanvas(BaseArviZWidget):
    """ArviZ diagnostic canvas for Bayesian visualization.

    Inherits from BaseArviZWidget to use the standardized figure swapping
    protocol with proper cleanup and performance tracking.

    Features:
        - All major ArviZ plot types (trace, pair, forest, etc.)
        - Interactive matplotlib toolbar for zoom/pan
        - Plot type selector dropdown
        - Export support (PNG, PDF, SVG)
        - HDI probability control
        - Performance metrics via PlotMetrics

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

    def __init__(self, parent: BaseArviZWidget | None = None) -> None:
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
        # Note: We don't set a layout engine here because ArviZ creates its own
        # figures which we swap in via _copy_arviz_figure(). The initial figure
        # is just a placeholder.
        self._figure = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        # Use Minimum policy so widget can grow beyond container for scrolling
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        # Set initial minimum size based on figure dimensions
        self._update_canvas_size()

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

    def _update_canvas_size(self) -> None:
        """Update canvas minimum size based on current figure dimensions.

        This ensures scrollbars appear when the figure is larger than the viewport.
        """
        if self._figure is None:
            return

        # Get figure size in pixels
        fig_width, fig_height = self._figure.get_size_inches()
        dpi = self._figure.get_dpi()
        width_px = int(fig_width * dpi)
        height_px = int(fig_height * dpi)

        # Set minimum size on canvas to enable scrolling
        self._canvas.setMinimumSize(width_px, height_px)

        # Update size hints
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        """Return recommended size based on figure dimensions.

        Returns
        -------
        QSize
            Recommended widget size
        """
        if self._figure is not None:
            fig_width, fig_height = self._figure.get_size_inches()
            dpi = self._figure.get_dpi()
            # Add space for toolbar and status label
            toolbar_height = 40
            return QSize(int(fig_width * dpi), int(fig_height * dpi) + toolbar_height)
        return QSize(800, 640)

    def minimumSizeHint(self) -> QSize:
        """Return minimum size for the widget.

        Returns
        -------
        QSize
            Minimum widget size (allows scrolling for larger plots)
        """
        # Allow widget to shrink to a reasonable minimum
        return QSize(400, 300)

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
        """Refresh the current plot with performance tracking."""
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
                # Use timed_render for performance tracking
                self.timed_render(self._current_plot_type, func)
                self._status_label.setText(f"Showing: {self._current_plot_type}")
                self._status_label.hide()

                logger.debug(
                    "plot_refresh_complete",
                    extra={
                        "plot_type": self._current_plot_type,
                        "has_inference_data": self._inference_data is not None,
                    },
                )
            except Exception as e:
                logger.warning(f"Plot generation failed: {e}", exc_info=True)
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

        # Sanitize any tab characters before drawing to avoid
        # "Glyph 9 missing from font" warnings
        self._sanitize_figure_text(self._figure)

        # Suppress font glyph warnings during draw - ArviZ may generate
        # text with tabs in tick labels that aren't accessible until draw time
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Glyph.*missing from font")
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

    def _copy_arviz_figure(self, arviz_fig: Figure) -> None:
        """Swap ArviZ-generated figure into our Qt canvas.

        ArviZ plotting functions create their own figures internally and don't
        accept a figure= parameter. This method uses the base class swap_figure
        protocol for thread-safe figure swapping with proper cleanup.

        Parameters
        ----------
        arviz_fig : Figure
            The figure created by ArviZ
        """
        # Sanitize text before swapping to avoid glyph warnings
        self._sanitize_figure_text(arviz_fig)

        # Use base class swap_figure for thread-safe figure replacement
        # This handles canvas transfer and schedules cleanup on main thread
        self.swap_figure(arviz_fig)

        # Update canvas size for scrolling based on new figure dimensions
        self._update_canvas_size()

        logger.debug(
            "arviz_figure_swapped",
            extra={
                "num_axes": len(arviz_fig.get_axes()),
                "plot_type": self._current_plot_type,
            },
        )

    def _sanitize_figure_text(self, fig: Figure) -> None:
        """Replace tab characters with spaces in all figure text.

        ArviZ sometimes uses tab characters in labels which causes
        "Glyph 9 missing from font" warnings with some fonts.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to sanitize
        """
        for ax in fig.get_axes():
            # Sanitize title
            title = ax.get_title()
            if title and "\t" in title:
                ax.set_title(title.replace("\t", "  "))

            # Sanitize axis labels
            xlabel = ax.get_xlabel()
            if xlabel and "\t" in xlabel:
                ax.set_xlabel(xlabel.replace("\t", "  "))

            ylabel = ax.get_ylabel()
            if ylabel and "\t" in ylabel:
                ax.set_ylabel(ylabel.replace("\t", "  "))

            # Sanitize tick labels
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                text = label.get_text()
                if text and "\t" in text:
                    label.set_text(text.replace("\t", "  "))

            # Sanitize legend if present
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    label_text = text.get_text()
                    if label_text and "\t" in label_text:
                        text.set_text(label_text.replace("\t", "  "))

        # Sanitize suptitle if present
        if fig._suptitle and fig._suptitle.get_text():
            suptitle = fig._suptitle.get_text()
            if "\t" in suptitle:
                fig.suptitle(suptitle.replace("\t", "  "))

    def _plot_trace(self) -> None:
        """Generate trace plot."""
        try:
            import arviz as az
            import matplotlib.pyplot as plt

            # Close any previous ArviZ figures to avoid memory leaks
            plt.close("all")

            # Let ArviZ create its own figure (it doesn't accept figure= kwarg)
            az.plot_trace(self._inference_data)

            # Get the figure ArviZ created and copy to our managed figure
            self._copy_arviz_figure(plt.gcf())
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_pair(self) -> None:
        """Generate pair plot."""
        try:
            import arviz as az
            import matplotlib.pyplot as plt

            plt.close("all")

            # Check if inference data has sample_stats with diverging info
            # before requesting divergences plot
            has_divergences = (
                self._inference_data is not None
                and hasattr(self._inference_data, "sample_stats")
                and self._inference_data.sample_stats is not None
                and "diverging" in self._inference_data.sample_stats
            )

            az.plot_pair(self._inference_data, divergences=has_divergences)
            self._copy_arviz_figure(plt.gcf())
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_forest(self) -> None:
        """Generate forest plot."""
        try:
            import arviz as az
            import matplotlib.pyplot as plt

            plt.close("all")
            az.plot_forest(self._inference_data, hdi_prob=self._hdi_prob, combined=True)
            self._copy_arviz_figure(plt.gcf())
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_posterior(self) -> None:
        """Generate posterior plot."""
        try:
            import arviz as az
            import matplotlib.pyplot as plt

            plt.close("all")
            az.plot_posterior(self._inference_data, hdi_prob=self._hdi_prob)
            self._copy_arviz_figure(plt.gcf())
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_energy(self) -> None:
        """Generate energy plot.

        Requires sample_stats group with energy diagnostic from MCMC sampling.
        Shows fallback message if sample_stats is not available.
        """
        try:
            import arviz as az
            import matplotlib.pyplot as plt

            # Check if sample_stats exists with energy diagnostic
            if not self._has_sample_stats_energy():
                self._plot_fallback(
                    "Energy plot requires MCMC sample statistics.\n\n"
                    "The current InferenceData only contains posterior samples.\n"
                    "To view energy diagnostics, ensure the Bayesian result\n"
                    "was created with full MCMC diagnostics (via fit_bayesian)."
                )
                return

            plt.close("all")
            az.plot_energy(self._inference_data)
            self._copy_arviz_figure(plt.gcf())
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _has_sample_stats_energy(self) -> bool:
        """Check if InferenceData has sample_stats with energy.

        Returns
        -------
        bool
            True if sample_stats.energy exists, False otherwise
        """
        if self._inference_data is None:
            return False

        sample_stats = getattr(self._inference_data, "sample_stats", None)
        if sample_stats is None:
            return False

        return "energy" in sample_stats

    def _plot_rank(self) -> None:
        """Generate rank plot."""
        try:
            import arviz as az
            import matplotlib.pyplot as plt

            plt.close("all")
            az.plot_rank(self._inference_data)
            self._copy_arviz_figure(plt.gcf())
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_ess(self) -> None:
        """Generate ESS plot."""
        try:
            import arviz as az
            import matplotlib.pyplot as plt

            plt.close("all")
            az.plot_ess(self._inference_data)
            self._copy_arviz_figure(plt.gcf())
        except ImportError:
            self._plot_fallback("ArviZ not installed")

    def _plot_autocorr(self) -> None:
        """Generate autocorrelation plot."""
        try:
            import arviz as az
            import matplotlib.pyplot as plt

            plt.close("all")
            az.plot_autocorr(self._inference_data)
            self._copy_arviz_figure(plt.gcf())
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
