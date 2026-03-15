"""
ArviZ Canvas Widget
==================

ArviZ diagnostic plot integration for Bayesian inference visualization.
"""

from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from rheojax.gui.compat import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    Signal,
)
from rheojax.gui.resources.styles.tokens import Spacing
from rheojax.gui.utils.layout_helpers import set_toolbar_margins
from rheojax.gui.widgets.base_arviz_widget import BaseArviZWidget
from rheojax.logging import get_logger

logger = get_logger(__name__)


def _filter_degenerate_vars(idata: Any) -> list[str] | None:
    """Filter out degenerate parameters (range < 1e-10) to prevent ArviZ KDE crashes.

    Returns filtered var_names list, or None if no posterior data available.
    """
    import numpy as np

    if not hasattr(idata, "posterior") or idata.posterior is None:
        return None
    var_names = list(idata.posterior.data_vars)
    filtered = []
    for v in var_names:
        vals = idata.posterior[v].values.ravel()
        if np.ptp(vals) > 1e-10:
            filtered.append(v)
        else:
            logger.debug("Skipping degenerate parameter in ArviZ plot", param=v)
    return filtered if filtered else None


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

    Example
    -------
    >>> canvas = ArvizCanvas()  # doctest: +SKIP
    >>> canvas.plot_trace(inference_data)  # doctest: +SKIP
    >>> canvas.set_plot_type("pair")  # doctest: +SKIP
    """

    plot_changed = Signal(str)

    def __init__(self, parent: BaseArviZWidget | None = None) -> None:
        """Initialize ArviZ canvas.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        logger.debug("Initializing", class_name=self.__class__.__name__)

        self._inference_data: Any | None = None
        self._current_plot_type = "trace"
        self._hdi_prob = 0.94

        self._setup_ui()
        self._connect_signals()

        logger.debug(
            "Initialization complete",
            class_name=self.__class__.__name__,
            num_plot_types=len(PLOT_TYPES),
        )

    def _setup_ui(self) -> None:
        """Set up the user interface.

        The toolbar stays fixed at the top.  The matplotlib canvas is
        wrapped in a QScrollArea so that tall ArviZ plots (e.g. trace
        plots with many parameters) display at a readable size and the
        user can scroll vertically instead of having everything squashed.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)

        # --- Fixed header: Plot Type selector + Refresh/Export buttons ---
        toolbar_layout = QHBoxLayout()
        set_toolbar_margins(toolbar_layout)

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

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Regenerate current plot")
        toolbar_layout.addWidget(self._refresh_btn)

        layout.addLayout(toolbar_layout)

        # --- Figure canvas inside a scroll area ---
        self._figure = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._nav_toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._nav_toolbar)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setWidget(self._canvas)
        layout.addWidget(self._scroll_area, 1)

        # Status / empty label
        self._status_label = QLabel(
            "No diagnostics yet. Run Bayesian inference to view plots."
        )
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: #94A3B8; padding: 6px;")
        layout.addWidget(self._status_label)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._refresh_btn.clicked.connect(self._refresh_plot)

    def _fit_figure_to_canvas(self) -> None:
        """Scale the figure to the viewport width, preserving aspect ratio.

        Instead of squashing tall ArviZ figures into the viewport, this
        method sets the figure width to match the scroll-area viewport
        and keeps the natural height (scaled proportionally).  The canvas
        widget is then resized to match so that the QScrollArea provides
        vertical scrolling for tall plots while short plots fill the
        viewport without scrollbars.
        """
        if self._figure is None or self._canvas is None:
            return

        viewport_w = self._scroll_area.viewport().width()
        viewport_h = self._scroll_area.viewport().height()
        if viewport_w <= 0 or viewport_h <= 0:
            return

        dpi = self._figure.get_dpi()
        fig_w, fig_h = self._figure.get_size_inches()

        # Scale figure width to viewport; scale height proportionally
        new_w = viewport_w / dpi
        scale = new_w / fig_w if fig_w > 0 else 1.0
        new_h = fig_h * scale

        # Ensure the figure is at least as tall as the viewport so that
        # short plots (e.g., forest, energy) fill the available space.
        min_h = viewport_h / dpi
        new_h = max(new_h, min_h)

        self._figure.set_size_inches(new_w, new_h, forward=False)

        try:
            self._figure.tight_layout(pad=0.5)
        except Exception:
            # tight_layout can fail with some axis configurations
            # (e.g., colorbars, inset axes) — safe to skip
            pass

        # Resize canvas widget to match figure so scroll area shows
        # scrollbars only when needed.
        canvas_w = int(new_w * dpi)
        canvas_h = int(new_h * dpi)
        self._canvas.setMinimumSize(canvas_w, canvas_h)
        self._canvas.resize(canvas_w, canvas_h)

    def _on_type_changed(self, index: int) -> None:
        """Handle plot type change.

        Parameters
        ----------
        index : int
            New combo box index
        """
        plot_type = self._type_combo.currentData()
        old_plot_type = self._current_plot_type
        self._current_plot_type = plot_type

        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="plot_type_changed",
            plot_type=plot_type,
        )

        logger.info(
            "Plot type changed",
            widget=self.__class__.__name__,
            old_type=old_plot_type,
            new_type=plot_type,
        )

        self.plot_changed.emit(plot_type)

        if self._inference_data is not None:
            self._refresh_plot()

    def _refresh_plot(self) -> None:
        """Refresh the current plot with performance tracking."""
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="refresh_plot",
            plot_type=self._current_plot_type,
        )

        if self._inference_data is None:
            self._status_label.show()
            return

        self._status_label.hide()

        # Generate plot based on type.
        # Each plot function calls _arviz_plot → _copy_arviz_figure which:
        #   1. Sanitizes text in the new ArviZ figure
        #   2. swap_figure() — transfers figure to canvas
        #   3. _fit_figure_to_canvas() — resizes figure to viewport
        #   4. draw_idle() — single deferred draw
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
                self.timed_render(self._current_plot_type, func)
                self._status_label.hide()
            except Exception as e:
                logger.error(
                    "Plot generation failed",
                    widget=self.__class__.__name__,
                    plot_type=self._current_plot_type,
                    exc_info=True,
                )
                self._status_label.setText(f"Error: {e}")
                self._status_label.show()
                self._figure.clear()
                ax = self._figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    f"Error generating plot:\n{e}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                self._canvas.draw_idle()
        else:
            self._status_label.setText("Unsupported plot type")
            self._status_label.show()

    def set_inference_data(self, idata: Any) -> None:
        """Set ArviZ InferenceData object.

        Parameters
        ----------
        idata : arviz.InferenceData
            Inference data from Bayesian fitting
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="set_inference_data",
            has_data=idata is not None,
        )

        self._inference_data = idata
        logger.info(
            "Inference data updated",
            widget=self.__class__.__name__,
            has_data=idata is not None,
        )
        # Disable plot types that are invalid for current data
        self._update_plot_type_availability()
        self._refresh_plot()

    def _update_plot_type_availability(self) -> None:
        """Enable/disable plot types based on available data."""
        if self._inference_data is None:
            return

        try:
            # Count number of parameters in posterior
            posterior = getattr(self._inference_data, "posterior", None)
            if posterior is not None:
                n_params = len(list(posterior.data_vars))
            else:
                n_params = 0
        except Exception:
            n_params = 0

        for idx in range(self._type_combo.count()):
            plot_id = self._type_combo.itemData(idx)
            # Pair plot requires at least 2 parameters
            enabled = True
            if plot_id == "pair" and n_params < 2:
                enabled = False
            # Energy plot requires sample_stats
            if plot_id == "energy" and not self._has_sample_stats_energy():
                enabled = False

            # Use Qt item flags to enable/disable
            model = self._type_combo.model()
            item = model.item(idx)
            if item is not None:
                if enabled:
                    item.setEnabled(True)
                else:
                    item.setEnabled(False)

    def set_hdi_prob(self, prob: float) -> None:
        """Set HDI probability for credible intervals.

        Parameters
        ----------
        prob : float
            HDI probability (0-1), e.g., 0.94 for 94% HDI
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="set_hdi_prob",
            prob=prob,
        )

        old_prob = self._hdi_prob
        self._hdi_prob = prob
        logger.info(
            "HDI probability changed",
            widget=self.__class__.__name__,
            old_prob=old_prob,
            new_prob=prob,
        )
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
        protocol for thread-safe figure swapping with proper cleanup, then
        resizes the figure to fit the canvas viewport so plots are never
        larger than the window.

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

        # Resize figure to fit the canvas viewport so the plot is never
        # larger than the window.  tight_layout is re-run to prevent
        # axis overlap at the new (potentially smaller) figsize.
        self._fit_figure_to_canvas()
        self._canvas.draw_idle()

        logger.debug(
            "Rendering",
            widget=self.__class__.__name__,
            action="arviz_figure_swapped",
            num_axes=len(arviz_fig.get_axes()),
            plot_type=self._current_plot_type,
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
        # GUI-R6-008: Use getattr for private _suptitle (may not exist in all matplotlib versions)
        suptitle_obj = getattr(fig, "_suptitle", None)
        if suptitle_obj is not None:
            suptitle_text = suptitle_obj.get_text()
            if suptitle_text and "\t" in suptitle_text:
                fig.suptitle(suptitle_text.replace("\t", "  "))

    @staticmethod
    def _close_new_figures(pre_fignums: set[int]) -> None:
        """Close only figures created since *pre_fignums* was captured."""
        import matplotlib.pyplot as plt

        for num in list(plt.get_fignums()):
            if num not in pre_fignums:
                plt.close(num)

    def _arviz_plot(self, plot_fn, *args, **kwargs) -> None:
        """Run an ArviZ plot function, copy its figure, and clean up only new figures.

        Extracts the figure from the ArviZ return value instead of relying on
        plt.gcf(), which is non-deterministic when multiple ArviZ calls run
        concurrently and may return the wrong figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        pre = set(plt.get_fignums())
        result = plot_fn(*args, **kwargs)

        # Extract figure from ArviZ return value rather than using plt.gcf()
        fig = None
        if hasattr(result, "figure"):
            fig = result.figure
        elif isinstance(result, np.ndarray) and result.size > 0:
            ax_item = result.ravel()[0]
            if hasattr(ax_item, "figure"):
                fig = ax_item.figure
        elif isinstance(result, list) and result:
            # GUI-R6-007: Some ArviZ versions return list-of-axes
            first = result[0]
            if isinstance(first, np.ndarray) and first.size > 0:
                fig = first.ravel()[0].figure
            elif hasattr(first, "figure"):
                fig = first.figure

        if fig is None:
            logger.error(
                "ArviZ plot function returned unrecognized result type: %s. "
                "Cannot extract figure safely under concurrency.",
                type(result).__name__,
            )
            raise RuntimeError(
                f"ArviZ plot function returned unrecognized result type: {type(result).__name__}. "
                "Cannot extract figure safely under concurrency."
            )

        self._copy_arviz_figure(fig)
        self._close_new_figures(pre)

    def _plot_trace(self) -> None:
        """Generate trace plot."""
        try:
            import arviz as az

            var_names = _filter_degenerate_vars(self._inference_data)
            if var_names is None:
                self._plot_fallback("No non-degenerate parameters to plot")
                return
            self._arviz_plot(
                az.plot_trace, self._inference_data, var_names=var_names
            )
        except ImportError:
            logger.error(
                "ArviZ import failed",
                widget=self.__class__.__name__,
                exc_info=True,
            )
            self._plot_fallback("ArviZ not installed")

    def _plot_pair(self) -> None:
        """Generate pair plot."""
        try:
            import arviz as az

            var_names = _filter_degenerate_vars(self._inference_data)
            if var_names is None:
                self._plot_fallback("No non-degenerate parameters to plot")
                return
            has_divergences = (
                self._inference_data is not None
                and hasattr(self._inference_data, "sample_stats")
                and self._inference_data.sample_stats is not None
                and "diverging" in self._inference_data.sample_stats
            )

            self._arviz_plot(
                az.plot_pair,
                self._inference_data,
                var_names=var_names,
                divergences=has_divergences,
            )
        except ImportError:
            logger.error(
                "ArviZ import failed",
                widget=self.__class__.__name__,
                exc_info=True,
            )
            self._plot_fallback("ArviZ not installed")

    def _plot_forest(self) -> None:
        """Generate forest plot."""
        try:
            import arviz as az

            var_names = _filter_degenerate_vars(self._inference_data)
            if var_names is None:
                self._plot_fallback("No non-degenerate parameters to plot")
                return
            self._arviz_plot(
                az.plot_forest,
                self._inference_data,
                var_names=var_names,
                hdi_prob=self._hdi_prob,
                combined=True,
            )
        except ImportError:
            logger.error(
                "ArviZ import failed",
                widget=self.__class__.__name__,
                exc_info=True,
            )
            self._plot_fallback("ArviZ not installed")

    def _plot_posterior(self) -> None:
        """Generate posterior plot."""
        try:
            import arviz as az

            var_names = _filter_degenerate_vars(self._inference_data)
            if var_names is None:
                self._plot_fallback("No non-degenerate parameters to plot")
                return
            self._arviz_plot(
                az.plot_posterior,
                self._inference_data,
                var_names=var_names,
                hdi_prob=self._hdi_prob,
            )
        except ImportError:
            logger.error(
                "ArviZ import failed",
                widget=self.__class__.__name__,
                exc_info=True,
            )
            self._plot_fallback("ArviZ not installed")

    def _plot_energy(self) -> None:
        """Generate energy plot.

        Requires sample_stats group with energy diagnostic from MCMC sampling.
        Shows fallback message if sample_stats is not available.
        """
        try:
            import arviz as az

            if not self._has_sample_stats_energy():
                self._plot_fallback(
                    "Energy plot requires MCMC sample statistics.\n\n"
                    "The current InferenceData only contains posterior samples.\n"
                    "To view energy diagnostics, ensure the Bayesian result\n"
                    "was created with full MCMC diagnostics (via fit_bayesian)."
                )
                return

            self._arviz_plot(az.plot_energy, self._inference_data)
        except ImportError:
            logger.error(
                "ArviZ import failed",
                widget=self.__class__.__name__,
                exc_info=True,
            )
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

            self._arviz_plot(az.plot_rank, self._inference_data)
        except ImportError:
            logger.error(
                "ArviZ import failed",
                widget=self.__class__.__name__,
                exc_info=True,
            )
            self._plot_fallback("ArviZ not installed")

    def _plot_ess(self) -> None:
        """Generate ESS plot."""
        try:
            import arviz as az

            self._arviz_plot(az.plot_ess, self._inference_data)
        except ImportError:
            logger.error(
                "ArviZ import failed",
                widget=self.__class__.__name__,
                exc_info=True,
            )
            self._plot_fallback("ArviZ not installed")

    def _plot_autocorr(self) -> None:
        """Generate autocorrelation plot."""
        try:
            import arviz as az

            self._arviz_plot(az.plot_autocorr, self._inference_data)
        except ImportError:
            logger.error(
                "ArviZ import failed",
                widget=self.__class__.__name__,
                exc_info=True,
            )
            self._plot_fallback("ArviZ not installed")

    def _plot_fallback(self, message: str) -> None:
        """Display fallback message when plotting fails.

        Parameters
        ----------
        message : str
            Message to display
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        self._sanitize_figure_text(self._figure)
        self._canvas.draw_idle()

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
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="export_figure",
            filepath=filepath,
            dpi=dpi,
        )

        try:
            self._figure.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.info(
                "Figure exported",
                widget=self.__class__.__name__,
                filepath=filepath,
                dpi=dpi,
                plot_type=self._current_plot_type,
            )
        except Exception:
            logger.error(
                "Failed to export figure",
                widget=self.__class__.__name__,
                filepath=filepath,
                exc_info=True,
            )
            raise

    def clear(self) -> None:
        """Clear the canvas."""
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="clear",
        )

        self._figure.clear()
        self._inference_data = None
        self._status_label.setText("No data loaded")
        if hasattr(self, "_nav_toolbar") and self._nav_toolbar is not None:
            self._nav_toolbar.update()
        self._canvas.draw_idle()

        logger.info(
            "Canvas cleared",
            widget=self.__class__.__name__,
        )

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
