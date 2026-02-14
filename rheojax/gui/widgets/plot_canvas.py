"""
Plot Canvas Widget
=================

Matplotlib canvas with interactive controls.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from rheojax.gui.compat import QVBoxLayout, QWidget
from rheojax.logging import get_logger

logger = get_logger(__name__)


class PlotCanvas(QWidget):
    """Interactive matplotlib canvas with log-scale aware zoom and tooltips.

    Features:
        - Zoom, pan, reset controls
        - Log-log, lin-lin, log-lin, lin-log scales
        - Mouse wheel zoom (log-scale aware)
        - Click-drag pan
        - Data point hover tooltips
        - Export to image

    Example
    -------
    >>> canvas = PlotCanvas()  # doctest: +SKIP
    >>> canvas.plot_data(x, y, label='Data')  # doctest: +SKIP
    >>> canvas.plot_fit(x_fit, y_fit, label='Fit')  # doctest: +SKIP
    >>> canvas.set_scale('log', 'log')  # doctest: +SKIP
    >>> canvas.add_legend()  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the plot canvas.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        logger.debug("Initializing", class_name=self.__class__.__name__)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        # Create navigation toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)

        # Track plot data for tooltips
        self._plot_data: list[tuple[np.ndarray, np.ndarray, str]] = []

        # Connect mouse events
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)

        # Pan state
        self._panning = False
        self._pan_start = None

        # Annotation for tooltips
        self._annotation = None

        logger.debug(
            "Initialization complete",
            class_name=self.__class__.__name__,
            figure_size=(8, 6),
            dpi=100,
        )

    def get_axes(self):
        """Return the primary matplotlib Axes for compatibility."""
        return self.axes

    def refresh(self) -> None:
        """Redraw the canvas (compat helper)."""
        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def plot_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str = "",
        color: str | None = None,
        marker: str = "o",
    ) -> None:
        """Plot data points.

        Parameters
        ----------
        x : np.ndarray
            X data
        y : np.ndarray
            Y data
        label : str, optional
            Data label for legend
        color : str, optional
            Line color
        marker : str, optional
            Marker style (default: 'o')
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="plot_data",
            label=label,
            data_size=len(x) if hasattr(x, "__len__") else 0,
        )

        self.axes.plot(
            x,
            y,
            marker=marker,
            linestyle="",
            label=label,
            color=color,
            markersize=6,
            alpha=0.7,
        )

        # Store data for tooltips
        self._plot_data.append((x, y, label))

        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def plot_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str = "",
        color: str | None = None,
        linestyle: str = "-",
    ) -> None:
        """Plot fitted curve.

        Parameters
        ----------
        x : np.ndarray
            X data
        y : np.ndarray
            Y data
        label : str, optional
            Line label for legend
        color : str, optional
            Line color
        linestyle : str, optional
            Line style (default: '-')
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="plot_fit",
            label=label,
            data_size=len(x) if hasattr(x, "__len__") else 0,
        )

        self.axes.plot(
            x, y, linestyle=linestyle, label=label, color=color, linewidth=2, alpha=0.9
        )

        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def plot_confidence_band(
        self,
        x: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        color: str | None = None,
        alpha: float = 0.3,
    ) -> None:
        """Plot confidence band.

        Parameters
        ----------
        x : np.ndarray
            X data
        y_lower : np.ndarray
            Lower bound
        y_upper : np.ndarray
            Upper bound
        color : str, optional
            Fill color
        alpha : float, optional
            Transparency (default: 0.3)
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="plot_confidence_band",
            data_size=len(x) if hasattr(x, "__len__") else 0,
        )

        self.axes.fill_between(
            x, y_lower, y_upper, color=color, alpha=alpha, linewidth=0
        )

        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def clear(self) -> None:
        """Clear all plot content."""
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="clear",
        )

        self.axes.clear()
        self._plot_data.clear()
        self._annotation = None

        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def set_labels(self, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
        """Set axis labels and title.

        Parameters
        ----------
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        title : str, optional
            Plot title
        """
        if xlabel:
            self.axes.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            self.axes.set_ylabel(ylabel, fontsize=11)
        if title:
            self.axes.set_title(title, fontsize=12, fontweight="bold")

        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def set_scale(self, xscale: str = "linear", yscale: str = "linear") -> None:
        """Set axis scales.

        Parameters
        ----------
        xscale : str, optional
            X-axis scale ('linear' or 'log')
        yscale : str, optional
            Y-axis scale ('linear' or 'log')
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="set_scale",
            xscale=xscale,
            yscale=yscale,
        )

        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)

        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def add_legend(self) -> None:
        """Add legend to plot."""
        handles, labels = self.axes.get_legend_handles_labels()
        if handles:
            self.axes.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)

        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def save_figure(self, path: str, dpi: int = 300, format: str | None = None) -> None:
        """Save figure to file.

        Parameters
        ----------
        path : str
            Output file path
        dpi : int, optional
            Resolution in dots per inch (default: 300)
        format : str, optional
            File format (inferred from path if not specified)
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="save_figure",
            path=path,
            dpi=dpi,
        )

        try:
            self.figure.savefig(path, dpi=dpi, format=format, bbox_inches="tight")
            logger.debug(
                "Figure saved",
                widget=self.__class__.__name__,
                path=path,
            )
        except Exception:
            logger.error(
                "Failed to save figure",
                widget=self.__class__.__name__,
                path=path,
                exc_info=True,
            )
            raise

    def _on_scroll(self, event) -> None:
        """Handle mouse wheel zoom (log-scale aware)."""
        if event.inaxes != self.axes:
            return

        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="scroll_zoom",
            direction="up" if event.button == "up" else "down",
        )

        # Zoom factor
        zoom_factor = 1.2 if event.button == "up" else 1 / 1.2

        # Get current limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        # Get mouse position in data coordinates
        xdata = event.xdata
        ydata = event.ydata

        # Calculate new limits based on scale type
        if self.axes.get_xscale() == "log":
            # Log scale: zoom in log space
            log_xlim = np.log10(xlim)
            log_xdata = np.log10(xdata)
            new_log_xlim = [
                log_xdata - (log_xdata - log_xlim[0]) / zoom_factor,
                log_xdata + (log_xlim[1] - log_xdata) / zoom_factor,
            ]
            new_xlim = 10 ** np.array(new_log_xlim)
        else:
            # Linear scale
            new_xlim = [
                xdata - (xdata - xlim[0]) / zoom_factor,
                xdata + (xlim[1] - xdata) / zoom_factor,
            ]

        if self.axes.get_yscale() == "log":
            log_ylim = np.log10(ylim)
            log_ydata = np.log10(ydata)
            new_log_ylim = [
                log_ydata - (log_ydata - log_ylim[0]) / zoom_factor,
                log_ydata + (log_ylim[1] - log_ydata) / zoom_factor,
            ]
            new_ylim = 10 ** np.array(new_log_ylim)
        else:
            new_ylim = [
                ydata - (ydata - ylim[0]) / zoom_factor,
                ydata + (ylim[1] - ydata) / zoom_factor,
            ]

        self.axes.set_xlim(new_xlim)
        self.axes.set_ylim(new_ylim)

        logger.debug("Rendering", widget=self.__class__.__name__)
        self.canvas.draw_idle()

    def _on_mouse_press(self, event) -> None:
        """Handle mouse button press for panning."""
        if event.inaxes != self.axes:
            return

        if event.button == 1:  # Left click
            logger.debug(
                "User interaction",
                widget=self.__class__.__name__,
                action="pan_start",
            )
            self._panning = True
            self._pan_start = (event.xdata, event.ydata)

    def _on_mouse_release(self, event) -> None:
        """Handle mouse button release."""
        if self._panning:
            logger.debug(
                "User interaction",
                widget=self.__class__.__name__,
                action="pan_end",
            )
        self._panning = False
        self._pan_start = None

    def _on_mouse_move(self, event) -> None:
        """Handle mouse movement for panning and tooltips."""
        if event.inaxes != self.axes:
            # Hide tooltip when mouse leaves axes
            if self._annotation is not None:
                self._annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        # Handle panning
        if self._panning and self._pan_start is not None:
            if self.axes.get_xscale() == "log":
                # Log scale panning
                factor_x = event.xdata / self._pan_start[0]
                xlim = self.axes.get_xlim()
                new_xlim = [xlim[0] / factor_x, xlim[1] / factor_x]
            else:
                dx = event.xdata - self._pan_start[0]
                xlim = self.axes.get_xlim()
                new_xlim = [xlim[0] - dx, xlim[1] - dx]

            if self.axes.get_yscale() == "log":
                factor_y = event.ydata / self._pan_start[1]
                ylim = self.axes.get_ylim()
                new_ylim = [ylim[0] / factor_y, ylim[1] / factor_y]
            else:
                dy = event.ydata - self._pan_start[1]
                ylim = self.axes.get_ylim()
                new_ylim = [ylim[0] - dy, ylim[1] - dy]

            self.axes.set_xlim(new_xlim)
            self.axes.set_ylim(new_ylim)
            self.canvas.draw_idle()
            return

        # Show tooltip for nearby data points
        self._show_tooltip(event.xdata, event.ydata)

    def _show_tooltip(self, x: float, y: float) -> None:
        """Show tooltip for nearby data points."""
        if not self._plot_data:
            return

        # Find nearest point
        min_dist = float("inf")
        nearest_point = None
        nearest_label = ""

        for data_x, data_y, label in self._plot_data:
            # Calculate distance in display coordinates
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()
            x_range = (
                (np.log10(xlim[1]) - np.log10(xlim[0]))
                if self.axes.get_xscale() == "log"
                else (xlim[1] - xlim[0])
            )
            y_range = (
                (np.log10(ylim[1]) - np.log10(ylim[0]))
                if self.axes.get_yscale() == "log"
                else (ylim[1] - ylim[0])
            )
            if abs(x_range) < 1e-30 or abs(y_range) < 1e-30:
                continue

            for i in range(len(data_x)):
                # Convert to display coordinates for distance calculation
                if self.axes.get_xscale() == "log":
                    dx = (np.log10(x) - np.log10(data_x[i])) / x_range
                else:
                    dx = (x - data_x[i]) / x_range

                if self.axes.get_yscale() == "log":
                    dy = (np.log10(y) - np.log10(data_y[i])) / y_range
                else:
                    dy = (y - data_y[i]) / y_range

                dist = np.sqrt(dx**2 + dy**2)

                if dist < min_dist:
                    min_dist = dist
                    nearest_point = (data_x[i], data_y[i])
                    nearest_label = label

        # Show tooltip if point is close enough (within 5% of plot range)
        if min_dist < 0.05:
            if self._annotation is None:
                self._annotation = self.axes.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox={"boxstyle": "round,pad=0.5", "fc": "yellow", "alpha": 0.9},
                    arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0"},
                )

            self._annotation.xy = nearest_point
            text = (
                f"{nearest_label}\nx={nearest_point[0]:.4g}\ny={nearest_point[1]:.4g}"
            )
            self._annotation.set_text(text)
            self._annotation.set_visible(True)
            self.canvas.draw_idle()
        else:
            if self._annotation is not None:
                self._annotation.set_visible(False)
                self.canvas.draw_idle()
