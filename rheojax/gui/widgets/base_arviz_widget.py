"""
Base ArviZ Widget
================

Base class for widgets that embed ArviZ/matplotlib figures with proper
figure swapping and cleanup protocols.
"""

import logging
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class PlotMetrics:
    """Track plot rendering performance metrics.

    This class collects timing data for plot rendering operations,
    enabling performance monitoring and optimization.

    Example
    -------
    >>> PlotMetrics.track("trace", 0.25)  # doctest: +SKIP
    >>> PlotMetrics.get_stats("trace")  # doctest: +SKIP
    {'count': 1, 'mean': 0.25, 'min': 0.25, 'max': 0.25}
    """

    render_times: dict[str, list[float]] = {}
    _max_samples: int = 100  # Keep last N samples per plot type

    @classmethod
    def track(cls, plot_type: str, duration: float) -> None:
        """Record a plot rendering duration.

        Parameters
        ----------
        plot_type : str
            Type of plot rendered (e.g., 'trace', 'pair', 'forest')
        duration : float
            Rendering time in seconds
        """
        if plot_type not in cls.render_times:
            cls.render_times[plot_type] = []

        times = cls.render_times[plot_type]
        times.append(duration)

        # Keep only recent samples to bound memory
        if len(times) > cls._max_samples:
            cls.render_times[plot_type] = times[-cls._max_samples :]

        logger.debug(
            "plot_render_metric",
            extra={
                "plot_type": plot_type,
                "duration_ms": round(duration * 1000, 2),
            },
        )

    @classmethod
    def get_stats(cls, plot_type: str) -> dict[str, float]:
        """Get statistics for a plot type.

        Parameters
        ----------
        plot_type : str
            Type of plot

        Returns
        -------
        dict
            Statistics including count, mean, min, max
        """
        times = cls.render_times.get(plot_type, [])
        if not times:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}

        return {
            "count": len(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }

    @classmethod
    def get_all_stats(cls) -> dict[str, dict[str, float]]:
        """Get statistics for all plot types.

        Returns
        -------
        dict
            Statistics per plot type
        """
        return {plot_type: cls.get_stats(plot_type) for plot_type in cls.render_times}

    @classmethod
    def reset(cls) -> None:
        """Clear all collected metrics."""
        cls.render_times.clear()


class BaseArviZWidget(QWidget):
    """Base class for widgets that embed ArviZ figures.

    Provides a standardized protocol for figure swapping with proper
    cleanup to prevent memory leaks and ensure thread-safe updates.

    Subclasses should set `_figure` and `_canvas` attributes.

    Attributes
    ----------
    _figure : Figure
        The current matplotlib figure
    _canvas : FigureCanvasQTAgg
        The Qt-embedded canvas for the figure
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize base widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._figure: Figure | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._pending_cleanup: list[Figure] = []

    def swap_figure(self, new_fig: Figure) -> None:
        """Thread-safe figure replacement with proper cleanup.

        This method handles the complex process of swapping matplotlib
        figures in a Qt environment, ensuring:
        - Old figure is properly cleaned up
        - Canvas reference is correctly transferred
        - Cleanup happens on the main thread

        Parameters
        ----------
        new_fig : Figure
            The new figure to display
        """
        if self._canvas is None:
            logger.warning("swap_figure called but _canvas is None")
            return

        old_fig = self._figure
        self._figure = new_fig

        # Transfer canvas reference
        self._canvas.figure = new_fig
        new_fig.set_canvas(self._canvas)

        # Schedule cleanup on main thread to avoid threading issues
        if old_fig is not None:
            # Store reference to prevent garbage collection before cleanup
            self._pending_cleanup.append(old_fig)
            QTimer.singleShot(0, lambda fig=old_fig: self._cleanup_figure(fig))

        logger.debug(
            "figure_swapped",
            extra={
                "widget": self.__class__.__name__,
                "new_fig_axes": len(new_fig.get_axes()),
            },
        )

    def _cleanup_figure(self, fig: Figure) -> None:
        """Clean up an old figure.

        Parameters
        ----------
        fig : Figure
            Figure to close
        """
        try:
            plt.close(fig)
            if fig in self._pending_cleanup:
                self._pending_cleanup.remove(fig)
        except Exception as e:
            logger.debug(f"Figure cleanup warning: {e}")

    def timed_render(self, plot_type: str, render_func: Any) -> Any:
        """Execute a render function with timing.

        Parameters
        ----------
        plot_type : str
            Type of plot being rendered
        render_func : callable
            Function to execute

        Returns
        -------
        Any
            Result of render_func
        """
        start = time.perf_counter()
        try:
            result = render_func()
            return result
        finally:
            duration = time.perf_counter() - start
            PlotMetrics.track(plot_type, duration)

    def refresh_canvas(self) -> None:
        """Redraw the canvas.

        Uses draw_idle() for thread-safe, non-blocking updates.
        """
        if self._canvas is not None:
            self._canvas.draw_idle()
