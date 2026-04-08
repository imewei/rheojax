"""
Plot Widget
===========

Unified plot backend abstraction wrapping either PlotCanvas (Matplotlib)
or PyQtGraphCanvas (PyQtGraph).  Pages select the backend at construction
time via the ``backend`` kwarg and interact only with the unified interface.
"""

from __future__ import annotations

from typing import Any

from rheojax.gui.compat import QVBoxLayout, QWidget
from rheojax.logging import get_logger

logger = get_logger(__name__)


class PlotWidget(QWidget):
    """Thin wrapper holding either a PlotCanvas or PyQtGraphCanvas.

    Exposes a unified interface so pages are decoupled from the concrete
    backend.  Only the operations common to both backends are exposed here;
    callers that need backend-specific functionality can access the
    underlying canvas via ``PlotWidget.canvas``.

    Parameters
    ----------
    backend : str
        ``"matplotlib"`` (default) or ``"pyqtgraph"``.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    canvas : PlotCanvas | PyQtGraphCanvas
        The underlying backend canvas widget.

    Example
    -------
    >>> pw = PlotWidget(backend="matplotlib")  # doctest: +SKIP
    >>> pw.plot(x, y_data, fit_result=result, style="default")  # doctest: +SKIP
    >>> pw.clear()  # doctest: +SKIP
    >>> pw.set_theme("dark")  # doctest: +SKIP
    """

    def __init__(
        self,
        backend: str = "matplotlib",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend.lower()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if self._backend == "pyqtgraph":
            from rheojax.gui.widgets.pyqtgraph_canvas import (
                PYQTGRAPH_AVAILABLE,
                PyQtGraphCanvas,
            )

            if not PYQTGRAPH_AVAILABLE:
                logger.warning(
                    "PyQtGraph not available — falling back to Matplotlib",
                    widget=self.__class__.__name__,
                )
                self._backend = "matplotlib"
                from rheojax.gui.widgets.plot_canvas import PlotCanvas

                self.canvas = PlotCanvas(parent=self)
            else:
                self.canvas = PyQtGraphCanvas(parent=self)
        else:
            from rheojax.gui.widgets.plot_canvas import PlotCanvas

            self.canvas = PlotCanvas(parent=self)

        layout.addWidget(self.canvas)
        logger.debug(
            "Initializing",
            class_name=self.__class__.__name__,
            backend=self._backend,
        )

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    def plot(
        self,
        data: Any,
        fit_result: Any = None,
        style: str = "default",
    ) -> None:
        """Plot data and optional fit result.

        Dispatches to the appropriate backend method.  ``data`` is expected
        to be an object with ``x_data`` and ``y_data`` attributes (e.g.
        ``DatasetState``), or a plain pair ``(x, y)`` tuple.

        Parameters
        ----------
        data : DatasetState or tuple[np.ndarray, np.ndarray]
            Data to plot.
        fit_result : FitResult, optional
            If provided, the fitted curve is overlaid.
        style : str
            Plot style hint (passed to backend where supported).
        """
        import numpy as np

        # Normalise data input
        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        else:
            x = getattr(data, "x_data", None)
            y = getattr(data, "y_data", None)

        if x is None or y is None:
            logger.warning("PlotWidget.plot: no data to display")
            return

        if self._backend == "matplotlib":
            self.canvas.clear()
            y_plot = np.real(y) if np.iscomplexobj(y) else y
            self.canvas.plot_data(np.asarray(x), np.asarray(y_plot), label="Data")

            if fit_result is not None:
                x_fit = getattr(fit_result, "x_fit", None)
                y_fit = getattr(fit_result, "y_fit", None)
                if x_fit is not None and y_fit is not None:
                    y_fit_plot = np.real(y_fit) if np.iscomplexobj(y_fit) else y_fit
                    self.canvas.plot_fit(
                        np.asarray(x_fit), np.asarray(y_fit_plot), label="Fit"
                    )

            self.canvas.add_legend()
        else:
            # PyQtGraphCanvas
            self.canvas.clear()
            y_plot = np.real(y) if np.iscomplexobj(y) else y
            self.canvas.plot_data(
                np.asarray(x, dtype=float), np.asarray(y_plot, dtype=float)
            )

            if fit_result is not None:
                x_fit = getattr(fit_result, "x_fit", None)
                y_fit = getattr(fit_result, "y_fit", None)
                if x_fit is not None and y_fit is not None:
                    y_fit_plot = np.real(y_fit) if np.iscomplexobj(y_fit) else y_fit
                    self.canvas.plot_fit(
                        np.asarray(x_fit, dtype=float),
                        np.asarray(y_fit_plot, dtype=float),
                    )

    def clear(self) -> None:
        """Clear all plot content."""
        self.canvas.clear()

    def set_theme(self, theme: str) -> None:
        """Update the plot theme.

        Parameters
        ----------
        theme : str
            ``"light"`` or ``"dark"``.
        """
        if self._backend == "matplotlib":
            # PlotCanvas does not have a set_theme — update the figure background
            # colour to match the requested theme.
            bg = "#1e1e1e" if theme == "dark" else "#ffffff"
            fg = "#e8eaed" if theme == "dark" else "#1a1a1a"
            try:
                self.canvas.figure.set_facecolor(bg)
                ax = self.canvas.axes
                ax.set_facecolor(bg)
                ax.tick_params(colors=fg)
                ax.xaxis.label.set_color(fg)
                ax.yaxis.label.set_color(fg)
                ax.title.set_color(fg)
                self.canvas.canvas.draw_idle()
            except Exception:
                logger.debug("Failed to apply matplotlib theme", exc_info=True)
        else:
            # PyQtGraphCanvas may expose set_theme / setBackground
            if hasattr(self.canvas, "set_theme"):
                self.canvas.set_theme(theme)
            elif hasattr(self.canvas, "setBackground"):
                bg = "#1e1e1e" if theme == "dark" else "#ffffff"
                self.canvas.setBackground(bg)

        logger.debug(
            "Theme applied",
            widget=self.__class__.__name__,
            backend=self._backend,
            theme=theme,
        )


__all__ = ["PlotWidget"]
