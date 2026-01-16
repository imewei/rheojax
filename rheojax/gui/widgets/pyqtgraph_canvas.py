"""PyQtGraph Canvas Widget for GPU-Accelerated Plotting.

This module provides GPU-accelerated real-time plotting using PyQtGraph,
complementing the matplotlib-based PlotCanvas for different use cases:

- PyQtGraphCanvas: Real-time streaming, large datasets (100k+ points), interactive exploration
- PlotCanvas (matplotlib): Publication-quality exports, complex annotations, ArviZ integration

Following Technical Guidelines §6.2 for GPU-accelerated visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph import PlotWidget, mkBrush, mkPen

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

from rheojax.gui.compat import QComboBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget
from rheojax.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PyQtGraphCanvas(QWidget):
    """GPU-accelerated interactive plot canvas using PyQtGraph.

    Optimized for:
        - Real-time streaming data visualization
        - Large datasets (100k+ points) with smooth pan/zoom
        - Interactive exploration with crosshairs and data inspection
        - Log-scale rheological data (storage/loss moduli, viscosity)

    Features:
        - OpenGL-accelerated rendering
        - Mouse wheel zoom (centered on cursor)
        - Click-drag pan
        - Crosshair with coordinate display
        - Automatic log-scale detection
        - Multiple data series with legend
        - Export to image/clipboard

    Example
    -------
    >>> canvas = PyQtGraphCanvas()  # doctest: +SKIP
    >>> canvas.plot_data(omega, G_storage, name='G\\'', color='blue')  # doctest: +SKIP
    >>> canvas.plot_data(omega, G_loss, name='G\\'\\'', color='red')  # doctest: +SKIP
    >>> canvas.set_log_scale(x=True, y=True)  # doctest: +SKIP
    """

    # Default colors for data series (rheology-friendly palette)
    COLORS = [
        "#1f77b4",  # Blue - typically G' (storage modulus)
        "#d62728",  # Red - typically G'' (loss modulus)
        "#2ca02c",  # Green - fits/predictions
        "#ff7f0e",  # Orange - secondary data
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the PyQtGraph canvas.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget

        Raises
        ------
        ImportError
            If pyqtgraph is not installed
        """
        super().__init__(parent)

        if not PYQTGRAPH_AVAILABLE:
            raise ImportError(
                "pyqtgraph is required for PyQtGraphCanvas. "
                "Install with: pip install pyqtgraph"
            )

        logger.debug("Initializing", class_name=self.__class__.__name__)

        # Configure pyqtgraph for scientific plotting
        pg.setConfigOptions(
            antialias=True,
            useOpenGL=True,  # GPU acceleration
            enableExperimental=True,
            background="w",  # White background
            foreground="k",  # Black foreground
        )

        # Create plot widget
        self._plot_widget = PlotWidget()
        self._plot_item = self._plot_widget.getPlotItem()

        # Enable grid
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)

        # Setup axes
        self._plot_item.setLabel("left", "Y")
        self._plot_item.setLabel("bottom", "X")

        # Add legend
        self._legend = self._plot_item.addLegend()

        # Track plot items for management
        self._data_items: list[pg.PlotDataItem] = []
        self._color_index = 0

        # Setup crosshair
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("g", width=1))
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("g", width=1))
        self._plot_item.addItem(self._crosshair_v, ignoreBounds=True)
        self._plot_item.addItem(self._crosshair_h, ignoreBounds=True)
        self._crosshair_v.hide()
        self._crosshair_h.hide()

        # Coordinate label
        self._coord_label = pg.TextItem(anchor=(0, 1))
        self._plot_item.addItem(self._coord_label, ignoreBounds=True)
        self._coord_label.hide()

        # Connect mouse events
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Create toolbar
        toolbar = self._create_toolbar()

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(toolbar)
        layout.addWidget(self._plot_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        logger.debug("PyQtGraph canvas initialized")

    def _create_toolbar(self) -> QWidget:
        """Create toolbar with scale and export controls."""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scale selector
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(["Log-Log", "Lin-Lin", "Log-Lin", "Lin-Log"])
        self._scale_combo.currentTextChanged.connect(self._on_scale_changed)
        layout.addWidget(self._scale_combo)

        # Auto-range button
        auto_btn = QPushButton("Auto Range")
        auto_btn.clicked.connect(self._plot_widget.autoRange)
        layout.addWidget(auto_btn)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        layout.addWidget(clear_btn)

        # Export button
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._on_export)
        layout.addWidget(export_btn)

        layout.addStretch()
        return toolbar

    def _on_scale_changed(self, scale_text: str) -> None:
        """Handle scale selection change."""
        log_x = "Log" in scale_text.split("-")[0]
        log_y = "Log" in scale_text.split("-")[1]
        self.set_log_scale(x=log_x, y=log_y)

    def set_log_scale(self, x: bool = True, y: bool = True) -> None:
        """Set logarithmic scale for axes.

        Parameters
        ----------
        x : bool
            Use log scale for X axis
        y : bool
            Use log scale for Y axis
        """
        self._plot_item.setLogMode(x=x, y=y)
        logger.debug("Scale set", log_x=x, log_y=y)

    def plot_data(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        name: str = "Data",
        color: str | None = None,
        symbol: str = "o",
        symbol_size: int = 6,
        line_width: int = 0,
    ) -> pg.PlotDataItem:
        """Plot data points.

        Parameters
        ----------
        x : NDArray
            X values
        y : NDArray
            Y values
        name : str
            Series name for legend
        color : str, optional
            Color (hex or name). Auto-assigned if None.
        symbol : str
            Point symbol ('o', 's', 't', 'd', '+', etc.)
        symbol_size : int
            Symbol size in pixels
        line_width : int
            Line width (0 for scatter only)

        Returns
        -------
        PlotDataItem
            The created plot item
        """
        if color is None:
            color = self.COLORS[self._color_index % len(self.COLORS)]
            self._color_index += 1

        pen = mkPen(color, width=line_width) if line_width > 0 else None
        brush = mkBrush(color)

        item = self._plot_widget.plot(
            x,
            y,
            name=name,
            pen=pen,
            symbol=symbol,
            symbolSize=symbol_size,
            symbolPen=mkPen(color),
            symbolBrush=brush,
        )
        self._data_items.append(item)

        logger.debug(
            "Data plotted",
            name=name,
            n_points=len(x),
            color=color,
        )
        return item

    def plot_line(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        name: str = "Fit",
        color: str | None = None,
        line_width: int = 2,
        line_style: str = "solid",
    ) -> pg.PlotDataItem:
        """Plot a line (typically for fits/predictions).

        Parameters
        ----------
        x : NDArray
            X values
        y : NDArray
            Y values
        name : str
            Series name for legend
        color : str, optional
            Color (hex or name). Auto-assigned if None.
        line_width : int
            Line width in pixels
        line_style : str
            Line style: 'solid', 'dash', 'dot', 'dashdot'

        Returns
        -------
        PlotDataItem
            The created plot item
        """
        if color is None:
            color = self.COLORS[self._color_index % len(self.COLORS)]
            self._color_index += 1

        # Map line style to Qt pen style
        style_map = {
            "solid": 1,  # Qt.SolidLine
            "dash": 2,  # Qt.DashLine
            "dot": 3,  # Qt.DotLine
            "dashdot": 4,  # Qt.DashDotLine
        }
        style = style_map.get(line_style, 1)

        pen = mkPen(color, width=line_width, style=style)
        item = self._plot_widget.plot(x, y, name=name, pen=pen)
        self._data_items.append(item)

        logger.debug(
            "Line plotted",
            name=name,
            n_points=len(x),
            color=color,
        )
        return item

    def clear(self) -> None:
        """Clear all plot data."""
        for item in self._data_items:
            self._plot_widget.removeItem(item)
        self._data_items.clear()
        self._color_index = 0
        self._legend.clear()
        logger.debug("Plot cleared")

    def set_labels(
        self,
        x_label: str | None = None,
        y_label: str | None = None,
        title: str | None = None,
    ) -> None:
        """Set axis labels and title.

        Parameters
        ----------
        x_label : str, optional
            X axis label
        y_label : str, optional
            Y axis label
        title : str, optional
            Plot title
        """
        if x_label:
            self._plot_item.setLabel("bottom", x_label)
        if y_label:
            self._plot_item.setLabel("left", y_label)
        if title:
            self._plot_item.setTitle(title)

    def _on_mouse_moved(self, pos) -> None:
        """Handle mouse movement for crosshair and coordinates."""
        if self._plot_item.sceneBoundingRect().contains(pos):
            mouse_point = self._plot_item.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Update crosshair
            self._crosshair_v.setPos(x)
            self._crosshair_h.setPos(y)
            self._crosshair_v.show()
            self._crosshair_h.show()

            # Update coordinate label
            self._coord_label.setText(f"x={x:.4g}, y={y:.4g}")
            self._coord_label.setPos(x, y)
            self._coord_label.show()
        else:
            self._crosshair_v.hide()
            self._crosshair_h.hide()
            self._coord_label.hide()

    def _on_export(self) -> None:
        """Export plot to file."""
        from rheojax.gui.compat import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Image (*.png);;SVG Vector (*.svg);;All Files (*)",
        )
        if file_path:
            exporter = pg.exporters.ImageExporter(self._plot_item)
            exporter.export(file_path)
            logger.info("Plot exported", path=file_path)

    def get_plot_widget(self) -> PlotWidget:
        """Get the underlying PlotWidget for advanced customization."""
        return self._plot_widget


def is_pyqtgraph_available() -> bool:
    """Check if PyQtGraph is available.

    Returns
    -------
    bool
        True if pyqtgraph can be imported
    """
    return PYQTGRAPH_AVAILABLE


__all__ = ["PyQtGraphCanvas", "is_pyqtgraph_available", "PYQTGRAPH_AVAILABLE"]
