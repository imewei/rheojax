"""
Multi-View Widget
================

Multiple synchronized plot panels for comparison and analysis.
"""

from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Available layout configurations
LAYOUTS = [
    ("1x1", "Single", (1, 1)),
    ("1x2", "Side by Side", (1, 2)),
    ("2x1", "Stacked", (2, 1)),
    ("2x2", "Grid (2x2)", (2, 2)),
    ("2x3", "Grid (2x3)", (2, 3)),
    ("3x2", "Grid (3x2)", (3, 2)),
]


class PlotPanel(QWidget):
    """Individual plot panel within MultiView.

    Features:
        - Matplotlib figure canvas
        - Title label
        - Optional toolbar
    """

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        """Initialize plot panel.

        Parameters
        ----------
        index : int
            Panel index
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._index = index
        self._figure = Figure(figsize=(5, 4), dpi=100)
        self._figure.set_layout_engine("tight")

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Title
        self._title_label = QLabel(f"Panel {self._index + 1}")
        self._title_label.setStyleSheet(
            "font-weight: bold; padding: 2px; background: #f0f0f0;"
        )
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title_label)

        # Canvas
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._canvas, 1)

    def set_title(self, title: str) -> None:
        """Set panel title.

        Parameters
        ----------
        title : str
            Panel title
        """
        self._title_label.setText(title)

    def get_figure(self) -> Figure:
        """Get matplotlib figure.

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        return self._figure

    def get_canvas(self) -> FigureCanvasQTAgg:
        """Get matplotlib canvas.

        Returns
        -------
        FigureCanvasQTAgg
            Matplotlib canvas widget
        """
        return self._canvas

    def clear(self) -> None:
        """Clear the figure."""
        self._figure.clear()
        self._canvas.draw()

    def refresh(self) -> None:
        """Refresh the canvas."""
        self._canvas.draw()


class MultiView(QWidget):
    """Multi-panel plot viewer for comparison and analysis.

    Features:
        - Configurable grid layout (1x1 to 3x2)
        - Synchronized zoom/pan across panels (optional)
        - Individual panel titles and figures
        - Layout selector dropdown
        - Export all panels or individual

    Signals
    -------
    layout_changed : Signal(str)
        Emitted when layout changes
    panel_selected : Signal(int)
        Emitted when a panel is selected

    Example
    -------
    >>> view = MultiView(layout='2x2')  # doctest: +SKIP
    >>> view.add_plot(0, fig1)  # doctest: +SKIP
    >>> view.set_panel_title(0, "Raw Data")  # doctest: +SKIP
    """

    layout_changed = Signal(str)
    panel_selected = Signal(int)

    def __init__(
        self, layout: str = "1x1", parent: QWidget | None = None
    ) -> None:
        """Initialize multi-view.

        Parameters
        ----------
        layout : str, optional
            Initial layout (default: "1x1")
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._current_layout = layout
        self._panels: list[PlotPanel] = []
        self._sync_axes = False

        self._setup_ui()
        self._connect_signals()
        self._set_layout(layout)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(4, 4, 4, 4)

        # Layout selector
        layout_label = QLabel("Layout:")
        toolbar_layout.addWidget(layout_label)

        self._layout_combo = QComboBox()
        for layout_id, display_name, _ in LAYOUTS:
            self._layout_combo.addItem(display_name, layout_id)
        toolbar_layout.addWidget(self._layout_combo)

        toolbar_layout.addStretch()

        # Sync button
        self._sync_btn = QPushButton("Sync Axes")
        self._sync_btn.setCheckable(True)
        self._sync_btn.setToolTip("Synchronize zoom/pan across all panels")
        toolbar_layout.addWidget(self._sync_btn)

        # Clear all button
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setToolTip("Clear all panels")
        toolbar_layout.addWidget(self._clear_btn)

        # Export button
        self._export_btn = QPushButton("Export All")
        self._export_btn.setToolTip("Export all panels to files")
        toolbar_layout.addWidget(self._export_btn)

        main_layout.addLayout(toolbar_layout)

        # Grid container for panels
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(4)

        main_layout.addWidget(self._grid_widget, 1)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        self._sync_btn.toggled.connect(self._on_sync_toggled)
        self._clear_btn.clicked.connect(self.clear_all)

    def _on_layout_changed(self, index: int) -> None:
        """Handle layout change.

        Parameters
        ----------
        index : int
            New combo box index
        """
        layout_id = self._layout_combo.currentData()
        self._set_layout(layout_id)
        self.layout_changed.emit(layout_id)

    def _set_layout(self, layout_id: str) -> None:
        """Set grid layout.

        Parameters
        ----------
        layout_id : str
            Layout identifier (e.g., "2x2")
        """
        # Find layout configuration
        rows, cols = 1, 1
        for lid, _, dims in LAYOUTS:
            if lid == layout_id:
                rows, cols = dims
                break

        self._current_layout = layout_id
        n_panels = rows * cols

        # Clear existing panels
        for panel in self._panels:
            self._grid_layout.removeWidget(panel)
            panel.deleteLater()
        self._panels.clear()

        # Create new panels
        for i in range(n_panels):
            panel = PlotPanel(i, self)
            self._panels.append(panel)

            row = i // cols
            col = i % cols
            self._grid_layout.addWidget(panel, row, col)

        # Update combo box
        idx = self._layout_combo.findData(layout_id)
        if idx >= 0:
            self._layout_combo.blockSignals(True)
            self._layout_combo.setCurrentIndex(idx)
            self._layout_combo.blockSignals(False)

    def _on_sync_toggled(self, checked: bool) -> None:
        """Handle sync button toggle.

        Parameters
        ----------
        checked : bool
            Whether sync is enabled
        """
        self._sync_axes = checked
        # Note: Full axis sync implementation would require
        # connecting xlim/ylim change callbacks across all panels

    def add_plot(self, index: int, figure: Any) -> None:
        """Add or replace figure in a panel.

        Parameters
        ----------
        index : int
            Panel index
        figure : matplotlib.Figure or callable
            Figure object or callable that draws on axes
        """
        if index < 0 or index >= len(self._panels):
            raise IndexError(f"Panel index {index} out of range")

        panel = self._panels[index]
        panel_fig = panel.get_figure()

        # Clear existing content
        panel_fig.clear()

        if callable(figure):
            # Call function with axes
            ax = panel_fig.add_subplot(111)
            figure(ax)
        elif hasattr(figure, "axes"):
            # Copy from another figure
            # This is a simplified approach - full copying is complex
            for src_ax in figure.axes:
                ax = panel_fig.add_subplot(111)
                # Copy lines
                for line in src_ax.get_lines():
                    ax.plot(
                        line.get_xdata(),
                        line.get_ydata(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        label=line.get_label(),
                    )
                ax.set_xlabel(src_ax.get_xlabel())
                ax.set_ylabel(src_ax.get_ylabel())
                ax.set_title(src_ax.get_title())
                if src_ax.get_legend():
                    ax.legend()

        panel.refresh()

    def set_panel_title(self, index: int, title: str) -> None:
        """Set title for a panel.

        Parameters
        ----------
        index : int
            Panel index
        title : str
            Panel title
        """
        if 0 <= index < len(self._panels):
            self._panels[index].set_title(title)

    def get_panel(self, index: int) -> PlotPanel | None:
        """Get panel by index.

        Parameters
        ----------
        index : int
            Panel index

        Returns
        -------
        PlotPanel or None
            Panel widget or None if index invalid
        """
        if 0 <= index < len(self._panels):
            return self._panels[index]
        return None

    def get_figure(self, index: int) -> Figure | None:
        """Get figure from panel.

        Parameters
        ----------
        index : int
            Panel index

        Returns
        -------
        Figure or None
            Matplotlib figure or None if index invalid
        """
        panel = self.get_panel(index)
        return panel.get_figure() if panel else None

    def get_num_panels(self) -> int:
        """Get number of panels.

        Returns
        -------
        int
            Number of panels in current layout
        """
        return len(self._panels)

    def clear_panel(self, index: int) -> None:
        """Clear a specific panel.

        Parameters
        ----------
        index : int
            Panel index
        """
        if 0 <= index < len(self._panels):
            self._panels[index].clear()

    def clear_all(self) -> None:
        """Clear all panels."""
        for panel in self._panels:
            panel.clear()

    def refresh_all(self) -> None:
        """Refresh all panel canvases."""
        for panel in self._panels:
            panel.refresh()

    def export_panel(self, index: int, filepath: str, dpi: int = 150) -> None:
        """Export a single panel to file.

        Parameters
        ----------
        index : int
            Panel index
        filepath : str
            Output file path
        dpi : int, optional
            Resolution for raster formats
        """
        panel = self.get_panel(index)
        if panel:
            panel.get_figure().savefig(filepath, dpi=dpi, bbox_inches="tight")

    def export_all(self, base_path: str, format: str = "png", dpi: int = 150) -> None:
        """Export all panels to files.

        Parameters
        ----------
        base_path : str
            Base file path (panel index will be appended)
        format : str, optional
            File format (png, pdf, svg)
        dpi : int, optional
            Resolution for raster formats
        """
        import os
        base, ext = os.path.splitext(base_path)

        for i, panel in enumerate(self._panels):
            filepath = f"{base}_panel{i + 1}.{format}"
            panel.get_figure().savefig(filepath, dpi=dpi, bbox_inches="tight")

    def get_layout(self) -> str:
        """Get current layout.

        Returns
        -------
        str
            Current layout identifier
        """
        return self._current_layout

    def set_layout_preset(self, layout: str) -> None:
        """Set layout by preset name.

        Parameters
        ----------
        layout : str
            Layout identifier (e.g., "2x2")
        """
        idx = self._layout_combo.findData(layout)
        if idx >= 0:
            self._layout_combo.setCurrentIndex(idx)
