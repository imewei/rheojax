"""
Custom Widgets
=============

Reusable UI components for the RheoJAX GUI.
"""

# Export enums from store for convenience
from rheojax.gui.foundation.state import PipelineStep, StepStatus
from rheojax.gui.widgets.arviz_canvas import ArviZCanvas, ArvizCanvas
from rheojax.gui.widgets.base_arviz_widget import BaseArviZWidget, PlotMetrics
from rheojax.gui.widgets.dropdown import RheoComboBox
from rheojax.gui.widgets.parameter_form import ParameterFormBuilder
from rheojax.gui.widgets.parameter_table import ParameterTable
from rheojax.gui.widgets.plot_canvas import PlotCanvas
from rheojax.gui.widgets.priors_editor import PriorsEditor
from rheojax.gui.widgets.pyqtgraph_canvas import (
    PYQTGRAPH_AVAILABLE,
    PyQtGraphCanvas,
    is_pyqtgraph_available,
)
from rheojax.gui.widgets.residuals_panel import ResidualsPanel

__all__ = [
    "BaseArviZWidget",
    "PlotCanvas",
    "PlotMetrics",
    "ParameterTable",
    "PipelineStep",
    "StepStatus",
    "ArvizCanvas",
    "ArviZCanvas",
    "PriorsEditor",
    "ResidualsPanel",
    "PyQtGraphCanvas",
    "is_pyqtgraph_available",
    "ParameterFormBuilder",
    "PYQTGRAPH_AVAILABLE",
    "RheoComboBox",
]
