"""
Custom Widgets
=============

Reusable UI components for the RheoJAX GUI.
"""

# Export enums from store for convenience
from rheojax.gui.state.store import PipelineStep, StepStatus
from rheojax.gui.widgets.arviz_canvas import ArviZCanvas, ArvizCanvas
from rheojax.gui.widgets.dataset_tree import DatasetTree
from rheojax.gui.widgets.jax_status import JAXStatusWidget
from rheojax.gui.widgets.model_browser import ModelBrowser
from rheojax.gui.widgets.multi_view import MultiView
from rheojax.gui.widgets.parameter_table import ParameterTable
from rheojax.gui.widgets.pipeline_chips import PipelineChips
from rheojax.gui.widgets.plot_canvas import PlotCanvas
from rheojax.gui.widgets.priors_editor import PriorsEditor
from rheojax.gui.app.toolbar import QuickFitStrip
from rheojax.gui.widgets.residuals_panel import ResidualsPanel

__all__ = [
    "PlotCanvas",
    "DatasetTree",
    "ParameterTable",
    "PipelineChips",
    "PipelineStep",
    "StepStatus",
    "JAXStatusWidget",
    "ModelBrowser",
    "QuickFitStrip",
    "ArvizCanvas",
    "ArviZCanvas",
    "PriorsEditor",
    "MultiView",
    "ResidualsPanel",
]
