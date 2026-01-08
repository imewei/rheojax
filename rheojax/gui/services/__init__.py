"""
Service Layer
============

Service layer for RheoJAX API integration with error handling and validation.
"""

__all__ = [
    "DataService",
    "TransformService",
    "ModelService",
    "BayesianService",
    "PlotService",
    "ExportService",
]


def __getattr__(name: str):
    """Lazy import for service components."""
    if name == "DataService":
        from rheojax.gui.services.data_service import DataService

        return DataService
    elif name == "TransformService":
        from rheojax.gui.services.transform_service import TransformService

        return TransformService
    elif name == "ModelService":
        from rheojax.gui.services.model_service import ModelService

        return ModelService
    elif name == "BayesianService":
        from rheojax.gui.services.bayesian_service import BayesianService

        return BayesianService
    elif name == "PlotService":
        from rheojax.gui.services.plot_service import PlotService

        return PlotService
    elif name == "ExportService":
        from rheojax.gui.services.export_service import ExportService

        return ExportService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
