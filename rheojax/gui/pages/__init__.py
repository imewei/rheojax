"""
Application Pages
================

Page-based navigation components for main workflow stages.
"""

__all__ = [
    "HomePage",
    "DataPage",
    "TransformPage",
    "FitPage",
    "BayesianPage",
    "DiagnosticsPage",
    "ExportPage",
]


def __getattr__(name: str):
    """Lazy import for page components."""
    if name == "HomePage":
        from rheojax.gui.pages.home_page import HomePage

        return HomePage
    elif name == "DataPage":
        from rheojax.gui.pages.data_page import DataPage

        return DataPage
    elif name == "TransformPage":
        from rheojax.gui.pages.transform_page import TransformPage

        return TransformPage
    elif name == "FitPage":
        from rheojax.gui.pages.fit_page import FitPage

        return FitPage
    elif name == "BayesianPage":
        from rheojax.gui.pages.bayesian_page import BayesianPage

        return BayesianPage
    elif name == "DiagnosticsPage":
        from rheojax.gui.pages.diagnostics_page import DiagnosticsPage

        return DiagnosticsPage
    elif name == "ExportPage":
        from rheojax.gui.pages.export_page import ExportPage

        return ExportPage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
