"""Unified plotting and visualization utilities for rheological data.

This module provides:
- Consistent plotting interface
- Pre-configured templates for common rheological plots
- Publication-quality matplotlib styling
- Export to PNG, PDF, SVG formats
"""

from rheo.visualization.plotter import (
    plot_flow_curve,
    plot_frequency_domain,
    plot_residuals,
    plot_rheo_data,
    plot_time_domain,
    save_figure,
)
from rheo.visualization.templates import (
    apply_template_style,
    plot_mastercurve,
    plot_model_fit,
    plot_modulus_frequency,
    plot_stress_strain,
)

__all__ = [
    # Core plotting functions
    "plot_rheo_data",
    "plot_time_domain",
    "plot_frequency_domain",
    "plot_flow_curve",
    "plot_residuals",
    "save_figure",
    # Template functions
    "plot_stress_strain",
    "plot_modulus_frequency",
    "plot_mastercurve",
    "plot_model_fit",
    "apply_template_style",
]
