"""Unified plotting and visualization utilities for rheological data.

This module provides:
- Consistent plotting interface
- Pre-configured templates for common rheological plots
- Publication-quality matplotlib styling
- Export to PNG, PDF, SVG formats
"""

import warnings

from rheojax.visualization.epm_plots import (
    animate_stress_evolution,
    animate_tensorial_evolution,
    plot_lattice_fields,
    plot_normal_stress_field,
    plot_normal_stress_ratio,
    plot_tensorial_fields,
    plot_von_mises_field,
)
from rheojax.visualization.plotter import (
    plot_flow_curve,
    plot_frequency_domain,
    plot_residuals,
    plot_rheo_data,
    plot_time_domain,
    save_figure,
)
from rheojax.visualization.templates import (
    apply_template_style,
    plot_mastercurve,
    plot_model_fit,
    plot_modulus_frequency,
    plot_stress_strain,
)

# Automatically suppress font glyph warnings on module import
# Justification: These warnings are purely cosmetic - plots render correctly,
# the glyph is just displayed as a box or skipped. This is harmless for
# headless batch runs. The warning provides no actionable information.
warnings.filterwarnings(
    "ignore",
    message="Glyph.*missing from.*font",
    category=UserWarning,
)


def configure_matplotlib(
    unicode_safe: bool = True,
    suppress_font_warnings: bool = True,
    style: str = "default",
) -> None:
    """Configure matplotlib for rheological plotting with proper font handling.

    This function configures matplotlib to handle Unicode characters properly
    (e.g., subscripts like σ₀, τ₀) and optionally suppresses font glyph warnings.

    Args:
        unicode_safe: If True, use DejaVu Sans font which has full Unicode support.
        suppress_font_warnings: If True, suppress "Glyph missing from font" warnings.
            These warnings are cosmetic - plots render correctly, just without
            the specific Unicode glyph. Justified suppression for headless runs.
        style: Matplotlib style to apply ("default", "seaborn-v0_8-whitegrid", etc.)

    Example:
        >>> from rheojax.visualization import configure_matplotlib
        >>> configure_matplotlib()  # Call once at notebook start
    """
    import matplotlib.pyplot as plt

    # Apply style if specified
    if style != "default":
        try:
            plt.style.use(style)
        except OSError:
            pass  # Style not available, use default

    # Configure font for Unicode support
    if unicode_safe:
        # DejaVu Sans has extensive Unicode coverage including subscripts
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Helvetica",
            "Arial",
            "sans-serif",
        ]

    # Suppress font glyph warnings if requested
    # Justification: These warnings are purely cosmetic - the plot renders
    # correctly, the glyph is just displayed as a box or skipped. This is
    # harmless for headless batch runs.
    if suppress_font_warnings:
        warnings.filterwarnings(
            "ignore",
            message="Glyph.*missing from.*font",
            category=UserWarning,
        )


__all__ = [
    # Configuration
    "configure_matplotlib",
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
    # EPM plots - scalar
    "plot_lattice_fields",
    "animate_stress_evolution",
    # EPM plots - tensorial
    "plot_tensorial_fields",
    "plot_normal_stress_field",
    "plot_von_mises_field",
    "plot_normal_stress_ratio",
    "animate_tensorial_evolution",
]
