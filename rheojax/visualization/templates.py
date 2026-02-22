"""Plot templates for common rheological visualizations.

This module provides template-based plotting functions for standard rheological
plots including stress-strain, modulus-frequency, and mastercurve plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from jax import Array

from rheojax.core.data import RheoData
from rheojax.logging import get_logger
from rheojax.visualization.plotter import (
    _apply_style,
    _ensure_numpy,
    _filter_positive,
    plot_frequency_domain,
    plot_residuals,
    plot_time_domain,
)

# Module logger
logger = get_logger(__name__)


def plot_stress_strain(
    data: RheoData, style: str = "default", **kwargs: Any
) -> tuple[Figure, Axes]:
    """Plot stress-strain or time-dependent rheological data.

    This template is designed for relaxation and creep tests, plotting
    stress or strain versus time.

    Args:
        data: RheoData object containing time-domain data
        style: Plotting style ('default', 'publication', 'presentation')
        **kwargs: Additional keyword arguments for matplotlib

    Returns:
        Tuple of (Figure, Axes)

    Examples:
        >>> time = np.linspace(0, 100, 200)
        >>> stress = 1000 * np.exp(-time / 20)
        >>> data = RheoData(x=time, y=stress, domain="time")
        >>> fig, ax = plot_stress_strain(data)
    """
    logger.debug("Generating plot", plot_type="stress_strain", style=style)

    try:
        test_mode = data.metadata.get("test_mode", "")

        # Determine if log scale is appropriate
        log_x = False
        log_y = False

        # For long time ranges, log scale is often more informative
        x_data = _ensure_numpy(data.x)
        if len(x_data) > 0 and np.min(x_data) > 0:
            x_range = np.max(x_data) / np.min(x_data)
            if x_range > 100:  # More than 2 decades
                log_x = True

        # Plot using time_domain plotter
        fig, ax = plot_time_domain(
            _ensure_numpy(data.x),
            _ensure_numpy(data.y),
            x_units=data.x_units,
            y_units=data.y_units,
            log_x=log_x,
            log_y=log_y,
            style=style,
            **kwargs,
        )

        # Update labels based on test mode
        if test_mode == "relaxation":
            ax.set_ylabel(f"Stress ({data.y_units})" if data.y_units else "Stress (Pa)")
            ax.set_title("Stress Relaxation")
        elif test_mode == "creep":
            ax.set_ylabel(f"Strain ({data.y_units})" if data.y_units else "Strain")
            ax.set_title("Creep Compliance")

        logger.debug("Figure created", plot_type="stress_strain")
        return fig, ax

    except Exception as e:
        logger.error(
            "Failed to generate stress_strain plot",
            plot_type="stress_strain",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_modulus_frequency(
    data: RheoData, separate_axes: bool = True, style: str = "default", **kwargs: Any
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot storage and loss modulus versus frequency.

    This template is designed for oscillatory (SAOS) test data, plotting
    G' and G'' versus frequency on log-log axes.

    Args:
        data: RheoData object containing frequency-domain data
        separate_axes: If True, plot G' and G'' on separate axes
        style: Plotting style
        **kwargs: Additional keyword arguments for matplotlib

    Returns:
        Tuple of (Figure, Axes) or (Figure, array of Axes)

    Examples:
        >>> frequency = np.logspace(-2, 2, 50)
        >>> G_complex = 1e5 / (1 + 1j * frequency)
        >>> data = RheoData(x=frequency, y=G_complex, domain="frequency")
        >>> fig, axes = plot_modulus_frequency(data)
    """
    logger.debug("Generating plot", plot_type="modulus_frequency", style=style)

    try:
        x_data = _ensure_numpy(data.x)
        y_data = _ensure_numpy(data.y)

        if separate_axes and np.iscomplexobj(y_data):
            # Two separate axes for G' and G''
            fig, axes = plot_frequency_domain(
                x_data,
                y_data,
                x_units=data.x_units,
                y_units=data.y_units,
                style=style,
                **kwargs,
            )

            axes[0].set_title("Storage Modulus (G')")
            axes[1].set_title("Loss Modulus (G'')")

            logger.debug("Figure created", plot_type="modulus_frequency")
            return fig, axes
        else:
            # Single axis (either real data or combined plot)
            style_params = _apply_style(style)

            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

            # Set font sizes
            plt.rcParams.update(
                {
                    "font.size": style_params["font.size"],
                    "axes.labelsize": style_params["axes.labelsize"],
                    "xtick.labelsize": style_params["xtick.labelsize"],
                    "ytick.labelsize": style_params["ytick.labelsize"],
                }
            )

            plot_kwargs = {
                "linewidth": style_params["lines.linewidth"],
                "marker": "o",
                "markersize": style_params["lines.markersize"],
                "markerfacecolor": "none",
                "markeredgewidth": 1.0,
            }
            plot_kwargs.update(kwargs)

            if np.iscomplexobj(y_data):
                # Plot both on same axes
                x_gp, gp = _filter_positive(x_data, np.real(y_data), warn=True)
                x_gpp, gpp = _filter_positive(x_data, np.imag(y_data), warn=True)
                ax.loglog(x_gp, gp, **plot_kwargs, label="G'")
                ax.loglog(x_gpp, gpp, **plot_kwargs, label='G"', color="C1")
                ax.legend()
            else:
                x_filtered, y_filtered = _filter_positive(x_data, y_data, warn=True)
                ax.loglog(x_filtered, y_filtered, **plot_kwargs)

            ax.set_xlabel(
                f"Frequency ({data.x_units})" if data.x_units else "Frequency (rad/s)"
            )
            ax.set_ylabel(
                f"Modulus ({data.y_units})" if data.y_units else "Modulus (Pa)"
            )
            ax.set_title("Dynamic Moduli")
            ax.grid(True, which="both", alpha=0.3, linestyle="--")

            fig.tight_layout()
            logger.debug("Figure created", plot_type="modulus_frequency")
            return fig, ax

    except Exception as e:
        logger.error(
            "Failed to generate modulus_frequency plot",
            plot_type="modulus_frequency",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_mastercurve(
    datasets: list[RheoData],
    reference_temp: float | None = None,
    shift_factors: dict[float, float] | None = None,
    show_shifts: bool = False,
    style: str = "default",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot mastercurve from multiple temperature datasets.

    This template creates a time-temperature superposition plot, overlaying
    data from multiple temperatures with optional shift factors.

    Args:
        datasets: List of RheoData objects at different temperatures
        reference_temp: Reference temperature (if None, uses first dataset)
        shift_factors: Dictionary mapping temperature to shift factor
        show_shifts: If True, display shift factors in legend
        style: Plotting style
        **kwargs: Additional keyword arguments for matplotlib

    Returns:
        Tuple of (Figure, Axes)

    Examples:
        >>> datasets = []
        >>> for temp in [20, 25, 30]:
        ...     freq = np.logspace(-2, 2, 50)
        ...     G = 1e5 / (1 + 1j * freq)
        ...     datasets.append(RheoData(x=freq, y=G, metadata={'temperature': temp}))
        >>> fig, ax = plot_mastercurve(datasets)
    """
    logger.debug(
        "Generating plot",
        plot_type="mastercurve",
        style=style,
        n_datasets=len(datasets),
    )

    try:
        style_params = _apply_style(style)

        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        # Set font sizes
        plt.rcParams.update(
            {
                "font.size": style_params["font.size"],
                "axes.labelsize": style_params["axes.labelsize"],
                "xtick.labelsize": style_params["xtick.labelsize"],
                "ytick.labelsize": style_params["ytick.labelsize"],
            }
        )

        # Get reference temperature
        if reference_temp is None:
            reference_temp = datasets[0].metadata.get("temperature", 25)

        # Plot each dataset
        colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))

        for i, data in enumerate(datasets):
            temp = data.metadata.get("temperature", None)
            x_data = _ensure_numpy(data.x)
            y_data = _ensure_numpy(data.y)

            # Apply shift factor if provided
            if shift_factors is not None and temp in shift_factors:
                shift = shift_factors[temp]
                x_shifted = x_data * shift
            else:
                x_shifted = x_data
                shift = 1.0

            # Create label
            if temp is not None:
                if show_shifts and shift != 1.0:
                    label = f"{temp}C (a_T={shift:.2e})"
                else:
                    label = f"{temp}C"
            else:
                label = f"Dataset {i+1}"

            # Plot (handle complex data)
            if np.iscomplexobj(y_data):
                x_filt, y_filt = _filter_positive(
                    x_shifted, np.real(y_data), warn=False
                )
                ax.loglog(
                    x_filt,
                    y_filt,
                    "o",
                    color=colors[i],
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    label=label,
                    **kwargs,
                )
            else:
                x_filt, y_filt = _filter_positive(x_shifted, y_data, warn=False)
                ax.loglog(
                    x_filt,
                    y_filt,
                    "o",
                    color=colors[i],
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    label=label,
                    **kwargs,
                )

        # Labels
        x_units = datasets[0].x_units if datasets[0].x_units else "rad/s"
        y_units = datasets[0].y_units if datasets[0].y_units else "Pa"

        # VIS-009: Fix inverted label semantics — shifted when shifts applied
        ax.set_xlabel(
            f"Shifted Frequency (a_T x {x_units})"
            if shift_factors
            else f"Frequency ({x_units})"
        )
        ax.set_ylabel(f"G' ({y_units})")
        ax.set_title(f"Master Curve (T_ref = {reference_temp}C)")
        ax.legend(loc="best", fontsize=style_params["legend.fontsize"])
        ax.grid(True, which="both", alpha=0.3, linestyle="--")

        fig.tight_layout()

        logger.debug("Figure created", plot_type="mastercurve")
        return fig, ax

    except Exception as e:
        logger.error(
            "Failed to generate mastercurve plot",
            plot_type="mastercurve",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_model_fit(
    data: RheoData,
    predictions: np.ndarray | Array,
    show_residuals: bool = True,
    style: str = "default",
    model_name: str | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot experimental data with model predictions and residuals.

    This template creates a standard model fitting visualization showing
    data, model predictions, and optionally residuals.

    Args:
        data: RheoData object with experimental data
        predictions: Model predictions
        show_residuals: If True, add residuals subplot
        style: Plotting style
        model_name: Name of the model (for title)
        **kwargs: Additional keyword arguments for matplotlib

    Returns:
        Tuple of (Figure, Axes) or (Figure, array of Axes)

    Examples:
        >>> freq = np.logspace(-2, 2, 50)
        >>> G_data = 1e5 / (1 + 1j * freq)
        >>> G_pred = G_data * 1.02  # Slight variation
        >>> data = RheoData(x=freq, y=G_data, domain="frequency")
        >>> fig, axes = plot_model_fit(data, G_pred)
    """
    logger.debug(
        "Generating plot",
        plot_type="model_fit",
        style=style,
        model_name=model_name,
        show_residuals=show_residuals,
    )

    try:
        style_params = _apply_style(style)

        # Set font sizes
        plt.rcParams.update(
            {
                "font.size": style_params["font.size"],
                "axes.labelsize": style_params["axes.labelsize"],
                "xtick.labelsize": style_params["xtick.labelsize"],
                "ytick.labelsize": style_params["ytick.labelsize"],
            }
        )

        x_data = _ensure_numpy(data.x)
        y_data = _ensure_numpy(data.y)
        y_pred = _ensure_numpy(predictions)

        if show_residuals:
            # Two subplots: fit and residuals
            if np.iscomplexobj(y_data):
                # For complex data, plot G' and G'' separately
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(
                        style_params["figure.figsize"][0] * 1.5,
                        style_params["figure.figsize"][1] * 1.5,
                    ),
                )

                # G' fit
                x_gp_data, gp_data = _filter_positive(
                    x_data, np.real(y_data), warn=True
                )
                x_gp_pred, gp_pred = _filter_positive(
                    x_data, np.real(y_pred), warn=False
                )
                axes[0, 0].loglog(
                    x_gp_data,
                    gp_data,
                    "o",
                    label="Data",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                )
                axes[0, 0].loglog(
                    x_gp_pred,
                    gp_pred,
                    "-",
                    label="Model",
                    linewidth=style_params["lines.linewidth"],
                )
                axes[0, 0].set_ylabel(
                    f"G' ({data.y_units})" if data.y_units else "G' (Pa)"
                )
                axes[0, 0].legend()
                axes[0, 0].grid(True, which="both", alpha=0.3, linestyle="--")

                # G'' fit
                x_gpp_data, gpp_data = _filter_positive(
                    x_data, np.imag(y_data), warn=True
                )
                x_gpp_pred, gpp_pred = _filter_positive(
                    x_data, np.imag(y_pred), warn=False
                )
                axes[0, 1].loglog(
                    x_gpp_data,
                    gpp_data,
                    "o",
                    label="Data",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    color="C1",
                )
                axes[0, 1].loglog(
                    x_gpp_pred,
                    gpp_pred,
                    "-",
                    label="Model",
                    linewidth=style_params["lines.linewidth"],
                    color="C1",
                )
                axes[0, 1].set_ylabel(
                    f'G" ({data.y_units})' if data.y_units else 'G" (Pa)'
                )
                axes[0, 1].legend()
                axes[0, 1].grid(True, which="both", alpha=0.3, linestyle="--")

                # G' residuals
                residuals_gp = np.real(y_data) - np.real(y_pred)
                gp_denom = np.where(
                    np.abs(np.real(y_data)) > 1e-12, np.real(y_data), 1.0
                )
                axes[1, 0].semilogx(
                    x_data,
                    residuals_gp / gp_denom * 100,
                    "o",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                )
                axes[1, 0].axhline(y=0, color="k", linestyle="--", linewidth=1.0)
                axes[1, 0].set_xlabel(
                    f"Frequency ({data.x_units})"
                    if data.x_units
                    else "Frequency (rad/s)"
                )
                axes[1, 0].set_ylabel("G' Residuals (%)")
                axes[1, 0].grid(True, alpha=0.3, linestyle="--")

                # G'' residuals
                residuals_gpp = np.imag(y_data) - np.imag(y_pred)
                gpp_denom = np.where(
                    np.abs(np.imag(y_data)) > 1e-12, np.imag(y_data), 1.0
                )
                axes[1, 1].semilogx(
                    x_data,
                    residuals_gpp / gpp_denom * 100,
                    "o",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    color="C1",
                )
                axes[1, 1].axhline(y=0, color="k", linestyle="--", linewidth=1.0)
                axes[1, 1].set_xlabel(
                    f"Frequency ({data.x_units})"
                    if data.x_units
                    else "Frequency (rad/s)"
                )
                axes[1, 1].set_ylabel('G" Residuals (%)')
                axes[1, 1].grid(True, alpha=0.3, linestyle="--")

                if model_name:
                    fig.suptitle(
                        f"Model Fit: {model_name}",
                        fontsize=style_params["axes.titlesize"],
                    )

                fig.tight_layout()
                logger.debug("Figure created", plot_type="model_fit")
                return fig, axes
            else:
                # Real data
                residuals = y_data - y_pred

                fig, axes = plot_residuals(
                    x_data,
                    residuals,
                    y_true=y_data,
                    y_pred=y_pred,
                    x_units=data.x_units,
                    style=style,
                )

                if model_name:
                    axes[0].set_title(f"Model Fit: {model_name}")

                logger.debug("Figure created", plot_type="model_fit")
                return fig, axes
        else:
            # Single plot: fit only
            if np.iscomplexobj(y_data):
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(
                        style_params["figure.figsize"][0] * 1.5,
                        style_params["figure.figsize"][1],
                    ),
                )

                # G' fit
                x_gp_data, gp_data = _filter_positive(
                    x_data, np.real(y_data), warn=True
                )
                x_gp_pred, gp_pred = _filter_positive(
                    x_data, np.real(y_pred), warn=False
                )
                axes[0].loglog(
                    x_gp_data,
                    gp_data,
                    "o",
                    label="Data",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                )
                axes[0].loglog(
                    x_gp_pred,
                    gp_pred,
                    "-",
                    label="Model",
                    linewidth=style_params["lines.linewidth"],
                )
                axes[0].set_xlabel(
                    f"Frequency ({data.x_units})"
                    if data.x_units
                    else "Frequency (rad/s)"
                )
                axes[0].set_ylabel(
                    f"G' ({data.y_units})" if data.y_units else "G' (Pa)"
                )
                axes[0].legend()
                axes[0].grid(True, which="both", alpha=0.3, linestyle="--")

                # G'' fit
                x_gpp_data, gpp_data = _filter_positive(
                    x_data, np.imag(y_data), warn=True
                )
                x_gpp_pred, gpp_pred = _filter_positive(
                    x_data, np.imag(y_pred), warn=False
                )
                axes[1].loglog(
                    x_gpp_data,
                    gpp_data,
                    "o",
                    label="Data",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    color="C1",
                )
                axes[1].loglog(
                    x_gpp_pred,
                    gpp_pred,
                    "-",
                    label="Model",
                    linewidth=style_params["lines.linewidth"],
                    color="C1",
                )
                axes[1].set_xlabel(
                    f"Frequency ({data.x_units})"
                    if data.x_units
                    else "Frequency (rad/s)"
                )
                axes[1].set_ylabel(
                    f'G" ({data.y_units})' if data.y_units else 'G" (Pa)'
                )
                axes[1].legend()
                axes[1].grid(True, which="both", alpha=0.3, linestyle="--")

                if model_name:
                    fig.suptitle(
                        f"Model Fit: {model_name}",
                        fontsize=style_params["axes.titlesize"],
                    )

                fig.tight_layout()
                logger.debug("Figure created", plot_type="model_fit")
                return fig, axes
            else:
                fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

                ax.plot(
                    x_data,
                    y_data,
                    "o",
                    label="Data",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                )
                ax.plot(
                    x_data,
                    y_pred,
                    "-",
                    label="Model",
                    linewidth=style_params["lines.linewidth"],
                )

                ax.set_xlabel(f"x ({data.x_units})" if data.x_units else "x")
                ax.set_ylabel(f"y ({data.y_units})" if data.y_units else "y")
                ax.legend()
                ax.grid(True, alpha=0.3, linestyle="--")

                if model_name:
                    ax.set_title(f"Model Fit: {model_name}")

                fig.tight_layout()
                logger.debug("Figure created", plot_type="model_fit")
                return fig, ax

    except Exception as e:
        logger.error(
            "Failed to generate model_fit plot",
            plot_type="model_fit",
            model_name=model_name,
            error=str(e),
            exc_info=True,
        )
        raise


def apply_template_style(ax: Axes, style: str = "default", **kwargs: Any) -> None:
    """Apply template styling to an existing axis.

    This function applies consistent styling to matplotlib axes based on
    the selected template style.

    Args:
        ax: Matplotlib axis to style
        style: Style name ('default', 'publication', 'presentation')
        **kwargs: Additional style parameters to override

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
        >>> apply_template_style(ax, style='publication')
    """
    logger.debug("Applying template style", style=style)

    try:
        style_params = _apply_style(style)
        style_params.update(kwargs)

        # Apply font sizes
        ax.xaxis.label.set_fontsize(style_params["axes.labelsize"])
        ax.yaxis.label.set_fontsize(style_params["axes.labelsize"])
        ax.title.set_fontsize(style_params["axes.titlesize"])

        for label in ax.get_xticklabels():
            label.set_fontsize(style_params["xtick.labelsize"])
        for label in ax.get_yticklabels():
            label.set_fontsize(style_params["ytick.labelsize"])

        # Update line widths and marker sizes
        for line in ax.get_lines():
            if line.get_linewidth() == plt.rcParams["lines.linewidth"]:
                line.set_linewidth(style_params["lines.linewidth"])
            if line.get_markersize() == plt.rcParams["lines.markersize"]:
                line.set_markersize(style_params["lines.markersize"])

        # Grid
        ax.grid(True, which="both", alpha=0.3, linestyle="--")

        logger.debug("Template style applied", style=style)

    except Exception as e:
        logger.error(
            "Failed to apply template style",
            style=style,
            error=str(e),
            exc_info=True,
        )
        raise
