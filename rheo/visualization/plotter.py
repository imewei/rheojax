"""Core plotting functions for rheological data visualization.

This module provides publication-quality plotting utilities for rheological data,
with automatic plot type selection based on data characteristics.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rheo.core.data import RheoData
from rheo.core.jax_config import safe_import_jax

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Default plotting style parameters
DEFAULT_STYLE = {
    "figure.figsize": (8, 6),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
}

PUBLICATION_STYLE = {
    "figure.figsize": (6, 4.5),
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
}

PRESENTATION_STYLE = {
    "figure.figsize": (10, 7),
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
}


def _apply_style(style: str = "default") -> dict[str, Any]:
    """Apply plotting style and return style parameters.

    Args:
        style: Style name ('default', 'publication', 'presentation')

    Returns:
        Dictionary of style parameters
    """
    if style == "publication":
        return PUBLICATION_STYLE.copy()
    elif style == "presentation":
        return PRESENTATION_STYLE.copy()
    else:
        return DEFAULT_STYLE.copy()


def _ensure_numpy(data: np.ndarray | jnp.ndarray) -> np.ndarray:
    """Ensure data is a NumPy array for plotting.

    Args:
        data: Input data array

    Returns:
        NumPy array
    """
    if isinstance(data, jnp.ndarray):
        return np.array(data)
    return np.asarray(data)


def plot_rheo_data(
    data: RheoData, style: str = "default", **kwargs: Any
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot RheoData with automatic plot type selection.

    This function automatically selects the appropriate plot type based on
    the data domain, test mode, and data characteristics.

    Args:
        data: RheoData object to plot
        style: Plotting style ('default', 'publication', 'presentation')
        **kwargs: Additional keyword arguments passed to matplotlib

    Returns:
        Tuple of (Figure, Axes) or (Figure, array of Axes)

    Examples:
        >>> time = np.linspace(0, 10, 100)
        >>> stress = 1000 * np.exp(-time / 2)
        >>> data = RheoData(x=time, y=stress, domain="time")
        >>> fig, ax = plot_rheo_data(data)
    """
    test_mode = data.metadata.get("test_mode", "")

    # Select plot type based on domain and test mode
    if data.domain == "frequency" or test_mode == "oscillation":
        # Complex modulus data
        if np.iscomplexobj(data.y):
            return plot_frequency_domain(
                _ensure_numpy(data.x),
                _ensure_numpy(data.y),
                x_units=data.x_units,
                y_units=data.y_units,
                style=style,
                **kwargs,
            )
        else:
            return plot_time_domain(
                _ensure_numpy(data.x),
                _ensure_numpy(data.y),
                x_units=data.x_units,
                y_units=data.y_units,
                style=style,
                **kwargs,
            )
    elif test_mode == "rotation" or data.x_units in ["1/s", "s^-1"]:
        # Flow curve data
        return plot_flow_curve(
            _ensure_numpy(data.x),
            _ensure_numpy(data.y),
            x_units=data.x_units,
            y_units=data.y_units,
            style=style,
            **kwargs,
        )
    else:
        # Time-domain data (relaxation, creep, etc.)
        return plot_time_domain(
            _ensure_numpy(data.x),
            _ensure_numpy(data.y),
            x_units=data.x_units,
            y_units=data.y_units,
            style=style,
            **kwargs,
        )


def plot_time_domain(
    x: np.ndarray,
    y: np.ndarray,
    x_units: str | None = None,
    y_units: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    style: str = "default",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot time-domain rheological data.

    Args:
        x: Independent variable (typically time)
        y: Dependent variable (stress, strain, etc.)
        x_units: Units for x-axis
        y_units: Units for y-axis
        log_x: Use logarithmic scale for x-axis
        log_y: Use logarithmic scale for y-axis
        style: Plotting style
        **kwargs: Additional keyword arguments for matplotlib plot

    Returns:
        Tuple of (Figure, Axes)
    """
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

    # Plot data
    plot_kwargs = {
        "linewidth": style_params["lines.linewidth"],
        "marker": "o",
        "markersize": style_params["lines.markersize"],
        "markerfacecolor": "none",
        "markeredgewidth": 1.0,
    }
    plot_kwargs.update(kwargs)

    ax.plot(x, y, **plot_kwargs)

    # Set labels
    x_label = f"Time ({x_units})" if x_units else "Time"
    y_label = (
        f"Stress ({y_units})"
        if y_units and "Pa" in y_units
        else f"y ({y_units})" if y_units else "y"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set scales
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Grid
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    fig.tight_layout()

    return fig, ax


def plot_frequency_domain(
    x: np.ndarray,
    y: np.ndarray,
    x_units: str | None = None,
    y_units: str | None = None,
    style: str = "default",
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot frequency-domain rheological data (complex modulus).

    For complex data, creates two subplots for G' (storage modulus) and
    G'' (loss modulus). For real data, creates a single plot.

    Args:
        x: Frequency data
        y: Complex modulus data (G* = G' + iG'')
        x_units: Units for frequency axis
        y_units: Units for modulus
        style: Plotting style
        **kwargs: Additional keyword arguments for matplotlib plot

    Returns:
        Tuple of (Figure, Axes) or (Figure, array of Axes) for complex data
    """
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

    if np.iscomplexobj(y):
        # Complex data - plot G' and G'' on separate subplots
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(
                style_params["figure.figsize"][0],
                style_params["figure.figsize"][1] * 1.5,
            ),
        )

        # Plot kwargs
        plot_kwargs = {
            "linewidth": style_params["lines.linewidth"],
            "marker": "o",
            "markersize": style_params["lines.markersize"],
            "markerfacecolor": "none",
            "markeredgewidth": 1.0,
        }
        plot_kwargs.update(kwargs)

        # G' (storage modulus)
        axes[0].loglog(x, np.real(y), **plot_kwargs, label="G'")
        axes[0].set_ylabel(f"G' ({y_units})" if y_units else "G' (Pa)")
        axes[0].grid(True, which="both", alpha=0.3, linestyle="--")
        axes[0].legend()

        # G'' (loss modulus)
        axes[1].loglog(x, np.imag(y), **plot_kwargs, label='G"', color="C1")
        axes[1].set_xlabel(f"Frequency ({x_units})" if x_units else "Frequency (rad/s)")
        axes[1].set_ylabel(f'G" ({y_units})' if y_units else 'G" (Pa)')
        axes[1].grid(True, which="both", alpha=0.3, linestyle="--")
        axes[1].legend()

        fig.tight_layout()
        return fig, axes
    else:
        # Real data - single plot
        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        plot_kwargs = {
            "linewidth": style_params["lines.linewidth"],
            "marker": "o",
            "markersize": style_params["lines.markersize"],
            "markerfacecolor": "none",
            "markeredgewidth": 1.0,
        }
        plot_kwargs.update(kwargs)

        ax.loglog(x, y, **plot_kwargs)
        ax.set_xlabel(f"Frequency ({x_units})" if x_units else "Frequency (rad/s)")
        ax.set_ylabel(f"Modulus ({y_units})" if y_units else "Modulus (Pa)")
        ax.grid(True, which="both", alpha=0.3, linestyle="--")

        fig.tight_layout()
        return fig, [ax]


def plot_flow_curve(
    x: np.ndarray,
    y: np.ndarray,
    x_units: str | None = None,
    y_units: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    style: str = "default",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot flow curve (viscosity or stress vs shear rate).

    Args:
        x: Shear rate data
        y: Viscosity or stress data
        x_units: Units for shear rate
        y_units: Units for y-axis
        x_label: Custom x-axis label
        y_label: Custom y-axis label
        style: Plotting style
        **kwargs: Additional keyword arguments for matplotlib plot

    Returns:
        Tuple of (Figure, Axes)
    """
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

    # Plot data on log-log scale
    plot_kwargs = {
        "linewidth": style_params["lines.linewidth"],
        "marker": "o",
        "markersize": style_params["lines.markersize"],
        "markerfacecolor": "none",
        "markeredgewidth": 1.0,
    }
    plot_kwargs.update(kwargs)

    ax.loglog(x, y, **plot_kwargs)

    # Set labels
    if x_label is None:
        x_label = f"Shear Rate ({x_units})" if x_units else "Shear Rate (1/s)"
    if y_label is None:
        y_label = f"Viscosity ({y_units})" if y_units else "Viscosity (Pa.s)"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Grid
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    fig.tight_layout()

    return fig, ax


def plot_residuals(
    x: np.ndarray,
    residuals: np.ndarray,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    x_units: str | None = None,
    style: str = "default",
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot residuals from model fitting.

    If y_true and y_pred are provided, creates two subplots: one with data
    and predictions, and one with residuals. Otherwise, plots only residuals.

    Args:
        x: Independent variable
        residuals: Residual values (y_true - y_pred)
        y_true: True y values (optional)
        y_pred: Predicted y values (optional)
        x_units: Units for x-axis
        style: Plotting style
        **kwargs: Additional keyword arguments for matplotlib plot

    Returns:
        Tuple of (Figure, Axes) or (Figure, array of Axes)
    """
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

    if y_true is not None and y_pred is not None:
        # Two subplots: data+predictions and residuals
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(
                style_params["figure.figsize"][0],
                style_params["figure.figsize"][1] * 1.5,
            ),
        )

        # Data and predictions
        axes[0].plot(
            x,
            y_true,
            "o",
            label="Data",
            markersize=style_params["lines.markersize"],
            markerfacecolor="none",
            markeredgewidth=1.0,
        )
        axes[0].plot(
            x, y_pred, "-", label="Model", linewidth=style_params["lines.linewidth"]
        )
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, linestyle="--")

        # Residuals
        axes[1].plot(
            x,
            residuals,
            "o",
            markersize=style_params["lines.markersize"],
            markerfacecolor="none",
            markeredgewidth=1.0,
        )
        axes[1].axhline(y=0, color="k", linestyle="--", linewidth=1.0)
        axes[1].set_xlabel(f"x ({x_units})" if x_units else "x")
        axes[1].set_ylabel("Residuals")
        axes[1].grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()
        return fig, axes
    else:
        # Single plot: residuals only
        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        ax.plot(
            x,
            residuals,
            "o",
            markersize=style_params["lines.markersize"],
            markerfacecolor="none",
            markeredgewidth=1.0,
            **kwargs,
        )
        ax.axhline(y=0, color="k", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"x ({x_units})" if x_units else "x")
        ax.set_ylabel("Residuals")
        ax.grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()
        return fig, ax
