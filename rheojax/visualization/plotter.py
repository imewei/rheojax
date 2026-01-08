"""Core plotting functions for rheological data visualization.

This module provides publication-quality plotting utilities for rheological data,
with automatic plot type selection based on data characteristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax

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


def _ensure_numpy(data: np.ndarray) -> np.ndarray:
    """Ensure data is a NumPy array for plotting.

    Args:
        data: Input data array

    Returns:
        NumPy array
    """
    if isinstance(data, jnp.ndarray):
        return np.array(data)
    return np.asarray(data)


def _filter_positive(
    x: np.ndarray, y: np.ndarray, warn: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Filter out non-positive y values for log-scale plotting.

    Args:
        x: X-axis data
        y: Y-axis data
        warn: If True, warn when filtering occurs

    Returns:
        Tuple of (filtered_x, filtered_y) with only positive y values
    """
    positive_mask = y > 0
    n_removed = len(y) - np.sum(positive_mask)

    if n_removed > 0 and warn:
        import warnings

        warnings.warn(
            f"Removed {n_removed} non-positive values from log-scale plot. "
            f"This is common for very small G'' values or measurement noise.",
            UserWarning,
            stacklevel=3,
        )

    return x[positive_mask], y[positive_mask]


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
        x_gp, gp = _filter_positive(x, np.real(y), warn=True)
        axes[0].loglog(x_gp, gp, **plot_kwargs, label="G'")
        axes[0].set_ylabel(f"G' ({y_units})" if y_units else "G' (Pa)")
        axes[0].grid(True, which="both", alpha=0.3, linestyle="--")
        axes[0].legend()

        # G'' (loss modulus)
        x_gpp, gpp = _filter_positive(x, np.imag(y), warn=True)
        axes[1].loglog(x_gpp, gpp, **plot_kwargs, label='G"', color="C1")
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

        x_filtered, y_filtered = _filter_positive(x, y, warn=True)
        ax.loglog(x_filtered, y_filtered, **plot_kwargs)
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

    # Filter positive values for log-log plot
    x_filtered, y_filtered = _filter_positive(x, y, warn=True)
    ax.loglog(x_filtered, y_filtered, **plot_kwargs)

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


def compute_uncertainty_band(
    model_fn: callable,
    x_pred: np.ndarray,
    popt: np.ndarray,
    pcov: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute prediction uncertainty band via error propagation.

    Uses the formula: σ_y(x) = sqrt(diag(J @ pcov @ J.T))
    where J is the Jacobian of the model with respect to parameters.

    For 95% confidence interval, the band is ±1.96 * σ_y(x).

    Args:
        model_fn: Model function that takes (x, params) and returns predictions.
            Must be compatible with JAX autodiff for Jacobian computation.
        x_pred: X values for prediction (n_points,)
        popt: Optimal parameter values (n_params,)
        pcov: Parameter covariance matrix (n_params x n_params)
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (y_fit, y_lower, y_upper) where:
            - y_fit: Fitted values at x_pred
            - y_lower: Lower bound of confidence interval
            - y_upper: Upper bound of confidence interval

    Example:
        >>> def model(x, params):
        ...     a, b = params
        ...     return a * x + b
        >>> x = np.linspace(0, 10, 50)
        >>> popt = np.array([2.0, 1.0])
        >>> pcov = np.array([[0.01, 0.0], [0.0, 0.05]])
        >>> y_fit, y_lo, y_hi = compute_uncertainty_band(model, x, popt, pcov)
    """
    from scipy.stats import norm

    x_pred = np.asarray(x_pred, dtype=np.float64)
    popt = np.asarray(popt, dtype=np.float64)
    pcov = np.asarray(pcov, dtype=np.float64)

    # Compute y_fit
    y_fit = np.asarray(model_fn(x_pred, popt), dtype=np.float64)

    # Handle complex output (e.g., G* = G' + iG'')
    if np.iscomplexobj(y_fit):
        # Return None for complex - uncertainty bands are harder to interpret
        return y_fit, None, None

    # Compute Jacobian: J[i, j] = ∂y[i] / ∂param[j]
    # Use finite differences for robustness
    n_points = len(x_pred)
    n_params = len(popt)
    jac = np.zeros((n_points, n_params), dtype=np.float64)

    eps = np.sqrt(np.finfo(np.float64).eps)
    for j in range(n_params):
        params_plus = popt.copy()
        params_plus[j] += eps * max(abs(popt[j]), 1.0)
        y_plus = np.asarray(model_fn(x_pred, params_plus), dtype=np.float64)

        params_minus = popt.copy()
        params_minus[j] -= eps * max(abs(popt[j]), 1.0)
        y_minus = np.asarray(model_fn(x_pred, params_minus), dtype=np.float64)

        jac[:, j] = (y_plus - y_minus) / (2 * eps * max(abs(popt[j]), 1.0))

    # Compute variance: var_y = diag(J @ pcov @ J.T)
    try:
        var_y = np.einsum("ij,jk,ik->i", jac, pcov, jac)
        sigma_y = np.sqrt(np.maximum(var_y, 0.0))
    except Exception:
        # Fallback if einsum fails
        return y_fit, None, None

    # Compute z-score for confidence interval
    z = norm.ppf(1 - (1 - confidence) / 2)

    y_lower = y_fit - z * sigma_y
    y_upper = y_fit + z * sigma_y

    return y_fit, y_lower, y_upper


def plot_fit_with_uncertainty(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
    log_x: bool = True,
    log_y: bool = True,
    data_label: str = "Data",
    fit_label: str = "Fit",
    band_label: str = "95% CI",
    x_label: str | None = None,
    y_label: str | None = None,
    style: str = "default",
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure | None, Axes]:
    """Plot data with fitted curve and optional uncertainty band.

    Creates publication-quality fit plots with:
    - Scatter data points
    - Solid fitted curve
    - Shaded uncertainty band (if y_lower/y_upper provided)
    - Legend with customizable labels

    Args:
        x_data: Experimental x values
        y_data: Experimental y values
        x_fit: X values for fitted curve (can be denser than data)
        y_fit: Fitted y values
        y_lower: Lower bound of uncertainty band (optional)
        y_upper: Upper bound of uncertainty band (optional)
        log_x: Use log scale for x-axis (default: True)
        log_y: Use log scale for y-axis (default: True)
        data_label: Legend label for data points
        fit_label: Legend label for fitted curve
        band_label: Legend label for uncertainty band
        x_label: X-axis label
        y_label: Y-axis label
        style: Plot style ('default', 'publication', 'presentation')
        ax: Optional existing axes to plot on
        **kwargs: Additional arguments passed to scatter/plot

    Returns:
        Tuple of (Figure, Axes). Figure is None if ax was provided.

    Example:
        >>> x = np.logspace(-1, 2, 20)
        >>> y = 100 * x ** -0.5 + np.random.randn(20) * 5
        >>> x_fit = np.logspace(-1, 2, 100)
        >>> y_fit = 100 * x_fit ** -0.5
        >>> fig, ax = plot_fit_with_uncertainty(x, y, x_fit, y_fit)
    """
    style_params = _apply_style(style)

    # Create figure if no axes provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

    x_data = _ensure_numpy(x_data)
    y_data = _ensure_numpy(y_data)
    x_fit = _ensure_numpy(x_fit)
    y_fit = _ensure_numpy(y_fit)

    def scatter_fn(x, y, **kw):
        return ax.scatter(x, y, **kw)

    # Determine plot functions based on log scales
    if log_x and log_y:
        plot_fn = ax.loglog
        # Filter positive values for log scale
        x_data_plot, y_data_plot = _filter_positive(x_data, y_data)
        x_fit_plot, y_fit_plot = _filter_positive(x_fit, y_fit, warn=False)
    elif log_x:
        plot_fn = ax.semilogx
        x_data_plot, y_data_plot = x_data, y_data
        x_fit_plot, y_fit_plot = x_fit, y_fit
    elif log_y:
        plot_fn = ax.semilogy
        x_data_plot, y_data_plot = _filter_positive(x_data, y_data)
        x_fit_plot, y_fit_plot = _filter_positive(x_fit, y_fit, warn=False)
    else:
        plot_fn = ax.plot
        x_data_plot, y_data_plot = x_data, y_data
        x_fit_plot, y_fit_plot = x_fit, y_fit

    # Plot uncertainty band first (so it's behind other elements)
    if y_lower is not None and y_upper is not None:
        y_lower = _ensure_numpy(y_lower)
        y_upper = _ensure_numpy(y_upper)
        if log_y:
            # Filter positive for log scale
            mask = (y_lower > 0) & (y_upper > 0) & (x_fit > 0)
            ax.fill_between(
                x_fit[mask],
                y_lower[mask],
                y_upper[mask],
                alpha=0.3,
                color="C0",
                label=band_label,
                zorder=1,
            )
        else:
            ax.fill_between(
                x_fit,
                y_lower,
                y_upper,
                alpha=0.3,
                color="C0",
                label=band_label,
                zorder=1,
            )

    # Plot fitted curve
    plot_fn(
        x_fit_plot,
        y_fit_plot,
        "-",
        linewidth=style_params["lines.linewidth"],
        color="C0",
        label=fit_label,
        zorder=2,
    )

    # Plot data points
    scatter_fn(
        x_data_plot,
        y_data_plot,
        s=style_params["lines.markersize"] ** 2,
        facecolors="none",
        edgecolors="C1",
        linewidths=1.5,
        label=data_label,
        zorder=3,
    )

    # Labels and legend
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    if fig is not None:
        fig.tight_layout()

    return fig, ax


def save_figure(
    fig: Figure,
    filepath: str | Path,
    format: str | None = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> Path:
    """
    Save matplotlib figure to file with publication-quality defaults.

    This convenience function wraps matplotlib's savefig() with sensible defaults
    for publication-quality figures, automatic format detection, and path validation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filepath : str or Path
        Output file path. Format inferred from extension if not specified.
    format : str, optional
        Output format ('pdf', 'svg', 'png', 'eps'). If None, inferred from filepath extension.
    dpi : int, default=300
        Resolution for raster formats (PNG). Ignored for vector formats (PDF, SVG, EPS).
        Common values:
        - 150: Draft quality
        - 300: Publication quality (default)
        - 600: High-resolution print
    bbox_inches : str, default='tight'
        Bounding box adjustment. 'tight' removes extra whitespace around figure.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib's savefig().
        Common options:
        - transparent : bool - Transparent background (default False)
        - facecolor : color - Figure background color
        - edgecolor : color - Figure edge color
        - pad_inches : float - Padding around figure

    Returns
    -------
    Path
        Absolute path to saved file (for confirmation/logging)

    Raises
    ------
    ValueError
        If format cannot be inferred from filepath or is unsupported
    OSError
        If filepath directory doesn't exist or lacks write permissions

    Examples
    --------
    Save figure to PDF with defaults:

    >>> fig, ax = plot_rheo_data(data)
    >>> save_figure(fig, 'analysis.pdf')
    PosixPath('/path/to/analysis.pdf')

    Save PNG with high resolution:

    >>> save_figure(fig, 'figure.png', dpi=600)
    PosixPath('/path/to/figure.png')

    Save SVG with transparent background:

    >>> save_figure(fig, 'diagram.svg', transparent=True)
    PosixPath('/path/to/diagram.svg')

    Explicit format specification:

    >>> save_figure(fig, 'output', format='pdf')
    PosixPath('/path/to/output.pdf')

    See Also
    --------
    plot_rheo_data : Automatic plot type selection
    Pipeline.save_figure : Fluent API integration

    Notes
    -----
    Supported formats:
    - PDF: Vector format, ideal for publications and LaTeX documents
    - SVG: Vector format, editable in Inkscape/Illustrator
    - PNG: Raster format, good for presentations and web
    - EPS: Vector format, legacy publication format

    DPI only affects raster formats (PNG). Vector formats (PDF, SVG, EPS) are
    resolution-independent.
    """
    filepath = Path(filepath)

    # Infer format from extension if not specified
    if format is None:
        if filepath.suffix:
            format = filepath.suffix.lstrip(".")
        else:
            raise ValueError(
                f"Cannot infer format from filepath '{filepath}' (no extension). "
                "Provide explicit format parameter or use file extension "
                "(e.g., 'output.pdf')."
            )

    # Validate format
    supported_formats = {"pdf", "svg", "png", "eps"}
    format_lower = format.lower()
    if format_lower not in supported_formats:
        raise ValueError(
            f"Unsupported format '{format}'. "
            f"Supported formats: {', '.join(sorted(supported_formats))}."
        )

    # Ensure directory exists
    if not filepath.parent.exists():
        raise OSError(
            f"Directory does not exist: {filepath.parent}. "
            "Create directory before saving."
        )

    # Save figure
    fig.savefig(
        filepath, format=format_lower, dpi=dpi, bbox_inches=bbox_inches, **kwargs
    )

    return filepath.resolve()
