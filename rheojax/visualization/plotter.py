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
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)


def _modulus_labels(
    data: RheoData | None = None,
    y_units: str | None = None,
) -> tuple[str, str, str]:
    """Return (storage_label, loss_label, generic_label) based on deformation mode.

    For tension/bending/compression → E'/E''/Modulus; otherwise G'/G''/Modulus.
    """
    deformation = None
    if data is not None:
        deformation = getattr(data, "deformation_mode", None) or (
            data.metadata.get("deformation_mode") if data.metadata else None
        )

    units = y_units or (data.y_units if data else None) or "Pa"

    if deformation in ("tension", "bending", "compression"):
        return f"E' ({units})", f'E" ({units})', f"Modulus ({units})"
    return f"G' ({units})", f'G" ({units})', f"Modulus ({units})"


# Default plotting style parameters
# Module constants for consistent plot styling
# Note: _GRID_ALPHA and _GRID_LINESTYLE are also used (with inline literals) in
# templates.py, spp_plots.py, and epm_plots.py.
_GRID_ALPHA = 0.3
_GRID_LINESTYLE = "--"

# Default figure sizes (can be overridden per-call via figsize parameter)
_DEFAULT_FIGSIZE = (8, 6)
_PUBLICATION_FIGSIZE = (6, 4.5)
_PRESENTATION_FIGSIZE = (10, 7)

# Explicit viscosity unit set for y-label heuristic in plot_time_domain()
_VISCOSITY_UNITS: frozenset[str] = frozenset(
    {"Pa·s", "Pa.s", "Pa*s", "Pa s", "mPa·s", "mPa.s"}
)

DEFAULT_STYLE = {
    "figure.figsize": _DEFAULT_FIGSIZE,
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
    "figure.figsize": _PUBLICATION_FIGSIZE,
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
    "figure.figsize": _PRESENTATION_FIGSIZE,
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


def _ensure_numpy(data: Any) -> np.ndarray:
    """Ensure data is a NumPy array for plotting.

    Args:
        data: Input data array

    Returns:
        NumPy array
    """
    # VIS-011: Use hasattr check instead of isinstance(data, jnp.ndarray)
    # which is unreliable on JAX >= 0.4.7
    if hasattr(data, "devices"):
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
    # VIS-P2-007: Guard against complex arrays — comparison y > 0 is undefined for complex
    if np.iscomplexobj(y):
        raise TypeError(
            "_filter_positive received complex y array; extract real/imag first"
        )

    positive_mask = np.isfinite(y) & (y > 0)
    n_removed = len(y) - np.sum(positive_mask)

    if n_removed > 0 and warn:
        if np.sum(positive_mask) == 0:
            # F-022: All values non-positive — return original data so the caller
            # gets something to plot rather than empty arrays that create blank plots.
            # Matplotlib will mask non-positive values on log scale by itself.
            import warnings

            logger.warning(
                "All y-values are non-positive; log-scale plot will be empty "
                "(matplotlib drops non-positive values). "
                "Check data scaling or use a linear-scale plot.",
            )
            warnings.warn(
                "All y-values are non-positive; log-scale plot will be empty. "
                "Check data scaling or use a linear-scale plot.",
                UserWarning,
                stacklevel=3,
            )
            return x, y
        else:
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
    logger.debug("Generating plot", plot_type="rheo_data", style=style)

    try:
        _meta = data.metadata or {}
        test_mode = _meta.get("test_mode", "")

        # Select plot type based on domain and test mode
        # VIS-P1-004: Forward deformation mode for E'/G' label selection
        kwargs = dict(kwargs)
        deformation_mode = kwargs.pop("deformation_mode", None)
        if deformation_mode is None:
            deformation_mode = getattr(data, "deformation_mode", None) or _meta.get(
                "deformation_mode"
            )

        # VIS-P2-003: Detect frequency-domain data even when y is real (e.g., only G' stored)
        is_freq_domain = getattr(data, "domain", None) == "frequency" or _meta.get(
            "test_mode"
        ) in (
            "oscillation",
            "frequency_sweep",
        )

        if is_freq_domain or np.iscomplexobj(data.y):
            # Frequency-domain data — pass deformation_mode for label selection
            freq_kwargs = dict(kwargs)
            if deformation_mode:
                freq_kwargs["deformation_mode"] = deformation_mode

            result = plot_frequency_domain(
                _ensure_numpy(data.x),
                _ensure_numpy(data.y),
                x_units=data.x_units,
                y_units=data.y_units,
                style=style,
                **freq_kwargs,
            )
        elif test_mode in ("startup", "laos"):
            # Startup and LAOS: time/strain vs stress (linear axes)
            result = plot_time_domain(
                _ensure_numpy(data.x),
                _ensure_numpy(data.y),
                x_units=data.x_units,
                y_units=data.y_units,
                style=style,
                **kwargs,
            )
        elif test_mode in ("rotation", "flow_curve") or data.x_units in [
            "1/s",
            "s^-1",
        ]:
            # Flow curve data
            result = plot_flow_curve(
                _ensure_numpy(data.x),
                _ensure_numpy(data.y),
                x_units=data.x_units,
                y_units=data.y_units,
                style=style,
                **kwargs,
            )
        else:
            # Time-domain data (relaxation, creep, etc.)
            result = plot_time_domain(
                _ensure_numpy(data.x),
                _ensure_numpy(data.y),
                x_units=data.x_units,
                y_units=data.y_units,
                style=style,
                **kwargs,
            )

        logger.debug("Figure created", plot_type="rheo_data")
        return result

    except Exception as e:
        logger.error(
            "Failed to generate rheo_data plot",
            plot_type="rheo_data",
            error=str(e),
            exc_info=True,
        )
        raise


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
    logger.debug("Generating plot", plot_type="time_domain", style=style)

    try:
        style_params = _apply_style(style)

        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        # VIS-010: Set font sizes on axes directly instead of mutating
        # global plt.rcParams (which permanently pollutes process state)

        # Plot data
        plot_kwargs = {
            "linewidth": style_params["lines.linewidth"],
            "marker": "o",
            "markersize": style_params["lines.markersize"],
            "markerfacecolor": "none",
            "markeredgewidth": 1.0,
        }
        plot_kwargs.update(kwargs)

        # VIZ-R6-003: Filter non-positive x before plotting when log_x is set,
        # otherwise t=0 anchors the x-axis at 0 and breaks the log scale display.
        x_plot, y_plot = x, y
        if log_x:
            pos_mask = np.isfinite(x) & (x > 0)
            if not np.all(pos_mask) and np.any(pos_mask):
                x_plot, y_plot = x[pos_mask], y[pos_mask]

        ax.plot(x_plot, y_plot, **plot_kwargs)

        # Set labels
        x_label = f"Time ({x_units})" if x_units else "Time"
        # VIS-P2-008 / F-035: Explicit unit sets prevent misclassification of edge cases
        if y_units:
            y_units_stripped = y_units.strip()
            if y_units_stripped in ("Pa", "kPa", "MPa", "GPa"):
                y_label = f"Stress ({y_units})"
            elif y_units_stripped in _VISCOSITY_UNITS:
                y_label = f"Viscosity ({y_units})"
            elif "Pa" in y_units_stripped and (
                "\u00b7s" in y_units_stripped
                or ".s" in y_units_stripped
                or "*s" in y_units_stripped
            ):
                y_label = f"Viscosity ({y_units})"
            else:
                y_label = f"y ({y_units})"
        else:
            y_label = "y"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Set scales
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        # Grid
        ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        fig.tight_layout()

        logger.debug("Figure created", plot_type="time_domain")
        return fig, ax

    except Exception as e:
        # VIZ-R6-002: Guard against fig not yet assigned (e.g. _apply_style raises)
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)
        logger.error(
            "Failed to generate time_domain plot",
            plot_type="time_domain",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_frequency_domain(
    x: np.ndarray,
    y: np.ndarray,
    x_units: str | None = None,
    y_units: str | None = None,
    style: str = "default",
    **kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """Plot frequency-domain rheological data (complex modulus).

    For complex data, creates two subplots for G' (storage modulus) and
    G'' (loss modulus). For real data, creates a single plot.

    Args:
        x: Frequency data
        y: Complex modulus data (G* = G' + iG'')
        x_units: Units for frequency axis
        y_units: Units for modulus
        style: Plotting style
        **kwargs: Additional keyword arguments for matplotlib plot.
            deformation_mode: str, optional — 'tension'/'bending'/'compression'
            causes labels to show E'/E'' instead of G'/G''.

    Returns:
        Tuple of (Figure, np.ndarray of Axes). For complex data the array has
        shape (2,); for real data it has shape (1,) so callers can always
        index the result uniformly (e.g. ``axes[0]``).
    """
    logger.debug("Generating plot", plot_type="frequency_domain", style=style)

    try:
        style_params = _apply_style(style)

        # VIS-010: Set font sizes on axes directly instead of mutating
        # global plt.rcParams (which permanently pollutes process state)

        # VIS-P1-004: Deformation-mode aware labels (E' vs G')
        kwargs = dict(kwargs)
        deformation_mode = kwargs.pop("deformation_mode", None)
        is_tensile = deformation_mode in ("tension", "bending", "compression")
        units = y_units or "Pa"
        storage_sym = "E'" if is_tensile else "G'"
        loss_sym = 'E"' if is_tensile else 'G"'

        if np.iscomplexobj(y):
            # Complex data - plot storage and loss on separate subplots
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

            # VIS-P0-001 / VIS-P0-002: Strip keys that are passed explicitly to avoid
            # "multiple values for keyword argument" TypeError.
            # VIZ-016: Prefer user-supplied label/color over the automatic defaults.
            user_storage_label = kwargs.pop("label", None)
            user_color = kwargs.pop("color", None)
            plot_kwargs_safe = {
                k: v for k, v in plot_kwargs.items() if k not in ("label", "color")
            }

            # Storage modulus
            x_gp, gp = _filter_positive(x, np.real(y), warn=True)
            axes[0].loglog(
                x_gp, gp, **plot_kwargs_safe, label=user_storage_label or storage_sym
            )
            axes[0].set_ylabel(f"{storage_sym} ({units})")
            axes[0].grid(
                True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE
            )
            axes[0].legend()

            # Loss modulus
            # VIZ-R6-001: Use distinct label for loss modulus subplot
            x_gpp, gpp = _filter_positive(x, np.imag(y), warn=True)
            loss_label = (
                f"{user_storage_label} (loss)" if user_storage_label else loss_sym
            )
            axes[1].loglog(
                x_gpp,
                gpp,
                **plot_kwargs_safe,
                label=loss_label,
                color=user_color or "C1",
            )
            axes[1].set_xlabel(
                f"Frequency ({x_units})" if x_units else "Frequency (rad/s)"
            )
            axes[1].set_ylabel(f"{loss_sym} ({units})")
            axes[1].grid(
                True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE
            )
            axes[1].legend()

            fig.tight_layout()
            logger.debug("Figure created", plot_type="frequency_domain")
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
            ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            fig.tight_layout()
            logger.debug("Figure created", plot_type="frequency_domain")
            # F-013: Wrap single ax in an array for consistent return type with complex branch
            return fig, np.array([ax])

    except Exception as e:
        # VIZ-R6-002: Guard against fig not yet assigned
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)
        logger.error(
            "Failed to generate frequency_domain plot",
            plot_type="frequency_domain",
            error=str(e),
            exc_info=True,
        )
        raise


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
    logger.debug("Generating plot", plot_type="flow_curve", style=style)

    try:
        style_params = _apply_style(style)

        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        # VIS-010: Set font sizes on axes directly instead of mutating
        # global plt.rcParams (which permanently pollutes process state)

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
            # VIZ-R6-005: Infer stress vs viscosity label from units
            if y_units:
                y_stripped = y_units.strip()
                if y_stripped in ("Pa", "kPa", "MPa", "GPa"):
                    y_label = f"Stress ({y_units})"
                else:
                    y_label = f"Viscosity ({y_units})"
            else:
                y_label = "Viscosity (Pa.s)"

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Grid
        ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        fig.tight_layout()

        logger.debug("Figure created", plot_type="flow_curve")
        return fig, ax

    except Exception as e:
        # VIZ-R6-002: Guard against fig not yet assigned
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)
        logger.error(
            "Failed to generate flow_curve plot",
            plot_type="flow_curve",
            error=str(e),
            exc_info=True,
        )
        raise


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
    logger.debug("Generating plot", plot_type="residuals", style=style)

    try:
        style_params = _apply_style(style)

        # VIS-010: Set font sizes on axes directly instead of mutating
        # global plt.rcParams (which permanently pollutes process state)

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
            logger.debug("Figure created", plot_type="residuals")
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
            logger.debug("Figure created", plot_type="residuals")
            return fig, ax

    except Exception as e:
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)  # prevent memory leak
        logger.error(
            "Failed to generate residuals plot",
            plot_type="residuals",
            error=str(e),
            exc_info=True,
        )
        raise


def compute_uncertainty_band(
    model_fn: Any,  # callable
    x_pred: np.ndarray,
    popt: np.ndarray,
    pcov: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute prediction uncertainty band via error propagation.

    Uses the formula: sigma_y(x) = sqrt(diag(J @ pcov @ J.T))
    where J is the Jacobian of the model with respect to parameters.

    For 95% confidence interval, the band is +/-1.96 * sigma_y(x).

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

    Note
    ----
    Complex-valued ``y_fit`` is not supported for uncertainty bands.
    If ``y_fit`` is complex, the function returns ``(y_fit, None, None)``
    and logs a debug message. Callers should check for ``None`` before
    using ``y_lower`` and ``y_upper``.
    """
    logger.debug(
        "Computing uncertainty band",
        n_points=len(x_pred),
        n_params=len(popt),
        confidence=confidence,
    )

    try:
        from scipy.stats import norm

        x_pred = np.asarray(x_pred, dtype=np.float64)
        popt = np.asarray(popt, dtype=np.float64)
        pcov = np.asarray(pcov, dtype=np.float64)

        # Compute y_fit — check complex BEFORE forcing float64 dtype,
        # since np.asarray(..., dtype=float64) silently truncates complex values.
        y_fit_raw = model_fn(x_pred, popt)
        if np.iscomplexobj(np.asarray(y_fit_raw)):
            logger.debug("Complex output detected, skipping uncertainty band")
            return np.asarray(y_fit_raw), None, None
        y_fit = np.asarray(y_fit_raw, dtype=np.float64)

        # Compute Jacobian: J[i, j] = dy[i] / dparam[j]
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
            logger.debug("Einsum failed for variance computation, returning None")
            return y_fit, None, None

        # VIS-P2-001: Guard against Inf/NaN from ill-conditioned covariance matrix
        if not np.all(np.isfinite(sigma_y)):
            logger.warning(
                "Non-finite uncertainty from ill-conditioned covariance matrix"
            )
            return y_fit, None, None

        # Compute z-score for confidence interval
        z = norm.ppf(1 - (1 - confidence) / 2)

        y_lower = y_fit - z * sigma_y
        y_upper = y_fit + z * sigma_y

        logger.debug("Uncertainty band computed successfully")
        return y_fit, y_lower, y_upper

    except Exception as e:
        logger.error(
            "Failed to compute uncertainty band",
            error=str(e),
            exc_info=True,
        )
        raise


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
    logger.debug("Generating plot", plot_type="fit_with_uncertainty", style=style)

    try:
        # VIZ-002: Reject complex inputs early — _filter_positive raises TypeError on complex
        # and log-scale operations are undefined. Callers should plot G'/G'' separately.
        if np.iscomplexobj(np.asarray(y_data)) or np.iscomplexobj(np.asarray(y_fit)):
            raise ValueError(
                "plot_fit_with_uncertainty does not support complex data. "
                "Plot G' and G'' components separately."
            )

        style_params = _apply_style(style)

        # Create figure if no axes provided
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        x_data = _ensure_numpy(x_data)
        y_data = _ensure_numpy(y_data)
        x_fit = _ensure_numpy(x_fit)
        y_fit = _ensure_numpy(y_fit)

        # F-030: Validate uncertainty bound lengths before attempting fill_between.
        # Log at error level so callers are clearly notified of the data problem.
        if y_lower is not None and y_upper is not None:
            if len(y_lower) != len(x_fit) or len(y_upper) != len(x_fit):
                logger.error(
                    "Uncertainty bounds length mismatch — bounds will not be plotted",
                    x_fit_len=len(x_fit),
                    y_lower_len=len(y_lower),
                    y_upper_len=len(y_upper),
                )
                y_lower = None
                y_upper = None

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
            # VIS-002: Filter positive x values for log-x scale
            x_mask = x_data > 0
            x_data_plot, y_data_plot = x_data[x_mask], y_data[x_mask]
            x_mask_fit = x_fit > 0
            x_fit_plot, y_fit_plot = x_fit[x_mask_fit], y_fit[x_mask_fit]
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

        logger.debug("Figure created", plot_type="fit_with_uncertainty")
        return fig, ax

    except Exception as e:
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)  # prevent memory leak (only when we created the figure)
        logger.error(
            "Failed to generate fit_with_uncertainty plot",
            plot_type="fit_with_uncertainty",
            error=str(e),
            exc_info=True,
        )
        raise


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
        File format. Supported: 'pdf', 'svg', 'png', 'eps', 'tiff', 'jpg', 'webp'.
        Auto-detected from file extension if None.
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
    - TIFF: Raster format, lossless, common in scientific publishing
    - JPG/JPEG: Raster format, lossy, good for photos
    - WEBP: Raster format, modern lossy/lossless, good for web

    DPI only affects raster formats (PNG, TIFF, JPG, WEBP). Vector formats
    (PDF, SVG, EPS) are resolution-independent.
    """
    logger.debug("Saving figure", filepath=str(filepath), format=format, dpi=dpi)

    try:
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
        # VIS-P3-002: Expanded to include all common matplotlib-supported raster formats
        supported_formats = {
            "pdf",
            "svg",
            "png",
            "eps",
            "tiff",
            "tif",
            "jpg",
            "jpeg",
            "webp",
        }
        format_lower = format.lower()
        if format_lower not in supported_formats:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported formats: {', '.join(sorted(supported_formats))}."
            )

        # VIS-PLT-001: Auto-create parent directory (consistent with save_hdf5
        # and export_spp_txt which both use mkdir). Raising OSError when the
        # directory is missing is surprising for callers saving to a new path.
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save figure
        fig.savefig(
            filepath, format=format_lower, dpi=dpi, bbox_inches=bbox_inches, **kwargs
        )

        resolved_path = filepath.resolve()
        logger.debug("Figure saved", filepath=str(resolved_path), format=format_lower)
        return resolved_path

    except Exception as e:
        logger.error(
            "Failed to save figure",
            filepath=str(filepath),
            error=str(e),
            exc_info=True,
        )
        raise
