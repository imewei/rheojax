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
    _modulus_labels,
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
        test_mode = (data.metadata or {}).get("test_mode", "")

        # Determine if log scale is appropriate
        log_x = False
        log_y = False

        # For long time ranges, log scale is often more informative
        x_data = _ensure_numpy(data.x)
        positive_x = x_data[x_data > 0]
        if len(positive_x) > 0:
            x_range = np.max(positive_x) / np.min(positive_x)
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
        # VIZ-R6-004: Close figure on error to prevent memory leak
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)
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

        # VIS-P1-004: Deformation-mode aware labels
        storage_label, loss_label, _generic = _modulus_labels(data)
        # Pop deformation_mode so it doesn't leak to matplotlib
        deformation_mode = kwargs.pop("deformation_mode", None)
        if deformation_mode is None:
            deformation_mode = getattr(data, "deformation_mode", None) or (
                data.metadata or {}
            ).get("deformation_mode")

        if separate_axes and np.iscomplexobj(y_data):
            # Two separate axes for storage/loss modulus
            freq_kwargs = dict(kwargs)
            if deformation_mode is not None:
                freq_kwargs["deformation_mode"] = deformation_mode
            fig, axes = plot_frequency_domain(
                x_data,
                y_data,
                x_units=data.x_units,
                y_units=data.y_units,
                style=style,
                **freq_kwargs,
            )

            axes[0].set_title(f"Storage Modulus ({storage_label.split(' ')[0]})")
            axes[1].set_title(f"Loss Modulus ({loss_label.split(' ')[0]})")

            logger.debug("Figure created", plot_type="modulus_frequency")
            return fig, axes
        else:
            # Single axis (either real data or combined plot)
            style_params = _apply_style(style)

            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

            # VIS-P0-001: Apply font sizes per-axes (not global rcParams)
            ax.xaxis.label.set_fontsize(style_params["axes.labelsize"])
            ax.yaxis.label.set_fontsize(style_params["axes.labelsize"])
            ax.tick_params(axis="x", labelsize=style_params["xtick.labelsize"])
            ax.tick_params(axis="y", labelsize=style_params["ytick.labelsize"])

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
                # VIZ-003: strip label/color from plot_kwargs to avoid TypeError on duplicate kwargs
                plot_kwargs_safe = {
                    k: v for k, v in plot_kwargs.items() if k not in ("label", "color")
                }
                ax.loglog(x_gp, gp, **plot_kwargs_safe, label=storage_label)
                ax.loglog(x_gpp, gpp, **plot_kwargs_safe, label=loss_label, color="C1")
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
        # VIZ-R6-004: Close figure on error to prevent memory leak
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)
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

    if not datasets:
        raise ValueError("plot_mastercurve requires at least one dataset")

    try:
        style_params = _apply_style(style)

        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        # VIS-P0-001: Apply font sizes per-axes (not global rcParams)
        ax.xaxis.label.set_fontsize(style_params["axes.labelsize"])
        ax.yaxis.label.set_fontsize(style_params["axes.labelsize"])
        ax.tick_params(axis="x", labelsize=style_params["xtick.labelsize"])
        ax.tick_params(axis="y", labelsize=style_params["ytick.labelsize"])

        # Get reference temperature
        if reference_temp is None:
            reference_temp = (datasets[0].metadata or {}).get("temperature", 25)

        # Plot each dataset
        colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))

        for i, data in enumerate(datasets):
            temp = (data.metadata or {}).get("temperature", None)
            x_data = _ensure_numpy(data.x)
            y_data = _ensure_numpy(data.y)

            # VIS-P0-003: strip keys that are passed explicitly to avoid conflicts
            mc_kwargs = {k: v for k, v in kwargs.items() if k not in ("color", "label")}

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
                # VIS-P2-004: Plot G' (storage modulus)
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
                    **mc_kwargs,
                )
                # VIS-P2-004: Plot G'' (loss modulus) with square markers
                x_filt_pp, y_filt_pp = _filter_positive(
                    x_shifted, np.imag(y_data), warn=False
                )
                if len(x_filt_pp) > 0:
                    ax.loglog(
                        x_filt_pp,
                        y_filt_pp,
                        "s",
                        color=colors[i],
                        alpha=0.6,
                        markersize=style_params["lines.markersize"],
                        markerfacecolor="none",
                        markeredgewidth=1.0,
                        label=f"{label} (loss)",
                        **mc_kwargs,
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
                    **mc_kwargs,
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
        # VIZ-013: use generic "Modulus" label for complex data (both G' and G'' plotted)
        # Use deformation-mode aware labels
        # _modulus_labels() already embeds units (e.g. "G' (Pa)"), so use directly
        mc_storage_label, _, mc_generic_label = _modulus_labels(datasets[0])
        has_complex = any(np.iscomplexobj(_ensure_numpy(d.y)) for d in datasets)
        if has_complex:
            ax.set_ylabel(mc_generic_label)
        else:
            ax.set_ylabel(mc_storage_label)
        ax.set_title(f"Master Curve (T_ref = {reference_temp}C)")
        ax.legend(loc="best", fontsize=style_params["legend.fontsize"])
        ax.grid(True, which="both", alpha=0.3, linestyle="--")

        fig.tight_layout()

        logger.debug("Figure created", plot_type="mastercurve")
        return fig, ax

    except Exception as e:
        # VIZ-R6-004: Close figure on error to prevent memory leak
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)
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

        x_data = _ensure_numpy(data.x)
        y_data = _ensure_numpy(data.y)
        y_pred = _ensure_numpy(predictions)

        # VIS-P1-005: Validate that data and predictions have matching shapes
        if len(y_data) != len(y_pred):
            raise ValueError(
                f"Data and predictions shape mismatch: data={y_data.shape}, predictions={y_pred.shape}"
            )

        # Deformation-mode aware labels (E'/E'' for DMTA, G'/G'' for shear)
        fit_storage_label, fit_loss_label, _ = _modulus_labels(data)

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
                    f"{fit_storage_label} ({data.y_units})"
                    if data.y_units
                    else f"{fit_storage_label} (Pa)"
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
                    f"{fit_loss_label} ({data.y_units})"
                    if data.y_units
                    else f"{fit_loss_label} (Pa)"
                )
                axes[0, 1].legend()
                axes[0, 1].grid(True, which="both", alpha=0.3, linestyle="--")

                # G' residuals
                # F-020: Use max(|data|) as fallback denominator to avoid huge % residuals
                residuals_gp = np.real(y_data) - np.real(y_pred)
                denom_fallback_gp = np.maximum(np.max(np.abs(np.real(y_data))), 1e-12)
                gp_denom = np.where(
                    np.abs(np.real(y_data)) > 1e-12, np.real(y_data), denom_fallback_gp
                )
                # VIZ-011: apply the same positive mask used when plotting G' data
                # R6-VIZ-001: Match data panel filter (y_data > 0 only), not
                # additionally y_pred > 0 which makes the residual panel shorter
                pos_mask_gp = np.isfinite(np.real(y_data)) & (np.real(y_data) > 0)
                axes[1, 0].semilogx(
                    x_data[pos_mask_gp],
                    (residuals_gp / gp_denom * 100)[pos_mask_gp],
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
                axes[1, 0].set_ylabel(f"{fit_storage_label} Residuals (%)")
                axes[1, 0].grid(True, alpha=0.3, linestyle="--")

                # G'' residuals
                # F-020: Use max(|data|) as fallback denominator to avoid huge % residuals
                residuals_gpp = np.imag(y_data) - np.imag(y_pred)
                denom_fallback_gpp = np.maximum(np.max(np.abs(np.imag(y_data))), 1e-12)
                gpp_denom = np.where(
                    np.abs(np.imag(y_data)) > 1e-12, np.imag(y_data), denom_fallback_gpp
                )
                # VIZ-011: apply the same positive mask used when plotting G'' data
                pos_mask_gpp = np.isfinite(np.imag(y_data)) & (np.imag(y_data) > 0)
                axes[1, 1].semilogx(
                    x_data[pos_mask_gpp],
                    (residuals_gpp / gpp_denom * 100)[pos_mask_gpp],
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
                axes[1, 1].set_ylabel(f"{fit_loss_label} Residuals (%)")
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
                    f"{fit_storage_label} ({data.y_units})"
                    if data.y_units
                    else f"{fit_storage_label} (Pa)"
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
                    f"{fit_loss_label} ({data.y_units})"
                    if data.y_units
                    else f"{fit_loss_label} (Pa)"
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

                # VIZ-012: infer log scale from domain / test_mode metadata
                is_log = getattr(data, "domain", None) == "frequency" or (
                    data.metadata or {}
                ).get("test_mode") in ("oscillation", "rotation", "flow_curve")

                # VIZ-R6-006: Filter non-positive values BEFORE plotting when log
                # scale will be applied, to prevent blank axes from t=0 or y<=0.
                xd, yd, xp, yp = x_data, y_data, x_data, y_pred
                if is_log:
                    pos_mask = (
                        np.isfinite(y_data)
                        & (y_data > 0)
                        & np.isfinite(x_data)
                        & (x_data > 0)
                    )
                    if not np.all(pos_mask) and np.any(pos_mask):
                        xd, yd = x_data[pos_mask], y_data[pos_mask]
                    pred_mask = (
                        np.isfinite(y_pred)
                        & (y_pred > 0)
                        & np.isfinite(x_data)
                        & (x_data > 0)
                    )
                    if not np.all(pred_mask) and np.any(pred_mask):
                        xp, yp = x_data[pred_mask], y_pred[pred_mask]

                ax.plot(
                    xd,
                    yd,
                    "o",
                    label="Data",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                )
                ax.plot(
                    xp,
                    yp,
                    "-",
                    label="Model",
                    linewidth=style_params["lines.linewidth"],
                )

                if is_log:
                    ax.set_xscale("log")
                    ax.set_yscale("log")

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
        # VIZ-R6-004: Close figure on error to prevent memory leak
        _fig = locals().get("fig")
        if _fig is not None:
            plt.close(_fig)
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

        # Update line widths and marker sizes (tolerance for float comparison)
        default_lw = plt.rcParams["lines.linewidth"]
        default_ms = plt.rcParams["lines.markersize"]
        for line in ax.get_lines():
            if abs(line.get_linewidth() - default_lw) < 0.01:
                line.set_linewidth(style_params["lines.linewidth"])
            if abs(line.get_markersize() - default_ms) < 0.01:
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
