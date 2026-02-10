"""Shared plotting utilities for RheoJAX example notebooks.

Provides standardized NLSQ uncertainty bands, ArviZ diagnostics,
and posterior predictive plots across all model families.
"""

import numpy as np


def _extract_y(result):
    """Extract y array from a predict result (handles RheoData and raw arrays)."""
    if hasattr(result, "y"):
        return np.asarray(result.y)
    return np.asarray(result)


# ---------------------------------------------------------------------------
# NLSQ Uncertainty Band
# ---------------------------------------------------------------------------


def compute_nlsq_uncertainty_band(model, x_pred, test_mode, confidence=0.95, **kwargs):
    """Compute NLSQ uncertainty band via error propagation from pcov.

    Uses the Jacobian-based formula: sigma_y = sqrt(diag(J @ pcov @ J.T))
    delegating to rheojax.visualization.plotter.compute_uncertainty_band.

    Args:
        model: Fitted RheoJAX model with _nlsq_result
        x_pred: X values for prediction (n_points,)
        test_mode: Test mode string (e.g. "flow_curve", "oscillation")
        confidence: Confidence level (default 0.95)
        **kwargs: Extra kwargs forwarded to model_function (e.g. gamma_dot)

    Returns:
        (y_fit, y_lower, y_upper) or (y_fit, None, None) if pcov unavailable
    """
    from rheojax.visualization.plotter import compute_uncertainty_band

    popt = model.popt_
    pcov = model.pcov_

    if popt is None or pcov is None:
        try:
            y_fit = _extract_y(model.predict(x_pred, test_mode=test_mode, **kwargs))
        except Exception:
            y_fit = _extract_y(model.predict(x_pred, **kwargs))
        return y_fit, None, None

    # Build a callable f(x, params) that wraps model_function
    def _model_fn(x, params):
        return model.model_function(x, params, test_mode=test_mode, **kwargs)

    try:
        y_fit, y_lower, y_upper = compute_uncertainty_band(
            _model_fn, x_pred, popt, pcov, confidence=confidence
        )
        return np.asarray(y_fit), y_lower, y_upper
    except Exception:
        try:
            y_fit = _extract_y(model.predict(x_pred, test_mode=test_mode, **kwargs))
        except Exception:
            y_fit = _extract_y(model.predict(x_pred, **kwargs))
        return y_fit, None, None


# ---------------------------------------------------------------------------
# Parameter Annotation
# ---------------------------------------------------------------------------


def format_param_annotation(model, param_names=None):
    """Format parameter annotation text box with uncertainties.

    Args:
        model: Fitted model
        param_names: List of parameter names (default: all)

    Returns:
        Multi-line string for matplotlib text box
    """
    if param_names is None:
        param_names = list(model.parameters.keys())

    uncertainties = model.get_parameter_uncertainties()
    lines = []
    for name in param_names:
        val = model.parameters.get_value(name)
        if uncertainties and name in uncertainties:
            se = uncertainties[name]
            # Use scientific notation for very small/large values
            if abs(val) > 0 and (abs(val) > 1e4 or abs(val) < 1e-2):
                lines.append(f"{name} = {val:.3e} \u00b1 {se:.2e}")
            else:
                lines.append(f"{name} = {val:.4g} \u00b1 {se:.3g}")
        else:
            if abs(val) > 0 and (abs(val) > 1e4 or abs(val) < 1e-2):
                lines.append(f"{name} = {val:.3e}")
            else:
                lines.append(f"{name} = {val:.4g}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NLSQ Fit Plot
# ---------------------------------------------------------------------------


def plot_nlsq_fit(
    x_data,
    y_data,
    model,
    test_mode,
    param_names=None,
    x_pred=None,
    confidence=0.95,
    log_scale=True,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(9, 6),
    band_color="C0",
    band_alpha=0.2,
    show_annotation=True,
    ax=None,
    **model_kwargs,
):
    """Plot NLSQ fit with uncertainty band and parameter annotation.

    Args:
        x_data: Measured x values
        y_data: Measured y values
        model: Fitted model
        test_mode: Test mode string
        param_names: Parameter names for annotation
        x_pred: X values for prediction line (default: auto logspace/linspace)
        confidence: Confidence level for band
        log_scale: Use log-log scale (True for flow_curve, oscillation, relaxation)
        xlabel, ylabel, title: Axis labels
        figsize: Figure size
        band_color: Color for fit line and band
        band_alpha: Alpha for band fill
        show_annotation: Show parameter text box
        ax: Existing axes (creates new figure if None)
        **model_kwargs: Extra kwargs for model_function (e.g. gamma_dot)

    Returns:
        (fig, ax) tuple
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # Build prediction x if not provided
    if x_pred is None:
        if log_scale and np.all(x_data > 0):
            x_pred = np.logspace(
                np.log10(x_data.min()) - 0.3,
                np.log10(x_data.max()) + 0.2,
                200,
            )
        else:
            margin = 0.05 * (x_data.max() - x_data.min())
            x_pred = np.linspace(x_data.min() - margin, x_data.max() + margin, 200)

    # Compute fit + uncertainty band
    y_fit, y_lower, y_upper = compute_nlsq_uncertainty_band(
        model, x_pred, test_mode, confidence=confidence, **model_kwargs
    )

    # Handle shape mismatch: some models return scalar/wrong-shape for fine grids
    y_fit = np.atleast_1d(np.asarray(y_fit)).ravel()
    if y_fit.shape[0] != x_pred.shape[0]:
        # Fall back to predicting at data points only
        x_pred = x_data
        y_fit, y_lower, y_upper = compute_nlsq_uncertainty_band(
            model, x_pred, test_mode, confidence=confidence, **model_kwargs
        )
        y_fit = np.atleast_1d(np.asarray(y_fit)).ravel()

    # Handle complex output (SAOS): use magnitude
    if np.iscomplexobj(y_fit):
        y_fit = np.abs(y_fit)
        y_lower = None
        y_upper = None

    # Plot
    plot_fn = ax.loglog if log_scale else ax.plot
    plot_fn(x_data, y_data, "ko", markersize=5, label="Data", zorder=3)
    plot_fn(x_pred, y_fit, "-", lw=2, color=band_color, label="NLSQ fit", zorder=2)

    if y_lower is not None and y_upper is not None:
        ax.fill_between(
            x_pred,
            y_lower,
            y_upper,
            alpha=band_alpha,
            color=band_color,
            label=f"{confidence*100:.0f}% CI",
            zorder=1,
        )

    # Annotation text box
    if show_annotation and param_names:
        text = format_param_annotation(model, param_names)
        props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7)
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=props,
            family="monospace",
        )

    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    return fig, ax


# ---------------------------------------------------------------------------
# Posterior Predictive Plot
# ---------------------------------------------------------------------------


def plot_posterior_predictive(
    x_data,
    y_data,
    model,
    result,
    test_mode,
    param_names,
    x_pred=None,
    n_draws=200,
    log_scale=True,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(9, 6),
    ci_color="C0",
    ci_alpha=0.3,
    ax=None,
    **model_kwargs,
):
    """Plot posterior predictive check with 95% credible interval.

    Args:
        x_data, y_data: Observed data
        model: Fitted model (parameters will be temporarily modified)
        result: BayesianResult with posterior_samples
        test_mode: Test mode string
        param_names: Parameter names to sample from posterior
        x_pred: Prediction x values (auto if None)
        n_draws: Number of posterior draws to use
        log_scale: Log-log scale
        xlabel, ylabel, title: Labels
        figsize: Figure size
        ci_color: Color for CI band
        ci_alpha: Alpha for CI fill
        ax: Existing axes
        **model_kwargs: Extra kwargs for predict

    Returns:
        (fig, ax) tuple
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    posterior = result.posterior_samples

    # Build prediction x
    if x_pred is None:
        if log_scale and np.all(x_data > 0):
            x_pred = np.logspace(
                np.log10(x_data.min()) - 0.3,
                np.log10(x_data.max()) + 0.2,
                100,
            )
        else:
            margin = 0.05 * (x_data.max() - x_data.min())
            x_pred = np.linspace(x_data.min() - margin, x_data.max() + margin, 100)

    n_available = len(posterior[param_names[0]])
    n_use = min(n_draws, n_available)

    pred_samples = []
    for i in range(n_use):
        for name in param_names:
            model.parameters.set_value(name, float(posterior[name][i]))
        pred_i = model.predict(x_pred, test_mode=test_mode, **model_kwargs)
        pred_arr = _extract_y(pred_i)
        if np.iscomplexobj(pred_arr):
            pred_arr = np.abs(pred_arr)
        pred_samples.append(pred_arr)

    pred_samples = np.array(pred_samples)
    pred_median = np.median(pred_samples, axis=0)
    pred_lo = np.percentile(pred_samples, 2.5, axis=0)
    pred_hi = np.percentile(pred_samples, 97.5, axis=0)

    # Plot
    plot_fn = ax.loglog if log_scale else ax.plot
    ax.fill_between(
        x_pred, pred_lo, pred_hi, alpha=ci_alpha, color=ci_color, label="95% CI"
    )
    plot_fn(x_pred, pred_median, "-", lw=2, color=ci_color, label="Posterior median")
    plot_fn(x_data, y_data, "ko", markersize=5, label="Data", zorder=3)

    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")
    ax.set_title(title or "Posterior Predictive Check")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    return fig, ax


# ---------------------------------------------------------------------------
# ArviZ Diagnostics
# ---------------------------------------------------------------------------


def filter_nondegenerate_params(idata, var_names):
    """Filter out degenerate parameters (constant chains) from var_names.

    Degenerate parameters (range < 1e-10) cause KDE failures in ArviZ.

    Args:
        idata: ArviZ InferenceData
        var_names: List of parameter names

    Returns:
        Filtered list of non-degenerate parameter names
    """
    filtered = []
    for name in var_names:
        try:
            vals = idata.posterior[name].values.ravel()
            if np.ptp(vals) > 1e-10:
                filtered.append(name)
        except (KeyError, AttributeError):
            pass
    return filtered


def plot_arviz_diagnostics(result, param_names, fast_mode=False):
    """Plot all 6 ArviZ diagnostic plots with degenerate parameter filtering.

    Plots: trace, pair, forest, energy, autocorrelation, rank.

    Args:
        result: BayesianResult
        param_names: Parameter names
        fast_mode: If True and single chain, skip energy/rank

    Returns:
        dict of {name: fig} for each diagnostic plot
    """
    import matplotlib.pyplot as plt

    try:
        import arviz as az
    except ImportError:
        return {}

    if result is None:
        return {}

    idata = result.to_inference_data()
    figs = {}

    # Filter degenerate params
    var_names = filter_nondegenerate_params(idata, param_names)
    if not var_names:
        return figs

    num_chains = result.num_chains if hasattr(result, "num_chains") else 1

    # 1. Trace plot
    try:
        axes = az.plot_trace(idata, var_names=var_names, figsize=(12, 2.5 * len(var_names)))
        fig = axes.ravel()[0].figure
        fig.suptitle("Trace Plots", fontsize=14, y=1.02)
        plt.tight_layout()
        figs["trace"] = fig
    except Exception:
        pass

    # 2. Pair plot
    if len(var_names) >= 2:
        try:
            axes = az.plot_pair(
                idata,
                var_names=var_names,
                kind="scatter",
                divergences=True,
                figsize=(10, 10),
            )
            if hasattr(axes, "ravel"):
                fig = axes.ravel()[0].figure
            else:
                fig = axes.figure
            fig.suptitle("Parameter Correlations", fontsize=14, y=1.02)
            plt.tight_layout()
            figs["pair"] = fig
        except Exception:
            pass

    # 3. Forest plot
    try:
        axes = az.plot_forest(
            idata,
            var_names=var_names,
            combined=True,
            hdi_prob=0.95,
            figsize=(10, max(3, 1.2 * len(var_names))),
        )
        if hasattr(axes, "ravel"):
            fig = axes.ravel()[0].figure
        else:
            fig = axes.figure
        plt.tight_layout()
        figs["forest"] = fig
    except Exception:
        pass

    # 4. Energy plot (requires > 1 chain for meaningful comparison, but works with 1)
    if not (fast_mode and num_chains == 1):
        try:
            ax_energy = az.plot_energy(idata, figsize=(10, 4))
            if hasattr(ax_energy, "figure"):
                fig = ax_energy.figure
            elif hasattr(ax_energy, "ravel"):
                fig = ax_energy.ravel()[0].figure
            else:
                fig = plt.gcf()
            fig.suptitle("Energy Diagnostic", fontsize=14, y=1.02)
            plt.tight_layout()
            figs["energy"] = fig
        except Exception:
            pass

    # 5. Autocorrelation plot
    try:
        axes = az.plot_autocorr(
            idata,
            var_names=var_names,
            figsize=(12, 2.5 * len(var_names)),
        )
        if hasattr(axes, "ravel"):
            fig = axes.ravel()[0].figure
        else:
            fig = axes.figure
        fig.suptitle("Autocorrelation", fontsize=14, y=1.02)
        plt.tight_layout()
        figs["autocorr"] = fig
    except Exception:
        pass

    # 6. Rank plot (requires >= 2 chains for meaningful interpretation)
    if num_chains >= 2 or not fast_mode:
        try:
            axes = az.plot_rank(
                idata,
                var_names=var_names,
                figsize=(12, 2.5 * len(var_names)),
            )
            if hasattr(axes, "ravel"):
                fig = axes.ravel()[0].figure
            else:
                fig = axes.figure
            fig.suptitle("Rank Plots", fontsize=14, y=1.02)
            plt.tight_layout()
            figs["rank"] = fig
        except Exception:
            pass

    return figs


def display_arviz_diagnostics(result, param_names, fast_mode=False):
    """Plot and display all ArviZ diagnostics, closing figures after display.

    This is the convenience function for notebooks — calls plot_arviz_diagnostics
    then displays and closes each figure.

    Args:
        result: BayesianResult
        param_names: Parameter names
        fast_mode: Skip energy/rank for single-chain fast runs
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    figs = plot_arviz_diagnostics(result, param_names, fast_mode=fast_mode)
    for name, fig in figs.items():
        display(fig)
        plt.close(fig)
