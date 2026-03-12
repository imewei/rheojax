"""Unified fit visualization for NLSQ and Bayesian results.

This module provides the FitPlotter class for creating publication-quality
visualizations of model fitting results, including:
- NLSQ fits with Jacobian-based uncertainty bands
- Bayesian posterior predictive with credible intervals
- ArviZ diagnostic suites (6-plot MCMC diagnostics)
- Side-by-side NLSQ vs Bayesian comparison
- Parameter summary tables
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger
from rheojax.visualization.plotter import (
    _apply_style,
    _ensure_numpy,
    _filter_positive,
    compute_uncertainty_band,
    save_figure,
)

jax, jnp = safe_import_jax()

logger = get_logger(__name__)

# Grid styling (consistent with plotter.py)
_GRID_ALPHA = 0.3
_GRID_LINESTYLE = "--"


def compute_credible_band(
    model_fn: Any,
    x_pred: np.ndarray,
    posterior_samples: dict[str, np.ndarray],
    param_names: list[str],
    credible_level: float = 0.95,
    test_mode: Any = None,
    max_draws: int = 500,
    **model_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Bayesian credible interval from posterior samples.

    Uses jax.vmap for GPU-accelerated evaluation of the model over all
    posterior draws, then extracts percentile-based credible bands.

    Parameters
    ----------
    model_fn : callable
        Model function with signature ``model_fn(X, params, test_mode, **kwargs)``.
        This is typically ``model.model_function``.
    x_pred : ndarray
        Prediction grid, shape ``(n_points,)``.
    posterior_samples : dict[str, ndarray]
        Parameter name to flat sample array, from ``BayesianResult.posterior_samples``.
    param_names : list[str]
        Ordered parameter names matching the model's parameter array layout.
        Noise parameters (sigma, sigma_real, sigma_imag) are automatically excluded.
    credible_level : float
        Credible interval level (default 0.95 for 95% CI).
    test_mode : TestMode or str, optional
        Protocol mode for model evaluation (relaxation, oscillation, etc.).
    max_draws : int
        Maximum number of posterior draws to use. If the total number of
        samples exceeds this, a random subset is selected for efficiency.
    **model_kwargs
        Additional keyword arguments forwarded to ``model_fn`` (e.g.,
        ``gamma_dot``, ``sigma_0``).

    Returns
    -------
    y_median : ndarray
        Posterior median predictions, shape ``(n_points,)`` or complex.
    y_lower : ndarray
        Lower bound of credible interval.
    y_upper : ndarray
        Upper bound of credible interval.

    Notes
    -----
    For complex-valued model outputs (oscillation data), the credible band
    is computed separately for the real (G') and imaginary (G'') parts.
    The returned arrays are complex: ``real = G' band``, ``imag = G'' band``.
    """
    logger.debug(
        "Computing credible band",
        n_params=len(param_names),
        credible_level=credible_level,
        max_draws=max_draws,
    )

    # Filter out noise parameters (startswith covers sigma_0, sigma_obs, etc.)
    model_param_names = [p for p in param_names if not p.startswith("sigma")]

    if not model_param_names:
        raise ValueError(
            "No model parameters found after excluding noise parameters. "
            f"param_names={param_names}"
        )

    # Stack posterior samples into (n_draws, n_params) array
    first_key = model_param_names[0]
    n_total = len(posterior_samples[first_key])

    # Subsample if needed
    if n_total > max_draws:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, size=max_draws, replace=False)
    else:
        indices = np.arange(n_total)

    samples_matrix = np.column_stack(
        [np.asarray(posterior_samples[name])[indices] for name in model_param_names]
    )

    # Convert to JAX arrays for vmap
    x_jax = jnp.asarray(x_pred)
    samples_jax = jnp.asarray(samples_matrix)

    # Vectorized model evaluation over posterior draws
    try:

        def _eval_single(params):
            return model_fn(x_jax, params, test_mode, **model_kwargs)

        predictions = jax.vmap(_eval_single)(samples_jax)
        predictions = np.asarray(predictions)
    except Exception:
        # Fallback to sequential evaluation if vmap fails
        # (e.g., model uses dynamic shapes incompatible with vmap)
        logger.debug("vmap failed, falling back to sequential evaluation")
        predictions = np.array(
            [
                np.asarray(model_fn(x_jax, samples_jax[i], test_mode, **model_kwargs))
                for i in range(len(samples_jax))
            ]
        )

    # Compute percentiles
    alpha = 1 - credible_level

    if np.iscomplexobj(predictions):
        # Handle complex output (oscillation): compute bands for G' and G'' separately
        real_preds = np.real(predictions)
        imag_preds = np.imag(predictions)

        median_real = np.median(real_preds, axis=0)
        median_imag = np.median(imag_preds, axis=0)
        lower_real = np.percentile(real_preds, 100 * alpha / 2, axis=0)
        upper_real = np.percentile(real_preds, 100 * (1 - alpha / 2), axis=0)
        lower_imag = np.percentile(imag_preds, 100 * alpha / 2, axis=0)
        upper_imag = np.percentile(imag_preds, 100 * (1 - alpha / 2), axis=0)

        y_median = median_real + 1j * median_imag
        y_lower = lower_real + 1j * lower_imag
        y_upper = upper_real + 1j * upper_imag
    else:
        y_median = np.median(predictions, axis=0)
        y_lower = np.percentile(predictions, 100 * alpha / 2, axis=0)
        y_upper = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

    logger.debug("Credible band computed", n_draws=len(indices))
    return y_median, y_lower, y_upper


def generate_diagnostic_suite(
    bayesian_result: Any,
    var_names: list[str] | None = None,
    style: str = "default",
    output_dir: str | Path | None = None,
    prefix: str = "mcmc",
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> dict[str, Figure | Path]:
    """Generate complete ArviZ MCMC diagnostic suite (6 plots).

    Produces the standard set of MCMC diagnostics:
    1. **Pair plot** — parameter correlations and marginals
    2. **Forest plot** — parameter estimates with 95% HDI
    3. **Energy plot** — NUTS energy diagnostics (E-BFMI)
    4. **Autocorrelation plot** — chain mixing quality
    5. **Rank plot** — convergence assessment
    6. **ESS plot** — effective sample size over iterations

    Parameters
    ----------
    bayesian_result : BayesianResult
        Output from ``model.fit_bayesian()``. Must have ``mcmc`` attribute
        for ArviZ conversion.
    var_names : list[str], optional
        Parameter names to include. If None, uses all non-noise parameters.
        Degenerate parameters (range < 1e-10) are automatically filtered
        to prevent KDE crashes in ArviZ.
    style : str
        Plot style ('default', 'publication', 'presentation').
    output_dir : str or Path, optional
        Directory for saving plots. If None, figures are returned without saving.
    prefix : str
        Filename prefix for saved plots (default: 'mcmc').
    formats : tuple[str, ...]
        Output file formats (default: ('pdf', 'png')).
    dpi : int
        Resolution for raster formats (default: 300).

    Returns
    -------
    dict[str, Figure or Path]
        If ``output_dir`` is None: mapping of plot type to Figure objects.
        If ``output_dir`` is set: mapping of ``'{type}_{format}'`` to file Paths.

    Raises
    ------
    ImportError
        If ArviZ is not installed.
    ValueError
        If ``bayesian_result.mcmc`` is None (required for ArviZ conversion).
    """
    try:
        import arviz as az
    except ImportError as err:
        raise ImportError(
            "ArviZ is required for diagnostic plots. Install with: uv add arviz"
        ) from err

    logger.debug("Generating diagnostic suite", prefix=prefix, style=style)

    if bayesian_result.mcmc is None:
        raise ValueError(
            "BayesianResult.mcmc is None — cannot convert to ArviZ InferenceData. "
            "Ensure fit_bayesian() was run with store_mcmc=True (the default)."
        )

    # Convert to ArviZ InferenceData
    trace = bayesian_result.to_inference_data()

    # Filter variable names
    if var_names is None:
        var_names = [v for v in trace.posterior.data_vars if not v.startswith("sigma")]

    # Filter degenerate parameters (range < 1e-10) to prevent KDE crashes
    filtered_vars = []
    for v in var_names:
        vals = trace.posterior[v].values.ravel()
        if np.ptp(vals) > 1e-10:
            filtered_vars.append(v)
        else:
            logger.debug(
                "Skipping degenerate parameter in diagnostics",
                param=v,
                ptp=float(np.ptp(vals)),
            )
    var_names = filtered_vars

    if not var_names:
        logger.warning("No non-degenerate parameters for diagnostic plots")
        return {}

    _apply_style(style)
    diagnostics: dict[str, Figure] = {}

    # 1. Pair plot (parameter correlations)
    try:
        axes = az.plot_pair(
            trace, var_names=var_names, marginals=True, divergences=True
        )
        fig_pair = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes.figure
        diagnostics["pair"] = fig_pair
    except Exception as e:
        logger.warning("Pair plot failed", error=str(e))

    # 2. Forest plot (parameter estimates with 95% HDI)
    try:
        axes = az.plot_forest(trace, var_names=var_names, combined=True, hdi_prob=0.95)
        fig_forest = (
            axes.ravel()[0].figure if hasattr(axes, "ravel") else axes[0].figure
        )
        diagnostics["forest"] = fig_forest
    except Exception as e:
        logger.warning("Forest plot failed", error=str(e))

    # 3. Energy plot (NUTS diagnostics)
    try:
        axes = az.plot_energy(trace)
        fig_energy = (
            axes.ravel()[0].figure
            if hasattr(axes, "ravel")
            else (axes.figure if hasattr(axes, "figure") else axes[0].figure)
        )
        diagnostics["energy"] = fig_energy
    except Exception as e:
        logger.warning("Energy plot failed", error=str(e))

    # 4. Autocorrelation plot
    try:
        axes = az.plot_autocorr(trace, var_names=var_names)
        fig_autocorr = (
            axes.ravel()[0].figure if hasattr(axes, "ravel") else axes[0].figure
        )
        diagnostics["autocorr"] = fig_autocorr
    except Exception as e:
        logger.warning("Autocorrelation plot failed", error=str(e))

    # 5. Rank plot (convergence check)
    try:
        axes = az.plot_rank(trace, var_names=var_names)
        fig_rank = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes[0].figure
        diagnostics["rank"] = fig_rank
    except Exception as e:
        logger.warning("Rank plot failed", error=str(e))

    # 6. ESS plot (effective sample size)
    try:
        axes = az.plot_ess(trace, var_names=var_names, kind="local")
        fig_ess = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes[0].figure
        diagnostics["ess"] = fig_ess
    except Exception as e:
        logger.warning("ESS plot failed", error=str(e))

    # Save if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}
        for plot_type, fig in diagnostics.items():
            for fmt in formats:
                filepath = output_dir / f"{prefix}_{plot_type}.{fmt}"
                save_figure(fig, filepath, format=fmt, dpi=dpi)
                saved_paths[f"{plot_type}_{fmt}"] = filepath
            plt.close(fig)

        logger.debug("Diagnostic suite saved", output_dir=str(output_dir))
        return saved_paths

    logger.debug("Diagnostic suite generated", n_plots=len(diagnostics))
    return diagnostics


class FitPlotter:
    """Unified plotting for NLSQ and Bayesian fit results.

    Provides methods for creating publication-quality fit visualizations
    that handle all RheoJAX data types (scalar, complex oscillation, 2D).

    All methods are stateless — data and results are passed explicitly.
    Style is controlled via the ``style`` parameter consistent with
    the rest of ``rheojax.visualization``.

    Examples
    --------
    NLSQ fit with uncertainty band:

    >>> plotter = FitPlotter()
    >>> fig, axes = plotter.plot_nlsq(
    ...     x_data, y_data, fit_result, model,
    ...     confidence=0.95, show_residuals=True
    ... )

    Bayesian posterior predictive:

    >>> fig, axes = plotter.plot_bayesian(
    ...     x_data, y_data, bayesian_result, model,
    ...     credible_level=0.95
    ... )
    """

    def plot_nlsq(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fit_result: Any,
        model: Any,
        confidence: float = 0.95,
        show_residuals: bool = True,
        show_uncertainty: bool = True,
        n_pred_points: int = 200,
        style: str = "default",
        deformation_mode: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray]:
        """Plot NLSQ fit with uncertainty band and optional residuals.

        Creates a comprehensive fit visualization with:
        - Data points (scatter)
        - Fitted curve (solid line)
        - Uncertainty band from covariance matrix (shaded, if available)
        - Optional residuals subplot

        For complex oscillation data, creates separate panels for G'/G''
        (or E'/E'' based on deformation_mode), each with its own uncertainty
        band and residuals.

        Parameters
        ----------
        x_data : ndarray
            Independent variable (time, frequency, shear rate).
        y_data : ndarray
            Dependent variable (stress, complex modulus, viscosity).
            Can be complex for oscillation data.
        fit_result : FitResult
            Output from ``model.fit()``. Must have ``optimization_result``
            with ``pcov`` for uncertainty bands.
        model : BaseModel
            Fitted model instance. Used for ``model_function`` evaluation
            on a denser prediction grid.
        confidence : float
            Confidence level for uncertainty band (default: 0.95).
        show_residuals : bool
            If True, add residuals subplot(s) below fit panel(s).
        show_uncertainty : bool
            If True and covariance is available, show uncertainty band.
        n_pred_points : int
            Number of points for the smooth prediction curve (default: 200).
        style : str
            Plot style ('default', 'publication', 'presentation').
        deformation_mode : str, optional
            'tension'/'bending'/'compression' for E'/E'' labels.
        **kwargs
            Additional arguments forwarded to model_function (e.g., test_mode).

        Returns
        -------
        tuple[Figure, Axes or ndarray]
            Figure and axes. For complex + residuals: 2x2 ndarray of Axes.
            For scalar + residuals: 1D ndarray of 2 Axes.
            For no residuals: single Axes or 1D ndarray.
        """
        logger.debug("Plotting NLSQ fit", style=style, confidence=confidence)

        x_data = _ensure_numpy(x_data)
        y_data = _ensure_numpy(y_data)
        style_params = _apply_style(style)
        is_complex = np.iscomplexobj(y_data)

        # Build prediction grid
        x_pred = self._make_pred_grid(x_data, n_pred_points)

        # Get covariance and optimal params
        pcov = None
        popt = None
        if fit_result.optimization_result is not None:
            pcov = fit_result.optimization_result.pcov
            popt = np.asarray(fit_result.optimization_result.x, dtype=np.float64)

        # Resolve test_mode
        test_mode = kwargs.pop("test_mode", None)
        if test_mode is None:
            test_mode = getattr(model, "_test_mode", None)
            if test_mode is None and fit_result.protocol:
                test_mode = fit_result.protocol

        # Compute predictions on dense grid
        try:
            y_pred_dense = np.asarray(
                model.model_function(
                    jnp.asarray(x_pred), jnp.asarray(popt), test_mode, **kwargs
                )
            )
        except Exception:
            # Fallback: use model.predict
            y_pred_dense = _ensure_numpy(model.predict(x_pred))

        # Compute predictions at data points (for residuals)
        y_pred_data = None
        if show_residuals and fit_result.fitted_curve is not None:
            y_pred_data = _ensure_numpy(fit_result.fitted_curve)
        elif show_residuals:
            try:
                y_pred_data = np.asarray(
                    model.model_function(
                        jnp.asarray(x_data), jnp.asarray(popt), test_mode, **kwargs
                    )
                )
            except Exception:
                y_pred_data = _ensure_numpy(model.predict(x_data))

        # Compute uncertainty band (NLSQ only works for real-valued output)
        y_lower, y_upper = None, None
        if (
            show_uncertainty
            and pcov is not None
            and popt is not None
            and not is_complex
        ):
            try:

                def _model_for_band(x, params):
                    return np.asarray(
                        model.model_function(
                            jnp.asarray(x), jnp.asarray(params), test_mode, **kwargs
                        )
                    )

                _, y_lower, y_upper = compute_uncertainty_band(
                    _model_for_band, x_pred, popt, pcov, confidence=confidence
                )
            except Exception as e:
                logger.debug("Uncertainty band computation failed", error=str(e))

        # Modulus labels
        storage_label, loss_label, _ = "G'", 'G"', "Modulus"
        if deformation_mode in ("tension", "bending", "compression"):
            storage_label, loss_label = "E'", 'E"'

        model_name = fit_result.model_name or fit_result.model_class_name or ""
        band_label = f"{int(confidence * 100)}% CI"

        if is_complex:
            return self._plot_complex_fit(
                x_data,
                y_data,
                x_pred,
                y_pred_dense,
                y_pred_data,
                y_lower=None,
                y_upper=None,  # NLSQ uncertainty not supported for complex
                show_residuals=show_residuals,
                storage_label=storage_label,
                loss_label=loss_label,
                model_name=model_name,
                band_label=band_label,
                style_params=style_params,
            )
        else:
            return self._plot_scalar_fit(
                x_data,
                y_data,
                x_pred,
                y_pred_dense,
                y_pred_data,
                y_lower=y_lower,
                y_upper=y_upper,
                show_residuals=show_residuals,
                model_name=model_name,
                band_label=band_label,
                style_params=style_params,
                log_x=self._infer_log_x(fit_result),
                log_y=self._infer_log_y(fit_result),
            )

    def plot_bayesian(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        bayesian_result: Any,
        model: Any,
        credible_level: float = 0.95,
        max_draws: int = 500,
        show_nlsq_overlay: bool = False,
        fit_result: Any = None,
        nlsq_confidence: float = 0.95,
        n_pred_points: int = 200,
        show_residuals: bool = False,
        style: str = "default",
        deformation_mode: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray]:
        """Plot Bayesian posterior predictive with credible interval.

        Creates a visualization showing:
        - Data points (scatter)
        - Posterior median curve (solid line)
        - Credible interval band (shaded region)
        - Optional NLSQ overlay for frequentist comparison

        Parameters
        ----------
        x_data : ndarray
            Independent variable.
        y_data : ndarray
            Dependent variable (can be complex for oscillation).
        bayesian_result : BayesianResult
            Output from ``model.fit_bayesian()``.
        model : BaseModel
            Fitted model instance.
        credible_level : float
            Credible interval level (default: 0.95).
        max_draws : int
            Maximum posterior draws for band computation (default: 500).
        show_nlsq_overlay : bool
            If True, overlay the NLSQ fit and its confidence band.
            Requires ``fit_result`` to be provided.
        fit_result : FitResult, optional
            NLSQ fit result for overlay. Required if ``show_nlsq_overlay=True``.
        nlsq_confidence : float
            Confidence level for NLSQ overlay band (default: 0.95).
        n_pred_points : int
            Number of prediction grid points (default: 200).
        show_residuals : bool
            If True, add residuals subplot using posterior median.
        style : str
            Plot style.
        deformation_mode : str, optional
            Deformation mode for axis labels.
        **kwargs
            Additional arguments forwarded to model_function.

        Returns
        -------
        tuple[Figure, Axes or ndarray]
        """
        logger.debug(
            "Plotting Bayesian fit",
            style=style,
            credible_level=credible_level,
            max_draws=max_draws,
        )

        x_data = _ensure_numpy(x_data)
        y_data = _ensure_numpy(y_data)
        style_params = _apply_style(style)
        is_complex = np.iscomplexobj(y_data)

        x_pred = self._make_pred_grid(x_data, n_pred_points)

        # Resolve test_mode
        test_mode = kwargs.pop("test_mode", None)
        if test_mode is None:
            test_mode = getattr(model, "_test_mode", None)

        # Get param names from model
        param_names = list(model.parameters.keys())

        # Compute credible band
        y_median, y_lower, y_upper = compute_credible_band(
            model_fn=model.model_function,
            x_pred=x_pred,
            posterior_samples=bayesian_result.posterior_samples,
            param_names=param_names,
            credible_level=credible_level,
            test_mode=test_mode,
            max_draws=max_draws,
            **kwargs,
        )

        # Compute residuals at data points using posterior median params
        y_pred_data = None
        if show_residuals:
            # Build median parameter vector
            model_param_names = [p for p in param_names if not p.startswith("sigma")]
            median_params = jnp.asarray(
                [
                    np.median(bayesian_result.posterior_samples[p])
                    for p in model_param_names
                ]
            )
            model_kwargs = dict(kwargs)  # snapshot to avoid mutation
            try:
                y_pred_data = np.asarray(
                    model.model_function(
                        jnp.asarray(x_data), median_params, test_mode, **model_kwargs
                    )
                )
            except Exception:
                y_pred_data = _ensure_numpy(model.predict(x_data))

        # Labels
        storage_label, loss_label = "G'", 'G"'
        if deformation_mode in ("tension", "bending", "compression"):
            storage_label, loss_label = "E'", 'E"'

        band_label = f"{int(credible_level * 100)}% CI"
        model_name = getattr(model, "__class__", type(model)).__name__

        if is_complex:
            fig, axes = self._plot_complex_fit(
                x_data,
                y_data,
                x_pred,
                y_median,
                y_pred_data,
                y_lower=y_lower,
                y_upper=y_upper,
                show_residuals=show_residuals,
                storage_label=storage_label,
                loss_label=loss_label,
                model_name=model_name,
                band_label=band_label,
                style_params=style_params,
                fit_label="Posterior median",
            )
        else:
            fig, axes = self._plot_scalar_fit(
                x_data,
                y_data,
                x_pred,
                y_median,
                y_pred_data,
                y_lower=y_lower,
                y_upper=y_upper,
                show_residuals=show_residuals,
                model_name=model_name,
                band_label=band_label,
                style_params=style_params,
                fit_label="Posterior median",
                band_color="C0",
            )

        # NLSQ overlay
        if show_nlsq_overlay and fit_result is not None and not is_complex:
            if fit_result.optimization_result is None:
                logger.debug("NLSQ overlay skipped: optimization_result is None")
            else:
                ax_main = axes[0] if isinstance(axes, np.ndarray) else axes
                popt = np.asarray(fit_result.optimization_result.x, dtype=np.float64)
                pcov = fit_result.optimization_result.pcov

                # Compute NLSQ prediction
                model_kwargs = dict(kwargs)
                try:
                    y_nlsq = np.asarray(
                        model.model_function(
                            jnp.asarray(x_pred),
                            jnp.asarray(popt),
                            test_mode,
                            **model_kwargs,
                        )
                    )
                except Exception:
                    y_nlsq = None

                if y_nlsq is not None:
                    ax_main.plot(
                        x_pred,
                        y_nlsq,
                        "--",
                        color="C3",
                        linewidth=style_params["lines.linewidth"] * 0.8,
                        label="NLSQ fit",
                        zorder=2,
                    )

                    # NLSQ uncertainty band
                    if pcov is not None:

                        def _nlsq_fn(x, params):
                            return np.asarray(
                                model.model_function(
                                    jnp.asarray(x),
                                    jnp.asarray(params),
                                    test_mode,
                                    **model_kwargs,
                                )
                            )

                        _, nlsq_lower, nlsq_upper = compute_uncertainty_band(
                            _nlsq_fn,
                            x_pred,
                            popt,
                            pcov,
                            confidence=nlsq_confidence,
                        )
                        if nlsq_lower is not None:
                            ax_main.fill_between(
                                x_pred,
                                nlsq_lower,
                                nlsq_upper,
                                alpha=0.15,
                                color="C3",
                                label=f"NLSQ {int(nlsq_confidence * 100)}% CI",
                                zorder=0,
                            )

                    ax_main.legend(loc="best")

        return fig, axes

    def plot_comparison(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fit_result: Any,
        bayesian_result: Any,
        model: Any,
        confidence: float = 0.95,
        credible_level: float = 0.95,
        max_draws: int = 500,
        n_pred_points: int = 200,
        style: str = "default",
        deformation_mode: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Side-by-side comparison of NLSQ and Bayesian fits.

        Creates a two-panel figure with shared y-axis:
        - Left: NLSQ fit + confidence band
        - Right: Bayesian posterior predictive + credible interval

        Both panels show the same data for direct visual comparison.

        Parameters
        ----------
        x_data, y_data : ndarray
            Experimental data.
        fit_result : FitResult
            NLSQ fit result.
        bayesian_result : BayesianResult
            Bayesian fit result.
        model : BaseModel
            Fitted model instance.
        confidence : float
            Confidence level for NLSQ band.
        credible_level : float
            Credible level for Bayesian band.
        max_draws : int
            Max posterior draws for Bayesian band.
        n_pred_points : int
            Prediction grid density.
        style : str
            Plot style.
        deformation_mode : str, optional
            For axis labels.
        **kwargs
            Forwarded to model_function.

        Returns
        -------
        tuple[Figure, ndarray]
            Figure and 1D array of 2 Axes.
        """
        logger.debug("Plotting NLSQ vs Bayesian comparison", style=style)

        x_data = _ensure_numpy(x_data)
        y_data = _ensure_numpy(y_data)
        style_params = _apply_style(style)
        is_complex = np.iscomplexobj(y_data)

        if is_complex:
            raise NotImplementedError(
                "Comparison plot for complex data is not yet supported. "
                "Use plot_nlsq() and plot_bayesian() separately."
            )

        x_pred = self._make_pred_grid(x_data, n_pred_points)

        # Resolve test_mode
        test_mode = kwargs.pop("test_mode", None)
        if test_mode is None:
            test_mode = getattr(model, "_test_mode", None)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(
                style_params["figure.figsize"][0] * 1.6,
                style_params["figure.figsize"][1],
            ),
            sharey=True,
        )

        marker_kw = {
            "s": style_params["lines.markersize"] ** 2,
            "facecolors": "none",
            "edgecolors": "k",
            "linewidths": 1.0,
            "zorder": 5,
        }

        # --- Left: NLSQ ---
        axes[0].scatter(x_data, y_data, label="Data", **marker_kw)

        if fit_result.optimization_result is None:
            raise ValueError(
                "plot_comparison requires a FitResult with a stored optimization_result. "
                "Fit with compute_diagnostics=True."
            )
        popt = np.asarray(fit_result.optimization_result.x, dtype=np.float64)
        pcov = fit_result.optimization_result.pcov

        try:
            y_nlsq = np.asarray(
                model.model_function(
                    jnp.asarray(x_pred), jnp.asarray(popt), test_mode, **kwargs
                )
            )
        except Exception:
            y_nlsq = _ensure_numpy(model.predict(x_pred))

        axes[0].plot(
            x_pred,
            y_nlsq,
            "-",
            color="C0",
            linewidth=style_params["lines.linewidth"],
            label="NLSQ fit",
            zorder=3,
        )

        if pcov is not None:

            def _nlsq_fn(x, params):
                return np.asarray(
                    model.model_function(
                        jnp.asarray(x), jnp.asarray(params), test_mode, **kwargs
                    )
                )

            _, nlsq_lower, nlsq_upper = compute_uncertainty_band(
                _nlsq_fn,
                x_pred,
                popt,
                pcov,
                confidence=confidence,
            )
            if nlsq_lower is not None:
                axes[0].fill_between(
                    x_pred,
                    nlsq_lower,
                    nlsq_upper,
                    alpha=0.25,
                    color="C0",
                    label=f"{int(confidence * 100)}% CI",
                    zorder=1,
                )

        axes[0].set_title("NLSQ (Frequentist)")
        axes[0].legend(loc="best", fontsize=style_params["legend.fontsize"])
        axes[0].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        # --- Right: Bayesian ---
        axes[1].scatter(x_data, y_data, label="Data", **marker_kw)

        param_names = list(model.parameters.keys())
        y_median, y_lower, y_upper = compute_credible_band(
            model_fn=model.model_function,
            x_pred=x_pred,
            posterior_samples=bayesian_result.posterior_samples,
            param_names=param_names,
            credible_level=credible_level,
            test_mode=test_mode,
            max_draws=max_draws,
            **kwargs,
        )

        axes[1].plot(
            x_pred,
            y_median,
            "-",
            color="C1",
            linewidth=style_params["lines.linewidth"],
            label="Posterior median",
            zorder=3,
        )
        axes[1].fill_between(
            x_pred,
            y_lower,
            y_upper,
            alpha=0.25,
            color="C1",
            label=f"{int(credible_level * 100)}% CI",
            zorder=1,
        )

        axes[1].set_title("Bayesian (NUTS)")
        axes[1].legend(loc="best", fontsize=style_params["legend.fontsize"])
        axes[1].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        # Log scales if appropriate
        if self._infer_log_x(fit_result):
            for ax in axes:
                ax.set_xscale("log")
        if self._infer_log_y(fit_result):
            for ax in axes:
                ax.set_yscale("log")

        fig.tight_layout()
        return fig, axes

    def plot_diagnostics(
        self,
        bayesian_result: Any,
        var_names: list[str] | None = None,
        style: str = "default",
        output_dir: str | Path | None = None,
        prefix: str = "mcmc",
        formats: tuple[str, ...] = ("pdf", "png"),
        dpi: int = 300,
    ) -> dict[str, Figure | Path]:
        """Generate ArviZ diagnostic suite. Delegates to generate_diagnostic_suite()."""
        return generate_diagnostic_suite(
            bayesian_result=bayesian_result,
            var_names=var_names,
            style=style,
            output_dir=output_dir,
            prefix=prefix,
            formats=formats,
            dpi=dpi,
        )

    def plot_parameter_table(
        self,
        fit_result: Any | None = None,
        bayesian_result: Any | None = None,
        param_names: list[str] | None = None,
        style: str = "default",
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Render a parameter summary table on a matplotlib axes.

        Shows parameter values with uncertainties from NLSQ and/or Bayesian fits
        in a clean tabular format suitable for publication figures.

        Parameters
        ----------
        fit_result : FitResult, optional
            NLSQ fit result. Provides values and standard errors from covariance.
        bayesian_result : BayesianResult, optional
            Bayesian fit result. Provides median and HDI from posterior.
        param_names : list[str], optional
            Parameter names to include. If None, uses all from fit_result.
        style : str
            Plot style for font sizing.
        ax : Axes, optional
            Existing axes to render on.

        Returns
        -------
        tuple[Figure or None, Axes]
        """
        style_params = _apply_style(style)

        if fit_result is None and bayesian_result is None:
            raise ValueError(
                "At least one of fit_result or bayesian_result must be provided"
            )

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(style_params["figure.figsize"][0], 2))

        ax.axis("off")

        # Build table data
        if param_names is None:
            if fit_result is not None:
                param_names = list(fit_result.params.keys())
            else:
                param_names = [
                    k
                    for k in bayesian_result.posterior_samples
                    if not k.startswith("sigma")
                ]

        columns = ["Parameter"]
        if fit_result is not None:
            columns.append("NLSQ ± σ")
        if bayesian_result is not None:
            columns.append("Bayesian [95% HDI]")

        rows = []
        for name in param_names:
            row = [name]

            if fit_result is not None:
                val = fit_result.params.get(name, float("nan"))
                # Get std error from covariance
                std_err = float("nan")
                if (
                    fit_result.optimization_result is not None
                    and fit_result.optimization_result.pcov is not None
                ):
                    idx = list(fit_result.params.keys()).index(name)
                    pcov = fit_result.optimization_result.pcov
                    if idx < pcov.shape[0]:
                        std_err = np.sqrt(pcov[idx, idx])
                row.append(f"{val:.4g} ± {std_err:.4g}")

            if bayesian_result is not None:
                if name in bayesian_result.posterior_samples:
                    samples = bayesian_result.posterior_samples[name]
                    median = np.median(samples)
                    q025 = np.percentile(samples, 2.5)
                    q975 = np.percentile(samples, 97.5)
                    row.append(f"{median:.4g} [{q025:.4g}, {q975:.4g}]")
                else:
                    row.append("—")

            rows.append(row)

        table = ax.table(
            cellText=rows,
            colLabels=columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(style_params["font.size"])
        table.scale(1.0, 1.5)

        # Style header row
        for j in range(len(columns)):
            table[0, j].set_facecolor("#e6e6e6")
            table[0, j].set_text_props(weight="bold")

        if fig is not None:
            fig.tight_layout()

        return fig, ax

    # --- Private helpers ---

    def _make_pred_grid(self, x_data: np.ndarray, n_points: int) -> np.ndarray:
        """Create a dense prediction grid matching the data range."""
        x_min, x_max = (
            np.min(x_data[x_data > 0]) if np.any(x_data > 0) else np.min(x_data)
        ), np.max(x_data)
        # Use log spacing if data spans > 1.5 decades
        if x_min > 0 and x_max / x_min > 30:
            return np.logspace(np.log10(x_min), np.log10(x_max), n_points)
        return np.linspace(x_min, x_max, n_points)

    def _infer_log_x(self, fit_result: Any) -> bool:
        """Infer whether log x-axis is appropriate from fit result metadata."""
        protocol = getattr(fit_result, "protocol", None) or ""
        return protocol in ("oscillation", "rotation", "flow_curve", "frequency_sweep")

    def _infer_log_y(self, fit_result: Any) -> bool:
        """Infer whether log y-axis is appropriate from fit result metadata."""
        protocol = getattr(fit_result, "protocol", None) or ""
        return protocol in (
            "oscillation",
            "rotation",
            "flow_curve",
            "frequency_sweep",
            "relaxation",
        )

    def _plot_scalar_fit(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_pred: np.ndarray,
        y_pred: np.ndarray,
        y_pred_data: np.ndarray | None,
        y_lower: np.ndarray | None,
        y_upper: np.ndarray | None,
        show_residuals: bool,
        model_name: str,
        band_label: str,
        style_params: dict[str, Any],
        fit_label: str = "Fit",
        band_color: str = "C0",
        log_x: bool = False,
        log_y: bool = False,
    ) -> tuple[Figure, Axes | np.ndarray]:
        """Plot scalar (non-complex) fit with optional uncertainty and residuals."""
        n_rows = 2 if (show_residuals and y_pred_data is not None) else 1

        if n_rows == 2:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(
                    style_params["figure.figsize"][0],
                    style_params["figure.figsize"][1] * 1.4,
                ),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
            ax_fit, ax_resid = axes[0], axes[1]
        else:
            fig, ax_fit = plt.subplots(figsize=style_params["figure.figsize"])
            axes = ax_fit

        # Uncertainty band (behind everything)
        if y_lower is not None and y_upper is not None:
            if log_y:
                mask = (y_lower > 0) & (y_upper > 0)
                ax_fit.fill_between(
                    x_pred[mask],
                    y_lower[mask],
                    y_upper[mask],
                    alpha=0.25,
                    color=band_color,
                    label=band_label,
                    zorder=1,
                )
            else:
                ax_fit.fill_between(
                    x_pred,
                    y_lower,
                    y_upper,
                    alpha=0.25,
                    color=band_color,
                    label=band_label,
                    zorder=1,
                )

        # Fit curve
        ax_fit.plot(
            x_pred,
            y_pred,
            "-",
            color=band_color,
            linewidth=style_params["lines.linewidth"],
            label=fit_label,
            zorder=2,
        )

        # Data points
        ax_fit.scatter(
            x_data,
            y_data,
            s=style_params["lines.markersize"] ** 2,
            facecolors="none",
            edgecolors="C1",
            linewidths=1.2,
            label="Data",
            zorder=3,
        )

        if log_x:
            ax_fit.set_xscale("log")
        if log_y:
            ax_fit.set_yscale("log")

        ax_fit.legend(loc="best", fontsize=style_params["legend.fontsize"])
        ax_fit.grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)
        if model_name:
            ax_fit.set_title(
                f"Model Fit: {model_name}",
                fontsize=style_params["axes.titlesize"],
            )

        # Residuals
        if n_rows == 2 and y_pred_data is not None:
            residuals = y_data - y_pred_data
            denom = np.maximum(np.abs(y_data), np.max(np.abs(y_data)) * 1e-10)
            pct_resid = residuals / denom * 100

            ax_resid.scatter(
                x_data,
                pct_resid,
                s=style_params["lines.markersize"] ** 2 * 0.7,
                facecolors="none",
                edgecolors="C2",
                linewidths=1.0,
            )
            ax_resid.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
            ax_resid.set_ylabel("Residuals (%)")
            ax_resid.grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        fig.tight_layout()
        return fig, axes

    def _plot_complex_fit(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_pred: np.ndarray,
        y_pred: np.ndarray,
        y_pred_data: np.ndarray | None,
        y_lower: np.ndarray | None,
        y_upper: np.ndarray | None,
        show_residuals: bool,
        storage_label: str,
        loss_label: str,
        model_name: str,
        band_label: str,
        style_params: dict[str, Any],
        fit_label: str = "Fit",
    ) -> tuple[Figure, np.ndarray]:
        """Plot complex (G'/G'') fit with optional uncertainty and residuals."""
        n_rows = 2 if (show_residuals and y_pred_data is not None) else 1
        n_cols = 2  # Always G' and G''

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                style_params["figure.figsize"][0] * 1.5,
                style_params["figure.figsize"][1] * (1.4 if n_rows == 2 else 1.0),
            ),
            squeeze=False,
        )

        components = [
            (np.real, storage_label, "C0"),
            (np.imag, loss_label, "C1"),
        ]

        for col, (extract_fn, label, color) in enumerate(components):
            ax = axes[0, col]

            # Data
            x_filt, y_filt = _filter_positive(x_data, extract_fn(y_data), warn=False)
            ax.scatter(
                x_filt,
                y_filt,
                s=style_params["lines.markersize"] ** 2,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                label="Data",
                zorder=3,
            )

            # Fit curve
            x_pred_filt, y_pred_filt = _filter_positive(
                x_pred, extract_fn(y_pred), warn=False
            )
            ax.loglog(
                x_pred_filt,
                y_pred_filt,
                "-",
                color=color,
                linewidth=style_params["lines.linewidth"],
                label=fit_label,
                zorder=2,
            )

            # Uncertainty band
            if y_lower is not None and y_upper is not None:
                lower_comp = extract_fn(y_lower)
                upper_comp = extract_fn(y_upper)
                mask = (lower_comp > 0) & (upper_comp > 0) & (x_pred > 0)
                if np.any(mask):
                    ax.fill_between(
                        x_pred[mask],
                        lower_comp[mask],
                        upper_comp[mask],
                        alpha=0.25,
                        color=color,
                        label=band_label,
                        zorder=1,
                    )

            ax.set_ylabel(label)
            ax.legend(loc="best", fontsize=style_params["legend.fontsize"])
            ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            # Residuals
            if n_rows == 2 and y_pred_data is not None:
                ax_r = axes[1, col]
                comp_data = extract_fn(y_data)
                comp_pred = extract_fn(y_pred_data)
                residuals = comp_data - comp_pred
                denom = np.maximum(np.abs(comp_data), np.max(np.abs(comp_data)) * 1e-10)
                pct = residuals / denom * 100

                pos_mask = np.isfinite(comp_data) & (comp_data > 0)
                ax_r.semilogx(
                    x_data[pos_mask],
                    pct[pos_mask],
                    "o",
                    color=color,
                    markersize=style_params["lines.markersize"] * 0.8,
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                )
                ax_r.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
                ax_r.set_ylabel(f"{label} Resid. (%)")
                ax_r.grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        if model_name:
            fig.suptitle(
                f"Model Fit: {model_name}",
                fontsize=style_params["axes.titlesize"],
            )

        fig.tight_layout()
        return fig, axes
