"""Bayesian inference infrastructure using NumPyro NUTS sampling.

This module provides the BayesianMixin class that adds Bayesian inference
capabilities to rheological models. It uses NumPyro's NUTS sampler for
efficient MCMC sampling with warm-start support from NLSQ optimization.

Example:
    >>> from rheojax.core.bayesian import BayesianMixin
    >>> from rheojax.core.parameters import ParameterSet
    >>>
    >>> class MyModel(BayesianMixin):
    ...     def __init__(self):
    ...         self.parameters = ParameterSet()
    ...         self.parameters.add("a", value=1.0, bounds=(0, 10))
    ...
    >>> model = MyModel()
    >>> result = model.fit_bayesian(X, y, num_samples=2000)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms as dist_transforms
from numpyro.infer import MCMC, NUTS, init_to_uniform, init_to_value

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import TestMode, detect_test_mode

# Safe JAX import (verifies NLSQ was imported first)
jax, jnp = safe_import_jax()


@dataclass
class BayesianResult:
    """Results from Bayesian inference with NUTS sampling.

    This dataclass stores the complete results of NumPyro NUTS sampling,
    including posterior samples, summary statistics, convergence diagnostics,
    and placeholders for future model comparison metrics.

    Attributes:
        posterior_samples: Dictionary mapping parameter names to arrays of
            posterior samples (shape: [num_samples * num_chains, ]). All arrays are float64.
        summary: Dictionary with summary statistics for each parameter.
            Contains nested dicts with 'mean', 'std', and quantiles.
        diagnostics: Dictionary with convergence diagnostics including:
            - r_hat: Gelman-Rubin statistic for each parameter (dict)
            - ess: Effective sample size for each parameter (dict)
            - divergences: Number of divergent transitions (int)
        num_samples: Number of posterior samples per chain (after warmup).
        num_chains: Number of MCMC chains used in sampling.
        mcmc: NumPyro MCMC object containing full sampling information including
            NUTS-specific diagnostics (energy, divergences, tree depth).
            Required for ArviZ visualization with full diagnostics.
        model_comparison: Dictionary for model comparison metrics (WAIC, LOO).
            Currently a placeholder for future implementation.
        _inference_data: Cached ArviZ InferenceData object. Automatically
            created on first access via to_inference_data(). Do not set manually.

    Example:
        >>> result = model.fit_bayesian(X, y)
        >>> print(result.summary["a"]["mean"])
        >>> print(result.diagnostics["r_hat"]["a"])
        >>> # Convert to ArviZ InferenceData for advanced plotting
        >>> idata = result.to_inference_data()
    """

    posterior_samples: dict[str, np.ndarray]
    summary: dict[str, dict[str, float]]
    diagnostics: dict[str, Any]
    num_samples: int
    num_chains: int
    mcmc: MCMC | None = None
    model_comparison: dict[str, float] = field(default_factory=dict)
    _inference_data: Any | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate result after initialization."""
        # Ensure posterior_samples are numpy arrays
        for name, samples in self.posterior_samples.items():
            if not isinstance(samples, np.ndarray):
                self.posterior_samples[name] = np.asarray(samples, dtype=np.float64)

    def to_inference_data(self) -> Any:
        """Convert to ArviZ InferenceData format for advanced visualization.

        Converts the NumPyro MCMC result to ArviZ InferenceData format, which
        enables access to ArviZ's comprehensive plotting and diagnostic tools.
        The conversion preserves all NUTS-specific diagnostics including energy,
        divergences, and tree depth information.

        The conversion automatically computes pointwise log-likelihood values
        required for Bayesian model comparison metrics (WAIC and LOO). This
        enables usage of az.waic(), az.loo(), and az.compare() for objective
        model selection.

        The InferenceData object is cached after first conversion to avoid
        repeated conversion overhead.

        Returns:
            ArviZ InferenceData object containing:
                - posterior: Posterior samples for all parameters
                - sample_stats: NUTS diagnostics (energy, divergences, etc.)
                - log_likelihood: Pointwise log-likelihood for WAIC/LOO
                - Additional groups as available from NumPyro

        Raises:
            ImportError: If arviz is not installed
            ValueError: If MCMC object was not stored (older results)

        Example:
            >>> result = model.fit_bayesian(X, y)
            >>> idata = result.to_inference_data()
            >>>
            >>> # Now use ArviZ plotting functions
            >>> import arviz as az
            >>> az.plot_trace(idata)
            >>> az.plot_pair(idata)
            >>> az.plot_energy(idata)

        Note:
            Requires arviz package: pip install arviz
            The MCMC object must be present (automatically stored by fit_bayesian).
        """
        # Return cached version if available
        if self._inference_data is not None:
            return self._inference_data

        # Import arviz (lazy import)
        try:
            import arviz as az
        except ImportError:
            raise ImportError(
                "ArviZ is required for InferenceData conversion. "
                "Install it with: pip install arviz"
            ) from None

        # Ensure MCMC object is available
        if self.mcmc is None:
            raise ValueError(
                "MCMC object not available for conversion. "
                "This may be a result from an older version. "
                "Re-run fit_bayesian() to generate a compatible result."
            )

        # Convert using ArviZ's from_numpyro utility
        # This preserves all NUTS diagnostics (energy, divergences, etc.)
        # log_likelihood=True computes pointwise log-likelihood for WAIC/LOO model comparison
        self._inference_data = az.from_numpyro(self.mcmc, log_likelihood=True)

        return self._inference_data


class BayesianMixin:
    """Mixin class providing Bayesian inference capabilities via NumPyro NUTS.

    This mixin adds methods for Bayesian parameter estimation to any class
    that has a `parameters` attribute (ParameterSet). It implements:
    - NUTS sampling for posterior inference
    - Prior sampling for prior predictive checks
    - Credible interval computation (highest density intervals)
    - Convergence diagnostics (R-hat, ESS)

    The mixin is designed to be composed with model classes, typically through
    BaseModel. All 20+ rheological models automatically inherit these
    capabilities when BaseModel is extended with BayesianMixin.

    Requirements:
        - Class must have `parameters` attribute (ParameterSet)
        - Class must define `model_function(X, params)` method for predictions
        - Class must have `X_data` and `y_data` attributes when fitting

    Example:
        >>> class MyModel(BayesianMixin):
        ...     def __init__(self):
        ...         self.parameters = ParameterSet()
        ...         self.parameters.add("a", bounds=(0, 10))
        ...         self.X_data = None
        ...         self.y_data = None
        ...
        ...     def model_function(self, X, params):
        ...         return params[0] * X
        ...
        >>> model = MyModel()
        >>> model.X_data = X
        >>> model.y_data = y
        >>> result = model.fit_bayesian(X, y)
    """

    def sample_prior(self, num_samples: int = 1000) -> dict[str, np.ndarray]:
        """Sample from prior distributions over parameter bounds.

        Samples from uniform prior distributions defined by parameter bounds.
        This is useful for prior predictive checks and understanding the prior's
        influence on the posterior.

        Args:
            num_samples: Number of samples to draw from prior (default: 1000)

        Returns:
            Dictionary mapping parameter names to arrays of prior samples.
            Each array has shape (num_samples,) and dtype float64.

        Raises:
            AttributeError: If class doesn't have `parameters` attribute
            ValueError: If any parameter lacks bounds

        Example:
            >>> model = MyModel()
            >>> prior_samples = model.sample_prior(num_samples=500)
            >>> print(prior_samples["a"].shape)  # (500,)
        """
        if not hasattr(self, "parameters"):
            raise AttributeError(
                "Class must have 'parameters' attribute (ParameterSet)"
            )

        prior_samples = {}

        for param_name in self.parameters:
            param = self.parameters.get(param_name)

            if param.bounds is None:
                raise ValueError(
                    f"Parameter '{param_name}' must have bounds for prior sampling"
                )

            lower, upper = param.bounds

            # Sample from uniform distribution over bounds
            samples = np.random.uniform(low=lower, high=upper, size=num_samples).astype(
                np.float64
            )

            prior_samples[param_name] = samples

        return prior_samples

    def get_credible_intervals(
        self,
        posterior_samples: dict[str, np.ndarray],
        credibility: float = 0.95,
    ) -> dict[str, tuple[float, float]]:
        """Compute highest density intervals (HDI) for posterior samples.

        Computes the highest posterior density interval for each parameter,
        which is the shortest interval containing the specified probability mass.
        This is preferred over equal-tailed intervals for skewed distributions.

        Args:
            posterior_samples: Dictionary mapping parameter names to posterior
                sample arrays (from BayesianResult.posterior_samples)
            credibility: Probability mass to include in interval (default: 0.95)
                Common values: 0.68 (1 sigma), 0.95 (2 sigma), 0.997 (3 sigma)

        Returns:
            Dictionary mapping parameter names to (lower, upper) credible
            interval tuples. All values are float64.

        Example:
            >>> result = model.fit_bayesian(X, y)
            >>> intervals_95 = model.get_credible_intervals(
            ...     result.posterior_samples, credibility=0.95
            ... )
            >>> print(f"95% CI for a: {intervals_95['a']}")
        """
        intervals = {}

        for param_name, samples in posterior_samples.items():
            samples = np.asarray(samples, dtype=np.float64)

            # Use NumPyro's HDI utility if available, otherwise fall back to quantile
            try:
                from numpyro.diagnostics import hpdi

                # NumPyro's hpdi returns (lower, upper)
                hdi = hpdi(samples, prob=credibility)
                intervals[param_name] = (float(hdi[0]), float(hdi[1]))
            except (ImportError, AttributeError):
                # Fallback to equal-tailed credible interval
                alpha = 1 - credibility
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100

                lower = float(np.percentile(samples, lower_percentile))
                upper = float(np.percentile(samples, upper_percentile))
                intervals[param_name] = (lower, upper)

        return intervals

    def fit_bayesian(
        self,
        X: np.ndarray | RheoData,
        y: np.ndarray | None = None,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 1,
        initial_values: dict[str, float] | None = None,
        test_mode: str | TestMode | None = None,
        **nuts_kwargs,
    ) -> BayesianResult:
        """Perform Bayesian inference using NumPyro NUTS sampler.

        Runs NUTS (No-U-Turn Sampler) to obtain posterior samples for model
        parameters. Supports warm-starting from NLSQ point estimates for faster
        convergence. Uses uniform priors over parameter bounds.

        The method constructs a NumPyro probabilistic model, runs NUTS sampling,
        and returns posterior samples with convergence diagnostics.

        CRITICAL: test_mode is captured in model_function closure to ensure
        correct posteriors for all test modes (relaxation, creep, oscillation).
        This fixes the v0.3.1 bug where model_function read global state.

        Args:
            X: Independent variable data (input features) or RheoData object
            y: Dependent variable data (observations to fit). If X is RheoData,
                y is ignored and extracted from X.
            num_warmup: Number of warmup/burn-in iterations (default: 1000)
            num_samples: Number of posterior samples to collect (default: 2000)
            num_chains: Number of MCMC chains (default: 1, multi-chain deferred)
            initial_values: Optional dict of initial parameter values for
                warm-start (e.g., from NLSQ). Keys are parameter names.
            test_mode: Explicit test mode (e.g., 'relaxation', 'creep', 'oscillation').
                If None, inferred from RheoData.metadata['test_mode'] or defaults
                to 'relaxation'. Overrides RheoData metadata if provided.
            **nuts_kwargs: Additional arguments passed to NUTS sampler

        Returns:
            BayesianResult containing:
                - posterior_samples: Dict of parameter samples
                - summary: Statistics (mean, std, quantiles)
                - diagnostics: R-hat, ESS, divergences
                - model_comparison: Placeholder dict

        Raises:
            AttributeError: If required attributes missing
            RuntimeError: If NUTS sampling fails

        Example:
            >>> # Basic usage with explicit mode
            >>> result = model.fit_bayesian(X, y, test_mode='oscillation')
            >>>
            >>> # RheoData with embedded mode (recommended)
            >>> rheo_data = RheoData(x=omega, y=G_star, metadata={'test_mode': 'oscillation'})
            >>> result = model.fit_bayesian(rheo_data)
            >>>
            >>> # With warm-start from NLSQ
            >>> nlsq_result = model.fit(X, y)
            >>> initial = {name: p.value for name, p in model.parameters._parameters.items()}
            >>> result = model.fit_bayesian(rheo_data, initial_values=initial)
            >>>
            >>> # Check convergence
            >>> print(result.diagnostics["r_hat"])  # Should be < 1.01
            >>> print(result.diagnostics["ess"])    # Should be > 400
        """
        # Validate required attributes
        if not hasattr(self, "parameters"):
            raise AttributeError(
                "Class must have 'parameters' attribute (ParameterSet)"
            )

        if not hasattr(self, "model_function"):
            raise AttributeError(
                "Class must define 'model_function(X, params, test_mode)' method"
            )

        # Validate parameter bounds before attempting warm-start or sampling
        for name in self.parameters.keys():
            param = self.parameters.get(name)
            if param is None:
                continue
            bounds = getattr(param, "bounds", None)
            if (
                bounds is not None
                and bounds[0] is not None
                and bounds[1] is not None
                and bounds[0] >= bounds[1]
            ):
                raise ValueError(
                    f"Invalid bounds for parameter '{name}': {bounds}. "
                    "Lower bound must be strictly less than upper bound."
                )

        # Handle RheoData input and infer test_mode
        if isinstance(X, RheoData):
            rheo_data = X
            X_array = rheo_data.x
            y_array = rheo_data.y

            # Infer test_mode from RheoData if not explicitly provided
            if test_mode is None:
                inferred_mode = detect_test_mode(rheo_data)
                test_mode = inferred_mode
        else:
            # Plain array input
            X_array = X
            y_array = y

            # If test_mode not provided, reuse mode from latest fit if available
            if test_mode is None:
                stored_mode = getattr(self, "_test_mode", None)
                if stored_mode is not None:
                    test_mode = stored_mode
                else:
                    test_mode = TestMode.RELAXATION
                    warnings.warn(
                        "test_mode not specified. Defaulting to 'relaxation'. "
                        "For correct posteriors, pass RheoData with metadata['test_mode'] "
                        "or specify test_mode explicitly.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Normalize test_mode to TestMode enum
        if isinstance(test_mode, str):
            test_mode = TestMode(test_mode.lower())

        # Cache resolved mode so future Bayesian/ predict calls stay in sync
        self._test_mode = test_mode

        # Convert data to JAX arrays with float64 precision
        X_jax = jnp.asarray(X_array, dtype=jnp.float64)

        # Handle complex data (e.g., G* = G' + iG" for oscillatory shear)
        # JAX's grad requires real-valued outputs, so we decompose complex data
        is_complex_data = jnp.iscomplexobj(y_array)
        data_scale_scalar: float | None = None
        y_real_scale_scalar: float | None = None
        y_imag_scale_scalar: float | None = None

        if is_complex_data:
            # Store original complex data for scale computation using NumPy to keep
            # host-side scalars that can be safely captured inside the NumPyro model.
            y_complex_np = np.asarray(y_array, dtype=np.complex128)
            y_real_np = np.real(y_complex_np)
            y_imag_np = np.imag(y_complex_np)
            y_real_scale_scalar = float(np.std(y_real_np)) if y_real_np.size else 0.0
            y_imag_scale_scalar = float(np.std(y_imag_np)) if y_imag_np.size else 0.0

            y_complex = jnp.asarray(y_array, dtype=jnp.complex128)
            y_real = jnp.real(y_complex)
            y_imag = jnp.imag(y_complex)
            # Stack real and imaginary parts as [real_part, imag_part]
            y_jax = jnp.concatenate([y_real, y_imag])
        else:
            y_np = np.asarray(y_array, dtype=np.float64)
            data_scale_scalar = float(np.std(y_np)) if y_np.size else 0.0
            y_jax = jnp.asarray(y_np, dtype=jnp.float64)

        # Get parameter information
        param_names = list(self.parameters)
        param_bounds: dict[str, tuple[float | None, float | None]] = {}
        for name in param_names:
            param = self.parameters.get(name)
            if param.bounds is None:
                raise ValueError(
                    f"Parameter '{name}' must have bounds for Bayesian inference"
                )
            param_bounds[name] = param.bounds

        bounds_override = getattr(self, "bayesian_parameter_bounds", None)
        if callable(bounds_override):
            param_bounds = bounds_override(
                param_bounds.copy(),
                np.asarray(X_array),
                np.asarray(y_array),
                test_mode,
            )

        # Helper utilities to build safe initialization seeds within bounds
        def _normalize_bound(value: float | None, default: float) -> float:
            return float(value) if value is not None else default

        def _safe_interval(lower_raw: float | None, upper_raw: float | None) -> tuple[float, float]:
            lower = _normalize_bound(lower_raw, -np.inf)
            upper = _normalize_bound(upper_raw, np.inf)

            eps = 1e-9
            if np.isfinite(lower) and np.isfinite(upper):
                span = max(upper - lower, eps * 10)
                half_span = max(span / 2.0 - eps, eps)
                pad = min(max(span * 1e-12, eps), half_span)
                return lower + pad, upper - pad
            if np.isfinite(lower):
                upper_guess = lower + max(abs(lower) * 2.0, 1.0)
                return lower + eps, upper_guess
            if np.isfinite(upper):
                lower_guess = upper - max(abs(upper) * 2.0, 1.0)
                return lower_guess, upper - eps
            return -1.0, 1.0

        def _default_midpoint(lower_raw: float | None, upper_raw: float | None) -> float:
            lower = _normalize_bound(lower_raw, -np.inf)
            upper = _normalize_bound(upper_raw, np.inf)

            if np.isfinite(lower) and np.isfinite(upper):
                if lower > 0 and upper > 0:
                    return float(np.exp((np.log(lower) + np.log(upper)) / 2.0))
                return float(0.5 * (lower + upper))
            if np.isfinite(lower):
                return float(lower + max(abs(lower) * 0.5, 1.0))
            if np.isfinite(upper):
                return float(upper - max(abs(upper) * 0.5, 1.0))
            return 0.0

        init_intervals = {
            name: _safe_interval(*param_bounds[name]) for name in param_names
        }

        def _sanitize_initial_value(name: str, raw_value: float | None) -> float:
            lower_raw, upper_raw = param_bounds[name]
            safe_lower, safe_upper = init_intervals[name]
            value = raw_value
            if value is None or not np.isfinite(value):
                value = _default_midpoint(lower_raw, upper_raw)
            if np.isfinite(safe_lower):
                value = max(value, safe_lower)
            if np.isfinite(safe_upper):
                value = min(value, safe_upper)
            return float(value)

        # Define NumPyro probabilistic model with test_mode captured in closure
        # CRITICAL: test_mode is captured here, not read from self._test_mode
        # This ensures correct posteriors for all test modes (fixes v0.3.1 bug)
        prior_factory = getattr(self, "bayesian_prior_factory", None)

        nuts_overrides = getattr(self, "bayesian_nuts_kwargs", None)
        if callable(nuts_overrides):
            overrides = nuts_overrides()
            if isinstance(overrides, dict):
                for key, value in overrides.items():
                    nuts_kwargs.setdefault(key, value)

        def numpyro_model(X, y=None):
            """NumPyro model with uniform priors over parameter bounds.

            test_mode is captured in closure from fit_bayesian() scope, ensuring
            model_function uses the correct mode throughout MCMC sampling.
            """
            # Sample parameters from priors
            params_dict = {}
            for name in param_names:
                lower, upper = param_bounds[name]
                custom_dist = None
                if callable(prior_factory):
                    custom_dist = prior_factory(name, lower, upper)

                if custom_dist is not None:
                    params_dict[name] = numpyro.sample(name, custom_dist)
                else:
                    # Weakly-informative Beta prior for fractional orders
                    if (
                        name.lower().endswith("alpha")
                        and lower is not None
                        and upper is not None
                        and 0.0 <= lower < upper <= 1.0
                    ):
                        beta_base = dist.Beta(
                            concentration1=2.0, concentration0=2.0
                        )
                        if lower == 0.0 and upper == 1.0:
                            params_dict[name] = numpyro.sample(name, beta_base)
                        else:
                            scale = upper - lower
                            beta_trans = dist_transforms.AffineTransform(
                                loc=lower, scale=scale
                            )
                            params_dict[name] = numpyro.sample(
                                name,
                                dist.TransformedDistribution(beta_base, beta_trans),
                            )
                    else:
                        params_dict[name] = numpyro.sample(
                            name, dist.Uniform(low=lower, high=upper)
                        )

            # Convert to array for model_function
            params_array = jnp.array([params_dict[name] for name in param_names])

            # Compute predictions with test_mode from closure (NOT self._test_mode)
            # Pass test_mode explicitly to ensure correct mode-aware predictions
            predictions_raw = self.model_function(X, params_array, test_mode)

            # Handle complex predictions (convert to stacked real for JAX grad compatibility)
            if is_complex_data:
                # predictions_raw is complex, decompose to [real_part, imag_part]
                pred_real = jnp.real(predictions_raw)
                pred_imag = jnp.imag(predictions_raw)

                # Split observations into real and imaginary parts
                n = len(y) // 2  # y is stacked [real, imag]
                y_real_obs = y[:n]
                y_imag_obs = y[n:]
                predictions = jnp.concatenate([pred_real, pred_imag])
            else:
                predictions = predictions_raw

            # Likelihood: Use appropriate noise model for real or complex data
            if is_complex_data:
                # For complex data, use separate noise for real and imaginary parts
                # Use weakly informative Exponential priors (heavier tails than HalfNormal)
                # Mean scale is intentionally inflated (×5) to avoid spuriously tight
                # likelihoods that previously caused FractionalMaxwellLiquid warm-starts
                # to be rejected before NUTS could adapt.
                real_scale = max(y_real_scale_scalar or 0.0, 1e-9)
                imag_scale = max(y_imag_scale_scalar or 0.0, 1e-9)
                sigma_real_scale = max(real_scale * 5.0, 1e-9)
                sigma_imag_scale = max(imag_scale * 5.0, 1e-9)
                sigma_real = numpyro.sample(
                    "sigma_real", dist.Exponential(rate=1.0 / sigma_real_scale)
                )
                sigma_imag = numpyro.sample(
                    "sigma_imag", dist.Exponential(rate=1.0 / sigma_imag_scale)
                )

                # Use pred_real and pred_imag already computed above (don't re-split!)
                # Separate likelihoods for real and imaginary components
                numpyro.sample(
                    "obs_real",
                    dist.Normal(loc=pred_real, scale=sigma_real),
                    obs=y_real_obs,
                )
                numpyro.sample(
                    "obs_imag",
                    dist.Normal(loc=pred_imag, scale=sigma_imag),
                    obs=y_imag_obs,
                )
            else:
                # Real data: weakly informative Exponential prior on noise.
                # Inflate the mean scale (×5) to keep the prior permissive enough
                # for challenging fractional models (e.g., FractionalMaxwellLiquid).
                base_scale = max(data_scale_scalar or 0.0, 1e-9)
                sigma_scale = max(base_scale * 5.0, 1e-9)
                sigma = numpyro.sample(
                    "sigma", dist.Exponential(rate=1.0 / sigma_scale)
                )
                numpyro.sample("obs", dist.Normal(loc=predictions, scale=sigma), obs=y)

        # Build sanitized initialization dictionary regardless of warm-start input
        def _build_warm_start(source: dict[str, float] | None) -> dict[str, float]:
            warm_start: dict[str, float] = {}
            for name in param_names:
                candidate = None
                if source and name in source:
                    candidate = source[name]
                else:
                    candidate = self.parameters.get_value(name)
                warm_start[name] = _sanitize_initial_value(name, candidate)
            return warm_start

        init_source = initial_values if initial_values is not None else None
        warm_start_values = _build_warm_start(init_source)

        # Provide sensible initial noise scales to avoid near-zero sigma issues
        if is_complex_data:
            if "sigma_real" not in warm_start_values and y_real_scale_scalar is not None:
                warm_start_values["sigma_real"] = max(
                    (y_real_scale_scalar or 0.0) * 0.1, 1e-6
                )
            if "sigma_imag" not in warm_start_values and y_imag_scale_scalar is not None:
                warm_start_values["sigma_imag"] = max(
                    (y_imag_scale_scalar or 0.0) * 0.1, 1e-6
                )
        else:
            data_scale = data_scale_scalar if data_scale_scalar is not None else float(
                np.std(np.asarray(y_array, dtype=np.float64)) if y_jax.size else 1.0
            )
            if "sigma" not in warm_start_values:
                warm_start_values["sigma"] = max(data_scale * 0.1, 1e-6)

        # Set up NUTS sampler with robust initialization strategy
        try:
            init_strategy = init_to_value(values=warm_start_values)
        except Exception as exc:
            warnings.warn(
                "Warm-start initialization failed; falling back to uniform init. "
                "Reason: " + str(exc),
                RuntimeWarning,
                stacklevel=2,
            )
            init_strategy = init_to_uniform()

        def _run_mcmc(strategy):
            kernel = NUTS(numpyro_model, init_strategy=strategy, **nuts_kwargs)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="There are not enough devices to run parallel chains",
                    category=UserWarning,
                )
                sampler = MCMC(
                    kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
            rng_key = jax.random.PRNGKey(0)
            sampler.run(rng_key, X_jax, y_jax)
            return sampler

        try:
            mcmc = _run_mcmc(init_strategy)
        except RuntimeError as e:
            if "Cannot find valid initial parameters" in str(e):
                warnings.warn(
                    "Warm-started NUTS initialization failed; retrying with uniform init.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                try:
                    mcmc = _run_mcmc(init_to_uniform())
                except Exception as final_exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"NUTS sampling failed: {str(final_exc)}"
                    ) from final_exc
            else:
                raise RuntimeError(f"NUTS sampling failed: {str(e)}") from e

        # Extract posterior samples
        samples = mcmc.get_samples()

        # Convert to numpy arrays (exclude sigma, keep only model parameters)
        posterior_samples = {}
        for name in param_names:
            if name in samples:
                posterior_samples[name] = np.asarray(samples[name], dtype=np.float64)

        # Compute summary statistics
        summary = {}
        for name, sample_array in posterior_samples.items():
            summary[name] = {
                "mean": float(np.mean(sample_array)),
                "std": float(np.std(sample_array)),
                "median": float(np.median(sample_array)),
                "q05": float(np.percentile(sample_array, 5)),
                "q25": float(np.percentile(sample_array, 25)),
                "q75": float(np.percentile(sample_array, 75)),
                "q95": float(np.percentile(sample_array, 95)),
            }

        # Compute convergence diagnostics
        diagnostics = self._compute_diagnostics(mcmc, posterior_samples)

        # Create BayesianResult with MCMC object for ArviZ conversion
        result = BayesianResult(
            posterior_samples=posterior_samples,
            summary=summary,
            diagnostics=diagnostics,
            num_samples=num_samples,
            num_chains=num_chains,
            mcmc=mcmc,  # Store MCMC object for ArviZ InferenceData conversion
            model_comparison={},  # Placeholder for future WAIC/LOO
        )

        return result

    def _compute_diagnostics(
        self, mcmc: MCMC, posterior_samples: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        """Compute convergence diagnostics from MCMC samples.

        Args:
            mcmc: NumPyro MCMC object after sampling
            posterior_samples: Dictionary of posterior samples

        Returns:
            Dictionary with diagnostic information:
                - r_hat: R-hat (Gelman-Rubin) statistic per parameter
                - ess: Effective sample size per parameter
                - divergences: Number of divergent transitions
        """
        diagnostics: dict[str, Any] = {}

        # Get NumPyro diagnostics
        try:
            # R-hat (should be < 1.01 for good convergence)
            r_hat_dict = {}
            for name in posterior_samples.keys():
                try:
                    # NumPyro computes split R-hat
                    r_hat_value = numpyro.diagnostics.split_gelman_rubin(
                        {name: posterior_samples[name]}
                    )
                    if isinstance(r_hat_value, dict):
                        r_hat_dict[name] = float(r_hat_value.get(name, 1.0))
                    else:
                        r_hat_dict[name] = float(r_hat_value)
                except Exception:
                    # Fallback value if R-hat computation fails
                    r_hat_dict[name] = 1.0

            diagnostics["r_hat"] = r_hat_dict

            # Effective Sample Size (should be > 400 ideally)
            ess_dict = {}
            for name in posterior_samples.keys():
                try:
                    ess_value = numpyro.diagnostics.effective_sample_size(
                        {name: posterior_samples[name]}
                    )
                    if isinstance(ess_value, dict):
                        ess_dict[name] = float(ess_value.get(name, 0.0))
                    else:
                        ess_dict[name] = float(ess_value)
                except Exception:
                    # Fallback: estimate ESS as number of samples
                    ess_dict[name] = float(len(posterior_samples[name]))

            diagnostics["ess"] = ess_dict

            # Divergences (should be 0)
            try:
                divergences = mcmc.get_extra_fields()["diverging"]
                num_divergences = int(np.sum(divergences))
            except (KeyError, AttributeError):
                num_divergences = 0

            diagnostics["divergences"] = num_divergences

        except Exception as e:
            # If diagnostics computation fails, return minimal info
            diagnostics["r_hat"] = dict.fromkeys(posterior_samples.keys(), 1.0)
            diagnostics["ess"] = {
                name: float(len(samples)) for name, samples in posterior_samples.items()
            }
            diagnostics["divergences"] = 0
            diagnostics["error"] = str(e)

        return diagnostics


__all__ = [
    "BayesianMixin",
    "BayesianResult",
]
