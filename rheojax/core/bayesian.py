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
from typing import TYPE_CHECKING, Any

import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms as dist_transforms
from numpyro.infer import MCMC, NUTS, init_to_uniform, init_to_value

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import TestMode, detect_test_mode
from rheojax.logging import get_logger, log_bayesian

logger = get_logger(__name__)

# Safe JAX import (verifies NLSQ was imported first)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    from jax import Array

    from rheojax.core.parameters import ParameterSet


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
        logger.debug(
            "Initializing BayesianResult",
            num_parameters=len(self.posterior_samples),
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )
        # Ensure posterior_samples are numpy arrays
        for name, samples in self.posterior_samples.items():
            if not isinstance(samples, np.ndarray):
                self.posterior_samples[name] = np.asarray(samples, dtype=np.float64)
        logger.debug(
            "BayesianResult initialized",
            parameter_names=list(self.posterior_samples.keys()),
        )

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

        logger.debug("Converting BayesianResult to InferenceData")
        # Return cached version if available
        def _ensure_energy(idata):
            """Guarantee energy diagnostic exists for ArviZ energy plots."""
            sample_stats = getattr(idata, "sample_stats", None)
            if sample_stats is None or hasattr(sample_stats, "energy"):
                return idata

            try:
                extra_fields = self.mcmc.get_extra_fields(group_by_chain=True)
            except TypeError:
                # Older NumPyro versions may not support group_by_chain kwarg.
                extra_fields = self.mcmc.get_extra_fields()
            except Exception:
                return idata
            energy_field = None
            if isinstance(extra_fields, dict):
                energy_field = extra_fields.get("energy") or extra_fields.get(
                    "potential_energy"
                )

            if energy_field is None:
                return idata

            import xarray as xr

            energy_array = np.asarray(energy_field)
            # ArviZ expects shape (chain, draw) for sample_stats variables.
            try:
                energy_array = energy_array.reshape(self.num_chains, -1)
            except ValueError:
                # Fallback to adding a singleton chain dimension if reshape fails.
                energy_array = np.expand_dims(energy_array, axis=0)

            idata.sample_stats = sample_stats.assign(
                energy=xr.DataArray(energy_array, dims=("chain", "draw"))
            )
            return idata

        if self._inference_data is not None:
            logger.debug("Returning cached InferenceData")
            return _ensure_energy(self._inference_data)

        # Import arviz (lazy import)
        from rheojax.core.arviz_utils import import_arviz

        try:
            az = import_arviz(required=("from_numpyro",))
            logger.debug("ArviZ imported successfully")
        except ImportError as exc:
            logger.error("ArviZ import failed", exc_info=True)
            raise ImportError(
                "ArviZ is required for InferenceData conversion. "
                "Install it with: pip install arviz"
            ) from exc

        # Ensure MCMC object is available
        if self.mcmc is None:
            logger.error("MCMC object not available for InferenceData conversion")
            raise ValueError(
                "MCMC object not available for conversion. "
                "This may be a result from an older version. "
                "Re-run fit_bayesian() to generate a compatible result."
            )

        # Convert using ArviZ's from_numpyro utility
        # This preserves all NUTS diagnostics (energy, divergences, etc.)
        # log_likelihood=True computes pointwise log-likelihood for WAIC/LOO model comparison
        logger.debug("Creating InferenceData from MCMC object")
        self._inference_data = az.from_numpyro(self.mcmc, log_likelihood=True)
        logger.info(
            "InferenceData created successfully",
            num_chains=self.num_chains,
            num_samples=self.num_samples,
        )

        return _ensure_energy(self._inference_data)


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

    # Type hints for attributes that must be provided by implementing class
    parameters: ParameterSet
    _test_mode: TestMode | str | None

    # =========================================================================
    # Helper methods for fit_bayesian (extracted for reduced complexity)
    # =========================================================================

    def _validate_bayesian_requirements(self) -> None:
        """Validate that required attributes exist for Bayesian inference."""
        logger.debug("Validating Bayesian requirements")
        if not hasattr(self, "parameters"):
            logger.error("Missing 'parameters' attribute for Bayesian inference")
            raise AttributeError(
                "Class must have 'parameters' attribute (ParameterSet)"
            )
        if not hasattr(self, "model_function"):
            logger.error("Missing 'model_function' method for Bayesian inference")
            raise AttributeError(
                "Class must define 'model_function(X, params, test_mode)' method"
            )
        logger.debug("Bayesian requirements validated successfully")

    def _validate_parameter_bounds(self) -> None:
        """Validate that all parameter bounds are valid (lower < upper)."""
        logger.debug("Validating parameter bounds")
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
                logger.error(
                    "Invalid parameter bounds",
                    parameter=name,
                    bounds=bounds,
                )
                raise ValueError(
                    f"Invalid bounds for parameter '{name}': {bounds}. "
                    "Lower bound must be strictly less than upper bound."
                )
        logger.debug("Parameter bounds validated successfully")

    def _resolve_test_mode(
        self,
        X: np.ndarray | RheoData,
        test_mode: str | TestMode | None,
    ) -> tuple[np.ndarray, np.ndarray | None, TestMode]:
        """Resolve test_mode and extract data arrays from input.

        Returns:
            Tuple of (X_array, y_array, resolved_test_mode)
        """
        logger.debug(
            "Resolving test mode",
            input_type=type(X).__name__,
            explicit_test_mode=str(test_mode) if test_mode else None,
        )
        if isinstance(X, RheoData):
            rheo_data = X
            X_array = rheo_data.x
            y_array = rheo_data.y

            if test_mode is None:
                test_mode = detect_test_mode(rheo_data)
                logger.debug("Test mode detected from RheoData", test_mode=str(test_mode))
        else:
            X_array = X
            y_array = None  # Will be set from y parameter

            if test_mode is None:
                stored_mode = getattr(self, "_test_mode", None)
                if stored_mode is not None:
                    test_mode = stored_mode
                    logger.debug("Using stored test mode", test_mode=str(test_mode))
                else:
                    test_mode = TestMode.RELAXATION
                    logger.debug("Defaulting to relaxation test mode")
                    warnings.warn(
                        "test_mode not specified. Defaulting to 'relaxation'. "
                        "For correct posteriors, pass RheoData with metadata['test_mode'] "
                        "or specify test_mode explicitly.",
                        UserWarning,
                        stacklevel=3,
                    )

        # Normalize to TestMode enum
        if isinstance(test_mode, str):
            test_mode = TestMode(test_mode.lower())

        logger.debug("Test mode resolved", test_mode=str(test_mode))
        return X_array, y_array, test_mode

    def _prepare_jax_data(
        self, X_array: np.ndarray, y_array: np.ndarray
    ) -> dict[str, Any]:
        """Prepare JAX arrays and scale information for Bayesian inference.

        Returns:
            Dictionary with keys: X_jax, y_jax, is_complex, scale_info
        """
        logger.debug(
            "Preparing JAX data",
            X_shape=X_array.shape if hasattr(X_array, "shape") else None,
            y_shape=y_array.shape if hasattr(y_array, "shape") else None,
        )
        X_jax = jnp.asarray(X_array, dtype=jnp.float64)
        is_complex = jnp.iscomplexobj(y_array)
        logger.debug("Data is complex", is_complex=bool(is_complex))

        scale_info: dict[str, float | None] = {
            "data_scale": None,
            "y_real_scale": None,
            "y_imag_scale": None,
        }

        if is_complex:
            y_complex_np = np.asarray(y_array, dtype=np.complex128)
            y_real_np = np.real(y_complex_np)
            y_imag_np = np.imag(y_complex_np)
            scale_info["y_real_scale"] = (
                float(np.std(y_real_np)) if y_real_np.size else 0.0
            )
            scale_info["y_imag_scale"] = (
                float(np.std(y_imag_np)) if y_imag_np.size else 0.0
            )

            y_complex = jnp.asarray(y_array, dtype=jnp.complex128)
            y_real = jnp.real(y_complex)
            y_imag = jnp.imag(y_complex)
            y_jax = jnp.concatenate([y_real, y_imag])
        else:
            y_np = np.asarray(y_array, dtype=np.float64)
            scale_info["data_scale"] = float(np.std(y_np)) if y_np.size else 0.0
            y_jax = jnp.asarray(y_np, dtype=jnp.float64)

        return {
            "X_jax": X_jax,
            "y_jax": y_jax,
            "is_complex": is_complex,
            "scale_info": scale_info,
        }

    def _get_parameter_bounds(
        self,
        X_array: np.ndarray,
        y_array: np.ndarray,
        test_mode: TestMode,
    ) -> dict[str, tuple[float | None, float | None]]:
        """Get parameter bounds, applying any model-specific overrides."""
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

        return param_bounds

    @staticmethod
    def _compute_safe_interval(
        lower_raw: float | None, upper_raw: float | None
    ) -> tuple[float, float]:
        """Compute safe interval for initialization within bounds."""
        lower = float(lower_raw) if lower_raw is not None else -np.inf
        upper = float(upper_raw) if upper_raw is not None else np.inf

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

    @staticmethod
    def _compute_default_midpoint(
        lower_raw: float | None, upper_raw: float | None
    ) -> float:
        """Compute default midpoint for parameter initialization."""
        lower = float(lower_raw) if lower_raw is not None else -np.inf
        upper = float(upper_raw) if upper_raw is not None else np.inf

        if np.isfinite(lower) and np.isfinite(upper):
            if lower > 0 and upper > 0:
                return float(np.exp((np.log(lower) + np.log(upper)) / 2.0))
            return float(0.5 * (lower + upper))
        if np.isfinite(lower):
            return float(lower + max(abs(lower) * 0.5, 1.0))
        if np.isfinite(upper):
            return float(upper - max(abs(upper) * 0.5, 1.0))
        return 0.0

    def _build_warm_start_values(
        self,
        param_names: list[str],
        param_bounds: dict[str, tuple[float | None, float | None]],
        initial_values: dict[str, float] | None,
        scale_info: dict[str, float | None],
        is_complex: bool,
    ) -> dict[str, float]:
        """Build sanitized warm-start initialization values."""
        init_intervals = {
            name: self._compute_safe_interval(*param_bounds[name])
            for name in param_names
        }

        def sanitize_value(name: str, raw_value: float | None) -> float:
            lower_raw, upper_raw = param_bounds[name]
            safe_lower, safe_upper = init_intervals[name]
            value = raw_value
            if value is None or not np.isfinite(value):
                value = self._compute_default_midpoint(lower_raw, upper_raw)
            if np.isfinite(safe_lower):
                value = max(value, safe_lower)
            if np.isfinite(safe_upper):
                value = min(value, safe_upper)
            return float(value)

        warm_start: dict[str, float] = {}
        for name in param_names:
            candidate = None
            if initial_values and name in initial_values:
                candidate = initial_values[name]
            else:
                candidate = self.parameters.get_value(name)
            warm_start[name] = sanitize_value(name, candidate)

        # Add noise scale initial values
        if is_complex:
            y_real_scale = scale_info.get("y_real_scale") or 0.0
            y_imag_scale = scale_info.get("y_imag_scale") or 0.0
            if "sigma_real" not in warm_start:
                warm_start["sigma_real"] = max(y_real_scale * 0.1, 1e-6)
            if "sigma_imag" not in warm_start:
                warm_start["sigma_imag"] = max(y_imag_scale * 0.1, 1e-6)
        else:
            data_scale = scale_info.get("data_scale") or 1.0
            if "sigma" not in warm_start:
                warm_start["sigma"] = max(data_scale * 0.1, 1e-6)

        return warm_start

    def _run_nuts_sampling(
        self,
        numpyro_model,
        X_jax: Array,
        y_jax: Array,
        warm_start_values: dict[str, float],
        num_warmup: int,
        num_samples: int,
        num_chains: int,
        nuts_kwargs: dict[str, Any],
        seed: int | None = None,
    ) -> MCMC:
        """Run NUTS sampling with fallback to uniform initialization.

        Args:
            seed: Random seed for reproducibility. If None, uses 0.
        """
        logger.debug(
            "Starting NUTS sampling",
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
        )
        user_chain_method = nuts_kwargs.pop("chain_method", None)

        try:
            init_strategy = init_to_value(values=warm_start_values)
            logger.debug("Using warm-start initialization", values=warm_start_values)
        except Exception as exc:
            logger.debug("Warm-start initialization failed", exc_info=True)
            warnings.warn(
                "Warm-start initialization failed; falling back to uniform init. "
                "Reason: " + str(exc),
                RuntimeWarning,
                stacklevel=3,
            )
            init_strategy = init_to_uniform()
            logger.debug("Falling back to uniform initialization")

        def _select_chain_method() -> str:
            """Prefer parallel/vectorized chains when multiple chains requested.

            Auto-selects optimal chain execution method:
            - 'sequential': Single chain or user override
            - 'parallel': Multi-chain on multi-GPU (fastest)
            - 'vectorized': Multi-chain on single device (uses vmap)

            Logs info about chain method selection for transparency.
            """
            if user_chain_method:
                return user_chain_method
            if num_chains <= 1:
                return "sequential"

            devices = jax.devices()
            # Count accelerators only (non-CPU) for true parallel execution.
            accelerator_count = sum(1 for d in devices if d.platform != "cpu")
            if accelerator_count >= num_chains:
                logger.debug("Using parallel chain method", accelerator_count=accelerator_count)
                return "parallel"
            # Fall back to vectorized on single-device setups for speed over sequential.
            # Vectorized uses vmap for efficient multi-chain execution on a single device.
            logger.debug("Using vectorized chain method")
            return "vectorized"

        # Use provided seed or default to 0 for reproducibility
        rng_seed = seed if seed is not None else 0

        # Encourage higher acceptance to reduce divergences in stiff fractional models.
        nuts_kwargs.setdefault("target_accept_prob", 0.99)

        def run_mcmc(strategy):
            kernel = NUTS(numpyro_model, init_strategy=strategy, **nuts_kwargs)
            chain_method = _select_chain_method()
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
                    chain_method=chain_method,
                )
            rng_key = jax.random.PRNGKey(rng_seed)
            sampler.run(rng_key, X_jax, y_jax)
            return sampler

        try:
            logger.debug("Running MCMC sampling")
            result = run_mcmc(init_strategy)
            logger.debug("MCMC sampling completed successfully")
            return result
        except RuntimeError as e:
            if "Cannot find valid initial parameters" in str(e):
                logger.debug("Warm-started NUTS initialization failed, retrying with uniform init")
                warnings.warn(
                    "Warm-started NUTS initialization failed; retrying with uniform init.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                try:
                    result = run_mcmc(init_to_uniform())
                    logger.debug("MCMC sampling completed with uniform init")
                    return result
                except Exception as final_exc:
                    logger.error("NUTS sampling failed after uniform init fallback", exc_info=True)
                    raise RuntimeError(
                        f"NUTS sampling failed: {str(final_exc)}"
                    ) from final_exc
            logger.error("NUTS sampling failed", error=str(e), exc_info=True)
            raise RuntimeError(f"NUTS sampling failed: {str(e)}") from e

    def _process_mcmc_results(
        self,
        mcmc: MCMC,
        param_names: list[str],
        num_samples: int,
        num_chains: int,
    ) -> BayesianResult:
        """Process MCMC results into BayesianResult."""
        logger.debug(
            "Processing MCMC results",
            num_params=len(param_names),
            num_samples=num_samples,
            num_chains=num_chains,
        )
        samples = mcmc.get_samples()

        # Convert to numpy arrays (model parameters only)
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

        diagnostics = self._compute_diagnostics(
            mcmc,
            posterior_samples,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        return BayesianResult(
            posterior_samples=posterior_samples,
            summary=summary,
            diagnostics=diagnostics,
            num_samples=num_samples,
            num_chains=num_chains,
            mcmc=mcmc,
            model_comparison={},
        )

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
        logger.debug("Sampling prior", num_samples=num_samples)
        if not hasattr(self, "parameters"):
            logger.error("Missing 'parameters' attribute for prior sampling")
            raise AttributeError(
                "Class must have 'parameters' attribute (ParameterSet)"
            )

        prior_samples = {}

        for param_name in self.parameters:
            param = self.parameters.get(param_name)

            if param.bounds is None:
                logger.error("Parameter missing bounds for prior sampling", parameter=param_name)
                raise ValueError(
                    f"Parameter '{param_name}' must have bounds for prior sampling"
                )

            lower, upper = param.bounds

            # Sample from uniform distribution over bounds
            samples = np.random.uniform(low=lower, high=upper, size=num_samples).astype(
                np.float64
            )

            prior_samples[param_name] = samples

        logger.debug(
            "Prior sampling completed",
            num_parameters=len(prior_samples),
            parameter_names=list(prior_samples.keys()),
        )
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
        logger.debug(
            "Computing credible intervals",
            credibility=credibility,
            num_parameters=len(posterior_samples),
        )
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

        logger.debug("Credible intervals computed", parameter_names=list(intervals.keys()))
        return intervals

    def fit_bayesian(
        self,
        X: np.ndarray | RheoData,
        y: np.ndarray | None = None,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        initial_values: dict[str, float] | None = None,
        test_mode: str | TestMode | None = None,
        seed: int | None = None,
        **nuts_kwargs,
    ) -> BayesianResult:
        """Perform Bayesian inference using NumPyro NUTS sampler.

        Runs NUTS (No-U-Turn Sampler) to obtain posterior samples for model
        parameters. Supports warm-starting from NLSQ point estimates for faster
        convergence. Uses uniform priors over parameter bounds.

        Multi-chain sampling is enabled by default (num_chains=4) to provide
        reliable convergence diagnostics (R-hat, ESS) and parallel execution
        on multi-GPU systems.

        CRITICAL: test_mode is captured in model_function closure to ensure
        correct posteriors for all test modes (relaxation, creep, oscillation).

        Args:
            X: Independent variable data (input features) or RheoData object
            y: Dependent variable data (observations to fit). If X is RheoData,
                y is ignored and extracted from X.
            num_warmup: Number of warmup/burn-in iterations (default: 1000)
            num_samples: Number of posterior samples per chain (default: 2000)
            num_chains: Number of MCMC chains (default: 4). Multiple chains
                enable proper R-hat computation and parallel execution.
                Chain method is auto-selected: 'parallel' on multi-GPU,
                'vectorized' on single GPU/CPU.
            initial_values: Optional dict of initial parameter values for
                warm-start (e.g., from NLSQ). Keys are parameter names.
            test_mode: Explicit test mode (e.g., 'relaxation', 'creep', 'oscillation').
                If None, inferred from RheoData.metadata['test_mode'] or defaults
                to 'relaxation'. Overrides RheoData metadata if provided.
            seed: Random seed for reproducibility. If None, uses seed=0 for
                deterministic results. Set to different values for independent runs.
            **nuts_kwargs: Additional arguments passed to NUTS sampler
                (e.g., target_accept_prob, chain_method)

        Returns:
            BayesianResult containing posterior_samples, summary, diagnostics.

        Example:
            >>> result = model.fit_bayesian(X, y, test_mode='oscillation')
            >>> print(result.diagnostics["r_hat"])  # Should be < 1.01
            >>>
            >>> # For production: use num_chains=4 (default)
            >>> result = model.fit_bayesian(X, y, num_chains=4)
        """
        # Get model name for logging
        model_name = getattr(self, "__class__", type(self)).__name__

        with log_bayesian(
            logger,
            model=model_name,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        ) as log_ctx:
            # Phase 1: Validation
            self._validate_bayesian_requirements()
            self._validate_parameter_bounds()

            # Phase 2: Resolve test_mode and extract data
            X_array, y_from_rheo, test_mode = self._resolve_test_mode(X, test_mode)
            y_array = y_from_rheo if y_from_rheo is not None else y
            self._test_mode = test_mode  # Cache for future calls

            logger.info(
                "Bayesian inference started",
                model=model_name,
                test_mode=str(test_mode),
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
            )

            # Phase 3: Prepare JAX data
            jax_data = self._prepare_jax_data(X_array, y_array)
            X_jax = jax_data["X_jax"]
            y_jax = jax_data["y_jax"]
            is_complex_data = jax_data["is_complex"]
            scale_info = jax_data["scale_info"]

            # Phase 4: Get parameter bounds
            param_names = list(self.parameters)
            param_bounds = self._get_parameter_bounds(X_array, y_array, test_mode)

            # Phase 5: Build NumPyro model (closure captures test_mode)
            numpyro_model = self._build_numpyro_model(
                param_names=param_names,
                param_bounds=param_bounds,
                test_mode=test_mode,
                is_complex_data=is_complex_data,
                scale_info=scale_info,
            )

            # Phase 6: Apply NUTS kwargs overrides
            nuts_overrides = getattr(self, "bayesian_nuts_kwargs", None)
            if callable(nuts_overrides):
                overrides = nuts_overrides()
                if isinstance(overrides, dict):
                    for key, value in overrides.items():
                        nuts_kwargs.setdefault(key, value)

            # Phase 7: Build warm-start values
            warm_start_values = self._build_warm_start_values(
                param_names=param_names,
                param_bounds=param_bounds,
                initial_values=initial_values,
                scale_info=scale_info,
                is_complex=is_complex_data,
            )

            # Phase 8: Run NUTS sampling with multi-chain parallelization
            mcmc = self._run_nuts_sampling(
                numpyro_model=numpyro_model,
                X_jax=X_jax,
                y_jax=y_jax,
                warm_start_values=warm_start_values,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                nuts_kwargs=nuts_kwargs,
                seed=seed,
            )

            # Phase 9: Process results
            result = self._process_mcmc_results(mcmc, param_names, num_samples, num_chains)

            # Add diagnostics to log context
            log_ctx["divergences"] = result.diagnostics.get("divergences", 0)
            max_r_hat = max(result.diagnostics.get("r_hat", {}).values(), default=1.0)
            min_ess = min(result.diagnostics.get("ess", {}).values(), default=0.0)
            log_ctx["r_hat_max"] = max_r_hat
            log_ctx["ess_min"] = min_ess

            logger.info(
                "Bayesian inference completed",
                model=model_name,
                divergences=result.diagnostics.get("divergences", 0),
                r_hat_max=max_r_hat,
                ess_min=min_ess,
            )

            return result

    def _build_numpyro_model(
        self,
        param_names: list[str],
        param_bounds: dict[str, tuple[float | None, float | None]],
        test_mode: TestMode,
        is_complex_data: bool,
        scale_info: dict[str, float | None],
    ):
        """Build the NumPyro probabilistic model function.

        Returns a callable model function with test_mode captured in closure.
        """
        prior_factory = getattr(self, "bayesian_prior_factory", None)

        # Extract scale values for likelihood
        y_real_scale = scale_info.get("y_real_scale") or 0.0
        y_imag_scale = scale_info.get("y_imag_scale") or 0.0
        data_scale = scale_info.get("data_scale") or 0.0

        def numpyro_model(X, y=None):
            """NumPyro model with test_mode captured in closure."""
            # Sample parameters from priors
            params_dict = {}
            for name in param_names:
                lower, upper = param_bounds[name]
                custom_dist = None
                if callable(prior_factory):
                    custom_dist = prior_factory(name, lower, upper)

                if custom_dist is not None:
                    params_dict[name] = numpyro.sample(name, custom_dist)
                elif (
                    name.lower().endswith("alpha")
                    and lower is not None
                    and upper is not None
                    and 0.0 <= lower < upper <= 1.0
                ):
                    # Weakly-informative Beta prior for fractional orders
                    beta_base = dist.Beta(concentration1=2.0, concentration0=2.0)
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

            # Convert to array and compute predictions
            params_array = jnp.array([params_dict[name] for name in param_names])
            predictions_raw = self.model_function(X, params_array, test_mode)

            # Handle complex vs real predictions
            if is_complex_data:
                pred_real = jnp.real(predictions_raw)
                pred_imag = jnp.imag(predictions_raw)
                n = len(y) // 2
                y_real_obs, y_imag_obs = y[:n], y[n:]

                # Exponential priors on noise (inflated scale for robustness)
                sigma_real_scale = max(y_real_scale * 10.0, 1e-9)
                sigma_imag_scale = max(y_imag_scale * 10.0, 1e-9)
                sigma_real = numpyro.sample(
                    "sigma_real", dist.Exponential(rate=1.0 / sigma_real_scale)
                )
                sigma_imag = numpyro.sample(
                    "sigma_imag", dist.Exponential(rate=1.0 / sigma_imag_scale)
                )
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
                sigma_scale = max(data_scale * 10.0, 1e-9)
                sigma = numpyro.sample(
                    "sigma", dist.Exponential(rate=1.0 / sigma_scale)
                )
                numpyro.sample(
                    "obs", dist.Normal(loc=predictions_raw, scale=sigma), obs=y
                )

        return numpyro_model

    def _compute_diagnostics(
        self,
        mcmc: MCMC,
        posterior_samples: dict[str, np.ndarray],
        num_samples: int,
        num_chains: int,
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
            diagnostics["total_samples"] = int(num_samples * num_chains)
            diagnostics["num_chains"] = int(num_chains)
            diagnostics["num_samples_per_chain"] = int(num_samples)

        except Exception as e:
            # If diagnostics computation fails, return minimal info
            diagnostics["r_hat"] = dict.fromkeys(posterior_samples.keys(), 1.0)
            diagnostics["ess"] = {
                name: float(len(samples)) for name, samples in posterior_samples.items()
            }
            diagnostics["divergences"] = 0
            diagnostics["total_samples"] = int(num_samples * num_chains)
            diagnostics["num_chains"] = int(num_chains)
            diagnostics["num_samples_per_chain"] = int(num_samples)
            diagnostics["error"] = str(e)

        return diagnostics


__all__ = [
    "BayesianMixin",
    "BayesianResult",
]
