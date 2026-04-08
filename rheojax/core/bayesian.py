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

import copy
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.bayesian_diagnostics import compute_diagnostics
from rheojax.core.bayesian_result import BayesianResult, DiagnosticsDict
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.numpyro_model_builder import build_numpyro_model, prior_dict_to_dist
from rheojax.core.test_modes import TestMode, detect_test_mode
from rheojax.logging import get_logger, log_bayesian

logger = get_logger(__name__)

# Safe JAX import (verifies NLSQ was imported first)
jax, jnp = safe_import_jax()


def _import_numpyro():
    """Lazy-import NumPyro and its submodules.

    Returns all NumPyro symbols needed by BayesianMixin. Python caches
    modules after the first import, so subsequent calls are a cheap dict
    lookup (~µs). This avoids the ~800ms startup cost when only NLSQ
    (non-Bayesian) workflows are used.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import transforms as dist_transforms
    from numpyro.infer import MCMC, NUTS, init_to_uniform, init_to_value

    return numpyro, dist, dist_transforms, MCMC, NUTS, init_to_uniform, init_to_value


if TYPE_CHECKING:
    from jax import Array
    from numpyro.infer import MCMC

    from rheojax.core.parameters import ParameterSet


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
        """Validate that required attributes exist for Bayesian inference.

        R12-B-007: The ordering — check 'parameters' before 'model_function' —
        is intentional and safe.  'parameters' is always set in __init__ so the
        check is fast and cheap.  Checking it first ensures a clear AttributeError
        (rather than a confusing downstream KeyError) when the mixin is used on a
        class that forgot to call super().__init__().
        """
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
        """Validate that all parameter bounds are valid.

        Parameters with near-equal lower and upper bounds are treated as
        **fixed** and silently skipped — they won't be sampled by NUTS.
        Near-equality uses relative tolerance ``1e-10 * max(|lo|, 1)`` to
        handle floating-point roundtrip from factory methods that set
        identical bounds.  Only bounds where lower > upper are invalid.
        """
        logger.debug("Validating parameter bounds")
        for name in self.parameters.keys():
            param = self.parameters.get(name)
            if param is None:
                continue
            bounds = getattr(param, "bounds", None)
            if bounds is None or bounds[0] is None or bounds[1] is None:
                continue
            # Equal bounds → fixed parameter, skip (not sampled by NUTS)
            if abs(bounds[1] - bounds[0]) < max(abs(bounds[0]), 1.0) * 1e-10:
                logger.debug(
                    "Parameter has equal bounds (fixed), skipping",
                    parameter=name,
                    value=bounds[0],
                )
                continue
            if bounds[0] > bounds[1]:
                logger.error(
                    "Invalid parameter bounds",
                    parameter=name,
                    bounds=bounds,
                )
                raise ValueError(
                    f"Invalid bounds for parameter '{name}': {bounds}. "
                    "Lower bound must be less than or equal to upper bound."
                )
        logger.debug("Parameter bounds validated successfully")

    def _resolve_test_mode(
        self,
        X: np.ndarray | RheoData,
        test_mode: str | TestMode | None,
    ) -> tuple[np.ndarray, np.ndarray | None, TestMode]:
        """Resolve test_mode and extract data arrays from input.

        Priority order for the returned y_array (R12-B-003):
          1. y extracted from a RheoData X object (highest priority — data and
             labels travel together, avoiding accidental mismatches).
          2. The caller's explicit ``y`` parameter (used when X is a plain array).
          3. ``None`` — the caller is responsible for supplying y from another source.

        The resolved test_mode follows a similar hierarchy:
          1. Explicit ``test_mode`` argument passed by the caller.
          2. Mode detected from RheoData metadata / auto-detection.
          3. ``self._test_mode`` stored from a prior NLSQ fit().
          4. ``self._last_fit_kwargs["test_mode"]`` as a secondary NLSQ fallback.
          5. ``TestMode.RELAXATION`` as the final default (with a UserWarning).

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
                logger.debug(
                    "Test mode detected from RheoData", test_mode=str(test_mode)
                )
        else:
            X_array = X
            y_array = None  # Will be set from y parameter

            if test_mode is None:
                # R12-B-013: three-tier fallback when X is a plain array and
                # no explicit test_mode was supplied:
                #   Tier 1 — self._test_mode set by a previous fit() call.
                #   Tier 2 — self._last_fit_kwargs["test_mode"] stored by _fit().
                #   Tier 3 — TestMode.RELAXATION hard default with a UserWarning.
                stored_mode = getattr(self, "_test_mode", None)
                # R10-BAY-002: Also check _last_fit_kwargs as a second fallback.
                # _fit() stores test_mode there; this covers the case where
                # _test_mode was not set but _fit() has run.
                if stored_mode is None:
                    _lfk = getattr(self, "_last_fit_kwargs", None)
                    if _lfk is None:
                        _lfk = {}
                    stored_mode = _lfk.get("test_mode", None)
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
        # Try model's _validate_test_mode first for special cases (e.g., 'laos' -> STARTUP)
        if isinstance(test_mode, str):
            if hasattr(self, "_validate_test_mode"):
                try:
                    test_mode = self._validate_test_mode(test_mode)
                except (ValueError, AttributeError):
                    # Fallback to standard TestMode conversion
                    test_mode = TestMode(test_mode.lower())
            else:
                test_mode = TestMode(test_mode.lower())

        logger.debug("Test mode resolved", test_mode=str(test_mode))
        return X_array, y_array, test_mode  # type: ignore[return-value]

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
        # R5-JAX-006: `isinstance(y_array, jnp.ndarray)` is unreliable in
        # JAX >= 0.4.7 where jnp.ndarray is an alias for jax.Array (an abstract
        # base class) — isinstance checks against it are deprecated and may
        # return False for valid JAX arrays.  Use duck-typing instead:
        # presence of `.devices` distinguishes a JAX array from NumPy/Python
        # objects (pattern established in optimization.py).
        if not (isinstance(y_array, np.ndarray) or hasattr(y_array, "devices")):
            y_array = np.asarray(y_array)
        is_complex = jnp.iscomplexobj(y_array)
        logger.debug("Data is complex", is_complex=bool(is_complex))

        scale_info: dict[str, float | None] = {
            "data_scale": None,
            "y_real_scale": None,
            "y_imag_scale": None,
        }

        if is_complex:
            # Single conversion to JAX, then compute real/imag components
            # Avoids redundant CPU→JAX conversion (10-20% overhead reduction)
            y_complex = jnp.asarray(y_array, dtype=jnp.complex128)
            if y_complex.dtype != jnp.complex128:
                logger.warning(
                    "complex128 downcast detected — GPU may not support float64 complex. "
                    "Bayesian oscillation fits may lose precision.",
                    actual_dtype=str(y_complex.dtype),
                )
            y_real = jnp.real(y_complex)
            y_imag = jnp.imag(y_complex)

            # SYS-09: Batch all four reductions into one jnp.stack + single
            # device→host transfer, avoiding 4 separate sync stalls.
            if y_real.size:
                _stats = jnp.stack(
                    [
                        jnp.std(y_real),
                        jnp.std(y_imag),
                        jnp.mean(jnp.abs(y_real)),
                        jnp.mean(jnp.abs(y_imag)),
                    ]
                ).tolist()
                scale_info["y_real_scale"] = _stats[0]
                scale_info["y_imag_scale"] = _stats[1]
                scale_info["y_real_mean"] = _stats[2]
                scale_info["y_imag_mean"] = _stats[3]
            else:
                scale_info["y_real_scale"] = 0.0
                scale_info["y_imag_scale"] = 0.0
                scale_info["y_real_mean"] = 0.0
                scale_info["y_imag_mean"] = 0.0

            scale_info["n_real"] = int(y_real.shape[0])
            y_jax = jnp.concatenate([y_real, y_imag])
        else:
            y_np = np.asarray(y_array, dtype=np.float64)
            scale_info["data_scale"] = float(np.std(y_np)) if y_np.size else 0.0
            # Mean magnitude for sigma prior flooring (constant-data guard)
            scale_info["data_mean"] = float(np.mean(np.abs(y_np))) if y_np.size else 0.0
            y_jax = jnp.asarray(y_np, dtype=jnp.float64)

        # R13-BAY-001: Populate n_points so _build_numpyro_model's probe uses
        # the actual data size instead of a hardcoded 10.
        scale_info["n_points"] = int(X_jax.shape[0])

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
            if param is None:
                raise ValueError(f"Parameter '{name}' not found in model parameters")
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
            safe_lower, safe_upper = lower + pad, upper - pad
            if safe_lower >= safe_upper:
                return lower, upper
            return safe_lower, safe_upper
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
            original = value
            if np.isfinite(safe_lower):
                value = max(value, safe_lower)
            if np.isfinite(safe_upper):
                value = min(value, safe_upper)
            if value != original:
                logger.debug(
                    "Bayesian warm-start clamped",
                    parameter=name,
                    original=original,
                    clamped=value,
                    interval=(float(safe_lower), float(safe_upper)),
                )
            return float(value)

        warm_start: dict[str, float] = {}
        _is_fitted = getattr(self, "fitted_", False)
        for name in param_names:
            candidate = None
            if initial_values is not None and name in initial_values:
                candidate = initial_values[name]
            else:
                candidate = self.parameters.get_value(name)
                # Warn if model claims to be fitted but parameter value is None,
                # indicating NLSQ didn't write back to ParameterSet for this param.
                if _is_fitted and candidate is None:
                    logger.warning(
                        "Fitted model has None parameter value — warm-start will "
                        "use bounds midpoint instead of NLSQ estimate",
                        parameter=name,
                    )
            warm_start[name] = sanitize_value(name, candidate)

        # Add noise scale initial values.
        # Use explicit None-check (not or-sentinel) so that a legitimately zero
        # std dev (constant-data edge case) is preserved rather than silently
        # replaced by 0.0 — both cases ultimately floor to 1e-6 via max().
        if is_complex:
            _yr = scale_info.get("y_real_scale")
            y_real_scale = _yr if _yr is not None else 0.0
            _yi = scale_info.get("y_imag_scale")
            y_imag_scale = _yi if _yi is not None else 0.0
            if "sigma_real" not in warm_start:
                warm_start["sigma_real"] = max(y_real_scale * 0.1, 1e-6)
            if "sigma_imag" not in warm_start:
                warm_start["sigma_imag"] = max(y_imag_scale * 0.1, 1e-6)
        else:
            _ds = scale_info.get("data_scale")
            data_scale = _ds if _ds is not None else 0.0
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
        _, _, _, MCMC, NUTS, init_to_uniform, init_to_value = _import_numpyro()

        logger.debug(
            "Starting NUTS sampling",
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
        )
        user_chain_method = nuts_kwargs.pop("chain_method", None)

        # Track which init strategy is actually used for diagnostics
        self._nuts_init_strategy = "warm_start"

        try:
            init_strategy = init_to_value(values=warm_start_values)
            logger.debug("Using warm-start initialization", values=warm_start_values)
        except Exception as exc:
            logger.warning(
                "Warm-start init_to_value construction failed — falling back "
                "to uniform init",
                error=str(exc),
            )
            warnings.warn(
                "Warm-start initialization failed; falling back to uniform init. "
                "Reason: " + str(exc),
                RuntimeWarning,
                stacklevel=3,
            )
            init_strategy = init_to_uniform()
            self._nuts_init_strategy = "uniform_fallback"

        def _select_chain_method() -> str:
            """Prefer parallel/vectorized chains when multiple chains requested."""
            if user_chain_method:
                return user_chain_method
            if num_chains <= 1:
                return "sequential"

            devices = jax.devices()
            accelerator_count = sum(1 for d in devices if d.platform != "cpu")
            if accelerator_count >= num_chains:
                logger.debug(
                    "Using parallel chain method", accelerator_count=accelerator_count
                )
                return "parallel"
            logger.debug("Using vectorized chain method")
            return "vectorized"

        # Use provided seed or default to 0 for reproducibility
        rng_seed = seed if seed is not None else 0

        # Warm-start-aware NUTS defaults
        has_warm_start = (
            bool(warm_start_values) and hasattr(self, "fitted_") and self.fitted_
        )
        if has_warm_start:
            nuts_kwargs.setdefault("target_accept_prob", 0.90)
            nuts_kwargs.setdefault("max_tree_depth", 8)
            logger.debug(
                "Using warm-start NUTS defaults",
                target_accept_prob=nuts_kwargs.get("target_accept_prob"),
                max_tree_depth=nuts_kwargs.get("max_tree_depth"),
            )
        else:
            # Conservative defaults for cold-start exploration
            nuts_kwargs.setdefault("target_accept_prob", 0.99)

        # Separate MCMC args from NUTS args
        mcmc_kwargs = {}
        if "progress_bar" in nuts_kwargs:
            mcmc_kwargs["progress_bar"] = nuts_kwargs.pop("progress_bar")
        if "jit_model_args" in nuts_kwargs:
            mcmc_kwargs["jit_model_args"] = nuts_kwargs.pop("jit_model_args")

        # Check if model requires forward-mode differentiation (for dynamic loops)
        use_forward_mode = nuts_kwargs.pop("forward_mode_differentiation", None)
        if use_forward_mode is None:
            use_forward_mode = getattr(self, "_use_forward_mode_ad", False)
        if use_forward_mode:
            nuts_kwargs["forward_mode_differentiation"] = True

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
                    **mcmc_kwargs,
                )
            rng_key = jax.random.PRNGKey(rng_seed)
            # R12-B-009: include "diverging" so _process_mcmc_results can report
            # per-transition divergence counts without a second MCMC.get_extra_fields() call.
            sampler.run(
                rng_key, X_jax, y_jax, extra_fields=("potential_energy", "diverging")
            )
            return sampler

        try:
            logger.debug("Running MCMC sampling")
            result = run_mcmc(init_strategy)
            logger.debug("MCMC sampling completed successfully")
            self._nuts_init_strategy = "warm_start"
            return result
        except RuntimeError as e:
            if "Cannot find valid initial parameters" in str(e):
                logger.warning(
                    "Warm-started NUTS initialization failed — falling back to "
                    "uniform init. Posterior samples may converge more slowly.",
                )
                warnings.warn(
                    "Warm-started NUTS initialization failed; retrying with uniform init. "
                    "NLSQ warm-start was discarded — check diagnostics carefully.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                try:
                    result = run_mcmc(init_to_uniform())
                    logger.debug("MCMC sampling completed with uniform init")
                    self._nuts_init_strategy = "uniform_fallback"
                    return result
                except Exception as final_exc:
                    logger.error(
                        "NUTS sampling failed after uniform init fallback",
                        exc_info=True,
                    )
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
        # BAY-03: call get_samples only once (group_by_chain=True), then
        # derive the flat form by reshaping — avoids a redundant device→host
        # transfer that get_samples() without group_by_chain would trigger.
        try:
            samples_grouped = mcmc.get_samples(group_by_chain=True)
            # Flat samples = concatenate along the chain axis (axis 0)
            samples = {
                k: v.reshape((-1,) + v.shape[2:]) for k, v in samples_grouped.items()
            }
        except Exception as exc:
            logger.debug(
                "group_by_chain failed, falling back to ungrouped samples",
                error=str(exc),
            )
            samples_grouped = None
            samples = mcmc.get_samples()

        # Convert to numpy arrays (model parameters only)
        posterior_samples = {}
        for name in param_names:
            if name in samples:
                posterior_samples[name] = np.asarray(samples[name], dtype=np.float64)

        noise_params = [
            k for k in samples if k not in param_names and k.startswith("sigma")
        ]
        for name in noise_params:
            posterior_samples[name] = np.asarray(samples[name], dtype=np.float64)

        # Chain-grouped samples for diagnostics (avoids reshape assumption)
        grouped_samples = {}
        if samples_grouped is not None:
            for name in param_names:
                if name in samples_grouped:
                    grouped_samples[name] = np.asarray(
                        samples_grouped[name], dtype=np.float64
                    )
            noise_params_grouped = [
                k
                for k in samples_grouped
                if k not in param_names and k.startswith("sigma")
            ]
            for name in noise_params_grouped:
                grouped_samples[name] = np.asarray(
                    samples_grouped[name], dtype=np.float64
                )

        # BAY-04: consolidate per-parameter percentile passes into one
        # np.quantile call per parameter (7 passes → 1 pass + 2 scalars).
        summary = {}
        for name, sample_array in posterior_samples.items():
            qs = np.quantile(sample_array, [0.05, 0.25, 0.50, 0.75, 0.95])
            summary[name] = {
                "mean": float(np.mean(sample_array)),
                "std": float(np.std(sample_array)),
                "median": float(qs[2]),
                "q05": float(qs[0]),
                "q25": float(qs[1]),
                "q75": float(qs[3]),
                "q95": float(qs[4]),
            }

        diagnostics = compute_diagnostics(
            mcmc,
            grouped_samples if len(grouped_samples) > 0 else posterior_samples,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        # Include which init strategy was actually used so users can detect
        # silent fallback from warm-start to uniform initialization.
        init_strategy = getattr(self, "_nuts_init_strategy", "unknown")
        diagnostics["init_strategy"] = init_strategy
        if init_strategy == "uniform_fallback":
            diagnostics["warm_start_failed"] = True

        return BayesianResult(
            posterior_samples=posterior_samples,
            summary=summary,
            diagnostics=diagnostics,
            num_samples=num_samples,
            num_chains=num_chains,
            mcmc=mcmc,
            model_comparison={},
        )

    def sample_prior(
        self, num_samples: int = 1000, seed: int | None = None
    ) -> dict[str, np.ndarray]:
        """Sample from prior distributions over parameter bounds.

        Samples from uniform prior distributions defined by parameter bounds.
        This is useful for prior predictive checks and understanding the prior's
        influence on the posterior.

        Args:
            num_samples: Number of samples to draw from prior (default: 1000)
            seed: Random seed for reproducibility (default: None)

        Returns:
            Dictionary mapping parameter names to arrays of prior samples.
            Each array has shape (num_samples,) and dtype float64.

        Raises:
            AttributeError: If class doesn't have `parameters` attribute
            ValueError: If any parameter lacks bounds

        Example:
            >>> model = MyModel()
            >>> prior_samples = model.sample_prior(num_samples=500, seed=42)
            >>> print(prior_samples["a"].shape)  # (500,)
        """
        logger.debug("Sampling prior", num_samples=num_samples, seed=seed)
        if not hasattr(self, "parameters"):
            logger.error("Missing 'parameters' attribute for prior sampling")
            raise AttributeError(
                "Class must have 'parameters' attribute (ParameterSet)"
            )

        rng = np.random.default_rng(seed)
        prior_samples = {}

        for param_name in self.parameters:
            param = self.parameters.get(param_name)

            if param is None or param.bounds is None:
                logger.error(
                    "Parameter missing bounds for prior sampling", parameter=param_name
                )
                raise ValueError(
                    f"Parameter '{param_name}' must have bounds for prior sampling"
                )

            lower, upper = param.bounds
            lower = float(lower) if lower is not None else -1e10
            upper = float(upper) if upper is not None else 1e10

            # Respect user-specified prior distribution if available
            prior = getattr(param, "prior", None)
            if isinstance(prior, dict) and prior.get("type") == "beta":
                a_val = float(prior.get("a", 2.0))
                b_val = float(prior.get("b", 2.0))
                unit_samples = rng.beta(a=a_val, b=b_val, size=num_samples)
                samples = (lower + unit_samples * (upper - lower)).astype(np.float64)
            else:
                samples = rng.uniform(low=lower, high=upper, size=num_samples).astype(
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
            except ImportError:
                # hpdi not available — use equal-tailed credible interval
                logger.debug(
                    "numpyro.diagnostics.hpdi not available, "
                    "using equal-tailed intervals",
                )
                alpha = 1 - credibility
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                lower = float(np.percentile(samples, lower_percentile))
                upper = float(np.percentile(samples, upper_percentile))
                intervals[param_name] = (lower, upper)
            except AttributeError as exc:
                logger.warning(
                    "hpdi computation failed, using equal-tailed intervals",
                    parameter=param_name,
                    error=str(exc),
                )
                alpha = 1 - credibility
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                lower = float(np.percentile(samples, lower_percentile))
                upper = float(np.percentile(samples, upper_percentile))
                intervals[param_name] = (lower, upper)

        logger.debug(
            "Credible intervals computed", parameter_names=list(intervals.keys())
        )
        return intervals

    # NOTE: Precompilation cache is stored per-instance (set in precompile_bayesian).
    # This avoids cross-model contamination from a shared class-level dict.

    def precompile_bayesian(
        self,
        X: np.ndarray | RheoData | None = None,
        y: np.ndarray | None = None,
        test_mode: str | TestMode | None = None,
        num_chains: int = 4,
    ) -> float:
        """Precompile NUTS kernel to eliminate JIT overhead in subsequent calls.

        Triggers JIT compilation of the NumPyro model by running a minimal
        sampling (1 warmup, 1 sample). This caches the compiled kernel so that
        subsequent fit_bayesian() calls are 2-5x faster.

        Parameters
        ----------
        X : ndarray or RheoData, optional
            Sample input data for determining array shapes. If None, uses
            a default 10-point linspace [0.01, 100].
        y : ndarray, optional
            Sample output data. If None, generates dummy data.
        test_mode : str or TestMode, optional
            Test mode to precompile for. If None, defaults to 'relaxation'.

        Returns
        -------
        float
            Compilation time in seconds.

        Example
        -------
        >>> model = Maxwell()
        >>> compile_time = model.precompile_bayesian(test_mode='relaxation')
        >>> print(f"Compiled in {compile_time:.1f}s")
        >>> # Now fit_bayesian() will be faster
        >>> result = model.fit_bayesian(X, y)  # No compilation overhead
        """
        import time

        logger.info("Starting Bayesian precompilation")

        # Generate sample data if not provided
        if X is None:
            X = np.logspace(-2, 2, 10, dtype=np.float64)
        if isinstance(X, RheoData):
            X_array = np.asarray(X.x)
            y_array = np.asarray(X.y)
            if test_mode is None:
                test_mode = detect_test_mode(X)
        else:
            X_array = np.asarray(X, dtype=np.float64)
            y_array = y if y is not None else np.ones_like(X_array, dtype=np.float64)

        # Resolve test mode
        if test_mode is None:
            test_mode = TestMode.RELAXATION
        elif isinstance(test_mode, str):
            test_mode = TestMode(test_mode.lower())

        # R10-BAY-005: Save/restore _test_mode to prevent permanent mutation.
        _had_test_mode = hasattr(self, "_test_mode")
        _saved_test_mode = getattr(self, "_test_mode", None)

        try:
            # Validate requirements
            self._validate_bayesian_requirements()
            self._validate_parameter_bounds()

            # Prepare JAX data
            jax_data = self._prepare_jax_data(X_array, y_array)
            is_complex_data = jax_data["is_complex"]

            # Get parameter info
            param_names = list(self.parameters)
            param_bounds = self._get_parameter_bounds(X_array, y_array, test_mode)

            # Build NumPyro model
            numpyro_model = self._build_numpyro_model(
                param_names=param_names,
                param_bounds=param_bounds,
                test_mode=test_mode,
                is_complex_data=is_complex_data,
                scale_info=jax_data["scale_info"],
            )

            # Build warm-start values
            warm_start_values = self._build_warm_start_values(
                param_names=param_names,
                param_bounds=param_bounds,
                initial_values=None,
                scale_info=jax_data["scale_info"],
                is_complex=is_complex_data,
            )

            # Time the compilation
            start_time = time.perf_counter()

            # Trigger JIT compilation with minimal sampling
            # BAY-007: use caller-specified num_chains to match production default (4)
            try:
                self._run_nuts_sampling(
                    numpyro_model=numpyro_model,
                    X_jax=jax_data["X_jax"],
                    y_jax=jax_data["y_jax"],
                    warm_start_values=warm_start_values,
                    num_warmup=1,
                    num_samples=1,
                    num_chains=num_chains,
                    nuts_kwargs={"progress_bar": False, "target_accept_prob": 0.5},
                    seed=0,
                )
            except Exception as e:
                logger.warning(
                    "Precompilation sampling failed — subsequent fit_bayesian() "
                    "may also fail",
                    error=str(e),
                )
                # Do NOT mark as precompiled if sampling failed
                return time.perf_counter() - start_time

            compile_time = time.perf_counter() - start_time

            # Cache the model key for reference (per-instance dict)
            precompile_key = (str(test_mode), is_complex_data)
            if not hasattr(self, "_precompiled_models"):
                self._precompiled_models = {}
            self._precompiled_models[precompile_key] = True

            logger.info(
                "Bayesian precompilation completed",
                compile_time_seconds=compile_time,
                test_mode=str(test_mode),
                is_complex=is_complex_data,
            )

            return compile_time
        finally:
            # Restore _test_mode regardless of success or failure
            if _had_test_mode:
                self._test_mode = _saved_test_mode
            elif hasattr(self, "_test_mode"):
                del self._test_mode

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

        # Capture protocol-specific arguments from nuts_kwargs
        # These must NOT be passed to NUTS — they go to model_function instead
        protocol_kwargs = {}
        protocol_keys_all = {
            "strain",
            "sigma_0",
            "sigma_applied",
            "gamma_0",
            "gamma_dot",
            "omega",
            "n_cycles",
            "gdot",
            "t_wait",
            # Structure/initial-state kwargs for thixotropic models (DMT, etc.)
            "lam_init",
            "lam_0",
            "sigma_init",
            # LAOS oscillation frequency (distinct from oscillation omega)
            "omega_laos",
            # Also accept non-underscore aliases
            "gamma0",
            "sigma0",
            "stress_target",
            # DMTA boundary kwargs — consumed by BaseModel.fit(), must not leak
            # to NUTS. E*→G* conversion is already applied to self.y_data by fit().
            "deformation_mode",
            "poisson_ratio",
        }

        for key in list(nuts_kwargs):
            if key in protocol_keys_all:
                protocol_kwargs[key] = nuts_kwargs.pop(key)

        # Save _test_mode BEFORE any mutation so it can be restored on error.
        _had_test_mode = hasattr(self, "_test_mode")  # BAY-004: track if attr existed
        _saved_test_mode = getattr(self, "_test_mode", None)
        # R12-B-005: deepcopy is intentional safety — _last_fit_kwargs may contain
        # mutable objects (arrays, sub-dicts) that get mutated during sampling.
        _raw_lfk = getattr(self, "_last_fit_kwargs", None)
        _saved_last_fit_kwargs = (
            copy.deepcopy(_raw_lfk) if _raw_lfk is not None else None
        )

        _fit_bayesian_succeeded = False

        with log_bayesian(
            logger,
            model=model_name,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        ) as log_ctx:
            try:
                # Phase 1: Validation
                self._validate_bayesian_requirements()
                self._validate_parameter_bounds()

                # Phase 2: Resolve test_mode and extract data
                X_array, y_from_rheo, test_mode = self._resolve_test_mode(X, test_mode)
                y_array = y_from_rheo if y_from_rheo is not None else y

                # Apply E*→G* conversion when explicit y is provided with a
                # tensile deformation_mode.
                _dm = protocol_kwargs.get("deformation_mode")
                if _dm is None:
                    _dm = getattr(self, "_deformation_mode", None)
                if _dm is not None and y_array is not None:
                    from rheojax.core.test_modes import DeformationMode

                    if isinstance(_dm, str):
                        _dm = DeformationMode(_dm)
                    if _dm.is_tensile():
                        from rheojax.utils.modulus_conversion import convert_modulus

                        _pr = protocol_kwargs.get("poisson_ratio")
                        if _pr is None:
                            _pr = getattr(self, "_poisson_ratio", 0.5)
                        y_array = convert_modulus(
                            y_array, _dm, DeformationMode.SHEAR, _pr
                        )
                        logger.debug(
                            "fit_bayesian: converted tensile modulus to shear",
                            from_mode=str(_dm),
                            poisson_ratio=_pr,
                        )

                # R12-B-004: On success, _test_mode is permanently updated.
                self._test_mode = test_mode

                # Merge protocol kwargs into _last_fit_kwargs.
                if protocol_kwargs:
                    if (
                        not hasattr(self, "_last_fit_kwargs")
                        or self._last_fit_kwargs is None
                    ):
                        self._last_fit_kwargs: dict = {}
                    self._last_fit_kwargs.update(protocol_kwargs)

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
                    **protocol_kwargs,
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
                result = self._process_mcmc_results(
                    mcmc, param_names, num_samples, num_chains
                )

                # Add diagnostics to log context
                log_ctx["divergences"] = result.diagnostics.get("divergences", 0)
                r_hat_vals = result.diagnostics.get("r_hat", {}).values()
                ess_vals = result.diagnostics.get("ess", {}).values()
                _finite_r_hats = [v for v in r_hat_vals if np.isfinite(v)]
                max_r_hat = max(_finite_r_hats) if _finite_r_hats else float("nan")
                _finite_ess = [v for v in ess_vals if np.isfinite(v)]
                min_ess = min(_finite_ess) if _finite_ess else float("nan")
                log_ctx["r_hat_max"] = max_r_hat
                log_ctx["ess_min"] = min_ess

                logger.info(
                    "Bayesian inference completed",
                    model=model_name,
                    divergences=result.diagnostics.get("divergences", 0),
                    r_hat_max=max_r_hat,
                    ess_min=min_ess,
                )

                _fit_bayesian_succeeded = True
                return result
            finally:
                # Only revert state on failure — on success, _test_mode and
                # _last_fit_kwargs must persist for subsequent predict() calls.
                if not _fit_bayesian_succeeded:
                    if _had_test_mode:
                        self._test_mode = _saved_test_mode
                    elif hasattr(self, "_test_mode"):
                        del self._test_mode
                    self._last_fit_kwargs = _saved_last_fit_kwargs

    @staticmethod
    def _prior_dict_to_dist(prior_spec: dict, dist_module):
        """Convert a prior specification dict to a NumPyro distribution.

        Delegates to numpyro_model_builder.prior_dict_to_dist for backward
        compatibility with subclasses that may call this static method directly.
        """
        return prior_dict_to_dist(prior_spec, dist_module)

    def _build_numpyro_model(
        self,
        param_names: list[str],
        param_bounds: dict[str, tuple[float | None, float | None]],
        test_mode: TestMode,
        is_complex_data: bool,
        scale_info: dict[str, float | None],
        **protocol_kwargs,
    ):
        """Build the NumPyro probabilistic model function.

        Delegates to numpyro_model_builder.build_numpyro_model, passing self
        as model_self for access to model_function, parameters, and caches.
        """
        return build_numpyro_model(
            model_self=self,
            param_names=param_names,
            param_bounds=param_bounds,
            test_mode=test_mode,
            is_complex_data=is_complex_data,
            scale_info=scale_info,
            **protocol_kwargs,
        )

    @staticmethod
    def _compute_per_param_diagnostic(
        posterior_samples: dict[str, np.ndarray],
        num_chains: int,
        num_samples: int,
        diagnostic_fn,
        label: str,
    ) -> tuple[dict[str, float], bool]:
        """Compute a per-parameter diagnostic (R-hat or ESS).

        Delegates to bayesian_diagnostics.compute_per_param_diagnostic.
        """
        from rheojax.core.bayesian_diagnostics import compute_per_param_diagnostic

        return compute_per_param_diagnostic(
            posterior_samples, num_chains, num_samples, diagnostic_fn, label
        )

    def _compute_diagnostics(
        self,
        mcmc: MCMC,
        posterior_samples: dict[str, np.ndarray],
        num_samples: int,
        num_chains: int,
    ) -> DiagnosticsDict:
        """Compute convergence diagnostics from MCMC samples.

        Delegates to bayesian_diagnostics.compute_diagnostics.
        """
        return compute_diagnostics(mcmc, posterior_samples, num_samples, num_chains)


__all__ = [
    "BayesianMixin",
    "BayesianResult",
]
