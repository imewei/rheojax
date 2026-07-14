"""NumPyro probabilistic model builder for Bayesian inference.

This module constructs the NumPyro model function (closure) used by NUTS sampling.
It handles prior distribution selection, model_function probing for output shape,
and the likelihood specification for both complex (oscillation) and real data.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import TestMode
from rheojax.logging import get_logger

logger = get_logger(__name__)

jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    from rheojax.core.parameters import ParameterSet


def _import_numpyro():
    """Lazy-import NumPyro and its submodules."""
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import transforms as dist_transforms

    return numpyro, dist, dist_transforms


def prior_dict_to_dist(prior_spec: dict, dist_module) -> Any | None:
    """Convert a prior specification dict to a NumPyro distribution.

    Supports prior dicts from the GUI PriorsEditor with format:
        {"type": "normal", "loc": 1000, "scale": 500}
        {"type": "uniform", "low": 0, "high": 100}
        {"type": "lognormal", "loc": 0, "scale": 1}
        {"type": "exponential", "rate": 0.01}
        {"type": "halfnormal", "scale": 500}
        {"type": "truncatednormal", "loc": 0, "scale": 1, "low": ..., "high": ...}
        {"type": "gamma", "concentration": 2.0, "rate": 1.0}
        {"type": "beta", "concentration0": 2.0, "concentration1": 2.0}

    Returns None if the spec is unrecognized or malformed (a warning is
    logged in both cases so a silently-wrong fallback prior is diagnosable).
    """
    dist_type = prior_spec.get("type", "").lower()
    try:
        if dist_type == "normal":
            return dist_module.Normal(
                loc=float(prior_spec["loc"]),
                scale=float(prior_spec["scale"]),
            )
        elif dist_type == "uniform":
            return dist_module.Uniform(
                low=float(prior_spec["low"]),
                high=float(prior_spec["high"]),
            )
        elif dist_type == "lognormal":
            return dist_module.LogNormal(
                loc=float(prior_spec.get("loc", 0)),
                scale=float(prior_spec.get("scale", 1)),
            )
        elif dist_type == "exponential":
            return dist_module.Exponential(
                rate=float(prior_spec["rate"]),
            )
        elif dist_type in ("halfnormal", "half_normal"):
            return dist_module.HalfNormal(
                scale=float(prior_spec["scale"]),
            )
        elif dist_type == "truncatednormal":
            return dist_module.TruncatedNormal(
                loc=float(prior_spec["loc"]),
                scale=float(prior_spec["scale"]),
                low=float(prior_spec.get("low", float("-inf"))),
                high=float(prior_spec.get("high", float("inf"))),
            )
        elif dist_type == "gamma":
            return dist_module.Gamma(
                concentration=float(prior_spec["concentration"]),
                rate=float(prior_spec["rate"]),
            )
        elif dist_type == "beta":
            return dist_module.Beta(
                concentration1=float(prior_spec["concentration1"]),
                concentration0=float(prior_spec["concentration0"]),
            )
        else:
            logger.warning(
                "Unrecognized prior dist_type; falling back to default prior",
                dist_type=dist_type,
                prior_spec=prior_spec,
            )
            return None
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "Malformed prior spec; falling back to default prior",
            dist_type=dist_type,
            prior_spec=prior_spec,
            error=str(exc),
        )
        return None


def build_numpyro_model(
    model_self: Any,
    param_names: list[str],
    param_bounds: dict[str, tuple[float | None, float | None]],
    test_mode: TestMode,
    is_complex_data: bool,
    scale_info: dict[str, float | None],
    **protocol_kwargs,
):
    """Build the NumPyro probabilistic model function.

    Returns a callable model function with test_mode captured in closure.
    Uses a per-instance cache keyed by all closure-captured state to avoid
    rebuilding identical closures during repeated fit_bayesian() calls.
    Skips caching when protocol_kwargs contains ndarrays (captured by ref).

    Args:
        model_self: The model instance (provides model_function, parameters,
            bayesian_prior_factory, _closure_cache). If it sets
            `_bayesian_nan_safe_grad_reeval = True`, non-finite
            model_function outputs trigger a second, gradient-safe
            re-evaluation (via stop_gradient) at ~2x the per-step cost —
            an opt-in escape hatch for models whose non-finite outputs come
            from a solver blow-up that can't be localized and guarded inline
            (see the finite_check NaN-gradient note in the model closure).
        param_names: List of parameter names to sample.
        param_bounds: Dict mapping parameter names to (lower, upper) bounds.
        test_mode: TestMode enum for the current protocol.
        is_complex_data: Whether the observation data is complex-valued.
        scale_info: Dict with data scale information for likelihood construction.
        **protocol_kwargs: Protocol-specific kwargs forwarded to model_function.

    Returns:
        A callable NumPyro model function suitable for NUTS sampling.
    """
    # Skip cache when ndarrays in kwargs — closure captures them by reference.
    # Check both NumPy and JAX arrays (JAX arrays are not hashable).
    has_ndarray_kwargs = any(
        isinstance(v, np.ndarray) or hasattr(v, "devices")
        for v in protocol_kwargs.values()
    )
    # _closure_cache is eagerly initialized in BaseModel.__init__; the
    # guard below is a safety net for non-BaseModel users of BayesianMixin.
    if not hasattr(model_self, "_closure_cache"):
        model_self._closure_cache = OrderedDict()

    prior_factory = getattr(model_self, "bayesian_prior_factory", None)
    # Check if any Parameter has a .prior dict set (from GUI PriorsEditor)
    _param_set: ParameterSet | None = getattr(model_self, "parameters", None)
    _has_param_priors = False
    if _param_set is not None and hasattr(_param_set, "values"):
        _has_param_priors = any(
            getattr(p, "prior", None) is not None for p in _param_set.values()
        )

    # R5-JAX-002: Determine if model returns 2-column real array [G', G'']
    # instead of complex G* (used for complex reconstruction in the likelihood).
    # Moved BEFORE cache key computation so scale_info is fully populated
    # when the key is built — prevents stale cache entries from key mismatch.
    if "model_returns_2col" not in scale_info:
        _model_returns_2col = False
        try:
            # Use the real data length when known so the probe's synthetic
            # call matches the actual invocation shape; only fall back to a
            # dummy size when n_points is unknown/non-positive. A mismatched
            # probe length can make an otherwise-fine model raise here for
            # reasons unrelated to its true (N,2)-vs-complex output shape.
            _n_known = scale_info.get("n_points", 0) or 0
            _n_probe = _n_known if _n_known > 0 else 10
            _test_X = jax.ShapeDtypeStruct((_n_probe,), jnp.float64)
            _test_params = jax.ShapeDtypeStruct((len(param_names),), jnp.float64)
            _probe_fn = functools.partial(
                model_self.model_function,
                test_mode=test_mode,
                **protocol_kwargs,
            )
            _out_shape = jax.eval_shape(
                _probe_fn,
                _test_X,
                _test_params,
            )
            _model_returns_2col = (
                hasattr(_out_shape, "ndim")
                and _out_shape.ndim == 2
                and _out_shape.shape[1] == 2
                and _out_shape.dtype != jnp.complex128
                and _out_shape.dtype != jnp.complex64
            )
        except Exception as exc:
            logger.warning(
                "model_function probe failed — defaulting _model_returns_2col=False",
                error=str(exc),
                test_mode=str(test_mode),
            )
        scale_info["model_returns_2col"] = int(_model_returns_2col)

    # Likelihood space read from the model instance, encoded as a numeric flag
    # (1.0 = log, 0.0 = linear) so it participates in the float-valued closure
    # cache key — otherwise a linear and a log fit of the same model would
    # collide in the cache.
    if "likelihood_log" not in scale_info:
        scale_info["likelihood_log"] = (
            1.0
            if getattr(model_self, "_bayes_likelihood_space", "linear") == "log"
            else 0.0
        )

    # Opt-in escape hatch for models whose non-finite predictions come from a
    # solver blow-up (rather than a clean algebraic singularity a model author
    # can localize with a safe_div/safe_log-style guard). See the NaN-gradient
    # note near the finite_check factor below. Off by default: it costs a full
    # second model_function evaluation per gradient step, so it must not be a
    # blanket cost on every NUTS gradient for every model.
    _nan_safe_grad_reeval = bool(
        getattr(model_self, "_bayesian_nan_safe_grad_reeval", False)
    )

    # Skip cache when prior_factory or param-level priors are set
    if not has_ndarray_kwargs and prior_factory is None and not _has_param_priors:
        # R5-JAX-007: scale_info values may be JAX Device arrays when a
        # subclass populates scale_info directly from JAX computations.
        # JAX arrays are not hashable (raises TypeError in tuple/sorted).
        # Coerce every value to a plain Python float so the cache key is
        # always hashable, while preserving None → 0.0 sentinel.
        def _to_hashable(v):
            if v is None:
                return 0.0
            try:
                return float(v)
            except (TypeError, ValueError):
                return repr(v)

        # R10-BAY-004: sorted() can raise TypeError when protocol_kwargs values
        # are non-comparable types (e.g., arrays, custom objects). Guard with
        # try/except and fall through to cache_key = None to skip caching.
        try:
            _pk_key = tuple(sorted(protocol_kwargs.items()))
        except TypeError:
            _pk_key = None

        if _pk_key is None:
            cache_key = None
        else:
            cache_key = (
                str(test_mode),
                is_complex_data,
                tuple(param_names),
                _pk_key,
                tuple(sorted(param_bounds.items())),
                tuple(
                    sorted(
                        (_to_hashable(k), _to_hashable(v))
                        for k, v in scale_info.items()
                    )
                ),
                _nan_safe_grad_reeval,
            )
        if cache_key in model_self._closure_cache:
            model_self._closure_cache.move_to_end(cache_key)
            return model_self._closure_cache[cache_key]
    else:
        cache_key = None

    numpyro, dist, dist_transforms = _import_numpyro()

    # Extract scale values for likelihood.  The `.get(key, 0.0)` call already
    # provides the default; the previous `or 0.0` would swallow legitimate
    # 0.0 values from scale_info, so it has been removed.
    # M-1 guard: if a subclass stores explicit None, fall back to 0.0.
    def _scale_val(key: str) -> float:
        v = scale_info.get(key, 0.0)
        return 0.0 if v is None else v

    y_real_scale = _scale_val("y_real_scale")
    y_imag_scale = _scale_val("y_imag_scale")
    data_scale = _scale_val("data_scale")
    y_real_mean = _scale_val("y_real_mean")
    y_imag_mean = _scale_val("y_imag_mean")
    data_mean = _scale_val("data_mean")

    # Capture priors at closure-build time (not re-read at trace time)
    _captured_priors = {}
    if hasattr(model_self, "parameters"):
        for _pname in param_names:
            _pobj = model_self.parameters.get(_pname)
            if _pobj is not None:
                _captured_priors[_pname] = getattr(_pobj, "prior", None)

    # model_returns_2col was already probed and cached in scale_info
    # before the cache key was computed (see block above).
    _model_returns_2col = bool(scale_info.get("model_returns_2col", 0))
    # Log-space likelihood flag (set from model_self._bayes_likelihood_space).
    _likelihood_log = bool(scale_info.get("likelihood_log", 0.0))

    def numpyro_model(X, y=None):
        """NumPyro model with test_mode captured in closure."""
        # Sample parameters from priors
        params_dict = {}
        for name in param_names:
            lower, upper = param_bounds[name]
            custom_dist = None
            if callable(prior_factory):
                custom_dist = prior_factory(name, lower, upper)

            # F-001 fix: Check Parameter.prior dict (set by GUI PriorsEditor)
            # Use _captured_priors to avoid re-reading self.parameters at trace time
            if custom_dist is None:
                param_prior = _captured_priors.get(name)
                if param_prior is not None and isinstance(param_prior, dict):
                    custom_dist = prior_dict_to_dist(param_prior, dist)

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
                    beta_trans = dist_transforms.AffineTransform(loc=lower, scale=scale)
                    params_dict[name] = numpyro.sample(
                        name,
                        dist.TransformedDistribution(beta_base, beta_trans),
                    )
            elif (
                lower is not None
                and upper is not None
                and abs(float(upper) - float(lower))
                < 1e-10 * max(abs(float(lower)), 1.0)
            ):
                # PARAMS-001: fixed parameter — use deterministic instead of Uniform.
                # Tolerance matches BayesianMixin._validate_parameter_bounds()
                # (bayesian.py) exactly, so both agree on what counts as "fixed"
                # instead of disagreeing in a narrow bound-width gap.
                param_val = numpyro.deterministic(name, lower)
                params_dict[name] = param_val
            else:
                params_dict[name] = numpyro.sample(
                    name, dist.Uniform(low=lower, high=upper)
                )

        # BAY-01: use jnp.stack instead of jnp.array(list-comp) so this
        # compiles to a single XLA concatenation op in the NUTS hot path.
        params_array = jnp.stack([params_dict[name] for name in param_names])

        # Forward protocol kwargs to model_function
        predictions_raw = model_self.model_function(
            X, params_array, test_mode, **protocol_kwargs
        )

        # R5-JAX-002 cross-check: the model_returns_2col probe above can
        # misclassify (e.g. its synthetic dummy call raised for a reason
        # unrelated to the model's true output shape, silently defaulting to
        # False). Catch that here with a cheap static-shape check on the real
        # call so the failure is a clear diagnostic instead of an opaque
        # NumPyro/JAX broadcasting error further down.
        if (
            not _model_returns_2col
            and predictions_raw.ndim == 2
            and predictions_raw.shape[-1] == 2
        ):
            raise ValueError(
                "model_function returned a 2-column (N, 2) array but the "
                "model_returns_2col probe misclassified it as not-2col; "
                "check why jax.eval_shape failed for this model_function "
                "(see the 'model_function probe failed' warning, if logged)."
            )

        # Guard against NaN/Inf from ODE-based models.
        #
        # NOTE: jnp.where(is_finite, predictions_raw, 0.0) below only sanitizes
        # the primal log-density value, not its gradient. If a non-finite
        # value here originates from a locally-singular op inside
        # model_function (e.g. dividing by a relaxation time that hit its
        # prior boundary), the local Jacobian at that internal node is itself
        # NaN/Inf, and 0 * inf = NaN under IEEE754 survives lax.select's
        # gradient rule — so NUTS can still receive a NaN gradient even though
        # this factor's value is finite. Model authors are responsible for
        # gradient-safe internals (guard hazards BEFORE the unsafe op, e.g.
        # `safe_tau = jnp.where(tau > eps, tau, eps)`), unless the model opts
        # into `_bayesian_nan_safe_grad_reeval` below.
        is_finite = jnp.isfinite(predictions_raw)
        not_finite = ~is_finite
        finite_penalty = jnp.where(is_finite, 0.0, -1e18).sum()
        numpyro.factor("finite_check", finite_penalty)
        numpyro.deterministic("num_nonfinite", not_finite.sum().astype(jnp.float64))
        if _nan_safe_grad_reeval:
            # Re-evaluate model_function with stop_gradient applied to the
            # sampled params whenever any output was non-finite. stop_gradient's
            # backward rule discards the incoming cotangent via a symbolic
            # zero (not an arithmetic multiplication), so this severs the
            # NaN/Inf gradient path instead of merely masking the primal
            # value. This costs a second full model_function evaluation, so
            # it only runs for models that explicitly opt in.
            params_safe = jnp.where(
                is_finite.all(), params_array, jax.lax.stop_gradient(params_array)
            )
            predictions_raw = model_self.model_function(
                X, params_safe, test_mode, **protocol_kwargs
            )
        predictions_raw = jnp.where(is_finite, predictions_raw, 0.0)

        # Normalize oscillation predictions: some models return (N,2) real
        # arrays [G', G''] instead of complex G' + 1j*G''
        if _model_returns_2col:
            if is_complex_data:
                predictions_raw = predictions_raw[:, 0] + 1j * predictions_raw[:, 1]
            else:
                predictions_raw = jnp.sqrt(
                    predictions_raw[:, 0] ** 2 + predictions_raw[:, 1] ** 2 + 1e-30
                )
                if y is not None and y.ndim == 2 and y.shape[1] == 2:
                    y = jnp.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2 + 1e-30)

        # Handle complex vs real predictions
        if is_complex_data:
            pred_real = jnp.real(predictions_raw)
            pred_imag = jnp.imag(predictions_raw)
            n = scale_info["n_real"]
            y_real_obs, y_imag_obs = (None, None) if y is None else (y[:n], y[n:])

            # Let prior_factory override noise priors if it provides them
            sigma_real_dist = None
            sigma_imag_dist = None
            if callable(prior_factory):
                sigma_real_dist = prior_factory("sigma_real", 0.0, None)
                sigma_imag_dist = prior_factory("sigma_imag", 0.0, None)

            if sigma_real_dist is None:
                # Scale noise prior to ~1x the data's peak-to-peak range,
                # matching the real-data branch below (P0-3 fix): a 10x
                # multiplier drowns the likelihood for small N, collapsing
                # the posterior to the prior.
                sigma_real_scale = max(y_real_scale * 1.0, y_real_mean * 0.01, 1e-3)
                sigma_real_dist = dist.Exponential(rate=1.0 / sigma_real_scale)
            if sigma_imag_dist is None:
                sigma_imag_scale = max(y_imag_scale * 1.0, y_imag_mean * 0.01, 1e-3)
                sigma_imag_dist = dist.Exponential(rate=1.0 / sigma_imag_scale)

            sigma_real = numpyro.sample("sigma_real", sigma_real_dist)
            sigma_imag = numpyro.sample("sigma_imag", sigma_imag_dist)
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
        elif _likelihood_log:
            # Log-space likelihood: log(y) ~ Normal(log(pred), sigma), i.e.
            # y ~ LogNormal(log(pred), sigma). Appropriate when y spans several
            # decades — residuals are relative, so every decade is weighted
            # equally instead of the largest points dominating the fit.
            sigma_dist = None
            if callable(prior_factory):
                sigma_dist = prior_factory("sigma", 0.0, None)
            if sigma_dist is None:
                # Log-space residual scale is dimensionless and O(0.1–1);
                # Exponential(mean=0.5) is weakly informative.
                sigma_dist = dist.Exponential(rate=1.0 / 0.5)

            sigma = numpyro.sample("sigma", sigma_dist)
            # Floor predictions away from zero so log() is finite (NaN-guarded
            # zeros above would otherwise map to -inf).
            pred_pos = jnp.maximum(predictions_raw, 1e-30)
            numpyro.sample(
                "obs", dist.LogNormal(loc=jnp.log(pred_pos), scale=sigma), obs=y
            )
        else:
            # Let prior_factory override noise prior if it provides one
            sigma_dist = None
            if callable(prior_factory):
                sigma_dist = prior_factory("sigma", 0.0, None)

            if sigma_dist is None:
                # Scale noise prior to ~1× the data's peak-to-peak range.
                # The old 10× multiplier drowned the likelihood for small N,
                # collapsing the posterior to the prior (P0-3 fix).
                sigma_scale = max(data_scale * 1.0, data_mean * 0.01, 1e-3)
                sigma_dist = dist.Exponential(rate=1.0 / sigma_scale)

            sigma = numpyro.sample("sigma", sigma_dist)
            numpyro.sample("obs", dist.Normal(loc=predictions_raw, scale=sigma), obs=y)

    if cache_key is not None:
        model_self._closure_cache[cache_key] = numpyro_model
        # LRU eviction: keep at most 32 cached closures
        while len(model_self._closure_cache) > 32:
            model_self._closure_cache.popitem(last=False)
    return numpyro_model


__all__ = [
    "build_numpyro_model",
    "prior_dict_to_dist",
]
