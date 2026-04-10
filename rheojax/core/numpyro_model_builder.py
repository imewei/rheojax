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

    Returns None if the spec is unrecognized.
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
    except (KeyError, TypeError, ValueError):
        pass
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
            bayesian_prior_factory, _closure_cache).
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
        model_self._closure_cache: OrderedDict = OrderedDict()

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
            _n_probe = max(scale_info.get("n_points", 0) or 0, 10)
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
                < 1e-9 * max(abs(float(lower)), abs(float(upper)), 1.0)
            ):
                # PARAMS-001: fixed parameter — use deterministic instead of Uniform.
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

        # Guard against NaN/Inf from ODE-based models
        is_finite = jnp.isfinite(predictions_raw)
        not_finite = ~is_finite
        finite_penalty = jnp.where(is_finite, 0.0, -1e18).sum()
        numpyro.factor("finite_check", finite_penalty)
        numpyro.deterministic("num_nonfinite", not_finite.sum().astype(jnp.float64))
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
                if y.ndim == 2 and y.shape[1] == 2:
                    y = jnp.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2 + 1e-30)

        # Handle complex vs real predictions
        if is_complex_data:
            pred_real = jnp.real(predictions_raw)
            pred_imag = jnp.imag(predictions_raw)
            n = scale_info["n_real"]
            y_real_obs, y_imag_obs = y[:n], y[n:]

            # Let prior_factory override noise priors if it provides them
            sigma_real_dist = None
            sigma_imag_dist = None
            if callable(prior_factory):
                sigma_real_dist = prior_factory("sigma_real", 0.0, None)
                sigma_imag_dist = prior_factory("sigma_imag", 0.0, None)

            if sigma_real_dist is None:
                sigma_real_scale = max(y_real_scale * 10.0, y_real_mean * 0.01, 1e-3)
                sigma_real_dist = dist.Exponential(rate=1.0 / sigma_real_scale)
            if sigma_imag_dist is None:
                sigma_imag_scale = max(y_imag_scale * 10.0, y_imag_mean * 0.01, 1e-3)
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
        else:
            # Let prior_factory override noise prior if it provides one
            sigma_dist = None
            if callable(prior_factory):
                sigma_dist = prior_factory("sigma", 0.0, None)

            if sigma_dist is None:
                # Floor at 1% of mean magnitude to handle constant-data edge case
                sigma_scale = max(data_scale * 10.0, data_mean * 0.01, 1e-3)
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
