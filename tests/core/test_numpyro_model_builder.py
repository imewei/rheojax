"""Unit tests for rheojax.core.numpyro_model_builder.

Targets prior_dict_to_dist() directly and build_numpyro_model()'s internal
branches (closure cache, model_returns_2col probe, fixed/alpha/default
parameter sampling) using synthetic model_self stand-ins — no full NUTS run
is needed to exercise any of this logic.
"""

from __future__ import annotations

import logging

import jax
import numpyro
import numpyro.distributions as dist
import pytest
from jax import numpy as jnp
from jax import random as jax_random

from rheojax.core.numpyro_model_builder import build_numpyro_model, prior_dict_to_dist
from rheojax.core.parameters import ParameterSet
from rheojax.core.test_modes import TestMode


class _FakeModel:
    """Minimal model_self stand-in: parameters + model_function only."""

    def __init__(self, names_and_bounds: dict[str, tuple[float, float]]):
        self.parameters = ParameterSet()
        for name, bounds in names_and_bounds.items():
            self.parameters.add(name, value=bounds[0], bounds=bounds)
        self.model_function = None  # set per-test


def _linear_model_function(X, params, test_mode, **kwargs):
    return params[0] * X


def _trace_model(numpyro_model, X, y):
    return numpyro.handlers.trace(
        numpyro.handlers.seed(numpyro_model, jax_random.PRNGKey(0))
    ).get_trace(X, y)


# ---------------------------------------------------------------------------
# prior_dict_to_dist
# ---------------------------------------------------------------------------


def test_prior_dict_to_dist_gamma():
    """GUI-configured Gamma priors must not be silently dropped (P0)."""
    d = prior_dict_to_dist(
        {"type": "gamma", "concentration": 2.0, "rate": 1.0}, dist
    )
    assert d is not None
    assert isinstance(d, dist.Gamma)
    assert float(d.concentration) == pytest.approx(2.0)
    assert float(d.rate) == pytest.approx(1.0)


def test_prior_dict_to_dist_beta():
    """GUI-configured Beta priors must not be silently dropped (P0)."""
    d = prior_dict_to_dist(
        {"type": "beta", "concentration0": 3.0, "concentration1": 5.0}, dist
    )
    assert d is not None
    assert isinstance(d, dist.Beta)
    assert float(d.concentration0) == pytest.approx(3.0)
    assert float(d.concentration1) == pytest.approx(5.0)


def test_prior_dict_to_dist_unrecognized_type_logs_warning(caplog):
    """Unrecognized dist_type returns None (existing contract) AND warns —
    the fall-through path raises no exception, so logging-via-except alone
    would not catch it."""
    with caplog.at_level(logging.WARNING, logger="rheojax.core.numpyro_model_builder"):
        result = prior_dict_to_dist({"type": "nope"}, dist)
    assert result is None
    assert any("Unrecognized prior dist_type" in r.message for r in caplog.records)


def test_prior_dict_to_dist_malformed_known_type_logs_warning(caplog):
    """A recognized dist_type with a missing required key still returns
    None but must now log a warning instead of silently swallowing it."""
    with caplog.at_level(logging.WARNING, logger="rheojax.core.numpyro_model_builder"):
        result = prior_dict_to_dist({"type": "normal", "loc": 1.0}, dist)  # no scale
    assert result is None
    assert any("Malformed prior spec" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# build_numpyro_model: parameter-branch selection
# ---------------------------------------------------------------------------


def test_build_numpyro_model_applies_gamma_prior_from_parameter():
    """End-to-end regression for the P0 bug: a Parameter.prior of type
    'gamma' (as set by the GUI PriorsEditor) must actually produce a Gamma
    sample site, not silently fall back to Uniform."""
    model = _FakeModel({"a": (0.1, 10.0)})
    model.parameters.get("a").prior = {
        "type": "gamma",
        "concentration": 2.0,
        "rate": 1.0,
    }
    model.model_function = _linear_model_function

    numpyro_model = build_numpyro_model(
        model, ["a"], {"a": (0.1, 10.0)}, TestMode.RELAXATION, False, {"n_points": 5}
    )
    X = jnp.linspace(0.1, 10.0, 5)
    y = X * 2.0
    trace = _trace_model(numpyro_model, X, y)
    assert isinstance(trace["a"]["fn"], dist.Gamma)


def test_build_numpyro_model_default_uniform_prior():
    """No custom prior, no alpha suffix, distinct bounds -> Uniform."""
    model = _FakeModel({"a": (0.1, 10.0)})
    model.model_function = _linear_model_function

    numpyro_model = build_numpyro_model(
        model, ["a"], {"a": (0.1, 10.0)}, TestMode.RELAXATION, False, {"n_points": 5}
    )
    X = jnp.linspace(0.1, 10.0, 5)
    y = X * 2.0
    trace = _trace_model(numpyro_model, X, y)
    assert trace["a"]["type"] == "sample"
    assert isinstance(trace["a"]["fn"], dist.Uniform)


def test_build_numpyro_model_alpha_param_uses_beta_prior():
    """Parameters whose name ends in 'alpha' with bounds in [0, 1] get the
    weakly-informative Beta(2, 2) heuristic."""
    model = _FakeModel({"alpha": (0.0, 1.0)})
    model.model_function = _linear_model_function

    numpyro_model = build_numpyro_model(
        model,
        ["alpha"],
        {"alpha": (0.0, 1.0)},
        TestMode.RELAXATION,
        False,
        {"n_points": 5},
    )
    X = jnp.linspace(0.1, 10.0, 5)
    y = X * 0.5
    trace = _trace_model(numpyro_model, X, y)
    assert trace["alpha"]["type"] == "sample"
    assert isinstance(trace["alpha"]["fn"], dist.Beta)


def test_build_numpyro_model_fixed_parameter_uses_deterministic():
    """Genuinely equal bounds -> parameter pinned via numpyro.deterministic,
    not sampled."""
    model = _FakeModel({"a": (5.0, 5.0)})
    model.model_function = _linear_model_function

    numpyro_model = build_numpyro_model(
        model, ["a"], {"a": (5.0, 5.0)}, TestMode.RELAXATION, False, {"n_points": 5}
    )
    X = jnp.linspace(0.1, 10.0, 5)
    y = X * 5.0
    trace = _trace_model(numpyro_model, X, y)
    assert trace["a"]["type"] == "deterministic"
    assert float(trace["a"]["value"]) == pytest.approx(5.0)


def test_build_numpyro_model_fixed_tolerance_matches_bounds_validation():
    """Regression: the builder's fixed-parameter tolerance must match
    BayesianMixin._validate_parameter_bounds's ``1e-10 * max(|lo|, 1)``
    formula. Before the fix the builder used a 10x looser
    ``1e-9 * max(|lo|, |hi|, 1)``, so a bound width of 5e-8 at lower=100.0
    was wrongly classified as fixed here while validate() treated it as a
    genuine, sampled parameter."""
    bounds = (100.0, 100.00000005)  # diff = 5e-8; new threshold = 1e-8 -> not fixed
    model = _FakeModel({"a": bounds})
    model.model_function = _linear_model_function

    numpyro_model = build_numpyro_model(
        model, ["a"], {"a": bounds}, TestMode.RELAXATION, False, {"n_points": 5}
    )
    X = jnp.linspace(0.1, 10.0, 5)
    y = X * 100.0
    trace = _trace_model(numpyro_model, X, y)
    assert trace["a"]["type"] == "sample"


# ---------------------------------------------------------------------------
# build_numpyro_model: model_returns_2col probe
# ---------------------------------------------------------------------------


def test_build_numpyro_model_2col_probe_uses_real_n_points_not_dummy():
    """The probe must synthesize its dummy call using the real data length
    (when known) rather than max(n_points, 10) — otherwise a model whose
    internals are sensitive to array length can raise on the mismatched
    dummy size and get misclassified as not-2col (P2 fix)."""
    model = _FakeModel({"a": (0.1, 10.0)})

    def shape_sensitive_2col_fn(X, params, test_mode, **kwargs):
        n = X.shape[0]
        if n % 6 != 0:
            raise ValueError("probe called with wrong length")
        g1 = jnp.sum(X) * params[0] * jnp.ones((n,))
        g2 = g1 * 0.5
        return jnp.column_stack([g1, g2])

    model.model_function = shape_sensitive_2col_fn
    scale_info = {"n_points": 6}
    build_numpyro_model(
        model, ["a"], {"a": (0.1, 10.0)}, TestMode.OSCILLATION, True, scale_info
    )
    assert scale_info["model_returns_2col"] == 1


def test_build_numpyro_model_runtime_check_catches_2col_misclassification():
    """If the probe result was (somehow) wrongly cached as not-2col, the
    runtime cross-check after the real model_function call must raise a
    clear, diagnosable error instead of an opaque broadcasting failure
    downstream (P2 fix)."""
    model = _FakeModel({"a": (0.1, 10.0)})

    def real_2col_fn(X, params, test_mode, **kwargs):
        n = X.shape[0]
        g1 = jnp.sum(X) * params[0] * jnp.ones((n,))
        g2 = g1 * 0.5
        return jnp.column_stack([g1, g2])

    model.model_function = real_2col_fn
    # Pre-seed scale_info so the probe block is skipped entirely, simulating
    # a probe that already ran and (wrongly) cached False.
    scale_info = {"n_points": 5, "model_returns_2col": 0}
    numpyro_model = build_numpyro_model(
        model, ["a"], {"a": (0.1, 10.0)}, TestMode.OSCILLATION, True, scale_info
    )
    X = jnp.linspace(0.1, 10.0, 5)
    y = jnp.zeros(10)
    with pytest.raises(ValueError, match="model_returns_2col probe misclassified"):
        _trace_model(numpyro_model, X, y)


# ---------------------------------------------------------------------------
# build_numpyro_model: closure cache
# ---------------------------------------------------------------------------


def test_build_numpyro_model_cache_hit_returns_same_closure():
    model = _FakeModel({"a": (0.1, 10.0)})
    model.model_function = _linear_model_function

    m1 = build_numpyro_model(
        model, ["a"], {"a": (0.1, 10.0)}, TestMode.RELAXATION, False, {"n_points": 5}
    )
    m2 = build_numpyro_model(
        model, ["a"], {"a": (0.1, 10.0)}, TestMode.RELAXATION, False, {"n_points": 5}
    )
    assert m1 is m2
    assert len(model._closure_cache) == 1


def test_build_numpyro_model_lru_cache_evicts_at_32():
    model = _FakeModel({"a": (0.1, 10.0)})
    model.model_function = _linear_model_function

    for i in range(40):
        bounds = {"a": (0.1, 10.0 + i)}
        build_numpyro_model(
            model, ["a"], bounds, TestMode.RELAXATION, False, {"n_points": 5}
        )
    assert len(model._closure_cache) == 32


# ---------------------------------------------------------------------------
# build_numpyro_model: NaN-safe primal guard poisons the gradient
# ---------------------------------------------------------------------------


def _singular_model_function(X, params, test_mode, **kwargs):
    """A model with a clean algebraic singularity: blows up at theta=0."""
    theta = params[0]
    return (1.0 / theta) * jnp.ones_like(X)


def _potential_at_theta(numpyro_model, X, y, theta_val):
    """-log_joint as a function of the "theta" sample site's value, with the
    site substituted directly (bypassing its prior sample) so jax.grad sees
    exactly the path NUTS differentiates through."""
    seeded = numpyro.handlers.seed(numpyro_model, jax_random.PRNGKey(0))
    substituted = numpyro.handlers.substitute(seeded, data={"theta": theta_val})
    from numpyro.infer.util import log_density

    log_joint, _ = log_density(substituted, (X, y), {}, {})
    return -log_joint


def test_finite_check_guard_poisons_gradient_at_singularity():
    """P0 regression for the shared NaN/Inf guard: jnp.where(is_finite,
    predictions_raw, 0.0) sanitizes the primal likelihood but not the
    gradient. When predictions_raw is non-finite because of a genuine
    algebraic singularity inside model_function (here 1/theta at theta=0),
    0 * inf = NaN survives back through lax.select, so jax.grad of the
    potential energy comes back NaN even though the guard's own factor is
    finite. This must reproduce with the default (opt-out) builder config."""
    model = _FakeModel({"theta": (-1.0, 1.0)})
    model.model_function = _singular_model_function
    numpyro_model = build_numpyro_model(
        model, ["theta"], {"theta": (-1.0, 1.0)}, TestMode.RELAXATION, False,
        {"n_points": 5},
    )
    X = jnp.linspace(0.1, 1.0, 5)
    y = jnp.ones(5)

    grad = jax.grad(lambda t: _potential_at_theta(numpyro_model, X, y, t))(0.0)
    assert jnp.isnan(grad)


def test_nan_safe_grad_reeval_opt_in_fixes_gradient_at_singularity():
    """With model_self._bayesian_nan_safe_grad_reeval = True, the builder
    re-evaluates model_function with stop_gradient applied to the sampled
    params whenever any output was non-finite, severing the NaN gradient
    path (stop_gradient's backward rule is a symbolic zero, not the
    arithmetic 0 * inf that leaks through jnp.where alone)."""
    model = _FakeModel({"theta": (-1.0, 1.0)})
    model.model_function = _singular_model_function
    model._bayesian_nan_safe_grad_reeval = True
    numpyro_model = build_numpyro_model(
        model, ["theta"], {"theta": (-1.0, 1.0)}, TestMode.RELAXATION, False,
        {"n_points": 5},
    )
    X = jnp.linspace(0.1, 1.0, 5)
    y = jnp.ones(5)

    grad = jax.grad(lambda t: _potential_at_theta(numpyro_model, X, y, t))(0.0)
    assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# build_numpyro_model: sigma prior scale (P0-3 fix parity, complex vs real)
# ---------------------------------------------------------------------------


def _complex_model_function(X, params, test_mode, **kwargs):
    """Returns true complex G* directly (not the (N,2) real form), so the
    is_complex_data branch is exercised without the model_returns_2col
    reconstruction path."""
    n = X.shape[0]
    real = params[0] * jnp.ones((n,))
    imag = 0.5 * params[0] * jnp.ones((n,))
    return real + 1j * imag


@pytest.mark.parametrize("is_complex_data", [True, False])
def test_sigma_prior_uses_1x_not_10x_data_scale(is_complex_data):
    """P0 regression: the complex-data (oscillation) sigma_real/sigma_imag
    priors must use the same 1x data-scale multiplier as the real-data
    sigma prior (P0-3 fix) — a stale 10x multiplier drowns the likelihood
    for small N and collapses the posterior to the prior."""
    model = _FakeModel({"a": (0.1, 10.0)})
    X = jnp.linspace(0.1, 10.0, 5)

    if is_complex_data:
        model.model_function = _complex_model_function
        scale_info = {
            "n_points": 5,
            "n_real": 5,
            "y_real_scale": 2.0,
            "y_imag_scale": 3.0,
            "y_real_mean": 1.0,
            "y_imag_mean": 1.0,
        }
        y = jnp.concatenate([jnp.ones(5), jnp.ones(5)])
        numpyro_model = build_numpyro_model(
            model, ["a"], {"a": (0.1, 10.0)}, TestMode.OSCILLATION, True, scale_info
        )
        trace = _trace_model(numpyro_model, X, y)

        expected_real_scale = max(2.0 * 1.0, 1.0 * 0.01, 1e-3)
        expected_imag_scale = max(3.0 * 1.0, 1.0 * 0.01, 1e-3)
        assert isinstance(trace["sigma_real"]["fn"], dist.Exponential)
        assert isinstance(trace["sigma_imag"]["fn"], dist.Exponential)
        assert float(trace["sigma_real"]["fn"].rate) == pytest.approx(
            1.0 / expected_real_scale
        )
        assert float(trace["sigma_imag"]["fn"].rate) == pytest.approx(
            1.0 / expected_imag_scale
        )
    else:
        model.model_function = _linear_model_function
        scale_info = {"n_points": 5, "data_scale": 2.0, "data_mean": 1.0}
        y = X * 2.0
        numpyro_model = build_numpyro_model(
            model, ["a"], {"a": (0.1, 10.0)}, TestMode.RELAXATION, False, scale_info
        )
        trace = _trace_model(numpyro_model, X, y)

        expected_scale = max(2.0 * 1.0, 1.0 * 0.01, 1e-3)
        assert isinstance(trace["sigma"]["fn"], dist.Exponential)
        assert float(trace["sigma"]["fn"].rate) == pytest.approx(1.0 / expected_scale)


# ---------------------------------------------------------------------------
# build_numpyro_model: y=None tolerated (unconditioned/Predictive-style call)
# ---------------------------------------------------------------------------


def test_numpyro_model_tolerates_y_none_for_2col_real_data():
    """P2 regression: with model_returns_2col=True and is_complex_data=False,
    the builder dereferenced y.ndim unconditionally, crashing with
    AttributeError on y=None — the exact call shape the `y=None` default in
    the signature implies should be supported (e.g. via Predictive)."""
    model = _FakeModel({"a": (0.1, 10.0)})

    def _real_2col_fn(X, params, test_mode, **kwargs):
        n = X.shape[0]
        g1 = params[0] * jnp.ones((n,))
        g2 = 0.5 * g1
        return jnp.column_stack([g1, g2])

    model.model_function = _real_2col_fn
    numpyro_model = build_numpyro_model(
        model, ["a"], {"a": (0.1, 10.0)}, TestMode.OSCILLATION, False, {"n_points": 5}
    )
    X = jnp.linspace(0.1, 10.0, 5)
    # Must not raise AttributeError: 'NoneType' object has no attribute 'ndim'
    _trace_model(numpyro_model, X, None)


def test_numpyro_model_tolerates_y_none_for_complex_data():
    """P2 regression: the is_complex_data branch dereferenced y[:n]/y[n:]
    unconditionally, crashing with TypeError on y=None."""
    model = _FakeModel({"a": (0.1, 10.0)})
    model.model_function = _complex_model_function
    numpyro_model = build_numpyro_model(
        model,
        ["a"],
        {"a": (0.1, 10.0)},
        TestMode.OSCILLATION,
        True,
        {"n_points": 5, "n_real": 5},
    )
    X = jnp.linspace(0.1, 10.0, 5)
    # Must not raise TypeError: 'NoneType' object is not subscriptable
    _trace_model(numpyro_model, X, None)
