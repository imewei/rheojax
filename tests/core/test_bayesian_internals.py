"""Unit tests for BayesianMixin internal helpers and orchestration wiring.

These target the pure/parametric helper methods (interval math, warm-start
sanitization, test-mode resolution, prior sampling, credible-interval
fallbacks) plus a few NUTS-adjacent code paths (precompilation, result
processing, protocol-kwarg roundtrip) that are cheap to exercise directly.

Most tests here avoid running full NUTS; the handful that do use minimal
warmup/samples to stay fast. Complements test_bayesian.py (end-to-end),
test_bayesian_diagnostics.py (RCA regressions), test_bayesian_edge_cases.py,
and test_bayesian_mode_closure.py (mode capture).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rheojax.core.bayesian import BayesianMixin, BayesianResult
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.test_modes import TestMode

jax, jnp = safe_import_jax()

# Keep NUTS-running tests in a single xdist worker (see test_bayesian.py).
pytestmark = pytest.mark.xdist_group(name="bayesian_nuts")


class LinearModel(BayesianMixin):
    """Minimal BayesianMixin model: y = a * X + b (accepts protocol kwargs)."""

    def __init__(self):
        self.parameters = ParameterSet()
        self.parameters.add("a", value=1.0, bounds=(0.1, 10.0))
        self.parameters.add("b", value=1.0, bounds=(0.1, 10.0))

    def model_function(self, X, params, test_mode=None, **kwargs):
        a, b = params
        return a * X + b


def _linear_data(n=25, seed=42):
    rng = np.random.default_rng(seed)
    X = np.linspace(0.5, 10.0, n)
    y = 2.0 * X + 1.0 + rng.normal(0, 0.1, n)
    return X, y


# ---------------------------------------------------------------------------
# _validate_bayesian_requirements (lines 119-128)
# ---------------------------------------------------------------------------


def test_validate_requirements_missing_parameters():
    """Missing 'parameters' attribute raises a clear AttributeError."""
    bare = BayesianMixin()
    with pytest.raises(AttributeError, match="parameters"):
        bare._validate_bayesian_requirements()


def test_validate_requirements_missing_model_function():
    """Has 'parameters' but no 'model_function' raises AttributeError."""

    class NoModelFunc(BayesianMixin):
        def __init__(self):
            self.parameters = ParameterSet()
            self.parameters.add("a", value=1.0, bounds=(0.1, 1.0))

    with pytest.raises(AttributeError, match="model_function"):
        NoModelFunc()._validate_bayesian_requirements()


# ---------------------------------------------------------------------------
# _validate_parameter_bounds (lines 144, 147)
# ---------------------------------------------------------------------------


def test_validate_bounds_skips_none_bounds():
    """A parameter with bounds=None is skipped (not sampled), no error."""
    model = LinearModel()
    model.parameters.add("c", value=1.0, bounds=None, overwrite=True)
    # Should not raise despite 'c' having no bounds.
    model._validate_parameter_bounds()


# ---------------------------------------------------------------------------
# _resolve_test_mode (lines 197-203, 216-237, 241-249)
# ---------------------------------------------------------------------------


def test_resolve_test_mode_from_rheodata_metadata():
    """RheoData with test_mode=None triggers detect_test_mode."""
    model = LinearModel()
    t = np.logspace(-2, 2, 20)
    G = 1000.0 * np.exp(-t / 1.0)  # monotonic decreasing -> relaxation
    rheo = RheoData(x=t, y=G, metadata={"test_mode": "relaxation"})

    X_array, y_array, mode = model._resolve_test_mode(rheo, None)
    assert mode == TestMode.RELAXATION
    np.testing.assert_allclose(np.asarray(X_array), t)
    np.testing.assert_allclose(np.asarray(y_array), G)


def test_resolve_test_mode_stored_on_self():
    """Tier-1 fallback: self._test_mode used when no explicit mode given."""
    model = LinearModel()
    model._test_mode = TestMode.OSCILLATION
    X = np.linspace(0, 1, 5)
    _, y_array, mode = model._resolve_test_mode(X, None)
    assert mode == TestMode.OSCILLATION
    assert y_array is None  # plain-array branch defers y to caller


def test_resolve_test_mode_from_last_fit_kwargs():
    """Tier-2 fallback: _last_fit_kwargs['test_mode'] used when _test_mode absent."""
    model = LinearModel()
    model._last_fit_kwargs = {"test_mode": TestMode.CREEP}
    X = np.linspace(0, 1, 5)
    _, _, mode = model._resolve_test_mode(X, None)
    assert mode == TestMode.CREEP


def test_resolve_test_mode_default_warns():
    """Tier-3 fallback: no info at all -> RELAXATION with a UserWarning."""
    model = LinearModel()
    X = np.linspace(0, 1, 5)
    with pytest.warns(UserWarning, match="test_mode not specified"):
        _, _, mode = model._resolve_test_mode(X, None)
    assert mode == TestMode.RELAXATION


def test_resolve_test_mode_string_normalized():
    """A plain string test_mode is normalized to the TestMode enum."""
    model = LinearModel()
    X = np.linspace(0, 1, 5)
    _, _, mode = model._resolve_test_mode(X, "relaxation")
    assert mode == TestMode.RELAXATION


def test_resolve_test_mode_uses_validate_hook():
    """A model-provided _validate_test_mode hook handles special strings."""

    class HookModel(LinearModel):
        def _validate_test_mode(self, mode):
            if mode == "laos":
                return TestMode.STARTUP
            raise ValueError("unknown")

    model = HookModel()
    X = np.linspace(0, 1, 5)
    # 'laos' routed through the hook -> STARTUP
    _, _, mode = model._resolve_test_mode(X, "laos")
    assert mode == TestMode.STARTUP
    # Hook raising ValueError falls back to standard TestMode() conversion.
    _, _, mode2 = model._resolve_test_mode(X, "creep")
    assert mode2 == TestMode.CREEP


# ---------------------------------------------------------------------------
# _prepare_jax_data (lines 275, 314-317)
# ---------------------------------------------------------------------------


def test_prepare_jax_data_coerces_python_list():
    """A non-array y (Python list) is coerced without error (real branch)."""
    model = LinearModel()
    X = np.linspace(0, 1, 3)
    out = model._prepare_jax_data(X, [1.0, 2.0, 3.0])
    assert out["is_complex"] is False or bool(out["is_complex"]) is False
    np.testing.assert_allclose(np.asarray(out["y_jax"]), [1.0, 2.0, 3.0])
    assert out["scale_info"]["data_scale"] is not None


def test_prepare_jax_data_empty_complex():
    """Empty complex y takes the zero-scale branch without dividing by zero."""
    model = LinearModel()
    X = np.asarray([], dtype=np.float64)
    y = np.asarray([], dtype=np.complex128)
    out = model._prepare_jax_data(X, y)
    assert bool(out["is_complex"]) is True
    si = out["scale_info"]
    assert si["y_real_scale"] == 0.0
    assert si["y_imag_scale"] == 0.0
    assert si["y_real_mean"] == 0.0
    assert si["y_imag_mean"] == 0.0


# ---------------------------------------------------------------------------
# _validate_xy_shapes
# ---------------------------------------------------------------------------


def test_validate_xy_shapes_accepts_matching_1d():
    """Matching 1-D X/y pass without error."""
    X, y = _linear_data(n=10)
    BayesianMixin._validate_xy_shapes(X, y)  # no raise


def test_validate_xy_shapes_rejects_column_vector_y():
    """A (N, 1) y has the same shape[0] as X but must still be rejected —
    shape[0] equality alone would silently pass through the exact
    broadcasting bug this check exists to catch.
    """
    X = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="1-dimensional"):
        BayesianMixin._validate_xy_shapes(X, y)


def test_validate_xy_shapes_accepts_n2_real_imag_columns():
    """The (N, 2) real/imag-columns convention (supported by
    numpyro_model_builder and RheoData) is not rejected.
    """
    X = np.linspace(0, 1, 10)
    y = np.zeros((10, 2))
    BayesianMixin._validate_xy_shapes(X, y)  # no raise


def test_validate_xy_shapes_rejects_length_mismatch():
    """Non-broadcastable length mismatch still raises."""
    X = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 50)
    with pytest.raises(ValueError, match="same length"):
        BayesianMixin._validate_xy_shapes(X, y)


# ---------------------------------------------------------------------------
# _get_parameter_bounds (lines 352-357, 361)
# ---------------------------------------------------------------------------


def test_get_parameter_bounds_missing_bounds_raises():
    """A parameter with bounds=None raises ValueError."""
    model = LinearModel()
    model.parameters.add("a", value=1.0, bounds=None, overwrite=True)
    X, y = _linear_data()
    with pytest.raises(ValueError, match="must have bounds"):
        model._get_parameter_bounds(X, y, TestMode.RELAXATION)


def test_get_parameter_bounds_override_applied():
    """A callable bayesian_parameter_bounds override rewrites the bounds dict."""
    model = LinearModel()

    def override(bounds, X, y, test_mode):
        bounds["a"] = (0.0, 5.0)
        return bounds

    model.bayesian_parameter_bounds = override
    X, y = _linear_data()
    out = model._get_parameter_bounds(X, y, TestMode.RELAXATION)
    assert out["a"] == (0.0, 5.0)


# ---------------------------------------------------------------------------
# _compute_safe_interval (lines 379-393)
# ---------------------------------------------------------------------------


def test_compute_safe_interval_branches():
    fn = BayesianMixin._compute_safe_interval

    # Both finite, well separated -> padded strictly inside.
    lo, hi = fn(0.0, 10.0)
    assert 0.0 < lo < hi < 10.0

    # Degenerate (equal) bounds -> returns original (safe_lower >= safe_upper).
    assert fn(5.0, 5.0) == (5.0, 5.0)

    # Lower finite only -> guesses an upper.
    lo, hi = fn(2.0, None)
    assert lo > 2.0 and hi > lo

    # Upper finite only -> guesses a lower.
    lo, hi = fn(None, -4.0)
    assert hi < -4.0 + 1e-6 and lo < hi

    # Neither finite -> unit interval.
    assert fn(None, None) == (-1.0, 1.0)


# ---------------------------------------------------------------------------
# _compute_default_midpoint (lines 400-411)
# ---------------------------------------------------------------------------


def test_compute_default_midpoint_branches():
    fn = BayesianMixin._compute_default_midpoint

    # Both positive -> geometric mean.
    np.testing.assert_allclose(fn(1.0, 100.0), 10.0, rtol=1e-9)
    # Spanning zero -> arithmetic mean.
    np.testing.assert_allclose(fn(-2.0, 4.0), 1.0, rtol=1e-9)
    # Lower finite only.
    np.testing.assert_allclose(fn(2.0, None), 3.0, rtol=1e-9)
    # Upper finite only.
    np.testing.assert_allclose(fn(None, -4.0), -6.0, rtol=1e-9)
    # Neither finite.
    assert fn(None, None) == 0.0


# ---------------------------------------------------------------------------
# _build_warm_start_values (lines 432, 459, 485)
# ---------------------------------------------------------------------------


def test_build_warm_start_defaults_to_midpoint_when_value_none():
    """None parameter value falls back to the bounds midpoint."""
    model = LinearModel()
    model.parameters.add("a", value=None, bounds=(0.1, 10.0), overwrite=True)
    model.parameters.add("b", value=None, bounds=(0.1, 10.0), overwrite=True)
    bounds = {"a": (0.1, 10.0), "b": (0.1, 10.0)}
    scale_info = {"data_scale": 1.0}
    ws = model._build_warm_start_values(
        ["a", "b"], bounds, None, scale_info, is_complex=False
    )
    # geometric midpoint of (0.1, 10) == 1.0
    np.testing.assert_allclose(ws["a"], 1.0, rtol=1e-9)
    np.testing.assert_allclose(ws["b"], 1.0, rtol=1e-9)
    assert ws["sigma"] > 0


def test_build_warm_start_fitted_with_none_value_warns(caplog):
    """A model claiming fitted_ with a None value logs a warning but proceeds."""
    model = LinearModel()
    model.fitted_ = True
    model.parameters.add("a", value=None, bounds=(0.1, 10.0), overwrite=True)
    model.parameters.add("b", value=None, bounds=(0.1, 10.0), overwrite=True)
    bounds = {"a": (0.1, 10.0), "b": (0.1, 10.0)}
    ws = model._build_warm_start_values(
        ["a", "b"], bounds, None, {"data_scale": 1.0}, is_complex=False
    )
    assert np.isfinite(ws["a"]) and np.isfinite(ws["b"])


def test_build_warm_start_log_space_sigma():
    """Log-space likelihood uses a dimensionless sigma init of 0.3."""
    model = LinearModel()
    model._bayes_likelihood_space = "log"
    bounds = {"a": (0.1, 10.0), "b": (0.1, 10.0)}
    ws = model._build_warm_start_values(
        ["a", "b"], bounds, {"a": 2.0, "b": 1.0}, {"data_scale": 5000.0}, False
    )
    assert ws["sigma"] == 0.3


def test_build_warm_start_complex_sigmas():
    """Complex data seeds sigma_real / sigma_imag from the scale info."""
    model = LinearModel()
    bounds = {"a": (0.1, 10.0), "b": (0.1, 10.0)}
    scale_info = {"y_real_scale": 2.0, "y_imag_scale": 4.0}
    ws = model._build_warm_start_values(
        ["a", "b"], bounds, {"a": 2.0, "b": 1.0}, scale_info, is_complex=True
    )
    np.testing.assert_allclose(ws["sigma_real"], 0.2, rtol=1e-9)
    np.testing.assert_allclose(ws["sigma_imag"], 0.4, rtol=1e-9)


def test_build_warm_start_out_of_bounds_value_warns_and_clamps():
    """An out-of-bounds initial_values entry is clamped (as before) but must
    now also surface a user-visible RuntimeWarning — previously this was
    logged at DEBUG only, silently defeating the warm-start contract.
    """
    model = LinearModel()
    bounds = {"a": (0.1, 10.0), "b": (0.1, 10.0)}
    with pytest.warns(RuntimeWarning, match="outside parameter bounds"):
        ws = model._build_warm_start_values(
            ["a", "b"], bounds, {"a": 100.0, "b": 1.0}, {"data_scale": 1.0}, False
        )
    assert ws["a"] <= 10.0


def test_build_warm_start_within_bounds_no_warning():
    """A value within bounds is not clamped and emits no warning."""
    model = LinearModel()
    bounds = {"a": (0.1, 10.0), "b": (0.1, 10.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ws = model._build_warm_start_values(
            ["a", "b"], bounds, {"a": 2.0, "b": 1.0}, {"data_scale": 1.0}, False
        )
    assert ws["a"] == 2.0


# ---------------------------------------------------------------------------
# fit_bayesian initial_values key validation
# ---------------------------------------------------------------------------


def test_fit_bayesian_rejects_unrecognized_initial_values_key():
    """A typo'd/stale key in initial_values must raise, not be silently
    dropped (which would fall back to a midpoint init while
    has_explicit_warm_start still reports a genuine warm start).
    """
    model = LinearModel()
    X, y = _linear_data()
    with pytest.raises(ValueError, match="unrecognized parameter"):
        model.fit_bayesian(
            X, y, test_mode="relaxation", initial_values={"aa": 2.0}
        )


# ---------------------------------------------------------------------------
# _run_nuts_sampling has_explicit_warm_start gating (lines 563-565)
# ---------------------------------------------------------------------------


def _prep_run_nuts_args(model, initial_values):
    X, y = _linear_data(n=10)
    jax_data = model._prepare_jax_data(X, y)
    param_names = list(model.parameters)
    param_bounds = model._get_parameter_bounds(X, y, TestMode.RELAXATION)
    numpyro_model = model._build_numpyro_model(
        param_names=param_names,
        param_bounds=param_bounds,
        test_mode=TestMode.RELAXATION,
        is_complex_data=False,
        scale_info=jax_data["scale_info"],
    )
    warm_start_values = model._build_warm_start_values(
        param_names, param_bounds, initial_values, jax_data["scale_info"], False
    )
    return jax_data, numpyro_model, warm_start_values


@pytest.mark.slow
def test_run_nuts_sampling_explicit_initial_values_gets_permissive_defaults():
    """A genuine caller-supplied initial_values (no prior .fit()) must select
    the permissive warm-start NUTS defaults, not the conservative cold-start
    ones. Regression for the bug where has_warm_start reduced to `fitted_`
    alone because _build_warm_start_values always returns a non-empty dict.
    """
    model = LinearModel()
    assert not hasattr(model, "fitted_")
    jax_data, numpyro_model, warm_start_values = _prep_run_nuts_args(
        model, {"a": 2.0, "b": 1.0}
    )
    nuts_kwargs = {}
    model._run_nuts_sampling(
        numpyro_model=numpyro_model,
        X_jax=jax_data["X_jax"],
        y_jax=jax_data["y_jax"],
        warm_start_values=warm_start_values,
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        nuts_kwargs=nuts_kwargs,
        seed=0,
        has_explicit_warm_start=True,
    )
    assert nuts_kwargs["target_accept_prob"] == 0.90
    assert nuts_kwargs["max_tree_depth"] == 8


@pytest.mark.slow
def test_run_nuts_sampling_no_explicit_warm_start_gets_conservative_defaults():
    """Without has_explicit_warm_start, cold-start defaults apply even though
    _build_warm_start_values still returns a fully-populated (midpoint) dict.
    """
    model = LinearModel()
    assert not hasattr(model, "fitted_")
    jax_data, numpyro_model, warm_start_values = _prep_run_nuts_args(model, None)
    assert bool(warm_start_values)  # always non-empty, per _build_warm_start_values
    nuts_kwargs = {}
    model._run_nuts_sampling(
        numpyro_model=numpyro_model,
        X_jax=jax_data["X_jax"],
        y_jax=jax_data["y_jax"],
        warm_start_values=warm_start_values,
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        nuts_kwargs=nuts_kwargs,
        seed=0,
        has_explicit_warm_start=False,
    )
    assert nuts_kwargs["target_accept_prob"] == 0.99
    assert "max_tree_depth" not in nuts_kwargs


@pytest.mark.slow
def test_run_nuts_sampling_construction_failure_preserves_uniform_fallback_flag(
    monkeypatch,
):
    """If init_to_value(values=...) raises during construction (a distinct
    failure surface from the runtime 'Cannot find valid initial parameters'
    RuntimeError handled separately below), _nuts_init_strategy is correctly
    downgraded to 'uniform_fallback'. It must stay there even though the
    subsequent run_mcmc succeeds — not get unconditionally clobbered back to
    'warm_start' on the happy-path return.
    """
    import rheojax.core.bayesian as bayesian_mod

    model = LinearModel()
    jax_data, numpyro_model, warm_start_values = _prep_run_nuts_args(
        model, {"a": 2.0, "b": 1.0}
    )

    real_import_numpyro = bayesian_mod._import_numpyro

    def _broken_init_to_value(*args, **kwargs):
        raise ValueError("simulated init_to_value construction failure")

    def _patched_import_numpyro():
        symbols = list(real_import_numpyro())
        symbols[-1] = _broken_init_to_value  # init_to_value is the last symbol
        return tuple(symbols)

    monkeypatch.setattr(bayesian_mod, "_import_numpyro", _patched_import_numpyro)

    with pytest.warns(RuntimeWarning, match="Warm-start initialization failed"):
        model._run_nuts_sampling(
            numpyro_model=numpyro_model,
            X_jax=jax_data["X_jax"],
            y_jax=jax_data["y_jax"],
            warm_start_values=warm_start_values,
            num_warmup=1,
            num_samples=1,
            num_chains=1,
            nuts_kwargs={},
            seed=0,
            has_explicit_warm_start=True,
        )

    assert model._nuts_init_strategy == "uniform_fallback"


# ---------------------------------------------------------------------------
# sample_prior (lines 785-786, 797-800, 811-814)
# ---------------------------------------------------------------------------


def test_sample_prior_missing_parameters_raises():
    with pytest.raises(AttributeError, match="parameters"):
        BayesianMixin().sample_prior(num_samples=10)


def test_sample_prior_missing_bounds_raises():
    model = LinearModel()
    model.parameters.add("a", value=1.0, bounds=None, overwrite=True)
    with pytest.raises(ValueError, match="must have bounds"):
        model.sample_prior(num_samples=10)


def test_sample_prior_unsupported_prior_dict_falls_back_to_uniform():
    """A prior dict type not recognized by prior_dict_to_dist (e.g. 'beta',
    which build_numpyro_model's own NUTS prior selection does not support
    either — it is not one of prior_dict_to_dist's recognized types) falls
    back to Uniform(lower, upper), matching NUTS exactly instead of
    sample_prior's old hand-rolled 'beta' special case that NUTS never
    actually implemented.
    """
    model = LinearModel()
    param = model.parameters.get("a")
    param.prior = {"type": "beta", "a": 5.0, "b": 1.0}
    samples = model.sample_prior(num_samples=5000, seed=0)["a"]
    lo, hi = param.bounds
    assert np.all(samples >= lo) and np.all(samples <= hi)
    # Falls back to Uniform — mean near midpoint, not skewed toward upper bound.
    assert samples.mean() == pytest.approx(0.5 * (lo + hi), abs=0.5)


def test_sample_prior_normal_prior_dict_matches_nuts():
    """A GUI-supported Parameter.prior dict (normal) is honored via the same
    prior_dict_to_dist conversion NUTS uses, not the old Uniform-only fallback.
    """
    model = LinearModel()
    param = model.parameters.get("a")
    param.prior = {"type": "normal", "loc": 5.0, "scale": 0.01}
    samples = model.sample_prior(num_samples=2000, seed=0)["a"]
    # Tight Normal(5, 0.01) clusters near 5, well inside bounds (0.1, 10.0).
    assert samples.mean() == pytest.approx(5.0, abs=0.05)
    assert samples.std() < 0.1


def test_sample_prior_alpha_suffix_uses_beta_default():
    """Alpha-suffixed bounded parameters get the same Beta(2,2) prior NUTS
    uses by default (numpyro_model_builder.py), not a Uniform.
    """
    model = LinearModel()
    model.parameters.add("alpha", value=0.5, bounds=(0.0, 1.0), overwrite=True)
    samples = model.sample_prior(num_samples=5000, seed=0)["alpha"]
    assert np.all(samples >= 0.0) and np.all(samples <= 1.0)
    # Beta(2,2) std = sqrt(1/20) ~ 0.2236, notably below Uniform(0,1)'s
    # std = 1/sqrt(12) ~ 0.2887 — distinguishes the two distributions.
    assert samples.std() < 0.26


def test_sample_prior_bayesian_prior_factory_override():
    """bayesian_prior_factory, when set, overrides both the alpha-suffix
    Beta default and any Parameter.prior dict, matching NUTS's priority order.
    """
    import numpyro.distributions as dist

    model = LinearModel()
    model.bayesian_prior_factory = lambda name, lower, upper: (
        dist.Normal(loc=3.0, scale=0.001) if name == "a" else None
    )
    samples = model.sample_prior(num_samples=1000, seed=0)["a"]
    assert samples.mean() == pytest.approx(3.0, abs=0.01)


def test_sample_prior_near_degenerate_tolerance_matches_nuts_builder():
    """sample_prior's degenerate-bounds tolerance must exactly match the NUTS
    model builder's (numpyro_model_builder.py) and _validate_parameter_bounds's:
    abs(upper-lower) < 1e-10 * max(abs(lower), 1.0). Bounds (100.0,
    100.00000005) have width 5e-8, which is real (non-degenerate) under that
    shared 1e-10 formula (threshold 1e-8) but was wrongly flagged as
    degenerate by sample_prior's previous 10x-looser, differently-shaped
    tolerance (1e-9 * max(|lower|, |upper|, 1) ~= 1e-7) -- silently returning
    a constant instead of the Uniform NUTS actually samples.
    """
    model = LinearModel()
    lower, upper = 100.0, 100.00000005
    model.parameters.add("a", value=lower, bounds=(lower, upper), overwrite=True)
    samples = model.sample_prior(num_samples=2000, seed=0)["a"]
    assert samples.std() > 0.0
    assert np.all(samples >= lower) and np.all(samples <= upper)


# ---------------------------------------------------------------------------
# get_credible_intervals fallbacks (lines 874-897)
# ---------------------------------------------------------------------------


def test_credible_intervals_attribute_error_fallback(monkeypatch):
    """If hpdi raises AttributeError, fall back to equal-tailed intervals."""
    import numpyro.diagnostics as npd

    def boom(*args, **kwargs):
        raise AttributeError("simulated hpdi failure")

    monkeypatch.setattr(npd, "hpdi", boom)
    model = LinearModel()
    rng = np.random.default_rng(0)
    samples = {"a": rng.normal(5.0, 1.0, 5000)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = model.get_credible_intervals(samples, credibility=0.9)
    lo, hi = ci["a"]
    # Equal-tailed 90% of N(5,1) ~ [3.36, 6.64]
    np.testing.assert_allclose([lo, hi], [3.355, 6.645], atol=0.15)


def test_credible_intervals_import_error_fallback(monkeypatch):
    """If hpdi import fails, fall back to percentile intervals."""
    import numpyro.diagnostics as npd

    monkeypatch.delattr(npd, "hpdi", raising=False)
    model = LinearModel()
    rng = np.random.default_rng(1)
    samples = {"a": rng.normal(0.0, 2.0, 5000)}
    ci = model.get_credible_intervals(samples, credibility=0.95)
    lo, hi = ci["a"]
    assert lo < 0.0 < hi
    np.testing.assert_allclose(hi - lo, 2 * 1.96 * 2.0, rtol=0.1)


# ---------------------------------------------------------------------------
# Static delegators (lines 1310, 1348-1350)
# ---------------------------------------------------------------------------


def test_prior_dict_to_dist_delegates():
    import numpyro.distributions as dist

    d = BayesianMixin._prior_dict_to_dist(
        {"type": "normal", "loc": 1.0, "scale": 2.0}, dist
    )
    assert d is not None
    assert hasattr(d, "log_prob")
    # Unrecognized spec returns None.
    assert BayesianMixin._prior_dict_to_dist({"type": "nope"}, dist) is None


def test_compute_per_param_diagnostic_delegates():
    from numpyro.diagnostics import effective_sample_size

    rng = np.random.default_rng(3)
    samples = {"a": rng.normal(0, 1, 400), "b": rng.normal(2, 1, 400)}
    result, ok = BayesianMixin._compute_per_param_diagnostic(
        samples, num_chains=2, num_samples=200,
        diagnostic_fn=effective_sample_size, label="ESS",
    )
    assert set(result) == {"a", "b"}
    assert all(np.isfinite(v) for v in result.values())
    assert isinstance(ok, bool)


# ---------------------------------------------------------------------------
# _process_mcmc_results with fake MCMC (lines 674-745)
# ---------------------------------------------------------------------------


class _FakeMCMC:
    """Stand-in for a NumPyro MCMC object exposing get_samples/get_extra_fields."""

    def __init__(self, flat, grouped=None, group_raises=False):
        self._flat = flat
        self._grouped = grouped
        self._group_raises = group_raises

    def get_samples(self, group_by_chain=False):
        if group_by_chain:
            if self._group_raises or self._grouped is None:
                raise RuntimeError("grouping unavailable")
            return self._grouped
        return self._flat

    def get_extra_fields(self, group_by_chain=False):
        n = next(iter(self._flat.values())).shape[0]
        return {"diverging": np.zeros(n, dtype=bool)}


def test_process_mcmc_results_grouped_path():
    """Grouped samples (with a sigma noise param) build summary + diagnostics."""
    model = LinearModel()
    rng = np.random.default_rng(7)
    grouped = {
        "a": rng.normal(2.0, 0.1, (2, 100)),
        "b": rng.normal(1.0, 0.1, (2, 100)),
        "sigma": np.abs(rng.normal(0.1, 0.01, (2, 100))),
    }
    flat = {k: v.reshape(-1) for k, v in grouped.items()}
    mcmc = _FakeMCMC(flat, grouped=grouped)

    result = model._process_mcmc_results(mcmc, ["a", "b"], num_samples=100, num_chains=2)
    assert isinstance(result, BayesianResult)
    assert set(result.posterior_samples) == {"a", "b", "sigma"}
    # Summary has the consolidated quantile fields.
    for k in ("mean", "std", "median", "q05", "q25", "q75", "q95"):
        assert k in result.summary["a"]
    np.testing.assert_allclose(result.summary["a"]["mean"], 2.0, atol=0.05)
    assert "r_hat" in result.diagnostics
    assert result.diagnostics["divergences"] == 0


def test_process_mcmc_results_group_by_chain_fallback():
    """When group_by_chain fails, fall back to ungrouped samples cleanly."""
    model = LinearModel()
    # Mark a uniform-fallback init so the warm_start_failed flag path is hit.
    model._nuts_init_strategy = "uniform_fallback"
    rng = np.random.default_rng(8)
    flat = {"a": rng.normal(2.0, 0.1, 200), "b": rng.normal(1.0, 0.1, 200)}
    mcmc = _FakeMCMC(flat, grouped=None, group_raises=True)

    result = model._process_mcmc_results(mcmc, ["a", "b"], num_samples=200, num_chains=1)
    assert set(result.posterior_samples) == {"a", "b"}
    assert result.diagnostics["init_strategy"] == "uniform_fallback"
    assert result.diagnostics["warm_start_failed"] is True


def test_process_mcmc_results_surfaces_nonfinite_draws_diagnostic():
    """num_nonfinite (the numpyro.deterministic site numpyro_model_builder.py
    records every draw to track NaN-guard activity) must reach
    diagnostics['nonfinite_draws'] instead of being silently dropped by the
    param_names/sigma-prefix filtering used to build posterior_samples.
    """
    model = LinearModel()
    rng = np.random.default_rng(11)
    grouped = {
        "a": rng.normal(2.0, 0.1, (2, 50)),
        "b": rng.normal(1.0, 0.1, (2, 50)),
        "num_nonfinite": np.zeros((2, 50)),
    }
    grouped["num_nonfinite"][0, :3] = 1.0  # 3 draws hit the NaN guard
    flat = {k: v.reshape(-1) for k, v in grouped.items()}
    mcmc = _FakeMCMC(flat, grouped=grouped)

    result = model._process_mcmc_results(mcmc, ["a", "b"], num_samples=50, num_chains=2)
    assert result.diagnostics["nonfinite_draws"] == 3
    assert "num_nonfinite" not in result.posterior_samples


def test_process_mcmc_results_nonfinite_draws_defaults_zero_when_absent():
    """Models that don't define the NaN-guard deterministic must not KeyError;
    diagnostics['nonfinite_draws'] defaults to 0.
    """
    model = LinearModel()
    rng = np.random.default_rng(12)
    flat = {"a": rng.normal(2.0, 0.1, 100), "b": rng.normal(1.0, 0.1, 100)}
    mcmc = _FakeMCMC(flat, grouped=None, group_raises=True)
    result = model._process_mcmc_results(mcmc, ["a", "b"], num_samples=100, num_chains=1)
    assert result.diagnostics["nonfinite_draws"] == 0


def test_compute_diagnostics_delegator_handles_missing_divergences():
    """_compute_diagnostics tolerates an mcmc lacking extra fields (-1 divergences)."""
    model = LinearModel()
    rng = np.random.default_rng(9)
    samples = {"a": rng.normal(0, 1, 200)}
    diag = model._compute_diagnostics(
        mcmc=None, posterior_samples=samples, num_samples=200, num_chains=1
    )
    assert "r_hat" in diag and "ess" in diag
    assert diag["divergences"] == -1


# ---------------------------------------------------------------------------
# fit_bayesian fast-fail validation (lines 1132, 1142, 1175, 1212-1221)
# ---------------------------------------------------------------------------


def test_fit_bayesian_invalid_likelihood_space():
    """likelihood_space other than linear/log raises (after protocol pop)."""
    model = LinearModel()
    X, y = _linear_data()
    with pytest.raises(ValueError, match="likelihood_space must be"):
        # strain=... exercises the protocol-kwarg pop before the validation.
        model.fit_bayesian(X, y, test_mode="relaxation", likelihood_space="bad", strain=1.0)


def test_fit_bayesian_missing_y_raises():
    """Plain-array X with y=None raises before any sampling."""
    model = LinearModel()
    X, _ = _linear_data()
    with pytest.raises(ValueError, match="requires `y`"):
        model.fit_bayesian(X, None, test_mode="relaxation")


def test_fit_bayesian_log_space_rejects_complex():
    """likelihood_space='log' is invalid for complex data."""
    model = LinearModel()
    omega = np.logspace(-1, 1, 10)
    y = (1.0 + 1j) * omega
    with pytest.raises(ValueError, match="not supported for complex"):
        model.fit_bayesian(omega, y, test_mode="oscillation", likelihood_space="log")


def test_fit_bayesian_log_space_rejects_nonpositive():
    """likelihood_space='log' requires strictly positive y."""
    model = LinearModel()
    X = np.linspace(0.5, 5.0, 10)
    y = X - 2.0  # contains negatives
    with pytest.raises(ValueError, match="strictly positive"):
        model.fit_bayesian(X, y, test_mode="relaxation", likelihood_space="log")


@pytest.mark.slow
def test_fit_bayesian_warns_on_low_ess_despite_clean_r_hat_and_no_divergences(
    monkeypatch,
):
    """A run with clean R-hat (<1.01), zero divergences, and
    diagnostics_valid=True but pathologically low ESS (a well-known MCMC
    pathology: R-hat converges on short/autocorrelated chains while ESS stays
    low) must still trigger the questionable-convergence RuntimeWarning. ESS
    was previously computed and logged but never included in the warning
    gate, so this case returned silently.
    """
    model = LinearModel()
    X, y = _linear_data(n=10)

    fake_result = BayesianResult(
        posterior_samples={"a": np.array([1.0]), "b": np.array([1.0])},
        summary={},
        diagnostics={
            "r_hat": {"a": 1.001, "b": 1.001},
            "ess": {"a": 3.0, "b": 3.0},
            "divergences": 0,
            "diagnostics_valid": True,
        },
        num_samples=1,
        num_chains=1,
        mcmc=None,
        model_comparison={},
    )
    monkeypatch.setattr(model, "_process_mcmc_results", lambda *a, **k: fake_result)

    with pytest.warns(RuntimeWarning, match="questionable convergence"):
        model.fit_bayesian(
            X, y, test_mode="relaxation", num_warmup=1, num_samples=1, num_chains=1
        )


def test_precompile_bayesian_rejects_nonfinite_input():
    """precompile_bayesian must reject NaN/Inf input the same way
    fit_bayesian does: its docstring explicitly invites passing the caller's
    real X/y to warm the JIT cache for the actual dataset, so bad input must
    fail loudly here too instead of only being caught by the generic
    except-Exception-and-warn wrapper around sampling.
    """
    model = LinearModel()
    X, y = _linear_data(n=10)
    y = y.copy()
    y[0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.precompile_bayesian(X, y, test_mode="relaxation", num_chains=1)


# ---------------------------------------------------------------------------
# NUTS-adjacent wiring: precompile + protocol/override/forward-mode roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_precompile_bayesian_success_and_restore():
    """precompile_bayesian compiles, caches, and restores absent _test_mode."""
    model = LinearModel()
    assert not hasattr(model, "_test_mode")
    X, y = _linear_data(n=10)
    t = model.precompile_bayesian(X, y, test_mode="relaxation", num_chains=1)
    assert isinstance(t, float) and t >= 0.0
    # Cache populated for the (mode, is_complex) key.
    assert getattr(model, "_precompiled_models", {}).get(("TestMode.relaxation", False)) \
        or any(k[0].endswith("relaxation") for k in model._precompiled_models)
    # _test_mode was absent before and must be restored to absent afterwards.
    assert not hasattr(model, "_test_mode")


@pytest.mark.slow
def test_precompile_bayesian_with_rheodata_default_shape():
    """precompile_bayesian accepts RheoData and default (None) inputs."""
    model = LinearModel()
    # Default X path (None -> logspace) with dummy y.
    t = model.precompile_bayesian(num_chains=1)
    assert isinstance(t, float)


@pytest.mark.slow
def test_precompile_bayesian_sampling_failure_returns_time_no_cache():
    """If model_function errors during sampling, no cache entry is recorded."""

    class BoomModel(LinearModel):
        def model_function(self, X, params, test_mode=None, **kwargs):
            raise ValueError("intentional failure during trace")

    model = BoomModel()
    X, y = _linear_data(n=10)
    t = model.precompile_bayesian(X, y, test_mode="relaxation", num_chains=1)
    assert isinstance(t, float)
    # Sampling failed -> model must NOT be marked precompiled.
    assert not getattr(model, "_precompiled_models", {})


@pytest.mark.slow
def test_precompile_bayesian_forwards_protocol_kwargs(monkeypatch):
    """precompile_bayesian must forward protocol_kwargs to _build_numpyro_model,
    mirroring fit_bayesian's Phase 5 call. Without this, the closure cache key
    computed during precompilation (built with an empty protocol_kwargs dict)
    never matches the key a real fit_bayesian(..., strain=...) call computes,
    silently defeating the precompile speedup for any model using protocol
    kwargs (LAOS, thixotropic lam_init/sigma_init, etc.).
    """
    model = LinearModel()
    captured = {}
    orig = model._build_numpyro_model

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return orig(*args, **kwargs)

    monkeypatch.setattr(model, "_build_numpyro_model", spy)
    X, y = _linear_data(n=10)
    model.precompile_bayesian(X, y, test_mode="relaxation", num_chains=1, strain=2.5)
    assert captured.get("strain") == 2.5


@pytest.mark.slow
def test_fit_bayesian_protocol_kwargs_and_overrides_roundtrip():
    """Protocol kwargs merge into _last_fit_kwargs; NUTS overrides + forward-mode wire through."""
    model = LinearModel()
    model.bayesian_nuts_kwargs = lambda: {"target_accept_prob": 0.8}
    model._use_forward_mode_ad = True
    X, y = _linear_data(n=20)

    result = model.fit_bayesian(
        X,
        y,
        test_mode="relaxation",
        num_warmup=20,
        num_samples=40,
        num_chains=2,
        chain_method="vectorized",
        jit_model_args=True,
        seed=0,
        strain=1.0,  # protocol kwarg -> merged into _last_fit_kwargs
    )
    assert isinstance(result, BayesianResult)
    assert model._last_fit_kwargs.get("strain") == 1.0
    assert len(result.posterior_samples["a"]) == 40 * 2
