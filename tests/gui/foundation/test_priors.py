"""Tests for rheojax.gui.foundation.priors."""

import math

from rheojax.gui.foundation.priors import adapt_prior, map_centered_priors

# ---------------------------------------------------------------------------
# adapt_prior
# ---------------------------------------------------------------------------


def test_adapt_prior_passthrough():
    """Passthrough: lognormal params survive unchanged."""
    out = adapt_prior(
        {"distribution": "lognormal", "params": {"loc": 2.0, "scale": 1.0}}
    )
    assert out["type"] == "lognormal"
    assert out["loc"] == 2.0
    assert out["scale"] == 1.0


def test_adapt_prior_exponential_scale_to_rate():
    """Exponential: PriorsEditor 'scale' must become 'rate = 1/scale'."""
    out = adapt_prior({"distribution": "exponential", "params": {"scale": 2.0}})
    assert out["type"] == "exponential"
    assert abs(out["rate"] - 0.5) < 1e-9
    assert "scale" not in out


def test_adapt_prior_normal():
    """Normal distribution passes loc/scale through without modification."""
    out = adapt_prior({"distribution": "normal", "params": {"loc": 0.0, "scale": 0.5}})
    assert out == {"type": "normal", "loc": 0.0, "scale": 0.5}


def test_adapt_prior_uniform():
    """Uniform distribution passes low/high through."""
    out = adapt_prior({"distribution": "uniform", "params": {"low": 0.0, "high": 10.0}})
    assert out == {"type": "uniform", "low": 0.0, "high": 10.0}


def test_adapt_prior_case_insensitive():
    """Distribution name is normalised to lowercase."""
    out = adapt_prior(
        {"distribution": "LogNormal", "params": {"loc": 1.0, "scale": 0.5}}
    )
    assert out["type"] == "lognormal"


def test_adapt_prior_halfnormal():
    """HalfNormal scale passes through unchanged."""
    out = adapt_prior({"distribution": "halfnormal", "params": {"scale": 1.0}})
    assert out == {"type": "halfnormal", "scale": 1.0}


def test_adapt_prior_exponential_zero_scale_guard():
    """Zero scale for exponential falls back to rate=1.0 instead of ZeroDivisionError."""
    out = adapt_prior({"distribution": "exponential", "params": {"scale": 0.0}})
    assert out["type"] == "exponential"
    assert out["rate"] == 1.0
    assert "scale" not in out


# ---------------------------------------------------------------------------
# map_centered_priors
# ---------------------------------------------------------------------------


def test_map_centered_priors_keys():
    """Output contains one key per MAP param plus 'sigma'."""
    pri = map_centered_priors({"G0": 1000.0, "eta": 50.0})
    assert set(pri) == {"G0", "eta", "sigma"}


def test_map_centered_priors_types():
    """MAP params → LogNormal; sigma → HalfNormal."""
    pri = map_centered_priors({"G0": 1000.0, "eta": 50.0})
    assert pri["G0"]["type"] == "lognormal"
    assert pri["eta"]["type"] == "lognormal"
    assert pri["sigma"]["type"] == "halfnormal"


def test_map_centered_priors_loc_is_log_of_value():
    """LogNormal loc = log(val) so the median equals the MAP estimate."""
    pri = map_centered_priors({"tau": 100.0})
    assert abs(pri["tau"]["loc"] - math.log(100.0)) < 1e-12
    assert pri["tau"]["scale"] == 1.0


def test_map_centered_priors_sigma_structure():
    """sigma prior has only type and scale keys."""
    pri = map_centered_priors({"x": 1.0})
    assert pri["sigma"] == {"type": "halfnormal", "scale": 1.0}


def test_map_centered_priors_zero_value_fallback():
    """Zero MAP value falls back to loc=0.0 without error."""
    pri = map_centered_priors({"x": 0.0})
    assert pri["x"]["loc"] == 0.0


def test_map_centered_priors_negative_value():
    """Negative MAP value uses abs() so loc = log(|val|)."""
    pri = map_centered_priors({"x": -50.0})
    assert abs(pri["x"]["loc"] - math.log(50.0)) < 1e-12


def test_map_centered_priors_empty():
    """Empty MAP estimate → only sigma key."""
    pri = map_centered_priors({})
    assert set(pri) == {"sigma"}
    assert pri["sigma"]["type"] == "halfnormal"
