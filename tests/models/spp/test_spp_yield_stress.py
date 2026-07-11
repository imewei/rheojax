"""Unit tests for SPPYieldStress model."""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.test_modes import TestMode
from rheojax.models import SPPYieldStress


def _synthetic_amplitude_sweep(scale: float = 50.0, exp: float = 0.6):
    gamma_0 = np.logspace(-2, 0, 8)
    sigma = scale * gamma_0**exp
    return gamma_0, sigma


def test_nlsq_converges_and_sets_exponent_static():
    gamma_0, sigma = _synthetic_amplitude_sweep(scale=80.0, exp=0.8)

    model = SPPYieldStress()
    model.fit(gamma_0, sigma, test_mode=TestMode.OSCILLATION, yield_type="static")

    est_scale = model.parameters.get_value("sigma_sy_scale")
    est_exp = model.parameters.get_value("sigma_sy_exp")

    np.testing.assert_allclose(est_exp, 0.8, rtol=0.1)
    np.testing.assert_allclose(est_scale, 80.0, rtol=0.25)


@pytest.mark.slow
def test_bayesian_warm_start_uses_nlsq_init_static():
    gamma_0, sigma = _synthetic_amplitude_sweep(scale=60.0, exp=0.7)

    model = SPPYieldStress()
    model.fit(gamma_0, sigma, test_mode=TestMode.OSCILLATION, yield_type="static")

    # Warm-started Bayesian run with tiny sample count for speed
    result = model.fit_bayesian(
        gamma_0,
        sigma,
        test_mode=TestMode.OSCILLATION,
        num_warmup=50,
        num_samples=80,
    )

    summary = result.summary
    # summary may be dict; normalize to dict access
    mean_scale = (
        summary.get("sigma_sy_scale", {}).get("mean")
        if isinstance(summary, dict)
        else float(summary.loc["sigma_sy_scale", "mean"])
    )
    mean_exp = (
        summary.get("sigma_sy_exp", {}).get("mean")
        if isinstance(summary, dict)
        else float(summary.loc["sigma_sy_exp", "mean"])
    )

    assert mean_exp is not None and 0.4 < mean_exp < 1.0
    assert mean_scale is not None and mean_scale > 10.0


@pytest.mark.slow
def test_fit_bayesian_accepts_yield_type_dynamic():
    """fit_bayesian(..., yield_type='dynamic') must not crash NUTS with a
    stray kwarg (regression: yield_type used to leak into NUTS.__init__)."""
    gamma_0, sigma = _synthetic_amplitude_sweep(scale=40.0, exp=0.5)

    model = SPPYieldStress()
    result = model.fit_bayesian(
        gamma_0,
        sigma,
        test_mode=TestMode.OSCILLATION,
        yield_type="dynamic",
        num_warmup=50,
        num_samples=80,
    )

    assert model._yield_type == "dynamic"
    summary = result.summary
    mean_scale = (
        summary.get("sigma_dy_scale", {}).get("mean")
        if isinstance(summary, dict)
        else float(summary.loc["sigma_dy_scale", "mean"])
    )
    assert mean_scale is not None and mean_scale > 0.0


def test_predict_amplitude_sweep_computes_sigma_max():
    """sigma_max = G_cage * gamma_0 + eta_inf * omega * gamma_0 per the class
    docstring's constitutive equation (regression: omega/G_cage/eta_inf used
    to be silently ignored)."""
    gamma_0, sigma = _synthetic_amplitude_sweep(scale=80.0, exp=0.8)

    model = SPPYieldStress()
    model.fit(gamma_0, sigma, test_mode=TestMode.OSCILLATION, yield_type="static")

    omega = 10.0
    result = model.predict_amplitude_sweep(gamma_0, omega=omega, yield_type="both")

    G_cage = model.parameters.get_value("G_cage")
    eta_inf = model.parameters.get_value("eta_inf")
    expected = G_cage * gamma_0 + eta_inf * omega * gamma_0

    assert "sigma_max" in result
    np.testing.assert_allclose(result["sigma_max"], expected, rtol=1e-8)

    # Changing omega must change sigma_max (was previously a no-op).
    result_other_omega = model.predict_amplitude_sweep(
        gamma_0, omega=omega * 5, yield_type="both"
    )
    assert not np.allclose(result["sigma_max"], result_other_omega["sigma_max"])
