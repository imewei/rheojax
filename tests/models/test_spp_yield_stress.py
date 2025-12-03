"""Tests for the SPPYieldStress model."""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.models.spp_yield_stress import SPPYieldStress


def _synthetic_yield_curve(scale: float = 5.0, exp: float = 0.2, n: int = 6):
    gamma_0 = np.logspace(-2, 0, n)
    sigma = scale * gamma_0**exp
    return gamma_0, sigma


def test_fit_oscillation_power_law():
    gamma_0, sigma = _synthetic_yield_curve(scale=8.0, exp=0.3, n=8)

    model = SPPYieldStress()
    model.fit(gamma_0, sigma, test_mode="oscillation", yield_type="static")

    assert model.fitted_
    assert 7.0 < model.parameters.get_value("sigma_sy_scale") < 9.5
    assert 0.2 < model.parameters.get_value("sigma_sy_exp") < 0.4


@pytest.mark.slow
def test_predict_matches_fit_trend():
    gamma_0, sigma = _synthetic_yield_curve(scale=4.0, exp=0.25, n=6)

    model = SPPYieldStress()
    model.fit(gamma_0, sigma, test_mode="oscillation", yield_type="static")

    preds = model._predict(gamma_0)
    # Stress should roughly follow the same scaling
    ratio = preds / sigma
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)


def test_rotation_mode_fallback():
    gamma_dot = np.logspace(-1, 2, 10)
    sigma = 2.0 + 0.5 * gamma_dot**0.6

    model = SPPYieldStress()
    model.fit(gamma_dot, sigma, test_mode="rotation")

    assert model.parameters.get_value("sigma_dy_scale") > 0
    assert 0.1 <= model.parameters.get_value("n_power_law") <= 1.5
