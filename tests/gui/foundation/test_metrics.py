"""Tests for rheojax.gui.foundation.metrics."""

import math

import numpy as np
import pytest

from rheojax.gui.foundation.metrics import (
    bfmi,
    param_uncertainties,
    reduced_chi_squared,
)


def test_reduced_chi_squared_basic():
    # rss=10, n=12, k=2, sigma2=1 → dof=10, result=1.0
    assert reduced_chi_squared(10.0, 12, 2, sigma2=1.0) == 1.0


def test_reduced_chi_squared_no_sigma2_returns_nan():
    result = reduced_chi_squared(10.0, 12, 2)
    assert math.isnan(result)


def test_reduced_chi_squared_dof_floor():
    # n==k → dof clamped to 1, not division-by-zero
    result = reduced_chi_squared(5.0, 3, 3, sigma2=1.0)
    assert result == pytest.approx(5.0)


def test_reduced_chi_squared_scales_with_sigma2():
    # sigma2=2 halves the result
    r1 = reduced_chi_squared(10.0, 12, 2, sigma2=1.0)
    r2 = reduced_chi_squared(10.0, 12, 2, sigma2=2.0)
    assert r2 == pytest.approx(r1 / 2)


def test_param_uncertainties_diagonal():
    cov = np.diag([4.0, 9.0])
    assert param_uncertainties(cov) == [2.0, 3.0]


def test_param_uncertainties_clamps_negative():
    # Negative diagonal (ill-conditioned cov) → clamped to 0, not NaN
    cov = np.array([[4.0, 0.0], [0.0, -1.0]])
    result = param_uncertainties(cov)
    assert result[0] == pytest.approx(2.0)
    assert result[1] == pytest.approx(0.0)


def test_param_uncertainties_list_input():
    cov = [[1.0, 0.5], [0.5, 4.0]]
    result = param_uncertainties(cov)
    assert result == pytest.approx([1.0, 2.0])


def test_bfmi_constant_energy_is_zero():
    assert bfmi(np.ones(100)) == 0.0


def test_bfmi_known_sequence():
    # E = [0, 1, 0, 1, ...]: diff^2 all 1, var=0.25 → bfmi=1/0.25=4.0
    e = np.tile([0.0, 1.0], 50)
    assert bfmi(e) == pytest.approx(4.0)


def test_bfmi_returns_float():
    result = bfmi(np.random.default_rng(0).standard_normal(200))
    assert isinstance(result, float)
