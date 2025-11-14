"""Unit tests for FractionalModelMixin.

Tests verify:
- Smart initialization delegation to concrete initializers
- Parameter validation
- Module name conversion
- Error handling for missing initializers
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.parameters import ParameterSet
from rheojax.core.test_modes import TestMode
from rheojax.models.fractional_mixin import FractionalModelMixin


class MockFractionalModel(FractionalModelMixin):
    """Mock fractional model for testing mixin functionality."""

    def __init__(self):
        self.parameters = ParameterSet()
        self.parameters.add("Ge", value=1000.0, bounds=(1e-3, 1e9))
        self.parameters.add("Gm", value=9000.0, bounds=(1e-3, 1e9))
        self.parameters.add("alpha", value=0.5, bounds=(0.01, 0.99))
        self.parameters.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))


class MockFZSS(FractionalModelMixin):
    """Mock FZSS model for testing."""

    def __init__(self):
        self.__class__.__name__ = "FractionalZenerSolidSolid"
        self.parameters = ParameterSet()
        self.parameters.add("Ge", value=1000.0, bounds=(1e-3, 1e9))
        self.parameters.add("Gm", value=9000.0, bounds=(1e-3, 1e9))
        self.parameters.add("alpha", value=0.5, bounds=(0.01, 0.99))
        self.parameters.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))


def test_module_name_conversion():
    """Test class name to module name conversion."""
    mixin = FractionalModelMixin()

    # Test special cases
    assert (
        mixin._class_to_module_name("FractionalZenerSolidSolid")
        == "fractional_zener_ss"
    )
    assert (
        mixin._class_to_module_name("FractionalZenerLiquidLiquid")
        == "fractional_zener_ll"
    )
    assert (
        mixin._class_to_module_name("FractionalZenerSolidLiquid")
        == "fractional_zener_sl"
    )
    assert mixin._class_to_module_name("FractionalKVZener") == "fractional_kv_zener"


def test_smart_initialization_skips_non_oscillation_mode():
    """Test that smart initialization is skipped for non-oscillation modes."""
    model = MockFractionalModel()

    omega = np.logspace(-2, 2, 50)
    G_star = np.ones((50, 2)) * 1e5

    # Should return False for relaxation mode
    result = model._apply_smart_initialization(
        omega, G_star, TestMode.RELAXATION, model.parameters
    )
    assert result is False

    # Should return False for creep mode
    result = model._apply_smart_initialization(
        omega, G_star, TestMode.CREEP, model.parameters
    )
    assert result is False


def test_smart_initialization_handles_string_test_mode():
    """Test that smart initialization accepts string test modes."""
    model = MockFractionalModel()

    omega = np.logspace(-2, 2, 50)
    G_star = np.ones((50, 2)) * 1e5

    # Should accept string "oscillation"
    result = model._apply_smart_initialization(
        omega, G_star, "oscillation", model.parameters
    )
    # Will fail to find initializer for MockFractionalModel, but should not crash
    assert result is False  # No initializer for mock model


def test_smart_initialization_with_real_model():
    """Test smart initialization with actual FZSS model."""
    model = MockFZSS()

    # Create realistic oscillation data
    omega = np.logspace(-2, 2, 50)
    Ge, Gm, alpha, tau = 1e5, 9e5, 0.5, 1.0

    # Generate synthetic oscillation data
    from scipy.special import gamma

    G_prime = Ge + Gm * (omega * tau) ** alpha * np.cos(alpha * np.pi / 2) / (
        1
        + 2 * (omega * tau) ** alpha * np.cos(alpha * np.pi / 2)
        + (omega * tau) ** (2 * alpha)
    )
    G_double_prime = (
        Gm
        * (omega * tau) ** alpha
        * np.sin(alpha * np.pi / 2)
        / (
            1
            + 2 * (omega * tau) ** alpha * np.cos(alpha * np.pi / 2)
            + (omega * tau) ** (2 * alpha)
        )
    )
    G_star = np.column_stack([G_prime, G_double_prime])

    # Apply smart initialization
    result = model._apply_smart_initialization(
        omega, G_star, TestMode.OSCILLATION, model.parameters
    )

    # Should succeed (initializer exists and data is valid)
    assert result is True

    # Parameters should be updated from their defaults
    # Note: exact values depend on initialization algorithm


def test_parameter_validation_alpha():
    """Test validation of alpha parameter."""
    mixin = FractionalModelMixin()
    params = ParameterSet()

    # Valid alpha
    params.add("alpha", value=0.5, bounds=(0.01, 0.99))
    mixin._validate_fractional_parameters(params)  # Should not raise

    # Invalid alpha (too small) - expect clamping warning
    params = ParameterSet()
    with pytest.warns(RuntimeWarning, match="Parameter 'alpha' initialized below bounds"):
        params.add("alpha", value=0.0, bounds=(0.01, 0.99))
    with pytest.raises(ValueError, match="alpha must be in"):
        mixin._validate_fractional_parameters(params)

    # Invalid alpha (too large) - expect clamping warning
    params = ParameterSet()
    with pytest.warns(RuntimeWarning, match="Parameter 'alpha' initialized above bounds"):
        params.add("alpha", value=1.0, bounds=(0.01, 0.99))
    with pytest.raises(ValueError, match="alpha must be in"):
        mixin._validate_fractional_parameters(params)


def test_parameter_validation_moduli():
    """Test validation of modulus parameters."""
    mixin = FractionalModelMixin()

    # Valid moduli
    params = ParameterSet()
    params.add("Ge", value=1000.0, bounds=(0, 1e9))
    params.add("Gm", value=9000.0, bounds=(0, 1e9))
    mixin._validate_fractional_parameters(params)  # Should not raise

    # Invalid modulus (negative)
    params = ParameterSet()
    params.add("Ge", value=-1000.0, bounds=(-1e9, 1e9))
    with pytest.raises(ValueError, match="Ge must be positive"):
        mixin._validate_fractional_parameters(params)

    # Invalid modulus (zero)
    params = ParameterSet()
    params.add("Gm", value=0.0, bounds=(0, 1e9))
    with pytest.raises(ValueError, match="Gm must be positive"):
        mixin._validate_fractional_parameters(params)


def test_parameter_validation_time_scales():
    """Test validation of time scale parameters."""
    mixin = FractionalModelMixin()

    # Valid time scale
    params = ParameterSet()
    params.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))
    mixin._validate_fractional_parameters(params)  # Should not raise

    # Invalid time scale (negative)
    params = ParameterSet()
    params.add("tau_alpha", value=-1.0, bounds=(-1e6, 1e6))
    with pytest.raises(ValueError, match="tau_alpha must be positive"):
        mixin._validate_fractional_parameters(params)

    # Invalid time scale (zero)
    params = ParameterSet()
    params.add("tau_beta", value=0.0, bounds=(0, 1e6))
    with pytest.raises(ValueError, match="tau_beta must be positive"):
        mixin._validate_fractional_parameters(params)


def test_initializer_map_coverage():
    """Test that all 11 fractional models have initializer mappings."""
    expected_models = [
        "FractionalZenerSolidSolid",
        "FractionalMaxwellLiquid",
        "FractionalMaxwellGel",
        "FractionalZenerLiquidLiquid",
        "FractionalZenerSolidLiquid",
        "FractionalKelvinVoigt",
        "FractionalKVZener",
        "FractionalMaxwellModel",
        "FractionalPoyntingThomson",
        "FractionalJeffreysModel",
        "FractionalBurgersModel",
    ]

    for model_name in expected_models:
        assert (
            model_name in FractionalModelMixin._INITIALIZER_MAP
        ), f"Missing initializer mapping for {model_name}"


def test_error_handling_missing_initializer():
    """Test graceful handling when initializer cannot be loaded."""

    class UnknownModel(FractionalModelMixin):
        def __init__(self):
            self.parameters = ParameterSet()

    model = UnknownModel()
    omega = np.logspace(-2, 2, 50)
    G_star = np.ones((50, 2)) * 1e5

    # Should handle missing initializer gracefully
    result = model._apply_smart_initialization(
        omega, G_star, TestMode.OSCILLATION, model.parameters
    )
    assert result is False  # Initialization failed but didn't crash
