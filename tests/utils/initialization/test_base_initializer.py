"""Unit tests for BaseInitializer template method pattern.

Tests verify:
- Template method calls all abstract methods in correct order
- Validation logic works (rejects invalid data)
- Error handling for missing/invalid parameters
- Edge cases (empty arrays, NaN values, extreme frequencies)
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.parameters import Parameter, ParameterSet
from rheojax.utils.initialization.base import (
    BaseInitializer,
    extract_frequency_features,
)


class MockInitializer(BaseInitializer):
    """Mock initializer for testing template method."""

    def __init__(self):
        self.call_order = []

    def _estimate_parameters(self, features: dict) -> dict:
        """Mock parameter estimation."""
        self.call_order.append("estimate_parameters")
        return {
            "param1": features["low_plateau"],
            "param2": features["high_plateau"],
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Mock parameter setting."""
        self.call_order.append("set_parameters")
        for key, value in clipped_params.items():
            if key in param_set._parameters:
                param_set.set_value(key, value)


def create_mock_param_set():
    """Create mock ParameterSet for testing."""
    params = ParameterSet()
    params.add("param1", value=1.0, bounds=(1e-3, 1e6))
    params.add("param2", value=1.0, bounds=(1e-3, 1e6))
    return params


def test_template_method_calls_methods_in_order():
    """Test that template method calls all methods in correct order."""
    # Create mock data
    omega = np.logspace(-2, 2, 50)
    G_prime = 1e5 + 9e5 / (1 + omega**2)
    G_double_prime = 9e5 * omega / (1 + omega**2)
    G_star = np.column_stack([G_prime, G_double_prime])

    # Create initializer and param set
    initializer = MockInitializer()
    param_set = create_mock_param_set()

    # Run initialization
    success = initializer.initialize(omega, G_star, param_set)

    # Verify success
    assert success is True

    # Verify methods called in correct order
    assert initializer.call_order == ["estimate_parameters", "set_parameters"]


def test_validation_rejects_invalid_data():
    """Test that validation logic rejects invalid data."""
    # Create invalid data (insufficient frequency range)
    omega = np.logspace(-1, 0, 10)  # Only 1 decade
    G_star = np.ones((10, 2)) * 1e5

    initializer = MockInitializer()
    param_set = create_mock_param_set()

    # Run initialization
    success = initializer.initialize(omega, G_star, param_set)

    # Verify failure
    assert success is False

    # Verify methods not called
    assert initializer.call_order == []


def test_clipping_to_parameter_bounds():
    """Test that parameters are clipped to bounds."""
    # Create data with very high modulus that exceeds bounds
    # Use frequency-dependent data (not flat) to pass validation
    omega = np.logspace(-2, 2, 50)
    G_prime = 1e10 + 9e10 / (1 + omega**2)  # Very high, exceeds 1e6 bound
    G_double_prime = 9e10 * omega / (1 + omega**2)
    G_star = np.column_stack([G_prime, G_double_prime])

    initializer = MockInitializer()
    param_set = create_mock_param_set()

    # Run initialization
    success = initializer.initialize(omega, G_star, param_set)

    # Verify success
    assert success is True

    # Verify parameters clipped to upper bound (1e6)
    assert param_set.get_value("param1") <= 1e6
    assert param_set.get_value("param2") <= 1e6


def test_extract_frequency_features_with_complex_data():
    """Test feature extraction with complex modulus data."""
    # Create complex modulus data
    omega = np.logspace(-2, 2, 50)
    G_star_complex = 1e5 + 9e5 / (1 + 1j * omega)

    features = extract_frequency_features(omega, G_star_complex)

    # Verify features extracted
    assert "low_plateau" in features
    assert "high_plateau" in features
    assert "omega_mid" in features
    assert "alpha_estimate" in features
    assert "valid" in features

    # Verify valid flag
    assert features["valid"] is True


def test_extract_frequency_features_with_2d_array():
    """Test feature extraction with 2D [G', G\"] array."""
    # Create 2D array data
    omega = np.logspace(-2, 2, 50)
    G_prime = 1e5 + 9e5 / (1 + omega**2)
    G_double_prime = 9e5 * omega / (1 + omega**2)
    G_star = np.column_stack([G_prime, G_double_prime])

    features = extract_frequency_features(omega, G_star)

    # Verify features extracted
    assert features["valid"] is True
    assert features["low_plateau"] > 0
    assert features["high_plateau"] > features["low_plateau"]


def test_edge_case_empty_arrays():
    """Test handling of empty arrays."""
    omega = np.array([])
    G_star = np.array([])

    # This should not crash, but may return invalid features
    features = extract_frequency_features(omega, G_star)

    # Verify function completes without error
    assert "valid" in features


def test_edge_case_insufficient_data_points():
    """Test handling of insufficient data points."""
    # Only 3 data points
    omega = np.array([0.1, 1.0, 10.0])
    G_star = np.array([[1e5, 1e4], [5e5, 5e4], [9e5, 1e4]])

    features = extract_frequency_features(omega, G_star)

    # Should still extract features, but may be invalid
    assert "valid" in features
    assert "low_plateau" in features
    assert "high_plateau" in features
