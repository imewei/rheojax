"""Edge case and error handling integration tests.

Tests boundary conditions, invalid inputs, and error recovery.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import patch

from rheo.core.data import RheoData
from rheo.core.parameters import Parameter, ParameterSet
from rheo.core.test_modes import detect_test_mode


class TestParameterBoundaryConditions:
    """Test parameter behavior at boundaries."""

    @pytest.mark.edge_case
    def test_parameter_at_lower_bound(self):
        """Test parameter value at lower bound."""
        param = Parameter("test", value=0.0, bounds=(0.0, 100.0))
        assert param.value == 0.0

    @pytest.mark.edge_case
    def test_parameter_at_upper_bound(self):
        """Test parameter value at upper bound."""
        param = Parameter("test", value=100.0, bounds=(0.0, 100.0))
        assert param.value == 100.0

    @pytest.mark.edge_case
    def test_parameter_zero_bounds_interval(self):
        """Test parameter with zero-width bounds interval."""
        # This should raise or handle gracefully
        with pytest.raises((ValueError, AssertionError)):
            Parameter("test", value=5.0, bounds=(5.0, 5.0))

    @pytest.mark.edge_case
    def test_parameter_very_small_values(self):
        """Test parameters with very small values."""
        param = Parameter("test", value=1e-15, bounds=(0.0, 1.0))
        assert param.value == 1e-15
        assert param.value > 0

    @pytest.mark.edge_case
    def test_parameter_very_large_values(self):
        """Test parameters with very large values."""
        param = Parameter("test", value=1e15, bounds=(0.0, 1e20))
        assert param.value == 1e15

    @pytest.mark.edge_case
    def test_parameter_set_empty_access(self):
        """Test accessing empty parameter set."""
        params = ParameterSet()

        assert len(params) == 0
        assert params.get("nonexistent") is None
        assert params.to_dict() == {}

    @pytest.mark.edge_case
    def test_parameter_set_single_element(self):
        """Test parameter set with single element."""
        params = ParameterSet()
        params.add("single", value=5.0)

        assert len(params) == 1
        assert "single" in params
        assert params.get("single").value == 5.0


class TestDataShapeEdgeCases:
    """Test RheoData with unusual shapes."""

    @pytest.mark.edge_case
    def test_single_point_data(self):
        """Test RheoData with single data point."""
        x = np.array([1.0])
        y = np.array([2.0])

        data = RheoData(x=x, y=y)

        assert len(data.x) == 1
        assert len(data.y) == 1
        assert data.shape == (1,)

    @pytest.mark.edge_case
    def test_two_point_data(self):
        """Test RheoData with two points (minimum for fitting)."""
        x = np.array([1.0, 2.0])
        y = np.array([10.0, 20.0])

        data = RheoData(x=x, y=y)

        assert len(data.x) == 2
        assert len(data.y) == 2

    @pytest.mark.edge_case
    def test_very_large_dataset(self):
        """Test RheoData with very large number of points."""
        x = np.linspace(0, 100, 100000)
        y = np.sin(x)

        data = RheoData(x=x, y=y)

        assert len(data.x) == 100000
        assert len(data.y) == 100000

    @pytest.mark.edge_case
    def test_mismatched_dimensions_error(self):
        """Test error on mismatched x and y dimensions."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0])

        with pytest.raises(ValueError, match="must have the same shape"):
            RheoData(x=x, y=y)

    @pytest.mark.edge_case
    def test_empty_data_error(self):
        """Test error on empty data."""
        with pytest.raises((ValueError, IndexError)):
            RheoData(x=np.array([]), y=np.array([]))

    @pytest.mark.edge_case
    def test_data_with_zeros(self):
        """Test data containing zeros."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 10.0, 20.0, 30.0])

        # Should not fail despite zero values
        data = RheoData(x=x, y=y, validate=False)
        assert data.y[0] == 0.0

    @pytest.mark.edge_case
    def test_constant_data(self):
        """Test data with constant y values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        data = RheoData(x=x, y=y)

        assert np.all(data.y == 5.0)
        assert np.var(data.y) == 0.0


class TestNumericalEdgeCases:
    """Test handling of numerical edge cases."""

    @pytest.mark.edge_case
    def test_very_small_differences(self):
        """Test data with very small differences between points."""
        x = np.array([1.0, 1.0 + 1e-10, 1.0 + 2e-10])
        y = np.array([1.0, 1.0 + 1e-10, 1.0 + 2e-10])

        data = RheoData(x=x, y=y, validate=False)

        assert data is not None
        assert len(data.x) == 3

    @pytest.mark.edge_case
    def test_very_large_value_ranges(self):
        """Test data spanning many orders of magnitude."""
        x = np.logspace(-10, 10, 50)
        y = np.logspace(-10, 10, 50)

        data = RheoData(x=x, y=y)

        assert np.min(data.x) < 1e-9
        assert np.max(data.x) > 1e9

    @pytest.mark.edge_case
    def test_data_near_machine_epsilon(self):
        """Test data near machine epsilon."""
        eps = np.finfo(float).eps
        x = np.array([1.0, 1.0 + eps, 1.0 + 2*eps])
        y = np.array([1.0, 1.0 + eps, 1.0 + 2*eps])

        # Should handle without error
        data = RheoData(x=x, y=y, validate=False)
        assert data is not None

    @pytest.mark.edge_case
    def test_negative_values_in_modulus(self):
        """Test handling of negative values where positive expected."""
        frequency = np.array([0.1, 1.0, 10.0])
        # Negative storage modulus (physically invalid)
        G_prime = np.array([-1e5, -2e4, -5e3])
        G_double_prime = np.array([1e4, 2e4, 3e4])

        G_complex = G_prime + 1j * G_double_prime

        # Should create but warn
        with pytest.warns(UserWarning):
            data = RheoData(
                x=frequency, y=G_complex,
                x_units="Hz", y_units="Pa",
                domain="frequency",
                validate=True
            )

        assert data is not None


class TestTestModeDetectionEdgeCases:
    """Test edge cases in test mode detection."""

    @pytest.mark.edge_case
    def test_ambiguous_monotonic_data(self):
        """Test detection with ambiguous monotonic data."""
        # Data that could be either relaxation or creep
        time = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # Flat response (neither increasing nor decreasing)
        response = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        data = RheoData(x=time, y=response, x_units="s", domain="time")

        
        detected = detect_test_mode(data)

        # Should default to something or return unknown
        assert detected is not None

    @pytest.mark.edge_case
    def test_detection_with_noisy_data(self, synthetic_noisy_data):
        """Test detection with noisy data."""
        clean, noisy = synthetic_noisy_data

        

        clean_mode = detect_test_mode(clean)
        noisy_mode = detect_test_mode(noisy)

        # Both should detect as relaxation despite noise
        assert clean_mode == "relaxation"
        assert noisy_mode == "relaxation"

    @pytest.mark.edge_case
    def test_detection_with_reversed_axis(self):
        """Test detection when independent variable is reversed."""
        # Normally time/frequency increases left-to-right
        # What if it's reversed?
        time = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        stress = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        data = RheoData(x=time, y=stress, x_units="s", domain="time")

        # Should still detect (or warn about non-monotonic axis)
        
        detected = detect_test_mode(data)

        assert detected is not None

    @pytest.mark.edge_case
    def test_detection_with_missing_metadata(self):
        """Test detection when metadata is minimal."""
        frequency = np.array([0.1, 1.0, 10.0])
        modulus = np.array([1e5, 2e5, 3e5])

        # Minimal metadata
        data = RheoData(
            x=frequency, y=modulus,
            x_units="Hz", y_units="Pa",
            domain="frequency",
            metadata={}
        )

        
        detected = detect_test_mode(data)

        # Should infer from domain/units
        assert detected == "oscillation"


class TestComplexNumberEdgeCases:
    """Test edge cases with complex numbers."""

    @pytest.mark.edge_case
    def test_purely_real_complex_modulus(self):
        """Test complex modulus with zero imaginary part."""
        frequency = np.array([0.1, 1.0, 10.0])
        G_complex = np.array([1e5 + 0j, 2e5 + 0j, 3e5 + 0j])

        data = RheoData(
            x=frequency, y=G_complex,
            x_units="Hz", y_units="Pa",
            domain="frequency"
        )

        # Extract components
        assert np.all(data.y.imag == 0)
        assert np.all(data.y.real > 0)

    @pytest.mark.edge_case
    def test_purely_imaginary_complex_modulus(self):
        """Test complex modulus with zero real part."""
        frequency = np.array([0.1, 1.0, 10.0])
        # Physically unusual but mathematically valid
        G_complex = np.array([0 + 1e5j, 0 + 2e5j, 0 + 3e5j])

        data = RheoData(
            x=frequency, y=G_complex,
            x_units="Hz", y_units="Pa",
            domain="frequency",
            validate=False  # Bypass validation
        )

        assert np.all(data.y.real == 0)
        assert np.all(data.y.imag > 0)

    @pytest.mark.edge_case
    def test_complex_with_very_large_imaginary(self):
        """Test complex number with dominant imaginary part."""
        frequency = np.array([0.1, 1.0, 10.0])
        # Loss dominates storage
        G_complex = np.array([1e3 + 1e6j, 2e3 + 2e6j, 3e3 + 3e6j])

        data = RheoData(
            x=frequency, y=G_complex,
            x_units="Hz", y_units="Pa",
            domain="frequency"
        )

        # Magnitude should be dominated by imaginary
        magnitude = np.abs(data.y)
        expected_dominance = np.abs(data.y.imag) > 10 * np.abs(data.y.real)
        assert np.all(expected_dominance)


class TestMemoryEdgeCases:
    """Test edge cases related to memory and array operations."""

    @pytest.mark.edge_case
    def test_large_array_operations(self):
        """Test operations on large arrays."""
        # Create a large but manageable dataset
        x = np.linspace(0, 100, 10000)
        y = np.sin(x)

        data = RheoData(x=x, y=y)

        # Should support basic operations
        assert data.size == 10000
        assert np.isfinite(np.sum(data.y))

    @pytest.mark.edge_case
    def test_jax_conversion_memory_efficiency(self, oscillation_data_large):
        """Test JAX conversion doesn't duplicate data unnecessarily."""
        data = oscillation_data_large

        # Convert to JAX
        jax_data = data.to_jax()

        # Arrays should be reference-compatible
        assert isinstance(jax_data.x, jnp.ndarray)
        assert isinstance(jax_data.y, jnp.ndarray)

    @pytest.mark.edge_case
    def test_repeated_conversions(self, oscillation_data_simple):
        """Test repeated conversions don't degrade data."""
        data = oscillation_data_simple
        original_y = data.y.copy()

        # Multiple conversions
        for _ in range(5):
            data = data.to_jax().to_numpy()

        # Data should be unchanged
        np.testing.assert_array_almost_equal(data.y, original_y)


class TestInvalidInputHandling:
    """Test error handling for invalid inputs."""

    @pytest.mark.edge_case
    def test_parameter_invalid_bounds_order(self):
        """Test error when lower bound > upper bound."""
        with pytest.raises((ValueError, AssertionError)):
            Parameter("test", value=5.0, bounds=(100.0, 0.0))

    @pytest.mark.edge_case
    def test_parameter_value_outside_initial_bounds(self):
        """Test error when initial value outside bounds."""
        with pytest.raises(ValueError):
            Parameter("test", value=150.0, bounds=(0.0, 100.0))

    @pytest.mark.edge_case
    def test_data_nan_values_validation(self):
        """Test validation catches NaN values."""
        x = np.array([1.0, 2.0, np.nan])
        y = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="NaN"):
            RheoData(x=x, y=y, validate=True)

    @pytest.mark.edge_case
    def test_data_inf_values_validation(self):
        """Test validation catches infinity values."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, np.inf, 30.0])

        with pytest.raises(ValueError, match="non-finite"):
            RheoData(x=x, y=y, validate=True)

    @pytest.mark.edge_case
    def test_parameter_set_duplicate_names(self):
        """Test handling of duplicate parameter names."""
        params = ParameterSet()
        params.add("test", value=1.0)

        # Adding duplicate should either overwrite or raise
        # Check what the actual behavior is
        params.add("test", value=2.0)

        # Should have consistent behavior
        assert params.get("test") is not None
