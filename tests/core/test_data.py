"""Tests for RheoData class - piblin Measurement wrapper.

This test suite ensures that RheoData maintains full compatibility
with piblin.Measurement while adding JAX support and additional features.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# We'll mock piblin for now since it's not installed yet

from rheojax.core.jax_config import safe_import_jax

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()
with patch.dict("sys.modules", {"piblin": MagicMock()}):
    from rheojax.core.data import RheoData


class TestRheoDataCreation:
    """Test RheoData creation and initialization."""

    def test_create_from_numpy_arrays(self):
        """Test creating RheoData from numpy arrays."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])
        metadata = {"experiment": "test", "temperature": 25.0}

        data = RheoData(x=x, y=y, metadata=metadata)

        assert data is not None
        assert np.array_equal(data.x, x)
        assert np.array_equal(data.y, y)
        assert data.metadata == metadata

    def test_create_from_jax_arrays(self):
        """Test creating RheoData from JAX arrays."""
        x = jnp.array([1, 2, 3, 4, 5])
        y = jnp.array([10, 20, 30, 40, 50])

        data = RheoData(x=x, y=y)

        assert data is not None
        assert jnp.array_equal(data.x, x)
        assert jnp.array_equal(data.y, y)

    def test_create_with_units(self):
        """Test creating RheoData with units."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        x_units = "Hz"
        y_units = "Pa"

        data = RheoData(x=x, y=y, x_units=x_units, y_units=y_units)

        assert data.x_units == x_units
        assert data.y_units == y_units

    def test_create_with_domain_type(self):
        """Test creating RheoData with domain specification."""
        x = np.array([0.1, 0.2, 0.3])
        y = np.array([100, 200, 300])

        data = RheoData(x=x, y=y, domain="frequency")

        assert data.domain == "frequency"

        data2 = RheoData(x=x, y=y, domain="time")
        assert data2.domain == "time"

    def test_create_empty_raises_error(self):
        """Test that creating empty RheoData raises appropriate error."""
        with pytest.raises(ValueError, match="x and y data must be provided"):
            RheoData()

    def test_create_mismatched_sizes_raises_error(self):
        """Test that mismatched array sizes raise error."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20])

        with pytest.raises(ValueError, match="x and y must have the same shape"):
            RheoData(x=x, y=y)


@pytest.mark.skip(
    reason="piblin is an optional dependency - install with: pip install piblin"
)
class TestRheoDataPiblinCompatibility:
    """Test piblin.Measurement compatibility.

    NOTE: These tests are skipped by default as piblin is an optional dependency.
    To enable these tests, install piblin: pip install piblin
    """

    @patch("rheo.core.data.piblin")
    def test_wraps_piblin_measurement(self, mock_piblin):
        """Test that RheoData properly wraps piblin.Measurement."""
        # Create a mock piblin measurement
        mock_measurement = Mock()
        mock_measurement.x = np.array([1, 2, 3])
        mock_measurement.y = np.array([10, 20, 30])
        mock_measurement.metadata = {"test": "data"}

        # Create RheoData from piblin measurement
        data = RheoData.from_piblin(mock_measurement)

        assert data._measurement is mock_measurement
        assert np.array_equal(data.x, mock_measurement.x)
        assert np.array_equal(data.y, mock_measurement.y)

    @patch("rheo.core.data.piblin")
    def test_to_piblin_returns_measurement(self, mock_piblin):
        """Test converting RheoData to piblin.Measurement."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])

        data = RheoData(x=x, y=y)
        measurement = data.to_piblin()

        assert measurement is not None
        # Verify that piblin.Measurement was called
        mock_piblin.Measurement.assert_called()

    def test_piblin_methods_forwarded(self):
        """Test that piblin methods are accessible through RheoData."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])

        data = RheoData(x=x, y=y)

        # Test that common piblin methods are accessible
        assert hasattr(data, "copy")
        assert hasattr(data, "slice")
        assert hasattr(data, "interpolate")


class TestRheoDataNumpyCompatibility:
    """Test NumPy array operation compatibility."""

    def test_numpy_array_interface(self):
        """Test that RheoData supports numpy array operations."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])

        data = RheoData(x=x, y=y)

        # Test array properties
        assert data.shape == y.shape
        assert data.ndim == y.ndim
        assert data.size == y.size
        assert data.dtype == y.dtype

    def test_arithmetic_operations(self):
        """Test arithmetic operations on RheoData."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])

        data1 = RheoData(x=x, y=y)
        data2 = RheoData(x=x, y=y * 2)

        # Test addition
        result = data1 + data2
        assert np.array_equal(result.y, y + y * 2)

        # Test multiplication by scalar
        result = data1 * 2
        assert np.array_equal(result.y, y * 2)

        # Test subtraction
        result = data2 - data1
        assert np.array_equal(result.y, y)

    def test_indexing_and_slicing(self):
        """Test indexing and slicing operations."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])

        data = RheoData(x=x, y=y)

        # Test indexing
        assert data[0] == (1, 10)
        assert data[-1] == (5, 50)

        # Test slicing
        sliced = data[1:3]
        assert np.array_equal(sliced.x, x[1:3])
        assert np.array_equal(sliced.y, y[1:3])

    def test_numpy_functions(self):
        """Test that numpy functions work with RheoData."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])

        data = RheoData(x=x, y=y)

        # Test mean
        assert np.mean(data.y) == np.mean(y)

        # Test std
        assert np.std(data.y) == np.std(y)

        # Test log
        log_data = np.log(data.y)
        assert np.array_equal(log_data, np.log(y))


class TestRheoDataJAXConversion:
    """Test JAX array conversion utilities."""

    def test_to_jax(self):
        """Test conversion to JAX arrays."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])

        data = RheoData(x=x, y=y)
        jax_data = data.to_jax()

        assert isinstance(jax_data.x, jnp.ndarray)
        assert isinstance(jax_data.y, jnp.ndarray)
        assert jnp.array_equal(jax_data.x, x)
        assert jnp.array_equal(jax_data.y, y)

    def test_from_jax(self):
        """Test conversion from JAX arrays."""
        x = jnp.array([1, 2, 3])
        y = jnp.array([10, 20, 30])

        data = RheoData(x=x, y=y)
        numpy_data = data.to_numpy()

        assert isinstance(numpy_data.x, np.ndarray)
        assert isinstance(numpy_data.y, np.ndarray)
        assert np.array_equal(numpy_data.x, np.array(x))
        assert np.array_equal(numpy_data.y, np.array(y))

    def test_jax_operations(self):
        """Test that JAX operations work with RheoData."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])

        data = RheoData(x=x, y=y)
        jax_data = data.to_jax()

        # Test JAX operations
        result = jnp.sum(jax_data.y)
        assert result == jnp.sum(y)


class TestRheoDataMetadata:
    """Test metadata management."""

    def test_metadata_storage(self):
        """Test storing and retrieving metadata."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        metadata = {
            "temperature": 25.0,
            "sample": "polymer_A",
            "operator": "John Doe",
            "date": "2024-01-01",
        }

        data = RheoData(x=x, y=y, metadata=metadata)

        assert data.metadata == metadata
        assert data.metadata["temperature"] == 25.0
        assert data.metadata["sample"] == "polymer_A"

    def test_metadata_update(self):
        """Test updating metadata."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])

        data = RheoData(x=x, y=y, metadata={"initial": "value"})

        # Update metadata
        data.update_metadata({"temperature": 30.0, "pressure": 101.3})

        assert "initial" in data.metadata
        assert data.metadata["temperature"] == 30.0
        assert data.metadata["pressure"] == 101.3

    def test_metadata_copy(self):
        """Test that metadata is properly copied."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        metadata = {"test": "data"}

        data1 = RheoData(x=x, y=y, metadata=metadata)
        data2 = data1.copy()

        # Modify original metadata
        data1.metadata["new_key"] = "new_value"

        # Check that copy is independent
        assert "new_key" not in data2.metadata


class TestRheoDataDomainSupport:
    """Test time/frequency domain support."""

    def test_frequency_domain(self):
        """Test frequency domain data."""
        freq = np.logspace(-2, 2, 50)
        g_star = np.array([100 + 10j] * 50)

        data = RheoData(x=freq, y=g_star, domain="frequency")

        assert data.domain == "frequency"
        assert data.is_complex

        # Test extracting modulus and phase
        modulus = data.modulus
        phase = data.phase

        assert modulus is not None
        assert phase is not None

    def test_time_domain(self):
        """Test time domain data."""
        time = np.linspace(0, 100, 100)
        strain = np.sin(2 * np.pi * 0.1 * time)

        data = RheoData(x=time, y=strain, domain="time")

        assert data.domain == "time"
        assert not data.is_complex

    def test_domain_conversion_interface(self):
        """Test that domain conversion methods exist."""
        time = np.linspace(0, 100, 100)
        signal = np.sin(2 * np.pi * 0.1 * time)

        data = RheoData(x=time, y=signal, domain="time")

        # Check that conversion methods exist
        assert hasattr(data, "to_frequency_domain")
        assert hasattr(data, "to_time_domain")


class TestRheoDataValidation:
    """Test data validation."""

    def test_validate_finite(self):
        """Test validation of finite values."""
        x = np.array([1, 2, np.inf])
        y = np.array([10, 20, 30])

        with pytest.raises(ValueError, match="contains non-finite values"):
            RheoData(x=x, y=y, validate=True)

    def test_validate_nan(self):
        """Test validation of NaN values."""
        x = np.array([1, 2, 3])
        y = np.array([10, np.nan, 30])

        with pytest.raises(ValueError, match="contains NaN values"):
            RheoData(x=x, y=y, validate=True)

    def test_validate_monotonic(self):
        """Test validation of monotonic x-axis."""
        x = np.array([1, 3, 2])  # Non-monotonic
        y = np.array([10, 20, 30])

        with pytest.warns(UserWarning, match="x-axis is not monotonic"):
            data = RheoData(x=x, y=y, validate=True)

    def test_validate_positive(self):
        """Test validation for positive values when required."""
        x = np.array([0.1, 1, 10])
        y = np.array([100, 200, -300])

        # For certain domains (like frequency), negative values might be invalid
        with pytest.warns(UserWarning, match="negative values"):
            data = RheoData(x=x, y=y, domain="frequency", validate=True)

    def test_skip_validation(self):
        """Test that validation can be skipped."""
        x = np.array([1, 2, np.inf])
        y = np.array([10, np.nan, 30])

        # Should not raise with validate=False
        data = RheoData(x=x, y=y, validate=False)
        assert data is not None


class TestRheoDataSerialization:
    """Test serialization/deserialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        metadata = {"test": "data"}

        data = RheoData(x=x, y=y, metadata=metadata)
        data_dict = data.to_dict()

        assert "x" in data_dict
        assert "y" in data_dict
        assert "metadata" in data_dict
        assert np.array_equal(data_dict["x"], x)
        assert np.array_equal(data_dict["y"], y)
        assert data_dict["metadata"] == metadata

    def test_from_dict(self):
        """Test creation from dictionary."""
        data_dict = {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
            "metadata": {"test": "data"},
            "x_units": "Hz",
            "y_units": "Pa",
            "domain": "frequency",
        }

        data = RheoData.from_dict(data_dict)

        assert np.array_equal(data.x, np.array([1, 2, 3]))
        assert np.array_equal(data.y, np.array([10, 20, 30]))
        assert data.metadata == {"test": "data"}
        assert data.x_units == "Hz"
        assert data.y_units == "Pa"
        assert data.domain == "frequency"

    def test_round_trip_serialization(self):
        """Test that serialization round-trip preserves data."""
        x = np.array([1.1, 2.2, 3.3])
        y = np.array([10.5, 20.5, 30.5])
        metadata = {"temperature": 25.0, "sample": "test"}

        original = RheoData(
            x=x, y=y, metadata=metadata, x_units="s", y_units="Pa", domain="time"
        )

        # Round trip
        data_dict = original.to_dict()
        restored = RheoData.from_dict(data_dict)

        assert np.allclose(restored.x, original.x)
        assert np.allclose(restored.y, original.y)
        assert restored.metadata == original.metadata
        assert restored.x_units == original.x_units
        assert restored.y_units == original.y_units
        assert restored.domain == original.domain


class TestRheoDataOperations:
    """Test data operations and transformations."""

    def test_interpolation(self):
        """Test data interpolation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])

        data = RheoData(x=x, y=y)

        # Test interpolation at new x values
        new_x = np.array([1.5, 2.5, 3.5])
        interpolated = data.interpolate(new_x)

        assert np.array_equal(interpolated.x, new_x)
        assert np.allclose(interpolated.y, [15, 25, 35])

    def test_resampling(self):
        """Test data resampling."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        data = RheoData(x=x, y=y)

        # Test downsampling
        resampled = data.resample(n_points=20)

        assert len(resampled.x) == 20
        assert len(resampled.y) == 20

    def test_smoothing(self):
        """Test data smoothing."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)

        data = RheoData(x=x, y=y)

        # Test that smoothing method exists
        smoothed = data.smooth(window_size=5)

        assert smoothed is not None
        assert len(smoothed.x) == len(x)

        # Smoothed data should have less variance
        assert np.var(smoothed.y) < np.var(y)

    def test_derivative(self):
        """Test numerical derivative."""
        x = np.linspace(0, 10, 100)
        y = x**2  # Derivative should be 2*x

        data = RheoData(x=x, y=y)
        derivative = data.derivative()

        # Check approximate equality (numerical derivative)
        expected = 2 * x
        assert np.allclose(derivative.y[1:-1], expected[1:-1], rtol=0.1)

    def test_integral(self):
        """Test numerical integral."""
        x = np.linspace(0, 10, 100)
        y = np.ones_like(x)  # Integral should be x

        data = RheoData(x=x, y=y)
        integral = data.integral()

        # Check approximate equality
        assert np.allclose(integral.y, x, rtol=0.1)
