"""Tests for RheoData class - JAX-native rheological data container.

This test suite ensures that RheoData provides a robust container for
rheological data with JAX array support and automatic type conversion.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()
from rheojax.core.data import RheoData, _coerce_ndarray


class TestRheoDataCreation:
    """Test RheoData creation and initialization."""

    @pytest.mark.smoke
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

    @pytest.mark.smoke
    def test_create_from_jax_arrays(self):
        """Test creating RheoData from JAX arrays."""
        x = jnp.array([1, 2, 3, 4, 5])
        y = jnp.array([10, 20, 30, 40, 50])

        data = RheoData(x=x, y=y)

        assert data is not None
        assert jnp.array_equal(data.x, x)
        assert jnp.array_equal(data.y, y)

    @pytest.mark.smoke
    def test_create_with_units(self):
        """Test creating RheoData with units."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        x_units = "Hz"
        y_units = "Pa"

        data = RheoData(x=x, y=y, x_units=x_units, y_units=y_units)

        assert data.x_units == x_units
        assert data.y_units == y_units

    @pytest.mark.smoke
    def test_modulus_labels_always_shear(self):
        """Test that RheoData storage/loss modulus labels are always shear (G'/G\")."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        data = RheoData(x=x, y=y)
        assert data.storage_modulus_label == "G'"
        assert data.loss_modulus_label == 'G"'

    @pytest.mark.smoke
    def test_create_with_domain_type(self):
        """Test creating RheoData with domain specification."""
        x = np.array([0.1, 0.2, 0.3])
        y = np.array([100, 200, 300])

        data = RheoData(x=x, y=y, domain="frequency")

        assert data.domain == "frequency"

        data2 = RheoData(x=x, y=y, domain="time")
        assert data2.domain == "time"

    @pytest.mark.smoke
    def test_create_empty_raises_error(self):
        """Test that creating empty RheoData raises appropriate error."""
        with pytest.raises(ValueError, match="x and y data must be provided"):
            RheoData()

    @pytest.mark.smoke
    def test_create_mismatched_sizes_raises_error(self):
        """Test that mismatched array sizes raise error."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20])

        with pytest.raises(
            ValueError, match="x and y must have the same first dimension"
        ):
            RheoData(x=x, y=y)


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

    def test_constructor_does_not_alias_caller_metadata(self):
        """Two RheoData built from the same dict must not share state, and
        the constructor must not mutate the caller's original dict either."""
        shared = {"temperature": 25.0}
        d1 = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]), metadata=shared)
        d2 = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]), metadata=shared)

        d1.update_metadata({"sample": "A"})

        assert d1.metadata is not d2.metadata
        assert "sample" not in d2.metadata
        assert "sample" not in shared


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

    def test_from_dict_validates_by_default(self):
        """from_dict() is a deserialization boundary and should validate
        by default, catching NaN in an externally-edited payload."""
        data_dict = {
            "x": [1, 2, 3],
            "y": [10, float("nan"), 30],
        }
        with pytest.raises(ValueError, match="NaN"):
            RheoData.from_dict(data_dict)

    def test_from_dict_validate_false_opt_out(self):
        """Callers may still opt out of validation explicitly."""
        data_dict = {
            "x": [1, 2, 3],
            "y": [10, float("nan"), 30],
        }
        data = RheoData.from_dict(data_dict, validate=False)
        assert np.isnan(data.y).any()


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


class TestCoerceNdarray:
    """Test the module-level _coerce_ndarray helper."""

    def test_none_raises(self):
        """None must raise rather than silently produce an empty array."""
        with pytest.raises(ValueError, match="must be initialized"):
            _coerce_ndarray(None)

    def test_list_input_converts(self):
        """List/tuple falls through to np.asarray."""
        out = _coerce_ndarray([1, 2, 3])
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, [1, 2, 3])

    def test_jax_array_converts(self):
        """JAX array is converted to NumPy for scalar ops."""
        out = _coerce_ndarray(jnp.array([1.0, 2.0, 3.0]))
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


class TestRheoDataConstructorEdges:
    """Constructor branches: metadata normalization, scalar coercion, test mode."""

    def test_metadata_none_normalized(self):
        """metadata=None is coerced to an empty dict (defensive path)."""
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]), metadata=None)
        assert data.metadata == {}

    def test_ensure_array_scalar_input(self):
        """_ensure_array wraps a non-list/tuple/array input via np.array."""
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        out = data._ensure_array(5)
        assert isinstance(out, np.ndarray)
        assert out.item() == 5

    def test_explicit_test_mode_populates_metadata(self):
        """initial_test_mode is mirrored into both metadata keys."""
        data = RheoData(
            x=np.array([1, 2, 3]),
            y=np.array([4, 5, 6]),
            initial_test_mode="relaxation",
        )
        assert data.metadata["test_mode"] == "relaxation"
        assert data.metadata["detected_test_mode"] == "relaxation"
        assert data.test_mode == "relaxation"

    def test_test_mode_from_metadata_prepopulated(self):
        """test_mode pre-set in metadata populates detected_test_mode."""
        data = RheoData(
            x=np.array([1, 2, 3]),
            y=np.array([4, 5, 6]),
            metadata={"test_mode": "creep"},
        )
        assert data.metadata["detected_test_mode"] == "creep"

    def test_single_point_array(self):
        """A single-point array is valid (monotonic check is skipped)."""
        data = RheoData(x=np.array([1.0]), y=np.array([2.0]))
        assert data.size == 1

    def test_list_input_converted(self):
        """Plain Python lists are converted to arrays via _ensure_array."""
        data = RheoData(x=[1, 2, 3], y=[4, 5, 6])
        assert isinstance(data.x, np.ndarray)
        assert isinstance(data.y, np.ndarray)
        np.testing.assert_allclose(data.y, [4, 5, 6])

    def test_scalar_xy_raises_value_error(self):
        """0-d scalar x/y must raise a clear ValueError, not IndexError."""
        with pytest.raises(ValueError, match="0-d"):
            RheoData(x=5, y=10)

    def test_update_metadata_test_mode_invalidates_detection(self):
        """Setting test_mode via update_metadata syncs explicit mode and clears cache."""
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        data.update_metadata({"test_mode": "creep"})
        assert data._explicit_test_mode == "creep"
        assert "detected_test_mode" not in data.metadata
        assert data.test_mode == "creep"

    def test_direct_metadata_mutation_overrides_test_mode(self):
        """Directly mutating metadata['test_mode'] (bypassing update_metadata)
        must be reflected by the test_mode property, per its own docstring
        ('If explicitly set in metadata["test_mode"], returns that value')."""
        data = RheoData(
            x=np.array([1, 2, 3]),
            y=np.array([4, 5, 6]),
            initial_test_mode="relaxation",
        )
        assert data.test_mode == "relaxation"
        data.metadata["test_mode"] = "creep"
        assert data.test_mode == "creep"


class TestRheoDataValidationJAXPaths:
    """Validation branches exercised with JAX arrays."""

    def test_jax_x_nan_raises(self):
        with pytest.raises(ValueError, match="x data contains NaN"):
            RheoData(x=jnp.array([1.0, jnp.nan, 3.0]), y=jnp.array([1.0, 2.0, 3.0]))

    def test_jax_x_inf_raises(self):
        with pytest.raises(ValueError, match="x data contains non-finite"):
            RheoData(x=jnp.array([1.0, jnp.inf, 3.0]), y=jnp.array([1.0, 2.0, 3.0]))

    def test_jax_y_nan_raises(self):
        with pytest.raises(ValueError, match="y data contains NaN"):
            RheoData(x=jnp.array([1.0, 2.0, 3.0]), y=jnp.array([1.0, jnp.nan, 3.0]))

    def test_jax_y_inf_raises(self):
        with pytest.raises(ValueError, match="y data contains non-finite"):
            RheoData(x=jnp.array([1.0, 2.0, 3.0]), y=jnp.array([1.0, jnp.inf, 3.0]))

    def test_jax_non_monotonic_warns(self):
        with pytest.warns(UserWarning, match="not monotonic"):
            RheoData(x=jnp.array([1.0, 3.0, 2.0]), y=jnp.array([1.0, 2.0, 3.0]))

    def test_jax_negative_frequency_warns(self):
        with pytest.warns(UserWarning, match="negative values"):
            RheoData(
                x=jnp.array([0.1, 1.0, 10.0]),
                y=jnp.array([1.0, 2.0, -3.0]),
                domain="frequency",
            )

    def test_numpy_x_nan_raises(self):
        """x NaN on the NumPy path (distinct from the existing y-NaN test)."""
        with pytest.raises(ValueError, match="x data contains NaN"):
            RheoData(x=np.array([1.0, np.nan, 3.0]), y=np.array([1.0, 2.0, 3.0]))

    def test_numpy_y_inf_raises(self):
        """y non-finite (inf, not NaN) on the NumPy path."""
        with pytest.raises(ValueError, match="y data contains non-finite"):
            RheoData(x=np.array([1.0, 2.0, 3.0]), y=np.array([1.0, np.inf, 3.0]))


class TestRheoDataConversionCaching:
    """to_jax caching and to_dict/from_dict complex handling."""

    def test_to_jax_cached(self):
        """Second to_jax() call returns the same cached object."""
        data = RheoData(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))
        first = data.to_jax()
        second = data.to_jax()
        assert first is second

    def test_to_jax_cache_invalidated_on_reassign(self):
        """Reassigning x invalidates the JAX cache."""
        data = RheoData(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))
        first = data.to_jax()
        data.x = np.array([10.0, 20.0, 30.0])
        second = data.to_jax()
        assert first is not second
        np.testing.assert_allclose(np.asarray(second.x), [10.0, 20.0, 30.0])

    def test_to_jax_complex_preserves_dtype(self):
        data = RheoData(
            x=np.array([1.0, 2.0]),
            y=np.array([1 + 2j, 3 + 4j]),
            domain="frequency",
        )
        jax_data = data.to_jax()
        assert jnp.iscomplexobj(jax_data.y)

    def test_to_dict_complex_splits_real_imag(self):
        data = RheoData(
            x=np.array([1.0, 2.0]),
            y=np.array([1 + 2j, 3 + 4j]),
            domain="frequency",
            initial_test_mode="oscillation",
        )
        d = data.to_dict()
        assert "y_real" in d and "y_imag" in d
        assert "y" not in d
        np.testing.assert_allclose(d["y_real"], [1.0, 3.0])
        np.testing.assert_allclose(d["y_imag"], [2.0, 4.0])
        assert d["test_mode"] == "oscillation"

    def test_from_dict_complex_reconstructs(self):
        d = {
            "x": [1.0, 2.0],
            "y_real": [1.0, 3.0],
            "y_imag": [2.0, 4.0],
            "domain": "frequency",
        }
        data = RheoData.from_dict(d)
        assert data.is_complex
        np.testing.assert_allclose(np.asarray(data.y), [1 + 2j, 3 + 4j])

    def test_complex_round_trip(self):
        original = RheoData(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1 + 1j, 2 + 2j, 3 + 3j]),
            domain="frequency",
        )
        restored = RheoData.from_dict(original.to_dict())
        np.testing.assert_allclose(np.asarray(restored.y), np.asarray(original.y))


class TestRheoDataComplexProperties:
    """Complex-modulus derived properties on both NumPy and JAX arrays."""

    def _complex_np(self):
        return RheoData(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([3 + 4j, 6 + 8j, 1 + 0j]),
            domain="frequency",
        )

    def _complex_jax(self):
        return RheoData(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([3 + 4j, 6 + 8j, 1 + 0j]),
            domain="frequency",
        )

    def test_modulus_numpy(self):
        np.testing.assert_allclose(self._complex_np().modulus, [5.0, 10.0, 1.0])

    def test_phase_numpy(self):
        data = self._complex_np()
        np.testing.assert_allclose(data.phase, np.angle(np.asarray(data.y)))

    def test_real_data_modulus_phase_none(self):
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        assert data.modulus is None
        assert data.phase is None

    def test_y_real_imag_numpy(self):
        data = self._complex_np()
        np.testing.assert_allclose(data.y_real, [3.0, 6.0, 1.0])
        np.testing.assert_allclose(data.y_imag, [4.0, 8.0, 0.0])

    def test_y_real_imag_jax(self):
        data = self._complex_jax()
        assert isinstance(data.y_real, jnp.ndarray)
        assert isinstance(data.y_imag, jnp.ndarray)
        np.testing.assert_allclose(np.asarray(data.y_real), [3.0, 6.0, 1.0])
        np.testing.assert_allclose(np.asarray(data.y_imag), [4.0, 8.0, 0.0])

    def test_y_real_returns_real_data_unchanged(self):
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        np.testing.assert_allclose(data.y_real, [4, 5, 6])

    def test_y_imag_real_data_zeros_numpy(self):
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(data.y_imag, [0.0, 0.0, 0.0])

    def test_y_imag_real_data_zeros_jax(self):
        data = RheoData(x=jnp.array([1.0, 2.0, 3.0]), y=jnp.array([4.0, 5.0, 6.0]))
        assert isinstance(data.y_imag, jnp.ndarray)
        np.testing.assert_allclose(np.asarray(data.y_imag), [0.0, 0.0, 0.0])

    def test_storage_loss_modulus_complex(self):
        data = self._complex_np()
        np.testing.assert_allclose(data.storage_modulus, [3.0, 6.0, 1.0])
        np.testing.assert_allclose(data.loss_modulus, [4.0, 8.0, 0.0])

    def test_storage_loss_modulus_real_none(self):
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        assert data.storage_modulus is None
        assert data.loss_modulus is None

    def test_tan_delta_numpy(self):
        data = self._complex_np()
        # G''/G' where G' > 0; last point G'=1 -> 0/1 = 0
        np.testing.assert_allclose(data.tan_delta, [4.0 / 3.0, 8.0 / 6.0, 0.0])

    def test_tan_delta_jax(self):
        data = self._complex_jax()
        np.testing.assert_allclose(
            np.asarray(data.tan_delta), [4.0 / 3.0, 8.0 / 6.0, 0.0]
        )

    def test_tan_delta_real_none(self):
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        assert data.tan_delta is None

    def test_tan_delta_zero_gprime_is_nan(self):
        data = RheoData(
            x=np.array([1.0, 2.0]),
            y=np.array([0 + 1j, 2 + 1j]),
            domain="frequency",
        )
        td = np.asarray(data.tan_delta)
        assert np.isnan(td[0])
        np.testing.assert_allclose(td[1], 0.5)


class TestRheoDataTestModeDetection:
    """test_mode caching and metadata resolution."""

    def test_detected_mode_cached(self):
        data = RheoData(x=np.linspace(0, 10, 50), y=np.exp(-np.linspace(0, 10, 50)))
        first = data.test_mode
        # Second call must hit the private cache, returning the same value
        assert data.test_mode == first

    def test_metadata_string_mode_normalized(self):
        data = RheoData(
            x=np.array([1, 2, 3]),
            y=np.array([4, 5, 6]),
            metadata={"test_mode": "RELAXATION"},
        )
        # Clear explicit mode to force the metadata-resolution branch
        data._explicit_test_mode = None
        assert data.test_mode == "relaxation"

    def test_metadata_unknown_mode_returned_verbatim(self):
        """An unrecognized test_mode string falls through and is returned as-is."""
        data = RheoData(
            x=np.array([1, 2, 3]),
            y=np.array([4, 5, 6]),
            metadata={"test_mode": "not_a_real_mode"},
        )
        data._explicit_test_mode = None
        assert data.test_mode == "not_a_real_mode"


class TestRheoDataArithmeticEdges:
    """Arithmetic operator branches: scalars, mismatched axes, RheoData*RheoData."""

    def test_add_scalar(self):
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        np.testing.assert_allclose((data + 10).y, [14, 15, 16])

    def test_sub_scalar(self):
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        np.testing.assert_allclose((data - 1).y, [3, 4, 5])

    def test_mul_rheodata(self):
        d1 = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        d2 = RheoData(x=np.array([1, 2, 3]), y=np.array([2, 2, 2]))
        np.testing.assert_allclose((d1 * d2).y, [8, 10, 12])

    def test_add_mismatched_axes_raises(self):
        d1 = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        d2 = RheoData(x=np.array([9, 8, 7]), y=np.array([4, 5, 6]))
        with pytest.raises(ValueError, match="x-axes must match for addition"):
            _ = d1 + d2

    def test_sub_mismatched_axes_raises(self):
        d1 = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        d2 = RheoData(x=np.array([9, 8, 7]), y=np.array([4, 5, 6]))
        with pytest.raises(ValueError, match="x-axes must match for subtraction"):
            _ = d1 - d2

    def test_mul_mismatched_axes_raises(self):
        d1 = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))
        d2 = RheoData(x=np.array([9, 8, 7]), y=np.array([4, 5, 6]))
        with pytest.raises(ValueError, match="x-axes must match for multiplication"):
            _ = d1 * d2


class TestRheoDataOperationBranches:
    """Interpolate/resample/smooth/derivative/integral branch coverage."""

    def test_interpolate_decreasing_numpy(self):
        # Strictly decreasing x is flipped internally for np.interp
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        data = RheoData(x=x, y=y, validate=False)
        out = data.interpolate(np.array([1.5, 2.5, 3.5]))
        np.testing.assert_allclose(out.y, [15.0, 25.0, 35.0])

    def test_interpolate_decreasing_jax(self):
        x = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
        y = jnp.array([50.0, 40.0, 30.0, 20.0, 10.0])
        data = RheoData(x=x, y=y, validate=False)
        out = data.interpolate(jnp.array([1.5, 2.5, 3.5]))
        np.testing.assert_allclose(np.asarray(out.y), [15.0, 25.0, 35.0])

    def test_interpolate_unsorted_mixed_sign_numpy(self):
        """x that is neither strictly increasing nor strictly decreasing
        (validate=True only warns) must still interpolate correctly."""
        x = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
        y = np.array([10.0, 50.0, 20.0, 40.0, 30.0])
        with pytest.warns(UserWarning, match="not monotonic"):
            data = RheoData(x=x, y=y, validate=True)
        out = data.interpolate(np.array([3.5, 4.5]))
        np.testing.assert_allclose(out.y, [35.0, 45.0])

    def test_interpolate_complex_numpy(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        data = RheoData(x=x, y=y, domain="frequency")
        out = data.interpolate(np.array([1.5, 2.5]))
        np.testing.assert_allclose(out.y, [1.5 + 1.5j, 2.5 + 2.5j])

    def test_interpolate_complex_jax(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1 + 1j, 2 + 2j, 3 + 3j])
        data = RheoData(x=x, y=y, domain="frequency")
        out = data.interpolate(jnp.array([1.5, 2.5]))
        np.testing.assert_allclose(np.asarray(out.y), [1.5 + 1.5j, 2.5 + 2.5j])

    def test_interpolate_jax_real(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = RheoData(x=x, y=y)
        out = data.interpolate(jnp.array([1.5, 2.5]))
        np.testing.assert_allclose(np.asarray(out.y), [15.0, 25.0])

    def test_resample_frequency_logspace(self):
        x = np.logspace(-1, 1, 20)
        y = x**2
        data = RheoData(x=x, y=y, domain="frequency")
        out = data.resample(10)
        assert len(out.x) == 10
        # log-spaced grid spans the original range
        np.testing.assert_allclose(out.x[0], x.min())
        np.testing.assert_allclose(out.x[-1], x.max())

    def test_resample_frequency_nonpositive_raises(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        data = RheoData(x=x, y=y, domain="frequency", validate=False)
        with pytest.raises(ValueError, match="non-positive values"):
            data.resample(5)

    def test_resample_time_domain_autodetects_log_spacing(self):
        """Log-sampled time-domain data (e.g. relaxation/creep) must resample
        onto a log-spaced grid, not linspace, or the flat late-time tail gets
        over-resolved at the expense of the fast-decay region."""
        x = np.logspace(-3, 3, 50)
        y = np.exp(-x)
        data = RheoData(x=x, y=y)  # domain defaults to "time"
        out = data.resample(n_points=20)

        new_x = np.asarray(out.x)

        def cv(a):
            return np.std(a) / abs(np.mean(a))

        log_cv = cv(np.diff(np.log(new_x)))
        linear_cv = cv(np.diff(new_x))
        # log-spacing selected: near-uniform steps in log-space, wildly
        # non-uniform steps in linear space
        assert log_cv < 1e-6
        assert linear_cv > 0.5

    def test_resample_time_domain_linear_stays_linear(self):
        """Regression guard: genuinely linear time-domain data must not be
        mistaken for log-spaced by the auto-detection heuristic."""
        x = np.linspace(1, 101, 50)
        y = np.exp(-x / 50)
        data = RheoData(x=x, y=y)  # domain defaults to "time"
        out = data.resample(n_points=20)

        new_x = np.asarray(out.x)

        def cv(a):
            return np.std(a) / abs(np.mean(a))

        linear_cv = cv(np.diff(new_x))
        log_cv = cv(np.diff(np.log(new_x)))
        # linear spacing selected: near-uniform steps in linear space
        assert linear_cv < 1e-6
        assert log_cv > 0.5

    def test_resample_time_domain_short_array_falls_back_linear(self):
        """size<=2 guard: too few points to compute a CV comparison, so
        detection must not run and resample must not crash."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 4.0])
        data = RheoData(x=x, y=y)
        out = data.resample(n_points=5)

        new_x = np.asarray(out.x)
        assert len(new_x) == 5
        np.testing.assert_allclose(new_x, np.linspace(1.0, 2.0, 5))

    def test_resample_time_domain_nonpositive_falls_back_linear(self):
        """positivity guard: a non-positive x value in time domain must skip
        log-detection (log undefined there) and fall back to linear without
        raising, unlike the frequency-domain path which requires positivity."""
        x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        data = RheoData(x=x, y=y)
        out = data.resample(n_points=5)

        new_x = np.asarray(out.x)
        assert len(new_x) == 5
        np.testing.assert_allclose(new_x, np.linspace(-1.0, 3.0, 5))

    def test_smooth_even_window_made_odd(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        data = RheoData(x=x, y=y)
        out = data.smooth(window_size=4)
        assert len(out.y) == len(y)

    def test_smooth_complex_numpy(self):
        x = np.linspace(1, 10, 20)
        y = (np.arange(20) + 1j * np.arange(20)).astype(complex)
        data = RheoData(x=x, y=y, domain="frequency")
        out = data.smooth(window_size=3)
        assert np.iscomplexobj(np.asarray(out.y))
        assert len(out.y) == 20

    def test_smooth_complex_jax(self):
        x = jnp.linspace(1, 10, 20)
        y = jnp.array(np.arange(20) + 1j * np.arange(20))
        data = RheoData(x=x, y=y, domain="frequency")
        out = data.smooth(window_size=3)
        assert jnp.iscomplexobj(out.y)

    def test_smooth_jax_real(self):
        x = jnp.linspace(0, 10, 30)
        y = jnp.sin(x)
        data = RheoData(x=x, y=y)
        out = data.smooth(window_size=3)
        assert isinstance(out.y, jnp.ndarray)

    def test_derivative_complex_numpy(self):
        x = np.linspace(0, 10, 50)
        y = (x + 1j * x).astype(complex)
        data = RheoData(x=x, y=y, domain="frequency")
        out = data.derivative()
        # d/dx (x + ix) = 1 + i
        np.testing.assert_allclose(out.y[1:-1], (1 + 1j), rtol=1e-6)

    def test_derivative_complex_jax(self):
        x = jnp.linspace(0, 10, 50)
        y = jnp.array(np.linspace(0, 10, 50) + 1j * np.linspace(0, 10, 50))
        data = RheoData(x=x, y=y, domain="frequency")
        out = data.derivative()
        np.testing.assert_allclose(np.asarray(out.y)[1:-1], (1 + 1j), rtol=1e-6)

    def test_derivative_jax_real(self):
        x = jnp.linspace(0, 10, 50)
        y = x**2
        data = RheoData(x=x, y=y)
        out = data.derivative()
        np.testing.assert_allclose(
            np.asarray(out.y)[1:-1], 2 * np.asarray(x)[1:-1], rtol=0.1
        )

    def test_derivative_units_label(self):
        x = np.linspace(0, 10, 20)
        y = x**2
        data = RheoData(x=x, y=y, x_units="s", y_units="Pa")
        out = data.derivative()
        assert out.y_units == "d(Pa)/d(s)"

    def test_derivative_duplicate_x_raises(self):
        """Duplicate adjacent x (warn-only in validation) causes zero
        spacing in np.gradient; derivative() must raise, not return NaN."""
        x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        with pytest.warns(UserWarning, match="not monotonic"):
            data = RheoData(x=x, y=y, validate=True)
        with pytest.raises(ValueError, match="non-finite"):
            data.derivative()

    def test_integral_jax(self):
        x = jnp.linspace(0, 10, 100)
        y = jnp.ones_like(x)
        data = RheoData(x=x, y=y)
        out = data.integral()
        np.testing.assert_allclose(np.asarray(out.y), np.asarray(x), rtol=0.1)

    def test_integral_units_label(self):
        x = np.linspace(0, 10, 20)
        y = np.ones_like(x)
        data = RheoData(x=x, y=y, x_units="s", y_units="Pa")
        out = data.integral()
        assert out.y_units == "∫Pa·ds"


class TestRheoDataDomainConversion:
    """Domain-conversion stubs: warn-and-copy vs NotImplementedError."""

    def test_to_frequency_already_frequency_warns(self):
        data = RheoData(
            x=np.array([0.1, 1.0, 10.0]), y=np.array([1.0, 2.0, 3.0]),
            domain="frequency",
        )
        with pytest.warns(UserWarning, match="already in frequency"):
            out = data.to_frequency_domain()
        np.testing.assert_allclose(out.y, [1.0, 2.0, 3.0])

    def test_to_frequency_from_time_not_implemented(self):
        data = RheoData(x=np.array([1.0, 2.0, 3.0]), y=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(NotImplementedError):
            data.to_frequency_domain()

    def test_to_time_already_time_warns(self):
        data = RheoData(x=np.array([1.0, 2.0, 3.0]), y=np.array([1.0, 2.0, 3.0]))
        with pytest.warns(UserWarning, match="already in time"):
            out = data.to_time_domain()
        np.testing.assert_allclose(out.y, [1.0, 2.0, 3.0])

    def test_to_time_from_frequency_not_implemented(self):
        data = RheoData(
            x=np.array([0.1, 1.0, 10.0]), y=np.array([1.0, 2.0, 3.0]),
            domain="frequency",
        )
        with pytest.raises(NotImplementedError):
            data.to_time_domain()

    def test_non_canonical_domain_not_implemented_not_warn(self):
        """A non-canonical domain (e.g. 'scalar', as produced by
        mutation_number transforms) is neither 'time' nor 'frequency', so
        both conversion methods must fall through to NotImplementedError
        instead of the '!=' bug that made them both claim success."""
        data = RheoData(
            x=np.array([0.0]), y=np.array([1.0]),
            domain="scalar", validate=False,
        )
        with pytest.raises(NotImplementedError):
            data.to_frequency_domain()
        with pytest.raises(NotImplementedError):
            data.to_time_domain()


class TestRheoDataSlice:
    """Slice-by-x-value coverage including the empty-result guard."""

    def test_slice_both_bounds(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = RheoData(x=x, y=y)
        out = data.slice(start=2.0, end=4.0)
        np.testing.assert_allclose(out.x, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(out.y, [20.0, 30.0, 40.0])

    def test_slice_start_only(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = RheoData(x=x, y=y)
        out = data.slice(start=3.0)
        np.testing.assert_allclose(out.x, [3.0, 4.0, 5.0])

    def test_slice_end_only(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = RheoData(x=x, y=y)
        out = data.slice(end=2.0)
        np.testing.assert_allclose(out.x, [1.0, 2.0])

    def test_slice_no_bounds_returns_all(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        data = RheoData(x=x, y=y)
        out = data.slice()
        np.testing.assert_allclose(out.x, x)

    def test_slice_empty_result_warns(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        data = RheoData(x=x, y=y)
        with pytest.warns(UserWarning, match="empty result"):
            out = data.slice(start=100.0)
        assert len(out.x) == 0

    def test_slice_jax(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = RheoData(x=x, y=y)
        out = data.slice(start=2.0, end=4.0)
        np.testing.assert_allclose(np.asarray(out.x), [2.0, 3.0, 4.0])


class TestRheoDataNoneGuards:
    """Integrity guards raise (not silently corrupt) when x/y are nulled out."""

    def _valid(self):
        return RheoData(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))

    def test_to_jax_none_raises(self):
        data = self._valid()
        data.x = None
        with pytest.raises(ValueError, match="requires non-None"):
            data.to_jax()

    def test_copy_none_raises(self):
        data = self._valid()
        data.y = None
        with pytest.raises(ValueError, match="requires non-None"):
            data.copy()

    def test_to_dict_none_raises(self):
        data = self._valid()
        data.x = None
        with pytest.raises(ValueError, match="requires non-None"):
            data.to_dict()

    def test_interpolate_none_raises(self):
        data = self._valid()
        data.y = None
        with pytest.raises(ValueError, match="requires non-None"):
            data.interpolate(np.array([1.5]))

    def test_derivative_none_raises(self):
        data = self._valid()
        data.x = None
        with pytest.raises(ValueError, match="requires non-None"):
            data.derivative()

    def test_integral_none_raises(self):
        data = self._valid()
        data.y = None
        with pytest.raises(ValueError, match="requires non-None"):
            data.integral()

    def test_slice_none_raises(self):
        data = self._valid()
        data.x = None
        with pytest.raises(ValueError, match="requires non-None"):
            data.slice(start=1.0)

    def test_smooth_none_raises(self):
        data = self._valid()
        data.y = None
        with pytest.raises(ValueError, match="requires non-None"):
            data.smooth()

    def test_y_real_none_raises(self):
        data = self._valid()
        data.y = None
        with pytest.raises(ValueError, match="requires non-None"):
            _ = data.y_real

    def test_y_imag_none_raises(self):
        data = self._valid()
        data.y = None
        with pytest.raises(ValueError, match="requires non-None"):
            _ = data.y_imag


class TestRheoData2DYContract:
    """RheoData's own accessors (is_complex/y_real/y_imag/modulus/phase/
    storage_modulus/loss_modulus) must correctly split the real-valued
    (N,2) DMTA/GMM convention (G', G'' as separate columns), not just
    complex-dtype y — this is the root-cause fix for the contract mismatch
    where __post_init__ documented/allowed (N,2) y but the accessors only
    ever checked np.iscomplexobj and silently returned the full array."""

    def _2d_data(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.column_stack([[3.0, 6.0, 1.0], [4.0, 8.0, 0.0]])
        return RheoData(x=x, y=y, domain="frequency")

    def test_2d_y_still_constructs_without_opt_in(self):
        """A real-valued (N,2) y must remain constructible (e.g. GMM/multimode
        model.fit(omega, G_star) with G_star built via np.column_stack passes
        raw 2-D y straight through FitOrchestrator's RheoData validation)."""
        data = self._2d_data()
        assert data.y.shape == (3, 2)

    def test_2d_y_is_complex_true(self):
        """is_complex must recognize the (N,2) convention, not just complex dtype."""
        assert self._2d_data().is_complex is True

    def test_2d_y_real_imag_split_columns(self):
        """y_real/y_imag must split the two columns (G', G''), not return
        the full (N,2) array unchanged (the original silent-bug behavior)."""
        data = self._2d_data()
        np.testing.assert_allclose(data.y_real, [3.0, 6.0, 1.0])
        np.testing.assert_allclose(data.y_imag, [4.0, 8.0, 0.0])

    def test_2d_y_modulus_phase(self):
        data = self._2d_data()
        np.testing.assert_allclose(data.modulus, [5.0, 10.0, 1.0])
        np.testing.assert_allclose(
            data.phase, np.arctan2([4.0, 8.0, 0.0], [3.0, 6.0, 1.0])
        )

    def test_2d_y_storage_loss_modulus(self):
        data = self._2d_data()
        np.testing.assert_allclose(data.storage_modulus, [3.0, 6.0, 1.0])
        np.testing.assert_allclose(data.loss_modulus, [4.0, 8.0, 0.0])

    def test_2d_y_matches_equivalent_complex_construction(self):
        """The (N,2) convention and the complex-dtype convention must agree
        on every accessor for the same underlying G'/G'' values."""
        x = np.array([1.0, 2.0, 3.0])
        two_d = RheoData(
            x=x, y=np.column_stack([[3.0, 6.0, 1.0], [4.0, 8.0, 0.0]]),
            domain="frequency",
        )
        complex_ = RheoData(
            x=x, y=np.array([3 + 4j, 6 + 8j, 1 + 0j]), domain="frequency"
        )
        np.testing.assert_allclose(two_d.y_real, complex_.y_real)
        np.testing.assert_allclose(two_d.y_imag, complex_.y_imag)
        np.testing.assert_allclose(two_d.modulus, complex_.modulus)
        np.testing.assert_allclose(two_d.phase, complex_.phase)


class TestRheoDataDtypeAndShapeValidation:
    """Dtype and x.ndim boundary checks at the RheoData construction contract."""

    def test_bool_x_rejected(self):
        x = np.array([True, False, True, True, False])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="x data must be"):
            RheoData(x=x, y=y)

    def test_bool_y_rejected(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([True, False, True])
        with pytest.raises(ValueError, match="y data must be"):
            RheoData(x=x, y=y)

    def test_object_dtype_x_rejected(self):
        x = np.array([1.0, 2.0, "3.0"], dtype=object)
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="x data must be"):
            RheoData(x=x, y=y)

    def test_int_x_and_y_still_allowed(self):
        """Integer dtype is legitimate numeric data, not a mask — must not
        be rejected by the new dtype check."""
        data = RheoData(x=np.array([0, 1, 2, 3, 4]), y=np.array([0, 1, 0, 1, 0]))
        assert data.x.dtype.kind in "iu"

    def test_complex_y_still_allowed(self):
        """Complex y (DMTA/GMM complex modulus) must remain valid."""
        data = RheoData(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1 + 2j, 3 + 4j, 5 + 6j]),
            domain="frequency",
        )
        assert data.is_complex

    def test_2d_x_rejected(self):
        """A (N,1) x (common pandas df[['col']].values footgun) must raise,
        not silently pass through to model.X_data unreshaped."""
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="1-dimensional"):
            RheoData(x=x, y=y)

    def test_y_3_columns_rejected(self):
        """y with shape (N, 3) is neither the plain (N,) nor the (N, 2)
        DMTA/GMM G'/G'' convention the class supports — must raise, not
        silently pass through as if it were valid scalar data."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        with pytest.raises(ValueError, match="unsupported|2-dimensional"):
            RheoData(x=x, y=y)

    def test_y_3d_rejected(self):
        """y with ndim > 2 must raise."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.zeros((3, 2, 2))
        with pytest.raises(ValueError, match="unsupported|2-dimensional"):
            RheoData(x=x, y=y)

    def test_y_2_columns_still_allowed(self):
        """The documented (N, 2) DMTA/GMM convention must still construct."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]])
        data = RheoData(x=x, y=y)
        assert data.y.shape == (3, 2)
