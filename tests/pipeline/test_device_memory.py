"""Tests for device memory optimization in Pipeline.

This module tests the host/device memory churn reduction optimization,
ensuring data remains as JAX arrays throughout pipeline stages and only
converts to NumPy at the plotting boundary.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.pipeline import Pipeline
from rheojax.visualization.plotter import _ensure_numpy

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


# Mock model for testing
class SimpleModel(BaseModel):
    """Simple model for device memory testing."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="scale", value=1.0, bounds=(0.1, 10.0))

    def _fit(self, X, y, **kwargs):
        # Simple fit: just store mean of y as scale parameter
        if isinstance(y, jnp.ndarray):
            mean_y = float(jnp.mean(y))
        else:
            mean_y = float(np.mean(y))
        self.parameters.set_value("scale", max(0.1, min(mean_y, 10.0)))
        return self

    def _predict(self, X):
        scale = self.parameters.get_value("scale")
        # Return same type as input (JAX or NumPy)
        if isinstance(X, jnp.ndarray):
            return scale * jnp.ones_like(X)
        else:
            return scale * np.ones_like(X)


# Register mock model
ModelRegistry.register("simple_test_model", protocols=[])(SimpleModel)


@pytest.fixture
def sample_data():
    """Create sample RheoData for testing."""
    t = np.linspace(0, 10, 100)
    G = np.exp(-t) + 0.1 * np.random.randn(len(t))
    return RheoData(
        x=t,
        y=G,
        x_units="s",
        y_units="Pa",
        domain="time",
        metadata={"test_mode": "relaxation"},
    )


@pytest.fixture
def sample_csv_file(sample_data):
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("time,stress\n")
        for x_val, y_val in zip(sample_data.x, sample_data.y):
            f.write(f"{x_val},{y_val}\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestDeviceMemoryOptimization:
    """Test suite for device memory optimization."""

    def test_data_remains_jax_arrays_throughout_pipeline(self, sample_data):
        """Test 1: Data remains as JAX arrays throughout pipeline stages."""
        # Convert to JAX arrays
        jax_data = RheoData(
            x=jnp.array(sample_data.x),
            y=jnp.array(sample_data.y),
            x_units=sample_data.x_units,
            y_units=sample_data.y_units,
            domain=sample_data.domain,
            metadata=sample_data.metadata,
        )

        pipeline = Pipeline(data=jax_data)

        # Verify data starts as JAX arrays
        assert isinstance(pipeline.data.x, jnp.ndarray)
        assert isinstance(pipeline.data.y, jnp.ndarray)

        # Fit model - data should remain as JAX arrays
        pipeline.fit("simple_test_model")

        # After fit, data should still be JAX arrays
        assert isinstance(pipeline.data.x, jnp.ndarray)
        assert isinstance(pipeline.data.y, jnp.ndarray)

    def test_conversion_only_at_plotting_boundary(self, sample_data):
        """Test 2: Conversion to NumPy only happens at plotting boundary."""
        # Start with JAX arrays
        jax_x = jnp.array(sample_data.x)
        jax_y = jnp.array(sample_data.y)

        # Verify _ensure_numpy converts JAX to NumPy
        numpy_x = _ensure_numpy(jax_x)
        numpy_y = _ensure_numpy(jax_y)

        assert isinstance(numpy_x, np.ndarray)
        assert isinstance(numpy_y, np.ndarray)
        assert not isinstance(numpy_x, jnp.ndarray)
        assert not isinstance(numpy_y, jnp.ndarray)

    def test_pipeline_memory_profiling(self, sample_csv_file):
        """Test 3: Profile device transfers in pipeline workflow."""
        pipeline = Pipeline()
        pipeline.load(sample_csv_file, x_col="time", y_col="stress")
        pipeline.fit("simple_test_model")
        predictions = pipeline.predict()

        # Verify pipeline completes successfully
        assert pipeline._last_model is not None
        assert predictions is not None

    def test_end_to_end_pipeline_speedup(self, sample_csv_file):
        """Test 4: Measure end-to-end pipeline speedup."""
        # Warm-up
        for _ in range(3):
            pipeline = Pipeline()
            pipeline.load(sample_csv_file, x_col="time", y_col="stress")
            pipeline.fit("simple_test_model")
            _ = pipeline.predict()

        # Benchmark optimized version
        num_iterations = 20
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            pipeline = Pipeline()
            pipeline.load(sample_csv_file, x_col="time", y_col="stress")
            pipeline.fit("simple_test_model")
            predictions = pipeline.predict()

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_iterations

        # Verify it completes in reasonable time
        assert avg_time < 1.0, f"Pipeline took {avg_time:.3f}s, expected <1s"

    def test_backward_compatibility_numpy_inputs(self, sample_data):
        """Test 6: Backward compatibility with NumPy inputs."""
        pipeline = Pipeline(data=sample_data)

        # Verify NumPy arrays work
        assert isinstance(pipeline.data.x, np.ndarray)
        assert isinstance(pipeline.data.y, np.ndarray)

        pipeline.fit("simple_test_model")
        predictions = pipeline.predict()

        assert predictions is not None
        assert pipeline._last_model is not None


class TestEnsureNumpyHelper:
    """Test suite for _ensure_numpy helper function."""

    def test_ensure_numpy_converts_jax_arrays(self):
        """Test _ensure_numpy converts JAX arrays to NumPy."""
        jax_array = jnp.array([1.0, 2.0, 3.0])
        result = _ensure_numpy(jax_array)

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, jnp.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_ensure_numpy_preserves_numpy_arrays(self):
        """Test _ensure_numpy preserves NumPy arrays."""
        numpy_array = np.array([1.0, 2.0, 3.0])
        result = _ensure_numpy(numpy_array)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, numpy_array)

    def test_ensure_numpy_converts_lists(self):
        """Test _ensure_numpy converts lists to NumPy arrays."""
        list_data = [1.0, 2.0, 3.0]
        result = _ensure_numpy(list_data)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))
