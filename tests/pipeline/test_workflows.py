"""Tests for specialized workflow pipelines.

This module tests the pre-configured workflow pipelines including
MastercurvePipeline, ModelComparisonPipeline, and conversion pipelines.
"""

import os
import tempfile

import numpy as np
import pytest

from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.registry import ModelRegistry
from rheojax.pipeline.workflows import (
    CreepToRelaxationPipeline,
    FrequencyToTimePipeline,
    MastercurvePipeline,
    ModelComparisonPipeline,
)


# Mock models for testing
class MockMaxwell(BaseModel):
    """Mock Maxwell model."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="G", value=1000.0, bounds=(0, 1e6))
        self.parameters.add(name="tau", value=1.0, bounds=(0, 100))

    def _fit(self, X, y, **kwargs):
        self.parameters.set_value("G", float(np.max(y)))
        self.parameters.set_value("tau", float(np.mean(X)))
        return self

    def _predict(self, X):
        G = self.parameters.get_value("G")
        tau = self.parameters.get_value("tau")
        return G * np.exp(-X / tau)


class MockZener(BaseModel):
    """Mock Zener model."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="G1", value=1000.0, bounds=(0, 1e6))
        self.parameters.add(name="G2", value=500.0, bounds=(0, 1e6))
        self.parameters.add(name="tau", value=1.0, bounds=(0, 100))

    def _fit(self, X, y, **kwargs):
        self.parameters.set_value("G1", float(np.max(y) * 0.6))
        self.parameters.set_value("G2", float(np.max(y) * 0.4))
        self.parameters.set_value("tau", float(np.mean(X)))
        return self

    def _predict(self, X):
        G1 = self.parameters.get_value("G1")
        G2 = self.parameters.get_value("G2")
        tau = self.parameters.get_value("tau")
        return G1 + G2 * np.exp(-X / tau)


@pytest.fixture(scope="module", autouse=True)
def register_mock_models():
    """Register mock models."""
    ModelRegistry.register("mock_maxwell")(MockMaxwell)
    ModelRegistry.register("mock_zener")(MockZener)
    yield
    ModelRegistry.unregister("mock_maxwell")
    ModelRegistry.unregister("mock_zener")


@pytest.fixture
def relaxation_data():
    """Create sample relaxation data."""
    t = np.logspace(-2, 2, 50)
    G_t = 1000 * np.exp(-t / 1.0)
    return RheoData(
        x=t,
        y=G_t,
        x_units="s",
        y_units="Pa",
        domain="time",
        metadata={"test_mode": "relaxation"},
        validate=False,
    )


@pytest.fixture
def creep_data():
    """Create sample creep compliance data."""
    t = np.logspace(-2, 2, 50)
    J_t = 1e-3 * (1 + t / 1.0)
    return RheoData(
        x=t,
        y=J_t,
        x_units="s",
        y_units="1/Pa",
        domain="time",
        metadata={"test_mode": "creep"},
        validate=False,
    )


@pytest.fixture
def frequency_data():
    """Create sample frequency domain data."""
    omega = np.logspace(-2, 2, 50)
    G_prime = 1000 * omega**2 / (1 + omega**2)
    G_double_prime = 1000 * omega / (1 + omega**2)
    G_star = G_prime + 1j * G_double_prime
    return RheoData(
        x=omega,
        y=G_star,
        x_units="rad/s",
        y_units="Pa",
        domain="frequency",
        validate=False,
    )


@pytest.fixture
def multi_temp_csv_files():
    """Create temporary CSV files for different temperatures."""
    temps = [273.15, 298.15, 323.15]
    files = []

    for temp in temps:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,modulus\n")
            t = np.logspace(-2, 2, 20)
            # Simulate temperature-dependent data
            shift = 10 ** ((temp - 298.15) / 20)
            G = 1000 * np.exp(-t * shift)
            for x, y in zip(t, G):
                f.write(f"{x},{y}\n")
            files.append(f.name)

    yield files, temps

    # Cleanup
    for f in files:
        if os.path.exists(f):
            os.unlink(f)


class TestMastercurvePipeline:
    """Test mastercurve pipeline."""

    def test_initialization(self):
        """Test mastercurve pipeline initialization."""
        pipeline = MastercurvePipeline(reference_temp=298.15)
        assert pipeline.reference_temp == 298.15
        assert len(pipeline.shift_factors) == 0

    def test_run_with_files(self, multi_temp_csv_files):
        """Test running mastercurve with multiple files."""
        files, temps = multi_temp_csv_files

        pipeline = MastercurvePipeline(reference_temp=298.15)
        pipeline.run(
            file_paths=files,
            temperatures=temps,
            format="csv",
            x_col="time",
            y_col="modulus",
        )

        assert pipeline.data is not None
        assert "mastercurve" in pipeline.data.metadata.get("type", "")

    def test_run_mismatched_lengths(self):
        """Test that mismatched file/temperature lengths raise error."""
        pipeline = MastercurvePipeline()
        with pytest.raises(ValueError, match="Number of files"):
            pipeline.run(["file1.csv"], [273.15, 298.15])

    def test_shift_factors(self, multi_temp_csv_files):
        """Test shift factor computation."""
        files, temps = multi_temp_csv_files

        pipeline = MastercurvePipeline(reference_temp=298.15)
        pipeline.run(files, temps, format="csv", x_col="time", y_col="modulus")

        shift_factors = pipeline.get_shift_factors()
        assert len(shift_factors) > 0
        # Reference temperature should have shift factor of 1.0
        assert 298.15 in shift_factors
        assert np.isclose(shift_factors[298.15], 1.0)


class TestModelComparisonPipeline:
    """Test model comparison pipeline."""

    def test_initialization(self):
        """Test model comparison pipeline initialization."""
        models = ["mock_maxwell", "mock_zener"]
        pipeline = ModelComparisonPipeline(models)

        assert pipeline.models == models
        assert len(pipeline.results) == 0

    def test_run_comparison(self, relaxation_data):
        """Test running model comparison."""
        pipeline = ModelComparisonPipeline(["mock_maxwell", "mock_zener"])
        pipeline.run(relaxation_data)

        assert len(pipeline.results) == 2
        assert "mock_maxwell" in pipeline.results
        assert "mock_zener" in pipeline.results

    def test_comparison_metrics(self, relaxation_data):
        """Test that comparison computes metrics."""
        pipeline = ModelComparisonPipeline(["mock_maxwell"])
        pipeline.run(relaxation_data)

        result = pipeline.results["mock_maxwell"]
        assert "rmse" in result
        assert "r_squared" in result
        assert "parameters" in result
        assert "aic" in result

    def test_get_best_model_rmse(self, relaxation_data):
        """Test getting best model by RMSE."""
        pipeline = ModelComparisonPipeline(["mock_maxwell", "mock_zener"])
        pipeline.run(relaxation_data)

        best = pipeline.get_best_model(metric="rmse", minimize=True)
        assert best in ["mock_maxwell", "mock_zener"]

    def test_get_best_model_r_squared(self, relaxation_data):
        """Test getting best model by R²."""
        pipeline = ModelComparisonPipeline(["mock_maxwell", "mock_zener"])
        pipeline.run(relaxation_data)

        best = pipeline.get_best_model(metric="r_squared", minimize=False)
        assert best in ["mock_maxwell", "mock_zener"]

    def test_get_comparison_table(self, relaxation_data):
        """Test getting comparison table."""
        pipeline = ModelComparisonPipeline(["mock_maxwell", "mock_zener"])
        pipeline.run(relaxation_data)

        table = pipeline.get_comparison_table()
        assert len(table) == 2
        assert "mock_maxwell" in table
        assert "rmse" in table["mock_maxwell"]
        assert "r_squared" in table["mock_maxwell"]

    def test_get_model_result(self, relaxation_data):
        """Test getting individual model result."""
        pipeline = ModelComparisonPipeline(["mock_maxwell"])
        pipeline.run(relaxation_data)

        result = pipeline.get_model_result("mock_maxwell")
        assert "model" in result
        assert "parameters" in result
        assert "predictions" in result

    def test_get_model_result_not_found(self, relaxation_data):
        """Test getting non-existent model raises error."""
        pipeline = ModelComparisonPipeline(["mock_maxwell"])
        pipeline.run(relaxation_data)

        with pytest.raises(KeyError):
            pipeline.get_model_result("nonexistent")


class TestCreepToRelaxationPipeline:
    """Test creep to relaxation conversion pipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = CreepToRelaxationPipeline()
        assert pipeline.data is None

    def test_approximate_conversion(self, creep_data):
        """Test approximate conversion method."""
        pipeline = CreepToRelaxationPipeline()
        pipeline.run(creep_data, method="approximate")

        assert pipeline.data is not None
        assert pipeline.data.metadata.get("test_mode") == "relaxation"
        assert pipeline.data.metadata.get("conversion_method") == "approximate"

    def test_conversion_result_positive(self, creep_data):
        """Test that conversion produces positive modulus."""
        pipeline = CreepToRelaxationPipeline()
        pipeline.run(creep_data, method="approximate")

        assert np.all(pipeline.data.y > 0)

    def test_exact_conversion_fallback(self, creep_data):
        """Test that exact method falls back to approximate."""
        pipeline = CreepToRelaxationPipeline()

        with pytest.warns(UserWarning, match="not fully implemented"):
            pipeline.run(creep_data, method="exact")

        assert "approximate" in pipeline.data.metadata.get("conversion_method", "")

    def test_invalid_method(self, creep_data):
        """Test invalid conversion method raises error."""
        pipeline = CreepToRelaxationPipeline()

        with pytest.raises(ValueError, match="Unknown method"):
            pipeline.run(creep_data, method="invalid")


class TestFrequencyToTimePipeline:
    """Test frequency to time conversion pipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = FrequencyToTimePipeline()
        assert pipeline.data is None

    def test_conversion(self, frequency_data):
        """Test frequency to time conversion."""
        pipeline = FrequencyToTimePipeline()
        pipeline.run(frequency_data, n_points=50)

        assert pipeline.data is not None
        assert pipeline.data.domain == "time"
        assert len(pipeline.data.x) == 50
        assert pipeline.data.metadata.get("conversion") == "frequency_to_time"

    def test_custom_time_range(self, frequency_data):
        """Test conversion with custom time range."""
        pipeline = FrequencyToTimePipeline()
        pipeline.run(frequency_data, time_range=(0.1, 10.0), n_points=30)

        assert len(pipeline.data.x) == 30
        assert np.min(pipeline.data.x) >= 0.1
        assert np.max(pipeline.data.x) <= 10.0

    def test_auto_time_range(self, frequency_data):
        """Test automatic time range generation."""
        pipeline = FrequencyToTimePipeline()
        pipeline.run(frequency_data, n_points=50)

        # Time range should be inversely related to frequency range
        assert pipeline.data is not None
        assert len(pipeline.data.x) == 50


class TestWorkflowIntegration:
    """Test integration between workflow pipelines."""

    def test_model_comparison_then_predict(self, relaxation_data):
        """Test using best model from comparison for prediction."""
        pipeline = ModelComparisonPipeline(["mock_maxwell", "mock_zener"])
        pipeline.run(relaxation_data)

        best_name = pipeline.get_best_model()
        best_result = pipeline.get_model_result(best_name)

        # Verify model can make predictions
        model = best_result["model"]
        predictions = model.predict(np.array(relaxation_data.x))
        assert len(predictions) == len(relaxation_data.x)

    def test_creep_conversion_then_fit(self, creep_data):
        """Test fitting model after creep conversion."""
        # Convert creep to relaxation
        conversion = CreepToRelaxationPipeline()
        conversion.run(creep_data)

        relaxation = conversion.get_result()

        # Fit model to converted data
        pipeline = ModelComparisonPipeline(["mock_maxwell"])
        pipeline.run(relaxation)

        assert len(pipeline.results) == 1
        assert pipeline.results["mock_maxwell"]["r_squared"] is not None
