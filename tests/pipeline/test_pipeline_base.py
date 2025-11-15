"""Tests for base Pipeline class.

This module tests the core Pipeline functionality including method chaining,
data loading, transforms, model fitting, and output generation.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.base import BaseModel, BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.registry import ModelRegistry, TransformRegistry
from rheojax.pipeline import Pipeline

# Check if h5py is available
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# Mock model for testing
class MockModel(BaseModel):
    """Simple mock model for testing."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="a", value=1.0, bounds=(0, 10))
        self.parameters.add(name="b", value=1.0, bounds=(0, 10))

    def _fit(self, X, y, **kwargs):
        # Simple fit: set parameters to clipped mean values (respect bounds)
        mean_y = float(np.mean(y))
        mean_X = float(np.mean(X))
        # Clip to bounds (0, 10)
        self.parameters.set_value("a", min(max(mean_y, 0.1), 9.9))
        self.parameters.set_value("b", min(max(mean_X, 0.1), 9.9))
        return self

    def _predict(self, X):
        a = self.parameters.get_value("a")
        b = self.parameters.get_value("b")
        return a * np.ones_like(X) + b * X * 0.1


# Mock transform for testing
class MockTransform(BaseTransform):
    """Simple mock transform for testing."""

    def _transform(self, data):
        return data * 2.0


@pytest.fixture(scope="module", autouse=True)
def register_mocks():
    """Register mock components."""
    ModelRegistry.register("mock_model")(MockModel)
    TransformRegistry.register("mock_transform")(MockTransform)
    yield
    # Cleanup
    ModelRegistry.unregister("mock_model")
    TransformRegistry.unregister("mock_transform")


@pytest.fixture
def sample_data():
    """Create sample RheoData for testing."""
    t = np.linspace(0, 10, 50)
    stress = 1000 * np.exp(-t / 2)
    return RheoData(
        x=t, y=stress, x_units="s", y_units="Pa", domain="time", validate=False
    )


@pytest.fixture
def temp_csv_file(sample_data):
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("time,stress\n")
        for x, y in zip(sample_data.x, sample_data.y):
            f.write(f"{x},{y}\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestPipelineInitialization:
    """Test Pipeline initialization."""

    @pytest.mark.smoke
    def test_init_empty(self):
        """Test initialization without data."""
        pipeline = Pipeline()
        assert pipeline.data is None
        assert len(pipeline.steps) == 0
        assert len(pipeline.history) == 0
        assert pipeline._last_model is None

    @pytest.mark.smoke
    def test_init_with_data(self, sample_data):
        """Test initialization with data."""
        pipeline = Pipeline(data=sample_data)
        assert pipeline.data is not None
        assert np.array_equal(pipeline.data.x, sample_data.x)
        assert np.array_equal(pipeline.data.y, sample_data.y)


class TestPipelineDataLoading:
    """Test data loading functionality."""

    @pytest.mark.smoke
    def test_load_csv(self, temp_csv_file):
        """Test loading CSV file."""
        pipeline = Pipeline()
        pipeline.load(temp_csv_file, format="csv", x_col="time", y_col="stress")

        assert pipeline.data is not None
        assert len(pipeline.data.x) > 0
        assert len(pipeline.history) == 1
        assert pipeline.history[0][0] == "load"

    @pytest.mark.smoke
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        pipeline = Pipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.load("nonexistent.csv")

    @pytest.mark.smoke
    def test_load_auto_format(self, temp_csv_file):
        """Test automatic format detection."""
        pipeline = Pipeline()
        pipeline.load(temp_csv_file, format="auto", x_col="time", y_col="stress")
        assert pipeline.data is not None


class TestPipelineTransforms:
    """Test transform application."""

    @pytest.mark.smoke
    def test_transform_with_string(self, sample_data):
        """Test applying transform by name."""
        pipeline = Pipeline(data=sample_data)
        original_y = sample_data.y.copy()

        pipeline.transform("mock_transform")

        assert pipeline.data is not None
        assert np.allclose(pipeline.data.y, original_y * 2.0)
        assert len(pipeline.history) == 1
        assert pipeline.history[0][0] == "transform"

    @pytest.mark.smoke
    def test_transform_without_data(self):
        """Test transform without data raises error."""
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No data loaded"):
            pipeline.transform("mock_transform")

    @pytest.mark.smoke
    def test_transform_with_instance(self, sample_data):
        """Test applying transform instance."""
        pipeline = Pipeline(data=sample_data)
        transform = MockTransform()

        pipeline.transform(transform)
        assert pipeline.data is not None


class TestPipelineModelFitting:
    """Test model fitting functionality."""

    def test_fit_with_string(self, sample_data):
        """Test fitting model by name."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        assert pipeline._last_model is not None
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "fit"
        assert len(pipeline.history) == 1

    def test_fit_without_data(self):
        """Test fit without data raises error."""
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No data loaded"):
            pipeline.fit("mock_model")

    def test_fit_with_instance(self, sample_data):
        """Test fitting model instance."""
        pipeline = Pipeline(data=sample_data)
        model = MockModel()

        pipeline.fit(model)
        assert pipeline._last_model is not None
        assert isinstance(pipeline._last_model, MockModel)


class TestPipelinePredictions:
    """Test prediction functionality."""

    def test_predict_after_fit(self, sample_data):
        """Test predictions after fitting."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        predictions = pipeline.predict()

        assert isinstance(predictions, RheoData)
        assert len(predictions.x) == len(sample_data.x)
        assert predictions.metadata["type"] == "prediction"

    def test_predict_without_fit(self, sample_data):
        """Test predict without fit raises error."""
        pipeline = Pipeline(data=sample_data)
        with pytest.raises(ValueError, match="No model fitted"):
            pipeline.predict()

    def test_predict_custom_x(self, sample_data):
        """Test predictions with custom X values."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        custom_x = np.linspace(0, 5, 25)
        predictions = pipeline.predict(X=custom_x)

        assert len(predictions.x) == 25


class TestPipelineMethodChaining:
    """Test method chaining functionality."""

    def test_basic_chain(self, temp_csv_file):
        """Test basic method chaining."""
        pipeline = (
            Pipeline()
            .load(temp_csv_file, format="csv", x_col="time", y_col="stress")
            .fit("mock_model")
        )

        assert pipeline.data is not None
        assert pipeline._last_model is not None
        assert len(pipeline.history) == 2

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
    def test_full_chain(self, temp_csv_file):
        """Test full workflow chain."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            output_path = f.name

        try:
            pipeline = (
                Pipeline()
                .load(temp_csv_file, format="csv", x_col="time", y_col="stress")
                .transform("mock_transform")
                .fit("mock_model")
                .save(output_path)
            )

            assert len(pipeline.history) == 4  # load, transform, fit, save
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestPipelineHistory:
    """Test history tracking."""

    def test_history_tracking(self, sample_data):
        """Test that history is tracked correctly."""
        pipeline = Pipeline(data=sample_data)
        pipeline.transform("mock_transform")
        pipeline.fit("mock_model")

        history = pipeline.get_history()
        assert len(history) == 2
        assert history[0][0] == "transform"
        assert history[1][0] == "fit"

    def test_get_history_copy(self, sample_data):
        """Test that get_history returns a copy."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        history1 = pipeline.get_history()
        history2 = pipeline.get_history()

        assert history1 is not history2
        assert history1 == history2


class TestPipelineUtilities:
    """Test utility methods."""

    def test_get_result(self, sample_data):
        """Test get_result method."""
        pipeline = Pipeline(data=sample_data)
        result = pipeline.get_result()

        assert isinstance(result, RheoData)
        assert np.array_equal(result.x, sample_data.x)

    def test_get_result_without_data(self):
        """Test get_result without data raises error."""
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No data available"):
            pipeline.get_result()

    def test_get_last_model(self, sample_data):
        """Test get_last_model method."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        model = pipeline.get_last_model()
        assert isinstance(model, MockModel)

    def test_get_all_models(self, sample_data):
        """Test get_all_models method."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        models = pipeline.get_all_models()
        assert len(models) == 1
        assert isinstance(models[0], MockModel)

    def test_clone(self, sample_data):
        """Test pipeline cloning."""
        pipeline1 = Pipeline(data=sample_data)
        pipeline1.fit("mock_model")

        pipeline2 = pipeline1.clone()

        assert pipeline2 is not pipeline1
        assert pipeline2.data is not pipeline1.data
        assert len(pipeline2.history) == len(pipeline1.history)

    def test_reset(self, sample_data):
        """Test pipeline reset."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        pipeline.reset()

        assert pipeline.data is None
        assert len(pipeline.steps) == 0
        assert len(pipeline.history) == 0
        assert pipeline._last_model is None

    def test_repr(self, sample_data):
        """Test string representation."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        repr_str = repr(pipeline)
        assert "Pipeline" in repr_str
        assert "has_data=True" in repr_str
        assert "has_model=True" in repr_str


class TestPipelinePlotting:
    """Test plotting functionality."""

    def test_plot_basic(self, sample_data):
        """Test basic plotting."""
        pipeline = Pipeline(data=sample_data)
        # Don't show plot in tests
        result = pipeline.plot(show=False)

        assert result is pipeline  # Check chaining
        assert len(pipeline.history) == 1
        assert pipeline.history[0][0] == "plot"

    def test_plot_without_data(self):
        """Test plot without data raises error."""
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No data loaded"):
            pipeline.plot(show=False)


class TestPipelineSaving:
    """Test data saving functionality."""

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
    def test_save_hdf5(self, sample_data):
        """Test saving to HDF5."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            output_path = f.name

        try:
            pipeline = Pipeline(data=sample_data)
            pipeline.save(output_path, format="hdf5")

            assert os.path.exists(output_path)
            assert len(pipeline.history) == 1
            assert pipeline.history[0][0] == "save"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_without_data(self):
        """Test save without data raises error."""
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No data to save"):
            pipeline.save("output.hdf5")
