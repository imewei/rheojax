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

    def test_get_fit_result(self, sample_data):
        """Test get_fit_result method (characterization: was only exercised
        indirectly via plot_fit/plot_diagnostics before this test)."""
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")

        fit_result = pipeline.get_fit_result()

        assert fit_result is not None
        assert hasattr(fit_result, "params")
        assert set(fit_result.params.keys()) == {"a", "b"}

    def test_get_fit_result_without_fit_raises(self, sample_data):
        """get_fit_result must raise before any model has been fitted."""
        pipeline = Pipeline(data=sample_data)

        with pytest.raises(ValueError, match="No model fitted"):
            pipeline.get_fit_result()


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


class TestPipelineFitBayesian:
    """Tests for Pipeline.fit_bayesian kwargs handling."""

    @pytest.mark.smoke
    def test_warm_start_stripped_before_model(self, sample_data):
        """warm_start must not reach model.fit_bayesian (NUTS rejects it)."""
        from unittest.mock import MagicMock, patch

        pipeline = Pipeline(data=sample_data)
        mock_model = MockModel()
        mock_model.fit_bayesian = MagicMock(return_value=MagicMock())
        pipeline._last_model = mock_model

        pipeline.fit_bayesian(warm_start=True, seed=42)

        mock_model.fit_bayesian.assert_called_once()
        _, kwargs = mock_model.fit_bayesian.call_args
        assert "warm_start" not in kwargs


# ---------------------------------------------------------------------------
# Helper models/transforms for the extended coverage tests below
# ---------------------------------------------------------------------------


class ScoreRaisingModel(MockModel):
    """Fits fine, but score() raises to exercise the NaN fallback."""

    def score(self, X, y):
        raise RuntimeError("score boom")


class FitRaisingModel(MockModel):
    """_fit raises to exercise fit()'s error-handling branch."""

    def _fit(self, X, y, **kwargs):
        raise RuntimeError("fit boom")


class TupleTransform(BaseTransform):
    """Returns a (data, extra) tuple so Pipeline unwraps result[0]."""

    def _transform(self, data):
        return (data * 2.0, {"note": "extra"})


class RaisingTransform(BaseTransform):
    """_transform raises to exercise transform()'s error branch."""

    def _transform(self, data):
        raise RuntimeError("transform boom")


class TestPipelineLoadFormats:
    """Exercise the format-specific loading branches."""

    def test_load_unknown_format_raises(self, temp_csv_file):
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="Unknown format"):
            pipeline.load(temp_csv_file, format="bogus")

    def test_load_excel_branch(self, sample_data):
        from unittest.mock import patch

        pipeline = Pipeline()
        with patch("rheojax.io.load_excel", return_value=sample_data) as m:
            pipeline.load("dummy.xlsx", format="excel")
        m.assert_called_once()
        assert pipeline.data is sample_data

    def test_load_hdf5_branch(self, sample_data):
        from unittest.mock import patch

        pipeline = Pipeline()
        with patch("rheojax.io.load_hdf5", return_value=sample_data) as m:
            pipeline.load("dummy.hdf5", format="hdf5")
        m.assert_called_once()

    def test_load_npz_branch(self, sample_data):
        from unittest.mock import patch

        pipeline = Pipeline()
        with patch(
            "rheojax.io.writers.npz_writer.load_npz", return_value=sample_data
        ) as m:
            pipeline.load("dummy.npz", format="npz")
        m.assert_called_once()

    def test_load_trios_single_segment(self, sample_data):
        from unittest.mock import patch

        pipeline = Pipeline()
        with patch("rheojax.io.load_trios", return_value=[sample_data]):
            pipeline.load("dummy.txt", format="trios")
        assert pipeline.data is sample_data

    def test_load_trios_multiple_segments_warns(self, sample_data):
        from unittest.mock import patch

        seg2 = RheoData(
            x=sample_data.x, y=sample_data.y * 2, domain="time", validate=False
        )
        pipeline = Pipeline()
        with patch("rheojax.io.load_trios", return_value=[sample_data, seg2]):
            with pytest.warns(UserWarning, match="Using first segment"):
                pipeline.load("dummy.txt", format="trios")
        assert pipeline.data is sample_data

    def test_load_attaches_test_mode(self, sample_data):
        from unittest.mock import patch

        pipeline = Pipeline()
        with patch("rheojax.io.load_excel", return_value=sample_data):
            pipeline.load("dummy.xlsx", format="excel", test_mode="relaxation")
        assert pipeline.data.metadata["test_mode"] == "relaxation"


class TestPipelineTransformBranches:
    """Cover transform edge cases and error handling."""

    def test_transform_no_x_raises(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        pipeline.data.x = None
        with pytest.raises(ValueError, match="no x values"):
            pipeline.transform("mock_transform")

    def test_transform_tuple_result_unwrapped(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        original_y = sample_data.y.copy()
        pipeline.transform(TupleTransform())
        assert np.allclose(pipeline.data.y, original_y * 2.0)

    def test_transform_error_propagates(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        with pytest.raises(RuntimeError, match="transform boom"):
            pipeline.transform(RaisingTransform())


class TestPipelineFitBranches:
    """Cover fit() score-fallback and error branches."""

    def test_fit_score_failure_records_nan(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        pipeline.fit(ScoreRaisingModel())
        # History stores (op, name, score); score should be NaN
        assert np.isnan(pipeline.history[-1][2])

    def test_fit_error_propagates(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        with pytest.raises(RuntimeError, match="fit boom"):
            pipeline.fit(FitRaisingModel())


class TestPipelinePredictBranches:
    """Cover predict() guard clauses and jax conversion."""

    def test_predict_no_data_raises(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")
        pipeline.data = None
        with pytest.raises(ValueError, match="No data available"):
            pipeline.predict()

    def test_predict_no_x_raises(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")
        pipeline.data.x = None
        with pytest.raises(ValueError, match="No data available"):
            pipeline.predict()

    def test_predict_jax_array_input(self, sample_data):
        import jax.numpy as jnp

        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")
        preds = pipeline.predict(X=jnp.asarray(sample_data.x))
        assert len(preds.x) == len(sample_data.x)


class TestPipelineSaveBranches:
    """Cover save() Excel/CSV branches and error handling."""

    def test_save_excel_with_fitted_model(self, sample_data):
        from unittest.mock import patch

        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")
        with patch("rheojax.io.save_excel") as m:
            pipeline.save("out.xlsx", format="excel")
        m.assert_called_once()
        payload = m.call_args[0][0]
        assert "parameters" in payload
        assert payload["parameters"]["model"] == "MockModel"
        # Fitted params were also stamped into data metadata (hdf5 path)
        assert pipeline.data.metadata["fitted_model"] == "MockModel"

    def test_save_csv_real(self, sample_data):
        import tempfile

        pipeline = Pipeline(data=sample_data)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline.save(path, format="csv")
            import pandas as pd

            df = pd.read_csv(path)
            assert list(df.columns) == ["x", "y"]
            assert len(df) == len(sample_data.x)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_csv_complex_y(self, sample_data):
        import tempfile

        cdata = RheoData(
            x=sample_data.x,
            y=sample_data.y + 1j * sample_data.y,
            domain="time",
            validate=False,
        )
        pipeline = Pipeline(data=cdata)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline.save(path, format="csv")
            import pandas as pd

            df = pd.read_csv(path)
            assert set(df.columns) == {"x", "y_real", "y_imag"}
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_csv_2d_y(self, sample_data):
        import tempfile

        y2d = np.column_stack([sample_data.y, sample_data.y * 2])
        data2d = RheoData(x=sample_data.x, y=y2d, domain="time", validate=False)
        pipeline = Pipeline(data=data2d)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline.save(path, format="csv")
            import pandas as pd

            df = pd.read_csv(path)
            assert set(df.columns) == {"x", "y_0", "y_1"}
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_unknown_format_raises(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        with pytest.raises(ValueError, match="Unknown format"):
            pipeline.save("out.bogus", format="bogus")


class TestPipelineSaveFigure:
    """Cover save_figure()."""

    def test_save_figure_without_plot_raises(self):
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No figure to save"):
            pipeline.save_figure("out.pdf")

    def test_save_figure_delegates(self, sample_data):
        from unittest.mock import MagicMock, patch

        pipeline = Pipeline(data=sample_data)
        pipeline._current_figure = MagicMock()
        with patch("rheojax.visualization.plotter.save_figure") as m:
            pipeline.save_figure("out.pdf")
        m.assert_called_once()
        assert pipeline.history[-1][0] == "save_figure"


class TestPipelineFitBayesianBranches:
    """Cover fit_bayesian model resolution, warm-start, and errors."""

    def test_no_model_raises(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        with pytest.raises(ValueError, match="No model available"):
            pipeline.fit_bayesian()

    def test_model_string_resolution(self, sample_data):
        from unittest.mock import MagicMock, patch

        fake = MockModel()
        fake.fit_bayesian = MagicMock(return_value=MagicMock())
        pipeline = Pipeline(data=sample_data)
        with patch.object(ModelRegistry, "create", return_value=fake) as create:
            pipeline.fit_bayesian(model="mock_model", seed=1)
        create.assert_called_once_with("mock_model")
        fake.fit_bayesian.assert_called_once()

    def test_warm_start_false_builds_midpoints_and_jax(self):
        from unittest.mock import MagicMock

        import jax.numpy as jnp

        data = RheoData(
            x=jnp.asarray(np.linspace(0.1, 5, 20)),
            y=jnp.asarray(np.linspace(1.0, 2.0, 20)),
            domain="time",
            validate=False,
        )
        pipeline = Pipeline(data=data)
        model = MockModel()
        model.fit_bayesian = MagicMock(return_value=MagicMock())
        pipeline._last_model = model

        pipeline.fit_bayesian(warm_start=False, seed=3)

        _, kwargs = model.fit_bayesian.call_args
        assert "initial_values" in kwargs
        # MockModel bounds are (0, 10) → arithmetic midpoint 5.0
        assert kwargs["initial_values"]["a"] == pytest.approx(5.0)

    def test_fit_bayesian_error_propagates(self, sample_data):
        from unittest.mock import MagicMock

        pipeline = Pipeline(data=sample_data)
        model = MockModel()
        model.fit_bayesian = MagicMock(side_effect=RuntimeError("nuts boom"))
        pipeline._last_model = model
        with pytest.raises(RuntimeError, match="nuts boom"):
            pipeline.fit_bayesian(seed=1)


class TestPipelineBayesianPlots:
    """Cover plot_bayesian / plot_diagnostics with the plotter mocked out."""

    def test_plot_bayesian_no_result_raises(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        with pytest.raises(ValueError, match="No Bayesian result"):
            pipeline.plot_bayesian(show=False)

    def test_plot_bayesian_delegates(self, sample_data):
        from unittest.mock import MagicMock, patch

        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")
        pipeline._last_bayesian_result = MagicMock()

        fake_plotter = MagicMock()
        fake_plotter.plot_bayesian.return_value = (MagicMock(), None)
        with patch(
            "rheojax.visualization.fit_plotter.FitPlotter",
            return_value=fake_plotter,
        ):
            pipeline.plot_bayesian(show=False, show_nlsq_overlay=True)
        fake_plotter.plot_bayesian.assert_called_once()
        assert pipeline.history[-1][0] == "plot_bayesian"

    def test_plot_diagnostics_no_result_raises(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        with pytest.raises(ValueError, match="No Bayesian result"):
            pipeline.plot_diagnostics()

    def test_plot_diagnostics_delegates(self, sample_data):
        from unittest.mock import MagicMock, patch

        pipeline = Pipeline(data=sample_data)
        pipeline._last_bayesian_result = MagicMock()

        fake_fig = MagicMock()  # has savefig -> becomes _current_figure
        with patch(
            "rheojax.visualization.fit_plotter.generate_diagnostic_suite",
            return_value={"trace": fake_fig},
        ) as gen:
            pipeline.plot_diagnostics(output_dir=None)
        gen.assert_called_once()
        assert pipeline._current_figure is fake_fig
        assert pipeline.history[-1][0] == "plot_diagnostics"


class TestPipelineGetFittedParameters:
    """Cover get_fitted_parameters."""

    def test_no_model_raises(self):
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No model fitted"):
            pipeline.get_fitted_parameters()

    def test_returns_param_dict(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")
        params = pipeline.get_fitted_parameters()
        assert set(params.keys()) == {"a", "b"}
        assert all(isinstance(v, float) for v in params.values())

    def test_none_value_raises(self):
        from unittest.mock import MagicMock

        pipeline = Pipeline()
        fake = MagicMock()
        fake.parameters.keys.return_value = ["a"]
        fake.parameters.get_value.return_value = None
        pipeline._last_model = fake
        with pytest.raises(ValueError, match="has no fitted value"):
            pipeline.get_fitted_parameters()


class TestPipelineCompareModels:
    """Cover compare_models."""

    def test_no_data_raises(self):
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No data loaded"):
            pipeline.compare_models(["mock_model"])

    def test_best_model_attached(self, sample_data):
        from unittest.mock import patch

        best = MockModel()

        class _R:
            model_name = "mock_model"
            _fitted_model = best

        class _Cmp:
            best_model = "mock_model"
            results = [_R()]

        pipeline = Pipeline(data=sample_data)
        pipeline.data.metadata["test_mode"] = "relaxation"
        with patch(
            "rheojax.utils.model_selection.compare_models", return_value=_Cmp()
        ) as cmp:
            pipeline.compare_models(["mock_model", "mock_model"])
        cmp.assert_called_once()
        # test_mode propagated from metadata into the call
        assert cmp.call_args.kwargs["test_mode"] == "relaxation"
        assert pipeline._last_model is best
        assert pipeline.steps[-1][0] == "compare_models"

    def test_best_model_missing_fitted_warns(self, sample_data):
        from unittest.mock import patch

        class _R:
            model_name = "mock_model"
            _fitted_model = None

        class _Cmp:
            best_model = "mock_model"
            results = [_R()]

        pipeline = Pipeline(data=sample_data)
        with patch(
            "rheojax.utils.model_selection.compare_models", return_value=_Cmp()
        ):
            pipeline.compare_models(["mock_model"])
        # No fitted model attached → _last_model stays None
        assert pipeline._last_model is None
        assert pipeline._last_comparison is not None


class TestPipelinePlotPrediction:
    """Cover plot() with prediction overlay (renders a real figure)."""

    def test_plot_with_prediction_overlay(self, sample_data):
        pipeline = Pipeline(data=sample_data)
        pipeline.fit("mock_model")
        try:
            pipeline.plot(show=False, include_prediction=True)
        except (RuntimeError, MemoryError) as exc:
            pytest.skip(f"matplotlib render unavailable: {exc}")
        assert pipeline._current_figure is not None
        assert pipeline.history[-1][0] == "plot"
