"""Tests for Pipeline visualization integration (plot_fit, plot_transform, etc.)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rheojax.core.base import BaseModel, BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry, TransformRegistry
from rheojax.pipeline import Pipeline

jax, jnp = safe_import_jax()


# ---------------------------------------------------------------------------
# Mock model & transform for testing
# ---------------------------------------------------------------------------

class _VizTestModel(BaseModel):
    """Simple model for visualization testing."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="a", value=2.0, bounds=(0.1, 100))
        self.parameters.add(name="b", value=0.5, bounds=(0.01, 10))

    def _fit(self, X, y, **kwargs):
        self.parameters.set_value("a", float(np.mean(np.abs(y))))
        self.parameters.set_value("b", 0.5)
        return self

    def _predict(self, X):
        a = self.parameters.get_value("a")
        b = self.parameters.get_value("b")
        return a * np.exp(-b * X)

    def model_function(self, X, params, test_mode=None, **kwargs):
        return params[0] * jnp.exp(-params[1] * X)


class _VizTestTransform(BaseTransform):
    """Simple transform that scales data by 2."""

    def _transform(self, data):
        return RheoData(
            x=data.x,
            y=data.y * 2.0,
            x_units=data.x_units,
            y_units=data.y_units,
            domain=data.domain,
            metadata={**(data.metadata or {}), "transform": "viz_test_scale"},
        )


@pytest.fixture(scope="module", autouse=True)
def register_mocks():
    """Register mock model and transform for pipeline tests."""
    try:
        ModelRegistry.create("viz_test_model")
    except (KeyError, ValueError):
        ModelRegistry.register("viz_test_model")(_VizTestModel)
    try:
        TransformRegistry.create("viz_test_scale")
    except (KeyError, ValueError):
        TransformRegistry.register("viz_test_scale")(_VizTestTransform)
    yield
    # Clean up to avoid polluting other test modules
    try:
        ModelRegistry.unregister("viz_test_model")
    except (KeyError, ValueError):
        pass
    try:
        TransformRegistry.unregister("viz_test_scale")
    except (KeyError, ValueError):
        pass


@pytest.fixture
def pipeline_with_data():
    """Pipeline with loaded time-domain data."""
    np.random.seed(42)
    t = np.linspace(0.1, 10, 50)
    y = 100 * np.exp(-0.5 * t) + np.random.normal(0, 2, 50)
    data = RheoData(
        x=t, y=y, x_units="s", y_units="Pa", domain="time",
        metadata={"test_mode": "relaxation"},
    )
    return Pipeline(data=data)


@pytest.fixture
def fitted_pipeline(pipeline_with_data):
    """Pipeline that has been fitted with a model."""
    pipeline_with_data.fit(_VizTestModel())
    return pipeline_with_data


# ---------------------------------------------------------------------------
# plot_fit tests
# ---------------------------------------------------------------------------

class TestPipelinePlotFit:
    """Tests for Pipeline.plot_fit()."""

    @pytest.mark.smoke
    def test_plot_fit_basic(self, fitted_pipeline):
        """plot_fit() creates a figure and records history."""
        result = fitted_pipeline.plot_fit(show_residuals=False, show_uncertainty=False)
        assert result is fitted_pipeline  # Returns self
        assert hasattr(fitted_pipeline, "_current_figure")
        assert fitted_pipeline._current_figure is not None
        assert ("plot_fit", "default") in fitted_pipeline.history
        plt.close(fitted_pipeline._current_figure)

    @pytest.mark.smoke
    def test_plot_fit_with_residuals(self, fitted_pipeline):
        """plot_fit() with residuals creates multi-panel figure."""
        fitted_pipeline.plot_fit(show_residuals=True)
        assert fitted_pipeline._current_figure is not None
        plt.close(fitted_pipeline._current_figure)

    def test_plot_fit_no_model_raises(self, pipeline_with_data):
        """plot_fit() before fit() raises ValueError."""
        with pytest.raises(ValueError, match="No model fitted"):
            pipeline_with_data.plot_fit()

    def test_plot_fit_no_data_raises(self):
        """plot_fit() without data raises ValueError."""
        p = Pipeline()
        with pytest.raises(ValueError, match="No data loaded"):
            p._last_model = MagicMock()
            p.plot_fit()

    def test_plot_fit_styles(self, fitted_pipeline):
        """plot_fit() works with all style presets."""
        for style in ("default", "publication", "presentation"):
            fitted_pipeline.plot_fit(style=style, show_residuals=False)
            plt.close(fitted_pipeline._current_figure)

    def test_plot_fit_chaining(self, fitted_pipeline):
        """plot_fit() supports fluent chaining."""
        result = (
            fitted_pipeline
            .plot_fit(show_residuals=False, show_uncertainty=False)
        )
        assert result is fitted_pipeline
        plt.close(fitted_pipeline._current_figure)


# ---------------------------------------------------------------------------
# plot_transform tests
# ---------------------------------------------------------------------------

class TestPipelinePlotTransform:
    """Tests for Pipeline.plot_transform()."""

    @pytest.mark.smoke
    def test_plot_transform_basic(self, pipeline_with_data):
        """plot_transform() after transform() creates a figure."""
        pipeline_with_data.transform(_VizTestTransform())
        result = pipeline_with_data.plot_transform()
        assert result is pipeline_with_data
        assert pipeline_with_data._current_figure is not None
        assert any(h[0] == "plot_transform" for h in pipeline_with_data.history)
        plt.close(pipeline_with_data._current_figure)

    @pytest.mark.smoke
    def test_transform_result_caching(self, pipeline_with_data):
        """Transform results are cached for later plotting."""
        pre_data = pipeline_with_data.data
        pipeline_with_data.transform(_VizTestTransform())

        assert "_VizTestTransform" in pipeline_with_data._transform_results
        cached_result, cached_pre = pipeline_with_data._transform_results["_VizTestTransform"]
        assert cached_pre is pre_data

    def test_plot_transform_no_transform_raises(self, pipeline_with_data):
        """plot_transform() without prior transform() raises ValueError."""
        with pytest.raises(ValueError, match="No cached result"):
            pipeline_with_data.plot_transform()

    def test_plot_transform_by_name(self, pipeline_with_data):
        """plot_transform() can reference a specific transform by name."""
        pipeline_with_data.transform(_VizTestTransform())
        pipeline_with_data.plot_transform(transform_name="_VizTestTransform")
        assert pipeline_with_data._current_figure is not None
        plt.close(pipeline_with_data._current_figure)

    def test_plot_transform_no_intermediate(self, pipeline_with_data):
        """plot_transform() with show_intermediate=False skips input panel."""
        pipeline_with_data.transform(_VizTestTransform())
        pipeline_with_data.plot_transform(show_intermediate=False)
        assert pipeline_with_data._current_figure is not None
        plt.close(pipeline_with_data._current_figure)


# ---------------------------------------------------------------------------
# fit_bayesian tests (lightweight — no real NUTS)
# ---------------------------------------------------------------------------

class TestPipelineFitBayesian:
    """Tests for Pipeline.fit_bayesian()."""

    def test_fit_bayesian_no_data_raises(self):
        """fit_bayesian() without data raises ValueError."""
        p = Pipeline()
        with pytest.raises(ValueError, match="No data loaded"):
            p.fit_bayesian()

    def test_fit_bayesian_no_model_raises(self, pipeline_with_data):
        """fit_bayesian() without fit or model arg raises ValueError."""
        with pytest.raises(ValueError, match="No model available"):
            pipeline_with_data.fit_bayesian()


# ---------------------------------------------------------------------------
# plot_bayesian tests
# ---------------------------------------------------------------------------

class TestPipelinePlotBayesian:
    """Tests for Pipeline.plot_bayesian()."""

    def test_plot_bayesian_no_result_raises(self, fitted_pipeline):
        """plot_bayesian() without fit_bayesian() raises ValueError."""
        with pytest.raises(ValueError, match="No Bayesian result"):
            fitted_pipeline.plot_bayesian()


# ---------------------------------------------------------------------------
# plot_diagnostics tests
# ---------------------------------------------------------------------------

class TestPipelinePlotDiagnostics:
    """Tests for Pipeline.plot_diagnostics()."""

    def test_plot_diagnostics_no_result_raises(self, fitted_pipeline):
        """plot_diagnostics() without fit_bayesian() raises ValueError."""
        with pytest.raises(ValueError, match="No Bayesian result"):
            fitted_pipeline.plot_diagnostics()


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestPipelineReset:
    """Tests for Pipeline.reset() clearing visualization state."""

    @pytest.mark.smoke
    def test_reset_clears_transform_cache(self, pipeline_with_data):
        """reset() clears cached transform results."""
        pipeline_with_data.transform(_VizTestTransform())
        assert len(pipeline_with_data._transform_results) > 0
        pipeline_with_data.reset()
        assert len(pipeline_with_data._transform_results) == 0
        assert pipeline_with_data._last_transform_name is None
        assert pipeline_with_data._last_bayesian_result is None


# ---------------------------------------------------------------------------
# Full workflow test
# ---------------------------------------------------------------------------

class TestPipelineFullWorkflow:
    """End-to-end workflow test: load → transform → fit → plot_*."""

    @pytest.mark.smoke
    def test_transform_then_fit_then_plot(self, pipeline_with_data):
        """Complete workflow: transform → plot_transform → fit → plot_fit."""
        result = (
            pipeline_with_data
            .transform(_VizTestTransform())
            .plot_transform(show_intermediate=True)
        )
        plt.close(result._current_figure)

        result = (
            result
            .fit(_VizTestModel())
            .plot_fit(show_residuals=True, show_uncertainty=False)
        )
        plt.close(result._current_figure)

        # Verify history
        ops = [h[0] for h in result.history]
        assert "transform" in ops
        assert "plot_transform" in ops
        assert "fit" in ops
        assert "plot_fit" in ops
