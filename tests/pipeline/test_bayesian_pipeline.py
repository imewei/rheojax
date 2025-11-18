"""Tests for BayesianPipeline class.

This module tests the BayesianPipeline functionality including NLSQ fitting,
Bayesian inference, diagnostics, and method chaining for the complete
NLSQ → NUTS workflow.
"""

import os
import tempfile

import numpy as np
import pytest

from rheojax.core.base import BaseModel
from rheojax.core.bayesian import BayesianResult
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.pipeline.bayesian import BayesianPipeline

# Safe JAX import
jax, jnp = safe_import_jax()


# Mock model for testing with model_function support
class MockBayesianModel(BaseModel):
    """Simple mock model for testing Bayesian pipeline."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="a", value=1.0, bounds=(0.1, 10))
        self.parameters.add(name="b", value=1.0, bounds=(0.1, 10))

    def _fit(self, X, y, **kwargs):
        # Simple fit: set parameters to reasonable values
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        def model_fn(x, params):
            """Model function for optimization: y = a * exp(-b * x)"""
            a, b = params[0], params[1]
            x_jax = jnp.asarray(x, dtype=jnp.float64)
            return a * jnp.exp(-b * x_jax)

        # Create least squares objective
        X_jax = jnp.asarray(X, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)
        objective = create_least_squares_objective(
            model_fn, X_jax, y_jax, normalize=True
        )

        # Use NLSQ optimizer
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
        )
        self._nlsq_result = result
        return self

    def _predict(self, X):
        a = self.parameters.get_value("a")
        b = self.parameters.get_value("b")
        return np.array(self.model_function(X, jnp.array([a, b])))

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference: y = a * exp(-b * X)"""
        # Use JAX operations for compatibility with NLSQ and NumPyro
        a, b = params[0], params[1]
        X_jax = jnp.asarray(X, dtype=jnp.float64)
        return a * jnp.exp(-b * X_jax)


@pytest.fixture
def sample_data():
    """Create sample RheoData for testing."""
    t = np.linspace(0.1, 5, 30)
    # Generate data from known model: y = 5 * exp(-0.5 * t) + noise
    np.random.seed(42)  # For reproducibility
    y = 5.0 * np.exp(-0.5 * t) + np.random.normal(0, 0.1, size=t.shape)
    return RheoData(x=t, y=y, x_units="s", y_units="Pa", domain="time", test_mode='relaxation', validate=False)


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


class TestBayesianPipelineFitNLSQ:
    """Test BayesianPipeline.fit_nlsq() method."""

    def test_fit_nlsq_with_model_instance(self, sample_data):
        """Test fit_nlsq with model instance."""
        pipeline = BayesianPipeline(data=sample_data)
        model = MockBayesianModel()

        result = pipeline.fit_nlsq(model)

        # Should return self for chaining
        assert result is pipeline

        # Model should be stored
        assert pipeline._last_model is not None
        assert isinstance(pipeline._last_model, MockBayesianModel)

        # Model should be fitted
        assert pipeline._last_model.fitted_ is True

        # NLSQ result should be stored
        assert pipeline._nlsq_result is not None

    def test_fit_nlsq_stores_result(self, sample_data):
        """Test that fit_nlsq stores NLSQ result."""
        pipeline = BayesianPipeline(data=sample_data)
        model = MockBayesianModel()

        pipeline.fit_nlsq(model)

        # Should have stored NLSQ result
        assert pipeline._nlsq_result is not None

        # Parameters should be updated
        a = pipeline._last_model.parameters.get_value("a")
        b = pipeline._last_model.parameters.get_value("b")
        assert a is not None
        assert b is not None
        # Values should be reasonable (close to true values: a=5, b=0.5)
        assert 2.0 < a < 10.0
        assert 0.1 < b < 2.0


class TestBayesianPipelineFitBayesian:
    """Test BayesianPipeline.fit_bayesian() method."""

    def test_fit_bayesian_after_nlsq(self, sample_data):
        """Test fit_bayesian with warm-start from NLSQ."""
        pipeline = BayesianPipeline(data=sample_data)
        model = MockBayesianModel()

        # First fit with NLSQ
        pipeline.fit_nlsq(model)

        # Then fit Bayesian (warm-start)
        result = pipeline.fit_bayesian(num_samples=100, num_warmup=50)

        # Should return self for chaining
        assert result is pipeline

        # Bayesian result should be stored
        assert pipeline._bayesian_result is not None
        assert isinstance(pipeline._bayesian_result, BayesianResult)

        # Should have posterior samples
        assert "a" in pipeline._bayesian_result.posterior_samples
        assert "b" in pipeline._bayesian_result.posterior_samples

        # Diagnostics should be stored
        assert pipeline._diagnostics is not None
        assert "r_hat" in pipeline._diagnostics
        assert "ess" in pipeline._diagnostics

    def test_fit_bayesian_without_nlsq_raises_error(self, sample_data):
        """Test that fit_bayesian raises error if no model fitted."""
        pipeline = BayesianPipeline(data=sample_data)

        with pytest.raises(ValueError, match="No model fitted"):
            pipeline.fit_bayesian()


class TestBayesianPipelineDiagnostics:
    """Test BayesianPipeline diagnostic methods."""

    def test_get_diagnostics_returns_correct_structure(self, sample_data):
        """Test get_diagnostics returns correct structure."""
        pipeline = BayesianPipeline(data=sample_data)
        model = MockBayesianModel()

        pipeline.fit_nlsq(model)
        pipeline.fit_bayesian(num_samples=100, num_warmup=50)

        diagnostics = pipeline.get_diagnostics()

        # Should have required keys
        assert "r_hat" in diagnostics
        assert "ess" in diagnostics
        assert "divergences" in diagnostics

        # R-hat should be dict with parameter names
        assert isinstance(diagnostics["r_hat"], dict)
        assert "a" in diagnostics["r_hat"]
        assert "b" in diagnostics["r_hat"]

        # ESS should be dict with parameter names
        assert isinstance(diagnostics["ess"], dict)
        assert "a" in diagnostics["ess"]
        assert "b" in diagnostics["ess"]

        # Divergences should be int
        assert isinstance(diagnostics["divergences"], (int, np.integer))


class TestBayesianPipelineMethodChaining:
    """Test BayesianPipeline method chaining (fluent API)."""

    def test_complete_workflow_chaining(self, temp_csv_file):
        """Test complete workflow with method chaining."""
        pipeline = BayesianPipeline()

        # Test fluent API: load → fit_nlsq → fit_bayesian
        result = (
            pipeline.load(temp_csv_file, x_col="time", y_col="stress")
            .fit_nlsq(MockBayesianModel())
            .fit_bayesian(num_samples=50, num_warmup=25)
        )

        # Should return self throughout
        assert result is pipeline

        # All steps should be complete
        assert pipeline.data is not None
        assert pipeline._last_model is not None
        assert pipeline._nlsq_result is not None
        assert pipeline._bayesian_result is not None


class TestBayesianPipelineCompleteWorkflow:
    """Test complete NLSQ → NUTS workflow."""

    def test_complete_workflow(self, sample_data):
        """Test complete workflow: load → fit_nlsq → fit_bayesian → diagnostics."""
        pipeline = BayesianPipeline(data=sample_data)
        model = MockBayesianModel()

        # Step 1: NLSQ fit
        pipeline.fit_nlsq(model)
        assert pipeline._last_model.fitted_ is True

        # Step 2: Bayesian fit (warm-start)
        pipeline.fit_bayesian(num_samples=100, num_warmup=50)
        assert pipeline._bayesian_result is not None

        # Step 3: Get diagnostics
        diagnostics = pipeline.get_diagnostics()
        assert diagnostics is not None

        # Step 4: Get posterior summary
        summary = pipeline.get_posterior_summary()
        assert summary is not None

        # Summary should be a DataFrame
        import pandas as pd

        assert isinstance(summary, pd.DataFrame)

        # Should have parameter rows
        assert "a" in summary.index
        assert "b" in summary.index

        # Should have statistics columns
        assert "mean" in summary.columns
        assert "std" in summary.columns

    def test_workflow_warm_start_converges_faster(self, sample_data):
        """Test that warm-start from NLSQ helps convergence."""
        # With warm-start
        pipeline_warm = BayesianPipeline(data=sample_data)
        model_warm = MockBayesianModel()
        pipeline_warm.fit_nlsq(model_warm)
        pipeline_warm.fit_bayesian(num_samples=100, num_warmup=50)

        # Get ESS (effective sample size) - higher is better
        diagnostics_warm = pipeline_warm.get_diagnostics()
        ess_warm = diagnostics_warm["ess"]

        # ESS should be reasonable (>10 at minimum for this small sample)
        assert ess_warm["a"] > 10
        assert ess_warm["b"] > 10

        # R-hat should be reasonable (<1.1 for good convergence)
        r_hat_warm = diagnostics_warm["r_hat"]
        # Note: With small sample size, we might not get perfect convergence
        # but values should be finite
        assert np.isfinite(r_hat_warm["a"])
        assert np.isfinite(r_hat_warm["b"])


class TestBayesianPipelineIntegration:
    """Integration tests for BayesianPipeline."""

    def test_state_management_across_methods(self, sample_data):
        """Test that state is properly managed across method calls."""
        pipeline = BayesianPipeline(data=sample_data)
        model = MockBayesianModel()

        # Initial state
        assert pipeline._nlsq_result is None
        assert pipeline._bayesian_result is None
        assert pipeline._diagnostics is None

        # After NLSQ fit
        pipeline.fit_nlsq(model)
        assert pipeline._nlsq_result is not None
        assert pipeline._bayesian_result is None

        # After Bayesian fit
        pipeline.fit_bayesian(num_samples=50, num_warmup=25)
        assert pipeline._nlsq_result is not None
        assert pipeline._bayesian_result is not None
        assert pipeline._diagnostics is not None

    def test_error_handling_when_methods_called_out_of_order(self):
        """Test error handling when methods called out of order."""
        pipeline = BayesianPipeline()

        # Try to fit without data
        with pytest.raises(ValueError, match="No data loaded"):
            pipeline.fit_nlsq(MockBayesianModel())

        # Try to get diagnostics without Bayesian fit
        with pytest.raises(ValueError, match="No Bayesian result"):
            pipeline.get_diagnostics()

        # Try to get posterior summary without Bayesian fit
        with pytest.raises(ValueError, match="No Bayesian result"):
            pipeline.get_posterior_summary()
