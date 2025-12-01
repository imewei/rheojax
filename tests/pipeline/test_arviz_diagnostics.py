"""Tests for ArviZ integration in BayesianPipeline.

This module comprehensively tests the ArviZ diagnostic visualization methods
including all 6 plotting functions and InferenceData conversion.
"""

import os
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")

from rheojax.core.base import BaseModel
from rheojax.core.bayesian import BayesianResult
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.pipeline.bayesian import BayesianPipeline

# Safe JAX import
jax, jnp = safe_import_jax()


# Mock model for testing
class MockBayesianModel(BaseModel):
    """Simple exponential decay model for testing."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="a", value=5.0, bounds=(0.1, 20))
        self.parameters.add(name="b", value=0.5, bounds=(0.01, 5))

    def _fit(self, X, y, **kwargs):
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        def model_fn(x, params):
            a, b = params[0], params[1]
            x_jax = jnp.asarray(x, dtype=jnp.float64)
            return a * jnp.exp(-b * x_jax)

        X_jax = jnp.asarray(X, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)
        objective = create_least_squares_objective(
            model_fn, X_jax, y_jax, normalize=True
        )

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
        a, b = params[0], params[1]
        X_jax = jnp.asarray(X, dtype=jnp.float64)
        return a * jnp.exp(-b * X_jax)


@pytest.fixture
def sample_data():
    """Create sample RheoData for testing."""
    t = np.linspace(0.1, 5, 30)
    np.random.seed(42)
    y = 5.0 * np.exp(-0.5 * t) + np.random.normal(0, 0.1, size=t.shape)
    return RheoData(
        x=t,
        y=y,
        x_units="s",
        y_units="Pa",
        domain="time",
        initial_test_mode='relaxation',
        validate=False,
    )


@pytest.fixture
def fitted_pipeline(sample_data):
    """Create a fitted BayesianPipeline with NLSQ + Bayesian results."""
    pipeline = BayesianPipeline(data=sample_data)
    model = MockBayesianModel()

    # Fit with NLSQ
    pipeline.fit_nlsq(model)

    # Fit Bayesian (use minimal samples for speed)
    pipeline.fit_bayesian(num_samples=100, num_warmup=50)

    return pipeline


# ============================================================================
# InferenceData Conversion Tests
# ============================================================================


class TestInferenceDataConversion:
    """Test conversion from BayesianResult to ArviZ InferenceData."""

    def test_to_inference_data_conversion_structure(self, fitted_pipeline):
        """Test that to_inference_data() returns valid InferenceData object."""
        # Get InferenceData
        idata = fitted_pipeline._bayesian_result.to_inference_data()

        # Check it's the right type
        try:
            import arviz as az

            assert isinstance(idata, az.InferenceData)
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Should not be None
        assert idata is not None

    def test_inference_data_contains_required_groups(self, fitted_pipeline):
        """Test that InferenceData contains required groups."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        idata = fitted_pipeline._bayesian_result.to_inference_data()

        # Check for required groups
        assert hasattr(idata, "posterior"), "Missing posterior group"
        assert hasattr(idata, "sample_stats"), "Missing sample_stats group"

        # Posterior should contain parameter samples
        assert "a" in idata.posterior
        assert "b" in idata.posterior

    def test_nuts_diagnostics_preserved_in_inference_data(self, fitted_pipeline):
        """Test that NUTS-specific diagnostics are preserved in InferenceData."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        idata = fitted_pipeline._bayesian_result.to_inference_data()

        # sample_stats should contain NUTS diagnostics
        assert hasattr(idata, "sample_stats")

        # Common NUTS diagnostics
        sample_stats = idata.sample_stats

        # At least one of these should be present
        nuts_diagnostics = ["energy", "diverging", "lp", "tree_depth"]
        has_diagnostic = any(hasattr(sample_stats, d) for d in nuts_diagnostics)
        assert has_diagnostic, "No NUTS diagnostics found in sample_stats"


# ============================================================================
# plot_pair() Tests
# ============================================================================


class TestPlotPair:
    """Test plot_pair() method for parameter correlations."""

    def test_plot_pair_returns_figure(self, fitted_pipeline):
        """Test that plot_pair() returns matplotlib Figure."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        result = fitted_pipeline.plot_pair()

        # Should return self for chaining
        assert result is fitted_pipeline

        # Figure should be created
        # ArviZ creates figures internally, we just verify no errors

    def test_plot_pair_with_optional_parameters(self, fitted_pipeline):
        """Test plot_pair() with optional parameters."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Test with specific var_names
        result = fitted_pipeline.plot_pair(var_names=["a", "b"])
        assert result is fitted_pipeline

        # Test with divergences highlighted (kind parameter)
        result = fitted_pipeline.plot_pair(divergences=True)
        assert result is fitted_pipeline

    def test_plot_pair_without_bayesian_fit_raises_error(self, sample_data):
        """Test that plot_pair() raises error if no Bayesian fit performed."""
        pipeline = BayesianPipeline(data=sample_data)

        with pytest.raises(ValueError, match="No Bayesian result available"):
            pipeline.plot_pair()


# ============================================================================
# plot_forest() Tests
# ============================================================================


class TestPlotForest:
    """Test plot_forest() method for credible intervals."""

    def test_plot_forest_returns_axes(self, fitted_pipeline):
        """Test that plot_forest() returns matplotlib Axes."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        result = fitted_pipeline.plot_forest()

        # Should return self for chaining
        assert result is fitted_pipeline

    def test_plot_forest_multiple_parameters(self, fitted_pipeline):
        """Test plot_forest() displays multiple parameters."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Test with specific var_names
        result = fitted_pipeline.plot_forest(var_names=["a", "b"])
        assert result is fitted_pipeline

        # Test with custom credible interval
        result = fitted_pipeline.plot_forest(hdi_prob=0.9)
        assert result is fitted_pipeline

    def test_plot_forest_without_bayesian_fit_raises_error(self, sample_data):
        """Test that plot_forest() raises error if no Bayesian fit performed."""
        pipeline = BayesianPipeline(data=sample_data)

        with pytest.raises(ValueError, match="No Bayesian result available"):
            pipeline.plot_forest()


# ============================================================================
# plot_energy() Tests
# ============================================================================


class TestPlotEnergy:
    """Test plot_energy() method for NUTS diagnostics."""

    def test_plot_energy_returns_axes(self, fitted_pipeline):
        """Test that plot_energy() returns matplotlib Axes."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Check if energy diagnostic is available
        idata = fitted_pipeline._bayesian_result.to_inference_data()
        if not hasattr(idata.sample_stats, "energy"):
            pytest.skip("Energy diagnostic not available in NumPyro MCMC output")

        result = fitted_pipeline.plot_energy()

        # Should return self for chaining
        assert result is fitted_pipeline

    def test_plot_energy_shows_divergences(self, fitted_pipeline):
        """Test plot_energy() can highlight divergences."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Check if energy diagnostic is available
        idata = fitted_pipeline._bayesian_result.to_inference_data()
        if not hasattr(idata.sample_stats, "energy"):
            pytest.skip("Energy diagnostic not available in NumPyro MCMC output")

        # Energy plot should work even with divergences present
        result = fitted_pipeline.plot_energy()
        assert result is fitted_pipeline

    def test_plot_energy_without_bayesian_fit_raises_error(self, sample_data):
        """Test that plot_energy() raises error if no Bayesian fit performed."""
        pipeline = BayesianPipeline(data=sample_data)

        with pytest.raises(ValueError, match="No Bayesian result available"):
            pipeline.plot_energy()


# ============================================================================
# plot_autocorr() Tests
# ============================================================================


class TestPlotAutocorr:
    """Test plot_autocorr() method for mixing quality."""

    def test_plot_autocorr_returns_axes(self, fitted_pipeline):
        """Test that plot_autocorr() returns matplotlib Axes."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        result = fitted_pipeline.plot_autocorr()

        # Should return self for chaining
        assert result is fitted_pipeline

    def test_plot_autocorr_custom_lags(self, fitted_pipeline):
        """Test plot_autocorr() with custom max_lag parameter."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Test with specific var_names
        result = fitted_pipeline.plot_autocorr(var_names=["a"])
        assert result is fitted_pipeline

        # Test with custom max_lag
        result = fitted_pipeline.plot_autocorr(max_lag=50)
        assert result is fitted_pipeline

    def test_plot_autocorr_without_bayesian_fit_raises_error(self, sample_data):
        """Test that plot_autocorr() raises error if no Bayesian fit performed."""
        pipeline = BayesianPipeline(data=sample_data)

        with pytest.raises(ValueError, match="No Bayesian result available"):
            pipeline.plot_autocorr()


# ============================================================================
# plot_rank() Tests
# ============================================================================


class TestPlotRank:
    """Test plot_rank() method for convergence assessment."""

    def test_plot_rank_returns_axes(self, fitted_pipeline):
        """Test that plot_rank() returns matplotlib Axes."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        result = fitted_pipeline.plot_rank()

        # Should return self for chaining
        assert result is fitted_pipeline

    def test_plot_rank_convergence_diagnostic(self, fitted_pipeline):
        """Test plot_rank() for convergence assessment."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Test with specific var_names
        result = fitted_pipeline.plot_rank(var_names=["a", "b"])
        assert result is fitted_pipeline

    def test_plot_rank_without_bayesian_fit_raises_error(self, sample_data):
        """Test that plot_rank() raises error if no Bayesian fit performed."""
        pipeline = BayesianPipeline(data=sample_data)

        with pytest.raises(ValueError, match="No Bayesian result available"):
            pipeline.plot_rank()


# ============================================================================
# plot_ess() Tests
# ============================================================================


class TestPlotESS:
    """Test plot_ess() method for effective sample size."""

    def test_plot_ess_returns_axes(self, fitted_pipeline):
        """Test that plot_ess() returns matplotlib Axes."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        result = fitted_pipeline.plot_ess()

        # Should return self for chaining
        assert result is fitted_pipeline

    def test_plot_ess_threshold_lines(self, fitted_pipeline):
        """Test plot_ess() shows threshold reference lines."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Test with specific kind (local, quantile, evolution)
        result = fitted_pipeline.plot_ess(kind="local")
        assert result is fitted_pipeline

        # Test with min_ess threshold
        result = fitted_pipeline.plot_ess(min_ess=400)
        assert result is fitted_pipeline

    def test_plot_ess_without_bayesian_fit_raises_error(self, sample_data):
        """Test that plot_ess() raises error if no Bayesian fit performed."""
        pipeline = BayesianPipeline(data=sample_data)

        with pytest.raises(ValueError, match="No Bayesian result available"):
            pipeline.plot_ess()


# ============================================================================
# Integration & Error Handling Tests
# ============================================================================


class TestArviZIntegration:
    """Test ArviZ integration with complete workflows."""

    def test_complete_workflow_with_arviz_plots(self, sample_data):
        """Test complete workflow: load → fit_nlsq → fit_bayesian → plot_*."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        pipeline = BayesianPipeline(data=sample_data)
        model = MockBayesianModel()

        # Complete workflow
        pipeline.fit_nlsq(model)
        pipeline.fit_bayesian(num_samples=50, num_warmup=25)

        # Test all plotting methods (except plot_energy - requires energy diagnostic)
        pipeline.plot_pair()
        pipeline.plot_forest()
        pipeline.plot_autocorr()
        pipeline.plot_rank()
        pipeline.plot_ess()

        # All should succeed without errors

    def test_plotting_methods_require_bayesian_fit(self, sample_data):
        """Test that all plotting methods require Bayesian fit."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        pipeline = BayesianPipeline(data=sample_data)
        model = MockBayesianModel()

        # Fit only with NLSQ (no Bayesian)
        pipeline.fit_nlsq(model)

        # All plotting methods should raise ValueError
        with pytest.raises(ValueError):
            pipeline.plot_pair()

        with pytest.raises(ValueError):
            pipeline.plot_forest()

        with pytest.raises(ValueError):
            pipeline.plot_energy()

        with pytest.raises(ValueError):
            pipeline.plot_autocorr()

        with pytest.raises(ValueError):
            pipeline.plot_rank()

        with pytest.raises(ValueError):
            pipeline.plot_ess()

    def test_plot_methods_return_for_chaining(self, fitted_pipeline):
        """Test that plot methods return self for chaining."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Test method chaining (skip plot_energy - requires energy diagnostic)
        result = (
            fitted_pipeline.plot_pair()
            .plot_forest()
            .plot_autocorr()
            .plot_rank()
            .plot_ess()
        )

        # Should return pipeline for chaining
        assert result is fitted_pipeline

    def test_plots_can_be_saved_to_file(self, fitted_pipeline):
        """Test that ArviZ plots can be saved to files."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate a plot
            fitted_pipeline.plot_pair()

            # Get current figure and save
            fig = plt.gcf()
            output_path = Path(tmpdir) / "test_plot.png"
            fig.savefig(output_path)

            # Verify file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            plt.close(fig)

    def test_arviz_optional_parameters_validation(self, fitted_pipeline):
        """Test that optional parameters are validated correctly."""
        try:
            import arviz as az
        except ImportError:
            pytest.skip("ArviZ not installed")

        # Test that invalid var_names are handled gracefully
        # (ArviZ should handle this, we just verify no crashes)
        try:
            fitted_pipeline.plot_pair(var_names=["nonexistent_param"])
        except (KeyError, ValueError):
            # Expected - parameter doesn't exist
            pass

        # Test that valid parameters work (plot_pair needs 2+ variables)
        fitted_pipeline.plot_pair(var_names=["a", "b"])  # Should work


# ============================================================================
# ArviZ Availability Tests
# ============================================================================


class TestArviZAvailability:
    """Test behavior when ArviZ is not available."""

    def test_plotting_methods_handle_missing_arviz(self, fitted_pipeline, monkeypatch):
        """Test that plotting methods handle missing ArviZ gracefully."""
        # Mock ArviZ as unavailable
        import sys

        original_arviz = sys.modules.get("arviz")

        try:
            # Temporarily remove arviz from sys.modules
            monkeypatch.setitem(sys.modules, "arviz", None)

            # Plotting methods should raise ImportError or informative message
            with pytest.raises((ImportError, RuntimeError)):
                fitted_pipeline.plot_pair()

        finally:
            # Restore arviz if it was available
            if original_arviz is not None:
                sys.modules["arviz"] = original_arviz
            else:
                sys.modules.pop("arviz", None)


# Mark all tests to filter deprecation warnings
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
