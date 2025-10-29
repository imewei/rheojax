"""Tests for Bayesian inference infrastructure.

This module tests the BayesianMixin class and related functionality for
NumPyro NUTS sampling with warm-start from NLSQ optimization.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheo.core.bayesian import BayesianMixin, BayesianResult
from rheo.core.parameters import ParameterSet


class SimpleBayesianModel(BayesianMixin):
    """Simple model for testing BayesianMixin functionality."""

    def __init__(self):
        """Initialize simple model."""
        self.parameters = ParameterSet()
        self.parameters.add("a", value=1.0, bounds=(0.1, 10.0))
        self.parameters.add("b", value=1.0, bounds=(0.1, 10.0))
        self.X_data = None
        self.y_data = None

    def model_function(self, X, params):
        """Simple linear model: y = a * X + b."""
        a, b = params
        return a * X + b


def test_bayesian_mixin_initialization():
    """Test that BayesianMixin can be instantiated with parameters."""
    model = SimpleBayesianModel()

    # BayesianMixin should work with ParameterSet
    assert hasattr(model, "parameters")
    assert isinstance(model.parameters, ParameterSet)
    assert len(model.parameters) == 2


def test_sample_prior_returns_correct_structure():
    """Test that sample_prior() returns Dict[str, np.ndarray] with correct shape."""
    model = SimpleBayesianModel()

    num_samples = 100
    prior_samples = model.sample_prior(num_samples=num_samples)

    # Should return dictionary with parameter names as keys
    assert isinstance(prior_samples, dict)
    assert "a" in prior_samples
    assert "b" in prior_samples

    # Each parameter should have num_samples samples
    assert isinstance(prior_samples["a"], np.ndarray)
    assert isinstance(prior_samples["b"], np.ndarray)
    assert len(prior_samples["a"]) == num_samples
    assert len(prior_samples["b"]) == num_samples

    # Samples should be within bounds
    assert np.all(prior_samples["a"] >= 0.1)
    assert np.all(prior_samples["a"] <= 10.0)
    assert np.all(prior_samples["b"] >= 0.1)
    assert np.all(prior_samples["b"] <= 10.0)


def test_get_credible_intervals_computes_hdi():
    """Test that get_credible_intervals() computes highest density intervals."""
    model = SimpleBayesianModel()

    # Create mock posterior samples (normal distribution)
    num_samples = 1000
    posterior_samples = {
        "a": np.random.normal(5.0, 0.5, num_samples),
        "b": np.random.normal(2.0, 0.3, num_samples),
    }

    # Compute 95% credible intervals
    intervals = model.get_credible_intervals(posterior_samples, credibility=0.95)

    # Should return dictionary with parameter names
    assert isinstance(intervals, dict)
    assert "a" in intervals
    assert "b" in intervals

    # Each interval should be a tuple of (lower, upper)
    assert isinstance(intervals["a"], tuple)
    assert len(intervals["a"]) == 2
    assert intervals["a"][0] < intervals["a"][1]

    # Intervals should roughly contain 95% of samples
    # Check that mean is within interval
    assert intervals["a"][0] < 5.0 < intervals["a"][1]
    assert intervals["b"][0] < 2.0 < intervals["b"][1]


def test_bayesian_result_structure():
    """Test that BayesianResult dataclass has required fields."""
    # Create a mock BayesianResult
    posterior_samples = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([4.0, 5.0, 6.0]),
    }

    summary = {
        "a": {"mean": 2.0, "std": 0.5},
        "b": {"mean": 5.0, "std": 0.5},
    }

    diagnostics = {
        "r_hat": {"a": 1.01, "b": 1.02},
        "ess": {"a": 450, "b": 480},
        "divergences": 0,
    }

    result = BayesianResult(
        posterior_samples=posterior_samples,
        summary=summary,
        diagnostics=diagnostics,
        model_comparison={},
    )

    # Check all required fields are present
    assert hasattr(result, "posterior_samples")
    assert hasattr(result, "summary")
    assert hasattr(result, "diagnostics")
    assert hasattr(result, "model_comparison")

    # Check field types
    assert isinstance(result.posterior_samples, dict)
    assert isinstance(result.summary, dict)
    assert isinstance(result.diagnostics, dict)
    assert isinstance(result.model_comparison, dict)


def test_fit_bayesian_basic_functionality():
    """Test basic fit_bayesian() workflow on simple problem."""
    model = SimpleBayesianModel()

    # Create simple synthetic data: y = 2*x + 3 + noise
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2.0 * X + 3.0 + np.random.normal(0, 0.5, 50)

    # Store data for model
    model.X_data = X
    model.y_data = y

    # Run Bayesian inference with minimal samples for speed
    result = model.fit_bayesian(
        X,
        y,
        num_warmup=50,
        num_samples=100,
        num_chains=1,
    )

    # Check result structure
    assert isinstance(result, BayesianResult)
    assert "a" in result.posterior_samples
    assert "b" in result.posterior_samples

    # Check sample dimensions
    assert len(result.posterior_samples["a"]) == 100
    assert len(result.posterior_samples["b"]) == 100

    # Check summary exists
    assert "a" in result.summary
    assert "b" in result.summary

    # Check diagnostics exist
    assert "r_hat" in result.diagnostics or "rhat" in result.diagnostics
    assert "ess" in result.diagnostics or "n_eff" in result.diagnostics


def test_float64_precision_in_nuts_sampling():
    """Test that NUTS sampling maintains float64 precision."""
    model = SimpleBayesianModel()

    # Create synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = 2.0 * X + 3.0 + np.random.normal(0, 0.5, 30)

    model.X_data = X
    model.y_data = y

    # Run Bayesian inference
    result = model.fit_bayesian(
        X,
        y,
        num_warmup=50,
        num_samples=100,
        num_chains=1,
    )

    # Check that posterior samples are float64
    assert result.posterior_samples["a"].dtype == np.float64
    assert result.posterior_samples["b"].dtype == np.float64


def test_warm_start_from_nlsq_initial_values():
    """Test that warm-starting from NLSQ initial values works."""
    model = SimpleBayesianModel()

    # Create synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = 2.0 * X + 3.0 + np.random.normal(0, 0.5, 30)

    model.X_data = X
    model.y_data = y

    # Provide initial values (simulating NLSQ results)
    initial_values = {"a": 2.0, "b": 3.0}

    # Run Bayesian inference with warm-start
    result = model.fit_bayesian(
        X,
        y,
        num_warmup=50,
        num_samples=100,
        num_chains=1,
        initial_values=initial_values,
    )

    # Check that result exists and has correct structure
    assert isinstance(result, BayesianResult)
    assert "a" in result.posterior_samples
    assert "b" in result.posterior_samples

    # Posterior means should be close to initial values (true parameters)
    mean_a = np.mean(result.posterior_samples["a"])
    mean_b = np.mean(result.posterior_samples["b"])

    # Allow generous tolerance since we're using few samples (100 samples + MCMC stochasticity)
    assert abs(mean_a - 2.0) < 1.5
    assert abs(mean_b - 3.0) < 1.5


# Mark tests that require NumPyro
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
