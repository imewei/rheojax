"""Tests for Bayesian inference infrastructure.

This module tests the BayesianMixin class and related functionality for
NumPyro NUTS sampling with warm-start from NLSQ optimization.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.bayesian import BayesianMixin, BayesianResult
from rheojax.core.parameters import ParameterSet


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
        num_samples=100,
        num_chains=1,
        model_comparison={},
    )

    # Check all required fields are present
    assert hasattr(result, "posterior_samples")
    assert hasattr(result, "summary")
    assert hasattr(result, "diagnostics")
    assert hasattr(result, "num_samples")
    assert hasattr(result, "num_chains")
    assert hasattr(result, "model_comparison")

    # Check field types
    assert isinstance(result.posterior_samples, dict)
    assert isinstance(result.summary, dict)
    assert isinstance(result.diagnostics, dict)
    assert isinstance(result.num_samples, int)
    assert isinstance(result.num_chains, int)
    assert isinstance(result.model_comparison, dict)

    # Check values
    assert result.num_samples == 100
    assert result.num_chains == 1


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


def test_warm_start_multichain_initial_values():
    """Test that warm-starting with initial values works correctly for multi-chain MCMC.

    This test ensures that the fix for the 'tuple index out of range' error
    when using num_chains > 1 with initial_values continues to work.
    Previously, NumPyro expected initial values with shape (num_chains,) but
    scalar values were being passed, causing the error.
    """
    model = SimpleBayesianModel()

    # Create synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = 2.0 * X + 3.0 + np.random.normal(0, 0.5, 30)

    model.X_data = X
    model.y_data = y

    # Provide initial values (simulating NLSQ results)
    initial_values = {"a": 2.0, "b": 3.0}

    # Run Bayesian inference with warm-start and MULTIPLE chains
    # This previously failed with "tuple index out of range"
    result = model.fit_bayesian(
        X,
        y,
        num_warmup=50,
        num_samples=100,
        num_chains=4,  # Multi-chain with warm-start (the bug scenario)
        initial_values=initial_values,
    )

    # Check that result exists and has correct structure
    assert isinstance(result, BayesianResult)
    assert "a" in result.posterior_samples
    assert "b" in result.posterior_samples

    # Total samples should be num_samples * num_chains
    assert len(result.posterior_samples["a"]) == 100 * 4
    assert len(result.posterior_samples["b"]) == 100 * 4

    # Posterior means should be close to initial values (true parameters)
    mean_a = np.mean(result.posterior_samples["a"])
    mean_b = np.mean(result.posterior_samples["b"])

    # Allow generous tolerance
    assert abs(mean_a - 2.0) < 1.5
    assert abs(mean_b - 3.0) < 1.5

    # Check that convergence diagnostics exist
    assert "r_hat" in result.diagnostics or "rhat" in result.diagnostics
    assert "ess" in result.diagnostics or "n_eff" in result.diagnostics


def test_convergence_diagnostics_rhat_computation():
    """Test that R-hat convergence diagnostic is computed correctly for multi-chain sampling."""
    model = SimpleBayesianModel()

    # Create synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = 2.0 * X + 3.0 + np.random.normal(0, 0.5, 30)

    model.X_data = X
    model.y_data = y

    # Run with multiple chains for R-hat computation
    result = model.fit_bayesian(
        X,
        y,
        num_warmup=100,
        num_samples=200,
        num_chains=2,  # Multi-chain for R-hat
    )

    # Check R-hat exists in diagnostics
    diagnostics = result.diagnostics
    assert "r_hat" in diagnostics or "rhat" in diagnostics

    # Extract R-hat values
    r_hat_key = "r_hat" if "r_hat" in diagnostics else "rhat"
    r_hat_values = diagnostics[r_hat_key]

    # R-hat should exist for all parameters
    assert "a" in r_hat_values
    assert "b" in r_hat_values

    # R-hat should be close to 1.0 for good convergence
    # Acceptable range: < 1.1 (excellent), < 1.2 (acceptable)
    assert r_hat_values["a"] < 1.2
    assert r_hat_values["b"] < 1.2

    # R-hat should not be exactly 1.0 (unrealistic)
    assert r_hat_values["a"] > 0.99
    assert r_hat_values["b"] > 0.99


def test_convergence_diagnostics_ess_computation():
    """Test that ESS (effective sample size) is computed correctly."""
    model = SimpleBayesianModel()

    # Create synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = 2.0 * X + 3.0 + np.random.normal(0, 0.5, 30)

    model.X_data = X
    model.y_data = y

    # Run with enough samples for ESS computation
    num_samples = 500
    result = model.fit_bayesian(
        X,
        y,
        num_warmup=100,
        num_samples=num_samples,
        num_chains=1,
    )

    # Check ESS exists in diagnostics
    diagnostics = result.diagnostics
    assert "ess" in diagnostics or "n_eff" in diagnostics

    # Extract ESS values
    ess_key = "ess" if "ess" in diagnostics else "n_eff"
    ess_values = diagnostics[ess_key]

    # ESS should exist for all parameters
    assert "a" in ess_values
    assert "b" in ess_values

    # ESS should be positive and reasonable (at least 25% of num_samples)
    assert ess_values["a"] > num_samples * 0.25
    assert ess_values["b"] > num_samples * 0.25

    # ESS should not exceed actual number of samples
    assert ess_values["a"] <= num_samples * 1.5  # Allow some margin
    assert ess_values["b"] <= num_samples * 1.5


def test_divergence_detection():
    """Test that divergences are detected and reported in diagnostics."""
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
        num_warmup=100,
        num_samples=200,
        num_chains=1,
    )

    # Check divergences are reported
    diagnostics = result.diagnostics
    assert "divergences" in diagnostics

    # Divergences should be a non-negative integer
    assert isinstance(diagnostics["divergences"], (int, np.integer))
    assert diagnostics["divergences"] >= 0

    # For this simple problem, divergences should be low or zero
    # Lenient threshold: < 10% of total samples
    total_samples = 200
    assert diagnostics["divergences"] < total_samples * 0.1


def test_multi_chain_sampling():
    """Test that multi-chain sampling produces consistent results across chains."""
    model = SimpleBayesianModel()

    # Create synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = 2.0 * X + 3.0 + np.random.normal(0, 0.5, 30)

    model.X_data = X
    model.y_data = y

    # Run with multiple chains
    num_chains = 3
    result = model.fit_bayesian(
        X,
        y,
        num_warmup=100,
        num_samples=200,
        num_chains=num_chains,
    )

    # Posterior samples should contain all chains combined
    total_samples = 200 * num_chains
    assert len(result.posterior_samples["a"]) == total_samples
    assert len(result.posterior_samples["b"]) == total_samples

    # R-hat should be good for multi-chain sampling
    r_hat_key = "r_hat" if "r_hat" in result.diagnostics else "rhat"
    r_hat_values = result.diagnostics[r_hat_key]

    # Multi-chain R-hat should be very close to 1.0
    assert r_hat_values["a"] < 1.1
    assert r_hat_values["b"] < 1.1


def test_error_handling_invalid_bounds():
    """Test that invalid parameter bounds raise appropriate errors."""
    model = SimpleBayesianModel()

    # Set invalid bounds (lower > upper)
    model.parameters.get("a").bounds = (10.0, 1.0)  # Invalid

    # Create synthetic data
    X = np.linspace(0, 10, 30)
    y = 2.0 * X + 3.0

    model.X_data = X
    model.y_data = y

    # Should raise RuntimeError or ValueError for invalid bounds
    with pytest.raises((RuntimeError, ValueError)):
        model.fit_bayesian(
            X,
            y,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
        )


def test_error_handling_mismatched_data_dimensions():
    """Test that mismatched X and y dimensions raise appropriate errors."""
    model = SimpleBayesianModel()

    # Create mismatched data
    X = np.linspace(0, 10, 30)
    y = np.linspace(0, 10, 50)  # Different length

    model.X_data = X
    model.y_data = y

    # Should raise RuntimeError or ValueError for mismatched dimensions
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        model.fit_bayesian(
            X,
            y,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
        )


def test_error_handling_invalid_initial_values():
    """Test that invalid initial values are handled gracefully."""
    model = SimpleBayesianModel()

    # Create synthetic data
    X = np.linspace(0, 10, 30)
    y = 2.0 * X + 3.0

    model.X_data = X
    model.y_data = y

    # Provide initial values outside bounds
    initial_values = {"a": 100.0, "b": -50.0}  # Outside bounds (0.1, 10.0)

    # Should either raise error or clip to bounds
    # Implementation detail: some libraries auto-clip, others raise error
    try:
        result = model.fit_bayesian(
            X,
            y,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            initial_values=initial_values,
        )
        # If it succeeds, check samples are within bounds
        assert np.all(result.posterior_samples["a"] >= 0.1)
        assert np.all(result.posterior_samples["a"] <= 10.0)
    except (ValueError, RuntimeError):
        # Expected error for invalid initial values
        pass


def test_posterior_summary_statistics():
    """Test that posterior summary statistics are computed correctly."""
    model = SimpleBayesianModel()

    # Create synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2.0 * X + 3.0 + np.random.normal(0, 0.5, 50)

    model.X_data = X
    model.y_data = y

    # Run Bayesian inference
    result = model.fit_bayesian(
        X,
        y,
        num_warmup=100,
        num_samples=500,
        num_chains=1,
    )

    # Check summary statistics
    summary = result.summary
    assert "a" in summary
    assert "b" in summary

    # Each parameter should have mean and std
    assert "mean" in summary["a"]
    assert "std" in summary["a"]
    assert "mean" in summary["b"]
    assert "std" in summary["b"]

    # Mean should be close to true values
    assert abs(summary["a"]["mean"] - 2.0) < 0.5
    assert abs(summary["b"]["mean"] - 3.0) < 0.5

    # Standard deviation should be positive and reasonable
    assert summary["a"]["std"] > 0
    assert summary["a"]["std"] < 2.0
    assert summary["b"]["std"] > 0
    assert summary["b"]["std"] < 2.0


def test_prior_sampling_respects_bounds():
    """Test that prior sampling respects parameter bounds."""
    model = SimpleBayesianModel()

    # Sample from prior
    num_samples = 1000
    prior_samples = model.sample_prior(num_samples=num_samples)

    # Check all samples are within bounds
    a_bounds = model.parameters.get("a").bounds
    b_bounds = model.parameters.get("b").bounds

    assert np.all(prior_samples["a"] >= a_bounds[0])
    assert np.all(prior_samples["a"] <= a_bounds[1])
    assert np.all(prior_samples["b"] >= b_bounds[0])
    assert np.all(prior_samples["b"] <= b_bounds[1])

    # Check that samples span a reasonable range of the bounds
    # For uniform prior, expect samples to cover most of the range
    a_range = np.ptp(prior_samples["a"])  # Peak-to-peak
    expected_range = a_bounds[1] - a_bounds[0]
    assert a_range > expected_range * 0.8


def test_credible_intervals_coverage():
    """Test that credible intervals have correct coverage."""
    model = SimpleBayesianModel()

    # Create many posterior samples with known distribution
    num_samples = 10000
    true_mean_a = 5.0
    true_std_a = 1.0

    posterior_samples = {
        "a": np.random.normal(true_mean_a, true_std_a, num_samples),
        "b": np.random.normal(2.0, 0.5, num_samples),
    }

    # Compute 95% credible interval
    intervals = model.get_credible_intervals(posterior_samples, credibility=0.95)

    # Check interval width is reasonable for 95% CI
    # For normal distribution, 95% CI ≈ mean ± 1.96 * std
    expected_width = 2 * 1.96 * true_std_a
    actual_width = intervals["a"][1] - intervals["a"][0]

    # Allow 20% tolerance due to HDI vs percentile differences
    assert abs(actual_width - expected_width) < expected_width * 0.2

    # Check that true mean is within interval
    assert intervals["a"][0] < true_mean_a < intervals["a"][1]


def test_reproducibility_with_seed():
    """Test that Bayesian inference is reproducible with random seed.

    Note: Skipped because NumPyro's fit_bayesian() doesn't expose rng_seed parameter.
    Reproducibility can be achieved by manually setting JAX random key, but that's
    not part of the public API.
    """
    pytest.skip("rng_seed parameter not supported in fit_bayesian() API")


# Mark tests that require NumPyro
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
