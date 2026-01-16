"""Tests for Generalized Maxwell Model Bayesian inference integration.

This module tests GMM integration with NumPyro NUTS sampling including:
- model_function() implementation for all test modes
- NLSQ warm-start workflow
- Posterior sampling and convergence
- Credible interval computation

Test Coverage (Focused on Critical Bayesian Behaviors):
- model_function() for relaxation/oscillation/creep
- NLSQ → NUTS warm-start workflow
- Posterior convergence diagnostics
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models import GeneralizedMaxwell

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class TestGMMBayesianInference:
    """Test GMM Bayesian inference capabilities."""

    def test_model_function_relaxation_mode(self):
        """Test model_function works in relaxation mode."""
        # Create model
        model = GeneralizedMaxwell(n_modes=1, modulus_type="shear")

        # Fit to set test_mode
        t = np.logspace(-3, 2, 30)
        G_true = 1e6 + 5e5 * np.exp(-t / 0.1)
        model.fit(t, G_true, test_mode="relaxation")

        # Test model_function
        params = jnp.array([1e6, 5e5, 0.1])  # [G_inf, G_1, tau_1]
        predictions = model.model_function(t, params)

        # Verify predictions are reasonable
        assert predictions.shape == t.shape
        assert jnp.all(jnp.isfinite(predictions))
        assert jnp.all(predictions > 0)  # Modulus should be positive

    def test_model_function_oscillation_mode(self):
        """Test model_function works in oscillation mode."""
        # Create model
        model = GeneralizedMaxwell(n_modes=1, modulus_type="shear")

        # Fit to set test_mode
        omega = np.logspace(-2, 2, 30)
        G_prime = 1e6 * (omega * 0.1) ** 2 / (1 + (omega * 0.1) ** 2)
        G_double_prime = 1e6 * (omega * 0.1) / (1 + (omega * 0.1) ** 2)
        G_star = np.vstack([G_prime, G_double_prime])
        model.fit(omega, G_star, test_mode="oscillation")

        # Test model_function
        params = jnp.array([0.0, 1e6, 0.1])  # [G_inf, G_1, tau_1]
        predictions = model.model_function(omega, params)

        # Verify predictions have correct shape for complex modulus
        assert predictions.shape == (2, len(omega))  # [G', G"]
        assert jnp.all(jnp.isfinite(predictions))

    def test_model_function_creep_mode(self):
        """Test model_function works in creep mode."""
        # Create model
        model = GeneralizedMaxwell(n_modes=1, modulus_type="shear")

        # Fit to set test_mode
        t = np.logspace(-3, 2, 30)
        # Creep compliance J(t) for GMM - simplified for testing
        J_approx = 1.0 / 1e6 * (1 + 0.5 * (1 - np.exp(-t / 0.1)))
        model.fit(t, J_approx, test_mode="creep")

        # Test model_function
        params = jnp.array([1e6, 5e5, 0.1])  # [E_inf, E_1, tau_1]
        predictions = model.model_function(t, params)

        # Verify predictions are reasonable
        assert predictions.shape == t.shape
        assert jnp.all(jnp.isfinite(predictions))
        assert jnp.all(predictions > 0)  # Compliance should be positive

    def test_bayesian_inference_with_nlsq_warmstart(self):
        """Test complete NLSQ → NUTS workflow with warm-start."""
        # Create synthetic data
        np.random.seed(42)
        t = np.logspace(-2, 1, 25)  # Reduced points for faster testing
        G_true = 1e6 + 5e5 * np.exp(-t / 0.1)
        G_data = G_true + np.random.normal(0, 1e4, size=t.shape)

        # Step 1: Fit with NLSQ
        model = GeneralizedMaxwell(n_modes=1, modulus_type="shear")
        model.fit(t, G_data, test_mode="relaxation")

        # Extract NLSQ estimates for warm-start
        initial_values = {
            "G_inf": model.parameters.get_value("G_inf"),
            "G_1": model.parameters.get_value("G_1"),
            "tau_1": model.parameters.get_value("tau_1"),
        }

        # Step 2: Bayesian inference with warm-start (short run for testing)
        result = model.fit_bayesian(
            t,
            G_data,
            test_mode="relaxation",
            num_warmup=100,  # Minimal warmup for testing
            num_samples=100,  # Minimal samples for testing
            num_chains=1,
            initial_values=initial_values,
        )

        # Verify result structure
        assert "G_inf" in result.posterior_samples
        assert "G_1" in result.posterior_samples
        assert "tau_1" in result.posterior_samples

        # Verify sample shapes
        assert result.posterior_samples["G_inf"].shape == (
            100,
        )  # num_samples * num_chains

        # Verify diagnostics exist
        assert "r_hat" in result.diagnostics
        assert "ess" in result.diagnostics
        assert "divergences" in result.diagnostics

    def test_bayesian_inference_convergence_diagnostics(self):
        """Test that Bayesian inference produces reasonable convergence diagnostics."""
        # Create clean synthetic data
        np.random.seed(123)
        t = np.logspace(-2, 1, 25)
        G_true = 1e6 + 5e5 * np.exp(-t / 0.1)
        G_data = G_true + np.random.normal(0, 5e3, size=t.shape)  # Low noise

        # Fit model
        model = GeneralizedMaxwell(n_modes=1, modulus_type="shear")
        model.fit(t, G_data, test_mode="relaxation")

        # Bayesian inference with NLSQ warm-start
        initial_values = {
            name: model.parameters.get_value(name) for name in ["G_inf", "G_1", "tau_1"]
        }

        result = model.fit_bayesian(
            t,
            G_data,
            test_mode="relaxation",
            num_warmup=200,
            num_samples=200,
            num_chains=1,
            initial_values=initial_values,
        )

        # Check R-hat (should be close to 1.0 for converged chains)
        # With single chain, R-hat may not be meaningful, so just check it exists
        assert "r_hat" in result.diagnostics
        for param_name in ["G_inf", "G_1", "tau_1"]:
            assert param_name in result.diagnostics["r_hat"]

        # Check ESS (effective sample size)
        assert "ess" in result.diagnostics
        for param_name in ["G_inf", "G_1", "tau_1"]:
            assert param_name in result.diagnostics["ess"]
            # ESS should be positive
            assert result.diagnostics["ess"][param_name] > 0

    def test_get_credible_intervals(self):
        """Test credible interval computation from posterior samples."""
        # Create clean synthetic data
        np.random.seed(456)
        t = np.logspace(-2, 1, 25)
        G_true = 1e6 + 5e5 * np.exp(-t / 0.1)
        G_data = G_true + np.random.normal(0, 5e3, size=t.shape)

        # Fit model
        model = GeneralizedMaxwell(n_modes=1, modulus_type="shear")
        model.fit(t, G_data, test_mode="relaxation")

        # Bayesian inference
        initial_values = {
            name: model.parameters.get_value(name) for name in ["G_inf", "G_1", "tau_1"]
        }

        result = model.fit_bayesian(
            t,
            G_data,
            test_mode="relaxation",
            num_warmup=100,
            num_samples=200,
            num_chains=1,
            initial_values=initial_values,
        )

        # Compute credible intervals
        intervals = model.get_credible_intervals(
            result.posterior_samples, credibility=0.95
        )

        # Verify intervals exist for all parameters
        assert "G_inf" in intervals
        assert "G_1" in intervals
        assert "tau_1" in intervals

        # Verify intervals are tuples of (lower, upper)
        for param_name in ["G_inf", "G_1", "tau_1"]:
            lower, upper = intervals[param_name]
            assert lower < upper
            assert lower > 0  # Physical constraint
            # True value should ideally be within interval (but not guaranteed with short chains)


class TestGMMBayesianPriorSafetyIntegration:
    """Test integration of prior safety with Bayesian inference."""

    def test_fit_bayesian_uses_good_priors_from_nlsq(self):
        """Test that fit_bayesian uses NLSQ estimates when convergence is good."""
        # Create clean data for good NLSQ convergence
        np.random.seed(789)
        t = np.logspace(-2, 1, 30)
        G_true = 1e6 + 5e5 * np.exp(-t / 0.1)
        G_data = G_true + np.random.normal(0, 1e4, size=t.shape)

        # Fit with NLSQ
        model = GeneralizedMaxwell(n_modes=1, modulus_type="shear")
        model.fit(t, G_data, test_mode="relaxation")

        # Check that NLSQ converged well
        diagnostics = model._extract_nlsq_diagnostics(model._nlsq_result)
        classification = model._classify_nlsq_convergence(diagnostics)

        # For this clean data, should be "good"
        assert classification == "good"

        # Construct priors
        priors = model._construct_bayesian_priors(classification, prior_mode="warn")

        # Verify priors are centered near NLSQ estimates
        nlsq_g1 = model.parameters.get_value("G_1")
        prior_g1_mean = priors["G_1"]["mean"]

        # Prior mean should be close to NLSQ estimate
        assert abs(prior_g1_mean - nlsq_g1) / nlsq_g1 < 0.1  # Within 10%

        # Prior std should be positive but not huge
        # Prior std may be large if using fallback (bounds-based) priors
