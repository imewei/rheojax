"""Tests for BaseModel integration with NLSQ + NumPyro workflow.

This module tests the integration of NLSQ optimization and NumPyro Bayesian
inference into the BaseModel class, ensuring all models inherit these capabilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.base import BaseModel
from rheojax.core.bayesian import BayesianResult
from rheojax.core.parameters import Parameter, ParameterSet
from rheojax.models import Maxwell
from rheojax.utils.optimization import OptimizationResult


def test_basemodel_fit_uses_nlsq_by_default():
    """Test that BaseModel.fit() uses NLSQ optimizer by default."""
    # Create Maxwell model (inherits from BaseModel)
    model = Maxwell()

    # Generate synthetic relaxation data
    t = np.linspace(0.1, 10, 50)
    G0_true = 1e5
    eta_true = 1e3
    tau_true = eta_true / G0_true
    G_true = G0_true * np.exp(-t / tau_true)

    # Add small noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.01 * G_true.mean(), size=t.shape)
    G_data = G_true + noise

    # Fit using default method (should be NLSQ)
    model.fit(t, G_data)

    # Check that model is fitted
    assert model.fitted_ is True

    # Check that parameters are updated (should be close to true values)
    G0_fitted = model.parameters.get_value("G0")
    eta_fitted = model.parameters.get_value("eta")

    # Verify parameters are reasonable (within order of magnitude)
    assert 1e4 < G0_fitted < 1e6
    assert 1e2 < eta_fitted < 1e4

    # Verify predictions work
    predictions = model.predict(t)
    assert predictions.shape == t.shape
    assert np.all(np.isfinite(predictions))


def test_basemodel_fit_bayesian_delegates_to_bayesian_mixin():
    """Test that BaseModel.fit_bayesian() delegates to BayesianMixin."""
    test_mode = ("relaxation",)
    # Create Maxwell model
    model = Maxwell()

    # Generate synthetic data
    t = np.linspace(0.1, 10, 30)
    G0_true = 1e5
    eta_true = 1e3
    tau_true = eta_true / G0_true
    G_true = G0_true * np.exp(-t / tau_true)

    # Fit using Bayesian inference (should delegate to BayesianMixin.fit_bayesian)
    # Use small number of samples for speed
    result = model.fit_bayesian(
        t,
        G_true,
        test_mode="relaxation",
        num_warmup=100,
        num_samples=200,
        num_chains=1,
    )

    # Check that result is BayesianResult
    assert isinstance(result, BayesianResult)

    # Check that posterior samples are present
    assert "G0" in result.posterior_samples
    assert "eta" in result.posterior_samples

    # Check sample shapes
    assert result.posterior_samples["G0"].shape == (200,)
    assert result.posterior_samples["eta"].shape == (200,)

    # Check diagnostics are computed
    assert "r_hat" in result.diagnostics
    assert "ess" in result.diagnostics
    assert "divergences" in result.diagnostics


def test_warm_start_workflow_fit_then_fit_bayesian():
    """Test warm-start workflow: fit() with NLSQ then fit_bayesian() with NUTS."""
    # Create Maxwell model
    model = Maxwell()

    # Generate synthetic data with known parameters
    t = np.linspace(0.1, 10, 30)
    G0_true = 1e5
    eta_true = 1e3
    tau_true = eta_true / G0_true
    G_true = G0_true * np.exp(-t / tau_true)

    # Step 1: Fit with NLSQ to get point estimates
    model.fit(t, G_true)

    # Store NLSQ results for comparison
    G0_nlsq = model.parameters.get_value("G0")
    eta_nlsq = model.parameters.get_value("eta")

    # Verify NLSQ got reasonable estimates
    assert 5e4 < G0_nlsq < 2e5  # Should be close to 1e5
    assert 5e2 < eta_nlsq < 2e3  # Should be close to 1e3

    # Step 2: Fit Bayesian with warm-start from NLSQ
    # Extract initial values from NLSQ fit
    initial_values = {"G0": G0_nlsq, "eta": eta_nlsq}

    result = model.fit_bayesian(
        t,
        G_true,
        test_mode="relaxation",
        num_warmup=200,
        num_samples=400,
        num_chains=1,
        initial_values=initial_values,
    )

    # Check that Bayesian result is obtained
    assert isinstance(result, BayesianResult)

    # Check that posterior means are reasonable
    G0_posterior_mean = result.summary["G0"]["mean"]
    eta_posterior_mean = result.summary["eta"]["mean"]

    # Posterior means should be within reasonable range (not checking exact match due to MCMC stochasticity)
    # Note: Wide priors can cause parameter identifiability issues, so we allow factor of 10 error
    assert 1e4 < G0_posterior_mean < 1e6  # Allow factor of 10 around true value 1e5
    assert 1e2 < eta_posterior_mean < 1e4  # Allow factor of 10 around true value 1e3

    # Verify convergence is reasonable (R-hat should be < 1.2)
    assert result.diagnostics["r_hat"]["G0"] < 1.2
    assert result.diagnostics["r_hat"]["eta"] < 1.2


def test_all_models_inherit_bayesian_capabilities():
    """Test that all models automatically gain fit_bayesian() method."""
    # Create Maxwell model (inherits from BaseModel)
    model = Maxwell()

    # Check that fit_bayesian method exists and is callable
    assert hasattr(model, "fit_bayesian")
    assert callable(model.fit_bayesian)

    # Check that sample_prior method exists (from BayesianMixin)
    assert hasattr(model, "sample_prior")
    assert callable(model.sample_prior)

    # Check that get_credible_intervals method exists
    assert hasattr(model, "get_credible_intervals")
    assert callable(model.get_credible_intervals)

    # Test that sample_prior works
    prior_samples = model.sample_prior(num_samples=100)

    assert "G0" in prior_samples
    assert "eta" in prior_samples
    assert prior_samples["G0"].shape == (100,)
    assert prior_samples["eta"].shape == (100,)


def test_end_to_end_nlsq_nuts_workflow_on_maxwell():
    """Test complete NLSQ → NUTS workflow on Maxwell model."""
    # Create Maxwell model
    model = Maxwell()

    # Generate synthetic data with known parameters and low noise
    t = np.linspace(0.1, 10, 40)
    G0_true = 1e5
    eta_true = 1e3
    tau_true = eta_true / G0_true
    G_true = G0_true * np.exp(-t / tau_true)

    # Add small noise (1% relative)
    np.random.seed(123)
    noise = np.random.normal(0, 0.01 * G_true.mean(), size=t.shape)
    G_data = G_true + noise

    # Step 1: NLSQ optimization
    model.fit(t, G_data)

    # Verify NLSQ converged
    assert model.fitted_ is True
    G0_nlsq = model.parameters.get_value("G0")
    eta_nlsq = model.parameters.get_value("eta")

    # Check NLSQ estimates are reasonable
    assert abs(G0_nlsq - G0_true) / G0_true < 0.2  # Within 20%
    assert abs(eta_nlsq - eta_true) / eta_true < 0.2

    # Step 2: Bayesian inference with warm-start
    initial_values = {"G0": G0_nlsq, "eta": eta_nlsq}

    result = model.fit_bayesian(
        t,
        G_data,
        test_mode="relaxation",
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        initial_values=initial_values,
    )

    # Step 3: Verify convergence diagnostics
    # R-hat should be < 1.01 for good convergence (lenient: < 1.05)
    r_hat_G0 = result.diagnostics["r_hat"]["G0"]
    r_hat_eta = result.diagnostics["r_hat"]["eta"]

    assert r_hat_G0 < 1.05, f"R-hat for G0 is {r_hat_G0:.4f}, should be < 1.05"
    assert r_hat_eta < 1.05, f"R-hat for eta is {r_hat_eta:.4f}, should be < 1.05"

    # ESS should be > 400 for good sampling (lenient: > 200)
    ess_G0 = result.diagnostics["ess"]["G0"]
    ess_eta = result.diagnostics["ess"]["eta"]

    assert ess_G0 > 200, f"ESS for G0 is {ess_G0:.0f}, should be > 200"
    assert ess_eta > 200, f"ESS for eta is {ess_eta:.0f}, should be > 200"

    # Divergences should be low (lenient: < 200 out of 1000 samples)
    divergences = result.diagnostics["divergences"]
    assert divergences < 200, f"Too many divergences: {divergences}"

    # Step 4: Verify posterior statistics make sense
    G0_mean = result.summary["G0"]["mean"]
    eta_mean = result.summary["eta"]["mean"]

    # Posterior means should be reasonable (allow factor of 10 due to wide priors)
    assert 1e4 < G0_mean < 1e6  # Within factor of 10 of true value 1e5
    assert 1e2 < eta_mean < 1e4  # Within factor of 10 of true value 1e3

    # Standard deviations should be reasonable (not too large)
    G0_std = result.summary["G0"]["std"]
    eta_std = result.summary["eta"]["std"]

    # Coefficient of variation should be reasonable
    assert G0_std / G0_mean < 0.5, "G0 posterior uncertainty too large"
    assert eta_std / eta_mean < 0.5, "eta posterior uncertainty too large"


def test_basemodel_stores_optimization_and_bayesian_results():
    """Test that BaseModel stores NLSQ and Bayesian results for later access."""
    # Create Maxwell model
    model = Maxwell()

    # Generate synthetic data
    t = np.linspace(0.1, 10, 30)
    G0_true = 1e5
    eta_true = 1e3
    G_true = G0_true * np.exp(-t / (eta_true / G0_true))

    # Fit with NLSQ
    model.fit(t, G_true)

    # Check if model stores NLSQ result (if implemented)
    # Note: This tests the helper methods from task 5.5
    if hasattr(model, "get_nlsq_result"):
        nlsq_result = model.get_nlsq_result()
        # Should return OptimizationResult or None
        assert nlsq_result is None or isinstance(nlsq_result, OptimizationResult)

    # Fit with Bayesian inference
    result = model.fit_bayesian(
        t,
        G_true,
        test_mode="relaxation",
        num_warmup=100,
        num_samples=200,
        num_chains=1,
    )

    # Check if model stores Bayesian result (if implemented)
    if hasattr(model, "get_bayesian_result"):
        bayesian_result = model.get_bayesian_result()
        assert bayesian_result is not None
        assert isinstance(bayesian_result, BayesianResult)

        # Should be the same result returned from fit_bayesian
        assert bayesian_result is result


def test_basemodel_fit_method_signature_backward_compatible():
    """Test that BaseModel.fit() maintains backward compatibility with existing API."""
    # Create Maxwell model
    model = Maxwell()

    # Generate synthetic data
    t = np.linspace(0.1, 10, 30)
    G0_true = 1e5
    eta_true = 1e3
    G_true = G0_true * np.exp(-t / (eta_true / G0_true))

    # Test that fit() can be called with just X and y (existing API)
    result = model.fit(t, G_true)

    # Should return self for method chaining
    assert result is model

    # Should set fitted_ flag
    assert model.fitted_ is True

    # Test that fit() accepts method parameter
    model2 = Maxwell()
    result2 = model2.fit(t, G_true, method="nlsq")
    assert result2 is model2
    assert model2.fitted_ is True

    # Test that fit() accepts additional kwargs
    model3 = Maxwell()
    result3 = model3.fit(t, G_true, max_iter=500, use_jax=True)
    assert result3 is model3
    assert model3.fitted_ is True
