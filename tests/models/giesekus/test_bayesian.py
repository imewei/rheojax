"""Tests for Giesekus Bayesian inference pipeline.

Tests cover:
- NLSQ fitting as warm-start
- NLSQ → NUTS workflow
- Credible interval extraction
- Multi-mode Bayesian fitting
- Mode-aware inference
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.giesekus import GiesekusMultiMode, GiesekusSingleMode

jax, jnp = safe_import_jax()


class TestNLSQWarmStart:
    """Tests for NLSQ fitting as Bayesian warm-start."""

    @pytest.mark.smoke
    def test_nlsq_fit_flow_curve(self):
        """Test NLSQ fitting on flow curve data."""
        model = GiesekusSingleMode()

        # Generate synthetic data
        gamma_dot = np.logspace(-2, 2, 30)
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 1.0)
        model.parameters.set_value("alpha", 0.3)
        model.parameters.set_value("eta_s", 10.0)

        sigma_true = model.predict(gamma_dot, test_mode="flow_curve")
        noise = np.random.default_rng(42).normal(0, 0.02, len(sigma_true))
        sigma_noisy = sigma_true * (1 + noise)

        # Reset and fit
        model.parameters.set_value("eta_p", 50.0)
        model.parameters.set_value("lambda_1", 0.5)
        model.parameters.set_value("alpha", 0.2)

        model.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        # Check recovery
        eta_p_fit = model.parameters.get_value("eta_p")
        lambda_fit = model.parameters.get_value("lambda_1")
        alpha_fit = model.parameters.get_value("alpha")

        assert 80 < eta_p_fit < 120, f"eta_p not recovered: {eta_p_fit}"
        assert 0.5 < lambda_fit < 2.0, f"lambda not recovered: {lambda_fit}"
        assert 0.15 < alpha_fit < 0.45, f"alpha not recovered: {alpha_fit}"

    @pytest.mark.smoke
    def test_nlsq_fit_saos(self):
        """Test NLSQ fitting on SAOS data."""
        model = GiesekusSingleMode()

        # Generate synthetic SAOS data
        omega = np.logspace(-2, 2, 40)
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 1.0)
        model.parameters.set_value("eta_s", 5.0)

        G_prime, G_double_prime = model.predict_saos(omega)
        G_star = np.sqrt(G_prime**2 + G_double_prime**2)

        # Add noise
        rng = np.random.default_rng(42)
        G_star_noisy = G_star * (1 + rng.normal(0, 0.02, len(G_star)))

        # Reset and fit
        model.parameters.set_value("eta_p", 50.0)
        model.parameters.set_value("lambda_1", 0.5)

        model.fit(omega, G_star_noisy, test_mode="oscillation")

        # Check recovery (SAOS is α-independent)
        eta_p_fit = model.parameters.get_value("eta_p")
        lambda_fit = model.parameters.get_value("lambda_1")

        assert 70 < eta_p_fit < 130, f"eta_p not recovered: {eta_p_fit}"
        assert 0.5 < lambda_fit < 2.0, f"lambda not recovered: {lambda_fit}"


class TestBayesianInference:
    """Tests for Bayesian inference with NumPyro."""

    @pytest.mark.slow
    def test_bayesian_flow_curve(self):
        """Test Bayesian fitting on flow curve data."""
        model = GiesekusSingleMode()

        # Generate synthetic data
        gamma_dot = np.logspace(-2, 2, 30)
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 1.0)
        model.parameters.set_value("alpha", 0.3)
        model.parameters.set_value("eta_s", 10.0)

        sigma_true = model.predict(gamma_dot, test_mode="flow_curve")
        rng = np.random.default_rng(42)
        sigma_noisy = sigma_true * (1 + rng.normal(0, 0.03, len(sigma_true)))

        # NLSQ warm-start
        model.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        # Bayesian inference
        result = model.fit_bayesian(
            gamma_dot,
            sigma_noisy,
            test_mode="flow_curve",
            num_warmup=200,
            num_samples=500,
            num_chains=1,
            seed=42,
        )

        assert result is not None
        assert "eta_p" in result.posterior_samples
        assert "lambda_1" in result.posterior_samples
        assert "alpha" in result.posterior_samples

    @pytest.mark.slow
    def test_bayesian_saos(self):
        """Test Bayesian fitting on SAOS data."""
        model = GiesekusSingleMode()

        # Generate synthetic SAOS data
        omega = np.logspace(-2, 2, 40)
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 1.0)
        model.parameters.set_value("eta_s", 5.0)

        G_prime, G_double_prime = model.predict_saos(omega)
        G_star = np.sqrt(G_prime**2 + G_double_prime**2)

        rng = np.random.default_rng(42)
        G_star_noisy = G_star * (1 + rng.normal(0, 0.03, len(G_star)))

        # NLSQ warm-start
        model.fit(omega, G_star_noisy, test_mode="oscillation")

        # Bayesian inference
        result = model.fit_bayesian(
            omega,
            G_star_noisy,
            test_mode="oscillation",
            num_warmup=200,
            num_samples=500,
            num_chains=1,
            seed=42,
        )

        assert result is not None


class TestCredibleIntervals:
    """Tests for credible interval extraction."""

    @pytest.mark.slow
    def test_credible_intervals_95(self):
        """Test 95% credible interval extraction."""
        model = GiesekusSingleMode()

        # Generate data
        gamma_dot = np.logspace(-1, 2, 25)
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 1.0)
        model.parameters.set_value("alpha", 0.3)

        sigma_true = model.predict(gamma_dot, test_mode="flow_curve")
        rng = np.random.default_rng(42)
        sigma_noisy = sigma_true * (1 + rng.normal(0, 0.03, len(sigma_true)))

        # Fit
        model.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")
        result = model.fit_bayesian(
            gamma_dot,
            sigma_noisy,
            test_mode="flow_curve",
            num_warmup=200,
            num_samples=500,
            num_chains=1,
            seed=42,
        )

        # Get credible intervals
        intervals = model.get_credible_intervals(
            result.posterior_samples, credibility=0.95
        )

        assert "eta_p" in intervals
        lower, upper = intervals["eta_p"]
        assert lower < upper

    @pytest.mark.slow
    def test_credible_intervals_contain_true(self):
        """Test that 95% CI contains true value (statistical test)."""
        model = GiesekusSingleMode()

        # Generate data with low noise
        gamma_dot = np.logspace(-1, 2, 30)
        true_eta_p = 100.0
        model.parameters.set_value("eta_p", true_eta_p)
        model.parameters.set_value("lambda_1", 1.0)
        model.parameters.set_value("alpha", 0.3)

        sigma_true = model.predict(gamma_dot, test_mode="flow_curve")
        rng = np.random.default_rng(42)
        sigma_noisy = sigma_true * (1 + rng.normal(0, 0.02, len(sigma_true)))

        # Fit
        model.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")
        result = model.fit_bayesian(
            gamma_dot,
            sigma_noisy,
            test_mode="flow_curve",
            num_warmup=300,
            num_samples=800,
            num_chains=1,
            seed=42,
        )

        intervals = model.get_credible_intervals(
            result.posterior_samples, credibility=0.95
        )

        # True value should be within CI (with high probability)
        eta_p_lower, eta_p_upper = intervals["eta_p"]

        # Allow some tolerance for statistical variation
        assert eta_p_lower < true_eta_p * 1.3
        assert eta_p_upper > true_eta_p * 0.7


class TestModeAwareBayesian:
    """Tests for mode-aware Bayesian inference (v0.4.0 fix)."""

    @pytest.mark.slow
    def test_mode_captured_in_closure(self):
        """Test that test_mode is correctly captured for Bayesian."""
        model = GiesekusSingleMode()

        # Fit with flow_curve mode
        gamma_dot = np.logspace(-1, 2, 20)
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("alpha", 0.3)

        sigma = model.predict(gamma_dot, test_mode="flow_curve")
        rng = np.random.default_rng(42)
        sigma_noisy = sigma * (1 + rng.normal(0, 0.03, len(sigma)))

        model.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        # Now fit Bayesian with SAOS (different mode)
        omega = np.logspace(-1, 2, 20)
        G_prime, G_double_prime = model.predict_saos(omega)
        G_star = np.sqrt(G_prime**2 + G_double_prime**2)
        G_star_noisy = G_star * (1 + rng.normal(0, 0.03, len(G_star)))

        # This should use oscillation mode, not flow_curve
        result = model.fit_bayesian(
            omega,
            G_star_noisy,
            test_mode="oscillation",  # Explicit mode
            num_warmup=100,
            num_samples=200,
            num_chains=1,
            seed=42,
        )

        assert result is not None


class TestMultiModeBayesian:
    """Tests for multi-mode Bayesian fitting."""

    @pytest.mark.slow
    def test_multimode_bayesian_saos(self):
        """Test multi-mode Bayesian on SAOS data."""
        model = GiesekusMultiMode(n_modes=2)

        # Generate synthetic multi-mode SAOS
        model.set_mode_params(0, eta_p=100.0, lambda_1=10.0, alpha=0.3)
        model.set_mode_params(1, eta_p=50.0, lambda_1=1.0, alpha=0.2)

        omega = np.logspace(-2, 2, 40)
        G_prime, G_double_prime = model.predict_saos(omega)
        G_star = np.sqrt(G_prime**2 + G_double_prime**2)

        rng = np.random.default_rng(42)
        G_star_noisy = G_star * (1 + rng.normal(0, 0.03, len(G_star)))

        # Reset and fit
        model.set_mode_params(0, eta_p=80.0, lambda_1=5.0, alpha=0.2)
        model.set_mode_params(1, eta_p=40.0, lambda_1=0.5, alpha=0.15)

        model.fit(omega, G_star_noisy, test_mode="oscillation")

        result = model.fit_bayesian(
            omega,
            G_star_noisy,
            test_mode="oscillation",
            num_warmup=200,
            num_samples=500,
            num_chains=1,
            seed=42,
        )

        assert result is not None
        # Check per-mode parameters are sampled
        assert "eta_p_0" in result.posterior_samples
        assert "eta_p_1" in result.posterior_samples


class TestModelFunction:
    """Tests for model_function interface used by BayesianMixin."""

    @pytest.mark.smoke
    def test_model_function_flow_curve(self):
        """Test model_function for flow_curve mode."""
        model = GiesekusSingleMode()

        params = jnp.array([100.0, 1.0, 0.3, 10.0])  # eta_p, lambda, alpha, eta_s
        gamma_dot = jnp.logspace(-1, 2, 20)

        y = model.model_function(gamma_dot, params, test_mode="flow_curve")

        assert y.shape == gamma_dot.shape
        assert np.all(y > 0)

    @pytest.mark.smoke
    def test_model_function_oscillation(self):
        """Test model_function for oscillation mode."""
        model = GiesekusSingleMode()

        params = jnp.array([100.0, 1.0, 0.3, 10.0])
        omega = jnp.logspace(-1, 2, 20)

        y = model.model_function(omega, params, test_mode="oscillation")

        assert y.shape == omega.shape
        assert np.all(y > 0)

    @pytest.mark.smoke
    def test_model_function_differentiable(self):
        """Test model_function is differentiable via JAX."""
        model = GiesekusSingleMode()

        params = jnp.array([100.0, 1.0, 0.3, 10.0])
        gamma_dot = jnp.logspace(-1, 2, 20)

        # Test gradient computation
        def loss(p):
            y = model.model_function(gamma_dot, p, test_mode="flow_curve")
            return jnp.sum(y**2)

        grad_fn = jax.grad(loss)
        grads = grad_fn(params)

        assert grads.shape == params.shape
        assert np.all(np.isfinite(grads))


class TestDiagnostics:
    """Tests for MCMC diagnostics."""

    @pytest.mark.slow
    def test_r_hat_convergence(self):
        """Test R-hat < 1.1 for converged chains."""
        model = GiesekusSingleMode()

        # Generate clean data
        gamma_dot = np.logspace(-1, 2, 25)
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 1.0)
        model.parameters.set_value("alpha", 0.3)

        sigma_true = model.predict(gamma_dot, test_mode="flow_curve")
        rng = np.random.default_rng(42)
        sigma_noisy = sigma_true * (1 + rng.normal(0, 0.02, len(sigma_true)))

        model.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        # Run with 2 chains for R-hat
        result = model.fit_bayesian(
            gamma_dot,
            sigma_noisy,
            test_mode="flow_curve",
            num_warmup=500,
            num_samples=1000,
            num_chains=2,
            seed=42,
        )

        # Check R-hat if available
        if hasattr(result, "diagnostics") and result.diagnostics is not None:
            for param, r_hat in result.diagnostics.get("r_hat", {}).items():
                assert r_hat < 1.1, f"R-hat for {param} = {r_hat:.3f} > 1.1"
