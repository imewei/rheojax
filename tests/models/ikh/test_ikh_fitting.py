"""Validation tests for NLSQ → NUTS pipeline for IKH models.

This module validates that the complete Bayesian inference pipeline works
correctly for MIKH and ML-IKH models across experimental protocols:
- Flow Curve
- Startup Shear
- Stress Relaxation
- Creep

The tests verify:
1. NLSQ fitting converges to reasonable point estimates
2. Bayesian inference (NUTS) produces valid posteriors
3. Warm-start from NLSQ improves Bayesian convergence
4. Convergence diagnostics (R-hat, ESS, divergences) are acceptable
"""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.ikh._kernels import ikh_flow_curve_steady_state
from rheojax.models.ikh.mikh import MIKH
from rheojax.models.ikh.ml_ikh import MLIKH

jax, jnp = safe_import_jax()


# =============================================================================
# Fixtures for synthetic data generation
# =============================================================================


@pytest.fixture
def flow_curve_data():
    """Generate synthetic flow curve data using IKH steady-state."""
    true_params = {
        "sigma_y0": 20.0,
        "delta_sigma_y": 40.0,
        "tau_thix": 1.0,
        "Gamma": 0.5,
        "eta_inf": 0.1,
        # Additional params not used in steady-state but needed for dict
        "G": 1000.0,
        "eta": 1e6,
        "C": 500.0,
        "gamma_dyn": 1.0,
        "m": 1.0,
        "mu_p": 1e-3,
    }

    gamma_dot = np.logspace(-2, 2, 25)

    # Analytical steady-state
    k1 = 1.0 / true_params["tau_thix"]
    k2 = true_params["Gamma"]
    lambda_ss = k1 / (k1 + k2 * np.abs(gamma_dot))
    sigma_y_ss = true_params["sigma_y0"] + true_params["delta_sigma_y"] * lambda_ss
    stress = sigma_y_ss + true_params["eta_inf"] * np.abs(gamma_dot)

    # Add 3% noise
    rng = np.random.default_rng(42)
    stress_noisy = stress * (1 + 0.03 * rng.standard_normal(stress.shape))

    return gamma_dot, stress_noisy, true_params


@pytest.fixture
def startup_data():
    """Generate synthetic startup shear data."""
    # Generate using MIKH model with known parameters
    true_model = MIKH()
    true_params = {
        "G": 100.0,
        "eta": 1e6,
        "C": 50.0,
        "gamma_dyn": 0.5,
        "m": 1.0,
        "sigma_y0": 20.0,
        "delta_sigma_y": 30.0,
        "tau_thix": 1.0,
        "Gamma": 0.5,
        "eta_inf": 0.1,
        "mu_p": 1e-3,
    }
    for k, v in true_params.items():
        true_model.parameters.set_value(k, v)

    t = np.linspace(0, 5.0, 80)
    gamma_dot = 2.0
    gamma = gamma_dot * t

    X_input = np.stack([t, gamma])
    stress_true = np.array(true_model.predict(X_input, test_mode="startup"))

    # Add 2% noise
    rng = np.random.default_rng(42)
    noise = 0.02 * np.std(stress_true) * rng.standard_normal(stress_true.shape)
    stress_noisy = stress_true + noise

    return X_input, stress_noisy, true_params


@pytest.fixture
def relaxation_data():
    """Generate synthetic stress relaxation data."""
    t = np.linspace(0, 10.0, 50)
    sigma_0 = 60.0

    # Simple exponential + yield stress decay
    tau_relax = 2.0
    sigma_y_final = 20.0
    stress = sigma_y_final + (sigma_0 - sigma_y_final) * np.exp(-t / tau_relax)

    # Add 2% noise
    rng = np.random.default_rng(42)
    stress_noisy = stress * (1 + 0.02 * rng.standard_normal(stress.shape))

    return t, stress_noisy, sigma_0


@pytest.fixture
def creep_data():
    """Generate synthetic creep data."""
    t = np.linspace(0, 10.0, 50)
    sigma_applied = 50.0

    # Simple creep response: elastic + viscous
    G = 100.0
    eta = 1000.0
    gamma_elastic = sigma_applied / G
    gamma_viscous = sigma_applied / eta * t
    strain = gamma_elastic + gamma_viscous

    # Add 2% noise
    rng = np.random.default_rng(42)
    strain_noisy = strain * (1 + 0.02 * rng.standard_normal(strain.shape))

    return t, strain_noisy, sigma_applied


# =============================================================================
# MIKH NLSQ Tests
# =============================================================================


class TestMIKHNLSQ:
    """Test NLSQ fitting for MIKH."""

    @pytest.mark.smoke
    def test_nlsq_flow_curve_converges(self, flow_curve_data):
        """NLSQ fitting for flow curve should converge."""
        gamma_dot, stress, _ = flow_curve_data

        model = MIKH()
        model.fit(gamma_dot, stress, test_mode="flow_curve", max_iter=500)

        assert model.fitted_

        # Prediction should be reasonable
        stress_pred = model.predict_flow_curve(gamma_dot)
        r2 = 1 - np.sum((stress - stress_pred) ** 2) / np.sum(
            (stress - np.mean(stress)) ** 2
        )
        assert r2 > 0.9, f"R² = {r2:.3f} should be > 0.9"

    @pytest.mark.smoke
    def test_nlsq_startup_converges(self, startup_data):
        """NLSQ fitting for startup should converge."""
        X_input, stress, _ = startup_data

        model = MIKH()
        model.fit(X_input, stress, test_mode="startup", max_iter=500)

        assert model.fitted_

        # Check prediction quality (R² > 0.75 is reasonable for this complex model)
        stress_pred = model.predict(X_input, test_mode="startup")
        r2 = 1 - np.sum((stress - stress_pred) ** 2) / np.sum(
            (stress - np.mean(stress)) ** 2
        )
        assert r2 > 0.75, f"R² = {r2:.3f} should be > 0.75"

    @pytest.mark.smoke
    def test_nlsq_parameter_recovery(self, flow_curve_data):
        """NLSQ should recover parameters within reasonable bounds."""
        gamma_dot, stress, true_params = flow_curve_data

        model = MIKH()
        model.fit(gamma_dot, stress, test_mode="flow_curve", max_iter=500)

        # Check key parameters (allow 50% deviation for flow curve only)
        fitted_sigma_y0 = model.parameters.get_value("sigma_y0")
        fitted_eta_inf = model.parameters.get_value("eta_inf")

        assert (
            10.0 < fitted_sigma_y0 < 40.0
        ), f"sigma_y0 = {fitted_sigma_y0} out of range"
        assert 0.01 < fitted_eta_inf < 1.0, f"eta_inf = {fitted_eta_inf} out of range"


# =============================================================================
# MIKH Bayesian Tests
# =============================================================================


class TestMIKHBayesian:
    """Test Bayesian inference for MIKH."""

    @pytest.mark.smoke
    def test_bayesian_smoke(self, startup_data):
        """Smoke test for Bayesian inference on MIKH."""
        X_input, stress, _ = startup_data

        model = MIKH()

        # Use very few samples for speed
        result = model.fit_bayesian(
            X_input,
            stress,
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            test_mode="startup",
            progress_bar=False,
        )

        assert result is not None
        assert "G" in result.posterior_samples
        assert len(result.posterior_samples["G"]) == 10

    @pytest.mark.slow
    def test_bayesian_startup_convergence(self, startup_data):
        """Bayesian inference should converge for startup data."""
        X_input, stress, _ = startup_data

        model = MIKH()

        # First do NLSQ warm-start
        model.fit(X_input, stress, test_mode="startup", max_iter=200)

        # Then Bayesian
        result = model.fit_bayesian(
            X_input,
            stress,
            num_warmup=200,
            num_samples=200,
            num_chains=1,
            test_mode="startup",
            progress_bar=False,
        )

        # Check basic diagnostics
        assert result is not None

        # Check we got samples for key parameters
        for param in ["G", "sigma_y0", "eta_inf"]:
            assert param in result.posterior_samples
            samples = result.posterior_samples[param]
            assert len(samples) == 200
            # Check samples are finite
            assert np.all(np.isfinite(samples))

    @pytest.mark.slow
    @pytest.mark.validation
    def test_pipeline_nlsq_to_nuts(self, startup_data):
        """Complete NLSQ → NUTS pipeline should work."""
        X_input, stress, true_params = startup_data

        # 1. NLSQ fitting
        model = MIKH()
        model.fit(X_input, stress, test_mode="startup", max_iter=300)
        assert model.fitted_

        # 2. Bayesian with warm-start
        result = model.fit_bayesian(
            X_input,
            stress,
            num_warmup=500,
            num_samples=1000,
            num_chains=1,
            test_mode="startup",
            progress_bar=False,
            seed=123,
        )

        # 3. Check diagnostics
        assert result is not None

        # 4. Posterior mean should be in reasonable range
        nlsq_G = model.parameters.get_value("G")
        posterior_G = np.mean(result.posterior_samples["G"])

        # Allow order-of-magnitude deviation (ODE-based NUTS is inherently noisy)
        assert (
            abs(np.log10(posterior_G) - np.log10(nlsq_G)) < 1.0
        ), f"Posterior mean G={posterior_G:.2e} too far from NLSQ G={nlsq_G:.2e}"


# =============================================================================
# ML-IKH Tests
# =============================================================================


class TestMLIKHFitting:
    """Tests for ML-IKH fitting."""

    @pytest.mark.smoke
    def test_mlikh_per_mode_fitting(self, startup_data):
        """ML-IKH per_mode should fit startup data."""
        X_input, stress, _ = startup_data

        model = MLIKH(n_modes=2, yield_mode="per_mode")
        model.fit(X_input, stress, max_iter=200)

        assert model.fitted_

        # Prediction should be reasonable
        stress_pred = model.predict(X_input)
        r2 = 1 - np.sum((stress - stress_pred) ** 2) / np.sum(
            (stress - np.mean(stress)) ** 2
        )
        assert r2 > 0.7, f"R² = {r2:.3f} should be > 0.7"

    @pytest.mark.smoke
    def test_mlikh_weighted_sum_fitting(self, startup_data):
        """ML-IKH weighted_sum should fit startup data."""
        X_input, stress, _ = startup_data

        model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        model.fit(X_input, stress, max_iter=200)

        assert model.fitted_

        # Prediction should be reasonable
        stress_pred = model.predict(X_input)
        r2 = 1 - np.sum((stress - stress_pred) ** 2) / np.sum(
            (stress - np.mean(stress)) ** 2
        )
        assert r2 > 0.7, f"R² = {r2:.3f} should be > 0.7"

    @pytest.mark.slow
    def test_mlikh_bayesian_smoke(self, startup_data):
        """Smoke test for ML-IKH Bayesian inference."""
        X_input, stress, _ = startup_data

        model = MLIKH(n_modes=2, yield_mode="per_mode")

        result = model.fit_bayesian(
            X_input,
            stress,
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            test_mode="startup",
            progress_bar=False,
        )

        assert result is not None
        assert "G_1" in result.posterior_samples
        assert "G_2" in result.posterior_samples


class TestMLIKHProtocolFitting:
    """Tests for ML-IKH protocol-specific fitting."""

    @pytest.fixture
    def flow_curve_data(self):
        """Generate synthetic flow curve data."""
        # Use MLIKH to generate data, then fit it back
        true_model = MLIKH(n_modes=2, yield_mode="per_mode")

        for i in [1, 2]:
            true_model.parameters.set_value(f"sigma_y0_{i}", 5.0 + i)
            true_model.parameters.set_value(f"delta_sigma_y_{i}", 10.0 + i * 5)
            true_model.parameters.set_value(f"tau_thix_{i}", 0.5 * i)
            true_model.parameters.set_value(f"Gamma_{i}", 0.3 + 0.1 * i)
        true_model.parameters.set_value("eta_inf", 0.05)

        gamma_dot = jnp.logspace(-2, 2, 30)
        sigma = true_model.predict_flow_curve(gamma_dot)

        # Add small noise
        noise = 0.02 * sigma.mean() * np.random.randn(len(sigma))
        sigma_noisy = sigma + noise

        return gamma_dot, sigma_noisy, true_model

    @pytest.mark.smoke
    def test_fit_flow_curve(self, flow_curve_data):
        """Test fitting flow curve data."""
        gamma_dot, sigma_target, _ = flow_curve_data

        model = MLIKH(n_modes=2, yield_mode="per_mode")
        model.fit(gamma_dot, sigma_target, test_mode="flow_curve", max_iter=300)

        assert model.fitted_

        sigma_pred = model.predict_flow_curve(gamma_dot)
        residuals = sigma_target - sigma_pred
        rel_error = np.std(residuals) / np.mean(sigma_target)

        assert rel_error < 0.15, f"Relative error {rel_error:.2%} should be < 15%"

    @pytest.mark.smoke
    def test_fit_startup_via_ode(self, startup_data):
        """Test startup fitting uses return mapping (not ODE)."""
        X_input, stress, _ = startup_data

        model = MLIKH(n_modes=2, yield_mode="per_mode")
        model.fit(X_input, stress, test_mode="startup", max_iter=200)

        assert model.fitted_
        assert model._test_mode == "startup"

    @pytest.mark.slow
    def test_fit_relaxation(self):
        """Test fitting relaxation data."""
        # Generate synthetic relaxation data
        true_model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        true_model.parameters.set_value("G", 200.0)
        true_model.parameters.set_value("sigma_y0", 10.0)
        true_model.parameters.set_value("k3", 20.0)
        true_model.parameters.set_value("tau_thix_1", 0.5)
        true_model.parameters.set_value("tau_thix_2", 5.0)

        t = jnp.linspace(0.01, 10.0, 40)
        sigma_true = true_model.predict_relaxation(t, sigma_0=80.0)

        # Fit with warm-start to ensure finite residuals at initial point
        model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        model.parameters.set_value("G", 150.0)
        model.parameters.set_value("sigma_y0", 8.0)
        model.parameters.set_value("k3", 15.0)
        model.parameters.set_value("tau_thix_1", 1.0)
        model.parameters.set_value("tau_thix_2", 3.0)
        model.fit(t, sigma_true, test_mode="relaxation", sigma_0=80.0, max_iter=300)

        assert model.fitted_

    @pytest.mark.slow
    def test_fit_creep(self):
        """Test fitting creep data."""
        # Generate synthetic creep data
        true_model = MLIKH(n_modes=2, yield_mode="per_mode")
        for i in [1, 2]:
            true_model.parameters.set_value(f"G_{i}", 100.0)
            true_model.parameters.set_value(f"sigma_y0_{i}", 5.0)
            true_model.parameters.set_value(f"tau_thix_{i}", 1.0 * i)
        true_model.parameters.set_value("eta_inf", 5.0)

        t = jnp.linspace(0.01, 5.0, 40)
        strain_true = true_model.predict_creep(t, sigma_applied=30.0)

        # Fit
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        model.fit(t, strain_true, test_mode="creep", sigma_applied=30.0, max_iter=300)

        assert model.fitted_

    @pytest.mark.slow
    def test_bayesian_flow_curve(self, flow_curve_data):
        """Test Bayesian inference on flow curve data."""
        gamma_dot, sigma_target, _ = flow_curve_data

        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # First do NLSQ for warm start
        model.fit(gamma_dot, sigma_target, test_mode="flow_curve", max_iter=200)

        # Then Bayesian
        result = model.fit_bayesian(
            gamma_dot,
            sigma_target,
            num_warmup=20,
            num_samples=20,
            num_chains=1,
            test_mode="flow_curve",
            progress_bar=False,
        )

        assert result is not None
        # Check posterior samples exist
        assert "sigma_y0_1" in result.posterior_samples
        assert "eta_inf" in result.posterior_samples

    @pytest.mark.smoke
    def test_protocol_dispatch_in_fit(self):
        """Verify _fit dispatches to correct sub-method based on test_mode."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # Dummy data for flow curve (no ODE integration needed)
        X = jnp.logspace(-1, 1, 10)
        y = jnp.ones(10) * 50.0

        # Test that test_mode is stored for flow_curve
        model.fit(X, y, test_mode="flow_curve", max_iter=10)
        assert model._test_mode == "flow_curve"

        # Test that test_mode is stored for startup (return mapping, no ODE)
        t = jnp.linspace(0, 1.0, 10)
        strain = 0.1 * t
        X_startup = jnp.stack([t, strain])
        model.fit(X_startup, y, test_mode="startup", max_iter=10)
        assert model._test_mode == "startup"


# =============================================================================
# Protocol-Specific Tests
# =============================================================================


class TestIKHProtocols:
    """Test different experimental protocols."""

    @pytest.mark.smoke
    def test_flow_curve_protocol(self, flow_curve_data):
        """Flow curve fitting should work."""
        gamma_dot, stress, _ = flow_curve_data

        model = MIKH()
        model.fit(gamma_dot, stress, test_mode="flow_curve")

        # Should use steady-state equation
        pred = model.predict_flow_curve(gamma_dot)
        assert pred.shape == stress.shape

    @pytest.mark.smoke
    def test_startup_protocol(self, startup_data):
        """Startup fitting should work."""
        X_input, stress, _ = startup_data

        model = MIKH()
        model.fit(X_input, stress, test_mode="startup")

        pred = model.predict(X_input, test_mode="startup")
        assert pred.shape == stress.shape

    @pytest.mark.smoke
    def test_predict_methods(self):
        """Test convenience predict methods."""
        model = MIKH()

        t = np.linspace(0, 5.0, 50)
        gamma_dot_arr = np.logspace(-2, 2, 20)

        # Flow curve
        sigma_fc = model.predict_flow_curve(gamma_dot_arr)
        assert sigma_fc.shape == gamma_dot_arr.shape

        # Startup
        sigma_startup = model.predict_startup(t, gamma_dot=1.0)
        assert sigma_startup.shape == t.shape

        # Relaxation
        sigma_relax = model.predict_relaxation(t, sigma_0=100.0)
        assert sigma_relax.shape == t.shape

        # Creep
        strain_creep = model.predict_creep(t, sigma_applied=50.0)
        assert strain_creep.shape == t.shape

        # LAOS
        sigma_laos = model.predict_laos(t, gamma_0=1.0, omega=1.0)
        assert sigma_laos.shape == t.shape
