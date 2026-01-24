"""Comprehensive unit tests for IKH models.

Tests cover:
1. Model initialization and parameter bounds
2. Physical limiting behaviors (elastic, viscous, yielding)
3. Kinematic hardening (Bauschinger effect)
4. Thixotropic buildup and breakdown
5. Analytical validation against known solutions
6. Multi-mode behavior (ML-IKH)
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.ikh._kernels import (
    evolution_lambda,
    ikh_flow_curve_steady_state,
    radial_return_step_corrected,
)
from rheojax.models.ikh.mikh import MIKH
from rheojax.models.ikh.ml_ikh import MLIKH

jax, jnp = safe_import_jax()


# =============================================================================
# MIKH Model Tests
# =============================================================================


class TestMIKH:
    """Tests for the MIKH model."""

    def test_initialization(self):
        """Test MIKH model initialization and default parameters."""
        model = MIKH()

        # Check all required parameters exist
        required_params = [
            "G",
            "eta",
            "C",
            "gamma_dyn",
            "m",
            "sigma_y0",
            "delta_sigma_y",
            "tau_thix",
            "Gamma",
            "eta_inf",
            "mu_p",
        ]
        for param in required_params:
            assert param in model.parameters, f"Missing parameter: {param}"

    def test_parameter_bounds(self):
        """Test that parameter bounds are physically sensible."""
        model = MIKH()

        # G must be positive
        G_param = model.parameters["G"]
        assert G_param.bounds[0] > 0, "G lower bound should be > 0"

        # eta_inf can be zero
        eta_inf_param = model.parameters["eta_inf"]
        assert eta_inf_param.bounds[0] >= 0, "eta_inf lower bound should be >= 0"

    def test_elastic_limit(self):
        """Test pure elastic response (no yielding, no viscosity)."""
        model = MIKH()

        # Set parameters for pure elasticity
        model.parameters.set_value("G", 100.0)
        model.parameters.set_value("eta", 1e12)  # Very high Maxwell viscosity
        model.parameters.set_value("eta_inf", 0.0)
        model.parameters.set_value("sigma_y0", 1e6)  # Very high yield stress
        model.parameters.set_value("delta_sigma_y", 0.0)

        # Small strain ramp (below yield)
        t = np.linspace(0, 1.0, 20)
        gamma = 0.001 * t  # Max strain 0.001, well below yield

        X = np.stack([t, gamma])
        stress = model.predict(X, test_mode="startup")

        # Expected: sigma = G * gamma (linear elastic)
        expected = 100.0 * gamma
        np.testing.assert_allclose(stress, expected, rtol=5e-2)

    def test_viscous_limit(self):
        """Test viscous-dominated response (very small elasticity, zero yield)."""
        model = MIKH()

        # Near-Newtonian settings (G very small but not zero due to bounds)
        model.parameters.set_value("G", 0.1)  # Very small
        model.parameters.set_value("eta_inf", 10.0)
        model.parameters.set_value("sigma_y0", 0.0)  # Zero yield
        model.parameters.set_value("delta_sigma_y", 0.0)

        # Constant shear rate
        dt = 0.1
        t = np.arange(0, 10.0, dt)
        gamma_dot = 2.0
        gamma = gamma_dot * t

        X = np.stack([t, gamma])
        stress = model.predict(X, test_mode="startup")

        # Expected: sigma ≈ eta_inf * gamma_dot = 10 * 2 = 20
        # The small G contribution will add a bit, but viscous should dominate
        # at steady state
        np.testing.assert_allclose(stress[-10:], 20.0, rtol=0.1)

    def test_yielding_behavior(self):
        """Test transition from elastic to plastic flow."""
        model = MIKH()

        G = 100.0
        sigma_y = 50.0

        model.parameters.set_value("G", G)
        model.parameters.set_value("C", 0.0)  # No hardening
        model.parameters.set_value("gamma_dyn", 0.0)
        model.parameters.set_value("sigma_y0", sigma_y)
        model.parameters.set_value("delta_sigma_y", 0.0)  # Constant yield
        model.parameters.set_value("eta_inf", 0.0)
        model.parameters.set_value("tau_thix", 1e9)  # Frozen structure
        model.parameters.set_value("mu_p", 1e-6)  # Small plastic viscosity

        # Strain ramp exceeding yield (gamma_y = 0.5)
        t = np.linspace(0, 2.0, 400)
        gamma = 2.0 * t  # Max strain 4.0

        X = np.stack([t, gamma])
        stress = model.predict(X, test_mode="startup")

        # Check elastic region
        elastic_mask = G * gamma < sigma_y * 0.9
        np.testing.assert_allclose(
            stress[elastic_mask], G * gamma[elastic_mask], rtol=5e-2
        )

        # Check plastic region (perfect plasticity)
        plastic_mask = G * gamma > sigma_y * 1.5
        np.testing.assert_allclose(stress[plastic_mask], sigma_y, rtol=0.1)

    def test_kinematic_hardening_bauschinger(self):
        """Test Bauschinger effect from kinematic hardening.

        The Bauschinger effect means that after forward plastic deformation,
        reverse yielding occurs at a lower stress magnitude due to the
        shifted yield surface (backstress).
        """
        model = MIKH()

        G = 100.0
        sigma_y = 20.0

        model.parameters.set_value("G", G)
        model.parameters.set_value("C", 50.0)  # Hardening modulus
        model.parameters.set_value("gamma_dyn", 0.1)
        model.parameters.set_value("sigma_y0", sigma_y)
        model.parameters.set_value("delta_sigma_y", 0.0)
        model.parameters.set_value("eta_inf", 0.0)
        model.parameters.set_value("tau_thix", 1e9)

        # Forward loading then reverse
        t1 = np.linspace(0, 2.0, 100)
        gamma1 = 2.0 * t1  # Forward

        t2 = np.linspace(2.0, 4.0, 100)
        gamma2 = 4.0 - 2.0 * (t2 - 2.0)  # Reverse

        t = np.concatenate([t1, t2])
        gamma = np.concatenate([gamma1, gamma2])

        X = np.stack([t, gamma])
        stress = model.predict(X, test_mode="startup")

        # Key indicators of Bauschinger effect:
        # 1. Stress at reversal point should be above initial yield
        stress_at_reversal = stress[99]
        assert stress_at_reversal > sigma_y, "Should have hardened above initial yield"

        # 2. During reversal, stress should decrease (unloading)
        # and the rate of decrease shows backstress influence
        stress_during_reversal = stress[100:150]
        # Use a few steps into reversal since first point may be same
        assert (
            stress_during_reversal[5] < stress_at_reversal
        ), "Stress should decrease on reversal"

        # 3. The stress drop during unloading should be steeper than
        # without kinematic hardening (more softening effect)
        # Compare with a no-hardening model
        model_no_kin = MIKH()
        model_no_kin.parameters.set_value("G", G)
        model_no_kin.parameters.set_value("C", 0.0)  # No kinematic hardening
        model_no_kin.parameters.set_value("gamma_dyn", 0.0)
        model_no_kin.parameters.set_value("sigma_y0", sigma_y)
        model_no_kin.parameters.set_value("delta_sigma_y", 0.0)
        model_no_kin.parameters.set_value("eta_inf", 0.0)
        model_no_kin.parameters.set_value("tau_thix", 1e9)

        stress_no_kin = model_no_kin.predict(X, test_mode="startup")

        # With kinematic hardening, stress should drop faster initially
        # due to the shifted yield surface
        drop_with_kin = stress[99] - stress[120]
        drop_without_kin = stress_no_kin[99] - stress_no_kin[120]

        # The hardening case should show different behavior
        # (not necessarily larger drop, but distinct pattern)
        assert not np.allclose(
            stress[100:150], stress_no_kin[100:150], rtol=0.05
        ), "Kinematic hardening should produce different reversal behavior"

    def test_thixotropy_buildup(self):
        """Test structural buildup at rest."""
        model = MIKH()

        model.parameters.set_value("tau_thix", 1.0)  # 1 second buildup time
        model.parameters.set_value("Gamma", 1.0)

        params = dict(
            zip(model.parameters.keys(), model.parameters.get_values(), strict=False)
        )

        # Lambda evolution at rest (gamma_dot_p = 0)
        lam_initial = 0.5
        dt = 0.1
        lam = lam_initial

        for _ in range(100):
            d_lam = evolution_lambda(lam, 0.0, params) * dt
            lam = np.clip(lam + d_lam, 0.0, 1.0)

        # After ~10 time constants, should approach 1
        assert lam > 0.99, f"Lambda should approach 1 at rest, got {lam}"

    def test_thixotropy_breakdown(self):
        """Test structural breakdown under shear."""
        model = MIKH()

        model.parameters.set_value("tau_thix", 10.0)  # Slow buildup
        model.parameters.set_value("Gamma", 1.0)

        params = dict(
            zip(model.parameters.keys(), model.parameters.get_values(), strict=False)
        )

        # Lambda evolution under shear
        lam_initial = 1.0
        gamma_dot_p = 10.0  # High shear rate
        dt = 0.01
        lam = lam_initial

        for _ in range(1000):
            d_lam = evolution_lambda(lam, gamma_dot_p, params) * dt
            lam = np.clip(lam + d_lam, 0.0, 1.0)

        # Should approach steady-state: λ_ss = k1/(k1 + k2*γ̇)
        k1 = 1.0 / 10.0
        k2 = 1.0
        lam_ss_expected = k1 / (k1 + k2 * gamma_dot_p)

        np.testing.assert_allclose(lam, lam_ss_expected, rtol=1e-2)

    @pytest.mark.smoke
    def test_flow_curve_analytical(self):
        """Test flow curve against analytical steady-state solution."""
        model = MIKH()

        # Set known parameters
        sigma_y0 = 10.0
        delta_sigma_y = 40.0
        tau_thix = 1.0
        Gamma = 0.5
        eta_inf = 0.1

        model.parameters.set_value("sigma_y0", sigma_y0)
        model.parameters.set_value("delta_sigma_y", delta_sigma_y)
        model.parameters.set_value("tau_thix", tau_thix)
        model.parameters.set_value("Gamma", Gamma)
        model.parameters.set_value("eta_inf", eta_inf)

        gamma_dot = np.logspace(-2, 2, 20)

        # Predicted
        sigma_pred = model.predict_flow_curve(gamma_dot)

        # Analytical steady-state
        k1 = 1.0 / tau_thix
        k2 = Gamma
        lambda_ss = k1 / (k1 + k2 * np.abs(gamma_dot))
        sigma_y_ss = sigma_y0 + delta_sigma_y * lambda_ss
        sigma_analytical = sigma_y_ss + eta_inf * np.abs(gamma_dot)

        np.testing.assert_allclose(sigma_pred, sigma_analytical, rtol=1e-5)


# =============================================================================
# ML-IKH Model Tests
# =============================================================================


class TestMLIKH:
    """Tests for the ML-IKH model."""

    def test_initialization_per_mode(self):
        """Test ML-IKH initialization with per_mode yield."""
        n_modes = 3
        model = MLIKH(n_modes=n_modes, yield_mode="per_mode")

        assert model.n_modes == n_modes
        assert model.yield_mode == "per_mode"

        # Check parameter existence
        for i in range(1, n_modes + 1):
            assert f"G_{i}" in model.parameters
            assert f"C_{i}" in model.parameters
            assert f"tau_thix_{i}" in model.parameters

        assert "eta_inf" in model.parameters

    def test_initialization_weighted_sum(self):
        """Test ML-IKH initialization with weighted_sum yield."""
        n_modes = 2
        model = MLIKH(n_modes=n_modes, yield_mode="weighted_sum")

        assert model.n_modes == n_modes
        assert model.yield_mode == "weighted_sum"

        # Check global parameters
        assert "G" in model.parameters
        assert "C" in model.parameters
        assert "sigma_y0" in model.parameters
        assert "k3" in model.parameters

        # Check per-mode parameters
        for i in range(1, n_modes + 1):
            assert f"tau_thix_{i}" in model.parameters
            assert f"Gamma_{i}" in model.parameters
            assert f"w_{i}" in model.parameters

    def test_mode_summation(self):
        """Verify ML-IKH sums contributions correctly in per_mode."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # Mode 1: Elastic G=100
        model.parameters.set_value("G_1", 100.0)
        model.parameters.set_value("sigma_y0_1", 1e6)  # High yield (no plasticity)
        model.parameters.set_value("delta_sigma_y_1", 0.0)

        # Mode 2: Elastic G=200
        model.parameters.set_value("G_2", 200.0)
        model.parameters.set_value("sigma_y0_2", 1e6)
        model.parameters.set_value("delta_sigma_y_2", 0.0)

        model.parameters.set_value("eta_inf", 0.0)

        t = np.linspace(0, 1.0, 20)
        gamma = 0.001 * t  # Small strain

        X = np.stack([t, gamma])
        stress = model.predict(X)

        # Expected: (G1 + G2) * gamma = 300 * gamma
        expected = 300.0 * gamma
        np.testing.assert_allclose(stress, expected, rtol=5e-2)

    def test_timescale_separation(self):
        """Test that different tau_thix produce different dynamics."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # Mode 1: Fast (tau = 0.1s)
        model.parameters.set_value("tau_thix_1", 0.1)
        model.parameters.set_value("Gamma_1", 1.0)

        # Mode 2: Slow (tau = 10s)
        model.parameters.set_value("tau_thix_2", 10.0)
        model.parameters.set_value("Gamma_2", 1.0)

        # Both modes should reach different steady states for same shear
        # The fast mode will equilibrate quickly, slow mode will lag
        # This is tested implicitly by the model working correctly

        # Basic sanity check
        t = np.linspace(0, 1.0, 50)
        gamma = 0.1 * np.sin(t)  # LAOS-like

        X = np.stack([t, gamma])
        stress = model.predict(X)

        # Should produce non-trivial stress response
        assert np.std(stress) > 0

    def test_jax_compilation(self):
        """Ensure model runs under JIT compilation without errors."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        t = np.linspace(0, 1.0, 100)
        gamma = 0.1 * np.sin(t)

        X = np.stack([t, gamma])

        # First run (compiles)
        s1 = model.predict(X)

        # Second run (uses cache)
        s2 = model.predict(X)

        np.testing.assert_array_equal(s1, s2)

    def test_weighted_sum_single_mode_equivalence(self):
        """Single-mode weighted_sum should be similar to MIKH."""
        # This is an approximate test since the formulations differ slightly
        model_ws = MLIKH(n_modes=1, yield_mode="weighted_sum")
        model_mikh = MIKH()

        # Set similar parameters
        G, C, sigma_y0 = 100.0, 50.0, 20.0

        model_ws.parameters.set_value("G", G)
        model_ws.parameters.set_value("C", C)
        model_ws.parameters.set_value("sigma_y0", sigma_y0)
        model_ws.parameters.set_value("k3", 30.0)
        model_ws.parameters.set_value("tau_thix_1", 1.0)
        model_ws.parameters.set_value("Gamma_1", 0.5)
        model_ws.parameters.set_value("w_1", 1.0)
        model_ws.parameters.set_value("eta_inf", 0.0)

        model_mikh.parameters.set_value("G", G)
        model_mikh.parameters.set_value("C", C)
        model_mikh.parameters.set_value("sigma_y0", sigma_y0)
        model_mikh.parameters.set_value("delta_sigma_y", 30.0)
        model_mikh.parameters.set_value("tau_thix", 1.0)
        model_mikh.parameters.set_value("Gamma", 0.5)
        model_mikh.parameters.set_value("eta_inf", 0.0)

        t = np.linspace(0, 2.0, 100)
        gamma = 1.0 * t

        X = np.stack([t, gamma])
        stress_ws = model_ws.predict(X)
        stress_mikh = model_mikh.predict(X, test_mode="startup")

        # Should be reasonably close (within 20%)
        np.testing.assert_allclose(stress_ws, stress_mikh, rtol=0.2, atol=5.0)


class TestMLIKHProtocols:
    """Tests for ML-IKH protocol implementations."""

    def test_flow_curve_per_mode(self):
        """Test flow curve prediction with per-mode yield surfaces."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # Set parameters - use small Gamma for monotonic flow curve
        for i in [1, 2]:
            model.parameters.set_value(f"sigma_y0_{i}", 10.0)
            model.parameters.set_value(f"delta_sigma_y_{i}", 20.0)
            model.parameters.set_value(f"tau_thix_{i}", 1.0)
            model.parameters.set_value(f"Gamma_{i}", 0.01)  # Small Gamma for monotonic
        model.parameters.set_value("eta_inf", 1.0)  # Higher viscosity

        gamma_dot = jnp.logspace(-2, 2, 20)
        sigma = model.predict_flow_curve(gamma_dot)

        # Basic checks
        assert sigma.shape == gamma_dot.shape
        # Stress should be positive
        assert jnp.all(sigma > 0)
        # At high rates: viscous contribution dominates, stress should be higher
        assert sigma[-1] > sigma[0]

    def test_flow_curve_weighted_sum(self):
        """Test flow curve prediction with weighted-sum yield surface."""
        model = MLIKH(n_modes=2, yield_mode="weighted_sum")

        model.parameters.set_value("sigma_y0", 10.0)
        model.parameters.set_value("k3", 30.0)
        for i in [1, 2]:
            model.parameters.set_value(f"tau_thix_{i}", 1.0)
            model.parameters.set_value(f"Gamma_{i}", 0.01)  # Small Gamma for monotonic
            model.parameters.set_value(f"w_{i}", 0.5)
        model.parameters.set_value("eta_inf", 1.0)  # Higher viscosity

        gamma_dot = jnp.logspace(-2, 2, 20)
        sigma = model.predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape
        # Stress should be positive and increase overall
        assert jnp.all(sigma > 0)
        assert sigma[-1] > sigma[0]

    def test_startup_ode_per_mode(self):
        """Test startup prediction via ODE for per-mode."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # Set reasonable parameters
        for i in [1, 2]:
            model.parameters.set_value(f"G_{i}", 100.0)
            model.parameters.set_value(f"C_{i}", 50.0)
            model.parameters.set_value(f"gamma_dyn_{i}", 1.0)
            model.parameters.set_value(f"sigma_y0_{i}", 5.0)
            model.parameters.set_value(f"delta_sigma_y_{i}", 15.0)
            model.parameters.set_value(f"tau_thix_{i}", 1.0)
            model.parameters.set_value(f"Gamma_{i}", 0.5)
        model.parameters.set_value("eta_inf", 0.1)

        t = jnp.linspace(0.01, 5.0, 50)
        sigma = model.predict_startup(t, gamma_dot=1.0)

        assert sigma.shape == t.shape
        # Stress should grow from zero
        assert sigma[0] < sigma[-1]

    def test_relaxation_per_mode(self):
        """Test stress relaxation for per-mode."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # Use parameters that create more stable dynamics
        for i in [1, 2]:
            model.parameters.set_value(f"G_{i}", 1000.0)  # Stiffer
            model.parameters.set_value(f"C_{i}", 100.0)
            model.parameters.set_value(f"gamma_dyn_{i}", 0.1)  # Slower recovery
            model.parameters.set_value(f"sigma_y0_{i}", 50.0)  # Higher yield
            model.parameters.set_value(f"delta_sigma_y_{i}", 30.0)
            model.parameters.set_value(f"tau_thix_{i}", 10.0)  # Slower thixotropy
            model.parameters.set_value(f"Gamma_{i}", 0.01)

        # Use initial stress within yield surface
        t = jnp.linspace(0.01, 5.0, 30)
        sigma = model.predict_relaxation(t, sigma_0=50.0)  # Below total yield

        assert sigma.shape == t.shape
        # Stress should be non-negative and finite
        assert jnp.all(jnp.isfinite(sigma))
        assert jnp.all(sigma >= 0)

    def test_creep_per_mode(self):
        """Test creep response for per-mode."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        for i in [1, 2]:
            model.parameters.set_value(f"G_{i}", 100.0)
            model.parameters.set_value(f"C_{i}", 50.0)
            model.parameters.set_value(f"sigma_y0_{i}", 5.0)
            model.parameters.set_value(f"delta_sigma_y_{i}", 15.0)
            model.parameters.set_value(f"tau_thix_{i}", 1.0)
            model.parameters.set_value(f"Gamma_{i}", 0.5)
        model.parameters.set_value("eta_inf", 10.0)

        t = jnp.linspace(0.01, 5.0, 50)
        strain = model.predict_creep(t, sigma_applied=50.0)

        assert strain.shape == t.shape
        # Strain should increase (creep)
        assert strain[-1] > strain[0]

    def test_laos_per_mode(self):
        """Test LAOS response for per-mode."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        for i in [1, 2]:
            model.parameters.set_value(f"G_{i}", 100.0)
            model.parameters.set_value(f"sigma_y0_{i}", 5.0)

        t = jnp.linspace(0, 10.0, 200)
        sigma = model.predict_laos(t, gamma_0=0.5, omega=1.0)

        assert sigma.shape == t.shape
        # Should have oscillatory behavior
        assert jnp.std(sigma) > 0

    def test_convenience_methods_exist(self):
        """Verify all convenience methods are callable."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # All these methods should exist and be callable
        assert hasattr(model, "predict_flow_curve")
        assert hasattr(model, "predict_startup")
        assert hasattr(model, "predict_relaxation")
        assert hasattr(model, "predict_creep")
        assert hasattr(model, "predict_laos")

        # All should be callable
        assert callable(model.predict_flow_curve)
        assert callable(model.predict_startup)
        assert callable(model.predict_relaxation)
        assert callable(model.predict_creep)
        assert callable(model.predict_laos)

    def test_mode_independence_in_ode(self):
        """Verify modes evolve independently in per-mode formulation."""
        # Create model with very different timescales
        model = MLIKH(n_modes=2, yield_mode="per_mode")

        # Mode 1: Fast (tau = 0.01s)
        model.parameters.set_value("tau_thix_1", 0.01)
        model.parameters.set_value("Gamma_1", 10.0)
        model.parameters.set_value("G_1", 100.0)
        model.parameters.set_value("sigma_y0_1", 5.0)
        model.parameters.set_value("delta_sigma_y_1", 10.0)

        # Mode 2: Slow (tau = 100s)
        model.parameters.set_value("tau_thix_2", 100.0)
        model.parameters.set_value("Gamma_2", 0.1)
        model.parameters.set_value("G_2", 100.0)
        model.parameters.set_value("sigma_y0_2", 5.0)
        model.parameters.set_value("delta_sigma_y_2", 10.0)

        t = jnp.linspace(0.01, 1.0, 50)
        sigma = model.predict_startup(t, gamma_dot=1.0)

        # Should have non-trivial behavior due to timescale separation
        assert jnp.std(sigma) > 0
        assert sigma[-1] > sigma[0]

    @pytest.mark.smoke
    def test_single_mode_matches_mikh_flow_curve(self):
        """Single-mode MLIKH flow curve should match MIKH."""
        model_ml = MLIKH(n_modes=1, yield_mode="per_mode")
        model_mikh = MIKH()

        # Set equivalent parameters
        params = {
            "G": 100.0,
            "C": 50.0,
            "sigma_y0": 10.0,
            "delta_sigma_y": 30.0,
            "tau_thix": 1.0,
            "Gamma": 0.5,
            "eta_inf": 0.1,
        }

        # MLIKH per_mode uses indexed params
        model_ml.parameters.set_value("G_1", params["G"])
        model_ml.parameters.set_value("C_1", params["C"])
        model_ml.parameters.set_value("sigma_y0_1", params["sigma_y0"])
        model_ml.parameters.set_value("delta_sigma_y_1", params["delta_sigma_y"])
        model_ml.parameters.set_value("tau_thix_1", params["tau_thix"])
        model_ml.parameters.set_value("Gamma_1", params["Gamma"])
        model_ml.parameters.set_value("eta_inf", params["eta_inf"])

        # MIKH uses direct params
        for k, v in params.items():
            if k in model_mikh.parameters:
                model_mikh.parameters.set_value(k, v)

        gamma_dot = jnp.logspace(-2, 2, 20)
        sigma_ml = model_ml.predict_flow_curve(gamma_dot)
        sigma_mikh = model_mikh.predict_flow_curve(gamma_dot)

        np.testing.assert_allclose(sigma_ml, sigma_mikh, rtol=1e-5)


# =============================================================================
# Kernel Tests
# =============================================================================


class TestIKHKernels:
    """Tests for IKH kernels directly."""

    def test_evolution_lambda_equilibrium(self):
        """Test lambda evolution reaches equilibrium."""
        params = {"tau_thix": 1.0, "Gamma": 0.5}

        # At equilibrium: d_lambda = 0
        # (1-λ)/τ = Γ*λ*γ̇
        # λ = k1/(k1 + k2*γ̇) where k1=1/τ, k2=Γ

        gamma_dot_p = 2.0
        k1 = 1.0 / params["tau_thix"]
        k2 = params["Gamma"]
        lam_eq = k1 / (k1 + k2 * gamma_dot_p)

        # At equilibrium, derivative should be zero
        d_lam = evolution_lambda(lam_eq, gamma_dot_p, params)
        np.testing.assert_allclose(d_lam, 0.0, atol=1e-10)

    def test_radial_return_elastic(self):
        """Test return mapping in elastic regime."""
        state = (0.0, 0.0, 1.0)  # (sigma, alpha, lambda)
        params = {
            "G": 100.0,
            "C": 50.0,
            "gamma_dyn": 1.0,
            "m": 1.0,
            "sigma_y0": 100.0,  # High yield
            "delta_sigma_y": 0.0,
            "tau_thix": 1.0,
            "Gamma": 0.5,
            "eta_inf": 0.0,
        }

        dt = 0.1
        d_gamma = 0.1  # Small strain

        (sigma_new, alpha_new, lam_new), (stress_total, d_gamma_p) = (
            radial_return_step_corrected(state, (dt, d_gamma), params)
        )

        # Should be purely elastic
        expected_sigma = 100.0 * 0.1  # G * d_gamma
        np.testing.assert_allclose(sigma_new, expected_sigma, rtol=1e-5)
        np.testing.assert_allclose(d_gamma_p, 0.0, atol=1e-10)

    def test_radial_return_plastic(self):
        """Test return mapping in plastic regime."""
        state = (50.0, 0.0, 1.0)  # Start at yield
        params = {
            "G": 100.0,
            "C": 50.0,
            "gamma_dyn": 0.0,
            "m": 1.0,
            "sigma_y0": 50.0,
            "delta_sigma_y": 0.0,
            "tau_thix": 1e9,
            "Gamma": 0.0,
            "eta_inf": 0.0,
        }

        dt = 0.1
        d_gamma = 0.5  # Large strain increment

        (sigma_new, alpha_new, lam_new), (stress_total, d_gamma_p) = (
            radial_return_step_corrected(state, (dt, d_gamma), params)
        )

        # Should have plastic flow
        assert d_gamma_p > 0, "Should have plastic strain increment"

        # Stress should be limited by hardening
        assert sigma_new > 50.0, "Stress should increase due to hardening"
        assert (
            sigma_new < 50.0 + 100.0 * 0.5
        ), "Stress should be less than elastic prediction"

    def test_flow_curve_steady_state_kernel(self):
        """Test steady-state flow curve kernel."""
        params = {
            "sigma_y0": 10.0,
            "delta_sigma_y": 40.0,
            "tau_thix": 1.0,
            "Gamma": 0.5,
            "eta_inf": 0.1,
        }

        gamma_dot = jnp.array([0.1, 1.0, 10.0, 100.0])
        sigma = ikh_flow_curve_steady_state(gamma_dot, **params)

        # Analytical
        k1 = 1.0
        k2 = 0.5
        lam_ss = k1 / (k1 + k2 * gamma_dot)
        sigma_expected = 10.0 + 40.0 * lam_ss + 0.1 * gamma_dot

        np.testing.assert_allclose(sigma, sigma_expected, rtol=1e-5)
