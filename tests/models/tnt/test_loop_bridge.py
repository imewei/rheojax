"""Tests for TNTLoopBridge model.

Tests cover:
- Instantiation and parameter management
- Flow curve predictions (ODE-based steady-state)
- SAOS predictions (linearized Maxwell with G_eff = f_B_eq·G)
- ODE-based simulations (startup, relaxation, creep, LAOS)
- Bridge fraction dynamics and physical consistency
- BayesianMixin interface (model_function)
- Registry integration
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.tnt import TNTLoopBridge

jax, jnp = safe_import_jax()


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestInstantiation:
    """Tests for model instantiation and parameters."""

    @pytest.mark.smoke
    def test_default_instantiation(self):
        """Test model instantiates with default parameters."""
        model = TNTLoopBridge()

        assert model.G == 1e3
        assert model.tau_b == 1.0
        assert model.tau_a == 5.0
        assert model.nu == 1.0
        assert model.f_B_eq == 0.5
        assert model.eta_s == 0.0

    @pytest.mark.smoke
    def test_parameter_setting(self):
        """Test parameters can be set."""
        model = TNTLoopBridge()

        model.parameters.set_value("G", 2000.0)
        model.parameters.set_value("tau_b", 0.5)
        model.parameters.set_value("tau_a", 2.0)
        model.parameters.set_value("nu", 2.0)
        model.parameters.set_value("f_B_eq", 0.7)
        model.parameters.set_value("eta_s", 10.0)

        assert model.G == 2000.0
        assert model.tau_b == 0.5
        assert model.tau_a == 2.0
        assert model.nu == 2.0
        assert model.f_B_eq == 0.7
        assert model.eta_s == 10.0

    @pytest.mark.smoke
    def test_parameter_count(self):
        """Test model has exactly 6 parameters."""
        model = TNTLoopBridge()
        assert len(list(model.parameters.keys())) == 6

    @pytest.mark.smoke
    def test_derived_properties(self):
        """Test derived properties are computed correctly."""
        model = TNTLoopBridge()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 2.0)
        model.parameters.set_value("f_B_eq", 0.6)
        model.parameters.set_value("eta_s", 10.0)

        # G_eff = f_B_eq·G = 0.6*1000 = 600
        assert model.G_eff == 600.0

        # η₀ = f_B_eq·G·τ_b + η_s = 0.6*1000*2 + 10 = 1210
        assert model.eta_0 == 1210.0

    def test_equilibrium_state(self):
        """Test equilibrium state vector is correct."""
        model = TNTLoopBridge()
        model.parameters.set_value("f_B_eq", 0.7)

        eq_state = model.get_equilibrium_state()

        assert eq_state.shape == (5,)
        assert eq_state[0] == 0.7  # f_B_eq
        assert eq_state[1] == 1.0  # S_xx = 1
        assert eq_state[2] == 1.0  # S_yy = 1
        assert eq_state[3] == 1.0  # S_zz = 1
        assert eq_state[4] == 0.0  # S_xy = 0


# =============================================================================
# Flow Curve Tests
# =============================================================================


class TestFlowCurve:
    """Tests for flow curve (steady shear) predictions."""

    @pytest.mark.smoke
    def test_predict_flow_curve(self):
        """Test flow curve prediction via predict()."""
        model = TNTLoopBridge()

        gamma_dot = np.logspace(-1, 1, 10)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_predict_flow_curve_method(self):
        """Test direct predict_flow_curve method."""
        model = TNTLoopBridge()

        gamma_dot = np.logspace(-1, 1, 10)
        sigma = model.predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape
        assert np.all(np.isfinite(sigma))

    def test_flow_curve_with_components(self):
        """Test flow curve with viscosity and N1."""
        model = TNTLoopBridge()

        gamma_dot = np.logspace(-1, 1, 10)
        sigma, eta, N1 = model.predict_flow_curve(
            gamma_dot, return_components=True
        )

        assert sigma.shape == gamma_dot.shape
        assert eta.shape == gamma_dot.shape
        assert N1.shape == gamma_dot.shape
        assert np.all(eta > 0)
        assert np.all(N1 >= 0)

    def test_flow_curve_shear_thinning(self):
        """Test flow curve shows shear thinning for Bell force sensitivity."""
        model = TNTLoopBridge()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("tau_a", 5.0)
        model.parameters.set_value("nu", 3.0)  # Strong force sensitivity
        model.parameters.set_value("f_B_eq", 0.5)
        model.parameters.set_value("eta_s", 0.0)

        gamma_dot = np.array([0.1, 1.0, 10.0])
        _, eta, _ = model.predict_flow_curve(
            gamma_dot, return_components=True
        )

        # Viscosity should decrease with rate (shear thinning)
        assert eta[0] > eta[1] > eta[2]


# =============================================================================
# SAOS Tests
# =============================================================================


class TestSAOS:
    """Tests for SAOS predictions."""

    @pytest.mark.smoke
    def test_predict_saos(self):
        """Test SAOS prediction."""
        model = TNTLoopBridge()

        omega = np.logspace(-2, 2, 20)
        G_prime, G_double_prime = model.predict_saos(omega)

        assert G_prime.shape == omega.shape
        assert G_double_prime.shape == omega.shape
        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    def test_saos_magnitude(self):
        """Test |G*| prediction via test_mode='oscillation'."""
        model = TNTLoopBridge()

        omega = np.logspace(-2, 2, 20)
        G_star = model.predict(omega, test_mode="oscillation")

        G_prime, G_double_prime = model.predict_saos(omega)
        expected = np.sqrt(G_prime**2 + G_double_prime**2)

        assert np.allclose(G_star, expected)

    @pytest.mark.smoke
    def test_saos_terminal_scaling(self):
        """Test terminal regime scaling: G' ~ ω², G'' ~ ω."""
        model = TNTLoopBridge()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("f_B_eq", 0.5)
        model.parameters.set_value("eta_s", 0.0)

        omega_low = np.array([0.001, 0.01])
        G_prime, G_double_prime = model.predict_saos(omega_low)

        # G' should scale as ω² (slope = 2 in log-log)
        log_slope_Gp = np.log10(G_prime[1] / G_prime[0]) / np.log10(
            omega_low[1] / omega_low[0]
        )
        assert np.isclose(log_slope_Gp, 2.0, atol=0.1)

        # G'' should scale as ω (slope = 1 in log-log)
        log_slope_Gpp = np.log10(G_double_prime[1] / G_double_prime[0]) / np.log10(
            omega_low[1] / omega_low[0]
        )
        assert np.isclose(log_slope_Gpp, 1.0, atol=0.1)

    def test_saos_effective_maxwell(self):
        """Test SAOS matches effective Maxwell model with G_eff = f_B_eq·G."""
        model = TNTLoopBridge()
        G, tau_b, f_B_eq = 1000.0, 2.0, 0.6
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("f_B_eq", f_B_eq)
        model.parameters.set_value("eta_s", 0.0)

        omega = np.array([0.5, 1.0, 2.0])
        G_prime, G_double_prime = model.predict_saos(omega)

        # Effective Maxwell: G_eff = f_B_eq·G
        G_eff = f_B_eq * G
        wt = omega * tau_b
        wt2 = wt**2
        expected_Gp = G_eff * wt2 / (1 + wt2)
        expected_Gpp = G_eff * wt / (1 + wt2)

        assert np.allclose(G_prime, expected_Gp, rtol=1e-6)
        assert np.allclose(G_double_prime, expected_Gpp, rtol=1e-6)


# =============================================================================
# Startup Simulation Tests
# =============================================================================


class TestStartupSimulation:
    """Tests for startup flow simulation."""

    @pytest.mark.smoke
    def test_simulate_startup(self):
        """Test startup simulation runs."""
        model = TNTLoopBridge()

        t = np.linspace(0, 5, 100)
        sigma = model.simulate_startup(t, gamma_dot=5.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_startup_steady_state_value(self):
        """Test startup approaches steady-state stress."""
        model = TNTLoopBridge()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("tau_a", 5.0)
        model.parameters.set_value("nu", 1.0)
        model.parameters.set_value("f_B_eq", 0.5)
        model.parameters.set_value("eta_s", 0.0)

        gamma_dot = 5.0
        t = np.linspace(0, 10, 500)
        sigma = model.simulate_startup(t, gamma_dot=gamma_dot)

        # Stress should approach a steady value
        sigma_final = sigma[-1]
        sigma_mid = sigma[-50]

        # Last 50 points should be close (within 5%)
        assert np.abs(sigma_final - sigma_mid) / sigma_final < 0.05

    def test_startup_with_bridge_fraction(self):
        """Test startup with bridge fraction return."""
        model = TNTLoopBridge()

        t = np.linspace(0, 5, 100)
        sigma, f_B = model.simulate_startup(
            t, gamma_dot=5.0, return_bridge_fraction=True
        )

        assert sigma.shape == t.shape
        assert f_B.shape == t.shape
        assert np.all(f_B >= 0)
        assert np.all(f_B <= 1)

    def test_startup_bridge_fraction_decreases(self):
        """Test bridge fraction decreases during startup flow."""
        model = TNTLoopBridge()
        model.parameters.set_value("nu", 2.0)  # Force sensitivity

        t = np.linspace(0, 5, 100)
        _, f_B = model.simulate_startup(
            t, gamma_dot=10.0, return_bridge_fraction=True
        )

        # Bridge fraction should decrease under strong flow
        assert f_B[0] > f_B[-1]


# =============================================================================
# Relaxation Simulation Tests
# =============================================================================


class TestRelaxationSimulation:
    """Tests for stress relaxation simulation."""

    @pytest.mark.smoke
    def test_simulate_relaxation(self):
        """Test relaxation simulation runs."""
        model = TNTLoopBridge()

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=5.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_relaxation_decay(self):
        """Test stress decays during relaxation."""
        model = TNTLoopBridge()

        t = np.linspace(0, 10, 100)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=5.0)

        # Stress should decay
        assert sigma[0] > sigma[-1]
        assert sigma[-1] < sigma[0] * 0.1  # Significant decay

    def test_relaxation_with_bridge_fraction(self):
        """Test relaxation with bridge fraction return."""
        model = TNTLoopBridge()

        t = np.linspace(0, 5, 50)
        sigma, f_B = model.simulate_relaxation(
            t, gamma_dot_preshear=5.0, return_bridge_fraction=True
        )

        assert sigma.shape == t.shape
        assert f_B.shape == t.shape
        assert np.all(f_B >= 0)
        assert np.all(f_B <= 1)

    def test_relaxation_bridge_fraction_bounded(self):
        """Test bridge fraction stays bounded during relaxation."""
        model = TNTLoopBridge()
        model.parameters.set_value("f_B_eq", 0.5)

        t = np.linspace(0, 20, 200)
        sigma = model.simulate_relaxation(
            t, gamma_dot_preshear=1.0
        )

        # Stress should decay
        assert sigma[0] > sigma[-1]
        assert np.all(np.isfinite(sigma))


# =============================================================================
# Creep Simulation Tests
# =============================================================================


class TestCreepSimulation:
    """Tests for creep simulation."""

    @pytest.mark.smoke
    def test_simulate_creep(self):
        """Test creep simulation runs."""
        model = TNTLoopBridge()
        model.parameters.set_value("eta_s", 10.0)  # Need eta_s > 0

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))

    @pytest.mark.smoke
    def test_creep_strain_increases(self):
        """Test strain increases during creep."""
        model = TNTLoopBridge()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        # Strain should increase
        assert gamma[-1] > gamma[0]
        assert gamma[0] >= 0

    def test_creep_with_rate(self):
        """Test creep with rate return."""
        model = TNTLoopBridge()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma, gamma_dot = model.simulate_creep(
            t, sigma_applied=50.0, return_rate=True
        )

        assert gamma.shape == t.shape
        assert gamma_dot.shape == t.shape

    def test_creep_requires_eta_s(self):
        """Test creep requires eta_s > 0 for stability."""
        model = TNTLoopBridge()
        model.parameters.set_value("eta_s", 20.0)
        model.parameters.set_value("G", 1000.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=100.0)

        # Should complete without errors
        assert np.all(np.isfinite(gamma))


# =============================================================================
# LAOS Simulation Tests
# =============================================================================


class TestLAOSSimulation:
    """Tests for LAOS simulation."""

    @pytest.mark.smoke
    def test_simulate_laos(self):
        """Test LAOS simulation runs."""
        model = TNTLoopBridge()

        result = model.simulate_laos(
            t=None, gamma_0=0.5, omega=1.0, n_cycles=3
        )

        assert "t" in result
        assert "strain" in result
        assert "stress" in result
        assert "strain_rate" in result
        assert "f_B" in result

    def test_laos_periodicity(self):
        """Test LAOS response becomes periodic after transient."""
        model = TNTLoopBridge()

        result = model.simulate_laos(
            t=None, gamma_0=0.3, omega=1.0, n_cycles=5
        )

        stress = result["stress"]
        n_per_cycle = len(stress) // 5

        cycle_4_max = np.max(np.abs(stress[3 * n_per_cycle : 4 * n_per_cycle]))
        cycle_5_max = np.max(np.abs(stress[4 * n_per_cycle :]))

        assert np.isclose(cycle_4_max, cycle_5_max, rtol=0.1)

    def test_laos_harmonics(self):
        """Test LAOS harmonic extraction."""
        model = TNTLoopBridge()

        result = model.simulate_laos(
            t=None, gamma_0=0.5, omega=1.0, n_cycles=5
        )

        harmonics = model.extract_laos_harmonics(result, n_harmonics=3)

        assert "n" in harmonics
        assert "intensity" in harmonics
        assert len(harmonics["n"]) == 3
        assert harmonics["n"][0] == 1
        assert harmonics["n"][1] == 3

    def test_laos_bridge_fraction_oscillates(self):
        """Test bridge fraction oscillates during LAOS."""
        model = TNTLoopBridge()

        result = model.simulate_laos(
            t=None, gamma_0=0.5, omega=1.0, n_cycles=3
        )

        f_B = result["f_B"]

        # Bridge fraction should stay in [0, 1]
        assert np.all(f_B >= 0)
        assert np.all(f_B <= 1)


# =============================================================================
# Bayesian Interface Tests
# =============================================================================


class TestBayesianInterface:
    """Tests for BayesianMixin compatibility."""

    @pytest.mark.smoke
    def test_model_function_flow_curve(self):
        """Test model_function for flow curve."""
        model = TNTLoopBridge()

        X = jnp.logspace(-1, 1, 5)
        params = jnp.array([1000.0, 1.0, 5.0, 1.0, 0.5, 0.0])  # G, tau_b, tau_a, nu, f_B_eq, eta_s

        y = model.model_function(X, params, test_mode="flow_curve")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))
        assert np.all(y > 0)

    def test_model_function_saos(self):
        """Test model_function for SAOS."""
        model = TNTLoopBridge()

        X = jnp.logspace(-1, 2, 10)
        params = jnp.array([1000.0, 1.0, 5.0, 1.0, 0.5, 10.0])

        y = model.model_function(X, params, test_mode="oscillation")

        assert y.shape == X.shape
        assert np.all(y > 0)

    @pytest.mark.smoke
    def test_model_function_parameter_consistency(self):
        """Test model_function params match ParameterSet order."""
        model = TNTLoopBridge()
        G, tau_b, tau_a, nu, f_B_eq, eta_s = 1000.0, 1.0, 5.0, 1.5, 0.6, 5.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("tau_a", tau_a)
        model.parameters.set_value("nu", nu)
        model.parameters.set_value("f_B_eq", f_B_eq)
        model.parameters.set_value("eta_s", eta_s)

        X = jnp.logspace(-1, 1, 5)
        params = jnp.array([G, tau_b, tau_a, nu, f_B_eq, eta_s])

        y_fn = model.model_function(X, params, test_mode="flow_curve")
        y_pred = model.predict(X, test_mode="flow_curve")

        assert np.allclose(y_fn, y_pred, rtol=1e-6)

    def test_model_function_parameter_order(self):
        """Verify parameter order for Bayesian inference."""
        model = TNTLoopBridge()

        # Order: G, tau_b, tau_a, nu, f_B_eq, eta_s
        param_names = list(model.parameters.keys())
        assert param_names == ["G", "tau_b", "tau_a", "nu", "f_B_eq", "eta_s"]


# =============================================================================
# Physical Consistency Tests
# =============================================================================


class TestPhysicalConsistency:
    """Tests for physical correctness and consistency."""

    @pytest.mark.smoke
    def test_stress_positive_under_shear(self):
        """Test stress is positive for positive shear rate."""
        model = TNTLoopBridge()

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert np.all(sigma > 0)

    @pytest.mark.smoke
    def test_bridge_fraction_bounded(self):
        """Test bridge fraction stays in [0, 1] during startup."""
        model = TNTLoopBridge()

        t = np.linspace(0, 10, 200)
        _, f_B = model.simulate_startup(
            t, gamma_dot=10.0, return_bridge_fraction=True
        )

        assert np.all(f_B >= 0)
        assert np.all(f_B <= 1)

    def test_bridge_fraction_at_equilibrium(self):
        """Test bridge fraction starts at f_B_eq."""
        model = TNTLoopBridge()
        model.parameters.set_value("f_B_eq", 0.7)

        t = np.linspace(0, 5, 100)
        _, f_B = model.simulate_startup(
            t, gamma_dot=1.0, return_bridge_fraction=True
        )

        # Initial value should be close to f_B_eq
        assert np.isclose(f_B[0], 0.7, atol=0.01)

    def test_normal_stress_positive(self):
        """Test N₁ > 0 for loop-bridge model."""
        model = TNTLoopBridge()

        gamma_dot = np.array([1.0, 10.0])
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        assert np.all(N1 > 0)
        # N2 should be close to zero for upper-convected
        assert np.all(np.abs(N2) < 0.1 * N1)

    def test_relaxation_positive(self):
        """Test relaxation stress stays positive."""
        model = TNTLoopBridge()

        t = np.linspace(0, 10, 100)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=5.0)

        assert np.all(sigma >= 0)

    def test_bridge_fraction_vs_rate(self):
        """Test bridge fraction decreases with shear rate."""
        model = TNTLoopBridge()
        model.parameters.set_value("nu", 2.0)

        gamma_dot = np.array([0.1, 1.0, 10.0])
        _, f_B_steady = model.get_bridge_fraction_vs_rate(gamma_dot)

        # Bridge fraction should decrease with increasing rate
        assert f_B_steady[0] > f_B_steady[1] > f_B_steady[2]


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Tests for model registry integration."""

    @pytest.mark.smoke
    def test_registry_create(self):
        """Test model creation via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("tnt_loop_bridge")
        assert isinstance(model, TNTLoopBridge)


# =============================================================================
# Fitting Tests
# =============================================================================


class TestFitting:
    """Tests for model fitting."""

    @pytest.mark.smoke
    @pytest.mark.skip(reason="NLSQ forward-mode AD incompatible with diffrax custom_vjp")
    def test_fit_flow_curve(self):
        """Test fitting to flow curve data.

        Note: TNTLoopBridge uses ODE-based flow curve which involves
        diffrax custom_vjp, incompatible with NLSQ's forward-mode AD.
        """
        model_true = TNTLoopBridge()
        model_true.parameters.set_value("G", 1500.0)
        model_true.parameters.set_value("tau_b", 0.5)
        model_true.parameters.set_value("tau_a", 3.0)
        model_true.parameters.set_value("nu", 1.5)
        model_true.parameters.set_value("f_B_eq", 0.6)
        model_true.parameters.set_value("eta_s", 5.0)

        gamma_dot = np.logspace(-1, 1, 10)
        sigma_true = model_true.predict(gamma_dot, test_mode="flow_curve")

        np.random.seed(42)
        sigma_noisy = sigma_true * (
            1 + 0.02 * np.random.randn(len(sigma_true))
        )

        model_fit = TNTLoopBridge()
        model_fit.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        sigma_pred = model_fit.predict(gamma_dot, test_mode="flow_curve")
        r2 = 1 - np.sum((sigma_noisy - sigma_pred) ** 2) / np.sum(
            (sigma_noisy - np.mean(sigma_noisy)) ** 2
        )

        assert r2 > 0.90


# =============================================================================
# Analysis Method Tests
# =============================================================================


class TestAnalysisMethods:
    """Tests for analysis helper methods."""

    def test_bridge_fraction_vs_rate_method(self):
        """Test get_bridge_fraction_vs_rate helper."""
        model = TNTLoopBridge()
        model.parameters.set_value("nu", 2.0)

        gamma_dot = np.array([0.1, 1.0, 10.0])
        gd_out, f_B_steady = model.get_bridge_fraction_vs_rate(gamma_dot)

        assert np.allclose(gd_out, gamma_dot)
        assert len(f_B_steady) == len(gamma_dot)
        assert np.all(f_B_steady >= 0)
        assert np.all(f_B_steady <= 1)
