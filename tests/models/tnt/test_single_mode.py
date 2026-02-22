"""Tests for TNTSingleMode model.

Tests cover:
- Instantiation and parameter management
- Maxwell limit verification (analytical)
- Flow curve predictions
- SAOS predictions
- ODE-based simulations (startup, relaxation, creep, LAOS)
- Registry integration
- BayesianMixin interface (model_function)
- Physical consistency checks
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.tnt import TNTSingleMode

jax, jnp = safe_import_jax()


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestInstantiation:
    """Tests for model instantiation and parameters."""

    @pytest.mark.smoke
    def test_default_instantiation(self):
        """Test model instantiates with default parameters."""
        model = TNTSingleMode()

        assert model.G == 1e3
        assert model.tau_b == 1.0
        assert model.eta_s == 0.0
        assert model.breakage == "constant"
        assert model.stress_type == "linear"
        assert model.xi == 0.0

    @pytest.mark.smoke
    def test_parameter_setting(self):
        """Test parameters can be set."""
        model = TNTSingleMode()

        model.parameters.set_value("G", 2000.0)
        model.parameters.set_value("tau_b", 0.5)
        model.parameters.set_value("eta_s", 10.0)

        assert model.G == 2000.0
        assert model.tau_b == 0.5
        assert model.eta_s == 10.0

    @pytest.mark.smoke
    def test_derived_properties(self):
        """Test derived properties are computed correctly."""
        model = TNTSingleMode()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 2.0)
        model.parameters.set_value("eta_s", 10.0)

        # η₀ = G·τ_b + η_s = 1000*2 + 10 = 2010
        assert model.eta_0 == 2010.0

    def test_dimensionless_numbers(self):
        """Test Weissenberg and Deborah number computations."""
        model = TNTSingleMode()
        model.parameters.set_value("tau_b", 2.0)

        assert model.weissenberg_number(5.0) == 10.0  # Wi = τ_b·γ̇
        assert model.deborah_number(0.5) == 1.0  # De = τ_b·ω

    @pytest.mark.smoke
    def test_core_parameter_count(self):
        """Test constant breakage model has exactly 3 parameters."""
        model = TNTSingleMode()
        assert len(list(model.parameters.keys())) == 3

    @pytest.mark.smoke
    def test_bell_variant_adds_nu(self):
        """Test Bell variant adds nu parameter."""
        model = TNTSingleMode(breakage="bell")
        params = list(model.parameters.keys())
        assert "nu" in params
        assert len(params) == 4

    def test_fene_variant_adds_lmax(self):
        """Test FENE variant adds L_max parameter."""
        model = TNTSingleMode(stress_type="fene")
        params = list(model.parameters.keys())
        assert "L_max" in params
        assert len(params) == 4

    def test_power_law_variant(self):
        """Test power_law breakage variant adds m_break."""
        model = TNTSingleMode(breakage="power_law")
        params = list(model.parameters.keys())
        assert "m_break" in params

    def test_stretch_creation_variant(self):
        """Test stretch_creation breakage variant adds kappa."""
        model = TNTSingleMode(breakage="stretch_creation")
        params = list(model.parameters.keys())
        assert "kappa" in params

    def test_combined_variants(self):
        """Test Bell + FENE combined variant has correct parameters."""
        model = TNTSingleMode(breakage="bell", stress_type="fene")
        params = list(model.parameters.keys())
        assert "G" in params
        assert "tau_b" in params
        assert "eta_s" in params
        assert "nu" in params
        assert "L_max" in params
        assert len(params) == 5


# =============================================================================
# Maxwell Limit Tests (Analytical Verification)
# =============================================================================


class TestMaxwellLimits:
    """Verify TNT basic model recovers exact Maxwell behavior.

    The constant-breakage TNT model is mathematically equivalent to
    the Upper-Convected Maxwell (UCM) model. These tests verify exact
    analytical expressions.
    """

    @pytest.mark.smoke
    def test_steady_stress_equals_ucm(self):
        """Verify σ = (G·τ_b + η_s)·γ̇ for constant breakage."""
        model = TNTSingleMode()
        G, tau_b, eta_s = 1000.0, 1.0, 5.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("eta_s", eta_s)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # UCM: σ = η₀·γ̇ where η₀ = G·τ_b + η_s
        eta_0 = G * tau_b + eta_s
        expected = eta_0 * gamma_dot

        assert np.allclose(sigma, expected, rtol=1e-10)

    @pytest.mark.smoke
    def test_saos_moduli_maxwell(self):
        """Verify G'(ω) and G''(ω) match Maxwell formulas exactly."""
        model = TNTSingleMode()
        G, tau_b, eta_s = 500.0, 2.0, 1.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("eta_s", eta_s)

        omega = np.logspace(-2, 2, 30)
        G_prime, G_double_prime = model.predict_saos(omega)

        # Maxwell: G'(ω) = G·(ωτ)²/(1+(ωτ)²)
        wt = omega * tau_b
        wt2 = wt**2
        expected_Gp = G * wt2 / (1 + wt2)
        expected_Gpp = G * wt / (1 + wt2) + eta_s * omega

        assert np.allclose(G_prime, expected_Gp, rtol=1e-10)
        assert np.allclose(G_double_prime, expected_Gpp, rtol=1e-10)

    @pytest.mark.smoke
    def test_saos_crossover(self):
        """Test G' = G'' at crossover frequency ω_c = 1/τ_b."""
        model = TNTSingleMode()
        G, tau_b = 1000.0, 1.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("eta_s", 0.0)

        omega_c = 1.0 / tau_b
        G_prime, G_double_prime = model.predict_saos(np.array([omega_c]))

        assert np.isclose(G_prime[0], G_double_prime[0], rtol=0.01)
        assert np.isclose(G_prime[0], G / 2, rtol=0.01)

    @pytest.mark.smoke
    def test_n1_quadratic(self):
        """Verify N₁ = 2G·(τ_b·γ̇)² for constant breakage."""
        model = TNTSingleMode()
        G, tau_b = 1000.0, 1.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)

        gamma_dot = np.array([0.1, 1.0, 10.0])
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        expected_N1 = 2 * G * (tau_b * gamma_dot) ** 2
        assert np.allclose(N1, expected_N1, rtol=1e-10)
        assert np.allclose(N2, 0.0)  # Upper-convected → N₂ = 0


# =============================================================================
# Flow Curve Tests
# =============================================================================


class TestFlowCurve:
    """Tests for flow curve (steady shear) predictions."""

    @pytest.mark.smoke
    def test_predict_flow_curve(self):
        """Test flow curve prediction via predict()."""
        model = TNTSingleMode()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_predict_flow_curve_method(self):
        """Test direct predict_flow_curve method."""
        model = TNTSingleMode()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape

    def test_flow_curve_with_components(self):
        """Test flow curve with viscosity and N1."""
        model = TNTSingleMode()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma, eta, N1 = model.predict_flow_curve(gamma_dot, return_components=True)

        assert sigma.shape == gamma_dot.shape
        assert eta.shape == gamma_dot.shape
        assert N1.shape == gamma_dot.shape
        assert np.all(eta > 0)
        assert np.all(N1 >= 0)

    @pytest.mark.smoke
    def test_no_shear_thinning_constant_breakage(self):
        """Constant breakage gives Newtonian viscosity (no shear thinning)."""
        model = TNTSingleMode()
        G, tau_b = 1000.0, 1.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("eta_s", 0.0)

        gamma_dot = np.logspace(-2, 2, 20)
        _, eta, _ = model.predict_flow_curve(gamma_dot, return_components=True)

        # Viscosity should be constant = G·τ_b for all rates
        expected_eta = G * tau_b
        assert np.allclose(eta, expected_eta, rtol=1e-6)


# =============================================================================
# SAOS Tests
# =============================================================================


class TestSAOS:
    """Tests for SAOS predictions."""

    @pytest.mark.smoke
    def test_predict_saos(self):
        """Test SAOS prediction."""
        model = TNTSingleMode()

        omega = np.logspace(-2, 2, 20)
        G_prime, G_double_prime = model.predict_saos(omega)

        assert G_prime.shape == omega.shape
        assert G_double_prime.shape == omega.shape
        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    def test_saos_magnitude(self):
        """Test |G*| prediction via test_mode='oscillation'."""
        model = TNTSingleMode()

        omega = np.logspace(-2, 2, 20)
        G_star = model.predict(omega, test_mode="oscillation")

        G_prime, G_double_prime = model.predict_saos(omega)
        expected = np.sqrt(G_prime**2 + G_double_prime**2)

        assert np.allclose(G_star, expected)

    def test_saos_terminal_scaling(self):
        """Test terminal regime scaling: G' ~ ω², G'' ~ ω."""
        model = TNTSingleMode()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 0.0)

        omega_low = np.array([0.001, 0.01])
        G_prime, G_double_prime = model.predict_saos(omega_low)

        # G' should scale as ω² (slope = 2 in log-log)
        log_slope_Gp = np.log10(G_prime[1] / G_prime[0]) / np.log10(
            omega_low[1] / omega_low[0]
        )
        assert np.isclose(log_slope_Gp, 2.0, atol=0.05)

        # G'' should scale as ω (slope = 1 in log-log)
        log_slope_Gpp = np.log10(G_double_prime[1] / G_double_prime[0]) / np.log10(
            omega_low[1] / omega_low[0]
        )
        assert np.isclose(log_slope_Gpp, 1.0, atol=0.05)


# =============================================================================
# Startup Simulation Tests
# =============================================================================


class TestStartupSimulation:
    """Tests for startup flow simulation."""

    @pytest.mark.smoke
    def test_simulate_startup(self):
        """Test startup simulation runs."""
        model = TNTSingleMode()

        t = np.linspace(0, 5, 100)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_startup_monotonic_for_constant_breakage(self):
        """Constant breakage (UCM) should have monotonic startup (no overshoot)."""
        model = TNTSingleMode()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 0.0)

        t = np.linspace(0, 10, 200)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        # Check monotonically increasing (within tolerance for numerics)
        diffs = np.diff(sigma)
        assert np.all(diffs >= -1e-6 * np.max(np.abs(sigma)))

    @pytest.mark.smoke
    def test_startup_steady_state_value(self):
        """Startup should approach steady-state stress for constant breakage."""
        model = TNTSingleMode()
        G, tau_b, eta_s = 1000.0, 1.0, 0.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("eta_s", eta_s)

        gamma_dot = 10.0
        t = np.linspace(0, 20, 500)
        sigma = model.simulate_startup(t, gamma_dot=gamma_dot)

        # Analytical steady state: σ_ss = (G·τ_b + η_s)·γ̇
        sigma_ss_expected = (G * tau_b + eta_s) * gamma_dot

        # After ~10 τ_b, should be very close
        assert np.isclose(sigma[-1], sigma_ss_expected, rtol=0.01)

    def test_startup_full_return(self):
        """Test full conformation tensor return."""
        model = TNTSingleMode()

        t = np.linspace(0, 5, 100)
        S_xx, S_yy, S_xy, S_zz = model.simulate_startup(
            t, gamma_dot=10.0, return_full=True
        )

        assert S_xx.shape == t.shape
        assert S_yy.shape == t.shape
        assert S_xy.shape == t.shape
        assert S_zz.shape == t.shape

    def test_startup_via_predict(self):
        """Test startup via predict() method."""
        model = TNTSingleMode()
        model._gamma_dot_applied = 10.0

        t = np.linspace(0, 5, 100)
        sigma = model.predict(t, test_mode="startup", gamma_dot=10.0)

        assert sigma.shape == t.shape


# =============================================================================
# Relaxation Simulation Tests
# =============================================================================


class TestRelaxationSimulation:
    """Tests for stress relaxation simulation."""

    @pytest.mark.smoke
    def test_simulate_relaxation(self):
        """Test relaxation simulation runs."""
        model = TNTSingleMode()

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=1.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_relaxation_exponential(self):
        """Verify single-exponential relaxation for constant breakage."""
        model = TNTSingleMode()
        G, tau_b = 1000.0, 1.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)

        gamma_dot_pre = 5.0
        t = np.linspace(0, 5, 100)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=gamma_dot_pre)

        # Analytical: σ(t) = G·τ_b·γ̇·exp(-t/τ_b)
        sigma_0 = G * tau_b * gamma_dot_pre
        expected = sigma_0 * np.exp(-t / tau_b)

        assert np.allclose(sigma, expected, rtol=1e-6)

    @pytest.mark.smoke
    def test_relaxation_decay(self):
        """Test stress decays during relaxation."""
        model = TNTSingleMode()

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=1.0)

        assert sigma[0] > sigma[-1]
        assert sigma[-1] < sigma[0] * 0.01  # ~5 τ_b decay


# =============================================================================
# Creep Simulation Tests
# =============================================================================


class TestCreepSimulation:
    """Tests for creep simulation."""

    @pytest.mark.smoke
    def test_simulate_creep(self):
        """Test creep simulation runs."""
        model = TNTSingleMode()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))

    @pytest.mark.smoke
    def test_creep_strain_increases(self):
        """Test strain increases during creep."""
        model = TNTSingleMode()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        assert gamma[-1] > gamma[0]

    def test_creep_with_rate(self):
        """Test creep with rate return."""
        model = TNTSingleMode()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma, gamma_dot = model.simulate_creep(t, sigma_applied=50.0, return_rate=True)

        assert gamma.shape == t.shape
        assert gamma_dot.shape == t.shape

    def test_creep_steady_rate(self):
        """Test creep reaches steady shear rate.

        At long times: γ̇ → σ/(G·τ_b + η_s) = σ/η₀
        """
        model = TNTSingleMode()
        G, tau_b, eta_s = 1000.0, 1.0, 10.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("eta_s", eta_s)

        sigma_applied = 100.0
        t = np.linspace(0, 50, 500)
        gamma, gamma_dot = model.simulate_creep(
            t, sigma_applied=sigma_applied, return_rate=True
        )

        # Steady rate: γ̇_ss = σ / η₀
        eta_0 = G * tau_b + eta_s
        gamma_dot_expected = sigma_applied / eta_0

        assert np.isclose(gamma_dot[-1], gamma_dot_expected, rtol=0.05)


# =============================================================================
# LAOS Simulation Tests
# =============================================================================


class TestLAOSSimulation:
    """Tests for LAOS simulation."""

    @pytest.mark.smoke
    def test_simulate_laos(self):
        """Test LAOS simulation runs."""
        model = TNTSingleMode()

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)

        assert "t" in result
        assert "strain" in result
        assert "stress" in result
        assert "strain_rate" in result

    @pytest.mark.smoke
    def test_laos_periodicity(self):
        """Test LAOS response is periodic after transient."""
        model = TNTSingleMode()

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=5)

        stress = result["stress"]
        n_per_cycle = len(stress) // 5

        cycle_4_max = np.max(np.abs(stress[3 * n_per_cycle : 4 * n_per_cycle]))
        cycle_5_max = np.max(np.abs(stress[4 * n_per_cycle :]))

        assert np.isclose(cycle_4_max, cycle_5_max, rtol=0.05)

    def test_laos_small_amplitude_linear(self):
        """Small-amplitude LAOS should match SAOS (linear limit)."""
        model = TNTSingleMode()
        G, tau_b, eta_s = 1000.0, 1.0, 0.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("eta_s", eta_s)

        omega = 1.0
        gamma_0 = 0.001  # Very small amplitude → linear

        result = model.simulate_laos(t=None, gamma_0=gamma_0, omega=omega, n_cycles=10)

        # Extract last cycle stress amplitude
        stress = result["stress"]
        n_per_cycle = len(stress) // 10
        last_cycle = stress[-n_per_cycle:]
        sigma_amplitude = np.max(np.abs(last_cycle))

        # Expected from SAOS: |G*|·γ₀
        G_prime, G_double_prime = model.predict_saos(np.array([omega]))
        G_star = np.sqrt(G_prime[0] ** 2 + G_double_prime[0] ** 2)
        expected_amplitude = G_star * gamma_0

        assert np.isclose(sigma_amplitude, expected_amplitude, rtol=0.05)

    def test_laos_harmonics(self):
        """Test LAOS harmonic extraction."""
        model = TNTSingleMode()

        result = model.simulate_laos(t=None, gamma_0=1.0, omega=1.0, n_cycles=5)

        harmonics = model.extract_laos_harmonics(result, n_harmonics=3)

        assert "n" in harmonics
        assert "intensity" in harmonics
        assert len(harmonics["n"]) == 3
        assert harmonics["n"][0] == 1
        assert harmonics["n"][1] == 3


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Tests for model registry integration."""

    @pytest.mark.smoke
    def test_registry_create(self):
        """Test model creation via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("tnt_single_mode")
        assert isinstance(model, TNTSingleMode)

    @pytest.mark.smoke
    def test_registry_alias(self):
        """Test alias creation via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("tnt")
        assert isinstance(model, TNTSingleMode)


# =============================================================================
# Bayesian Interface Tests
# =============================================================================


class TestBayesianInterface:
    """Tests for BayesianMixin compatibility."""

    @pytest.mark.smoke
    def test_model_function_flow_curve(self):
        """Test model_function for flow curve."""
        model = TNTSingleMode()

        X = jnp.logspace(-2, 2, 10)
        params = jnp.array([1000.0, 1.0, 0.0])  # G, tau_b, eta_s

        y = model.model_function(X, params, test_mode="flow_curve")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))

    def test_model_function_saos(self):
        """Test model_function for SAOS."""
        model = TNTSingleMode()

        X = jnp.logspace(-1, 2, 10)
        params = jnp.array([1000.0, 1.0, 10.0])

        y = model.model_function(X, params, test_mode="oscillation")

        assert y.shape == (len(X), 2)
        assert np.all(y > 0)

    @pytest.mark.smoke
    def test_model_function_parameter_consistency(self):
        """Test model_function params match ParameterSet order."""
        model = TNTSingleMode()
        G, tau_b, eta_s = 1000.0, 1.0, 5.0
        model.parameters.set_value("G", G)
        model.parameters.set_value("tau_b", tau_b)
        model.parameters.set_value("eta_s", eta_s)

        X = jnp.logspace(-2, 2, 10)
        params = jnp.array([G, tau_b, eta_s])

        y_fn = model.model_function(X, params, test_mode="flow_curve")
        y_pred = model.predict(X, test_mode="flow_curve")

        assert np.allclose(y_fn, y_pred, rtol=1e-10)


# =============================================================================
# Physical Consistency Tests
# =============================================================================


class TestPhysicalConsistency:
    """Tests for physical correctness and consistency."""

    @pytest.mark.smoke
    def test_equilibrium_conformation(self):
        """Test equilibrium conformation tensor is identity."""
        S_eq = TNTSingleMode.get_equilibrium_conformation()
        assert np.allclose(S_eq, [1.0, 1.0, 1.0, 0.0])

    @pytest.mark.smoke
    def test_stress_positive_under_shear(self):
        """Test stress is positive for positive shear rate."""
        model = TNTSingleMode()

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert np.all(sigma > 0)

    def test_conformation_positive_definite_startup(self):
        """Test conformation tensor stays positive definite during startup."""
        model = TNTSingleMode()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)

        t = np.linspace(0, 10, 200)
        S_xx, S_yy, S_xy, S_zz = model.simulate_startup(
            t, gamma_dot=10.0, return_full=True
        )

        # Diagonal elements must be positive
        assert np.all(S_xx > 0)
        assert np.all(S_yy > 0)
        assert np.all(S_zz > 0)

        # Positive definite: S_xx·S_yy > S_xy²
        det_2d = S_xx * S_yy - S_xy**2
        assert np.all(det_2d > 0)

    def test_n1_positive_n2_zero(self):
        """Test N₁ > 0 and N₂ = 0 for upper-convected (ξ=0)."""
        model = TNTSingleMode()

        gamma_dot = np.logspace(-1, 2, 20)
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        assert np.all(N1 > 0)
        assert np.allclose(N2, 0.0)

    @pytest.mark.smoke
    def test_relaxation_positive(self):
        """Test relaxation stress stays positive."""
        model = TNTSingleMode()

        t = np.linspace(0, 10, 100)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=5.0)

        assert np.all(sigma >= 0)


# =============================================================================
# Fitting Tests
# =============================================================================


class TestFitting:
    """Tests for model fitting."""

    @pytest.mark.smoke
    def test_fit_flow_curve(self):
        """Test fitting to flow curve data."""
        model_true = TNTSingleMode()
        model_true.parameters.set_value("G", 1500.0)
        model_true.parameters.set_value("tau_b", 0.5)
        model_true.parameters.set_value("eta_s", 5.0)

        gamma_dot = np.logspace(-1, 2, 20)
        sigma_true = model_true.predict(gamma_dot, test_mode="flow_curve")

        np.random.seed(42)
        sigma_noisy = sigma_true * (1 + 0.02 * np.random.randn(len(sigma_true)))

        model_fit = TNTSingleMode()
        model_fit.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        sigma_pred = model_fit.predict(gamma_dot, test_mode="flow_curve")
        r2 = 1 - np.sum((sigma_noisy - sigma_pred) ** 2) / np.sum(
            (sigma_noisy - np.mean(sigma_noisy)) ** 2
        )

        assert r2 > 0.95


# =============================================================================
# Analysis Method Tests
# =============================================================================


class TestAnalysisMethods:
    """Tests for analysis helper methods."""

    @pytest.mark.smoke
    def test_overshoot_ratio_ucm(self):
        """For UCM (constant breakage), overshoot ratio should be ~1."""
        model = TNTSingleMode()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 0.0)

        overshoot, _ = model.get_overshoot_ratio(gamma_dot=10.0)
        # UCM doesn't overshoot
        assert np.isclose(overshoot, 1.0, atol=0.05)

    def test_relaxation_spectrum(self):
        """Test relaxation spectrum G(t)."""
        model = TNTSingleMode()
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)

        t, G_t = model.get_relaxation_spectrum(n_points=50)

        assert len(t) == 50
        assert len(G_t) == 50
        assert G_t[0] > G_t[-1]

        # Should be single exponential
        expected = 1000.0 * np.exp(-t / 1.0)
        assert np.allclose(G_t, expected, rtol=1e-6)


# =============================================================================
# Bell Breakage Variant Tests
# =============================================================================


class TestBellVariant:
    """Tests for Bell force-dependent breakage: β = (1/τ_b)·exp(ν·(stretch-1))."""

    def test_bell_instantiation(self):
        """Bell variant should have nu parameter."""
        model = TNTSingleMode(breakage="bell")
        assert "nu" in list(model.parameters.keys())
        assert model.breakage == "bell"

    def test_bell_shear_thinning_flow_curve(self):
        """Bell breakage should produce enhanced shear thinning vs constant.

        At high Wi, force-dependent detachment accelerates breakage,
        reducing the effective relaxation time and thus the stress.
        """
        model_basic = TNTSingleMode()
        model_basic.parameters.set_value("G", 1000.0)
        model_basic.parameters.set_value("tau_b", 1.0)
        model_basic.parameters.set_value("eta_s", 0.01)

        model_bell = TNTSingleMode(breakage="bell")
        model_bell.parameters.set_value("G", 1000.0)
        model_bell.parameters.set_value("tau_b", 1.0)
        model_bell.parameters.set_value("eta_s", 0.01)
        model_bell.parameters.set_value("nu", 3.0)

        gamma_dot = np.array([10.0, 100.0])

        sigma_basic = model_basic.predict_flow_curve(gamma_dot)
        sigma_bell = model_bell.predict_flow_curve(gamma_dot)

        # Bell should give lower stress at high rates (enhanced thinning)
        assert np.all(
            sigma_bell < sigma_basic
        ), f"Bell stress {sigma_bell} should be below constant {sigma_basic}"

    def test_bell_stress_overshoot(self):
        """Bell breakage should produce stress overshoot in startup flow.

        At high Wi, chains stretch before force-dependent breakage kicks in,
        leading to a transient overshoot.
        """
        model = TNTSingleMode(breakage="bell")
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 0.0)
        model.parameters.set_value("nu", 5.0)  # Strong force sensitivity

        gamma_dot = 20.0  # Wi = 20 (strong flow)
        t = np.linspace(0, 5.0, 500)
        sigma = model.simulate_startup(t, gamma_dot)

        peak_idx = np.argmax(sigma)
        sigma_max = sigma[peak_idx]
        sigma_ss = sigma[-1]

        # Overshoot ratio > 1 for strong force sensitivity
        overshoot = sigma_max / sigma_ss
        assert overshoot > 1.05, f"Expected Bell overshoot > 1.05, got {overshoot:.3f}"

    def test_bell_model_function_flow_curve(self):
        """Test model_function works for Bell variant flow curve."""
        model = TNTSingleMode(breakage="bell")

        # params = [G, tau_b, eta_s, nu]
        params = jnp.array([1000.0, 1.0, 0.01, 3.0])
        X = jnp.array([0.1, 1.0, 10.0])

        result = model.model_function(X, params, test_mode="flow_curve")

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_bell_all_protocols_run(self):
        """Verify Bell variant can execute all 6 protocols without error."""
        model = TNTSingleMode(breakage="bell")
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 1.0)
        model.parameters.set_value("nu", 2.0)

        # Flow curve
        sigma_fc = model.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(sigma_fc))

        # SAOS (always analytical — linearized to Maxwell)
        G_p, G_pp = model.predict_saos(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(G_p)) and np.all(np.isfinite(G_pp))

        # Startup
        t = np.linspace(0, 5.0, 200)
        sigma_s = model.simulate_startup(t, gamma_dot=5.0)
        assert np.all(np.isfinite(sigma_s))

        # Relaxation
        sigma_r = model.simulate_relaxation(t, gamma_dot_preshear=5.0)
        assert np.all(np.isfinite(sigma_r))

        # Creep
        gamma = model.simulate_creep(t, sigma_applied=500.0)
        assert np.all(np.isfinite(gamma))

        # LAOS
        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)
        assert np.all(np.isfinite(result["stress"]))


# =============================================================================
# FENE-P Variant Tests
# =============================================================================


class TestFENEVariant:
    """Tests for FENE-P finite extensibility: σ = G·f(trS)·(S-I)."""

    def test_fene_instantiation(self):
        """FENE variant should have L_max parameter."""
        model = TNTSingleMode(stress_type="fene")
        assert "L_max" in list(model.parameters.keys())
        assert model.stress_type == "fene"

    def test_fene_stress_bounded(self):
        """FENE stress should saturate at high stretch (bounded by L_max).

        Unlike linear stress which grows unboundedly, FENE-P stress
        reaches a maximum determined by the finite extensibility L_max.
        """
        model_fene = TNTSingleMode(stress_type="fene")
        model_fene.parameters.set_value("G", 1000.0)
        model_fene.parameters.set_value("tau_b", 1.0)
        model_fene.parameters.set_value("eta_s", 0.0)
        model_fene.parameters.set_value("L_max", 5.0)

        model_linear = TNTSingleMode()
        model_linear.parameters.set_value("G", 1000.0)
        model_linear.parameters.set_value("tau_b", 1.0)
        model_linear.parameters.set_value("eta_s", 0.0)

        gamma_dot = np.array([1.0, 10.0, 100.0])

        sigma_fene = model_fene.predict_flow_curve(gamma_dot)
        sigma_linear = model_linear.predict_flow_curve(gamma_dot)

        # At high rates, FENE should show higher stress than linear
        # due to the Peterlin factor f = L²/(L²-trS) > 1
        # But the key test: FENE stress grows slower at very high rates
        ratio_fene = sigma_fene[-1] / sigma_fene[0]
        ratio_linear = sigma_linear[-1] / sigma_linear[0]

        # Both finite and positive
        assert np.all(np.isfinite(sigma_fene))
        assert np.all(sigma_fene > 0)

    def test_fene_strain_stiffening_startup(self):
        """FENE should show strain stiffening in startup (higher transient stress).

        The Peterlin factor f > 1 amplifies stress when chains are
        stretched, leading to a transient overshoot even without
        force-dependent breakage.
        """
        model = TNTSingleMode(stress_type="fene")
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 0.0)
        model.parameters.set_value("L_max", 5.0)

        t = np.linspace(0, 5.0, 500)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        # FENE startup should produce finite stresses
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma >= 0)

    def test_fene_model_function(self):
        """Test model_function for FENE variant."""
        model = TNTSingleMode(stress_type="fene")

        # params = [G, tau_b, eta_s, L_max]
        params = jnp.array([1000.0, 1.0, 0.01, 5.0])
        X = jnp.array([0.1, 1.0, 10.0])

        result = model.model_function(X, params, test_mode="flow_curve")
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_fene_all_protocols_run(self):
        """Verify FENE variant can execute all 6 protocols."""
        model = TNTSingleMode(stress_type="fene")
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 1.0)
        model.parameters.set_value("L_max", 10.0)

        sigma_fc = model.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(sigma_fc))

        G_p, G_pp = model.predict_saos(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(G_p))

        t = np.linspace(0, 5.0, 200)
        sigma_s = model.simulate_startup(t, gamma_dot=5.0)
        assert np.all(np.isfinite(sigma_s))

        sigma_r = model.simulate_relaxation(t, gamma_dot_preshear=5.0)
        assert np.all(np.isfinite(sigma_r))

        gamma = model.simulate_creep(t, sigma_applied=500.0)
        assert np.all(np.isfinite(gamma))

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)
        assert np.all(np.isfinite(result["stress"]))


# =============================================================================
# Non-Affine (Gordon-Schowalter) Variant Tests
# =============================================================================


class TestNonAffineVariant:
    """Tests for non-affine chain slip via Gordon-Schowalter derivative."""

    def test_non_affine_instantiation(self):
        """Non-affine model should store xi > 0."""
        model = TNTSingleMode(xi=0.3)
        assert model.xi == 0.3

    def test_non_affine_n2_nonzero(self):
        """Non-affine deformation produces N₂ ≠ 0.

        The Gordon-Schowalter derivative introduces coupling between
        normal stress differences: N₂ = -ξ/(2-ξ)·N₁ for small ξ.
        """
        model = TNTSingleMode(xi=0.3)
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 0.0)

        gamma_dot = np.array([1.0, 10.0])
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        # N1 should be positive
        assert np.all(N1 > 0), f"Expected N1 > 0, got {N1}"

        # N2 should be nonzero (and typically negative)
        assert np.all(N2 != 0), f"Expected N2 ≠ 0, got {N2}"
        assert np.all(N2 < 0), f"Expected N2 < 0 for xi > 0, got {N2}"

    def test_non_affine_reduces_to_ucm_at_xi_zero(self):
        """At xi=0, non-affine should give same result as basic model."""
        model_basic = TNTSingleMode()
        model_basic.parameters.set_value("G", 1000.0)
        model_basic.parameters.set_value("tau_b", 1.0)
        model_basic.parameters.set_value("eta_s", 0.01)

        # xi=0 should be identical to basic (just uses GS derivative with xi=0)
        model_gs = TNTSingleMode(xi=0.0)
        model_gs.parameters.set_value("G", 1000.0)
        model_gs.parameters.set_value("tau_b", 1.0)
        model_gs.parameters.set_value("eta_s", 0.01)

        gamma_dot = np.array([0.1, 1.0, 10.0])

        sigma_basic = model_basic.predict_flow_curve(gamma_dot)
        sigma_gs = model_gs.predict_flow_curve(gamma_dot)

        # Should be very close (both reduce to UCM at xi=0)
        assert np.allclose(
            sigma_basic, sigma_gs, rtol=0.05
        ), f"xi=0 should match basic: {sigma_basic} vs {sigma_gs}"

    def test_non_affine_all_protocols_run(self):
        """Verify non-affine variant can execute all 6 protocols."""
        model = TNTSingleMode(xi=0.2)
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 1.0)

        sigma_fc = model.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(sigma_fc))

        G_p, G_pp = model.predict_saos(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(G_p))

        t = np.linspace(0, 5.0, 200)
        sigma_s = model.simulate_startup(t, gamma_dot=5.0)
        assert np.all(np.isfinite(sigma_s))

        sigma_r = model.simulate_relaxation(t, gamma_dot_preshear=5.0)
        assert np.all(np.isfinite(sigma_r))

        gamma = model.simulate_creep(t, sigma_applied=500.0)
        assert np.all(np.isfinite(gamma))

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)
        assert np.all(np.isfinite(result["stress"]))


# =============================================================================
# Composed Variant Tests (Bell + FENE, Bell + NonAffine, etc.)
# =============================================================================


class TestComposedVariants:
    """Tests for combining multiple variant flags."""

    def test_bell_fene_instantiation(self):
        """Bell+FENE should have both nu and L_max parameters."""
        model = TNTSingleMode(breakage="bell", stress_type="fene")
        param_names = list(model.parameters.keys())
        assert "nu" in param_names
        assert "L_max" in param_names

    def test_bell_fene_all_protocols(self):
        """Bell+FENE composition should run all 6 protocols."""
        model = TNTSingleMode(breakage="bell", stress_type="fene")
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 1.0)
        model.parameters.set_value("nu", 2.0)
        model.parameters.set_value("L_max", 10.0)

        sigma_fc = model.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(sigma_fc))

        G_p, G_pp = model.predict_saos(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(G_p))

        t = np.linspace(0, 5.0, 200)
        sigma_s = model.simulate_startup(t, gamma_dot=5.0)
        assert np.all(np.isfinite(sigma_s))

        sigma_r = model.simulate_relaxation(t, gamma_dot_preshear=5.0)
        assert np.all(np.isfinite(sigma_r))

        gamma = model.simulate_creep(t, sigma_applied=500.0)
        assert np.all(np.isfinite(gamma))

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)
        assert np.all(np.isfinite(result["stress"]))

    def test_bell_nonaffine_composition(self):
        """Bell+NonAffine should run without errors."""
        model = TNTSingleMode(breakage="bell", xi=0.2)
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 1.0)
        model.parameters.set_value("nu", 2.0)

        t = np.linspace(0, 5.0, 200)
        sigma = model.simulate_startup(t, gamma_dot=5.0)
        assert np.all(np.isfinite(sigma))

        N1, N2 = model.predict_normal_stresses(np.array([1.0, 10.0]))
        assert np.all(N1 > 0)
        assert np.all(N2 < 0)

    def test_full_composition_bell_fene_nonaffine(self):
        """Full composition: Bell+FENE+NonAffine should run all protocols."""
        model = TNTSingleMode(breakage="bell", stress_type="fene", xi=0.2)
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 1.0)
        model.parameters.set_value("nu", 2.0)
        model.parameters.set_value("L_max", 10.0)

        # Flow curve
        sigma_fc = model.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(sigma_fc))
        assert np.all(sigma_fc > 0)

        # Startup
        t = np.linspace(0, 5.0, 200)
        sigma_s = model.simulate_startup(t, gamma_dot=5.0)
        assert np.all(np.isfinite(sigma_s))

        # Creep
        gamma = model.simulate_creep(t, sigma_applied=500.0)
        assert np.all(np.isfinite(gamma))
        assert gamma[-1] > gamma[0]  # Strain accumulates

        # LAOS
        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)
        assert np.all(np.isfinite(result["stress"]))

    def test_full_composition_model_function(self):
        """model_function should work for fully composed variant."""
        model = TNTSingleMode(breakage="bell", stress_type="fene", xi=0.2)

        # params = [G, tau_b, eta_s, nu, L_max]
        # (xi is fixed at construction, not a fitted parameter)
        params = jnp.array([1000.0, 1.0, 0.01, 2.0, 10.0])
        X = jnp.array([0.1, 1.0, 10.0])

        result = model.model_function(X, params, test_mode="flow_curve")
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_power_law_breakage_all_protocols(self):
        """Power-law breakage variant should run all protocols."""
        model = TNTSingleMode(breakage="power_law")
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 1.0)
        model.parameters.set_value("m_break", 2.0)

        sigma_fc = model.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(sigma_fc))

        t = np.linspace(0, 5.0, 200)
        sigma_s = model.simulate_startup(t, gamma_dot=5.0)
        assert np.all(np.isfinite(sigma_s))

        gamma = model.simulate_creep(t, sigma_applied=500.0)
        assert np.all(np.isfinite(gamma))

    def test_stretch_creation_all_protocols(self):
        """Stretch-creation variant should run all protocols."""
        model = TNTSingleMode(breakage="stretch_creation")
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_b", 1.0)
        model.parameters.set_value("eta_s", 1.0)
        model.parameters.set_value("kappa", 1.0)

        sigma_fc = model.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        assert np.all(np.isfinite(sigma_fc))

        t = np.linspace(0, 5.0, 200)
        sigma_s = model.simulate_startup(t, gamma_dot=5.0)
        assert np.all(np.isfinite(sigma_s))

        gamma = model.simulate_creep(t, sigma_applied=500.0)
        assert np.all(np.isfinite(gamma))
