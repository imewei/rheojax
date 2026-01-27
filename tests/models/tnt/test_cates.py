"""Tests for TNTCates model (Cates living polymer / wormlike micelle).

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
from rheojax.models.tnt import TNTCates

jax, jnp = safe_import_jax()


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestInstantiation:
    """Tests for model instantiation and parameters."""

    @pytest.mark.smoke
    def test_default_instantiation(self):
        """Test model instantiates with default parameters."""
        model = TNTCates()

        assert model.G_0 == 1e3
        assert model.tau_rep == 10.0
        assert model.tau_break == 0.1
        assert model.eta_s == 0.0

    @pytest.mark.smoke
    def test_derived_tau_d(self):
        """Test derived τ_d = √(τ_rep · τ_break)."""
        model = TNTCates()
        model.parameters.set_value("tau_rep", 10.0)
        model.parameters.set_value("tau_break", 0.1)

        # τ_d = √(10 * 0.1) = √1 = 1.0
        assert np.isclose(model.tau_d, 1.0, rtol=1e-10)

    @pytest.mark.smoke
    def test_parameter_setting(self):
        """Test parameters can be set."""
        model = TNTCates()

        model.parameters.set_value("G_0", 2000.0)
        model.parameters.set_value("tau_rep", 5.0)
        model.parameters.set_value("tau_break", 0.2)
        model.parameters.set_value("eta_s", 10.0)

        assert model.G_0 == 2000.0
        assert model.tau_rep == 5.0
        assert model.tau_break == 0.2
        assert model.eta_s == 10.0

        # τ_d = √(5 * 0.2) = 1.0
        assert np.isclose(model.tau_d, 1.0, rtol=1e-10)

    def test_derived_eta_0(self):
        """Test derived η₀ = G₀·τ_d + η_s."""
        model = TNTCates()
        model.parameters.set_value("G_0", 1000.0)
        model.parameters.set_value("tau_rep", 4.0)
        model.parameters.set_value("tau_break", 1.0)
        model.parameters.set_value("eta_s", 10.0)

        # τ_d = √(4 * 1) = 2.0
        # η₀ = 1000 * 2.0 + 10 = 2010
        assert np.isclose(model.tau_d, 2.0)
        assert np.isclose(model.eta_0, 2010.0)

    @pytest.mark.smoke
    def test_parameter_count(self):
        """Test Cates model has exactly 4 parameters."""
        model = TNTCates()
        params = list(model.parameters.keys())
        assert len(params) == 4
        assert "G_0" in params
        assert "tau_rep" in params
        assert "tau_break" in params
        assert "eta_s" in params


# =============================================================================
# Cates Maxwell Limit Tests (Analytical Verification)
# =============================================================================


class TestCatesMaxwellLimit:
    """Verify Cates model reduces to Maxwell behavior with effective τ_d.

    The Cates living polymer model in the fast-breaking limit behaves as
    a single Maxwell mode with effective relaxation time τ_d = √(τ_rep·τ_break).
    """

    @pytest.mark.smoke
    def test_tau_d_derived_correctly(self):
        """Test τ_d = √(τ_rep · τ_break)."""
        model = TNTCates()
        model.parameters.set_value("tau_rep", 16.0)
        model.parameters.set_value("tau_break", 0.25)

        # τ_d = √(16 * 0.25) = √4 = 2.0
        assert np.isclose(model.tau_d, 2.0, rtol=1e-10)

    @pytest.mark.smoke
    def test_steady_stress_equals_maxwell(self):
        """Verify σ = (G₀·τ_d + η_s)·γ̇ for Cates model."""
        model = TNTCates()
        G_0, tau_rep, tau_break, eta_s = 1000.0, 4.0, 1.0, 5.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)
        model.parameters.set_value("eta_s", eta_s)

        # τ_d = √(4*1) = 2.0
        # η₀ = G₀·τ_d + η_s = 1000*2 + 5 = 2005
        tau_d = np.sqrt(tau_rep * tau_break)
        eta_0 = G_0 * tau_d + eta_s

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # UCM-like steady state: σ = η₀·γ̇
        expected = eta_0 * gamma_dot

        assert np.allclose(sigma, expected, rtol=1e-10)

    @pytest.mark.smoke
    def test_saos_moduli_match_maxwell(self):
        """Verify G'(ω) and G''(ω) match Maxwell with τ_d."""
        model = TNTCates()
        G_0, tau_rep, tau_break, eta_s = 500.0, 4.0, 1.0, 1.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)
        model.parameters.set_value("eta_s", eta_s)

        # τ_d = √(4*1) = 2.0
        tau_d = np.sqrt(tau_rep * tau_break)

        omega = np.logspace(-2, 2, 30)
        G_prime, G_double_prime = model.predict_saos(omega)

        # Maxwell: G'(ω) = G₀·(ωτ_d)²/(1+(ωτ_d)²)
        wt = omega * tau_d
        wt2 = wt**2
        expected_Gp = G_0 * wt2 / (1 + wt2)
        expected_Gpp = G_0 * wt / (1 + wt2) + eta_s * omega

        assert np.allclose(G_prime, expected_Gp, rtol=1e-10)
        assert np.allclose(G_double_prime, expected_Gpp, rtol=1e-10)

    @pytest.mark.smoke
    def test_crossover_at_one_over_tau_d(self):
        """Test G' = G'' crossover at ω_c = 1/τ_d."""
        model = TNTCates()
        G_0, tau_rep, tau_break = 1000.0, 1.0, 1.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)
        model.parameters.set_value("eta_s", 0.0)

        # τ_d = √(1*1) = 1.0
        tau_d = model.tau_d
        omega_c = 1.0 / tau_d

        G_prime, G_double_prime = model.predict_saos(np.array([omega_c]))

        assert np.isclose(G_prime[0], G_double_prime[0], rtol=0.01)
        assert np.isclose(G_prime[0], G_0 / 2, rtol=0.01)


# =============================================================================
# Flow Curve Tests
# =============================================================================


class TestFlowCurve:
    """Tests for flow curve (steady shear) predictions."""

    @pytest.mark.smoke
    def test_predict_flow_curve(self):
        """Test flow curve prediction via predict()."""
        model = TNTCates()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_predict_flow_curve_method(self):
        """Test direct predict_flow_curve method."""
        model = TNTCates()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    def test_flow_curve_with_components(self):
        """Test flow curve with viscosity and N₁."""
        model = TNTCates()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma, eta, N1 = model.predict_flow_curve(
            gamma_dot, return_components=True
        )

        assert sigma.shape == gamma_dot.shape
        assert eta.shape == gamma_dot.shape
        assert N1.shape == gamma_dot.shape
        assert np.all(eta > 0)
        assert np.all(N1 >= 0)

    @pytest.mark.smoke
    def test_newtonian_viscosity_constant(self):
        """Cates with constant breakage gives Newtonian viscosity."""
        model = TNTCates()
        G_0, tau_rep, tau_break = 1000.0, 4.0, 1.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)
        model.parameters.set_value("eta_s", 0.0)

        # τ_d = 2.0, η₀ = 1000 * 2.0 = 2000
        tau_d = np.sqrt(tau_rep * tau_break)
        expected_eta = G_0 * tau_d

        gamma_dot = np.logspace(-2, 2, 20)
        _, eta, _ = model.predict_flow_curve(
            gamma_dot, return_components=True
        )

        # Viscosity should be constant for all rates
        assert np.allclose(eta, expected_eta, rtol=1e-6)


# =============================================================================
# SAOS Tests
# =============================================================================


class TestSAOS:
    """Tests for SAOS predictions."""

    @pytest.mark.smoke
    def test_predict_saos(self):
        """Test SAOS prediction."""
        model = TNTCates()

        omega = np.logspace(-2, 2, 20)
        G_prime, G_double_prime = model.predict_saos(omega)

        assert G_prime.shape == omega.shape
        assert G_double_prime.shape == omega.shape
        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    def test_saos_magnitude(self):
        """Test |G*| prediction via test_mode='oscillation'."""
        model = TNTCates()

        omega = np.logspace(-2, 2, 20)
        G_star = model.predict(omega, test_mode="oscillation")

        G_prime, G_double_prime = model.predict_saos(omega)
        expected = np.sqrt(G_prime**2 + G_double_prime**2)

        assert np.allclose(G_star, expected)

    @pytest.mark.smoke
    def test_saos_terminal_scaling(self):
        """Test terminal regime scaling: G' ~ ω², G'' ~ ω."""
        model = TNTCates()
        model.parameters.set_value("G_0", 1000.0)
        model.parameters.set_value("tau_rep", 4.0)
        model.parameters.set_value("tau_break", 1.0)
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
        model = TNTCates()

        t = np.linspace(0, 5, 100)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_startup_monotonic(self):
        """Cates model should have monotonic startup (no overshoot)."""
        model = TNTCates()
        model.parameters.set_value("G_0", 1000.0)
        model.parameters.set_value("tau_rep", 4.0)
        model.parameters.set_value("tau_break", 1.0)
        model.parameters.set_value("eta_s", 0.0)

        t = np.linspace(0, 10, 200)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        # Check monotonically increasing (within tolerance for numerics)
        diffs = np.diff(sigma)
        assert np.all(diffs >= -1e-6 * np.max(np.abs(sigma)))

    @pytest.mark.smoke
    def test_startup_approaches_steady_state(self):
        """Startup should approach steady-state stress σ_ss = η₀·γ̇."""
        model = TNTCates()
        G_0, tau_rep, tau_break, eta_s = 1000.0, 4.0, 1.0, 0.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)
        model.parameters.set_value("eta_s", eta_s)

        gamma_dot = 10.0
        t = np.linspace(0, 20, 500)
        sigma = model.simulate_startup(t, gamma_dot=gamma_dot)

        # Analytical steady state: σ_ss = (G₀·τ_d + η_s)·γ̇
        tau_d = model.tau_d
        sigma_ss_expected = (G_0 * tau_d + eta_s) * gamma_dot

        # After ~10 τ_d, should be very close
        assert np.isclose(sigma[-1], sigma_ss_expected, rtol=0.01)

    def test_startup_full_return(self):
        """Test full conformation tensor return."""
        model = TNTCates()

        t = np.linspace(0, 5, 100)
        S_xx, S_yy, S_xy, S_zz = model.simulate_startup(
            t, gamma_dot=10.0, return_full=True
        )

        assert S_xx.shape == t.shape
        assert S_yy.shape == t.shape
        assert S_xy.shape == t.shape
        assert S_zz.shape == t.shape


# =============================================================================
# Relaxation Simulation Tests
# =============================================================================


class TestRelaxationSimulation:
    """Tests for stress relaxation simulation."""

    @pytest.mark.smoke
    def test_simulate_relaxation(self):
        """Test relaxation simulation runs."""
        model = TNTCates()

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=1.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_relaxation_decay(self):
        """Test stress decays during relaxation."""
        model = TNTCates()

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=1.0)

        assert sigma[0] > sigma[-1]
        assert sigma[-1] < sigma[0] * 0.01  # ~5 τ_d decay

    @pytest.mark.smoke
    def test_relaxation_single_exponential(self):
        """Verify single-exponential relaxation: σ(t) = G₀·τ_d·γ̇·exp(-t/τ_d)."""
        model = TNTCates()
        G_0, tau_rep, tau_break = 1000.0, 4.0, 1.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)

        # τ_d = √(4*1) = 2.0
        tau_d = model.tau_d
        gamma_dot_pre = 5.0

        t = np.linspace(0, 5, 100)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=gamma_dot_pre)

        # Analytical: σ(t) = G₀·τ_d·γ̇·exp(-t/τ_d)
        sigma_0 = G_0 * tau_d * gamma_dot_pre
        expected = sigma_0 * np.exp(-t / tau_d)

        assert np.allclose(sigma, expected, rtol=1e-6)


# =============================================================================
# Creep Simulation Tests
# =============================================================================


class TestCreepSimulation:
    """Tests for creep simulation."""

    @pytest.mark.smoke
    def test_simulate_creep(self):
        """Test creep simulation runs."""
        model = TNTCates()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))

    @pytest.mark.smoke
    def test_creep_strain_increases(self):
        """Test strain increases during creep."""
        model = TNTCates()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        assert gamma[-1] > gamma[0]

    def test_creep_with_rate(self):
        """Test creep with rate return."""
        model = TNTCates()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma, gamma_dot = model.simulate_creep(
            t, sigma_applied=50.0, return_rate=True
        )

        assert gamma.shape == t.shape
        assert gamma_dot.shape == t.shape

    def test_creep_steady_rate(self):
        """Test creep reaches steady shear rate γ̇ → σ/η₀."""
        model = TNTCates()
        G_0, tau_rep, tau_break, eta_s = 1000.0, 4.0, 1.0, 10.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)
        model.parameters.set_value("eta_s", eta_s)

        sigma_applied = 100.0
        t = np.linspace(0, 50, 500)
        gamma, gamma_dot = model.simulate_creep(
            t, sigma_applied=sigma_applied, return_rate=True
        )

        # Steady rate: γ̇_ss = σ / η₀
        eta_0 = model.eta_0
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
        model = TNTCates()

        result = model.simulate_laos(
            t=None, gamma_0=0.5, omega=1.0, n_cycles=3
        )

        assert "t" in result
        assert "strain" in result
        assert "stress" in result
        assert "strain_rate" in result

    def test_laos_periodicity(self):
        """Test LAOS response is periodic after transient."""
        model = TNTCates()

        result = model.simulate_laos(
            t=None, gamma_0=0.5, omega=1.0, n_cycles=5
        )

        stress = result["stress"]
        n_per_cycle = len(stress) // 5

        cycle_4_max = np.max(np.abs(stress[3 * n_per_cycle : 4 * n_per_cycle]))
        cycle_5_max = np.max(np.abs(stress[4 * n_per_cycle :]))

        assert np.isclose(cycle_4_max, cycle_5_max, rtol=0.05)

    def test_laos_small_amplitude_linear(self):
        """Small-amplitude LAOS should match SAOS (linear limit)."""
        model = TNTCates()
        G_0, tau_rep, tau_break, eta_s = 1000.0, 1.0, 1.0, 0.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)
        model.parameters.set_value("eta_s", eta_s)

        omega = 1.0
        gamma_0 = 0.001  # Very small amplitude

        result = model.simulate_laos(
            t=None, gamma_0=gamma_0, omega=omega, n_cycles=10
        )

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
        model = TNTCates()

        result = model.simulate_laos(
            t=None, gamma_0=1.0, omega=1.0, n_cycles=5
        )

        harmonics = model.extract_laos_harmonics(result, n_harmonics=3)

        assert "n" in harmonics
        assert "intensity" in harmonics
        assert len(harmonics["n"]) == 3
        assert harmonics["n"][0] == 1
        assert harmonics["n"][1] == 3


# =============================================================================
# Bayesian Interface Tests
# =============================================================================


class TestBayesianInterface:
    """Tests for BayesianMixin compatibility."""

    @pytest.mark.smoke
    def test_model_function_flow_curve(self):
        """Test model_function for flow curve."""
        model = TNTCates()

        X = jnp.logspace(-2, 2, 10)
        params = jnp.array([1000.0, 4.0, 1.0, 0.0])  # G_0, tau_rep, tau_break, eta_s

        y = model.model_function(X, params, test_mode="flow_curve")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))
        assert np.all(y > 0)

    @pytest.mark.smoke
    def test_model_function_saos(self):
        """Test model_function for SAOS."""
        model = TNTCates()

        X = jnp.logspace(-1, 2, 10)
        params = jnp.array([1000.0, 4.0, 1.0, 10.0])

        y = model.model_function(X, params, test_mode="oscillation")

        assert y.shape == X.shape
        assert np.all(y > 0)

    @pytest.mark.smoke
    def test_model_function_parameter_consistency(self):
        """Test model_function params match ParameterSet order."""
        model = TNTCates()
        G_0, tau_rep, tau_break, eta_s = 1000.0, 4.0, 1.0, 5.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_rep", tau_rep)
        model.parameters.set_value("tau_break", tau_break)
        model.parameters.set_value("eta_s", eta_s)

        X = jnp.logspace(-2, 2, 10)
        params = jnp.array([G_0, tau_rep, tau_break, eta_s])

        y_fn = model.model_function(X, params, test_mode="flow_curve")
        y_pred = model.predict(X, test_mode="flow_curve")

        assert np.allclose(y_fn, y_pred, rtol=1e-10)


# =============================================================================
# Physical Consistency Tests
# =============================================================================


class TestPhysicalConsistency:
    """Tests for physical correctness and consistency."""

    @pytest.mark.smoke
    def test_stress_positive_under_shear(self):
        """Test stress is positive for positive shear rate."""
        model = TNTCates()

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert np.all(sigma > 0)

    @pytest.mark.smoke
    def test_n1_positive(self):
        """Test first normal stress difference N₁ >= 0."""
        model = TNTCates()

        gamma_dot = np.logspace(-1, 2, 20)
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        assert np.all(N1 >= 0)
        assert np.allclose(N2, 0.0)  # UCM-like → N₂ = 0

    def test_conformation_positive_definite_startup(self):
        """Test conformation tensor stays positive definite during startup."""
        model = TNTCates()
        model.parameters.set_value("G_0", 1000.0)
        model.parameters.set_value("tau_rep", 4.0)
        model.parameters.set_value("tau_break", 1.0)

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

    @pytest.mark.smoke
    def test_relaxation_positive(self):
        """Test relaxation stress stays positive."""
        model = TNTCates()

        t = np.linspace(0, 10, 100)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=5.0)

        assert np.all(sigma >= 0)


# =============================================================================
# Analysis Method Tests
# =============================================================================


class TestAnalysisMethods:
    """Tests for analysis helper methods."""

    @pytest.mark.smoke
    def test_relaxation_spectrum(self):
        """Test relaxation spectrum G(t)."""
        model = TNTCates()
        model.parameters.set_value("G_0", 1000.0)
        model.parameters.set_value("tau_rep", 4.0)
        model.parameters.set_value("tau_break", 1.0)

        # τ_d = 2.0
        tau_d = model.tau_d

        t, G_t = model.get_relaxation_spectrum(n_points=50)

        assert len(t) == 50
        assert len(G_t) == 50
        assert G_t[0] > G_t[-1]

        # Should be single exponential: G(t) = G₀·exp(-t/τ_d)
        expected = 1000.0 * np.exp(-t / tau_d)
        assert np.allclose(G_t, expected, rtol=1e-6)


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Tests for model registry integration."""

    @pytest.mark.smoke
    def test_registry_create(self):
        """Test model creation via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("tnt_cates")
        assert isinstance(model, TNTCates)
