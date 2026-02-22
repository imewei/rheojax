"""Tests for TNTMultiSpecies model.

Tests cover:
- Instantiation and parameter management (2-species, 3-species)
- Maxwell limit verification (multi-mode analytical)
- Flow curve predictions
- SAOS predictions
- ODE-based simulations (startup, relaxation, creep, LAOS)
- BayesianMixin interface (model_function)
- Physical consistency checks
- Multi-species specific features
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.tnt import TNTMultiSpecies

jax, jnp = safe_import_jax()


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestInstantiation:
    """Tests for model instantiation and parameters."""

    @pytest.mark.smoke
    def test_default_2_species(self):
        """Test default instantiation creates 2-species model with 5 parameters."""
        model = TNTMultiSpecies()

        assert model.n_species == 2
        assert len(list(model.parameters.keys())) == 5

        # Check parameter names: G_0, tau_b_0, G_1, tau_b_1, eta_s
        param_names = list(model.parameters.keys())
        assert "G_0" in param_names
        assert "tau_b_0" in param_names
        assert "G_1" in param_names
        assert "tau_b_1" in param_names
        assert "eta_s" in param_names

    @pytest.mark.smoke
    def test_default_3_species(self):
        """Test 3-species model has 7 parameters."""
        model = TNTMultiSpecies(n_species=3)

        assert model.n_species == 3
        assert len(list(model.parameters.keys())) == 7

        param_names = list(model.parameters.keys())
        assert "G_0" in param_names
        assert "G_1" in param_names
        assert "G_2" in param_names
        assert "tau_b_2" in param_names

    @pytest.mark.smoke
    def test_parameter_setting(self):
        """Test parameters can be set."""
        model = TNTMultiSpecies(n_species=2)

        model.parameters.set_value("G_0", 500.0)
        model.parameters.set_value("tau_b_0", 0.1)
        model.parameters.set_value("G_1", 500.0)
        model.parameters.set_value("tau_b_1", 1.0)
        model.parameters.set_value("eta_s", 10.0)

        assert float(model.parameters.get_value("G_0")) == 500.0
        assert float(model.parameters.get_value("tau_b_0")) == 0.1
        assert model.eta_s == 10.0

    def test_n_species_must_be_positive(self):
        """Test n_species >= 1 is enforced."""
        with pytest.raises(ValueError, match="n_species must be ≥ 1"):
            TNTMultiSpecies(n_species=0)

    @pytest.mark.smoke
    def test_derived_properties(self):
        """Test derived properties (G_total, eta_0)."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("G_0", 500.0)
        model.parameters.set_value("tau_b_0", 0.1)
        model.parameters.set_value("G_1", 500.0)
        model.parameters.set_value("tau_b_1", 1.0)
        model.parameters.set_value("eta_s", 10.0)

        # G_total = 500 + 500 = 1000
        assert model.G_total == 1000.0

        # η₀ = Σ G_i·τ_b_i + η_s = 500*0.1 + 500*1.0 + 10 = 50 + 500 + 10 = 560
        assert model.eta_0 == 560.0

    def test_default_parameter_distribution(self):
        """Test default parameters have logarithmic spacing for tau_b."""
        model = TNTMultiSpecies(n_species=3)

        # Default tau_b_i = 10^(-i)
        assert float(model.parameters.get_value("tau_b_0")) == 1.0
        assert float(model.parameters.get_value("tau_b_1")) == 0.1
        assert float(model.parameters.get_value("tau_b_2")) == 0.01


# =============================================================================
# Maxwell Limit Tests (Multi-Mode Analytical)
# =============================================================================


class TestMaxwellLimit:
    """Verify TNTMultiSpecies recovers multi-mode Maxwell (generalized UCM).

    For constant breakage, each species evolves independently and the
    total response is a superposition of N Maxwell modes.
    """

    @pytest.mark.smoke
    def test_single_species_equals_single_mode(self):
        """Single species (N=1) should behave identically to single Maxwell mode."""
        model = TNTMultiSpecies(n_species=1)
        G, tau_b, eta_s = 1000.0, 1.0, 5.0
        model.parameters.set_value("G_0", G)
        model.parameters.set_value("tau_b_0", tau_b)
        model.parameters.set_value("eta_s", eta_s)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # Single Maxwell mode: σ = (G·τ_b + η_s)·γ̇
        eta_0 = G * tau_b + eta_s
        expected = eta_0 * gamma_dot

        assert np.allclose(sigma, expected, rtol=1e-10)

    @pytest.mark.smoke
    def test_steady_stress_superposition(self):
        """Verify σ = Σ(G_i·τ_b_i)·γ̇ + η_s·γ̇ for 2 species."""
        model = TNTMultiSpecies(n_species=2)
        G_0, tau_b_0 = 500.0, 0.1
        G_1, tau_b_1 = 500.0, 1.0
        eta_s = 10.0

        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_b_0", tau_b_0)
        model.parameters.set_value("G_1", G_1)
        model.parameters.set_value("tau_b_1", tau_b_1)
        model.parameters.set_value("eta_s", eta_s)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # Multi-mode UCM (conformation tensor): σ = Σ G_i·τ_b_i·γ̇ + η_s·γ̇ = η₀·γ̇
        eta_0 = G_0 * tau_b_0 + G_1 * tau_b_1 + eta_s
        expected = eta_0 * gamma_dot

        assert np.allclose(sigma, expected, rtol=1e-10)

    @pytest.mark.smoke
    def test_saos_moduli_superposition(self):
        """Verify G'(ω) and G''(ω) match multi-mode Maxwell formulas."""
        model = TNTMultiSpecies(n_species=2)
        G_0, tau_b_0 = 400.0, 0.1
        G_1, tau_b_1 = 600.0, 1.0
        eta_s = 2.0

        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_b_0", tau_b_0)
        model.parameters.set_value("G_1", G_1)
        model.parameters.set_value("tau_b_1", tau_b_1)
        model.parameters.set_value("eta_s", eta_s)

        omega = np.logspace(-2, 2, 30)
        G_prime, G_double_prime = model.predict_saos(omega)

        # Multi-mode Maxwell:
        # G'(ω) = Σ G_i·(ωτ_i)²/(1+(ωτ_i)²)
        # G''(ω) = Σ G_i·(ωτ_i)/(1+(ωτ_i)²) + η_s·ω
        wt_0 = omega * tau_b_0
        wt_1 = omega * tau_b_1

        expected_Gp = G_0 * wt_0**2 / (1 + wt_0**2) + G_1 * wt_1**2 / (1 + wt_1**2)
        expected_Gpp = (
            G_0 * wt_0 / (1 + wt_0**2) + G_1 * wt_1 / (1 + wt_1**2) + eta_s * omega
        )

        assert np.allclose(G_prime, expected_Gp, rtol=1e-10)
        assert np.allclose(G_double_prime, expected_Gpp, rtol=1e-10)

    def test_broader_spectrum_than_single_mode(self):
        """2-species should show broader relaxation spectrum than 1-species."""
        model_1 = TNTMultiSpecies(n_species=1)
        model_1.parameters.set_value("G_0", 1000.0)
        model_1.parameters.set_value("tau_b_0", 1.0)
        model_1.parameters.set_value("eta_s", 0.0)

        model_2 = TNTMultiSpecies(n_species=2)
        model_2.parameters.set_value("G_0", 500.0)
        model_2.parameters.set_value("tau_b_0", 0.1)
        model_2.parameters.set_value("G_1", 500.0)
        model_2.parameters.set_value("tau_b_1", 10.0)
        model_2.parameters.set_value("eta_s", 0.0)

        omega = np.logspace(-2, 2, 50)
        G_p_1, G_pp_1 = model_1.predict_saos(omega)
        G_p_2, G_pp_2 = model_2.predict_saos(omega)

        # tan(δ) = G''/G'
        tan_delta_1 = G_pp_1 / np.maximum(G_p_1, 1e-12)
        tan_delta_2 = G_pp_2 / np.maximum(G_p_2, 1e-12)

        # Two modes should have broader tan(δ) peak
        peak_width_2 = np.sum(tan_delta_2 > 0.5 * np.max(tan_delta_2))
        peak_width_1 = np.sum(tan_delta_1 > 0.5 * np.max(tan_delta_1))

        assert peak_width_2 >= peak_width_1


# =============================================================================
# Flow Curve Tests
# =============================================================================


class TestFlowCurve:
    """Tests for flow curve (steady shear) predictions."""

    @pytest.mark.smoke
    def test_predict_flow_curve(self):
        """Test flow curve prediction returns correct shape, positive, finite."""
        model = TNTMultiSpecies(n_species=2)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_predict_flow_curve_method(self):
        """Test direct predict_flow_curve method."""
        model = TNTMultiSpecies(n_species=2)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape

    def test_flow_curve_with_components(self):
        """Test flow curve with viscosity and N1."""
        model = TNTMultiSpecies(n_species=2)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma, eta, N1 = model.predict_flow_curve(gamma_dot, return_components=True)

        assert sigma.shape == gamma_dot.shape
        assert eta.shape == gamma_dot.shape
        assert N1.shape == gamma_dot.shape
        assert np.all(eta > 0)
        assert np.all(N1 >= 0)

    @pytest.mark.smoke
    def test_newtonian_at_low_rates(self):
        """Multi-mode with constant breakage should be Newtonian (constant viscosity)."""
        model = TNTMultiSpecies(n_species=2)
        G_0, tau_b_0 = 500.0, 0.1
        G_1, tau_b_1 = 500.0, 1.0
        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_b_0", tau_b_0)
        model.parameters.set_value("G_1", G_1)
        model.parameters.set_value("tau_b_1", tau_b_1)
        model.parameters.set_value("eta_s", 0.0)

        gamma_dot = np.logspace(-2, 2, 20)
        _, eta, _ = model.predict_flow_curve(gamma_dot, return_components=True)

        # η₀ = Σ G_i·τ_b_i
        expected_eta_0 = G_0 * tau_b_0 + G_1 * tau_b_1

        # All viscosities should be constant (multi-mode UCM is Newtonian for constant breakage)
        assert np.allclose(eta, expected_eta_0, rtol=1e-6)


# =============================================================================
# SAOS Tests
# =============================================================================


class TestSAOS:
    """Tests for SAOS predictions."""

    @pytest.mark.smoke
    def test_predict_saos(self):
        """Test SAOS prediction returns correct shapes, positive."""
        model = TNTMultiSpecies(n_species=2)

        omega = np.logspace(-2, 2, 20)
        G_prime, G_double_prime = model.predict_saos(omega)

        assert G_prime.shape == omega.shape
        assert G_double_prime.shape == omega.shape
        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    def test_saos_magnitude(self):
        """Test |G*| prediction via test_mode='oscillation'."""
        model = TNTMultiSpecies(n_species=2)

        omega = np.logspace(-2, 2, 20)
        G_star = model.predict(omega, test_mode="oscillation")

        G_prime, G_double_prime = model.predict_saos(omega)
        expected = np.sqrt(G_prime**2 + G_double_prime**2)

        assert np.allclose(G_star, expected)

    @pytest.mark.smoke
    def test_saos_terminal_scaling(self):
        """Test terminal regime scaling: G' ~ ω², G'' ~ ω."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("G_0", 500.0)
        model.parameters.set_value("tau_b_0", 1.0)
        model.parameters.set_value("G_1", 500.0)
        model.parameters.set_value("tau_b_1", 10.0)
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


# =============================================================================
# Startup Simulation Tests
# =============================================================================


class TestStartupSimulation:
    """Tests for startup flow simulation."""

    @pytest.mark.smoke
    def test_simulate_startup(self):
        """Test startup simulation runs and returns correct shape, finite."""
        model = TNTMultiSpecies(n_species=2)

        t = np.linspace(0, 5, 100)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_startup_approaches_steady_state(self):
        """Startup should approach steady-state stress at long times."""
        model = TNTMultiSpecies(n_species=2)
        G_0, tau_b_0 = 500.0, 0.1
        G_1, tau_b_1 = 500.0, 1.0
        eta_s = 0.0

        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_b_0", tau_b_0)
        model.parameters.set_value("G_1", G_1)
        model.parameters.set_value("tau_b_1", tau_b_1)
        model.parameters.set_value("eta_s", eta_s)

        gamma_dot = 10.0
        t = np.linspace(0, 20, 500)
        sigma = model.simulate_startup(t, gamma_dot=gamma_dot)

        # Analytical steady state (UCM conformation): σ_ss = Σ G_i·τ_b_i·γ̇ + η_s·γ̇
        sigma_ss_expected = (G_0 * tau_b_0 + G_1 * tau_b_1 + eta_s) * gamma_dot

        assert np.isclose(sigma[-1], sigma_ss_expected, rtol=0.02)

    def test_startup_full_return(self):
        """Test full conformation tensor return for all modes."""
        model = TNTMultiSpecies(n_species=2)

        t = np.linspace(0, 5, 100)
        result = model.simulate_startup(t, gamma_dot=10.0, return_full=True)

        # Each component should have shape (T, N)
        assert result["S_xx"].shape == (len(t), 2)
        assert result["S_yy"].shape == (len(t), 2)
        assert result["S_xy"].shape == (len(t), 2)
        assert result["S_zz"].shape == (len(t), 2)


# =============================================================================
# Relaxation Simulation Tests
# =============================================================================


class TestRelaxationSimulation:
    """Tests for stress relaxation simulation."""

    @pytest.mark.smoke
    def test_simulate_relaxation(self):
        """Test relaxation simulation runs and returns correct shape, finite."""
        model = TNTMultiSpecies(n_species=2)

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=1.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_relaxation_decay(self):
        """Test stress decays during relaxation (σ[0] > σ[-1])."""
        model = TNTMultiSpecies(n_species=2)

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=1.0)

        assert sigma[0] > sigma[-1]
        assert sigma[-1] < sigma[0] * 0.05

    def test_relaxation_multi_exponential(self):
        """Verify multi-exponential relaxation for 2 species."""
        model = TNTMultiSpecies(n_species=2)
        G_0, tau_b_0 = 500.0, 0.5
        G_1, tau_b_1 = 500.0, 5.0

        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_b_0", tau_b_0)
        model.parameters.set_value("G_1", G_1)
        model.parameters.set_value("tau_b_1", tau_b_1)
        model.parameters.set_value("eta_s", 0.0)

        gamma_dot_pre = 5.0
        t = np.linspace(0, 10, 100)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=gamma_dot_pre)

        # Analytical: σ(t) = Σ σ₀_i·exp(-t/τ_b_i)
        # σ₀_i = G_i·τ_b_i·γ̇ / (1 + (τ_b_i·γ̇)²)
        wi_0 = tau_b_0 * gamma_dot_pre
        wi_1 = tau_b_1 * gamma_dot_pre
        sigma_0_mode_0 = G_0 * wi_0 / (1.0 + wi_0**2)
        sigma_0_mode_1 = G_1 * wi_1 / (1.0 + wi_1**2)

        expected = sigma_0_mode_0 * np.exp(-t / tau_b_0) + sigma_0_mode_1 * np.exp(
            -t / tau_b_1
        )

        assert np.allclose(sigma, expected, rtol=1e-6)


# =============================================================================
# Creep Simulation Tests
# =============================================================================


class TestCreepSimulation:
    """Tests for creep simulation."""

    @pytest.mark.smoke
    def test_simulate_creep(self):
        """Test creep simulation runs and returns correct shape, finite."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))

    @pytest.mark.smoke
    def test_creep_strain_increases(self):
        """Test strain increases during creep."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        assert gamma[-1] > gamma[0]

    def test_creep_with_rate(self):
        """Test creep with rate return."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma, gamma_dot = model.simulate_creep(t, sigma_applied=50.0, return_rate=True)

        assert gamma.shape == t.shape
        assert gamma_dot.shape == t.shape

    def test_creep_requires_positive_eta_s(self):
        """Creep simulation needs eta_s > 0 to be well-posed."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        # Should run without numerical issues
        assert np.all(np.isfinite(gamma))


# =============================================================================
# LAOS Simulation Tests
# =============================================================================


class TestLAOSSimulation:
    """Tests for LAOS simulation."""

    @pytest.mark.smoke
    def test_simulate_laos(self):
        """Test LAOS simulation runs and returns dict with expected keys."""
        model = TNTMultiSpecies(n_species=2)

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)

        assert "t" in result
        assert "strain" in result
        assert "stress" in result
        assert "strain_rate" in result

    def test_laos_periodicity(self):
        """Test LAOS response is periodic after transient."""
        model = TNTMultiSpecies(n_species=2)

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=5)

        stress = result["stress"]
        n_per_cycle = len(stress) // 5

        cycle_4_max = np.max(np.abs(stress[3 * n_per_cycle : 4 * n_per_cycle]))
        cycle_5_max = np.max(np.abs(stress[4 * n_per_cycle :]))

        assert np.isclose(cycle_4_max, cycle_5_max, rtol=0.1)

    def test_laos_harmonics(self):
        """Test LAOS harmonic extraction."""
        model = TNTMultiSpecies(n_species=2)

        result = model.simulate_laos(t=None, gamma_0=1.0, omega=1.0, n_cycles=5)

        harmonics = model.extract_laos_harmonics(result, n_harmonics=3)

        assert "n" in harmonics
        assert "intensity" in harmonics
        assert len(harmonics["n"]) == 3


# =============================================================================
# Bayesian Interface Tests
# =============================================================================


class TestBayesianInterface:
    """Tests for BayesianMixin compatibility."""

    @pytest.mark.smoke
    def test_model_function_flow_curve_2_species(self):
        """Test model_function for flow curve with 2 species (5 params)."""
        model = TNTMultiSpecies(n_species=2)

        X = jnp.logspace(-2, 2, 10)
        # params = [G_0, tau_b_0, G_1, tau_b_1, eta_s]
        params = jnp.array([500.0, 0.1, 500.0, 1.0, 0.0])

        y = model.model_function(X, params, test_mode="flow_curve")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))

    def test_model_function_saos(self):
        """Test model_function for SAOS."""
        model = TNTMultiSpecies(n_species=2)

        X = jnp.logspace(-1, 2, 10)
        params = jnp.array([500.0, 0.1, 500.0, 1.0, 10.0])

        y = model.model_function(X, params, test_mode="oscillation")

        assert y.shape == (len(X), 2)
        assert np.all(y > 0)

    @pytest.mark.smoke
    def test_model_function_parameter_order(self):
        """Test model_function params match ParameterSet.keys() order."""
        model = TNTMultiSpecies(n_species=2)
        G_0, tau_b_0 = 500.0, 0.1
        G_1, tau_b_1 = 500.0, 1.0
        eta_s = 5.0

        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_b_0", tau_b_0)
        model.parameters.set_value("G_1", G_1)
        model.parameters.set_value("tau_b_1", tau_b_1)
        model.parameters.set_value("eta_s", eta_s)

        X = jnp.logspace(-2, 2, 10)
        # Parameter order: [G_0, tau_b_0, G_1, tau_b_1, eta_s]
        params = jnp.array([G_0, tau_b_0, G_1, tau_b_1, eta_s])

        y_fn = model.model_function(X, params, test_mode="flow_curve")
        y_pred = model.predict(X, test_mode="flow_curve")

        assert np.allclose(y_fn, y_pred, rtol=1e-10)

    def test_model_function_3_species(self):
        """Test model_function for 3 species (7 params)."""
        model = TNTMultiSpecies(n_species=3)

        X = jnp.logspace(-2, 2, 10)
        # params = [G_0, tau_b_0, G_1, tau_b_1, G_2, tau_b_2, eta_s]
        params = jnp.array([300.0, 0.01, 400.0, 0.1, 300.0, 1.0, 0.0])

        y = model.model_function(X, params, test_mode="flow_curve")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))


# =============================================================================
# Physical Consistency Tests
# =============================================================================


class TestPhysicalConsistency:
    """Tests for physical correctness and consistency."""

    @pytest.mark.smoke
    def test_equilibrium_conformation_multimode(self):
        """Test equilibrium conformation for 2 modes is [1,1,1,0, 1,1,1,0]."""
        model = TNTMultiSpecies(n_species=2)
        S_eq = model.get_equilibrium_conformation_multimode()

        # 2 modes: 4*2 = 8 components
        expected = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        assert np.allclose(S_eq, expected)

    @pytest.mark.smoke
    def test_stress_positive_under_shear(self):
        """Test stress is positive for positive shear rate."""
        model = TNTMultiSpecies(n_species=2)

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert np.all(sigma > 0)

    def test_n1_positive(self):
        """Test N₁ >= 0 for multi-mode UCM."""
        model = TNTMultiSpecies(n_species=2)

        gamma_dot = np.logspace(-1, 2, 20)
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        assert np.all(N1 >= 0)
        assert np.allclose(N2, 0.0)

    def test_relaxation_positive(self):
        """Test relaxation stress stays positive."""
        model = TNTMultiSpecies(n_species=2)

        t = np.linspace(0, 10, 100)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=5.0)

        assert np.all(sigma >= 0)

    def test_conformation_positive_definite_startup(self):
        """Test conformation tensor stays positive definite during startup."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("G_0", 500.0)
        model.parameters.set_value("tau_b_0", 0.1)
        model.parameters.set_value("G_1", 500.0)
        model.parameters.set_value("tau_b_1", 1.0)

        t = np.linspace(0, 10, 200)
        result = model.simulate_startup(t, gamma_dot=10.0, return_full=True)

        S_xx = result["S_xx"]
        S_yy = result["S_yy"]
        S_xy = result["S_xy"]

        # Check for both modes
        for mode in range(2):
            assert np.all(S_xx[:, mode] > 0)
            assert np.all(S_yy[:, mode] > 0)

            # Positive definite: S_xx·S_yy > S_xy²
            det_2d = S_xx[:, mode] * S_yy[:, mode] - S_xy[:, mode] ** 2
            assert np.all(det_2d > 0)


# =============================================================================
# Analysis Method Tests
# =============================================================================


class TestAnalysisMethods:
    """Tests for analysis helper methods."""

    @pytest.mark.smoke
    def test_relaxation_spectrum(self):
        """Test relaxation modulus G(t) for multi-species."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("G_0", 500.0)
        model.parameters.set_value("tau_b_0", 0.5)
        model.parameters.set_value("G_1", 500.0)
        model.parameters.set_value("tau_b_1", 5.0)

        t, G_t = model.get_relaxation_spectrum(n_points=50)

        assert len(t) == 50
        assert len(G_t) == 50
        assert G_t[0] > G_t[-1]

        # Should be sum of two exponentials
        expected = 500.0 * np.exp(-t / 0.5) + 500.0 * np.exp(-t / 5.0)
        assert np.allclose(G_t, expected, rtol=1e-6)

    def test_3_species_relaxation_spectrum(self):
        """Test relaxation spectrum for 3 species."""
        model = TNTMultiSpecies(n_species=3)
        model.parameters.set_value("G_0", 300.0)
        model.parameters.set_value("tau_b_0", 0.1)
        model.parameters.set_value("G_1", 400.0)
        model.parameters.set_value("tau_b_1", 1.0)
        model.parameters.set_value("G_2", 300.0)
        model.parameters.set_value("tau_b_2", 10.0)

        t, G_t = model.get_relaxation_spectrum(n_points=100)

        # Should be sum of three exponentials
        expected = (
            300.0 * np.exp(-t / 0.1)
            + 400.0 * np.exp(-t / 1.0)
            + 300.0 * np.exp(-t / 10.0)
        )
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

        model = ModelRegistry.create("tnt_multi_species")
        assert isinstance(model, TNTMultiSpecies)
        assert model.n_species == 2

    def test_registry_with_kwargs(self):
        """Test registry creation with n_species argument."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("tnt_multi_species", n_species=3)
        assert isinstance(model, TNTMultiSpecies)
        assert model.n_species == 3


# =============================================================================
# Fitting Tests
# =============================================================================


class TestFitting:
    """Tests for model fitting."""

    @pytest.mark.smoke
    def test_fit_flow_curve(self):
        """Test fitting to flow curve data."""
        model_true = TNTMultiSpecies(n_species=2)
        model_true.parameters.set_value("G_0", 600.0)
        model_true.parameters.set_value("tau_b_0", 0.2)
        model_true.parameters.set_value("G_1", 400.0)
        model_true.parameters.set_value("tau_b_1", 2.0)
        model_true.parameters.set_value("eta_s", 5.0)

        gamma_dot = np.logspace(-1, 2, 20)
        sigma_true = model_true.predict(gamma_dot, test_mode="flow_curve")

        np.random.seed(42)
        sigma_noisy = sigma_true * (1 + 0.02 * np.random.randn(len(sigma_true)))

        model_fit = TNTMultiSpecies(n_species=2)
        model_fit.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        sigma_pred = model_fit.predict(gamma_dot, test_mode="flow_curve")
        r2 = 1 - np.sum((sigma_noisy - sigma_pred) ** 2) / np.sum(
            (sigma_noisy - np.mean(sigma_noisy)) ** 2
        )

        assert r2 > 0.90


# =============================================================================
# String Representation Tests
# =============================================================================


class TestStringRepresentation:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ includes n_species, G_total, tau_range, eta_s."""
        model = TNTMultiSpecies(n_species=2)
        model.parameters.set_value("G_0", 500.0)
        model.parameters.set_value("tau_b_0", 0.1)
        model.parameters.set_value("G_1", 500.0)
        model.parameters.set_value("tau_b_1", 10.0)
        model.parameters.set_value("eta_s", 5.0)

        repr_str = repr(model)

        assert "n_species=2" in repr_str
        assert "G_total=1.00e+03" in repr_str
        assert "tau_range=[1.00e-01, 1.00e+01]" in repr_str
        assert "η_s=5.00e+00" in repr_str


# =============================================================================
# F-TNT-001 Regression: LAOS model_function kwargs
# =============================================================================


class TestLAOSKwargsRegression:
    """Regression tests for F-TNT-001: LAOS branch must use kwargs, not self._."""

    @pytest.mark.smoke
    def test_model_function_laos_uses_kwargs(self):
        """Test that model_function LAOS branch respects kwargs over self._."""
        model = TNTMultiSpecies(n_species=2)
        # Set self._ to sentinel values
        model._gamma_0 = 999.0
        model._omega_laos = 999.0

        X = jnp.linspace(0.01, 10.0, 50)
        # [G_0, tau_b_0, G_1, tau_b_1, eta_s]
        params = jnp.array([500.0, 0.1, 500.0, 10.0, 5.0])

        # Pass different values via kwargs — these should be used, not self._
        y_kwargs = model.model_function(
            X, params, test_mode="laos", gamma_0=0.1, omega=1.0
        )
        assert y_kwargs.shape == X.shape
        assert np.all(np.isfinite(y_kwargs))

    def test_model_function_laos_kwargs_differ_from_self(self):
        """Test that kwargs-passed gamma_0/omega actually override self._."""
        model = TNTMultiSpecies(n_species=2)
        X = jnp.linspace(0.01, 6.0, 30)
        params = jnp.array([500.0, 0.1, 500.0, 10.0, 5.0])

        # First call: store via self._
        model._gamma_0 = 0.01
        model._omega_laos = 1.0
        y_small = model.model_function(X, params, test_mode="laos")

        # Second call: override via kwargs with larger amplitude
        y_large = model.model_function(
            X, params, test_mode="laos", gamma_0=1.0, omega=1.0
        )

        # Larger amplitude should produce larger peak stress
        assert np.max(np.abs(y_large)) > np.max(np.abs(y_small))
