"""Tests for TNTStickyRouse model.

Tests cover:
- Instantiation and parameter management (n_modes >= 1)
- Sticker floor physics (tau_eff_k = max(tau_R_k, tau_s))
- Flow curve predictions
- SAOS predictions (multi-mode Maxwell with sticker constraint)
- ODE-based simulations (startup, relaxation, creep, LAOS)
- Bayesian interface (model_function)
- Physical consistency checks
- Derived properties (plateau modulus, zero-shear viscosity, terminal time)
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.tnt import TNTStickyRouse

jax, jnp = safe_import_jax()


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestInstantiation:
    """Tests for model instantiation and parameters."""

    @pytest.mark.smoke
    def test_default_instantiation(self):
        """Test model instantiates with default 3 modes and 8 parameters."""
        model = TNTStickyRouse()

        assert model.n_modes == 3
        assert model.tau_s == 0.1
        assert model.eta_s == 0.0

        # 3 modes: G_0, tau_R_0, G_1, tau_R_1, G_2, tau_R_2, tau_s, eta_s
        assert len(list(model.parameters.keys())) == 8

    @pytest.mark.smoke
    def test_n_modes_property_returns_correct_value(self):
        """Test n_modes property returns the correct number of modes."""
        model = TNTStickyRouse(n_modes=5)
        assert model.n_modes == 5

    @pytest.mark.smoke
    def test_parameter_setting_works(self):
        """Test parameters can be set."""
        model = TNTStickyRouse(n_modes=2)

        model.parameters.set_value("G_0", 500.0)
        model.parameters.set_value("tau_R_0", 5.0)
        model.parameters.set_value("G_1", 300.0)
        model.parameters.set_value("tau_R_1", 0.5)
        model.parameters.set_value("tau_s", 0.2)
        model.parameters.set_value("eta_s", 10.0)

        assert model.parameters.get_value("G_0") == 500.0
        assert model.parameters.get_value("tau_R_0") == 5.0
        assert model.parameters.get_value("G_1") == 300.0
        assert model.parameters.get_value("tau_R_1") == 0.5
        assert model.tau_s == 0.2
        assert model.eta_s == 10.0

    @pytest.mark.smoke
    def test_n_modes_at_least_one_enforced(self):
        """Test n_modes >= 1 is enforced."""
        with pytest.raises(ValueError, match="n_modes must be >= 1"):
            TNTStickyRouse(n_modes=0)

        with pytest.raises(ValueError, match="n_modes must be >= 1"):
            TNTStickyRouse(n_modes=-1)

    def test_single_mode(self):
        """Test single-mode instantiation (N=1, 4 params)."""
        model = TNTStickyRouse(n_modes=1)

        assert model.n_modes == 1
        # G_0, tau_R_0, tau_s, eta_s
        assert len(list(model.parameters.keys())) == 4

    def test_five_modes(self):
        """Test five-mode instantiation (N=5, 12 params)."""
        model = TNTStickyRouse(n_modes=5)

        assert model.n_modes == 5
        # G_0, tau_R_0, ..., G_4, tau_R_4, tau_s, eta_s
        assert len(list(model.parameters.keys())) == 12

    def test_default_mode_spacing(self):
        """Test default Rouse times are logarithmically spaced."""
        model = TNTStickyRouse(n_modes=3)

        tau_R_0 = model.parameters.get_value("tau_R_0")
        tau_R_1 = model.parameters.get_value("tau_R_1")
        tau_R_2 = model.parameters.get_value("tau_R_2")

        # Default: tau_R_k = 10^(1-k)
        assert np.isclose(tau_R_0, 10.0)
        assert np.isclose(tau_R_1, 1.0)
        assert np.isclose(tau_R_2, 0.1)

    def test_default_modulus_equal_weight(self):
        """Test default modulus is equally split across modes."""
        model = TNTStickyRouse(n_modes=3)

        G_0 = model.parameters.get_value("G_0")
        G_1 = model.parameters.get_value("G_1")
        G_2 = model.parameters.get_value("G_2")

        # Default: G_k = 1e3 / n_modes
        expected_G = 1e3 / 3
        assert np.isclose(G_0, expected_G)
        assert np.isclose(G_1, expected_G)
        assert np.isclose(G_2, expected_G)


# =============================================================================
# Sticker Floor Physics Tests
# =============================================================================


class TestStickerFloorPhysics:
    """Tests for sticker floor constraint: tau_eff_k = max(tau_R_k, tau_s)."""

    @pytest.mark.smoke
    def test_effective_times_sticker_floor(self):
        """Test effective times are max(tau_R_k, tau_s)."""
        model = TNTStickyRouse(n_modes=3)

        # Default tau_s = 0.1, tau_R = [10.0, 1.0, 0.1]
        tau_eff = model.get_effective_times()

        # Mode 0: max(10.0, 0.1) = 10.0
        # Mode 1: max(1.0, 0.1) = 1.0
        # Mode 2: max(0.1, 0.1) = 0.1
        assert np.isclose(tau_eff[0], 10.0)
        assert np.isclose(tau_eff[1], 1.0)
        assert np.isclose(tau_eff[2], 0.1)

    @pytest.mark.smoke
    def test_sticker_floor_limits_fast_modes(self):
        """When tau_s > tau_R_k, sticker limits relaxation."""
        model = TNTStickyRouse(n_modes=3)
        model.parameters.set_value("tau_s", 5.0)  # Longer than tau_R_1 and tau_R_2

        tau_eff = model.get_effective_times()

        # Mode 0: max(10.0, 5.0) = 10.0
        # Mode 1: max(1.0, 5.0) = 5.0 ← sticker-limited
        # Mode 2: max(0.1, 5.0) = 5.0 ← sticker-limited
        assert np.isclose(tau_eff[0], 10.0)
        assert np.isclose(tau_eff[1], 5.0)
        assert np.isclose(tau_eff[2], 5.0)

    def test_rouse_limit_when_tau_s_small(self):
        """When tau_s ≪ all tau_R_k, model reduces to multi-mode Maxwell."""
        model = TNTStickyRouse(n_modes=3)
        model.parameters.set_value("tau_s", 1e-6)  # Very small (lower bound)

        tau_eff = model.get_effective_times()

        # All modes should relax at their natural Rouse times
        tau_R_0 = model.parameters.get_value("tau_R_0")
        tau_R_1 = model.parameters.get_value("tau_R_1")
        tau_R_2 = model.parameters.get_value("tau_R_2")

        assert np.isclose(tau_eff[0], tau_R_0)
        assert np.isclose(tau_eff[1], tau_R_1)
        assert np.isclose(tau_eff[2], tau_R_2)


# =============================================================================
# Flow Curve Tests
# =============================================================================


class TestFlowCurve:
    """Tests for flow curve (steady shear) predictions."""

    @pytest.mark.smoke
    def test_predict_flow_curve_correct_shape(self):
        """Test flow curve prediction returns correct shape."""
        model = TNTStickyRouse()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_flow_curve_positive(self):
        """Test flow curve stress is positive."""
        model = TNTStickyRouse()

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert np.all(sigma > 0)

    @pytest.mark.smoke
    def test_flow_curve_finite(self):
        """Test flow curve stress is finite."""
        model = TNTStickyRouse()

        gamma_dot = np.logspace(-2, 3, 30)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert np.all(np.isfinite(sigma))

    def test_newtonian_at_low_rates(self):
        """Test flow curve approaches Newtonian behavior at low shear rates.

        At low Wi, σ → η₀·γ̇ where η₀ = Σ G_k·τ_eff_k + η_s.
        """
        model = TNTStickyRouse(n_modes=3)
        G_total = 1e3  # Default: 3 modes × 1e3/3 each
        tau_eff = model.get_effective_times()
        eta_s = model.eta_s

        # Compute expected zero-shear viscosity
        eta_0_expected = np.sum(
            [
                model.parameters.get_value(f"G_{k}") * tau_eff[k]
                for k in range(model.n_modes)
            ]
        ) + eta_s

        # Low shear rate
        gamma_dot_low = 1e-4
        sigma = model.predict(np.array([gamma_dot_low]), test_mode="flow_curve")

        # At low rates: σ ≈ η₀·γ̇
        expected_stress = eta_0_expected * gamma_dot_low
        assert np.isclose(sigma[0], expected_stress, rtol=0.01)


# =============================================================================
# SAOS Tests
# =============================================================================


class TestSAOS:
    """Tests for SAOS predictions."""

    @pytest.mark.smoke
    def test_predict_oscillation_correct_shapes(self):
        """Test SAOS prediction returns correct shapes."""
        model = TNTStickyRouse()

        omega = np.logspace(-2, 2, 20)
        G_star = model.predict(omega, test_mode="oscillation")

        assert G_star.shape == omega.shape
        assert np.all(np.isfinite(G_star))

    @pytest.mark.smoke
    def test_saos_moduli_positive(self):
        """Test G' and G'' are positive."""
        model = TNTStickyRouse()

        omega = np.logspace(-2, 2, 20)
        G_star = model.predict(omega, test_mode="oscillation")

        # G* = G' + i·G'', both should be positive
        G_prime = np.real(G_star)
        G_double_prime = np.imag(G_star)

        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    def test_multimode_broader_plateau_than_single_mode(self):
        """Multi-mode should give broader plateau than single mode.

        The multi-mode spectrum creates a plateau region where multiple
        modes contribute, unlike a single-mode Maxwell which has a sharp
        crossover.
        """
        model_multi = TNTStickyRouse(n_modes=3)
        model_multi.parameters.set_value("tau_s", 1.0)

        omega = np.logspace(-2, 2, 50)
        G_star = model_multi.predict(omega, test_mode="oscillation")
        G_prime = np.real(G_star)

        # Check for plateau: G' should be relatively flat over some range
        # Find frequency range where G' varies less than 20%
        G_prime_normalized = G_prime / np.max(G_prime)
        plateau_mask = G_prime_normalized > 0.8

        # Multi-mode should have at least a few points in plateau region
        assert np.sum(plateau_mask) >= 3


# =============================================================================
# Startup Simulation Tests
# =============================================================================


class TestStartupSimulation:
    """Tests for startup flow simulation."""

    @pytest.mark.smoke
    def test_simulate_startup_runs(self):
        """Test startup simulation runs without error."""
        model = TNTStickyRouse()

        t = np.linspace(0, 5, 100)
        sigma = model.predict(t, test_mode="startup", gamma_dot=10.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_startup_stress_correct_shape(self):
        """Test startup returns correct shape."""
        model = TNTStickyRouse()

        t = np.linspace(0, 10, 200)
        sigma = model.predict(t, test_mode="startup", gamma_dot=5.0)

        assert sigma.shape == t.shape

    @pytest.mark.smoke
    def test_startup_stress_finite(self):
        """Test startup stress is finite."""
        model = TNTStickyRouse()

        t = np.linspace(0, 10, 200)
        sigma = model.predict(t, test_mode="startup", gamma_dot=5.0)

        assert np.all(np.isfinite(sigma))

    def test_startup_approaches_steady_state(self):
        """Test startup approaches steady-state stress.

        At long times, σ(t) → σ_ss = η₀·γ̇ where η₀ = Σ G_k·τ_eff_k + η_s.
        """
        model = TNTStickyRouse(n_modes=3)
        gamma_dot = 1.0

        # Long time simulation
        t = np.linspace(0, 50, 500)
        sigma = model.predict(t, test_mode="startup", gamma_dot=gamma_dot)

        # Compare to analytical steady state
        sigma_ss = model.predict(np.array([gamma_dot]), test_mode="flow_curve")[0]

        # After ~10 τ_terminal, should be close
        assert np.isclose(sigma[-1], sigma_ss, rtol=0.05)


# =============================================================================
# Relaxation Simulation Tests
# =============================================================================


class TestRelaxationSimulation:
    """Tests for stress relaxation simulation."""

    @pytest.mark.smoke
    def test_relaxation_runs(self):
        """Test relaxation simulation runs without error."""
        model = TNTStickyRouse()

        t = np.linspace(0, 5, 50)
        sigma = model.predict(t, test_mode="relaxation")

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_relaxation_stress_correct_shape(self):
        """Test relaxation returns correct shape."""
        model = TNTStickyRouse()

        t = np.linspace(0, 10, 100)
        sigma = model.predict(t, test_mode="relaxation")

        assert sigma.shape == t.shape

    @pytest.mark.smoke
    def test_relaxation_stress_finite(self):
        """Test relaxation stress is finite."""
        model = TNTStickyRouse()

        t = np.linspace(0, 10, 100)
        sigma = model.predict(t, test_mode="relaxation")

        assert np.all(np.isfinite(sigma))

    def test_stress_decays(self):
        """Test stress decays during relaxation."""
        model = TNTStickyRouse()

        t = np.linspace(0, 20, 200)
        sigma = model.predict(t, test_mode="relaxation")

        # Stress should decay
        assert sigma[0] > sigma[-1]
        # Should decay significantly (multi-exponential)
        assert sigma[-1] < sigma[0] * 0.1


# =============================================================================
# Creep Simulation Tests
# =============================================================================


class TestCreepSimulation:
    """Tests for creep simulation."""

    @pytest.mark.smoke
    def test_simulate_creep_runs(self):
        """Test creep simulation runs without error."""
        model = TNTStickyRouse()
        model.parameters.set_value("eta_s", 10.0)  # Need eta_s > 0

        t = np.linspace(0, 10, 100)
        gamma = model.predict(t, test_mode="creep", sigma_applied=50.0)

        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))

    @pytest.mark.smoke
    def test_creep_strain_correct_shape(self):
        """Test creep returns correct shape."""
        model = TNTStickyRouse()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.predict(t, test_mode="creep", sigma_applied=50.0)

        assert gamma.shape == t.shape

    @pytest.mark.smoke
    def test_creep_strain_finite(self):
        """Test creep strain is finite."""
        model = TNTStickyRouse()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.predict(t, test_mode="creep", sigma_applied=50.0)

        assert np.all(np.isfinite(gamma))

    def test_strain_increases(self):
        """Test strain increases during creep."""
        model = TNTStickyRouse()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 20, 200)
        gamma = model.predict(t, test_mode="creep", sigma_applied=100.0)

        # Strain should increase monotonically
        assert gamma[-1] > gamma[0]
        # Check mostly monotonic (within numerical tolerance)
        diffs = np.diff(gamma)
        assert np.sum(diffs >= 0) > 0.95 * len(diffs)

    def test_creep_needs_eta_s_positive(self):
        """Test creep with eta_s > 0 works properly."""
        model = TNTStickyRouse()
        # Ensure eta_s > 0 for well-posed creep ODE
        model.parameters.set_value("eta_s", 1.0)

        t = np.linspace(0, 10, 100)
        gamma = model.predict(t, test_mode="creep", sigma_applied=50.0)

        # Should run without error and produce physical results
        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))


# =============================================================================
# LAOS Simulation Tests
# =============================================================================


class TestLAOSSimulation:
    """Tests for LAOS simulation."""

    @pytest.mark.smoke
    def test_laos_runs(self):
        """Test LAOS simulation runs without error."""
        model = TNTStickyRouse()

        t = np.linspace(0, 10, 200)
        sigma = model.predict(t, test_mode="laos", gamma_0=0.5, omega=1.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    def test_laos_returns_expected_structure(self):
        """Test LAOS simulation returns expected output."""
        model = TNTStickyRouse()

        t = np.linspace(0, 10, 200)
        sigma = model.predict(t, test_mode="laos", gamma_0=0.5, omega=1.0)

        # Should return stress array
        assert isinstance(sigma, np.ndarray)
        assert sigma.shape == t.shape

    def test_laos_trajectory_stored(self):
        """Test LAOS stores trajectory data."""
        model = TNTStickyRouse()

        t = np.linspace(0, 10, 200)
        sigma = model.predict(t, test_mode="laos", gamma_0=0.5, omega=1.0)

        # Check trajectory was stored
        trajectory = model.get_trajectory()
        assert trajectory is not None
        assert "time" in trajectory
        assert "stress" in trajectory
        assert "strain" in trajectory


# =============================================================================
# Bayesian Interface Tests
# =============================================================================


class TestBayesianInterface:
    """Tests for BayesianMixin compatibility."""

    @pytest.mark.smoke
    def test_model_function_flow_curve(self):
        """Test model_function for flow curve."""
        model = TNTStickyRouse(n_modes=2)

        X = jnp.logspace(-2, 2, 10)
        # params = [G_0, tau_R_0, G_1, tau_R_1, tau_s, eta_s]
        params = jnp.array([500.0, 10.0, 500.0, 1.0, 0.1, 0.0])

        y = model.model_function(X, params, test_mode="flow_curve")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))
        assert np.all(y > 0)

    def test_model_function_oscillation(self):
        """Test model_function for SAOS."""
        model = TNTStickyRouse(n_modes=2)

        X = jnp.logspace(-1, 2, 10)
        params = jnp.array([500.0, 10.0, 500.0, 1.0, 0.1, 5.0])

        y = model.model_function(X, params, test_mode="oscillation")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))

    @pytest.mark.smoke
    def test_model_function_parameter_order_consistency(self):
        """Test model_function params match ParameterSet order.

        Parameter order: [G_0, tau_R_0, G_1, tau_R_1, ..., tau_s, eta_s]
        """
        model = TNTStickyRouse(n_modes=2)
        G_0, tau_R_0 = 500.0, 10.0
        G_1, tau_R_1 = 300.0, 1.0
        tau_s, eta_s = 0.2, 5.0

        model.parameters.set_value("G_0", G_0)
        model.parameters.set_value("tau_R_0", tau_R_0)
        model.parameters.set_value("G_1", G_1)
        model.parameters.set_value("tau_R_1", tau_R_1)
        model.parameters.set_value("tau_s", tau_s)
        model.parameters.set_value("eta_s", eta_s)

        X = jnp.logspace(-2, 2, 10)
        params = jnp.array([G_0, tau_R_0, G_1, tau_R_1, tau_s, eta_s])

        y_fn = model.model_function(X, params, test_mode="flow_curve")
        y_pred = model.predict(X, test_mode="flow_curve")

        assert np.allclose(y_fn, y_pred, rtol=1e-8)

    def test_model_function_three_modes(self):
        """Test model_function with default 3 modes (8 params)."""
        model = TNTStickyRouse(n_modes=3)

        X = jnp.logspace(-1, 2, 10)
        # params = [G_0, tau_R_0, G_1, tau_R_1, G_2, tau_R_2, tau_s, eta_s]
        params = jnp.array([
            333.33, 10.0,  # Mode 0
            333.33, 1.0,   # Mode 1
            333.33, 0.1,   # Mode 2
            0.1,           # tau_s
            0.0            # eta_s
        ])

        y = model.model_function(X, params, test_mode="flow_curve")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))


# =============================================================================
# Physical Consistency Tests
# =============================================================================


class TestPhysicalConsistency:
    """Tests for physical correctness and consistency."""

    @pytest.mark.smoke
    def test_stress_positive_under_shear(self):
        """Test stress is positive for positive shear rate."""
        model = TNTStickyRouse()

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert np.all(sigma > 0)

    def test_plateau_modulus_positive(self):
        """Test plateau modulus is positive when sticker floor is active."""
        model = TNTStickyRouse(n_modes=3)
        # Set tau_s > some tau_R_k so modes become sticker-limited
        model.parameters.set_value("tau_s", 5.0)  # > tau_R_1=1.0, tau_R_2=0.1

        G_plateau = model.predict_plateau_modulus()

        assert G_plateau > 0
        assert np.isfinite(G_plateau)

    def test_zero_shear_viscosity_positive(self):
        """Test zero-shear viscosity is positive."""
        model = TNTStickyRouse()

        eta_0 = model.predict_zero_shear_viscosity()

        assert eta_0 > 0
        assert np.isfinite(eta_0)

    def test_terminal_time_positive(self):
        """Test terminal time is positive."""
        model = TNTStickyRouse()

        tau_terminal = model.predict_terminal_time()

        assert tau_terminal > 0
        assert np.isfinite(tau_terminal)

    def test_normal_stress_positive(self):
        """Test first normal stress difference is positive."""
        model = TNTStickyRouse()

        gamma_dot = np.logspace(-1, 2, 20)
        N1 = model.predict_normal_stress_difference(gamma_dot)

        assert np.all(N1 > 0)
        assert np.all(np.isfinite(N1))

    def test_plateau_modulus_sum_of_sticker_limited_modes(self):
        """Test plateau modulus equals sum of modes where tau_R < tau_s.

        Plateau is formed by modes dominated by sticker lifetime.
        """
        model = TNTStickyRouse(n_modes=3)
        model.parameters.set_value("tau_s", 2.0)  # Between tau_R_0 and tau_R_1

        # tau_R = [10.0, 1.0, 0.1], tau_s = 2.0
        # Modes with tau_R < tau_s: modes 1 and 2
        G_1 = model.parameters.get_value("G_1")
        G_2 = model.parameters.get_value("G_2")
        expected_plateau = G_1 + G_2

        G_plateau = model.predict_plateau_modulus()

        assert np.isclose(G_plateau, expected_plateau)

    def test_zero_shear_viscosity_sum_rule(self):
        """Test η₀ = Σ G_k·τ_eff_k + η_s."""
        model = TNTStickyRouse(n_modes=3)
        model.parameters.set_value("eta_s", 10.0)

        tau_eff = model.get_effective_times()
        expected_eta_0 = sum(
            model.parameters.get_value(f"G_{k}") * tau_eff[k]
            for k in range(model.n_modes)
        ) + model.eta_s

        eta_0 = model.predict_zero_shear_viscosity()

        assert np.isclose(eta_0, expected_eta_0)

    def test_terminal_time_is_max_effective_time(self):
        """Test terminal time is the longest effective relaxation time."""
        model = TNTStickyRouse(n_modes=3)

        tau_eff = model.get_effective_times()
        expected_terminal = np.max(tau_eff)

        tau_terminal = model.predict_terminal_time()

        assert np.isclose(tau_terminal, expected_terminal)


# =============================================================================
# Derived Properties Tests
# =============================================================================


class TestDerivedProperties:
    """Tests for derived property methods."""

    def test_predict_plateau_modulus_no_sticker_floor(self):
        """When tau_s very small, plateau modulus should be near zero.

        With no sticker floor, fast modes relax at their natural rate,
        no plateau forms.
        """
        model = TNTStickyRouse(n_modes=3)
        model.parameters.set_value("tau_s", 1e-6)  # Very small (lower bound)

        G_plateau = model.predict_plateau_modulus()

        # All modes have tau_R > tau_s, so plateau modulus should be zero
        # (no modes are sticker-limited)
        assert G_plateau == 0.0

    def test_predict_plateau_modulus_all_modes_limited(self):
        """When tau_s very large, all modes contribute to plateau."""
        model = TNTStickyRouse(n_modes=3)
        model.parameters.set_value("tau_s", 100.0)  # Larger than all tau_R

        G_plateau = model.predict_plateau_modulus()

        # All modes have tau_R < tau_s, so all contribute
        G_total = sum(
            model.parameters.get_value(f"G_{k}") for k in range(model.n_modes)
        )

        assert np.isclose(G_plateau, G_total)

    def test_predict_zero_shear_viscosity_consistency(self):
        """Test η₀ from method matches low-rate flow curve."""
        model = TNTStickyRouse(n_modes=3)

        eta_0 = model.predict_zero_shear_viscosity()

        # Compare to very low shear rate
        gamma_dot_low = 1e-6
        sigma = model.predict(np.array([gamma_dot_low]), test_mode="flow_curve")[0]
        eta_from_flow = sigma / gamma_dot_low

        assert np.isclose(eta_0, eta_from_flow, rtol=0.01)

    def test_predict_normal_stress_scalar_and_array(self):
        """Test N1 prediction works for both scalar and array input."""
        model = TNTStickyRouse()

        # Scalar
        N1_scalar = model.predict_normal_stress_difference(1.0)
        assert np.isscalar(N1_scalar) or N1_scalar.shape == ()

        # Array
        gamma_dot = np.array([0.1, 1.0, 10.0])
        N1_array = model.predict_normal_stress_difference(gamma_dot)
        assert N1_array.shape == gamma_dot.shape


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Tests for model registry integration."""

    @pytest.mark.smoke
    def test_registry_create(self):
        """Test model creation via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("tnt_sticky_rouse")
        assert isinstance(model, TNTStickyRouse)

    def test_registry_create(self):
        """Test model creation via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("tnt_sticky_rouse")
        assert isinstance(model, TNTStickyRouse)


# =============================================================================
# Fitting Tests
# =============================================================================


class TestFitting:
    """Tests for model fitting."""

    @pytest.mark.smoke
    def test_fit_flow_curve(self):
        """Test fitting to flow curve data."""
        model_true = TNTStickyRouse(n_modes=2)
        model_true.parameters.set_value("G_0", 600.0)
        model_true.parameters.set_value("tau_R_0", 5.0)
        model_true.parameters.set_value("G_1", 400.0)
        model_true.parameters.set_value("tau_R_1", 0.5)
        model_true.parameters.set_value("tau_s", 0.2)
        model_true.parameters.set_value("eta_s", 5.0)

        gamma_dot = np.logspace(-1, 2, 20)
        sigma_true = model_true.predict(gamma_dot, test_mode="flow_curve")

        np.random.seed(42)
        sigma_noisy = sigma_true * (1 + 0.02 * np.random.randn(len(sigma_true)))

        model_fit = TNTStickyRouse(n_modes=2)
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
    """Tests for __repr__ method."""

    def test_repr_contains_key_info(self):
        """Test __repr__ contains n_modes, tau_s, and G_plateau."""
        model = TNTStickyRouse(n_modes=3)

        repr_str = repr(model)

        assert "n_modes=3" in repr_str
        assert "tau_s" in repr_str
        assert "G_plateau" in repr_str
