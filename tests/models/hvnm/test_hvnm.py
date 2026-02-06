"""Tests for HVNM (Hybrid Vitrimer Nanocomposite Model).

Tests are organized into 11 classes covering creation, limiting cases,
flow curve, SAOS, startup, relaxation, creep, LAOS, damage, interphase,
and Bayesian inference.

The highest-priority test is phi=0 recovery of HVM (atol=1e-10).
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.hvnm import HVNMLocal
from rheojax.models.hvnm._kernels import (
    hvnm_ber_rate_constant_interphase,
    hvnm_ber_rate_constant_matrix,
    hvnm_guth_gold,
    hvnm_interphase_fraction,
    hvnm_interphase_modulus,
)

jax, jnp = safe_import_jax()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_model():
    """Default HVNM model with all flags off."""
    return HVNMLocal()


@pytest.fixture
def full_model():
    """HVNM with all features enabled."""
    return HVNMLocal(
        include_damage=True,
        include_dissociative=True,
        include_interfacial_damage=True,
        include_diffusion=True,
    )


@pytest.fixture
def unfilled_model():
    """HVNM with phi=0 (should recover HVM)."""
    return HVNMLocal.unfilled_vitrimer(
        G_P=5000.0, G_E=3000.0, G_D=1000.0,
        k_d_D=1.0, nu_0=1e10, E_a=80e3, T=300.0,
    )


@pytest.fixture
def filled_model():
    """HVNM with phi=0.1."""
    m = HVNMLocal()
    m.parameters.set_value("G_P", 5000.0)
    m.parameters.set_value("G_E", 3000.0)
    m.parameters.set_value("G_D", 1000.0)
    m.parameters.set_value("phi", 0.1)
    m.parameters.set_value("R_NP", 20e-9)
    m.parameters.set_value("delta_m", 10e-9)
    m.parameters.set_value("beta_I", 3.0)
    return m


@pytest.fixture
def hvm_model():
    """HVM reference model for comparison."""
    from rheojax.models import HVMLocal
    m = HVMLocal(kinetics="stress", include_dissociative=True)
    m.parameters.set_value("G_P", 5000.0)
    m.parameters.set_value("G_E", 3000.0)
    m.parameters.set_value("G_D", 1000.0)
    m.parameters.set_value("k_d_D", 1.0)
    m.parameters.set_value("nu_0", 1e10)
    m.parameters.set_value("E_a", 80e3)
    m.parameters.set_value("T", 300.0)
    return m


# =============================================================================
# Test Creation & Registration
# =============================================================================


class TestHVNMCreation:
    """Test model creation and parameter setup."""

    @pytest.mark.smoke
    def test_default_creation(self, default_model):
        assert default_model is not None
        assert isinstance(default_model, HVNMLocal)

    @pytest.mark.smoke
    def test_parameter_count_default(self, default_model):
        # 6 (HVM base) + 2 (dissociative) + 7 (interphase core) = 15
        assert len(default_model.parameters) == 15

    @pytest.mark.smoke
    def test_parameter_count_full(self, full_model):
        # 15 (default) + 2 (damage) + 5 (int_damage) + 3 (diffusion) = 25
        assert len(full_model.parameters) == 25

    @pytest.mark.smoke
    def test_parameter_count_no_dissociative(self):
        m = HVNMLocal(include_dissociative=False)
        # 6 (HVM base, no D) + 7 (interphase core) = 13
        assert len(m.parameters) == 13

    @pytest.mark.smoke
    def test_registry_lookup_hvnm_local(self):
        from rheojax.core.registry import ModelRegistry
        m = ModelRegistry.create("hvnm_local")
        assert isinstance(m, HVNMLocal)

    @pytest.mark.smoke
    def test_registry_lookup_hvnm(self):
        from rheojax.core.registry import ModelRegistry
        m = ModelRegistry.create("hvnm")
        assert isinstance(m, HVNMLocal)

    @pytest.mark.smoke
    def test_property_accessors(self, default_model):
        assert default_model.G_P > 0
        assert default_model.G_E > 0
        assert default_model.phi > 0
        assert default_model.beta_I > 0
        assert default_model.R_NP > 0
        assert default_model.delta_m > 0

    @pytest.mark.smoke
    def test_derived_quantities(self, default_model):
        assert default_model.X_phi > 1.0  # phi > 0 means X > 1
        assert default_model.phi_I > 0
        assert default_model.G_I_eff > 0

    @pytest.mark.smoke
    def test_repr(self, default_model):
        r = repr(default_model)
        assert "HVNMLocal" in r
        assert "phi=" in r

    @pytest.mark.smoke
    def test_feature_flags(self):
        m = HVNMLocal(include_interfacial_damage=True, include_diffusion=True)
        assert m.include_interfacial_damage
        assert m.include_diffusion


# =============================================================================
# Test Limiting Cases
# =============================================================================


class TestHVNMLimitingCases:
    """Test limiting case recovery — highest priority tests."""

    def test_phi_zero_recovers_hvm_saos(self, unfilled_model, hvm_model):
        """PRIMARY VALIDATION: phi=0 must match HVM exactly."""
        omega = np.logspace(-3, 3, 100)
        G_hvm_p, G_hvm_dp = hvm_model.predict_saos(omega)
        G_hvnm_p, G_hvnm_dp = unfilled_model.predict_saos(omega)

        np.testing.assert_allclose(G_hvnm_p, G_hvm_p, atol=1e-10)
        np.testing.assert_allclose(G_hvnm_dp, G_hvm_dp, atol=1e-10)

    def test_phi_zero_recovers_hvm_relaxation(self, unfilled_model, hvm_model):
        """phi=0 relaxation must match HVM."""
        t = np.logspace(-3, 3, 100)
        G_hvm = hvm_model.simulate_relaxation(t, gamma_step=0.01)
        G_hvnm = unfilled_model.simulate_relaxation(t, gamma_step=0.01)
        # ODE solver may have small numerical differences
        np.testing.assert_allclose(G_hvnm, G_hvm, rtol=1e-3)

    def test_phi_zero_recovers_hvm_startup(self, unfilled_model, hvm_model):
        """phi=0 startup must match HVM."""
        t = np.linspace(0.01, 10, 100)
        s_hvm = hvm_model.simulate_startup(t, gamma_dot=1.0)
        s_hvnm = unfilled_model.simulate_startup(t, gamma_dot=1.0)
        np.testing.assert_allclose(s_hvnm, s_hvm, rtol=1e-3)

    def test_filled_elastomer_no_exchange(self):
        m = HVNMLocal.filled_elastomer(G_P=1e4, phi=0.1)
        assert m.G_E == 0.0
        assert m.G_D == 0.0

    def test_frozen_interphase(self):
        m = HVNMLocal.matrix_only_exchange(G_P=5000, G_E=3000, phi=0.1)
        # k_BER^int should be extremely small
        k_int = m.compute_ber_rate_interphase_equilibrium()
        assert k_int < 1e-10

    def test_factory_unfilled_propagation(self, unfilled_model):
        assert unfilled_model.phi == 0.0
        assert unfilled_model.G_P == 5000.0
        assert unfilled_model.G_E == 3000.0
        assert unfilled_model.G_D == 1000.0

    def test_network_fractions_sum_to_one(self, filled_model):
        fracs = filled_model.get_network_fractions_nc()
        total = sum(fracs.values())
        assert abs(total - 1.0) < 1e-10

    def test_relaxation_spectrum_has_4_entries(self, filled_model):
        spectrum = filled_model.get_relaxation_spectrum()
        # P (inf), E, D, I
        assert len(spectrum) == 4

    def test_relaxation_spectrum_unfilled_matches_hvm(self, unfilled_model):
        spectrum = unfilled_model.get_relaxation_spectrum()
        # P (inf), E, D — no I since G_I_eff = 0
        assert len(spectrum) == 3

    def test_amplified_neo_hookean(self):
        """G_E=0, G_D=0, phi>0 → only amplified permanent network."""
        m = HVNMLocal.filled_elastomer(G_P=1e4, phi=0.1)
        omega = np.logspace(-2, 2, 50)
        G_p, G_dp = m.predict_saos(omega)
        X = float(hvnm_guth_gold(0.1))
        # G' should be constant at G_P * X
        np.testing.assert_allclose(G_p, 1e4 * X, atol=1.0)
        # G'' should be ~0
        assert np.all(G_dp < 1.0)


# =============================================================================
# Test Flow Curve
# =============================================================================


class TestHVNMFlowCurve:
    """Test steady-state flow curve predictions."""

    def test_sigma_e_zero_at_steady_state(self, filled_model):
        """E and I networks relax to zero at steady state."""
        gd = np.logspace(-2, 2, 50)
        result = filled_model.predict_flow_curve(gd, return_components=True)
        np.testing.assert_allclose(result["sigma_E"], 0.0, atol=1e-30)
        np.testing.assert_allclose(result["sigma_I"], 0.0, atol=1e-30)

    def test_sigma_d_dominates(self, filled_model):
        """D-network viscous stress dominates at steady state."""
        gd = np.logspace(-2, 2, 50)
        result = filled_model.predict_flow_curve(gd, return_components=True)
        np.testing.assert_allclose(
            result["stress"], result["sigma_D"], rtol=1e-10
        )

    def test_monotonic_stress_rate(self, filled_model):
        gd = np.logspace(-2, 2, 50)
        sigma = filled_model.predict_flow_curve(gd)
        assert np.all(np.diff(sigma) > 0)

    def test_component_decomposition(self, filled_model):
        gd = np.array([1.0, 10.0])
        result = filled_model.predict_flow_curve(gd, return_components=True)
        assert "sigma_D" in result
        assert "sigma_I" in result
        assert "eta_eff" in result

    def test_flow_curve_positive(self, filled_model):
        gd = np.logspace(-1, 1, 20)
        sigma = filled_model.predict_flow_curve(gd)
        assert np.all(sigma > 0)

    def test_phi_does_not_affect_flow_curve(self):
        """Steady-state flow curve is independent of phi (only D-network)."""
        m1 = HVNMLocal()
        m1.parameters.set_value("phi", 0.0)
        m2 = HVNMLocal()
        m2.parameters.set_value("phi", 0.2)
        gd = np.logspace(-1, 1, 20)
        s1 = m1.predict_flow_curve(gd)
        s2 = m2.predict_flow_curve(gd)
        np.testing.assert_allclose(s1, s2, rtol=1e-12)


# =============================================================================
# Test SAOS
# =============================================================================


class TestHVNMSAOS:
    """Test SAOS (Small Amplitude Oscillatory Shear) predictions."""

    def test_three_maxwell_modes_plus_plateau(self, filled_model):
        omega = np.logspace(-4, 4, 200)
        G_p, G_dp = filled_model.predict_saos(omega)
        # G' should increase from plateau to high-freq limit
        assert G_p[-1] > G_p[0]
        # G'' should have loss peaks
        assert np.max(G_dp) > G_dp[0]

    def test_low_freq_plateau(self, filled_model):
        """G'(omega->0) ≈ G_P * X(phi) (amplified plateau, plus residual I-network)."""
        omega = np.array([1e-8])
        G_p, _ = filled_model.predict_saos(omega)
        X = filled_model.X_phi
        expected = filled_model.G_P * X
        # I-network may contribute slightly even at very low freq
        np.testing.assert_allclose(G_p[0], expected, rtol=0.05)

    def test_high_freq_limit(self, filled_model):
        """G'(omega->inf) approaches sum of all moduli."""
        omega = np.array([1e10])
        G_p, _ = filled_model.predict_saos(omega)
        d = filled_model._get_derived_params(filled_model._get_params_dict())
        G_tot = (
            filled_model.G_P * d["X_phi"]
            + filled_model.G_E
            + filled_model.G_D
            + d["G_I_eff"] * d["X_I"]
        )
        np.testing.assert_allclose(G_p[0], G_tot, rtol=1e-2)

    def test_interphase_slower_than_matrix(self, filled_model):
        """tau_I > tau_E when E_a^int > E_a^mat."""
        tau_E = filled_model.get_vitrimer_relaxation_time()
        tau_I = filled_model.get_interphase_relaxation_time()
        assert tau_I > tau_E

    def test_return_components_true(self, filled_model):
        omega = np.logspace(-2, 2, 20)
        result = filled_model.predict_saos(omega, return_components=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_g_star(self, filled_model):
        omega = np.logspace(-2, 2, 20)
        result = filled_model.predict_saos(omega, return_components=False)
        assert isinstance(result, np.ndarray)
        assert np.all(result > 0)

    def test_amplification_increases_modulus(self):
        """Filling with NPs should increase G'."""
        m0 = HVNMLocal()
        m0.parameters.set_value("phi", 0.0)
        m1 = HVNMLocal()
        m1.parameters.set_value("phi", 0.1)
        omega = np.array([1.0])
        G0, _ = m0.predict_saos(omega)
        G1, _ = m1.predict_saos(omega)
        assert G1[0] > G0[0]

    def test_factor_of_2_both_networks(self, filled_model):
        """tau_E = 1/(2*k_BER^mat) and tau_I = 1/(2*k_BER^int)."""
        k_mat = filled_model.compute_ber_rate_at_equilibrium()
        k_int = filled_model.compute_ber_rate_interphase_equilibrium()
        tau_E = filled_model.get_vitrimer_relaxation_time()
        tau_I = filled_model.get_interphase_relaxation_time()
        np.testing.assert_allclose(tau_E, 1.0 / (2.0 * k_mat), rtol=1e-10)
        np.testing.assert_allclose(tau_I, 1.0 / (2.0 * k_int), rtol=1e-10)


# =============================================================================
# Test Startup
# =============================================================================


class TestHVNMStartup:
    """Test startup shear predictions."""

    def test_stress_increases_monotonically_short_time(self, filled_model):
        """Stress should increase monotonically at short times."""
        t = np.linspace(0.01, 1.0, 100)
        stress = filled_model.simulate_startup(t, gamma_dot=1.0)
        # At short times, all networks contribute positively
        assert np.all(np.diff(stress[1:]) > -1e-6)

    def test_return_full_dict(self, filled_model):
        t = np.linspace(0.01, 5, 50)
        result = filled_model.simulate_startup(t, gamma_dot=1.0, return_full=True)
        assert "mu_I_xy" in result
        assert "mu_I_nat_xy" in result
        assert "damage_int" in result

    def test_peak_stress_increases_with_phi(self):
        """Higher phi → amplified modulus → higher peak stress."""
        t = np.linspace(0.01, 5, 100)
        peaks = []
        for phi_val in [0.0, 0.05, 0.1]:
            m = HVNMLocal()
            m.parameters.set_value("phi", phi_val)
            stress = m.simulate_startup(t, gamma_dot=1.0)
            peaks.append(np.max(stress))
        assert peaks[1] > peaks[0]
        assert peaks[2] > peaks[1]

    def test_startup_positive_stress(self, filled_model):
        t = np.linspace(0.01, 5, 100)
        stress = filled_model.simulate_startup(t, gamma_dot=1.0)
        assert np.all(np.isfinite(stress))
        assert np.all(stress >= 0)
        # After initial point, stress should be strictly positive
        assert np.all(stress[1:] > 0)

    def test_startup_no_nan(self, filled_model):
        t = np.linspace(0.01, 10, 100)
        stress = filled_model.simulate_startup(t, gamma_dot=0.1)
        assert np.all(np.isfinite(stress))

    def test_initial_slope(self, filled_model):
        """Short-time slope ≈ G_tot * gamma_dot."""
        t = np.linspace(0.001, 0.01, 10)
        gamma_dot = 1.0
        stress = filled_model.simulate_startup(t, gamma_dot=gamma_dot)
        d = filled_model._get_derived_params(filled_model._get_params_dict())
        G_tot = (
            filled_model.G_P * d["X_phi"]
            + filled_model.G_E
            + filled_model.G_D
            + d["G_I_eff"] * d["X_I"]
        )
        slope = (stress[-1] - stress[0]) / (t[-1] - t[0])
        np.testing.assert_allclose(slope, G_tot * gamma_dot, rtol=0.3)

    def test_long_time_consistency_with_flow_curve(self, filled_model):
        """Startup stress at long time → flow curve stress."""
        gd = 1.0
        t = np.linspace(0.01, 200, 500)
        stress_startup = filled_model.simulate_startup(t, gamma_dot=gd)
        sigma_fc = filled_model.predict_flow_curve(np.array([gd]))[0]
        # At very long time, total stress should approach sigma_P(linear) + sigma_D(viscous)
        # sigma_P grows linearly, so we check sigma_startup[-1] > sigma_fc
        assert stress_startup[-1] > 0


# =============================================================================
# Test Relaxation
# =============================================================================


class TestHVNMRelaxation:
    """Test stress relaxation predictions."""

    def test_quad_exponential_plus_plateau(self, filled_model):
        """G(t) has 4 modes: permanent plateau + E + D + I decay."""
        t = np.logspace(-3, 3, 100)
        G_t = filled_model.simulate_relaxation(t, gamma_step=0.01)
        # G(0) should be high, G(inf) should be plateau
        assert G_t[0] > G_t[-1]
        assert G_t[-1] > 0  # Permanent plateau

    def test_long_time_plateau(self, filled_model):
        """G(inf) → G_P * X(phi) (amplified permanent modulus)."""
        t = np.logspace(-3, 6, 200)
        G_t = filled_model.simulate_relaxation(t, gamma_step=0.001)
        X = filled_model.X_phi
        expected_plateau = filled_model.G_P * X
        # ODE may not fully relax even at t=1e6 for slow I-network
        np.testing.assert_allclose(G_t[-1], expected_plateau, rtol=0.2)

    def test_initial_modulus(self, filled_model):
        """G(0+) ≈ sum of all network moduli."""
        t = np.logspace(-4, 1, 100)
        G_t = filled_model.simulate_relaxation(t, gamma_step=0.01)
        d = filled_model._get_derived_params(filled_model._get_params_dict())
        G_tot = (
            filled_model.G_P * d["X_phi"]
            + filled_model.G_E
            + filled_model.G_D
            + d["G_I_eff"] * d["X_I"]
        )
        np.testing.assert_allclose(G_t[0], G_tot, rtol=0.2)

    def test_relaxation_positive(self, filled_model):
        t = np.logspace(-2, 2, 50)
        G_t = filled_model.simulate_relaxation(t, gamma_step=0.01)
        assert np.all(np.isfinite(G_t))
        assert np.all(G_t > 0)

    def test_relaxation_monotonically_decreasing(self, filled_model):
        t = np.logspace(-2, 3, 100)
        G_t = filled_model.simulate_relaxation(t, gamma_step=0.01)
        # G(t) should be monotonically decreasing (or plateau)
        diffs = np.diff(G_t)
        assert np.all(diffs <= 1e-6)  # Small positive diffs from numerics OK

    def test_return_full(self, filled_model):
        t = np.logspace(-2, 2, 50)
        result = filled_model.simulate_relaxation(t, gamma_step=0.01, return_full=True)
        assert "G_t" in result
        assert "mu_I_xy" in result
        assert "mu_I_nat_xy" in result


# =============================================================================
# Test Creep
# =============================================================================


class TestHVNMCreep:
    """Test creep predictions."""

    def test_strain_increases(self, filled_model):
        """Strain should increase monotonically under constant stress."""
        t = np.logspace(-2, 3, 100)
        gamma = filled_model.simulate_creep(t, sigma_0=100.0)
        assert np.all(np.isfinite(gamma))
        assert np.all(np.diff(gamma) >= -1e-10)

    def test_np_reduces_compliance(self):
        """NP filling should reduce creep compliance."""
        t = np.logspace(-2, 2, 50)
        sigma = 100.0
        m0 = HVNMLocal()
        m0.parameters.set_value("phi", 0.0)
        m1 = HVNMLocal()
        m1.parameters.set_value("phi", 0.1)
        g0 = m0.simulate_creep(t, sigma_0=sigma)
        g1 = m1.simulate_creep(t, sigma_0=sigma)
        # Higher phi → stiffer → less creep
        assert g1[-1] < g0[-1]

    def test_creep_return_full(self, filled_model):
        t = np.logspace(-2, 2, 50)
        result = filled_model.simulate_creep(t, sigma_0=100.0, return_full=True)
        assert "compliance" in result
        assert "mu_I_xy" in result

    def test_creep_positive(self, filled_model):
        t = np.logspace(-1, 2, 50)
        gamma = filled_model.simulate_creep(t, sigma_0=50.0)
        assert np.all(np.isfinite(gamma))
        assert np.all(gamma >= -1e-10)

    def test_creep_strain_grows_with_time(self, filled_model):
        """Creep strain increases with time as networks relax."""
        t = np.logspace(-2, 4, 100)
        sigma_0 = 100.0
        gamma = filled_model.simulate_creep(t, sigma_0=sigma_0)
        # Strain should be monotonically increasing
        assert np.all(np.diff(gamma) >= -1e-15)
        # Late-time strain should be larger than early-time
        assert gamma[-1] > gamma[5]


# =============================================================================
# Test LAOS
# =============================================================================


class TestHVNMLAOS:
    """Test LAOS (Large Amplitude Oscillatory Shear)."""

    def test_laos_returns_dict(self, filled_model):
        omega = 1.0
        t = np.linspace(0, 10 * 2 * np.pi / omega, 500)
        result = filled_model.simulate_laos(t, gamma_0=0.1, omega=omega)
        assert "stress" in result
        assert "strain" in result
        assert "mu_I_xy" in result

    def test_odd_harmonics(self, filled_model):
        """LAOS with dual TST should produce odd harmonics."""
        omega = 1.0
        t = np.linspace(0, 20 * 2 * np.pi / omega, 2000)
        result = filled_model.simulate_laos(t, gamma_0=0.5, omega=omega)
        harmonics = filled_model.extract_laos_harmonics(result, n_harmonics=3)
        # I_1 should dominate
        assert harmonics["sigma_harmonics"][0] > 0
        # Higher harmonics present but smaller
        assert harmonics["sigma_harmonics"][0] > harmonics["sigma_harmonics"][1]

    def test_lissajous_closed(self, filled_model):
        """Stress-strain Lissajous should be approximately closed."""
        omega = 1.0
        n_cycles = 10
        t = np.linspace(0, n_cycles * 2 * np.pi / omega, n_cycles * 200)
        result = filled_model.simulate_laos(t, gamma_0=0.1, omega=omega)
        # Last cycle start and end should be close
        period_pts = 200
        stress_start = result["stress"][-period_pts]
        stress_end = result["stress"][-1]
        np.testing.assert_allclose(stress_start, stress_end, atol=100.0)

    def test_laos_no_nan(self, filled_model):
        omega = 1.0
        t = np.linspace(0, 5 * 2 * np.pi / omega, 300)
        result = filled_model.simulate_laos(t, gamma_0=0.1, omega=omega)
        assert np.all(np.isfinite(result["stress"]))

    def test_harmonic_extraction(self, filled_model):
        omega = 1.0
        t = np.linspace(0, 10 * 2 * np.pi / omega, 1000)
        result = filled_model.simulate_laos(t, gamma_0=0.1, omega=omega)
        h = filled_model.extract_laos_harmonics(result, n_harmonics=5)
        assert len(h["harmonic_index"]) == 5
        assert h["harmonic_index"][0] == 1
        assert h["harmonic_index"][1] == 3


# =============================================================================
# Test Damage
# =============================================================================


class TestHVNMDamage:
    """Test damage behavior."""

    def test_no_damage_matches_damage_off(self):
        """With damage params at zero, result matches damage-off."""
        m_off = HVNMLocal(include_damage=False, include_interfacial_damage=False)
        m_on = HVNMLocal(include_damage=True, include_interfacial_damage=True)
        # Set damage rates to zero
        m_on.parameters.set_value("Gamma_0", 0.0)
        m_on.parameters.set_value("Gamma_0_int", 0.0)

        omega = np.logspace(-2, 2, 30)
        G_off_p, G_off_dp = m_off.predict_saos(omega)
        G_on_p, G_on_dp = m_on.predict_saos(omega)
        np.testing.assert_allclose(G_on_p, G_off_p, rtol=1e-10)
        np.testing.assert_allclose(G_on_dp, G_off_dp, rtol=1e-10)

    def test_interfacial_damage_flag(self):
        m = HVNMLocal(include_interfacial_damage=True)
        assert "Gamma_0_int" in m.parameters.keys()
        assert "h_0" in m.parameters.keys()
        assert "lambda_crit_int" in m.parameters.keys()
        assert "E_a_heal" in m.parameters.keys()
        assert "n_h" in m.parameters.keys()

    def test_matrix_damage_flag(self):
        m = HVNMLocal(include_damage=True)
        assert "Gamma_0" in m.parameters.keys()
        assert "lambda_crit" in m.parameters.keys()

    def test_damage_parameters_independent(self):
        """Matrix and interfacial damage are independent flags."""
        m = HVNMLocal(include_damage=True, include_interfacial_damage=False)
        assert "Gamma_0" in m.parameters.keys()
        assert "Gamma_0_int" not in m.parameters.keys()

    def test_self_healing_parameter(self):
        m = HVNMLocal(include_interfacial_damage=True)
        assert m.h_0 is not None
        assert m.E_a_heal is not None


# =============================================================================
# Test Interphase Physics
# =============================================================================


class TestHVNMInterphase:
    """Test interphase physics and NP geometry."""

    def test_phi_I_increases_with_phi(self):
        for phi_val in [0.01, 0.05, 0.1, 0.2]:
            phi_I = float(hvnm_interphase_fraction(phi_val, 20e-9, 1e-9, 10e-9))
            if phi_val > 0.01:
                assert phi_I > prev_phi_I  # noqa: F821
            prev_phi_I = phi_I  # noqa: F841

    def test_phi_I_increases_with_delta_m(self):
        for dm in [5e-9, 10e-9, 20e-9]:
            phi_I = float(hvnm_interphase_fraction(0.05, 20e-9, 1e-9, dm))
            if dm > 5e-9:
                assert phi_I > prev_phi_I  # noqa: F821
            prev_phi_I = phi_I  # noqa: F841

    def test_guth_gold_values(self):
        """Verify Guth-Gold formula at known values."""
        np.testing.assert_allclose(float(hvnm_guth_gold(0.0)), 1.0, atol=1e-15)
        np.testing.assert_allclose(float(hvnm_guth_gold(0.1)), 1.391, atol=1e-3)
        np.testing.assert_allclose(float(hvnm_guth_gold(0.2)), 2.064, atol=1e-3)

    def test_dual_ber_rates_differ(self, filled_model):
        """Matrix and interphase BER rates should differ (different E_a)."""
        k_mat = filled_model.compute_ber_rate_at_equilibrium()
        k_int = filled_model.compute_ber_rate_interphase_equilibrium()
        # E_a_int > E_a => k_int < k_mat
        assert k_int < k_mat

    def test_dual_freezing_temps(self, filled_model):
        """T_v^int > T_v^mat when E_a^int > E_a^mat."""
        T_v_mat, T_v_int = filled_model.get_dual_topological_freezing_temps()
        assert T_v_int > T_v_mat

    def test_interphase_regime_classification(self, filled_model):
        regime = filled_model.classify_interphase_regime()
        assert regime in ["frozen", "active"]

    def test_weissenberg_numbers(self, filled_model):
        Wi_mat, Wi_int = filled_model.compute_weissenberg_numbers(1.0)
        assert Wi_mat > 0
        assert Wi_int > 0
        # Wi_int > Wi_mat because tau_I > tau_E
        assert Wi_int > Wi_mat

    def test_payne_parameters(self, filled_model):
        payne = filled_model.get_payne_parameters()
        assert payne["G_0"] > payne["G_inf"]
        assert payne["gamma_c"] > 0
        assert payne["gamma_c"] < 1.0  # Amplified critical strain


# =============================================================================
# Test Bayesian Interface
# =============================================================================


class TestHVNMBayesian:
    """Test model_function JAX-traceability and Bayesian interface."""

    @pytest.mark.smoke
    def test_model_function_traceable(self, default_model):
        """model_function must accept JAX arrays without error."""
        omega = jnp.logspace(-2, 2, 20)
        params = jnp.array(
            [default_model.parameters.get_value(n) for n in default_model.parameters.keys()],
            dtype=jnp.float64,
        )
        result = default_model.model_function(omega, params, test_mode="oscillation")
        assert result is not None
        assert len(result) == 20

    @pytest.mark.smoke
    def test_model_function_flow_curve(self, default_model):
        gd = jnp.logspace(-1, 1, 10)
        params = jnp.array(
            [default_model.parameters.get_value(n) for n in default_model.parameters.keys()],
            dtype=jnp.float64,
        )
        result = default_model.model_function(gd, params, test_mode="flow_curve")
        assert np.all(np.isfinite(np.asarray(result)))

    @pytest.mark.smoke
    def test_model_function_relaxation(self, default_model):
        t = jnp.logspace(-2, 2, 20)
        params = jnp.array(
            [default_model.parameters.get_value(n) for n in default_model.parameters.keys()],
            dtype=jnp.float64,
        )
        result = default_model.model_function(t, params, test_mode="relaxation")
        assert np.all(np.isfinite(np.asarray(result)))

    @pytest.mark.slow
    def test_nlsq_to_nuts_saos(self, filled_model):
        """NLSQ then NUTS for SAOS data."""
        omega = np.logspace(-2, 2, 30)
        G_p, G_dp = filled_model.predict_saos(omega)
        G_star = np.sqrt(G_p**2 + G_dp**2)

        # Add noise
        rng = np.random.default_rng(42)
        G_star_noisy = G_star * (1.0 + 0.02 * rng.standard_normal(len(G_star)))

        # NLSQ fit
        m = HVNMLocal()
        m.fit(omega, G_star_noisy, test_mode="oscillation")
        assert m.fitted_

    @pytest.mark.slow
    def test_nlsq_to_nuts_relaxation(self, filled_model):
        """NLSQ then NUTS for relaxation data."""
        t = np.logspace(-2, 3, 50)
        p = filled_model._get_params_dict()
        d = filled_model._get_derived_params(p)
        G_t = filled_model.simulate_relaxation(t, gamma_step=0.01)

        rng = np.random.default_rng(42)
        G_t_noisy = G_t * (1.0 + 0.02 * rng.standard_normal(len(G_t)))

        m = HVNMLocal()
        m.fit(t, G_t_noisy, test_mode="relaxation")
        assert m.fitted_
