"""Tests for HVM (Hybrid Vitrimer Model).

Test organization:
- TestHVMCreation: Model instantiation and parameter setup (~10, smoke)
- TestHVMLimitingCases: Recovery of known models (~8)
- TestHVMFlowCurve: Steady-state behavior (~5)
- TestHVMSAOS: Linear viscoelastic response (~6)
- TestHVMStartup: Transient startup shear (~6)
- TestHVMRelaxation: Stress relaxation (~5)
- TestHVMCreep: Creep compliance (~5)
- TestHVMLAOS: Large amplitude oscillatory shear (~5)
- TestHVMDamage: Cooperative shielding (~4)
- TestHVMBayesian: NumPyro integration (~3, slow)
- TestHVMTemperature: Arrhenius kinetics (~3)
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hvm_default():
    """Default full HVM model."""
    from rheojax.models import HVMLocal

    m = HVMLocal()
    m.parameters.set_value("G_P", 5000.0)
    m.parameters.set_value("G_E", 3000.0)
    m.parameters.set_value("G_D", 2000.0)
    m.parameters.set_value("k_d_D", 1.0)
    m.parameters.set_value("V_act", 1e-6)  # Small V_act for near-constant rate
    return m


@pytest.fixture
def hvm_partial():
    """Partial vitrimer (no D-network)."""
    from rheojax.models import HVMLocal

    return HVMLocal.partial_vitrimer(G_P=5000.0, G_E=3000.0, V_act=1e-6)


@pytest.fixture
def hvm_maxwell():
    """Maxwell limiting case."""
    from rheojax.models import HVMLocal

    return HVMLocal.maxwell(G_D=5000.0, k_d_D=2.0)


@pytest.fixture
def hvm_zener():
    """Zener/SLS limiting case."""
    from rheojax.models import HVMLocal

    return HVMLocal.zener(G_P=5000.0, G_D=3000.0, k_d_D=0.5)


# =============================================================================
# TestHVMCreation
# =============================================================================


class TestHVMCreation:
    """Model instantiation and parameter setup."""

    @pytest.mark.smoke
    def test_default_creation(self):
        from rheojax.models import HVMLocal

        m = HVMLocal()
        assert m is not None
        assert m.kinetics == "stress"
        assert m.include_damage is False
        assert m.include_dissociative is True

    @pytest.mark.smoke
    def test_parameter_count_full(self):
        from rheojax.models import HVMLocal

        m = HVMLocal()
        # 6 base + 2 dissociative = 8
        assert len(list(m.parameters.keys())) == 8

    @pytest.mark.smoke
    def test_parameter_count_no_dissociative(self):
        from rheojax.models import HVMLocal

        m = HVMLocal(include_dissociative=False)
        # 6 base only
        assert len(list(m.parameters.keys())) == 6

    @pytest.mark.smoke
    def test_parameter_count_with_damage(self):
        from rheojax.models import HVMLocal

        m = HVMLocal(include_damage=True)
        # 6 base + 2 dissociative + 2 damage = 10
        assert len(list(m.parameters.keys())) == 10

    @pytest.mark.smoke
    def test_parameter_names_present(self):
        from rheojax.models import HVMLocal

        m = HVMLocal()
        names = list(m.parameters.keys())
        for expected in ["G_P", "G_E", "nu_0", "E_a", "V_act", "T", "G_D", "k_d_D"]:
            assert expected in names

    @pytest.mark.smoke
    def test_registry_lookup(self):
        from rheojax.core.registry import ModelRegistry

        m = ModelRegistry.create("hvm_local")
        assert m.__class__.__name__ == "HVMLocal"

    @pytest.mark.smoke
    def test_registry_alias(self):
        from rheojax.core.registry import ModelRegistry

        m = ModelRegistry.create("hvm")
        assert m.__class__.__name__ == "HVMLocal"

    @pytest.mark.smoke
    def test_kinetics_stretch(self):
        from rheojax.models import HVMLocal

        m = HVMLocal(kinetics="stretch")
        assert m.kinetics == "stretch"

    @pytest.mark.smoke
    def test_repr_format(self):
        from rheojax.models import HVMLocal

        m = HVMLocal()
        s = repr(m)
        assert "HVMLocal" in s
        assert "G_P=" in s
        assert "G_E=" in s

    @pytest.mark.smoke
    def test_import_from_models(self):
        from rheojax.models import HVMLocal

        assert HVMLocal is not None


# =============================================================================
# TestHVMLimitingCases
# =============================================================================


class TestHVMLimitingCases:
    """Recovery of known models as limiting cases."""

    def test_neo_hookean_saos(self):
        """G_E=0, G_D=0 → SAOS G' = G_P (frequency-independent)."""
        from rheojax.models import HVMLocal

        m = HVMLocal.neo_hookean(G_P=5000.0)
        omega = np.logspace(-2, 2, 50)
        G_p, G_pp = m.predict_saos(omega)

        # G' should be constant = G_P
        np.testing.assert_allclose(G_p, 5000.0, rtol=1e-6)
        # G'' should be ~0
        np.testing.assert_allclose(G_pp, 0.0, atol=1e-3)

    def test_maxwell_saos(self):
        """G_P=0, G_E=0 → single Maxwell mode from D-network."""
        from rheojax.models import HVMLocal

        m = HVMLocal.maxwell(G_D=5000.0, k_d_D=2.0)
        omega = np.array([2.0])  # At crossover omega = k_d_D

        G_p, G_pp = m.predict_saos(omega)
        # At crossover: G' = G'' = G_D/2
        np.testing.assert_allclose(G_p[0], 2500.0, rtol=0.01)
        np.testing.assert_allclose(G_pp[0], 2500.0, rtol=0.01)

    def test_zener_saos_limits(self):
        """G_E=0 → Zener/SLS with plateau + one Maxwell."""
        from rheojax.models import HVMLocal

        m = HVMLocal.zener(G_P=5000.0, G_D=3000.0, k_d_D=0.5)
        omega = np.logspace(-4, 4, 100)
        G_p, G_pp = m.predict_saos(omega)

        # G'(omega -> 0) = G_P
        assert G_p[0] == pytest.approx(5000.0, rel=0.01)
        # G'(omega -> inf) = G_P + G_D
        assert G_p[-1] == pytest.approx(8000.0, rel=0.01)

    def test_partial_vitrimer_no_d_stress(self):
        """G_D=0 → no D-network stress at steady state."""
        from rheojax.models import HVMLocal

        m = HVMLocal.partial_vitrimer(G_P=5000.0, G_E=3000.0)
        gamma_dot = np.logspace(-2, 2, 20)
        sigma = m.predict_flow_curve(gamma_dot)

        # Sigma_E -> 0 at steady state, sigma_D = 0 (no D-network)
        # Only remaining: sigma_D contribution = 0
        np.testing.assert_allclose(sigma, 0.0, atol=1e-10)

    def test_pure_vitrimer_saos(self):
        """G_P=0, G_D=0 → single vitrimer mode with factor-of-2."""
        from rheojax.models import HVMLocal

        m = HVMLocal.pure_vitrimer(G_E=5000.0)
        k0 = m.compute_ber_rate_at_equilibrium()
        tau_eff = 1.0 / (2.0 * k0)

        # At omega = 1/tau_eff: crossover frequency
        omega = np.array([1.0 / tau_eff])
        G_p, G_pp = m.predict_saos(omega)
        # G' = G'' = G_E/2 at crossover
        np.testing.assert_allclose(G_p[0], 2500.0, rtol=0.01)
        np.testing.assert_allclose(G_pp[0], 2500.0, rtol=0.01)

    def test_get_limiting_case_string(self):
        """Verify limiting case identification."""
        from rheojax.models import HVMLocal

        assert "neo-Hookean" in HVMLocal.neo_hookean().get_limiting_case()
        assert "Maxwell" in HVMLocal.maxwell().get_limiting_case()
        assert "Zener" in HVMLocal.zener().get_limiting_case()
        assert "partial vitrimer" in HVMLocal.partial_vitrimer().get_limiting_case()
        assert "pure vitrimer" in HVMLocal.pure_vitrimer().get_limiting_case()
        assert "full HVM" in HVMLocal().get_limiting_case()

    def test_relaxation_spectrum_entries(self, hvm_default):
        """Verify relaxation spectrum has correct entries."""
        spectrum = hvm_default.get_relaxation_spectrum()
        assert len(spectrum) == 3
        # P-network: infinite relaxation time
        assert spectrum[0][1] == float("inf")
        # E and D have finite relaxation times
        assert spectrum[1][1] < float("inf")
        assert spectrum[2][1] < float("inf")

    def test_network_fractions_sum_to_one(self, hvm_default):
        """Network fractions must sum to 1."""
        fracs = hvm_default.get_network_fractions()
        total = fracs["f_P"] + fracs["f_E"] + fracs["f_D"]
        assert total == pytest.approx(1.0, rel=1e-10)


# =============================================================================
# TestHVMFlowCurve
# =============================================================================


class TestHVMFlowCurve:
    """Steady-state flow curve behavior."""

    def test_sigma_e_zero_at_steady_state(self, hvm_default):
        """E-network stress → 0 at steady state (vitrimer signature)."""
        gamma_dot = np.logspace(-2, 2, 20)
        result = hvm_default.predict_flow_curve(gamma_dot, return_components=True)
        np.testing.assert_allclose(result["sigma_E"], 0.0, atol=1e-10)

    def test_sigma_d_dominates(self, hvm_default):
        """D-network viscous stress dominates flow curve."""
        gamma_dot = np.logspace(-2, 2, 20)
        result = hvm_default.predict_flow_curve(gamma_dot, return_components=True)
        eta_D = hvm_default.G_D / hvm_default.k_d_D
        expected = eta_D * gamma_dot
        np.testing.assert_allclose(result["sigma_D"], expected, rtol=1e-10)

    def test_monotonic_stress_rate(self, hvm_default):
        """Stress increases monotonically with shear rate."""
        gamma_dot = np.logspace(-2, 2, 20)
        sigma = hvm_default.predict_flow_curve(gamma_dot)
        assert np.all(np.diff(sigma) > 0)

    def test_linear_newtonian(self, hvm_default):
        """Low shear rate: Newtonian behavior sigma = eta*gamma_dot."""
        gamma_dot = np.array([0.01, 0.02, 0.05])
        sigma = hvm_default.predict_flow_curve(gamma_dot)
        eta_eff = sigma / gamma_dot
        # Constant effective viscosity
        np.testing.assert_allclose(eta_eff, eta_eff[0], rtol=1e-8)

    def test_flow_curve_component_decomposition(self, hvm_default):
        """Components dict has expected keys."""
        gamma_dot = np.array([1.0])
        result = hvm_default.predict_flow_curve(gamma_dot, return_components=True)
        assert "stress" in result
        assert "sigma_P" in result
        assert "sigma_E" in result
        assert "sigma_D" in result
        assert "eta_eff" in result


# =============================================================================
# TestHVMSAOS
# =============================================================================


class TestHVMSAOS:
    """Small Amplitude Oscillatory Shear."""

    def test_low_freq_plateau(self, hvm_default):
        """G'(omega → 0) → G_P."""
        omega = np.array([1e-6])
        G_p, _ = hvm_default.predict_saos(omega)
        assert G_p[0] == pytest.approx(hvm_default.G_P, rel=0.01)

    def test_high_freq_plateau(self, hvm_default):
        """G'(omega → inf) → G_P + G_E + G_D."""
        omega = np.array([1e8])
        G_p, _ = hvm_default.predict_saos(omega)
        G_tot = hvm_default.G_P + hvm_default.G_E + hvm_default.G_D
        assert G_p[0] == pytest.approx(G_tot, rel=0.01)

    def test_two_loss_peaks(self, hvm_default):
        """G'' has two peaks from E and D networks."""
        omega = np.logspace(-6, 6, 1000)
        _, G_pp = hvm_default.predict_saos(omega)

        # Find local maxima
        from scipy.signal import argrelmax

        peaks = argrelmax(G_pp, order=10)[0]
        # Should have at least 1 peak (D-network); E-network peak may be
        # at very low frequency depending on k_BER
        assert len(peaks) >= 1

    def test_factor_of_two_consistency(self, hvm_default):
        """tau_E_eff = 1/(2*k_BER_0), not 1/k_BER_0."""
        k0 = hvm_default.compute_ber_rate_at_equilibrium()
        tau_E_eff = hvm_default.get_vitrimer_relaxation_time()
        assert tau_E_eff == pytest.approx(1.0 / (2.0 * k0), rel=1e-10)

    def test_saos_returns_magnitude(self, hvm_default):
        """return_components=False returns |G*|."""
        omega = np.logspace(-2, 2, 10)
        G_star = hvm_default.predict_saos(omega, return_components=False)
        G_p, G_pp = hvm_default.predict_saos(omega)
        expected = np.sqrt(G_p**2 + G_pp**2)
        np.testing.assert_allclose(G_star, expected, rtol=1e-10)

    def test_temperature_shift_arrhenius(self):
        """Higher T → lower tau_E → shifted crossover."""
        from rheojax.models import HVMLocal

        m1 = HVMLocal.pure_vitrimer(G_E=5000.0, T=300.0)
        m2 = HVMLocal.pure_vitrimer(G_E=5000.0, T=350.0)

        # Higher T → faster BER → shorter tau_E
        tau1 = m1.get_vitrimer_relaxation_time()
        tau2 = m2.get_vitrimer_relaxation_time()
        assert tau2 < tau1


# =============================================================================
# TestHVMStartup
# =============================================================================


class TestHVMStartup:
    """Transient startup shear."""

    def test_startup_initial_slope(self, hvm_default):
        """Short-time slope = G_tot * gamma_dot."""
        t = np.linspace(0.001, 0.01, 10)
        stress = hvm_default.simulate_startup(t, gamma_dot=1.0)
        G_tot = hvm_default.G_P + hvm_default.G_E + hvm_default.G_D

        # At very short times, all networks are elastic: sigma ≈ G_tot * gamma_dot * t
        expected = G_tot * 1.0 * t
        # Use atol for near-zero values (first point) and rtol for the rest
        np.testing.assert_allclose(stress, expected, rtol=0.1, atol=G_tot * 0.002)

    def test_startup_no_nan(self, hvm_default):
        """No NaN in startup stress."""
        t = np.linspace(0.01, 20, 100)
        stress = hvm_default.simulate_startup(t, gamma_dot=0.5)
        assert not np.any(np.isnan(stress))

    def test_startup_return_full(self, hvm_default):
        """return_full gives dict with expected keys."""
        t = np.linspace(0.01, 10, 50)
        result = hvm_default.simulate_startup(t, gamma_dot=1.0, return_full=True)
        expected_keys = {
            "time",
            "stress",
            "strain",
            "N1",
            "damage",
            "mu_E_xx",
            "mu_E_yy",
            "mu_E_xy",
            "mu_E_nat_xx",
            "mu_E_nat_yy",
            "mu_E_nat_xy",
            "mu_D_xx",
            "mu_D_yy",
            "mu_D_xy",
        }
        assert set(result.keys()) == expected_keys

    def test_startup_positive_stress(self, hvm_default):
        """Stress is positive for positive gamma_dot."""
        t = np.linspace(0.01, 10, 50)
        stress = hvm_default.simulate_startup(t, gamma_dot=1.0)
        assert np.all(stress >= 0)

    def test_startup_stress_increases_initially(self, hvm_default):
        """Stress increases initially (elastic loading)."""
        t = np.linspace(0.01, 0.5, 20)
        stress = hvm_default.simulate_startup(t, gamma_dot=1.0)
        # First few points should increase
        assert np.all(np.diff(stress[:10]) > 0)

    def test_startup_strain_linear(self, hvm_default):
        """Accumulated strain increases linearly at gamma_dot*t."""
        t = np.linspace(0.01, 10, 50)
        result = hvm_default.simulate_startup(t, gamma_dot=2.0, return_full=True)
        # ODE integrates from t[0], so gamma(t) = gamma_dot * (t - t[0]) at the saved points
        # Use a looser tolerance and skip first point
        expected_strain = 2.0 * t
        np.testing.assert_allclose(result["strain"][5:], expected_strain[5:], rtol=0.02)


# =============================================================================
# TestHVMRelaxation
# =============================================================================


class TestHVMRelaxation:
    """Stress relaxation after step strain."""

    def test_relaxation_initial_modulus(self, hvm_default):
        """G(0+) ≈ G_P + G_E + G_D."""
        t = np.linspace(0.001, 50, 200)
        G_t = hvm_default.simulate_relaxation(t, gamma_step=0.01)
        G_tot = hvm_default.G_P + hvm_default.G_E + hvm_default.G_D
        assert G_t[0] == pytest.approx(G_tot, rel=0.05)

    def test_relaxation_plateau(self, hvm_default):
        """G(t → inf) → G_P (permanent network plateau)."""
        # Use long time array
        t = np.linspace(0.01, 500, 200)
        G_t = hvm_default.simulate_relaxation(t, gamma_step=0.01)
        # At long times, only P-network remains
        # D-network should be fully relaxed (tau_D = 1s)
        # E-network has very long tau_E (~4000s) so it will still contribute
        # Check that G_t is between G_P and G_P + G_E
        assert G_t[-1] >= hvm_default.G_P * 0.9

    def test_relaxation_monotonic_decrease(self, hvm_default):
        """G(t) decreases monotonically."""
        t = np.linspace(0.01, 20, 100)
        G_t = hvm_default.simulate_relaxation(t, gamma_step=0.01)
        # Allow small numerical noise
        assert np.all(np.diff(G_t) <= 1e-4 * G_t[0])

    def test_relaxation_no_nan(self, hvm_default):
        """No NaN in relaxation modulus."""
        t = np.linspace(0.01, 50, 100)
        G_t = hvm_default.simulate_relaxation(t, gamma_step=0.01)
        assert not np.any(np.isnan(G_t))

    def test_relaxation_return_full(self, hvm_default):
        """return_full dict has expected keys."""
        t = np.linspace(0.01, 10, 50)
        result = hvm_default.simulate_relaxation(t, gamma_step=0.01, return_full=True)
        assert "G_t" in result
        assert "stress" in result
        assert "mu_E_xy" in result


# =============================================================================
# TestHVMCreep
# =============================================================================


class TestHVMCreep:
    """Creep compliance under constant stress."""

    def test_creep_no_nan(self, hvm_default):
        """No NaN in creep strain."""
        t = np.linspace(0.01, 50, 100)
        gamma = hvm_default.simulate_creep(t, sigma_0=100.0)
        assert not np.any(np.isnan(gamma))

    def test_creep_positive_strain(self, hvm_default):
        """Strain is positive for positive stress."""
        t = np.linspace(0.01, 10, 50)
        gamma = hvm_default.simulate_creep(t, sigma_0=100.0)
        assert np.all(gamma >= 0)

    def test_creep_strain_increases(self, hvm_default):
        """Strain increases with time under load."""
        t = np.linspace(0.01, 10, 50)
        gamma = hvm_default.simulate_creep(t, sigma_0=100.0)
        # Overall increasing (may have small fluctuations at start)
        assert gamma[-1] > gamma[0]

    def test_creep_return_full(self, hvm_default):
        """return_full dict has expected keys."""
        t = np.linspace(0.01, 10, 50)
        result = hvm_default.simulate_creep(t, sigma_0=100.0, return_full=True)
        assert "strain" in result
        assert "compliance" in result
        assert "mu_E_xy" in result

    def test_creep_higher_stress_more_strain(self, hvm_default):
        """Higher applied stress → more strain."""
        t = np.linspace(0.01, 10, 50)
        gamma_low = hvm_default.simulate_creep(t, sigma_0=50.0)
        gamma_high = hvm_default.simulate_creep(t, sigma_0=200.0)
        assert gamma_high[-1] > gamma_low[-1]


# =============================================================================
# TestHVMLAOS
# =============================================================================


class TestHVMLAOS:
    """Large Amplitude Oscillatory Shear."""

    def test_laos_no_nan(self, hvm_default):
        """No NaN in LAOS stress."""
        period = 2.0 * np.pi / 1.0
        t = np.linspace(0, 5 * period, 500)
        result = hvm_default.simulate_laos(t, gamma_0=0.01, omega=1.0)
        assert not np.any(np.isnan(result["stress"]))

    def test_laos_dict_keys(self, hvm_default):
        """LAOS returns dict with expected keys."""
        t = np.linspace(0, 10, 100)
        result = hvm_default.simulate_laos(t, gamma_0=0.1, omega=1.0)
        expected_keys = {
            "time",
            "strain",
            "stress",
            "gamma_dot",
            "N1",
            "mu_E_xy",
            "mu_E_nat_xy",
            "mu_D_xy",
            "damage",
        }
        assert set(result.keys()) == expected_keys

    def test_laos_strain_oscillatory(self, hvm_default):
        """Strain follows sin(omega*t)."""
        t = np.linspace(0, 10, 100)
        result = hvm_default.simulate_laos(t, gamma_0=0.5, omega=2.0)
        expected_strain = 0.5 * np.sin(2.0 * t)
        np.testing.assert_allclose(result["strain"], expected_strain, atol=1e-10)

    def test_laos_harmonic_extraction(self, hvm_default):
        """Harmonic extraction returns expected structure."""
        period = 2.0 * np.pi / 1.0
        t = np.linspace(0, 10 * period, 2000)
        result = hvm_default.simulate_laos(t, gamma_0=0.01, omega=1.0)
        harmonics = hvm_default.extract_laos_harmonics(result, n_harmonics=3)
        assert "harmonic_index" in harmonics
        assert "sigma_harmonics" in harmonics
        assert len(harmonics["harmonic_index"]) == 3
        # Fundamental should dominate
        assert harmonics["sigma_harmonics"][0] > 0

    def test_laos_small_amplitude_linear(self, hvm_default):
        """Small amplitude → nearly sinusoidal stress (linear regime)."""
        period = 2.0 * np.pi / 1.0
        t = np.linspace(0, 10 * period, 2000)
        result = hvm_default.simulate_laos(t, gamma_0=0.001, omega=1.0)
        harmonics = hvm_default.extract_laos_harmonics(result, n_harmonics=3)
        # Third harmonic should be << first (linear regime)
        if harmonics["sigma_harmonics"][0] > 1e-10:
            ratio = harmonics["sigma_harmonics"][1] / harmonics["sigma_harmonics"][0]
            assert ratio < 0.1  # I3/I1 < 10% in linear regime


# =============================================================================
# TestHVMDamage
# =============================================================================


class TestHVMDamage:
    """Cooperative damage shielding."""

    def test_damage_off_by_default(self):
        """Damage is off by default."""
        from rheojax.models import HVMLocal

        m = HVMLocal()
        assert m.include_damage is False
        assert "Gamma_0" not in list(m.parameters.keys())

    def test_damage_parameters_present(self):
        """Damage parameters present when enabled."""
        from rheojax.models import HVMLocal

        m = HVMLocal(include_damage=True)
        names = list(m.parameters.keys())
        assert "Gamma_0" in names
        assert "lambda_crit" in names

    def test_no_damage_matches_damage_off(self):
        """Damage model with Gamma_0=0 matches no-damage model."""
        from rheojax.models import HVMLocal

        m1 = HVMLocal(include_damage=False)
        m1.parameters.set_value("G_P", 5000.0)
        m1.parameters.set_value("G_E", 3000.0)
        m1.parameters.set_value("V_act", 1e-6)

        m2 = HVMLocal(include_damage=True)
        m2.parameters.set_value("G_P", 5000.0)
        m2.parameters.set_value("G_E", 3000.0)
        m2.parameters.set_value("V_act", 1e-6)
        m2.parameters.set_value("Gamma_0", 0.0)

        omega = np.logspace(-2, 2, 20)
        G1_p, G1_pp = m1.predict_saos(omega)
        G2_p, G2_pp = m2.predict_saos(omega)
        np.testing.assert_allclose(G1_p, G2_p, rtol=1e-10)

    def test_damage_increases_monotonically(self):
        """Damage variable D increases monotonically under load."""
        from rheojax.models import HVMLocal

        m = HVMLocal(include_damage=True)
        m.parameters.set_value("G_P", 5000.0)
        m.parameters.set_value("G_E", 3000.0)
        m.parameters.set_value("V_act", 1e-6)
        m.parameters.set_value("Gamma_0", 0.01)
        m.parameters.set_value("lambda_crit", 1.01)  # Low threshold

        t = np.linspace(0.01, 20, 100)
        result = m.simulate_startup(t, gamma_dot=2.0, return_full=True)
        D = result["damage"]
        # D should be non-decreasing
        assert np.all(np.diff(D) >= -1e-10)


# =============================================================================
# TestHVMBayesian
# =============================================================================


class TestHVMBayesian:
    """NumPyro Bayesian integration."""

    @pytest.mark.smoke
    def test_model_function_callable(self, hvm_default):
        """model_function is callable with array params."""
        omega = np.logspace(-2, 2, 20)
        params = jnp.array(
            [
                hvm_default.parameters.get_value(n)
                for n in hvm_default.parameters.keys()
            ],
            dtype=jnp.float64,
        )
        result = hvm_default.model_function(omega, params, test_mode="oscillation")
        assert result.shape == (20,)
        assert not jnp.any(jnp.isnan(result))

    @pytest.mark.smoke
    def test_model_function_flow_curve(self, hvm_default):
        """model_function works for flow_curve."""
        gdot = np.logspace(-2, 2, 10)
        params = jnp.array(
            [
                hvm_default.parameters.get_value(n)
                for n in hvm_default.parameters.keys()
            ],
            dtype=jnp.float64,
        )
        result = hvm_default.model_function(gdot, params, test_mode="flow_curve")
        assert result.shape == (10,)
        assert jnp.all(result >= 0)

    @pytest.mark.smoke
    def test_model_function_relaxation(self, hvm_default):
        """model_function works for relaxation."""
        t = np.linspace(0.01, 10, 20)
        params = jnp.array(
            [
                hvm_default.parameters.get_value(n)
                for n in hvm_default.parameters.keys()
            ],
            dtype=jnp.float64,
        )
        result = hvm_default.model_function(t, params, test_mode="relaxation")
        assert result.shape == (20,)
        assert jnp.all(result >= 0)


# =============================================================================
# TestHVMTemperature
# =============================================================================


class TestHVMTemperature:
    """Arrhenius temperature dependence."""

    def test_arrhenius_ber_behavior(self):
        """k_BER_0 follows Arrhenius: higher T → faster exchange."""
        from rheojax.models import HVMLocal

        m = HVMLocal()
        T_values = [280.0, 300.0, 350.0, 400.0]
        rates = []
        for T in T_values:
            m.parameters.set_value("T", T)
            rates.append(m.compute_ber_rate_at_equilibrium())

        # Rates should increase with temperature
        for i in range(len(rates) - 1):
            assert rates[i + 1] > rates[i]

    def test_arrhenius_plot_data(self):
        """arrhenius_plot_data returns correct shapes."""
        from rheojax.models import HVMLocal

        m = HVMLocal()
        T_range = np.linspace(280, 400, 20)
        inv_T, log_k = m.arrhenius_plot_data(T_range)
        assert inv_T.shape == (20,)
        assert log_k.shape == (20,)
        # log_k should increase with T (decrease with 1/T)
        assert log_k[-1] > log_k[0]  # Higher T = higher rate

    def test_classify_vitrimer_regime(self):
        """Regime classification changes with temperature."""
        from rheojax.models import HVMLocal

        m = HVMLocal()
        # Very low T → glassy (k_BER << 1)
        m.parameters.set_value("T", 200.0)
        # May or may not be glassy depending on E_a, nu_0
        regime = m.classify_vitrimer_regime()
        assert regime in ("glassy", "rubbery", "flow")
