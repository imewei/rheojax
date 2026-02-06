"""Unit tests for VLB (Vernerey-Long-Brighenti) transient network models.

Tests cover:
- Model creation and parameter setup
- Analytical flow curve predictions
- Startup shear transients
- Stress relaxation (single exponential)
- Creep compliance (Maxwell and SLS)
- SAOS moduli (Maxwell)
- LAOS simulation and harmonic analysis
- Uniaxial extension
- Multi-network superposition
- Cross-protocol consistency
- Bayesian inference (JAX traceability)
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.vlb import VLBLocal, VLBMultiNetwork

jax, jnp = safe_import_jax()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def vlb_local():
    """VLBLocal with known parameters: G0=1000 Pa, k_d=1 1/s."""
    model = VLBLocal()
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d", 1.0)
    return model


@pytest.fixture
def vlb_multi_2():
    """VLBMultiNetwork with 2 modes, no permanent, no solvent."""
    model = VLBMultiNetwork(n_modes=2)
    model.parameters.set_value("G_0", 500.0)
    model.parameters.set_value("k_d_0", 0.1)
    model.parameters.set_value("G_1", 500.0)
    model.parameters.set_value("k_d_1", 10.0)
    model.parameters.set_value("eta_s", 0.0)
    return model


@pytest.fixture
def vlb_multi_perm():
    """VLBMultiNetwork with 1 mode + permanent."""
    model = VLBMultiNetwork(n_modes=1, include_permanent=True)
    model.parameters.set_value("G_0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    model.parameters.set_value("G_e", 500.0)
    return model


# =============================================================================
# VLBLocal Creation Tests
# =============================================================================


class TestVLBLocalCreation:
    """Test VLBLocal model instantiation and parameters."""

    @pytest.mark.smoke
    def test_creation_default(self):
        """Test default creation."""
        model = VLBLocal()
        assert "G0" in model.parameters.keys()
        assert "k_d" in model.parameters.keys()
        assert len(list(model.parameters.keys())) == 2

    @pytest.mark.smoke
    def test_parameter_bounds(self):
        """Test parameter bounds are reasonable."""
        model = VLBLocal()
        G0_bounds = model.parameters["G0"].bounds
        k_d_bounds = model.parameters["k_d"].bounds
        assert G0_bounds[0] > 0
        assert k_d_bounds[0] > 0

    @pytest.mark.smoke
    def test_parameter_units(self):
        """Test parameter units."""
        model = VLBLocal()
        assert model.parameters["G0"].units == "Pa"
        assert model.parameters["k_d"].units == "1/s"

    @pytest.mark.smoke
    def test_registry_lookup(self):
        """Test model can be found in registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("vlb_local")
        assert isinstance(model, VLBLocal)
        model2 = ModelRegistry.create("vlb")
        assert isinstance(model2, VLBLocal)

    @pytest.mark.smoke
    def test_repr(self):
        """Test string representation."""
        model = VLBLocal()
        repr_str = repr(model)
        assert "VLBLocal" in repr_str
        assert "G0=" in repr_str
        assert "k_d=" in repr_str

    def test_properties(self, vlb_local):
        """Test computed properties."""
        assert vlb_local.relaxation_time == pytest.approx(1.0, rel=1e-10)
        assert vlb_local.viscosity == pytest.approx(1000.0, rel=1e-10)
        assert vlb_local.G0 == pytest.approx(1000.0, rel=1e-10)
        assert vlb_local.k_d == pytest.approx(1.0, rel=1e-10)


# =============================================================================
# VLBLocal Flow Curve Tests
# =============================================================================


class TestVLBLocalFlowCurve:
    """Test VLBLocal steady shear predictions."""

    @pytest.mark.smoke
    def test_newtonian_linearity(self, vlb_local):
        """Sigma should be proportional to gamma_dot (Newtonian)."""
        gd = np.logspace(-2, 2, 20)
        sigma = vlb_local.predict(gd, test_mode="flow_curve")
        sigma = np.asarray(sigma)
        eta = sigma / gd
        # Should be constant
        np.testing.assert_allclose(eta, eta[0], rtol=1e-10)

    @pytest.mark.smoke
    def test_viscosity_value(self, vlb_local):
        """Viscosity should equal G0/k_d."""
        gd = np.array([1.0])
        sigma = vlb_local.predict(gd, test_mode="flow_curve")
        eta = float(sigma[0]) / gd[0]
        assert eta == pytest.approx(1000.0, rel=1e-10)

    @pytest.mark.smoke
    def test_n1_quadratic(self, vlb_local):
        """N1 should be proportional to gamma_dot^2."""
        gd = np.array([1.0, 2.0, 4.0])
        N1, N2 = vlb_local.predict_normal_stresses(gd)
        # N1 = 2*G0*(gdot/k_d)^2
        expected_N1 = 2 * 1000.0 * (gd / 1.0) ** 2
        np.testing.assert_allclose(N1, expected_N1, rtol=1e-10)
        # N2 = 0 for upper-convected
        np.testing.assert_allclose(N2, 0.0, atol=1e-15)

    def test_flow_curve_components(self, vlb_local):
        """Test predict_flow_curve with return_components."""
        gd = np.logspace(-1, 1, 10)
        sigma, eta, N1 = vlb_local.predict_flow_curve(gd, return_components=True)
        np.testing.assert_allclose(eta, 1000.0, rtol=1e-10)
        np.testing.assert_allclose(sigma, 1000.0 * gd, rtol=1e-10)


# =============================================================================
# VLBLocal Startup Tests
# =============================================================================


class TestVLBLocalStartup:
    """Test VLBLocal startup shear predictions."""

    def test_analytical_formula(self, vlb_local):
        """Compare analytical formula against known values."""
        t = np.linspace(0, 10, 100)
        sigma = vlb_local.simulate_startup(t, gamma_dot=1.0)
        expected = 1000.0 * (1.0 - np.exp(-t))
        np.testing.assert_allclose(sigma, expected, rtol=1e-10)

    def test_initial_zero(self, vlb_local):
        """Stress at t=0 should be zero."""
        t = np.array([0.0])
        sigma = vlb_local.simulate_startup(t, gamma_dot=1.0)
        assert float(sigma[0]) == pytest.approx(0.0, abs=1e-12)

    def test_steady_state(self, vlb_local):
        """Stress at long time should equal steady-state."""
        t = np.array([100.0])  # >> t_R = 1
        sigma = vlb_local.simulate_startup(t, gamma_dot=2.0)
        expected_ss = 1000.0 * 2.0 / 1.0  # G0*gdot/k_d
        assert float(sigma[0]) == pytest.approx(expected_ss, rel=1e-6)

    def test_n1_startup(self, vlb_local):
        """Test N1 during startup."""
        t = np.linspace(0, 10, 100)
        sigma, N1, strain = vlb_local.simulate_startup(
            t, gamma_dot=1.0, return_full=True
        )
        # N1(inf) = 2*G0*(gdot/k_d)^2 = 2000
        assert float(N1[-1]) == pytest.approx(2000.0, rel=1e-3)
        # N1(0) = 0
        assert float(N1[0]) == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# VLBLocal Relaxation Tests
# =============================================================================


class TestVLBLocalRelaxation:
    """Test VLBLocal stress relaxation."""

    @pytest.mark.smoke
    def test_single_exponential(self, vlb_local):
        """G(t) should be a single exponential."""
        t = np.linspace(0, 5, 50)
        G_t = vlb_local.simulate_relaxation(t)
        expected = 1000.0 * np.exp(-t)
        np.testing.assert_allclose(G_t, expected, rtol=1e-10)

    @pytest.mark.smoke
    def test_initial_modulus(self, vlb_local):
        """G(0) = G0."""
        G_0 = vlb_local.simulate_relaxation(np.array([0.0]))
        assert float(G_0[0]) == pytest.approx(1000.0, rel=1e-10)

    @pytest.mark.smoke
    def test_long_time_zero(self, vlb_local):
        """G(t) -> 0 at long times."""
        G_inf = vlb_local.simulate_relaxation(np.array([100.0]))
        assert abs(float(G_inf[0])) < 1e-30


# =============================================================================
# VLBLocal Creep Tests
# =============================================================================


class TestVLBLocalCreep:
    """Test VLBLocal creep compliance."""

    def test_maxwell_compliance(self, vlb_local):
        """J(t) = (1 + k_d*t) / G0 for Maxwell."""
        t = np.linspace(0, 10, 50)
        gamma, J = vlb_local.simulate_creep(t, sigma_0=1.0, return_full=True)
        expected_J = (1.0 + t) / 1000.0
        np.testing.assert_allclose(J, expected_J, rtol=1e-10)

    def test_elastic_jump(self, vlb_local):
        """J(0) = 1/G0."""
        _, J = vlb_local.simulate_creep(np.array([0.0]), sigma_0=1.0, return_full=True)
        assert float(J[0]) == pytest.approx(1.0 / 1000.0, rel=1e-10)

    def test_viscous_slope(self, vlb_local):
        """dJ/dt = k_d/G0 = 1/eta."""
        t = np.array([5.0, 10.0])
        _, J = vlb_local.simulate_creep(t, sigma_0=1.0, return_full=True)
        slope = (float(J[1]) - float(J[0])) / (t[1] - t[0])
        expected_slope = 1.0 / 1000.0  # k_d/G0 = 1/eta
        assert slope == pytest.approx(expected_slope, rel=1e-10)


# =============================================================================
# VLBLocal SAOS Tests
# =============================================================================


class TestVLBLocalSAOS:
    """Test VLBLocal small-amplitude oscillatory shear."""

    @pytest.mark.smoke
    def test_maxwell_g_prime(self, vlb_local):
        """G'(omega) = G0*omega^2*t_R^2 / (1 + omega^2*t_R^2)."""
        omega = np.logspace(-2, 2, 50)
        Gp, _ = vlb_local.predict_saos(omega)
        wt = omega * 1.0  # t_R = 1/k_d = 1
        expected = 1000.0 * wt**2 / (1 + wt**2)
        np.testing.assert_allclose(Gp, expected, rtol=1e-10)

    @pytest.mark.smoke
    def test_maxwell_g_double_prime(self, vlb_local):
        """G''(omega) = G0*omega*t_R / (1 + omega^2*t_R^2)."""
        omega = np.logspace(-2, 2, 50)
        _, Gpp = vlb_local.predict_saos(omega)
        wt = omega * 1.0
        expected = 1000.0 * wt / (1 + wt**2)
        np.testing.assert_allclose(Gpp, expected, rtol=1e-10)

    @pytest.mark.smoke
    def test_crossover(self, vlb_local):
        """G' = G'' at omega = k_d."""
        omega = np.array([1.0])  # k_d = 1
        Gp, Gpp = vlb_local.predict_saos(omega)
        assert float(Gp[0]) == pytest.approx(float(Gpp[0]), rel=1e-10)
        assert float(Gp[0]) == pytest.approx(500.0, rel=1e-10)

    @pytest.mark.smoke
    def test_complex_modulus(self, vlb_local):
        """Test |G*| via predict with test_mode=oscillation."""
        omega = np.array([1.0])
        G_star = vlb_local.predict(omega, test_mode="oscillation")
        expected = np.sqrt(500.0**2 + 500.0**2)
        assert float(G_star[0]) == pytest.approx(expected, rel=1e-10)


# =============================================================================
# VLBLocal LAOS Tests
# =============================================================================


class TestVLBLocalLAOS:
    """Test VLBLocal large-amplitude oscillatory shear."""

    def test_small_gamma_saos_limit(self, vlb_local):
        """At small gamma_0, LAOS should match SAOS."""
        omega = 1.0
        gamma_0 = 0.001  # Very small strain -> linear
        n_cycles = 10
        period = 2 * np.pi / omega
        t = np.linspace(0, n_cycles * period, 5000)

        result = vlb_local.simulate_laos(t, gamma_0=gamma_0, omega=omega)

        # Extract last 2 cycles for steady state
        stress = result["stress"]
        strain = result["strain"]
        t_arr = result["time"]
        start = np.searchsorted(t_arr, (n_cycles - 2) * period)

        # Compute amplitude ratio -> should match |G*|
        stress_amp = (np.max(stress[start:]) - np.min(stress[start:])) / 2
        strain_amp = gamma_0

        G_star_laos = stress_amp / strain_amp
        Gp, Gpp = vlb_local.predict_saos(np.array([omega]))
        G_star_saos = np.sqrt(float(Gp[0])**2 + float(Gpp[0])**2)

        assert G_star_laos == pytest.approx(G_star_saos, rel=0.02)

    def test_laos_model_function(self, vlb_local):
        """Test LAOS via model_function interface."""
        omega = 1.0
        gamma_0 = 0.5
        period = 2 * np.pi / omega
        t = np.linspace(0, 5 * period, 500)

        stress = vlb_local.predict(
            t, test_mode="laos", gamma_0=gamma_0, omega=omega
        )
        assert stress.shape == t.shape
        assert not np.any(np.isnan(np.asarray(stress)))

    def test_n1_present_in_laos(self, vlb_local):
        """N1 should be nonzero during LAOS (second harmonic)."""
        omega = 1.0
        gamma_0 = 0.5
        period = 2 * np.pi / omega
        t = np.linspace(0, 10 * period, 2000)

        result = vlb_local.simulate_laos(t, gamma_0=gamma_0, omega=omega)
        N1 = result["N1"]
        # N1 should have time-varying component
        assert np.std(N1[-500:]) > 0


# =============================================================================
# VLBLocal Extension Tests
# =============================================================================


class TestVLBLocalExtension:
    """Test VLBLocal uniaxial extension."""

    def test_trouton_ratio_3(self, vlb_local):
        """Trouton ratio should approach 3 at low extension rate."""
        eps_dot = np.array([1e-4])
        _, Tr = vlb_local.predict_uniaxial_extension(eps_dot, return_trouton=True)
        assert float(Tr[0]) == pytest.approx(3.0, rel=1e-3)

    def test_singularity_region(self, vlb_local):
        """High extension rate near k_d/2 should give large stress."""
        eps_dot = np.array([0.4])  # Close to k_d/2 = 0.5
        sigma_E = vlb_local.predict_uniaxial_extension(eps_dot)
        # Should be large (near singularity)
        assert float(sigma_E[0]) > 1000.0

    def test_analytical_transient(self, vlb_local):
        """Test transient extensional stress."""
        t = np.linspace(0, 10, 50)
        sigma_E, eta_E = vlb_local.simulate_uniaxial_extension(t, epsilon_dot=0.1)
        # At t=0: sigma_E = 0
        assert abs(float(sigma_E[0])) < 1.0
        # At long time: should approach steady state
        sigma_ss = vlb_local.predict_uniaxial_extension(np.array([0.1]))
        assert float(sigma_E[-1]) == pytest.approx(float(sigma_ss[0]), rel=0.01)


# =============================================================================
# VLBMultiNetwork Creation Tests
# =============================================================================


class TestVLBMultiNetworkCreation:
    """Test VLBMultiNetwork creation."""

    @pytest.mark.smoke
    def test_creation_2_modes(self):
        """Test creation with 2 modes."""
        model = VLBMultiNetwork(n_modes=2)
        assert model.n_modes == 2
        assert not model.include_permanent
        # 2*2 + 1 = 5 params
        assert len(list(model.parameters.keys())) == 5

    @pytest.mark.smoke
    def test_creation_with_permanent(self):
        """Test creation with permanent network."""
        model = VLBMultiNetwork(n_modes=2, include_permanent=True)
        assert model.include_permanent
        assert "G_e" in model.parameters.keys()
        # 2*2 + 2 = 6 params
        assert len(list(model.parameters.keys())) == 6

    @pytest.mark.smoke
    def test_parameter_count(self):
        """Test parameter count for various configurations."""
        for n in [1, 2, 3, 5]:
            m = VLBMultiNetwork(n_modes=n)
            assert len(list(m.parameters.keys())) == 2 * n + 1
            m_p = VLBMultiNetwork(n_modes=n, include_permanent=True)
            assert len(list(m_p.parameters.keys())) == 2 * n + 2

    @pytest.mark.smoke
    def test_single_mode_recovery(self):
        """Single mode VLBMultiNetwork should match VLBLocal."""
        m1 = VLBLocal()
        m1.parameters.set_value("G0", 1000.0)
        m1.parameters.set_value("k_d", 2.0)

        m2 = VLBMultiNetwork(n_modes=1)
        m2.parameters.set_value("G_0", 1000.0)
        m2.parameters.set_value("k_d_0", 2.0)
        m2.parameters.set_value("eta_s", 0.0)

        omega = np.logspace(-2, 2, 20)
        Gp1, Gpp1 = m1.predict_saos(omega)
        Gp2, Gpp2 = m2.predict_saos(omega)
        np.testing.assert_allclose(Gp1, Gp2, rtol=1e-10)
        np.testing.assert_allclose(Gpp1, Gpp2, rtol=1e-10)

    def test_invalid_n_modes(self):
        """Test error on invalid n_modes."""
        with pytest.raises(ValueError, match="n_modes must be >= 1"):
            VLBMultiNetwork(n_modes=0)


# =============================================================================
# VLBMultiNetwork Analytical Tests
# =============================================================================


class TestVLBMultiNetworkAnalytical:
    """Test multi-network analytical predictions."""

    def test_saos_superposition(self, vlb_multi_2):
        """G' and G'' should be sum of Maxwell contributions."""
        omega = np.logspace(-2, 2, 50)
        Gp, Gpp = vlb_multi_2.predict_saos(omega)

        # Manual computation
        G0, kd0 = 500.0, 0.1
        G1, kd1 = 500.0, 10.0
        t0, t1 = 1.0 / kd0, 1.0 / kd1
        wt0 = omega * t0
        wt1 = omega * t1

        Gp_expected = G0 * wt0**2 / (1 + wt0**2) + G1 * wt1**2 / (1 + wt1**2)
        Gpp_expected = G0 * wt0 / (1 + wt0**2) + G1 * wt1 / (1 + wt1**2)

        np.testing.assert_allclose(Gp, Gp_expected, rtol=1e-10)
        np.testing.assert_allclose(Gpp, Gpp_expected, rtol=1e-10)

    def test_relaxation_prony(self, vlb_multi_2):
        """G(t) should be a multi-exponential Prony series."""
        t = np.linspace(0, 20, 100)
        G_t = vlb_multi_2.predict(t, test_mode="relaxation")
        G_t = np.asarray(G_t)

        expected = 500.0 * np.exp(-0.1 * t) + 500.0 * np.exp(-10.0 * t)
        np.testing.assert_allclose(G_t, expected, rtol=1e-10)

    def test_startup_superposition(self, vlb_multi_2):
        """Startup stress is sum of individual mode transients."""
        t = np.linspace(0, 50, 200)
        gamma_dot = 1.0
        sigma = vlb_multi_2.predict(t, test_mode="startup", gamma_dot=gamma_dot)
        sigma = np.asarray(sigma)

        G0, kd0 = 500.0, 0.1
        G1, kd1 = 500.0, 10.0
        expected = (
            G0 * gamma_dot / kd0 * (1 - np.exp(-kd0 * t))
            + G1 * gamma_dot / kd1 * (1 - np.exp(-kd1 * t))
        )
        np.testing.assert_allclose(sigma, expected, rtol=1e-8)

    def test_flow_curve_newtonian(self, vlb_multi_2):
        """Flow curve should be Newtonian: sigma = eta_0 * gdot."""
        gd = np.logspace(-1, 1, 20)
        sigma = vlb_multi_2.predict(gd, test_mode="flow_curve")
        sigma = np.asarray(sigma)

        eta_0 = 500.0 / 0.1 + 500.0 / 10.0  # = 5050
        np.testing.assert_allclose(sigma, eta_0 * gd, rtol=1e-10)

    def test_sls_creep(self, vlb_multi_perm):
        """1 mode + permanent should give SLS creep (bounded)."""
        t = np.linspace(0, 50, 200)
        gamma = vlb_multi_perm.predict(
            t, test_mode="creep", sigma_applied=100.0
        )
        gamma = np.asarray(gamma)

        # SLS: J(t) = 1/(G0+Ge) + [G0/(Ge*(G0+Ge))]*(1-exp(-t/tau_ret))
        G0, k_d, G_e = 1000.0, 1.0, 500.0
        G_total = G0 + G_e
        tau_ret = G_total / (k_d * G_e)
        J_inf = 1.0 / G_e

        # At long time: J -> 1/G_e, gamma -> sigma_0/G_e
        gamma_inf = 100.0 * J_inf
        assert float(gamma[-1]) == pytest.approx(gamma_inf, rel=0.02)

    def test_permanent_equilibrium(self, vlb_multi_perm):
        """G(inf) should equal G_e for network with permanent component."""
        t = np.array([100.0])  # Very long time
        G_t = vlb_multi_perm.predict(t, test_mode="relaxation")
        assert float(G_t[0]) == pytest.approx(500.0, rel=1e-3)


# =============================================================================
# Cross-Protocol Consistency Tests
# =============================================================================


class TestVLBCrossProtocol:
    """Test cross-protocol consistency of VLB models."""

    def test_startup_steady_vs_flow_curve(self, vlb_local):
        """Steady-state startup stress should match flow curve."""
        gamma_dot = 5.0
        # Startup at long time
        t = np.array([100.0])
        sigma_startup = vlb_local.simulate_startup(t, gamma_dot=gamma_dot)
        # Flow curve
        sigma_flow = vlb_local.predict(np.array([gamma_dot]), test_mode="flow_curve")
        assert float(sigma_startup[0]) == pytest.approx(
            float(sigma_flow[0]), rel=1e-6
        )

    def test_saos_crossover_vs_relaxation(self, vlb_local):
        """Crossover frequency should match relaxation rate."""
        # Crossover at omega = k_d = 1 for Maxwell
        omega_c = 1.0
        Gp, Gpp = vlb_local.predict_saos(np.array([omega_c]))
        assert float(Gp[0]) == pytest.approx(float(Gpp[0]), rel=1e-10)

        # Relaxation time scale
        t = np.array([1.0 / omega_c])  # t = t_R
        G_t = vlb_local.simulate_relaxation(t)
        assert float(G_t[0]) == pytest.approx(1000.0 / np.e, rel=1e-10)

    def test_eta0_consistency(self, vlb_local):
        """Zero-shear viscosity from flow curve, creep, and SAOS should agree."""
        # From flow curve
        gd = np.array([1e-4])
        sigma = vlb_local.predict(gd, test_mode="flow_curve")
        eta_flow = float(sigma[0]) / gd[0]

        # From viscosity property
        eta_prop = vlb_local.viscosity

        # From SAOS: eta_0 = lim(G''/omega) as omega -> 0
        omega = np.array([1e-4])
        _, Gpp = vlb_local.predict_saos(omega)
        eta_saos = float(Gpp[0]) / omega[0]

        assert eta_flow == pytest.approx(1000.0, rel=1e-8)
        assert eta_prop == pytest.approx(1000.0, rel=1e-10)
        assert eta_saos == pytest.approx(1000.0, rel=1e-4)

    def test_instantaneous_modulus(self, vlb_local):
        """G(0) from relaxation should match G0 from SAOS high-freq limit."""
        G_0 = vlb_local.simulate_relaxation(np.array([0.0]))
        assert float(G_0[0]) == pytest.approx(1000.0, rel=1e-10)

        # High-freq SAOS: G'(inf) -> G0
        omega = np.array([1e6])
        Gp, _ = vlb_local.predict_saos(omega)
        assert float(Gp[0]) == pytest.approx(1000.0, rel=1e-4)


# =============================================================================
# Bayesian Tests
# =============================================================================


class TestVLBBayesian:
    """Test VLB Bayesian inference compatibility."""

    @pytest.mark.smoke
    def test_model_function_jax_traceable(self):
        """model_function should be JAX-traceable (no Python conditionals on params)."""
        model = VLBLocal()

        # Test with JAX arrays (simulating what NLSQ/NUTS would pass)
        X = jnp.linspace(0.01, 100.0, 20)
        params = jnp.array([1000.0, 1.0])

        # Should not raise
        result = model.model_function(X, params, test_mode="oscillation")
        assert result.shape == (20,)
        assert not jnp.any(jnp.isnan(result))

        # Also test flow curve
        result_fc = model.model_function(X, params, test_mode="flow_curve")
        assert result_fc.shape == (20,)

        # And relaxation
        t = jnp.linspace(0.01, 10.0, 20)
        result_rel = model.model_function(t, params, test_mode="relaxation")
        assert result_rel.shape == (20,)

    @pytest.mark.smoke
    def test_multi_model_function_jax_traceable(self):
        """Multi-network model_function should be JAX-traceable."""
        model = VLBMultiNetwork(n_modes=2)

        X = jnp.linspace(0.01, 100.0, 20)
        params = jnp.array([500.0, 0.1, 500.0, 10.0, 0.0])

        result = model.model_function(X, params, test_mode="oscillation")
        assert result.shape == (20,)
        assert not jnp.any(jnp.isnan(result))

    @pytest.mark.slow
    def test_fit_bayesian_oscillation(self, mcmc_config):
        """Test full NLSQ -> NUTS pipeline for SAOS."""
        model = VLBLocal()

        # Generate synthetic data
        G0_true, k_d_true = 2000.0, 5.0
        omega = np.logspace(-1, 2, 30)
        t_R = 1.0 / k_d_true
        wt = omega * t_R
        G_prime = G0_true * wt**2 / (1 + wt**2)
        G_double_prime = G0_true * wt / (1 + wt**2)
        G_star = np.sqrt(G_prime**2 + G_double_prime**2)

        # Add small noise
        rng = np.random.default_rng(42)
        noise = 1 + 0.02 * rng.standard_normal(len(G_star))
        G_star_noisy = G_star * noise

        # Fit NLSQ
        model.fit(omega, G_star_noisy, test_mode="oscillation")

        # Check NLSQ recovers parameters
        assert model.G0 == pytest.approx(G0_true, rel=0.1)
        assert model.k_d == pytest.approx(k_d_true, rel=0.1)

        # Bayesian
        result = model.fit_bayesian(
            omega,
            G_star_noisy,
            test_mode="oscillation",
            num_warmup=mcmc_config["num_warmup"],
            num_samples=mcmc_config["num_samples"],
            num_chains=mcmc_config["num_chains"],
        )

        assert result is not None
        assert result.posterior_samples is not None

    @pytest.mark.slow
    def test_fit_bayesian_relaxation(self, mcmc_config):
        """Test NLSQ -> NUTS for relaxation data."""
        model = VLBLocal()

        G0_true, k_d_true = 5000.0, 2.0
        t = np.logspace(-2, 1, 30)
        G_t = G0_true * np.exp(-k_d_true * t)

        rng = np.random.default_rng(123)
        noise = 1 + 0.02 * rng.standard_normal(len(G_t))
        G_t_noisy = G_t * noise

        model.fit(t, G_t_noisy, test_mode="relaxation")
        assert model.G0 == pytest.approx(G0_true, rel=0.1)

        result = model.fit_bayesian(
            t,
            G_t_noisy,
            test_mode="relaxation",
            num_warmup=mcmc_config["num_warmup"],
            num_samples=mcmc_config["num_samples"],
            num_chains=mcmc_config["num_chains"],
        )

        assert result is not None


# =============================================================================
# Additional Utility Tests
# =============================================================================


class TestVLBUtilities:
    """Test utility methods."""

    def test_relaxation_spectrum_local(self, vlb_local):
        """Test relaxation spectrum for single mode."""
        spectrum = vlb_local.get_relaxation_spectrum()
        assert len(spectrum) == 1
        G, tau = spectrum[0]
        assert G == pytest.approx(1000.0, rel=1e-10)
        assert tau == pytest.approx(1.0, rel=1e-10)

    def test_relaxation_spectrum_multi(self, vlb_multi_2):
        """Test relaxation spectrum for multi-mode."""
        spectrum = vlb_multi_2.get_relaxation_spectrum()
        assert len(spectrum) == 2
        G0, tau0 = spectrum[0]
        G1, tau1 = spectrum[1]
        assert G0 == pytest.approx(500.0, rel=1e-10)
        assert tau0 == pytest.approx(10.0, rel=1e-10)  # 1/0.1
        assert G1 == pytest.approx(500.0, rel=1e-10)
        assert tau1 == pytest.approx(0.1, rel=1e-10)  # 1/10.0

    def test_weissenberg_deborah(self, vlb_local):
        """Test dimensionless number calculations."""
        Wi = vlb_local.weissenberg_number(2.0)
        assert Wi == pytest.approx(2.0, rel=1e-10)  # t_R * gdot = 1 * 2

        De = vlb_local.deborah_number(3.0)
        assert De == pytest.approx(3.0, rel=1e-10)  # t_R * omega = 1 * 3

    def test_equilibrium_distribution(self):
        """Test equilibrium distribution tensor."""
        mu_eq = VLBLocal.get_equilibrium_distribution()
        np.testing.assert_array_equal(np.asarray(mu_eq), [1.0, 1.0, 1.0, 0.0])

    def test_set_get_parameter_dict(self, vlb_local):
        """Test parameter dict round-trip."""
        params = vlb_local.get_parameter_dict()
        assert params["G0"] == pytest.approx(1000.0)
        assert params["k_d"] == pytest.approx(1.0)

        vlb_local.set_parameter_dict({"G0": 2000.0, "k_d": 0.5})
        assert vlb_local.G0 == pytest.approx(2000.0)
        assert vlb_local.k_d == pytest.approx(0.5)
