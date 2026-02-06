"""Unit tests for VLBVariant (Bell + FENE-P + Temperature extensions).

Tests cover:
- Model creation and parameter setup for all variant combinations
- Regression: VLBVariant(constant, linear) matches VLBLocal for all 6 protocols
- Bell breakage physics: shear thinning, overshoot, LAOS harmonics
- FENE-P physics: bounded extensional stress, strain hardening
- Bell + FENE combined behavior
- Temperature: Arrhenius scaling, T_ref identity
- Bayesian inference (JAX traceability)
- Protocol fitting (NLSQ)
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.vlb import VLBLocal, VLBVariant

jax, jnp = safe_import_jax()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def vlb_constant():
    """VLBVariant with constant breakage, linear stress (= VLBLocal)."""
    model = VLBVariant(breakage="constant", stress_type="linear")
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    return model


@pytest.fixture
def vlb_bell():
    """VLBVariant with Bell breakage."""
    model = VLBVariant(breakage="bell")
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    model.parameters.set_value("nu", 3.0)
    return model


@pytest.fixture
def vlb_fene():
    """VLBVariant with FENE-P stress."""
    model = VLBVariant(stress_type="fene")
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    model.parameters.set_value("L_max", 10.0)
    return model


@pytest.fixture
def vlb_bell_fene():
    """VLBVariant with Bell + FENE-P."""
    model = VLBVariant(breakage="bell", stress_type="fene")
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    model.parameters.set_value("nu", 3.0)
    model.parameters.set_value("L_max", 10.0)
    return model


@pytest.fixture
def vlb_temp():
    """VLBVariant with temperature dependence."""
    model = VLBVariant(temperature=True)
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    model.parameters.set_value("E_a", 50e3)
    model.parameters.set_value("T_ref", 298.15)
    return model


@pytest.fixture
def vlb_local_ref():
    """VLBLocal reference model for regression tests."""
    model = VLBLocal()
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d", 1.0)
    return model


# =============================================================================
# Test Creation (@smoke)
# =============================================================================


class TestVLBVariantCreation:
    """Test model creation for all variant combinations."""

    @pytest.mark.smoke
    def test_creation_constant_linear(self, vlb_constant):
        """Default variant has 3 params: G0, k_d_0, eta_s."""
        assert len(vlb_constant.parameters) == 3
        assert vlb_constant.G0 == 1000.0
        assert vlb_constant.k_d_0 == 1.0
        assert vlb_constant.nu is None
        assert vlb_constant.L_max is None

    @pytest.mark.smoke
    def test_creation_bell(self, vlb_bell):
        """Bell variant adds nu parameter."""
        assert len(vlb_bell.parameters) == 4
        assert vlb_bell.nu == 3.0
        assert vlb_bell.L_max is None

    @pytest.mark.smoke
    def test_creation_fene(self, vlb_fene):
        """FENE variant adds L_max parameter."""
        assert len(vlb_fene.parameters) == 4
        assert vlb_fene.nu is None
        assert vlb_fene.L_max == 10.0

    @pytest.mark.smoke
    def test_creation_bell_fene(self, vlb_bell_fene):
        """Bell + FENE has 5 params."""
        assert len(vlb_bell_fene.parameters) == 5
        assert vlb_bell_fene.nu == 3.0
        assert vlb_bell_fene.L_max == 10.0

    @pytest.mark.smoke
    def test_creation_temperature(self, vlb_temp):
        """Temperature adds E_a and T_ref."""
        assert len(vlb_temp.parameters) == 5
        assert vlb_temp.parameters.get_value("E_a") == 50e3
        assert vlb_temp.parameters.get_value("T_ref") == 298.15

    @pytest.mark.smoke
    def test_registry_lookup(self):
        """VLBVariant registered as 'vlb_variant'."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("vlb_variant")
        assert isinstance(model, VLBVariant)

    @pytest.mark.smoke
    def test_properties(self, vlb_constant):
        """Test relaxation_time and viscosity properties."""
        assert vlb_constant.relaxation_time == pytest.approx(1.0)
        assert vlb_constant.viscosity == pytest.approx(1000.0)

    @pytest.mark.smoke
    def test_repr(self, vlb_bell_fene):
        """Repr includes active flags."""
        r = repr(vlb_bell_fene)
        assert "bell" in r
        assert "fene" in r


# =============================================================================
# Regression Tests (@smoke)
# =============================================================================


class TestVLBVariantRegression:
    """VLBVariant(constant, linear) must match VLBLocal for all 6 protocols."""

    def _params_pair(self):
        """Return (local_params, variant_params)."""
        return jnp.array([1000.0, 1.0]), jnp.array([1000.0, 1.0, 0.0])

    def _relerr(self, a, b):
        return float(jnp.max(jnp.abs(a - b) / jnp.maximum(jnp.abs(a), 1e-20)))

    @pytest.mark.smoke
    def test_flow_curve_regression(self, vlb_constant, vlb_local_ref):
        """Flow curve matches VLBLocal to rtol < 1e-4."""
        p_l, p_v = self._params_pair()
        gdot = np.logspace(-2, 2, 20)
        local = vlb_local_ref.model_function(gdot, p_l, test_mode="flow_curve")
        variant = vlb_constant.model_function(gdot, p_v, test_mode="flow_curve")
        assert self._relerr(local, variant) < 1e-4

    @pytest.mark.smoke
    def test_saos_regression(self, vlb_constant, vlb_local_ref):
        """SAOS matches VLBLocal exactly (both analytical)."""
        p_l, p_v = self._params_pair()
        omega = np.logspace(-1, 2, 20)
        local = vlb_local_ref.model_function(omega, p_l, test_mode="oscillation")
        variant = vlb_constant.model_function(omega, p_v, test_mode="oscillation")
        assert self._relerr(local, variant) < 1e-10

    @pytest.mark.smoke
    def test_startup_regression(self, vlb_constant, vlb_local_ref):
        """Startup matches VLBLocal to rtol < 1e-6."""
        p_l, p_v = self._params_pair()
        t = np.linspace(0.01, 10.0, 50)
        local = vlb_local_ref.model_function(
            t, p_l, test_mode="startup", gamma_dot=1.0
        )
        variant = vlb_constant.model_function(
            t, p_v, test_mode="startup", gamma_dot=1.0
        )
        assert self._relerr(local, variant) < 1e-6

    @pytest.mark.smoke
    def test_relaxation_regression(self, vlb_constant, vlb_local_ref):
        """Relaxation matches VLBLocal exactly (both analytical for constant)."""
        p_l, p_v = self._params_pair()
        t = np.linspace(0.01, 10.0, 50)
        local = vlb_local_ref.model_function(t, p_l, test_mode="relaxation")
        variant = vlb_constant.model_function(t, p_v, test_mode="relaxation")
        assert self._relerr(local, variant) < 1e-10

    @pytest.mark.smoke
    def test_creep_regression(self, vlb_constant, vlb_local_ref):
        """Creep matches VLBLocal to rtol < 1e-6."""
        p_l, p_v = self._params_pair()
        t = np.linspace(0.01, 10.0, 50)
        local = vlb_local_ref.model_function(
            t, p_l, test_mode="creep", sigma_applied=100.0
        )
        variant = vlb_constant.model_function(
            t, p_v, test_mode="creep", sigma_applied=100.0
        )
        assert self._relerr(local, variant) < 1e-6

    @pytest.mark.smoke
    def test_laos_regression(self, vlb_constant, vlb_local_ref):
        """LAOS matches VLBLocal exactly (same ODE approach)."""
        p_l, p_v = self._params_pair()
        t = np.linspace(0.001, 10 * 2 * np.pi, 500)
        local = vlb_local_ref.model_function(
            t, p_l, test_mode="laos", gamma_0=0.1, omega=1.0
        )
        variant = vlb_constant.model_function(
            t, p_v, test_mode="laos", gamma_0=0.1, omega=1.0
        )
        assert self._relerr(local, variant) < 1e-6


# =============================================================================
# Bell Physics Tests
# =============================================================================


class TestVLBBellPhysics:
    """Test Bell breakage physics: force-dependent k_d."""

    def test_shear_thinning(self, vlb_bell):
        """Bell gives shear-thinning: eta(gamma_dot) decreasing."""
        sigma, eta = vlb_bell.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        assert float(eta[0]) > float(eta[1]) > float(eta[2])

    def test_nu_zero_is_newtonian(self):
        """Bell with nu=0 recovers constant k_d (Newtonian)."""
        model = VLBVariant(breakage="bell")
        model.parameters.set_value("G0", 1000.0)
        model.parameters.set_value("k_d_0", 1.0)
        model.parameters.set_value("eta_s", 0.0)
        model.parameters.set_value("nu", 0.0)

        sigma, eta = model.predict_flow_curve(np.array([0.1, 1.0, 10.0]))
        np.testing.assert_allclose(eta, 1000.0, rtol=1e-3)

    def test_stress_overshoot_startup(self, vlb_bell):
        """Bell startup shows stress overshoot at high Wi."""
        t = np.linspace(0.01, 20.0, 500)
        stress = vlb_bell.simulate_startup(t, gamma_dot=10.0)
        # Overshoot: max stress > final stress
        sigma_max = float(np.max(stress))
        sigma_final = float(stress[-1])
        assert sigma_max > sigma_final * 1.01  # At least 1% overshoot

    def test_n1_bell_nonlinear(self, vlb_bell):
        """Bell N1 deviates from quadratic at high shear rates."""
        gdot = np.array([0.01, 1.0, 10.0])
        n1 = vlb_bell.predict_normal_stresses(gdot)
        # For linear VLB: N1 = 2*G0*(gdot/k_d)^2
        n1_linear = 2.0 * 1000.0 * (gdot / 1.0) ** 2
        # At low gdot should be close, at high should differ
        np.testing.assert_allclose(n1[0], n1_linear[0], rtol=0.1)
        # At high gdot, Bell N1 should be LESS than linear (faster relaxation)
        assert float(n1[2]) < float(n1_linear[2])

    def test_laos_harmonics_nonzero(self, vlb_bell):
        """Bell LAOS produces nonzero higher harmonics (I3/I1 > 0)."""
        t = np.linspace(0.001, 20 * 2 * np.pi, 2000)
        vlb_bell.simulate_laos(t, gamma_0=0.5, omega=1.0)
        harmonics = vlb_bell.extract_laos_harmonics(n_harmonics=3)
        assert harmonics["I3_I1"] > 1e-4

    def test_relaxation_stress_decays(self, vlb_bell):
        """Bell relaxation from pre-sheared state decays to zero."""
        t = np.linspace(0.01, 20.0, 100)
        stress = vlb_bell.simulate_relaxation(t, gamma_dot_preshear=10.0)
        # Stress should decay
        assert float(stress[0]) > float(stress[-1])
        # Should be near zero at long times (20/k_d_0 >> 1)
        assert float(stress[-1]) < 0.01 * float(stress[0])

    def test_wi_dependent_viscosity(self, vlb_bell):
        """At Wi → 0, viscosity approaches eta_0 = G0/k_d_0."""
        sigma, eta = vlb_bell.predict_flow_curve(np.array([1e-3]))
        assert eta[0] == pytest.approx(1000.0, rel=0.01)

    def test_extension_softening(self, vlb_bell):
        """Bell gives softer extensional response than constant k_d."""
        eps_dot = np.array([0.01, 0.1, 0.3])
        sigma_bell = vlb_bell.predict_uniaxial_extension(eps_dot)

        linear = VLBVariant()
        linear.parameters.set_value("G0", 1000.0)
        linear.parameters.set_value("k_d_0", 1.0)
        sigma_lin = linear.predict_uniaxial_extension(eps_dot)

        # Bell should give lower extensional stress (faster relaxation at stretch)
        assert float(sigma_bell[-1]) < float(sigma_lin[-1])


# =============================================================================
# FENE-P Physics Tests
# =============================================================================


class TestVLBFenePhysics:
    """Test FENE-P finite extensibility physics."""

    def test_bounded_extensional_stress(self, vlb_fene):
        """FENE-P prevents divergence in extensional stress."""
        # Near the k_d/2 singularity for constant k_d
        eps_dot = np.array([0.01, 0.1, 0.45, 0.49])
        sigma = vlb_fene.predict_uniaxial_extension(eps_dot)
        # All should be finite
        assert np.all(np.isfinite(sigma))

    def test_strain_hardening(self, vlb_fene):
        """FENE-P gives strain hardening relative to linear at moderate rates."""
        gdot = np.logspace(-1, 0.5, 10)
        sigma_fene, _ = vlb_fene.predict_flow_curve(gdot)

        linear = VLBVariant()
        linear.parameters.set_value("G0", 1000.0)
        linear.parameters.set_value("k_d_0", 1.0)
        sigma_lin, _ = linear.predict_flow_curve(gdot)

        # FENE stress should exceed linear at high rates
        assert float(sigma_fene[-1]) > float(sigma_lin[-1])

    def test_large_L_max_recovers_linear(self):
        """Large L_max makes FENE-P approach linear stress."""
        model = VLBVariant(stress_type="fene")
        model.parameters.set_value("G0", 1000.0)
        model.parameters.set_value("k_d_0", 1.0)
        model.parameters.set_value("eta_s", 0.0)
        model.parameters.set_value("L_max", 1000.0)

        linear = VLBVariant()
        linear.parameters.set_value("G0", 1000.0)
        linear.parameters.set_value("k_d_0", 1.0)

        gdot = np.logspace(-1, 1, 10)
        sigma_fene, _ = model.predict_flow_curve(gdot)
        sigma_lin, _ = linear.predict_flow_curve(gdot)

        np.testing.assert_allclose(sigma_fene, sigma_lin, rtol=0.01)

    def test_fene_factor_at_equilibrium(self):
        """FENE factor f(tr(mu)=3) = L^2/L^2 = 1."""
        from rheojax.models.vlb._kernels import vlb_fene_factor

        f = vlb_fene_factor(1.0, 1.0, 1.0, 10.0)
        assert float(f) == pytest.approx(1.0, abs=1e-10)

    def test_fene_n1_bounded(self, vlb_fene):
        """FENE N1 values are all finite (no divergence to inf)."""
        gdot = np.logspace(0, 1.5, 10)
        n1 = vlb_fene.predict_normal_stresses(gdot)
        # All values should be finite (FENE prevents divergence)
        assert np.all(np.isfinite(n1))
        # N1 should increase with shear rate
        assert float(n1[-1]) > float(n1[0])


# =============================================================================
# Bell + FENE Combined Tests
# =============================================================================


class TestVLBBellFeneCombined:
    """Test combined Bell + FENE-P behavior."""

    def test_combined_shear_thinning_plus_hardening(self, vlb_bell_fene):
        """Bell+FENE: shear thinning (viscosity) but stress hardening."""
        sigma, eta = vlb_bell_fene.predict_flow_curve(
            np.array([0.1, 1.0, 10.0])
        )
        # Shear thinning from Bell
        assert float(eta[0]) > float(eta[2])
        # Stress still increases with shear rate
        assert float(sigma[2]) > float(sigma[0])

    def test_combined_extensional_bounded(self, vlb_bell_fene):
        """Bell+FENE: extensional stress is bounded."""
        eps_dot = np.array([0.01, 0.1, 0.3, 0.45])
        sigma = vlb_bell_fene.predict_uniaxial_extension(eps_dot)
        assert np.all(np.isfinite(sigma))

    def test_combined_laos_harmonics(self, vlb_bell_fene):
        """Bell+FENE LAOS produces higher harmonics."""
        t = np.linspace(0.001, 20 * 2 * np.pi, 2000)
        vlb_bell_fene.simulate_laos(t, gamma_0=0.5, omega=1.0)
        harmonics = vlb_bell_fene.extract_laos_harmonics(n_harmonics=3)
        assert harmonics["I3_I1"] > 1e-4


# =============================================================================
# Temperature Tests
# =============================================================================


class TestVLBTemperature:
    """Test Arrhenius temperature dependence."""

    def test_arrhenius_kd_scaling(self, vlb_temp):
        """Higher T increases k_d (more thermal energy)."""
        from rheojax.models.vlb._kernels import vlb_arrhenius_shift

        k_d_low = vlb_arrhenius_shift(1.0, 50e3, 280.0, 298.15)
        k_d_ref = vlb_arrhenius_shift(1.0, 50e3, 298.15, 298.15)
        k_d_high = vlb_arrhenius_shift(1.0, 50e3, 350.0, 298.15)
        assert float(k_d_low) < float(k_d_ref) < float(k_d_high)

    def test_g0_thermal_scaling(self, vlb_temp):
        """G0 scales linearly with T (rubber elasticity)."""
        from rheojax.models.vlb._kernels import vlb_thermal_modulus

        G_low = vlb_thermal_modulus(1000.0, 280.0, 298.15)
        G_ref = vlb_thermal_modulus(1000.0, 298.15, 298.15)
        G_high = vlb_thermal_modulus(1000.0, 350.0, 298.15)
        assert float(G_ref) == pytest.approx(1000.0)
        assert float(G_low) < float(G_ref) < float(G_high)

    def test_t_ref_identity(self, vlb_temp):
        """At T=T_ref, temperature model matches non-temp model."""
        params = vlb_temp._build_params_array()
        omega = np.logspace(-1, 2, 20)

        saos_temp = vlb_temp.model_function(
            omega, params, test_mode="oscillation", T=298.15
        )

        linear = VLBVariant()
        linear.parameters.set_value("G0", 1000.0)
        linear.parameters.set_value("k_d_0", 1.0)
        params_lin = jnp.array([1000.0, 1.0, 0.0])
        saos_ref = linear.model_function(omega, params_lin, test_mode="oscillation")

        np.testing.assert_allclose(
            np.asarray(saos_temp), np.asarray(saos_ref), rtol=1e-10
        )

    def test_shift_factor_recovery(self, vlb_temp):
        """Time-temperature superposition: a_T = k_d(T)/k_d(T_ref)."""
        from rheojax.models.vlb._kernels import vlb_arrhenius_shift

        T = 350.0
        T_ref = 298.15
        E_a = 50e3
        a_T = float(
            vlb_arrhenius_shift(1.0, E_a, T, T_ref)
            / vlb_arrhenius_shift(1.0, E_a, T_ref, T_ref)
        )
        # a_T should be > 1 for T > T_ref
        assert a_T > 1.0
        # Should match Arrhenius formula
        R = 8.314
        expected = np.exp(-E_a / R * (1.0 / T - 1.0 / T_ref))
        assert a_T == pytest.approx(expected, rel=1e-10)

    def test_bayesian_with_temperature(self, vlb_temp):
        """model_function is JAX-traceable with temperature kwarg."""
        params = vlb_temp._build_params_array()
        omega = jnp.logspace(-1, 2, 20)

        @jax.jit
        def predict(p):
            return vlb_temp.model_function(omega, p, test_mode="oscillation", T=320.0)

        result = predict(params)
        assert result.shape == (20,)
        assert jnp.all(jnp.isfinite(result))


# =============================================================================
# Bayesian Tests
# =============================================================================


class TestVLBVariantBayesian:
    """Test JAX traceability for Bayesian inference."""

    @pytest.mark.smoke
    def test_model_function_jax_traceable(self, vlb_constant):
        """model_function compiles under jax.jit for all modes."""
        params = vlb_constant._build_params_array()
        t = jnp.linspace(0.01, 5.0, 20)
        omega = jnp.logspace(-1, 2, 20)
        gdot = jnp.logspace(-2, 2, 20)

        @jax.jit
        def pred_fc(p):
            return vlb_constant.model_function(gdot, p, test_mode="flow_curve")

        @jax.jit
        def pred_saos(p):
            return vlb_constant.model_function(omega, p, test_mode="oscillation")

        @jax.jit
        def pred_startup(p):
            return vlb_constant.model_function(
                t, p, test_mode="startup", gamma_dot=1.0
            )

        assert jnp.all(jnp.isfinite(pred_fc(params)))
        assert jnp.all(jnp.isfinite(pred_saos(params)))
        assert jnp.all(jnp.isfinite(pred_startup(params)))

    def test_bell_model_function_traceable(self, vlb_bell):
        """Bell model_function compiles under jax.jit."""
        params = vlb_bell._build_params_array()
        gdot = jnp.logspace(-2, 2, 10)

        @jax.jit
        def pred(p):
            return vlb_bell.model_function(gdot, p, test_mode="flow_curve")

        result = pred(params)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.slow
    def test_fit_bayesian_bell_flow_curve(self, vlb_bell):
        """Bayesian inference on Bell shear-thinning flow curve."""
        # Generate synthetic data
        gdot = np.logspace(-1, 1, 15)
        sigma_true, _ = vlb_bell.predict_flow_curve(gdot)
        noise = np.random.default_rng(42).normal(0, 0.02 * sigma_true)
        sigma_data = sigma_true + noise

        # Fit — Bell flow curve has G0/k_d_0/nu correlations,
        # so start close to truth for reliable convergence
        model = VLBVariant(breakage="bell")
        model.parameters.set_value("G0", 900.0)
        model.parameters.set_value("k_d_0", 0.8)
        model.parameters.set_value("nu", 2.5)
        model.fit(gdot, sigma_data, test_mode="flow_curve")

        # NLSQ should recover parameters approximately
        # (Bell has strong parameter correlations, so wide tolerance)
        assert model.G0 == pytest.approx(1000.0, rel=0.3)
        assert model.k_d_0 == pytest.approx(1.0, rel=0.5)

        # Bayesian
        result = model.fit_bayesian(
            gdot,
            sigma_data,
            test_mode="flow_curve",
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )
        assert result is not None


# =============================================================================
# Protocol Fitting Tests
# =============================================================================


class TestVLBVariantProtocols:
    """Test NLSQ fitting for various protocols."""

    def test_fit_flow_curve_bell(self, vlb_bell):
        """NLSQ fits shear-thinning flow curve."""
        gdot = np.logspace(-1, 1, 20)
        sigma_true, _ = vlb_bell.predict_flow_curve(gdot)
        noise = np.random.default_rng(42).normal(0, 0.01 * sigma_true)

        model = VLBVariant(breakage="bell")
        model.fit(gdot, sigma_true + noise, test_mode="flow_curve")
        assert model.fitted_

    def test_fit_oscillation(self, vlb_constant):
        """SAOS fitting (analytical Maxwell)."""
        omega = np.logspace(-1, 2, 30)
        Gp, Gpp = vlb_constant.predict_saos(omega)
        G_star = np.sqrt(Gp**2 + Gpp**2)

        model = VLBVariant()
        model.fit(omega, G_star, test_mode="oscillation")
        assert model.G0 == pytest.approx(1000.0, rel=0.1)
        assert model.k_d_0 == pytest.approx(1.0, rel=0.1)

    def test_predict_method(self, vlb_bell):
        """predict() works after fit."""
        gdot = np.logspace(-1, 1, 15)
        sigma, _ = vlb_bell.predict_flow_curve(gdot)

        vlb_bell.fit(gdot, sigma, test_mode="flow_curve")
        pred = vlb_bell.predict(gdot, test_mode="flow_curve")
        assert pred.shape == sigma.shape

    def test_fit_startup_bell(self, vlb_bell):
        """Startup fitting recovers Bell parameters."""
        t = np.linspace(0.01, 10.0, 50)
        stress = vlb_bell.simulate_startup(t, gamma_dot=1.0)

        model = VLBVariant(breakage="bell")
        model.fit(t, stress, test_mode="startup", gamma_dot=1.0)
        assert model.fitted_

    def test_fit_relaxation(self, vlb_constant):
        """Relaxation fitting for constant k_d."""
        t = np.linspace(0.01, 10.0, 50)
        params = vlb_constant._build_params_array()
        G_t = vlb_constant.model_function(t, params, test_mode="relaxation")

        model = VLBVariant()
        model.fit(t, np.asarray(G_t), test_mode="relaxation")
        assert model.G0 == pytest.approx(1000.0, rel=0.1)
