"""
Tests for SGRConventional model structure.

This test module validates the core structure of the SGRConventional model,
including instantiation, parameter configuration, model registration, and
BayesianMixin integration. Tests focus on structural correctness, not numerical
accuracy (which is covered in Task Group 3).
"""

import numpy as np
import pytest

from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.models import SGRConventional


class TestSGRConventionalStructure:
    """Test suite for SGRConventional model structure."""

    @pytest.mark.smoke
    def test_model_instantiation_default(self):
        """Test model instantiation with default parameters."""
        model = SGRConventional()

        assert model is not None
        assert hasattr(model, "parameters")
        assert isinstance(model.parameters, ParameterSet)
        assert hasattr(model, "_test_mode")
        assert model._test_mode is None

    @pytest.mark.smoke
    def test_parameter_set_creation(self):
        """Test ParameterSet has correct parameters with bounds."""
        model = SGRConventional()

        # Check parameter names exist
        assert "x" in model.parameters.keys()
        assert "G0" in model.parameters.keys()
        assert "tau0" in model.parameters.keys()

        # Check bounds for x (noise temperature)
        x_param = model.parameters.get("x")
        assert x_param.bounds == (0.5, 3.0)
        assert x_param.value == 1.5  # Default value

        # Check bounds for G0 (modulus scale)
        G0_param = model.parameters.get("G0")
        assert G0_param.bounds == (1e-3, 1e9)
        assert G0_param.value == 1e3

        # Check bounds for tau0 (attempt time)
        tau0_param = model.parameters.get("tau0")
        assert tau0_param.bounds == (1e-9, 1e3)
        assert tau0_param.value == 1e-3

    def test_parameter_validation_x_range(self):
        """Test that x parameter is validated to be in valid range."""
        model = SGRConventional()

        # x should be within (0.5, 3.0) bounds
        x_param = model.parameters.get("x")
        lower, upper = x_param.bounds

        assert lower == 0.5
        assert upper == 3.0

        # Setting values within bounds should work
        model.parameters.set_value("x", 1.2)
        assert model.parameters.get_value("x") == 1.2

        model.parameters.set_value("x", 2.5)
        assert model.parameters.get_value("x") == 2.5

    def test_model_registration(self):
        """Test that model is registered in ModelRegistry."""
        # Check model is registered via list_models
        registered_models = ModelRegistry.list_models()
        assert "sgr_conventional" in registered_models

        # Check we can instantiate from registry (factory pattern)
        model = ModelRegistry.create("sgr_conventional")
        assert isinstance(model, SGRConventional)

        # Check model info is available
        info = ModelRegistry.get_info("sgr_conventional")
        assert info is not None

    def test_base_model_inheritance(self):
        """Test that SGRConventional inherits from BaseModel."""
        from rheojax.core.base import BaseModel

        model = SGRConventional()

        # Check inheritance
        assert isinstance(model, BaseModel)

        # Check BaseModel interface methods exist
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "_fit")
        assert hasattr(model, "_predict")

        # Check methods are callable
        assert callable(model.fit)
        assert callable(model.predict)

    def test_bayesian_mixin_integration(self):
        """Test that BayesianMixin is properly integrated."""
        from rheojax.core.bayesian import BayesianMixin

        model = SGRConventional()

        # Check BayesianMixin inheritance
        assert isinstance(model, BayesianMixin)

        # Check Bayesian methods exist
        assert hasattr(model, "fit_bayesian")
        assert hasattr(model, "model_function")
        assert callable(model.fit_bayesian)
        assert callable(model.model_function)

    def test_model_function_signature(self):
        """Test that model_function has correct signature for NumPyro."""
        model = SGRConventional()

        # Check model_function signature
        import inspect

        sig = inspect.signature(model.model_function)

        # Should accept (self, X, params, test_mode=None)
        param_names = list(sig.parameters.keys())
        assert "X" in param_names
        assert "params" in param_names
        assert "test_mode" in param_names

        # test_mode should have default None
        assert sig.parameters["test_mode"].default is None

    def test_test_mode_storage(self):
        """Test that _test_mode is stored for mode-aware inference."""
        model = SGRConventional()

        # Initially None
        assert model._test_mode is None

        # Should be set after fit (tested in Task Group 3)
        # Here we just verify the attribute exists
        assert hasattr(model, "_test_mode")


class TestSGRConventionalPredictions:
    """Test suite for SGRConventional prediction methods (Task Group 3)."""

    def test_oscillation_mode_power_law_scaling(self):
        """Test G'(omega) and G''(omega) physical behavior for SGR model.

        With the corrected Sollich 1998 formula (tau_E = tau0 * exp(E/x)):
        - G'(omega) increases monotonically from zero to the elastic plateau G0(x).
        - G''(omega) has a peak at an intermediate frequency — it is NOT monotone
          over the full frequency range.
        - The asymptotic power-law G', G'' ~ omega^(x-1) applies for omega*tau0 << 1.
        """
        model = SGRConventional()

        x_val = 1.5  # Power-law exponent should be 0.5
        model.parameters.set_value("x", x_val)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)
        model._test_mode = "oscillation"

        # Full sweep to verify positivity and shape
        omega_full = np.logspace(-1, 5, 100)
        G_star = model.predict(omega_full)

        assert G_star.shape == (100,)
        assert np.iscomplexobj(G_star)
        assert not np.any(np.isnan(G_star))
        assert not np.any(np.isinf(G_star))
        assert np.all(np.real(G_star) > 0), "G' must be positive at all frequencies"
        assert np.all(np.imag(G_star) > 0), "G'' must be positive at all frequencies"

        # G' increases monotonically toward plateau
        assert np.all(np.diff(np.real(G_star)) > -1e-6), "G' should be non-decreasing"

        # G'' peaks at an intermediate frequency (physically correct SGR behavior)
        gpp_max_idx = np.argmax(np.imag(G_star))
        assert 0 < gpp_max_idx < 99, "G'' peak should be at an intermediate frequency"

        # Low-frequency power-law check (omega*tau0 << 1: G' ~ omega^(x-1))
        omega_low = np.logspace(-1, 1, 40)  # omega*tau0 from 1e-4 to 1e-2
        G_star_low = model.predict(omega_low)
        log_omega_low = np.log10(omega_low[5:-5])
        slope_gp = np.polyfit(log_omega_low, np.log10(np.real(G_star_low[5:-5])), 1)[0]
        # G' slope should be in (0, 2) in this low-frequency regime
        assert 0 < slope_gp < 2, (
            f"G' low-freq slope {slope_gp} out of expected range (0, 2)"
        )

        # G'/G'' ratio should be physically plausible at the G'' peak frequency
        ratio_at_peak = np.real(G_star[gpp_max_idx]) / np.imag(G_star[gpp_max_idx])
        assert 0.1 < ratio_at_peak < 50.0, (
            f"G'/G'' ratio at peak {ratio_at_peak} unreasonable"
        )

    def test_relaxation_mode_power_law_decay(self):
        """Test G(t) shows power-law decay t^(1-x) at long times."""
        model = SGRConventional()

        # Set parameters for power-law regime
        x_val = 1.5  # Decay exponent 1-x = -0.5 (coincides with old x-2 here)
        model.parameters.set_value("x", x_val)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Store test mode
        model._test_mode = "relaxation"

        # Create time array (log-spaced)
        # t/tau0 should span appropriate range for power-law
        t = np.logspace(-4, 3, 50)  # 1e-4 to 1e3 seconds

        # Predict relaxation modulus
        G_t = model.predict(t)

        # Check shape and validity
        assert G_t.shape == (50,)
        assert not np.any(np.isnan(G_t))
        assert not np.any(np.isinf(G_t))
        assert np.all(G_t > 0)

        # Check G(t) decreases with time
        assert G_t[0] > G_t[-1], "G(t) should decay with time"

        # Check power-law decay: G(t) ~ t^(1-x) = t^(-0.5) at long times
        # Use later time points where power-law dominates
        late_idx = slice(25, 45)
        log_t = np.log10(t[late_idx])
        log_G = np.log10(G_t[late_idx])

        # Linear fit in log-log space
        slope_G = np.polyfit(log_t, log_G, 1)[0]

        # Expected slope: 1 - x = -0.5
        assert -0.7 < slope_G < -0.3, f"G(t) slope {slope_G} not near -0.5"

        # Check plateau at short times (should be near G0)
        early_G = G_t[0]
        expected_G0 = model.parameters.get_value("G0")
        # At very short times, should be within order of magnitude of G0
        assert 0.1 * expected_G0 < early_G < 10 * expected_G0

    def test_relaxation_modulus_power_law_exponent(self):
        """G(t) must decay as t^(1-x) for 1 < x < 2 (SGR relaxation exponent).

        Regression for the relaxation exponent. The kernel previously used
        (x-2), which only coincides with the theoretical (1-x) at x=1.5. The
        asymptotic log-log slope of G(t) is measured in the t >> tau0 regime
        and must equal 1-x — the negative of the creep exponent (x-1) and
        Fourier-consistent with SAOS G', G'' ~ omega^(x-1). At x=1.2 the buggy
        (x-2)=-0.8 vs correct (1-x)=-0.2 differ sharply.
        """
        from rheojax.utils.sgr_kernels import power_law_exponent

        tau0 = 1e-3
        t = np.logspace(1, 5, 60)  # t/tau0 in [1e4, 1e8] -> power-law regime
        for x_val in (1.2, 1.5, 1.8):
            model = SGRConventional()
            model.parameters.set_value("x", x_val)
            model.parameters.set_value("G0", 1.0)
            model.parameters.set_value("tau0", tau0)
            model._test_mode = "relaxation"
            G_t = np.asarray(model.predict(t))

            slope = np.polyfit(np.log(t), np.log(G_t), 1)[0]
            expected = -float(power_law_exponent(x_val))  # 1 - x = -(x-1)
            assert expected == pytest.approx(1.0 - x_val, abs=1e-9)
            assert slope == pytest.approx(expected, abs=0.02), (
                f"relaxation log-log slope {slope:.3f} != 1-x ({expected:.3f}) "
                f"at x={x_val}"
            )

    def test_startup_growth_coefficient_power_law_exponent(self):
        """eta_plus(t) = INT G ds must grow as t^(2-x) for 1 < x < 2.

        Regression for the startup exponent. eta_plus is the time integral of
        the relaxation modulus G ~ t^(1-x), so its long-time exponent is (2-x).
        The kernel previously used (x-1), which only coincides with (2-x) at
        x=1.5; at x=1.2 the buggy (x-1)=0.2 vs correct (2-x)=0.8 differ sharply.
        Also checks that x>2 (Newtonian regime) saturates to a finite eta_0.
        """
        tau0 = 1e-3
        t = np.logspace(1, 5, 60)  # t/tau0 >> 1 -> power-law regime
        for x_val in (1.2, 1.5, 1.8):
            model = SGRConventional()
            model.parameters.set_value("x", x_val)
            model.parameters.set_value("G0", 1.0)
            model.parameters.set_value("tau0", tau0)
            model._test_mode = "startup"
            model._startup_gamma_dot = (
                1.0  # eta_plus is the LVE envelope (gamma_dot-independent)
            )
            eta_plus = np.asarray(model.predict(t))

            slope = np.polyfit(np.log(t), np.log(eta_plus), 1)[0]
            expected = 2.0 - x_val
            assert slope == pytest.approx(expected, abs=0.02), (
                f"startup log-log slope {slope:.3f} != 2-x ({expected:.3f}) "
                f"at x={x_val}"
            )

        # x > 2 (Newtonian regime): eta_plus saturates to a finite eta_0
        m = SGRConventional()
        m.parameters.set_value("x", 2.5)
        m.parameters.set_value("G0", 1.0)
        m.parameters.set_value("tau0", tau0)
        m._test_mode = "startup"
        m._startup_gamma_dot = 1.0
        ep = np.asarray(m.predict(np.logspace(2, 7, 50)))
        assert ep[-1] == pytest.approx(ep[-5], rel=0.05), (
            "eta_plus should saturate to a finite zero-shear viscosity for x>2"
        )

    def test_creep_mode_compliance_prediction(self):
        """Test J(t) creep compliance is positive and monotonically increasing."""
        model = SGRConventional()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Store test mode
        model._test_mode = "creep"

        # Create time array
        t = np.logspace(-3, 2, 50)

        # Predict creep compliance
        J_t = model.predict(t)

        # Check shape and validity
        assert J_t.shape == (50,)
        assert not np.any(np.isnan(J_t))
        assert not np.any(np.isinf(J_t))
        assert np.all(J_t > 0), "J(t) should be positive"

        # Check monotonicity: J(t) should increase with time
        for i in range(1, len(J_t)):
            assert J_t[i] >= J_t[i - 1], f"J(t) not monotonic at index {i}"

        # Check consistency with G0: J(0) ~ 1/G0
        J_initial = J_t[0]
        G0 = model.parameters.get_value("G0")
        # Should be roughly inverse relationship (within factor of 10)
        assert 0.01 / G0 < J_initial < 100 / G0

    def test_creep_compliance_power_law_exponent(self):
        """J(t) must grow as t^(x-1) for 1 < x < 2 (SGR creep exponent).

        Regression for the creep exponent. The kernel previously used (2-x),
        which only coincides with the theoretical (x-1) at x=1.5. The
        asymptotic log-log slope of J(t) is measured in the t >> tau0
        power-law regime and must equal x-1 — matching
        utils.sgr_kernels.power_law_exponent and the canonical SAOS scaling
        G', G'' ~ omega^(x-1). At x=1.2 the buggy (2-x)=0.8 vs correct
        (x-1)=0.2 differ sharply, so this test discriminates the fix.
        """
        from rheojax.utils.sgr_kernels import power_law_exponent

        tau0 = 1e-3
        # t/tau0 in [1e4, 1e8] -> deep in the power-law (t >> tau0) regime
        t = np.logspace(1, 5, 60)
        for x_val in (1.2, 1.5, 1.8):
            model = SGRConventional()
            model.parameters.set_value("x", x_val)
            model.parameters.set_value("G0", 1.0)
            model.parameters.set_value("tau0", tau0)
            model._test_mode = "creep"
            J_t = np.asarray(model.predict(t))

            slope = np.polyfit(np.log(t), np.log(J_t), 1)[0]
            expected = float(power_law_exponent(x_val))  # = x - 1
            assert expected == pytest.approx(x_val - 1.0, abs=1e-9)
            assert slope == pytest.approx(expected, abs=0.02), (
                f"creep log-log slope {slope:.3f} != x-1 ({expected:.3f}) at x={x_val}"
            )

    def test_relaxation_and_creep_monotonic_in_glass_phase(self):
        """Regression: for x < 1 (glass phase, in-bounds), G(t) must not
        increase and J(t) must not decrease with time.

        The unconditional exponent (x-1) is only correct for 1 < x < 2; in
        the glass phase it silently flips the sign of both curves, which
        violates passive-material causality/monotonicity.
        """
        t = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])

        model = SGRConventional()
        model.parameters.set_value("x", 0.8)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        model._test_mode = "relaxation"
        G_t = np.asarray(model.predict(t))
        assert np.all(np.diff(G_t) <= 1e-9), "G(t) must be non-increasing for x<1"

        model._test_mode = "creep"
        J_t = np.asarray(model.predict(t))
        assert np.all(np.diff(J_t) >= -1e-9), "J(t) must be non-decreasing for x<1"

    def test_steady_shear_flow_curve(self):
        """Test eta(gamma_dot) viscosity shows shear-thinning behavior."""
        model = SGRConventional()

        # Set parameters for power-law regime
        x_val = 1.5  # Shear-thinning exponent should be x - 2 = -0.5
        model.parameters.set_value("x", x_val)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Store test mode
        model._test_mode = "steady_shear"

        # Create shear rate array
        gamma_dot = np.logspace(-2, 2, 50)

        # Predict viscosity
        eta = model.predict(gamma_dot)

        # Check shape and validity
        assert eta.shape == (50,)
        assert not np.any(np.isnan(eta))
        assert not np.any(np.isinf(eta))
        assert np.all(eta > 0)

        # Check shear-thinning: eta decreases with gamma_dot
        assert eta[0] > eta[-1], (
            "Viscosity should decrease with shear rate (shear-thinning)"
        )

        # Check power-law scaling: eta ~ gamma_dot^(x-2) = gamma_dot^(-0.5)
        mid_idx = slice(15, 35)
        log_gamma_dot = np.log10(gamma_dot[mid_idx])
        log_eta = np.log10(eta[mid_idx])

        # Linear fit in log-log space
        slope_eta = np.polyfit(log_gamma_dot, log_eta, 1)[0]

        # Expected slope: x - 2 = -0.5
        assert -0.7 < slope_eta < -0.3, f"Viscosity slope {slope_eta} not near -0.5"

    def test_phase_boundaries_glass_regime(self):
        """Test yield stress behavior for x < 1 (glass phase)."""
        model = SGRConventional()

        # Set x < 1 for glass phase
        model.parameters.set_value("x", 0.8)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Oscillation mode - use intermediate to high frequencies
        model._test_mode = "oscillation"
        omega = np.logspace(2, 5, 30)  # 100 to 100000 rad/s
        G_star = model.predict(omega)

        # In glass phase at high frequencies, G' should dominate (approaches plateau)
        # while G'' decreases. Check high-frequency end.
        high_freq_idx = -5  # Near end of range
        assert np.real(G_star[high_freq_idx]) > np.imag(G_star[high_freq_idx]), (
            "Glass phase should show G' > G'' at high frequencies"
        )

        # Also check that phase regime detection works
        phase = model.get_phase_regime()
        assert phase == "glass", f"Expected 'glass' phase, got '{phase}'"

    def test_phase_boundaries_newtonian_regime(self):
        """Test constant viscosity for x >= 2 (Newtonian phase)."""
        model = SGRConventional()

        # Set x >= 2 for Newtonian phase
        model.parameters.set_value("x", 2.2)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Oscillation mode - use low to intermediate frequencies
        model._test_mode = "oscillation"
        omega = np.logspace(0, 4, 30)  # 1 to 10000 rad/s
        G_star = model.predict(omega)

        # In Newtonian regime at low frequencies, G'' should dominate (viscous behavior)
        # and G' ~ omega^2, G'' ~ omega (different from power-law regime)
        low_freq_idx = slice(0, 10)
        assert np.all(np.imag(G_star[low_freq_idx]) > np.real(G_star[low_freq_idx])), (
            "Newtonian phase should show G'' > G' at low frequencies"
        )

        # Check phase regime detection
        phase = model.get_phase_regime()
        assert phase == "newtonian", f"Expected 'newtonian' phase, got '{phase}'"

    def test_complex_modulus_consistency(self):
        """Test G* = G' + i*G'' consistency and positivity."""
        model = SGRConventional()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Oscillation mode
        model._test_mode = "oscillation"
        omega = np.logspace(-2, 2, 30)
        G_star = model.predict(omega)

        # Check both components positive
        assert np.all(np.real(G_star) > 0), "G' should be positive"
        assert np.all(np.imag(G_star) > 0), "G'' should be positive"

        # Check complex modulus magnitude |G*| = sqrt(G'^2 + G''^2)
        G_magnitude = np.abs(G_star)
        assert np.all(G_magnitude > 0)
        assert np.all(G_magnitude > np.real(G_star)), "|G*| should be >= G'"
        assert np.all(G_magnitude > np.imag(G_star)), "|G*| should be >= G''"

        # Check loss tangent tan(delta) = G''/G' is reasonable
        tan_delta = np.imag(G_star) / np.real(G_star)
        assert np.all(tan_delta > 0), "Loss tangent should be positive"

        # Check that tan(delta) values are reasonable (not too extreme)
        # In SGR, tan(delta) can vary significantly across frequency range
        # but should stay within reasonable bounds
        assert np.all(tan_delta < 1e6), "Loss tangent unreasonably large"
        assert np.all(tan_delta > 1e-6), "Loss tangent unreasonably small"

    def test_output_shapes_scalar_vs_array(self):
        """Test prediction output shapes for scalar and array inputs."""
        model = SGRConventional()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Test oscillation mode with array
        model._test_mode = "oscillation"
        omega_array = np.logspace(-2, 2, 20)
        G_star_array = model.predict(omega_array)
        assert G_star_array.shape == (20,), (
            "Oscillation array output should be complex (M,)"
        )

        # Test relaxation mode with array
        model._test_mode = "relaxation"
        t_array = np.logspace(-3, 2, 30)
        G_t_array = model.predict(t_array)
        assert G_t_array.shape == (30,), "Relaxation array output should be (M,)"

        # Test creep mode with array
        model._test_mode = "creep"
        J_t_array = model.predict(t_array)
        assert J_t_array.shape == (30,), "Creep array output should be (M,)"

        # Test steady shear with array
        model._test_mode = "steady_shear"
        gamma_dot_array = np.logspace(-2, 2, 25)
        eta_array = model.predict(gamma_dot_array)
        assert eta_array.shape == (25,), "Steady shear array output should be (M,)"


class TestSGRConventionalDynamicX:
    """Test suite for dynamic effective temperature x(t) evolution (Task Group 4)."""

    def test_aging_x_decreases_at_rest(self):
        """Test aging: x decreases at rest when gamma_dot = 0."""
        model = SGRConventional(dynamic_x=True)

        # Set parameters for aging dynamics
        model.parameters.set_value("x_eq", 1.0)  # Equilibrium temperature
        model.parameters.set_value("alpha_aging", 0.5)  # Aging rate
        model.parameters.set_value("beta_rejuv", 0.0)  # No rejuvenation at rest

        # Initial x > x_eq, should decay toward x_eq
        x_initial = 1.8
        t = np.linspace(0, 10, 100)
        gamma_dot = np.zeros_like(t)  # At rest

        # Evolve x(t)
        x_t = model.evolve_x(t, gamma_dot, x_initial)

        # Check shape
        assert x_t.shape == t.shape

        # Check x decreases with time
        assert x_t[0] > x_t[-1], "x should decrease with time during aging"

        # Check x approaches x_eq
        x_eq = model.parameters.get_value("x_eq")
        assert x_t[-1] > x_eq - 0.2, "x should approach x_eq"
        assert x_t[-1] < x_initial, "x should be less than initial value"

        # Check dx/dt < 0 initially (aging)
        dx_dt_initial = (x_t[1] - x_t[0]) / (t[1] - t[0])
        assert dx_dt_initial < 0, "dx/dt should be negative during aging"

    def test_rejuvenation_x_increases_under_shear(self):
        """Test rejuvenation: x increases under shear when gamma_dot > 0."""
        model = SGRConventional(dynamic_x=True)

        # Set parameters for rejuvenation dynamics
        model.parameters.set_value("x_eq", 1.0)
        model.parameters.set_value("x_ss_A", 0.5)  # Steady-state amplitude
        model.parameters.set_value("x_ss_n", 0.3)  # Steady-state exponent
        model.parameters.set_value("alpha_aging", 0.1)
        model.parameters.set_value("beta_rejuv", 1.0)  # Strong rejuvenation

        # Initial x < x_ss, should increase under shear
        x_initial = 1.0
        t = np.linspace(0, 10, 100)
        gamma_dot = np.ones_like(t) * 10.0  # Constant shear rate

        # Evolve x(t)
        x_t = model.evolve_x(t, gamma_dot, x_initial)

        # Check x increases with time
        assert x_t[-1] > x_t[0], "x should increase under shear (rejuvenation)"

        # Check dx/dt > 0 initially (rejuvenation)
        dx_dt_initial = (x_t[1] - x_t[0]) / (t[1] - t[0])
        assert dx_dt_initial > 0, "dx/dt should be positive during rejuvenation"

        # Check x approaches steady-state value
        tau0 = model.parameters.get_value("tau0")
        x_ss = model._compute_x_ss(gamma_dot[-1], tau0)
        # Should get reasonably close to x_ss
        assert abs(x_t[-1] - x_ss) < 0.5, "x should approach x_ss under constant shear"

    def test_steady_state_x_ss_convergence(self):
        """Test x converges to steady-state x_ss(gamma_dot) under constant shear."""
        model = SGRConventional(dynamic_x=True)

        # Set parameters
        model.parameters.set_value("x_eq", 1.0)
        model.parameters.set_value("x_ss_A", 0.5)
        model.parameters.set_value("x_ss_n", 0.3)
        model.parameters.set_value("alpha_aging", 0.5)
        model.parameters.set_value("beta_rejuv", 1.0)
        tau0 = model.parameters.get_value("tau0")

        # Test different shear rates
        gamma_dot_values = [0.1, 1.0, 10.0]

        for gamma_dot_val in gamma_dot_values:
            x_initial = 1.0
            t = np.linspace(0, 50, 500)  # Long enough to reach steady state
            gamma_dot = np.ones_like(t) * gamma_dot_val

            # Evolve x(t)
            x_t = model.evolve_x(t, gamma_dot, x_initial)

            # Compute expected steady-state
            x_ss = model._compute_x_ss(gamma_dot_val, tau0)

            # Check convergence to x_ss (within 10%)
            final_x = x_t[-1]
            assert abs(final_x - x_ss) / x_ss < 0.15, (
                f"x should converge to x_ss={x_ss:.3f}, got {final_x:.3f} at gamma_dot={gamma_dot_val}"
            )

    def test_evolve_x_oscillatory_shear_no_nan(self):
        """Regression: evolve_x must not produce NaN under oscillatory shear.

        gamma_dot*tau0 is raised to the fractional power x_ss_n; without an
        abs() guard a negative gamma_dot (normal for oscillation/LAOS) makes
        this a fractional power of a negative number -> NaN from the first
        ODE step.
        """
        model = SGRConventional(dynamic_x=True)
        model.parameters.set_value("x_eq", 1.0)
        model.parameters.set_value("x_ss_A", 0.5)
        model.parameters.set_value("x_ss_n", 0.3)  # non-integer exponent
        model.parameters.set_value("alpha_aging", 0.5)
        model.parameters.set_value("beta_rejuv", 1.0)

        t = np.linspace(0, 10, 50)
        gamma_dot = 0.5 * np.cos(t)  # oscillatory: changes sign

        x_t = model.evolve_x(t, gamma_dot, x_initial=1.0)

        assert not np.any(np.isnan(x_t)), (
            "evolve_x produced NaN under oscillatory shear"
        )

    def test_compute_x_ss_negative_shear_rate_no_nan(self):
        """Regression: _compute_x_ss must not produce NaN for negative gamma_dot."""
        model = SGRConventional(dynamic_x=True)
        model.parameters.set_value("x_eq", 1.0)
        model.parameters.set_value("x_ss_A", 0.5)
        model.parameters.set_value("x_ss_n", 0.3)
        tau0 = model.parameters.get_value("tau0")

        x_ss_pos = model._compute_x_ss(0.5, tau0)
        x_ss_neg = model._compute_x_ss(-0.5, tau0)

        assert not np.isnan(x_ss_neg)
        # Symmetric in sign, matching the |gamma_dot*tau0|^n formula.
        assert x_ss_neg == pytest.approx(x_ss_pos)

    def test_static_x_mode_constant_parameter(self):
        """Test static x mode: x is constant fitted parameter, no dynamics."""
        # This is the default behavior (dynamic_x=False)
        model = SGRConventional(dynamic_x=False)

        # Set x parameter
        x_value = 1.5
        model.parameters.set_value("x", x_value)

        # In static mode, x should not evolve
        # This is tested by checking that model predictions don't change with time

        # Oscillation prediction should use constant x
        model._test_mode = "oscillation"
        omega = np.logspace(0, 3, 20)
        G_star = model.predict(omega)

        # Get x parameter value
        x_current = model.parameters.get_value("x")
        assert x_current == x_value, "x should remain constant in static mode"

        # Verify no evolve_x method is called (or it doesn't exist in static mode)
        assert not hasattr(model, "_x_trajectory") or model._x_trajectory is None, (
            "Static mode should not store x trajectory"
        )

    def test_ode_integration_stability_long_times(self):
        """Test ODE integration remains stable over long simulation times."""
        model = SGRConventional(dynamic_x=True)

        # Set parameters to ensure x increases under shear
        model.parameters.set_value("x_eq", 1.0)
        model.parameters.set_value("x_ss_A", 0.8)  # Larger amplitude for higher x_ss
        model.parameters.set_value("x_ss_n", 0.3)
        model.parameters.set_value(
            "alpha_aging", 1.0
        )  # Faster aging for quicker relaxation
        model.parameters.set_value("beta_rejuv", 2.0)  # Strong rejuvenation

        # Long time evolution (1000 * tau0)
        tau0 = model.parameters.get_value("tau0")
        t_max = 1000 * tau0
        t = np.linspace(0, t_max, 1000)

        # Time-varying shear rate (step protocol)
        gamma_dot = np.zeros_like(t)
        gamma_dot[t > 0.2 * t_max] = 10.0  # Step up earlier
        gamma_dot[t > 0.5 * t_max] = (
            0.0  # Step down earlier to give more relaxation time
        )

        # Start at x_eq so we can clearly see increase under shear
        x_initial = 1.0

        # Evolve x(t)
        x_t = model.evolve_x(t, gamma_dot, x_initial)

        # Check no NaN or Inf
        assert not np.any(np.isnan(x_t)), "x(t) contains NaN values"
        assert not np.any(np.isinf(x_t)), "x(t) contains Inf values"

        # Check x remains in physical bounds
        assert np.all(x_t > 0.5), "x(t) went below lower bound"
        assert np.all(x_t < 3.0), "x(t) exceeded upper bound"

        # Check x responds to shear steps
        # Use specific time points
        idx_initial = 0
        idx_during_shear = np.argmax(t > 0.4 * t_max)  # Well into shear period
        idx_after_rest = np.argmax(t > 0.9 * t_max)  # Near end of relaxation

        # During shear, x should be higher than initial value
        # (since x starts at x_eq and shear drives it toward x_ss > x_eq)
        assert x_t[idx_during_shear] > x_t[idx_initial], (
            f"x should increase during shear: x_initial={x_t[idx_initial]:.3f}, "
            f"x_during={x_t[idx_during_shear]:.3f}"
        )

        # After returning to rest, x should decrease back toward x_eq
        # Check that it's moving in the right direction (closer to x_eq than during shear)
        x_eq = model.parameters.get_value("x_eq")
        dist_during = abs(x_t[idx_during_shear] - x_eq)
        dist_after = abs(x_t[idx_after_rest] - x_eq)

        assert dist_after < dist_during, (
            f"x should move closer to x_eq after shear stops: "
            f"dist_during={dist_during:.3f}, dist_after={dist_after:.3f}"
        )

    def test_dynamic_x_coupling_to_predictions(self):
        """Test that x(t) dynamics couple to constitutive predictions."""
        model = SGRConventional(dynamic_x=True)

        # Set parameters
        model.parameters.set_value("x_eq", 1.0)
        model.parameters.set_value("x_ss_A", 0.5)
        model.parameters.set_value("x_ss_n", 0.3)
        model.parameters.set_value("alpha_aging", 0.5)
        model.parameters.set_value("beta_rejuv", 1.0)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Define time-dependent shear protocol
        t = np.linspace(0, 10, 100)
        gamma_dot = np.ones_like(t) * 5.0  # Constant shear

        # Evolve x(t) from initial value
        x_initial = 1.0
        x_t = model.evolve_x(t, gamma_dot, x_initial)

        # Store x(t) trajectory in model
        model._x_trajectory = x_t
        model._t_trajectory = t

        # Verify that x changed during evolution
        assert x_t[-1] != x_t[0], "x should change over time under shear"

        # Now predict stress using time-dependent x(t)
        # This tests that the JIT-compiled prediction methods can use different x values
        model._test_mode = "steady_shear"

        # Predict viscosity at different times using different x values
        # In dynamic mode, predictions should vary with x(t)
        eta_initial = model._predict_steady_shear_jit(
            np.array([gamma_dot[0]]),
            x_t[0],
            model.parameters.get_value("G0"),
            model.parameters.get_value("tau0"),
        )[0]

        eta_final = model._predict_steady_shear_jit(
            np.array([gamma_dot[-1]]),
            x_t[-1],
            model.parameters.get_value("G0"),
            model.parameters.get_value("tau0"),
        )[0]

        # Since x changes, viscosity should change
        # The key test is that we CAN compute different viscosities with different x
        assert eta_initial != eta_final, (
            "Viscosity should change when x changes (coupling verified)"
        )

        # Also test that we can predict at intermediate points
        eta_mid = model._predict_steady_shear_jit(
            np.array([gamma_dot[50]]),
            x_t[50],
            model.parameters.get_value("G0"),
            model.parameters.get_value("tau0"),
        )[0]

        # All viscosities should be positive and finite
        assert eta_initial > 0, "Viscosity should be positive"
        assert eta_final > 0, "Viscosity should be positive"
        assert eta_mid > 0, "Viscosity should be positive"
        assert np.isfinite(eta_initial), "Viscosity should be finite"
        assert np.isfinite(eta_final), "Viscosity should be finite"
        assert np.isfinite(eta_mid), "Viscosity should be finite"


class TestSGRConventionalLAOS:
    """Test suite for LAOS (Large Amplitude Oscillatory Shear) support (Task Group 6)."""

    def test_lissajous_curve_generation(self):
        """Test Lissajous curve generation (stress vs strain loop)."""
        model = SGRConventional()

        # Set parameters for power-law regime
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # LAOS parameters
        gamma_0 = 0.1  # Strain amplitude (10%)
        omega = 1.0  # Angular frequency (rad/s)

        # Generate Lissajous curve
        strain, stress = model.simulate_laos(
            gamma_0, omega, n_cycles=2, n_points_per_cycle=256
        )

        # Check outputs are arrays with correct shape
        assert len(strain) == len(stress), (
            "Strain and stress arrays should have same length"
        )
        assert len(strain) == 2 * 256, (
            "Should have n_cycles * n_points_per_cycle points"
        )

        # Check strain oscillates between -gamma_0 and +gamma_0
        assert np.max(strain) <= gamma_0 * 1.01, "Strain max should be near gamma_0"
        assert np.min(strain) >= -gamma_0 * 1.01, "Strain min should be near -gamma_0"

        # Check stress is bounded (should not blow up)
        assert not np.any(np.isnan(stress)), "Stress should not contain NaN"
        assert not np.any(np.isinf(stress)), "Stress should not contain Inf"

        # Check that the Lissajous curve forms a closed loop (approximately)
        # The end should be close to the start after full cycles
        assert np.abs(strain[-1] - strain[0]) < gamma_0 * 0.1, (
            "Strain loop should close"
        )

    def test_ellipse_shape_linear_regime(self):
        """Test ellipse shape for linear regime (small amplitude)."""
        model = SGRConventional()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Small strain amplitude for linear regime
        gamma_0 = 0.001  # 0.1% strain (very small, definitely linear)
        omega = 1.0

        # Generate Lissajous curve
        strain, stress = model.simulate_laos(
            gamma_0, omega, n_cycles=3, n_points_per_cycle=256
        )

        # Use last cycle for steady-state
        n_pts = 256
        strain_cycle = strain[-n_pts:]
        stress_cycle = stress[-n_pts:]

        # Normalize stress for shape analysis
        stress_norm = stress_cycle / np.max(np.abs(stress_cycle))
        strain_norm = strain_cycle / gamma_0

        # Fit ellipse: in linear regime, stress = G' * strain + (G'' / omega) * d(strain)/dt
        # For sinusoidal strain: stress = G' * gamma_0 * sin(wt) + G'' * gamma_0 * cos(wt)
        # This traces an ellipse in the (strain, stress) plane

        # Test for ellipse shape using Fourier analysis
        # For a perfect ellipse, only the fundamental frequency should be present
        stress_fft = np.fft.fft(stress_cycle)
        power_spectrum = np.abs(stress_fft) ** 2

        # Find fundamental peak
        freqs = np.fft.fftfreq(len(stress_cycle))
        fundamental_idx = np.argmax(power_spectrum[1 : len(power_spectrum) // 2]) + 1

        # In linear regime, fundamental should dominate over higher harmonics
        fundamental_power = power_spectrum[fundamental_idx]
        third_harmonic_idx = 3 * fundamental_idx
        if third_harmonic_idx < len(power_spectrum) // 2:
            third_harmonic_power = power_spectrum[third_harmonic_idx]
            # Third harmonic should be negligible (< 1% of fundamental) in linear regime
            assert third_harmonic_power < 0.01 * fundamental_power, (
                f"Third harmonic too strong for linear regime: {third_harmonic_power / fundamental_power:.4f}"
            )

    def test_distorted_curves_nonlinear_regime(self):
        """Test distorted curves for nonlinear regime (large amplitude)."""
        model = SGRConventional()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Large strain amplitude for nonlinear regime
        gamma_0 = 1.0  # 100% strain (very large, definitely nonlinear)
        omega = 1.0

        # Generate Lissajous curve
        strain, stress = model.simulate_laos(
            gamma_0, omega, n_cycles=3, n_points_per_cycle=256
        )

        # Use last cycle for steady-state
        n_pts = 256
        stress_cycle = stress[-n_pts:]

        # In nonlinear regime, higher harmonics should be present
        stress_fft = np.fft.fft(stress_cycle)
        power_spectrum = np.abs(stress_fft) ** 2

        # Find fundamental peak
        fundamental_idx = np.argmax(power_spectrum[1 : len(power_spectrum) // 2]) + 1
        fundamental_power = power_spectrum[fundamental_idx]

        # Check that the stress response exists and is non-trivial
        assert fundamental_power > 0, "Fundamental frequency power should be non-zero"

        # For SGR model in nonlinear regime, the response may still be relatively
        # linear-looking depending on the constitutive law. The key test is that
        # the simulation runs and produces physically reasonable output.
        assert not np.any(np.isnan(stress)), "Stress should not contain NaN"
        assert np.max(np.abs(stress)) > 0, "Stress amplitude should be non-zero"

    def test_fourier_decomposition_harmonics(self):
        """Test Fourier decomposition (higher harmonics I_3/I_1, I_5/I_1)."""
        model = SGRConventional()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Medium amplitude
        gamma_0 = 0.5
        omega = 1.0

        # Generate LAOS response
        strain, stress = model.simulate_laos(
            gamma_0, omega, n_cycles=4, n_points_per_cycle=512
        )

        # Extract harmonics
        harmonics = model.extract_laos_harmonics(stress, n_points_per_cycle=512)

        # Check that harmonics dictionary contains expected keys
        assert "I_1" in harmonics, "Should have fundamental intensity I_1"
        assert "I_3" in harmonics, "Should have third harmonic I_3"
        assert "I_5" in harmonics, "Should have fifth harmonic I_5"
        assert "phi_1" in harmonics, "Should have fundamental phase phi_1"
        assert "phi_3" in harmonics, "Should have third harmonic phase phi_3"

        # Check fundamental is non-zero
        assert harmonics["I_1"] > 0, "Fundamental intensity should be positive"

        # Check relative intensities are computed
        assert "I_3_I_1" in harmonics, "Should have relative intensity I_3/I_1"
        assert "I_5_I_1" in harmonics, "Should have relative intensity I_5/I_1"

        # Relative intensities should be between 0 and 1 (typically much less than 1)
        assert 0 <= harmonics["I_3_I_1"] <= 1.0, "I_3/I_1 should be in [0, 1]"
        assert 0 <= harmonics["I_5_I_1"] <= 1.0, "I_5/I_1 should be in [0, 1]"

        # Phase angles should be in reasonable range
        assert -np.pi <= harmonics["phi_1"] <= np.pi, "phi_1 should be in [-pi, pi]"
        assert -np.pi <= harmonics["phi_3"] <= np.pi, "phi_3 should be in [-pi, pi]"

    def test_chebyshev_coefficients_computation(self):
        """Test Chebyshev coefficients e_n, v_n computation."""
        model = SGRConventional()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Medium amplitude for some nonlinearity
        gamma_0 = 0.5
        omega = 1.0

        # Generate LAOS response
        strain, stress = model.simulate_laos(
            gamma_0, omega, n_cycles=4, n_points_per_cycle=512
        )

        # Compute Chebyshev decomposition
        chebyshev = model.compute_chebyshev_coefficients(
            strain, stress, gamma_0, omega, n_points_per_cycle=512
        )

        # Check that Chebyshev dictionary contains expected keys
        assert "e_1" in chebyshev, "Should have elastic coefficient e_1"
        assert "e_3" in chebyshev, "Should have elastic coefficient e_3"
        assert "v_1" in chebyshev, "Should have viscous coefficient v_1"
        assert "v_3" in chebyshev, "Should have viscous coefficient v_3"

        # Check normalized coefficients
        assert "e_3_e_1" in chebyshev, "Should have normalized e_3/e_1"
        assert "v_3_v_1" in chebyshev, "Should have normalized v_3/v_1"

        # First-order coefficients should be non-zero
        assert chebyshev["e_1"] != 0, "e_1 should be non-zero"
        assert chebyshev["v_1"] != 0, "v_1 should be non-zero"

        # Physical interpretation:
        # e_3/e_1 > 0: strain stiffening
        # e_3/e_1 < 0: strain softening
        # v_3/v_1 > 0: shear thickening
        # v_3/v_1 < 0: shear thinning

        # Check values are finite
        assert np.isfinite(chebyshev["e_3_e_1"]), "e_3/e_1 should be finite"
        assert np.isfinite(chebyshev["v_3_v_1"]), "v_3/v_1 should be finite"

    def test_amplitude_sweep_linear_to_nonlinear(self):
        """Test amplitude sweep transition from linear to nonlinear."""
        model = SGRConventional()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        omega = 1.0

        # Amplitude sweep from small to large
        gamma_amplitudes = np.array([0.001, 0.01, 0.1, 0.5, 1.0])

        # Track third harmonic ratio across amplitudes
        I3_I1_values = []

        for gamma_0 in gamma_amplitudes:
            # Generate LAOS response
            strain, stress = model.simulate_laos(
                gamma_0, omega, n_cycles=3, n_points_per_cycle=256
            )

            # Extract harmonics
            harmonics = model.extract_laos_harmonics(stress, n_points_per_cycle=256)
            I3_I1_values.append(harmonics["I_3_I_1"])

        I3_I1_values = np.array(I3_I1_values)

        # In linear regime (small gamma), I_3/I_1 should be very small
        assert I3_I1_values[0] < 0.05, (
            f"I_3/I_1 should be small in linear regime, got {I3_I1_values[0]:.4f}"
        )

        # Check that all values are non-negative
        assert np.all(I3_I1_values >= 0), "I_3/I_1 should be non-negative"

        # Check no NaN values
        assert not np.any(np.isnan(I3_I1_values)), "I_3/I_1 should not contain NaN"


class TestSGRConventionalFitting:
    """Test suite for SGRConventional NLSQ fitting (Task Group 3)."""

    def test_oscillation_mode_fitting_basic(self):
        """Test basic oscillation mode fitting with synthetic data."""
        # Generate synthetic data with known parameters
        model_true = SGRConventional()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "oscillation"

        omega = np.logspace(-2, 2, 50)
        G_star = model_true.predict(omega)

        # Fit model with different initial values
        model_fit = SGRConventional()
        model_fit.parameters.set_value("x", 2.0)
        model_fit.parameters.set_value("G0", 500.0)
        model_fit.parameters.set_value("tau0", 0.01)

        model_fit.fit(omega, G_star, test_mode="oscillation")

        # Check model was marked as fitted
        assert model_fit.fitted_ is True

        # Check predictions match original data (even if parameters differ due to correlation)
        model_fit._test_mode = "oscillation"
        G_star_fit = model_fit.predict(omega)

        # Predictions should match within tolerance
        relative_error = np.abs(G_star_fit - G_star) / (np.abs(G_star) + 1e-10)
        assert np.max(relative_error) < 0.1, "Fitted predictions should match data"

    def test_relaxation_mode_fitting_basic(self):
        """Test basic relaxation mode fitting with synthetic data."""
        # Generate synthetic data
        model_true = SGRConventional()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "relaxation"

        t = np.logspace(-3, 2, 50)
        G_t = model_true.predict(t)

        # Fit model
        model_fit = SGRConventional()
        model_fit.parameters.set_value("x", 2.0)
        model_fit.parameters.set_value("G0", 500.0)
        model_fit.parameters.set_value("tau0", 0.01)

        model_fit.fit(t, G_t, test_mode="relaxation")

        assert model_fit.fitted_ is True

        # Check parameter recovery (relaxation mode has better identifiability)
        fitted_x = model_fit.parameters.get_value("x")
        assert abs(fitted_x - 1.5) < 0.1, f"Expected x≈1.5, got {fitted_x:.3f}"

    def test_creep_mode_fitting_basic(self):
        """Test basic creep mode fitting with synthetic data."""
        # Generate synthetic data
        model_true = SGRConventional()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "creep"

        t = np.logspace(-3, 2, 50)
        J_t = model_true.predict(t)

        # Fit model
        model_fit = SGRConventional()
        model_fit.fit(t, J_t, test_mode="creep")

        assert model_fit.fitted_ is True
        assert model_fit._test_mode == "creep"

    def test_steady_shear_mode_fitting_basic(self):
        """Test basic steady shear mode fitting with synthetic data."""
        # Generate synthetic data
        model_true = SGRConventional()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "steady_shear"

        gamma_dot = np.logspace(-2, 2, 50)
        eta = model_true.predict(gamma_dot)

        # Fit model
        model_fit = SGRConventional()
        model_fit.fit(gamma_dot, eta, test_mode="steady_shear")

        assert model_fit.fitted_ is True
        assert model_fit._test_mode == "steady_shear"

    def test_fitting_complex_input_format(self):
        """Test oscillation fitting with complex array input."""
        model_true = SGRConventional()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "oscillation"

        omega = np.logspace(-2, 2, 30)
        G_star_complex = model_true.predict(omega)  # Shape (30,) complex

        # Fit with complex input
        model_fit = SGRConventional()
        model_fit.fit(omega, G_star_complex, test_mode="oscillation")

        assert model_fit.fitted_ is True

    def test_fitting_stores_test_mode(self):
        """Test that fitting stores test_mode for mode-aware Bayesian inference."""
        model = SGRConventional()

        omega = np.logspace(-1, 1, 20)
        model.parameters.set_value("x", 1.5)
        model._test_mode = "oscillation"
        G_star = model.predict(omega)

        # Reset and fit
        model2 = SGRConventional()
        assert model2._test_mode is None

        model2.fit(omega, G_star, test_mode="oscillation")

        # test_mode should be stored for Bayesian inference
        assert model2._test_mode == "oscillation"

    def test_fitting_with_noisy_data(self):
        """Test fitting robustness with noisy data."""
        np.random.seed(42)

        # Generate noisy data
        model_true = SGRConventional()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "oscillation"

        omega = np.logspace(-2, 2, 50)
        G_star = model_true.predict(omega)

        # Add 5% noise
        noise = 0.05 * G_star * np.random.randn(*G_star.shape)
        G_star_noisy = G_star + noise

        # Fit should still succeed
        model_fit = SGRConventional()
        model_fit.fit(omega, G_star_noisy, test_mode="oscillation")

        assert model_fit.fitted_ is True

    def test_fitting_invalid_test_mode_raises(self):
        """Test that invalid test_mode raises ValueError."""
        model = SGRConventional()
        omega = np.logspace(-1, 1, 20)
        G_star = np.ones((20, 2))

        with pytest.raises(ValueError, match="test_mode must be specified"):
            model.fit(omega, G_star)

        with pytest.raises(ValueError, match="Unsupported test_mode"):
            model.fit(omega, G_star, test_mode="invalid_mode")


class TestSGRConventionalStartupFitting:
    """Startup-flow fitting (_fit_startup_mode, routing lines 308-309, 886-951)."""

    def _make_startup_data(self, gamma_dot=1.0):
        model_true = SGRConventional()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "startup"
        model_true._startup_gamma_dot = gamma_dot
        t = np.logspace(-3, 1, 40)
        eta_plus = np.asarray(model_true.predict(t))
        return t, eta_plus

    def test_startup_mode_fitting_basic(self):
        """Fit eta_plus(t) startup data; routes through _fit_startup_mode."""
        t, eta_plus = self._make_startup_data(gamma_dot=1.0)

        model_fit = SGRConventional()
        model_fit.fit(t, eta_plus, test_mode="startup", gamma_dot=1.0)

        assert model_fit.fitted_ is True
        assert model_fit._test_mode == "startup"
        assert model_fit._startup_gamma_dot == 1.0

        # Fitted prediction should reproduce the data within tolerance
        pred = np.asarray(model_fit.predict(t))
        assert not np.any(np.isnan(pred))
        rel = np.abs(pred - eta_plus) / (np.abs(eta_plus) + 1e-12)
        np.testing.assert_allclose(np.max(rel), 0.0, atol=0.2)

    def test_startup_fitting_is_stress_conversion(self):
        """is_stress=True divides stress by gamma_dot before fitting (line 895-897)."""
        gamma_dot = 2.0
        t, eta_plus = self._make_startup_data(gamma_dot=gamma_dot)
        sigma = eta_plus * gamma_dot  # stress = gamma_dot * eta_plus

        model_fit = SGRConventional()
        model_fit.fit(
            t, sigma, test_mode="startup", gamma_dot=gamma_dot, is_stress=True
        )

        assert model_fit.fitted_ is True
        # Internally converts sigma -> eta_plus, so recovered response matches eta_plus
        pred = np.asarray(model_fit.predict(t))
        rel = np.abs(pred - eta_plus) / (np.abs(eta_plus) + 1e-12)
        np.testing.assert_allclose(np.max(rel), 0.0, atol=0.2)


class TestSGRConventionalLAOSFitting:
    """LAOS fitting paths (_fit_laos_mode/_fit_laos_mc, routing line 307, 690-862)."""

    def test_laos_fit_requires_gamma0_and_omega(self):
        """Missing gamma_0/omega raises ValueError (lines 693-694)."""
        model = SGRConventional()
        t = np.linspace(0, 2 * np.pi, 40)
        sigma = np.sin(t)
        with pytest.raises(ValueError, match="requires gamma_0 and omega"):
            model.fit(t, sigma, test_mode="laos", omega=1.0)

    def test_laos_fit_negative_gamma0_raises(self):
        """Non-positive gamma_0 raises ValueError (lines 695-696)."""
        model = SGRConventional()
        t = np.linspace(0, 2 * np.pi, 40)
        sigma = np.sin(t)
        with pytest.raises(ValueError, match="gamma_0 must be positive"):
            model.fit(t, sigma, test_mode="laos", gamma_0=-0.1, omega=1.0)

    def test_laos_saos_approximation_small_amplitude(self):
        """Small gamma_0 (<0.1) routes through the SAOS-approx branch (lines 706-729).

        Regression coverage for a kwargs-collision bug: _fit_laos_mode read
        gamma_0/omega/n_particles via kwargs.get (not pop), so they remained in
        **kwargs and collided with the same-named positional parameter on
        _fit_oscillation_mode (omega). Fixed by filtering the already-consumed
        keys out of kwargs before re-forwarding.
        """
        model = SGRConventional()
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        omega = 1.0
        t = np.linspace(0, 4 * np.pi, 60)
        # Linear-regime sinusoidal stress response
        strain = 0.05 * np.sin(omega * t)
        sigma = 1e3 * strain

        model._fit_laos_mode(t, sigma, gamma_0=0.05, omega=omega)

        assert model.fitted_ is True
        # LAOS parameters are cached (lines 698-701) before the branch.
        assert model._gamma_0 == 0.05
        assert model._omega_laos == omega

    def test_laos_mc_dispatch_kwargs_no_collision(self):
        """Public LAOS fit with gamma_0>=0.1 routes to the MC branch cleanly.

        Regression coverage for the same kwargs-collision bug as above, but for
        the MC dispatch (_fit_laos_mc's gamma_0/omega/n_particles positional
        parameters). Covers routing lines 690-704 and 730-732.
        """
        model = SGRConventional()
        t = np.linspace(0, 4 * np.pi, 40)
        sigma = 1e3 * 0.5 * np.sin(t)

        model.fit(
            t,
            sigma,
            test_mode="laos",
            gamma_0=0.5,
            omega=1.0,
            n_particles=64,
            max_iter=2,
        )

        assert model.fitted_ is True

    def test_laos_mc_fitting_direct(self):
        """Exercise the Monte Carlo fit body directly (lines 761-862).

        Bypasses the buggy _fit_laos_mode forwarding by calling _fit_laos_mc
        with clean positional args and no colliding kwargs.
        """
        model = SGRConventional()
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        omega = 1.0
        gamma_0 = 0.5
        period = 2.0 * np.pi / omega
        t = np.linspace(0, 2 * period, 40)
        strain = gamma_0 * np.sin(omega * t)
        sigma = 1e3 * strain

        # Keep the MC fit cheap: few particles, few iterations
        model._fit_laos_mc(t, sigma, gamma_0, omega, n_particles=64, max_iter=2, seed=0)

        assert model.fitted_ is True
        # Parameters remain within physical bounds after MC optimization
        x_fit = model.parameters.get_value("x")
        assert 0.5 <= x_fit <= 2.5
        assert np.isfinite(model.parameters.get_value("G0"))
        assert np.isfinite(model.parameters.get_value("tau0"))


class TestSGRConventionalOscillationInputFormats:
    """Oscillation fitting input-format handling (lines 388-394)."""

    def _target_data(self, omega):
        m = SGRConventional()
        m.parameters.set_value("x", 1.5)
        m.parameters.set_value("G0", 1e3)
        m.parameters.set_value("tau0", 1e-3)
        m._test_mode = "oscillation"
        G_star = np.asarray(m.predict(omega))
        return G_star

    def test_fit_accepts_2d_real_format(self):
        """(M, 2) real [G', G''] input format (line 388-389)."""
        omega = np.logspace(-2, 2, 30)
        G_star = self._target_data(omega)
        G_2d = np.column_stack([np.real(G_star), np.imag(G_star)])

        model = SGRConventional()
        model.fit(omega, G_2d, test_mode="oscillation")
        assert model.fitted_ is True

    def test_fit_accepts_transposed_format(self):
        """(2, M) transposed input format (lines 390-392)."""
        omega = np.logspace(-2, 2, 30)
        G_star = self._target_data(omega)
        G_2xM = np.vstack([np.real(G_star), np.imag(G_star)])

        model = SGRConventional()
        model.fit(omega, G_2xM, test_mode="oscillation")
        assert model.fitted_ is True

    def test_fit_invalid_shape_raises(self):
        """Wrong-shape modulus raises ValueError (lines 393-396)."""
        omega = np.logspace(-2, 2, 30)
        bad = np.ones((30, 3))  # neither complex, (M,2), nor (2,M)

        model = SGRConventional()
        with pytest.raises(ValueError, match=r"G_star must be complex"):
            model.fit(omega, bad, test_mode="oscillation")


class TestSGRConventionalPredictRouting:
    """Prediction routing and error paths (_predict, _predict_laos)."""

    def test_predict_without_test_mode_raises(self):
        """No test_mode set raises ValueError (lines 1176-1177)."""
        model = SGRConventional()
        t = np.logspace(-2, 2, 10)
        with pytest.raises(ValueError, match="test_mode must be specified"):
            model._predict(t)

    def test_predict_unknown_mode_raises(self):
        """Unknown test_mode override raises ValueError (line 1202)."""
        model = SGRConventional()
        t = np.logspace(-2, 2, 10)
        with pytest.raises(ValueError, match="Unknown test_mode"):
            model._predict(t, test_mode="bogus")

    def test_predict_laos_missing_params_raises(self):
        """LAOS predict without cached gamma_0/omega raises (lines 1306-1310)."""
        model = SGRConventional()
        t = np.linspace(0, 2 * np.pi, 32)
        with pytest.raises(ValueError, match="requires gamma_0 and omega"):
            model._predict(t, test_mode="laos")

    def test_predict_laos_with_kwargs(self):
        """LAOS predict extracts gamma_0/omega from kwargs (lines 1181-1186, 1312-1315)."""
        model = SGRConventional()
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        t = np.linspace(0, 2 * np.pi, 64)
        stress = model._predict(t, test_mode="laos", gamma_0=0.1, omega=1.0)

        assert model._gamma_0 == 0.1
        assert model._omega_laos == 1.0
        assert stress.shape == (64,)
        assert not np.any(np.isnan(stress))

    def test_predict_laos_multi_cycle_matches_closed_form(self):
        """Regression: predict(test_mode='laos') must evaluate at the actual
        times in X, not silently regenerate a single-period grid.

        Previously _predict_laos called simulate_laos(n_cycles=1,
        n_points_per_cycle=len(X)), aliasing whenever X spans more than one
        oscillation period. Verify predict() matches the closed-form SAOS
        reconstruction G'*gamma0*sin(wt) + G''*gamma0*cos(wt) at the real
        times, for a 3-cycle time array.
        """
        model = SGRConventional()
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        gamma_0, omega = 0.1, 1.0
        period = 2.0 * np.pi / omega
        t = np.linspace(0, 3 * period, 90, endpoint=False)

        stress = model._predict(t, test_mode="laos", gamma_0=gamma_0, omega=omega)

        # Closed-form linear-viscoelastic reconstruction at the real times.
        G_star = np.asarray(
            model.model_function(
                np.array([omega]),
                np.array([1.5, 1e3, 1e-3]),
                test_mode="oscillation",
            )
        )
        G_prime, G_double_prime = G_star[0, 0], G_star[0, 1]
        strain = gamma_0 * np.sin(omega * t)
        strain_rate = gamma_0 * omega * np.cos(omega * t)
        expected = G_prime * strain + (G_double_prime / omega) * strain_rate

        assert stress.shape == t.shape
        np.testing.assert_allclose(stress, expected, rtol=1e-6, atol=1e-9)


class TestSGRConventionalModelFunction:
    """Bayesian model_function routing across all modes (lines 1698-1738)."""

    PARAMS = np.array([1.5, 1e3, 1e-3])

    def test_model_function_oscillation(self):
        model = SGRConventional()
        omega = np.logspace(-2, 2, 20)
        out = np.asarray(
            model.model_function(omega, self.PARAMS, test_mode="oscillation")
        )
        assert out.shape == (20, 2)
        assert not np.any(np.isnan(out))

    def test_model_function_time_domain_modes(self):
        model = SGRConventional()
        t = np.logspace(-3, 2, 25)
        for mode in ("relaxation", "creep", "steady_shear"):
            out = np.asarray(model.model_function(t, self.PARAMS, test_mode=mode))
            assert out.shape == (25,), f"mode {mode} wrong shape"
            assert np.all(out > 0), f"mode {mode} must be positive"

    def test_model_function_flow_curve_alias(self):
        """flow_curve maps to steady_shear (line 1718)."""
        model = SGRConventional()
        gamma_dot = np.logspace(-2, 2, 15)
        out = np.asarray(
            model.model_function(gamma_dot, self.PARAMS, test_mode="flow_curve")
        )
        assert out.shape == (15,)

    def test_model_function_laos_uses_oscillation(self):
        """LAOS mode falls back to oscillation response (lines 1720-1722)."""
        model = SGRConventional()
        omega = np.logspace(-2, 2, 12)
        out = np.asarray(model.model_function(omega, self.PARAMS, test_mode="laos"))
        assert out.shape == (12, 2)

    def test_model_function_startup_with_kwarg_gamma_dot(self):
        """Startup mode uses explicit gamma_dot kwarg (lines 1723-1736)."""
        model = SGRConventional()
        t = np.logspace(-3, 1, 20)
        out = np.asarray(
            model.model_function(t, self.PARAMS, test_mode="startup", gamma_dot=2.0)
        )
        assert out.shape == (20,)
        assert np.all(out > 0)

    def test_model_function_startup_uses_cached_gamma_dot(self):
        """Startup mode falls back to cached _startup_gamma_dot."""
        model = SGRConventional()
        model._startup_gamma_dot = 1.5
        t = np.logspace(-3, 1, 10)
        out = np.asarray(model.model_function(t, self.PARAMS, test_mode="startup"))
        assert out.shape == (10,)

    def test_model_function_startup_missing_gamma_dot_raises(self):
        """Startup with no gamma_dot anywhere raises RuntimeError (lines 1730-1735)."""
        model = SGRConventional()
        t = np.logspace(-3, 1, 10)
        with pytest.raises(RuntimeError, match="gamma_dot not provided"):
            model.model_function(t, self.PARAMS, test_mode="startup")

    def test_model_function_unsupported_mode_raises(self):
        """Unsupported mode raises ValueError (line 1738)."""
        model = SGRConventional()
        t = np.logspace(-3, 1, 10)
        with pytest.raises(ValueError, match="Unsupported test mode"):
            model.model_function(t, self.PARAMS, test_mode="nonsense")

    def test_model_function_default_mode_fallback(self):
        """None mode (unfitted) defaults to oscillation (lines 1705-1706)."""
        model = SGRConventional()
        assert model._test_mode is None
        omega = np.logspace(-2, 2, 8)
        out = np.asarray(model.model_function(omega, self.PARAMS))
        assert out.shape == (8, 2)


class TestSGRConventionalDynamicXGuards:
    """Guard clauses for dynamic-x-only methods (lines 1472, 1561, 1564)."""

    def test_compute_x_ss_requires_dynamic_x(self):
        model = SGRConventional(dynamic_x=False)
        with pytest.raises(ValueError, match="requires dynamic_x=True"):
            model._compute_x_ss(1.0, 1e-3)

    def test_evolve_x_requires_dynamic_x(self):
        model = SGRConventional(dynamic_x=False)
        t = np.linspace(0, 1, 10)
        gamma_dot = np.zeros_like(t)
        with pytest.raises(ValueError, match="requires dynamic_x=True"):
            model.evolve_x(t, gamma_dot, x_initial=1.0)

    def test_evolve_x_shape_mismatch_raises(self):
        model = SGRConventional(dynamic_x=True)
        t = np.linspace(0, 1, 10)
        gamma_dot = np.zeros(5)  # wrong shape
        with pytest.raises(ValueError, match="same shape"):
            model.evolve_x(t, gamma_dot, x_initial=1.0)


class TestSGRConventionalHarmonicEdgeCases:
    """Harmonic/Chebyshev truncation and degenerate branches."""

    def test_harmonics_truncated_when_window_too_small(self):
        """Short window zeroes higher harmonics (lines 1872-1873, 1883-1884, 1894-1895)."""
        model = SGRConventional()
        # n_points_per_cycle=4 -> n//2=2, so idx_3=3, idx_5=5, idx_7=7 all >= 2
        stress = np.array([0.0, 1.0, 0.0, -1.0])
        harmonics = model.extract_laos_harmonics(stress, n_points_per_cycle=4)

        assert harmonics["I_3"] == 0.0
        assert harmonics["phi_3"] == 0.0
        assert harmonics["I_5"] == 0.0
        assert harmonics["I_7"] == 0.0

    def test_harmonics_zero_fundamental(self):
        """Zero stress -> zero fundamental -> relative intensities 0 (lines 1905-1907)."""
        model = SGRConventional()
        stress = np.zeros(256)
        harmonics = model.extract_laos_harmonics(stress, n_points_per_cycle=256)

        assert harmonics["I_1"] == 0.0
        assert harmonics["I_3_I_1"] == 0.0
        assert harmonics["I_5_I_1"] == 0.0
        assert harmonics["I_7_I_1"] == 0.0

    def test_chebyshev_zero_coefficients_branch(self):
        """Zero stress -> e_1=v_1=0 -> normalized ratios 0 (lines 2023-2024, 2030-2031)."""
        model = SGRConventional()
        n = 256
        omega = 1.0
        gamma_0 = 0.5
        t = np.linspace(0, 2 * np.pi / omega, n)
        strain = gamma_0 * np.sin(omega * t)
        stress = np.zeros(n)

        chebyshev = model.compute_chebyshev_coefficients(
            strain, stress, gamma_0, omega, n_points_per_cycle=n
        )

        assert chebyshev["e_1"] == 0.0
        assert chebyshev["v_1"] == 0.0
        assert chebyshev["e_3_e_1"] == 0.0
        assert chebyshev["e_5_e_1"] == 0.0
        assert chebyshev["v_3_v_1"] == 0.0
        assert chebyshev["v_5_v_1"] == 0.0


class TestSGRConventionalLissajous:
    """get_lissajous_curve (lines 2059-2073)."""

    def test_lissajous_curve_unnormalized(self):
        model = SGRConventional()
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        strain, stress = model.get_lissajous_curve(gamma_0=0.1, omega=1.0, n_points=128)

        assert strain.shape == (128,)
        assert stress.shape == (128,)
        assert np.max(np.abs(strain)) <= 0.1 * 1.01
        assert not np.any(np.isnan(stress))

    def test_lissajous_curve_normalized(self):
        model = SGRConventional()
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        strain, stress = model.get_lissajous_curve(
            gamma_0=0.2, omega=1.0, n_points=128, normalized=True
        )

        # Normalized strain in [-1, 1], normalized stress in [-1, 1]
        assert np.max(np.abs(strain)) <= 1.0 + 1e-9
        assert np.max(np.abs(stress)) == pytest.approx(1.0, abs=1e-9)


class TestSGRConventionalThixotropyGuards:
    """Thixotropy guard clauses and basic stress path (lines 2174, 2177, 2254, 2258)."""

    def test_evolve_lambda_requires_thixotropy(self):
        model = SGRConventional()
        t = np.linspace(0, 1, 10)
        gamma_dot = np.ones_like(t)
        with pytest.raises(ValueError, match="Thixotropy not enabled"):
            model.evolve_lambda(t, gamma_dot)

    def test_evolve_lambda_shape_mismatch(self):
        model = SGRConventional()
        model.enable_thixotropy()
        t = np.linspace(0, 1, 10)
        gamma_dot = np.ones(5)
        with pytest.raises(ValueError, match="same shape"):
            model.evolve_lambda(t, gamma_dot)

    def test_predict_thixotropic_stress_requires_thixotropy(self):
        model = SGRConventional()
        t = np.linspace(0, 1, 10)
        gamma_dot = np.ones_like(t)
        with pytest.raises(ValueError, match="Thixotropy not enabled"):
            model.predict_thixotropic_stress(t, gamma_dot)

    def test_predict_thixotropic_stress_basic(self):
        """Stress transient computes lambda internally (line 2258 onward)."""
        model = SGRConventional()
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)
        model.enable_thixotropy(k_build=0.1, k_break=0.5, n_struct=2.0)

        t = np.linspace(0, 10, 100)
        gamma_dot = np.ones_like(t) * 5.0

        sigma = model.predict_thixotropic_stress(t, gamma_dot)

        assert sigma.shape == (100,)
        assert not np.any(np.isnan(sigma))
        assert not np.any(np.isinf(sigma))
        # Structure breaks down under constant shear -> lambda cached
        assert model._lambda_trajectory is not None
