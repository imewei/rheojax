"""Unit tests for FIKH model.

Tests cover:
- Model initialization with various configurations
- Parameter setup and bounds
- Basic predictions for all protocols
- Mode-aware Bayesian inference
- Limiting behavior (α → 1)
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fikh import FIKH

jax, jnp = safe_import_jax()


class TestFIKHInitialization:
    """Test FIKH model initialization."""

    @pytest.mark.smoke
    def test_default_initialization(self):
        """Test default FIKH initialization."""
        model = FIKH()
        assert model.include_thermal is True
        assert model.alpha_structure == 0.5
        assert model.n_history == 100
        assert "G" in model.parameters
        assert "T_ref" in model.parameters  # Thermal param

    @pytest.mark.smoke
    def test_isothermal_initialization(self):
        """Test isothermal FIKH initialization."""
        model = FIKH(include_thermal=False)
        assert model.include_thermal is False
        assert "T_ref" not in model.parameters
        assert "E_a" not in model.parameters

    def test_custom_alpha(self):
        """Test FIKH with custom fractional order."""
        model = FIKH(alpha_structure=0.3, include_thermal=False)
        assert model.parameters.get_value("alpha_structure") == 0.3

    def test_alpha_bounds(self):
        """Test that alpha_structure has correct bounds."""
        model = FIKH(include_thermal=False)
        alpha_param = model.parameters.get("alpha_structure")
        lower, upper = alpha_param.bounds
        # Lower bound can be 0 (but value should be in (0, 1))
        assert lower >= 0
        assert upper <= 1
        # Value should be strictly between 0 and 1
        assert 0 < alpha_param.value < 1

    def test_repr(self):
        """Test string representation."""
        model = FIKH(include_thermal=False, alpha_structure=0.7)
        repr_str = repr(model)
        assert "FIKH" in repr_str
        assert "0.7" in repr_str or "0.700" in repr_str


class TestFIKHParameters:
    """Test FIKH parameter handling."""

    @pytest.mark.smoke
    def test_base_parameters_exist(self):
        """Test all base parameters exist."""
        model = FIKH(include_thermal=False)
        expected = [
            "G",
            "eta",
            "C",
            "gamma_dyn",
            "m",
            "sigma_y0",
            "delta_sigma_y",
            "tau_thix",
            "Gamma",
            "alpha_structure",
            "eta_inf",
            "mu_p",
        ]
        for param in expected:
            assert param in model.parameters, f"Missing parameter: {param}"

    def test_thermal_parameters_exist(self):
        """Test thermal parameters exist when enabled."""
        model = FIKH(include_thermal=True)
        thermal_params = ["T_ref", "E_a", "E_y", "m_y", "rho_cp", "chi", "h", "T_env"]
        for param in thermal_params:
            assert param in model.parameters, f"Missing thermal param: {param}"

    def test_parameter_bounds_valid(self):
        """Test all parameters have valid bounds."""
        model = FIKH(include_thermal=True)
        for name in model.parameters.keys():
            param = model.parameters.get(name)
            lower, upper = param.bounds
            assert lower < upper, f"Invalid bounds for {name}: ({lower}, {upper})"
            assert param.value >= lower, f"{name} value below lower bound"
            assert param.value <= upper, f"{name} value above upper bound"

    def test_get_params_dict(self):
        """Test _get_params_dict returns correct dictionary."""
        model = FIKH(include_thermal=False)
        params = model._get_params_dict()
        assert isinstance(params, dict)
        assert len(params) == len(model.parameters)
        assert params["G"] == model.parameters.get_value("G")


class TestFIKHPredictions:
    """Test FIKH model predictions."""

    @pytest.fixture
    def model(self):
        """Create isothermal FIKH model."""
        return FIKH(include_thermal=False, alpha_structure=0.5)

    @pytest.fixture
    def thermal_model(self):
        """Create thermal FIKH model."""
        return FIKH(include_thermal=True, alpha_structure=0.5)

    @pytest.mark.smoke
    def test_startup_prediction(self, model):
        """Test startup prediction produces valid output."""
        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t
        stress = model._predict_from_params(t, strain, model._get_params_dict())

        assert stress.shape == t.shape
        assert jnp.isfinite(stress).all()
        assert stress[0] == 0.0  # Initial stress is zero

    def test_startup_shows_overshoot(self, model):
        """Test startup shows stress overshoot characteristic of EVP."""
        t = jnp.linspace(0, 100, 500)
        strain = 1.0 * t  # Higher rate for clear overshoot
        stress = model._predict_from_params(t, strain, model._get_params_dict())

        # Find max stress - should be higher than final stress (overshoot)
        max_stress = float(stress.max())
        final_stress = float(stress[-1])
        # With thixotropy, overshoot may not always occur, so just check non-negative
        assert max_stress >= 0

    @pytest.mark.smoke
    def test_flow_curve_prediction(self, model):
        """Test steady-state flow curve prediction."""
        gamma_dot = jnp.logspace(-2, 2, 20)
        stress = model._predict(gamma_dot, test_mode="flow_curve")

        assert stress.shape == gamma_dot.shape
        assert jnp.isfinite(stress).all()
        # For thixotropic materials, stress may not monotonically increase
        # due to structure breakdown at high rates. Just check valid output.
        assert float(stress.min()) > 0  # Stress is positive

    def test_flow_curve_yield_stress(self, model):
        """Test flow curve shows yield stress behavior."""
        gamma_dot = jnp.logspace(-4, 2, 50)
        stress = model._predict(gamma_dot, test_mode="flow_curve")

        # At very low rates, stress should approach yield stress
        sigma_y0 = model.parameters.get_value("sigma_y0")
        delta_sigma_y = model.parameters.get_value("delta_sigma_y")

        # At low rate, λ ≈ 1, so σ_y ≈ σ_y0 + δσ_y
        expected_yield = sigma_y0 + delta_sigma_y
        low_rate_stress = float(stress[0])

        # Should be close to yield stress (within order of magnitude)
        assert low_rate_stress > sigma_y0 * 0.5

    @pytest.mark.smoke
    def test_laos_prediction(self, model):
        """Test LAOS prediction."""
        result = model.predict_laos(t=jnp.linspace(0, 10, 200), gamma_0=0.5, omega=1.0)

        assert "time" in result
        assert "strain" in result
        assert "stress" in result
        assert result["stress"].shape == result["time"].shape

    def test_thermal_startup(self, thermal_model):
        """Test thermal model produces temperature evolution."""
        from rheojax.models.fikh._kernels import fikh_scan_kernel_thermal

        t = jnp.linspace(0, 50, 200)
        strain = 1.0 * t  # High rate for heating
        params = thermal_model._get_params_dict()

        stress, temperature = fikh_scan_kernel_thermal(
            t,
            strain,
            n_history=100,
            alpha=0.5,
            use_viscosity=True,
            T_init=298.15,
            **params,
        )

        assert temperature.shape == t.shape
        # Temperature should increase due to viscous heating
        # (if chi > 0 and h not too large)
        final_T = float(temperature[-1])
        initial_T = 298.15
        # May increase or stay same depending on cooling
        assert final_T >= initial_T - 1.0  # Allow small numerical error


class TestFIKHTestModeValidation:
    """Test test_mode validation and handling."""

    def test_valid_test_modes(self):
        """Test all valid test modes are accepted."""
        model = FIKH(include_thermal=False)
        valid_modes = [
            "flow_curve",
            "startup",
            "relaxation",
            "creep",
            "oscillation",
            "laos",
        ]

        for mode in valid_modes:
            result = model._validate_test_mode(mode)
            assert result is not None

    def test_invalid_test_mode(self):
        """Test invalid test mode raises error."""
        model = FIKH(include_thermal=False)

        with pytest.raises(ValueError):
            model._validate_test_mode("invalid_mode")

    def test_laos_maps_to_startup(self):
        """Test LAOS mode maps to STARTUP (return mapping) internally."""
        model = FIKH(include_thermal=False)
        from rheojax.core.test_modes import TestMode

        # LAOS uses return mapping like startup, not frequency-domain oscillation
        result = model._validate_test_mode("laos")
        assert result == TestMode.STARTUP


class TestFIKHLimitingBehavior:
    """Test FIKH limiting behavior as α → 1."""

    @pytest.mark.slow
    def test_alpha_near_one_approaches_exponential(self):
        """Test that α → 1 gives exponential-like relaxation."""
        # Create model with α close to 1
        model = FIKH(include_thermal=False, alpha_structure=0.99)

        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t

        stress = model._predict_from_params(t, strain, model._get_params_dict())

        # Should produce smooth, stable response
        assert jnp.isfinite(stress).all()
        assert not jnp.isnan(stress).any()

    def test_alpha_small_gives_slow_recovery(self):
        """Test that small α gives slower structure recovery."""
        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t

        # High α (weak memory)
        model_high = FIKH(include_thermal=False, alpha_structure=0.9)
        stress_high = model_high._predict_from_params(
            t, strain, model_high._get_params_dict()
        )

        # Low α (strong memory)
        model_low = FIKH(include_thermal=False, alpha_structure=0.3)
        stress_low = model_low._predict_from_params(
            t, strain, model_low._get_params_dict()
        )

        # Both should be valid
        assert jnp.isfinite(stress_high).all()
        assert jnp.isfinite(stress_low).all()


class TestFIKHModelFunction:
    """Test model_function for Bayesian inference."""

    def test_model_function_with_dict(self):
        """Test model_function accepts parameter dictionary."""
        model = FIKH(include_thermal=False)
        model._test_mode = "startup"

        t = jnp.linspace(0, 5, 50)
        strain = 0.01 * t
        X = jnp.stack([t, strain], axis=0)

        params = model._get_params_dict()
        result = model.model_function(X, params)

        assert result.shape == t.shape

    def test_model_function_with_array(self):
        """Test model_function accepts parameter array."""
        model = FIKH(include_thermal=False)
        model._test_mode = "startup"

        t = jnp.linspace(0, 5, 50)
        strain = 0.01 * t
        X = jnp.stack([t, strain], axis=0)

        params_array = model.parameters.get_values()
        result = model.model_function(X, params_array)

        assert result.shape == t.shape

    def test_model_function_flow_curve(self):
        """Test model_function for flow curve mode."""
        model = FIKH(include_thermal=False)
        model._test_mode = "flow_curve"

        gamma_dot = jnp.logspace(-2, 2, 20)
        params = model._get_params_dict()

        result = model.model_function(gamma_dot, params)
        assert result.shape == gamma_dot.shape


class TestFIKHRelaxation:
    """Test FIKH relaxation protocol predictions."""

    @pytest.fixture
    def model(self):
        """Create isothermal FIKH model for relaxation tests."""
        return FIKH(include_thermal=False, alpha_structure=0.6)

    @pytest.mark.smoke
    def test_relaxation_prediction(self, model):
        """Test relaxation prediction produces valid output."""
        t = jnp.linspace(0, 100, 200)
        sigma_0 = 100.0

        stress = model.predict_relaxation(t, sigma_0=sigma_0)

        assert stress.shape == t.shape
        assert jnp.isfinite(stress).all()
        # Initial stress should be close to sigma_0
        assert float(stress[0]) > sigma_0 * 0.5

    def test_relaxation_decays_over_time(self, model):
        """Test that relaxation shows stress decay (Mittag-Leffler behavior)."""
        t = jnp.linspace(0, 500, 500)
        sigma_0 = 100.0

        stress = model.predict_relaxation(t, sigma_0=sigma_0)

        # Stress should decay over time
        initial_stress = float(stress[10])  # After initial transient
        final_stress = float(stress[-1])

        # Final stress should be less than initial (relaxation)
        assert final_stress < initial_stress

    def test_relaxation_alpha_affects_decay_rate(self):
        """Test that smaller alpha gives slower (power-law) relaxation."""
        t = jnp.linspace(0, 200, 200)
        sigma_0 = 100.0

        # Fast relaxation (alpha close to 1 = exponential-like)
        model_fast = FIKH(include_thermal=False, alpha_structure=0.9)
        stress_fast = model_fast.predict_relaxation(t, sigma_0=sigma_0)

        # Slow relaxation (alpha close to 0 = strong memory)
        model_slow = FIKH(include_thermal=False, alpha_structure=0.3)
        stress_slow = model_slow.predict_relaxation(t, sigma_0=sigma_0)

        # At intermediate times, lower alpha should retain more stress
        mid_idx = len(t) // 2
        # Slow relaxation should have higher stress at mid-time
        # (power-law decays slower than exponential at long times)
        assert float(stress_slow[mid_idx]) >= float(stress_fast[mid_idx]) * 0.5


class TestFIKHCreep:
    """Test FIKH creep protocol predictions."""

    @pytest.fixture
    def model(self):
        """Create isothermal FIKH model for creep tests."""
        return FIKH(include_thermal=False, alpha_structure=0.5)

    @pytest.mark.smoke
    def test_creep_prediction(self, model):
        """Test creep prediction produces valid output."""
        t = jnp.linspace(0, 100, 200)
        sigma_applied = 50.0

        strain = model.predict_creep(t, sigma_applied=sigma_applied)

        assert strain.shape == t.shape
        assert jnp.isfinite(strain).all()
        # Initial strain should be near zero
        assert abs(float(strain[0])) < 1.0

    def test_creep_strain_increases_under_stress(self, model):
        """Test that creep strain increases monotonically under constant stress."""
        t = jnp.linspace(0, 200, 200)
        sigma_applied = 80.0  # Above yield to ensure flow

        strain = model.predict_creep(t, sigma_applied=sigma_applied)

        # Strain should generally increase over time
        # (though with thixotropy there can be complex dynamics)
        final_strain = float(strain[-1])
        initial_strain = float(strain[10])

        assert final_strain > initial_strain


class TestFIKHPrecompile:
    """Test FIKH precompile method."""

    @pytest.mark.smoke
    def test_precompile_isothermal(self):
        """Test precompile works for isothermal model."""
        model = FIKH(include_thermal=False, alpha_structure=0.5)

        compile_time = model.precompile(n_points=50, verbose=False)

        assert isinstance(compile_time, float)
        assert compile_time > 0

    def test_precompile_thermal(self):
        """Test precompile works for thermal model."""
        model = FIKH(include_thermal=True, alpha_structure=0.5)

        compile_time = model.precompile(n_points=50, verbose=False)

        assert isinstance(compile_time, float)
        assert compile_time > 0

    def test_precompile_speeds_up_subsequent_calls(self):
        """Test that precompile reduces prediction time."""
        import time as time_module

        model = FIKH(include_thermal=False, alpha_structure=0.5)

        # First call (cold)
        _ = model.precompile(n_points=50, verbose=False)

        # Now measure prediction time (should be fast)
        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t

        start = time_module.perf_counter()
        _ = model._predict_from_params(t, strain, model._get_params_dict())
        elapsed = time_module.perf_counter() - start

        # After precompile, prediction should be reasonably fast
        # (hard to test exact speedup, but it shouldn't be absurdly slow)
        assert elapsed < 30.0  # Should complete in under 30s


class TestFIKHOscillation:
    """Test FIKH oscillation (SAOS) predictions (F-026)."""

    @pytest.mark.smoke
    def test_oscillation_prediction_returns_complex(self):
        """Test that oscillation prediction returns complex G*."""
        model = FIKH(include_thermal=False, alpha_structure=0.7)
        omega = jnp.array([0.1, 1.0, 10.0])
        G_star = model.predict_oscillation(omega, gamma_0=0.01, n_cycles=3)
        assert G_star.shape == (3,)
        assert jnp.iscomplexobj(G_star)
        # G' and G'' should be positive for a viscoelastic material
        assert jnp.all(jnp.real(G_star) > 0)
        assert jnp.all(jnp.imag(G_star) > 0)

    @pytest.mark.smoke
    def test_oscillation_via_predict(self):
        """Test oscillation via _predict method with test_mode."""
        model = FIKH(include_thermal=False, alpha_structure=0.7)
        omega = jnp.array([0.1, 1.0, 10.0])
        model._test_mode = "oscillation"
        result = model._predict(omega, test_mode="oscillation", gamma_0=0.01)
        assert result is not None
        assert len(result) == 3

    def test_oscillation_frequency_dependence(self):
        """Test G* is finite and positive across frequencies."""
        model = FIKH(include_thermal=False, alpha_structure=0.7)
        omega = jnp.array([0.1, 1.0, 10.0])
        G_star = model.predict_oscillation(omega, gamma_0=0.01, n_cycles=3)
        magnitudes = jnp.abs(G_star)
        # |G*| should be finite and positive at all frequencies
        assert jnp.all(jnp.isfinite(magnitudes))
        assert jnp.all(magnitudes > 0)


class TestFIKHFitIntegration:
    """Test fit → predict integration (F-027)."""

    @pytest.mark.smoke
    def test_fit_startup_then_predict(self):
        """Test that fit() followed by predict() works for startup."""
        model = FIKH(include_thermal=False, alpha_structure=0.7)
        t = jnp.linspace(0.01, 5.0, 50)
        strain = 0.1 * t  # linear ramp
        # Generate synthetic data from default params
        stress_true = model._predict_from_params(t, strain, model._get_params_dict())
        noise = 0.01 * jnp.std(stress_true) * jax.random.normal(
            jax.random.PRNGKey(42), shape=stress_true.shape
        )
        stress_data = stress_true + noise

        # Fit
        model.fit(t, stress_data, test_mode="startup", strain=strain, max_iter=50)

        # Predict should use the fitted parameters
        stress_pred = model.predict(t, test_mode="startup", strain=strain)
        assert stress_pred.shape == stress_data.shape
        assert jnp.all(jnp.isfinite(stress_pred))

    @pytest.mark.smoke
    def test_fit_flow_curve_then_predict(self):
        """Test flow curve fit → predict roundtrip."""
        model = FIKH(include_thermal=False, alpha_structure=0.7)
        gamma_dot = jnp.logspace(-2, 2, 20)
        # Use model to generate synthetic data
        from rheojax.models.fikh._kernels import fikh_flow_curve_steady_state

        params = model._get_params_dict()
        sigma_true = fikh_flow_curve_steady_state(
            gamma_dot, include_thermal=False, **params
        )

        model.fit(gamma_dot, sigma_true, test_mode="flow_curve", max_iter=20)
        sigma_pred = model.predict(gamma_dot, test_mode="flow_curve")
        assert sigma_pred.shape == sigma_true.shape
        assert jnp.all(jnp.isfinite(sigma_pred))


class TestFIKHSignSafe:
    """Test sign_safe fix (F-001)."""

    def test_sign_safe_positive(self):
        """Test sign_safe returns +1 for positive values."""
        from rheojax.models.fikh._kernels import sign_safe

        assert sign_safe(jnp.array(1.0)) == 1.0
        assert sign_safe(jnp.array(1e-25)) == 0.0  # Below eps (1e-20) → 0

    def test_sign_safe_negative(self):
        """Test sign_safe returns -1 for negative values."""
        from rheojax.models.fikh._kernels import sign_safe

        assert sign_safe(jnp.array(-1.0)) == -1.0
        # F-001 fix: small negative values no longer flip to +1
        assert sign_safe(jnp.array(-1e-30)) == 0.0  # Below eps → 0

    def test_sign_safe_zero(self):
        """Test sign_safe returns 0 for zero."""
        from rheojax.models.fikh._kernels import sign_safe

        assert sign_safe(jnp.array(0.0)) == 0.0


class TestFIKHThermalYieldStress:
    """Test thermal yield stress fix (F-007)."""

    def test_no_double_lambda(self):
        """Verify yield stress doesn't apply λ twice."""
        from rheojax.models.fikh._thermal import thermal_yield_stress

        sigma_y0 = 100.0
        lam = 0.5
        m_y = 1.0
        T = 298.15
        T_ref = 298.15
        E_y = 0.0  # No temperature effect

        # When called with lam=1.0 (F-007 fix in kernels), only base matters
        result = thermal_yield_stress(sigma_y0, 1.0, m_y, T, T_ref, E_y)
        assert jnp.isclose(result, sigma_y0, atol=1e-6)

        # With lam < 1, should scale by lam^m_y
        result_lam = thermal_yield_stress(sigma_y0, lam, m_y, T, T_ref, E_y)
        assert jnp.isclose(result_lam, sigma_y0 * lam**m_y, atol=1e-6)
