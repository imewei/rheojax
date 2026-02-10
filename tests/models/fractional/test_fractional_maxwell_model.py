"""Tests for Fractional Maxwell Model (FMM).

Most general fractional Maxwell model with two independent fractional orders.
Tests cover initialization, all test modes, limit cases, JAX operations,
numerical stability, and two-parameter fractional behavior.
"""

import numpy as np
import pytest

import rheojax.models  # Import to trigger all model registrations
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.models import FractionalMaxwellModel

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class TestFractionalMaxwellModelInitialization:
    """Test model initialization."""

    def test_model_creation(self):
        model = FractionalMaxwellModel()
        assert model is not None

    def test_parameters_exist(self):
        model = FractionalMaxwellModel()
        assert "c1" in model.parameters
        assert "alpha" in model.parameters
        assert "beta" in model.parameters
        assert "tau" in model.parameters

    def test_parameter_defaults(self):
        model = FractionalMaxwellModel()
        assert model.parameters.get_value("c1") == 1e5
        assert model.parameters.get_value("alpha") == 0.5
        assert model.parameters.get_value("beta") == 0.5
        assert model.parameters.get_value("tau") == 1.0

    def test_parameter_bounds(self):
        model = FractionalMaxwellModel()
        assert model.parameters.get("c1").bounds == (1e-3, 1e9)
        assert model.parameters.get("alpha").bounds == (0.0, 1.0)
        assert model.parameters.get("beta").bounds == (0.0, 1.0)
        assert model.parameters.get("tau").bounds == (1e-6, 1e6)

    def test_registry_registration(self):
        assert "fractional_maxwell_model" in ModelRegistry.list_models()


class TestFractionalMaxwellModelRelaxation:
    """Test relaxation modulus."""

    def test_relaxation_basic(self):
        model = FractionalMaxwellModel()
        t = np.array([0.01, 0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_relaxation_monotonic_decrease(self):
        model = FractionalMaxwellModel()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        diffs = np.diff(result.y)
        assert np.sum(diffs < 0) > 0.75 * len(diffs)

    def test_relaxation_short_time_power_law(self):
        """G(t) ~ c1 * t^(-α) at short times."""
        model = FractionalMaxwellModel()
        model.parameters.set_value("c1", 1e5)
        model.parameters.set_value("alpha", 0.5)
        model.parameters.set_value("beta", 0.7)
        model.parameters.set_value("tau", 10.0)

        t = np.logspace(-3, -1, 20)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"
        result = model.predict(data)

        log_t = np.log10(t)
        log_G = np.log10(result.y)
        slope = np.polyfit(log_t, log_G, 1)[0]
        assert np.abs(slope - (-0.5)) < 0.3

    def test_relaxation_two_parameter_effect(self):
        """Test effect of having two independent fractional orders."""
        model = FractionalMaxwellModel()
        t = np.logspace(-2, 2, 50)

        # Case 1: alpha=beta
        model.parameters.set_value("alpha", 0.5)
        model.parameters.set_value("beta", 0.5)
        data1 = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data1.metadata["test_mode"] = "relaxation"
        result1 = model.predict(data1)

        # Case 2: alpha≠beta
        model.parameters.set_value("alpha", 0.3)
        model.parameters.set_value("beta", 0.7)
        data2 = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data2.metadata["test_mode"] = "relaxation"
        result2 = model.predict(data2)

        # Should give different results
        assert not np.allclose(result1.y, result2.y, rtol=0.1)


class TestFractionalMaxwellModelCreep:
    """Test creep compliance."""

    def test_creep_basic(self):
        model = FractionalMaxwellModel()
        t = np.array([0.01, 0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "creep"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_creep_monotonic_increase(self):
        model = FractionalMaxwellModel()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "creep"

        result = model.predict(data)
        diffs = np.diff(result.y)
        assert np.all(diffs > -1e-10)

    def test_creep_power_law_behavior(self):
        """J(t) ~ t^α at short times."""
        model = FractionalMaxwellModel()
        model.parameters.set_value("c1", 1e5)
        model.parameters.set_value("alpha", 0.5)

        t = np.logspace(-3, -1, 20)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "creep"
        result = model.predict(data)

        log_t = np.log10(t)
        log_J = np.log10(result.y)
        slope = np.polyfit(log_t, log_J, 1)[0]
        assert 0.3 < slope < 0.7


class TestFractionalMaxwellModelOscillation:
    """Test complex modulus."""

    def test_oscillation_basic(self):
        model = FractionalMaxwellModel()
        omega = np.array([0.1, 1.0, 10.0, 100.0])
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data.metadata["test_mode"] = "oscillation"

        result = model.predict(data)
        assert np.iscomplexobj(result.y)
        assert np.all(np.isfinite(result.y))

    def test_oscillation_moduli_positive(self):
        model = FractionalMaxwellModel()
        omega = np.logspace(-2, 2, 50)
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data.metadata["test_mode"] = "oscillation"

        result = model.predict(data)
        assert np.all(np.real(result.y) > 0)
        assert np.all(np.imag(result.y) > 0)

    def test_oscillation_low_frequency_scaling(self):
        """Test power-law scaling ~ ω^α at low frequency."""
        model = FractionalMaxwellModel()
        model.parameters.set_value("alpha", 0.5)
        model.parameters.set_value("beta", 0.7)

        omega = np.logspace(-3, -1, 20)
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data.metadata["test_mode"] = "oscillation"

        result = model.predict(data)
        log_omega = np.log10(omega)
        log_G_abs = np.log10(np.abs(result.y))

        slope = np.polyfit(log_omega, log_G_abs, 1)[0]
        assert 0.3 < slope < 0.7

    def test_oscillation_two_parameter_independence(self):
        """Test that alpha and beta have independent effects."""
        model = FractionalMaxwellModel()
        omega = np.logspace(-2, 2, 50)

        # Vary alpha, fix beta
        model.parameters.set_value("alpha", 0.3)
        model.parameters.set_value("beta", 0.5)
        data1 = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data1.metadata["test_mode"] = "oscillation"
        result1 = model.predict(data1)

        model.parameters.set_value("alpha", 0.7)
        model.parameters.set_value("beta", 0.5)
        data2 = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data2.metadata["test_mode"] = "oscillation"
        result2 = model.predict(data2)

        # Should be different
        assert not np.allclose(np.abs(result1.y), np.abs(result2.y), rtol=0.1)


class TestFractionalMaxwellModelLimitCases:
    """Test limit cases."""

    def test_alpha_beta_equal(self):
        """Test when alpha = beta (reduced symmetry)."""
        model = FractionalMaxwellModel()
        model.parameters.set_value("alpha", 0.5)
        model.parameters.set_value("beta", 0.5)

        t = np.logspace(-2, 2, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_alpha_near_zero_beta_near_one(self):
        """Test extreme parameter combinations."""
        model = FractionalMaxwellModel()
        model.parameters.set_value("alpha", 0.05)
        model.parameters.set_value("beta", 0.95)

        t = np.logspace(-2, 2, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_both_alpha_beta_near_one(self):
        """Test when both approach 1 (should approach Maxwell)."""
        model = FractionalMaxwellModel()
        model.parameters.set_value("alpha", 0.95)
        model.parameters.set_value("beta", 0.95)

        t = np.logspace(-2, 2, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_zero_time_handling(self):
        model = FractionalMaxwellModel()
        t = np.array([0.0, 0.01, 0.1, 1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))


class TestFractionalMaxwellModelJAX:
    """Test JAX functionality."""

    def test_jit_compilation(self):
        model = FractionalMaxwellModel()
        t = jnp.array([0.1, 1.0, 10.0])
        result = model._predict_relaxation_jax(t, 1e5, 0.5, 0.5, 1.0)
        assert result.shape == t.shape

    def test_gradient_computation(self):
        model = FractionalMaxwellModel()

        def loss_fn(c1):
            t = jnp.array([1.0])
            result = model._predict_relaxation_jax(t, c1, 0.5, 0.5, 1.0)
            return jnp.sum(result**2)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(1e5)
        assert np.isfinite(gradient)

    @pytest.mark.xfail(
        reason="vmap over alpha not supported - alpha must be concrete for Mittag-Leffler"
    )
    def test_vmap_over_alpha_beta(self):
        """Test vectorization over two parameters simultaneously."""
        model = FractionalMaxwellModel()
        t = jnp.array([1.0])
        alphas = jnp.array([0.3, 0.5, 0.7])
        betas = jnp.array([0.4, 0.6, 0.8])

        def predict_for_params(alpha, beta):
            return model._predict_relaxation_jax(t, 1e5, alpha, beta, 1.0)[0]

        vmapped = jax.vmap(predict_for_params)
        results = vmapped(alphas, betas)
        assert results.shape == (3,)
        assert np.all(np.isfinite(results))

    @pytest.mark.xfail(
        reason="Mittag-Leffler function derivatives w.r.t. tau produce NaN - numerical issue to be investigated"
    )
    def test_hessian_computation(self):
        """Test second derivatives for two parameters."""
        model = FractionalMaxwellModel()

        def loss_fn(params):
            c1, tau = params
            t = jnp.array([1.0])
            result = model._predict_relaxation_jax(t, c1, 0.5, 0.5, tau)
            return jnp.sum(result**2)

        hessian_fn = jax.hessian(loss_fn)
        params = jnp.array([1e5, 1.0])
        hessian = hessian_fn(params)
        assert hessian.shape == (2, 2)
        assert np.all(np.isfinite(hessian))


class TestFractionalMaxwellModelNumericalStability:
    """Test numerical stability."""

    def test_all_modes_stable(self):
        model = FractionalMaxwellModel()
        t = np.logspace(-2, 2, 30)

        for mode in ["relaxation", "creep"]:
            data = RheoData(x=t, y=np.zeros_like(t), domain="time")
            data.metadata["test_mode"] = mode
            result = model.predict(data)
            assert np.all(np.isfinite(result.y))

    def test_extreme_parameter_combinations(self):
        """Test stability with extreme parameter values."""
        model = FractionalMaxwellModel()

        test_cases = [
            {"c1": 1e-2, "alpha": 0.1, "beta": 0.1, "tau": 1e-4},
            {"c1": 1e8, "alpha": 0.9, "beta": 0.9, "tau": 1e5},
            {"c1": 1e5, "alpha": 0.1, "beta": 0.9, "tau": 1.0},
        ]

        t = np.logspace(-2, 2, 20)

        for params in test_cases:
            for key, value in params.items():
                model.parameters.set_value(key, value)

            data = RheoData(x=t, y=np.zeros_like(t), domain="time")
            data.metadata["test_mode"] = "relaxation"
            result = model.predict(data)
            assert np.all(np.isfinite(result.y))

    def test_parameter_continuity(self):
        """Test that small parameter changes give small result changes."""
        model = FractionalMaxwellModel()
        t = np.array([1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        model.parameters.set_value("alpha", 0.5)
        result1 = model.predict(data)

        model.parameters.set_value("alpha", 0.51)
        result2 = model.predict(data)

        relative_change = np.abs(result2.y[0] - result1.y[0]) / result1.y[0]
        assert relative_change < 0.1


class TestFractionalMaxwellModelRheoDataIntegration:
    """Test RheoData integration."""

    def test_rheodata_input_output(self):
        model = FractionalMaxwellModel()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert isinstance(result, RheoData)

    def test_metadata_preservation(self):
        model = FractionalMaxwellModel()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"
        data.metadata["param_set"] = "A"

        result = model.predict(data)
        assert result.metadata["param_set"] == "A"

    def test_auto_detect_frequency_domain(self):
        model = FractionalMaxwellModel()
        omega = np.logspace(-2, 2, 50)
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )

        result = model.predict(data)
        assert np.iscomplexobj(result.y)


class TestFractionalMaxwellModelErrorHandling:
    """Test error handling."""

    def test_invalid_test_mode(self):
        model = FractionalMaxwellModel()
        t = np.array([0.1, 1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "invalid"

        with pytest.raises(ValueError, match="Unknown test mode"):
            model.predict(data)

    def test_parameter_bounds_validation(self):
        model = FractionalMaxwellModel()

        with pytest.raises(ValueError):
            model.parameters.set_value("alpha", 1.5)

        with pytest.raises(ValueError):
            model.parameters.set_value("beta", -0.1)
