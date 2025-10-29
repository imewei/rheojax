"""Tests for Fractional Maxwell Liquid (FML) model.

Comprehensive test suite covering initialization, all test modes, limit cases,
JAX operations, numerical stability, and RheoData integration.
"""

import jax
import numpy as np
import pytest

import rheo.models  # Import to trigger all model registrations
from rheo.core.data import RheoData
from rheo.core.registry import ModelRegistry
from rheo.models.fractional_maxwell_liquid import FractionalMaxwellLiquid



from rheo.core.jax_config import safe_import_jax

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()
class TestFractionalMaxwellLiquidInitialization:
    """Test model initialization and parameters."""

    def test_model_creation(self):
        model = FractionalMaxwellLiquid()
        assert model is not None

    def test_parameters_exist(self):
        model = FractionalMaxwellLiquid()
        assert "Gm" in model.parameters
        assert "alpha" in model.parameters
        assert "tau_alpha" in model.parameters

    def test_parameter_defaults(self):
        model = FractionalMaxwellLiquid()
        assert model.parameters.get_value("Gm") == 1e6
        assert model.parameters.get_value("alpha") == 0.5
        assert model.parameters.get_value("tau_alpha") == 1.0

    def test_parameter_bounds(self):
        model = FractionalMaxwellLiquid()
        assert model.parameters.get("Gm").bounds == (1e-3, 1e9)
        assert model.parameters.get("alpha").bounds == (0.0, 1.0)
        assert model.parameters.get("tau_alpha").bounds == (1e-6, 1e6)

    def test_registry_registration(self):
        assert "fractional_maxwell_liquid" in ModelRegistry.list_models()

    def test_factory_creation(self):
        model = ModelRegistry.create("fractional_maxwell_liquid")
        assert isinstance(model, FractionalMaxwellLiquid)


class TestFractionalMaxwellLiquidRelaxation:
    """Test relaxation modulus predictions."""

    def test_relaxation_basic(self):
        model = FractionalMaxwellLiquid()
        t = np.array([0.01, 0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert isinstance(result, RheoData)
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_relaxation_monotonic_decrease(self):
        model = FractionalMaxwellLiquid()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        diffs = np.diff(result.y)
        assert np.sum(diffs < 0) > 0.8 * len(diffs)

    def test_relaxation_short_time_power_law(self):
        """G(t) ~ Gm * t^(-α) at short times."""
        model = FractionalMaxwellLiquid()
        model.parameters.set_value("Gm", 1e6)
        model.parameters.set_value("alpha", 0.5)
        model.parameters.set_value("tau_alpha", 10.0)

        t = np.logspace(-3, -1, 20)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"
        result = model.predict(data)

        log_t = np.log10(t)
        log_G = np.log10(result.y)
        slope = np.polyfit(log_t, log_G, 1)[0]
        assert np.abs(slope - (-0.5)) < 0.3

    def test_relaxation_alpha_dependence(self):
        model = FractionalMaxwellLiquid()
        t = np.logspace(-2, 2, 50)

        model.parameters.set_value("alpha", 0.3)
        data_low = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data_low.metadata["test_mode"] = "relaxation"
        result_low = model.predict(data_low)

        model.parameters.set_value("alpha", 0.7)
        data_high = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data_high.metadata["test_mode"] = "relaxation"
        result_high = model.predict(data_high)

        # Different alpha should give different results
        assert not np.allclose(result_low.y, result_high.y)


class TestFractionalMaxwellLiquidCreep:
    """Test creep compliance predictions."""

    def test_creep_basic(self):
        model = FractionalMaxwellLiquid()
        t = np.array([0.01, 0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "creep"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_creep_monotonic_increase(self):
        model = FractionalMaxwellLiquid()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "creep"

        result = model.predict(data)
        diffs = np.diff(result.y)
        assert np.all(diffs > -1e-10)

    def test_creep_instant_compliance(self):
        """Test that J(t) starts at 1/Gm."""
        model = FractionalMaxwellLiquid()
        Gm = 1e6
        model.parameters.set_value("Gm", Gm)

        t = np.array([1e-6])  # Very small time
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "creep"

        result = model.predict(data)
        # Should be close to 1/Gm
        assert np.abs(result.y[0] - 1.0 / Gm) / (1.0 / Gm) < 0.5


class TestFractionalMaxwellLiquidOscillation:
    """Test complex modulus predictions."""

    def test_oscillation_basic(self):
        model = FractionalMaxwellLiquid()
        omega = np.array([0.1, 1.0, 10.0, 100.0])
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data.metadata["test_mode"] = "oscillation"

        result = model.predict(data)
        assert np.iscomplexobj(result.y)
        assert np.all(np.isfinite(result.y))

    def test_oscillation_moduli_positive(self):
        model = FractionalMaxwellLiquid()
        omega = np.logspace(-2, 2, 50)
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data.metadata["test_mode"] = "oscillation"

        result = model.predict(data)
        assert np.all(np.real(result.y) > 0)
        assert np.all(np.imag(result.y) > 0)

    def test_oscillation_frequency_scaling(self):
        """Test power-law scaling at low frequency."""
        model = FractionalMaxwellLiquid()
        model.parameters.set_value("alpha", 0.5)

        omega = np.logspace(-3, -1, 20)
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data.metadata["test_mode"] = "oscillation"

        result = model.predict(data)
        log_omega = np.log10(omega)
        log_G_abs = np.log10(np.abs(result.y))

        slope = np.polyfit(log_omega, log_G_abs, 1)[0]
        assert 0.3 < slope < 0.7  # Should be close to alpha=0.5

    def test_oscillation_high_frequency_plateau(self):
        """Test that G* approaches Gm at high frequency."""
        model = FractionalMaxwellLiquid()
        Gm = 1e6
        model.parameters.set_value("Gm", Gm)

        omega = np.array([1e6])  # Very high frequency
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data.metadata["test_mode"] = "oscillation"

        result = model.predict(data)
        # Should approach Gm
        assert np.abs(result.y[0].real - Gm) / Gm < 0.5


class TestFractionalMaxwellLiquidLimitCases:
    """Test limit cases."""

    def test_alpha_near_zero(self):
        model = FractionalMaxwellLiquid()
        model.parameters.set_value("alpha", 0.05)

        t = np.logspace(-2, 2, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_alpha_near_one(self):
        """As alpha→1, should approach Maxwell model."""
        model = FractionalMaxwellLiquid()
        model.parameters.set_value("alpha", 0.95)

        t = np.logspace(-2, 2, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_zero_time_handling(self):
        model = FractionalMaxwellLiquid()
        t = np.array([0.0, 0.01, 0.1, 1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))


class TestFractionalMaxwellLiquidJAX:
    """Test JAX functionality."""

    def test_jit_compilation(self):
        model = FractionalMaxwellLiquid()
        t = jnp.array([0.1, 1.0, 10.0])
        result = model._predict_relaxation_jax(t, 1e6, 0.5, 1.0)
        assert result.shape == t.shape

    def test_gradient_computation(self):
        model = FractionalMaxwellLiquid()

        def loss_fn(Gm):
            t = jnp.array([1.0])
            result = model._predict_relaxation_jax(t, Gm, 0.5, 1.0)
            return jnp.sum(result**2)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(1e6)
        assert np.isfinite(gradient)

    def test_vmap_over_frequencies(self):
        model = FractionalMaxwellLiquid()
        omega = jnp.logspace(-2, 2, 20)

        def predict_single(w):
            return model._predict_oscillation_jax(jnp.array([w]), 1e6, 0.5, 1.0)[0]

        vmapped = jax.vmap(predict_single)
        results = vmapped(omega)
        assert results.shape == omega.shape


class TestFractionalMaxwellLiquidNumericalStability:
    """Test numerical stability."""

    def test_all_modes_finite(self):
        model = FractionalMaxwellLiquid()
        t = np.logspace(-2, 2, 30)

        for mode in ["relaxation", "creep"]:
            data = RheoData(x=t, y=np.zeros_like(t), domain="time")
            data.metadata["test_mode"] = mode
            result = model.predict(data)
            assert np.all(np.isfinite(result.y))

        omega = np.logspace(-2, 2, 30)
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        data.metadata["test_mode"] = "oscillation"
        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_parameter_sensitivity(self):
        """Test small parameter changes give small result changes."""
        model = FractionalMaxwellLiquid()
        t = np.array([1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        model.parameters.set_value("Gm", 1e6)
        result1 = model.predict(data)

        model.parameters.set_value("Gm", 1e6 * 1.01)  # 1% change
        result2 = model.predict(data)

        relative_change = np.abs(result2.y[0] - result1.y[0]) / result1.y[0]
        assert relative_change < 0.1  # Less than 10% change


class TestFractionalMaxwellLiquidRheoDataIntegration:
    """Test RheoData integration."""

    def test_rheodata_input_output(self):
        model = FractionalMaxwellLiquid()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"

        result = model.predict(data)
        assert isinstance(result, RheoData)
        assert result.x.shape == data.x.shape

    def test_metadata_preservation(self):
        model = FractionalMaxwellLiquid()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "relaxation"
        data.metadata["sample_id"] = "test123"

        result = model.predict(data)
        assert result.metadata["sample_id"] == "test123"

    def test_auto_detect_test_mode(self):
        model = FractionalMaxwellLiquid()
        omega = np.logspace(-2, 2, 50)
        data = RheoData(
            x=omega, y=np.zeros_like(omega, dtype=complex), domain="frequency"
        )
        result = model.predict(data)
        assert np.iscomplexobj(result.y)


class TestFractionalMaxwellLiquidErrorHandling:
    """Test error handling."""

    def test_invalid_test_mode(self):
        model = FractionalMaxwellLiquid()
        t = np.array([0.1, 1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain="time")
        data.metadata["test_mode"] = "invalid"

        with pytest.raises(ValueError, match="Unknown test mode"):
            model.predict(data)

    def test_parameter_bounds_validation(self):
        model = FractionalMaxwellLiquid()

        with pytest.raises(ValueError):
            model.parameters.set_value("alpha", 1.5)

        with pytest.raises(ValueError):
            model.parameters.set_value("alpha", -0.1)
