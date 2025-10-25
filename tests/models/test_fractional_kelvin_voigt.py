"""Tests for Fractional Kelvin-Voigt (FKV) model.

Tests spring and SpringPot in parallel configuration. This model describes
solid-like materials with power-law creep behavior.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import rheo.models  # Import to trigger all model registrations
from rheo.models.fractional_kelvin_voigt import FractionalKelvinVoigt
from rheo.core.data import RheoData
from rheo.core.registry import ModelRegistry


class TestFractionalKelvinVoigtInitialization:
    """Test model initialization."""

    def test_model_creation(self):
        model = FractionalKelvinVoigt()
        assert model is not None

    def test_parameters_exist(self):
        model = FractionalKelvinVoigt()
        assert 'Ge' in model.parameters
        assert 'c_alpha' in model.parameters
        assert 'alpha' in model.parameters

    def test_parameter_defaults(self):
        model = FractionalKelvinVoigt()
        assert model.parameters.get_value('Ge') == 1e6
        assert model.parameters.get_value('c_alpha') == 1e4
        assert model.parameters.get_value('alpha') == 0.5

    def test_parameter_bounds(self):
        model = FractionalKelvinVoigt()
        assert model.parameters.get('Ge').bounds == (1e-3, 1e9)
        assert model.parameters.get('c_alpha').bounds == (1e-3, 1e9)
        assert model.parameters.get('alpha').bounds == (0.0, 1.0)

    def test_registry_registration(self):
        assert 'fractional_kelvin_voigt' in ModelRegistry.list_models()


class TestFractionalKelvinVoigtRelaxation:
    """Test relaxation modulus."""

    def test_relaxation_basic(self):
        model = FractionalKelvinVoigt()
        t = np.array([0.01, 0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_relaxation_has_elastic_floor(self):
        """Test that G(t) ≥ Ge (elastic modulus is floor)."""
        model = FractionalKelvinVoigt()
        Ge = 1e6
        model.parameters.set_value('Ge', Ge)

        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        # All values should be >= Ge
        assert np.all(result.y >= Ge * 0.95)  # Allow 5% numerical tolerance

    def test_relaxation_power_law_plus_constant(self):
        """Test G(t) = Ge + c_α t^(-α) / Γ(1-α)."""
        model = FractionalKelvinVoigt()
        model.parameters.set_value('Ge', 1e6)
        model.parameters.set_value('c_alpha', 1e4)
        model.parameters.set_value('alpha', 0.5)

        t = np.logspace(-3, -1, 20)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)

        # At short times, should see power-law decay from high values
        assert result.y[0] > result.y[-1]

    def test_relaxation_long_time_plateau(self):
        """Test that G(t) → Ge at long times."""
        model = FractionalKelvinVoigt()
        Ge = 1e6
        model.parameters.set_value('Ge', Ge)
        model.parameters.set_value('c_alpha', 1e4)
        model.parameters.set_value('alpha', 0.5)

        t = np.array([1e6])  # Very long time
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        # Should approach Ge
        assert np.abs(result.y[0] - Ge) / Ge < 0.1


class TestFractionalKelvinVoigtCreep:
    """Test creep compliance."""

    def test_creep_basic(self):
        model = FractionalKelvinVoigt()
        t = np.array([0.01, 0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'creep'

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_creep_bounded(self):
        """Test that J(t) is bounded (doesn't flow to infinity)."""
        model = FractionalKelvinVoigt()
        Ge = 1e6
        model.parameters.set_value('Ge', Ge)

        t = np.logspace(-2, 4, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'creep'

        result = model.predict(data)
        # Should be bounded by 1/Ge
        assert np.all(result.y <= 1.0/Ge * 1.1)  # Allow 10% numerical tolerance

    def test_creep_monotonic_increase(self):
        """Test that J(t) increases monotonically."""
        model = FractionalKelvinVoigt()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'creep'

        result = model.predict(data)
        diffs = np.diff(result.y)
        assert np.all(diffs > -1e-12)  # Monotonic with numerical tolerance

    def test_creep_plateau_at_long_time(self):
        """Test J(t) → 1/Ge at long times."""
        model = FractionalKelvinVoigt()
        Ge = 1e6
        model.parameters.set_value('Ge', Ge)

        t = np.array([1e6])  # Very long time
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'creep'

        result = model.predict(data)
        # Should approach 1/Ge
        assert np.abs(result.y[0] - 1.0/Ge) / (1.0/Ge) < 0.1


class TestFractionalKelvinVoigtOscillation:
    """Test complex modulus."""

    def test_oscillation_basic(self):
        model = FractionalKelvinVoigt()
        omega = np.array([0.1, 1.0, 10.0, 100.0])
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)
        assert np.iscomplexobj(result.y)
        assert np.all(np.isfinite(result.y))

    def test_oscillation_moduli_positive(self):
        model = FractionalKelvinVoigt()
        omega = np.logspace(-2, 2, 50)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)
        assert np.all(np.real(result.y) > 0)
        assert np.all(np.imag(result.y) > 0)

    def test_oscillation_storage_modulus_floor(self):
        """Test that G' ≥ Ge at all frequencies."""
        model = FractionalKelvinVoigt()
        Ge = 1e6
        model.parameters.set_value('Ge', Ge)

        omega = np.logspace(-2, 2, 50)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)
        G_prime = np.real(result.y)
        # G' should be at least Ge
        assert np.all(G_prime >= Ge * 0.95)

    def test_oscillation_power_law_scaling(self):
        """Test G'' ~ ω^α at low frequency."""
        model = FractionalKelvinVoigt()
        model.parameters.set_value('alpha', 0.5)

        omega = np.logspace(-3, -1, 20)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)
        G_double_prime = np.imag(result.y)

        log_omega = np.log10(omega)
        log_G_dp = np.log10(G_double_prime)

        slope = np.polyfit(log_omega, log_G_dp, 1)[0]
        assert 0.3 < slope < 0.7  # Should be close to alpha=0.5

    def test_oscillation_simple_sum_form(self):
        """Test that G* = Ge + c_α (iω)^α."""
        model = FractionalKelvinVoigt()
        Ge = 1e6
        c_alpha = 1e4
        alpha = 0.5
        model.parameters.set_value('Ge', Ge)
        model.parameters.set_value('c_alpha', c_alpha)
        model.parameters.set_value('alpha', alpha)

        omega = np.array([1.0])
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)

        # Manual calculation
        i_omega_alpha = (omega[0] ** alpha) * np.exp(1j * alpha * np.pi / 2.0)
        G_star_expected = Ge + c_alpha * i_omega_alpha

        # Compare
        assert np.abs(result.y[0] - G_star_expected) / np.abs(G_star_expected) < 0.01


class TestFractionalKelvinVoigtLimitCases:
    """Test limit cases."""

    def test_alpha_near_zero(self):
        """As alpha→0, SpringPot → spring, total → two springs."""
        model = FractionalKelvinVoigt()
        model.parameters.set_value('alpha', 0.05)

        t = np.logspace(-2, 2, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_alpha_near_one(self):
        """As alpha→1, SpringPot → dashpot (parallel spring-dashpot)."""
        model = FractionalKelvinVoigt()
        model.parameters.set_value('alpha', 0.95)

        t = np.logspace(-2, 2, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_very_low_c_alpha(self):
        """Test with very small SpringPot contribution."""
        model = FractionalKelvinVoigt()
        model.parameters.set_value('Ge', 1e6)
        model.parameters.set_value('c_alpha', 1.0)  # Very small
        model.parameters.set_value('alpha', 0.5)

        t = np.logspace(-2, 2, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        # Should be close to Ge everywhere
        assert np.allclose(result.y, 1e6, rtol=0.1)

    def test_zero_time_handling(self):
        model = FractionalKelvinVoigt()
        t = np.array([0.0, 0.01, 0.1, 1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))


class TestFractionalKelvinVoigtJAX:
    """Test JAX functionality."""

    def test_jit_compilation(self):
        model = FractionalKelvinVoigt()
        t = jnp.array([0.1, 1.0, 10.0])
        result = model._predict_relaxation_jax(t, 1e6, 1e4, 0.5)
        assert result.shape == t.shape

    def test_gradient_computation(self):
        model = FractionalKelvinVoigt()

        def loss_fn(Ge):
            t = jnp.array([1.0])
            result = model._predict_relaxation_jax(t, Ge, 1e4, 0.5)
            return jnp.sum(result ** 2)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(1e6)
        assert np.isfinite(gradient)

    @pytest.mark.xfail(reason="vmap over alpha not supported - alpha must be concrete for Mittag-Leffler")
    def test_vmap_over_alpha(self):
        model = FractionalKelvinVoigt()
        t = jnp.array([1.0])
        alphas = jnp.array([0.3, 0.5, 0.7])

        def predict_for_alpha(alpha):
            return model._predict_relaxation_jax(t, 1e6, 1e4, alpha)[0]

        vmapped = jax.vmap(predict_for_alpha)
        results = vmapped(alphas)
        assert results.shape == (3,)
        assert np.all(np.isfinite(results))

    def test_gradient_wrt_all_parameters(self):
        """Test gradients with respect to all three parameters."""
        model = FractionalKelvinVoigt()

        def loss_fn(params):
            Ge, c_alpha, alpha = params
            t = jnp.array([1.0])
            result = model._predict_oscillation_jax(t, Ge, c_alpha, alpha)
            return jnp.sum(jnp.abs(result) ** 2)

        grad_fn = jax.grad(loss_fn)
        params = jnp.array([1e6, 1e4, 0.5])
        gradients = grad_fn(params)

        assert gradients.shape == (3,)
        assert np.all(np.isfinite(gradients))


class TestFractionalKelvinVoigtNumericalStability:
    """Test numerical stability."""

    def test_all_modes_stable(self):
        model = FractionalKelvinVoigt()
        t = np.logspace(-2, 2, 30)

        for mode in ['relaxation', 'creep']:
            data = RheoData(x=t, y=np.zeros_like(t), domain='time')
            data.metadata['test_mode'] = mode
            result = model.predict(data)
            assert np.all(np.isfinite(result.y))

        omega = np.logspace(-2, 2, 30)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'
        result = model.predict(data)
        assert np.all(np.isfinite(result.y))

    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        model = FractionalKelvinVoigt()

        test_cases = [
            {'Ge': 1e-2, 'c_alpha': 1e-2, 'alpha': 0.1},
            {'Ge': 1e8, 'c_alpha': 1e8, 'alpha': 0.9},
            {'Ge': 1e6, 'c_alpha': 1.0, 'alpha': 0.5},
        ]

        t = np.logspace(-2, 2, 20)

        for params in test_cases:
            for key, value in params.items():
                model.parameters.set_value(key, value)

            data = RheoData(x=t, y=np.zeros_like(t), domain='time')
            data.metadata['test_mode'] = 'relaxation'
            result = model.predict(data)
            assert np.all(np.isfinite(result.y))

    def test_consistency_across_modes(self):
        """Test that different modes give consistent physical behavior."""
        model = FractionalKelvinVoigt()
        Ge = 1e6
        model.parameters.set_value('Ge', Ge)

        # Relaxation should give G(∞) = Ge
        t_long = np.array([1e6])
        data_relax = RheoData(x=t_long, y=np.zeros_like(t_long), domain='time')
        data_relax.metadata['test_mode'] = 'relaxation'
        result_relax = model.predict(data_relax)

        # Creep should give J(∞) = 1/Ge
        data_creep = RheoData(x=t_long, y=np.zeros_like(t_long), domain='time')
        data_creep.metadata['test_mode'] = 'creep'
        result_creep = model.predict(data_creep)

        # Check consistency: G(∞) * J(∞) ≈ 1
        product = result_relax.y[0] * result_creep.y[0]
        assert np.abs(product - 1.0) < 0.2


class TestFractionalKelvinVoigtRheoDataIntegration:
    """Test RheoData integration."""

    def test_rheodata_input_output(self):
        model = FractionalKelvinVoigt()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert isinstance(result, RheoData)

    def test_metadata_preservation(self):
        model = FractionalKelvinVoigt()
        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'
        data.metadata['material'] = 'filled_polymer'

        result = model.predict(data)
        assert result.metadata['material'] == 'filled_polymer'

    def test_auto_detect_test_mode(self):
        model = FractionalKelvinVoigt()

        # Frequency domain auto-detection
        omega = np.logspace(-2, 2, 50)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        result = model.predict(data)
        assert np.iscomplexobj(result.y)

    def test_explicit_test_mode_override(self):
        model = FractionalKelvinVoigt()
        t = np.array([0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')

        result = model.predict(data, test_mode='creep')
        assert isinstance(result, RheoData)


class TestFractionalKelvinVoigtErrorHandling:
    """Test error handling."""

    def test_invalid_test_mode(self):
        model = FractionalKelvinVoigt()
        t = np.array([0.1, 1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'invalid'

        with pytest.raises(ValueError, match="Unknown test mode"):
            model.predict(data)

    def test_parameter_bounds_validation(self):
        model = FractionalKelvinVoigt()

        with pytest.raises(ValueError):
            model.parameters.set_value('alpha', 1.5)

        with pytest.raises(ValueError):
            model.parameters.set_value('alpha', -0.1)

    def test_negative_modulus_rejected(self):
        model = FractionalKelvinVoigt()

        with pytest.raises(ValueError):
            model.parameters.set_value('Ge', -1000)


class TestFractionalKelvinVoigtPhysicalBehavior:
    """Test physical behavior specific to KV model."""

    def test_solid_like_behavior(self):
        """FKV should show solid-like behavior (bounded compliance)."""
        model = FractionalKelvinVoigt()

        t = np.logspace(-2, 4, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'creep'

        result = model.predict(data)

        # Compliance should plateau, not flow indefinitely
        late_time_values = result.y[-10:]
        early_late_diff = np.max(late_time_values) - np.min(late_time_values)
        assert early_late_diff / np.mean(late_time_values) < 0.1  # < 10% variation

    def test_elastic_dominance_at_high_frequency(self):
        """At high frequency, storage modulus should dominate."""
        model = FractionalKelvinVoigt()

        omega = np.logspace(2, 4, 30)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)

        G_prime = np.real(result.y)
        G_double_prime = np.imag(result.y)

        # G' should be much larger than G'' at high frequency
        assert np.all(G_prime > G_double_prime)
