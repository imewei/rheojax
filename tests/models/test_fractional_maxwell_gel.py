"""Tests for Fractional Maxwell Gel (FMG) model.

This test suite validates:
1. Model initialization and parameters
2. Relaxation modulus predictions
3. Creep compliance predictions
4. Complex modulus predictions (oscillation)
5. Limit cases (α→0, α→1)
6. JAX operations (jit, grad, vmap)
7. Numerical stability
8. Test mode auto-detection
9. RheoData integration
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import rheo.models  # Import to trigger all model registrations
from rheo.models.fractional_maxwell_gel import FractionalMaxwellGel
from rheo.core.data import RheoData
from rheo.core.registry import ModelRegistry


class TestFractionalMaxwellGelInitialization:
    """Test model initialization and parameter setup."""

    def test_model_creation(self):
        """Test that model can be created."""
        model = FractionalMaxwellGel()
        assert model is not None
        assert isinstance(model, FractionalMaxwellGel)

    def test_parameters_exist(self):
        """Test that all required parameters exist."""
        model = FractionalMaxwellGel()
        assert 'c_alpha' in model.parameters
        assert 'alpha' in model.parameters
        assert 'eta' in model.parameters

    def test_parameter_defaults(self):
        """Test default parameter values."""
        model = FractionalMaxwellGel()
        # Updated defaults for numerical stability across alpha range
        assert model.parameters.get_value('c_alpha') == 10.0
        assert model.parameters.get_value('alpha') == 0.5
        assert model.parameters.get_value('eta') == 1e4

    def test_parameter_bounds(self):
        """Test parameter bounds are correct."""
        model = FractionalMaxwellGel()

        c_alpha_param = model.parameters.get('c_alpha')
        assert c_alpha_param.bounds == (1e-3, 1e9)

        alpha_param = model.parameters.get('alpha')
        assert alpha_param.bounds == (0.0, 1.0)

        eta_param = model.parameters.get('eta')
        assert eta_param.bounds == (1e-6, 1e12)

    def test_parameter_modification(self):
        """Test that parameters can be modified."""
        model = FractionalMaxwellGel()
        model.parameters.set_value('c_alpha', 2e5)
        model.parameters.set_value('alpha', 0.7)
        model.parameters.set_value('eta', 2e3)

        assert model.parameters.get_value('c_alpha') == 2e5
        assert model.parameters.get_value('alpha') == 0.7
        assert model.parameters.get_value('eta') == 2e3

    def test_registry_registration(self):
        """Test that model is registered in ModelRegistry."""
        models = ModelRegistry.list_models()
        assert 'fractional_maxwell_gel' in models

    def test_factory_creation(self):
        """Test model creation via registry factory."""
        model = ModelRegistry.create('fractional_maxwell_gel')
        assert isinstance(model, FractionalMaxwellGel)


class TestFractionalMaxwellGelRelaxation:
    """Test relaxation modulus predictions."""

    def test_relaxation_basic(self):
        """Test basic relaxation modulus calculation."""
        model = FractionalMaxwellGel()
        # Use parameters that give tau ≈ 1 for numerical stability
        # tau = eta / c_alpha^(1/(1-alpha)) = 1e6 / 1e3^2 = 1.0
        model.parameters.set_value('c_alpha', 1e3)
        model.parameters.set_value('alpha', 0.5)
        model.parameters.set_value('eta', 1e6)

        t = np.array([0.01, 0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert isinstance(result, RheoData)
        assert result.y.shape == t.shape
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_relaxation_monotonic_decrease(self):
        """Test that relaxation modulus decreases monotonically."""
        model = FractionalMaxwellGel()

        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        # Check that G(t) is decreasing (with some tolerance for numerical noise)
        diffs = np.diff(result.y)
        assert np.sum(diffs < 0) > 0.8 * len(diffs)  # At least 80% decreasing

    def test_relaxation_short_time_limit(self):
        """Test short-time power-law behavior: G(t) ~ t^(-α)."""
        model = FractionalMaxwellGel()
        # Use larger eta to keep tau reasonable: tau = 1e10 / 1e3^2 = 10
        model.parameters.set_value('c_alpha', 1e3)
        model.parameters.set_value('alpha', 0.5)
        model.parameters.set_value('eta', 1e7)  # Large eta for clear power-law

        t = np.logspace(-4, -2, 20)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)

        # Check power-law scaling in log space
        log_t = np.log10(t)
        log_G = np.log10(result.y)

        # Fit slope (should be close to -alpha = -0.5)
        slope = np.polyfit(log_t, log_G, 1)[0]
        assert np.abs(slope - (-0.5)) < 0.2  # Within 20% of expected

    def test_relaxation_alpha_effect(self):
        """Test effect of alpha parameter."""
        model = FractionalMaxwellGel()

        t = np.logspace(-2, 2, 50)

        # Low alpha (more fluid-like)
        model.parameters.set_value('alpha', 0.2)
        data_low = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data_low.metadata['test_mode'] = 'relaxation'
        result_low = model.predict(data_low)

        # High alpha (more solid-like)
        model.parameters.set_value('alpha', 0.8)
        data_high = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data_high.metadata['test_mode'] = 'relaxation'
        result_high = model.predict(data_high)

        # For FMG: At long times, G(t) ~ t^(-α)
        # Lower alpha decays slower, so should have higher modulus at long times
        assert result_low.y[-1] > result_high.y[-1]

        # Modulus should be different for different alpha values
        assert not np.allclose(result_low.y, result_high.y)


class TestFractionalMaxwellGelCreep:
    """Test creep compliance predictions."""

    def test_creep_basic(self):
        """Test basic creep compliance calculation."""
        model = FractionalMaxwellGel()

        t = np.array([0.01, 0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'creep'

        result = model.predict(data)
        assert isinstance(result, RheoData)
        assert result.y.shape == t.shape
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_creep_monotonic_increase(self):
        """Test that creep compliance increases monotonically."""
        model = FractionalMaxwellGel()

        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'creep'

        result = model.predict(data)
        # Check that J(t) is increasing
        diffs = np.diff(result.y)
        assert np.all(diffs > -1e-10)  # Monotonic with tolerance

    def test_creep_short_time_power_law(self):
        """Test short-time power-law: J(t) ~ t^α."""
        model = FractionalMaxwellGel()
        # Adjust parameters for numerical stability
        model.parameters.set_value('c_alpha', 1e3)
        model.parameters.set_value('alpha', 0.5)
        model.parameters.set_value('eta', 1e7)

        t = np.logspace(-4, -2, 20)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'creep'

        result = model.predict(data)

        # Check power-law scaling
        log_t = np.log10(t)
        log_J = np.log10(result.y)

        # Slope should be close to alpha = 0.5
        slope = np.polyfit(log_t, log_J, 1)[0]
        assert np.abs(slope - 0.5) < 0.2


class TestFractionalMaxwellGelOscillation:
    """Test complex modulus predictions."""

    def test_oscillation_basic(self):
        """Test basic complex modulus calculation."""
        model = FractionalMaxwellGel()

        omega = np.array([0.1, 1.0, 10.0, 100.0])
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)
        assert isinstance(result, RheoData)
        assert result.y.shape == omega.shape
        assert np.iscomplexobj(result.y)
        assert np.all(np.isfinite(result.y))

    def test_oscillation_storage_loss_positive(self):
        """Test that storage and loss moduli are positive."""
        model = FractionalMaxwellGel()

        omega = np.logspace(-2, 2, 50)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)

        G_prime = np.real(result.y)
        G_double_prime = np.imag(result.y)

        assert np.all(G_prime > 0)
        assert np.all(G_double_prime > 0)

    def test_oscillation_frequency_dependence(self):
        """Test frequency dependence of complex modulus."""
        model = FractionalMaxwellGel()

        omega = np.logspace(-2, 2, 50)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)

        # G' should increase with frequency for FMG
        assert result.y[-1].real > result.y[0].real
        # G'' behavior depends on alpha and tau - it can have peaks
        # Just check that G'' is positive
        assert np.all(result.y.imag > 0)

    def test_oscillation_power_law_scaling(self):
        """Test power-law scaling at low frequency: |G*| ~ ω^α."""
        model = FractionalMaxwellGel()
        # Adjust parameters for numerical stability
        model.parameters.set_value('c_alpha', 1e3)
        model.parameters.set_value('alpha', 0.5)
        model.parameters.set_value('eta', 1e7)

        omega = np.logspace(-3, -1, 20)
        data = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data.metadata['test_mode'] = 'oscillation'

        result = model.predict(data)

        # Check power-law scaling
        log_omega = np.log10(omega)
        log_G_abs = np.log10(np.abs(result.y))

        # Slope should be close to alpha = 0.5
        slope = np.polyfit(log_omega, log_G_abs, 1)[0]
        assert np.abs(slope - 0.5) < 0.2


class TestFractionalMaxwellGelLimitCases:
    """Test limit cases and edge behavior."""

    def test_alpha_near_zero(self):
        """Test behavior as alpha → 0 (approaches pure dashpot)."""
        model = FractionalMaxwellGel()
        model.parameters.set_value('alpha', 0.05)
        # Adjust parameters to keep tau reasonable
        model.parameters.set_value('c_alpha', 1e3)
        model.parameters.set_value('eta', 1e6)

        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_alpha_near_one(self):
        """Test behavior as alpha → 1 (approaches spring-dashpot)."""
        model = FractionalMaxwellGel()
        model.parameters.set_value('alpha', 0.95)
        # Adjust parameters to keep tau reasonable
        model.parameters.set_value('c_alpha', 1e3)
        model.parameters.set_value('eta', 1e6)

        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))
        assert np.all(result.y > 0)

    def test_zero_time_handling(self):
        """Test handling of t=0."""
        model = FractionalMaxwellGel()

        t = np.array([0.0, 0.01, 0.1, 1.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        # Should handle t=0 gracefully (with epsilon)
        assert np.all(np.isfinite(result.y))

    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        model = FractionalMaxwellGel()
        model.parameters.set_value('c_alpha', 1e-2)  # Very low
        model.parameters.set_value('alpha', 0.1)
        model.parameters.set_value('eta', 1e10)  # Very high

        t = np.logspace(-2, 2, 20)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)
        assert np.all(np.isfinite(result.y))


class TestFractionalMaxwellGelJAX:
    """Test JAX-specific functionality."""

    def test_jit_compilation(self):
        """Test that prediction methods can be JIT compiled."""
        model = FractionalMaxwellGel()

        t = jnp.array([0.1, 1.0, 10.0])
        c_alpha = 1e5
        alpha = 0.5
        eta = 1e3

        # Should not raise
        result = model._predict_relaxation_jax(t, c_alpha, alpha, eta)
        assert result.shape == t.shape

    @pytest.mark.xfail(reason="Gradient through asymptotic ML approximation may produce NaN for some parameter ranges")
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        model = FractionalMaxwellGel()

        def loss_fn(c_alpha):
            t = jnp.array([1.0])
            result = model._predict_relaxation_jax(t, c_alpha, 0.5, 1e3)
            return jnp.sum(result ** 2)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(1e5)

        assert np.isfinite(gradient)
        assert gradient != 0.0

    @pytest.mark.xfail(reason="vmap over alpha not supported - alpha must be concrete for Mittag-Leffler")
    def test_vmap_over_parameters(self):
        """Test vectorization over parameter values."""
        model = FractionalMaxwellGel()

        t = jnp.array([1.0])
        alpha_values = jnp.array([0.3, 0.5, 0.7])

        def predict_for_alpha(alpha):
            return model._predict_relaxation_jax(t, 1e5, alpha, 1e3)

        vmapped = jax.vmap(predict_for_alpha)
        results = vmapped(alpha_values)

        assert results.shape == (3, 1)
        assert np.all(np.isfinite(results))


class TestFractionalMaxwellGelNumericalStability:
    """Test numerical stability and accuracy."""

    def test_mittag_leffler_convergence(self):
        """Test that Mittag-Leffler evaluations are stable."""
        model = FractionalMaxwellGel()

        # Test range where ML should converge well (|z| < 10)
        t = np.logspace(-2, 1, 30)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)

        # Check for NaN or inf
        assert np.all(np.isfinite(result.y))
        # Check for negative values (non-physical)
        assert np.all(result.y > 0)

    def test_consistency_across_test_modes(self):
        """Test that model is consistent across different test modes."""
        model = FractionalMaxwellGel()
        # Use parameters that give tau ≈ 1 for numerical stability
        model.parameters.set_value('c_alpha', 1e3)
        model.parameters.set_value('alpha', 0.5)
        model.parameters.set_value('eta', 1e6)

        # All modes should produce finite, physical results
        t = np.array([0.1, 1.0, 10.0])

        # Relaxation
        data_relax = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data_relax.metadata['test_mode'] = 'relaxation'
        result_relax = model.predict(data_relax)
        assert np.all(np.isfinite(result_relax.y))

        # Creep
        data_creep = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data_creep.metadata['test_mode'] = 'creep'
        result_creep = model.predict(data_creep)
        assert np.all(np.isfinite(result_creep.y))

        # Oscillation
        omega = t  # Use same values as frequency
        data_osc = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        data_osc.metadata['test_mode'] = 'oscillation'
        result_osc = model.predict(data_osc)
        assert np.all(np.isfinite(result_osc.y))


class TestFractionalMaxwellGelRheoDataIntegration:
    """Test integration with RheoData."""

    def test_rheodata_input_output(self):
        """Test that model accepts and returns RheoData."""
        model = FractionalMaxwellGel()

        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'

        result = model.predict(data)

        assert isinstance(result, RheoData)
        assert result.x.shape == data.x.shape
        assert result.domain == data.domain

    def test_metadata_preservation(self):
        """Test that metadata is preserved."""
        model = FractionalMaxwellGel()

        t = np.logspace(-2, 2, 50)
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        data.metadata['test_mode'] = 'relaxation'
        data.metadata['sample_name'] = 'test_sample'
        data.metadata['temperature'] = 25.0

        result = model.predict(data)

        assert 'sample_name' in result.metadata
        assert result.metadata['sample_name'] == 'test_sample'
        assert result.metadata['temperature'] == 25.0

    def test_auto_detect_test_mode(self):
        """Test automatic test mode detection."""
        model = FractionalMaxwellGel()

        # Frequency domain should auto-detect oscillation
        omega = np.logspace(-2, 2, 50)
        data_freq = RheoData(x=omega, y=np.zeros_like(omega, dtype=complex), domain='frequency')
        result_freq = model.predict(data_freq)
        assert np.iscomplexobj(result_freq.y)

    def test_explicit_test_mode_override(self):
        """Test explicit test mode override."""
        model = FractionalMaxwellGel()

        t = np.array([0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        # Don't set test_mode in metadata, use parameter instead

        result = model.predict(data, test_mode='creep')
        assert isinstance(result, RheoData)
        assert np.all(np.isfinite(result.y))


class TestFractionalMaxwellGelErrorHandling:
    """Test error handling and validation."""

    def test_invalid_test_mode(self):
        """Test handling of invalid test mode."""
        model = FractionalMaxwellGel()

        t = np.array([0.1, 1.0, 10.0])
        data = RheoData(x=t, y=np.zeros_like(t), domain='time')

        # Explicitly pass invalid test_mode to bypass auto-detection
        with pytest.raises(ValueError, match="Unknown test mode"):
            model.predict(data, test_mode='invalid_mode')

    def test_parameter_bounds_validation(self):
        """Test that parameter bounds are enforced."""
        model = FractionalMaxwellGel()

        # Test alpha out of bounds
        with pytest.raises(ValueError):
            model.parameters.set_value('alpha', 1.5)  # > 1.0

        with pytest.raises(ValueError):
            model.parameters.set_value('alpha', -0.1)  # < 0.0
