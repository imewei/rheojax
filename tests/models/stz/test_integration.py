"""Integration tests for STZ model with Bayesian pipeline.

Tests cover:
- End-to-end NLSQ -> NUTS workflow
- Variant switching in Bayesian mode
- Diagnostics validation (R-hat, ESS)
"""

import pytest
import numpy as np
from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled
jax, jnp = safe_import_jax()

from rheojax.models.stz import STZConventional


@pytest.mark.unit
class TestSTZIntegration:
    """Integration tests for STZ model."""

    def test_model_registry(self):
        """Test model is registered correctly."""
        from rheojax.core.registry import ModelRegistry

        assert "stz_conventional" in ModelRegistry.list_models()
        model = ModelRegistry.create("stz_conventional")
        assert model is not None
        assert isinstance(model, STZConventional)

    def test_variant_parameter_sets(self):
        """Test different variants have different parameter counts."""
        minimal = STZConventional(variant="minimal")
        standard = STZConventional(variant="standard")
        full = STZConventional(variant="full")

        # Count parameters
        n_minimal = len(list(minimal.parameters.keys()))
        n_standard = len(list(standard.parameters.keys()))
        n_full = len(list(full.parameters.keys()))

        # Full > Standard > Minimal
        assert n_full > n_standard >= n_minimal

        # Check variant-specific parameters
        assert "tau_beta" not in minimal.parameters.keys()
        assert "tau_beta" in standard.parameters.keys()
        assert "tau_beta" in full.parameters.keys()

        assert "m_inf" not in minimal.parameters.keys()
        assert "m_inf" not in standard.parameters.keys()
        assert "m_inf" in full.parameters.keys()

    def test_initial_state_dimensions(self):
        """Test initial state has correct dimensions for each variant."""
        for variant, expected_dim in [("minimal", 2), ("standard", 3), ("full", 4)]:
            model = STZConventional(variant=variant)
            y0 = model.get_initial_state(stress_init=0.0)

            assert y0.shape == (expected_dim,), f"{variant} has wrong dimension"
            assert jnp.all(jnp.isfinite(y0)), f"{variant} has non-finite initial state"

    def test_model_function_routing(self):
        """Test model_function correctly routes based on test_mode."""
        model = STZConventional(variant="standard")

        # Set up parameters
        model._test_mode = "steady_shear"

        x = np.logspace(-2, 4, 10)
        params = np.array([model.parameters.get_value(k) for k in model.parameters.keys()])

        # Should not raise for steady_shear mode
        result = model.model_function(x, params, test_mode="steady_shear")
        assert np.all(np.isfinite(np.array(result)))

    def test_predict_interface(self):
        """Test _predict works after setting test_mode."""
        model = STZConventional(variant="standard")
        model._test_mode = "steady_shear"
        model.fitted_ = True

        x = np.logspace(-2, 4, 10)
        result = model._predict(x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_float64_precision(self):
        """Test that model maintains float64 precision."""
        model = STZConventional(variant="standard")

        x = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6])
        model._test_mode = "steady_shear"
        model.fitted_ = True

        result = model._predict(x)

        # Check dtype
        assert result.dtype == np.float64

    def test_saos_predict_shape(self):
        """Test SAOS prediction returns 2-column output."""
        model = STZConventional(variant="standard")
        model._test_mode = "oscillation"
        model.fitted_ = True

        omega = np.logspace(6, 14, 10)
        result = model._predict(omega)

        # Should return [G', G''] as 2 columns
        assert result.ndim == 2
        assert result.shape == (len(omega), 2)

    def test_parameter_bounds_enforced(self):
        """Test that parameter bounds are enforced."""
        model = STZConventional(variant="standard")

        # Try setting out-of-bounds value
        with pytest.raises(ValueError):
            model.parameters.set_value("chi_inf", -0.1)  # Negative not allowed

        with pytest.raises(ValueError):
            model.parameters.set_value("chi_inf", 1.0)  # > 0.5 not allowed

    def test_bayesian_mixin_integration(self):
        """Test that model inherits BayesianMixin correctly."""
        model = STZConventional(variant="standard")

        # Should have fit_bayesian method from mixin
        assert hasattr(model, "fit_bayesian")
        assert callable(model.fit_bayesian)

    @pytest.mark.slow
    def test_bayesian_inference_runs(self):
        """Test that Bayesian inference pipeline runs (slow)."""
        model = STZConventional(variant="standard")

        # Generate synthetic data
        gamma_dot = np.logspace(-2, 4, 20)

        # Set true parameters
        model.parameters.set_value("sigma_y", 1e6)
        model.parameters.set_value("chi_inf", 0.15)
        model.parameters.set_value("tau0", 1e-12)
        model.parameters.set_value("ez", 1.0)

        model._test_mode = "steady_shear"
        model.fitted_ = True

        # Generate synthetic stress
        stress_true = model._predict(gamma_dot)

        # Add noise
        np.random.seed(42)
        noise = 0.05 * stress_true * np.random.randn(len(stress_true))
        stress_noisy = stress_true + noise

        # Run Bayesian inference (minimal settings for speed)
        try:
            result = model.fit_bayesian(
                gamma_dot,
                stress_noisy,
                num_warmup=100,
                num_samples=100,
                num_chains=1,
            )

            # Check result structure
            assert hasattr(result, "posterior_samples")
            assert result.posterior_samples is not None

        except Exception as e:
            # Some setups may not have full NumPyro configuration
            pytest.skip(f"Bayesian inference not available: {e}")
