"""Tests for FluidityLocal model.

Tests cover instantiation, registry, parameters, and protocol implementations.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled
jax, jnp = safe_import_jax()

from rheojax.core.registry import ModelRegistry
from rheojax.models.fluidity import FluidityLocal


@pytest.mark.smoke
class TestFluidityLocalSmoke:
    """Smoke tests for basic FluidityLocal functionality."""

    def test_instantiation(self):
        """Test FluidityLocal can be instantiated."""
        model = FluidityLocal()
        assert model is not None
        assert hasattr(model, "parameters")

    def test_registry_entry(self):
        """Test FluidityLocal is registered correctly."""
        model = ModelRegistry.create("fluidity_local")
        assert isinstance(model, FluidityLocal)

    def test_parameters_exist(self):
        """Test expected parameters are present."""
        model = FluidityLocal()
        expected_params = [
            "G",
            "tau_y",
            "K",
            "n_flow",
            "f_eq",
            "f_inf",
            "theta",
            "a",
            "n_rejuv",
        ]
        for param in expected_params:
            assert param in model.parameters.keys()

    def test_parameter_bounds_valid(self):
        """Test parameter bounds are valid (lower < upper)."""
        model = FluidityLocal()
        for name in model.parameters.keys():
            param = model.parameters._parameters[name]
            if param.bounds is not None:
                lower, upper = param.bounds
                assert lower < upper, f"Invalid bounds for {name}"


@pytest.mark.unit
class TestFluidityLocalFlowCurve:
    """Tests for FluidityLocal flow curve (steady-state) protocol."""

    def test_flow_curve_prediction_shape(self):
        """Test flow curve prediction returns correct shape."""
        model = FluidityLocal()
        gamma_dot = np.logspace(-3, 2, 20)
        sigma = model._predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape
        assert np.all(np.isfinite(sigma))

    def test_flow_curve_monotonic_increase(self):
        """Test flow curve stress increases with shear rate."""
        model = FluidityLocal()
        gamma_dot = np.logspace(-3, 2, 50)
        sigma = model._predict_flow_curve(gamma_dot)

        # Should be monotonically increasing
        assert np.all(np.diff(sigma) >= 0)

    def test_flow_curve_yield_stress_plateau(self):
        """Test flow curve shows yield stress at low rates."""
        model = FluidityLocal()
        # Set parameters for clear yield behavior
        model.parameters.set_value("tau_y", 1000.0)
        model.parameters.set_value("f_eq", 1e-9)  # Very low rest fluidity
        model.parameters.set_value("f_inf", 1e-3)

        gamma_dot = np.logspace(-6, -3, 20)  # Very low rates
        sigma = model._predict_flow_curve(gamma_dot)

        # At low rates, stress should be close to yield stress
        # The relationship depends on model, but stress should be significant
        assert np.mean(sigma) > 50  # Non-trivial stress

    def test_flow_curve_high_rate_behavior(self):
        """Test flow curve at high rates approaches power-law."""
        model = FluidityLocal()
        gamma_dot = np.logspace(1, 3, 20)  # High rates
        sigma = model._predict_flow_curve(gamma_dot)

        # Should still be monotonic
        assert np.all(np.diff(sigma) >= 0)


@pytest.mark.unit
class TestFluidityLocalTransient:
    """Tests for FluidityLocal transient protocols."""

    def test_startup_stress_overshoot(self):
        """Test startup shows stress increase from zero."""
        model = FluidityLocal()
        t = np.linspace(0, 10, 100)

        sigma = model._simulate_transient(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="startup",
            gamma_dot=1.0,
            sigma_applied=None,
            sigma_0=None,
        )

        sigma = np.array(sigma)

        # Stress should increase from 0
        assert sigma[0] < sigma[-1]
        assert sigma[0] < 1.0  # Close to zero at t=0

    def test_relaxation_stress_decay(self):
        """Test relaxation shows stress decay."""
        model = FluidityLocal()
        t = np.linspace(0, 100, 100)

        sigma = model._simulate_transient(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="relaxation",
            gamma_dot=None,
            sigma_applied=None,
            sigma_0=1000.0,
        )

        sigma = np.array(sigma)

        # Stress should decay
        assert sigma[0] > sigma[-1]
        assert sigma[0] > 500  # Started at 1000

    def test_creep_strain_increases(self):
        """Test creep shows strain accumulation."""
        model = FluidityLocal()
        # Set high fluidity for observable creep
        model.parameters.set_value("f_eq", 1e-4)
        model.parameters.set_value("f_inf", 1e-2)

        t = np.linspace(0, 10, 100)

        gamma = model._simulate_transient(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="creep",
            gamma_dot=None,
            sigma_applied=1000.0,
            sigma_0=None,
        )

        gamma = np.array(gamma)

        # Strain should increase
        assert gamma[-1] > gamma[0]
        assert gamma[0] < 0.1  # Started near zero


@pytest.mark.unit
class TestFluidityLocalOscillation:
    """Tests for FluidityLocal oscillation protocols."""

    def test_saos_prediction_shape(self):
        """Test SAOS prediction returns [G', G''] shape."""
        model = FluidityLocal()
        omega = np.logspace(-2, 2, 20)

        G_star = model._predict_saos_jit(
            jnp.asarray(omega),
            model.parameters.get_value("G"),
            model.parameters.get_value("f_eq"),
            model.parameters.get_value("theta"),
        )

        G_star = np.array(G_star)
        assert G_star.shape == (len(omega), 2)
        assert np.all(np.isfinite(G_star))
        assert np.all(G_star >= 0)

    def test_saos_maxwell_behavior(self):
        """Test SAOS shows Maxwell-like behavior."""
        model = FluidityLocal()
        omega = np.logspace(-4, 4, 50)

        G_star = model._predict_saos_jit(
            jnp.asarray(omega),
            model.parameters.get_value("G"),
            model.parameters.get_value("f_eq"),
            model.parameters.get_value("theta"),
        )

        G_star = np.array(G_star)
        G_prime = G_star[:, 0]
        G_double_prime = G_star[:, 1]

        # G' should increase with frequency
        assert G_prime[-1] > G_prime[0]

        # G'' should have a maximum (characteristic of Maxwell)
        max_idx = np.argmax(G_double_prime)
        assert max_idx > 0 and max_idx < len(omega) - 1

    def test_laos_simulation(self):
        """Test LAOS simulation returns valid output."""
        model = FluidityLocal()

        strain, stress = model.simulate_laos(
            gamma_0=0.1,
            omega=1.0,
            n_cycles=2,
            n_points_per_cycle=64,
        )

        assert strain.shape == stress.shape
        assert len(strain) == 2 * 64
        assert np.all(np.isfinite(strain))
        assert np.all(np.isfinite(stress))

    def test_extract_harmonics(self):
        """Test harmonic extraction from LAOS."""
        model = FluidityLocal()

        _, stress = model.simulate_laos(
            gamma_0=0.5,  # Larger amplitude for nonlinear response
            omega=1.0,
            n_cycles=4,  # More cycles for better FFT
            n_points_per_cycle=256,
        )

        harmonics = model.extract_harmonics(stress, n_points_per_cycle=256)

        assert "I_1" in harmonics
        assert "I_3" in harmonics
        assert "I_3_I_1" in harmonics
        assert harmonics["I_1"] > 0


@pytest.mark.unit
class TestFluidityLocalModelFunction:
    """Tests for Bayesian model_function interface."""

    def test_model_function_flow_curve(self):
        """Test model_function works for flow curve mode."""
        model = FluidityLocal()
        model._test_mode = "flow_curve"

        X = np.logspace(-2, 1, 10)
        params = list(model.parameters.get_values())

        result = model.model_function(X, params, test_mode="flow_curve")

        assert result.shape == X.shape
        assert np.all(np.isfinite(result))

    def test_model_function_oscillation(self):
        """Test model_function works for oscillation mode."""
        model = FluidityLocal()
        model._test_mode = "oscillation"

        X = np.logspace(-1, 2, 10)
        params = list(model.parameters.get_values())

        result = model.model_function(X, params, test_mode="oscillation")

        assert result.shape == (len(X), 2)
        assert np.all(np.isfinite(result))


@pytest.mark.unit
class TestFluidityLocalFitting:
    """Tests for FluidityLocal fitting capability."""

    def test_fit_flow_curve_synthetic(self):
        """Test fitting synthetic flow curve data."""
        # Generate synthetic data
        model_true = FluidityLocal()
        model_true.parameters.set_value("tau_y", 500.0)
        model_true.parameters.set_value("a", 2.0)

        gamma_dot = np.logspace(-2, 2, 30)
        sigma_true = model_true._predict_flow_curve(gamma_dot)

        # Add noise
        np.random.seed(42)
        sigma_noisy = sigma_true * (1 + 0.05 * np.random.randn(len(sigma_true)))

        # Fit with new model
        model_fit = FluidityLocal()
        model_fit.fit(gamma_dot, sigma_noisy, test_mode="flow_curve", max_iter=100)

        # Check fit
        assert model_fit.fitted_

    def test_fit_sets_test_mode(self):
        """Test fitting sets test_mode correctly."""
        model = FluidityLocal()
        gamma_dot = np.logspace(-1, 1, 10)
        sigma = model._predict_flow_curve(gamma_dot)

        model.fit(gamma_dot, sigma, test_mode="flow_curve", max_iter=10)

        assert model._test_mode == "flow_curve"
