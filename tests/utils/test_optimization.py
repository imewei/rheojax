"""Tests for optimization utilities.

This module tests the optimization wrapper for model fitting,
including JAX gradient integration, parameter bounds handling,
and optimization convergence.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from rheo.core.parameters import ParameterSet
from rheo.utils.optimization import nlsq_optimize, OptimizationResult


class TestOptimizationBasics:
    """Test basic optimization functionality."""

    def test_simple_quadratic_optimization(self):
        """Test optimization on a simple quadratic function."""
        # Set up parameters
        params = ParameterSet()
        params.add(name="x", value=5.0, bounds=(-10.0, 10.0))
        params.add(name="y", value=5.0, bounds=(-10.0, 10.0))

        # Define quadratic objective: (x-1)^2 + (y-2)^2
        def objective(values):
            x, y = values
            return (x - 1.0) ** 2 + (y - 2.0) ** 2

        # Optimize
        result = nlsq_optimize(objective, params, method="auto")

        # Check convergence
        assert result.success, "Optimization should converge"
        assert result.fun < 1e-6, "Should reach near-zero minimum"

        # Check optimal values
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-4)

    def test_rosenbrock_optimization(self):
        """Test optimization on Rosenbrock function."""
        # Set up parameters
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-5.0, 5.0))
        params.add(name="y", value=0.0, bounds=(-5.0, 5.0))

        # Rosenbrock function: (1-x)^2 + 100*(y-x^2)^2
        def objective(values):
            x, y = values
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        # Optimize
        result = nlsq_optimize(objective, params, method="auto", max_iter=1000)

        # Check convergence (Rosenbrock is hard, so tolerance is higher)
        assert result.success or result.fun < 1e-3, "Should converge or get close"
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-2)


class TestBoundsHandling:
    """Test parameter bounds enforcement."""

    def test_bounds_respected(self):
        """Test that optimization respects parameter bounds."""
        # Set up parameters with tight bounds
        params = ParameterSet()
        params.add(name="x", value=0.5, bounds=(0.0, 1.0))
        params.add(name="y", value=0.5, bounds=(0.0, 1.0))

        # Objective with minimum outside bounds: (x-5)^2 + (y-5)^2
        def objective(values):
            x, y = values
            return (x - 5.0) ** 2 + (y - 5.0) ** 2

        # Optimize
        result = nlsq_optimize(objective, params, method="auto")

        # Check that bounds are respected
        assert 0.0 <= result.x[0] <= 1.0, "x should be within bounds"
        assert 0.0 <= result.x[1] <= 1.0, "y should be within bounds"

        # Should converge to boundary
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)

    def test_unbounded_optimization(self):
        """Test optimization without bounds."""
        # Set up parameters without bounds
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=None)
        params.add(name="y", value=0.0, bounds=None)

        # Simple quadratic
        def objective(values):
            x, y = values
            return x**2 + y**2

        # Optimize
        result = nlsq_optimize(objective, params, method="auto")

        # Should converge to zero
        assert result.success
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-6)


class TestJAXIntegration:
    """Test JAX automatic differentiation integration."""

    def test_jax_gradients_used(self):
        """Test that JAX gradients are computed correctly."""
        # Set up parameters
        params = ParameterSet()
        params.add(name="a", value=2.0, bounds=(0.1, 10.0))
        params.add(name="b", value=2.0, bounds=(0.1, 10.0))

        # JAX-compatible objective
        def objective(values):
            a, b = values
            return jnp.sum((a - 3.0) ** 2 + (b - 4.0) ** 2)

        # Optimize with JAX gradients
        result = nlsq_optimize(objective, params, use_jax=True, method="auto")

        # Should converge
        assert result.success
        np.testing.assert_allclose(result.x, [3.0, 4.0], atol=1e-4)

    def test_jit_compiled_objective(self):
        """Test optimization with JIT-compiled objective."""

        # JIT-compiled objective
        @jax.jit
        def objective(values):
            x, y = values
            return (x - 1.5) ** 2 + (y - 2.5) ** 2

        # Set up parameters
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-5.0, 5.0))
        params.add(name="y", value=0.0, bounds=(-5.0, 5.0))

        # Optimize
        result = nlsq_optimize(objective, params, use_jax=True, method="auto")

        # Should converge
        assert result.success
        np.testing.assert_allclose(result.x, [1.5, 2.5], atol=1e-4)


class TestConvergenceCriteria:
    """Test convergence criteria and stopping conditions."""

    def test_max_iterations_respected(self):
        """Test that max_iter is respected."""
        # Set up parameters
        params = ParameterSet()
        params.add(name="x", value=10.0, bounds=None)

        # Difficult objective
        def objective(values):
            x = values[0]
            return x**4  # Very flat gradient near zero

        # Optimize with small max_iter
        result = nlsq_optimize(objective, params, method="auto", max_iter=5)

        # Should stop at max iterations
        assert result.nit <= 5, "Should stop at max iterations"

    def test_ftol_convergence(self):
        """Test convergence based on function tolerance."""
        # Set up parameters
        params = ParameterSet()
        params.add(name="x", value=5.0, bounds=None)
        params.add(name="y", value=5.0, bounds=None)

        # Simple quadratic
        def objective(values):
            x, y = values
            return (x - 1.0) ** 2 + (y - 2.0) ** 2

        # Optimize with tight ftol
        result = nlsq_optimize(objective, params, method="auto", ftol=1e-10)

        # Should converge very tightly
        assert result.success
        assert result.fun < 1e-8


class TestOptimizationResult:
    """Test OptimizationResult structure."""

    def test_result_fields(self):
        """Test that result has all required fields."""
        # Set up simple optimization
        params = ParameterSet()
        params.add(name="x", value=1.0, bounds=None)

        def objective(values):
            return values[0] ** 2

        # Optimize
        result = nlsq_optimize(objective, params, method="auto")

        # Check required fields
        assert hasattr(result, "x"), "Should have x (optimal values)"
        assert hasattr(result, "fun"), "Should have fun (objective value)"
        assert hasattr(result, "success"), "Should have success flag"
        assert hasattr(result, "nit"), "Should have iteration count"
        assert hasattr(result, "message"), "Should have status message"

    def test_result_updates_parameters(self):
        """Test that result updates ParameterSet."""
        # Set up parameters
        params = ParameterSet()
        params.add(name="x", value=5.0, bounds=None)
        params.add(name="y", value=5.0, bounds=None)

        # Store initial values
        initial_values = params.get_values()

        # Simple objective
        def objective(values):
            x, y = values
            return (x - 1.0) ** 2 + (y - 2.0) ** 2

        # Optimize
        result = nlsq_optimize(objective, params, method="auto")

        # Check that parameters were updated
        final_values = params.get_values()
        assert not np.allclose(
            initial_values, final_values
        ), "Parameters should be updated"
        np.testing.assert_allclose(final_values, result.x, atol=1e-12)


class TestMaxwellModelFitting:
    """Test optimization on a realistic rheology example (Maxwell model)."""

    def test_maxwell_parameter_fitting(self):
        """Test fitting Maxwell model parameters to synthetic data."""
        # True parameters
        G_s_true = 1e5  # Pa
        eta_s_true = 1e4  # Pa·s

        # Generate synthetic oscillation data
        omega = jnp.logspace(-2, 2, 50)  # rad/s

        # Maxwell model: G* = G_s / (1 + i*omega*tau), tau = eta_s/G_s
        tau = eta_s_true / G_s_true

        def maxwell_modulus(omega_val, G_s, eta_s):
            tau_val = eta_s / G_s
            G_star = G_s / (1 + 1j * omega_val * tau_val)
            return G_star

        # Compute true moduli
        G_star_true = maxwell_modulus(omega, G_s_true, eta_s_true)
        G_prime_true = jnp.real(G_star_true)
        G_double_prime_true = jnp.imag(G_star_true)

        # Set up parameters for fitting with initial guess far from truth
        params = ParameterSet()
        params.add(name="G_s", value=2e5, bounds=(1e3, 1e8))
        params.add(name="eta_s", value=5e3, bounds=(1e2, 1e6))

        # Objective: minimize residual sum of squares
        def objective(values):
            G_s, eta_s = values
            G_star_pred = maxwell_modulus(omega, G_s, eta_s)
            G_prime_pred = jnp.real(G_star_pred)
            G_double_prime_pred = jnp.imag(G_star_pred)

            # Weighted RSS (relative error)
            residual_G_prime = (G_prime_pred - G_prime_true) / G_prime_true
            residual_G_double_prime = (
                G_double_prime_pred - G_double_prime_true
            ) / G_double_prime_true

            rss = jnp.sum(residual_G_prime**2 + residual_G_double_prime**2)
            return rss

        # Optimize with more iterations
        result = nlsq_optimize(objective, params, use_jax=True, method="auto", max_iter=2000)

        # Check convergence
        assert result.success, "Maxwell fitting should converge"

        # Check recovered parameters (within 5% tolerance - more realistic for optimization)
        G_s_fit, eta_s_fit = result.x
        np.testing.assert_allclose(G_s_fit, G_s_true, rtol=5e-2)
        np.testing.assert_allclose(eta_s_fit, eta_s_true, rtol=5e-2)
