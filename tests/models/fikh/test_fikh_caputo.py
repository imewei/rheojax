"""Tests for Caputo fractional derivative implementation.

Tests cover:
- GL weights computation
- L1 coefficient computation
- Caputo derivative accuracy
- History buffer operations
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fikh._caputo import (
    caputo_derivative_l1,
    compute_gl_weights,
    compute_l1_coefficients,
    create_history_buffer,
    fractional_derivative_with_short_memory,
    initialize_history_with_value,
    update_history_buffer,
)

jax, jnp = safe_import_jax()


class TestGLWeights:
    """Test Grünwald-Letnikov weights computation."""

    @pytest.mark.smoke
    def test_first_weight_is_one(self):
        """Test w_0 = 1 for all alpha."""
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            weights = compute_gl_weights(alpha, 10)
            assert jnp.isclose(weights[0], 1.0)

    def test_weights_decrease(self):
        """Test weights generally decrease in magnitude."""
        weights = compute_gl_weights(0.5, 20)
        # First few weights should decrease in absolute value
        for i in range(1, 5):
            assert abs(float(weights[i])) < abs(float(weights[i - 1]))

    def test_weights_sign_pattern(self):
        """Test weights have expected sign pattern for 0 < α < 1."""
        weights = compute_gl_weights(0.5, 10)
        # w_0 = 1 (positive)
        # w_k = (1 - (1+α)/k) * w_{k-1}
        # For α = 0.5: w_1 = (1 - 1.5) * 1 = -0.5
        # w_2 = (1 - 0.75) * (-0.5) = -0.125
        # First weight is positive, rest become negative
        assert float(weights[0]) > 0  # w_0 = 1
        assert float(weights[1]) < 0  # w_1 is negative

    def test_weights_sum_converges(self):
        """Test GL weights sum converges to 0 as n → ∞."""
        # For large n, sum should approach 0
        n_large = 1000
        weights = compute_gl_weights(0.5, n_large)
        weight_sum = float(jnp.sum(weights))
        # Sum should be close to 0 (within tolerance)
        assert abs(weight_sum) < 0.1


class TestL1Coefficients:
    """Test L1 scheme coefficients."""

    @pytest.mark.smoke
    def test_coefficients_positive(self):
        """Test L1 coefficients are positive."""
        for alpha in [0.1, 0.5, 0.9]:
            coeffs = compute_l1_coefficients(alpha, 20)
            assert (coeffs >= 0).all()

    def test_first_coefficient(self):
        """Test b_0 = 1 for all alpha."""
        for alpha in [0.1, 0.5, 0.9]:
            coeffs = compute_l1_coefficients(alpha, 10)
            assert jnp.isclose(coeffs[0], 1.0)

    def test_coefficients_decrease(self):
        """Test coefficients decrease monotonically."""
        coeffs = compute_l1_coefficients(0.5, 20)
        for i in range(1, 10):
            assert float(coeffs[i]) < float(coeffs[i - 1])


class TestCaputoDerivativeL1:
    """Test L1 scheme Caputo derivative computation."""

    @pytest.mark.smoke
    def test_constant_function_zero_derivative(self):
        """Test derivative of constant is zero."""
        n = 50
        dt = 0.1
        alpha = 0.5

        # Constant function history
        f_history = jnp.ones(n) * 5.0
        b_coeffs = compute_l1_coefficients(alpha, n)

        D_alpha_f = caputo_derivative_l1(f_history, dt, alpha, b_coeffs)

        # Derivative of constant should be zero
        assert jnp.isclose(D_alpha_f, 0.0, atol=1e-10)

    def test_linear_function(self):
        """Test derivative of linear function."""
        n = 100
        dt = 0.1
        alpha = 0.5

        # Linear function: f(t) = t
        t = jnp.arange(n) * dt
        f_history = t
        b_coeffs = compute_l1_coefficients(alpha, n)

        D_alpha_f = caputo_derivative_l1(f_history, dt, alpha, b_coeffs)

        # Analytical result for D^α t = t^{1-α} / Γ(2-α)
        t_final = float(t[-1])
        gamma_2_minus_alpha = jax.scipy.special.gamma(2.0 - alpha)
        expected = t_final ** (1 - alpha) / gamma_2_minus_alpha

        # Allow some numerical error
        assert jnp.isclose(D_alpha_f, expected, rtol=0.1)

    def test_alpha_near_one_matches_integer(self):
        """Test α → 1 approaches integer derivative."""
        n = 100
        dt = 0.01
        alpha = 0.99

        # Linear function: f(t) = 2*t
        t = jnp.arange(n) * dt
        f_history = 2.0 * t
        b_coeffs = compute_l1_coefficients(alpha, n)

        D_alpha_f = caputo_derivative_l1(f_history, dt, alpha, b_coeffs)

        # For α → 1, D^α f → df/dt = 2
        assert abs(float(D_alpha_f) - 2.0) < 0.5  # Allow error due to discretization


class TestHistoryBuffer:
    """Test history buffer operations."""

    @pytest.mark.smoke
    def test_create_buffer(self):
        """Test buffer creation."""
        buffer = create_history_buffer(100, 1)
        assert buffer.shape == (100,)
        assert (buffer == 0).all()

    def test_create_multidim_buffer(self):
        """Test multi-dimensional buffer creation."""
        buffer = create_history_buffer(50, 3)
        assert buffer.shape == (50, 3)

    @pytest.mark.smoke
    def test_update_buffer(self):
        """Test buffer update shifts correctly."""
        buffer = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        new_value = jnp.array(6.0)

        updated = update_history_buffer(buffer, new_value)

        expected = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        assert jnp.allclose(updated, expected)

    def test_initialize_with_value(self):
        """Test buffer initialization with constant."""
        buffer = create_history_buffer(10, 1)
        initialized = initialize_history_with_value(buffer, jnp.array(0.5))

        assert (initialized == 0.5).all()


class TestFractionalDerivativeWithMemory:
    """Test fractional derivative with short memory truncation."""

    @pytest.mark.smoke
    def test_short_memory_produces_result(self):
        """Test short memory computation produces valid output."""
        n_history = 50
        alpha = 0.5
        dt = 0.1

        # Create history and current value
        f_history = jnp.ones(n_history) * 0.5
        f_current = jnp.array(0.6)
        b_coeffs = compute_l1_coefficients(alpha, n_history + 1)

        result = fractional_derivative_with_short_memory(
            f_current, f_history, alpha, dt, b_coeffs
        )

        assert jnp.isfinite(result)

    def test_memory_truncation_effect(self):
        """Test that longer memory gives different (more accurate) result."""
        alpha = 0.5
        dt = 0.1

        # Create declining history (like relaxation)
        n_short = 20
        n_long = 100

        history_short = jnp.exp(-jnp.arange(n_short) * dt * 0.1)
        history_long = jnp.exp(-jnp.arange(n_long) * dt * 0.1)

        f_current = jnp.array(jnp.exp(-n_long * dt * 0.1))

        b_short = compute_l1_coefficients(alpha, n_short + 1)
        b_long = compute_l1_coefficients(alpha, n_long + 1)

        result_short = fractional_derivative_with_short_memory(
            f_current, history_short, alpha, dt, b_short
        )
        result_long = fractional_derivative_with_short_memory(
            f_current, history_long, alpha, dt, b_long
        )

        # Results should be different (longer memory is more accurate)
        assert not jnp.isclose(result_short, result_long, rtol=0.01)
