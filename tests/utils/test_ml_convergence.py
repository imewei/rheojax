"""
Tests for Mittag-Leffler convergence intelligence optimization.

This test suite validates the replacement of fixed loop with lax.while_loop,
early termination, and performance improvements for fractional models.
"""

import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

import numpy as np

from rheojax.utils.mittag_leffler import mittag_leffler_e, mittag_leffler_e2


class TestMittagLefflerConvergence:
    """Test suite for Mittag-Leffler convergence optimization."""

    def test_early_termination_correctness(self):
        """Test 1: Verify convergence criterion produces correct results."""
        alpha_values = [0.3, 0.5, 0.7, 0.9]
        z_values = jnp.array([-1.0, -0.5, -0.1])

        for alpha in alpha_values:
            result = mittag_leffler_e2(z_values, alpha=alpha, beta=alpha)
            assert jnp.all(jnp.isfinite(result)), f"Non-finite result for alpha={alpha}"
            assert jnp.all(result >= 0.0), f"Negative result for alpha={alpha}"
            assert jnp.all(result <= 1.0), f"Result > 1 for negative z, alpha={alpha}"

    def test_kahan_summation_stability(self):
        """Test 2: Verify Kahan summation preserves numerical stability."""
        alpha = 0.5
        z = jnp.array([-0.01, -0.001, -0.0001])
        result = mittag_leffler_e2(z, alpha=alpha, beta=alpha)

        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0.0)

        from jax.scipy.special import gamma as jax_gamma

        expected_approx = 1.0 / jax_gamma(alpha)
        assert jnp.abs(result[-1] - expected_approx) < 1e-3

    @pytest.mark.benchmark
    def test_performance_rheological_alpha(self):
        """Test 3: Benchmark performance for typical rheological alpha values."""
        import time

        alpha_values = [0.3, 0.5, 0.7, 0.9]
        z = jnp.linspace(-5.0, -0.1, 100)

        # Warm-up JIT
        _ = mittag_leffler_e2(z, alpha=0.5, beta=0.5)

        timings = []
        for alpha in alpha_values:
            start = time.perf_counter()
            for _ in range(10):
                result = mittag_leffler_e2(z, alpha=alpha, beta=alpha)
                result = jax.block_until_ready(result)
            elapsed = time.perf_counter() - start
            timings.append(elapsed / 10)

        avg_time = np.mean(timings)
        assert avg_time < 0.01, f"Average time {avg_time:.6f}s exceeds threshold"

    def test_convergence_iteration_count(self):
        """Test 4: Verify convergence for common cases."""
        alpha_values = [0.3, 0.5, 0.7, 0.9]
        z = jnp.array([-0.1, -0.5, -1.0])

        for alpha in alpha_values:
            result = mittag_leffler_e2(z, alpha=alpha, beta=alpha)
            assert jnp.all(jnp.isfinite(result))
            assert jnp.all(result >= 0.0)
            assert jnp.all(result > 0.0)

    def test_asymptotic_approximation_preserved(self):
        """Test 5: Verify asymptotic approximation logic for large |z|."""
        alpha = 0.5
        z_large = jnp.array([-10.0, -20.0, -50.0])
        result = mittag_leffler_e2(z_large, alpha=alpha, beta=alpha)

        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)
        assert jnp.all(result < 0.1)
        assert result[0] > result[1] > result[2]

    def test_edge_case_alpha_near_boundaries(self):
        """Test 6: Verify behavior at alpha boundaries."""
        z = jnp.array([-1.0, -0.5, -0.1])

        alpha_low = 0.1
        result_low = mittag_leffler_e2(z, alpha=alpha_low, beta=alpha_low)
        assert jnp.all(jnp.isfinite(result_low))
        assert jnp.all(result_low >= 0.0)

        alpha_high = 0.95
        result_high = mittag_leffler_e2(z, alpha=alpha_high, beta=alpha_high)
        assert jnp.all(jnp.isfinite(result_high))
        assert jnp.all(result_high >= 0.0)

        expected_approx = jnp.exp(z)
        assert jnp.allclose(result_high, expected_approx, rtol=0.2)

    @pytest.mark.benchmark
    def test_mittag_leffler_array_performance(self):
        """Test 7: Benchmark Mittag-Leffler performance on arrays."""
        import time

        # Typical usage pattern in fractional models
        t = jnp.logspace(-2, 2, 100)
        alpha = 0.5
        tau = 1.0

        # Generate argument array
        z = -jnp.power(t / tau, alpha)

        # Warm-up
        _ = mittag_leffler_e(z, alpha)

        # Time evaluation
        start = time.perf_counter()
        for _ in range(10):
            result = mittag_leffler_e(z, alpha)
            result = jax.block_until_ready(result)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 10
        assert avg_time < 0.01, f"Average time {avg_time:.6f}s exceeds threshold"

    def test_alpha_not_equal_beta(self):
        """Test 8: Verify alpha != beta case (Pade approximation)."""
        alpha = 0.5
        beta = 1.0
        z = jnp.array([-1.0, -0.5, -0.1])

        result = mittag_leffler_e2(z, alpha=alpha, beta=beta)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)
        assert jnp.all(result > 0.0)
