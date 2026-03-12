"""Performance benchmarks for F9 compliance.

Tests that NLSQ fits complete within 2s for N ≤ 1000 on CPU,
and that JIT compilation overhead does not exceed 10s.
"""

import time

import numpy as np
import pytest

NLSQ_TIME_LIMIT = 2.0  # seconds
JIT_TIME_LIMIT = 10.0  # seconds


def _maxwell_data(n=500):
    """Generate Maxwell relaxation data."""
    t = np.logspace(-2, 2, n)
    G_t = 1000.0 * np.exp(-t / 1.0) + np.random.default_rng(42).normal(0, 1.0, n)
    G_t = np.maximum(G_t, 1e-3)
    return t, G_t


def _maxwell_osc_data(n=500):
    """Generate Maxwell oscillation data."""
    omega = np.logspace(-2, 2, n)
    G0, tau = 1000.0, 1.0
    wt2 = (omega * tau) ** 2
    G_prime = G0 * wt2 / (1.0 + wt2)
    G_dbl = G0 * omega * tau / (1.0 + wt2)
    return omega, G_prime + 1j * G_dbl


@pytest.mark.slow
class TestNLSQPerformance:
    """NLSQ fits must complete in < 2s for N ≤ 1000."""

    def test_maxwell_relaxation_500pts(self):
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t, G_t = _maxwell_data(500)

        start = time.perf_counter()
        model.fit(t, G_t, test_mode="relaxation")
        elapsed = time.perf_counter() - start

        assert (
            elapsed < NLSQ_TIME_LIMIT
        ), f"Maxwell relaxation fit took {elapsed:.2f}s (limit {NLSQ_TIME_LIMIT}s)"

    def test_maxwell_relaxation_1000pts(self):
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t, G_t = _maxwell_data(1000)

        start = time.perf_counter()
        model.fit(t, G_t, test_mode="relaxation")
        elapsed = time.perf_counter() - start

        assert (
            elapsed < NLSQ_TIME_LIMIT
        ), f"Maxwell relaxation 1000pts took {elapsed:.2f}s (limit {NLSQ_TIME_LIMIT}s)"

    def test_maxwell_oscillation_500pts(self):
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        omega, G_star = _maxwell_osc_data(500)

        start = time.perf_counter()
        model.fit(omega, G_star, test_mode="oscillation")
        elapsed = time.perf_counter() - start

        assert (
            elapsed < NLSQ_TIME_LIMIT
        ), f"Maxwell oscillation fit took {elapsed:.2f}s (limit {NLSQ_TIME_LIMIT}s)"

    def test_zener_relaxation_500pts(self):
        from rheojax.models.classical import Zener

        model = Zener()
        t = np.logspace(-2, 2, 500)
        G_t = 500.0 + 500.0 * np.exp(-t / 1.0)

        start = time.perf_counter()
        model.fit(t, G_t, test_mode="relaxation")
        elapsed = time.perf_counter() - start

        assert (
            elapsed < NLSQ_TIME_LIMIT
        ), f"Zener relaxation fit took {elapsed:.2f}s (limit {NLSQ_TIME_LIMIT}s)"

    def test_cross_flow_curve_500pts(self):
        from rheojax.models.flow import Cross

        model = Cross()
        gd = np.logspace(-2, 2, 500)
        eta0, lam, n = 100.0, 1.0, 0.5
        eta = eta0 / (1.0 + (lam * gd) ** (1.0 - n))
        sigma = eta * gd

        start = time.perf_counter()
        model.fit(gd, sigma, test_mode="flow_curve")
        elapsed = time.perf_counter() - start

        assert (
            elapsed < NLSQ_TIME_LIMIT
        ), f"Cross flow_curve fit took {elapsed:.2f}s (limit {NLSQ_TIME_LIMIT}s)"


@pytest.mark.slow
class TestJITOverhead:
    """JIT compilation overhead must not exceed 10s."""

    def test_maxwell_first_call_jit(self):
        """First fit (includes JIT compilation) must finish within 10s."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t, G_t = _maxwell_data(100)

        start = time.perf_counter()
        model.fit(t, G_t, test_mode="relaxation")
        elapsed = time.perf_counter() - start

        assert (
            elapsed < JIT_TIME_LIMIT
        ), f"First Maxwell fit (JIT) took {elapsed:.2f}s (limit {JIT_TIME_LIMIT}s)"
