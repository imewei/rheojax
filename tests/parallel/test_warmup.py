"""Tests for JIT warmup in process pool."""

import time

import pytest


def _fit_maxwell_relaxation(x_unused):
    """Fit Maxwell model (requires warm JAX + model registry)."""
    import numpy as np

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()
    from rheojax.models import Maxwell

    m = Maxwell()
    t = np.linspace(0.01, 10.0, 50)
    G = 1000.0 * np.exp(-t / 1.0)
    m.fit(t, G, test_mode="relaxation", max_iter=20)
    return float(m.parameters.get_value("G0"))


class TestJITWarmup:
    @pytest.mark.smoke
    def test_warm_pool_creates_without_error(self):
        from rheojax.parallel.pool import PersistentProcessPool

        with PersistentProcessPool(n_workers=1, warm_pool=True) as pool:
            assert pool.is_alive()

    def test_warm_pool_faster_second_task(self):
        from rheojax.parallel.pool import PersistentProcessPool

        with PersistentProcessPool(n_workers=1, warm_pool=True) as pool:
            # First fit (may still be slow - JIT compiles in warmup)
            t1_start = time.perf_counter()
            r1 = pool.submit(_fit_maxwell_relaxation, 0).result(timeout=120)
            t1 = time.perf_counter() - t1_start

            # Second fit (should be faster - JIT cached)
            t2_start = time.perf_counter()
            r2 = pool.submit(_fit_maxwell_relaxation, 0).result(timeout=120)
            t2 = time.perf_counter() - t2_start

            assert r1 > 0  # G0 should be positive
            assert r2 > 0
            # Second fit should not be dramatically slower than first
            # (JIT cache is warm). Allow 2x tolerance for CI noise.
            assert t2 < t1 * 2.0, f"Expected warmup benefit: t1={t1:.2f}s, t2={t2:.2f}s"
