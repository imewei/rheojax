"""Tests for PersistentProcessPool."""

import time

import pytest


# Module-level work functions (must be picklable on macOS spawn context)
def _add_one(x):
    return x + 1


def _slow_add(x):
    import time

    time.sleep(0.5)
    return x + 1


def _raise_error(x):
    raise ValueError(f"intentional error: {x}")


class TestPersistentProcessPool:
    """Test persistent process pool lifecycle and task execution."""

    @pytest.mark.smoke
    def test_pool_creates_workers(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        assert pool.n_workers == 2
        assert pool.is_alive()
        pool.shutdown()

    def test_submit_and_get_result(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        try:
            future = pool.submit(_add_one, 41)
            result = future.result(timeout=10)
            assert result == 42
        finally:
            pool.shutdown()

    def test_multiple_tasks_round_robin(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        try:
            futures = [pool.submit(_add_one, i) for i in range(10)]
            results = sorted(f.result(timeout=10) for f in futures)
            assert results == list(range(1, 11))
        finally:
            pool.shutdown()

    def test_parallel_speedup(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        try:
            start = time.perf_counter()
            futures = [pool.submit(_slow_add, i) for i in range(4)]
            results = [f.result(timeout=10) for f in futures]
            elapsed = time.perf_counter() - start
            # 4 tasks x 0.5s each, 2 workers -> ~1.0s, not 2.0s
            assert elapsed < 1.8, f"Expected parallel speedup, got {elapsed:.1f}s"
            assert sorted(results) == [1, 2, 3, 4]
        finally:
            pool.shutdown()

    def test_worker_error_propagates(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        try:
            future = pool.submit(_raise_error, "test")
            with pytest.raises(Exception, match="intentional error"):
                future.result(timeout=10)
        finally:
            pool.shutdown()

    def test_shutdown_terminates_workers(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        pool.shutdown(timeout=5)
        assert not pool.is_alive()

    def test_pool_rejects_after_shutdown(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        pool.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            pool.submit(_add_one, 1)

    def test_context_manager(self):
        from rheojax.parallel.pool import PersistentProcessPool

        with PersistentProcessPool(n_workers=1) as pool:
            result = pool.submit(_add_one, 99).result(timeout=10)
            assert result == 100
        assert not pool.is_alive()

    def test_map_convenience(self):
        from rheojax.parallel.pool import PersistentProcessPool

        with PersistentProcessPool(n_workers=2) as pool:
            results = list(pool.map(_add_one, range(5), timeout=10))
            assert sorted(results) == [1, 2, 3, 4, 5]
