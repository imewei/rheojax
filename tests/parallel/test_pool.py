"""Tests for PersistentProcessPool."""

import threading
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

    @pytest.mark.slow
    def test_parallel_speedup(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        try:
            # First submit a warmup task to absorb spawn overhead
            pool.submit(_add_one, 0).result(timeout=30)

            start = time.perf_counter()
            futures = [pool.submit(_slow_add, i) for i in range(4)]
            results = [f.result(timeout=30) for f in futures]
            elapsed = time.perf_counter() - start
            # 4 tasks x 0.5s each, 2 workers -> ~1.0s, not 2.0s
            # Allow generous margin for CI environments
            assert elapsed < 3.0, f"Expected parallel speedup, got {elapsed:.1f}s"
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

    def test_submit_holds_lock_through_queue_put(self):
        """PARALLEL-004 regression test (deterministic).

        submit() must hold self._lock for the entire span from the
        shutdown-check to task_queue.put() -- that's what makes the
        check-then-act atomic and closes the TOCTOU window where a
        concurrent shutdown() could close the queue between the check and
        the put(). Verify this directly by wrapping put() and asserting the
        lock is held at the moment it's called, instead of relying on a
        timing-based race to happen to trigger.
        """
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        try:
            observed_locked = []
            orig_put = pool._task_queue.put

            def wrapped_put(*args, **kwargs):
                observed_locked.append(pool._lock.locked())
                return orig_put(*args, **kwargs)

            pool._task_queue.put = wrapped_put

            future = pool.submit(_add_one, 1)
            assert future.result(timeout=10) == 2
            assert observed_locked == [True], (
                "submit() must call task_queue.put() while holding self._lock"
            )
        finally:
            pool._task_queue.put = orig_put
            pool.shutdown()

    def test_concurrent_submit_during_shutdown_no_race_exception(self):
        """PARALLEL-004 regression test (timing-sensitive, probabilistic).

        Before the fix, submit()'s shutdown-check and task_queue.put() were
        not atomic: a submit() from one thread could pass the shutdown
        check, then shutdown() (running concurrently on another thread)
        could close the queue, then submit()'s put() would raise a raw
        queue-related exception (e.g. ValueError/OSError from a closed
        pipe) instead of the documented RuntimeError. Hammer submit() from
        a background thread while shutdown() runs concurrently on the main
        thread, and confirm the only exception type that ever escapes
        submit() is the documented post-shutdown RuntimeError.
        """
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        unexpected: list[BaseException] = []
        stop = threading.Event()

        def submit_loop():
            i = 0
            while not stop.is_set():
                try:
                    pool.submit(_add_one, i)
                except RuntimeError as e:
                    if "shut down" in str(e):
                        stop.set()
                    else:
                        unexpected.append(e)
                        stop.set()
                except Exception as e:  # noqa: BLE001 - intentionally broad
                    unexpected.append(e)
                    stop.set()
                i += 1

        t = threading.Thread(target=submit_loop)
        t.start()
        time.sleep(0.01)
        pool.shutdown()
        stop.set()
        t.join(timeout=10)

        assert not t.is_alive(), "submit loop thread failed to stop"
        assert unexpected == [], (
            f"Unexpected exception type(s) escaped submit() during concurrent "
            f"shutdown: {unexpected!r}"
        )

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

    def test_shutdown_poisons_pending_futures(self):
        """Futures submitted but not completed before shutdown get an error."""
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        # Submit a slow task, then shut down before it completes
        future = pool.submit(_slow_add, 0)
        pool.shutdown(timeout=0.1)  # Very short timeout forces termination
        # Future should resolve (either with result or error), not deadlock
        with pytest.raises((RuntimeError, TimeoutError)):
            future.result(timeout=5)
