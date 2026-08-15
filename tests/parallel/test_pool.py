"""Tests for PersistentProcessPool."""

import os
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


def _die_hard(x):
    """Simulate a worker crash (segfault, OOM-kill) -- no Python exception,
    the process just disappears."""
    import os

    os._exit(1)  # noqa: SLF001


class TestPinWorkerGpu:
    """Unit tests for _pin_worker_gpu -- pure function, no subprocess needed."""

    def test_no_gpus_is_noop(self, monkeypatch):
        from rheojax.parallel.pool import _pin_worker_gpu

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _pin_worker_gpu(worker_id=0, gpu_count=0)
        assert "CUDA_VISIBLE_DEVICES" not in os.environ  # nosec B101

    def test_single_gpu_is_noop(self, monkeypatch):
        from rheojax.parallel.pool import _pin_worker_gpu

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _pin_worker_gpu(worker_id=0, gpu_count=1)
        assert "CUDA_VISIBLE_DEVICES" not in os.environ  # nosec B101

    def test_multi_gpu_pins_by_worker_id(self, monkeypatch):
        from rheojax.parallel.pool import _pin_worker_gpu

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _pin_worker_gpu(worker_id=3, gpu_count=2)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"  # nosec B101

    def test_preset_single_device_left_untouched(self, monkeypatch):
        """A pre-set single-device value (already exclusive) has nothing to
        partition -- must be left as-is, not overwritten from gpu_count."""
        from rheojax.parallel.pool import _pin_worker_gpu

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
        _pin_worker_gpu(worker_id=2, gpu_count=8)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"  # nosec B101

    def test_preset_multi_device_list_partitioned_by_worker_id(self, monkeypatch):
        """Regression test: a scheduler-set CUDA_VISIBLE_DEVICES (SLURM,
        Kubernetes GPU device plugin) is what the parent process sees and
        every spawned worker inherits unchanged -- if this were left
        untouched (the pre-fix behavior), every worker would still see and
        preallocate on the whole allocation, silently defeating pinning on
        exactly the managed-cluster hosts it targets. gpu_count is
        irrelevant here -- the preset list is authoritative."""
        from rheojax.parallel.pool import _pin_worker_gpu

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7")
        _pin_worker_gpu(worker_id=0, gpu_count=0)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"  # nosec B101

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7")
        _pin_worker_gpu(worker_id=1, gpu_count=0)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"  # nosec B101

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7")
        _pin_worker_gpu(worker_id=3, gpu_count=0)  # wraps: 3 % 3 == 0
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"  # nosec B101


class TestPersistentProcessPool:
    """Test persistent process pool lifecycle and task execution."""

    @pytest.mark.smoke
    def test_pool_creates_workers(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        assert pool.n_workers == 2  # nosec B101
        assert pool.is_alive()  # nosec B101
        pool.shutdown()

    def test_submit_and_get_result(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        try:
            future = pool.submit(_add_one, 41)
            result = future.result(timeout=10)
            assert result == 42  # nosec B101
        finally:
            pool.shutdown()

    def test_multiple_tasks_round_robin(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        try:
            futures = [pool.submit(_add_one, i) for i in range(10)]
            results = sorted(f.result(timeout=10) for f in futures)
            assert results == list(range(1, 11))  # nosec B101
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
            assert elapsed < 3.0, (  # nosec B101
                f"Expected parallel speedup, got {elapsed:.1f}s"
            )
            assert sorted(results) == [1, 2, 3, 4]  # nosec B101
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

    def test_submit_rejects_unpicklable_fn(self):
        """A lambda can't be pickled onto a spawn worker -- submit() must
        fail fast with TypeError instead of silently dropping the task and
        letting future.result() hang until timeout."""
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        try:
            with pytest.raises(TypeError, match="not picklable"):
                pool.submit(lambda x: x + 1, 1)
        finally:
            pool.shutdown()

    def test_del_without_shutdown_stops_collector_thread(self):
        """Dropping a pool without calling shutdown() must eventually stop
        its background collector thread. A bound-method Thread target
        (`target=self._collect_results`) keeps the pool alive via
        threading's internal registry for as long as the thread runs,
        which makes __del__ itself unreachable -- the weakref-based
        collector this guards against regressing to exists specifically
        to break that cycle."""
        import gc

        from rheojax.parallel.pool import PersistentProcessPool

        baseline = threading.active_count()
        pool = PersistentProcessPool(n_workers=1)
        if threading.active_count() != baseline + 1:
            raise AssertionError(
                f"Expected {baseline + 1} active threads, got {threading.active_count()}"
            )

        del pool
        gc.collect()

        deadline = time.time() + 5.0
        while time.time() < deadline and threading.active_count() != baseline:
            time.sleep(0.1)

        if threading.active_count() != baseline:
            raise AssertionError(
                "collector thread leaked after the pool was garbage-collected "
                "without an explicit shutdown()"
            )

    def test_shutdown_terminates_workers(self):
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        pool.shutdown(timeout=5)
        assert not pool.is_alive()  # nosec B101

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
            assert future.result(timeout=10) == 2  # nosec B101
            assert observed_locked == [True], (  # nosec B101
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

        assert not t.is_alive(), "submit loop thread failed to stop"  # nosec B101
        assert unexpected == [], (  # nosec B101
            f"Unexpected exception type(s) escaped submit() during concurrent "
            f"shutdown: {unexpected!r}"
        )

    def test_context_manager(self):
        from rheojax.parallel.pool import PersistentProcessPool

        with PersistentProcessPool(n_workers=1) as pool:
            result = pool.submit(_add_one, 99).result(timeout=10)
            assert result == 100  # nosec B101
        assert not pool.is_alive()  # nosec B101

    def test_map_convenience(self):
        from rheojax.parallel.pool import PersistentProcessPool

        with PersistentProcessPool(n_workers=2) as pool:
            results = list(pool.map(_add_one, range(5), timeout=10))
            assert sorted(results) == [1, 2, 3, 4, 5]  # nosec B101

    def test_worker_crash_poisons_future_and_respawns(self):
        """A worker that dies outright (segfault/OOM-kill, no Python
        exception) must not leave its future hanging under the default
        timeout=None, and the pool must recover to serve subsequent
        submissions rather than silently running below capacity forever."""
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=2)
        try:
            future = pool.submit(_die_hard, 1)
            with pytest.raises(RuntimeError, match="crashed"):
                future.result(timeout=15)

            # The dead slot must have been respawned -- confirm the pool
            # still completes tasks at full capacity, not just via the
            # one worker that never crashed.
            deadline = time.time() + 10.0
            last_exc: Exception | None = None
            while time.time() < deadline:
                try:
                    result = pool.submit(_add_one, 41).result(timeout=10)
                    assert result == 42  # nosec B101
                    break
                except Exception as e:  # replacement worker still starting
                    last_exc = e
                    time.sleep(0.2)
            else:
                raise AssertionError(f"pool never recovered after crash: {last_exc}")
        finally:
            pool.shutdown()

    def test_check_worker_health_pops_poisoned_futures(self):
        """A poisoned future must be removed from pool._futures, not just
        marked done -- otherwise a late 'ok' result for the same task_id
        (delivered by a surviving worker) would call _set_result() on top
        of the already-set _error, and PoolFuture.result() checks _error
        first, permanently masking a real success behind a stale crash
        message."""
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        try:
            future = pool.submit(_slow_add, 0)
            tid = future.task_id
            assert tid in pool._futures  # nosec B101

            pool._workers[0].kill()
            pool._workers[0].join(timeout=5)
            pool._check_worker_health()

            with pytest.raises(RuntimeError, match="crashed"):
                future.result(timeout=5)
            assert tid not in pool._futures, (  # nosec B101
                "poisoned future must be popped from pool._futures"
            )
        finally:
            pool.shutdown()

    def test_check_worker_health_survives_respawn_failure(self):
        """If respawning a crashed worker's slot itself raises (e.g. OSError
        from resource exhaustion, plausible right after an OOM-kill), the
        collector thread that calls _check_worker_health() must not die --
        it is the sole consumer of result_queue, so losing it would
        silently hang every future .result() call for the rest of the
        pool's life."""
        from rheojax.parallel.pool import PersistentProcessPool

        pool = PersistentProcessPool(n_workers=1)
        try:
            orig_spawn = pool._spawn_worker
            call_count = {"n": 0}

            def flaky_spawn(worker_id):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise OSError("simulated resource exhaustion")
                return orig_spawn(worker_id)

            pool._spawn_worker = flaky_spawn

            pool._workers[0].kill()
            pool._workers[0].join(timeout=5)

            # First health-check tick (inside _collect_results) hits the
            # simulated failure; the collector thread must survive it and
            # succeed on a later tick instead of dying silently.
            deadline = time.time() + 10.0
            while time.time() < deadline and pool._result_thread.is_alive():
                if call_count["n"] >= 2:
                    break
                time.sleep(0.2)

            assert pool._result_thread.is_alive(), (  # nosec B101
                "collector thread must not die from a failed respawn"
            )
            assert call_count["n"] >= 2, "respawn must be retried"  # nosec B101

            # Pool must still be usable after recovering.
            result = pool.submit(_add_one, 41).result(timeout=10)
            assert result == 42  # nosec B101
        finally:
            pool._spawn_worker = orig_spawn
            pool.shutdown()

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
