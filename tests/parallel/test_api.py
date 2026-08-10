"""Tests for parallel public API."""

from unittest.mock import patch

import pytest


# Module-level function for pickling
def _test_add_one(x):
    return x + 1


class TestParallelLoad:
    """Test parallel_load() for multi-file I/O."""

    @pytest.mark.smoke
    def test_parallel_load_empty_list(self):
        from rheojax.parallel.api import parallel_load

        results = parallel_load([])
        assert results == []  # nosec B101

    def test_parallel_load_returns_list_of_rheodata(self, tmp_path):
        from rheojax.parallel.api import parallel_load

        # Create minimal CSV files
        for i in range(3):
            f = tmp_path / f"data_{i}.csv"
            f.write_text("time,stress\n0.1,100\n1.0,50\n10.0,10\n")
        files = sorted(tmp_path.glob("*.csv"))
        results = parallel_load(files, x_col="time", y_col="stress")
        assert len(results) == 3  # nosec B101
        for r in results:
            assert hasattr(r, "x")  # nosec B101
            assert hasattr(r, "y")  # nosec B101

    def test_parallel_load_sequential_fallback(self, tmp_path):
        from rheojax.parallel.api import parallel_load

        for i in range(2):
            f = tmp_path / f"data_{i}.csv"
            f.write_text("time,stress\n0.1,100\n1.0,50\n")
        files = sorted(tmp_path.glob("*.csv"))
        with patch.dict("os.environ", {"RHEOJAX_SEQUENTIAL": "1"}):
            results = parallel_load(files, x_col="time", y_col="stress")
        assert len(results) == 2  # nosec B101


class TestParallelMap:
    """Test parallel_map() generic fan-out."""

    @pytest.mark.smoke
    def test_parallel_map_sequential_fallback(self):
        from rheojax.parallel.api import parallel_map

        with patch.dict("os.environ", {"RHEOJAX_SEQUENTIAL": "1"}):
            results = list(parallel_map(_test_add_one, [1, 2, 3]))
            assert sorted(results) == [2, 3, 4]  # nosec B101

    def test_parallel_map_with_workers(self):
        from rheojax.parallel.api import parallel_map

        results = list(parallel_map(_test_add_one, range(10), n_workers=2))
        assert sorted(results) == list(range(1, 11))  # nosec B101

    def test_parallel_map_empty(self):
        from rheojax.parallel.api import parallel_map

        results = list(parallel_map(_test_add_one, []))
        assert results == []  # nosec B101

    def test_parallel_map_thread_isolation(self):
        """RHEOJAX_WORKER_ISOLATION=thread must route through a
        ThreadPoolExecutor, not subprocesses. A closure over a local
        variable is unpicklable (subprocess submit() would reject it) but
        works fine in threads -- proving which path actually ran.

        is_sequential_mode is forced False: under pytest-xdist,
        PYTEST_XDIST_WORKER makes it auto-True, which would route through
        the plain sequential fallback and pass vacuously without ever
        exercising the thread-isolation branch this test targets.
        """
        from rheojax.parallel.api import parallel_map

        factor = 3

        def _mul_by_factor(x):
            return x * factor

        with (
            patch.dict("os.environ", {"RHEOJAX_WORKER_ISOLATION": "thread"}),
            patch("rheojax.parallel.api.is_sequential_mode", return_value=False),
        ):
            results = list(parallel_map(_mul_by_factor, [1, 2, 3]))
        assert sorted(results) == [3, 6, 9]  # nosec B101

    def test_parallel_map_forwards_warm_pool_config(self):
        """configure(warm_pool=True) / RHEOJAX_WARM_POOL=1 must actually
        reach PersistentProcessPool -- previously parallel_map() ignored
        get_parallel_config()["warm_pool"] entirely.

        is_sequential_mode is forced False for the same xdist reason as
        test_parallel_map_thread_isolation -- otherwise this test would
        KeyError on captured["warm_pool"] instead of testing anything.
        """
        from rheojax.parallel import configure
        from rheojax.parallel.api import parallel_map

        captured = {}

        class _FakePool:
            def __init__(self, n_workers=None, warm_pool=False):
                captured["warm_pool"] = warm_pool

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, fn, items, timeout=None):
                return [fn(i) for i in items]

        try:
            configure(warm_pool=True)
            with (
                patch("rheojax.parallel.pool.PersistentProcessPool", _FakePool),
                patch("rheojax.parallel.api.is_sequential_mode", return_value=False),
            ):
                list(parallel_map(_test_add_one, [1, 2]))
            assert captured["warm_pool"] is True  # nosec B101
        finally:
            configure()

    def test_parallel_map_exception_type_consistent_across_isolation_modes(self):
        """A task failure must raise the same exception type regardless of
        which isolation mode is active -- otherwise `except SomeError:`
        around parallel_map() works under one env var setting and silently
        stops catching under another. PersistentProcessPool.PoolFuture
        always wraps failures as RuntimeError (pool.py); the thread branch
        must match that contract rather than leaking the raw exception
        type a plain concurrent.futures.Future would otherwise preserve.
        """
        from rheojax.parallel.api import parallel_map

        def _boom(x):
            raise ValueError(f"boom {x}")

        with (
            patch.dict("os.environ", {"RHEOJAX_WORKER_ISOLATION": "thread"}),
            patch("rheojax.parallel.api.is_sequential_mode", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                list(parallel_map(_boom, [1]))
