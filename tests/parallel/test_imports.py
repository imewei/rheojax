"""Test public API imports."""

import pytest


@pytest.mark.smoke
class TestParallelImports:
    def test_config_functions(self):
        from rheojax.parallel import (  # noqa: F401
            configure,
            get_default_workers,
            get_parallel_config,
            get_worker_isolation,
            is_sequential_mode,
        )

        assert callable(configure)
        assert callable(get_parallel_config)
        assert callable(is_sequential_mode)
        assert callable(get_default_workers)
        assert callable(get_worker_isolation)

    def test_pool_class(self):
        from rheojax.parallel import PersistentProcessPool

        assert callable(PersistentProcessPool)

    def test_convenience_functions(self):
        from rheojax.parallel import parallel_load, parallel_map

        assert callable(parallel_load)
        assert callable(parallel_map)

    def test_all_exports(self):
        import rheojax.parallel

        for name in rheojax.parallel.__all__:
            obj = getattr(rheojax.parallel, name)
            assert obj is not None, f"{name} is None"
