"""Tests for parallel configuration."""

import os

import pytest
from unittest.mock import patch


class TestParallelConfig:
    """Test adaptive parallelism configuration."""

    def test_default_workers_returns_positive_int(self):
        from rheojax.parallel.config import get_default_workers

        n = get_default_workers()
        assert isinstance(n, int)
        assert n >= 1

    def test_env_override_workers(self):
        from rheojax.parallel.config import get_default_workers

        with patch.dict(os.environ, {"RHEOJAX_PARALLEL_WORKERS": "8"}):
            assert get_default_workers() == 8

    def test_sequential_mode(self):
        from rheojax.parallel.config import is_sequential_mode

        assert not is_sequential_mode()
        with patch.dict(os.environ, {"RHEOJAX_SEQUENTIAL": "1"}):
            assert is_sequential_mode()

    def test_worker_isolation_mode_default(self):
        from rheojax.parallel.config import get_worker_isolation

        assert get_worker_isolation() == "subprocess"

    def test_worker_isolation_env_override(self):
        from rheojax.parallel.config import get_worker_isolation

        with patch.dict(os.environ, {"RHEOJAX_WORKER_ISOLATION": "thread"}):
            assert get_worker_isolation() == "thread"

    def test_get_parallel_config_returns_dict(self):
        from rheojax.parallel.config import get_parallel_config

        cfg = get_parallel_config()
        assert "n_workers" in cfg
        assert "isolation" in cfg
        assert "sequential" in cfg
        assert "warm_pool" in cfg

    def test_configure_overrides(self):
        from rheojax.parallel.config import configure, get_parallel_config

        configure(n_workers=6, warm_pool=True)
        cfg = get_parallel_config()
        assert cfg["n_workers"] == 6
        assert cfg["warm_pool"] is True
        # Reset
        configure(n_workers=None, warm_pool=False)

    def test_max_workers_capped_by_cpu(self):
        from rheojax.parallel.config import get_default_workers
        import multiprocessing

        n = get_default_workers()
        assert n <= multiprocessing.cpu_count()

    @pytest.mark.smoke
    def test_config_import(self):
        from rheojax.parallel.config import (  # noqa: F401
            get_default_workers,
            get_parallel_config,
            get_worker_isolation,
            is_sequential_mode,
            configure,
        )
