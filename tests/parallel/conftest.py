"""Shared fixtures for parallel module tests."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clean_parallel_config():
    """Reset parallel config state before and after each test.

    Ensures no test pollutes _overrides or env vars for subsequent tests.
    """
    from rheojax.parallel.config import configure

    configure()  # Reset to defaults

    # Also ensure env vars are clean
    env_keys = [
        "RHEOJAX_PARALLEL_WORKERS",
        "RHEOJAX_SEQUENTIAL",
        "RHEOJAX_WORKER_ISOLATION",
        "RHEOJAX_WARM_POOL",
    ]
    clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
    with patch.dict(os.environ, clean_env, clear=True):
        yield

    # Reset again after test
    configure()
