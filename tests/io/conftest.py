"""Shared fixtures for tests/io/."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "rheocompass"


@pytest.fixture(scope="session")
def rheocompass_fixtures():
    """Skip tests if rheocompass fixtures are not available."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Rheocompass fixtures not found at {FIXTURES_DIR}")
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def _io_test_gc():
    """Free memory after each io test to prevent xdist worker OOM.

    Several io test modules (test_analysis_exporter, test_trios_chunked_integrity)
    import JAX and model code at module level.  Under --dist=loadscope all 19 io
    test files may land in one worker; matplotlib figures and JAX caches accumulate
    without explicit cleanup.
    """
    yield
    gc.collect()
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass
    try:
        import jax

        jax.clear_caches()
    except ImportError:
        pass
