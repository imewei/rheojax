"""Parallel execution layer for RheoJAX.

Provides process-based parallelism for model fitting, Bayesian inference,
and transforms. Thread parallelism for I/O only.

Configuration:
    RHEOJAX_PARALLEL_WORKERS=N  -- worker count (default: auto)
    RHEOJAX_SEQUENTIAL=1        -- disable all parallelism
    RHEOJAX_WORKER_ISOLATION=subprocess|thread
    RHEOJAX_WARM_POOL=1         -- pre-initialize workers

Example:
    >>> from rheojax.parallel import configure, parallel_load, parallel_map
    >>>
    >>> # Load files in parallel threads
    >>> datasets = parallel_load(['data1.csv', 'data2.csv'], x_col='time', y_col='stress')
    >>>
    >>> # Process items in parallel subprocesses
    >>> results = list(parallel_map(fit_func, items, n_workers=4))
    >>>
    >>> # Configure at startup
    >>> configure(n_workers=4, warm_pool=True)
"""

from rheojax.parallel.api import parallel_load, parallel_map
from rheojax.parallel.config import (
    configure,
    get_default_workers,
    get_parallel_config,
    get_worker_isolation,
    is_sequential_mode,
)

__all__ = [
    "PersistentProcessPool",
    "configure",
    "get_default_workers",
    "get_parallel_config",
    "get_worker_isolation",
    "is_sequential_mode",
    "parallel_load",
    "parallel_map",
]


def __getattr__(name: str):
    """Lazy import for PersistentProcessPool (avoids multiprocessing import at package level)."""
    if name == "PersistentProcessPool":
        from rheojax.parallel.pool import PersistentProcessPool

        return PersistentProcessPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
