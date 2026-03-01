"""Public API for parallel execution.

High-level convenience functions that hide pool management.
These are the primary entry points for users and internal pipeline code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rheojax.parallel.config import get_default_workers, is_sequential_mode

logger = logging.getLogger(__name__)


def parallel_load(
    file_paths: list[str | Path],
    n_workers: int | None = None,
    **load_kwargs: Any,
) -> list:
    """Load multiple data files in parallel using threads.

    Uses ThreadPoolExecutor (I/O-bound, no JAX involved).
    Falls back to sequential when RHEOJAX_SEQUENTIAL=1.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to data files.
    n_workers : int, optional
        Number of I/O threads. Defaults to min(len(files), 8).
    **load_kwargs
        Keyword arguments forwarded to auto_load (e.g., x_col, y_col).

    Returns
    -------
    list of RheoData
        Loaded datasets, one per file, preserving input order.
    """
    if not file_paths:
        return []

    from rheojax.io import auto_load

    if is_sequential_mode():
        return [auto_load(fp, **load_kwargs) for fp in file_paths]

    n = n_workers or min(len(file_paths), 8)

    def _load_one(fp: str | Path):
        return auto_load(fp, **load_kwargs)

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(_load_one, fp) for fp in file_paths]
        return [f.result() for f in futures]


def parallel_map(
    fn: Callable,
    items: Iterable,
    n_workers: int | None = None,
    timeout: float = 300,
) -> Iterator:
    """Execute a function over items using process-based parallelism.

    Each invocation runs in a separate subprocess with its own JIT cache.
    Falls back to sequential when RHEOJAX_SEQUENTIAL=1.

    Parameters
    ----------
    fn : callable
        Must be a module-level function (picklable on spawn context).
    items : iterable
        Items to process.
    n_workers : int, optional
        Number of worker processes. Defaults to auto-detection.
    timeout : float
        Timeout per task in seconds.

    Yields
    ------
    Results in input order.
    """
    items_list = list(items)
    if not items_list:
        return

    if is_sequential_mode():
        for item in items_list:
            yield fn(item)
        return

    from rheojax.parallel.pool import PersistentProcessPool

    n = n_workers or get_default_workers()
    with PersistentProcessPool(n_workers=n) as pool:
        yield from pool.map(fn, items_list, timeout=timeout)
