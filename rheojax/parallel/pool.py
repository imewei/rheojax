"""Persistent process pool for JAX-safe parallel execution.

Long-lived subprocesses with pre-loaded JAX and model registry.
Tasks arrive via mp.Queue, results return via mp.Queue.
Uses 'spawn' context for macOS/Linux safety.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import traceback
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from rheojax.parallel.config import get_default_workers

logger = logging.getLogger(__name__)

# Unique sentinel for shutdown — string constant survives pickling across spawn
_SHUTDOWN_SENTINEL = "RHEOJAX_POOL_SHUTDOWN_v1"


def _warmup_jax(_unused=None):
    """Pre-import JAX and register models in a worker process.

    This triggers JIT compilation of common kernels so subsequent
    tasks start faster.
    """
    from rheojax.core.jax_config import safe_import_jax

    safe_import_jax()
    from rheojax.models import _ensure_all_registered

    _ensure_all_registered()
    return True


def _worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_id: int,
    shutdown_sentinel: object,
) -> None:
    """Worker main loop — runs in a subprocess.

    Receives (task_id, fn, args, kwargs) from task_queue.
    Sends (task_id, "ok", result) or (task_id, "error", error_str) to result_queue.
    Exits when it receives the shutdown sentinel.
    """
    while True:
        try:
            task = task_queue.get()
        except (EOFError, OSError):
            break

        if task == shutdown_sentinel:
            break

        task_id, fn, args, kwargs = task
        try:
            result = fn(*args, **kwargs)
            result_queue.put((task_id, "ok", result))
        except Exception as exc:
            tb = traceback.format_exc()
            # Truncate traceback to prevent unbounded payload size
            tb_truncated = tb[:4096] if len(tb) > 4096 else tb
            result_queue.put((task_id, "error", f"{exc}\n{tb_truncated}"))


class PoolFuture:
    """Future-like object for pool task results."""

    def __init__(self, task_id: int) -> None:
        self._task_id = task_id
        self._event = threading.Event()
        self._result: Any = None
        self._error: str | None = None

    @property
    def task_id(self) -> int:
        return self._task_id

    def _set_result(self, result: Any) -> None:
        self._result = result
        self._event.set()

    def _set_error(self, error: str) -> None:
        self._error = error
        self._event.set()

    def result(self, timeout: float | None = None) -> Any:
        """Block until result is available, then return it."""
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(
                f"Task {self._task_id} did not complete within {timeout}s"
            )
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._result

    def done(self) -> bool:
        return self._event.is_set()


class PersistentProcessPool:
    """Process pool with long-lived workers for JAX-safe parallel execution.

    Each worker runs in its own subprocess with independent JIT cache.
    Workers stay alive between tasks to avoid cold start overhead.
    """

    def __init__(
        self,
        n_workers: int | None = None,
        warm_pool: bool = False,
    ) -> None:
        self._n_workers = n_workers or get_default_workers()
        self._ctx = mp.get_context("spawn")
        # String sentinel survives pickling across spawn boundaries
        self._shutdown_sentinel = _SHUTDOWN_SENTINEL
        self._task_queue: mp.Queue = self._ctx.Queue()
        self._result_queue: mp.Queue = self._ctx.Queue()
        self._workers: list[mp.process.BaseProcess] = []
        self._futures: dict[int, PoolFuture] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        self._shutdown = False
        self._result_thread: threading.Thread | None = None

        # Spawn workers
        for i in range(self._n_workers):
            p = self._ctx.Process(
                target=_worker_loop,
                args=(self._task_queue, self._result_queue, i, self._shutdown_sentinel),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Start result collector thread
        self._result_thread = threading.Thread(
            target=self._collect_results, daemon=True
        )
        self._result_thread.start()

        logger.debug("PersistentProcessPool started with %d workers", self._n_workers)

        # Optionally warm up workers with JAX + model registry imports
        if warm_pool:
            warmup_futures = []
            for _ in range(self._n_workers):
                warmup_futures.append(self.submit(_warmup_jax))
            for f in warmup_futures:
                try:
                    f.result(timeout=120)
                except Exception as e:
                    logger.warning("Worker warmup failed: %s", e)
            logger.debug("Worker warmup completed")

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def is_alive(self) -> bool:
        """Check if any worker processes are still running."""
        return any(p.is_alive() for p in self._workers)

    def submit(
        self,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> PoolFuture:
        """Submit a task to the pool.

        The function must be module-level (picklable on spawn context).
        Returns a PoolFuture that can be awaited for the result.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Cannot submit to a pool that has been shut down")

            task_id = self._task_counter
            self._task_counter += 1
            future = PoolFuture(task_id)
            self._futures[task_id] = future

        self._task_queue.put((task_id, fn, args, kwargs))
        return future

    def map(
        self,
        fn: Callable,
        items: Iterable,
        timeout: float | None = None,
    ) -> Iterator:
        """Submit multiple tasks and yield results in order."""
        futures = [self.submit(fn, item) for item in items]
        for f in futures:
            yield f.result(timeout=timeout)

    def shutdown(self, timeout: float = 10) -> None:
        """Shutdown all workers gracefully."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

        # Send shutdown sentinels
        for _ in self._workers:
            try:
                self._task_queue.put(self._shutdown_sentinel)
            except (OSError, ValueError):
                pass

        # Join workers
        for p in self._workers:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        # Drain remaining results from queue before closing
        import queue as _queue

        while True:
            try:
                msg = self._result_queue.get_nowait()
                task_id, status, payload = msg
                future = self._futures.get(task_id)
                if future is not None:
                    if status == "ok":
                        future._set_result(payload)
                    else:
                        future._set_error(payload)
            except (_queue.Empty, EOFError, OSError):
                break

        # Poison any remaining pending futures so callers don't deadlock
        with self._lock:
            for future in self._futures.values():
                if not future.done():
                    future._set_error("Pool was shut down before task completed")
            self._futures.clear()

        # Clean up queues
        for q in (self._task_queue, self._result_queue):
            try:
                q.close()
                q.join_thread()
            except (OSError, ValueError):
                # OSError: queue pipe already closed
                # ValueError: queue already closed
                pass

        logger.debug("PersistentProcessPool shut down")

    def _collect_results(self) -> None:
        """Background thread that routes results to futures."""
        import queue as _queue

        while not self._shutdown:
            try:
                msg = self._result_queue.get(timeout=0.1)
            except (_queue.Empty, EOFError, OSError):
                # Empty: normal timeout, no results pending
                # EOFError/OSError: queue closed during shutdown
                continue

            task_id, status, payload = msg
            with self._lock:
                future = self._futures.pop(task_id, None)
            if future is None:
                logger.warning("Result for unknown task_id=%d", task_id)
                continue

            if status == "ok":
                future._set_result(payload)
            else:
                future._set_error(payload)

    def __enter__(self) -> PersistentProcessPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
