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
from typing import Any, Callable, Iterable, Iterator

from rheojax.parallel.config import get_default_workers

logger = logging.getLogger(__name__)

# Sentinel for shutdown
_SHUTDOWN = None


def _worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_id: int,
) -> None:
    """Worker main loop — runs in a subprocess.

    Receives (task_id, fn, args, kwargs) from task_queue.
    Sends (task_id, "ok", result) or (task_id, "error", error_str) to result_queue.
    Exits when it receives None sentinel.
    """
    while True:
        try:
            task = task_queue.get()
        except (EOFError, OSError):
            break

        if task is _SHUTDOWN:
            break

        task_id, fn, args, kwargs = task
        try:
            result = fn(*args, **kwargs)
            result_queue.put((task_id, "ok", result))
        except Exception as exc:
            tb = traceback.format_exc()
            result_queue.put((task_id, "error", f"{exc}\n{tb}"))


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
        self._task_queue: mp.Queue = self._ctx.Queue()
        self._result_queue: mp.Queue = self._ctx.Queue()
        self._workers: list[mp.Process] = []
        self._futures: dict[int, PoolFuture] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        self._shutdown = False
        self._result_thread: threading.Thread | None = None

        # Spawn workers
        for i in range(self._n_workers):
            p = self._ctx.Process(
                target=_worker_loop,
                args=(self._task_queue, self._result_queue, i),
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
        if self._shutdown:
            raise RuntimeError("Cannot submit to a pool that has been shut down")

        with self._lock:
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
        if self._shutdown:
            return
        self._shutdown = True

        # Send shutdown sentinels
        for _ in self._workers:
            try:
                self._task_queue.put(_SHUTDOWN)
            except (OSError, ValueError):
                pass

        # Join workers
        for p in self._workers:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        # Clean up queues
        for q in (self._task_queue, self._result_queue):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        logger.debug("PersistentProcessPool shut down")

    def _collect_results(self) -> None:
        """Background thread that routes results to futures."""
        while not self._shutdown:
            try:
                msg = self._result_queue.get(timeout=0.1)
            except Exception:
                continue

            task_id, status, payload = msg
            future = self._futures.get(task_id)
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
