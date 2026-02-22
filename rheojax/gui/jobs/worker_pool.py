"""
Worker Pool
==========

Thread pool for executing background jobs with progress tracking using PySide6.
"""

import time
import uuid
from collections.abc import Callable
from threading import Lock
from typing import Any

try:
    from rheojax.gui.compat import QObject, QRunnable, Qt, QThreadPool, Signal, Slot

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

    # Provide stub classes for type checking when PySide6 not available
    class QObject:  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Signal:  # type: ignore
        def __init__(self, *args, **kwargs):
            self._slots = []

        def connect(self, slot):
            self._slots.append(slot)

        def emit(self, *args, **kwargs):
            for slot in list(self._slots):
                try:
                    slot(*args, **kwargs)
                except Exception:
                    # Best-effort stub; ignore downstream errors
                    continue

    class QRunnable:  # type: ignore
        def run(self):  # pragma: no cover - stub
            return None

    class QThreadPool:  # type: ignore
        @staticmethod
        def globalInstance():
            return QThreadPool()

        def start(self, runnable):
            # Run immediately in current thread as fallback
            if hasattr(runnable, "run"):
                runnable.run()

    def Slot(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn

        return decorator

    class Qt:  # type: ignore
        QueuedConnection = None


from rheojax.gui.jobs.cancellation import CancellationToken
from rheojax.logging import get_logger

logger = get_logger(__name__)


class WorkerPool(QObject):
    """Manages background workers using QThreadPool.

    This is a singleton class - all pages share the same WorkerPool instance
    to avoid resource duplication and ensure consistent job tracking.

    Features:
        - PySide6 QThreadPool-based execution
        - Job tracking with unique IDs
        - Progress and status signals
        - Cancellation support
        - Error handling and recovery
        - Automatic cleanup
        - Singleton pattern for resource efficiency

    Signals:
        job_started(str): Emitted when a job starts (job_id)
        job_progress(str, int, int, str): Progress update (job_id, current, total, message)
        job_completed(str, object): Job finished successfully (job_id, result)
        job_failed(str, str): Job failed with error (job_id, error_message)
        job_cancelled(str): Job was cancelled (job_id)

    Example
    -------
    >>> pool = WorkerPool.instance()  # doctest: +SKIP
    >>> worker = FitWorker(...)  # doctest: +SKIP
    >>> job_id = pool.submit(worker)  # doctest: +SKIP
    >>> pool.cancel(job_id)  # doctest: +SKIP
    """

    # Singleton instance
    _instance: "WorkerPool | None" = None
    _initialized: bool = False

    # Signals
    job_started = Signal(str)  # job_id
    job_progress = Signal(str, int, int, str)  # job_id, current, total, message
    job_completed = Signal(str, object)  # job_id, result
    job_failed = Signal(str, str)  # job_id, error_message
    job_cancelled = Signal(str)  # job_id

    _singleton_lock = Lock()

    def __new__(cls, max_threads: int = 4) -> "WorkerPool":
        """Create or return singleton instance."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls, max_threads: int = 4) -> "WorkerPool":
        """Get the singleton WorkerPool instance.

        Parameters
        ----------
        max_threads : int, default=4
            Maximum number of concurrent worker threads (only used on first call)

        Returns
        -------
        WorkerPool
            The singleton WorkerPool instance
        """
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(max_threads)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        logger.debug("Resetting worker pool singleton")
        if cls._instance is not None:
            cls._instance.shutdown(wait=False)
        cls._instance = None
        cls._initialized = False
        logger.debug("Worker pool singleton reset complete")

    def __init__(self, max_threads: int = 4):
        """Initialize worker pool.

        Parameters
        ----------
        max_threads : int, default=4
            Maximum number of concurrent worker threads
        """
        # Skip re-initialization for singleton
        if WorkerPool._initialized:
            return

        if not HAS_PYSIDE6:
            logger.error("PySide6 not available, cannot initialize WorkerPool")
            raise ImportError(
                "PySide6 is required for WorkerPool. "
                "Install with: pip install PySide6"
            )

        super().__init__()
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(max_threads)
        self._active_jobs: dict[str, CancellationToken] = {}
        self._job_signals: dict[str, QObject] = {}
        self._job_start_times: dict[str, float] = {}
        self._job_lock = Lock()

        WorkerPool._initialized = True
        logger.info("Worker pool initialized", max_workers=max_threads)

    def submit(
        self, worker: QRunnable, on_job_registered: Callable[[str], None] | None = None
    ) -> str:
        """Submit a worker to the pool.

        Parameters
        ----------
        worker : QRunnable
            Worker to execute (e.g., FitWorker, BayesianWorker)

        Returns
        -------
        str
            Unique job ID for tracking

        Notes
        -----
        The worker must have a `cancel_token` attribute of type
        CancellationToken and a `signals` attribute with appropriate
        signals (completed, failed, cancelled).
        """
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Get cancellation token from worker
        if not hasattr(worker, "cancel_token"):
            logger.warning(
                "Worker missing cancel_token attribute, creating default",
                job_id=job_id,
                worker_type=type(worker).__name__,
            )
            worker.cancel_token = CancellationToken()  # type: ignore[attr-defined]

        # Register job
        with self._job_lock:
            self._active_jobs[job_id] = worker.cancel_token  # type: ignore[attr-defined]
            self._job_start_times[job_id] = time.monotonic()
            active_count = len(self._active_jobs)

        logger.debug(
            "Worker utilization",
            active_jobs=active_count,
            max_threads=self._pool.maxThreadCount(),
        )

        if on_job_registered is not None:
            try:
                on_job_registered(job_id)
            except Exception as exc:
                logger.warning(
                    "on_job_registered hook failed",
                    job_id=job_id,
                    error=str(exc),
                )

        # Connect worker signals to pool signals.
        #
        # Workers emit from background threads; always use queued connections into
        # WorkerPool's thread (GUI thread). Avoid connecting worker signals directly
        # to ad-hoc Python callables (e.g. lambdas), which can lead to shiboken
        # EXC_BAD_ACCESS crashes when posted events are delivered.
        if hasattr(worker, "signals"):
            signals = worker.signals
            # Preserve job_id on the worker for debugging.
            try:
                worker.job_id = job_id  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("Could not set job_id on worker", error=str(exc))
            if isinstance(signals, QObject):
                self._job_signals[job_id] = signals
                # Tag signals object with job_id for direct lookup
                # when sender() works but dict reverse-lookup is ambiguous.
                signals._pool_job_id = job_id  # type: ignore[attr-defined]

            conn_type = Qt.QueuedConnection if HAS_PYSIDE6 else None
            if conn_type is None:
                logger.warning(
                    "PySide6 not available — worker signals connected without "
                    "QueuedConnection; cross-thread safety not guaranteed",
                )

            if hasattr(signals, "completed"):
                if conn_type is not None:
                    signals.completed.connect(self._on_worker_completed, conn_type)
                else:
                    signals.completed.connect(self._on_worker_completed)
            if hasattr(signals, "failed"):
                if conn_type is not None:
                    signals.failed.connect(self._on_worker_failed, conn_type)
                else:
                    signals.failed.connect(self._on_worker_failed)
            if hasattr(signals, "cancelled"):
                if conn_type is not None:
                    signals.cancelled.connect(self._on_worker_cancelled, conn_type)
                else:
                    signals.cancelled.connect(self._on_worker_cancelled)
            if hasattr(signals, "progress"):
                if conn_type is not None:
                    signals.progress.connect(self._on_worker_progress, conn_type)
                else:
                    signals.progress.connect(self._on_worker_progress)

        # Submit to thread pool
        self._pool.start(worker)
        self.job_started.emit(job_id)
        logger.debug("Job submitted", job_id=job_id)

        return job_id

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a job.

        Parameters
        ----------
        job_id : str
            Job ID to cancel

        Returns
        -------
        bool
            True if job was found and cancellation requested

        Notes
        -----
        This only requests cancellation - the job must check
        its cancellation token periodically to actually stop.
        """
        with self._job_lock:
            if job_id in self._active_jobs:
                token = self._active_jobs[job_id]
                token.cancel()
                logger.info("Cancellation requested for job", job_id=job_id)
                return True

        logger.warning("Job not found for cancellation", job_id=job_id)
        return False

    def cancel_all(self) -> None:
        """Cancel all active jobs.

        Requests cancellation for every currently running job.
        """
        with self._job_lock:
            job_ids = list(self._active_jobs.keys())

        for job_id in job_ids:
            self.cancel(job_id)

        logger.info("Cancellation requested for all jobs", job_count=len(job_ids))

    def is_busy(self) -> bool:
        """Check if any jobs are running.

        Returns
        -------
        bool
            True if any jobs are active
        """
        return self.get_active_count() > 0

    def get_active_count(self) -> int:
        """Get number of active jobs.

        Returns
        -------
        int
            Number of currently running jobs
        """
        with self._job_lock:
            return len(self._active_jobs)

    def shutdown(self, wait: bool = True, timeout_ms: int = 30000) -> None:
        """Shutdown the worker pool.

        Parameters
        ----------
        wait : bool, default=True
            Wait for running jobs to complete
        timeout_ms : int, default=30000
            Maximum wait time in milliseconds

        Notes
        -----
        If wait=True, this will block until all jobs complete
        or the timeout expires. Active jobs will be cancelled
        before waiting.
        """
        with self._job_lock:
            active_count = len(self._active_jobs)

        logger.info(
            "Shutting down worker pool",
            active_jobs=active_count,
            wait=wait,
            timeout_ms=timeout_ms,
        )

        # Cancel all active jobs
        self.cancel_all()

        # Wait for completion if requested
        if wait:
            success = self._pool.waitForDone(timeout_ms)
            if not success:
                logger.warning(
                    "Worker pool shutdown timed out",
                    timeout_ms=timeout_ms,
                )

        # Clear active jobs
        with self._job_lock:
            self._active_jobs.clear()
            self._job_signals.clear()
            self._job_start_times.clear()

        logger.info("Worker pool shut down")

    def _get_job_elapsed(self, job_id: str) -> float | None:
        """Get elapsed time for a job in seconds."""
        with self._job_lock:
            start_time = self._job_start_times.get(job_id)
        if start_time is not None:
            return time.monotonic() - start_time
        return None

    @Slot(str, object)
    def _on_job_completed(self, job_id: str, result: Any) -> None:
        """Handle job completion."""
        elapsed = self._get_job_elapsed(job_id)
        self._cleanup_job(job_id)
        self.job_completed.emit(job_id, result)
        logger.debug("Job complete", job_id=job_id, elapsed=elapsed)

    @Slot(str, str)
    def _on_job_failed(self, job_id: str, error_message: str) -> None:
        """Handle job failure."""
        elapsed = self._get_job_elapsed(job_id)
        self._cleanup_job(job_id)
        self.job_failed.emit(job_id, error_message)
        logger.error(
            "Job failed",
            job_id=job_id,
            error=error_message,
            elapsed=elapsed,
            exc_info=True,
        )

    @Slot(str)
    def _on_job_cancelled(self, job_id: str) -> None:
        """Handle job cancellation."""
        elapsed = self._get_job_elapsed(job_id)
        self._cleanup_job(job_id)
        self.job_cancelled.emit(job_id)
        logger.info("Job cancelled", job_id=job_id, elapsed=elapsed)

    @Slot(str, int, int, str)
    def _on_job_progress(
        self, job_id: str, current: int, total: int, message: str
    ) -> None:
        """Handle job progress update."""
        self.job_progress.emit(job_id, current, total, message)

    def _cleanup_job(self, job_id: str) -> None:
        """Remove job from active tracking."""
        with self._job_lock:
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
            self._job_signals.pop(job_id, None)
            self._job_start_times.pop(job_id, None)
            remaining_jobs = len(self._active_jobs)

        logger.debug(
            "Job cleaned up",
            job_id=job_id,
            remaining_jobs=remaining_jobs,
        )

    def _job_id_from_sender(self) -> str | None:
        """Best-effort reverse lookup for job_id based on Qt sender().

        Uses a two-tier strategy:
        1. Direct attribute lookup (O(1)) via ``_pool_job_id`` tag set in submit()
        2. Dict reverse-lookup (O(n)) scanning ``_job_signals``

        This resolves the race condition where ``_job_id_from_result_fallback``
        returned None with 2+ concurrent jobs.
        """
        try:
            sender = self.sender()
        except Exception:
            sender = None

        if sender is None:
            return None

        # Fast path: direct attribute lookup (set in submit)
        pool_job_id = getattr(sender, "_pool_job_id", None)
        if pool_job_id is not None:
            with self._job_lock:
                if pool_job_id in self._active_jobs:
                    return pool_job_id

        # Slow path: dict reverse-lookup
        for job_id, signals in list(self._job_signals.items()):
            if signals is sender:
                return job_id
        return None

    @Slot(object)
    def _on_worker_completed(self, result: Any) -> None:
        job_id = self._job_id_from_sender()
        if job_id:
            self._on_job_completed(job_id, result)
        else:
            # Fallback: try to find job_id from worker attribute
            job_id = self._job_id_from_result_fallback()
            if job_id:
                self._on_job_completed(job_id, result)
            else:
                logger.error(
                    "Worker completed but job_id lookup failed (sender=%s). "
                    "Emitting with synthetic job_id to avoid silent loss.",
                    self.sender(),
                )
                # Emit with empty job_id so downstream handlers can still
                # process the result rather than silently dropping it.
                self.job_completed.emit("", result)

    @Slot(str)
    def _on_worker_failed(self, error_message: str) -> None:
        job_id = self._job_id_from_sender()
        if job_id:
            self._on_job_failed(job_id, error_message)
        else:
            job_id = self._job_id_from_result_fallback()
            if job_id:
                self._on_job_failed(job_id, error_message)
            else:
                logger.error(
                    "Worker failed but job_id lookup failed (sender=%s). Error: %s",
                    self.sender(),
                    error_message,
                )
                self.job_failed.emit("", error_message)

    @Slot()
    def _on_worker_cancelled(self) -> None:
        job_id = self._job_id_from_sender()
        if job_id:
            self._on_job_cancelled(job_id)
        else:
            job_id = self._job_id_from_result_fallback()
            if job_id:
                self._on_job_cancelled(job_id)

    def _job_id_from_result_fallback(self) -> str | None:
        """Fallback job_id lookup when sender() fails.

        Checks if there is exactly one active job, and returns its id.
        This handles the common case of a single concurrent job.
        """
        with self._job_lock:
            if len(self._active_jobs) == 1:
                return next(iter(self._active_jobs))
        return None

    @Slot(int, int, str)
    def _on_worker_progress(self, *args: object) -> None:
        """Normalize worker progress signals and route them by sender.

        Workers may emit progress as:
        - (current, total, message)
        - (current, message)
        - (current,)
        """
        job_id = self._job_id_from_sender()
        if not job_id:
            return

        current = args[0] if len(args) > 0 else 0
        if len(args) >= 3:
            total = args[1]
            message = args[2]
        elif len(args) == 2:
            total = 0
            message = args[1]
        else:
            total = 0
            message = ""

        self._on_job_progress(job_id, int(current), int(total), str(message))
