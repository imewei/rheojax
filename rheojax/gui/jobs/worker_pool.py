"""
Worker Pool
==========

Thread pool for executing background jobs with progress tracking using PySide6.
"""

import logging
import uuid
from threading import Lock
from typing import Any

try:
    from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    # Provide stub classes for type checking when PySide6 not available
    class QObject:  # type: ignore
        pass
    class QThreadPool:  # type: ignore
        pass
    class QRunnable:  # type: ignore
        pass
    class Signal:  # type: ignore
        def __init__(self, *args): pass
    def Slot(*args):  # type: ignore
        def decorator(func):
            return func
        return decorator

from rheojax.gui.jobs.cancellation import CancellationToken

logger = logging.getLogger(__name__)


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

    def __new__(cls, max_threads: int = 4) -> "WorkerPool":
        """Create or return singleton instance."""
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
        if cls._instance is None:
            cls._instance = cls(max_threads)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        if cls._instance is not None:
            cls._instance.shutdown(wait=False)
        cls._instance = None
        cls._initialized = False

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
            raise ImportError(
                "PySide6 is required for WorkerPool. "
                "Install with: pip install PySide6"
            )

        super().__init__()
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(max_threads)
        self._active_jobs: dict[str, CancellationToken] = {}
        self._job_lock = Lock()

        WorkerPool._initialized = True
        logger.info(f"WorkerPool initialized with {max_threads} threads")

    def submit(self, worker: QRunnable) -> str:
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
        if not hasattr(worker, 'cancel_token'):
            logger.warning(f"Worker {worker} has no cancel_token attribute")
            worker.cancel_token = CancellationToken()  # type: ignore[attr-defined]

        # Register job
        with self._job_lock:
            self._active_jobs[job_id] = worker.cancel_token  # type: ignore[attr-defined]

        # Connect worker signals to pool signals
        if hasattr(worker, 'signals'):
            # Store job_id on worker for signal emission
            worker.job_id = job_id  # type: ignore[attr-defined]

            # Connect completion/failure/cancellation signals
            # IMPORTANT: Use default arguments to capture job_id by value, not by reference.
            # Without this, rapid job submissions could cause the wrong job_id to be used
            # when the signal is emitted (closure would capture the last job_id value).
            if hasattr(worker.signals, 'completed'):
                worker.signals.completed.connect(
                    lambda result, jid=job_id: self._on_job_completed(jid, result)
                )
            if hasattr(worker.signals, 'failed'):
                worker.signals.failed.connect(
                    lambda error, jid=job_id: self._on_job_failed(jid, error)
                )
            if hasattr(worker.signals, 'cancelled'):
                worker.signals.cancelled.connect(
                    lambda jid=job_id: self._on_job_cancelled(jid)
                )
            if hasattr(worker.signals, 'progress'):
                worker.signals.progress.connect(
                    lambda current, total, msg, jid=job_id: self._on_job_progress(
                        jid, current, total, msg
                    )
                )

        # Submit to thread pool
        self._pool.start(worker)
        self.job_started.emit(job_id)
        logger.debug(f"Job {job_id} submitted to pool")

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
                logger.info(f"Cancellation requested for job {job_id}")
                return True

        logger.warning(f"Job {job_id} not found for cancellation")
        return False

    def cancel_all(self) -> None:
        """Cancel all active jobs.

        Requests cancellation for every currently running job.
        """
        with self._job_lock:
            job_ids = list(self._active_jobs.keys())

        for job_id in job_ids:
            self.cancel(job_id)

        logger.info(f"Cancellation requested for {len(job_ids)} jobs")

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
        logger.info("Shutting down worker pool")

        # Cancel all active jobs
        self.cancel_all()

        # Wait for completion if requested
        if wait:
            success = self._pool.waitForDone(timeout_ms)
            if not success:
                logger.warning(
                    f"Worker pool shutdown timed out after {timeout_ms}ms"
                )

        # Clear active jobs
        with self._job_lock:
            self._active_jobs.clear()

        logger.info("Worker pool shut down")

    @Slot(str, object)
    def _on_job_completed(self, job_id: str, result: Any) -> None:
        """Handle job completion."""
        self._cleanup_job(job_id)
        self.job_completed.emit(job_id, result)
        logger.debug(f"Job {job_id} completed")

    @Slot(str, str)
    def _on_job_failed(self, job_id: str, error_message: str) -> None:
        """Handle job failure."""
        self._cleanup_job(job_id)
        self.job_failed.emit(job_id, error_message)
        logger.error(f"Job {job_id} failed: {error_message}")

    @Slot(str)
    def _on_job_cancelled(self, job_id: str) -> None:
        """Handle job cancellation."""
        self._cleanup_job(job_id)
        self.job_cancelled.emit(job_id)
        logger.info(f"Job {job_id} cancelled")

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
