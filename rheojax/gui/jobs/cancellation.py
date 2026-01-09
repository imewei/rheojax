"""
Cancellation Token
=================

Thread-safe cancellation mechanism for background jobs.
"""

import time
from threading import Event

from rheojax.logging import get_logger

logger = get_logger(__name__)


class CancellationError(Exception):
    """Exception raised when a job is cancelled."""

    pass


class CancellationToken:
    """Thread-safe cancellation token for async operations.

    Features:
        - Thread-safe cancellation signaling
        - Non-blocking status checks
        - Error storage and retrieval
        - Reusable token state
        - Timeout support for waiting

    Example
    -------
    >>> token = CancellationToken()  # doctest: +SKIP
    >>> # In worker thread:
    >>> token.check()  # Raises CancellationError if cancelled
    >>> # In UI thread:
    >>> token.cancel()  # Request cancellation
    """

    def __init__(self, job_id: str | None = None) -> None:
        """Initialize cancellation token.

        Parameters
        ----------
        job_id : str, optional
            Job identifier for logging purposes
        """
        self._cancelled = Event()
        self._error: Exception | None = None
        self._job_id = job_id
        self._cancel_start_time: float | None = None

    def cancel(self) -> None:
        """Request cancellation.

        This sets the cancellation flag. Worker threads should
        check is_cancelled() or call check() periodically.
        """
        self._cancel_start_time = time.perf_counter()
        logger.debug("Cancellation requested", job_id=self._job_id)
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested.

        Returns
        -------
        bool
            True if cancelled, False otherwise

        Notes
        -----
        This is a non-blocking check that can be called
        frequently in tight loops.
        """
        return self._cancelled.is_set()

    def check(self) -> None:
        """Check and raise if cancelled.

        Raises
        ------
        CancellationError
            If cancellation was requested

        Notes
        -----
        This is the recommended way to check for cancellation
        in worker code, as it propagates the cancellation as
        an exception that can be caught by the worker pool.
        """
        if self.is_cancelled():
            elapsed = None
            if self._cancel_start_time is not None:
                elapsed = time.perf_counter() - self._cancel_start_time
            logger.debug("Cancellation complete", job_id=self._job_id, elapsed=elapsed)
            raise CancellationError("Operation cancelled by user")

    def set_error(self, error: Exception) -> None:
        """Store an error that occurred during execution.

        Parameters
        ----------
        error : Exception
            The error to store

        Notes
        -----
        This is useful for passing errors from worker threads
        back to the main thread without raising them immediately.
        """
        logger.error(
            "Error stored in cancellation token",
            job_id=self._job_id,
            error_type=type(error).__name__,
            exc_info=True,
        )
        self._error = error

    def get_error(self) -> Exception | None:
        """Get any error that occurred.

        Returns
        -------
        Exception or None
            The stored error, or None if no error occurred
        """
        return self._error

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation.

        Parameters
        ----------
        timeout : float, optional
            Wait timeout in seconds. If None, wait indefinitely.

        Returns
        -------
        bool
            True if cancelled within timeout, False if timeout expired

        Notes
        -----
        This blocks the calling thread until either cancellation
        is requested or the timeout expires.
        """
        return self._cancelled.wait(timeout)

    def reset(self) -> None:
        """Reset cancellation state.

        Clears both the cancellation flag and any stored error.
        This allows the token to be reused for another operation.

        Notes
        -----
        Be careful when reusing tokens - ensure the previous
        operation has fully completed before resetting.
        """
        self._cancelled.clear()
        self._error = None
        self._cancel_start_time = None
