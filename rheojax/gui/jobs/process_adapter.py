"""
Process Worker Adapter
=====================

Subprocess-based worker isolation for GUI jobs.

Wraps arbitrary work functions in a ``multiprocessing.Process`` so they can be
terminated cleanly via escalation (cancel event -> SIGTERM -> SIGKILL) without
leaving the parent Qt thread pool in a broken state.

IPC Protocol
------------
The child process communicates with the parent via a single
``multiprocessing.Queue``.  Messages are dicts with a ``"type"`` key:

Progress messages (non-terminal):
    ``{"type": "progress", "percent": int, "total": int, "message": str}``
    ``{"type": "stage_changed", "stage": str}``
    ``{"type": "divergence_detected", "count": int}``

Terminal messages (exactly one per run):
    ``{"type": "completed", "result": <picklable object>}``
    ``{"type": "failed", "error": str, "traceback": str}``
    ``{"type": "cancelled"}``
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from queue import Empty
from typing import Any, Callable

try:
    from rheojax.gui.compat import QObject, QRunnable, Signal

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

    class QObject:  # type: ignore[no-redef]
        pass

    class QRunnable:  # type: ignore[no-redef]
        pass

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args: Any) -> None:
            pass

from rheojax.gui.jobs.cancellation import CancellationError, ProcessCancellationToken
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Terminal message types that end the poll loop
_TERMINAL_TYPES = frozenset({"completed", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# Subprocess entry point (must be module-level for pickling with "spawn")
# ---------------------------------------------------------------------------


def _subprocess_entry(
    work_fn: Callable[..., Any],
    result_queue: mp.Queue,
    cancel_event: mp.synchronize.Event,
) -> None:
    """Entry point executed inside the child process.

    Calls *work_fn(result_queue, cancel_event)* and wraps the outcome in a
    terminal IPC message placed on *result_queue*.

    Parameters
    ----------
    work_fn : callable
        The user-supplied work function.  Signature:
        ``work_fn(progress_queue, cancel_event) -> result``.
        May put non-terminal messages on *progress_queue* for progress
        reporting.
    result_queue : mp.Queue
        Shared queue for IPC messages (progress + terminal).
    cancel_event : mp.synchronize.Event
        Shared event; set by parent to request cancellation.
    """
    try:
        result = work_fn(result_queue, cancel_event)
        result_queue.put({"type": "completed", "result": result})
    except CancellationError:
        result_queue.put({"type": "cancelled"})
    except Exception as exc:
        result_queue.put(
            {
                "type": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


# ---------------------------------------------------------------------------
# Qt signal bundle
# ---------------------------------------------------------------------------


class ProcessWorkerSignals(QObject):
    """Signals emitted by :class:`ProcessWorkerAdapter`.

    Mirrors the signal shape used by ``FitWorkerSignals`` and
    ``BayesianWorkerSignals`` so callers can connect the same slots.

    Signals
    -------
    progress : Signal(int, int, str)
        Progress update: (percent, total, message).
    stage_changed : Signal(str)
        Bayesian sampling stage (e.g. ``'warmup'``, ``'sampling'``).
    completed : Signal(object)
        Work completed successfully; payload is the return value.
    failed : Signal(str)
        Work failed; payload is the error message.
    cancelled : Signal()
        Work was cancelled via the cancellation token.
    divergence_detected : Signal(int)
        Number of MCMC divergences detected so far.
    """

    progress = Signal(int, int, str)
    stage_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    divergence_detected = Signal(int)


# ---------------------------------------------------------------------------
# QRunnable adapter
# ---------------------------------------------------------------------------


class ProcessWorkerAdapter(QRunnable):
    """Run a work function inside a ``multiprocessing.Process``.

    The adapter starts the child process, polls the IPC queue in its
    ``run()`` method (which executes on a QThreadPool thread), and
    re-emits messages as Qt signals.

    Cancellation follows an escalation chain:
    1. Set the ``mp.Event`` so the child can exit cooperatively.
    2. Wait *process_timeout* seconds.
    3. Send ``SIGTERM``.
    4. Wait *kill_timeout* seconds.
    5. Send ``SIGKILL``.

    Parameters
    ----------
    work_fn : callable
        ``work_fn(progress_queue, cancel_event) -> result``.
        Must be picklable (module-level function with "spawn" start method).
    process_timeout : float
        Seconds to wait after setting the cancel event before sending
        SIGTERM.  Default ``5.0``.
    kill_timeout : float
        Seconds to wait after SIGTERM before sending SIGKILL.
        Default ``2.0``.
    """

    def __init__(
        self,
        work_fn: Callable[..., Any],
        process_timeout: float = 5.0,
        kill_timeout: float = 2.0,
    ) -> None:
        if not HAS_PYSIDE6:
            raise ImportError(
                "PySide6 is required for ProcessWorkerAdapter. "
                "Install with: pip install PySide6"
            )
        super().__init__()
        self.signals = ProcessWorkerSignals()
        self._work_fn = work_fn
        self._process_timeout = process_timeout
        self._kill_timeout = kill_timeout

        # Cancellation token backed by mp.Event
        self._cancel_token = ProcessCancellationToken()

        # Will be set when the process is started
        self._process: mp.Process | None = None
        self._result_queue: mp.Queue | None = None

    @property
    def cancel_token(self) -> ProcessCancellationToken:
        """The cancellation token for this worker."""
        return self._cancel_token

    # ---- public API --------------------------------------------------------

    def run(self) -> None:  # noqa: D401 – QRunnable override
        """Execute the work function in a child process.

        Called by QThreadPool.  Blocks the pool thread until the child
        process finishes (or is killed).
        """
        self._result_queue = mp.Queue()
        self._process = mp.Process(
            target=_subprocess_entry,
            args=(self._work_fn, self._result_queue, self._cancel_token.event),
            daemon=True,
        )

        try:
            logger.debug("Starting subprocess worker")
            self._process.start()
            self._poll_loop(self._result_queue)
        except Exception as exc:
            logger.error("ProcessWorkerAdapter.run() failed", error=str(exc))
            self.signals.failed.emit(f"Subprocess worker error: {exc}")
        finally:
            self._ensure_process_dead()

    def cancel(self) -> None:
        """Request cancellation with escalation to SIGTERM/SIGKILL."""
        logger.debug("Cancel requested for subprocess worker")
        self._cancel_token.cancel()
        self._escalate_kill()

    # ---- internals ---------------------------------------------------------

    def _poll_loop(self, result_queue: mp.Queue) -> None:
        """Poll *result_queue* until a terminal message arrives or the
        child dies unexpectedly.
        """
        while True:
            # Try to get a message with a short timeout so we can also
            # check whether the child process is still alive.
            try:
                msg = result_queue.get(timeout=0.25)
            except Empty:
                # No message yet -- check if child is still running
                if self._process is not None and not self._process.is_alive():
                    # Child died without sending a terminal message
                    self._drain_queue(result_queue)
                    exitcode = self._process.exitcode
                    if self._cancel_token.is_cancelled():
                        self.signals.cancelled.emit()
                    elif exitcode is not None and exitcode != 0:
                        self.signals.failed.emit(
                            f"Subprocess exited unexpectedly (exit code {exitcode})"
                        )
                    else:
                        # exitcode 0 but no terminal message -- odd, treat as failure
                        self.signals.failed.emit(
                            "Subprocess exited without sending a result"
                        )
                    return
                continue

            msg_type = msg.get("type", "")

            # Non-terminal messages ----------------------------------------
            if msg_type == "progress":
                self.signals.progress.emit(
                    msg.get("percent", 0),
                    msg.get("total", 0),
                    msg.get("message", ""),
                )
            elif msg_type == "stage_changed":
                self.signals.stage_changed.emit(msg.get("stage", ""))
            elif msg_type == "divergence_detected":
                self.signals.divergence_detected.emit(msg.get("count", 0))

            # Terminal messages --------------------------------------------
            elif msg_type == "completed":
                self.signals.completed.emit(msg.get("result"))
                return
            elif msg_type == "failed":
                error = msg.get("error", "Unknown error")
                tb = msg.get("traceback", "")
                if tb:
                    logger.debug("Subprocess traceback:\n%s", tb)
                self.signals.failed.emit(error)
                return
            elif msg_type == "cancelled":
                self.signals.cancelled.emit()
                return
            else:
                logger.warning("Unknown IPC message type: %s", msg_type)

    def _drain_queue(self, result_queue: mp.Queue) -> None:
        """Drain any remaining messages after the process has exited.

        If a terminal message is found while draining, emit the
        corresponding signal.
        """
        while True:
            try:
                msg = result_queue.get_nowait()
            except Empty:
                break

            msg_type = msg.get("type", "")
            if msg_type == "completed":
                self.signals.completed.emit(msg.get("result"))
                return
            elif msg_type == "failed":
                self.signals.failed.emit(msg.get("error", "Unknown error"))
                return
            elif msg_type == "cancelled":
                self.signals.cancelled.emit()
                return
            # Ignore non-terminal messages during drain

    def _escalate_kill(self) -> None:
        """Escalation chain: cancel event -> SIGTERM -> SIGKILL."""
        proc = self._process
        if proc is None or not proc.is_alive():
            return

        # Step 1: cooperative cancellation already set via _cancel_token.cancel()
        # Wait for the process to exit on its own.
        proc.join(timeout=self._process_timeout)
        if not proc.is_alive():
            return

        # Step 2: SIGTERM
        logger.debug("Subprocess did not exit cooperatively; sending SIGTERM")
        pid = proc.pid
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                return  # Process already gone
        proc.join(timeout=self._kill_timeout)
        if not proc.is_alive():
            return

        # Step 3: SIGKILL
        logger.debug("Subprocess did not respond to SIGTERM; sending SIGKILL")
        try:
            proc.kill()  # SIGKILL on Unix, TerminateProcess on Windows
        except OSError:
            pass  # Already dead
        proc.join(timeout=2.0)

    def _ensure_process_dead(self) -> None:
        """Best-effort cleanup called in the finally block of run()."""
        proc = self._process
        if proc is None:
            return
        if proc.is_alive():
            logger.warning("Process still alive in cleanup; killing")
            try:
                proc.kill()
            except OSError:
                pass
            proc.join(timeout=2.0)
        # Close the process handle to free resources
        try:
            proc.close()
        except (ValueError, OSError):
            pass  # Already closed or not started
