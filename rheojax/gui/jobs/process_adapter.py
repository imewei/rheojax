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
import traceback
from collections.abc import Callable
from queue import Empty
from typing import Any

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
                if self._process is not None and not self._proc_is_alive(self._process):
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

    @staticmethod
    def _proc_is_alive(proc: mp.Process) -> bool:
        """Check if *proc* is alive, returning False if already closed."""
        try:
            return proc.is_alive()
        except ValueError:
            # Process was already .close()'d — it's dead.
            return False

    def _escalate_kill(self) -> None:
        """Escalation chain: cancel event -> SIGTERM -> SIGKILL."""
        proc = self._process
        if proc is None or not self._proc_is_alive(proc):
            return

        # Step 1: cooperative cancellation already set via _cancel_token.cancel()
        # Wait for the process to exit on its own.
        proc.join(timeout=self._process_timeout)
        if not self._proc_is_alive(proc):
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
        if not self._proc_is_alive(proc):
            return

        # Step 3: SIGKILL
        logger.debug("Subprocess did not respond to SIGTERM; sending SIGKILL")
        try:
            proc.kill()  # SIGKILL on Unix, TerminateProcess on Windows
        except OSError:
            pass  # Already dead
        proc.join(timeout=2.0)

    def _ensure_process_dead(self) -> None:
        """Best-effort cleanup called in the finally block of run().

        Cleans up all IPC resources (process handle, queue, event) to
        prevent leaked POSIX semaphores at interpreter shutdown.
        """
        proc = self._process
        if proc is not None:
            if self._proc_is_alive(proc):
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
            self._process = None

        # Close the IPC queue — releases its internal semaphore.
        queue = self._result_queue
        if queue is not None:
            try:
                queue.close()
                queue.join_thread()
            except (OSError, ValueError, BrokenPipeError):
                pass
            self._result_queue = None


# ---------------------------------------------------------------------------
# Worker isolation mode
# ---------------------------------------------------------------------------


def get_worker_isolation_mode() -> str:
    """Return the configured worker isolation mode.

    Reads the ``RHEOJAX_WORKER_ISOLATION`` environment variable.

    Returns
    -------
    str
        ``"subprocess"`` (default) or ``"thread"``.
    """
    return os.environ.get("RHEOJAX_WORKER_ISOLATION", "subprocess").lower()


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _extract_data(data: Any) -> tuple[Any, Any, Any, str, dict[str, Any]]:
    """Extract x, y, y2, test_mode, metadata from RheoData or DatasetState.

    Parameters
    ----------
    data : RheoData or DatasetState
        The data source.

    Returns
    -------
    tuple
        (x_data, y_data, y2_data, test_mode, metadata)
    """
    import numpy as np

    # DatasetState has .x_data / .y_data / .y2_data / .test_mode / .metadata
    if hasattr(data, "x_data"):
        x = data.x_data
        y = data.y_data
        y2 = getattr(data, "y2_data", None)
        test_mode = getattr(data, "test_mode", "relaxation")
        metadata = getattr(data, "metadata", {})
    # RheoData has .x / .y (and possibly .y2)
    elif hasattr(data, "x"):
        x = data.x
        y = data.y
        y2 = getattr(data, "y2", None)
        # RheoData stores test_mode in metadata or _explicit_test_mode
        test_mode = getattr(data, "_explicit_test_mode", None)
        if test_mode is None:
            md = getattr(data, "metadata", {})
            test_mode = md.get("test_mode", "relaxation")
        metadata = getattr(data, "metadata", {})
    else:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            "Expected RheoData or DatasetState."
        )

    # Ensure NumPy arrays for cross-process pickling
    if x is not None and not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if y is not None and not isinstance(y, np.ndarray):
        y = np.asarray(y)
    if y2 is not None and not isinstance(y2, np.ndarray):
        y2 = np.asarray(y2)

    return x, y, y2, str(test_mode or "relaxation"), dict(metadata) if metadata else {}


# ---------------------------------------------------------------------------
# Module-level entry points for subprocess work functions.
# These MUST be at module level for pickling with macOS spawn context.
# ---------------------------------------------------------------------------


def _fit_work_entry(
    progress_queue: Any,
    cancel_event: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Module-level entry point for fit subprocess (picklable)."""
    from rheojax.gui.jobs.subprocess_fit import run_fit_isolated

    return run_fit_isolated(
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        **kwargs,
    )


def _bayesian_work_entry(
    progress_queue: Any,
    cancel_event: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Module-level entry point for Bayesian subprocess (picklable)."""
    from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated

    return run_bayesian_isolated(
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# FitWorker factory
# ---------------------------------------------------------------------------


def make_fit_worker(
    model_name: str,
    data: Any,
    initial_params: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
    cancel_token: Any | None = None,
    dataset_id: str = "",
) -> Any:
    """Create a fit worker (subprocess or thread mode).

    Parameters
    ----------
    model_name : str
        Registered model name (e.g. ``"maxwell"``).
    data : RheoData or DatasetState
        Input data for fitting.
    initial_params : dict, optional
        Initial parameter values.
    options : dict, optional
        Fitting options (max_iter, ftol, etc.).
    cancel_token : CancellationToken, optional
        Only used in thread mode.
    dataset_id : str, optional
        Dataset identifier.

    Returns
    -------
    QRunnable
        Either a ``FitWorker`` (thread) or ``ProcessWorkerAdapter`` (subprocess).
    """
    mode = get_worker_isolation_mode()

    if mode == "thread":
        from rheojax.gui.jobs.fit_worker import FitWorker

        return FitWorker(
            model_name=model_name,
            data=data,
            initial_params=initial_params,
            options=options,
            cancel_token=cancel_token,
            dataset_id=dataset_id,
        )

    # Subprocess mode — use functools.partial with module-level entry
    # point for macOS spawn-context pickling compatibility.
    from functools import partial

    x_data, y_data, y2_data, test_mode, metadata = _extract_data(data)

    # Get deformation_mode and poisson_ratio from metadata or options
    deformation_mode = metadata.get("deformation_mode")
    poisson_ratio = metadata.get("poisson_ratio")

    work_fn = partial(
        _fit_work_entry,
        model_name=model_name,
        x_data=x_data,
        y_data=y_data,
        test_mode=test_mode,
        initial_params=initial_params or {},
        options=options or {},
        y2_data=y2_data,
        metadata=metadata,
        deformation_mode=deformation_mode,
        poisson_ratio=poisson_ratio,
        dataset_id=dataset_id,
    )

    return ProcessWorkerAdapter(work_fn)


# ---------------------------------------------------------------------------
# FitResult reconstruction
# ---------------------------------------------------------------------------


def fit_result_from_dict(d: dict[str, Any]) -> Any:
    """Reconstruct a ``FitResult`` dataclass from a subprocess result dict.

    Parameters
    ----------
    d : dict
        Dictionary returned by ``run_fit_isolated``.

    Returns
    -------
    FitResult
        The reconstructed dataclass.
    """
    from datetime import datetime

    from rheojax.gui.state.store import FitResult

    timestamp = d.get("timestamp")
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            timestamp = datetime.now()
    elif timestamp is None:
        timestamp = datetime.now()

    return FitResult(
        model_name=d.get("model_name", ""),
        parameters=d.get("parameters", {}),
        chi_squared=float(d.get("chi_squared", 0.0) or 0.0),
        success=bool(d.get("success", False)),
        message=d.get("message", ""),
        timestamp=timestamp,
        dataset_id=d.get("dataset_id", ""),
        r_squared=float(d.get("r_squared", 0.0) or 0.0),
        mpe=float(d.get("mpe", 0.0) or 0.0),
        fit_time=float(d.get("fit_time", 0.0) or 0.0),
        num_iterations=int(d.get("num_iterations", 0) or 0),
        convergence_message=d.get("convergence_message", ""),
        x_fit=d.get("x_fit"),
        y_fit=d.get("y_fit"),
        residuals=d.get("residuals"),
        pcov=d.get("pcov"),
        rmse=d.get("rmse"),
        mae=d.get("mae"),
        aic=d.get("aic"),
        bic=d.get("bic"),
        metadata=d.get("metadata"),
    )


# ---------------------------------------------------------------------------
# BayesianWorker factory
# ---------------------------------------------------------------------------


def make_bayesian_worker(
    model_name: str,
    data: Any,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    warm_start: dict[str, float] | None = None,
    priors: dict[str, Any] | None = None,
    seed: int = 42,
    cancel_token: Any | None = None,
    deformation_mode: str | None = None,
    poisson_ratio: float | None = None,
    fitted_model_state: dict[str, Any] | None = None,
    dataset_id: str = "",
) -> Any:
    """Create a Bayesian worker (subprocess or thread mode).

    Parameters
    ----------
    model_name : str
        Registered model name.
    data : RheoData or DatasetState
        Input data for inference.
    num_warmup : int
        Number of MCMC warmup iterations.
    num_samples : int
        Number of MCMC sampling iterations.
    num_chains : int
        Number of MCMC chains.
    warm_start : dict, optional
        Warm-start parameter values from NLSQ.
    priors : dict, optional
        Custom prior distributions.
    seed : int
        Random seed for reproducibility.
    cancel_token : CancellationToken, optional
        Only used in thread mode.
    deformation_mode : str, optional
        Deformation mode for DMTA.
    poisson_ratio : float, optional
        Poisson ratio for E-to-G conversion.
    fitted_model_state : dict, optional
        Full model state for warm-start.
    dataset_id : str
        Dataset identifier.

    Returns
    -------
    QRunnable
        Either a ``BayesianWorker`` (thread) or ``ProcessWorkerAdapter`` (subprocess).
    """
    mode = get_worker_isolation_mode()

    if mode == "thread":
        from rheojax.gui.jobs.bayesian_worker import BayesianWorker

        return BayesianWorker(
            model_name=model_name,
            data=data,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            warm_start=warm_start,
            priors=priors,
            seed=seed,
            cancel_token=cancel_token,
            deformation_mode=deformation_mode,
            poisson_ratio=poisson_ratio,
            fitted_model_state=fitted_model_state,
            dataset_id=dataset_id,
        )

    # Subprocess mode — use functools.partial with module-level entry
    # point for macOS spawn-context pickling compatibility.
    from functools import partial

    import numpy as np

    x_data, y_data, y2_data, test_mode, metadata = _extract_data(data)

    # Convert fitted_model_state JAX arrays to NumPy for pickling
    safe_model_state = None
    if fitted_model_state is not None:
        safe_model_state = {}
        for k, v in fitted_model_state.items():
            if hasattr(v, "numpy"):
                safe_model_state[k] = np.asarray(v)
            elif hasattr(v, "__jax_array__"):
                safe_model_state[k] = np.asarray(v)
            else:
                safe_model_state[k] = v

    work_fn = partial(
        _bayesian_work_entry,
        model_name=model_name,
        x_data=x_data,
        y_data=y_data,
        test_mode=test_mode,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        warm_start=warm_start,
        priors=priors or {},
        seed=seed,
        y2_data=y2_data,
        metadata=metadata,
        deformation_mode=deformation_mode,
        poisson_ratio=poisson_ratio,
        fitted_model_state=safe_model_state,
        dataset_id=dataset_id,
    )

    return ProcessWorkerAdapter(work_fn)


# ---------------------------------------------------------------------------
# BayesianResult reconstruction
# ---------------------------------------------------------------------------


def bayesian_result_from_dict(d: dict[str, Any]) -> Any:
    """Reconstruct a ``BayesianResult`` dataclass from a subprocess result dict.

    Parameters
    ----------
    d : dict
        Dictionary returned by ``run_bayesian_isolated``.

    Returns
    -------
    BayesianResult
        The reconstructed dataclass.
    """
    from datetime import datetime

    from rheojax.gui.state.store import BayesianResult

    timestamp = d.get("timestamp")
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            timestamp = datetime.now()
    elif timestamp is None:
        timestamp = datetime.now()

    return BayesianResult(
        model_name=d.get("model_name", ""),
        dataset_id=d.get("dataset_id", ""),
        posterior_samples=d.get("posterior_samples"),
        summary=d.get("summary"),
        r_hat=d.get("r_hat", {}),
        ess=d.get("ess", {}),
        divergences=int(d.get("divergences", 0) or 0),
        credible_intervals=d.get("credible_intervals", {}),
        mcmc_time=float(d.get("mcmc_time", 0.0) or 0.0),
        timestamp=timestamp,
        num_warmup=int(d.get("num_warmup", 0) or 0),
        num_samples=int(d.get("num_samples", 0) or 0),
        num_chains=int(d.get("num_chains", 4) or 4),
        inference_data=d.get("inference_data"),
        sample_stats=d.get("sample_stats"),
        diagnostics_valid=bool(d.get("diagnostics_valid", True)),
    )
