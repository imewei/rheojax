"""Qt signal definitions for state management.

This module provides Qt signals for reactive UI updates on state changes.
"""

from collections import deque
from collections.abc import Callable
from typing import Any

from rheojax.gui.compat import QObject, Signal
from rheojax.logging import get_logger

try:
    from PySide6.QtCore import Qt, Slot
except ImportError:
    # Fallback stubs when PySide6 is unavailable (headless/test environment)
    Qt = None  # type: ignore[assignment]

    def Slot(*args, **kwargs):  # type: ignore[misc]
        """No-op Slot decorator for non-Qt environments."""

        def decorator(fn):
            return fn

        return decorator


logger = get_logger(__name__)


class StateSignals(QObject):
    """Qt signals for state change notifications.

    Signals
    -------
    state_changed : Signal()
        General state update signal
    dataset_added : Signal(str)
        Emitted when dataset added (dataset_id)
    dataset_removed : Signal(str)
        Emitted when dataset removed (dataset_id)
    dataset_updated : Signal(str)
        Emitted when dataset modified (dataset_id)
    dataset_selected : Signal(str)
        Emitted when active dataset changes (dataset_id)
    model_selected : Signal(str)
        Emitted when model selected (model_name)
    model_params_changed : Signal(str)
        Emitted when model parameters change (model_name)
    fit_started : Signal(str, str)
        Emitted when fit starts (model_name, dataset_id)
    fit_progress : Signal(str, int)
        Emitted during fit progress (job_id, progress 0-100)
    fit_completed : Signal(str, str)
        Emitted when fit completes (model_name, dataset_id)
    fit_failed : Signal(str, str, str)
        Emitted when fit fails (model_name, dataset_id, error)
    bayesian_started : Signal(str, str)
        Emitted when Bayesian inference starts (model_name, dataset_id)
    bayesian_progress : Signal(str, int)
        Emitted during Bayesian progress (job_id, progress 0-100)
    bayesian_completed : Signal(str, str)
        Emitted when Bayesian inference completes (model_name, dataset_id)
    bayesian_failed : Signal(str, str, str)
        Emitted when Bayesian inference fails (model_name, dataset_id, error)
    pipeline_step_changed : Signal(str, str)
        Emitted when pipeline step changes (step_name, status)
    pipeline_structure_changed : Signal()
        Emitted on any structural change to the visual pipeline (add/remove/reorder)
    pipeline_step_selected : Signal(str)
        Emitted when a visual pipeline step is selected (step_id, empty for deselect)
    pipeline_step_status_changed : Signal(str, str)
        Emitted when a visual pipeline step status changes (step_id, status_name)
    pipeline_execution_started : Signal()
        Emitted when visual pipeline execution begins
    pipeline_execution_completed : Signal()
        Emitted when visual pipeline execution finishes
    pipeline_name_changed : Signal(str)
        Emitted when the visual pipeline name changes (new_name)
    transform_applied : Signal(str, str)
        Emitted when transform applied (transform_name, dataset_id)
    jax_device_changed : Signal(str)
        Emitted when JAX device changes (device_name)
    jax_memory_updated : Signal(int, int)
        Emitted when JAX memory updates (used_bytes, total_bytes)
    theme_changed : Signal(str)
        Emitted when theme changes (theme_name)
    project_loaded : Signal(str)
        Emitted when project loaded (project_path)
    project_saved : Signal(str)
        Emitted when project saved (project_path)
    """

    # General
    state_changed = Signal()

    # Datasets
    dataset_added = Signal(str)
    dataset_removed = Signal(str)
    dataset_updated = Signal(str)
    dataset_selected = Signal(str)

    # Models
    model_selected = Signal(str)
    model_params_changed = Signal(str)

    # Fitting
    fit_started = Signal(str, str)  # model_name, dataset_id
    fit_progress = Signal(str, int)  # job_id, progress
    fit_completed = Signal(str, str)  # model_name, dataset_id
    fit_failed = Signal(str, str, str)  # model_name, dataset_id, error

    # Bayesian
    bayesian_started = Signal(str, str)  # model_name, dataset_id
    bayesian_progress = Signal(str, int)  # job_id, progress
    bayesian_completed = Signal(str, str)  # model_name, dataset_id
    bayesian_failed = Signal(str, str, str)  # model_name, dataset_id, error

    # Pipeline
    pipeline_step_changed = Signal(str, str)  # step_name, status

    # Visual Pipeline signals
    pipeline_step_added = Signal(str)  # step_id
    pipeline_step_removed = Signal(str)  # step_id
    pipeline_step_selected = Signal(str)  # step_id (empty string for deselection)
    pipeline_step_config_changed = Signal(str)  # step_id
    pipeline_step_status_changed = Signal(str, str)  # step_id, status_name
    pipeline_structure_changed = Signal()  # any structural change (add/remove/reorder)
    pipeline_execution_started = Signal()
    pipeline_execution_completed = Signal()
    pipeline_name_changed = Signal(str)  # new_name

    # Transforms
    transform_applied = Signal(str, str)  # transform_name, dataset_id

    # JAX
    jax_device_changed = Signal(str)  # device_name
    jax_memory_updated = Signal(int, int)  # used_bytes, total_bytes

    # UI
    theme_changed = Signal(str)  # theme_name
    os_theme_changed = Signal(str)  # resolved OS color scheme ("light" or "dark")

    # Project
    project_loaded = Signal(str)  # project_path
    project_saved = Signal(str)  # project_path

    def __init__(self) -> None:
        """Initialize signal emitters."""
        super().__init__()
        # GUI-P1-002: thread-safe queue for subscriber notifications deferred
        # to the GUI thread via QMetaObject.invokeMethod(QueuedConnection).
        # Using a deque prevents buffer overwrite when multiple worker-thread
        # updates are posted before the event loop processes them (M2 fix).
        self._pending_notifications: deque[tuple[list, Any]] = deque()

    @Slot()
    def _emit_state_changed(self) -> None:
        """Slot invoked via QMetaObject.invokeMethod to emit state_changed.

        This method is the target of QueuedConnection invocations from
        update_state() so that the signal is always emitted on the main
        Qt thread, regardless of which thread triggered the state update.
        """
        self.state_changed.emit()

    @Slot()
    def _run_subscriber_notifications(self) -> None:
        """GUI-P1-002 fix: call subscriber callbacks on the GUI thread.

        Invoked exclusively via QMetaObject.invokeMethod(QueuedConnection)
        from StateStore.update_state() when the state update originates on a
        worker thread.  By routing through this slot, subscriber callbacks
        (which update Qt widgets) always run on the main Qt thread.

        M2 fix: drains all queued (subscribers, snapshot) pairs so rapid
        worker-thread updates are never lost.
        M3 fix: reentrancy guard mirrors the TLS guard in the synchronous
        path of update_state() — nested dispatches triggered by subscribers
        are properly ordered.
        """
        # Drain all queued notifications (M2)
        while self._pending_notifications:
            subscribers, snapshot = self._pending_notifications.popleft()
            for subscriber in subscribers:
                try:
                    subscriber(snapshot)
                except Exception:
                    logger.error(
                        "Subscriber callback failed (main-thread relay)",
                        exc_info=True,
                    )

    def emit_signal(self, signal_name: str, *args: Any) -> None:
        """Emit a signal by name with logging.

        Parameters
        ----------
        signal_name : str
            Name of the signal to emit
        *args : Any
            Arguments to pass to the signal
        """
        signal = getattr(self, signal_name, None)
        if signal is not None:
            logger.debug("Signal emitted", signal=signal_name, args=args)
            signal.emit(*args)
        else:
            # GUI-R6-004: Remove exc_info=True — there is no active exception here,
            # so exc_info produces a misleading NoneType traceback in logs.
            logger.error(
                "Attempted to emit unknown signal",
                signal=signal_name,
            )

    def connect_signal(self, signal_name: str, handler: Callable[..., Any]) -> bool:
        """Connect a handler to a signal by name with logging.

        Parameters
        ----------
        signal_name : str
            Name of the signal to connect
        handler : Callable[..., Any]
            Handler function to connect

        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        signal = getattr(self, signal_name, None)
        if signal is not None:
            handler_name = getattr(handler, "__name__", repr(handler))
            logger.debug("Signal connected", signal=signal_name, handler=handler_name)
            # Use AutoConnection (Qt default) — it selects DirectConnection
            # for same-thread and QueuedConnection for cross-thread automatically.
            # Worker-originated signals that need explicit QueuedConnection are
            # connected at the call site (e.g. MainWindow._init_worker_pool).
            signal.connect(handler)
            return True
        else:
            # GUI-R6-008: Removed exc_info=True — no active exception here.
            # The signal_name lookup failure is a programming error, not a
            # caught exception.
            logger.error(
                "Attempted to connect to unknown signal",
                signal=signal_name,
            )
            return False

    def disconnect_signal(
        self, signal_name: str, handler: Callable[..., Any] | None = None
    ) -> bool:
        """Disconnect a handler from a signal by name with logging.

        Parameters
        ----------
        signal_name : str
            Name of the signal to disconnect
        handler : Callable[..., Any] | None
            Handler function to disconnect, or None to disconnect all

        Returns
        -------
        bool
            True if disconnection successful, False otherwise
        """
        signal = getattr(self, signal_name, None)
        if signal is not None:
            handler_name = (
                getattr(handler, "__name__", repr(handler)) if handler else "all"
            )
            logger.debug(
                "Signal disconnected", signal=signal_name, handler=handler_name
            )
            try:
                if handler is not None:
                    signal.disconnect(handler)
                else:
                    signal.disconnect()
                return True
            except RuntimeError as e:
                logger.error(
                    "Failed to disconnect signal",
                    signal=signal_name,
                    handler=handler_name,
                    error=str(e),
                    exc_info=True,
                )
                return False
        else:
            # R11-SIG-001: Removed exc_info=True — no active exception here;
            # matches the corrected pattern in connect_signal() and emit_signal().
            logger.error(
                "Attempted to disconnect from unknown signal",
                signal=signal_name,
            )
            return False
