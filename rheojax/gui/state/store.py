"""Central state management for RheoJAX GUI.

This module implements a singleton state store with immutable state updates
and signal-based reactivity for the Qt GUI.

Thread Safety
-------------
The StateStore uses a threading.RLock to ensure thread-safe access when
background workers update state from worker threads. All public methods
that read or modify state are protected by the lock.
"""

import copy
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Optional

from rheojax.gui.state.signals import StateSignals
from rheojax.logging import get_logger

logger = get_logger(__name__)


class PipelineStep(Enum):
    """Pipeline execution steps."""

    LOAD = auto()
    TRANSFORM = auto()
    FIT = auto()
    BAYESIAN = auto()
    EXPORT = auto()


class WorkflowMode(Enum):
    """Application workflow modes."""

    FITTING = auto()
    TRANSFORM = auto()


class StepStatus(Enum):
    """Status of each pipeline step."""

    PENDING = auto()
    ACTIVE = auto()
    COMPLETE = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class ParameterState:
    """State for a single model parameter."""

    name: str
    value: float
    min_bound: float
    max_bound: float
    fixed: bool = False
    unit: str = ""
    description: str = ""

    def clone(self) -> "ParameterState":
        """Create a deep copy of this parameter state."""
        return replace(self)


@dataclass
class DatasetState:
    """State for a loaded dataset."""

    id: str
    name: str
    file_path: Path | None
    test_mode: str  # oscillation, relaxation, creep, rotation
    x_data: Any | None = None  # Will be jax array
    y_data: Any | None = None
    y2_data: Any | None = None  # For G'' in oscillation
    metadata: dict = field(default_factory=dict)
    is_modified: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def clone(self) -> "DatasetState":
        """Create a deep copy of this dataset state.

        NOTE: x_data/y_data/y2_data are shared references by design to avoid
        copying large arrays. Data should be converted to NumPy at the import
        boundary (DataService/ImportWorker) so these are never live JAX device arrays.
        """
        import numpy as np

        # R12-C-011: Enforce NumPy-at-boundary contract — x_data/y_data must be
        # NumPy arrays (or None) by the time they reach the store.  JAX device
        # arrays must be converted in DataService/ImportWorker before storage.
        assert isinstance(self.x_data, (np.ndarray, type(None))), (
            f"x_data must be a NumPy array or None, got {type(self.x_data).__name__}. "
            "Convert to NumPy at the import boundary before storing in the state."
        )
        assert isinstance(self.y_data, (np.ndarray, type(None))), (
            f"y_data must be a NumPy array or None, got {type(self.y_data).__name__}. "
            "Convert to NumPy at the import boundary before storing in the state."
        )
        return replace(
            self,
            metadata=copy.deepcopy(self.metadata),
            x_data=self.x_data,  # Keep reference to avoid copying large arrays (GUI-010)
            y_data=self.y_data,
            y2_data=self.y2_data,
        )


@dataclass
class FitResult:
    """Canonical result from NLSQ point estimation fit.

    This is the single source of truth for fit results across the GUI.
    All GUI modules (fit_worker, model_service, pages) import from here.
    """

    model_name: str
    parameters: dict[str, float]
    chi_squared: float
    success: bool
    message: str
    timestamp: datetime
    # Optional fields — populated when available
    dataset_id: str = ""
    r_squared: float = 0.0
    mpe: float = 0.0
    fit_time: float = 0.0
    num_iterations: int = 0
    convergence_message: str = ""
    x_fit: Any | None = None
    y_fit: Any | None = None
    residuals: Any | None = None
    pcov: Any | None = None
    rmse: float | None = None
    mae: float | None = None
    aic: float | None = None
    bic: float | None = None
    metadata: dict[str, Any] | None = None

    def clone(self) -> "FitResult":
        """Create a deep copy of this fit result."""
        return replace(
            self,
            parameters=copy.deepcopy(self.parameters),
            x_fit=self.x_fit,
            y_fit=self.y_fit,
            residuals=self.residuals,
            metadata=copy.deepcopy(self.metadata) if self.metadata else None,
        )


@dataclass
class BayesianResult:
    """Result from Bayesian NUTS inference."""

    model_name: str
    dataset_id: str
    posterior_samples: Any | None  # Dict of posterior samples
    summary: dict[str, dict[str, float]] | None
    r_hat: dict[str, float]
    ess: dict[str, float]
    divergences: int
    credible_intervals: dict[str, tuple[float, float]]
    mcmc_time: float
    timestamp: datetime
    num_warmup: int = 1000
    num_samples: int = 2000
    num_chains: int = 4
    inference_data: Any | None = None  # Full ArviZ InferenceData with sample_stats
    sample_stats: dict[str, Any] | None = None  # Raw energy/diverging arrays from MCMC
    diagnostics_valid: bool = True  # False when R-hat/ESS fell back to NaN/defaults

    @property
    def sampling_time(self) -> float:
        """Alias for mcmc_time (compatibility with worker BayesianResult)."""
        return self.mcmc_time

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Computed diagnostics dict (compatibility with worker BayesianResult)."""
        return {
            "r_hat": self.r_hat,
            "ess": self.ess,
            "divergences": self.divergences,
            "diagnostics_valid": self.diagnostics_valid,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Synthesized metadata dict for backward compatibility."""
        return {
            "model_name": self.model_name,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
        }

    def clone(self) -> "BayesianResult":
        """Create a deep copy of this Bayesian result.

        Large array data (posterior_samples, inference_data, summary) is
        kept as a reference to avoid expensive deep-copies of JAX/NumPy
        arrays and ArviZ InferenceData objects.  Small metadata dicts
        (r_hat, ess, credible_intervals) are deep-copied for isolation.
        """
        return replace(
            self,
            r_hat=copy.deepcopy(self.r_hat),
            ess=copy.deepcopy(self.ess),
            credible_intervals=copy.deepcopy(self.credible_intervals),
            posterior_samples=self.posterior_samples,  # Large arrays — reference
            inference_data=self.inference_data,  # ArviZ InferenceData — reference
            # R12-C-010: shallow-copy summary so that two clones do not share the
            # same dict object; callers can safely add/remove top-level keys.
            # Values (nested dicts of floats) remain shared references.
            summary=self.summary.copy() if self.summary else None,
        )


@dataclass
class PipelineState:
    """State of the analysis pipeline."""

    steps: dict[PipelineStep, StepStatus] = field(
        default_factory=lambda: dict.fromkeys(PipelineStep, StepStatus.PENDING)
    )
    current_step: PipelineStep | None = None
    error_message: str | None = None

    def clone(self) -> "PipelineState":
        """Create a deep copy of this pipeline state."""
        return replace(self, steps=copy.deepcopy(self.steps))


@dataclass
class TransformRecord:
    """Record of a transform operation for provenance tracking."""

    timestamp: datetime
    source_dataset_id: str
    target_dataset_id: str
    transform_name: str
    parameters: dict[str, Any]
    seed: int | None

    def clone(self) -> "TransformRecord":
        """Create a deep copy of this transform record."""
        return replace(self, parameters=copy.deepcopy(self.parameters))


@dataclass
class PipelineStepConfig:
    """Configuration for a single visual pipeline step."""

    id: str  # UUID
    step_type: str  # "load", "transform", "fit", "bayesian", "export"
    name: str  # Display name
    config: dict[str, Any] = field(default_factory=dict)
    status: "StepStatus" = field(default_factory=lambda: StepStatus.PENDING)
    result_cache_key: str | None = None
    error_message: str | None = None
    position: int = 0

    def clone(self) -> "PipelineStepConfig":
        """Create a deep copy of this pipeline step config."""
        return PipelineStepConfig(
            id=self.id,
            step_type=self.step_type,
            name=self.name,
            config=copy.deepcopy(self.config),
            status=self.status,
            result_cache_key=self.result_cache_key,
            error_message=self.error_message,
            position=self.position,
        )


@dataclass
class VisualPipelineState:
    """State for the visual pipeline builder in the GUI."""

    steps: list["PipelineStepConfig"] = field(default_factory=list)
    selected_step_id: str | None = None
    is_running: bool = False
    current_running_step_id: str | None = None
    pipeline_name: str = "Untitled Pipeline"
    step_results: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "VisualPipelineState":
        """Create a deep copy of this visual pipeline state."""
        return VisualPipelineState(
            steps=[s.clone() for s in self.steps],
            selected_step_id=self.selected_step_id,
            is_running=self.is_running,
            current_running_step_id=self.current_running_step_id,
            pipeline_name=self.pipeline_name,
            step_results=copy.deepcopy(self.step_results),
        )


@dataclass
class AppState:
    """Root application state."""

    # Project
    project_path: Path | None = None
    project_name: str = "Untitled"
    is_modified: bool = False

    # Datasets
    datasets: dict[str, DatasetState] = field(default_factory=dict)
    active_dataset_id: str | None = None

    # Models
    active_model_name: str | None = None
    model_params: dict[str, ParameterState] = field(default_factory=dict)

    # Fitting results
    fit_results: dict[str, FitResult] = field(default_factory=dict)
    bayesian_results: dict[str, BayesianResult] = field(default_factory=dict)

    # UI State
    current_tab: str = "home"
    pipeline_state: PipelineState = field(default_factory=PipelineState)
    visual_pipeline: VisualPipelineState = field(default_factory=VisualPipelineState)

    # JAX
    jax_device: str = "cpu"
    jax_memory_used: int = 0
    jax_memory_total: int = 0

    # Provenance
    transform_history: list[TransformRecord] = field(default_factory=list)

    # DMTA / Deformation
    deformation_mode: str = "shear"  # shear, tension, bending, compression
    poisson_ratio: float = 0.5  # rubber default

    # Settings
    workflow_mode: WorkflowMode = WorkflowMode.FITTING
    current_seed: int = 42
    auto_save_enabled: bool = True
    theme: str = "light"
    os_theme: str = "light"  # Resolved OS color scheme ("light" or "dark")
    last_export_dir: Path | None = None
    recent_projects: list[Path] = field(default_factory=list)

    def clone(self) -> "AppState":
        """Create a deep copy of the entire app state."""
        return replace(
            self,
            datasets={k: v.clone() for k, v in self.datasets.items()},
            model_params={k: v.clone() for k, v in self.model_params.items()},
            fit_results={k: v.clone() for k, v in self.fit_results.items()},
            bayesian_results={k: v.clone() for k, v in self.bayesian_results.items()},
            pipeline_state=self.pipeline_state.clone(),
            visual_pipeline=self.visual_pipeline.clone(),
            transform_history=[t.clone() for t in self.transform_history],
            recent_projects=copy.copy(self.recent_projects),
        )


class StateStore:
    """Singleton state store with Qt signal integration.

    This store implements immutable state updates with signals for reactive UI.
    All state modifications go through update_state() which creates a new state
    object and notifies subscribers.

    Thread Safety
    -------------
    All public methods are protected by an RLock to ensure safe access from
    background worker threads. The lock is reentrant to allow nested calls
    (e.g., a subscriber callback that reads state during an update).
    """

    _instance: Optional["StateStore"] = None
    _singleton_lock: Lock = Lock()

    # Class-level type annotations for instance attributes (singleton pattern)
    _state: AppState
    _signals: Any
    _subscribers: list[Callable[[AppState], None]]
    _undo_stack: list[AppState]
    _redo_stack: list[AppState]
    _max_undo_size: int
    _lock: RLock

    def __new__(cls) -> "StateStore":
        with cls._singleton_lock:
            if cls._instance is None:
                # G-015: Verify StateStore is created on the main Qt thread so all
                # signals have correct thread affinity for cross-thread delivery.
                try:
                    from PySide6.QtCore import QCoreApplication, QThread

                    app = QCoreApplication.instance()
                    if app is not None and QThread.currentThread() != app.thread():
                        logger.warning(
                            "StateStore created from non-main thread. "
                            "Qt signals may have incorrect thread affinity.",
                            current_thread=str(QThread.currentThread()),
                        )
                except ImportError:
                    pass  # PySide6 not available (e.g., headless test environment)
                cls._instance = super().__new__(cls)
                cls._instance._state = AppState()
                cls._instance._signals = StateSignals()
                cls._instance._subscribers = []
                cls._instance._undo_stack = []
                cls._instance._redo_stack = []
                cls._instance._max_undo_size = 50
                cls._instance._lock = RLock()  # Thread safety lock
                # R10-STO-004: per-thread reentrancy guard — threading.local
                # ensures that a worker-thread dispatch does not block the
                # main-thread dispatch or vice versa, while still preventing
                # recursive subscriber notification within the same thread.
                cls._instance._dispatch_tls = threading.local()
                logger.debug(
                    "Initializing store", class_name=cls.__name__, max_undo_size=50
                )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        # GUI-R6-006: Acquire _singleton_lock for the entire operation to
        # prevent races where another thread calls instance() between the
        # check and clear (mirrors WorkerPool.reset() pattern).
        with cls._singleton_lock:
            logger.debug("Resetting store singleton", class_name=cls.__name__)
            cls._instance = None

    def get_state(self) -> AppState:
        """Get the current application state (read-only).

        Thread-safe: Protected by RLock.
        """
        with self._lock:
            return self._state.clone()

    def set_signals(self, signals: Any) -> None:
        """Set the Qt signals object for state change notifications.

        Thread-safe: Protected by RLock to prevent a worker thread from reading
        a partially-assigned signals object while the main thread writes it.

        Parameters
        ----------
        signals : StateSignals
            StateSignals instance with Qt signals
        """
        logger.debug("Setting signals object", signals_type=type(signals).__name__)
        # R12-C-009: Wrap assignment in lock so concurrent readers never observe
        # a partially-initialised signals object.
        with self._lock:
            self._signals = signals

    @property
    def signals(self) -> Any:
        """Get the Qt signals object.

        Returns
        -------
        StateSignals or None
            StateSignals instance if set, None otherwise
        """
        with self._lock:
            return self._signals

    def _atomic_state_and_signal(
        self,
        updater: Callable[[AppState], AppState],
        signal_name: str,
        *signal_args: Any,
        track_undo: bool = True,
    ) -> None:
        """Atomically update state and emit a domain signal.

        Wraps ``update_state()`` and ``emit_signal()`` so that no concurrent
        dispatch can interleave between the state mutation and the signal
        emission (R12-C-002 fix).

        Parameters
        ----------
        updater : Callable[[AppState], AppState]
            Pure function producing the next state.
        signal_name : str
            Name of the domain signal to emit after state mutation.
        *signal_args : Any
            Arguments forwarded to the signal's ``emit()`` method.
        track_undo : bool
            Whether to record the pre-mutation state on the undo stack.
        """
        with self._lock:
            self.update_state(updater, track_undo=track_undo, emit_signal=False)
        # Emit signals outside the lock to prevent reentrant deadlocks.
        # The general state_changed signal is emitted first, then the
        # domain-specific signal so subscribers see the updated state.
        self.emit_signal("state_changed")
        self.emit_signal(signal_name, *signal_args)

    def emit_signal(self, signal_name: str, *args) -> None:
        """Emit a named signal if available.

        Thread-safe: if called from a non-main thread, defers emission to the
        main-thread event loop via QTimer.singleShot to prevent cross-thread
        signal delivery violations (R10-STO-001).

        Parameters
        ----------
        signal_name : str
            Name of the signal attribute on the StateSignals instance.
        *args
            Arguments forwarded to the signal's ``emit()`` method.
        """
        if not self._signals or not hasattr(self._signals, signal_name):
            return
        signal = getattr(self._signals, signal_name)

        # R10-STO-001: ensure domain signals are always emitted on the main thread.
        try:
            from PySide6.QtCore import QThread
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None and QThread.currentThread() != app.thread():
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, lambda s=signal, a=args: s.emit(*a))
                return
        except (ImportError, RuntimeError):
            pass  # Qt unavailable (headless/test) — fall through to direct emit

        signal.emit(*args)

    def dispatch(self, action: Any, payload: Any | None = None) -> None:
        """Dispatch an action to update state.

        Supports two call patterns:
        - dispatch({"type": "SET_THEME", "theme": "dark"})
        - dispatch("SET_THEME", {"theme": "dark"})

        Note: State mutation is atomic (protected by RLock), but subscriber
        notifications and Qt signal emissions occur outside the lock.
        Signal ordering is not guaranteed to be strictly sequential with
        respect to concurrent dispatches.  Subscribers should read state
        via ``get_state()`` rather than relying on signal arguments to
        avoid observing stale values from interleaved dispatches.

        Parameters
        ----------
        action : Any
            Action to dispatch. If dict with 'type' key, processes as action.
            If str, combined with payload into an action dict.
        payload : Any, optional
            Optional payload when using string-based dispatch.
        """
        # R8-NEW-007: warn if dispatched from non-main thread
        _worker_thread_violation = False
        _action_label = ""
        try:
            from PySide6.QtCore import QThread
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app and QThread.currentThread() != app.thread():
                import logging as _logging

                _action_label = (
                    action if isinstance(action, str) else action.get("type", "?")
                )
                _logging.getLogger(__name__).warning(
                    "dispatch() called from worker thread for action '%s'. "
                    "Use QMetaObject.invokeMethod for thread safety.",
                    _action_label,
                )
                _worker_thread_violation = True
        except (ImportError, RuntimeError):
            pass
        # R11-STO-002: Hard-fail in debug mode to catch thread violations
        # (raised outside the try/except so it is not swallowed)
        if _worker_thread_violation:
            import os

            if os.environ.get("RHEOJAX_DEBUG"):
                raise RuntimeError(
                    f"dispatch('{_action_label}') called from worker thread. "
                    "Use QMetaObject.invokeMethod for thread safety."
                )

        if isinstance(action, str):
            if isinstance(payload, dict):
                action = {"type": action, **payload}
            elif payload is None:
                action = {"type": action}
            else:
                action = {"type": action, "payload": payload}

        if isinstance(action, dict) and "type" in action:
            action_type = action["type"]
            payload_keys = [k for k in action if k != "type"]
            logger.debug(
                "Dispatching action",
                action_type=action_type,
                payload_keys=payload_keys,
            )
            reducer = self._reduce_action(action_type, action)
            # STORE-001: capture the active dataset id before state mutation
            # so we can emit dataset_removed with the correct id after deletion.
            _pre_delete_dataset_id: str = ""
            if action_type == "DELETE_SELECTED_DATASET":
                with self._lock:
                    _pre_delete_dataset_id = self._state.active_dataset_id or ""
            if reducer is not None:
                # STATE-009 enforcement: CACHE_STEP_RESULT must never push
                # large JAX arrays onto the undo stack regardless of what
                # the call-site passes.
                _track = action_type != "CACHE_STEP_RESULT"
                self.update_state(reducer, track_undo=_track, emit_signal=True)

            # Domain signals are emitted intentionally outside the lock to avoid
            # deadlocks: a signal consumer that calls dispatch() or get_state()
            # would deadlock if the lock were still held here.  Consumers must
            # therefore snapshot the values they need from the action payload
            # directly rather than re-reading store state from within a signal
            # handler, as a concurrent dispatch may have already advanced
            # self._state by the time the handler runs.
            if action_type == "SET_ACTIVE_MODEL":
                model_name = action.get("model_name", "")
                self.emit_signal("model_selected", model_name)

            elif action_type == "START_FITTING":
                model_name = action.get("model_name", "")
                dataset_id = action.get("dataset_id", "")
                self.emit_signal("fit_started", model_name, dataset_id)

            elif action_type == "FIT_PROGRESS":
                progress = action.get("progress", 0)
                job_id = action.get("job_id", "")
                self.emit_signal("fit_progress", job_id, int(progress))

            elif action_type == "FITTING_COMPLETED":
                self.emit_signal(
                    "fit_completed",
                    str(action.get("model_name", "")),
                    str(action.get("dataset_id", "")),
                )

            elif action_type == "FITTING_FAILED":
                error = action.get("error", "")
                model_name = action.get("model_name", "")
                dataset_id = action.get("dataset_id", "")
                # GUI-R6-008: Removed exc_info=True — there is no active
                # exception here; this is a dispatch handler, not a catch block.
                logger.error(
                    "Fitting failed",
                    action_type=action_type,
                    model_name=model_name,
                    dataset_id=dataset_id,
                    error=error,
                )
                # F-005 fix: pass model_name and dataset_id (not empty strings)
                self.emit_signal("fit_failed", model_name, dataset_id, error)

            elif action_type == "START_BAYESIAN":
                model_name = action.get("model_name", "")
                dataset_id = action.get("dataset_id", "")
                self.emit_signal("bayesian_started", model_name, dataset_id)

            elif action_type == "BAYESIAN_PROGRESS":
                progress = action.get("progress", 0)
                self.emit_signal("bayesian_progress", "", int(progress))

            elif action_type == "BAYESIAN_COMPLETED":
                # Symmetric with FITTING_COMPLETED: emit the signal so callers
                # using only bayesian_completed() action creator get notified.
                # STORE_BAYESIAN_RESULT also emits this signal, but duplicate
                # emissions are harmless (UI idempotent).
                self.emit_signal(
                    "bayesian_completed",
                    str(action.get("model_name", "")),
                    str(action.get("dataset_id", "")),
                )

            elif action_type == "BAYESIAN_FAILED":
                error = action.get("error", "")
                # GUI-R6-005: Extract model_name/dataset_id so subscribers can
                # update the correct UI state (empty strings caused stale "in
                # progress" indicators).
                model_name = action.get("model_name", "")
                dataset_id = action.get("dataset_id", "")
                # GUI-R6-008: Removed exc_info=True — no active exception here
                logger.error(
                    "Bayesian inference failed",
                    action_type=action_type,
                    model_name=model_name,
                    dataset_id=dataset_id,
                    error=error,
                )
                self.emit_signal("bayesian_failed", model_name, dataset_id, error)

            elif action_type == "STORE_BAYESIAN_RESULT":
                payload = action.get("payload", action)
                model_name = payload.get("model_name", "")
                dataset_id = payload.get("dataset_id", "")
                if model_name and dataset_id:
                    self.emit_signal(
                        "bayesian_completed", str(model_name), str(dataset_id)
                    )

            elif action_type == "SET_THEME":
                theme = action.get("theme", "light")
                self.emit_signal("theme_changed", theme)

            elif action_type == "SET_OS_THEME":
                os_theme = action.get("os_theme", "light")
                self.emit_signal("os_theme_changed", os_theme)

            elif action_type == "SET_PIPELINE_STEP":
                step = action.get("step", "")
                status = action.get("status", "")
                self.emit_signal("pipeline_step_changed", step, status)

            elif action_type == "TRANSFORM_APPLIED":
                transform = action.get("transform", "")
                dataset_id = action.get("dataset_id", "")
                self.emit_signal("transform_applied", transform, dataset_id)

            elif action_type == "IMPORT_DATA_SUCCESS":
                dataset_id = action.get("dataset_id")
                if dataset_id:
                    self.emit_signal("dataset_added", dataset_id)
                    self.emit_signal("dataset_selected", dataset_id)

            elif action_type == "DELETE_SELECTED_DATASET":
                # STORE-001: Emit dataset_removed with the id that was active
                # *before* the reducer ran (captured above).  Subscribers such
                # as the data tree use this to purge the deleted entry.
                self.emit_signal("dataset_removed", _pre_delete_dataset_id)

            elif action_type == "IMPORT_DATA_FAILED":
                error = action.get("error", "")
                # GUI-R6-008: Removed exc_info=True — no active exception here
                logger.error(
                    "Data import failed",
                    action_type=action_type,
                    error=error,
                )

            elif action_type == "ADD_PIPELINE_STEP":
                # STATE-003: Emit specific signal before the general structural one
                # so subscribers that track individual steps can react correctly.
                sc = action.get("step_config")
                if sc is not None:
                    self.emit_signal("pipeline_step_added", sc.id)
                self.emit_signal("pipeline_structure_changed")

            elif action_type == "REMOVE_PIPELINE_STEP":
                # STATE-003: Emit specific removal signal before the general one.
                step_id = action.get("step_id", "")
                self.emit_signal("pipeline_step_removed", step_id)
                self.emit_signal("pipeline_structure_changed")

            elif action_type == "REORDER_PIPELINE_STEP":
                self.emit_signal("pipeline_structure_changed")

            elif action_type == "SELECT_PIPELINE_STEP":
                step_id = action.get("step_id") or ""
                self.emit_signal("pipeline_step_selected", step_id)

            elif action_type == "UPDATE_STEP_CONFIG":
                # STATE-002: Emit pipeline_step_config_changed so subscribers
                # (e.g. step config panels) can refresh when config is updated.
                step_id = action.get("step_id", "")
                self.emit_signal("pipeline_step_config_changed", step_id)

            elif action_type == "UPDATE_STEP_STATUS":
                step_id = action.get("step_id", "")
                status = action.get("status")
                status_name = status.name if hasattr(status, "name") else str(status)
                self.emit_signal("pipeline_step_status_changed", step_id, status_name)

            elif action_type == "SET_PIPELINE_RUNNING":
                is_running = action.get("is_running", False)
                if is_running:
                    self.emit_signal("pipeline_execution_started")
                else:
                    self.emit_signal("pipeline_execution_completed")

            elif action_type == "SET_PIPELINE_NAME":
                name = action.get("name", "")
                self.emit_signal("pipeline_name_changed", name)

    def update_state(
        self,
        updater: Callable[[AppState], AppState],
        track_undo: bool = True,
        emit_signal: bool = True,
    ) -> None:
        """Update state immutably with undo tracking.

        Thread-safe: Protected by RLock.

        Parameters
        ----------
        updater : Callable
            Function that takes current state and returns new state
        track_undo : bool
            Whether to add current state to undo stack
        emit_signal : bool
            Whether to emit state_changed signal
        """
        with self._lock:
            old_state = self._state
            if track_undo and len(self._undo_stack) < self._max_undo_size:
                self._undo_stack.append(self._state.clone())
                self._redo_stack.clear()  # Clear redo on new action

            self._state = updater(self._state)

            # Compute changed keys for logging
            changed_keys = self._get_changed_keys(old_state, self._state)
            if changed_keys:
                logger.debug(
                    "State updated",
                    changed_keys=changed_keys,
                    track_undo=track_undo,
                    emit_signal=emit_signal,
                )

            # Copy subscriber list to avoid mutation during iteration
            subscribers = list(self._subscribers)
            state_snapshot = self._state.clone()

        # Notify subscribers outside the lock to prevent deadlocks.
        # Pass a deep clone (snapshot) so that subscribers cannot mutate
        # the live state object. Nested dispatches triggered by a subscriber
        # will still update self._state and be visible to subsequent reads.
        # IMPORTANT (GUI-003): The snapshot passed to subscribers may become
        # stale if a subscriber triggers a nested dispatch that further mutates
        # self._state.  Subscribers that need the latest state must call
        # self.get_state() rather than relying on the snapshot argument.
        #
        # GUI-P1-002 fix: if update_state() is called from a worker thread,
        # defer subscriber notifications to the main Qt thread via
        # QMetaObject.invokeMethod(QueuedConnection).  Subscriber callbacks
        # update Qt widgets and MUST run on the main thread.
        #
        # R12-C-003: The subscriber list is captured once (above, inside the
        # lock); subscribe/unsubscribe calls during iteration take effect on
        # the next dispatch, not the current one.
        #
        # R10-STO-004: per-thread reentrancy guard — if a subscriber triggers
        # a nested dispatch (which calls update_state again on the SAME thread),
        # we queue the nested snapshot for delivery after the outer loop finishes
        # instead of silently dropping it.  Using threading.local ensures that
        # a worker-thread dispatch does not interfere with a concurrent
        # main-thread dispatch.
        _on_main_thread = True
        try:
            from PySide6.QtCore import QThread
            from PySide6.QtWidgets import QApplication

            _app = QApplication.instance()
            if _app is not None and QThread.currentThread() != _app.thread():
                _on_main_thread = False
        except (ImportError, RuntimeError):
            pass

        if not _on_main_thread:
            # Worker thread: defer subscriber calls to GUI thread.
            try:
                from PySide6.QtCore import QMetaObject, Qt

                # M2 fix: append to deque instead of overwriting single-slot buffer
                self._signals._pending_notifications.append(
                    (subscribers, state_snapshot)
                )
                QMetaObject.invokeMethod(
                    self._signals,
                    "_run_subscriber_notifications",
                    Qt.ConnectionType.QueuedConnection,
                )
                # Fall through to emit_signal block below.
            except (ImportError, RuntimeError, AttributeError) as exc:
                logger.warning(
                    "Failed to defer subscriber notifications to GUI thread; "
                    "falling back to synchronous dispatch",
                    error=str(exc),
                )
                _on_main_thread = True  # Fall back to synchronous path

        if _on_main_thread:
            tls = self._dispatch_tls
            if getattr(tls, "dispatching", False):
                # Queue nested state change for delivery after outer loop
                if not hasattr(tls, "pending"):
                    tls.pending = []
                tls.pending.append((subscribers, state_snapshot))
                return
            tls.dispatching = True
            try:
                for subscriber in subscribers:
                    try:
                        subscriber(state_snapshot)
                    except Exception:
                        logger.error(
                            "Subscriber callback failed",
                            subscriber=getattr(subscriber, "__name__", str(subscriber)),
                            exc_info=True,
                        )
                # Drain any nested dispatches that were queued during this loop
                while hasattr(tls, "pending") and tls.pending:
                    pending_subs, pending_snap = tls.pending.pop(0)
                    for subscriber in pending_subs:
                        try:
                            subscriber(pending_snap)
                        except Exception:
                            logger.error(
                                "Subscriber callback failed (queued)",
                                subscriber=getattr(
                                    subscriber, "__name__", str(subscriber)
                                ),
                                exc_info=True,
                            )
            finally:
                tls.dispatching = False
                if hasattr(tls, "pending"):
                    tls.pending.clear()

        # Emit Qt signal via QueuedConnection to ensure delivery on the main
        # thread regardless of which thread calls update_state() (GUI-005).
        # QueuedConnection posts the call to the event loop of the signal's
        # owning thread (main thread) so it is safe to call from workers.
        if emit_signal:
            try:
                from PySide6.QtCore import QMetaObject, Qt

                QMetaObject.invokeMethod(
                    self._signals,
                    "_emit_state_changed",
                    Qt.ConnectionType.QueuedConnection,
                )
            except (ImportError, RuntimeError, AttributeError, TypeError):
                # Fall back to direct emit when Qt is unavailable (headless/tests)
                self.emit_signal("state_changed")

    def _get_changed_keys(self, old_state: AppState, new_state: AppState) -> list[str]:
        """Compute which top-level keys changed between two states."""
        changed = []
        for attr in [
            "project_path",
            "project_name",
            "is_modified",
            "datasets",
            "active_dataset_id",
            "active_model_name",
            "model_params",
            "fit_results",
            "bayesian_results",
            "current_tab",
            "pipeline_state",
            "jax_device",
            "jax_memory_used",
            "jax_memory_total",
            "transform_history",
            "workflow_mode",
            "current_seed",
            "auto_save_enabled",
            "theme",
            "os_theme",
            "last_export_dir",
            "recent_projects",
            "deformation_mode",
            "poisson_ratio",
            "visual_pipeline",
        ]:
            old_val = getattr(old_state, attr, None)
            new_val = getattr(new_state, attr, None)
            if old_val is new_val:
                continue
            try:
                if old_val != new_val:
                    changed.append(attr)
            except (ValueError, TypeError):
                changed.append(attr)
        return changed

    def _reduce_action(
        self, action_type: str, action: dict[str, Any]
    ) -> Callable[[AppState], AppState] | None:
        """Translate an action into a state updater.

        Delegates to sub-reducer modules via the REDUCER_DISPATCH table.
        A small number of actions that require access to the store instance
        (UNDO, REDO, UPDATE_PREFERENCES) are handled inline before the
        table lookup.
        """
        from rheojax.gui.state.reducers import REDUCER_DISPATCH, SIGNAL_ONLY_ACTIONS
        from rheojax.gui.state.reducers.ui_reducers import reduce_update_preferences

        # --- Store-dependent actions (need `self`) ---

        if action_type == "UNDO":
            # R12-C-005: UNDO bypasses update_state() — undo() acquires the
            # lock and runs subscriber notifications itself via the TLS guard.
            # Returning None tells dispatch() not to call update_state() again.
            self.undo()
            return None

        if action_type == "REDO":
            # R12-C-005: Same as UNDO — redo() manages its own lock and
            # subscriber notification path.
            self.redo()
            return None

        if action_type == "UPDATE_PREFERENCES":
            return reduce_update_preferences(action, self)

        # --- Signal-only actions (no state mutation) ---

        if action_type in SIGNAL_ONLY_ACTIONS:
            return None

        # --- Dispatch-table lookup ---

        reducer_fn = REDUCER_DISPATCH.get(action_type)
        if reducer_fn is not None:
            return reducer_fn(action)

        logger.warning(
            "Unhandled action type — no reducer registered",
            action_type=action_type,
        )
        return None

    def subscribe(self, callback: Callable[[AppState], None]) -> None:
        """Subscribe to state changes.

        Thread-safe: Protected by RLock.

        Parameters
        ----------
        callback : Callable
            Function called with new state on every update
        """
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
                logger.debug(
                    "Subscriber added",
                    subscriber=getattr(callback, "__name__", str(callback)),
                    total_subscribers=len(self._subscribers),
                )

    def unsubscribe(self, callback: Callable[[AppState], None]) -> None:
        """Unsubscribe from state changes.

        Thread-safe: Protected by RLock.

        Parameters
        ----------
        callback : Callable
            Previously subscribed callback
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
                logger.debug(
                    "Subscriber removed",
                    subscriber=getattr(callback, "__name__", str(callback)),
                    total_subscribers=len(self._subscribers),
                )

    def undo(self) -> bool:
        """Undo the last state change.

        Thread-safe: Protected by RLock.

        Note (R12-C-006): undo/redo emit state_changed but do NOT re-emit
        domain signals (dataset_added, fit_completed, etc.).  Subscribers
        should read from the store rather than relying on domain signals for
        state tracking, as domain signals are only emitted by the original
        action dispatchers and will not fire again on undo/redo.

        Returns
        -------
        bool
            True if undo was successful, False if nothing to undo
        """
        with self._lock:
            if not self._undo_stack:
                logger.debug("Undo requested but stack is empty")
                return False

            # Push current state to redo
            self._redo_stack.append(self._state.clone())

            # Restore previous state
            self._state = self._undo_stack.pop()

            logger.debug(
                "State undone",
                undo_stack_size=len(self._undo_stack),
                redo_stack_size=len(self._redo_stack),
            )

            # Copy subscribers list and capture snapshot inside the lock
            subscribers = list(self._subscribers)
            state_snapshot = self._state.clone()

        # R11-STO-001: Notify subscribers outside the lock using the same TLS
        # reentrancy guard as update_state() to prevent recursive subscriber
        # notification when a subscriber triggers a nested dispatch.
        tls = self._dispatch_tls
        if getattr(tls, "dispatching", False):
            if not hasattr(tls, "pending"):
                tls.pending = []
            tls.pending.append((subscribers, state_snapshot))
        else:
            tls.dispatching = True
            try:
                for subscriber in subscribers:
                    try:
                        subscriber(state_snapshot)
                    except Exception:
                        logger.error(
                            "Subscriber callback failed during undo",
                            subscriber=getattr(subscriber, "__name__", str(subscriber)),
                            exc_info=True,
                        )
                while hasattr(tls, "pending") and tls.pending:
                    pending_subs, pending_snap = tls.pending.pop(0)
                    for sub in pending_subs:
                        try:
                            sub(pending_snap)
                        except Exception:
                            logger.error(
                                "Subscriber callback failed (queued)",
                                subscriber=getattr(sub, "__name__", str(sub)),
                                exc_info=True,
                            )
            finally:
                tls.dispatching = False
                if hasattr(tls, "pending"):
                    tls.pending.clear()

        # Emit via QueuedConnection for thread safety (matches update_state path)
        try:
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                self._signals,
                "_emit_state_changed",
                Qt.ConnectionType.QueuedConnection,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError):
            self.emit_signal("state_changed")

        return True

    def redo(self) -> bool:
        """Redo the last undone state change.

        Thread-safe: Protected by RLock.

        Returns
        -------
        bool
            True if redo was successful, False if nothing to redo
        """
        with self._lock:
            if not self._redo_stack:
                logger.debug("Redo requested but stack is empty")
                return False

            # Push current state to undo
            self._undo_stack.append(self._state.clone())

            # Restore next state
            self._state = self._redo_stack.pop()

            logger.debug(
                "State redone",
                undo_stack_size=len(self._undo_stack),
                redo_stack_size=len(self._redo_stack),
            )

            # Copy subscribers list and capture snapshot inside the lock
            subscribers = list(self._subscribers)
            state_snapshot = self._state.clone()

        # R11-STO-001: Same TLS reentrancy guard as undo() and update_state().
        tls = self._dispatch_tls
        if getattr(tls, "dispatching", False):
            if not hasattr(tls, "pending"):
                tls.pending = []
            tls.pending.append((subscribers, state_snapshot))
        else:
            tls.dispatching = True
            try:
                for subscriber in subscribers:
                    try:
                        subscriber(state_snapshot)
                    except Exception:
                        logger.error(
                            "Subscriber callback failed during redo",
                            subscriber=getattr(subscriber, "__name__", str(subscriber)),
                            exc_info=True,
                        )
                while hasattr(tls, "pending") and tls.pending:
                    pending_subs, pending_snap = tls.pending.pop(0)
                    for sub in pending_subs:
                        try:
                            sub(pending_snap)
                        except Exception:
                            logger.error(
                                "Subscriber callback failed (queued)",
                                subscriber=getattr(sub, "__name__", str(sub)),
                                exc_info=True,
                            )
            finally:
                tls.dispatching = False
                if hasattr(tls, "pending"):
                    tls.pending.clear()

        # Emit via QueuedConnection for thread safety (matches update_state path)
        try:
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                self._signals,
                "_emit_state_changed",
                Qt.ConnectionType.QueuedConnection,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError):
            self.emit_signal("state_changed")

        return True

    def can_undo(self) -> bool:
        """Check if undo is available. Thread-safe."""
        with self._lock:
            return bool(self._undo_stack)

    def can_redo(self) -> bool:
        """Check if redo is available. Thread-safe."""
        with self._lock:
            return bool(self._redo_stack)

    def clear_history(self) -> None:
        """Clear undo/redo history. Thread-safe."""
        with self._lock:
            self._undo_stack.clear()
            self._redo_stack.clear()
            logger.debug("Undo/redo history cleared")

    def set_max_undo_size(self, size: int) -> None:
        """Set the maximum undo stack depth. Thread-safe."""
        with self._lock:
            self._max_undo_size = max(10, min(size, 500))
            # Trim if current stack exceeds new limit
            while len(self._undo_stack) > self._max_undo_size:
                self._undo_stack.pop(0)
            logger.debug("Max undo size updated", max_undo_size=self._max_undo_size)

    def batch_update(
        self, updaters: list[Callable[[AppState], AppState]], track_undo: bool = True
    ) -> None:
        """Apply multiple state updates in a single transaction.

        Thread-safe: Protected by RLock.

        This is more efficient than multiple update_state() calls as it only
        emits one signal at the end.

        Parameters
        ----------
        updaters : list[Callable]
            List of update functions to apply in sequence
        track_undo : bool
            Whether to track this batch as one undo action
        """
        logger.debug(
            "Starting batch update",
            num_updaters=len(updaters),
            track_undo=track_undo,
        )
        with self._lock:
            old_state = self._state
            if track_undo and len(self._undo_stack) < self._max_undo_size:
                self._undo_stack.append(self._state.clone())
                self._redo_stack.clear()

            # Apply all updates (rollback on failure)
            for updater in updaters:
                try:
                    self._state = updater(self._state)
                except Exception:
                    logger.error(
                        "Batch updater failed, rolling back",
                        updater=getattr(updater, "__name__", str(updater)),
                        exc_info=True,
                    )
                    self._state = old_state
                    if track_undo and self._undo_stack:
                        self._undo_stack.pop()
                    raise

            # Compute changed keys for logging
            changed_keys = self._get_changed_keys(old_state, self._state)
            if changed_keys:
                logger.debug(
                    "Batch update completed",
                    num_updaters=len(updaters),
                    changed_keys=changed_keys,
                )

            # Copy subscriber list and capture snapshot inside the lock to
            # prevent stale references if a nested dispatch replaces
            # self._state while subscribers are being notified.
            subscribers = list(self._subscribers)
            state_snapshot = self._state.clone()

        # M4 fix: mirror update_state() worker-thread deferral for batch_update().
        # Subscriber callbacks update Qt widgets and MUST run on the main thread.
        _on_main_thread = True
        try:
            from PySide6.QtCore import QThread
            from PySide6.QtWidgets import QApplication

            _app = QApplication.instance()
            if _app is not None and QThread.currentThread() != _app.thread():
                _on_main_thread = False
        except (ImportError, RuntimeError):
            pass

        if not _on_main_thread:
            try:
                from PySide6.QtCore import QMetaObject, Qt

                self._signals._pending_notifications.append(
                    (subscribers, state_snapshot)
                )
                QMetaObject.invokeMethod(
                    self._signals,
                    "_run_subscriber_notifications",
                    Qt.ConnectionType.QueuedConnection,
                )
            except (ImportError, RuntimeError, AttributeError) as exc:
                logger.warning(
                    "Failed to defer batch subscriber notifications; "
                    "falling back to synchronous dispatch",
                    error=str(exc),
                )
                _on_main_thread = True

        if _on_main_thread:
            # Notify subscribers using the TLS reentrancy guard to prevent
            # recursive subscriber notification (R12-C-004).
            tls = self._dispatch_tls
            if getattr(tls, "dispatching", False):
                if not hasattr(tls, "pending"):
                    tls.pending = []
                tls.pending.append((subscribers, state_snapshot))
            else:
                tls.dispatching = True
                try:
                    for subscriber in subscribers:
                        try:
                            subscriber(state_snapshot)
                        except Exception:
                            logger.error(
                                "Subscriber callback failed during batch update",
                                subscriber=getattr(
                                    subscriber, "__name__", str(subscriber)
                                ),
                                exc_info=True,
                            )
                    while hasattr(tls, "pending") and tls.pending:
                        pending_subs, pending_snap = tls.pending.pop(0)
                        for subscriber in pending_subs:
                            try:
                                subscriber(pending_snap)
                            except Exception:
                                logger.error(
                                    "Subscriber callback failed in batch_update (pending)",
                                    subscriber=getattr(
                                        subscriber, "__name__", str(subscriber)
                                    ),
                                    exc_info=True,
                                )
                finally:
                    tls.dispatching = False
                    if hasattr(tls, "pending"):
                        tls.pending.clear()

        # R8-NEW-004: use QueuedConnection for thread safety (matches update_state pattern)
        try:
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                self._signals,
                "_emit_state_changed",
                Qt.ConnectionType.QueuedConnection,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError):
            self.emit_signal("state_changed")

    def get_dataset(self, dataset_id: str) -> DatasetState | None:
        """Get dataset by ID.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        DatasetState | None
            Cloned dataset state or None if not found
        """
        with self._lock:
            ds = self._state.datasets.get(dataset_id)
            return ds.clone() if ds is not None else None

    def get_active_dataset(self) -> DatasetState | None:
        """Get the currently active dataset.

        Returns
        -------
        DatasetState | None
            Cloned active dataset state or None
        """
        with self._lock:
            if self._state.active_dataset_id:
                ds = self._state.datasets.get(self._state.active_dataset_id)
                return ds.clone() if ds is not None else None
            return None

    def get_fit_result(self, key: str) -> FitResult | None:
        """Get fit result by key.

        Parameters
        ----------
        key : str
            Result key (typically f"{model_name}_{dataset_id}")

        Returns
        -------
        FitResult | None
            Cloned fit result or None if not found
        """
        with self._lock:
            fr = self._state.fit_results.get(key)
            return fr.clone() if fr is not None else None

    def get_bayesian_result(self, key: str) -> BayesianResult | None:
        """Get Bayesian result by key.

        Parameters
        ----------
        key : str
            Result key (typically f"{model_name}_{dataset_id}")

        Returns
        -------
        BayesianResult | None
            Cloned Bayesian result or None if not found
        """
        with self._lock:
            br = self._state.bayesian_results.get(key)
            return br.clone() if br is not None else None

    def get_active_fit_result(self) -> FitResult | None:
        """Return cloned fit result for the active model/dataset combo."""
        with self._lock:
            state = self._state
            if not state.active_model_name or not state.active_dataset_id:
                return None
            key = f"{state.active_model_name}_{state.active_dataset_id}"
            fr = state.fit_results.get(key)
            return fr.clone() if fr is not None else None

    def get_active_bayesian_result(self) -> BayesianResult | None:
        """Return cloned Bayesian result for the active model/dataset combo."""
        with self._lock:
            state = self._state
            if not state.active_model_name or not state.active_dataset_id:
                return None
            key = f"{state.active_model_name}_{state.active_dataset_id}"
            br = state.bayesian_results.get(key)
            return br.clone() if br is not None else None

    def get_pipeline_state(self) -> PipelineState:
        """Return cloned pipeline state."""
        with self._lock:
            return self._state.pipeline_state.clone()
