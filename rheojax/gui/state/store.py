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
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from threading import RLock
from typing import Any, Optional


class PipelineStep(Enum):
    """Pipeline execution steps."""

    LOAD = auto()
    TRANSFORM = auto()
    FIT = auto()
    BAYESIAN = auto()
    EXPORT = auto()


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
        """Create a deep copy of this dataset state."""
        return replace(
            self,
            metadata=copy.deepcopy(self.metadata),
            x_data=self.x_data,  # Keep reference to JAX arrays
            y_data=self.y_data,
            y2_data=self.y2_data,
        )


@dataclass
class FitResult:
    """Result from NLSQ point estimation fit."""

    model_name: str
    dataset_id: str
    parameters: dict[str, float]
    r_squared: float
    mpe: float
    chi_squared: float
    fit_time: float
    timestamp: datetime
    num_iterations: int = 0
    convergence_message: str = ""

    def clone(self) -> "FitResult":
        """Create a deep copy of this fit result."""
        return replace(self, parameters=copy.deepcopy(self.parameters))


@dataclass
class BayesianResult:
    """Result from Bayesian NUTS inference."""

    model_name: str
    dataset_id: str
    posterior_samples: Any | None  # ArviZ InferenceData
    r_hat: dict[str, float]
    ess: dict[str, float]
    divergences: int
    credible_intervals: dict[str, tuple[float, float]]
    mcmc_time: float
    timestamp: datetime
    num_warmup: int = 1000
    num_samples: int = 2000

    def clone(self) -> "BayesianResult":
        """Create a deep copy of this Bayesian result."""
        return replace(
            self,
            r_hat=copy.deepcopy(self.r_hat),
            ess=copy.deepcopy(self.ess),
            credible_intervals=copy.deepcopy(self.credible_intervals),
            posterior_samples=self.posterior_samples,  # Keep reference
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

    # JAX
    jax_device: str = "cpu"
    jax_memory_used: int = 0
    jax_memory_total: int = 0

    # Provenance
    transform_history: list[TransformRecord] = field(default_factory=list)

    # Settings
    current_seed: int = 42
    auto_save_enabled: bool = True
    theme: str = "light"
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

    # Class-level type annotations for instance attributes (singleton pattern)
    _state: AppState
    _signals: Any
    _subscribers: list[Callable[[AppState], None]]
    _undo_stack: list[AppState]
    _redo_stack: list[AppState]
    _max_undo_size: int
    _lock: RLock

    def __new__(cls) -> "StateStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._state = AppState()
            cls._instance._signals = None  # Will be set by GUI
            cls._instance._subscribers = []
            cls._instance._undo_stack = []
            cls._instance._redo_stack = []
            cls._instance._max_undo_size = 50
            cls._instance._lock = RLock()  # Thread safety lock
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def get_state(self) -> AppState:
        """Get the current application state (read-only).

        Thread-safe: Protected by RLock.
        """
        with self._lock:
            return self._state

    def set_signals(self, signals: Any) -> None:
        """Set the Qt signals object for state change notifications.

        Parameters
        ----------
        signals : StateSignals
            StateSignals instance with Qt signals
        """
        self._signals = signals

    @property
    def signals(self) -> Any:
        """Get the Qt signals object.

        Returns
        -------
        StateSignals or None
            StateSignals instance if set, None otherwise
        """
        return self._signals

    def dispatch(self, action: Any) -> None:
        """Dispatch an action to update state.

        This is a convenience method for Redux-style action dispatching.
        Actions can be dicts with a 'type' key, or any other value.

        Parameters
        ----------
        action : Any
            Action to dispatch. If dict with 'type' key, processes as action.
            Otherwise, treated as a simple state notification.
        """
        if isinstance(action, dict) and "type" in action:
            action_type = action["type"]

            # Handle specific action types
            if action_type == "SET_ACTIVE_MODEL":
                model_name = action.get("model_name", "")
                if self._signals:
                    self._signals.model_selected.emit(model_name)

            elif action_type == "START_FITTING":
                model_name = action.get("model_name", "")
                dataset_id = action.get("dataset_id", "")
                if self._signals:
                    self._signals.fit_started.emit(model_name, dataset_id)

            elif action_type == "FIT_PROGRESS":
                progress = action.get("progress", 0)
                if self._signals:
                    self._signals.fit_progress.emit("", int(progress))

            elif action_type == "FITTING_COMPLETED":
                if self._signals:
                    self._signals.fit_completed.emit("", "")

            elif action_type == "FITTING_FAILED":
                error = action.get("error", "")
                if self._signals:
                    self._signals.fit_failed.emit("", "", error)

            elif action_type == "START_BAYESIAN":
                model_name = action.get("model_name", "")
                dataset_id = action.get("dataset_id", "")
                if self._signals:
                    self._signals.bayesian_started.emit(model_name, dataset_id)

            elif action_type == "BAYESIAN_PROGRESS":
                progress = action.get("progress", 0)
                if self._signals:
                    self._signals.bayesian_progress.emit("", int(progress))

            elif action_type == "BAYESIAN_COMPLETED":
                if self._signals:
                    self._signals.bayesian_completed.emit("", "")

            elif action_type == "BAYESIAN_FAILED":
                error = action.get("error", "")
                if self._signals:
                    self._signals.bayesian_failed.emit("", "", error)

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
            if track_undo and len(self._undo_stack) < self._max_undo_size:
                self._undo_stack.append(self._state.clone())
                self._redo_stack.clear()  # Clear redo on new action

            self._state = updater(self._state)

            # Copy subscribers list to avoid modification during iteration
            subscribers = list(self._subscribers)

        # Notify subscribers outside the lock to prevent deadlocks
        for subscriber in subscribers:
            subscriber(self._state)

        # Emit Qt signal if available
        if emit_signal and self._signals is not None:
            self._signals.state_changed.emit()

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

    def undo(self) -> bool:
        """Undo the last state change.

        Thread-safe: Protected by RLock.

        Returns
        -------
        bool
            True if undo was successful, False if nothing to undo
        """
        with self._lock:
            if not self._undo_stack:
                return False

            # Push current state to redo
            self._redo_stack.append(self._state.clone())

            # Restore previous state
            self._state = self._undo_stack.pop()

            # Copy subscribers list
            subscribers = list(self._subscribers)

        # Notify subscribers outside the lock
        for subscriber in subscribers:
            subscriber(self._state)

        if self._signals is not None:
            self._signals.state_changed.emit()

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
                return False

            # Push current state to undo
            self._undo_stack.append(self._state.clone())

            # Restore next state
            self._state = self._redo_stack.pop()

            # Copy subscribers list
            subscribers = list(self._subscribers)

        # Notify subscribers outside the lock
        for subscriber in subscribers:
            subscriber(self._state)

        if self._signals is not None:
            self._signals.state_changed.emit()

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
        with self._lock:
            if track_undo and len(self._undo_stack) < self._max_undo_size:
                self._undo_stack.append(self._state.clone())
                self._redo_stack.clear()

            # Apply all updates
            for updater in updaters:
                self._state = updater(self._state)

            # Copy subscribers list
            subscribers = list(self._subscribers)

        # Notify subscribers outside the lock
        for subscriber in subscribers:
            subscriber(self._state)

        if self._signals is not None:
            self._signals.state_changed.emit()

    def get_dataset(self, dataset_id: str) -> DatasetState | None:
        """Get dataset by ID.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        DatasetState | None
            Dataset state or None if not found
        """
        return self._state.datasets.get(dataset_id)

    def get_active_dataset(self) -> DatasetState | None:
        """Get the currently active dataset.

        Returns
        -------
        DatasetState | None
            Active dataset state or None
        """
        if self._state.active_dataset_id:
            return self._state.datasets.get(self._state.active_dataset_id)
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
            Fit result or None if not found
        """
        return self._state.fit_results.get(key)

    def get_bayesian_result(self, key: str) -> BayesianResult | None:
        """Get Bayesian result by key.

        Parameters
        ----------
        key : str
            Result key (typically f"{model_name}_{dataset_id}")

        Returns
        -------
        BayesianResult | None
            Bayesian result or None if not found
        """
        return self._state.bayesian_results.get(key)
