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
import uuid
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
            summary=self.summary,  # Summary dict — reference (read-only)
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

    # DMTA / Deformation
    deformation_mode: str = "shear"  # shear, tension, bending, compression
    poisson_ratio: float = 0.5  # rubber default

    # Settings
    workflow_mode: WorkflowMode = WorkflowMode.FITTING
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
                cls._instance = super().__new__(cls)
                cls._instance._state = AppState()
                cls._instance._signals = StateSignals()
                cls._instance._subscribers = []
                cls._instance._undo_stack = []
                cls._instance._redo_stack = []
                cls._instance._max_undo_size = 50
                cls._instance._lock = RLock()  # Thread safety lock
                logger.debug(
                    "Initializing store", class_name=cls.__name__, max_undo_size=50
                )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
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

        Parameters
        ----------
        signals : StateSignals
            StateSignals instance with Qt signals
        """
        logger.debug("Setting signals object", signals_type=type(signals).__name__)
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

    def emit_signal(self, signal_name: str, *args) -> None:
        """Emit a named signal if available.

        Parameters
        ----------
        signal_name : str
            Name of the signal attribute on the StateSignals instance.
        *args
            Arguments forwarded to the signal's ``emit()`` method.
        """
        if self._signals and hasattr(self._signals, signal_name):
            getattr(self._signals, signal_name).emit(*args)

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
        if isinstance(action, str):
            if isinstance(payload, dict):
                action = {"type": action, **payload}
            elif payload is None:
                action = {"type": action}
            else:
                action = {"type": action, "payload": payload}

        if isinstance(action, dict) and "type" in action:
            action_type = action["type"]
            logger.debug("Dispatching action", action_type=action_type)
            reducer = self._reduce_action(action_type, action)
            if reducer is not None:
                self.update_state(reducer, emit_signal=True)

            # Emit relevant signals for UI reactivity
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
                logger.error(
                    "Fitting failed",
                    action_type=action_type,
                    model_name=model_name,
                    dataset_id=dataset_id,
                    error=error,
                    exc_info=True,
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
                # Signal emission handled by STORE_BAYESIAN_RESULT to avoid
                # duplicate bayesian_completed signals when both actions fire.
                pass

            elif action_type == "BAYESIAN_FAILED":
                error = action.get("error", "")
                logger.error(
                    "Bayesian inference failed",
                    action_type=action_type,
                    error=error,
                    exc_info=True,
                )
                self.emit_signal("bayesian_failed", "", "", error)

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

            elif action_type == "IMPORT_DATA_FAILED":
                error = action.get("error", "")
                logger.error(
                    "Data import failed",
                    action_type=action_type,
                    error=error,
                    exc_info=True,
                )

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

        # Notify subscribers outside the lock to prevent deadlocks.
        # Always read self._state (not a snapshot) so that if a subscriber
        # triggers a nested dispatch, remaining subscribers see the latest
        # state rather than a stale pre-nested-dispatch snapshot.
        for subscriber in subscribers:
            try:
                subscriber(self._state)
            except Exception:
                logger.error(
                    "Subscriber callback failed",
                    subscriber=getattr(subscriber, "__name__", str(subscriber)),
                    exc_info=True,
                )

        # Emit Qt signal if available
        if emit_signal:
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
            "last_export_dir",
            "recent_projects",
            "deformation_mode",
            "poisson_ratio",
        ]:
            old_val = getattr(old_state, attr, None)
            new_val = getattr(new_state, attr, None)
            if old_val != new_val:
                changed.append(attr)
        return changed

    def _reduce_action(
        self, action_type: str, action: dict[str, Any]
    ) -> Callable[[AppState], AppState] | None:
        """Translate an action into a state updater.

        Only a minimal set of actions are implemented here to keep the GUI
        responsive while the full reducer pattern is built out.
        """

        if action_type == "SET_THEME":
            theme = action.get("theme", "light")

            def updater(state: AppState) -> AppState:
                if state.theme == theme:
                    return state
                return replace(state, theme=theme, is_modified=True)

            return updater

        if action_type == "SET_WORKFLOW_MODE":
            mode = action.get("mode")

            def updater(state: AppState) -> AppState:
                new_mode = WorkflowMode[mode.upper()] if isinstance(mode, str) else mode
                if isinstance(new_mode, WorkflowMode):
                    return replace(state, workflow_mode=new_mode)
                return state

            return updater

        if action_type == "SET_DEFORMATION_MODE":
            mode = action.get("deformation_mode", "shear")

            def updater(state: AppState) -> AppState:
                if state.deformation_mode == mode:
                    return state
                return replace(state, deformation_mode=mode, is_modified=True)

            return updater

        if action_type == "SET_POISSON_RATIO":
            ratio = action.get("poisson_ratio", 0.5)

            def updater(state: AppState) -> AppState:
                if state.poisson_ratio == ratio:
                    return state
                return replace(state, poisson_ratio=ratio, is_modified=True)

            return updater

        if action_type == "SET_TEST_MODE":
            mode = action.get("test_mode", "oscillation")
            target_id = action.get("dataset_id")

            def updater(state: AppState) -> AppState:
                dataset_id = target_id or state.active_dataset_id
                datasets = state.datasets.copy()
                if dataset_id and dataset_id in datasets:
                    ds = datasets[dataset_id].clone()
                    ds.test_mode = mode
                    ds.metadata = {**ds.metadata, "test_mode": mode}
                    ds.is_modified = True
                    datasets[dataset_id] = ds
                # Only update datasets — do NOT change current_tab as a side-effect.
                # Tab navigation should be an explicit user action, not implicit.
                return replace(
                    state, datasets=datasets, is_modified=True
                )

            return updater

        if action_type == "AUTO_DETECT_TEST_MODE":

            def updater(state: AppState) -> AppState:
                dataset_id = state.active_dataset_id
                if not dataset_id or dataset_id not in state.datasets:
                    return state

                ds = state.datasets[dataset_id].clone()
                if ds.x_data is None or ds.y_data is None:
                    return state
                try:
                    from rheojax.core.data import RheoData
                    from rheojax.gui.services.data_service import DataService

                    svc = DataService()
                    inferred = svc.detect_test_mode(
                        RheoData(
                            x=ds.x_data,
                            y=ds.y_data,
                            y_units=None,
                            x_units=None,
                            domain=ds.metadata.get("domain", "time"),
                            metadata=ds.metadata,
                            validate=False,
                        )
                    )
                    if inferred:
                        ds.test_mode = inferred
                        ds.metadata = {**ds.metadata, "test_mode": inferred}
                        ds.is_modified = True
                except Exception:
                    logger.error(
                        "Auto-detect test mode failed",
                        dataset_id=dataset_id,
                        exc_info=True,
                    )
                    # Leave dataset unchanged on failure
                    return state

                datasets = state.datasets.copy()
                datasets[dataset_id] = ds
                return replace(
                    state, datasets=datasets, current_tab="data", is_modified=True
                )

            return updater

        if action_type == "CANCEL_JOBS":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                if pipeline.current_step:
                    pipeline.steps[pipeline.current_step] = StepStatus.WARNING
                pipeline.error_message = "Cancelled by user"
                pipeline.current_step = None
                return replace(state, pipeline_state=pipeline, is_modified=True)

            return updater

        if action_type == "SET_ACTIVE_MODEL":
            model_name = action.get("model_name")

            def updater(state: AppState) -> AppState:
                return replace(state, active_model_name=model_name, is_modified=True)

            return updater

        if action_type == "SET_TAB" or action_type == "NAVIGATE_TAB":
            default_tab = getattr(self, "_state", AppState()).current_tab
            tab = action.get("tab", default_tab)

            def updater(state: AppState) -> AppState:
                return replace(state, current_tab=tab)

            return updater

        if action_type == "SET_PIPELINE_STEP":
            step = action.get("step")
            status = action.get("status")

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                try:
                    step_enum = PipelineStep[step.upper()] if step else None
                except (KeyError, AttributeError):
                    valid = [s.name for s in PipelineStep]
                    logger.error(
                        "Invalid pipeline step",
                        step=step,
                        valid_steps=valid,
                    )
                    step_enum = None
                try:
                    status_enum = StepStatus[status.upper()] if status else None
                except (KeyError, AttributeError):
                    valid = [s.name for s in StepStatus]
                    logger.error(
                        "Invalid step status",
                        status=status,
                        valid_statuses=valid,
                    )
                    status_enum = None
                if step_enum and status_enum:
                    pipeline.steps[step_enum] = status_enum
                    pipeline.current_step = step_enum
                return replace(state, pipeline_state=pipeline)

            return updater

        if action_type == "START_FITTING":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.FIT] = StepStatus.ACTIVE
                pipeline.current_step = PipelineStep.FIT
                return replace(state, pipeline_state=pipeline)

            return updater

        if action_type == "START_BAYESIAN":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.BAYESIAN] = StepStatus.ACTIVE
                pipeline.current_step = PipelineStep.BAYESIAN
                return replace(state, pipeline_state=pipeline)

            return updater

        if action_type == "LOAD_PROJECT":
            path = Path(action.get("file_path")) if action.get("file_path") else None

            def updater(state: AppState) -> AppState:
                recent = list(state.recent_projects)
                if path and path not in recent:
                    recent = [path] + recent[:9]
                return replace(
                    state,
                    project_path=path,
                    project_name=path.name if path else state.project_name,
                    recent_projects=recent,
                )

            return updater

        if action_type == "NEW_PROJECT":

            def updater(state: AppState) -> AppState:
                return AppState()

            return updater

        if action_type == "UNDO":
            self.undo()
            return None

        if action_type == "REDO":
            self.redo()
            return None

        if action_type == "IMPORT_DATA":
            config = action.get("payload", action)

            def updater(state: AppState) -> AppState:
                # Mark pipeline as active for load step
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.LOAD] = StepStatus.ACTIVE
                pipeline.current_step = PipelineStep.LOAD
                return replace(state, pipeline_state=pipeline, is_modified=True)

            return updater

        if action_type == "IMPORT_DATA_SUCCESS":
            config = action.get("payload", action)
            dataset_id = config.get("dataset_id") or str(uuid.uuid4())
            name = config.get("name", "Dataset")
            test_mode = config.get("test_mode", "oscillation")
            file_path = (
                Path(config.get("file_path")) if config.get("file_path") else None
            )

            dataset = DatasetState(
                id=dataset_id,
                name=name,
                file_path=file_path,
                test_mode=test_mode,
                x_data=config.get("x_data"),
                y_data=config.get("y_data"),
                y2_data=config.get("y2_data"),
                metadata=config.get("metadata", {}),
                is_modified=False,
            )

            def updater(state: AppState) -> AppState:
                datasets = state.datasets.copy()
                datasets[dataset_id] = dataset
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.LOAD] = StepStatus.COMPLETE
                pipeline.current_step = PipelineStep.LOAD
                return replace(
                    state,
                    datasets=datasets,
                    active_dataset_id=dataset_id,
                    pipeline_state=pipeline,
                    is_modified=True,
                )

            return updater

        if action_type == "IMPORT_DATA_FAILED":
            error = action.get("error")

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.LOAD] = StepStatus.ERROR
                pipeline.error_message = error or pipeline.error_message
                pipeline.current_step = PipelineStep.LOAD
                return replace(state, pipeline_state=pipeline, is_modified=True)

            return updater

        if action_type == "SET_ACTIVE_DATASET":

            def updater(state: AppState) -> AppState:
                dataset_id = action.get("dataset_id")
                if dataset_id in state.datasets:
                    return replace(state, active_dataset_id=dataset_id)
                return state

            return updater

        if action_type == "APPLY_TRANSFORM":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.TRANSFORM] = StepStatus.ACTIVE
                pipeline.current_step = PipelineStep.TRANSFORM
                return replace(state, pipeline_state=pipeline, is_modified=True)

            return updater

        if action_type == "TRANSFORM_COMPLETED":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.TRANSFORM] = StepStatus.COMPLETE
                pipeline.current_step = PipelineStep.TRANSFORM
                history = list(state.transform_history)
                transform_name = action.get("transform_id") or action.get("transform")
                source = action.get("source_dataset_id")
                target = action.get("target_dataset_id")
                params = action.get("parameters", {})
                seed = action.get("seed")
                if transform_name and source and target:
                    history.append(
                        TransformRecord(
                            timestamp=datetime.now(),
                            source_dataset_id=str(source),
                            target_dataset_id=str(target),
                            transform_name=str(transform_name),
                            parameters=params,
                            seed=seed,
                        )
                    )
                return replace(
                    state, pipeline_state=pipeline, transform_history=history
                )

            return updater

        if action_type == "EXPORT_RESULTS":
            export_path = action.get("file_path")

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.EXPORT] = StepStatus.COMPLETE
                pipeline.current_step = PipelineStep.EXPORT
                last_export_dir = (
                    Path(export_path).parent if export_path else state.last_export_dir
                )
                return replace(
                    state, pipeline_state=pipeline, last_export_dir=last_export_dir
                )

            return updater

        if action_type == "SAVE_PROJECT":
            file_path = action.get("file_path")

            def updater(state: AppState) -> AppState:
                project_path = Path(file_path) if file_path else state.project_path
                recent = list(state.recent_projects)
                if project_path and project_path not in recent:
                    recent = [project_path] + recent[:9]
                return replace(
                    state,
                    project_path=project_path,
                    project_name=(
                        project_path.name if project_path else state.project_name
                    ),
                    is_modified=False,
                    recent_projects=recent,
                )

            return updater

        if action_type == "RECORD_PROVENANCE":
            payload = action.get("payload", action)
            record = payload.get("record")

            def updater(state: AppState) -> AppState:
                if not isinstance(record, TransformRecord):
                    return state
                history = list(state.transform_history)
                history.append(record.clone())
                return replace(state, transform_history=history, is_modified=True)

            return updater

        if action_type == "FITTING_COMPLETED":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.FIT] = StepStatus.COMPLETE
                pipeline.current_step = PipelineStep.FIT
                return replace(state, pipeline_state=pipeline)

            return updater

        if action_type == "BAYESIAN_COMPLETED":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.BAYESIAN] = StepStatus.COMPLETE
                pipeline.current_step = PipelineStep.BAYESIAN
                return replace(state, pipeline_state=pipeline)

            return updater

        if action_type == "FITTING_FAILED":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.FIT] = StepStatus.ERROR
                pipeline.current_step = PipelineStep.FIT
                return replace(state, pipeline_state=pipeline)

            return updater

        if action_type == "BAYESIAN_FAILED":

            def updater(state: AppState) -> AppState:
                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.BAYESIAN] = StepStatus.ERROR
                pipeline.current_step = PipelineStep.BAYESIAN
                return replace(state, pipeline_state=pipeline)

            return updater

        if action_type == "STORE_FIT_RESULT":
            payload = action.get("payload", action)
            model_name = payload.get("model_name")
            dataset_id = payload.get("dataset_id")
            result = payload.get("result")

            def updater(state: AppState) -> AppState:
                if not model_name or not dataset_id or result is None:
                    return state

                fit_state = (
                    result
                    if isinstance(result, FitResult)
                    else FitResult(
                        model_name=model_name,
                        dataset_id=str(dataset_id),
                        parameters=getattr(result, "parameters", {}),
                        r_squared=float(getattr(result, "r_squared", 0.0)),
                        mpe=float(getattr(result, "mpe", 0.0)),
                        chi_squared=float(getattr(result, "chi_squared", 0.0)),
                        fit_time=float(getattr(result, "fit_time", 0.0)),
                        timestamp=getattr(result, "timestamp", datetime.now()),
                        num_iterations=getattr(
                            result,
                            "num_iterations",
                            getattr(result, "n_iterations", 0) or 0,
                        ),
                        convergence_message=getattr(result, "message", ""),
                        x_fit=getattr(result, "x_fit", None),
                        y_fit=getattr(result, "y_fit", None),
                        residuals=getattr(result, "residuals", None),
                    )
                )
                fits = state.fit_results.copy()
                key = f"{model_name}_{dataset_id}"
                fits[key] = fit_state

                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.FIT] = StepStatus.COMPLETE
                pipeline.current_step = PipelineStep.FIT

                return replace(
                    state,
                    fit_results=fits,
                    active_dataset_id=dataset_id,
                    active_model_name=model_name,
                    pipeline_state=pipeline,
                    is_modified=True,
                )

            return updater

        if action_type == "STORE_BAYESIAN_RESULT":
            payload = action.get("payload", action)
            model_name = payload.get("model_name")
            dataset_id = payload.get("dataset_id")
            result = payload.get("result")

            def updater(state: AppState) -> AppState:
                if not model_name or not dataset_id or result is None:
                    return state

                if isinstance(result, BayesianResult):
                    bayes_state = result
                else:
                    # Extract diagnostics from nested dict if present
                    diag = getattr(result, "diagnostics", {}) or {}
                    bayes_state = BayesianResult(
                        model_name=model_name,
                        dataset_id=str(dataset_id),
                        posterior_samples=getattr(result, "posterior_samples", {}),
                        r_hat=diag.get("r_hat", diag.get("rhat", {})) or getattr(result, "r_hat", {}),
                        ess=diag.get("ess", {}) or getattr(result, "ess", {}),
                        divergences=int(
                            diag.get("divergences", 0)
                            or getattr(result, "divergences", 0)
                        ),
                        credible_intervals=getattr(result, "credible_intervals", {}),
                        mcmc_time=float(
                            getattr(
                                result,
                                "mcmc_time",
                                getattr(result, "sampling_time", 0.0),
                            )
                        ),
                        timestamp=getattr(result, "timestamp", datetime.now()),
                        summary=getattr(result, "summary", None),
                        num_warmup=int(getattr(result, "num_warmup", 0)),
                        num_samples=int(getattr(result, "num_samples", 0)),
                        num_chains=int(getattr(result, "num_chains", 4)),
                        inference_data=getattr(result, "inference_data", None),
                    )

                bayes = state.bayesian_results.copy()
                key = f"{model_name}_{dataset_id}"
                bayes[key] = bayes_state

                pipeline = state.pipeline_state.clone()
                pipeline.steps[PipelineStep.BAYESIAN] = StepStatus.COMPLETE
                pipeline.current_step = PipelineStep.BAYESIAN

                return replace(
                    state,
                    bayesian_results=bayes,
                    active_dataset_id=dataset_id,
                    active_model_name=model_name,
                    pipeline_state=pipeline,
                    is_modified=True,
                )

            return updater

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

            # Copy subscribers list
            subscribers = list(self._subscribers)

        # Notify subscribers outside the lock
        for subscriber in subscribers:
            try:
                subscriber(self._state)
            except Exception:
                logger.error(
                    "Subscriber callback failed during undo",
                    subscriber=getattr(subscriber, "__name__", str(subscriber)),
                    exc_info=True,
                )

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

            # Copy subscribers list
            subscribers = list(self._subscribers)

        # Notify subscribers outside the lock
        for subscriber in subscribers:
            try:
                subscriber(self._state)
            except Exception:
                logger.error(
                    "Subscriber callback failed during redo",
                    subscriber=getattr(subscriber, "__name__", str(subscriber)),
                    exc_info=True,
                )

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

            # Copy subscriber list to avoid mutation during iteration
            subscribers = list(self._subscribers)

        # Notify subscribers outside the lock (read self._state for freshness)
        for subscriber in subscribers:
            try:
                subscriber(self._state)
            except Exception:
                logger.error(
                    "Subscriber callback failed during batch update",
                    subscriber=getattr(subscriber, "__name__", str(subscriber)),
                    exc_info=True,
                )

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
            Dataset state or None if not found
        """
        with self._lock:
            return self._state.datasets.get(dataset_id)

    def get_active_dataset(self) -> DatasetState | None:
        """Get the currently active dataset.

        Returns
        -------
        DatasetState | None
            Active dataset state or None
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            return self._state.bayesian_results.get(key)

    def get_active_fit_result(self) -> FitResult | None:
        """Return the fit result for the active model/dataset combo."""
        with self._lock:
            state = self._state
            if not state.active_model_name or not state.active_dataset_id:
                return None
            key = f"{state.active_model_name}_{state.active_dataset_id}"
            return state.fit_results.get(key)

    def get_active_bayesian_result(self) -> BayesianResult | None:
        """Return the Bayesian result for the active model/dataset combo."""
        with self._lock:
            state = self._state
            if not state.active_model_name or not state.active_dataset_id:
                return None
            key = f"{state.active_model_name}_{state.active_dataset_id}"
            return state.bayesian_results.get(key)

    def get_pipeline_state(self) -> PipelineState:
        """Return the current pipeline state."""
        with self._lock:
            return self._state.pipeline_state
