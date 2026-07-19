from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from rheojax.gui.foundation.library import DatasetLibrary


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


@dataclass
class DatasetState:
    """State for a loaded dataset."""

    id: str
    name: str
    file_path: Path | None
    test_mode: str  # oscillation, relaxation, creep, rotation
    x_data: Any | None = None  # NumPy array
    y_data: Any | None = None
    y2_data: Any | None = None  # For G'' in oscillation
    metadata: dict = field(default_factory=dict)
    is_modified: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FitResult:
    """Canonical result from NLSQ point estimation fit.

    Single source of truth for fit results across the GUI -- fit_worker,
    model_service, and pages all import from here.
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


@dataclass
class ParameterConfig:
    name: str
    value: float
    lower: float
    upper: float
    fixed: bool


@dataclass
class NlsqConfig:
    multi_start: bool = False
    n_starts: int = 8
    parameters: list[ParameterConfig] = field(default_factory=list)


@dataclass
class NutsConfig:
    run_nuts: bool = True
    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 4
    target_accept: float = 0.8
    seed: int = 0
    warm_start: bool = True
    max_tree_depth: int | None = None
    priors: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitState:
    protocol: str | None = None
    model_key: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    data_ref: str | None = None
    column_map: dict[str, Any] = field(default_factory=dict)
    # ponytail: plumbed through the invalidation cascade (contract.py declares
    # per-protocol required names like "sigma0"/"gamma0") but no GUI step
    # writes to it, and no fit/sample call reads it -- real runs still get
    # protocol kwargs from data.metadata (ModelService.fit()'s
    # _PROTOCOL_KWARGS). Building a working control_vars UI needs the
    # contract.py names reconciled with ModelService's actual kwarg
    # vocabulary first (they don't match 1:1 today, e.g. "gamma_dot0" vs
    # "gdot"); do that reconciliation before wiring a widget to this field.
    control_vars: dict[str, float] = field(default_factory=dict)
    nlsq_config: NlsqConfig = field(default_factory=NlsqConfig)
    nuts_config: NutsConfig = field(default_factory=NutsConfig)
    nlsq_result: Any | None = (
        None  # FitResult from ModelService.fit(); typed Any to avoid circular import
    )
    nuts_result: dict | None = None
    step: int = 0
    revision: int = 0


@dataclass
class TransformState:
    transform_key: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    # Populated by SlotsStep's ParameterFormBuilder (step2_slots.py); RunStep
    # (step3_run.py) reads whatever this dict holds when it runs.
    config: dict[str, Any] = field(default_factory=dict)
    result: dict | None = None
    step: int = 0
    revision: int = 0


@dataclass
class PipelineStepConfig:
    id: str
    step_type: str  # "transform" | "fit" | "export"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResultRef:
    result_id: str


@dataclass
class ActiveJobsState:
    by_id: dict[str, dict] = field(default_factory=dict)
    # PipelineBatchRunner.run() writes into `by_id` from a QThreadPool worker
    # thread (see batch_runner.py) while GUI-thread code reads/iterates it --
    # this lock is the one piece of real cross-thread synchronization for that
    # shared dict; GUI-thread-only mutators (fit_controller.py, controller.py)
    # don't need it since the Qt event loop already serializes them.
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


@dataclass
class JobHistoryState:
    by_id: dict[str, dict] = field(default_factory=dict)


@dataclass
class PipelineState:
    steps: list[PipelineStepConfig] = field(default_factory=list)
    selected_dataset_ids: list[str] = field(default_factory=list)
    name: str | None = None
    job_id: str | None = None


@dataclass
class UiState:
    mode: str = "fit"
    theme: str = "system"
    inspector_tab: str = "log"


@dataclass
class ProjectState:
    path: str | None = None
    name: str | None = None
    dirty: bool = False


@dataclass
class AppState:
    library: DatasetLibrary = field(default_factory=DatasetLibrary)
    fit: FitState = field(default_factory=FitState)
    transform: TransformState = field(default_factory=TransformState)
    pipeline: PipelineState = field(default_factory=PipelineState)
    active_jobs: ActiveJobsState = field(default_factory=ActiveJobsState)
    job_history: JobHistoryState = field(default_factory=JobHistoryState)
    project: ProjectState = field(default_factory=ProjectState)
    ui: UiState = field(default_factory=UiState)


# re-export replace for invalidation.py
__all__ = [
    "ParameterConfig",
    "NlsqConfig",
    "NutsConfig",
    "FitState",
    "TransformState",
    "PipelineStepConfig",
    "JobResultRef",
    "ActiveJobsState",
    "JobHistoryState",
    "PipelineState",
    "UiState",
    "ProjectState",
    "AppState",
    "replace",
]
