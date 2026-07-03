from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from rheojax.gui.foundation.library import DatasetLibrary


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
    nlsq_result: Any | None = None  # FitResult from ModelService.fit(); typed Any to avoid circular import
    nuts_result: dict | None = None
    step: int = 0
    revision: int = 0

@dataclass
class TransformState:
    transform_key: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    result: dict | None = None
    step: int = 0
    revision: int = 0

@dataclass
class PipelineStepConfig:
    id: str
    step_type: str            # "transform" | "fit" | "export"
    config: dict[str, Any] = field(default_factory=dict)

@dataclass
class JobResultRef:
    result_id: str

@dataclass
class ActiveJobsState:
    by_id: dict[str, dict] = field(default_factory=dict)

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
    "ParameterConfig", "NlsqConfig", "NutsConfig", "FitState", "TransformState",
    "PipelineStepConfig", "JobResultRef", "ActiveJobsState", "JobHistoryState",
    "PipelineState", "UiState", "ProjectState", "AppState", "replace",
]
