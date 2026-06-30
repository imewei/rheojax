from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from rheojax.gui.foundation.library import DatasetLibrary


@dataclass
class FitState:
    protocol: str | None = None
    model_key: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    data_ref: str | None = None
    column_map: dict[str, Any] = field(default_factory=dict)
    control_vars: dict[str, float] = field(default_factory=dict)
    nlsq_config: dict = field(default_factory=dict)       # multi-start / solver settings
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
class JobsState:
    by_id: dict[str, dict] = field(default_factory=dict)

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
    jobs: JobsState = field(default_factory=JobsState)
    project: ProjectState = field(default_factory=ProjectState)
    ui: dict[str, Any] = field(default_factory=lambda: {"mode": "fit"})

# re-export replace for invalidation.py
__all__ = ["FitState", "TransformState", "JobsState", "ProjectState", "AppState", "replace"]
