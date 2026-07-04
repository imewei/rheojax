"""Runtime result/artifact shapes for Pipeline mode execution.

Distinct from the persisted AppState dataclasses in foundation/state.py: PhaseResult.result
holds a RAW result dict while a run is active/in-memory; PipelineBatchRunner._prepare_job_record
(batch_runner.py) is what converts it into a JobResultRef (foundation/state.py) for persistence
into AppState.job_history -- see design spec §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PhaseResult:
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class FitStepResult:
    nlsq: PhaseResult
    nuts: PhaseResult | None


@dataclass
class PipelineRunResult:
    step_results: dict[str, FitStepResult | dict] = field(default_factory=dict)
    status: Literal["completed", "failed", "cancelled"] = "completed"
    error: str | None = None


@dataclass
class DatasetArtifact:
    dataset_id: str
    source_step_id: str
    produced_at_revision: int


@dataclass
class FitArtifact:
    step_id: str
    source_dataset_id: str
    result: FitStepResult


@dataclass
class FileArtifact:
    step_id: str
    paths: list[str] = field(default_factory=list)
