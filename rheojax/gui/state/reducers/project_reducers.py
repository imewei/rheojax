"""Project action reducers.

Handles: LOAD_PROJECT, NEW_PROJECT, SAVE_PROJECT, RECORD_PROVENANCE.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rheojax.gui.state.store import AppState

from rheojax.gui.state.store import AppState as _AppState
from rheojax.gui.state.store import TransformRecord
from rheojax.logging import get_logger

logger = get_logger(__name__)


def reduce_load_project(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
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


def reduce_new_project(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def updater(state: AppState) -> AppState:
        # Reset only project-scoped fields; global/persistent settings
        # (recent_projects, theme, os_theme, auto_save_enabled, current_seed,
        # jax_device, ...) must survive across project lifetimes.
        defaults = _AppState()
        return replace(
            state,
            project_path=defaults.project_path,
            project_name=defaults.project_name,
            is_modified=defaults.is_modified,
            datasets=defaults.datasets,
            active_dataset_id=defaults.active_dataset_id,
            active_model_name=defaults.active_model_name,
            model_params=defaults.model_params,
            fit_results=defaults.fit_results,
            bayesian_results=defaults.bayesian_results,
            current_tab=defaults.current_tab,
            pipeline_state=defaults.pipeline_state,
            visual_pipeline=defaults.visual_pipeline,
            transform_history=defaults.transform_history,
            workflow_mode=defaults.workflow_mode,
        )

    return updater


def reduce_save_project(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    file_path = action.get("file_path")

    def updater(state: AppState) -> AppState:
        project_path = Path(file_path) if file_path else state.project_path
        recent = list(state.recent_projects)
        if project_path and project_path not in recent:
            recent = [project_path] + recent[:9]
        return replace(
            state,
            project_path=project_path,
            project_name=(project_path.name if project_path else state.project_name),
            is_modified=False,
            recent_projects=recent,
        )

    return updater


def reduce_record_provenance(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    payload = action.get("payload", action)
    record = payload.get("record")

    def updater(state: AppState) -> AppState:
        if not isinstance(record, TransformRecord):
            return state
        history = list(state.transform_history)
        history.append(record.clone())
        return replace(state, transform_history=history, is_modified=True)

    return updater
