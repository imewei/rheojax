"""Pipeline action reducers (classic workflow + visual pipeline).

Classic pipeline: SET_PIPELINE_STEP, APPLY_TRANSFORM, TRANSFORM_COMPLETED,
EXPORT_RESULTS.

Visual pipeline: ADD_PIPELINE_STEP, REMOVE_PIPELINE_STEP,
REORDER_PIPELINE_STEP, SELECT_PIPELINE_STEP, UPDATE_STEP_CONFIG,
UPDATE_STEP_STATUS, CACHE_STEP_RESULT, SET_PIPELINE_RUNNING,
SET_PIPELINE_NAME, CLEAR_PIPELINE, LOAD_PIPELINE.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rheojax.gui.state.store import AppState

from rheojax.gui.state.store import (
    PipelineStep,
    StepStatus,
    TransformRecord,
    VisualPipelineState,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


# --- Classic pipeline reducers ---


def reduce_set_pipeline_step(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
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


def reduce_apply_transform(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def updater(state: AppState) -> AppState:
        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.TRANSFORM] = StepStatus.ACTIVE
        pipeline.current_step = PipelineStep.TRANSFORM
        return replace(state, pipeline_state=pipeline, is_modified=True)

    return updater


def reduce_transform_completed(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
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


def reduce_export_results(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
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


# --- Visual pipeline reducers ---


def reduce_add_pipeline_step(
    action: dict[str, Any],
) -> Callable[[AppState], AppState] | None:
    step_config = action.get("step_config")
    # STATE-001: Guard against missing step_config to prevent AttributeError
    # on None.clone() when the action is dispatched without a step_config key.
    if step_config is None:
        return lambda state: state

    def _add_step(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        vp.steps.append(step_config.clone())
        for i, s in enumerate(vp.steps):
            s.position = i
        return replace(state, visual_pipeline=vp, is_modified=True)

    return _add_step


def reduce_remove_pipeline_step(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    step_id = action.get("step_id")

    def _remove_step(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        vp.steps = [s for s in vp.steps if s.id != step_id]
        for i, s in enumerate(vp.steps):
            s.position = i
        if vp.selected_step_id == step_id:
            vp.selected_step_id = None
        if step_id in vp.step_results:
            del vp.step_results[step_id]
        return replace(state, visual_pipeline=vp, is_modified=True)

    return _remove_step


def reduce_reorder_pipeline_step(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    step_id = action.get("step_id")
    new_position = action.get("new_position")
    # STATE-006: Guard against None/non-int new_position. The step would
    # have already been removed from the list before list.insert() raises
    # TypeError, causing permanent data loss without this guard.
    if new_position is None or not isinstance(new_position, int):
        return lambda state: state

    def _reorder_step(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        step = next((s for s in vp.steps if s.id == step_id), None)
        if step is None:
            return state
        vp.steps.remove(step)
        vp.steps.insert(new_position, step)
        for i, s in enumerate(vp.steps):
            s.position = i
        # When reordering, clear ALL cached results -- pipeline semantics
        # change fundamentally and any cached output may be invalid.
        vp.step_results.clear()
        return replace(state, visual_pipeline=vp, is_modified=True)

    return _reorder_step


def reduce_select_pipeline_step(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    step_id = action.get("step_id")

    def _select_step(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        vp.selected_step_id = step_id
        return replace(state, visual_pipeline=vp)

    return _select_step


def reduce_update_step_config(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    step_id = action.get("step_id")
    config_updates = action.get("config", {})

    def _update_config(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        for step in vp.steps:
            if step.id == step_id:
                step.config.update(config_updates)
                break
        idx = next((i for i, s in enumerate(vp.steps) if s.id == step_id), None)
        if idx is not None:
            # Clear the edited step's cached result (config changed
            # -> old result is stale), but preserve its status so an
            # in-progress execution is not disrupted.
            vp.step_results.pop(step_id, None)
            # STATE-007: Use idx+1 so only *downstream* steps are
            # invalidated. The edited step keeps its status.
            for s in vp.steps[idx + 1 :]:
                vp.step_results.pop(s.id, None)
                if s.status == StepStatus.COMPLETE:
                    s.status = StepStatus.PENDING
        return replace(state, visual_pipeline=vp, is_modified=True)

    return _update_config


def reduce_update_step_status(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    step_id = action.get("step_id")
    status = action.get("status")
    error_message = action.get("error_message")

    def _update_status(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        for step in vp.steps:
            if step.id == step_id:
                step.status = status
                step.error_message = error_message
                break
        return replace(state, visual_pipeline=vp)

    return _update_status


def reduce_cache_step_result(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    step_id = action.get("step_id")
    result = action.get("result")

    def _cache_result(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        vp.step_results[step_id] = result
        return replace(state, visual_pipeline=vp)

    return _cache_result


def reduce_set_pipeline_running(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    is_running = action.get("is_running", False)
    current_step_id = action.get("current_step_id")

    def _set_running(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        vp.is_running = is_running
        vp.current_running_step_id = current_step_id
        return replace(state, visual_pipeline=vp)

    return _set_running


def reduce_set_pipeline_name(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    name = action.get("name", "Untitled Pipeline")

    def _set_name(state: AppState) -> AppState:
        vp = state.visual_pipeline.clone()
        vp.pipeline_name = name
        return replace(state, visual_pipeline=vp, is_modified=True)

    return _set_name


def reduce_clear_pipeline(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def _clear_pipeline(state: AppState) -> AppState:
        return replace(
            state,
            visual_pipeline=VisualPipelineState(),
            is_modified=True,
        )

    return _clear_pipeline


def reduce_load_pipeline(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    visual_pipeline = action.get("visual_pipeline")

    def _load_pipeline(state: AppState) -> AppState:
        return replace(
            state,
            visual_pipeline=visual_pipeline.clone(),
            is_modified=False,
        )

    return _load_pipeline
