"""Model action reducers.

Handles: SET_ACTIVE_MODEL, CANCEL_JOBS.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rheojax.gui.state.store import AppState

from rheojax.gui.state.store import StepStatus
from rheojax.logging import get_logger

logger = get_logger(__name__)


def reduce_set_active_model(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    model_name = action.get("model_name")
    model_params = action.get("model_params")

    def updater(state: AppState) -> AppState:
        kwargs: dict[str, Any] = {
            "active_model_name": model_name,
            "is_modified": True,
        }
        if model_params is not None:
            kwargs["model_params"] = model_params
        return replace(state, **kwargs)

    return updater


def reduce_cancel_jobs(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def updater(state: AppState) -> AppState:
        pipeline = state.pipeline_state.clone()
        if pipeline.current_step:
            pipeline.steps[pipeline.current_step] = StepStatus.WARNING
        pipeline.error_message = "Cancelled by user"
        pipeline.current_step = None
        return replace(state, pipeline_state=pipeline, is_modified=True)

    return updater
