"""Fitting action reducers.

Handles: START_FITTING, FITTING_COMPLETED, FITTING_FAILED, STORE_FIT_RESULT.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rheojax.gui.state.store import AppState

from rheojax.gui.state.store import (
    FitResult,
    PipelineStep,
    StepStatus,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


def reduce_start_fitting(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def updater(state: AppState) -> AppState:
        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.FIT] = StepStatus.ACTIVE
        pipeline.current_step = PipelineStep.FIT
        return replace(state, pipeline_state=pipeline)

    return updater


def reduce_fitting_completed(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    # STATE-004: Do not mutate pipeline step status here.
    # STORE_FIT_RESULT is the single source of truth for FIT=COMPLETE,
    # which prevents duplicate undo entries from both actions writing the
    # same state transition.  This action exists solely to trigger the
    # fit_completed signal in the dispatch() signal chain.
    #
    # INVARIANT: Every caller that dispatches FITTING_COMPLETED must
    # dispatch STORE_FIT_RESULT first (see main_window.py
    # _on_job_completed).  If STORE_FIT_RESULT is omitted, the step
    # will silently remain non-COMPLETE.
    return lambda state: state


def reduce_fitting_failed(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def updater(state: AppState) -> AppState:
        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.FIT] = StepStatus.ERROR
        pipeline.current_step = PipelineStep.FIT
        return replace(state, pipeline_state=pipeline)

    return updater


def reduce_store_fit_result(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
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
                parameters=getattr(result, "parameters", {}),
                chi_squared=float(getattr(result, "chi_squared", 0.0)),
                success=bool(getattr(result, "success", True)),
                message=str(getattr(result, "message", "")),
                timestamp=getattr(result, "timestamp", datetime.now()),
                dataset_id=str(dataset_id),
                r_squared=float(getattr(result, "r_squared", 0.0)),
                mpe=float(getattr(result, "mpe", 0.0)),
                fit_time=float(getattr(result, "fit_time", 0.0)),
                num_iterations=getattr(
                    result,
                    "num_iterations",
                    getattr(result, "n_iterations", 0) or 0,
                ),
                convergence_message=getattr(result, "message", ""),
                x_fit=getattr(result, "x_fit", None),
                y_fit=getattr(result, "y_fit", None),
                residuals=getattr(result, "residuals", None),
                # STORE-005: Propagate uncertainty/info-criterion fields
                # so that export and diagnostics can access them.
                pcov=getattr(result, "pcov", None),
                rmse=getattr(result, "rmse", None),
                mae=getattr(result, "mae", None),
                aic=getattr(result, "aic", None),
                bic=getattr(result, "bic", None),
                metadata=getattr(result, "metadata", None),
            )
        )
        fits = state.fit_results.copy()
        key = f"{model_name}_{dataset_id}"
        fits[key] = fit_state

        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.FIT] = StepStatus.COMPLETE
        pipeline.current_step = PipelineStep.FIT

        # R10-STO-003: Do NOT overwrite active_dataset_id / active_model_name
        # here -- the user may have switched selection while the worker was
        # running, and clobbering their choice causes silent TOCTOU bugs.
        # Downstream UI components that need the fit context should use
        # get_fit_result(key) with the submitted identifiers directly.
        return replace(
            state,
            fit_results=fits,
            pipeline_state=pipeline,
            is_modified=True,
        )

    return updater
