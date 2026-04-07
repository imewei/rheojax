"""Bayesian action reducers.

Handles: START_BAYESIAN, BAYESIAN_COMPLETED, BAYESIAN_FAILED,
STORE_BAYESIAN_RESULT.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rheojax.gui.state.store import AppState

from rheojax.gui.state.store import (
    BayesianResult,
    PipelineStep,
    StepStatus,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


def reduce_start_bayesian(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def updater(state: AppState) -> AppState:
        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.BAYESIAN] = StepStatus.ACTIVE
        pipeline.current_step = PipelineStep.BAYESIAN
        return replace(state, pipeline_state=pipeline)

    return updater


def reduce_bayesian_completed(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    # STATE-004: Same pattern -- STORE_BAYESIAN_RESULT owns BAYESIAN=COMPLETE.
    # BAYESIAN_COMPLETED exists only to drive the bayesian_completed signal.
    #
    # INVARIANT: same as FITTING_COMPLETED above -- callers must dispatch
    # STORE_BAYESIAN_RESULT before (or instead of) this action.
    return lambda state: state


def reduce_bayesian_failed(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def updater(state: AppState) -> AppState:
        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.BAYESIAN] = StepStatus.ERROR
        pipeline.current_step = PipelineStep.BAYESIAN
        return replace(state, pipeline_state=pipeline)

    return updater


def reduce_store_bayesian_result(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
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
                r_hat=diag.get("r_hat", {}) or getattr(result, "r_hat", {}),
                ess=diag.get("ess", {}) or getattr(result, "ess", {}),
                divergences=int(
                    diag.get("divergences")
                    if diag.get("divergences") is not None
                    else getattr(result, "divergences", 0)
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
                # STORE-004: Propagate diagnostics_valid so that the UI
                # can distinguish valid R-hat/ESS from NaN fallbacks.
                # M-6: When diag exists but lacks the key, fall through
                # to the result-level attribute to avoid defaulting True.
                diagnostics_valid=bool(
                    diag["diagnostics_valid"]
                    if diag and "diagnostics_valid" in diag
                    else getattr(result, "diagnostics_valid", True)
                ),
            )

        bayes = state.bayesian_results.copy()
        key = f"{model_name}_{dataset_id}"
        bayes[key] = bayes_state

        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.BAYESIAN] = StepStatus.COMPLETE
        pipeline.current_step = PipelineStep.BAYESIAN

        # R8-NEW-009: don't overwrite active_model_name / active_dataset_id
        # on Bayesian completion -- the user may have switched selection
        # while inference was running in the background.
        return replace(
            state,
            bayesian_results=bayes,
            pipeline_state=pipeline,
            is_modified=True,
        )

    return updater
