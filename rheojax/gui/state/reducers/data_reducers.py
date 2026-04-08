"""Data action reducers.

Handles: SET_TEST_MODE, AUTO_DETECT_TEST_MODE, SET_ACTIVE_DATASET,
IMPORT_DATA, IMPORT_DATA_SUCCESS, IMPORT_DATA_FAILED,
DELETE_SELECTED_DATASET.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rheojax.gui.state.store import AppState

from rheojax.gui.state.store import (
    DatasetState,
    PipelineStep,
    StepStatus,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


def reduce_set_test_mode(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
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
        # Only update datasets -- do NOT change current_tab as a side-effect.
        # Tab navigation should be an explicit user action, not implicit.
        return replace(state, datasets=datasets, is_modified=True)

    return updater


def reduce_auto_detect_test_mode(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    target_id = action.get("dataset_id")
    inferred = action.get("inferred_mode")

    def updater(state: AppState) -> AppState:
        dataset_id = target_id or state.active_dataset_id
        if not dataset_id or dataset_id not in state.datasets:
            return state
        if not inferred:
            return state
        ds = state.datasets[dataset_id].clone()
        ds.test_mode = inferred
        ds.metadata = {**ds.metadata, "test_mode": inferred}
        ds.is_modified = True
        datasets = state.datasets.copy()
        datasets[dataset_id] = ds
        return replace(state, datasets=datasets, is_modified=True)

    return updater


def reduce_set_active_dataset(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    # Support both flat action and payload-wrapped dispatch (GUI-017)
    dataset_id = action.get("dataset_id") or (action.get("payload") or {}).get(
        "dataset_id"
    )

    def updater(state: AppState) -> AppState:
        if dataset_id and dataset_id in (state.datasets or {}):
            return replace(state, active_dataset_id=dataset_id)
        return state

    return updater


def reduce_import_data(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    def updater(state: AppState) -> AppState:
        # Mark pipeline as active for load step
        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.LOAD] = StepStatus.ACTIVE
        pipeline.current_step = PipelineStep.LOAD
        return replace(state, pipeline_state=pipeline, is_modified=True)

    return updater


def reduce_import_data_success(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    config = action.get("payload", action)
    dataset_id = config.get("dataset_id") or str(uuid.uuid4())
    name = config.get("name", "Dataset")
    test_mode = config.get("test_mode", "oscillation")
    file_path = Path(config.get("file_path")) if config.get("file_path") else None

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


def reduce_import_data_failed(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    error = action.get("error")

    def updater(state: AppState) -> AppState:
        pipeline = state.pipeline_state.clone()
        pipeline.steps[PipelineStep.LOAD] = StepStatus.ERROR
        pipeline.error_message = error or pipeline.error_message
        pipeline.current_step = PipelineStep.LOAD
        return replace(state, pipeline_state=pipeline, is_modified=True)

    return updater


def reduce_delete_selected_dataset(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    # STORE-001: reducer for deleting the currently active dataset.
    # Removes the dataset, its associated fit/Bayesian results, and
    # selects the next available dataset as active (or None).

    def updater(state: AppState) -> AppState:
        dataset_id = state.active_dataset_id
        if not dataset_id or dataset_id not in (state.datasets or {}):
            return state

        new_datasets = state.datasets.copy()
        del new_datasets[dataset_id]

        # Select the next dataset (first remaining) as the new active one
        new_active = next(iter(new_datasets.keys()), None)

        # Purge all fit/Bayesian results associated with the deleted dataset
        new_fit_results = {
            k: v for k, v in state.fit_results.items() if v.dataset_id != dataset_id
        }
        new_bayesian_results = {
            k: v
            for k, v in state.bayesian_results.items()
            if v.dataset_id != dataset_id
        }

        # M-3: Purge transform_history entries referencing the
        # deleted dataset to avoid dangling references.
        new_transform_history = [
            tr
            for tr in state.transform_history
            if tr.source_dataset_id != dataset_id and tr.target_dataset_id != dataset_id
        ]

        return replace(
            state,
            datasets=new_datasets,
            active_dataset_id=new_active,
            fit_results=new_fit_results,
            bayesian_results=new_bayesian_results,
            transform_history=new_transform_history,
            is_modified=True,
        )

    return updater
