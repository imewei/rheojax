"""State selectors for efficient queries.

This module provides computed properties and derived state queries
for the RheoJAX GUI state.
"""

from pathlib import Path

from rheojax.gui.state.store import (
    BayesianResult,
    DatasetState,
    FitResult,
    PipelineStep,
    StateStore,
    StepStatus,
)

# Dataset Selectors


def get_active_dataset() -> DatasetState | None:
    """Get the currently active dataset.

    Returns
    -------
    DatasetState | None
        Active dataset or None
    """
    store = StateStore()
    return store.get_active_dataset()


def get_dataset(dataset_id: str) -> DatasetState | None:
    """Get dataset by ID.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    DatasetState | None
        Dataset or None if not found
    """
    store = StateStore()
    return store.get_dataset(dataset_id)


def get_all_datasets() -> dict[str, DatasetState]:
    """Get all loaded datasets.

    Returns
    -------
    dict[str, DatasetState]
        All datasets
    """
    store = StateStore()
    return store.get_state().datasets


def get_dataset_count() -> int:
    """Get number of loaded datasets.

    Returns
    -------
    int
        Dataset count
    """
    store = StateStore()
    return len(store.get_state().datasets)


# Model Selectors


def get_active_model_name() -> str | None:
    """Get the currently selected model name.

    Returns
    -------
    str | None
        Model name or None
    """
    store = StateStore()
    return store.get_state().active_model_name


def get_model_param_dict() -> dict[str, float]:
    """Get current model parameters as a dict.

    Returns
    -------
    dict[str, float]
        Parameter name -> value mapping
    """
    store = StateStore()
    params = store.get_state().model_params
    return {name: param.value for name, param in params.items()}


def get_model_param_bounds() -> dict[str, tuple[float, float]]:
    """Get parameter bounds for current model.

    Returns
    -------
    dict[str, tuple[float, float]]
        Parameter name -> (min, max) mapping
    """
    store = StateStore()
    params = store.get_state().model_params
    return {name: (param.min_bound, param.max_bound) for name, param in params.items()}


# Fit Result Selectors


def get_active_fit_result() -> FitResult | None:
    """Get fit result for active model and dataset.

    Returns
    -------
    FitResult | None
        Fit result or None
    """
    store = StateStore()
    state = store.get_state()

    if not state.active_model_name or not state.active_dataset_id:
        return None

    key = f"{state.active_model_name}_{state.active_dataset_id}"
    return state.fit_results.get(key)


def get_fit_result(model_name: str, dataset_id: str) -> FitResult | None:
    """Get fit result for specific model and dataset.

    Parameters
    ----------
    model_name : str
        Model name
    dataset_id : str
        Dataset identifier

    Returns
    -------
    FitResult | None
        Fit result or None
    """
    store = StateStore()
    key = f"{model_name}_{dataset_id}"
    return store.get_state().fit_results.get(key)


def get_all_fit_results() -> dict[str, FitResult]:
    """Get all fit results.

    Returns
    -------
    dict[str, FitResult]
        All fit results
    """
    store = StateStore()
    return store.get_state().fit_results


def is_fit_available() -> bool:
    """Check if a fit result is available for active model and dataset.

    Returns
    -------
    bool
        True if fit available
    """
    return get_active_fit_result() is not None


# Bayesian Result Selectors


def get_active_bayesian_result() -> BayesianResult | None:
    """Get Bayesian result for active model and dataset.

    Returns
    -------
    BayesianResult | None
        Bayesian result or None
    """
    store = StateStore()
    state = store.get_state()

    if not state.active_model_name or not state.active_dataset_id:
        return None

    key = f"{state.active_model_name}_{state.active_dataset_id}"
    return state.bayesian_results.get(key)


def get_bayesian_result(model_name: str, dataset_id: str) -> BayesianResult | None:
    """Get Bayesian result for specific model and dataset.

    Parameters
    ----------
    model_name : str
        Model name
    dataset_id : str
        Dataset identifier

    Returns
    -------
    BayesianResult | None
        Bayesian result or None
    """
    store = StateStore()
    key = f"{model_name}_{dataset_id}"
    return store.get_state().bayesian_results.get(key)


def get_all_bayesian_results() -> dict[str, BayesianResult]:
    """Get all Bayesian results.

    Returns
    -------
    dict[str, BayesianResult]
        All Bayesian results
    """
    store = StateStore()
    return store.get_state().bayesian_results


def is_bayesian_available() -> bool:
    """Check if Bayesian result is available for active model and dataset.

    Returns
    -------
    bool
        True if Bayesian result available
    """
    return get_active_bayesian_result() is not None


# Pipeline Selectors


def get_pipeline_progress() -> float:
    """Calculate overall pipeline progress as a fraction.

    Returns
    -------
    float
        Progress from 0.0 to 1.0
    """
    store = StateStore()
    state = store.get_state()

    total_steps = len(PipelineStep)
    completed_steps = sum(
        1
        for status in state.pipeline_state.steps.values()
        if status == StepStatus.COMPLETE
    )

    return completed_steps / total_steps if total_steps > 0 else 0.0


def get_pipeline_step_status(step: PipelineStep) -> StepStatus:
    """Get status of a specific pipeline step.

    Parameters
    ----------
    step : PipelineStep
        Pipeline step

    Returns
    -------
    StepStatus
        Step status
    """
    store = StateStore()
    return store.get_state().pipeline_state.steps.get(step, StepStatus.PENDING)


def get_current_pipeline_step() -> PipelineStep | None:
    """Get the current active pipeline step.

    Returns
    -------
    PipelineStep | None
        Current step or None
    """
    store = StateStore()
    return store.get_state().pipeline_state.current_step


# Project Selectors


def get_recent_projects() -> list[Path]:
    """Get list of recent project paths.

    Returns
    -------
    list[Path]
        Recent project paths (most recent first)
    """
    store = StateStore()
    return store.get_state().recent_projects


def get_project_name() -> str:
    """Get current project name.

    Returns
    -------
    str
        Project name
    """
    store = StateStore()
    return store.get_state().project_name


def is_project_modified() -> bool:
    """Check if project has unsaved changes.

    Returns
    -------
    bool
        True if modified
    """
    store = StateStore()
    return store.get_state().is_modified


# JAX Selectors


def get_jax_device() -> str:
    """Get current JAX device.

    Returns
    -------
    str
        Device name (cpu, cuda, tpu)
    """
    store = StateStore()
    return store.get_state().jax_device


def get_jax_memory_usage() -> tuple[int, int]:
    """Get JAX memory usage.

    Returns
    -------
    tuple[int, int]
        (used_bytes, total_bytes)
    """
    store = StateStore()
    state = store.get_state()
    return (state.jax_memory_used, state.jax_memory_total)


def get_jax_memory_percent() -> float:
    """Get JAX memory usage as percentage.

    Returns
    -------
    float
        Memory usage percentage (0-100)
    """
    used, total = get_jax_memory_usage()
    if total == 0:
        return 0.0
    return (used / total) * 100.0


# Settings Selectors


def get_theme() -> str:
    """Get current UI theme.

    Returns
    -------
    str
        Theme name (light, dark)
    """
    store = StateStore()
    return store.get_state().theme


def get_current_seed() -> int:
    """Get current random seed.

    Returns
    -------
    int
        Random seed
    """
    store = StateStore()
    return store.get_state().current_seed


def is_auto_save_enabled() -> bool:
    """Check if auto-save is enabled.

    Returns
    -------
    bool
        True if auto-save enabled
    """
    store = StateStore()
    return store.get_state().auto_save_enabled


# Provenance Selectors


def get_transform_history_for_dataset(dataset_id: str) -> list:
    """Get transform history for a specific dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    list[TransformRecord]
        Transform records where this dataset is the target
    """
    store = StateStore()
    state = store.get_state()

    return [
        record
        for record in state.transform_history
        if record.target_dataset_id == dataset_id
    ]


def get_dataset_lineage(dataset_id: str) -> list[str]:
    """Get lineage of dataset through transforms.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    list[str]
        List of dataset IDs in lineage (oldest to newest)
    """
    store = StateStore()
    state = store.get_state()

    lineage = [dataset_id]
    current_id = dataset_id

    # Trace back through transform history
    while True:
        # Find record where current_id is the target
        record = next(
            (r for r in state.transform_history if r.target_dataset_id == current_id),
            None,
        )

        if record is None:
            break

        lineage.insert(0, record.source_dataset_id)
        current_id = record.source_dataset_id

    return lineage


# Undo/Redo Selectors


def can_undo() -> bool:
    """Check if undo is available.

    Returns
    -------
    bool
        True if undo available
    """
    store = StateStore()
    return store.can_undo()


def can_redo() -> bool:
    """Check if redo is available.

    Returns
    -------
    bool
        True if redo available
    """
    store = StateStore()
    return store.can_redo()
