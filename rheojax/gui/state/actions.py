"""State action creators for RheoJAX GUI.

This module provides action functions that modify state through the StateStore
and emit appropriate signals for UI reactivity.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from rheojax.gui.state.store import (
    AppState,
    BayesianResult,
    DatasetState,
    FitResult,
    ParameterState,
    PipelineStep,
    StateStore,
    StepStatus,
    TransformRecord,
)

# Dataset Actions


def load_dataset(
    file_path: Path,
    name: str,
    test_mode: str,
    x_data: Any,
    y_data: Any,
    y2_data: Any | None = None,
    metadata: dict | None = None,
) -> str:
    """Load a dataset into the state.

    Parameters
    ----------
    file_path : Path
        Source file path
    name : str
        Dataset name
    test_mode : str
        Test mode (oscillation, relaxation, creep, rotation)
    x_data : Any
        X-axis data (JAX array)
    y_data : Any
        Y-axis data (JAX array)
    y2_data : Any, optional
        Second Y-axis data for oscillation (G'')
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    str
        Dataset ID
    """
    store = StateStore()
    dataset_id = str(uuid.uuid4())

    dataset = DatasetState(
        id=dataset_id,
        name=name,
        file_path=file_path,
        test_mode=test_mode,
        x_data=x_data,
        y_data=y_data,
        y2_data=y2_data,
        metadata=metadata or {},
        is_modified=False,
    )

    def updater(state: AppState) -> AppState:
        new_datasets = state.datasets.copy()
        new_datasets[dataset_id] = dataset
        return AppState(
            **{
                **state.__dict__,
                "datasets": new_datasets,
                "active_dataset_id": dataset_id,
                "is_modified": True,
            }
        )

    store.update_state(updater)

    if store._signals:
        store._signals.dataset_added.emit(dataset_id)
        store._signals.dataset_selected.emit(dataset_id)

    return dataset_id


def remove_dataset(dataset_id: str) -> None:
    """Remove a dataset from state.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        new_datasets = state.datasets.copy()
        if dataset_id in new_datasets:
            del new_datasets[dataset_id]

        # Clear active if removed
        new_active = state.active_dataset_id
        if new_active == dataset_id:
            new_active = next(iter(new_datasets.keys()), None)

        # Remove associated fit results
        new_fit_results = {
            k: v for k, v in state.fit_results.items() if v.dataset_id != dataset_id
        }
        new_bayesian_results = {
            k: v
            for k, v in state.bayesian_results.items()
            if v.dataset_id != dataset_id
        }

        return AppState(
            **{
                **state.__dict__,
                "datasets": new_datasets,
                "active_dataset_id": new_active,
                "fit_results": new_fit_results,
                "bayesian_results": new_bayesian_results,
                "is_modified": True,
            }
        )

    store.update_state(updater)

    if store._signals:
        store._signals.dataset_removed.emit(dataset_id)


def set_active_dataset(dataset_id: str) -> None:
    """Set the active dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        if dataset_id not in state.datasets:
            return state
        return AppState(**{**state.__dict__, "active_dataset_id": dataset_id})

    store.update_state(updater)

    if store._signals:
        store._signals.dataset_selected.emit(dataset_id)


def update_dataset(dataset_id: str, **updates) -> None:
    """Update dataset properties.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    **updates : dict
        Properties to update
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        if dataset_id not in state.datasets:
            return state

        dataset = state.datasets[dataset_id]
        updated_dataset = DatasetState(**{**dataset.__dict__, **updates})

        new_datasets = state.datasets.copy()
        new_datasets[dataset_id] = updated_dataset

        return AppState(
            **{**state.__dict__, "datasets": new_datasets, "is_modified": True}
        )

    store.update_state(updater)

    if store._signals:
        store._signals.dataset_updated.emit(dataset_id)


# Model Actions


def select_model(model_name: str, parameters: dict[str, ParameterState]) -> None:
    """Select a model and initialize parameters.

    Parameters
    ----------
    model_name : str
        Model name
    parameters : dict[str, ParameterState]
        Model parameters
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        return AppState(
            **{
                **state.__dict__,
                "active_model_name": model_name,
                "model_params": parameters,
            }
        )

    store.update_state(updater)

    if store._signals:
        store._signals.model_selected.emit(model_name)


def update_parameter(name: str, value: float) -> None:
    """Update a model parameter value.

    Parameters
    ----------
    name : str
        Parameter name
    value : float
        New parameter value
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        if name not in state.model_params:
            return state

        param = state.model_params[name]
        updated_param = ParameterState(**{**param.__dict__, "value": value})

        new_params = state.model_params.copy()
        new_params[name] = updated_param

        return AppState(**{**state.__dict__, "model_params": new_params})

    store.update_state(updater)

    if store._signals and store._state.active_model_name:
        store._signals.model_params_changed.emit(store._state.active_model_name)


def update_parameter_bounds(name: str, min_bound: float, max_bound: float) -> None:
    """Update bounds for a model parameter.

    Parameters
    ----------
    name : str
        Parameter name
    min_bound : float
        New minimum bound
    max_bound : float
        New maximum bound
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        if name not in state.model_params:
            return state

        param = state.model_params[name]
        updated_param = ParameterState(
            **{
                **param.__dict__,
                "min_bound": float(min_bound),
                "max_bound": float(max_bound),
            }
        )

        new_params = state.model_params.copy()
        new_params[name] = updated_param

        return AppState(**{**state.__dict__, "model_params": new_params})

    store.update_state(updater)

    if store._signals and store._state.active_model_name:
        store._signals.model_params_changed.emit(store._state.active_model_name)


def toggle_parameter_fixed(name: str, fixed: bool) -> None:
    """Toggle fixed state for a model parameter.

    Parameters
    ----------
    name : str
        Parameter name
    fixed : bool
        Whether the parameter is fixed
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        if name not in state.model_params:
            return state

        param = state.model_params[name]
        updated_param = ParameterState(**{**param.__dict__, "fixed": bool(fixed)})

        new_params = state.model_params.copy()
        new_params[name] = updated_param

        return AppState(**{**state.__dict__, "model_params": new_params})

    store.update_state(updater)

    if store._signals and store._state.active_model_name:
        store._signals.model_params_changed.emit(store._state.active_model_name)


def reset_parameters(default_params: dict[str, ParameterState]) -> None:
    """Reset parameters to default values.

    Parameters
    ----------
    default_params : dict[str, ParameterState]
        Default parameter states
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        return AppState(**{**state.__dict__, "model_params": default_params})

    store.update_state(updater)

    if store._signals and store._state.active_model_name:
        store._signals.model_params_changed.emit(store._state.active_model_name)


# Fit Actions


def start_fit(model_name: str, dataset_id: str) -> None:
    """Mark fit as started.

    Parameters
    ----------
    model_name : str
        Model name
    dataset_id : str
        Dataset identifier
    """
    store = StateStore()

    if store._signals:
        store._signals.fit_started.emit(model_name, dataset_id)


def store_fit_result(result: FitResult) -> None:
    """Store a completed fit result.

    Parameters
    ----------
    result : FitResult
        Fit result
    """
    store = StateStore()
    result_key = f"{result.model_name}_{result.dataset_id}"

    def updater(state: AppState) -> AppState:
        new_results = state.fit_results.copy()
        new_results[result_key] = result
        return AppState(
            **{**state.__dict__, "fit_results": new_results, "is_modified": True}
        )

    store.update_state(updater)

    if store._signals:
        store._signals.fit_completed.emit(result.model_name, result.dataset_id)


def fail_fit(model_name: str, dataset_id: str, error: str) -> None:
    """Mark fit as failed.

    Parameters
    ----------
    model_name : str
        Model name
    dataset_id : str
        Dataset identifier
    error : str
        Error message
    """
    store = StateStore()

    if store._signals:
        store._signals.fit_failed.emit(model_name, dataset_id, error)


def set_active_model(model_name: str) -> dict:
    """Create action to set the active model.

    Parameters
    ----------
    model_name : str
        Model name

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {"type": "SET_ACTIVE_MODEL", "model_name": model_name}


def start_fitting(model_name: str, dataset_id: str) -> dict:
    """Create action to start fitting.

    Parameters
    ----------
    model_name : str
        Model name
    dataset_id : str
        Dataset identifier

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {"type": "START_FITTING", "model_name": model_name, "dataset_id": dataset_id}


def update_fit_progress(progress: float) -> dict:
    """Create action to update fit progress.

    Parameters
    ----------
    progress : float
        Progress percentage (0-100)

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {"type": "FIT_PROGRESS", "progress": progress}


def fitting_completed(result: FitResult) -> dict:
    """Create action for fit completion.

    Parameters
    ----------
    result : FitResult
        Fit result

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {"type": "FITTING_COMPLETED", "result": result}


def fitting_failed(error: str) -> dict:
    """Create action for fit failure.

    Parameters
    ----------
    error : str
        Error message

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {"type": "FITTING_FAILED", "error": error}


# Bayesian Actions


def start_bayesian(model_name: str, dataset_id: str) -> dict:
    """Create action to start Bayesian inference.

    Parameters
    ----------
    model_name : str
        Model name
    dataset_id : str
        Dataset identifier

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {
        "type": "START_BAYESIAN",
        "model_name": model_name,
        "dataset_id": dataset_id,
    }


def store_bayesian_result(result: BayesianResult) -> None:
    """Store a completed Bayesian result.

    Parameters
    ----------
    result : BayesianResult
        Bayesian result
    """
    store = StateStore()
    result_key = f"{result.model_name}_{result.dataset_id}"

    def updater(state: AppState) -> AppState:
        new_results = state.bayesian_results.copy()
        new_results[result_key] = result
        return AppState(
            **{**state.__dict__, "bayesian_results": new_results, "is_modified": True}
        )

    store.update_state(updater)

    if store._signals:
        store._signals.bayesian_completed.emit(result.model_name, result.dataset_id)


def fail_bayesian(model_name: str, dataset_id: str, error: str) -> None:
    """Mark Bayesian inference as failed.

    Parameters
    ----------
    model_name : str
        Model name
    dataset_id : str
        Dataset identifier
    error : str
        Error message
    """
    store = StateStore()

    if store._signals:
        store._signals.bayesian_failed.emit(model_name, dataset_id, error)


def update_bayesian_progress(progress: float) -> dict:
    """Create a Bayesian progress update action.

    Parameters
    ----------
    progress : float
        Progress percentage (0-100)

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {"type": "BAYESIAN_PROGRESS", "progress": progress}


def bayesian_completed(result: BayesianResult) -> dict:
    """Create a Bayesian completed action.

    Parameters
    ----------
    result : BayesianResult
        Bayesian inference result

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {"type": "BAYESIAN_COMPLETED", "result": result}


def bayesian_failed(error: str) -> dict:
    """Create a Bayesian failed action.

    Parameters
    ----------
    error : str
        Error message

    Returns
    -------
    dict
        Action dict for dispatch
    """
    return {"type": "BAYESIAN_FAILED", "error": error}


# Pipeline Actions


def set_pipeline_step(step: PipelineStep, status: StepStatus) -> None:
    """Update pipeline step status.

    Parameters
    ----------
    step : PipelineStep
        Pipeline step
    status : StepStatus
        New status
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        from rheojax.gui.state.store import PipelineState

        new_steps = state.pipeline_state.steps.copy()
        new_steps[step] = status

        new_pipeline = PipelineState(
            steps=new_steps,
            current_step=step,
            error_message=state.pipeline_state.error_message,
        )

        return AppState(**{**state.__dict__, "pipeline_state": new_pipeline})

    store.update_state(updater)

    if store._signals:
        store._signals.pipeline_step_changed.emit(step.name, status.name)


# JAX Actions


def set_jax_device(device: str) -> None:
    """Set JAX device.

    Parameters
    ----------
    device : str
        Device name (cpu, cuda, tpu)
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        return AppState(**{**state.__dict__, "jax_device": device})

    store.update_state(updater)

    if store._signals:
        store._signals.jax_device_changed.emit(device)


def update_jax_memory(used: int, total: int) -> None:
    """Update JAX memory usage.

    Parameters
    ----------
    used : int
        Used memory in bytes
    total : int
        Total memory in bytes
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        return AppState(
            **{**state.__dict__, "jax_memory_used": used, "jax_memory_total": total}
        )

    store.update_state(updater, track_undo=False, emit_signal=False)

    if store._signals:
        store._signals.jax_memory_updated.emit(used, total)


# Settings Actions


def set_theme(theme: str) -> None:
    """Set UI theme.

    Parameters
    ----------
    theme : str
        Theme name (light, dark)
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        return AppState(**{**state.__dict__, "theme": theme})

    store.update_state(updater, track_undo=False)

    if store._signals:
        store._signals.theme_changed.emit(theme)


def set_seed(seed: int) -> None:
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        return AppState(**{**state.__dict__, "current_seed": seed})

    store.update_state(updater, track_undo=False)


def set_auto_save(enabled: bool) -> None:
    """Enable/disable auto-save.

    Parameters
    ----------
    enabled : bool
        Auto-save enabled
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        return AppState(**{**state.__dict__, "auto_save_enabled": enabled})

    store.update_state(updater, track_undo=False)


# Project Actions


def save_project(path: Path) -> None:
    """Mark project as saved.

    Parameters
    ----------
    path : Path
        Project file path
    """
    store = StateStore()

    def updater(state: AppState) -> AppState:
        # Add to recent projects
        recent = [p for p in state.recent_projects if p != path]
        recent.insert(0, path)
        recent = recent[:10]  # Keep last 10

        return AppState(
            **{
                **state.__dict__,
                "project_path": path,
                "project_name": path.stem,
                "is_modified": False,
                "recent_projects": recent,
            }
        )

    store.update_state(updater, track_undo=False)

    if store._signals:
        store._signals.project_saved.emit(str(path))


def load_project(path: Path, state: AppState) -> None:
    """Load project from file.

    Parameters
    ----------
    path : Path
        Project file path
    state : AppState
        Loaded application state
    """
    store = StateStore()

    # Add to recent projects
    recent = [p for p in state.recent_projects if p != path]
    recent.insert(0, path)
    recent = recent[:10]

    updated_state = AppState(
        **{
            **state.__dict__,
            "project_path": path,
            "project_name": path.stem,
            "is_modified": False,
            "recent_projects": recent,
        }
    )

    store.update_state(lambda _: updated_state, track_undo=False)
    store.clear_history()  # Clear undo history on load

    if store._signals:
        store._signals.project_loaded.emit(str(path))


def add_transform_record(
    source_id: str,
    target_id: str,
    transform_name: str,
    parameters: dict[str, Any],
    seed: int | None = None,
) -> None:
    """Add a transform record for provenance tracking.

    Parameters
    ----------
    source_id : str
        Source dataset ID
    target_id : str
        Target dataset ID
    transform_name : str
        Transform name
    parameters : dict[str, Any]
        Transform parameters
    seed : int, optional
        Random seed used
    """
    store = StateStore()

    record = TransformRecord(
        timestamp=datetime.now(),
        source_dataset_id=source_id,
        target_dataset_id=target_id,
        transform_name=transform_name,
        parameters=parameters,
        seed=seed,
    )

    def updater(state: AppState) -> AppState:
        new_history = state.transform_history.copy()
        new_history.append(record)
        return AppState(**{**state.__dict__, "transform_history": new_history})

    store.update_state(updater)

    if store._signals:
        store._signals.transform_applied.emit(transform_name, target_id)
