"""Qt signal definitions for state management.

This module provides Qt signals for reactive UI updates on state changes.
"""

from PySide6.QtCore import QObject, Signal


class StateSignals(QObject):
    """Qt signals for state change notifications.

    Signals
    -------
    state_changed : Signal()
        General state update signal
    dataset_added : Signal(str)
        Emitted when dataset added (dataset_id)
    dataset_removed : Signal(str)
        Emitted when dataset removed (dataset_id)
    dataset_updated : Signal(str)
        Emitted when dataset modified (dataset_id)
    dataset_selected : Signal(str)
        Emitted when active dataset changes (dataset_id)
    model_selected : Signal(str)
        Emitted when model selected (model_name)
    model_params_changed : Signal(str)
        Emitted when model parameters change (model_name)
    fit_started : Signal(str, str)
        Emitted when fit starts (model_name, dataset_id)
    fit_progress : Signal(str, int)
        Emitted during fit progress (job_id, progress 0-100)
    fit_completed : Signal(str, str)
        Emitted when fit completes (model_name, dataset_id)
    fit_failed : Signal(str, str, str)
        Emitted when fit fails (model_name, dataset_id, error)
    bayesian_started : Signal(str, str)
        Emitted when Bayesian inference starts (model_name, dataset_id)
    bayesian_progress : Signal(str, int)
        Emitted during Bayesian progress (job_id, progress 0-100)
    bayesian_completed : Signal(str, str)
        Emitted when Bayesian inference completes (model_name, dataset_id)
    bayesian_failed : Signal(str, str, str)
        Emitted when Bayesian inference fails (model_name, dataset_id, error)
    pipeline_step_changed : Signal(str, str)
        Emitted when pipeline step changes (step_name, status)
    transform_applied : Signal(str, str)
        Emitted when transform applied (transform_name, dataset_id)
    jax_device_changed : Signal(str)
        Emitted when JAX device changes (device_name)
    jax_memory_updated : Signal(int, int)
        Emitted when JAX memory updates (used_bytes, total_bytes)
    theme_changed : Signal(str)
        Emitted when theme changes (theme_name)
    project_loaded : Signal(str)
        Emitted when project loaded (project_path)
    project_saved : Signal(str)
        Emitted when project saved (project_path)
    """

    # General
    state_changed = Signal()

    # Datasets
    dataset_added = Signal(str)
    dataset_removed = Signal(str)
    dataset_updated = Signal(str)
    dataset_selected = Signal(str)

    # Models
    model_selected = Signal(str)
    model_params_changed = Signal(str)

    # Fitting
    fit_started = Signal(str, str)  # model_name, dataset_id
    fit_progress = Signal(str, int)  # job_id, progress
    fit_completed = Signal(str, str)  # model_name, dataset_id
    fit_failed = Signal(str, str, str)  # model_name, dataset_id, error

    # Bayesian
    bayesian_started = Signal(str, str)  # model_name, dataset_id
    bayesian_progress = Signal(str, int)  # job_id, progress
    bayesian_completed = Signal(str, str)  # model_name, dataset_id
    bayesian_failed = Signal(str, str, str)  # model_name, dataset_id, error

    # Pipeline
    pipeline_step_changed = Signal(str, str)  # step_name, status

    # Transforms
    transform_applied = Signal(str, str)  # transform_name, dataset_id

    # JAX
    jax_device_changed = Signal(str)  # device_name
    jax_memory_updated = Signal(int, int)  # used_bytes, total_bytes

    # UI
    theme_changed = Signal(str)  # theme_name

    # Project
    project_loaded = Signal(str)  # project_path
    project_saved = Signal(str)  # project_path

    def __init__(self) -> None:
        """Initialize signal emitters."""
        super().__init__()
