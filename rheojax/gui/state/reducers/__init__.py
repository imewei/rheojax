"""Sub-reducer dispatch table for the StateStore.

Each entry maps an action type string to a function that takes
``(action_dict)`` and returns ``Callable[[AppState], AppState] | None``.

Special cases that require access to the ``StateStore`` instance
(UNDO, REDO, UPDATE_PREFERENCES) are handled separately in
``StateStore._reduce_action`` before consulting this table.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rheojax.gui.state.reducers.bayesian_reducers import (
    reduce_bayesian_completed,
    reduce_bayesian_failed,
    reduce_start_bayesian,
    reduce_store_bayesian_result,
)
from rheojax.gui.state.reducers.data_reducers import (
    reduce_auto_detect_test_mode,
    reduce_delete_selected_dataset,
    reduce_import_data,
    reduce_import_data_failed,
    reduce_import_data_success,
    reduce_set_active_dataset,
    reduce_set_test_mode,
)
from rheojax.gui.state.reducers.fitting_reducers import (
    reduce_fitting_completed,
    reduce_fitting_failed,
    reduce_start_fitting,
    reduce_store_fit_result,
)
from rheojax.gui.state.reducers.model_reducers import (
    reduce_cancel_jobs,
    reduce_set_active_model,
)
from rheojax.gui.state.reducers.pipeline_reducers import (
    reduce_add_pipeline_step,
    reduce_apply_transform,
    reduce_cache_step_result,
    reduce_clear_pipeline,
    reduce_export_results,
    reduce_load_pipeline,
    reduce_remove_pipeline_step,
    reduce_reorder_pipeline_step,
    reduce_select_pipeline_step,
    reduce_set_pipeline_name,
    reduce_set_pipeline_running,
    reduce_set_pipeline_step,
    reduce_transform_completed,
    reduce_update_step_config,
    reduce_update_step_status,
)
from rheojax.gui.state.reducers.project_reducers import (
    reduce_load_project,
    reduce_new_project,
    reduce_record_provenance,
    reduce_save_project,
)
from rheojax.gui.state.reducers.ui_reducers import (
    reduce_check_compatibility,
    reduce_set_deformation_mode,
    reduce_set_os_theme,
    reduce_set_poisson_ratio,
    reduce_set_tab,
    reduce_set_theme,
    reduce_set_workflow_mode,
)

# Type alias for reducer functions: (action_dict) -> updater | None
ReducerFn = Callable[[dict[str, Any]], Callable | None]

# Dispatch table: action_type -> reducer function.
# All entries take a single ``action`` dict argument and return either
# a ``Callable[[AppState], AppState]`` updater or ``None``.
#
# Actions that need access to the store instance (UNDO, REDO,
# UPDATE_PREFERENCES) are NOT in this table -- they are handled inline
# in ``StateStore._reduce_action`` before this table is consulted.
REDUCER_DISPATCH: dict[str, ReducerFn] = {
    # UI / Settings
    "SET_THEME": reduce_set_theme,
    "SET_OS_THEME": reduce_set_os_theme,
    "SET_WORKFLOW_MODE": reduce_set_workflow_mode,
    "SET_DEFORMATION_MODE": reduce_set_deformation_mode,
    "SET_POISSON_RATIO": reduce_set_poisson_ratio,
    "SET_TAB": reduce_set_tab,
    "NAVIGATE_TAB": reduce_set_tab,
    "CHECK_COMPATIBILITY": reduce_check_compatibility,
    # Data
    "SET_TEST_MODE": reduce_set_test_mode,
    "AUTO_DETECT_TEST_MODE": reduce_auto_detect_test_mode,
    "SET_ACTIVE_DATASET": reduce_set_active_dataset,
    "IMPORT_DATA": reduce_import_data,
    "IMPORT_DATA_SUCCESS": reduce_import_data_success,
    "IMPORT_DATA_FAILED": reduce_import_data_failed,
    "DELETE_SELECTED_DATASET": reduce_delete_selected_dataset,
    # Model
    "SET_ACTIVE_MODEL": reduce_set_active_model,
    "CANCEL_JOBS": reduce_cancel_jobs,
    # Fitting
    "START_FITTING": reduce_start_fitting,
    "FITTING_COMPLETED": reduce_fitting_completed,
    "FITTING_FAILED": reduce_fitting_failed,
    "STORE_FIT_RESULT": reduce_store_fit_result,
    # Bayesian
    "START_BAYESIAN": reduce_start_bayesian,
    "BAYESIAN_COMPLETED": reduce_bayesian_completed,
    "BAYESIAN_FAILED": reduce_bayesian_failed,
    "STORE_BAYESIAN_RESULT": reduce_store_bayesian_result,
    # Classic pipeline
    "SET_PIPELINE_STEP": reduce_set_pipeline_step,
    "APPLY_TRANSFORM": reduce_apply_transform,
    "TRANSFORM_COMPLETED": reduce_transform_completed,
    "EXPORT_RESULTS": reduce_export_results,
    # Visual pipeline
    "ADD_PIPELINE_STEP": reduce_add_pipeline_step,
    "REMOVE_PIPELINE_STEP": reduce_remove_pipeline_step,
    "REORDER_PIPELINE_STEP": reduce_reorder_pipeline_step,
    "SELECT_PIPELINE_STEP": reduce_select_pipeline_step,
    "UPDATE_STEP_CONFIG": reduce_update_step_config,
    "UPDATE_STEP_STATUS": reduce_update_step_status,
    "CACHE_STEP_RESULT": reduce_cache_step_result,
    "SET_PIPELINE_RUNNING": reduce_set_pipeline_running,
    "SET_PIPELINE_NAME": reduce_set_pipeline_name,
    "CLEAR_PIPELINE": reduce_clear_pipeline,
    "LOAD_PIPELINE": reduce_load_pipeline,
    # Project
    "LOAD_PROJECT": reduce_load_project,
    "NEW_PROJECT": reduce_new_project,
    "SAVE_PROJECT": reduce_save_project,
    "RECORD_PROVENANCE": reduce_record_provenance,
}

# Signal-only actions: dispatch() emits domain signals for these but
# they do not mutate state.  Listed separately so _reduce_action can
# return None without logging a warning.
SIGNAL_ONLY_ACTIONS: frozenset[str] = frozenset({
    "BAYESIAN_PROGRESS",
    "FIT_PROGRESS",
    "TRANSFORM_APPLIED",
})
