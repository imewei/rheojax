"""UI and settings action reducers.

Handles: SET_THEME, SET_OS_THEME, SET_WORKFLOW_MODE, SET_DEFORMATION_MODE,
SET_POISSON_RATIO, SET_TAB, NAVIGATE_TAB, UPDATE_PREFERENCES,
CHECK_COMPATIBILITY.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rheojax.gui.state.store import AppState, StateStore

from rheojax.logging import get_logger

logger = get_logger(__name__)


def reduce_set_theme(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    theme = action.get("theme", "light")

    def updater(state: AppState) -> AppState:
        if state.theme == theme:
            return state
        return replace(state, theme=theme, is_modified=True)

    return updater


def reduce_set_os_theme(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    os_theme = action.get("os_theme", "light")

    def updater(state: AppState) -> AppState:
        if state.os_theme == os_theme:
            return state
        return replace(state, os_theme=os_theme)

    return updater


def reduce_set_workflow_mode(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    from rheojax.gui.state.store import WorkflowMode

    mode = action.get("mode")

    def updater(state: AppState) -> AppState:
        new_mode = WorkflowMode[mode.upper()] if isinstance(mode, str) else mode
        if isinstance(new_mode, WorkflowMode):
            return replace(state, workflow_mode=new_mode)
        return state

    return updater


def reduce_set_deformation_mode(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    mode = action.get("deformation_mode", "shear")

    def updater(state: AppState) -> AppState:
        if state.deformation_mode == mode:
            return state
        return replace(state, deformation_mode=mode, is_modified=True)

    return updater


def reduce_set_poisson_ratio(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    ratio = action.get("poisson_ratio", 0.5)

    def updater(state: AppState) -> AppState:
        if state.poisson_ratio == ratio:
            return state
        return replace(state, poisson_ratio=ratio, is_modified=True)

    return updater


def reduce_set_tab(
    action: dict[str, Any],
) -> Callable[[AppState], AppState]:
    tab = action.get("tab")

    def updater(state: AppState) -> AppState:
        resolved_tab = tab if tab is not None else state.current_tab
        return replace(state, current_tab=resolved_tab)

    return updater


def reduce_update_preferences(
    action: dict[str, Any], store: StateStore,
) -> Callable[[AppState], AppState]:
    prefs = action

    def updater(state: AppState) -> AppState:
        updates: dict[str, Any] = {}
        for key in (
            "theme",
            "auto_save_enabled",
            "last_export_dir",
            "current_seed",
        ):
            if key in prefs:
                updates[key] = prefs[key]
        return replace(state, **updates) if updates else state

    # Apply runtime settings that live outside AppState
    if "max_undo_steps" in prefs:
        store.set_max_undo_size(int(prefs["max_undo_steps"]))
    if "worker_isolation_mode" in prefs:
        import os

        os.environ["RHEOJAX_WORKER_ISOLATION"] = prefs[
            "worker_isolation_mode"
        ]

    return updater


def reduce_check_compatibility(
    action: dict[str, Any],
) -> Callable[[AppState], AppState] | None:
    # STORE-002: CHECK_COMPATIBILITY is a UI-only trigger (opens the
    # diagnostics tab) and does not need to mutate state.  Return
    # None so dispatch() skips update_state entirely (no undo entry,
    # no state clone, no state_changed signal).
    return None
