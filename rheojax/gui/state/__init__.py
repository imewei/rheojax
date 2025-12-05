"""State management for RheoJAX GUI.

This module provides Redux-inspired state management with immutable updates
and Qt signals for reactive UI.

Architecture
------------
- StateStore: Singleton state container with immutable updates
- StateSignals: Qt signal emitters for state changes
- actions: State mutation functions
- selectors: State query functions

Example
-------
>>> from rheojax.gui.state import StateStore, StateSignals
>>> from rheojax.gui.state import actions, selectors
>>>
>>> # Initialize store with signals
>>> store = StateStore()
>>> signals = StateSignals()
>>> store.set_signals(signals)
>>>
>>> # Load a dataset
>>> dataset_id = actions.load_dataset(
...     path, "My Dataset", "oscillation", x_data, y_data
... )
>>>
>>> # Query state
>>> dataset = selectors.get_active_dataset()
>>> is_fit_ready = selectors.is_fit_available()
"""

from rheojax.gui.state import actions, selectors
from rheojax.gui.state.signals import StateSignals
from rheojax.gui.state.store import (
    AppState,
    BayesianResult,
    DatasetState,
    FitResult,
    ParameterState,
    PipelineState,
    PipelineStep,
    StateStore,
    StepStatus,
    TransformRecord,
)

__all__ = [
    # Core
    "StateStore",
    "StateSignals",
    "actions",
    "selectors",
    # State Types
    "AppState",
    "DatasetState",
    "ParameterState",
    "FitResult",
    "BayesianResult",
    "PipelineState",
    "TransformRecord",
    # Enums
    "PipelineStep",
    "StepStatus",
]
