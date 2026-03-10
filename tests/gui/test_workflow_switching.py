import pytest

from rheojax.gui.app.main_window import RheoJAXMainWindow
from rheojax.gui.state.store import WorkflowMode


def test_workflow_switching(qtbot):
    """Test switching between Fitting and Transform workflows.

    The main window now uses a PipelineSidebar + WorkspaceContainer (QStackedWidget)
    instead of a QTabWidget.  Workflow mode changes are reflected in the
    _current_workflow_mode attribute; the sidebar and workspace always show all
    step types (the mode is informational for diagnostics/logging only).
    """
    window = RheoJAXMainWindow()
    qtbot.addWidget(window)

    # -------------------------------------------------------------------------
    # Initial State (Fitting Mode)
    # -------------------------------------------------------------------------
    assert window._current_workflow_mode == WorkflowMode.FITTING

    # Workspace container and sidebar exist
    assert window.workspace is not None
    assert window.sidebar is not None

    # -------------------------------------------------------------------------
    # Switch to Transform Mode
    # -------------------------------------------------------------------------
    window.store.dispatch("SET_WORKFLOW_MODE", {"mode": "transform"})
    assert window._current_workflow_mode == WorkflowMode.TRANSFORM

    # -------------------------------------------------------------------------
    # Switch back to Fitting Mode
    # -------------------------------------------------------------------------
    window.store.dispatch("SET_WORKFLOW_MODE", {"mode": "fitting"})
    assert window._current_workflow_mode == WorkflowMode.FITTING
