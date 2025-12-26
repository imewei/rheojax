
import pytest
from PySide6.QtCore import Qt
from rheojax.gui.app.main_window import RheoJAXMainWindow
from rheojax.gui.state.store import WorkflowMode

def test_workflow_switching(qtbot):
    """Test switching between Fitting and Transform workflows."""
    window = RheoJAXMainWindow()
    qtbot.addWidget(window)
    
    # -------------------------------------------------------------------------
    # Initial State (Fitting Mode)
    # -------------------------------------------------------------------------
    assert window._current_workflow_mode == WorkflowMode.FITTING
    
    # Check tab visibility for Fitting mode
    # Indices: 0:Home, 1:Data, 2:Transform, 3:Fit, 4:Bayesian, 5:Diagnostics, 6:Export
    # Visible: 0, 1, 3, 4, 5, 6
    # Hidden: 2 (Transform)
    
    assert window.tabs.isTabVisible(0)  # Home
    assert window.tabs.isTabVisible(1)  # Data
    assert not window.tabs.isTabVisible(2)  # Transform (HIDDEN)
    assert window.tabs.isTabVisible(3)  # Fit
    assert window.tabs.isTabVisible(4)  # Bayesian
    assert window.tabs.isTabVisible(5)  # Diagnostics
    assert window.tabs.isTabVisible(6)  # Export

    # -------------------------------------------------------------------------
    # Switch to Transform Mode
    # -------------------------------------------------------------------------
    # Simulate clicking "Transform Workflow" on Home Page
    window.store.dispatch("SET_WORKFLOW_MODE", {"mode": "transform"})
    
    # Check state update
    assert window._current_workflow_mode == WorkflowMode.TRANSFORM
    
    # Check tab visibility for Transform mode
    # Visible: 0, 1, 2, 6
    # Hidden: 3, 4, 5
    
    assert window.tabs.isTabVisible(0)  # Home
    assert window.tabs.isTabVisible(1)  # Data
    assert window.tabs.isTabVisible(2)  # Transform (VISIBLE)
    assert not window.tabs.isTabVisible(3)  # Fit (HIDDEN)
    assert not window.tabs.isTabVisible(4)  # Bayesian (HIDDEN)
    assert not window.tabs.isTabVisible(5)  # Diagnostics (HIDDEN)
    assert window.tabs.isTabVisible(6)  # Export
    
    # -------------------------------------------------------------------------
    # Switch back to Fitting Mode
    # -------------------------------------------------------------------------
    window.store.dispatch("SET_WORKFLOW_MODE", {"mode": "fitting"})
    
    assert window._current_workflow_mode == WorkflowMode.FITTING
    assert not window.tabs.isTabVisible(2)  # Transform (HIDDEN)
    assert window.tabs.isTabVisible(3)  # Fit (VISIBLE)
