"""Tests for PipelineSidebar.select_step_by_type and navigate_to re-entry guard.

These tests verify:
- select_step_by_type() selects the correct list item by step type
- navigate_to re-entry guard prevents infinite recursion
"""

import os

# Set offscreen platform BEFORE any Qt imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

pytestmark = pytest.mark.skipif(
    not HAS_PYSIDE6,
    reason="PySide6 not installed",
)


@pytest.fixture(scope="module")
def qapp_module():
    if not HAS_PYSIDE6:
        return None
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def reset_store():
    from rheojax.gui.state.store import StateStore

    StateStore.reset()
    yield
    StateStore.reset()


# ---------------------------------------------------------------------------
# TestSelectStepByType
# ---------------------------------------------------------------------------


class TestSelectStepByType:
    @pytest.mark.smoke
    def test_select_step_by_type_selects_correct_item(self, qapp_module):
        """select_step_by_type must select the first item matching the given type."""
        from rheojax.gui.state.actions import add_pipeline_step
        from rheojax.gui.widgets.pipeline_sidebar import PipelineSidebar
        from rheojax.gui.widgets.pipeline_step_delegate import ROLE_STEP_TYPE

        sidebar = PipelineSidebar()

        add_pipeline_step("load", "Load Data", config={"file": "data.csv"})
        add_pipeline_step("fit", "Fit Maxwell", config={"model": "maxwell"})
        sidebar._refresh_list()

        sidebar.select_step_by_type("fit")

        current = sidebar._list.currentItem()
        assert current is not None
        assert current.data(ROLE_STEP_TYPE) == "fit"

    @pytest.mark.unit
    def test_select_step_by_type_none_deselects(self, qapp_module):
        """select_step_by_type(None) must deselect all items."""
        from rheojax.gui.state.actions import add_pipeline_step
        from rheojax.gui.widgets.pipeline_sidebar import PipelineSidebar

        sidebar = PipelineSidebar()
        add_pipeline_step("load", "Load", config={"file": "data.csv"})
        sidebar._refresh_list()

        sidebar.select_step_by_type(None)
        assert sidebar._list.currentItem() is None

    @pytest.mark.unit
    def test_select_step_by_type_no_match_does_nothing(self, qapp_module):
        """select_step_by_type with a non-existent type must not crash."""
        from rheojax.gui.state.actions import add_pipeline_step
        from rheojax.gui.widgets.pipeline_sidebar import PipelineSidebar

        sidebar = PipelineSidebar()
        add_pipeline_step("load", "Load", config={"file": "data.csv"})
        sidebar._refresh_list()

        # Should not raise
        sidebar.select_step_by_type("bayesian")


# ---------------------------------------------------------------------------
# TestNavigateToReEntryGuard
# ---------------------------------------------------------------------------


class TestNavigateToReEntryGuard:
    @pytest.mark.smoke
    def test_navigate_to_is_not_reentrant(self, qapp_module):
        """Calling navigate_to while already navigating must be a no-op."""
        from unittest.mock import patch

        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()

        # Simulate re-entrant call: set _navigating=True before calling
        window._navigating = True
        with patch.object(window.workspace, "show_step") as mock_show:
            window.navigate_to("data")

        # show_step must NOT have been called (re-entry guard blocked it)
        mock_show.assert_not_called()

        # Clean up
        window._navigating = False
        window.close()
