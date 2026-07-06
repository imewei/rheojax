import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import AppState
from rheojax.gui.widgets.log_dock import LogDockWidget
from rheojax.gui.workspace.window import WorkspaceWindow


def _win(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    return win


def test_workspace_window_has_log_dock(qtbot):
    win = _win(qtbot)
    assert isinstance(win.log_dock, LogDockWidget)
    assert win.log_dock.isVisible() is False


def test_workspace_window_log_appends_to_dock(qtbot):
    win = _win(qtbot)
    win.log("hello from workspace")
    assert "hello from workspace" in win.log_dock.text_edit.toPlainText()


def test_view_menu_toggle_shows_log_dock(qtbot):
    win = _win(qtbot)
    win.show()
    win.view_log_dock_action.trigger()
    assert win.log_dock.isVisible() is True
