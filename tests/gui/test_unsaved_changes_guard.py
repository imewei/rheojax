"""Regression tests for the shared unsaved-changes guard on File>New/Open/Open Recent.

Prior to this fix, File>New ignored `_on_save_file()`'s return value (so
choosing "Save" silently wiped the project when save wasn't implemented),
and File>Open / Open Recent skipped the unsaved-changes prompt entirely,
silently discarding the current project. All three now route through
`RheoJAXMainWindow._confirm_unsaved_changes`, the same guard `closeEvent`
already used correctly.
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gui

try:
    from PySide6.QtWidgets import QApplication, QMessageBox

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_new_file_does_not_wipe_project_when_save_fails(qtbot, qapp) -> None:
    """File>New must NOT dispatch NEW_PROJECT if the user picks Save and it fails."""
    from rheojax.gui.app.main_window import RheoJAXMainWindow

    window = RheoJAXMainWindow()
    qtbot.addWidget(window)
    window._has_unsaved_changes = True

    with (
        patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Save
        ),
        patch.object(window, "_on_save_file", return_value=False),
        patch.object(window.store, "dispatch") as mock_dispatch,
    ):
        window._on_new_file()

    mock_dispatch.assert_not_called()


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_new_file_proceeds_when_save_succeeds(qtbot, qapp) -> None:
    """File>New dispatches NEW_PROJECT once the chosen Save actually succeeds."""
    from rheojax.gui.app.main_window import RheoJAXMainWindow

    window = RheoJAXMainWindow()
    qtbot.addWidget(window)
    window._has_unsaved_changes = True

    with (
        patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Save
        ),
        patch.object(window, "_on_save_file", return_value=True),
        patch.object(window.store, "dispatch") as mock_dispatch,
    ):
        window._on_new_file()

    mock_dispatch.assert_any_call("NEW_PROJECT")


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_open_file_guards_unsaved_changes(qtbot, qapp) -> None:
    """File>Open must prompt for unsaved changes before replacing the project."""
    from rheojax.gui.app.main_window import RheoJAXMainWindow
    from rheojax.gui.compat import QFileDialog

    window = RheoJAXMainWindow()
    qtbot.addWidget(window)
    window._has_unsaved_changes = True

    with (
        patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel
        ),
        patch.object(QFileDialog, "getOpenFileName") as mock_dialog,
        patch.object(window.store, "dispatch") as mock_dispatch,
    ):
        window._on_open_file()

    # Cancelling the unsaved-changes prompt must abort before the file picker
    # even opens, and LOAD_PROJECT must never be dispatched.
    mock_dialog.assert_not_called()
    mock_dispatch.assert_not_called()


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_open_recent_project_guards_unsaved_changes(qtbot, qapp, tmp_path) -> None:
    """Open Recent must prompt for unsaved changes before replacing the project."""
    from rheojax.gui.app.main_window import RheoJAXMainWindow

    window = RheoJAXMainWindow()
    qtbot.addWidget(window)
    window._has_unsaved_changes = True

    recent_project = tmp_path / "project.rheojax"
    recent_project.write_text("placeholder")

    with (
        patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel
        ),
        patch.object(window.store, "dispatch") as mock_dispatch,
    ):
        window._on_open_recent_project(recent_project)

    mock_dispatch.assert_not_called()


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_open_file_does_not_dispatch_load_project(qtbot, qapp, tmp_path) -> None:
    """LOAD_PROJECT's reducer never restores datasets/results in this legacy
    window (no codec bridges its AppState to the .rheojax v2 archive), so
    picking a file must not dispatch LOAD_PROJECT nor claim success -- it
    must show a not-yet-implemented message instead (GUI-P1)."""
    from rheojax.gui.app.main_window import RheoJAXMainWindow
    from rheojax.gui.compat import QFileDialog

    window = RheoJAXMainWindow()
    qtbot.addWidget(window)

    picked = tmp_path / "project.rheojax"
    picked.write_text("placeholder")

    with (
        patch.object(
            QFileDialog, "getOpenFileName", return_value=(str(picked), "")
        ),
        patch.object(QMessageBox, "information") as mock_info,
        patch.object(window.store, "dispatch") as mock_dispatch,
    ):
        window._on_open_file()

    mock_dispatch.assert_not_called()
    mock_info.assert_called_once()


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_auto_save_does_not_dispatch_save_project(qtbot, qapp, tmp_path) -> None:
    """SAVE_PROJECT's reducer only clears is_modified -- it writes nothing to
    disk in this legacy window. Auto-save must not dispatch it (that would
    mark unsaved work as saved without persisting it) (GUI-P1)."""
    from rheojax.gui.app.main_window import RheoJAXMainWindow

    window = RheoJAXMainWindow()
    qtbot.addWidget(window)

    project_path = tmp_path / "project.rheojax"
    window.store.dispatch("LOAD_PROJECT", {"file_path": str(project_path)})
    assert window.store.get_state().auto_save_enabled is True

    with patch.object(window.store, "dispatch") as mock_dispatch:
        window._do_auto_save()

    for call in mock_dispatch.call_args_list:
        assert call.args[0] != "SAVE_PROJECT"
