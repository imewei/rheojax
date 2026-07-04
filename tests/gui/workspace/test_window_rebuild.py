import warnings

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def test_rebuild_increments_epoch(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    epoch_before = win._epoch
    win._rebuild(AppState())
    assert win._epoch == epoch_before + 1


def test_rebuild_replaces_controllers_with_new_state(qtbot):
    state_a = AppState()
    win = WorkspaceWindow(state_a)
    qtbot.addWidget(win)
    state_b = AppState()
    state_b.fit.protocol = "creep"
    win._rebuild(state_b)
    assert win._controllers["fit"] is not None
    assert win._state is state_b


def test_toolbar_widgets_are_same_instance_across_rebuild(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    fit_btn_before = win._fit_btn
    win._rebuild(AppState())
    assert win._fit_btn is fit_btn_before


def test_notifier_is_same_instance_across_rebuild(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    notifier_before = win._notifier
    win._rebuild(AppState())
    assert win._notifier is notifier_before


def test_guard_ignores_callback_after_epoch_changed(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    calls = []
    guarded = win._guard(win._epoch, lambda: calls.append(1))
    win._epoch += 1  # simulate a rebuild happening before the callback fires
    guarded()
    assert calls == []


def test_guard_runs_callback_when_epoch_unchanged(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    calls = []
    guarded = win._guard(win._epoch, lambda: calls.append(1))
    guarded()
    assert calls == [1]


def test_build_workspace_never_marks_dirty(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    assert state.project.dirty is False


def test_dispose_workspace_does_not_warn_on_cross_handler_disconnect(qtbot):
    """Regression: _dispose_workspace used to iterate all bodies (fit +
    transform + pipeline) combined and try disconnecting all three modes'
    edited-handlers from every body, regardless of which mode it actually
    belonged to. PySide6 emits a "Failed to disconnect" RuntimeWarning (not
    a Python exception) for the handlers that were never connected, so the
    surrounding try/except (RuntimeError, TypeError) never caught it."""
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        win._rebuild(AppState())

    disconnect_warnings = [
        w for w in caught if "Failed to disconnect" in str(w.message)
    ]
    assert disconnect_warnings == []
