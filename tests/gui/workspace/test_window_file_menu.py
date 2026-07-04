import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QMessageBox

from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def _win(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    return win


def test_maybe_confirm_unsaved_save_cancelled_does_not_proceed(qtbot, monkeypatch):
    # Simulates the user clicking "Save" but then cancelling the file dialog
    # inside _on_save/_on_save_as: dirty stays True, so proceed() must not run.
    win = _win(qtbot)
    win._state.project.dirty = True
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save
    )
    monkeypatch.setattr(win, "_on_save", lambda: None)
    calls = []
    win._maybe_confirm_unsaved_changes(lambda: calls.append(1))
    assert calls == []


def test_maybe_confirm_unsaved_save_success_proceeds(qtbot, monkeypatch):
    win = _win(qtbot)
    win._state.project.dirty = True
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save
    )

    def _fake_save():
        win._state.project.dirty = False

    monkeypatch.setattr(win, "_on_save", _fake_save)
    calls = []
    win._maybe_confirm_unsaved_changes(lambda: calls.append(1))
    assert calls == [1]


def test_maybe_confirm_unsaved_save_blocked_by_active_jobs_warns_and_aborts(
    qtbot, monkeypatch
):
    # Reachable when _maybe_confirm_active_jobs gave up waiting (30s poll timeout) with
    # active_jobs still non-empty: Save is unconditionally blocked in that state, so
    # picking "Save" here must not silently no-op -- it must warn the user that the
    # whole Close/New/Open action was aborted, not just fail to save.
    win = _win(qtbot)
    win._state.project.dirty = True
    win._state.active_jobs.by_id["d1"] = {"status": "running"}
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save
    )
    warning_calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warning_calls.append(1))
    save_calls = []
    monkeypatch.setattr(win, "_on_save", lambda: save_calls.append(1))

    calls = []
    win._maybe_confirm_unsaved_changes(lambda: calls.append(1))

    assert warning_calls == [1]  # user told the action was aborted
    assert save_calls == []  # _on_save (and its own separate dialog) never invoked
    assert calls == []  # proceed() never called -- Close/New/Open did not happen


def test_maybe_confirm_unsaved_discard_still_proceeds(qtbot, monkeypatch):
    win = _win(qtbot)
    win._state.project.dirty = True
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard
    )
    calls = []
    win._maybe_confirm_unsaved_changes(lambda: calls.append(1))
    assert calls == [1]


def test_maybe_confirm_unsaved_clean_state_proceeds_without_dialog(qtbot, monkeypatch):
    win = _win(qtbot)
    assert win._state.project.dirty is False

    def _fail(*a, **k):
        raise AssertionError("dialog should not be shown when not dirty")

    monkeypatch.setattr(QMessageBox, "question", _fail)
    calls = []
    win._maybe_confirm_unsaved_changes(lambda: calls.append(1))
    assert calls == [1]


def test_on_save_shows_critical_dialog_on_value_error(qtbot, monkeypatch):
    win = _win(qtbot)
    win._state.project.path = "/tmp/whatever.rheojax"
    win._state.project.dirty = True

    def _raise(*a, **k):
        raise ValueError("corrupt archive")

    monkeypatch.setattr("rheojax.gui.foundation.project_codec.save_project_v2", _raise)
    critical_calls = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *a, **k: critical_calls.append(a)
    )

    win._on_save()  # must not raise

    assert len(critical_calls) == 1
    assert win._state.project.dirty is True


def test_on_save_as_shows_critical_dialog_on_os_error(qtbot, monkeypatch):
    win = _win(qtbot)

    def _raise(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("rheojax.gui.foundation.project_codec.save_project_v2", _raise)
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: ("/tmp/whatever.rheojax", ""),
    )
    critical_calls = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *a, **k: critical_calls.append(a)
    )

    win._on_save_as()  # must not raise

    assert len(critical_calls) == 1
    assert win._state.project.path is None


def test_on_open_shows_critical_dialog_on_value_error(qtbot, monkeypatch):
    win = _win(qtbot)

    def _raise(*a, **k):
        raise ValueError("unsupported project version")

    monkeypatch.setattr("rheojax.gui.foundation.project_codec.load_project_v2", _raise)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *a, **k: ("/tmp/whatever.rheojax", ""),
    )
    critical_calls = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *a, **k: critical_calls.append(a)
    )

    win._on_open()  # must not raise

    assert len(critical_calls) == 1
