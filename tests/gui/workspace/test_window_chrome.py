import pytest

pytest.importorskip("PySide6")

from rheojax.gui.app.status_bar import StatusBar
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def _win(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    return win


def test_status_bar_is_installed(qtbot):
    win = _win(qtbot)
    assert isinstance(win.statusBar(), StatusBar)


def test_refresh_status_bar_calls_update_jax_status(qtbot, monkeypatch):
    win = _win(qtbot)
    calls = []
    monkeypatch.setattr(
        win.statusBar(),
        "update_jax_status",
        lambda **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "rheojax.gui.utils.jax_utils.get_jax_info",
        lambda: {
            "default_device": "cuda:0",
            "memory_used_mb": 100.0,
            "memory_total_mb": 8192.0,
            "float64_enabled": True,
        },
    )
    win._refresh_status_bar()
    assert calls == [
        {
            "device": "cuda:0",
            "memory_used": 100.0,
            "memory_total": 8192.0,
            "float64_enabled": True,
        }
    ]


def test_refresh_status_bar_survives_get_jax_info_failure(qtbot, monkeypatch):
    win = _win(qtbot)

    def _raise():
        raise RuntimeError("no JAX backend")

    monkeypatch.setattr("rheojax.gui.utils.jax_utils.get_jax_info", _raise)
    win._refresh_status_bar()  # must not raise


def test_on_new_shows_status_message(qtbot):
    win = _win(qtbot)
    win._on_new()
    assert win.statusBar().message_label.text() == "New project created"


def test_on_save_as_shows_status_message(qtbot, monkeypatch, tmp_path):
    win = _win(qtbot)
    path = str(tmp_path / "proj.rheojax")
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (path, "")
    )
    win._on_save_as()
    assert win.statusBar().message_label.text() == "Project saved"


def test_on_save_shows_status_message(qtbot, tmp_path):
    win = _win(qtbot)
    win._state.project.path = str(tmp_path / "proj.rheojax")
    win._on_save()
    assert win.statusBar().message_label.text() == "Project saved"


def test_on_open_shows_status_message(qtbot, monkeypatch, tmp_path):
    win = _win(qtbot)
    path = str(tmp_path / "proj.rheojax")
    from PySide6.QtWidgets import QFileDialog

    from rheojax.gui.foundation.project_codec import save_project_v2

    save_project_v2(win._state, path)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (path, "")
    )
    win._on_open()
    assert win.statusBar().message_label.text() == "Project opened"
