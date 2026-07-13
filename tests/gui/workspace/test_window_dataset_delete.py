from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def _ref(id_: str = "d1") -> DatasetRef:
    return DatasetRef(
        id=id_,
        name=id_,
        protocol_type="oscillation",
        origin="imported",
        units={},
        row_count=1,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_delete_confirmed_removes_dataset_and_refreshes_rail(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    state = AppState()
    state.library.add(_ref("d1"))
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    assert win._rail.count() == 1
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )

    win._on_dataset_delete_requested("d1")

    assert win._rail.count() == 0
    with pytest.raises(KeyError):
        state.library.get("d1")


def test_delete_cancelled_keeps_dataset(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    state = AppState()
    state.library.add(_ref("d1"))
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )

    win._on_dataset_delete_requested("d1")

    assert state.library.get("d1").id == "d1"
    assert win._rail.count() == 1


def test_delete_blocked_when_dataset_has_active_job(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    state = AppState()
    state.library.add(_ref("d1"))
    state.active_jobs.by_id["d1"] = {"job_id": "j1", "step_type": "fit"}
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    prompted = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: prompted.append(True) or QMessageBox.StandardButton.Yes,
    )
    warned = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *a, **k: warned.append(True),
    )

    win._on_dataset_delete_requested("d1")

    assert not prompted  # never got to the confirm-delete prompt
    assert warned
    assert state.library.get("d1").id == "d1"  # dataset untouched
    assert win._rail.count() == 1


def test_delete_unknown_dataset_id_logs_warning_and_does_not_prompt(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    logged = []
    monkeypatch.setattr(
        win.log_dock, "append_record", lambda level, msg: logged.append((level, msg))
    )
    prompted = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: prompted.append(True) or QMessageBox.StandardButton.Yes,
    )

    win._on_dataset_delete_requested("does-not-exist")

    assert not prompted
    assert logged and logged[0][0] == logging.WARNING
