import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def test_maybe_confirm_active_jobs_proceeds_immediately_when_empty(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    calls = []
    win._maybe_confirm_active_jobs(lambda: calls.append(1))
    assert calls == [1]


def test_maybe_confirm_active_jobs_blocks_when_nonempty(qtbot, monkeypatch):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.active_jobs.by_id["d1"] = {"status": "running"}

    # Simulate the user choosing "Wait" then the job finishing -- monkeypatch the dialog to
    # avoid a real blocking modal in the test, matching this file's need to control the flow.
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
    )
    calls = []
    win._maybe_confirm_active_jobs(lambda: calls.append(1))
    assert calls == []  # blocked -- user chose Cancel (the dialog's overall "abort this
    # action" option, distinct from "Cancel Jobs")


def test_second_action_while_first_is_pending_does_not_double_dialog(
    qtbot, monkeypatch
):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.active_jobs.by_id["d1"] = {"status": "running"}

    from PySide6.QtWidgets import QMessageBox

    call_count = []

    def _counting_question(*a, **k):
        call_count.append(1)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", _counting_question)
    win._maybe_confirm_active_jobs(lambda: None)
    win._maybe_confirm_active_jobs(
        lambda: None
    )  # fired again before the first poll resolves
    assert call_count == [1]  # only one dialog shown, not two


def test_phase_worker_ready_updates_only_its_own_dataset(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.active_jobs.by_id = {
        "d1": {"status": "running"},
        "d2": {"status": "running"},
    }

    class _FakeWorker:
        pass

    worker_for_d1 = _FakeWorker()
    win._pipeline_service.phase_worker_ready.emit("d1", "s1", "nlsq", worker_for_d1)
    qtbot.wait(50)
    assert win._state.active_jobs.by_id["d1"].get("worker") is worker_for_d1
    assert (
        "worker" not in win._state.active_jobs.by_id["d2"]
    )  # NOT overwritten by d1's worker


@pytest.mark.parametrize("trigger", ["_on_new", "_on_close"])
def test_on_new_and_close_do_not_rebuild_while_jobs_active(qtbot, monkeypatch, trigger):
    # Regression test for the stale-controller scenario Task 10 flagged: rebuilding
    # window.py's workspace (via _rebuild -> _dispose_workspace/_build_workspace) while a
    # Pipeline batch is still mid-flight on a worker thread would leave the OLD
    # PipelineController connected to the persistent _pipeline_service, so it would keep
    # receiving dataset_run_started/dataset_run_finished and writing into an AppState
    # that's already been discarded. Blocking the rebuild outright (this task) is what
    # prevents that -- so _rebuild must never be called while active_jobs is non-empty.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.active_jobs.by_id["d1"] = {"status": "running"}

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
    )
    rebuild_calls = []
    monkeypatch.setattr(win, "_rebuild", lambda *a, **k: rebuild_calls.append(1))

    getattr(win, trigger)()

    assert rebuild_calls == []


@pytest.mark.parametrize("trigger", ["_on_save", "_on_save_as"])
def test_save_and_save_as_blocked_while_jobs_active(
    qtbot, monkeypatch, trigger, tmp_path
):
    # Spec §3.3: "Save snapshots job_history only; blocked while active_jobs is
    # non-empty" -- a running Pipeline batch mutates the library/job_history from a
    # worker thread, so a concurrent Save could serialize a torn snapshot.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.project.path = str(tmp_path / "p.rheojax")
    win._state.active_jobs.by_id["d1"] = {"status": "running"}

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    info_calls = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: info_calls.append(1)
    )
    save_calls = []
    monkeypatch.setattr(
        "rheojax.gui.foundation.project_codec.save_project_v2",
        lambda *a, **k: save_calls.append(1),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(tmp_path / "x.rheojax"), ""),
    )

    getattr(win, trigger)()

    assert info_calls == [1]  # blocked -- user was told, not silently ignored
    assert save_calls == []  # save_project_v2 never invoked


def test_on_open_does_not_rebuild_while_jobs_active(qtbot, monkeypatch):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.active_jobs.by_id["d1"] = {"status": "running"}

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
    )
    dialog_calls = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (dialog_calls.append(1), ("", ""))[1],
    )
    rebuild_calls = []
    monkeypatch.setattr(win, "_rebuild", lambda *a, **k: rebuild_calls.append(1))

    win._on_open()

    assert rebuild_calls == []
    assert dialog_calls == []  # blocked before even reaching the file-open dialog
