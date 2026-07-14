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


def test_stale_poll_chain_timer_is_parented_to_window(qtbot):
    # Regression: this test's active_jobs never empties, so _poll_active_jobs_then's
    # QTimer chain never naturally resolves within a single test. If that timer
    # isn't parented to win, Qt won't cancel it when win is destroyed -- it
    # keeps re-arming on the (process-wide, shared) QApplication event loop and
    # can eventually fire the real QMessageBox.warning() during a LATER,
    # unrelated test's teardown, crashing the whole xdist worker with a fatal
    # Qt/shiboken abort. This is exactly what happened under the full suite
    # (see PR history). Rather than racing the real 30s chain or Qt's/CPython's
    # GC timing to reproduce the crash, assert the architectural property that
    # prevents it: the scheduled QTimer must be a child of win, so Qt's
    # parent-child ownership destroys it (and stops it firing) the moment win
    # is destroyed, regardless of exactly when that happens.
    from PySide6.QtCore import QTimer

    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.active_jobs.by_id["d1"] = {"status": "running"}
    win._active_jobs_action_pending = True

    created_timers = []
    real_init = QTimer.__init__

    def _capturing_init(self, *a, **k):
        real_init(self, *a, **k)
        created_timers.append(self)

    QTimer.__init__ = _capturing_init
    try:
        win._poll_active_jobs_then(lambda: None, remaining_polls=1)
    finally:
        QTimer.__init__ = real_init

    assert created_timers, "_poll_active_jobs_then did not create a QTimer"
    assert created_timers[-1].parent() is win, (
        "The active-jobs poll QTimer is not parented to the window, so Qt "
        "cannot cancel it when the window is destroyed -- it will keep "
        "firing (and can call the real QMessageBox.warning()) into whatever "
        "unrelated test happens to be running when its 250ms tick lands."
    )


def test_poll_timeout_does_not_show_modal_when_window_not_visible(qtbot, monkeypatch):
    # Regression: a stale/orphaned WorkspaceWindow (never shown, or already
    # closed -- e.g. left behind by a previous test whose poll chain never
    # resolved) must not pop a real modal QMessageBox.warning() once its 30s
    # budget expires. _is_qobject_alive only catches the case where the C++
    # object is already destroyed; it does NOT catch a window that is still
    # fully alive but orphaned. Showing a modal from an orphaned window's
    # timer callback, reentrant into whatever unrelated test happens to be
    # running at that moment, is exactly what caused a reproducible
    # Fatal Python error: Aborted crash under the full GUI test suite.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    assert not win.isVisible()  # never shown -- the orphaned-window case

    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(1))

    win._state.active_jobs.by_id["d1"] = {"status": "running"}
    win._active_jobs_action_pending = True
    win._poll_active_jobs_then(lambda: None, remaining_polls=0)

    assert warned == []
    assert win._active_jobs_action_pending is False


def test_poll_timeout_shows_modal_when_window_visible(qtbot, monkeypatch):
    # Complement to the test above: a real, visible window (the only case
    # that occurs in actual usage -- a user closing/starting-new always has
    # this window shown) must still get the warning.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)

    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(1))

    win._state.active_jobs.by_id["d1"] = {"status": "running"}
    win._active_jobs_action_pending = True
    win._poll_active_jobs_then(lambda: None, remaining_polls=0)

    assert warned == [1]


def test_phase_worker_ready_takes_active_jobs_lock(qtbot, monkeypatch):
    # Regression: _on_phase_worker_ready used to read/write
    # active_jobs.by_id without taking active_jobs.lock, unlike every other
    # call site in this file -- a real TOCTOU gap against
    # PipelineBatchRunner, which mutates the same dict from a worker thread.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.active_jobs.by_id["d1"] = {"status": "running"}

    acquired = []
    real_lock = win._state.active_jobs.lock

    class _SpyLock:
        def __enter__(self):
            acquired.append(1)
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)

    monkeypatch.setattr(win._state.active_jobs, "lock", _SpyLock())

    class _FakeWorker:
        pass

    win._on_phase_worker_ready("d1", "s1", "nlsq", _FakeWorker())

    assert acquired == [1]
    assert win._state.active_jobs.by_id["d1"].get("worker") is not None


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


def test_on_close_actually_closes_window_not_reset(qtbot, monkeypatch):
    # Regression: _on_close used to call self._rebuild(AppState()) (copy-pasted from
    # _on_new), which just blanked the project instead of closing the window -- File>Close
    # appeared to do nothing. It must route through the same _confirmed_close chain
    # closeEvent uses for the OS window controls.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)

    confirmed_close_calls = []
    monkeypatch.setattr(
        win, "_confirmed_close", lambda: confirmed_close_calls.append(1)
    )
    rebuild_calls = []
    monkeypatch.setattr(win, "_rebuild", lambda *a, **k: rebuild_calls.append(1))

    win._on_close()

    assert confirmed_close_calls == [1]
    assert rebuild_calls == []


def test_on_close_declines_active_jobs_cancel_does_not_close(qtbot, monkeypatch):
    # Coverage gap flagged in review: the test above only exercises _on_close's
    # trivial fast path (no jobs, not dirty). This exercises the active-jobs branch
    # specifically -- user declines to cancel running jobs, so _confirmed_close must
    # never fire and the window must stay open.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win._state.active_jobs.by_id["d1"] = {"status": "running"}

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
    )
    confirmed_close_calls = []
    monkeypatch.setattr(
        win, "_confirmed_close", lambda: confirmed_close_calls.append(1)
    )

    win._on_close()

    assert confirmed_close_calls == []


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
