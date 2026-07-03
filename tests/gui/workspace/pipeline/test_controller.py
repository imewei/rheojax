import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import AppState
from rheojax.gui.services.pipeline_execution_service import PipelineExecutionService
from rheojax.gui.workspace.pipeline.controller import PipelineController
from rheojax.gui.workspace.window import WorkspaceWindow


def test_commit_job_result_moves_active_to_history_and_marks_dirty(qtbot):
    state = AppState()
    state.active_jobs.by_id["d1"] = {"status": "running"}
    svc = PipelineExecutionService()
    dirty_calls = []
    ctl = PipelineController(state, svc, on_dirty=lambda: dirty_calls.append(1))

    record = {"dataset_id": "d1", "status": "completed", "error": None, "step_results": {}}
    ctl._commit_job_result("d1", record)

    assert "d1" not in state.active_jobs.by_id
    assert len(state.job_history.by_id) == 1
    stored = next(iter(state.job_history.by_id.values()))
    assert stored["status"] == "completed"
    assert state.pipeline.job_id is not None
    assert dirty_calls == [1]


def test_commit_job_result_connected_to_dataset_run_finished(qtbot):
    state = AppState()
    state.active_jobs.by_id["d1"] = {"status": "running"}
    svc = PipelineExecutionService()
    ctl = PipelineController(state, svc, on_dirty=lambda: None)

    record = {"dataset_id": "d1", "status": "completed", "error": None, "step_results": {}}
    with qtbot.waitSignal(svc.dataset_run_finished, timeout=1000):
        svc.dataset_run_finished.emit("d1", record)
    assert "d1" not in state.active_jobs.by_id


def test_dataset_run_started_registers_active_job(qtbot):
    state = AppState()
    svc = PipelineExecutionService()
    ctl = PipelineController(state, svc, on_dirty=lambda: None)
    assert "d1" not in state.active_jobs.by_id

    with qtbot.waitSignal(svc.dataset_run_started, timeout=1000):
        svc.dataset_run_started.emit("d1")
    assert state.active_jobs.by_id["d1"] == {"status": "running"}


def test_stale_controller_guarded_after_rebuild_does_not_mutate_discarded_state(qtbot):
    """Reproduces the Plan 2 Task 11 gap: a PipelineController connected against the
    persistent PipelineExecutionService must stop mutating its AppState once
    WorkspaceWindow rebuilds past it, even though Qt keeps the old signal connection
    alive (only the Python-level self._controllers reference is dropped on rebuild)."""
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)

    stale_state = win._state
    stale_ctl = win._controllers["pipeline"]
    stale_state.active_jobs.by_id["d1"] = {"status": "running"}

    win._rebuild(AppState())  # bumps win._epoch; stale_ctl's connections survive regardless
    fresh_state = win._state
    fresh_ctl = win._controllers["pipeline"]
    assert fresh_ctl is not stale_ctl
    assert fresh_state is not stale_state

    record = {"dataset_id": "d1", "status": "completed", "error": None, "step_results": {}}
    with qtbot.waitSignal(win._pipeline_service.dataset_run_finished, timeout=1000):
        win._pipeline_service.dataset_run_finished.emit("d1", record)

    # Stale controller's handler was a no-op: the discarded state is untouched.
    assert stale_state.active_jobs.by_id == {"d1": {"status": "running"}}
    assert stale_state.job_history.by_id == {}

    # Fresh controller still commits normally.
    assert "d1" not in fresh_state.active_jobs.by_id
    assert len(fresh_state.job_history.by_id) == 1
