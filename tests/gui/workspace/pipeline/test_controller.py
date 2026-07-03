import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import AppState
from rheojax.gui.services.pipeline_execution_service import PipelineExecutionService
from rheojax.gui.workspace.pipeline.controller import PipelineController


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
