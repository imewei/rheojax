import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def test_pipeline_mode_is_available(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    assert "pipeline" in win.MODES
    win.set_mode("pipeline")
    assert win.mode() == "pipeline"


def test_pipeline_toolbar_button_exists(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    assert win._pipeline_btn.text() == "Pipeline"


def test_run_all_populates_active_jobs(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "subprocess")
    from rheojax.core.data import RheoData
    from rheojax.gui.foundation.library import DatasetRef

    state = AppState()
    ref = DatasetRef(id="d1", name="d1", protocol_type="oscillation", origin="imported",
                      units={}, row_count=1, hash="h", provenance={}, lineage=[])
    state.library.add(ref)
    state.library.store_payload("d1", RheoData(x=[0.1, 1.0], y=[1.0, 2.0], initial_test_mode="oscillation"))
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    win.set_mode("pipeline")
    pipeline_body = win._pipeline_bodies[0]
    pipeline_body.add_step("export", {"path": str(tmp_path / "out.csv"), "format": "csv"})
    pipeline_body.set_selected_dataset_ids(["d1"])

    # For a single-dataset, single-export-step batch, dataset_run_started and
    # dataset_run_finished can both land on the GUI thread within the same event-loop
    # tick -- a fixed post-signal sleep races the batch's real completion and is
    # flaky-by-construction (it can observe active_jobs after PipelineController's
    # dataset_run_finished handler has already popped the entry). Capture active_jobs
    # synchronously inside the dataset_run_started handler instead: Qt delivers
    # same-signal slots in connection order, and PipelineController's own
    # dataset_run_started handler (connected in build_pipeline_controller, before this
    # test connects) always runs first, so by the time this probe fires, active_jobs
    # is guaranteed to already be populated.
    captured: dict[str, dict] = {}
    win._pipeline_service.dataset_run_started.connect(
        lambda dataset_id: captured.setdefault(dataset_id, dict(state.active_jobs.by_id))
    )

    with qtbot.waitSignal(win._pipeline_service.dataset_run_finished, timeout=5000):
        pipeline_body.run_requested.emit()

    assert "d1" in captured
    assert "d1" in captured["d1"]


def test_run_all_ignored_while_batch_already_running(qtbot, monkeypatch):
    from PySide6.QtCore import QThreadPool

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    win.set_mode("pipeline")

    start_calls = []
    monkeypatch.setattr(
        QThreadPool.globalInstance(), "start", lambda *a, **kw: start_calls.append((a, kw))
    )

    # Simulate a batch already in flight.
    state.active_jobs.by_id["d1"] = {"worker": None}

    win._on_pipeline_run_requested()

    assert start_calls == []
