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
    ref = DatasetRef(
        id="d1",
        name="d1",
        protocol_type="oscillation",
        origin="imported",
        units={},
        row_count=1,
        hash="h",
        provenance={},
        lineage=[],
    )
    state.library.add(ref)
    state.library.store_payload(
        "d1", RheoData(x=[0.1, 1.0], y=[1.0, 2.0], initial_test_mode="oscillation")
    )
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    win.set_mode("pipeline")
    pipeline_body = win._pipeline_bodies[0]
    pipeline_body.add_step(
        "export", {"path": str(tmp_path / "out.csv"), "format": "csv"}
    )
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
        lambda dataset_id: captured.setdefault(
            dataset_id, dict(state.active_jobs.by_id)
        )
    )

    with qtbot.waitSignal(win._pipeline_service.dataset_run_finished, timeout=5000):
        pipeline_body.run_requested.emit()

    assert "d1" in captured
    assert "d1" in captured["d1"]


def test_pipeline_run_requested_passes_step_list_copies_not_live_refs(
    qtbot, monkeypatch
):
    # Regression: PipelineBatchRunner used to receive the live
    # pipeline_state.steps/.selected_dataset_ids list objects by reference
    # and iterate them on a worker thread across the whole (potentially
    # long) batch, while Add/Remove Step and the dataset picker in
    # step1_configure_run.py stayed clickable on the GUI thread and mutated
    # those same lists in place -- an in-flight batch could silently skip,
    # duplicate, or misconfigure steps for whichever dataset happened to be
    # substituting when the edit landed. The window must now pass shallow
    # copies at construction time.
    from PySide6.QtCore import QThreadPool

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    win.set_mode("pipeline")
    pipeline_body = win._pipeline_bodies[0]
    pipeline_body.add_step(
        "export", {"path": "out.csv", "format": "csv"}
    )
    pipeline_body.set_selected_dataset_ids(["d1"])

    started_runners = []
    monkeypatch.setattr(
        QThreadPool.globalInstance(),
        "start",
        lambda runner: started_runners.append(runner),
    )

    win._on_pipeline_run_requested()

    assert len(started_runners) == 1
    runner = started_runners[0]
    assert runner._steps is not state.pipeline.steps
    assert runner._selected_dataset_ids is not state.pipeline.selected_dataset_ids
    assert runner._steps == state.pipeline.steps  # same content, distinct object
    assert runner._selected_dataset_ids == state.pipeline.selected_dataset_ids

    # Mutating the live state after the runner was constructed must not
    # reach the runner's copies.
    state.pipeline.steps.clear()
    state.pipeline.selected_dataset_ids.clear()
    assert len(runner._steps) == 1
    assert runner._selected_dataset_ids == ["d1"]


def test_run_all_ignored_while_batch_already_running(qtbot, monkeypatch):
    from PySide6.QtCore import QThreadPool

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    win.set_mode("pipeline")

    start_calls = []
    monkeypatch.setattr(
        QThreadPool.globalInstance(),
        "start",
        lambda *a, **kw: start_calls.append((a, kw)),
    )

    # Simulate a batch already in flight.
    state.active_jobs.by_id["d1"] = {"worker": None}

    win._on_pipeline_run_requested()

    assert start_calls == []


def test_run_all_double_click_starts_only_one_runner(qtbot, monkeypatch):
    # Regression test: PipelineBatchRunner.run() only writes its first per-dataset
    # active_jobs entry once it actually executes on a worker thread -- QThreadPool.start()
    # doesn't guarantee that happens before it returns. Mocking start() as a no-op
    # (like test_run_all_ignored_while_batch_already_running above) simulates exactly
    # that gap: active_jobs stays whatever _on_pipeline_run_requested itself wrote,
    # since the runner body never runs. A second call in that gap must still be
    # ignored, not start a second runner.
    from PySide6.QtCore import QThreadPool

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    win.set_mode("pipeline")

    start_calls = []
    monkeypatch.setattr(
        QThreadPool.globalInstance(),
        "start",
        lambda *a, **kw: start_calls.append((a, kw)),
    )

    win._on_pipeline_run_requested()
    win._on_pipeline_run_requested()

    assert len(start_calls) == 1


def test_pipeline_progress_signals_are_wired_to_log_dock(qtbot):
    # Regression: step_started/step_completed/step_failed, pipeline_started/
    # completed/failed, and step_phase_started/completed/failed used to have
    # zero listeners anywhere -- real per-step/per-phase progress was
    # computed and silently discarded during a pipeline run, with no
    # user-visible feedback beyond the coarse per-dataset start/finish.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    service = win._pipeline_service

    service.pipeline_started.emit()
    service.step_started.emit("s1")
    service.step_phase_started.emit("s1", "nlsq")
    service.step_phase_completed.emit("s1", "nlsq")
    service.step_completed.emit("s1")
    service.step_failed.emit("s2", "boom")
    service.step_phase_failed.emit("s3", "nuts", "kaboom")
    service.pipeline_failed.emit("fatal")
    service.pipeline_completed.emit()
    qtbot.wait(50)  # these connections are QueuedConnection

    messages = [msg for _, msg in win.log_dock._records]
    assert any("Pipeline run started" in m for m in messages)
    assert any("Step s1 started" in m for m in messages)
    assert any("s1: nlsq phase started" in m for m in messages)
    assert any("s1: nlsq phase completed" in m for m in messages)
    assert any("Step s1 completed" in m for m in messages)
    assert any("Step s2 failed: boom" in m for m in messages)
    assert any("s3: nuts phase failed: kaboom" in m for m in messages)
    assert any("Pipeline run failed: fatal" in m for m in messages)
    assert any("Pipeline run completed" in m for m in messages)
