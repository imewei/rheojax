import threading

import pytest

pytest.importorskip("PySide6")

from rheojax.core.data import RheoData
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.project_codec import load_project_v2, save_project_v2
from rheojax.gui.foundation.state import AppState, PipelineStepConfig
from rheojax.gui.services.pipeline_execution_service import PipelineExecutionService
from rheojax.gui.workspace.pipeline.batch_runner import PipelineBatchRunner


def _ref(id_, ptype):
    return DatasetRef(
        id=id_,
        name=id_,
        protocol_type=ptype,
        origin="imported",
        units={},
        row_count=3,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_batch_runner_processes_each_selected_dataset(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "subprocess")
    lib = DatasetLibrary()
    for id_ in ("d1", "d2"):
        lib.add(_ref(id_, "oscillation"))
        lib.store_payload(
            id_, RheoData(x=[0.1, 1.0], y=[1.0, 2.0], initial_test_mode="oscillation")
        )

    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(
            id="s1",
            step_type="export",
            config={"path": str(tmp_path / "{id}.csv"), "format": "csv"},
        )
    ]
    runner = PipelineBatchRunner(
        service=svc,
        steps=steps,
        selected_dataset_ids=["d1", "d2"],
        library=lib,
        stop_requested=threading.Event(),
    )

    started, finished = [], []
    svc.dataset_run_started.connect(lambda dsid: started.append(dsid))
    svc.dataset_run_finished.connect(lambda dsid, record: finished.append(dsid))
    with qtbot.waitSignal(svc.dataset_run_finished, timeout=5000):
        runner.run()
    assert started == ["d1", "d2"]  # started signal fires per-dataset, in order
    assert set(finished) == {"d1", "d2"}


def test_batch_runner_prepare_job_record_keeps_raw_result_uncompressed(
    qtbot, tmp_path, monkeypatch
):
    # _prepare_job_record() must NOT write any HDF5 file or split the result -- that's
    # save_project_v2()'s job at save time, not the batch runner's job at execution time.
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "subprocess")
    lib = DatasetLibrary()
    lib.add(_ref("d1", "relaxation"))
    lib.store_payload(
        "d1", RheoData(x=[0.1, 1.0], y=[100.0, 50.0], initial_test_mode="relaxation")
    )
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(
            id="s1",
            step_type="fit",
            config={"model_name": "maxwell", "run_nuts": False},
        )
    ]
    runner = PipelineBatchRunner(
        service=svc,
        steps=steps,
        selected_dataset_ids=["d1"],
        library=lib,
        stop_requested=threading.Event(),
    )
    records = []
    svc.dataset_run_finished.connect(lambda dsid, record: records.append(record))
    with qtbot.waitSignal(svc.dataset_run_finished, timeout=5000):
        runner.run()
    phase = records[0]["step_results"]["s1"]["nlsq"]
    assert "result_ref" not in phase and "result_meta" not in phase
    assert isinstance(phase["result"], dict)  # still the raw, unpersisted result dict


def test_batch_runner_processes_datasets_in_order_and_skips_after_stop(
    qtbot, tmp_path, monkeypatch
):
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "subprocess")
    lib = DatasetLibrary()
    for id_ in ("d1", "d2", "d3"):
        lib.add(_ref(id_, "oscillation"))
        lib.store_payload(
            id_, RheoData(x=[0.1, 1.0], y=[1.0, 2.0], initial_test_mode="oscillation")
        )

    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(
            id="s1",
            step_type="export",
            config={"path": str(tmp_path / "{id}.csv"), "format": "csv"},
        )
    ]
    stop_requested = threading.Event()
    runner = PipelineBatchRunner(
        service=svc,
        steps=steps,
        selected_dataset_ids=["d1", "d2", "d3"],
        library=lib,
        stop_requested=stop_requested,
    )

    started = []

    def _on_started(dsid):
        started.append(dsid)
        if dsid == "d2":
            stop_requested.set()

    svc.dataset_run_started.connect(_on_started)
    runner.run()
    assert started == ["d1", "d2"]  # d3 never started once stop_requested was set


def test_batch_runner_job_record_round_trips_through_save_project_v2(
    qtbot, tmp_path, monkeypatch
):
    # Cross-plan integration: _prepare_job_record()'s output dict must slot directly into
    # AppState.job_history.by_id and survive Plan 1's save_project_v2/load_project_v2 round
    # trip -- the exact shape contract described in project_codec.py's job_history-walking
    # section (search "job_history_out"/"step_results_out").
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "subprocess")
    lib = DatasetLibrary()
    lib.add(_ref("d1", "relaxation"))
    lib.store_payload(
        "d1", RheoData(x=[0.1, 1.0], y=[100.0, 50.0], initial_test_mode="relaxation")
    )
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(
            id="s1",
            step_type="fit",
            config={"model_name": "maxwell", "run_nuts": False},
        )
    ]
    runner = PipelineBatchRunner(
        service=svc,
        steps=steps,
        selected_dataset_ids=["d1"],
        library=lib,
        stop_requested=threading.Event(),
    )
    records = []
    svc.dataset_run_finished.connect(lambda dsid, record: records.append(record))
    with qtbot.waitSignal(svc.dataset_run_finished, timeout=5000):
        runner.run()
    record = records[0]
    assert record["status"] == "completed"

    state = AppState()
    state.job_history.by_id["job1"] = record
    path = tmp_path / "project.rheojax"
    save_project_v2(
        state, path
    )  # must not raise -- proves the shape matches project_codec's
    # expectations (fit phase result_ref/result_meta split)
    loaded = load_project_v2(path)

    loaded_phase = loaded.job_history.by_id["job1"]["step_results"]["s1"]["nlsq"]
    assert loaded_phase["status"] == "completed"
    assert isinstance(loaded_phase["result"], dict)
    assert "result_ref" not in loaded_phase


def test_batch_runner_emits_finished_with_failed_record_on_fatal_precondition_error(
    qtbot, monkeypatch
):
    # A WorkerIsolationRequiredError (misconfigured environment) escaping execute() must
    # still result in dataset_run_finished being emitted with a terminal "failed" record
    # -- otherwise this dataset's active_jobs entry (registered by dataset_run_started,
    # which already fired) is never cleared by PipelineController._commit_job_result(),
    # leaving it stuck "running" forever. The batch must also stop rather than repeat the
    # identical fatal error for every remaining dataset.
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "thread")
    lib = DatasetLibrary()
    for id_ in ("d1", "d2"):
        lib.add(_ref(id_, "relaxation"))
        lib.store_payload(
            id_, RheoData(x=[0.1, 1.0], y=[100.0, 50.0], initial_test_mode="relaxation")
        )
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(
            id="s1",
            step_type="fit",
            config={"model_name": "maxwell", "run_nuts": False},
        )
    ]
    runner = PipelineBatchRunner(
        service=svc,
        steps=steps,
        selected_dataset_ids=["d1", "d2"],
        library=lib,
        stop_requested=threading.Event(),
    )
    started, records = [], []
    svc.dataset_run_started.connect(lambda dsid: started.append(dsid))
    svc.dataset_run_finished.connect(lambda dsid, record: records.append(record))
    with qtbot.waitSignal(svc.dataset_run_finished, timeout=5000):
        runner.run()
    assert started == [
        "d1"
    ]  # fatal precondition error stops the batch -- d2 never started
    assert len(records) == 2
    assert records[0]["status"] == "failed"
    assert "subprocess" in records[0]["error"]
    assert records[0]["step_results"] == {}
    # d2 never ran (never got dataset_run_started), but must still get an explicit
    # "skipped" record -- not silent absence -- so it's distinguishable from
    # "not yet attempted" when reviewing job_history afterwards.
    assert records[1]["dataset_id"] == "d2"
    assert records[1]["status"] == "skipped"
    assert records[1]["step_results"] == {}
    # The skip record must surface *why* -- both the fixed "Batch aborted"
    # framing and the underlying failure reason text (not just a generic
    # placeholder), so reviewing job_history afterwards explains the skip.
    assert "Batch aborted" in records[1]["error"]
    assert "subprocess" in records[1]["error"]


def test_batch_runner_skips_all_datasets_queued_behind_fatal_precondition_error(
    qtbot, monkeypatch
):
    # With 3+ datasets, every dataset after the fatal one must get a skipped
    # record -- not just the immediate next one. This exercises the
    # `selected_dataset_ids[idx + 1 :]` slice for a queue longer than 2.
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "thread")
    lib = DatasetLibrary()
    for id_ in ("d1", "d2", "d3", "d4"):
        lib.add(_ref(id_, "relaxation"))
        lib.store_payload(
            id_, RheoData(x=[0.1, 1.0], y=[100.0, 50.0], initial_test_mode="relaxation")
        )
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(
            id="s1",
            step_type="fit",
            config={"model_name": "maxwell", "run_nuts": False},
        )
    ]
    runner = PipelineBatchRunner(
        service=svc,
        steps=steps,
        selected_dataset_ids=["d1", "d2", "d3", "d4"],
        library=lib,
        stop_requested=threading.Event(),
    )
    started, records = [], []
    svc.dataset_run_started.connect(lambda dsid: started.append(dsid))
    svc.dataset_run_finished.connect(lambda dsid, record: records.append(record))
    with qtbot.waitSignal(svc.dataset_run_finished, timeout=5000):
        runner.run()
    assert started == ["d1"]  # fatal precondition error stops the batch
    assert len(records) == 4
    assert records[0]["status"] == "failed"
    for record, dataset_id in zip(records[1:], ("d2", "d3", "d4")):
        assert record["dataset_id"] == dataset_id
        assert record["status"] == "skipped"
        assert record["step_results"] == {}
        assert "Batch aborted" in record["error"]


def test_batch_runner_transform_step_record_round_trips_output_as_rheodata(
    qtbot, tmp_path, monkeypatch
):
    # A "transform" step's record must carry its RheoData under literally "output" so
    # save_project_v2's `elif "output" in step_record` branch detects and persists it.
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "subprocess")
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    lib.store_payload(
        "d1", RheoData(x=[0.1, 1.0], y=[1.0, 2.0], initial_test_mode="oscillation")
    )
    svc = PipelineExecutionService()
    steps = [PipelineStepConfig(id="s1", step_type="transform", config={"name": "fft"})]
    runner = PipelineBatchRunner(
        service=svc,
        steps=steps,
        selected_dataset_ids=["d1"],
        library=lib,
        stop_requested=threading.Event(),
    )
    records = []
    svc.dataset_run_finished.connect(lambda dsid, record: records.append(record))
    with qtbot.waitSignal(svc.dataset_run_finished, timeout=5000):
        runner.run()
    record = records[0]
    step_record = record["step_results"]["s1"]
    assert step_record["step_type"] == "other"
    assert isinstance(step_record["output"], RheoData)

    state = AppState()
    state.job_history.by_id["job1"] = record
    path = tmp_path / "project.rheojax"
    save_project_v2(
        state, path
    )  # must not raise TypeError trying to json.dumps() a RheoData
    loaded = load_project_v2(path)

    loaded_step = loaded.job_history.by_id["job1"]["step_results"]["s1"]
    assert "output_ref" not in loaded_step
    import numpy as np

    np.testing.assert_array_equal(loaded_step["output"].x, step_record["output"].x)
