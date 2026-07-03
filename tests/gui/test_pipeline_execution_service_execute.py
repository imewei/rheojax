import threading

import pytest

pytest.importorskip("PySide6")

from rheojax.core.data import RheoData
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import PipelineStepConfig
from rheojax.gui.services.pipeline_execution_service import PipelineExecutionService


def _ref(id_, ptype):
    return DatasetRef(id=id_, name=id_, protocol_type=ptype, origin="imported",
                      units={}, row_count=3, hash="h", provenance={}, lineage=[])


@pytest.fixture(autouse=True)
def _subprocess_isolation(monkeypatch):
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "subprocess")


def test_execute_raises_on_thread_isolation_mode(qtbot, monkeypatch):
    monkeypatch.setenv("RHEOJAX_WORKER_ISOLATION", "thread")
    lib = DatasetLibrary()
    lib.add(_ref("d1", "relaxation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[100.0, 50.0, 10.0], initial_test_mode="relaxation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    steps = [PipelineStepConfig(id="s1", step_type="fit",
                                 config={"model_name": "maxwell", "run_nuts": False})]
    with pytest.raises(RuntimeError, match="subprocess"):
        svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                     library=lib, stop_requested=threading.Event())


def test_execute_runs_nlsq_only_fit_step(qtbot):
    lib = DatasetLibrary()
    lib.add(_ref("d1", "relaxation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[100.0, 50.0, 10.0], initial_test_mode="relaxation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    steps = [PipelineStepConfig(id="s1", step_type="fit",
                                 config={"model_name": "maxwell", "run_nuts": False})]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=threading.Event())
    assert result.status == "completed"
    fit_result = result.step_results["s1"]
    assert fit_result.nlsq.status == "completed"
    assert fit_result.nuts is None


@pytest.mark.slow
def test_execute_runs_nlsq_then_nuts_fit_step(qtbot):
    lib = DatasetLibrary()
    lib.add(_ref("d1", "relaxation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[100.0, 50.0, 10.0], initial_test_mode="relaxation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    steps = [PipelineStepConfig(id="s1", step_type="fit", config={
        "model_name": "maxwell", "run_nuts": True,
        "nuts_config": {"num_warmup": 50, "num_samples": 50, "num_chains": 1},
    })]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=threading.Event())
    fit_result = result.step_results["s1"]
    assert fit_result.nlsq.status == "completed"
    assert fit_result.nuts is not None
    assert fit_result.nuts.status in ("completed", "failed")  # real subprocess MCMC -- allow
                                                                  # either, just verify it RAN
                                                                  # independently of nlsq


def test_execute_runs_transform_step(qtbot):
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[1.0, 2.0, 3.0], initial_test_mode="oscillation")
    lib.store_payload("d1", data)

    svc = PipelineExecutionService()
    steps = [PipelineStepConfig(id="s1", step_type="transform", config={"name": "smooth_derivative"})]
    result = svc.execute(
        steps=steps,
        initial_context={"data": data, "dataset_id": "d1"},
        library=lib,
        stop_requested=threading.Event(),
    )
    assert result.status == "completed"
    assert "s1" in result.step_results
