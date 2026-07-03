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


def test_execute_runs_export_step(qtbot, tmp_path):
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[1.0, 2.0, 3.0], initial_test_mode="oscillation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    out_path = str(tmp_path / "out.csv")
    steps = [PipelineStepConfig(id="s1", step_type="export", config={"path": out_path, "format": "csv"})]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=threading.Event())
    assert result.status == "completed"
    assert result.step_results["s1"]["paths"] == [out_path]
    assert (tmp_path / "out.csv").exists()


def test_execute_reports_failed_when_export_raises(qtbot, tmp_path, monkeypatch):
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[1.0, 2.0, 3.0], initial_test_mode="oscillation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()

    from rheojax.gui.services.export_service import ExportService

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ExportService, "export_data", _raise)

    out_path = str(tmp_path / "out.csv")
    steps = [PipelineStepConfig(id="s1", step_type="export", config={"path": out_path, "format": "csv"})]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=threading.Event())
    assert result.status == "failed"
    assert "disk full" in result.error
    assert "s1" not in result.step_results
    assert not (tmp_path / "out.csv").exists()


def test_execute_returns_cancelled_when_stop_requested_set(qtbot):
    lib = DatasetLibrary()
    data = RheoData(x=[0.1, 1.0], y=[1.0, 2.0], initial_test_mode="oscillation")
    svc = PipelineExecutionService()
    stop = threading.Event()
    stop.set()
    steps = [PipelineStepConfig(id="s1", step_type="export", config={"path": "unused.csv"})]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=stop)
    assert result.status == "cancelled"
    assert result.step_results == {}


def test_transform_output_persisted_when_export_step_follows(qtbot, tmp_path):
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[1.0, 2.0, 3.0], initial_test_mode="oscillation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(id="s1", step_type="transform", config={"name": "smooth_derivative"}),
        PipelineStepConfig(id="s2", step_type="export", config={"path": str(tmp_path / "out.csv"), "format": "csv"}),
    ]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=threading.Event())
    assert result.status == "completed"
    # exactly one new library entry beyond the original "d1"
    new_ids = {r.id for r in lib.all()} - {"d1"}
    assert len(new_ids) == 1


def test_transform_output_not_persisted_when_no_later_step_consumes_it(qtbot):
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[1.0, 2.0, 3.0], initial_test_mode="oscillation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    steps = [PipelineStepConfig(id="s1", step_type="transform", config={"name": "smooth_derivative"})]
    svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                library=lib, stop_requested=threading.Event())
    assert {r.id for r in lib.all()} == {"d1"}


def test_terminal_transform_after_unrelated_fit_step_is_not_persisted(qtbot):
    # A terminal transform preceded only by an unrelated fit step (not another transform)
    # has nothing downstream reading its output -- it must NOT be persisted, even though
    # len(steps) > 1.
    lib = DatasetLibrary()
    lib.add(_ref("d1", "relaxation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[100.0, 50.0, 10.0], initial_test_mode="relaxation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(id="s1", step_type="fit",
                            config={"model_name": "maxwell", "run_nuts": False}),
        PipelineStepConfig(id="s2", step_type="transform", config={"name": "smooth_derivative"}),
    ]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=threading.Event())
    assert result.status == "completed"
    assert result.step_results["s2"]["dataset_id"] is None
    assert {r.id for r in lib.all()} == {"d1"}


def test_transform_output_persisted_when_a_second_transform_follows(qtbot):
    # A transform->transform chain must also count as "consumed downstream" -- the first
    # transform's output is exactly what the second transform reads from context["data"].
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[1.0, 2.0, 3.0], initial_test_mode="oscillation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(id="s1", step_type="transform", config={"name": "smooth_derivative"}),
        PipelineStepConfig(id="s2", step_type="transform", config={"name": "smooth_derivative"}),
    ]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=threading.Event())
    assert result.status == "completed"
    new_ids = {r.id for r in lib.all()} - {"d1"}
    assert len(new_ids) == 2  # both transforms' outputs persisted


def test_second_transform_lineage_points_at_first_transforms_output_not_original(qtbot):
    # context["dataset_id"] must be updated to the newly-persisted id after each persisted
    # transform, so a later step's infer_output_protocol() lookup (and any lineage it records)
    # reflects the immediately-preceding derived dataset, not the original input dataset.
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    data = RheoData(x=[0.1, 1.0, 10.0], y=[1.0, 2.0, 3.0], initial_test_mode="oscillation")
    lib.store_payload("d1", data)
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(id="s1", step_type="transform", config={"name": "smooth_derivative"}),
        PipelineStepConfig(id="s2", step_type="transform", config={"name": "smooth_derivative"}),
    ]
    result = svc.execute(steps=steps, initial_context={"data": data, "dataset_id": "d1"},
                          library=lib, stop_requested=threading.Event())
    first_new_id = result.step_results["s1"]["dataset_id"]
    second_ref = lib.get(result.step_results["s2"]["dataset_id"])
    assert second_ref.lineage == [first_new_id]  # not ["d1"]


def test_duplicate_step_ids_rejected(qtbot):
    lib = DatasetLibrary()
    svc = PipelineExecutionService()
    steps = [
        PipelineStepConfig(id="s1", step_type="export", config={"path": "a.csv"}),
        PipelineStepConfig(id="s1", step_type="export", config={"path": "b.csv"}),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        svc.execute(steps=steps, initial_context={"data": object(), "dataset_id": "d1"},
                    library=lib, stop_requested=threading.Event())
