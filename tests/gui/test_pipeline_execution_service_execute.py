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
