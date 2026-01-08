import numpy as np
import pytest

from rheojax.gui.pages.bayesian_page import BayesianPage
from rheojax.gui.pages.export_page import ExportPage
from rheojax.gui.state.store import (
    AppState,
    DatasetState,
    PipelineState,
    PipelineStep,
    StateStore,
    StepStatus,
)


def setup_function() -> None:
    StateStore.reset()


def test_get_state_returns_clone_and_set_test_mode_updates_dataset():
    store = StateStore()
    ds = DatasetState(id="d1", name="ds", file_path=None, test_mode="oscillation")
    store._state = AppState(datasets={"d1": ds}, active_dataset_id="d1")

    store.dispatch("SET_TEST_MODE", {"test_mode": "creep"})

    state = store.get_state()
    assert state.datasets["d1"].test_mode == "creep"
    # Mutating clone should not affect store
    state.datasets["d1"].test_mode = "relaxation"
    assert store.get_state().datasets["d1"].test_mode == "creep"


def test_cancel_jobs_marks_pipeline_warning():
    store = StateStore()
    pipeline = PipelineState(steps=dict.fromkeys(PipelineStep, StepStatus.PENDING))
    pipeline.current_step = PipelineStep.FIT
    pipeline.steps[PipelineStep.FIT] = StepStatus.ACTIVE
    store._state = AppState(pipeline_state=pipeline)

    store.dispatch("CANCEL_JOBS")

    state = store.get_state()
    assert state.pipeline_state.steps[PipelineStep.FIT] == StepStatus.WARNING
    assert state.pipeline_state.current_step is None


def test_export_raw_data_converts_dataset(qtbot, tmp_path):
    page = ExportPage()
    qtbot.addWidget(page)
    ds = DatasetState(
        id="d1",
        name="sample",
        file_path=None,
        test_mode="oscillation",
        x_data=np.array([1.0, 2.0]),
        y_data=np.array([3.0, 4.0]),
        metadata={"domain": "time"},
    )

    rheo = page._dataset_to_rheodata(ds)
    assert np.allclose(rheo.x, ds.x_data)
    assert np.allclose(rheo.y, ds.y_data)

    out = tmp_path / "data.csv"
    page._export_service.export_data(rheo, out, "csv")
    assert out.exists()


def test_bayesian_warm_start_picks_matching_fit(qtbot):
    store = StateStore()
    ds = DatasetState(id="d1", name="ds", file_path=None, test_mode="oscillation")
    store._state = AppState(
        datasets={"d1": ds}, active_dataset_id="d1", fit_results={}, current_tab="home"
    )

    # Insert two fit results with distinct keys
    class DummyFit:
        def __init__(self, params):
            self.parameters = params

    store.dispatch(
        "STORE_FIT_RESULT",
        {"model_name": "maxwell", "dataset_id": "d1", "result": DummyFit({"a": 1.0})},
    )
    store.dispatch(
        "STORE_FIT_RESULT",
        {"model_name": "zener", "dataset_id": "d2", "result": DummyFit({"b": 2.0})},
    )

    page = BayesianPage()
    qtbot.addWidget(page)
    warm = page._select_warm_start_params("maxwell", "d1")
    assert warm == {"a": 1.0}
    assert page._select_warm_start_params("zener", "d1") is None
