from datetime import datetime

import numpy as np
import pytest

from rheojax.gui.pages.bayesian_page import BayesianPage
from rheojax.gui.pages.export_page import ExportPage
from rheojax.gui.state.actions import update_dataset
from rheojax.gui.state.store import (
    AppState,
    DatasetState,
    FitResult,
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


@pytest.mark.parametrize(
    "mutation", ["update_dataset", "set_test_mode", "auto_detect_test_mode"]
)
def test_dataset_mutation_invalidates_results_for_that_dataset(mutation):
    store = StateStore()
    ds = DatasetState(id="d1", name="ds", file_path=None, test_mode="oscillation")

    class Result:
        def __init__(self, dataset_id):
            self.dataset_id = dataset_id

        def clone(self):
            return Result(self.dataset_id)

    store._state = AppState(
        datasets={"d1": ds},
        active_dataset_id="d1",
        fit_results={"maxwell_d1": Result("d1"), "maxwell_d2": Result("d2")},
        bayesian_results={"maxwell_d1": Result("d1"), "maxwell_d2": Result("d2")},
    )

    if mutation == "update_dataset":
        update_dataset("d1", name="edited")
    elif mutation == "set_test_mode":
        store.dispatch("SET_TEST_MODE", {"dataset_id": "d1", "test_mode": "creep"})
    else:
        store.dispatch(
            "AUTO_DETECT_TEST_MODE",
            {"dataset_id": "d1", "inferred_mode": "creep"},
        )

    state = store.get_state()
    assert set(state.fit_results) == {"maxwell_d2"}
    assert set(state.bayesian_results) == {"maxwell_d2"}


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

    rheojax = page._dataset_to_rheodata(ds)
    assert np.allclose(rheojax.x, ds.x_data)
    assert np.allclose(rheojax.y, ds.y_data)

    out = tmp_path / "data.csv"
    page._export_service.export_data(rheojax, out, "csv")
    assert out.exists()


def test_export_all_fit_results_writes_real_files(qtbot, tmp_path, monkeypatch):
    """Regression: ExportPage called a nonexistent ExportService.export_fit_result,
    so every export raised AttributeError (swallowed per-result) and "Export All
    Fit Results" always reported "0 exported" with no file ever written.
    """
    from rheojax.gui.compat import QMessageBox

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    page = ExportPage()
    qtbot.addWidget(page)

    fit_result = FitResult(
        model_name="maxwell",
        parameters={"G0": 1000.0, "eta": 50.0},
        chi_squared=0.01,
        success=True,
        message="ok",
        timestamp=datetime.now(),
        dataset_id="d1",
    )
    store = page._store
    store._state = AppState(fit_results={"maxwell:d1": fit_result})

    page._output_dir_edit.setText(str(tmp_path))
    page._data_format_combo.setCurrentText("CSV")

    page._export_all_fit_results()

    written = list(tmp_path.glob("fit_*.csv"))
    assert len(written) == 1


def test_export_page_prepare_for_close_cancels_and_guards_callbacks(qtbot):
    import threading

    page = ExportPage()
    qtbot.addWidget(page)

    cancel_event = threading.Event()
    page._active_cancel_events.add(cancel_event)
    assert page._closing is False

    page.prepare_for_close()

    assert page._closing is True
    assert cancel_event.is_set()


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

    # Second dispatch changed active_dataset_id to "d2"; restore to "d1"
    # so the warm-start mismatch guard doesn't reject our lookup.
    from dataclasses import replace as _replace

    store._state = _replace(store._state, active_dataset_id="d1")

    page = BayesianPage()
    qtbot.addWidget(page)
    warm = page._select_warm_start_params("maxwell", "d1")
    assert warm == {"a": 1.0}
    assert page._select_warm_start_params("zener", "d1") is None
