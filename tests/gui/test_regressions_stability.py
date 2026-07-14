import pytest

from rheojax.gui.state.actions import update_dataset
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
