"""Regression: SET_ACTIVE_DATASET / DELETE_SELECTED_DATASET never emitted
dataset_selected, so pages subscribed only to that signal (FitPage has no
state_changed fallback) went stale on every dataset switch made via the tree.
"""

from rheojax.gui.state.store import AppState, DatasetState, StateStore


def setup_function() -> None:
    StateStore.reset()


def test_set_active_dataset_emits_dataset_selected() -> None:
    store = StateStore()
    ds1 = DatasetState(id="d1", name="ds1", file_path=None, test_mode="oscillation")
    ds2 = DatasetState(id="d2", name="ds2", file_path=None, test_mode="oscillation")
    store._state = AppState(datasets={"d1": ds1, "d2": ds2}, active_dataset_id="d1")

    selected: list[str] = []
    store.signals.dataset_selected.connect(selected.append)

    store.dispatch("SET_ACTIVE_DATASET", {"dataset_id": "d2"})

    assert selected == ["d2"]
    assert store.get_state().active_dataset_id == "d2"


def test_delete_selected_dataset_emits_dataset_selected_for_new_active() -> None:
    store = StateStore()
    ds1 = DatasetState(id="d1", name="ds1", file_path=None, test_mode="oscillation")
    ds2 = DatasetState(id="d2", name="ds2", file_path=None, test_mode="oscillation")
    store._state = AppState(datasets={"d1": ds1, "d2": ds2}, active_dataset_id="d1")

    removed: list[str] = []
    selected: list[str] = []
    store.signals.dataset_removed.connect(removed.append)
    store.signals.dataset_selected.connect(selected.append)

    store.dispatch("DELETE_SELECTED_DATASET")

    assert removed == ["d1"]
    assert selected == ["d2"]
    assert store.get_state().active_dataset_id == "d2"


def test_delete_last_dataset_does_not_emit_dataset_selected() -> None:
    """No datasets remain -> no dataset to select, so no emission."""
    store = StateStore()
    ds1 = DatasetState(id="d1", name="ds1", file_path=None, test_mode="oscillation")
    store._state = AppState(datasets={"d1": ds1}, active_dataset_id="d1")

    selected: list[str] = []
    store.signals.dataset_selected.connect(selected.append)

    store.dispatch("DELETE_SELECTED_DATASET")

    assert selected == []
    assert store.get_state().active_dataset_id is None
