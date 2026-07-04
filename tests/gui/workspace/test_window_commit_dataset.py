import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def _ref(id_):
    return DatasetRef(id=id_, name=id_, protocol_type="oscillation", origin="derived",
                      units={}, row_count=1, hash="h", provenance={}, lineage=[])


def test_commit_dataset_adds_ref_and_marks_dirty(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    assert state.project.dirty is False

    win._commit_dataset(_ref("d1"), payload=None, overwrite=False)

    assert state.library.get("d1").id == "d1"
    assert state.project.dirty is True


def test_commit_dataset_emits_notifier_changed_once(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    with qtbot.waitSignal(win._notifier.changed, timeout=1000, raising=True):
        win._commit_dataset(_ref("d1"), payload=None, overwrite=False)


def test_commit_dataset_overwrite_false_raises_on_collision(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    win._commit_dataset(_ref("d1"), payload=None, overwrite=False)
    with pytest.raises(ValueError):
        win._commit_dataset(_ref("d1"), payload=None, overwrite=False)


def test_fit_body_dataset_commit_requested_routes_to_commit_dataset(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    fit_export_body = next(b for b in win._fit_bodies if hasattr(b, "dataset_commit_requested"))
    fit_export_body.dataset_commit_requested.emit(_ref("d2"), None, True)
    assert state.library.get("d2").id == "d2"


def test_commit_dataset_refreshes_library_rail(qtbot):
    # LibraryRail.refresh() is otherwise never called after construction -- without the
    # notifier -> rail.refresh wiring, a newly committed dataset would render nowhere
    # until the next full _rebuild() (Open/New/Close).
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    assert win._rail.count() == 0

    win._commit_dataset(_ref("d1"), payload=None, overwrite=False)

    assert win._rail.count() == 1


@pytest.mark.parametrize("edited_signal", [
    "_on_fit_body_edited", "_on_transform_body_edited", "_on_pipeline_body_edited",
])
def test_body_edited_marks_project_dirty(qtbot, edited_signal):
    # Fit/Transform/Pipeline step config edits (nlsq_config, priors, pipeline steps) are
    # all persisted into the project archive -- silently NOT dirtying on edit means a
    # user can Close/New/Open without a save prompt and lose those changes.
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    assert state.project.dirty is False

    getattr(win, edited_signal)()

    assert state.project.dirty is True
