from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from rheojax.core.data import RheoData
from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def _ref(id_: str = "d1", row_count: int = 0) -> DatasetRef:
    return DatasetRef(
        id=id_,
        name=id_,
        protocol_type="oscillation",
        origin="derived",
        units={},
        row_count=row_count,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_preview_unknown_dataset_id_leaves_dialog_none(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    win._on_dataset_preview_requested("does-not-exist")

    assert win._preview_dialog is None


def test_preview_unknown_dataset_id_leaves_existing_dialog_untouched(qtbot):
    # Distinct from the test above: here a dialog is already open showing a
    # real dataset, and a stale/unknown id must not clear or replace it.
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", RheoData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), validate=False)
    )
    win._on_dataset_preview_requested("d1")
    existing_dialog = win._preview_dialog
    assert existing_dialog is not None

    win._on_dataset_preview_requested("does-not-exist")

    assert win._preview_dialog is existing_dialog
    assert win._preview_dialog._model.rowCount() == 2


def test_preview_unknown_dataset_id_logs_warning(qtbot, monkeypatch):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    logged = []
    monkeypatch.setattr(
        win.log_dock, "append_record", lambda level, msg: logged.append((level, msg))
    )

    win._on_dataset_preview_requested("does-not-exist")

    assert len(logged) == 1
    level, message = logged[0]
    assert level == logging.WARNING
    assert "does-not-exist" in message


def test_preview_known_dataset_opens_dialog_with_table(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", RheoData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), validate=False)
    )

    win._on_dataset_preview_requested("d1")

    assert win._preview_dialog is not None
    assert win._preview_dialog.isVisible()
    assert win._preview_dialog._model.rowCount() == 2


def test_rail_signal_emission_reaches_handler_through_real_wiring(qtbot):
    # Every other test in this file calls _on_dataset_preview_requested()
    # directly. A missing or broken
    # `_rail.dataset_preview_requested.connect(...)` in _build_workspace
    # would leave all of them passing while the actual user-facing feature
    # (double-click/right-click in LibraryRail) silently did nothing -- this
    # test goes through the real signal instead of calling the handler.
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", RheoData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), validate=False)
    )

    win._rail.dataset_preview_requested.emit("d1")

    assert win._preview_dialog is not None
    assert win._preview_dialog._model.rowCount() == 2


def test_preview_ref_with_no_payload_shows_no_data_state(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))  # never store_payload()'d

    win._on_dataset_preview_requested("d1")

    assert win._preview_dialog._no_data_label.isVisible() is True


def test_preview_simplenamespace_payload_normalizes_without_crashing(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", SimpleNamespace(x=np.array([1.0, 2.0]), y=np.array([5.0, 6.0]))
    )

    win._on_dataset_preview_requested("d1")

    assert win._preview_dialog._model.rowCount() == 2
    assert win._preview_dialog._domain_label.text() == "time"


def test_preview_mismatched_length_payload_shows_no_data_state(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", SimpleNamespace(x=np.array([1.0, 2.0, 3.0]), y=np.array([5.0]))
    )

    win._on_dataset_preview_requested("d1")

    assert win._preview_dialog._no_data_label.isVisible() is True


def test_preview_2d_y_payload_shows_no_data_state(qtbot):
    # Round-2/3 review: RheoData's own shape validation tolerates a real-valued
    # (N,2) `y` (e.g. a hand-crafted/restored payload), but no interactive
    # import/transform/fit path in this codebase produces it -- this design
    # treats it as unsupported rather than claiming it can't occur.
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1",
        RheoData(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            validate=False,
        ),
    )

    win._on_dataset_preview_requested("d1")

    assert win._preview_dialog._no_data_label.isVisible() is True


def test_preview_nan_dataset_surfaces_real_validation_warning(qtbot):
    # End-to-end with the REAL DataService.validate_data() (not mocked), to
    # prove the full pipeline surfaces an actual validation warning, not just
    # that the dialog can render whatever warnings list it's handed.
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1",
        RheoData(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, np.nan, 3.0]),
            validate=False,
        ),
    )

    win._on_dataset_preview_requested("d1")

    warnings_shown = [
        win._preview_dialog._warnings_layout.itemAt(i).widget().text()
        for i in range(win._preview_dialog._warnings_layout.count())
    ]
    assert "Y-axis contains NaN or Inf values" in warnings_shown


def test_preview_object_dtype_payload_does_not_crash_rendering(qtbot):
    # Non-numeric values can pass the shape/length/ndim checks (they're 1-D,
    # matching length) but must still render without crashing -- exercises
    # the real validate_data() AND the real table model, no mocking.
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1",
        RheoData(
            x=np.array([1.0, 2.0]),
            y=np.array(["not", "numeric"], dtype=object),
            validate=False,
        ),
    )

    win._on_dataset_preview_requested("d1")  # must not raise

    dialog = win._preview_dialog
    assert dialog._model.data(dialog._model.index(0, 1)) == "not"
    warnings_shown = [
        dialog._warnings_layout.itemAt(i).widget().text()
        for i in range(dialog._warnings_layout.count())
    ]
    assert len(warnings_shown) == 1
    assert warnings_shown[0].startswith("Validation check failed:")


def test_preview_zero_row_dataset_skips_validate_data(qtbot, monkeypatch):
    from rheojax.gui.services.data_service import DataService

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", RheoData(x=np.array([]), y=np.array([]), validate=False)
    )
    calls = {"n": 0}

    def fake_validate(self, data):
        calls["n"] += 1
        return []

    monkeypatch.setattr(DataService, "validate_data", fake_validate)

    win._on_dataset_preview_requested("d1")

    assert calls["n"] == 0
    assert win._preview_dialog._warnings_layout.count() == 1
    assert (
        win._preview_dialog._warnings_layout.itemAt(0).widget().text()
        == "Dataset contains no rows"
    )


def test_preview_validate_data_exception_becomes_warning_not_crash(qtbot, monkeypatch):
    from rheojax.gui.services.data_service import DataService

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", RheoData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), validate=False)
    )

    def broken_validate(self, data):
        raise TypeError("boom")

    monkeypatch.setattr(DataService, "validate_data", broken_validate)

    win._on_dataset_preview_requested("d1")  # must not raise

    assert win._preview_dialog._warnings_layout.count() == 1


def test_preview_reuses_same_dialog_instance_after_closing(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", RheoData(x=np.array([1.0]), y=np.array([2.0]), validate=False)
    )

    win._on_dataset_preview_requested("d1")
    first_dialog = win._preview_dialog
    first_dialog.close()
    assert first_dialog.isVisible() is False

    win._on_dataset_preview_requested("d1")

    assert win._preview_dialog is first_dialog
    assert win._preview_dialog.isVisible() is True


def test_preview_holds_library_lock_across_get_and_load_payload(qtbot):
    # A shallow "was library.lock entered at all" check would still pass if
    # get()/load_payload() were accidentally moved outside the `with` block
    # -- DatasetLibrary.get()/load_payload() acquire the SAME RLock via the
    # private `_lock` attribute (`.lock` is just a public alias to it).
    # Recording depth inside __enter__ itself (rather than in a wrapper
    # called *before* the wrapped method reacquires the lock) captures every
    # acquisition, including the reentrant ones -- proving the outer
    # `with library.lock:` in the handler is still held when get()/
    # load_payload() acquire it internally, not merely that it was entered
    # at some point.
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", RheoData(x=np.array([1.0]), y=np.array([2.0]), validate=False)
    )
    depths_seen = []

    class ReentrantSpyLock:
        def __init__(self, real_lock):
            self._real_lock = real_lock
            self.depth = 0

        def __enter__(self):
            self._real_lock.__enter__()
            self.depth += 1
            depths_seen.append(self.depth)
            return self

        def __exit__(self, *exc):
            self.depth -= 1
            return self._real_lock.__exit__(*exc)

    spy = ReentrantSpyLock(state.library.lock)
    state.library.lock = spy
    state.library._lock = spy

    win._on_dataset_preview_requested("d1")

    # [1, 2, 2]: the handler's outer scope acquires first (depth 1), then
    # get() and load_payload() each reacquire the SAME lock from inside it
    # (depth 2 each), one after the other.
    assert depths_seen == [1, 2, 2]


def test_rebuild_closes_and_clears_open_preview_dialog(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    state.library.add(_ref("d1"))
    state.library.store_payload(
        "d1", RheoData(x=np.array([1.0]), y=np.array([2.0]), validate=False)
    )
    win._on_dataset_preview_requested("d1")
    old_dialog = win._preview_dialog
    assert old_dialog is not None

    win._rebuild(AppState())

    # Not just "the attribute was cleared" -- the old dialog itself must
    # actually be closed, not merely dereferenced while still on screen.
    assert win._preview_dialog is None
    assert old_dialog.isVisible() is False


def test_rebuild_with_no_preview_ever_opened_does_not_raise(qtbot):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    assert win._preview_dialog is None

    win._rebuild(AppState())  # must not raise AttributeError

    assert win._preview_dialog is None
