import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from rheojax.core.data import RheoData
from rheojax.gui.dialogs.column_mapper import ColumnMapperDialog
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def test_library_rail_import_click_invokes_file_dialog(qtbot, monkeypatch):
    # Regression: LibraryRail's "+ Import data..." button emitted
    # import_requested, but WorkspaceWindow never connected it to anything,
    # so clicking it silently did nothing.
    calls = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *a, **k: (calls.append(1), ([], ""))[1],
    )

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    win._rail._import.click()

    assert calls == [1]


def test_import_completed_adds_dataset_to_library(qtbot, tmp_path):
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)
    assert win._rail.count() == 0

    source = tmp_path / "sample.csv"
    source.write_text("omega,g_prime\n1.0,2.0\n2.0,3.0\n")
    data = RheoData(
        x=np.array([1.0, 2.0]),
        y=np.array([2.0, 3.0]),
        domain="frequency",
        metadata={"_source_file": str(source)},
    )

    win._on_import_completed([data])

    assert win._rail.count() == 1
    ref = next(iter(state.library.all()))
    assert ref.name == "sample"
    assert ref.protocol_type == "oscillation"
    assert ref.row_count == 2
    assert state.library.load_payload(ref.id) is data
    assert "_source_file" not in data.metadata


def test_import_completed_maps_flow_test_mode_to_flow_curve_protocol(qtbot, tmp_path):
    # DataService.detect_test_mode() returns "flow", but the Protocol enum
    # (and DatasetLibrary.datasets_of_type filtering) uses "flow_curve" --
    # without normalizing, imported flow-curve data would never match any
    # protocol's Data step.
    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    source = tmp_path / "flow.csv"
    source.write_text("shear_rate,viscosity\n1.0,10.0\n")
    data = RheoData(
        x=np.array([1.0, 10.0, 100.0]),
        y=np.array([100.0, 10.0, 1.0]),
        domain="time",
        metadata={"_source_file": str(source), "test_mode": "flow"},
    )

    win._on_import_completed([data])

    ref = next(iter(state.library.all()))
    assert ref.protocol_type == "flow_curve"


def test_import_failed_shows_message_box(qtbot, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: messages.append(a[-1]))

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    win._on_import_failed("boom")

    assert messages == ["boom"]


def test_import_failed_with_undetected_columns_offers_column_mapper(
    qtbot, tmp_path, monkeypatch
):
    # Regression: a single-file import whose headers don't match auto_load's
    # heuristic column-name list used to dead-end with a raw error dialog
    # and no way to specify columns. It should offer ColumnMapperDialog and
    # retry with the user's mapping instead.
    source = tmp_path / "weird.csv"
    source.write_text("foo,bar\n1.0,2.0\n2.0,3.0\n")

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    monkeypatch.setattr(
        ColumnMapperDialog, "exec", lambda self: QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(
        ColumnMapperDialog, "get_mapping", lambda self: {"x": "foo", "y": "bar"}
    )
    relaunches = []
    monkeypatch.setattr(
        win, "_launch_import", lambda *a, **k: relaunches.append((a, k))
    )

    win._on_import_failed(
        "Could not parse CSV as TRIOS or generic CSV: Could not auto-detect "
        "columns: Could not auto-detect x and y columns. "
        "Please specify x_col and y_col.. Please specify x_col and y_col.",
        [source],
    )

    assert len(relaunches) == 1
    args, kwargs = relaunches[0]
    assert args == ([source],)
    assert kwargs == {
        "x_col": "foo",
        "y_col": "bar",
        "y2_col": None,
        "temp_col": None,
    }


def test_import_failed_undetected_columns_cancelled_shows_message_box(
    qtbot, tmp_path, monkeypatch
):
    source = tmp_path / "weird.csv"
    source.write_text("foo,bar\n1.0,2.0\n")

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    monkeypatch.setattr(
        ColumnMapperDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: messages.append(a[-1]))

    win._on_import_failed("Could not auto-detect x and y columns.", [source])

    assert messages == ["Could not auto-detect x and y columns."]


def test_import_failed_non_mappable_extension_skips_column_mapper(
    qtbot, tmp_path, monkeypatch
):
    # Regression: auto_load's unknown-extension fallback (_try_all_readers)
    # aggregates every reader's error into one message, so a corrupt/
    # unsupported .tri file can still produce a message containing
    # "auto-detect" from the CSV reader's fallback attempt even though the
    # real failure has nothing to do with CSV column mapping.
    # ColumnMapperDialog can't parse this file either (no branch for .tri in
    # its _load_data) -- don't offer it for extensions it doesn't understand.
    source = tmp_path / "weird.tri"
    source.write_bytes(b"not real trios data")

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    opened = []
    monkeypatch.setattr(
        ColumnMapperDialog, "__init__", lambda self, *a, **k: opened.append(1)
    )
    monkeypatch.setattr(
        ColumnMapperDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: messages.append(a[-1]))

    win._on_import_failed(
        "Could not parse file with any available reader:\n"
        "csv: Could not auto-detect columns: ...",
        [source],
    )

    assert opened == []
    assert messages == [
        "Could not parse file with any available reader:\n"
        "csv: Could not auto-detect columns: ..."
    ]


def test_overlapping_imports_do_not_cross_wire_failure_paths(
    qtbot, tmp_path, monkeypatch
):
    # Regression: _on_import_failed used to read a single shared
    # `_pending_import_paths` instance attribute set by _launch_import. If a
    # second import was launched before the first worker's queued "failed"
    # signal was delivered, the second call's paths silently overwrote the
    # first's, so the failure handler retried against the wrong file.
    # file_paths must now be bound per-connection so each worker's failure
    # always carries its own paths regardless of what else launched after it.
    monkeypatch.setattr(QThreadPool.globalInstance(), "start", lambda worker: None)

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    source_a = tmp_path / "a.csv"
    source_a.write_text("foo,bar\n1.0,2.0\n")
    source_b = tmp_path / "b.csv"
    source_b.write_text("foo,bar\n1.0,2.0\n")

    win._launch_import([source_a])
    worker_a = win._active_import_worker
    win._launch_import([source_b])

    calls = []
    monkeypatch.setattr(win, "_on_import_failed", lambda *a, **k: calls.append(a))

    worker_a.signals.failed.emit("boom")
    qtbot.wait(50)

    assert calls == [("boom", [source_a])]


def test_import_failed_unparseable_file_skips_broken_column_mapper(
    qtbot, tmp_path, monkeypatch
):
    # Regression: some files that fail auto_load's column-detection (and so
    # match the "auto-detect" retry gate) are also genuinely unparseable by
    # pandas itself -- e.g. a preamble with inconsistent field counts per
    # row, like a RepTate Prony-mode export. ColumnMapperDialog re-reads the
    # file independently and hits the same parse error; its own except
    # branch only shows a warning and leaves the dialog open with empty
    # column lists (self.columns stays []). Opening that broken dialog on
    # top of the warning was confusing -- it should be skipped so the user
    # sees just the one original error.
    source = tmp_path / "malformed.csv"
    source.write_text("line1\nline2\na,b,c\n1,2,3\n4,5,6\n")

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    exec_calls = []
    monkeypatch.setattr(
        ColumnMapperDialog,
        "exec",
        lambda self: exec_calls.append(1) or QDialog.DialogCode.Accepted,
    )
    warnings_shown = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: warnings_shown.append(a[-1])
    )
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: messages.append(a[-1]))

    win._on_import_failed(
        "Could not parse CSV as TRIOS or generic CSV: Could not auto-detect "
        "columns: Error tokenizing data. Please specify x_col and y_col.",
        [source],
    )

    assert exec_calls == []
    assert len(warnings_shown) == 1  # ColumnMapperDialog's own load-failure warning
    assert len(messages) == 1
