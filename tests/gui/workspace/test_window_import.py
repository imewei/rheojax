import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QMessageBox

from rheojax.core.data import RheoData
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
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *a, **k: messages.append(a[-1])
    )

    state = AppState()
    win = WorkspaceWindow(state)
    qtbot.addWidget(win)

    win._on_import_failed("boom")

    assert messages == ["boom"]
