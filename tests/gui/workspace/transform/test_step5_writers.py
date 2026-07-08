from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.compat import QFileDialog
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.step5_export import TransformExportStep


class _RheoData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_export_bundle_writes_output_result_and_provenance(qtbot, tmp_path):
    st = TransformState(
        transform_key="derivative",
        slots={"input": "d1"},
        result={
            "output": _RheoData([1, 2], [3, 4]),
            "result": {"n": 2},
            "protocol_type": "flow_curve",
        },
    )
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)

    written = step.export_bundle(tmp_path)

    assert written["result"].exists()
    assert written["provenance"].exists()
    # ExportService.export_data's CSV path only touches data.x / data.y (no
    # isinstance(data, RheoData) check exists in export_service.py), so the
    # duck-typed _RheoData fake above works fine here.
    assert written["output"].exists()


def test_export_bundle_skips_output_when_no_result(qtbot, tmp_path):
    st = TransformState(transform_key="derivative", slots={"input": "d1"}, result=None)
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)
    written = step.export_bundle(tmp_path)
    assert "output" not in written
    assert "result" not in written
    assert written["provenance"].exists()


def test_export_button_prompts_for_directory_instead_of_using_cwd(
    qtbot, monkeypatch, tmp_path
):
    """Regression: the Export button used to be wired directly to
    export_bundle(Path.cwd()), silently writing files into the process's
    working directory with no dialog and no feedback."""
    st = TransformState(
        transform_key="derivative",
        slots={"input": "d1"},
        result={"result": {"n": 2}, "protocol_type": "flow_curve"},
    )
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )

    step._on_export_clicked()

    assert (tmp_path / "provenance.json").exists()
    assert str(tmp_path) in step._export_status.text()


def test_export_button_reports_failure_instead_of_raising(qtbot, monkeypatch):
    """Export errors (e.g. an unwritable directory) must surface in the
    status label, not propagate uncaught out of the Qt slot."""
    st = TransformState(transform_key="derivative", slots={"input": "d1"}, result=None)
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: "/nonexistent/path"
    )

    def _raise(_directory):
        raise OSError("disk full")

    monkeypatch.setattr(step, "export_bundle", _raise)

    step._on_export_clicked()  # must not raise
    assert "failed" in step._export_status.text().lower()
