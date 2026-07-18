from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.compat import QFileDialog
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import ActiveJobsState, TransformState
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

    qtbot.waitUntil(lambda: str(tmp_path) in step._export_status.text())
    assert (tmp_path / "provenance.json").exists()


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
    qtbot.waitUntil(lambda: "failed" in step._export_status.text().lower())


def test_export_holds_worker_ref_and_registers_active_job(qtbot, monkeypatch, tmp_path):
    """Regression: _active_export_workers/active_jobs registration must
    happen synchronously inside _on_export_clicked() and clear exactly once
    the export finishes -- see the mirrored test in test_step6.py for why
    this line looks like a dead store without this assertion."""
    st = TransformState(
        transform_key="derivative",
        slots={"input": "d1"},
        result={"result": {"n": 2}, "protocol_type": "flow_curve"},
    )
    lib = DatasetLibrary()
    active_jobs = ActiveJobsState()
    step = TransformExportStep(st, lib, active_jobs)
    qtbot.addWidget(step)

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )

    step._on_export_clicked()
    assert len(step._active_export_workers) == 1
    assert len(active_jobs.by_id) == 1

    qtbot.waitUntil(lambda: str(tmp_path) in step._export_status.text())
    assert step._active_export_workers == {}
    assert active_jobs.by_id == {}
