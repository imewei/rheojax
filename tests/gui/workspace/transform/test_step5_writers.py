from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

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
