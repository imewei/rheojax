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


def test_save_to_library_stores_payload(qtbot):
    st = TransformState(
        transform_key="prony_conversion",
        slots={"input": "rel1"},
        result={"output": _RheoData([1], [2]), "protocol_type": "oscillation"},
    )
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)
    new_id = step.save_to_library()
    assert new_id is not None
    payload = lib.load_payload(new_id)  # must not raise KeyError
    assert payload.x == [1]
