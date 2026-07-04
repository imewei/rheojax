import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.step2_slots import SlotsStep


def _ref(i, t):
    return DatasetRef(
        id=i,
        name=i,
        protocol_type=t,
        origin="imported",
        units={},
        row_count=1,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_typed_slot_filtering_and_ready(qtbot):
    st = TransformState(transform_key="cox_merz")
    lib = DatasetLibrary()
    lib.add(_ref("o", "oscillation"))
    lib.add(_ref("f", "flow_curve"))
    step = SlotsStep(st, lib)
    qtbot.addWidget(step)
    assert step.candidates("oscillation") == ["o"]
    assert step.candidates("flow_curve") == ["f"]
    step.fill("oscillation", "o")
    assert step.is_ready() is False
    step.fill("flow_curve", "f")
    assert step.is_ready() is True and st.slots == {
        "oscillation": "o",
        "flow_curve": "f",
    }
