from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.step2_slots import SlotsStep


def _ref(i, protocol):
    return DatasetRef(id=i, name=i, protocol_type=protocol, origin="imported",
                       units={}, row_count=1, hash="h", provenance={}, lineage=[])


def test_single_slot_combo_box_selection_fills_state(qtbot):
    st = TransformState(transform_key="fft_analysis")
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    step = SlotsStep(st, lib)
    qtbot.addWidget(step)
    combo = step._combo_widgets["input"]
    assert "d1" in [combo.itemText(i) for i in range(combo.count())]
    combo.setCurrentText("d1")
    assert st.slots["input"] == "d1"


def test_typed_pair_slot_uses_two_combo_boxes(qtbot):
    st = TransformState(transform_key="cox_merz")
    lib = DatasetLibrary()
    lib.add(_ref("o1", "oscillation"))
    lib.add(_ref("f1", "flow_curve"))
    step = SlotsStep(st, lib)
    qtbot.addWidget(step)
    step._combo_widgets["oscillation"].setCurrentText("o1")
    step._combo_widgets["flow_curve"].setCurrentText("f1")
    assert st.slots == {"oscillation": "o1", "flow_curve": "f1"}


def test_list_slot_add_and_remove(qtbot):
    st = TransformState(transform_key="mastercurve")
    lib = DatasetLibrary()
    lib.add(_ref("s1", "oscillation"))
    lib.add(_ref("s2", "oscillation"))
    step = SlotsStep(st, lib)
    qtbot.addWidget(step)

    step._list_add_combos["datasets"].setCurrentText("s1")
    step._list_add_buttons["datasets"].click()
    assert st.slots["datasets"] == ["s1"]
    assert step._list_widgets["datasets"].count() == 1

    step._list_add_combos["datasets"].setCurrentText("s2")
    step._list_add_buttons["datasets"].click()
    assert st.slots["datasets"] == ["s1", "s2"]

    step._list_widgets["datasets"].setCurrentRow(0)
    step._list_remove_buttons["datasets"].click()
    assert st.slots["datasets"] == ["s2"]


def test_refresh_drops_selection_no_longer_in_candidates(qtbot):
    st = TransformState(transform_key="fft_analysis", slots={"input": "gone"})
    lib = DatasetLibrary()  # "gone" was never added -> not a valid candidate
    step = SlotsStep(st, lib)
    qtbot.addWidget(step)
    assert "input" not in st.slots
