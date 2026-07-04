import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step2_data import DataStep


def _ref(i, t):
    return DatasetRef(
        id=i,
        name=i,
        protocol_type=t,
        origin="imported",
        units={"x": "Hz"},
        row_count=64,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_data_step_contract_and_filter(qtbot):
    st = FitState(protocol="oscillation", model_key="generalized_maxwell")
    lib = DatasetLibrary()
    lib.add(_ref("osc1", "oscillation"))
    lib.add(_ref("c1", "creep"))
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    assert step.expected_columns() == ["omega", "G_prime", "G_double_prime"]
    assert step.available_datasets() == ["osc1"]  # only oscillation
    step.select_dataset("osc1")
    assert st.data_ref == "osc1"
    assert step.needs_hz_conversion() is True  # units x=Hz, contract wants rad/s


def test_data_step_none_contract_does_not_raise(qtbot):
    st = FitState()  # protocol never set -> self._contract is None
    lib = DatasetLibrary()
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    assert step.needs_hz_conversion() is False
    step._on_select("some_id")  # must not raise AttributeError on None contract


def test_data_step_refresh_rebuilds_after_protocol_set(qtbot):
    # Regression: DataStep built before Step 1 sets protocol/model_key must
    # rebuild its contract/combo once refresh() is called (the real app wires
    # this to ProtocolModelStep.edited) -- it must not stay stuck forever.
    st = FitState()  # fresh state, protocol=None
    lib = DatasetLibrary()
    lib.add(_ref("osc1", "oscillation"))
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    assert step.expected_columns() == []
    assert step.available_datasets() == []

    # Step 1 completes "elsewhere" -> state mutated directly, then refresh() called
    st.protocol = "oscillation"
    st.model_key = "generalized_maxwell"
    step.refresh()

    assert step.expected_columns() == ["omega", "G_prime", "G_double_prime"]
    assert step.available_datasets() == ["osc1"]
    assert [step._source.itemText(i) for i in range(step._source.count())] == [
        "",
        "osc1",
    ]


def test_data_step_refresh_clears_stale_selection(qtbot):
    # If the previously selected dataset is no longer valid for the new
    # protocol, refresh() must clear data_ref/column_map and notify via edited.
    st = FitState(protocol="oscillation", model_key="generalized_maxwell")
    lib = DatasetLibrary()
    lib.add(_ref("osc1", "oscillation"))
    lib.add(_ref("c1", "creep"))
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("osc1")
    assert st.data_ref == "osc1"
    assert st.column_map

    st.protocol = "creep"  # osc1 is no longer a valid dataset for this protocol
    with qtbot.waitSignal(step.edited, timeout=1000):
        step.refresh()

    assert st.data_ref is None
    assert st.column_map == {}
    assert step.available_datasets() == ["c1"]
