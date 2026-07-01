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
