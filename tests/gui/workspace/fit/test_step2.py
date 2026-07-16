import numpy as np
import pytest

pytest.importorskip("PySide6")
from rheojax.core.data import RheoData
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step2_data import DataStep, _classify_y_unit


def _ref(i, t, units=None):
    return DatasetRef(
        id=i,
        name=i,
        protocol_type=t,
        origin="imported",
        units=units if units is not None else {"x": "Hz"},
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


def test_data_step_flags_viscosity_data_for_stress_model(qtbot):
    # Regression: MIKH (flow_quantity="stress") silently fit raw viscosity
    # values as stress when a flow_curve CSV had no explicit unit conversion.
    st = FitState(protocol="flow_curve", model_key="mikh")
    lib = DatasetLibrary()
    lib.add(_ref("flow1", "flow_curve", units={"y": "Pa.s"}))
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("flow1")
    assert step.needs_quantity_conversion() is True
    assert "viscosity" in step._quantity_mismatch_message()
    assert "stress" in step._quantity_mismatch_message()


def test_data_step_no_quantity_mismatch_when_units_match(qtbot):
    st = FitState(protocol="flow_curve", model_key="mikh")
    lib = DatasetLibrary()
    lib.add(_ref("flow1", "flow_curve", units={"y": "Pa"}))
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("flow1")
    assert step.needs_quantity_conversion() is False


def test_data_step_apply_quantity_conversion_viscosity_to_stress(qtbot):
    st = FitState(protocol="flow_curve", model_key="mikh")
    lib = DatasetLibrary()
    lib.add(_ref("flow1", "flow_curve", units={"y": "Pa.s"}))
    gamma_dot = np.array([0.1, 1.0, 10.0])
    viscosity = np.array([7348.0, 831.83, 94.353])
    lib.store_payload(
        "flow1", RheoData(x=gamma_dot, y=viscosity, x_units="1/s", y_units="Pa.s")
    )
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("flow1")
    assert step.needs_quantity_conversion() is True

    step.apply_quantity_conversion()

    assert step.needs_quantity_conversion() is False
    assert step.quantity_conversion_applied() is True
    converted = lib.load_payload("flow1")
    np.testing.assert_allclose(
        np.asarray(converted.y), viscosity * gamma_dot, rtol=1e-10
    )
    assert lib.get("flow1").units["y"] == "Pa"


def test_data_step_apply_quantity_conversion_stress_to_viscosity(qtbot):
    # power_law has flow_quantity="viscosity" -- the mirror direction of the
    # MIKH (flow_quantity="stress") case above.
    st = FitState(protocol="flow_curve", model_key="power_law")
    lib = DatasetLibrary()
    lib.add(_ref("flow1", "flow_curve", units={"y": "Pa"}))
    gamma_dot = np.array([0.1, 1.0, 10.0])
    stress = np.array([734.8, 831.83, 943.53])
    lib.store_payload(
        "flow1", RheoData(x=gamma_dot, y=stress, x_units="1/s", y_units="Pa")
    )
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("flow1")
    assert step.needs_quantity_conversion() is True

    step.apply_quantity_conversion()

    assert step.needs_quantity_conversion() is False
    assert step.quantity_conversion_applied() is True
    converted = lib.load_payload("flow1")
    np.testing.assert_allclose(np.asarray(converted.y), stress / gamma_dot, rtol=1e-10)
    assert lib.get("flow1").units["y"] == "Pa.s"


def test_data_step_apply_quantity_conversion_zero_gamma_dot_surfaces_error(qtbot):
    # Regression: dividing by a gamma_dot=0 point (a legitimate flow-curve
    # value) produces inf silently -- apply_quantity_conversion() must
    # re-validate the mutated payload so this reaches validation_errors(),
    # not a fit that silently trains on inf.
    st = FitState(protocol="flow_curve", model_key="power_law")
    lib = DatasetLibrary()
    lib.add(_ref("flow1", "flow_curve", units={"y": "Pa"}))
    gamma_dot = np.array([0.0, 1.0, 10.0])
    stress = np.array([10.0, 20.0, 30.0])
    lib.store_payload(
        "flow1", RheoData(x=gamma_dot, y=stress, x_units="1/s", y_units="Pa")
    )
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("flow1")
    assert step.validation_errors() == []  # pre-conversion data is valid

    step.apply_quantity_conversion()

    converted = lib.load_payload("flow1")
    assert not np.isfinite(np.asarray(converted.y)).all()
    assert step.validation_errors() != []
    assert step.is_ready() is False


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("Pa", "stress"),
        ("kPa", "stress"),
        ("MPa", "stress"),
        ("GPa", "stress"),
        ("Pa.s", "viscosity"),
        ("Pa·s", "viscosity"),
        ("mPa*s", "viscosity"),
        ("cP", "viscosity"),
        ("poise", "viscosity"),
        ("", None),
        ("rad/s", None),
        ("G'", None),
    ],
)
def test_classify_y_unit(unit, expected):
    assert _classify_y_unit(unit) == expected
