import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import TransformState
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.workspace.transform.step2_slots import SlotsStep


def test_param_form_seeds_config_with_defaults(qtbot):
    # Regression guard: state.config previously stayed {} forever -- no UI
    # ever populated it, so every transform silently ran with library
    # defaults and the user had no way to see or change them.
    st = TransformState(transform_key="mastercurve")
    step = SlotsStep(st, DatasetLibrary())
    qtbot.addWidget(step)

    assert st.config == {
        "reference_temp": 25.0,
        "auto_shift": True,
        "shift_method": "wlf",
    }


def test_param_form_edit_updates_config_and_emits_edited(qtbot):
    st = TransformState(transform_key="mastercurve")
    step = SlotsStep(st, DatasetLibrary())
    qtbot.addWidget(step)

    with qtbot.waitSignal(step.edited, timeout=1000):
        step._param_form._widgets["reference_temp"].setValue(40.0)

    assert st.config["reference_temp"] == 40.0


def test_no_param_form_for_transform_without_configurable_params(qtbot):
    st = TransformState(transform_key="fft_analysis")
    step = SlotsStep(st, DatasetLibrary())
    qtbot.addWidget(step)

    assert step._param_form is None
    assert st.config == {}
