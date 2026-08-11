import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.step2_slots import SlotsStep


def test_param_form_seeds_config_with_defaults(qtbot):
    # Regression guard: state.config previously stayed {} forever -- no UI
    # ever populated it, so every transform silently ran with library
    # defaults and the user had no way to see or change them.
    st = TransformState(transform_key="mastercurve")
    step = SlotsStep(st, DatasetLibrary())
    qtbot.addWidget(step)

    assert st.config == {
        "reference_temp": 298.15,
        "auto_shift": False,
        "shift_method": "wlf",
    }


def test_param_form_edit_updates_config_and_emits_edited(qtbot):
    st = TransformState(transform_key="mastercurve")
    step = SlotsStep(st, DatasetLibrary())
    qtbot.addWidget(step)

    with qtbot.waitSignal(step.edited, timeout=1000):
        step._param_form._widgets["reference_temp"].setValue(310.0)

    assert st.config["reference_temp"] == 310.0


def test_no_param_form_for_transform_without_configurable_params(qtbot):
    # Uses an unregistered key rather than a real one: every registered
    # transform declares configurable params. This previously passed with
    # "fft_analysis" only because get_transform_params() skipped the
    # registry-key -> legacy-key normalization and returned {} for it, which
    # also meant FFT/SPP/derivative rendered no parameter UI at all. The
    # behaviour under test is the empty-params guard, not that any particular
    # transform lacks params.
    st = TransformState(transform_key="not_a_registered_transform")
    step = SlotsStep(st, DatasetLibrary())
    qtbot.addWidget(step)

    assert step._param_form is None
    assert st.config == {}


def test_param_form_renders_for_registry_key_not_just_legacy_alias(qtbot):
    # Regression guard for the normalization bug above: the picker writes the
    # registry key ("fft_analysis"), while the param spec table is keyed on the
    # legacy dispatch name ("fft"). Both must resolve to the same spec.
    st = TransformState(transform_key="fft_analysis")
    step = SlotsStep(st, DatasetLibrary())
    qtbot.addWidget(step)

    assert step._param_form is not None
    assert st.config["direction"] == "forward"


def test_edited_param_value_survives_a_refresh_triggered_by_list_slot_edit(qtbot):
    # Regression: refresh() (called by _add_list_slot_widgets' add/remove
    # closures whenever mastercurve/srfs's list slot changes) used to
    # rebuild ParameterFormBuilder from the spec defaults unconditionally,
    # silently resetting the visible widget value even though state.config
    # (via setdefault) still held the edited one -- UI and state disagreed.
    st = TransformState(transform_key="mastercurve")
    step = SlotsStep(st, DatasetLibrary())
    qtbot.addWidget(step)

    step._param_form._widgets["reference_temp"].setValue(310.0)
    assert st.config["reference_temp"] == 310.0

    step.refresh()

    assert st.config["reference_temp"] == 310.0
    assert step._param_form._widgets["reference_temp"].value() == 310.0
