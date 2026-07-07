import pytest

pytest.importorskip("PySide6")
import rheojax.models  # noqa
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step1_protocol_model import ProtocolModelStep


def test_model_list_filters_by_protocol(qtbot):
    st = FitState()
    step = ProtocolModelStep(st)
    qtbot.addWidget(step)
    assert step.is_ready() is False
    step.set_protocol("oscillation")
    keys = step.model_keys()
    assert (
        "generalized_maxwell" in keys and "power_law" not in keys
    )  # power_law is flow-only
    with qtbot.waitSignal(step.edited, timeout=1000):
        step.set_model("generalized_maxwell")
    assert st.protocol == "oscillation" and st.model_key == "generalized_maxwell"
    assert step.is_ready() is True


def test_model_combo_is_grouped_by_family_not_a_flat_list(qtbot):
    # Regression: the model combo used to be a flat sorted(ModelRegistry.find())
    # list -- reuse fit_page.py's family-grouping so users can scan by family.
    st = FitState()
    step = ProtocolModelStep(st)
    qtbot.addWidget(step)
    step.set_protocol("oscillation")

    combo = step._model
    header_texts = [
        combo.itemText(i)
        for i in range(combo.count())
        if combo.itemData(i) is None and combo.itemText(i)
    ]
    assert any(text.startswith("── ") for text in header_texts)
    # Header rows are disabled (non-selectable) separators, not real models.
    for i in range(combo.count()):
        if combo.itemData(i) is None and combo.itemText(i):
            assert combo.model().item(i).isEnabled() is False
    # model_keys() must still return only real model names, never headers.
    assert all(not k.startswith("──") for k in step.model_keys())
