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
