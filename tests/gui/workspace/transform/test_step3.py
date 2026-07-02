import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.step3_run import RunStep


def test_run_stores_result(qtbot):
    st = TransformState(
        transform_key="cox_merz",
        slots={"oscillation": "o", "flow_curve": "f"},
    )
    step = RunStep(
        st, run_fn=lambda k, s, c: {"output": "rd", "result": {"pass": True}}
    )
    qtbot.addWidget(step)
    assert step.is_ready() is False
    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()
    assert st.result["result"]["pass"] is True and step.is_ready() is True
