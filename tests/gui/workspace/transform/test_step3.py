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


def test_run_without_wired_run_fn_does_not_crash(qtbot):
    """Clicking Run before the real transform worker is wired must not raise
    NotImplementedError from inside the Qt slot (which can abort the process
    under PySide6 6.x)."""
    st = TransformState(
        transform_key="cox_merz",
        slots={"oscillation": "o", "flow_curve": "f"},
    )
    step = RunStep(st)  # no run_fn injected -> falls back to _default_run_fn stub
    qtbot.addWidget(step)
    step.run()  # must not raise
    assert st.result is None
    assert step.is_ready() is False
    assert "not wired up yet" in step._status.text()
