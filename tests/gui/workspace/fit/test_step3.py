import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep


def test_nlsq_runs_and_stores_result(qtbot):
    st = FitState(protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0})
    calls = {}
    def fake_fit(model_key, model_config, data_ref, column_map):
        calls["args"] = (model_key, model_config, data_ref)
        return {"params": {"G0": 1000.0}, "r_squared": 0.99,
                "reduced_chi_squared": 0.8, "uncertainties": [10.0]}
    step = NlsqStep(st, fit_fn=fake_fit); qtbot.addWidget(step)
    assert step.is_ready() is False
    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()
    assert st.nlsq_result["r_squared"] == 0.99
    assert calls["args"][0] == "maxwell"
    assert step.is_ready() is True
