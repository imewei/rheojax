import types

import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep


def test_nlsq_runs_and_stores_result(qtbot):
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    calls = {}

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        calls["args"] = (model_key, model_config, data_ref)
        return {
            "params": {"G0": 1000.0},
            "r_squared": 0.99,
            "reduced_chi_squared": 0.8,
            "uncertainties": [10.0],
        }

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    assert step.is_ready() is False
    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()
    assert st.nlsq_result["r_squared"] == 0.99
    assert calls["args"][0] == "maxwell"
    assert step.is_ready() is True


def test_nlsq_reads_params_from_non_dict_result(qtbot):
    """Non-dict fit_fn results (e.g. FitResult) must be read via `.params`, not `.parameters`."""
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    fake_result = types.SimpleNamespace(
        params={"G0": 1000.0}, r_squared=0.9, success=True
    )

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        return fake_result

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()
    assert st.nlsq_result["params"] == {"G0": 1000.0}
    assert st.nlsq_result["r_squared"] == 0.9


def test_nlsq_handles_none_r_squared(qtbot):
    """A present-but-None r_squared must not crash the result label formatting."""
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        return {"params": {"G0": 1000.0}, "r_squared": None}

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()
    assert st.nlsq_result["r_squared"] is None


def test_nlsq_run_without_wired_fit_fn_does_not_crash(qtbot):
    """Clicking Run before the real solver is wired must not raise NotImplementedError
    from inside the Qt slot (which can abort the process under PySide6 6.x)."""
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    step = NlsqStep(st)  # no fit_fn injected -> falls back to _default_fit_fn stub
    qtbot.addWidget(step)
    step.run()  # must not raise
    assert st.nlsq_result is None
    assert step.is_ready() is False


def test_nlsq_solver_failure_is_reported_without_escaping_qt_slot(qtbot):
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )

    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    step = NlsqStep(st, fit_fn=fail)
    qtbot.addWidget(step)
    step.run()
    assert st.nlsq_result is None
    assert "optimizer failed" in step._result.text()
