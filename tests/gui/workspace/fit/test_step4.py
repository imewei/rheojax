import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step4_nuts import NutsStep


def test_nuts_warmstart_priors_and_skip(qtbot):
    st = FitState(nlsq_result={"params": {"G0": 1000.0, "eta": 50.0}})
    step = NutsStep(st, sample_fn=lambda *a, **k: {"rhat": 1.0})
    qtbot.addWidget(step)
    pri = step.suggested_priors()  # MAP-centered
    assert set(pri) == {"G0", "eta", "sigma"} and pri["G0"]["type"] == "lognormal"
    step.skip()
    assert st.nuts_result is None and step.is_ready() is True  # skip => ready


def test_nuts_run_without_wired_sample_fn_does_not_crash(qtbot):
    """Clicking Sample before the real sampler is wired must not raise NotImplementedError
    from inside the Qt slot (which can abort the process under PySide6 6.x)."""
    st = FitState(nlsq_result={"params": {"G0": 1000.0, "eta": 50.0}})
    step = NutsStep(st)  # no sample_fn injected -> falls back to _default_sample_fn stub
    qtbot.addWidget(step)
    step.run()  # must not raise
    assert st.nuts_result is None
    assert step.is_ready() is False
