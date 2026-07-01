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
