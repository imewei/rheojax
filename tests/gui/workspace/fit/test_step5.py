import numpy as np
import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step5_visualize import VisualizeStep


def test_diagnostics_tab_only_with_nuts(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    assert v.tab_names() == ["Fit overlay", "Residuals"]  # no Diagnostics yet


def test_diagnostics_tab_added_by_refresh(qtbot):
    # Simulate the real flow: built with nuts_result=None, then NUTS completes
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    assert "Diagnostics" not in v.tab_names()
    st.nuts_result = {"rhat": 1.0}  # NUTS worker sets this
    v.refresh()  # controller calls this on NutsStep.finished
    assert "Diagnostics" in v.tab_names()
    assert v.arviz_plots() == ["pair", "forest", "energy", "autocorr", "rank", "ess"]


def test_diagnostics_builds_with_real_nuts_result(qtbot):
    # Shaped like the worker output: NlsqStep normalizes FitResult to a plain
    # dict (see step3_nlsq.py); by analogy the NUTS sample_fn wired in
    # build_fit_controller normalizes BayesianResult the same way — see
    # BayesianResult in rheojax/gui/state/store.py and its consumption in
    # rheojax/gui/pages/diagnostics_page.py::_get_inference_data.
    rng = np.random.default_rng(0)
    num_chains, num_draws = 2, 50
    nuts_result = {
        "posterior_samples": {
            "G0": rng.normal(1000.0, 10.0, size=num_chains * num_draws),
            "eta": rng.normal(50.0, 1.0, size=num_chains * num_draws),
        },
        "sample_stats": {
            "energy": rng.normal(10.0, 1.0, size=num_chains * num_draws),
            "diverging": np.zeros(num_chains * num_draws, dtype=bool),
        },
        "num_chains": num_chains,
        "r_hat": {"G0": 1.01, "eta": 1.0},
        "ess": {"G0": 400.0, "eta": 420.0},
        "divergences": 0,
    }
    st = FitState(nlsq_result={"params": {"G0": 1000.0, "eta": 50.0}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)

    st.nuts_result = nuts_result
    v.refresh()  # must build 6 ArvizCanvas widgets without raising

    assert "Diagnostics" in v.tab_names()
    assert v.arviz_plots() == ["pair", "forest", "energy", "autocorr", "rank", "ess"]
