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


def test_diagnostics_tab_removed_when_nuts_result_invalidated(qtbot):
    # Regression: refresh() added the Diagnostics tab when nuts_result appears
    # but never removed it when nuts_result is later cleared (e.g. by the
    # upstream invalidation cascade) -- stale plots would persist forever.
    st = FitState(nlsq_result={"params": {}}, nuts_result={"rhat": 1.0})
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    assert "Diagnostics" in v.tab_names()

    st.nuts_result = None  # invalidation cascade cleared it
    v.refresh()

    assert "Diagnostics" not in v.tab_names()
    assert v.tab_names() == ["Fit overlay", "Residuals"]
    assert v._arviz_canvases == {}


def test_refresh_overlay_handles_ndarray_y_band_without_raising(qtbot, monkeypatch):
    # Regression: `if band:` on a raw numpy ndarray raises
    # "ValueError: The truth value of an array...is ambiguous". y_band isn't
    # populated by any current code path, but must be guarded defensively.
    #
    # NOTE: line_style="dash" plotting goes through a separate, pre-existing,
    # unrelated bug in PyQtGraphCanvas.plot_line (raw int passed to
    # QPen.setStyle instead of a Qt.PenStyle enum) -- out of scope here. Stub
    # plot_line so this test isolates the ambiguous-truth-value fix only.
    from rheojax.gui.widgets.pyqtgraph_canvas import PyQtGraphCanvas

    # __init__ -> refresh() -> _refresh_overlay() runs immediately, before a
    # test could patch the instance -- patch the class ahead of construction.
    monkeypatch.setattr(PyQtGraphCanvas, "plot_line", lambda *a, **k: None)

    st = FitState(
        nlsq_result={
            "params": {},
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([1.0, 2.0, 3.0]),
            "y_fit": np.array([1.1, 2.1, 3.1]),
        },
        nuts_result={"y_band": np.array([[0.9, 1.9, 2.9], [1.2, 2.2, 3.2]])},
    )
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    v._refresh_overlay()  # must not raise ValueError on ambiguous truth value
