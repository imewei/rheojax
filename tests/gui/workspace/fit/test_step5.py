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
    assert v.arviz_plots() == [
        "pair", "forest", "energy", "autocorr", "rank", "ess", "trace", "posterior"
    ]


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
    v.refresh()  # must build 8 ArvizCanvas widgets without raising

    assert "Diagnostics" in v.tab_names()
    assert v.arviz_plots() == [
        "pair", "forest", "energy", "autocorr", "rank", "ess", "trace", "posterior"
    ]


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


def test_diagnostics_summary_labels_ok_status(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "r_hat": {"a": 1.0, "b": 1.02},
        "ess": {"a": 800.0, "b": 900.0},
        "divergences": 0,
    }
    v.refresh()
    assert v._rhat_label.text() == "R-hat (max): 1.0200 [OK]"
    assert v._ess_label.text() == "ESS (min): 800 [OK]"
    assert v._divergence_label.text() == "Divergences: 0"


def test_diagnostics_summary_labels_warning_status(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "r_hat": {"a": 1.2},
        "ess": {"a": 100.0},
        "divergences": 5,
    }
    v.refresh()
    assert v._rhat_label.text() == "R-hat (max): 1.2000 [WARNING]"
    assert v._ess_label.text() == "ESS (min): 100 [LOW]"
    assert v._divergence_label.text() == "Divergences: 5"


def test_diagnostics_summary_labels_mixed_finite_and_nan_is_failing(qtbot):
    # Regression: max()/min() are order-dependent with NaN present -- a NaN
    # never compares >/< a finite value, so `max([1.0, nan])` silently
    # returns 1.0 (the finite value), not nan. Checking isnan() only on the
    # aggregate result (rather than on every dict entry, like
    # step4_nuts.py::_diagnostics_verdict() already does) would let a
    # genuinely unverifiable diagnostic read as "OK". An all-NaN dict can't
    # catch this bug -- it must be a MIX of finite and NaN values.
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "r_hat": {"finite": 1.0, "nan": float("nan")},
        "ess": {"finite": 800.0, "nan": float("nan")},
    }
    v.refresh()
    assert "[WARNING]" in v._rhat_label.text()
    assert "[LOW]" in v._ess_label.text()


def test_diagnostics_summary_labels_none_entry_excluded(qtbot):
    # Regression: individual r_hat/ess dict VALUES can be None (not just
    # NaN) -- step4_nuts.py::_diagnostics_verdict() already guards this
    # exact case (`if r_hat is None: continue`), confirming it's a real,
    # reachable shape from the NumPyro/ArviZ pipeline. max()/min() raise
    # TypeError comparing None to a float, so None entries must be
    # filtered out before aggregating, not just checked for NaN.
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "r_hat": {"a": 1.0, "b": None},
        "ess": {"a": 800.0, "b": None},
    }
    v.refresh()  # must not raise TypeError
    assert v._rhat_label.text() == "R-hat (max): 1.0000 [OK]"
    assert v._ess_label.text() == "ESS (min): 800 [OK]"


def test_diagnostics_summary_labels_missing_fallback(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {}  # populated dict with no r_hat/ess/divergences keys
    v.refresh()
    assert v._rhat_label.text() == "R-hat: --"
    assert v._ess_label.text() == "ESS: --"
    assert v._divergence_label.text() == "Divergences: 0"


def test_diagnostics_summary_labels_style_reset_after_warning_clears(qtbot):
    # Regression: transitioning from a WARNING-colored label back to the
    # "--" placeholder must not leave the stale warning color behind.
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {"r_hat": {"a": 1.2}}  # over threshold -> WARNING color
    v.refresh()
    assert "color:" in v._rhat_label.styleSheet()

    st.nuts_result = {}  # r_hat now missing -> falls back to "--"
    v.refresh()
    assert v._rhat_label.text() == "R-hat: --"
    assert "color:" not in v._rhat_label.styleSheet()


def test_diagnostics_summary_divergences_unknown_sentinel(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {"divergences": -1}
    v.refresh()
    assert v._divergence_label.text() == "Divergences: unknown"


def test_diagnostics_summary_updates_on_second_nuts_run(qtbot):
    # A refresh() cycle that replaces one non-None nuts_result with another
    # (re-running NUTS with different settings), not just None -> result.
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {"r_hat": {"a": 1.0}, "ess": {"a": 800.0}, "divergences": 0}
    v.refresh()
    assert v._divergence_label.text() == "Divergences: 0"

    st.nuts_result = {"r_hat": {"a": 1.0}, "ess": {"a": 800.0}, "divergences": 3}
    v.refresh()
    assert v._divergence_label.text() == "Divergences: 3"


def test_diagnostics_intervals_table_three_tuple(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {"credible_intervals": {"G0": (900.0, 1000.0, 1100.0)}}
    v.refresh()
    assert v._intervals_table.rowCount() == 1
    assert v._intervals_table.item(0, 0).text() == "G0"
    assert v._intervals_table.item(0, 1).text() == "1000"
    assert v._intervals_table.item(0, 2).text() == "900"
    assert v._intervals_table.item(0, 3).text() == "1100"


def test_diagnostics_intervals_table_dict_shape(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "credible_intervals": {"eta": {"lower": 45.0, "median": 50.0, "upper": 55.0}}
    }
    v.refresh()
    assert v._intervals_table.item(0, 1).text() == "50"
    assert v._intervals_table.item(0, 2).text() == "45"
    assert v._intervals_table.item(0, 3).text() == "55"


def test_diagnostics_intervals_table_dict_shape_missing_fields_default_zero(qtbot):
    # Spec requires dict-shape entries to default absent fields to 0.0
    # (not skip the row) -- distinct from "malformed" (wrong type/length).
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {"credible_intervals": {"eta": {"lower": 45.0}}}
    v.refresh()
    assert v._intervals_table.rowCount() == 1
    assert v._intervals_table.item(0, 1).text() == "0"
    assert v._intervals_table.item(0, 2).text() == "45"
    assert v._intervals_table.item(0, 3).text() == "0"


def test_diagnostics_intervals_table_malformed_entry_skipped(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "credible_intervals": {"good": (1.0, 2.0, 3.0), "bad": "not-a-valid-shape"}
    }
    v.refresh()  # must not raise
    assert v._intervals_table.rowCount() == 1
    assert v._intervals_table.item(0, 0).text() == "good"


def test_diagnostics_intervals_table_two_tuple_uses_summary_mean(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "credible_intervals": {"eta": (45.0, 55.0)},
        "summary": {"eta": {"mean": 50.0}},
    }
    v.refresh()
    assert v._intervals_table.item(0, 1).text() == "50"
    assert v._intervals_table.item(0, 2).text() == "45"
    assert v._intervals_table.item(0, 3).text() == "55"


def test_diagnostics_intervals_table_two_tuple_summary_entry_is_none(qtbot):
    # Regression: summary.get(param_name) can return an explicit None (key
    # present, value None) rather than merely absent -- `(x or {}).get(...)`
    # must guard this, or `None.get("mean", 0.0)` raises AttributeError.
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "credible_intervals": {"eta": (45.0, 55.0)},
        "summary": {"eta": None},
    }
    v.refresh()  # must not raise AttributeError
    assert v._intervals_table.item(0, 1).text() == "0"
    assert v._intervals_table.item(0, 2).text() == "45"
    assert v._intervals_table.item(0, 3).text() == "55"


def test_diagnostics_summary_and_table_both_update_on_second_nuts_run(qtbot):
    # Fuller version of Task 2's test_diagnostics_summary_updates_on_second_
    # nuts_run, now that the credible-intervals table exists too -- the spec
    # requires a refresh() cycle replacing one non-None nuts_result with
    # another to update BOTH the labels and the table, not just labels.
    st = FitState(nlsq_result={"params": {}}, nuts_result=None)
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    st.nuts_result = {
        "r_hat": {"a": 1.0},
        "ess": {"a": 800.0},
        "divergences": 0,
        "credible_intervals": {"a": (0.9, 1.0, 1.1)},
    }
    v.refresh()
    assert v._divergence_label.text() == "Divergences: 0"
    assert v._intervals_table.item(0, 1).text() == "1"

    st.nuts_result = {
        "r_hat": {"a": 1.2},
        "ess": {"a": 100.0},
        "divergences": 3,
        "credible_intervals": {"a": (1.9, 2.0, 2.1)},
    }
    v.refresh()
    assert v._divergence_label.text() == "Divergences: 3"
    assert "[WARNING]" in v._rhat_label.text()
    assert "[LOW]" in v._ess_label.text()
    assert v._intervals_table.item(0, 1).text() == "2"


def test_diagnostics_tab_cleanup_no_dangling_refs_on_removal(qtbot):
    st = FitState(nlsq_result={"params": {}}, nuts_result={"r_hat": {"a": 1.0}})
    v = VisualizeStep(st)
    qtbot.addWidget(v)
    assert hasattr(v, "_rhat_label")
    assert hasattr(v, "_intervals_table")

    st.nuts_result = None
    v.refresh()  # removes the Diagnostics tab

    assert not hasattr(v, "_rhat_label")
    assert not hasattr(v, "_ess_label")
    assert not hasattr(v, "_divergence_label")
    assert not hasattr(v, "_intervals_table")

    # Regression: a later refresh() must not touch a widget whose tab was
    # already removed, before deleteLater()'s deferred cleanup runs.
    st.nuts_result = {"r_hat": {"a": 1.0}}
    v.refresh()
    assert hasattr(v, "_rhat_label")
