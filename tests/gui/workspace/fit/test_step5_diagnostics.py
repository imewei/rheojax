from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step4_nuts import NutsStep, _diagnostics_verdict
from rheojax.gui.workspace.fit.step5_visualize import VisualizeStep


def test_diagnostics_verdict_converged():
    result = {
        "r_hat": {"a": 1.0, "b": 1.01},
        "ess": {"a": 800, "b": 900},
        "sample_stats": {"energy": [1.0, 1.05, 0.98, 1.02] * 25, "diverging": [False] * 100},
    }
    v = _diagnostics_verdict(result)
    assert v["converged"] is True
    assert v["reasons"] == []


def test_diagnostics_verdict_not_converged_high_rhat():
    result = {
        "r_hat": {"a": 1.2},
        "ess": {"a": 800},
        "sample_stats": {"energy": [1.0] * 100, "diverging": [False] * 100},
    }
    v = _diagnostics_verdict(result)
    assert v["converged"] is False
    assert any("r_hat" in r for r in v["reasons"])


def test_diagnostics_verdict_numpy_arrays_converged():
    # Regression: sample_stats["energy"]/["diverging"] are real numpy arrays
    # in production (see subprocess_bayesian.py's sample_stats_np build-up),
    # not Python lists. `if energy:` and `... or []` both call bool() on the
    # array, which raises ValueError for arrays with more than one element.
    result = {
        "r_hat": {"a": 1.0, "b": 1.01},
        "ess": {"a": 800, "b": 900},
        "sample_stats": {
            "energy": np.array([1.0, 1.05, 0.98, 1.02] * 25),
            "diverging": np.array([False] * 100),
        },
    }
    v = _diagnostics_verdict(result)
    assert v["converged"] is True
    assert v["reasons"] == []


def test_diagnostics_verdict_numpy_arrays_not_converged_divergences():
    result = {
        "r_hat": {"a": 1.0},
        "ess": {"a": 800},
        "sample_stats": {
            "energy": np.array([1.0, 1.05, 0.98, 1.02] * 25),
            "diverging": np.array([False] * 95 + [True] * 5),
        },
    }
    v = _diagnostics_verdict(result)
    assert v["converged"] is False
    assert any("divergent" in r for r in v["reasons"])


def test_diagnostics_verdict_2d_multichain_diverging_does_not_crash():
    # Regression: real ArviZ sample_stats["diverging"] is shape
    # (num_chains, num_samples), e.g. (4, 1000) for the GUI's 4-chain NUTS
    # runs (fit_controller.py num_chains=4). `sum(1 for d in diverging if d)`
    # iterates per-chain rows (each a >1-element bool array) and `if d:`
    # raises "truth value of an array with more than one element is
    # ambiguous" -- this must count divergences across the whole array
    # instead of crashing.
    diverging = np.zeros((4, 1000), dtype=bool)
    diverging[0, 5] = True
    diverging[2, 100:103] = True
    result = {
        "r_hat": {"a": 1.0},
        "ess": {"a": 800},
        "bfmi": 0.5,
        "sample_stats": {"diverging": diverging},
    }
    v = _diagnostics_verdict(result)
    assert v["converged"] is False
    assert "4 divergent transitions" in v["reasons"]


def test_diagnostics_verdict_nan_bfmi_flags_unavailable_not_converged():
    # Regression: subprocess_bayesian.run_bayesian_isolated() sets
    # bfmi=float("nan") when the InferenceData had no "energy" sample_stat.
    # `nan < _BFMI_MIN` is always False in Python, so the old code silently
    # treated a genuinely unverifiable diagnostic as passing.
    result = {
        "r_hat": {"a": 1.0},
        "ess": {"a": 800},
        "bfmi": float("nan"),
        "sample_stats": {"diverging": np.zeros((4, 1000), dtype=bool)},
    }
    v = _diagnostics_verdict(result)
    assert v["converged"] is False
    assert any("BFMI unavailable" in r for r in v["reasons"])


def test_nuts_step_run_attaches_verdict(qtbot):
    st = FitState(nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9})

    def fake_sample_fn(priors, warm_start, config):
        # NOTE: energy must actually vary -- a constant energy trace makes
        # bfmi() return 0.0 (its explicit zero-variance special case), which
        # trips the "BFMI too low" reason and would make this fixture
        # not-converged regardless of r_hat/ess. Mirrors the fixture in
        # test_diagnostics_verdict_converged above.
        return {"r_hat": {"a": 1.0}, "ess": {"a": 900},
                "sample_stats": {
                    "energy": [1.0, 1.05, 0.98, 1.02] * 25,
                    "diverging": [False] * 100,
                }}

    step = NutsStep(st, sample_fn=fake_sample_fn)
    qtbot.addWidget(step)
    step.run()
    assert st.nuts_result["verdict"]["converged"] is True
    assert step.is_ready() is True


def test_visualize_diagnostics_badge_text(qtbot):
    st = FitState(
        nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9},
        nuts_result={
            "r_hat": {"a": 1.2}, "ess": {"a": 800},
            "sample_stats": {"energy": [1.0] * 100, "diverging": [False] * 100},
            "verdict": {"converged": False, "reasons": ["r_hat too high for a"]},
        },
    )
    step = VisualizeStep(st)
    qtbot.addWidget(step)
    assert step.diagnostics_badge_text() == "⚠ not-converged: r_hat too high for a"


def test_visualize_diagnostics_badge_converged(qtbot):
    st = FitState(
        nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9},
        nuts_result={"verdict": {"converged": True, "reasons": []}},
    )
    step = VisualizeStep(st)
    qtbot.addWidget(step)
    assert step.diagnostics_badge_text() == "✓ converged"


def test_visualize_diagnostics_badge_no_nuts(qtbot):
    st = FitState(nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9})
    step = VisualizeStep(st)
    qtbot.addWidget(step)
    assert step.diagnostics_badge_text() == ""
