from __future__ import annotations

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
