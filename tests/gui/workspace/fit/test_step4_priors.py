from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step4_nuts import NutsStep


def test_load_suggested_priors_seeds_editor(qtbot):
    st = FitState(nlsq_result={"params": {"a": 1.0, "b": 2.0}, "r_squared": 0.9})
    step = NutsStep(st)
    qtbot.addWidget(step)
    step.load_suggested_priors()
    priors = step.priors_editor().get_all_priors()
    assert set(priors.keys()) >= {"a", "b", "sigma"}


def test_run_falls_back_to_suggested_priors_when_editor_never_seeded(qtbot):
    # load_suggested_priors() is never called here -- the editor stays empty,
    # so to_numpyro_priors() returns {} and run() must fall back to the raw
    # suggested_priors() (map_centered_priors of the NLSQ MAP estimate)
    # rather than sampling with no priors at all.
    st = FitState(nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9})
    captured = {}

    def fake_sample_fn(priors, warm_start, config):
        captured["priors"] = priors
        return {"posterior_samples": {"a": [1.0]}}

    step = NutsStep(st, sample_fn=fake_sample_fn)
    qtbot.addWidget(step)

    step.run()

    assert captured["priors"] == step.suggested_priors()
    assert captured["priors"]["a"]["type"] == "lognormal"
    assert "sigma" in captured["priors"]


def test_run_uses_edited_priors(qtbot):
    st = FitState(nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9})
    captured = {}

    def fake_sample_fn(priors, warm_start, config):
        captured["priors"] = priors
        return {"posterior_samples": {"a": [1.0]}}

    step = NutsStep(st, sample_fn=fake_sample_fn)
    qtbot.addWidget(step)
    step.load_suggested_priors()
    step.priors_editor().set_prior("a", "normal", loc=5.0, scale=1.0)  # user override

    step.run()
    assert captured["priors"]["a"]["type"] == "normal"
    assert captured["priors"]["a"]["loc"] == 5.0
