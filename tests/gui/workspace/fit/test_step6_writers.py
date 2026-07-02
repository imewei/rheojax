from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step6_export import ExportStep


class _RheoData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_export_bundle_writes_expected_files(qtbot, tmp_path, monkeypatch):
    st = FitState(
        model_key="power_law", protocol="flow_curve", data_ref="d1",
        nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9,
                     "x": [1.0, 2.0], "y": [1.0, 2.0], "y_fit": [1.1, 1.9]},
    )
    lib = DatasetLibrary()
    step = ExportStep(st, lib)
    qtbot.addWidget(step)

    written = step.export_bundle(tmp_path)

    assert written["parameters"].exists()
    assert written["fitted_curve"].exists()
    assert "posterior_samples" not in written  # no NUTS result -> no posterior file


def test_export_bundle_includes_posterior_when_nuts_ran(qtbot, tmp_path):
    import numpy as np

    st = FitState(
        model_key="power_law", protocol="flow_curve", data_ref="d1",
        nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9,
                     "x": [1.0, 2.0], "y": [1.0, 2.0], "y_fit": [1.1, 1.9]},
        nuts_result={
            "posterior_samples": {"a": list(np.random.default_rng(0).normal(size=400))},
            "sample_stats": {"diverging": [False] * 400},
            "num_chains": 4,
            "verdict": {"converged": True, "reasons": []},
        },
    )
    lib = DatasetLibrary()
    step = ExportStep(st, lib)
    qtbot.addWidget(step)

    written = step.export_bundle(tmp_path)
    assert written["posterior_samples"].exists()
    assert written["posterior_samples"].suffix == ".nc"
