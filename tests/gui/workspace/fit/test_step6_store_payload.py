from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step6_export import ExportStep


def test_save_to_library_stores_fitted_curve_payload(qtbot):
    st = FitState(
        model_key="power_law", protocol="flow_curve",
        nlsq_result={"params": {"a": 1.0}, "r_squared": 0.9,
                     "x": [1.0, 2.0], "y_fit": [1.1, 1.9]},
    )
    lib = DatasetLibrary()
    step = ExportStep(st, lib)
    qtbot.addWidget(step)
    new_id = step.save_to_library()
    payload = lib.load_payload(new_id)  # must not raise KeyError
    assert list(payload.x) == [1.0, 2.0]
