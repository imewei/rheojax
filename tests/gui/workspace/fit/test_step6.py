import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step6_export import ExportStep


def test_bundle_and_save(qtbot):
    st = FitState(
        protocol="oscillation",
        model_key="maxwell",
        model_config={},
        data_ref="d",
        nlsq_result={"params": {"G0": 1.0}},
        nuts_result=None,
        revision=2,
    )
    lib = DatasetLibrary()
    step = ExportStep(st, lib)
    qtbot.addWidget(step)
    man = step.bundle_manifest()
    assert "parameters" in man and "posterior_samples" not in man  # no NUTS
    new_id = step.save_to_library()
    assert lib.get(new_id).origin == "derived"
    assert step.provenance()["model_key"] == "maxwell"
