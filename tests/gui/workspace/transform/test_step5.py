import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.step5_export import TransformExportStep


def test_save_derived_with_protocol_type(qtbot):
    st = TransformState(
        transform_key="prony_conversion",
        slots={"input": "rel1"},
        result={"output": "rd", "protocol_type": "oscillation"}
    )
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)
    new_id = step.save_to_library()
    assert new_id is not None and lib.get(new_id).origin == "derived"
    assert lib.get(new_id).protocol_type == "oscillation"


def test_scalar_output_not_saved(qtbot):
    st = TransformState(
        transform_key="mutation_number",
        slots={"input": "r"},
        result={"output": 0.42}
    )
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)
    assert step.save_to_library() is None
