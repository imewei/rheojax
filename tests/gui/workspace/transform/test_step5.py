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


def test_save_to_library_does_not_overwrite_on_repeat_transform_key(qtbot):
    lib = DatasetLibrary()

    st1 = TransformState(
        transform_key="cox_merz",
        slots={"input": "datasetA"},
        result={"output": "rd1", "protocol_type": "oscillation"},
    )
    step1 = TransformExportStep(st1, lib)
    qtbot.addWidget(step1)
    id1 = step1.save_to_library()

    st2 = TransformState(
        transform_key="cox_merz",
        slots={"input": "datasetB"},
        result={"output": "rd2", "protocol_type": "oscillation"},
    )
    step2 = TransformExportStep(st2, lib)
    qtbot.addWidget(step2)
    id2 = step2.save_to_library()

    assert id1 is not None and id2 is not None
    assert id1 != id2
    assert lib.get(id1) is not None
    assert lib.get(id2) is not None
    all_ids = {ref.id for ref in lib.all()}
    assert {id1, id2} <= all_ids


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
