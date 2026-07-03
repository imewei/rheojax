import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.step5_export import TransformExportStep


def _commit_on_request(step, lib):
    # Simulates the Task-10 WorkspaceWindow._commit_dataset handler that will
    # eventually own this: the widget only requests a commit, it no longer
    # performs one itself.
    def _handle(ref, payload, overwrite):
        lib.add(ref, overwrite=overwrite)
        if payload is not None:
            lib.store_payload(ref.id, payload)

    step.dataset_commit_requested.connect(_handle)


def test_save_emits_dataset_commit_requested_not_exported(qtbot):
    st = TransformState(
        transform_key="prony_conversion",
        slots={"input": "rel1"},
        result={"output": "rd", "protocol_type": "oscillation"},
    )
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)
    assert not hasattr(step, "exported")
    with qtbot.waitSignal(step.dataset_commit_requested, timeout=1000) as blocker:
        step.save_to_library()
    ref, payload, overwrite = blocker.args
    assert ref.origin == "derived"
    assert overwrite is False
    # The library must NOT have been mutated by the widget itself:
    with pytest.raises(KeyError):
        step._library.get(ref.id)


def test_save_derived_with_protocol_type(qtbot):
    st = TransformState(
        transform_key="prony_conversion",
        slots={"input": "rel1"},
        result={"output": "rd", "protocol_type": "oscillation"}
    )
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)
    _commit_on_request(step, lib)
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
    _commit_on_request(step1, lib)
    id1 = step1.save_to_library()

    st2 = TransformState(
        transform_key="cox_merz",
        slots={"input": "datasetB"},
        result={"output": "rd2", "protocol_type": "oscillation"},
    )
    step2 = TransformExportStep(st2, lib)
    qtbot.addWidget(step2)
    _commit_on_request(step2, lib)
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


def test_save_domain_changing_output_with_empty_protocol_type_still_stores(qtbot):
    # Regression: transform_controller._infer_protocol_type() deliberately
    # returns "" (not None) for domain-changing transforms (spectral/
    # decomposition) whose output has a real payload but no determinable
    # rheological protocol. save_to_library() used to no-op on ANY falsy
    # protocol_type (including ""), so these outputs could never be saved to
    # the library at all -- contradicting design §7's "stored but not
    # offered to typed Fit slots" for such outputs.
    st = TransformState(
        transform_key="fft_analysis",
        slots={"input": "rel1"},
        result={"output": "rd", "protocol_type": ""},
    )
    lib = DatasetLibrary()
    step = TransformExportStep(st, lib)
    qtbot.addWidget(step)
    _commit_on_request(step, lib)
    new_id = step.save_to_library()
    assert new_id is not None
    ref = lib.get(new_id)
    assert ref.origin == "derived"
    assert ref.protocol_type == ""
    # "" never matches a real protocol query -> not offered to typed Fit slots.
    assert new_id not in [r.id for r in lib.datasets_of_type("oscillation")]
    assert lib.load_payload(new_id) == "rd"
