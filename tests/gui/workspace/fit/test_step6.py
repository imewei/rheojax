import pytest

pytest.importorskip("PySide6")

from rheojax.gui.compat import QFileDialog
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step6_export import ExportStep


def _commit_on_request(step, lib):
    # Simulates the Task-10 WorkspaceWindow._commit_dataset handler that will
    # eventually own this: the widget only requests a commit, it no longer
    # performs one itself.
    def _handle(ref, payload, overwrite):
        lib.add(ref, overwrite=overwrite)
        if payload is not None:
            lib.store_payload(ref.id, payload)

    step.dataset_commit_requested.connect(_handle)


def test_save_emits_dataset_commit_requested_with_overwrite_true(qtbot):
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
    assert not hasattr(step, "exported")
    with qtbot.waitSignal(step.dataset_commit_requested, timeout=1000) as blocker:
        step.save_to_library()
    ref, payload, overwrite = blocker.args
    assert ref.origin == "derived"
    assert overwrite is True
    # The library must NOT have been mutated by the widget itself:
    with pytest.raises(KeyError):
        step._library.get(ref.id)


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
    _commit_on_request(step, lib)
    man = step.bundle_manifest()
    assert "parameters" in man and "posterior_samples" not in man  # no NUTS
    new_id = step.save_to_library()
    assert lib.get(new_id).origin == "derived"
    assert step.provenance()["model_key"] == "maxwell"


def test_save_to_library_reruns_at_same_revision_overwrite(qtbot):
    # step6_export.py's id policy is `{model_key}_fit_{revision}`, reused
    # across re-saves at the same revision (revision only bumps on an
    # edit-invalidation, not on every save). overwrite=True lets a repeat
    # save at the same revision replace the prior entry instead of raising.
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
    _commit_on_request(step, lib)
    id1 = step.save_to_library()
    id2 = step.save_to_library()
    assert id1 == id2
    assert lib.get(id2).origin == "derived"


def test_on_export_clicked_reports_failure_instead_of_raising(qtbot, monkeypatch, tmp_path):
    """Regression: _on_export_clicked called export_bundle() unguarded, so an
    unwritable directory or export failure propagated uncaught out of the Qt
    slot instead of updating the status label with an error."""
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

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )

    def _raise(_directory):
        raise OSError("disk full")

    monkeypatch.setattr(step, "export_bundle", _raise)

    step._on_export_clicked()  # must not raise
    qtbot.waitUntil(lambda: "failed" in step._export_status.text().lower())
    assert "disk full" in step._export_status.text()
