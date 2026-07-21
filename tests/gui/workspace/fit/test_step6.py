import pytest

pytest.importorskip("PySide6")

from rheojax.gui.compat import QFileDialog
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import ActiveJobsState, FitState
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


def test_export_holds_worker_ref_and_registers_active_job(qtbot, monkeypatch, tmp_path):
    """Regression: _active_export_workers/active_jobs registration must
    happen synchronously inside _on_export_clicked() and clear exactly once
    the export finishes. self._active_export_workers[job_id] = worker looks
    like a dead store to a future cleanup pass (assigned, later popped,
    never read in between) -- deleting it would silently reintroduce the
    GC-lifetime bug where the worker's parentless `.signals` QObject could
    be collected mid-run, and nothing else in this suite would catch that
    deletion."""
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
    active_jobs = ActiveJobsState()
    step = ExportStep(st, lib, active_jobs)
    qtbot.addWidget(step)

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )

    step._on_export_clicked()
    # Synchronous up to QThreadPool.start(): both registrations must already
    # be in place before the worker thread has any chance to run.
    assert len(step._active_export_workers) == 1
    assert len(active_jobs.by_id) == 1

    qtbot.waitUntil(lambda: "Exported to" in step._export_status.text())
    assert step._active_export_workers == {}
    assert active_jobs.by_id == {}


def _non_converged_state(**overrides):
    kwargs = dict(
        protocol="oscillation",
        model_key="maxwell",
        model_config={},
        data_ref="d",
        nlsq_result={"params": {"G0": 1.0}},
        nuts_result={
            "verdict": {"converged": False, "reasons": ["ESS too low for G0"]}
        },
        revision=2,
    )
    kwargs.update(overrides)
    return FitState(**kwargs)


def test_save_button_confirm_gate_blocks_on_non_converged_cancel(qtbot, monkeypatch):
    """PR #104 gate: clicking the real Save button (not calling
    save_to_library() directly) with a non-converged NUTS verdict must show
    a confirm dialog, and Cancel must prevent the save from proceeding."""
    from rheojax.gui.compat import QMessageBox

    st = _non_converged_state()
    lib = DatasetLibrary()
    step = ExportStep(st, lib)
    qtbot.addWidget(step)
    _commit_on_request(step, lib)

    captured = {}

    def fake_question(self_, title, text, *args, **kwargs):
        captured["text"] = text
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", fake_question)

    emitted = {"called": False}
    step.dataset_commit_requested.connect(lambda *a: emitted.__setitem__("called", True))

    step._save_btn.click()

    assert "did not converge" in captured["text"].lower()
    assert emitted["called"] is False


def test_save_button_confirm_gate_proceeds_on_non_converged_yes(qtbot, monkeypatch):
    """Positive-path guard: answering Yes to the non-converged confirm must
    let the real Save button proceed to save_to_library()."""
    from rheojax.gui.compat import QMessageBox

    st = _non_converged_state()
    lib = DatasetLibrary()
    step = ExportStep(st, lib)
    qtbot.addWidget(step)
    _commit_on_request(step, lib)

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )

    with qtbot.waitSignal(step.dataset_commit_requested, timeout=1000):
        step._save_btn.click()

    assert lib.get(step.provenance()["model_key"] + "_fit_2").origin == "derived"


def test_export_button_confirm_gate_blocks_on_non_converged_cancel(
    qtbot, monkeypatch, tmp_path
):
    """The real Export Bundle button must also honor the non-converged
    confirm gate -- Cancel must stop before even prompting for a directory."""
    from rheojax.gui.compat import QFileDialog, QMessageBox

    st = _non_converged_state()
    lib = DatasetLibrary()
    step = ExportStep(st, lib)
    qtbot.addWidget(step)

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    dir_dialog_called = {"called": False}
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: dir_dialog_called.__setitem__("called", True) or str(tmp_path),
    )

    step._export_btn.click()

    assert dir_dialog_called["called"] is False
    assert step._export_status.text() == ""


def test_export_button_confirm_gate_proceeds_on_non_converged_yes(
    qtbot, monkeypatch, tmp_path
):
    """Answering Yes to the non-converged confirm must let Export Bundle
    proceed to the directory prompt and the export itself."""
    from rheojax.gui.compat import QFileDialog, QMessageBox

    st = _non_converged_state()
    lib = DatasetLibrary()
    step = ExportStep(st, lib)
    qtbot.addWidget(step)

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )

    step._export_btn.click()
    qtbot.waitUntil(lambda: "Exported to" in step._export_status.text())
