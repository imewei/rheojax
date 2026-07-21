import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.pipeline.step1_configure_run import PipelineConfigureRunStep


def _ref(id_):
    return DatasetRef(
        id=id_,
        name=id_,
        protocol_type="oscillation",
        origin="imported",
        units={},
        row_count=1,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_add_step_appends_to_state(qtbot):
    state = AppState()
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)
    step.add_step("export", {"path": "out.csv"})
    assert len(state.pipeline.steps) == 1
    assert state.pipeline.steps[0].step_type == "export"


def test_transform_type_picker_excludes_multi_slot_transforms(qtbot):
    state = AppState()
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)
    offered = step.available_transform_keys()
    assert "cox_merz" not in offered
    assert "mastercurve" not in offered
    assert "srfs" not in offered


def test_is_ready_requires_steps_and_selected_datasets(qtbot):
    state = AppState()
    state.library.add(_ref("d1"))
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)
    assert step.is_ready() is False
    step.add_step("export", {"path": "out.csv"})
    assert step.is_ready() is False  # steps set, but no dataset selected yet
    step.set_selected_dataset_ids(["d1"])
    assert step.is_ready() is True


def test_edited_signal_fires_on_add_step(qtbot):
    state = AppState()
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)
    with qtbot.waitSignal(step.edited, timeout=1000):
        step.add_step("export", {"path": "out.csv"})


def test_add_step_from_ui_collects_fit_config(qtbot):
    # A user-created fit step must have a real model_name -- an empty {} config (as an earlier
    # version of this widget produced) fails at execute() time with KeyError: 'model_name'.
    state = AppState()
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)
    step._step_type_combo.setCurrentText("fit")
    step._fit_model_combo.setCurrentText("maxwell")
    step._fit_run_nuts_checkbox.setChecked(False)
    step._on_add_step_clicked()
    assert state.pipeline.steps[0].config == {
        "model_name": "maxwell",
        "run_nuts": False,
    }


def test_add_step_from_ui_collects_export_config(qtbot):
    state = AppState()
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)
    step._step_type_combo.setCurrentText("export")
    step._export_path_edit.setText("/tmp/out_{id}.csv")
    step._export_format_combo.setCurrentText("csv")
    step._on_add_step_clicked()
    assert state.pipeline.steps[0].config == {
        "path": "/tmp/out_{id}.csv",
        "format": "csv",
    }


def test_is_ready_false_when_a_step_has_incomplete_config(qtbot):
    # A step added with an empty/incomplete config must not make is_ready() True -- otherwise
    # "Run All" is enabled against a step that will raise at execute() time.
    state = AppState()
    state.library.add(_ref("d1"))
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)
    step.add_step("fit", {})  # incomplete -- no model_name
    step.set_selected_dataset_ids(["d1"])
    assert step.is_ready() is False


def test_browse_button_populates_export_path(monkeypatch, qtbot):
    state = AppState()
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)

    monkeypatch.setattr(
        "rheojax.gui.workspace.pipeline.step1_configure_run.QFileDialog.getSaveFileName",
        lambda *a, **k: ("/tmp/chosen_{id}.csv", ""),
    )
    step._export_browse_btn.click()

    assert step._export_path_edit.text() == "/tmp/chosen_{id}.csv"


def test_browse_button_cancel_leaves_export_path_unchanged(monkeypatch, qtbot):
    # QFileDialog.getSaveFileName returns ("", "") when the user cancels the
    # native dialog -- the `if path:` guard must keep an existing value
    # instead of clobbering it with an empty string.
    state = AppState()
    step = PipelineConfigureRunStep(state.pipeline, state.library)
    qtbot.addWidget(step)
    step._export_path_edit.setText("/tmp/existing.csv")

    monkeypatch.setattr(
        "rheojax.gui.workspace.pipeline.step1_configure_run.QFileDialog.getSaveFileName",
        lambda *a, **k: ("", ""),
    )
    step._export_browse_btn.click()

    assert step._export_path_edit.text() == "/tmp/existing.csv"
