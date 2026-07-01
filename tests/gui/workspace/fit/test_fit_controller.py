import pytest

pytest.importorskip("PySide6")
import rheojax.models  # noqa
from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.fit.fit_controller import build_fit_controller
from rheojax.gui.workspace.window import WorkspaceWindow


def test_controller_gating_and_invalidation(qtbot):
    app = AppState()
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    assert ctl.can_advance() is False  # step 1 not satisfied
    bodies[0].set_protocol("oscillation")
    bodies[0].set_model("maxwell")
    assert ctl.can_advance() is True  # step 1 ready -> can advance
    # simulate progress then an upstream edit re-locks downstream
    ctl.advance()  # at step 2 (data), reached {0,1}
    bodies[0].set_model("zener")  # edit step 1
    assert ctl.reached == {0}  # downstream re-locked


def test_window_uses_real_fit_steps(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    assert win.active_step_count() == 6  # fit controller has 6 real steps
    assert hasattr(win.fit_bodies()[0], "set_protocol")


def test_window_auto_advances_when_step_becomes_ready(qtbot):
    # Regression: nothing in production wiring ever called
    # WorkflowController.advance() -- the workflow was stuck on step 0 forever.
    # Driving the real Step 1 body to "ready" must auto-advance the fit
    # controller's `reached`/`current` WITHOUT the test calling .advance().
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    ctl = win._controllers["fit"]
    assert ctl.reached == {0}
    assert win.current_step() == 0

    win.fit_bodies()[0].set_protocol("oscillation")
    win.fit_bodies()[0].set_model("maxwell")

    assert 1 in ctl.reached
    assert win.current_step() == 1


def test_model_only_edit_restores_still_valid_dataset_selection(qtbot):
    # Regression: a Step-1 edit that changes only the model (protocol
    # unchanged) triggers the invalidation cascade (fit_controller._on_edit)
    # BEFORE DataStep.refresh() runs, clearing state.data_ref/column_map.
    # Since dataset filtering is by protocol only, the already-selected
    # dataset is still valid for the new model -- refresh()'s still_valid
    # branch must restore data_ref/column_map to match it, not leave them
    # cleared forever (which silently defeats Step 2's is_ready()/auto-advance).
    app = AppState()
    app.library.add(
        DatasetRef(
            id="osc1",
            name="osc1",
            protocol_type="oscillation",
            origin="imported",
            units={"x": "rad/s"},
            row_count=64,
            hash="h",
            provenance={},
            lineage=[],
        )
    )
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)

    bodies[0].set_protocol("oscillation")
    bodies[0].set_model("maxwell")
    bodies[1].select_dataset("osc1")
    assert app.fit.data_ref == "osc1"
    assert app.fit.column_map
    assert bodies[1].is_ready() is True

    bodies[0].set_model("zener")  # model-only edit; protocol unchanged

    assert app.fit.data_ref == "osc1"
    assert app.fit.column_map
    assert bodies[1].is_ready() is True
    assert bodies[1]._source.currentText() == "osc1"
