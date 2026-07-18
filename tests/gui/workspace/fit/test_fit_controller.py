import queue
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
import rheojax.models  # noqa
from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.fit.fit_controller import _run_on_thread, build_fit_controller
from rheojax.gui.workspace.window import WorkspaceWindow


class _RheoData:
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)


def test_run_on_thread_drains_progress_queue(qtbot):
    # Regression: progress_queue was created in _fit_fn_body/_sample_fn and
    # handed to run_fit_isolated/run_bayesian_isolated, but _run_on_thread
    # never read from it -- every progress_callback() message piled up
    # unread and never reached the UI. _run_on_thread must poll and forward
    # messages (via on_progress) while the worker runs, not just discard
    # them.
    q = queue.Queue()
    seen = []

    def fn():
        q.put({"type": "progress", "percent": 1})
        time.sleep(0.05)
        q.put({"type": "progress", "percent": 50})
        time.sleep(0.25)  # spans multiple QTimer drain ticks (100ms each)
        q.put({"type": "progress", "percent": 100})
        return "done"

    result = _run_on_thread(fn, progress_queue=q, on_progress=seen.append)

    assert result == "done"
    assert [m["percent"] for m in seen] == [1, 50, 100]
    assert q.empty()


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


def test_config_widget_edit_survives_invalidation_cascade(qtbot):
    # Regression: build_fit_controller wired ProtocolModelStep's constructor-
    # config widgets to the same `edited` signal used for protocol/model
    # swaps, which invalidates downstream with the "model_key" cascade key.
    # That cascade clears model_config back to {} (invalidation.py's _CLEAR),
    # so every real n_modes/kinetics/etc widget edit was silently discarded
    # the instant it fired -- app_state.fit.model_config never reflected
    # what Step 1 displayed. Constructor-config widgets must use the
    # narrower "model_config" cascade (via config_edited) which leaves
    # model_config itself untouched.
    app = AppState()
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    bodies[0].set_protocol("relaxation")
    bodies[0].set_model("generalized_maxwell")
    assert app.fit.model_config == {}

    bodies[0].set_model_config({"n_modes": 4})

    assert app.fit.model_config == {"n_modes": 4}


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
    app.library.store_payload(
        "osc1", _RheoData(np.linspace(0.1, 10.0, 64), np.linspace(1.0, 5.0, 64))
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


def test_refresh_on_unrelated_notify_preserves_fit_results(qtbot):
    # End-to-end regression: DataStep.refresh() also runs when
    # WorkspaceWindow wires DatasetLibraryNotifier.changed -> refresh() for
    # events unrelated to Step 1 (e.g. "Save fit to library", another
    # dataset import) -- not just via the Step-1 edit cascade exercised by
    # test_model_only_edit_restores_still_valid_dataset_selection above.
    # Before the fix, refresh() unconditionally re-ran _on_select(), which
    # unconditionally emits `edited`; through this real controller's wiring
    # (body.edited -> _cascade_and_relock(1, "column_map") ->
    # invalidate_downstream), that silently wiped nlsq_result/nuts_result
    # for a selection that never changed. This drives the actual wired
    # cascade (unlike test_step2.py's DataStep-in-isolation test, which
    # only proves `edited` doesn't fire and can't see this invalidation).
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
    app.library.store_payload(
        "osc1", _RheoData(np.linspace(0.1, 10.0, 64), np.linspace(1.0, 5.0, 64))
    )
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)

    bodies[0].set_protocol("oscillation")
    bodies[0].set_model("maxwell")
    bodies[1].select_dataset("osc1")
    app.fit.nlsq_result = {"success": True, "params": {}}
    app.fit.nuts_result = {"success": True}

    bodies[1].refresh()  # simulates DatasetLibraryNotifier.changed -> refresh()

    assert app.fit.data_ref == "osc1"
    assert app.fit.nlsq_result is not None
    assert app.fit.nuts_result is not None


def test_export_step_unlocked_once_visualize_reached(qtbot):
    # Regression: ExportStep (index 5) was structurally unreachable.
    # WorkflowController.advance() is only ever invoked from
    # WorkspaceWindow._on_fit_body_edited, which is only wired to bodies with
    # an `edited` signal (indices 0-3). VisualizeStep (4) and ExportStep (5)
    # have no `edited` signal, so nothing ever pushed `reached` past index 4
    # once the user landed there after NUTS finished. Both steps fall back to
    # the trivial `is_ready=lambda: True` default in build_fit_controller, so
    # the gap is purely navigational: index 5 must be unlocked (added to
    # `reached`) as soon as index 4 is reached, without forcing `current`
    # to jump there.
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
    app.library.store_payload(
        "osc1", _RheoData(np.linspace(0.1, 10.0, 64), np.linspace(1.0, 5.0, 64))
    )
    win = WorkspaceWindow(app)
    qtbot.addWidget(win)
    ctl = win._controllers["fit"]
    assert ctl.reached == {0}  # nothing unlocked prematurely before Step 1 is filled in

    bodies = win.fit_bodies()
    bodies[2]._fit_fn = lambda *a, **k: {
        "params": {"a": 1.0},
        "r_squared": 0.99,
        "success": True,
    }
    bodies[3]._sample_fn = lambda *a, **k: {
        "posterior_samples": {},
        "sample_stats": None,
    }

    bodies[0].set_protocol("oscillation")
    bodies[0].set_model("maxwell")
    bodies[1].select_dataset("osc1")
    bodies[2].run()  # NLSQ
    bodies[3].run()  # NUTS

    assert ctl.current == 4  # landed on Visualize; not force-advanced past it
    assert 5 in ctl.reached  # Export unlocked because Visualize is trivially ready
    assert ctl.goto(5) is True
    assert ctl.current == 5
