from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
import rheojax.transforms  # noqa
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.transform.slots_spec import SlotSpec
from rheojax.gui.workspace.transform.transform_controller import build_transform_controller
from rheojax.gui.workspace.window import WorkspaceWindow


def test_transform_controller_gating(qtbot):
    app = AppState()
    ctl, bodies = build_transform_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    assert ctl.can_advance() is False
    bodies[0].set_transform("cox_merz")
    assert ctl.can_advance() is True


def test_picking_transform_invalidates_downstream_state(qtbot):
    # Regression guard for the invalidation rule: re-picking the transform
    # (step 0) must clear slots/config/result and re-lock everything past it,
    # even after the workflow already progressed further.
    app = AppState()
    ctl, bodies = build_transform_controller(app)
    for b in bodies:
        qtbot.addWidget(b)

    bodies[0].set_transform("cox_merz")
    bodies[1].fill("oscillation", "ds1")
    bodies[1].fill("flow_curve", "ds2")
    app.transform.config["k"] = 1
    app.transform.result = {"output": "x"}
    ctl.advance()
    ctl.advance()
    assert ctl.reached == {0, 1, 2}

    bodies[0].set_transform("mastercurve")  # re-pick -> invalidation

    assert app.transform.slots == {}
    assert app.transform.config == {}
    assert app.transform.result is None
    assert ctl.reached == {0}
    # step bodies must observe the cleared state through their existing
    # `state` reference (no stale dataclass copy left behind)
    assert bodies[1].is_ready() is False


def test_window_transform_export_step_reachable_without_forced_navigation(qtbot):
    # Regression: TransformExportStep (index 4) has only an `exported`
    # signal, not `edited`; TransformVisualizeStep (index 3) is read-only
    # with no signal at all. If window.py only wired `edited` from bodies
    # 0-2 to advance()+refresh with no forward-unlock loop, index 4 would
    # stay permanently unreachable once the workflow lands on Visualize --
    # the exact bug already found (and fixed) for the sibling Fit workflow's
    # Export step (see tests/gui/workspace/fit/test_fit_controller.py::
    # test_export_step_unlocked_once_visualize_reached).
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win.set_mode("transform")
    ctl = win._controllers["transform"]
    assert ctl.reached == {0}  # nothing unlocked prematurely

    bodies = win.transform_bodies()
    bodies[2]._run_fn = lambda *a, **k: {"input": None, "output": None}

    bodies[0].set_transform("cox_merz")
    bodies[1].fill("oscillation", "ds1")
    bodies[1].fill("flow_curve", "ds2")
    bodies[2].run()

    assert ctl.current == 3  # landed on Visualize; not force-advanced past it
    assert 4 in ctl.reached  # Export unlocked because Visualize is trivially ready
    assert ctl.goto(4) is True
    assert ctl.current == 4


def test_picking_transform_rebuilds_slots_specs(qtbot):
    # Regression guard: bodies are all constructed eagerly at controller-build
    # time, before the user has picked a transform (transform_key is None).
    # SlotsStep._specs used to be computed once in __init__ from that None
    # key and never recomputed, so picking a typed-pair transform like
    # cox_merz left `_specs` frozen at the generic ["input"] fallback --
    # `candidates("oscillation")` would raise StopIteration.
    app = AppState()
    ctl, bodies = build_transform_controller(app)
    for b in bodies:
        qtbot.addWidget(b)

    assert bodies[1].slot_specs() == [SlotSpec("input", None, False)]

    bodies[0].set_transform("cox_merz")

    specs = bodies[1].slot_specs()
    assert [s.name for s in specs] == ["oscillation", "flow_curve"]
    # Must not raise StopIteration now that the typed slots exist.
    assert bodies[1].candidates("oscillation") == []
    assert bodies[1].candidates("flow_curve") == []
