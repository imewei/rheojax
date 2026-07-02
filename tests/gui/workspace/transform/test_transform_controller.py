from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
import rheojax.transforms  # noqa
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.widgets.pyqtgraph_canvas import PyQtGraphCanvas
from rheojax.gui.workspace.transform.slots_spec import SlotSpec
from rheojax.gui.workspace.transform.transform_controller import (
    _infer_protocol_type,
    build_transform_controller,
)
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


def test_editing_slots_invalidates_stale_result(qtbot):
    # Regression: refilling a slot (step 1) with a different dataset must
    # invalidate a previously-computed `result`. Before the fix, only idx==0
    # (re-picking the transform) cleared `result`; refilling a slot only
    # called ctl.on_edit(1), which re-locks `reached` to {0, 1} but leaves
    # `state.result` untouched. RunStep.is_ready() checks `result is not
    # None`, so it stayed True, and the window's `_advance_and_unlock`
    # forward-unlock loop (wired to fire right after `_on_edit`) immediately
    # re-added 2/3/4 to `reached`, undoing the re-lock -- leaving stale
    # output reachable/exportable for a dataset that was never actually run.
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    win.set_mode("transform")
    ctl = win._controllers["transform"]

    bodies = win.transform_bodies()
    bodies[2]._run_fn = lambda *a, **k: {"input": None, "output": None}

    bodies[0].set_transform("cox_merz")
    bodies[1].fill("oscillation", "ds1")
    bodies[1].fill("flow_curve", "ds2")
    bodies[2].run()

    assert win._state.transform.result is not None
    assert 4 in ctl.reached  # Export reachable, mirrors the sibling test above

    bodies[1].fill("oscillation", "ds1-different")  # refill with a new dataset

    assert win._state.transform.result is None
    assert bodies[2].is_ready() is False
    assert 4 not in ctl.reached
    assert ctl.goto(4) is False  # Export no longer trivially reachable


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


def test_run_finish_refreshes_visualize_step(qtbot):
    # Regression guard: TransformVisualizeStep's tabs/view_mode are computed
    # from transform_key/result, both unset (None) at controller-build time
    # (bodies are constructed eagerly before the user drives the workflow).
    # Nothing rebuilt the widget after RunStep produced a result, so it
    # stayed frozen in its construction-time "overlay, no data" shape
    # forever -- the same bug class already fixed for the Fit Workflow's
    # VisualizeStep and this plan's own SlotsStep.
    app = AppState()
    ctl, bodies = build_transform_controller(app)
    for b in bodies:
        qtbot.addWidget(b)

    pick_body, slots_body, run_body, visualize_body = (
        bodies[0],
        bodies[1],
        bodies[2],
        bodies[3],
    )

    # Constructed against transform_key=None -> defaults to overlay mode.
    assert visualize_body.view_mode() == "overlay"
    assert visualize_body.tab_names()[0] == "Input vs output"

    pick_body.set_transform("fft_analysis")  # domain-changing -> "separate"
    slots_body.fill("input", "ds1")

    run_body._run_fn = lambda k, s, c: {
        "output": {"x": [0.1, 1.0], "y": [1.0, 2.0]},
        "result": {"n_peaks": 3},
    }
    with qtbot.waitSignal(run_body.finished, timeout=2000):
        run_body.run()

    assert visualize_body.view_mode() == "separate"
    assert visualize_body.tab_names()[0] == "Output"
    primary = visualize_body._tabs.widget(0)
    assert len(primary.findChildren(PyQtGraphCanvas)) == 2
    result_tab = visualize_body._tabs.widget(visualize_body._tabs.count() - 1)
    assert "n_peaks" in result_tab.text()


def test_infer_protocol_type_returns_empty_string_not_none_for_domain_changing():
    # Regression: domain-changing transforms (spectral/decomposition) must
    # still return a real `str` ("", not None) so save_to_library() can tell
    # "genuinely unresolvable" apart from "known but typeless" -- per design
    # §7, such outputs are "stored but not offered to typed Fit slots", not
    # silently dropped from the library entirely.
    lib = DatasetLibrary()
    lib.add(
        DatasetRef(
            id="rel1", name="rel1", protocol_type="relaxation", origin="imported",
            units={}, row_count=3, hash="h", provenance={}, lineage=[],
        )
    )
    ptype = _infer_protocol_type(lib, "fft_analysis", {"input": "rel1"})
    assert ptype == ""
    assert ptype is not None


def test_infer_protocol_type_same_domain_still_resolves_real_type():
    lib = DatasetLibrary()
    lib.add(
        DatasetRef(
            id="osc1", name="osc1", protocol_type="oscillation", origin="imported",
            units={}, row_count=3, hash="h", provenance={}, lineage=[],
        )
    )
    ptype = _infer_protocol_type(lib, "smooth_derivative", {"input": "osc1"})
    assert ptype == "oscillation"
