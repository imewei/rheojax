from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
import rheojax.transforms  # noqa: F401
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.transform.transform_controller import (
    build_transform_controller,
)


def test_pick_step_validate_requires_transform(qtbot):
    app = AppState()
    ctl, bodies = build_transform_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    assert ctl.steps[0].validate() is False
    bodies[0].set_transform("derivative")
    assert ctl.steps[0].validate() is True


def test_slots_step_validate_requires_all_slots_filled(qtbot):
    app = AppState()
    ctl, bodies = build_transform_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    bodies[0].set_transform("cox_merz")
    assert ctl.steps[1].validate() is False
    bodies[1].fill("oscillation", "o1")
    assert ctl.steps[1].validate() is False
    bodies[1].fill("flow_curve", "f1")
    assert ctl.steps[1].validate() is True


def test_run_step_validate_requires_result(qtbot):
    app = AppState()
    ctl, bodies = build_transform_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    assert ctl.steps[2].validate() is False
    bodies[2]._run_fn = lambda *a, **k: {"output": None, "result": {}}
    bodies[2].run()
    assert ctl.steps[2].validate() is True
