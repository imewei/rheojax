from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.fit.fit_controller import build_fit_controller


def _ref(i, protocol):
    return DatasetRef(id=i, name=i, protocol_type=protocol, origin="imported",
                       units={}, row_count=2, hash="h", provenance={}, lineage=[])


class _RheoData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_data_step_validate_fails_on_invalid_data(qtbot):
    app = AppState()
    app.library.add(_ref("bad", "oscillation"))
    app.library.store_payload("bad", _RheoData([1, 3, 2], [1, 2, 3]))  # non-monotonic
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    bodies[0].set_protocol("oscillation")
    bodies[1].select_dataset("bad")
    assert ctl.steps[1].validate() is False


def test_data_step_validate_passes_on_valid_data(qtbot):
    app = AppState()
    app.library.add(_ref("good", "oscillation"))
    app.library.store_payload("good", _RheoData([1, 2, 3], [1, 2, 3]))
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    bodies[0].set_protocol("oscillation")
    bodies[1].select_dataset("good")
    assert ctl.steps[1].validate() is True


def test_nuts_step_validate_true_when_skipped(qtbot):
    app = AppState()
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    bodies[3].skip()
    assert ctl.steps[3].validate() is True


def test_protocol_model_step_validate_requires_model(qtbot):
    app = AppState()
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    assert ctl.steps[0].validate() is False
    bodies[0].set_protocol("flow_curve")
    bodies[0].set_model("power_law")
    assert ctl.steps[0].validate() is True


def test_nuts_step_validate_true_when_not_converged(qtbot):
    """A not-converged verdict must never hard-block validate() -- it flags,
    it does not gate. Only presence of a result (or an explicit skip) may
    gate NutsStep.is_ready()/validate()."""
    app = AppState()
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    bodies[3]._state.nuts_result = {
        "params": {"a": 1.0},
        "verdict": {"converged": False, "reasons": ["r_hat too high for a"]},
    }
    assert ctl.steps[3].validate() is True
