import pytest

pytest.importorskip("PySide6")
import rheojax.models  # noqa
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
