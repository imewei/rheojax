import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.workspace.library_rail import LibraryRail


def _flush_deleted_widgets() -> None:
    # deleteLater() posts a DeferredDelete event that plain wait()/processEvents()
    # don't drain — it must be requested explicitly.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


def _ref(i, t):
    return DatasetRef(
        id=i, name=i, protocol_type=t, origin="imported",
        units={}, row_count=1, hash="h", provenance={}, lineage=[],
    )


def test_library_rail_lists_and_emits(qtbot):
    lib = DatasetLibrary()
    lib.add(_ref("a", "oscillation"))
    lib.add(_ref("b", "creep"))
    rail = LibraryRail(lib)
    qtbot.addWidget(rail)
    rail.refresh()
    assert rail.count() == 2
    with qtbot.waitSignal(rail.dataset_selected, timeout=1000) as blocker:
        rail.select_row(0)
    assert blocker.args == ["a"]


def test_library_rail_select_row_empty_list_does_not_crash(qtbot):
    rail = LibraryRail(DatasetLibrary())  # no datasets added
    qtbot.addWidget(rail)
    rail.select_row(0)                    # must not raise AttributeError


from rheojax.gui.workspace.controller import Step, WorkflowController
from rheojax.gui.workspace.stepper_canvas import StepperCanvas


def test_stepper_rail_reflects_reached(qtbot):
    steps = [Step(str(i), f"S{i}", lambda: True, lambda: True) for i in range(3)]
    ctl = WorkflowController(steps); ctl.advance()       # reached {0,1}
    canvas = StepperCanvas(ctl); qtbot.addWidget(canvas); canvas.refresh()
    assert canvas.is_enabled(1) is True
    assert canvas.is_enabled(2) is False                 # unreached
    with qtbot.waitSignal(canvas.step_clicked, timeout=1000) as b:
        canvas.click_step(1)
    assert b.args == [1]


def test_stepper_active_step_is_highlighted(qtbot):
    steps = [Step(str(i), f"S{i}", lambda: True, lambda: True) for i in range(3)]
    ctl = WorkflowController(steps); ctl.advance()       # current == 1
    canvas = StepperCanvas(ctl); qtbot.addWidget(canvas); canvas.refresh()
    assert canvas.is_active(1) is True
    assert canvas.is_active(0) is False
    assert canvas.is_active(2) is False


def test_stepper_set_body_deletes_old_widget(qtbot):
    from PySide6.QtWidgets import QWidget

    steps = [Step("0", "S0", lambda: True, lambda: True)]
    ctl = WorkflowController(steps)
    canvas = StepperCanvas(ctl); qtbot.addWidget(canvas)
    old = canvas._stack.widget(0)          # the auto-created placeholder
    destroyed = []
    old.destroyed.connect(lambda: destroyed.append(True))
    canvas.set_body(0, QWidget())
    assert canvas._stack.indexOf(old) == -1  # removed from the stack synchronously
    _flush_deleted_widgets()
    assert destroyed == [True]               # ...and actually freed, not just hidden


from PySide6.QtWidgets import QLabel

from rheojax.gui.workspace.inspector import InspectorPanel


def test_inspector_tabs(qtbot):
    ins = InspectorPanel(); qtbot.addWidget(ins)
    assert ins.tab_names() == ["params", "priors", "log"]
    ins.set_tab_widget("log", QLabel("hi"))
    ins.show_tab("log")
    assert ins.current_tab() == "log"


def test_inspector_set_tab_widget_preserves_selection_and_deletes_old(qtbot):
    ins = InspectorPanel(); qtbot.addWidget(ins)
    ins.show_tab("params")
    old = ins._tabs.widget(ins._index["params"])
    destroyed = []
    old.destroyed.connect(lambda: destroyed.append(True))

    ins.set_tab_widget("params", QLabel("new params"))

    assert ins.current_tab() == "params"   # selection must not jump away
    _flush_deleted_widgets()
    assert destroyed == [True]             # old widget must not leak


from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def test_window_mode_switch(qtbot):
    win = WorkspaceWindow(AppState()); qtbot.addWidget(win)
    assert win.mode() == "fit"
    with qtbot.waitSignal(win.mode_changed, timeout=1000) as b:
        win.set_mode("transform")
    assert b.args == ["transform"] and win.mode() == "transform"
    assert win.active_step_count() == 5


def test_window_mode_buttons_reflect_active_mode(qtbot):
    win = WorkspaceWindow(AppState()); qtbot.addWidget(win)
    assert win._fit_btn.isChecked() is True and win._tx_btn.isChecked() is False
    win.set_mode("transform")
    assert win._fit_btn.isChecked() is False and win._tx_btn.isChecked() is True


def test_window_set_mode_rejects_unknown_mode(qtbot):
    win = WorkspaceWindow(AppState()); qtbot.addWidget(win)
    with pytest.raises(ValueError):
        win.set_mode("unknown")
    assert win.mode() == "fit"             # rejected — state unchanged


def test_window_invalid_persisted_mode_falls_back_to_fit(qtbot):
    state = AppState()
    state.ui["mode"] = "not-a-real-mode"
    win = WorkspaceWindow(state); qtbot.addWidget(win)
    assert win.mode() == "fit"             # invalid persisted state doesn't crash construction


def test_window_step_click_navigates(qtbot):
    # Clicking a reached step button must move controller.current AND the
    # displayed stack index — regression test for the click -> goto() wiring.
    # Uses transform mode: its steps are still trivially-ready skeleton stubs,
    # so this generic navigation-mechanics test doesn't need to satisfy the
    # real fit workflow's per-step readiness gating (protocol/model/data/...).
    win = WorkspaceWindow(AppState()); qtbot.addWidget(win)
    win.set_mode("transform")
    ctl = win._controllers[win.mode()]
    ctl.advance(); ctl.advance()          # reached {0,1,2}, current == 2
    win._canvas.refresh()
    assert win.current_step() == 2 and win._canvas.current_index() == 2

    with qtbot.waitSignal(win._canvas.step_clicked, timeout=1000) as b:
        win._canvas.click_step(0)
    assert b.args == [0]
    assert win.current_step() == 0        # controller navigated back
    assert win._canvas.current_index() == 0  # displayed body followed
