import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.workspace.library_rail import LibraryRail


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


from PySide6.QtWidgets import QLabel
from rheojax.gui.workspace.inspector import InspectorPanel


def test_inspector_tabs(qtbot):
    ins = InspectorPanel(); qtbot.addWidget(ins)
    assert ins.tab_names() == ["params", "priors", "log"]
    ins.set_tab_widget("log", QLabel("hi"))
    ins.show_tab("log")
    assert ins.current_tab() == "log"
