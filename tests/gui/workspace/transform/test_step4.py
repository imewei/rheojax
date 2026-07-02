import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel

from rheojax.gui.foundation.state import TransformState
from rheojax.gui.widgets.pyqtgraph_canvas import PyQtGraphCanvas
from rheojax.gui.workspace.transform.step4_visualize import TransformVisualizeStep


def test_view_mode_by_transform(qtbot):
    over = TransformVisualizeStep(
        TransformState(transform_key="cox_merz", result={"output": "rd"})
    )
    qtbot.addWidget(over)
    assert over.view_mode() == "overlay"
    sep = TransformVisualizeStep(
        TransformState(transform_key="fft_analysis", result={"output": "rd"})
    )
    qtbot.addWidget(sep)
    assert sep.view_mode() == "separate"  # spectral changes domain
    assert "Result" in sep.tab_names()


def test_view_mode_defaults_safely_with_no_transform_key(qtbot):
    step = TransformVisualizeStep(TransformState())
    qtbot.addWidget(step)
    assert step.view_mode() == "overlay"


def test_overlay_mode_uses_single_canvas(qtbot):
    st = TransformState(
        transform_key="cox_merz",
        result={
            "output": {"x": [0.1, 1.0, 10.0], "y": [1.0, 2.0, 3.0]},
            "result": {"deviation": 0.05, "status": "PASS"},
        },
    )
    step = TransformVisualizeStep(st)
    qtbot.addWidget(step)
    primary = step._tabs.widget(0)
    assert isinstance(primary, PyQtGraphCanvas)


def test_separate_mode_uses_two_canvases(qtbot):
    st = TransformState(
        transform_key="fft_analysis",
        result={
            "output": {"x": [0.1, 1.0], "y": [1.0, 2.0]},
            "result": {"n_peaks": 3},
        },
    )
    step = TransformVisualizeStep(st)
    qtbot.addWidget(step)
    primary = step._tabs.widget(0)
    assert len(primary.findChildren(PyQtGraphCanvas)) == 2


def test_result_tab_renders_result_dict_contents(qtbot):
    st = TransformState(
        transform_key="prony_conversion",
        result={
            "output": {"x": [0.1, 1.0], "y": [1.0, 2.0]},
            "result": {"G_i": [1.0, 2.0], "tau_i": [0.1, 1.0]},
        },
    )
    step = TransformVisualizeStep(st)
    qtbot.addWidget(step)
    result_tab = step._tabs.widget(step._tabs.count() - 1)
    assert isinstance(result_tab, QLabel)
    assert "G_i" in result_tab.text() and "tau_i" in result_tab.text()


def test_result_tab_handles_missing_result_without_crashing(qtbot):
    st = TransformState(transform_key="cox_merz", result=None)
    step = TransformVisualizeStep(st)
    qtbot.addWidget(step)
    assert step._tabs.count() == 2


def test_category_handles_unregistered_transform_key_without_crashing(qtbot, monkeypatch):
    # Regression: TransformRegistry.get_info() can return None for a key that
    # isn't registered (mirrors the guard in transform_controller.py's
    # _is_same_domain, added in the same task). Before the fix, _category()
    # called str(info.transform_type) unconditionally and raised AttributeError.
    from rheojax.core.registry import TransformRegistry

    monkeypatch.setattr(TransformRegistry, "get_info", classmethod(lambda cls, key: None))

    st = TransformState(transform_key="not_a_real_transform")
    step = TransformVisualizeStep(st)
    qtbot.addWidget(step)
    assert step._category() == "analysis"
    assert step.view_mode() == "overlay"
