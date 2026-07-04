from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from typing import Literal

from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step1_protocol_model import (
    ProtocolModelStep,
    _constructor_params,
)


class _FakeModel:
    def __init__(
        self,
        kinetics: Literal["stress", "stretch"] = "stress",
        include_damage: bool = False,
        gap_width: float = 1e-3,
    ) -> None:
        pass


class _FakeInfo:
    plugin_class = _FakeModel


class _FakeInstance:
    parameters = {"a": None, "b": None}


def test_constructor_params_introspection(monkeypatch):
    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.step1_protocol_model.ModelRegistry.get_info",
        lambda name: _FakeInfo(),
    )
    params = _constructor_params("fake_model")
    names = {p[0] for p in params}
    assert names == {"kinetics", "include_damage", "gap_width"}


def test_config_widgets_rebuild_and_apply(qtbot, monkeypatch):
    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.step1_protocol_model.ModelRegistry.get_info",
        lambda name: _FakeInfo(),
    )
    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.step1_protocol_model.ModelRegistry.create",
        lambda name, **kw: _FakeInstance(),
    )
    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.step1_protocol_model.ModelRegistry.find",
        lambda protocol=None, **kw: ["fake_model"],
    )
    st = FitState()
    step = ProtocolModelStep(st)
    qtbot.addWidget(step)
    step.set_protocol("flow_curve")
    step.set_model("fake_model")
    assert "kinetics" in step._config_widgets
    assert "include_damage" in step._config_widgets
    # Regression: float-typed constructor params (e.g.
    # NonLocalFluidityModel.gap_width) were previously skipped entirely by
    # _make_config_widget() ("stays at model default"), so they were never
    # exposed in the UI at all.
    assert "gap_width" in step._config_widgets
    from PySide6.QtWidgets import QDoubleSpinBox

    assert isinstance(step._config_widgets["gap_width"][0], QDoubleSpinBox)

    step.set_model_config(
        {"kinetics": "stretch", "include_damage": True, "gap_width": 2.5e-3}
    )
    assert st.model_config == {
        "kinetics": "stretch",
        "include_damage": True,
        "gap_width": 2.5e-3,
    }
