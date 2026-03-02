"""Tests for ParameterFormBuilder widget."""

import os
import pytest

# Skip entirely if no display and no offscreen platform
if not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from rheojax.gui.widgets.parameter_form import ParameterFormBuilder


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_specs():
    return {
        "temperature": {
            "type": "float",
            "default": 25.0,
            "range": (-100.0, 300.0),
            "label": "Temperature",
            "description": "Reference temperature",
        },
        "auto_detect": {
            "type": "bool",
            "default": True,
            "label": "Auto-detect",
            "description": "Enable auto-detection",
        },
        "method": {
            "type": "choice",
            "default": "wlf",
            "choices": ["wlf", "arrhenius", "manual"],
            "label": "Method",
            "description": "Calculation method",
        },
        "order": {
            "type": "int",
            "default": 1,
            "range": (1, 4),
            "label": "Order",
            "description": "Derivative order",
        },
    }


def test_form_creates_from_specs(qapp, sample_specs):
    """ParameterFormBuilder creates widgets from param specs."""
    form = ParameterFormBuilder(sample_specs)
    values = form.get_values()
    assert values["temperature"] == 25.0
    assert values["auto_detect"] is True
    assert values["method"] == "wlf"
    assert values["order"] == 1


def test_form_handles_empty_specs(qapp):
    """Empty spec dict produces empty form."""
    form = ParameterFormBuilder({})
    assert form.get_values() == {}


def test_form_respects_ranges(qapp):
    """Float/int widgets enforce specified ranges."""
    specs = {
        "val": {
            "type": "float",
            "default": 5.0,
            "range": (0.0, 10.0),
            "label": "Value",
        }
    }
    form = ParameterFormBuilder(specs)
    widget = form._widgets["val"]
    assert widget.minimum() == 0.0
    assert widget.maximum() == 10.0


def test_form_values_changed_signal(qapp, sample_specs):
    """values_changed signal fires on parameter modification."""
    form = ParameterFormBuilder(sample_specs)
    changed = []
    form.values_changed.connect(lambda: changed.append(True))
    form._widgets["temperature"].setValue(50.0)
    assert len(changed) >= 1
    assert form.get_values()["temperature"] == 50.0


def test_form_with_real_transform_specs(qapp):
    """Form works with actual TransformService specs."""
    from rheojax.gui.services.transform_service import TransformService
    service = TransformService()
    for key in service.get_available_transforms():
        specs = service.get_transform_params(key)
        form = ParameterFormBuilder(specs)
        values = form.get_values()
        assert len(values) == len(specs), f"Widget count mismatch for {key}"
