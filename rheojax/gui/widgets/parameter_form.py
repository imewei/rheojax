"""Auto-generated parameter forms from spec dictionaries.

Given a dict of parameter specs (type, default, range, label, choices),
builds a QFormLayout with the appropriate widget per type and emits
values_changed on any modification.
"""

from __future__ import annotations

import math
from typing import Any

from rheojax.gui.compat import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
    Signal,
)
from rheojax.gui.resources.styles.tokens import Spacing, Typography
from rheojax.logging import get_logger

logger = get_logger(__name__)


class ParameterFormBuilder(QWidget):
    """Dynamically build a parameter form from spec dicts.

    Parameters
    ----------
    specs : dict[str, dict]
        Mapping of param_name -> spec. Each spec has:
        - type: "float" | "int" | "bool" | "choice"
        - default: default value
        - label: human-readable label
        - range: (min, max) for float/int (optional)
        - choices: list[str] for choice type
        - description: tooltip text (optional)
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    values_changed()
        Emitted when any parameter value changes.
    """

    values_changed = Signal()

    def __init__(
        self, specs: dict[str, dict[str, Any]], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._specs = specs
        self._widgets: dict[str, QWidget] = {}
        self._build_form()

    def _build_form(self) -> None:
        layout = QFormLayout(self)
        layout.setSpacing(Spacing.SM)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        for name, spec in self._specs.items():
            ptype = spec["type"]
            label = spec.get("label", name)
            tooltip = spec.get("description", "")

            if ptype == "float":
                widget = QDoubleSpinBox()
                widget.blockSignals(True)
                lo, hi = spec.get("range", (0.0, 1e6))
                widget.setRange(lo, hi)
                # Adaptive decimals: 4 for small ranges, 2 for large
                decimals = 4 if (hi - lo) < 10 else 2
                widget.setDecimals(decimals)
                # Step size: ~1% of range or based on magnitude
                step = 10 ** (math.floor(math.log10(max(abs(hi - lo), 1e-10))) - 2)
                widget.setSingleStep(step)
                widget.setValue(spec["default"])
                widget.blockSignals(False)
                widget.valueChanged.connect(self._on_change)
            elif ptype == "int":
                widget = QSpinBox()
                widget.blockSignals(True)
                lo, hi = spec.get("range", (0, 1000))
                widget.setRange(int(lo), int(hi))
                widget.setValue(int(spec["default"]))
                widget.blockSignals(False)
                widget.valueChanged.connect(self._on_change)
            elif ptype == "bool":
                widget = QCheckBox()
                widget.blockSignals(True)
                widget.setChecked(bool(spec["default"]))
                widget.blockSignals(False)
                widget.stateChanged.connect(self._on_change)
            elif ptype == "choice":
                widget = QComboBox()
                widget.blockSignals(True)
                widget.addItems(spec.get("choices", []))
                widget.setCurrentText(str(spec["default"]))
                widget.blockSignals(False)
                widget.currentTextChanged.connect(self._on_change)
            else:
                logger.warning("Unknown param type", param=name, type=ptype)
                continue

            widget.setToolTip(tooltip)
            widget.setStyleSheet(f"font-size: {Typography.SIZE_MD_SM}pt;")
            self._widgets[name] = widget
            layout.addRow(label + ":", widget)

    def _on_change(self, *_args: object) -> None:
        self.values_changed.emit()

    def get_values(self) -> dict[str, Any]:
        """Read current values from all widgets.

        Returns
        -------
        dict[str, Any]
            Mapping of param_name -> current value.
        """
        values: dict[str, Any] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
        return values


__all__ = ["ParameterFormBuilder"]
