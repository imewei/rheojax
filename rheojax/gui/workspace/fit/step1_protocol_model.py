from __future__ import annotations

import inspect
import typing
from typing import Any, Literal, get_args, get_origin

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheojax.core.registry import ModelRegistry
from rheojax.gui.foundation.state import FitState
from rheojax.gui.pages.fit_page import _FAMILY_LABELS
from rheojax.gui.resources.styles.tokens import field_label_style
from rheojax.gui.services.model_service import ModelService
from rheojax.gui.utils.layout_helpers import set_panel_margins

_PROTOCOLS = ["flow_curve", "creep", "relaxation", "startup", "oscillation", "laos"]


def _grouped_models(protocol: str) -> dict[str, list[str]]:
    """Models compatible with `protocol`, grouped by family (same grouping
    fit_page.py's model selector already uses) instead of one flat list.

    ModelRegistry.find(protocol=...) and ModelService.get_available_models()
    are two independent registry queries -- a name present in the former but
    absent from the latter's categorization (e.g. a name it doesn't
    recognize) must still surface, under "other", rather than silently
    disappear from the picker.
    """
    allowed = set(ModelRegistry.find(protocol=protocol))
    grouped = {
        family: [name for name in names if name in allowed]
        for family, names in ModelService().get_available_models().items()
    }
    categorized = {name for names in grouped.values() for name in names}
    leftover = sorted(allowed - categorized)
    if leftover:
        grouped.setdefault("other", []).extend(leftover)
    return grouped


def _constructor_params(model_key: str) -> list[tuple[str, Any, Any]]:
    """Introspect a model class's __init__ for configurable constructor kwargs.

    Returns (name, annotation, default) tuples, skipping `self` and any
    parameter without a default (required positional args are data, not
    config, and are never asked for here).
    """
    cls = ModelRegistry.get_info(model_key).plugin_class
    sig = inspect.signature(cls.__init__)
    # Model modules use `from __future__ import annotations`, so
    # param.annotation is a plain string at runtime; resolve it to the real
    # type via get_type_hints (falls back to the unresolved param.annotation
    # if resolution fails, e.g. a missing import in the model's module).
    try:
        hints = typing.get_type_hints(cls.__init__)
    except Exception:
        hints = {}
    out: list[tuple[str, Any, Any]] = []
    for name, param in sig.parameters.items():
        if name == "self" or param.default is inspect.Parameter.empty:
            continue
        out.append((name, hints.get(name, param.annotation), param.default))
    return out


def _make_config_widget(annotation: Any, default: Any):
    """Return (widget, getter) for a supported annotation, or (None, None)."""
    if annotation is bool:
        w = QCheckBox()
        w.setChecked(bool(default))
        return w, w.isChecked
    if get_origin(annotation) is Literal:
        w = QComboBox()
        w.addItems([str(a) for a in get_args(annotation)])
        w.setCurrentText(str(default))
        return w, w.currentText
    if annotation is int:
        w = QSpinBox()
        w.setRange(-1_000_000, 1_000_000)
        w.setValue(int(default))
        return w, w.value
    if annotation is float:
        w = QDoubleSpinBox()
        w.setRange(-1_000_000.0, 1_000_000.0)
        w.setDecimals(6)
        w.setValue(float(default))
        return w, w.value
    return None, None


class ProtocolModelStep(QWidget):
    edited = Signal()
    # Fired only for constructor-config widget edits (model/protocol
    # unchanged). Kept separate from `edited` so fit_controller.py can
    # invalidate downstream with the narrower "model_config" cascade key
    # instead of "model_key" -- the latter clears model_config itself,
    # which would wipe out the very edit this signal reports.
    config_edited = Signal()

    def __init__(self, state: FitState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Model registration is lazy; ensure it happened before find()/create() are used.
        from rheojax.models import _ensure_all_registered

        _ensure_all_registered()
        self._state = state
        self._config_widgets: dict[str, tuple[QWidget, Any]] = {}
        self._protocol = QComboBox(self)
        self._protocol.addItems([""] + _PROTOCOLS)
        self._model = QComboBox(self)
        self._config_form = QFormLayout()
        self._params = QLabel("", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        protocol_caption = QLabel("Protocol")
        protocol_caption.setStyleSheet(field_label_style())
        model_caption = QLabel("Model")
        model_caption.setStyleSheet(field_label_style())
        for w in (
            protocol_caption,
            self._protocol,
            model_caption,
            self._model,
        ):
            lay.addWidget(w)
        lay.addLayout(self._config_form)
        params_caption = QLabel("Parameters")
        params_caption.setStyleSheet(field_label_style())
        lay.addWidget(params_caption)
        lay.addWidget(self._params)
        self._protocol.currentTextChanged.connect(self._on_protocol)
        self._model.currentIndexChanged.connect(
            lambda _idx: self._on_model(self._model.currentData())
        )

    def set_protocol(self, p: str) -> None:
        self._protocol.setCurrentText(p)

    def _on_protocol(self, p: str) -> None:
        self._state.protocol = p or None
        # Block signals across clear()+addItems(): if a model was already
        # selected, clear() alone would synchronously fire
        # currentTextChanged("") -> _on_model("") -> an `edited` emission,
        # then this method would emit a second time below -- double-running
        # fit_controller.py's invalidation cascade for one user action.
        # Instead, resync explicitly via a single _on_model() call after
        # repopulating, so exactly one `edited` emission covers both the
        # protocol change and the resulting model reset.
        self._model.blockSignals(True)
        self._model.clear()
        self._model.addItem("", None)
        if p:
            for family, names in _grouped_models(p).items():
                if not names:
                    continue
                label = _FAMILY_LABELS.get(family, family.replace("_", " ").title())
                self._model.addItem(f"── {label} ──", None)
                header_idx = self._model.count() - 1
                item_model = self._model.model()
                if item_model is not None:
                    item = item_model.item(header_idx)
                    if item is not None:
                        item.setEnabled(False)
                for name in names:
                    self._model.addItem(f"  {name}", name)
        self._model.blockSignals(False)
        self._on_model(self._model.currentData())

    def model_keys(self) -> list[str]:
        return [
            self._model.itemData(i)
            for i in range(self._model.count())
            if self._model.itemData(i)
        ]

    def set_model(self, key: str) -> None:
        idx = self._model.findData(key)
        if idx >= 0:
            self._model.setCurrentIndex(idx)

    def _on_model(self, key: str | None) -> None:
        self._state.model_key = key or None
        self._state.model_config = {}
        self._rebuild_config_widgets(key)
        self._refresh_preview()
        self.edited.emit()

    def _rebuild_config_widgets(self, key: str) -> None:
        while self._config_form.rowCount():
            self._config_form.removeRow(0)
        self._config_widgets.clear()
        if not key:
            return
        for name, annotation, default in _constructor_params(key):
            widget, getter = _make_config_widget(annotation, default)
            if widget is None:
                continue  # unsupported annotation type — stays at model default
            self._config_widgets[name] = (widget, getter)
            self._config_form.addRow(name, widget)
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_config_changed)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._on_config_changed)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self._on_config_changed)

    def _on_config_changed(self, *_args: Any) -> None:
        self._state.model_config = {
            name: getter() for name, (_w, getter) in self._config_widgets.items()
        }
        self._refresh_preview()
        self.config_edited.emit()

    def set_model_config(self, config: dict[str, Any]) -> None:
        """Test/programmatic helper: apply a config dict to the widgets, then commit it."""
        for name, value in config.items():
            entry = self._config_widgets.get(name)
            if entry is None:
                continue
            widget, _getter = entry
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
        self._on_config_changed()

    def _refresh_preview(self) -> None:
        if not self._state.model_key:
            self._params.setText("")
            return
        instance = ModelRegistry.create(
            self._state.model_key, **self._state.model_config
        )
        self._params.setText(", ".join(instance.parameters.keys()))

    def is_ready(self) -> bool:
        return bool(self._state.protocol and self._state.model_key)
