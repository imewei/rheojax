from __future__ import annotations
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel
from rheojax.core.registry import ModelRegistry
from rheojax.gui.foundation.state import FitState

_PROTOCOLS = ["flow_curve", "creep", "relaxation", "startup", "oscillation", "laos"]

class ProtocolModelStep(QWidget):
    edited = Signal()

    def __init__(self, state: FitState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._protocol = QComboBox(self); self._protocol.addItems([""] + _PROTOCOLS)
        self._model = QComboBox(self)
        self._params = QLabel("", self)
        lay = QVBoxLayout(self)
        for w in (QLabel("Protocol"), self._protocol, QLabel("Model"), self._model,
                  QLabel("Parameters"), self._params):
            lay.addWidget(w)
        self._protocol.currentTextChanged.connect(self._on_protocol)
        self._model.currentTextChanged.connect(self._on_model)

    def set_protocol(self, p: str) -> None:
        self._protocol.setCurrentText(p)

    def _on_protocol(self, p: str) -> None:
        self._state.protocol = p or None
        self._model.clear()
        if p:
            self._model.addItems([""] + sorted(ModelRegistry.find(protocol=p)))
        self.edited.emit()

    def model_keys(self) -> list[str]:
        return [self._model.itemText(i) for i in range(self._model.count()) if self._model.itemText(i)]

    def set_model(self, key: str) -> None:
        self._model.setCurrentText(key)

    def _on_model(self, key: str) -> None:
        self._state.model_key = key or None
        self._state.model_config = {}
        if key:
            params = list(ModelRegistry.create(key).parameters.keys())
            self._params.setText(", ".join(params))
        self.edited.emit()

    def is_ready(self) -> bool:
        return bool(self._state.protocol and self._state.model_key)
