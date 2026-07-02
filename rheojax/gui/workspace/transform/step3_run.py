from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.foundation.state import TransformState


def _default_run_fn(transform_key, slots, config):
    # NOTE: TransformService has no .run() method. The real implementation must:
    # 1. Load RheoData payloads from the library for each slot id
    # 2. Use TransformService().apply_transform(transform_key, data_or_list, params=config)
    # 3. For non-blocking: use TransformWorker instead of calling synchronously
    # Tests inject a fake run_fn; wire the real default in build_transform_controller.
    raise NotImplementedError(
        "Wire _default_run_fn via build_transform_controller; inject fake for tests"
    )


class RunStep(QWidget):
    edited = Signal()
    finished = Signal()

    def __init__(
        self,
        state: TransformState,
        run_fn: Callable | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._run_fn = run_fn or _default_run_fn
        self._btn = QPushButton("▶ Run transform", self)
        self._status = QLabel("", self)
        lay = QVBoxLayout(self)
        lay.addWidget(self._btn)
        lay.addWidget(self._status)
        self._btn.clicked.connect(self.run)

    def run(self) -> None:
        self._state.result = self._run_fn(
            self._state.transform_key, self._state.slots, self._state.config
        )
        self._status.setText("✓ done")
        self.edited.emit()
        self.finished.emit()

    def is_ready(self) -> bool:
        return self._state.result is not None
