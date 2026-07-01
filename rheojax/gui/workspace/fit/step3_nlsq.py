from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.foundation.state import FitState


def _default_fit_fn(model_key, model_config, data_ref, column_map):
    # NOTE: `data_ref` is a str id — the real implementation must:
    # 1. Load RheoData: `library.load_payload(data_ref)` (pass library to NlsqStep at build time)
    # 2. Use real signature: ModelService().fit(model_key, rheodata, {}, model_config=model_config)
    # 3. Run via subprocess worker (non-blocking) — see build_fit_controller in Task 7.
    # Tests inject fake_fit instead of this default; wire the real default in build_fit_controller.
    raise NotImplementedError(
        "Wire _default_fit_fn via build_fit_controller; inject fake for tests"
    )


class NlsqStep(QWidget):
    edited = Signal()
    finished = Signal()

    def __init__(
        self,
        state: FitState,
        fit_fn: Callable | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._fit_fn = fit_fn or _default_fit_fn
        self._ms_enabled = QCheckBox("multi-start", self)
        self._ms_count = QSpinBox(self)
        self._ms_count.setRange(1, 64)
        self._ms_count.setValue(8)
        self._run_btn = QPushButton("▶ Run NLSQ", self)
        self._result = QLabel("", self)
        lay = QVBoxLayout(self)
        for w in (self._ms_enabled, self._ms_count, self._run_btn, self._result):
            lay.addWidget(w)
        self._run_btn.clicked.connect(self.run)

    def set_multistart(self, enabled: bool, count: int) -> None:
        self._ms_enabled.setChecked(enabled)
        self._ms_count.setValue(count)

    def run(self) -> None:
        # Multi-start config is transient widget state — not stored in FitState
        res = self._fit_fn(
            self._state.model_key,
            self._state.model_config,
            self._state.data_ref,
            self._state.column_map,
        )
        # Normalize to dict: ModelService.fit() returns FitResult (dataclass), fakes return dict
        if isinstance(res, dict):
            self._state.nlsq_result = res
        else:
            self._state.nlsq_result = {
                "params": res.parameters,
                "r_squared": res.r_squared,
                "success": getattr(res, "success", True),
            }
        r2 = self._state.nlsq_result.get("r_squared", float("nan"))
        self._result.setText(f"R²={r2:.3f}")
        self.edited.emit()
        self.finished.emit()

    def is_ready(self) -> bool:
        return self._state.nlsq_result is not None
