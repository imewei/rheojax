from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.foundation.priors import map_centered_priors
from rheojax.gui.foundation.state import FitState


def _default_sample_fn(priors, init, config):
    # NOTE: The real implementation must:
    # 1. Load RheoData from the library (pass library to NutsStep at build time)
    # 2. Construct BayesianWorker(model_name, data, priors=priors, warm_start=init,
    #    target_accept=config.get("target_accept", 0.8), ...) — see bayesian_worker.py
    # 3. Run via a QThread, not synchronously on the GUI thread.
    # Tests inject a fake lambda instead; wire the real default in build_fit_controller.
    raise NotImplementedError(
        "Wire _default_sample_fn via build_fit_controller; inject fake for tests"
    )


class NutsStep(QWidget):
    edited = Signal()
    finished = Signal()

    def __init__(
        self,
        state: FitState,
        sample_fn: Callable | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._sample_fn = sample_fn or _default_sample_fn
        self._skipped = False
        self._banner = QLabel("⚡ warm-started from NLSQ MAP", self)
        self._target = QDoubleSpinBox(self)
        self._target.setRange(0.5, 0.999)
        self._target.setValue(0.8)
        self._skip_btn = QPushButton("Skip NUTS", self)
        self._run_btn = QPushButton("▶ Sample", self)
        lay = QVBoxLayout(self)
        for w in (self._banner, self._target, self._skip_btn, self._run_btn):
            lay.addWidget(w)
        self._skip_btn.clicked.connect(self.skip)
        self._run_btn.clicked.connect(self.run)

    def suggested_priors(self) -> dict:
        # nlsq_result is a normalized dict (see NlsqStep.run()); key is "params"
        params = (self._state.nlsq_result or {}).get("params", {})
        return map_centered_priors(params)  # safe: FitResult was normalized to dict in step 3

    def set_target_accept(self, v: float) -> None:
        self._target.setValue(v)

    def skip(self) -> None:
        self._skipped = True
        self._state.nuts_result = None
        self.edited.emit()
        self.finished.emit()

    def run(self) -> None:
        self._skipped = False
        cfg = {"target_accept": self._target.value()}
        warm_start = (self._state.nlsq_result or {}).get("params", {})
        self._state.nuts_result = self._sample_fn(
            self.suggested_priors(), warm_start, cfg
        )
        self.edited.emit()
        self.finished.emit()

    def is_ready(self) -> bool:
        return self._skipped or self._state.nuts_result is not None
