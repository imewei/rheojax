from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.foundation.priors import adapt_prior, map_centered_priors
from rheojax.gui.foundation.state import FitState
from rheojax.gui.widgets.priors_editor import PriorsEditor


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
        self._priors_editor = PriorsEditor(self)
        self._target = QDoubleSpinBox(self)
        self._target.setRange(0.5, 0.999)
        self._target.setValue(0.8)
        self._skip_btn = QPushButton("Skip NUTS", self)
        self._run_btn = QPushButton("▶ Sample", self)
        self._result = QLabel("", self)
        lay = QVBoxLayout(self)
        for w in (
            self._banner,
            self._priors_editor,
            self._target,
            self._skip_btn,
            self._run_btn,
            self._result,
        ):
            lay.addWidget(w)
        self._skip_btn.clicked.connect(self.skip)
        self._run_btn.clicked.connect(self.run)

    def priors_editor(self) -> PriorsEditor:
        return self._priors_editor

    def suggested_priors(self) -> dict:
        # nlsq_result is a normalized dict (see NlsqStep.run()); key is "params"
        params = (self._state.nlsq_result or {}).get("params", {})
        return map_centered_priors(params)  # safe: FitResult was normalized to dict in step 3

    def load_suggested_priors(self) -> None:
        """Seed the editable PriorsEditor from the NLSQ MAP estimate.

        Converts map_centered_priors()'s {"type": ..., ...} shape into
        PriorsEditor.set_prior(name, dist, **params) calls.
        """
        suggested = self.suggested_priors()
        self._priors_editor.set_parameters(list(suggested.keys()))
        for name, prior in suggested.items():
            dist = prior.get("type", "normal")
            params = {k: v for k, v in prior.items() if k != "type"}
            self._priors_editor.set_prior(name, dist, **params)

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
        # Build priors from the (possibly user-edited) PriorsEditor; if the editor
        # was never seeded via load_suggested_priors(), it's empty and we fall back
        # to the raw suggested_priors(), matching pre-Task-7 behavior exactly.
        edited_priors = self._priors_editor.get_all_priors()
        priors = {
            name: adapt_prior(entry) for name, entry in edited_priors.items()
        } or self.suggested_priors()
        try:
            result = self._sample_fn(priors, warm_start, cfg)
        except NotImplementedError:
            # ponytail: real sampler wiring is out of scope here (tracked separately);
            # this guard only keeps an unwired Sample button from crashing the Qt slot.
            self._result.setText("NUTS sampler is not wired up yet.")
            return
        self._state.nuts_result = result
        self.edited.emit()
        self.finished.emit()

    def is_ready(self) -> bool:
        return self._skipped or self._state.nuts_result is not None

    def reset_skip(self) -> None:
        """Clear a stale skip decision when an upstream NLSQ re-run invalidates
        nuts_result (see _FIT_CASCADE["nlsq_result"] in invalidation.py)."""
        self._skipped = False
