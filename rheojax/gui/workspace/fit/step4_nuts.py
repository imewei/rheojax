from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.foundation.metrics import bfmi
from rheojax.gui.foundation.priors import adapt_prior, map_centered_priors
from rheojax.gui.foundation.state import FitState
from rheojax.gui.widgets.priors_editor import PriorsEditor

_R_HAT_THRESHOLD = 1.05
_ESS_MIN = 400
_BFMI_MIN = 0.3


def _diagnostics_verdict(nuts_result: dict) -> dict:
    """Summarize NUTS convergence diagnostics into a pass/fail verdict.

    Per the design (Step 4), this never blocks progress -- callers only use
    it to flag a warning badge; NutsStep.is_ready() ignores it entirely.
    """
    reasons: list[str] = []
    for name, r_hat in (nuts_result.get("r_hat") or {}).items():
        if r_hat is not None and r_hat > _R_HAT_THRESHOLD:
            reasons.append(f"r_hat too high for {name}")
    for name, ess in (nuts_result.get("ess") or {}).items():
        if ess is not None and ess < _ESS_MIN:
            reasons.append(f"ESS too low for {name}")
    sample_stats = nuts_result.get("sample_stats") or {}
    # Prefer run_bayesian_isolated's own per-chain-averaged "bfmi" (correct);
    # only recompute from the flattened multi-chain energy array (biased --
    # mixes cross-chain energy-level offsets into one variance calc) when
    # that top-level key is absent, e.g. in standalone/unit-test callers.
    top_level_bfmi = nuts_result.get("bfmi")
    if top_level_bfmi is not None:
        if top_level_bfmi < _BFMI_MIN:
            reasons.append("BFMI too low")
    else:
        energy = sample_stats.get("energy")
        if energy is not None and len(energy) > 0:
            b = bfmi(energy)
            if b < _BFMI_MIN:
                reasons.append("BFMI too low")
    diverging = sample_stats.get("diverging")
    diverging = [] if diverging is None else diverging
    n_divergences = sum(1 for d in diverging if d)
    if n_divergences:
        reasons.append(f"{n_divergences} divergent transitions")
    return {
        "converged": not reasons,
        "reasons": reasons,
    }


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
        result["verdict"] = _diagnostics_verdict(result)
        self._state.nuts_result = result
        self.edited.emit()
        self.finished.emit()

    def is_ready(self) -> bool:
        return self._skipped or self._state.nuts_result is not None

    def reset_skip(self) -> None:
        """Clear a stale skip decision when an upstream NLSQ re-run invalidates
        nuts_result (see _FIT_CASCADE["nlsq_result"] in invalidation.py)."""
        self._skipped = False
