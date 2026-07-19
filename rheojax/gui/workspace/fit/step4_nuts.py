from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.foundation.metrics import bfmi
from rheojax.gui.foundation.priors import map_centered_priors
from rheojax.gui.foundation.state import FitState
from rheojax.gui.jobs.cancellation import CancellationError
from rheojax.gui.utils.layout_helpers import set_panel_margins
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
        if r_hat is None:
            continue
        # `nan > threshold` is always False in Python, so a degenerate chain
        # (e.g. zero-variance) whose az.rhat() legitimately returns NaN would
        # otherwise silently read as "converged" instead of "unverifiable".
        if (isinstance(r_hat, float) and math.isnan(r_hat)) or r_hat > _R_HAT_THRESHOLD:
            reasons.append(f"r_hat too high for {name}")
    for name, ess in (nuts_result.get("ess") or {}).items():
        if ess is None:
            continue
        if (isinstance(ess, float) and math.isnan(ess)) or ess < _ESS_MIN:
            reasons.append(f"ESS too low for {name}")
    sample_stats = nuts_result.get("sample_stats") or {}
    # Prefer run_bayesian_isolated's own per-chain-averaged "bfmi" (correct);
    # only recompute from the flattened multi-chain energy array (biased --
    # mixes cross-chain energy-level offsets into one variance calc) when
    # that top-level key is absent, e.g. in standalone/unit-test callers.
    top_level_bfmi = nuts_result.get("bfmi")
    if top_level_bfmi is not None:
        if isinstance(top_level_bfmi, float) and math.isnan(top_level_bfmi):
            # subprocess_bayesian.run_bayesian_isolated() sets bfmi=nan when
            # the ArviZ InferenceData had no "energy" sample_stat to compute
            # it from -- that's a genuinely unverifiable diagnostic, not a
            # passing one; `nan < _BFMI_MIN` is always False in Python, which
            # would otherwise silently report "converged".
            reasons.append("BFMI unavailable (energy stat missing)")
        elif top_level_bfmi < _BFMI_MIN:
            reasons.append("BFMI too low")
    else:
        energy = sample_stats.get("energy")
        if energy is not None and len(energy) > 0:
            b = bfmi(energy)
            if b < _BFMI_MIN:
                reasons.append("BFMI too low")
    # Prefer run_bayesian_isolated's own top-level "divergences" count --
    # it comes straight from mcmc.get_extra_fields() (BayesianService's
    # get_diagnostics()), independent of ArviZ InferenceData conversion.
    # sample_stats["diverging"] only exists when result.to_inference_data()
    # succeeded (bayesian_service.py wraps that call in a try/except with a
    # None fallback); when it fails, sample_stats comes back {} and the
    # fallback path below would wrongly report zero divergences even though
    # the authoritative count is sitting unused right here.
    top_level_divergences = nuts_result.get("divergences")
    if top_level_divergences is not None:
        n_divergences = int(top_level_divergences)
    else:
        diverging = sample_stats.get("diverging")
        # diverging may be a flat list (test fixtures), or a real (num_chains,
        # num_samples) NumPy array from ArviZ sample_stats -- `sum(1 for d in
        # ... if d)` would iterate chain rows and raise ValueError on the
        # multi-element truthiness check. np.asarray(...).sum() counts True
        # entries correctly regardless of dimensionality.
        n_divergences = int(np.asarray(diverging).sum()) if diverging is not None else 0
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
        active_jobs=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._sample_fn = sample_fn or _default_sample_fn
        self._active_jobs = active_jobs
        self._skipped = False
        self._banner = QLabel("⚡ warm-started from NLSQ MAP", self)
        self._banner.setAccessibleName("Warm-started from NLSQ MAP")
        self._priors_editor = PriorsEditor(self)

        # Sampler settings (previously hardcoded in fit_controller.py's
        # _make_sample_fn with no UI to change them). Initial values and
        # live edits both go through state.nuts_config -- previously a
        # fully-unused FitState field with its own copy of these same
        # defaults (num_warmup=500/num_samples=1000/num_chains=4/seed=0/
        # target_accept=0.8) that nothing ever read or wrote.
        nuts_cfg = self._state.nuts_config
        self._warmup = QSpinBox(self)
        self._warmup.setRange(50, 100_000)
        self._warmup.setValue(nuts_cfg.num_warmup)
        self._samples = QSpinBox(self)
        self._samples.setRange(50, 100_000)
        self._samples.setValue(nuts_cfg.num_samples)
        self._chains = QSpinBox(self)
        self._chains.setRange(1, 16)
        self._chains.setValue(nuts_cfg.num_chains)
        self._seed = QSpinBox(self)
        self._seed.setRange(0, 2**31 - 1)
        self._seed.setValue(nuts_cfg.seed)
        self._target = QDoubleSpinBox(self)
        self._target.setRange(0.5, 0.999)
        self._target.setValue(nuts_cfg.target_accept)
        self._max_tree_depth = QSpinBox(self)
        # 0 means "unset" -- library default (10) applies; see run()/_make_sample_fn.
        self._max_tree_depth.setRange(0, 20)
        self._max_tree_depth.setValue(nuts_cfg.max_tree_depth or 0)
        self._max_tree_depth.setSpecialValueText("default")
        settings_form = QFormLayout()
        settings_form.addRow("Warmup:", self._warmup)
        settings_form.addRow("Samples:", self._samples)
        settings_form.addRow("Chains:", self._chains)
        settings_form.addRow("Seed:", self._seed)
        settings_form.addRow("Target accept:", self._target)
        settings_form.addRow("Max tree depth:", self._max_tree_depth)

        self._skip_btn = QPushButton("Skip NUTS", self)
        self._run_btn = QPushButton("▶ Sample", self)
        self._run_btn.setAccessibleName("Sample")
        # Only path to stop a running NUTS sample used to be Close/New/Open's
        # heavyweight "Jobs Running -- Cancel them and continue?" dialog.
        # This reuses the same active_jobs "worker" token + CancelWorkerRunnable
        # that dialog already dispatches, just from a direct per-step button.
        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._result = QLabel("", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        lay.addWidget(self._banner)
        lay.addWidget(self._priors_editor)
        lay.addLayout(settings_form)
        for w in (self._skip_btn, self._run_btn, self._cancel_btn, self._result):
            lay.addWidget(w)
        lay.addStretch()  # see step1_protocol_model.py's addStretch() comment
        self._skip_btn.clicked.connect(self.skip)
        self._run_btn.clicked.connect(self.run)
        for spin in (
            self._warmup,
            self._samples,
            self._chains,
            self._seed,
            self._max_tree_depth,
        ):
            spin.valueChanged.connect(self._on_settings_changed)
        self._target.valueChanged.connect(self._on_settings_changed)

    def _on_settings_changed(self, *_args: object) -> None:
        cfg = self._state.nuts_config
        cfg.num_warmup = self._warmup.value()
        cfg.num_samples = self._samples.value()
        cfg.num_chains = self._chains.value()
        cfg.seed = self._seed.value()
        cfg.target_accept = self._target.value()
        cfg.max_tree_depth = self._max_tree_depth.value() or None

    def _on_cancel_clicked(self) -> None:
        if self._active_jobs is None:
            return
        job_id = f"{self._state.data_ref}:nuts"
        job = self._active_jobs.by_id.get(job_id)
        worker = job.get("worker") if job else None
        if worker is None:
            return
        from PySide6.QtCore import QThreadPool

        from rheojax.gui.workspace.pipeline.cancel_runnable import (
            CancelWorkerRunnable,
        )

        QThreadPool.globalInstance().start(CancelWorkerRunnable(worker))

    def priors_editor(self) -> PriorsEditor:
        return self._priors_editor

    def suggested_priors(self) -> dict:
        # nlsq_result is a normalized dict (see NlsqStep.run()); key is "params"
        params = (self._state.nlsq_result or {}).get("params", {})
        return map_centered_priors(
            params
        )  # safe: FitResult was normalized to dict in step 3

    def load_suggested_priors(self) -> None:
        """Seed the editable PriorsEditor from the NLSQ MAP estimate."""
        self._priors_editor.load_numpyro_priors(self.suggested_priors())

    def set_target_accept(self, v: float) -> None:
        self._target.setValue(v)

    def skip(self) -> None:
        if not self._run_btn.isEnabled():
            # A run is in flight (see run()'s busy-guard) -- skipping now
            # would set nuts_result=None and emit finished immediately, then
            # the still-running sample_fn would later resume and overwrite
            # nuts_result with the real result plus a second finished
            # emission, silently un-skipping what the user believed was
            # skipped.
            return
        self._skipped = True
        self._state.nuts_result = None
        self.edited.emit()
        self.finished.emit()

    def run(self) -> None:
        self._skipped = False
        cfg = {
            "target_accept": self._target.value(),
            "num_warmup": self._warmup.value(),
            "num_samples": self._samples.value(),
            "num_chains": self._chains.value(),
            "seed": self._seed.value(),
            "max_tree_depth": self._max_tree_depth.value() or None,
        }
        warm_start = (self._state.nlsq_result or {}).get("params", {})
        # Build priors from the (possibly user-edited) PriorsEditor; if the editor
        # was never seeded via load_suggested_priors(), it's empty and we fall back
        # to the raw suggested_priors(), matching pre-Task-7 behavior exactly.
        priors = self._priors_editor.to_numpyro_priors() or self.suggested_priors()
        # Guard against re-entrant clicks: _sample_fn pumps a nested event
        # loop (see fit_controller._run_on_thread) while staying responsive,
        # so both buttons must be disabled for the duration -- otherwise a
        # second Sample click (or a Skip click) mid-run collides with this
        # run on the same active_jobs tracking entry and/or silently
        # overwrites/un-skips nuts_result once one of the two calls
        # completes. Mirrors NlsqStep.run()'s identical guard.
        self._run_btn.setEnabled(False)
        self._skip_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        # See NlsqStep.run()'s matching comment: discard a result that no
        # longer corresponds to the current selection if state moved on
        # while this run was in flight.
        revision_at_start = self._state.revision
        try:
            try:
                result = self._sample_fn(priors, warm_start, cfg)
            except NotImplementedError:
                # ponytail: real sampler wiring is out of scope here (tracked separately);
                # this guard only keeps an unwired Sample button from crashing the Qt slot.
                self._result.setText("NUTS sampler is not wired up yet.")
                if self._state.revision == revision_at_start:
                    self._state.nuts_result = None
                    self.edited.emit()
                return
            except CancellationError:
                # A user-initiated cancel is not a sampling failure -- show
                # a clean cancelled state and leave a stale successful
                # nuts_result untouched (mirrors NlsqStep.run()'s identical
                # branch).
                self._result.setText("Cancelled.")
                return
            except Exception as exc:
                self._result.setText(f"NUTS failed: {exc}")
                # A prior successful nuts_result must not survive a failed
                # re-run -- is_ready() reads nuts_result live, so leaving
                # the old result in place would misrepresent this run as
                # having succeeded.
                if self._state.revision == revision_at_start:
                    self._state.nuts_result = None
                    self.edited.emit()
                return
            if self._state.revision != revision_at_start:
                return
            verdict = _diagnostics_verdict(result)
            result["verdict"] = verdict
            self._state.nuts_result = result
            status = (
                "✓ converged"
                if verdict["converged"]
                else "⚠ " + "; ".join(verdict["reasons"])
            )
            self._result.setText(status)
            self.edited.emit()
            self.finished.emit()
        finally:
            self._run_btn.setEnabled(True)
            self._skip_btn.setEnabled(True)
            self._cancel_btn.setVisible(False)

    def is_ready(self) -> bool:
        return self._skipped or self._state.nuts_result is not None

    def reset_skip(self) -> None:
        """Clear a stale skip decision when an upstream NLSQ re-run invalidates
        nuts_result (see _FIT_CASCADE["nlsq_result"] in invalidation.py)."""
        self._skipped = False
        self._result.setText("")
