from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheojax.core.registry import ModelRegistry
from rheojax.gui.foundation.state import FitState, ParameterState
from rheojax.gui.jobs.cancellation import CancellationError
from rheojax.gui.utils.layout_helpers import set_panel_margins
from rheojax.gui.widgets.parameter_table import ParameterTable
from rheojax.logging import get_logger

logger = get_logger(__name__)


def _default_fit_fn(
    model_key,
    model_config,
    data_ref,
    column_map,
    initial_params=None,
    multi_start=None,
    options=None,
):
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
    # Mirrors ProtocolModelStep's edited/config_edited split: `edited` means
    # nlsq_result itself changed (fit_controller.py registers it with
    # changed="nlsq_result", which cascades-clears nuts_result and relocks
    # downstream). Multi-start is a setting for the NEXT run, not a result
    # change -- emitting `edited` for it would silently wipe an existing
    # NUTS result every time the user toggles the checkbox. config_edited
    # only marks the project dirty (window.py's generic hasattr(body,
    # "config_edited") wiring) without touching the cascade.
    config_edited = Signal()
    finished = Signal()

    def __init__(
        self,
        state: FitState,
        fit_fn: Callable | None = None,
        active_jobs=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._fit_fn = fit_fn or _default_fit_fn
        self._active_jobs = active_jobs
        self._current_job_id: str | None = None
        self._table = ParameterTable(self)
        # Seeded from FitState.nlsq_config (mirrors NutsStep's nuts_config
        # pattern) rather than a hardcoded default -- previously this was
        # transient widget state only, so navigating away and back (or a
        # workspace rebuild) silently reset it with no warning.
        ms_cfg = self._state.nlsq_config
        self._ms_enabled = QCheckBox("multi-start", self)
        self._ms_enabled.setChecked(ms_cfg.multi_start)
        self._ms_enabled.setAccessibleDescription(
            "Run multiple randomized restarts and keep the best fit by R²."
        )
        self._ms_count = QSpinBox(self)
        self._ms_count.setRange(1, 64)
        self._ms_count.setValue(ms_cfg.n_starts)
        self._ms_count.setAccessibleName("Number of multi-start restarts")
        self._fit_options: dict = {}
        self._options_btn = QPushButton("⚙ Fit Options", self)
        self._options_btn.setAccessibleName("Fit Options")
        self._options_btn.clicked.connect(self._on_options_clicked)
        self._run_btn = QPushButton("▶ Run NLSQ", self)
        self._run_btn.setAccessibleName("Run NLSQ")
        # Only path to stop a running NLSQ fit used to be Close/New/Open's
        # heavyweight "Jobs Running -- Cancel them and continue?" dialog.
        # This reuses the same active_jobs "worker" token + CancelWorkerRunnable
        # that dialog already dispatches, just from a direct per-step button.
        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        # NLSQ can take minutes (e.g. multi-start restarts, EPM lattice sims);
        # the only prior feedback was a disabled Run button. run_fit_isolated
        # (subprocess_fit.py) does emit a real iteration/max_iter percent, but
        # multi-start restarts reset that percent each restart and nothing
        # here currently wires the queue through to a widget-local bar --
        # indeterminate + elapsed time is the honest, always-correct fallback
        # per the audit's own guidance, and matches step4_nuts.py's identical
        # treatment for NUTS (which has no per-iteration percent at all).
        # In-body (not just status-bar) feedback: the status bar's progress
        # sliver lives 300px wide at the bottom of the window and is easy to
        # miss during a multi-minute run; this sits right next to Cancel.
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 0)  # indeterminate/busy
        self._progress_bar.setVisible(False)
        self._elapsed_label = QLabel("", self)
        self._elapsed_label.setVisible(False)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)
        self._run_start_time = 0.0
        self._result = QLabel("", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        for w in (
            self._table,
            self._ms_enabled,
            self._ms_count,
            self._options_btn,
            self._run_btn,
            self._cancel_btn,
            self._progress_bar,
            self._elapsed_label,
            self._result,
        ):
            lay.addWidget(w)
        self.setTabOrder(self._ms_enabled, self._ms_count)
        self.setTabOrder(self._ms_count, self._options_btn)
        self.setTabOrder(self._options_btn, self._run_btn)
        self._ms_enabled.toggled.connect(self._on_multistart_changed)
        self._ms_count.valueChanged.connect(self._on_multistart_changed)
        self._run_btn.clicked.connect(self.run)

    def _on_elapsed_tick(self) -> None:
        elapsed = time.monotonic() - self._run_start_time
        self._elapsed_label.setText(f"Running... ({elapsed:.0f}s elapsed)")

    def _on_multistart_changed(self, *_args: object) -> None:
        cfg = self._state.nlsq_config
        cfg.multi_start = self._ms_enabled.isChecked()
        cfg.n_starts = self._ms_count.value()
        # Multi-start is now persisted into FitState.nlsq_config (see
        # __init__'s seeding comment), which project_codec serializes into
        # fit.json -- without this emit, window.py's dirty-tracking never
        # sees the edit, so New/Open/Close would silently discard an
        # un-rerun multi-start change with no unsaved-changes prompt.
        # config_edited, not edited -- see the class-level comment on why.
        self.config_edited.emit()

    def _on_cancel_clicked(self) -> None:
        if self._active_jobs is None or self._current_job_id is None:
            return
        # Look up the job_id captured when THIS run started (not recomputed
        # from live state.data_ref) -- run()'s own comments document that the
        # nested event loop lets the user navigate to Step 2 and pick a
        # different dataset while a fit is in flight, which would otherwise
        # make this button silently miss and do nothing.
        job = self._active_jobs.by_id.get(self._current_job_id)
        worker = job.get("worker") if job else None
        if worker is None:
            # Job already finished (or never registered) -- not an error,
            # just a stale click, but worth a trace since it's the one path
            # that makes this button silently do nothing.
            logger.debug(
                "Cancel clicked with no live worker",
                job_id=self._current_job_id,
            )
            return
        from PySide6.QtCore import QThreadPool

        from rheojax.gui.workspace.pipeline.cancel_runnable import (
            CancelWorkerRunnable,
        )

        QThreadPool.globalInstance().start(CancelWorkerRunnable(worker))

    def parameter_table(self) -> ParameterTable:
        return self._table

    def load_parameters_from_model(self) -> None:
        if not self._state.model_key:
            return
        # Same fallible call as step1_protocol_model.py's _refresh_preview()
        # (an out-of-range constructor-config value, e.g. N_y=1, can raise) --
        # this method is wired directly to Step 1's edited/config_edited
        # signals (fit_controller.py), so it runs in the same tick as a bad
        # config edit. Without this guard, step1's own try/except only
        # prevented ITS crash; this identical unguarded call one signal
        # handler later would still raise uncaught, leaving the parameter
        # table showing the PREVIOUS model's params with no indication why.
        try:
            instance = ModelRegistry.create(
                self._state.model_key, **self._state.model_config
            )
        except Exception:
            return
        # instance.parameters is a ParameterSet of core.parameters.Parameter
        # (`.bounds` tuple, no `.fixed`) -- ParameterTable needs foundation.state
        # .ParameterState (`.min_bound`/`.max_bound`/`.fixed`). Convert here,
        # mirroring ModelService.get_parameter_defaults()'s same conversion.
        params = {
            name: ParameterState(
                name=name,
                value=float(getattr(p, "value", 0.0)),
                min_bound=float(p.bounds[0]),
                max_bound=float(p.bounds[1]),
                fixed=False,
                unit=getattr(p, "units", ""),
                description=getattr(p, "description", ""),
            )
            for name, p in dict(instance.parameters).items()
        }
        self._table.set_parameters(params)

    def set_multistart(self, enabled: bool, count: int) -> None:
        self._ms_enabled.setChecked(enabled)
        self._ms_count.setValue(count)

    def fit_options(self) -> dict:
        return dict(self._fit_options)

    def _on_options_clicked(self) -> None:
        from rheojax.gui.dialogs.fitting_options import FittingOptionsDialog

        dialog = FittingOptionsDialog(current_options=self._fit_options, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._fit_options = dialog.get_options()

    def run(self) -> None:
        # get_parameters() silently skips invalid rows (out-of-range/non-
        # numeric text) so a fit could previously launch with those
        # parameters silently defaulted -- refuse to run until the user
        # fixes the highlighted cell(s) instead.
        if self._table.has_invalid_rows():
            from rheojax.gui.compat import QMessageBox

            QMessageBox.warning(
                self,
                "Invalid Parameters",
                "One or more parameter values or bounds are invalid "
                "(non-numeric, out of range, or min > max). Fix the "
                "highlighted cell(s) before running.",
            )
            return
        table_params = self._table.get_parameters()
        initial_params = {
            name: {
                "value": p.value,
                "bounds": (p.min_bound, p.max_bound),
                "fixed": p.fixed,
            }
            for name, p in table_params.items()
        } or None
        # Guard against re-entrant clicks: _fit_fn pumps a nested event loop
        # (see fit_controller._run_on_thread) while staying responsive, so the
        # button must be disabled for the duration or a second click launches
        # an overlapping fit that corrupts shared active_jobs tracking.
        self._run_btn.setEnabled(False)
        self._run_start_time = time.monotonic()
        self._progress_bar.setVisible(True)
        self._elapsed_label.setVisible(True)
        self._elapsed_label.setText("Running... (0s elapsed)")
        self._elapsed_timer.start()
        # job_id mirrors fit_controller._make_fit_fn's own f"{data_ref}:nlsq"
        # exactly, captured here (not recomputed from live state in
        # _on_cancel_clicked) so a dataset switch mid-run can't make Cancel
        # silently miss the job it's actually trying to stop.
        self._current_job_id = f"{self._state.data_ref}:nlsq"
        self._cancel_btn.setVisible(self._active_jobs is not None)
        # Snapshot: _fit_fn pumps a nested QEventLoop, so the user can
        # navigate elsewhere/edit Step 1 while this run is in flight --
        # that invalidation bumps FitState.revision (invalidation.py). If it
        # changed by the time the fit returns, the result no longer
        # corresponds to the current model/protocol/data selection and must
        # be discarded rather than silently written over the newer state.
        revision_at_start = self._state.revision
        try:
            # This step's own _ms_enabled/_ms_count widgets are the single
            # source of truth for multi-start (outer restart loop in
            # fit_controller). FittingOptionsDialog can also set
            # "multistart"/"num_starts" in self._fit_options, which
            # ModelService.fit() would translate into its OWN, separate
            # backend-level multi-start. Passing both through would nest
            # them: the outer loop's `count` restarts each triggering the
            # backend's `n_starts` restarts, silently multiplying fit time.
            # Strip a copy here (never mutate self._fit_options) so the
            # dialog's stored state is preserved for the user to see if
            # they reopen it, but never reaches the backend.
            fit_options = {
                k: v
                for k, v in self._fit_options.items()
                if k not in ("multistart", "num_starts")
            }
            try:
                res = self._fit_fn(
                    self._state.model_key,
                    self._state.model_config,
                    self._state.data_ref,
                    self._state.column_map,
                    initial_params=initial_params,
                    multi_start={
                        "enabled": self._ms_enabled.isChecked(),
                        "count": self._ms_count.value(),
                    },
                    options=fit_options,
                )
            except NotImplementedError:
                # ponytail: real solver wiring is out of scope here (tracked separately);
                # this guard only keeps an unwired Run button from crashing the Qt slot.
                self._result.setText("NLSQ solver is not wired up yet.")
                # A first-ever failure with nlsq_result already None needs no
                # change -- is_ready() already reads that as not-ready. Only
                # a STALE successful result (from an earlier run) needs to
                # be cleared here; leaving it as a fresh {"success": False}
                # dict rather than None would be an unrelated contract
                # change with no is_ready() difference.
                if (
                    self._state.revision == revision_at_start
                    and self._state.nlsq_result is not None
                ):
                    self._state.nlsq_result = {
                        "success": False,
                        "message": "NLSQ solver is not wired up yet.",
                    }
                    self.edited.emit()
                return
            except CancellationError:
                # A user-initiated cancel is not a fit failure -- show a
                # clean cancelled state and leave a stale successful result
                # in place untouched (unlike a real failure, cancelling
                # doesn't invalidate a prior success the user hasn't acted
                # on yet).
                self._result.setText("Cancelled.")
                return
            except Exception as exc:
                self._result.setText(f"NLSQ failed: {exc}")
                # Discarded if state moved on while this fit was running
                # (see revision_at_start above); otherwise a prior
                # successful nlsq_result must NOT survive a failed re-fit --
                # is_ready() reads nlsq_result live, so leaving the old
                # success dict in place would let the wizard advance to
                # NUTS (warm-starting from stale params) while the screen
                # plainly says the current attempt failed. A first-ever
                # failure (nlsq_result already None) needs no change --
                # is_ready() already reads that as not-ready.
                if (
                    self._state.revision == revision_at_start
                    and self._state.nlsq_result is not None
                ):
                    self._state.nlsq_result = {"success": False, "message": str(exc)}
                    self.edited.emit()
                return
            if self._state.revision != revision_at_start:
                # State was invalidated (model/protocol/data changed) while
                # this fit was running in the nested event loop -- the
                # result belongs to a selection that no longer applies.
                return
            # Normalize to dict: ModelService.fit() returns FitResult (dataclass), fakes return dict
            if isinstance(res, dict):
                self._state.nlsq_result = res
            else:
                self._state.nlsq_result = {
                    "params": res.params,
                    "r_squared": res.r_squared,
                    "success": getattr(res, "success", True),
                }
            self.refresh_display()
            self.edited.emit()
            self.finished.emit()
        finally:
            self._run_btn.setEnabled(True)
            self._cancel_btn.setVisible(False)
            self._current_job_id = None
            self._elapsed_timer.stop()
            self._progress_bar.setVisible(False)
            self._elapsed_label.setVisible(False)

    def refresh_display(self) -> None:
        """Sync the result label to current state.

        Called both after a run and whenever an upstream edit invalidates
        nlsq_result -- without this, the label kept showing a stale
        "R²=..." readout for a fit that was already discarded.
        """
        if self._state.nlsq_result is None:
            self._result.setText("")
            return
        result = self._state.nlsq_result
        r2 = result.get("r_squared", float("nan"))
        if r2 is None:
            r2 = float("nan")
        lines = [f"R²={r2:.3f}"]

        chi2 = result.get("chi_squared")
        if chi2 is not None:
            lines.append(f"chi²={float(chi2):.6g}")

        mpe = result.get("mpe")
        if mpe is not None:
            lines.append(f"MPE={float(mpe):.2f}%")

        fit_time = result.get("fit_time")
        if fit_time is not None:
            lines.append(f"time={float(fit_time):.2f}s")

        # ModelService.fit() appends a poor-fit note to `message` when
        # r_squared < 0.5 despite nlsq reporting success (model_service.py) --
        # without surfacing it here, that's the only place the warning was
        # ever written, and nothing else in the GUI reads it. A plain
        # "Fit successful" is the boring default and not worth a line.
        message = result.get("message")
        if message and message != "Fit successful":
            lines.append(f"⚠ {message}")

        # Recompute locally from pcov (sorted param names + shape-checked
        # sqrt(diag)) rather than trusting a precomputed uncertainties list's
        # ordering against params -- mirrors the legacy shell's existing,
        # working pattern (rheojax/gui/pages/fit_page.py:1095-1122).
        params = result.get("params") or {}
        pcov = result.get("pcov")
        if pcov is not None and params:
            param_names = sorted(params.keys())
            try:
                pcov_arr = np.asarray(pcov)
                if pcov_arr.ndim == 2 and pcov_arr.shape[0] == len(param_names):
                    sigma_parts = [
                        f"{name}=±{np.sqrt(pcov_arr[i, i]):.3g}"
                        for i, name in enumerate(param_names)
                        if pcov_arr[i, i] > 0
                    ]
                    if sigma_parts:
                        lines.append("  ".join(sigma_parts))
            except (ValueError, TypeError, IndexError):
                pass

        self._result.setText("  ".join(lines))

    def is_ready(self) -> bool:
        return bool(
            self._state.nlsq_result is not None
            and self._state.nlsq_result.get("success", True)
        )
