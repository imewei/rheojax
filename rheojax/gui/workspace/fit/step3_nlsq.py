from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheojax.core.registry import ModelRegistry
from rheojax.gui.foundation.state import FitState
from rheojax.gui.state.store import ParameterState
from rheojax.gui.utils.layout_helpers import set_panel_margins
from rheojax.gui.widgets.parameter_table import ParameterTable


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
        self._table = ParameterTable(self)
        self._ms_enabled = QCheckBox("multi-start", self)
        self._ms_count = QSpinBox(self)
        self._ms_count.setRange(1, 64)
        self._ms_count.setValue(8)
        self._fit_options: dict = {}
        self._options_btn = QPushButton("⚙ Fit Options", self)
        self._options_btn.clicked.connect(self._on_options_clicked)
        self._run_btn = QPushButton("▶ Run NLSQ", self)
        self._result = QLabel("", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        for w in (
            self._table,
            self._ms_enabled,
            self._ms_count,
            self._options_btn,
            self._run_btn,
            self._result,
        ):
            lay.addWidget(w)
        self._run_btn.clicked.connect(self.run)

    def parameter_table(self) -> ParameterTable:
        return self._table

    def load_parameters_from_model(self) -> None:
        if not self._state.model_key:
            return
        instance = ModelRegistry.create(
            self._state.model_key, **self._state.model_config
        )
        # instance.parameters is a ParameterSet of core.parameters.Parameter
        # (`.bounds` tuple, no `.fixed`) -- ParameterTable needs gui.state.store
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
        # Multi-start config is transient widget state — not stored in FitState
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
