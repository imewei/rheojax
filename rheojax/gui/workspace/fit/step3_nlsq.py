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

from rheojax.core.registry import ModelRegistry
from rheojax.gui.foundation.state import FitState
from rheojax.gui.state.store import ParameterState
from rheojax.gui.widgets.parameter_table import ParameterTable


def _default_fit_fn(model_key, model_config, data_ref, column_map, initial_params=None,
                     multi_start=None):
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
        self._run_btn = QPushButton("▶ Run NLSQ", self)
        self._result = QLabel("", self)
        lay = QVBoxLayout(self)
        for w in (self._table, self._ms_enabled, self._ms_count, self._run_btn, self._result):
            lay.addWidget(w)
        self._run_btn.clicked.connect(self.run)

    def parameter_table(self) -> ParameterTable:
        return self._table

    def load_parameters_from_model(self) -> None:
        if not self._state.model_key:
            return
        instance = ModelRegistry.create(self._state.model_key, **self._state.model_config)
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

    def run(self) -> None:
        # Multi-start config is transient widget state — not stored in FitState
        table_params = self._table.get_parameters()
        initial_params = {
            name: {"value": p.value, "bounds": (p.min_bound, p.max_bound), "fixed": p.fixed}
            for name, p in table_params.items()
        } or None
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
            )
        except NotImplementedError:
            # ponytail: real solver wiring is out of scope here (tracked separately);
            # this guard only keeps an unwired Run button from crashing the Qt slot.
            self._result.setText("NLSQ solver is not wired up yet.")
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

    def refresh_display(self) -> None:
        """Sync the result label to current state.

        Called both after a run and whenever an upstream edit invalidates
        nlsq_result -- without this, the label kept showing a stale
        "R²=..." readout for a fit that was already discarded.
        """
        if self._state.nlsq_result is None:
            self._result.setText("")
            return
        r2 = self._state.nlsq_result.get("r_squared", float("nan"))
        if r2 is None:
            r2 = float("nan")
        self._result.setText(f"R²={r2:.3f}")

    def is_ready(self) -> bool:
        return bool(
            self._state.nlsq_result is not None
            and self._state.nlsq_result.get("success", True)
        )
