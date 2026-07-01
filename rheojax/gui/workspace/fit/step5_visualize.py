from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QGridLayout, QTabWidget, QVBoxLayout, QWidget

from rheojax.core.arviz_utils import inference_data_from_dict
from rheojax.gui.foundation.state import FitState
from rheojax.gui.widgets.arviz_canvas import ArvizCanvas
from rheojax.gui.widgets.pyqtgraph_canvas import PyQtGraphCanvas
from rheojax.gui.widgets.residuals_panel import ResidualsPanel

_ARVIZ_PLOTS = ["pair", "forest", "energy", "autocorr", "rank", "ess"]


class VisualizeStep(QWidget):
    """Fit-workflow step 5: Fit overlay / Residuals / Diagnostics tabs.

    ``state.nlsq_result`` is the plain dict NlsqStep normalizes FitResult
    into (see step3_nlsq.py) — currently ``{"params", "r_squared",
    "success"}``. The overlay/residuals panels additionally read optional
    ``"x"``, ``"y"``, ``"y_fit"`` keys when a later controller populates
    them; until then those two tabs are constructed but stay empty.

    ``state.nuts_result`` is expected to be normalized the same way, by
    analogy with ``BayesianResult`` (rheojax/gui/state/store.py) and its
    ``inference_data`` consumption in
    ``rheojax/gui/pages/diagnostics_page.py::_get_inference_data``:
    ``{"inference_data": arviz.InferenceData | None, "posterior_samples":
    dict, "sample_stats": dict | None, "num_chains": int, "r_hat": dict,
    "ess": dict, "divergences": int}``.
    """

    def __init__(self, state: FitState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._tabs = QTabWidget(self)
        self._names: list[str] = []
        self._arviz_canvases: dict[str, ArvizCanvas] = {}

        self._overlay = PyQtGraphCanvas(self)
        self._overlay.set_labels(x_label="x", y_label="y", title="Fit overlay")
        self._add("Fit overlay", self._overlay)

        self._residuals = ResidualsPanel(self)
        self._add("Residuals", self._residuals)

        # NOTE: Do NOT add Diagnostics here — nuts_result is always None at
        # controller build time. refresh() adds it once NUTS completes.
        QVBoxLayout(self).addWidget(self._tabs)
        self.refresh()

    def _add(self, name: str, widget: QWidget) -> None:
        self._tabs.addTab(widget, name)
        self._names.append(name)

    def refresh(self) -> None:
        """Re-populate tabs from current state (call after NLSQ/NUTS updates)."""
        self._refresh_overlay()
        self._refresh_residuals()
        if self._state.nuts_result is not None and "Diagnostics" not in self._names:
            self._add_diagnostics_tab()
        self._refresh_diagnostics()

    def tab_names(self) -> list[str]:
        return list(self._names)

    def arviz_plots(self) -> list[str]:
        return list(_ARVIZ_PLOTS) if self._state.nuts_result is not None else []

    def _refresh_overlay(self) -> None:
        result = self._state.nlsq_result or {}
        x, y, y_fit = result.get("x"), result.get("y"), result.get("y_fit")
        if x is None or y is None:
            return
        self._overlay.clear()
        self._overlay.plot_data(np.asarray(x), np.asarray(y), name="Data")
        if y_fit is not None:
            self._overlay.plot_line(np.asarray(x), np.asarray(y_fit), name="Fit")
        nuts = self._state.nuts_result or {}
        band = nuts.get("y_band")
        if band:
            lo, hi = band
            self._overlay.plot_line(
                np.asarray(x), np.asarray(lo), name="Posterior lo", line_style="dash"
            )
            self._overlay.plot_line(
                np.asarray(x), np.asarray(hi), name="Posterior hi", line_style="dash"
            )

    def _refresh_residuals(self) -> None:
        result = self._state.nlsq_result or {}
        y, y_fit = result.get("y"), result.get("y_fit")
        if y is None or y_fit is None:
            return
        self._residuals.plot_residuals(np.asarray(y), np.asarray(y_fit), result.get("x"))

    def _add_diagnostics_tab(self) -> None:
        page = QWidget(self)
        grid = QGridLayout(page)
        for i, plot_name in enumerate(_ARVIZ_PLOTS):
            canvas = ArvizCanvas(page)
            canvas.set_plot_type(plot_name)
            grid.addWidget(canvas, i // 2, i % 2)
            self._arviz_canvases[plot_name] = canvas
        self._add("Diagnostics", page)

    def _refresh_diagnostics(self) -> None:
        if not self._arviz_canvases:
            return
        idata = self._inference_data()
        if idata is None:
            return
        for canvas in self._arviz_canvases.values():
            canvas.set_inference_data(idata)

    def _inference_data(self) -> Any | None:
        """Build (or pass through) an ArviZ InferenceData from nuts_result.

        Mirrors diagnostics_page.py::_get_inference_data's contract but
        reads a plain dict (nuts_result) instead of a BayesianResult
        dataclass.
        """
        result = self._state.nuts_result
        if not result:
            return None

        idata = result.get("inference_data")
        if idata is not None and hasattr(idata, "posterior"):
            return idata

        posterior_samples = result.get("posterior_samples")
        if not posterior_samples:
            return None

        num_chains = result.get("num_chains", 1) or 1
        posterior = {
            name: self._to_chain_draw(samples, num_chains)
            for name, samples in posterior_samples.items()
        }
        sample_stats = result.get("sample_stats")
        sample_stats_dict = (
            {
                name: self._to_chain_draw(stat, num_chains)
                for name, stat in sample_stats.items()
            }
            if sample_stats
            else None
        )
        try:
            return inference_data_from_dict(
                {"posterior": posterior, "sample_stats": sample_stats_dict}
            )
        except Exception:
            return None

    @staticmethod
    def _to_chain_draw(samples: Any, num_chains: int) -> np.ndarray:
        """Reshape a flat (chains*draws,) array to (chains, draws) for ArviZ."""
        arr = np.asarray(samples)
        if arr.ndim != 1:
            return arr
        if num_chains > 1 and arr.size % num_chains == 0:
            return arr.reshape(num_chains, -1)
        return arr.reshape(1, -1)
