from __future__ import annotations

import math
from typing import Any

import numpy as np

from rheojax.core.arviz_utils import inference_data_from_dict
from rheojax.gui.compat import (
    QGridLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from rheojax.gui.foundation.state import FitState
from rheojax.gui.resources.styles.tokens import Typography, themed
from rheojax.gui.utils.layout_helpers import set_panel_margins
from rheojax.gui.widgets.arviz_canvas import ArvizCanvas
from rheojax.gui.widgets.pyqtgraph_canvas import PyQtGraphCanvas
from rheojax.gui.widgets.residuals_panel import ResidualsPanel
from rheojax.gui.workspace.fit.step4_nuts import _ESS_MIN, _R_HAT_THRESHOLD

_ARVIZ_PLOTS = [
    "pair", "forest", "energy", "autocorr", "rank", "ess", "trace", "posterior"
]


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
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        lay.addWidget(self._tabs)
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
        elif self._state.nuts_result is None and "Diagnostics" in self._names:
            self._remove_diagnostics_tab()
        self._refresh_diagnostics()

    def tab_names(self) -> list[str]:
        return list(self._names)

    def arviz_plots(self) -> list[str]:
        return list(_ARVIZ_PLOTS) if self._state.nuts_result is not None else []

    def diagnostics_badge_text(self) -> str:
        nuts = self._state.nuts_result
        if not nuts or "verdict" not in nuts:
            return ""
        verdict = nuts["verdict"]
        if verdict.get("converged"):
            return "✓ converged"
        return "⚠ not-converged: " + "; ".join(verdict.get("reasons", []))

    def _refresh_overlay(self) -> None:
        result = self._state.nlsq_result or {}
        x, y, y_fit = result.get("x"), result.get("y"), result.get("y_fit")
        if x is None or y is None:
            return
        x = np.asarray(x)
        y = np.asarray(y)
        self._overlay.clear()
        # PyQtGraph rejects complex dtype arrays outright. Oscillation-mode
        # data/fits carry y = G' + i*G'' as a genuine complex array (see
        # RheoData.is_complex) -- split into separate real/imag traces
        # instead of collapsing to abs() (as ResidualsPanel does for a
        # single scalar residual), since G' and G'' are both independently
        # meaningful curves here.
        if np.iscomplexobj(y):
            self._overlay.plot_data(x, y.real, name="Data (G')")
            self._overlay.plot_data(x, y.imag, name="Data (G'')")
        else:
            self._overlay.plot_data(x, y, name="Data")
        if y_fit is not None:
            y_fit = np.asarray(y_fit)
            if np.iscomplexobj(y_fit):
                self._overlay.plot_line(x, y_fit.real, name="Fit (G')")
                self._overlay.plot_line(x, y_fit.imag, name="Fit (G'')")
            else:
                self._overlay.plot_line(x, y_fit, name="Fit")
        nuts = self._state.nuts_result or {}
        band = nuts.get("y_band")
        if band is not None:
            lo, hi = band
            lo = np.asarray(lo)
            hi = np.asarray(hi)
            if np.iscomplexobj(lo) or np.iscomplexobj(hi):
                self._overlay.plot_line(
                    x, lo.real, name="Posterior lo (G')", line_style="dash"
                )
                self._overlay.plot_line(
                    x, hi.real, name="Posterior hi (G')", line_style="dash"
                )
                self._overlay.plot_line(
                    x, lo.imag, name="Posterior lo (G'')", line_style="dash"
                )
                self._overlay.plot_line(
                    x, hi.imag, name="Posterior hi (G'')", line_style="dash"
                )
            else:
                self._overlay.plot_line(x, lo, name="Posterior lo", line_style="dash")
                self._overlay.plot_line(x, hi, name="Posterior hi", line_style="dash")

    def _refresh_residuals(self) -> None:
        result = self._state.nlsq_result or {}
        y, y_fit = result.get("y"), result.get("y_fit")
        if y is None or y_fit is None:
            return
        self._residuals.plot_residuals(
            np.asarray(y), np.asarray(y_fit), result.get("x")
        )

    def _add_diagnostics_tab(self) -> None:
        page = QWidget(self)
        outer = QVBoxLayout(page)
        badge = QLabel(self.diagnostics_badge_text(), page)
        outer.addWidget(badge)

        self._rhat_label = QLabel("R-hat: --", page)
        self._ess_label = QLabel("ESS: --", page)
        self._divergence_label = QLabel("Divergences: 0", page)
        for label in (self._rhat_label, self._ess_label):
            label.setStyleSheet(f"font-family: {Typography.FONT_FAMILY_MONO};")
        outer.addWidget(self._rhat_label)
        outer.addWidget(self._ess_label)
        outer.addWidget(self._divergence_label)

        grid_widget = QWidget(page)
        grid = QGridLayout(grid_widget)
        for i, plot_name in enumerate(_ARVIZ_PLOTS):
            canvas = ArvizCanvas(grid_widget)
            canvas.set_plot_type(plot_name)
            grid.addWidget(canvas, i // 2, i % 2)
            self._arviz_canvases[plot_name] = canvas
        outer.addWidget(grid_widget)
        self._badge_label = badge
        self._add("Diagnostics", page)

    def _refresh_diagnostics_summary(self) -> None:
        """Populate the R-hat/ESS/Divergences labels from nuts_result.

        No new computation -- state.nuts_result already carries r_hat, ess,
        and divergences (see run_bayesian_isolated() in subprocess_bayesian.py).
        Thresholds are imported from step4_nuts.py so these labels can never
        disagree with the convergence-verdict badge on the same screen.
        """
        result = self._state.nuts_result or {}
        mono_style = f"font-family: {Typography.FONT_FAMILY_MONO};"

        r_hat = result.get("r_hat") or {}
        # Individual dict values can be None (step4_nuts.py::_diagnostics_
        # verdict() already guards this exact case), so filter those before
        # aggregating -- max() raises TypeError comparing None to a float.
        r_hat_vals = [v for v in r_hat.values() if v is not None]
        if r_hat_vals:
            max_rhat = max(r_hat_vals)
            # Check every value for NaN individually rather than only the
            # aggregate: max()/min() are order-dependent with NaN present
            # (a NaN never compares >/< a finite value, so it can silently
            # lose to a finite value depending on dict iteration order and
            # never surface in the aggregate at all).
            has_nan = any(isinstance(v, float) and math.isnan(v) for v in r_hat_vals)
            failing = has_nan or max_rhat > _R_HAT_THRESHOLD
            status = "WARNING" if failing else "OK"
            color = themed("WARNING") if failing else themed("SUCCESS")
            self._rhat_label.setText(f"R-hat (max): {max_rhat:.4f} [{status}]")
            self._rhat_label.setStyleSheet(f"color: {color}; {mono_style}")
        else:
            self._rhat_label.setText("R-hat: --")
            self._rhat_label.setStyleSheet(mono_style)

        ess = result.get("ess") or {}
        ess_vals = [v for v in ess.values() if v is not None]
        if ess_vals:
            min_ess = min(ess_vals)
            has_nan = any(isinstance(v, float) and math.isnan(v) for v in ess_vals)
            failing = has_nan or min_ess < _ESS_MIN
            status = "LOW" if failing else "OK"
            color = themed("WARNING") if failing else themed("SUCCESS")
            self._ess_label.setText(f"ESS (min): {min_ess:.0f} [{status}]")
            self._ess_label.setStyleSheet(f"color: {color}; {mono_style}")
        else:
            self._ess_label.setText("ESS: --")
            self._ess_label.setStyleSheet(mono_style)

        divergences = result.get("divergences", 0)
        if divergences is None:
            divergences = 0
        if divergences == -1:
            self._divergence_label.setText("Divergences: unknown")
            self._divergence_label.setStyleSheet(
                f"color: {themed('WARNING')}; font-weight: bold;"
            )
        elif divergences > 0:
            self._divergence_label.setText(f"Divergences: {divergences}")
            self._divergence_label.setStyleSheet(
                f"color: {themed('ERROR')}; font-weight: bold;"
            )
        else:
            self._divergence_label.setText("Divergences: 0")
            self._divergence_label.setStyleSheet(
                f"color: {themed('SUCCESS')}; font-weight: bold;"
            )

    def _remove_diagnostics_tab(self) -> None:
        idx = self._names.index("Diagnostics")
        page = self._tabs.widget(idx)
        self._tabs.removeTab(idx)
        if page is not None:
            page.deleteLater()
        self._arviz_canvases.clear()
        self._names.remove("Diagnostics")
        # ponytail: drop the dangling reference so a later refresh() (before
        # deleteLater's deferred Qt cleanup actually runs) can't touch a
        # widget whose parent tab no longer exists.
        if hasattr(self, "_badge_label"):
            del self._badge_label

    def _refresh_diagnostics(self) -> None:
        if hasattr(self, "_badge_label"):
            self._badge_label.setText(self.diagnostics_badge_text())
        if hasattr(self, "_rhat_label"):
            self._refresh_diagnostics_summary()
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
