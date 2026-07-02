from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget

import rheojax.transforms  # noqa: F401
from rheojax.core.registry import TransformRegistry
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.widgets.pyqtgraph_canvas import PyQtGraphCanvas

_DOMAIN_CHANGING = {"spectral", "decomposition"}


def _plot_as_line(canvas: PyQtGraphCanvas, x: Any, y: Any, name: str) -> None:
    # ponytail: PyQtGraphCanvas.plot_line() has a pre-existing bug (raw int
    # passed to QPen.setStyle instead of Qt.PenStyle; see
    # rheojax/gui/workspace/fit/step5_visualize.py's test_step5.py comment for
    # the same finding) that crashes on any call. Use plot_data() with
    # line_width>0 and no symbol instead — same visual result, working code
    # path. Switch back to plot_line() once that bug is fixed upstream.
    canvas.plot_data(x, y, name=name, symbol=None, line_width=2)


def _xy(entry: Any) -> tuple[Any, Any] | None:
    """Best-effort extraction of (x, y) from a RheoData-like object or dict.

    `state.result` has no fixed schema across transforms, so this tolerates
    missing keys/attributes and plain placeholder values (e.g. strings used
    in unit tests) by returning None instead of raising.
    """
    if entry is None:
        return None
    x = getattr(entry, "x", None)
    y = getattr(entry, "y", None)
    if x is None and isinstance(entry, dict):
        x, y = entry.get("x"), entry.get("y")
    if x is None or y is None:
        return None
    return x, y


class TransformVisualizeStep(QWidget):
    def __init__(self, state: TransformState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._tabs = QTabWidget(self)
        self._names: list[str] = []
        QVBoxLayout(self).addWidget(self._tabs)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild tabs from the current transform_key/result.

        Bodies are constructed eagerly (before a transform is picked and
        before RunStep has produced a result), so the tab layout must be
        recomputed here rather than frozen in __init__ -- otherwise it
        stays stuck at construction-time (empty, transform_key=None)
        forever. The view mode itself can change (overlay <-> separate
        canvases) once the real transform_key/result are known, so this
        rebuilds from scratch (mirrors SlotsStep.refresh()) rather than
        re-plotting onto persistent canvases. Call after RunStep emits
        `finished`.
        """
        while self._tabs.count():
            widget = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()
        self._names = []
        primary = "Input vs output" if self.view_mode() == "overlay" else "Output"
        self._add(primary, self._build_primary_tab())
        self._add("Result", self._build_result_tab())

    def _add(self, name: str, widget: QWidget) -> None:
        self._tabs.addTab(widget, name)
        self._names.append(name)

    def _category(self) -> str:
        if self._state.transform_key is None:
            return "analysis"  # safe default; view_mode() → "overlay"
        info = TransformRegistry.get_info(self._state.transform_key)
        if info is None:
            return "analysis"  # unrecognized key -> same safe default as above
        return str(info.transform_type).split(".")[-1].lower()

    def view_mode(self) -> str:
        return "separate" if self._category() in _DOMAIN_CHANGING else "overlay"

    def tab_names(self) -> list[str]:
        return list(self._names)

    def _payload(self) -> dict:
        result = self._state.result
        return result if isinstance(result, dict) else {}

    def _build_primary_tab(self) -> QWidget:
        payload = self._payload()
        input_xy = _xy(payload.get("input"))
        output_xy = _xy(payload.get("output"))

        if self.view_mode() == "overlay":
            canvas = PyQtGraphCanvas(self)
            if input_xy is not None:
                canvas.plot_data(*input_xy, name="Input")
            if output_xy is not None:
                _plot_as_line(canvas, *output_xy, name="Output")
            return canvas

        # Domain-changing transforms (e.g. fft_analysis, spp_decomposer): the
        # input and output axes are not comparable, so use two side-by-side
        # canvases instead of overlaying mismatched domains.
        panel = QWidget(self)
        layout = QHBoxLayout(panel)
        input_canvas = PyQtGraphCanvas(panel)
        output_canvas = PyQtGraphCanvas(panel)
        input_canvas.set_labels(title="Input")
        output_canvas.set_labels(title="Output")
        if input_xy is not None:
            input_canvas.plot_data(*input_xy, name="Input")
        if output_xy is not None:
            _plot_as_line(output_canvas, *output_xy, name="Output")
        layout.addWidget(input_canvas)
        layout.addWidget(output_canvas)
        return panel

    def _build_result_tab(self) -> QWidget:
        result = self._payload().get("result")
        label = QLabel(self)
        label.setWordWrap(True)
        if isinstance(result, dict) and result:
            label.setText("\n".join(f"{k}: {v}" for k, v in result.items()))
        else:
            label.setText("No result yet.")
        return label
