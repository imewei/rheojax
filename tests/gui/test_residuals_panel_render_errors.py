"""Regression: ResidualsPanel._refresh_plot() guarded the plot-generation
function against exceptions but left the immediately-following
self._canvas.draw() call unguarded. Agg/FreeType rendering can legitimately
raise (e.g. a host DPI/font-cache combination triggering a "raster overflow"
when the tight-layout engine can't fit axis decorations) for reasons outside
this widget's control, and that raised straight through _refresh_plot() into
plot_residuals(), crashing whatever called it instead of degrading to an
error state like the plot-generation path already does.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from rheojax.gui.widgets.residuals_panel import ResidualsPanel


def test_refresh_plot_survives_canvas_draw_failure(qtbot, monkeypatch):
    panel = ResidualsPanel()
    qtbot.addWidget(panel)

    def _raise():
        raise RuntimeError("FT_Render_Glyph failed with error 0x62: raster overflow")

    monkeypatch.setattr(panel._canvas, "draw", _raise)

    panel.plot_residuals(
        np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1]), np.array([1.0, 2.0, 3.0])
    )  # must not raise

    assert "render error" in panel._empty_label.text().lower()
