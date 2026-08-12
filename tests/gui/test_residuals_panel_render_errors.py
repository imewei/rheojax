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


@pytest.mark.parametrize("plot_type", ["qq", "histogram"])
def test_set_plot_type_renders_without_error(qtbot, plot_type):
    """Functional (non-golden-image) coverage for the qq/histogram dispatch
    branches: PR #111 dropped the pixel-comparison goldens for these plot
    types as environment-fragile, but no replacement asserted the dispatch
    itself still runs without hitting the error-label fallback path.

    Per the project's established FreeType-skip convention (this host's
    matplotlib/FreeType glyph rasterizer can corrupt after enough Figure
    churn in-process, independent of this widget's code — see
    test_visual_regression.py's `_render_widget_figure` docstring), a render
    failure whose message matches that known signature is skipped rather
    than failed; any other failure is a real regression.
    """
    panel = ResidualsPanel()
    qtbot.addWidget(panel)

    rng = np.random.default_rng(0)
    observed = np.linspace(1.0, 10.0, 30)
    predicted = observed + rng.normal(0, 0.1, size=30)

    panel.set_plot_type(plot_type)
    panel.plot_residuals(observed, predicted, observed)

    if not panel._empty_label.isHidden():
        error_text = panel._empty_label.text()
        if "FT_Render_Glyph" in error_text or "raster overflow" in error_text:
            pytest.skip(f"Known host FreeType rendering limitation: {error_text}")
        pytest.fail(
            f"Unexpected render error for plot_type={plot_type!r}: {error_text}"
        )


def test_arviz_canvas_get_figure_on_empty_state_does_not_raise(qtbot):
    """Functional (non-golden-image) coverage for ArvizCanvas with no
    inference data set: PR #111 dropped the golden-image empty-state test
    as font-metric-sensitive without a replacement asserting get_figure()
    itself stays safe to call before any data is loaded.
    """
    from rheojax.gui.widgets.arviz_canvas import ArvizCanvas

    canvas = ArvizCanvas()
    qtbot.addWidget(canvas)

    figure = canvas.get_figure()  # must not raise

    assert figure is not None
