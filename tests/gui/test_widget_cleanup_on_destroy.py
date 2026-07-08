"""Regression: closeEvent() only fires for top-level widgets. ResidualsPanel
and MultiView are usually embedded as children (VisualizeStep, the Compare
Models QDialog) and get torn down via Qt's parent-cascade deleteLater()
instead of their own .close() -- closeEvent never reaches them there, so
cleanup() (which releases matplotlib figures/canvas callbacks) silently never
ran, leaking resources across repeated fit/NUTS runs. Wiring cleanup() to the
`destroyed` signal instead covers every teardown path.
"""

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.widgets.multi_view import MultiView
from rheojax.gui.widgets.residuals_panel import ResidualsPanel


def test_residuals_panel_cleanup_fires_on_destroy_without_close(qtbot):
    panel = ResidualsPanel()
    calls = []
    panel.cleanup = lambda: calls.append(True)

    panel.deleteLater()
    qtbot.wait(50)

    assert calls, "cleanup() was not called when the widget was destroyed without close()"


def test_multi_view_cleanup_fires_on_destroy_without_close(qtbot):
    view = MultiView()
    calls = []
    view.cleanup = lambda: calls.append(True)

    view.deleteLater()
    qtbot.wait(50)

    assert calls, "cleanup() was not called when the widget was destroyed without close()"
