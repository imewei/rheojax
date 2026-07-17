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

from rheojax.gui.widgets.base_arviz_widget import BaseArviZWidget
from rheojax.gui.widgets.residuals_panel import ResidualsPanel


def test_residuals_panel_cleanup_fires_on_destroy_without_close(qtbot):
    panel = ResidualsPanel()
    calls = []
    panel.cleanup = lambda: calls.append(True)

    panel.deleteLater()
    qtbot.wait(50)

    assert calls, "cleanup() was not called when the widget was destroyed without close()"


def test_base_arviz_widget_cleanup_fires_on_destroy_without_close(qtbot):
    # Same class of bug as above, for the widget step5_visualize.py's
    # Diagnostics tab actually builds 8 of per NUTS run (ArvizCanvas
    # subclasses BaseArviZWidget) -- _remove_diagnostics_tab() tears its
    # container down via deleteLater() cascade, not .close().
    widget = BaseArviZWidget()
    calls = []
    widget.cleanup = lambda: calls.append(True)

    widget.deleteLater()
    qtbot.wait(50)

    assert calls, "cleanup() was not called when the widget was destroyed without close()"
