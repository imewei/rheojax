"""Regression test: cleanup() must still close figures when the canvas's
C++ object has already been deleted (RuntimeError on attribute access).
"""

import matplotlib.pyplot as plt

from rheojax.gui.widgets.base_arviz_widget import BaseArviZWidget


class _DeadCanvas:
    """Stand-in for a FigureCanvasQTAgg whose C++ object was already freed."""

    @property
    def callbacks(self):
        raise RuntimeError("wrapped C/C++ object of type FigureCanvasQTAgg has been deleted")


def test_cleanup_closes_figures_when_canvas_is_dead(qapp):
    widget = BaseArviZWidget()
    fig = plt.figure()
    pending_fig = plt.figure()

    widget._figure = fig
    widget._pending_cleanup = [pending_fig]
    widget._canvas = _DeadCanvas()

    widget.cleanup()

    assert not plt.fignum_exists(fig.number)
    assert not plt.fignum_exists(pending_fig.number)
    assert widget._pending_cleanup == []
