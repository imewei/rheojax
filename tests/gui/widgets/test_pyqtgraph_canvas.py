"""Regression tests for PyQtGraphCanvas export behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from rheojax.gui.widgets.pyqtgraph_canvas import (  # noqa: E402
    PYQTGRAPH_AVAILABLE,
    PyQtGraphCanvas,
)

pytestmark = pytest.mark.skipif(
    not PYQTGRAPH_AVAILABLE, reason="pyqtgraph not available"
)


def test_on_export_uses_svg_exporter_for_svg_path(qtbot, monkeypatch, tmp_path):
    """Regression: _on_export used ImageExporter (PNG) unconditionally, even
    when the user chose the SVG filter and a .svg path, corrupting the file.
    """
    import pyqtgraph as pg

    from rheojax.gui.compat import QFileDialog

    canvas = PyQtGraphCanvas()
    qtbot.addWidget(canvas)
    canvas.plot_data([1, 2, 3], [1, 4, 9], name="test")

    svg_path = str(tmp_path / "plot.svg")
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (svg_path, "SVG Vector (*.svg)")
    )

    used_exporter_types = []
    for exporter_cls_name in ("SVGExporter", "ImageExporter"):
        orig_cls = getattr(pg.exporters, exporter_cls_name)

        def make_tracker(cls, name=exporter_cls_name):
            def _tracker(item):
                used_exporter_types.append(name)
                instance = cls(item)
                instance.export = lambda path: None  # avoid touching disk twice
                return instance

            return _tracker

        monkeypatch.setattr(pg.exporters, exporter_cls_name, make_tracker(orig_cls))

    canvas._on_export()

    assert used_exporter_types == ["SVGExporter"]


def test_on_export_uses_image_exporter_for_png_path(qtbot, monkeypatch, tmp_path):
    """Non-SVG paths (e.g. .png) must still use ImageExporter."""
    import pyqtgraph as pg

    from rheojax.gui.compat import QFileDialog

    canvas = PyQtGraphCanvas()
    qtbot.addWidget(canvas)
    canvas.plot_data([1, 2, 3], [1, 4, 9], name="test")

    png_path = str(tmp_path / "plot.png")
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (png_path, "PNG Image (*.png)")
    )

    used_exporter_types = []
    for exporter_cls_name in ("SVGExporter", "ImageExporter"):
        orig_cls = getattr(pg.exporters, exporter_cls_name)

        def make_tracker(cls, name=exporter_cls_name):
            def _tracker(item):
                used_exporter_types.append(name)
                instance = cls(item)
                instance.export = lambda path: None
                return instance

            return _tracker

        monkeypatch.setattr(pg.exporters, exporter_cls_name, make_tracker(orig_cls))

    canvas._on_export()

    assert used_exporter_types == ["ImageExporter"]
