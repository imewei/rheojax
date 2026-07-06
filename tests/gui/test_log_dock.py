"""Tests for the shared GUI log dock widget."""

import logging

import pytest

pytestmark = pytest.mark.gui


def test_append_record_buffers_and_renders(qtbot):
    from rheojax.gui.widgets.log_dock import LogDockWidget

    dock = LogDockWidget()
    qtbot.addWidget(dock)

    dock.append_record(logging.INFO, "hello world")

    assert "hello world" in dock.text_edit.toPlainText()
    assert len(dock._records) == 1


def test_level_filter_hides_below_threshold(qtbot):
    from rheojax.gui.widgets.log_dock import LogDockWidget

    dock = LogDockWidget()
    qtbot.addWidget(dock)

    dock.append_record(logging.DEBUG, "debug message")
    dock.append_record(logging.ERROR, "error message")

    # Buffer keeps both regardless of filter.
    assert len(dock._records) == 2
    # Default filter is INFO, so the DEBUG line should not render.
    assert "debug message" not in dock.text_edit.toPlainText()
    assert "error message" in dock.text_edit.toPlainText()

    dock.level_combo.setCurrentText("DEBUG")
    assert "debug message" in dock.text_edit.toPlainText()


def test_buffer_is_bounded(qtbot):
    from rheojax.gui.widgets.log_dock import LogDockWidget

    dock = LogDockWidget()
    qtbot.addWidget(dock)

    for i in range(2100):
        dock.append_record(logging.INFO, f"line {i}")

    assert len(dock._records) == 2000
    assert dock._records[0][1] == "line 100"
