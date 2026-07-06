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


def test_message_with_angle_brackets_is_not_corrupted(qtbot):
    from rheojax.gui.widgets.log_dock import LogDockWidget

    dock = LogDockWidget()
    qtbot.addWidget(dock)

    dock.append_record(logging.ERROR, "Solver failed: <lambda> at 0x7f0000")

    assert "<lambda>" in dock.text_edit.toPlainText()


def test_save_writes_full_buffer_not_filtered_view(qtbot, monkeypatch, tmp_path):
    from rheojax.gui.compat import QFileDialog
    from rheojax.gui.widgets.log_dock import LogDockWidget

    dock = LogDockWidget()
    qtbot.addWidget(dock)

    dock.append_record(logging.DEBUG, "debug message")
    dock.append_record(logging.ERROR, "error message")
    dock.level_combo.setCurrentText("ERROR")  # hides the DEBUG line on screen

    out_path = tmp_path / "saved.log"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), "")
    )
    dock._on_save_clicked()

    saved = out_path.read_text(encoding="utf-8")
    assert "debug message" in saved
    assert "error message" in saved
