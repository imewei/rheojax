"""Logging utilities tests for RheoJAX GUI."""

import logging

import pytest

from rheojax.gui.utils.logging import install_gui_log_handler

pytestmark = pytest.mark.gui


def test_install_gui_log_handler_forwards_records() -> None:
    """Log records should reach the GUI append callback."""
    from PySide6.QtWidgets import QApplication

    captured: list[str] = []
    handler = install_gui_log_handler(captured.append, level=logging.DEBUG)

    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

    # Temporarily remove stream handlers to avoid "I/O operation on closed
    # file" errors when xdist workers close stderr.
    _removed: list[logging.Handler] = []
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler) and h is not handler:
            root_logger.removeHandler(h)
            _removed.append(h)

    logger = logging.getLogger("rheojax.gui.logging_test")
    logger.setLevel(logging.DEBUG)

    try:
        logger.info("hello gui logger")
        # Qt signal delivery may be queued — flush the event loop.
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    finally:
        logging.getLogger().removeHandler(handler)
        root_logger.setLevel(original_level)
        for h in _removed:
            root_logger.addHandler(h)

    assert any("hello gui logger" in line for line in captured)
