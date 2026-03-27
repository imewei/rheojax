"""Logging utilities tests for RheoJAX GUI."""

import logging

import pytest

from rheojax.gui.utils.logging import install_gui_log_handler

pytestmark = pytest.mark.gui


@pytest.mark.flaky(reruns=2)
def test_install_gui_log_handler_forwards_records() -> None:
    """Log records should reach the GUI append callback."""
    from PySide6.QtWidgets import QApplication

    captured: list[str] = []
    handler = install_gui_log_handler(captured.append, level=logging.DEBUG)

    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

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

    assert any("hello gui logger" in line for line in captured)
