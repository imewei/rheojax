"""Logging utilities tests for RheoJAX GUI."""

import logging

import pytest

from rheojax.gui.utils.logging import install_gui_log_handler

pytestmark = pytest.mark.gui


def test_install_gui_log_handler_forwards_records() -> None:
    """Log records should reach the GUI append callback."""
    captured: list[str] = []
    handler = install_gui_log_handler(captured.append, level=logging.DEBUG)

    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

    try:
        # Log directly on root logger to avoid structlog interception
        # on child loggers that may be configured in the full test suite.
        root_logger.info("hello gui logger")
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)

    assert any("hello gui logger" in line for line in captured)
