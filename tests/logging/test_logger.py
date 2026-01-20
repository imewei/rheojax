"""Tests for rheojax.logging.logger module."""

import logging
from unittest import mock

import pytest

from rheojax.logging.config import reset_config
from rheojax.logging.logger import (
    RheoJAXLogger,
    clear_logger_cache,
    get_logger,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before and after each test."""
    clear_logger_cache()
    reset_config()
    yield
    clear_logger_cache()
    reset_config()


class TestRheoJAXLogger:
    """Tests for RheoJAXLogger class."""

    def test_logger_creation(self):
        """Test basic logger creation."""
        base_logger = logging.getLogger("test.logger")
        logger = RheoJAXLogger(base_logger)
        assert logger.logger is base_logger

    def test_logger_with_extra(self):
        """Test logger with default extra context."""
        base_logger = logging.getLogger("test.logger")
        logger = RheoJAXLogger(base_logger, {"model": "Maxwell"})
        assert logger.extra == {"model": "Maxwell"}

    def test_bind_creates_new_logger(self):
        """Test bind creates new logger with merged context."""
        base_logger = logging.getLogger("test.logger")
        logger = RheoJAXLogger(base_logger, {"model": "Maxwell"})
        bound_logger = logger.bind(test_mode="relaxation")

        # Original unchanged
        assert logger.extra == {"model": "Maxwell"}
        # New logger has merged context
        assert bound_logger.extra == {"model": "Maxwell", "test_mode": "relaxation"}

    def test_bind_overwrites_existing(self):
        """Test bind overwrites existing context keys."""
        base_logger = logging.getLogger("test.logger")
        logger = RheoJAXLogger(base_logger, {"model": "Maxwell"})
        bound_logger = logger.bind(model="Zener")

        assert bound_logger.extra == {"model": "Zener"}

    def test_process_extracts_kwargs(self):
        """Test process extracts non-standard kwargs to extra."""
        base_logger = logging.getLogger("test.logger")
        logger = RheoJAXLogger(base_logger)

        msg, kwargs = logger.process("Test message", {"R2": 0.998, "iterations": 100})

        assert msg == "Test message"
        assert "extra" in kwargs
        assert kwargs["extra"]["extra"]["R2"] == 0.998
        assert kwargs["extra"]["extra"]["iterations"] == 100


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_basic(self):
        """Test basic get_logger call."""
        logger = get_logger("test.module")
        assert isinstance(logger, RheoJAXLogger)
        assert logger.logger.name == "test.module"

    def test_get_logger_with_context(self):
        """Test get_logger with initial context."""
        logger = get_logger("test.module", model="Maxwell")
        assert logger.extra == {"model": "Maxwell"}

    def test_get_logger_caches(self):
        """Test get_logger returns cached instance."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test get_logger creates different loggers for different names."""
        logger1 = get_logger("test.module1")
        logger2 = get_logger("test.module2")
        assert logger1 is not logger2

    def test_get_logger_different_context(self):
        """Test get_logger creates different loggers for different context."""
        logger1 = get_logger("test.module", model="Maxwell")
        logger2 = get_logger("test.module", model="Zener")
        assert logger1 is not logger2


class TestLoggerMethods:
    """Tests for logger method calls."""

    @pytest.fixture
    def mock_logger(self):
        """Create a logger with mocked underlying logger."""
        base_logger = mock.MagicMock(spec=logging.Logger)
        base_logger.name = "test.logger"
        return RheoJAXLogger(base_logger)

    def test_debug_calls_log(self, mock_logger):
        """Test debug method calls log with DEBUG level."""
        mock_logger.debug("Test message")
        mock_logger.logger.log.assert_called()
        call_args = mock_logger.logger.log.call_args
        assert call_args[0][0] == logging.DEBUG

    def test_info_calls_log(self, mock_logger):
        """Test info method calls log with INFO level."""
        mock_logger.info("Test message")
        mock_logger.logger.log.assert_called()
        call_args = mock_logger.logger.log.call_args
        assert call_args[0][0] == logging.INFO

    def test_warning_calls_log(self, mock_logger):
        """Test warning method calls log with WARNING level."""
        mock_logger.warning("Test message")
        mock_logger.logger.log.assert_called()
        call_args = mock_logger.logger.log.call_args
        assert call_args[0][0] == logging.WARNING

    def test_error_calls_log(self, mock_logger):
        """Test error method calls log with ERROR level."""
        mock_logger.error("Test message")
        mock_logger.logger.log.assert_called()
        call_args = mock_logger.logger.log.call_args
        assert call_args[0][0] == logging.ERROR

    def test_critical_calls_log(self, mock_logger):
        """Test critical method calls log with CRITICAL level."""
        mock_logger.critical("Test message")
        mock_logger.logger.log.assert_called()
        call_args = mock_logger.logger.log.call_args
        assert call_args[0][0] == logging.CRITICAL


class TestClearLoggerCache:
    """Tests for clear_logger_cache function."""

    def test_clear_logger_cache(self):
        """Test clear_logger_cache clears the cache."""
        logger1 = get_logger("test.module")
        clear_logger_cache()
        logger2 = get_logger("test.module")
        # After clearing, should get a new instance
        assert logger1 is not logger2
