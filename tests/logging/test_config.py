"""Tests for rheojax.logging.config module."""

import logging
import os
from pathlib import Path
from unittest import mock

import pytest

from rheojax.logging.config import (
    DEFAULT_SUBSYSTEM_LEVELS,
    LogConfig,
    LogFormat,
    configure_logging,
    get_config,
    is_configured,
    reset_config,
)


@pytest.fixture(autouse=True)
def reset_logging_config():
    """Reset logging configuration before and after each test."""
    reset_config()
    yield
    reset_config()


class TestLogFormat:
    """Tests for LogFormat enum."""

    def test_log_format_values(self):
        """Test that all expected formats exist."""
        assert LogFormat.STANDARD.value == "standard"
        assert LogFormat.DETAILED.value == "detailed"
        assert LogFormat.JSON.value == "json"
        assert LogFormat.SCIENTIFIC.value == "scientific"

    def test_log_format_from_string(self):
        """Test creating LogFormat from string."""
        assert LogFormat("standard") == LogFormat.STANDARD
        assert LogFormat("json") == LogFormat.JSON


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.format == LogFormat.STANDARD
        assert config.console is True
        assert config.file is None
        assert config.file_max_bytes == 10_000_000
        assert config.file_backup_count == 5
        assert config.lazy_formatting is True
        assert config.colorize is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LogConfig(
            level="DEBUG",
            format=LogFormat.JSON,
            file=Path("/tmp/test.log"),
            colorize=False,
        )
        assert config.level == "DEBUG"
        assert config.format == LogFormat.JSON
        assert config.file == Path("/tmp/test.log")
        assert config.colorize is False

    def test_string_format_conversion(self):
        """Test that string format is converted to enum."""
        config = LogConfig(format="json")
        assert config.format == LogFormat.JSON

    def test_string_path_conversion(self):
        """Test that string path is converted to Path."""
        config = LogConfig(file="/tmp/test.log")
        assert config.file == Path("/tmp/test.log")

    def test_invalid_level_raises(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            LogConfig(level="INVALID")

    def test_from_env_defaults(self):
        """Test from_env with no environment variables."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = LogConfig.from_env()
            assert config.level == "INFO"
            assert config.format == LogFormat.STANDARD
            assert config.file is None

    def test_from_env_custom_level(self):
        """Test from_env with custom log level."""
        with mock.patch.dict(os.environ, {"RHEOJAX_LOG_LEVEL": "DEBUG"}):
            config = LogConfig.from_env()
            assert config.level == "DEBUG"

    def test_from_env_custom_file(self):
        """Test from_env with custom log file."""
        with mock.patch.dict(os.environ, {"RHEOJAX_LOG_FILE": "/tmp/rheojax.log"}):
            config = LogConfig.from_env()
            assert config.file == Path("/tmp/rheojax.log")

    def test_from_env_custom_format(self):
        """Test from_env with custom format."""
        with mock.patch.dict(os.environ, {"RHEOJAX_LOG_FORMAT": "json"}):
            config = LogConfig.from_env()
            assert config.format == LogFormat.JSON

    def test_from_env_subsystem_levels(self):
        """Test from_env with per-subsystem levels."""
        with mock.patch.dict(os.environ, {"RHEOJAX_LOG_MODELS": "DEBUG"}):
            config = LogConfig.from_env()
            assert config.subsystem_levels["rheojax.models"] == "DEBUG"

    def test_get_level_exact_match(self):
        """Test get_level with exact subsystem match."""
        config = LogConfig(
            level="INFO",
            subsystem_levels={"rheojax.models": "DEBUG"}
        )
        assert config.get_level("rheojax.models") == logging.DEBUG

    def test_get_level_parent_match(self):
        """Test get_level with parent subsystem match."""
        config = LogConfig(
            level="INFO",
            subsystem_levels={"rheojax.models": "DEBUG"}
        )
        assert config.get_level("rheojax.models.maxwell") == logging.DEBUG

    def test_get_level_fallback(self):
        """Test get_level falls back to global level."""
        config = LogConfig(level="WARNING")
        assert config.get_level("rheojax.unknown") == logging.WARNING


class TestConfigureFunctions:
    """Tests for configuration functions."""

    def test_configure_logging_basic(self):
        """Test basic configure_logging call."""
        config = configure_logging(level="DEBUG")
        assert config.level == "DEBUG"
        assert is_configured() is True

    def test_configure_logging_with_file(self, tmp_path):
        """Test configure_logging with file output."""
        log_file = tmp_path / "test.log"
        config = configure_logging(level="INFO", file=str(log_file))
        assert config.file == log_file

    def test_get_config_returns_default(self):
        """Test get_config returns default when not configured."""
        config = get_config()
        assert isinstance(config, LogConfig)

    def test_is_configured_false_initially(self):
        """Test is_configured returns False initially."""
        assert is_configured() is False

    def test_reset_config(self):
        """Test reset_config clears configuration."""
        configure_logging(level="DEBUG")
        assert is_configured() is True
        reset_config()
        assert is_configured() is False
