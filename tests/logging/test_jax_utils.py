"""Tests for rheojax.logging.jax_utils module."""

import logging
from unittest import mock

import numpy as np
import pytest

from rheojax.logging.config import reset_config
from rheojax.logging.jax_utils import (
    log_array_info,
    log_array_stats,
    log_device_transfer,
    log_jax_config,
    log_numerical_issue,
)
from rheojax.logging.logger import clear_logger_cache


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before and after each test."""
    clear_logger_cache()
    reset_config()
    yield
    clear_logger_cache()
    reset_config()


class TestLogArrayInfo:
    """Tests for log_array_info function."""

    def test_extracts_shape(self):
        """Test that shape is extracted."""
        arr = np.random.randn(100, 50)
        info = log_array_info(arr, "test")
        assert info["test_shape"] == (100, 50)

    def test_extracts_dtype(self):
        """Test that dtype is extracted."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        info = log_array_info(arr, "test")
        assert info["test_dtype"] == "float32"

    def test_extracts_size(self):
        """Test that size is extracted."""
        arr = np.random.randn(100, 50)
        info = log_array_info(arr, "test")
        assert info["test_size"] == 5000

    def test_custom_name(self):
        """Test that custom name is used in keys."""
        arr = np.random.randn(10)
        info = log_array_info(arr, "my_array")
        assert "my_array_shape" in info
        assert "my_array_dtype" in info


class TestLogArrayStats:
    """Tests for log_array_stats function."""

    def test_computes_statistics(self):
        """Test that statistics are computed."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        info = log_array_stats(arr, "test")

        assert info["test_min"] == 1.0
        assert info["test_max"] == 5.0
        assert info["test_mean"] == 3.0
        assert info["test_has_nan"] is False
        assert info["test_has_inf"] is False

    def test_detects_nan(self):
        """Test that NaN is detected."""
        arr = np.array([1.0, np.nan, 3.0])
        info = log_array_stats(arr, "test")
        assert info["test_has_nan"] is True

    def test_detects_inf(self):
        """Test that Inf is detected."""
        arr = np.array([1.0, np.inf, 3.0])
        # Suppress warning from np.std when array contains inf
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered")
            info = log_array_stats(arr, "test")
        assert info["test_has_inf"] is True

    def test_logs_immediately_if_logger_provided(self):
        """Test that stats are logged if logger provided."""
        logger = mock.MagicMock(spec=logging.Logger)
        arr = np.array([1.0, 2.0, 3.0])

        log_array_stats(arr, "test", logger=logger)

        logger.log.assert_called_once()


class TestLogNumericalIssue:
    """Tests for log_numerical_issue function."""

    def test_returns_false_for_clean_array(self):
        """Test that clean arrays return False."""
        logger = mock.MagicMock(spec=logging.Logger)
        arr = np.array([1.0, 2.0, 3.0])

        result = log_numerical_issue(logger, arr, "test")

        assert result is False
        logger.warning.assert_not_called()

    def test_returns_true_for_nan(self):
        """Test that NaN arrays return True."""
        logger = mock.MagicMock(spec=logging.Logger)
        arr = np.array([1.0, np.nan, 3.0])

        result = log_numerical_issue(logger, arr, "test")

        assert result is True
        logger.warning.assert_called()

    def test_returns_true_for_inf(self):
        """Test that Inf arrays return True."""
        logger = mock.MagicMock(spec=logging.Logger)
        arr = np.array([1.0, np.inf, 3.0])

        result = log_numerical_issue(logger, arr, "test")

        assert result is True
        logger.warning.assert_called()

    def test_includes_context(self):
        """Test that context is included in log."""
        logger = mock.MagicMock(spec=logging.Logger)
        arr = np.array([1.0, np.nan, 3.0])

        log_numerical_issue(logger, arr, "residuals", "during fitting")

        call_args = logger.warning.call_args
        extra = call_args.kwargs.get("extra", {})
        assert extra.get("context") == "during fitting"
        assert extra.get("array_name") == "residuals"


class TestLogJaxConfig:
    """Tests for log_jax_config function."""

    def test_returns_config_dict(self):
        """Test that config dict is returned."""
        config = log_jax_config()

        assert isinstance(config, dict)
        assert "jax_version" in config
        assert "default_backend" in config
        assert "float64_enabled" in config

    def test_logs_if_logger_provided(self):
        """Test that config is logged if logger provided."""
        logger = mock.MagicMock(spec=logging.Logger)

        log_jax_config(logger)

        logger.info.assert_called()


class TestLogDeviceTransfer:
    """Tests for log_device_transfer function."""

    def test_logs_transfer_info(self):
        """Test that transfer info is logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        arr = np.random.randn(100, 100)

        log_device_transfer(logger, arr, "result", "host")

        logger.debug.assert_called()
        call_args = logger.debug.call_args
        extra = call_args.kwargs.get("extra", {})
        assert extra.get("array_name") == "result"
        assert extra.get("target") == "host"
        assert "size_mb" in extra
