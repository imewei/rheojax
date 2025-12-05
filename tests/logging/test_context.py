"""Tests for rheojax.logging.context module."""

import logging
from unittest import mock

import pytest

from rheojax.logging.config import configure_logging, reset_config
from rheojax.logging.context import (
    log_bayesian,
    log_fit,
    log_io,
    log_operation,
    log_pipeline_stage,
    log_transform,
)
from rheojax.logging.logger import get_logger, clear_logger_cache


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before and after each test."""
    clear_logger_cache()
    reset_config()
    yield
    clear_logger_cache()
    reset_config()


class TestLogOperation:
    """Tests for log_operation context manager."""

    def test_logs_start_and_end(self):
        """Test that start and end are logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_operation(logger, "test_operation"):
            pass

        # Should have 2 log calls (start and end)
        assert logger.log.call_count == 2

    def test_logs_elapsed_time(self):
        """Test that elapsed time is logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_operation(logger, "test_operation"):
            pass

        # Check end log has elapsed_seconds
        end_call = logger.log.call_args_list[-1]
        extra = end_call.kwargs.get("extra", {})
        assert "elapsed_seconds" in extra

    def test_logs_exception_on_error(self):
        """Test that exceptions are logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with pytest.raises(ValueError):
            with log_operation(logger, "test_operation"):
                raise ValueError("Test error")

        # Should have error logged
        logger.error.assert_called_once()

    def test_context_dict_available(self):
        """Test that context dict can be updated."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_operation(logger, "test_operation") as ctx:
            ctx["result"] = "success"

        # Check end log has the context
        end_call = logger.log.call_args_list[-1]
        extra = end_call.kwargs.get("extra", {})
        assert extra.get("result") == "success"

    def test_custom_level(self):
        """Test that custom log level is used."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_operation(logger, "test_operation", level=logging.DEBUG):
            pass

        # Both calls should be DEBUG level
        for call in logger.log.call_args_list:
            assert call[0][0] == logging.DEBUG

    def test_additional_context(self):
        """Test that additional context is included."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_operation(logger, "test_operation", model="Maxwell"):
            pass

        # Check start log has model
        start_call = logger.log.call_args_list[0]
        extra = start_call.kwargs.get("extra", {})
        assert extra.get("model") == "Maxwell"


class TestLogFit:
    """Tests for log_fit context manager."""

    def test_logs_fit_context(self):
        """Test that fit-specific context is logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_fit(
            logger,
            model="Maxwell",
            data_shape=(100,),
            test_mode="relaxation"
        ):
            pass

        # Check context
        start_call = logger.log.call_args_list[0]
        extra = start_call.kwargs.get("extra", {})
        assert extra.get("model") == "Maxwell"
        assert extra.get("data_shape") == (100,)
        assert extra.get("test_mode") == "relaxation"

    def test_fit_completion_context(self):
        """Test that completion context is included."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_fit(logger, model="Maxwell") as ctx:
            ctx["R2"] = 0.9987
            ctx["n_iterations"] = 42

        end_call = logger.log.call_args_list[-1]
        extra = end_call.kwargs.get("extra", {})
        assert extra.get("R2") == 0.9987
        assert extra.get("n_iterations") == 42


class TestLogBayesian:
    """Tests for log_bayesian context manager."""

    def test_logs_bayesian_context(self):
        """Test that Bayesian-specific context is logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_bayesian(
            logger,
            model="Maxwell",
            num_warmup=1000,
            num_samples=2000,
            num_chains=4
        ):
            pass

        start_call = logger.log.call_args_list[0]
        extra = start_call.kwargs.get("extra", {})
        assert extra.get("num_warmup") == 1000
        assert extra.get("num_samples") == 2000
        assert extra.get("num_chains") == 4

    def test_bayesian_completion_context(self):
        """Test that MCMC diagnostics can be added."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_bayesian(logger, "Maxwell", 1000, 2000) as ctx:
            ctx["r_hat_max"] = 1.002
            ctx["ess_min"] = 450

        end_call = logger.log.call_args_list[-1]
        extra = end_call.kwargs.get("extra", {})
        assert extra.get("r_hat_max") == 1.002


class TestLogTransform:
    """Tests for log_transform context manager."""

    def test_logs_transform_context(self):
        """Test that transform-specific context is logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_transform(
            logger,
            transform="mastercurve",
            input_shape=(10, 100)
        ):
            pass

        start_call = logger.log.call_args_list[0]
        extra = start_call.kwargs.get("extra", {})
        assert extra.get("transform") == "mastercurve"
        assert extra.get("input_shape") == (10, 100)


class TestLogIO:
    """Tests for log_io context manager."""

    def test_logs_io_context(self):
        """Test that I/O-specific context is logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_io(
            logger,
            operation="read",
            filepath="/path/to/file.csv"
        ):
            pass

        start_call = logger.log.call_args_list[0]
        extra = start_call.kwargs.get("extra", {})
        assert extra.get("io_operation") == "read"
        assert extra.get("filepath") == "/path/to/file.csv"


class TestLogPipelineStage:
    """Tests for log_pipeline_stage context manager."""

    def test_logs_pipeline_context(self):
        """Test that pipeline-specific context is logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.name = "test"

        with log_pipeline_stage(
            logger,
            stage="fit",
            pipeline_id="pipe_001"
        ):
            pass

        start_call = logger.log.call_args_list[0]
        extra = start_call.kwargs.get("extra", {})
        assert extra.get("stage") == "fit"
        assert extra.get("pipeline_id") == "pipe_001"
