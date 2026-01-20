"""Tests for rheojax.logging.metrics module."""

import logging
import time
from unittest import mock

import pytest

from rheojax.logging.config import reset_config
from rheojax.logging.logger import clear_logger_cache
from rheojax.logging.metrics import (
    ConvergenceTracker,
    IterationLogger,
    log_memory,
    timed,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before and after each test."""
    clear_logger_cache()
    reset_config()
    yield
    clear_logger_cache()
    reset_config()


class TestTimedDecorator:
    """Tests for timed decorator."""

    def test_timed_logs_completion(self):
        """Test that timed decorator logs completion."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = True

        @timed(logger=logger, level=logging.DEBUG)
        def sample_function():
            return 42

        result = sample_function()

        assert result == 42
        logger.log.assert_called()

    def test_timed_logs_elapsed_time(self):
        """Test that elapsed time is logged."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = True

        @timed(logger=logger, level=logging.DEBUG)
        def sample_function():
            time.sleep(0.01)
            return 42

        sample_function()

        call_args = logger.log.call_args
        extra = call_args.kwargs.get("extra", {})
        assert "elapsed_seconds" in extra
        assert extra["elapsed_seconds"] > 0

    def test_timed_logs_error_on_exception(self):
        """Test that errors are logged on exception."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = True

        @timed(logger=logger)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        logger.error.assert_called()

    def test_timed_preserves_function_name(self):
        """Test that function name is preserved."""
        logger = mock.MagicMock(spec=logging.Logger)

        @timed(logger=logger)
        def named_function():
            pass

        assert named_function.__name__ == "named_function"

    def test_timed_with_include_args(self):
        """Test that arguments can be included in logs."""
        logger = mock.MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = True

        @timed(logger=logger, include_args=True)
        def sample_function(x, y=10):
            return x + y

        sample_function(5, y=20)

        call_args = logger.log.call_args
        extra = call_args.kwargs.get("extra", {})
        assert "args" in extra
        assert "kwargs" in extra


class TestLogMemory:
    """Tests for log_memory context manager."""

    def test_log_memory_tracks_allocation(self):
        """Test that memory allocation is tracked."""
        logger = mock.MagicMock(spec=logging.Logger)

        with log_memory(logger, "test_operation"):
            # Allocate some memory
            data = [i for i in range(10000)]

        logger.log.assert_called()
        call_args = logger.log.call_args
        extra = call_args.kwargs.get("extra", {})
        assert "current_mb" in extra
        assert "peak_mb" in extra

    def test_log_memory_includes_operation_name(self):
        """Test that operation name is included."""
        logger = mock.MagicMock(spec=logging.Logger)

        with log_memory(logger, "my_operation"):
            pass

        call_args = logger.log.call_args
        extra = call_args.kwargs.get("extra", {})
        assert extra.get("operation") == "my_operation"


class TestIterationLogger:
    """Tests for IterationLogger class."""

    def test_logs_at_interval(self):
        """Test that logging occurs at specified interval."""
        logger = mock.MagicMock(spec=logging.Logger)
        iter_logger = IterationLogger(logger, log_every=5)

        for i in range(12):
            iter_logger.log(cost=1.0 / (i + 1))

        # Should log at iterations 5 and 10
        assert logger.log.call_count == 2

    def test_tracks_iteration_count(self):
        """Test that iteration count is tracked."""
        logger = mock.MagicMock(spec=logging.Logger)
        iter_logger = IterationLogger(logger, log_every=5)

        for i in range(7):
            iter_logger.log()

        assert iter_logger.iteration == 7

    def test_force_logs_immediately(self):
        """Test that force=True logs immediately."""
        logger = mock.MagicMock(spec=logging.Logger)
        iter_logger = IterationLogger(logger, log_every=100)

        iter_logger.log(force=True)

        assert logger.log.call_count == 1

    def test_log_final_uses_info_level(self):
        """Test that log_final uses INFO level."""
        logger = mock.MagicMock(spec=logging.Logger)
        iter_logger = IterationLogger(logger)

        for i in range(5):
            iter_logger.log(cost=1.0)

        iter_logger.log_final(status="converged")

        logger.info.assert_called()

    def test_reset_clears_state(self):
        """Test that reset clears iteration count."""
        logger = mock.MagicMock(spec=logging.Logger)
        iter_logger = IterationLogger(logger)

        for i in range(10):
            iter_logger.log()

        assert iter_logger.iteration == 10

        iter_logger.reset()
        assert iter_logger.iteration == 0


class TestConvergenceTracker:
    """Tests for ConvergenceTracker class."""

    def test_detects_convergence(self):
        """Test that convergence is detected."""
        logger = mock.MagicMock(spec=logging.Logger)
        tracker = ConvergenceTracker(
            logger, tolerance=1e-3, patience=3, min_iterations=5  # Looser tolerance
        )

        # Costs with clear convergence pattern - last 4 values are nearly identical
        costs = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0625, 0.0625, 0.0625]
        converged = False

        for cost in costs:
            if tracker.update(cost):
                converged = True
                break

        assert converged
        logger.info.assert_called()

    def test_requires_min_iterations(self):
        """Test that convergence requires minimum iterations."""
        logger = mock.MagicMock(spec=logging.Logger)
        tracker = ConvergenceTracker(
            logger, tolerance=1e-4, patience=1, min_iterations=10
        )

        # Same cost for 5 iterations shouldn't converge
        for _ in range(5):
            result = tracker.update(0.0)

        assert result is False

    def test_reset_clears_history(self):
        """Test that reset clears history."""
        logger = mock.MagicMock(spec=logging.Logger)
        tracker = ConvergenceTracker(logger)

        for i in range(10):
            tracker.update(1.0 / (i + 1))

        assert len(tracker.history) == 10

        tracker.reset()
        assert len(tracker.history) == 0

    def test_improvement_rate(self):
        """Test improvement rate calculation."""
        logger = mock.MagicMock(spec=logging.Logger)
        tracker = ConvergenceTracker(logger)

        costs = [1.0, 0.8, 0.6, 0.4, 0.2]
        for cost in costs:
            tracker.update(cost)

        rate = tracker.improvement_rate
        assert rate is not None
        assert rate > 0  # Cost is decreasing

    def test_improvement_rate_none_for_insufficient_data(self):
        """Test improvement rate returns None with insufficient data."""
        logger = mock.MagicMock(spec=logging.Logger)
        tracker = ConvergenceTracker(logger)

        tracker.update(1.0)

        assert tracker.improvement_rate is None
