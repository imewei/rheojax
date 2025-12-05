"""
RheoJAX Performance Metrics Logging.

Utilities for timing, memory tracking, and iteration logging.
"""

import functools
import logging
import time
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from rheojax.logging.logger import RheoJAXLogger, get_logger

F = TypeVar("F", bound=Callable[..., Any])
LoggerType = logging.Logger | RheoJAXLogger | None


def timed(
    logger: LoggerType = None,
    level: int = logging.DEBUG,
    include_args: bool = False
) -> Callable[[F], F]:
    """Decorator to log function execution time.

    Args:
        logger: Logger to use. If None, uses function's module logger.
        level: Log level (default DEBUG).
        include_args: Include function arguments in log output.

    Returns:
        Decorator function.

    Example:
        >>> @timed()
        ... def compute_something(x, y):
        ...     return x + y

        >>> @timed(level=logging.INFO, include_args=True)
        ... def fit_model(data):
        ...     return model.fit(data)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start = time.perf_counter()

            extra = {
                "function": func.__name__,
                "module": func.__module__,
            }

            if include_args:
                # Only include serializable args (skip large arrays)
                safe_args = []
                for arg in args:
                    if hasattr(arg, "shape"):
                        safe_args.append(f"array{arg.shape}")
                    elif isinstance(arg, (str, int, float, bool, type(None))):
                        safe_args.append(arg)
                    else:
                        safe_args.append(type(arg).__name__)
                extra["args"] = safe_args
                extra["kwargs"] = {
                    k: v if isinstance(v, (str, int, float, bool, type(None)))
                    else type(v).__name__
                    for k, v in kwargs.items()
                }

            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start

                logger.log(
                    level,
                    f"{func.__name__} completed",
                    extra={
                        **extra,
                        "elapsed_seconds": round(elapsed, 6),
                        "elapsed_ms": round(elapsed * 1000, 3),
                        "status": "success",
                    }
                )
                return result

            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {e}",
                    extra={
                        **extra,
                        "elapsed_seconds": round(elapsed, 6),
                        "status": "error",
                        "error_type": type(e).__name__,
                    }
                )
                raise

        return wrapper  # type: ignore
    return decorator


@contextmanager
def log_memory(
    logger: LoggerType = None,
    operation: str = "operation",
    level: int = logging.DEBUG,
    trace_lines: bool = False
):
    """Context manager for tracking memory usage.

    Uses tracemalloc to measure memory allocation during an operation.

    Args:
        logger: Logger to use.
        operation: Name of operation being measured.
        level: Log level (default DEBUG).
        trace_lines: Include top memory-allocating lines.

    Yields:
        None

    Example:
        >>> with log_memory(logger, "large_computation"):
        ...     result = compute_large_array()
        DEBUG | rheojax.core | large_computation memory | current_mb=45.2 | peak_mb=128.5
    """
    actual_logger: logging.Logger | RheoJAXLogger = logger if logger is not None else get_logger("rheojax.metrics")

    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()

        extra = {
            "operation": operation,
            "current_mb": round(current / 1024 / 1024, 2),
            "peak_mb": round(peak / 1024 / 1024, 2),
        }

        if trace_lines:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:5]
            extra["top_allocations"] = [
                f"{stat.traceback.format()[0]}: {stat.size / 1024:.1f} KB"
                for stat in top_stats
            ]

        tracemalloc.stop()

        actual_logger.log(level, f"{operation} memory usage", extra=extra)


class IterationLogger:
    """Logger for optimization iterations with rate limiting.

    Logs iteration progress at configurable intervals to avoid
    flooding logs during long-running optimizations.

    Attributes:
        logger: Logger instance.
        log_every: Log every N iterations.
        level: Log level.
        iteration: Current iteration count.
        start_time: Time when logging started.

    Example:
        >>> iter_logger = IterationLogger(logger, log_every=100)
        >>> for i in range(1000):
        ...     cost = optimizer.step()
        ...     iter_logger.log(cost=cost, grad_norm=grad_norm)
        DEBUG | rheojax.opt | Iteration 100 | cost=0.0234 | grad_norm=0.001
        DEBUG | rheojax.opt | Iteration 200 | cost=0.0189 | grad_norm=0.0008
    """

    def __init__(
        self,
        logger: LoggerType = None,
        log_every: int = 10,
        level: int = logging.DEBUG,
        operation: str = "optimization"
    ) -> None:
        """Initialize the iteration logger.

        Args:
            logger: Logger instance (creates default if None).
            log_every: Log every N iterations (default 10).
            level: Log level (default DEBUG).
            operation: Operation name for log messages.
        """
        self.logger = logger or get_logger("rheojax.optimization")
        self.log_every = log_every
        self.level = level
        self.operation = operation
        self.iteration = 0
        self.start_time = time.perf_counter()
        self._last_cost: float | None = None

    def log(
        self,
        cost: float | None = None,
        force: bool = False,
        **metrics
    ) -> None:
        """Log iteration if at logging interval.

        Args:
            cost: Current cost/loss value.
            force: Force logging regardless of interval.
            **metrics: Additional metrics to log.
        """
        self.iteration += 1
        self._last_cost = cost

        if force or self.iteration % self.log_every == 0:
            elapsed = time.perf_counter() - self.start_time
            iter_per_sec = self.iteration / elapsed if elapsed > 0 else 0

            extra = {
                "iteration": self.iteration,
                "elapsed_seconds": round(elapsed, 3),
                "iterations_per_second": round(iter_per_sec, 2),
            }

            if cost is not None:
                extra["cost"] = cost

            extra.update(metrics)

            self.logger.log(
                self.level,
                f"Iteration {self.iteration}",
                extra=extra
            )

    def log_final(self, **metrics) -> None:
        """Log final iteration summary.

        Args:
            **metrics: Final metrics to include.
        """
        elapsed = time.perf_counter() - self.start_time
        iter_per_sec = self.iteration / elapsed if elapsed > 0 else 0

        extra = {
            "total_iterations": self.iteration,
            "total_elapsed_seconds": round(elapsed, 3),
            "average_iterations_per_second": round(iter_per_sec, 2),
        }

        if self._last_cost is not None:
            extra["final_cost"] = self._last_cost

        extra.update(metrics)

        self.logger.info(
            f"{self.operation} completed",
            extra=extra
        )

    def reset(self) -> None:
        """Reset iteration counter and timer."""
        self.iteration = 0
        self.start_time = time.perf_counter()
        self._last_cost = None


class ConvergenceTracker:
    """Track and log convergence metrics for optimization.

    Monitors cost progression and determines when convergence
    criteria are met.

    Example:
        >>> tracker = ConvergenceTracker(logger, tolerance=1e-6)
        >>> for i in range(1000):
        ...     cost = optimizer.step()
        ...     if tracker.update(cost):
        ...         print("Converged!")
        ...         break
    """

    def __init__(
        self,
        logger: LoggerType = None,
        tolerance: float = 1e-6,
        patience: int = 5,
        min_iterations: int = 10
    ) -> None:
        """Initialize the convergence tracker.

        Args:
            logger: Logger instance.
            tolerance: Convergence tolerance for cost improvement.
            patience: Number of iterations with small improvement before converged.
            min_iterations: Minimum iterations before convergence check.
        """
        self.logger = logger or get_logger("rheojax.optimization")
        self.tolerance = tolerance
        self.patience = patience
        self.min_iterations = min_iterations
        self.history: list[float] = []
        self._small_improvement_count = 0

    def update(self, cost: float) -> bool:
        """Update with new cost and check for convergence.

        Args:
            cost: Current cost/loss value.

        Returns:
            True if convergence criteria met.
        """
        self.history.append(cost)

        if len(self.history) < self.min_iterations:
            return False

        # Check improvement
        if len(self.history) >= 2:
            improvement = abs(self.history[-2] - self.history[-1])

            if improvement < self.tolerance:
                self._small_improvement_count += 1
            else:
                self._small_improvement_count = 0

            if self._small_improvement_count >= self.patience:
                self.logger.info(
                    "Convergence achieved",
                    extra={
                        "final_cost": cost,
                        "iterations": len(self.history),
                        "last_improvement": improvement,
                        "tolerance": self.tolerance,
                    }
                )
                return True

        return False

    def reset(self) -> None:
        """Reset the tracker."""
        self.history.clear()
        self._small_improvement_count = 0

    @property
    def improvement_rate(self) -> float | None:
        """Calculate average improvement rate.

        Returns:
            Average cost reduction per iteration, or None if insufficient data.
        """
        if len(self.history) < 2:
            return None
        return (self.history[0] - self.history[-1]) / len(self.history)
