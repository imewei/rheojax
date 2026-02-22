"""
RheoJAX Logging Context Managers.

Context managers for automatically logging operation start/end,
timing, and exception handling.

Includes specialized context managers for:
- General operations (log_operation)
- Model fitting (log_fit)
- Bayesian inference (log_bayesian)
- Data transforms (log_transform)
- File I/O (log_io)
- Pipeline stages (log_pipeline_stage)
- GUI user actions (log_gui_action)
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rheojax.logging.logger import RheoJAXLogger


@contextmanager
def log_operation(
    logger: logging.Logger | RheoJAXLogger,
    operation: str,
    level: int = logging.INFO,
    **context,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for logging operation start/end with timing.

    Automatically logs when an operation starts and completes,
    including elapsed time and any exceptions that occur.

    Args:
        logger: Logger instance to use.
        operation: Name of the operation being performed.
        level: Log level for start/end messages (default INFO).
        **context: Additional context to include in log messages.

    Yields:
        Dictionary that can be used to add additional context
        to the completion log message.

    Example:
        >>> with log_operation(logger, "fitting", model="Maxwell"):
        ...     result = model.fit(x, y)
        14:32:05 | INFO | rheojax.models | fitting started | model=Maxwell
        14:32:07 | INFO | rheojax.models | fitting completed | model=Maxwell | elapsed_seconds=2.15

    Example with additional context:
        >>> with log_operation(logger, "fitting", model="Maxwell") as ctx:
        ...     result = model.fit(x, y)
        ...     ctx["R2"] = result.r_squared
        14:32:05 | INFO | rheojax.models | fitting started | model=Maxwell
        14:32:07 | INFO | rheojax.models | fitting completed | model=Maxwell | R2=0.9987 | elapsed_seconds=2.15
    """
    start_time = time.perf_counter()
    completion_context: dict[str, Any] = {}

    # Log start
    logger.log(
        level,
        f"{operation} started",
        extra={"operation": operation, "phase": "start", **context},
    )

    try:
        yield completion_context
        elapsed = time.perf_counter() - start_time

        # Log successful completion
        logger.log(
            level,
            f"{operation} completed",
            extra={
                "operation": operation,
                "phase": "end",
                "elapsed_seconds": round(elapsed, 4),
                "status": "success",
                **context,
                **completion_context,
            },
        )

    except Exception as e:
        elapsed = time.perf_counter() - start_time

        # Log failure (wrapped to avoid masking the original exception)
        try:
            logger.error(
                f"{operation} failed: {e}",
                extra={
                    "operation": operation,
                    "phase": "end",
                    "elapsed_seconds": round(elapsed, 4),
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    **context,
                    **completion_context,
                },
            )
        except Exception:
            pass  # Never mask the original exception
        raise


@contextmanager
def log_fit(
    logger: logging.Logger | RheoJAXLogger,
    model: str,
    data_shape: tuple[int, ...] | None = None,
    test_mode: str = "unknown",
    level: int = logging.INFO,
    **kwargs,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for model fitting operations.

    Specialized wrapper around log_operation for model fitting.

    Args:
        logger: Logger instance to use.
        model: Model name or class name.
        data_shape: Shape of input data (optional).
        test_mode: Test mode (relaxation, creep, oscillation, flow).
        level: Log level (default INFO).
        **kwargs: Additional context.

    Yields:
        Dictionary for adding completion context (e.g., R2, parameters).

    Example:
        >>> with log_fit(logger, "Maxwell", data_shape=(100,), test_mode="relaxation") as ctx:
        ...     result = model._fit(x, y)
        ...     ctx["R2"] = result.r_squared
        ...     ctx["n_iterations"] = result.iterations
    """
    context = {"model": model, "test_mode": test_mode, **kwargs}
    if data_shape is not None:
        context["data_shape"] = data_shape

    with log_operation(logger, "model_fit", level=level, **context) as ctx:
        yield ctx


@contextmanager
def log_bayesian(
    logger: logging.Logger | RheoJAXLogger,
    model: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int = 1,
    level: int = logging.INFO,
    **kwargs,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for Bayesian inference operations.

    Specialized wrapper for MCMC sampling operations.

    Args:
        logger: Logger instance to use.
        model: Model name.
        num_warmup: Number of warmup samples.
        num_samples: Number of posterior samples.
        num_chains: Number of MCMC chains.
        level: Log level (default INFO).
        **kwargs: Additional context.

    Yields:
        Dictionary for adding completion context (e.g., R-hat, ESS).

    Example:
        >>> with log_bayesian(logger, "Maxwell", num_warmup=1000, num_samples=2000) as ctx:
        ...     result = model.fit_bayesian(x, y)
        ...     ctx["r_hat_max"] = compute_rhat(result)
        ...     ctx["ess_min"] = compute_ess(result)
        ...     ctx["divergences"] = result.divergences
    """
    context = {
        "model": model,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        **kwargs,
    }

    with log_operation(logger, "bayesian_inference", level=level, **context) as ctx:
        yield ctx


@contextmanager
def log_transform(
    logger: logging.Logger | RheoJAXLogger,
    transform: str,
    input_shape: tuple[int, ...] | None = None,
    level: int = logging.INFO,
    **kwargs,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for transform operations.

    Specialized wrapper for data transformation operations.

    Args:
        logger: Logger instance to use.
        transform: Transform name.
        input_shape: Shape of input data.
        level: Log level (default INFO).
        **kwargs: Additional context.

    Yields:
        Dictionary for adding completion context (e.g., output_shape).

    Example:
        >>> with log_transform(logger, "mastercurve", input_shape=(10, 100)) as ctx:
        ...     result = transform.transform(datasets)
        ...     ctx["output_shape"] = result.shape
        ...     ctx["shift_factors"] = len(shift_factors)
    """
    context = {"transform": transform, **kwargs}
    if input_shape is not None:
        context["input_shape"] = input_shape

    with log_operation(logger, "transform", level=level, **context) as ctx:
        yield ctx


@contextmanager
def log_io(
    logger: logging.Logger | RheoJAXLogger,
    operation: str,
    filepath: str | None = None,
    level: int = logging.INFO,
    **kwargs,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for I/O operations.

    Specialized wrapper for file read/write operations.

    Args:
        logger: Logger instance to use.
        operation: I/O operation type (read, write, load, save).
        filepath: Path to file being accessed.
        level: Log level (default INFO).
        **kwargs: Additional context.

    Yields:
        Dictionary for adding completion context (e.g., records, file_size).

    Example:
        >>> with log_io(logger, "read", filepath="data.csv") as ctx:
        ...     data = read_csv(filepath)
        ...     ctx["records"] = len(data)
        ...     ctx["columns"] = list(data.columns)
    """
    context = {"io_operation": operation, **kwargs}
    if filepath is not None:
        context["filepath"] = str(filepath)

    with log_operation(logger, f"io_{operation}", level=level, **context) as ctx:
        yield ctx


@contextmanager
def log_pipeline_stage(
    logger: logging.Logger | RheoJAXLogger,
    stage: str,
    pipeline_id: str | None = None,
    level: int = logging.INFO,
    **kwargs,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for pipeline stage execution.

    Args:
        logger: Logger instance to use.
        stage: Pipeline stage name.
        pipeline_id: Optional pipeline identifier.
        level: Log level (default INFO).
        **kwargs: Additional context.

    Yields:
        Dictionary for adding completion context.

    Example:
        >>> with log_pipeline_stage(logger, "fit", pipeline_id="pipe_001") as ctx:
        ...     result = pipeline.fit()
        ...     ctx["model"] = result.model_name
    """
    context = {"stage": stage, **kwargs}
    if pipeline_id is not None:
        context["pipeline_id"] = pipeline_id

    with log_operation(logger, f"pipeline_{stage}", level=level, **context) as ctx:
        yield ctx


@contextmanager
def log_gui_action(
    logger: logging.Logger | RheoJAXLogger,
    action: str,
    widget: str | None = None,
    page: str | None = None,
    level: int = logging.DEBUG,
    **kwargs,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for GUI user interaction logging.

    Specialized wrapper for logging user interactions in the GUI.
    Defaults to DEBUG level since GUI actions are typically verbose.

    Args:
        logger: Logger instance to use.
        action: Type of action (click, select, navigate, etc.).
        widget: Widget identifier or class name.
        page: Page where action occurred.
        level: Log level (default DEBUG for GUI actions).
        **kwargs: Additional context (button_id, value, etc.).

    Yields:
        Dictionary for adding completion context.

    Example:
        >>> with log_gui_action(logger, "button_click", widget="FitButton", page="FitPage") as ctx:
        ...     self._perform_fit()
        ...     ctx["result"] = "success"

    Example without context manager (for simple actions):
        >>> logger.debug("Button clicked", action="click", widget="FitButton", page="FitPage")
    """
    context = {"action": action, **kwargs}
    if widget is not None:
        context["widget"] = widget
    if page is not None:
        context["page"] = page

    with log_operation(logger, f"gui_{action}", level=level, **context) as ctx:
        yield ctx
