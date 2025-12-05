"""
RheoJAX Logging System.

Comprehensive logging for monitoring and debugging RheoJAX operations.
Provides structured logging, JAX-safe utilities, and performance metrics.

Basic Usage:
    >>> from rheojax.logging import configure_logging, get_logger
    >>> configure_logging(level="INFO")
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting analysis", model="Maxwell")

Environment Variables:
    RHEOJAX_LOG_LEVEL: Global log level (DEBUG, INFO, WARNING, ERROR)
    RHEOJAX_LOG_FILE: Path to log file (enables file logging)
    RHEOJAX_LOG_FORMAT: Output format (standard, detailed, json)

Context Managers:
    >>> from rheojax.logging import log_fit
    >>> with log_fit(logger, "Maxwell", data_shape=(100,)) as ctx:
    ...     result = model.fit(x, y)
    ...     ctx["R2"] = result.r_squared

Performance Tracking:
    >>> from rheojax.logging import timed, log_memory
    >>> @timed(level=logging.INFO)
    ... def expensive_operation():
    ...     pass
"""

from rheojax.logging.config import (
    LogConfig,
    LogFormat,
    configure_logging,
    get_config,
    is_configured,
    reset_config,
)
from rheojax.logging.context import (
    log_bayesian,
    log_fit,
    log_io,
    log_operation,
    log_pipeline_stage,
    log_transform,
)
from rheojax.logging.formatters import (
    DetailedFormatter,
    JSONFormatter,
    ScientificFormatter,
    StandardFormatter,
    get_formatter,
)
from rheojax.logging.handlers import (
    NullHandler,
    RheoJAXMemoryHandler,
    RheoJAXRotatingFileHandler,
    RheoJAXStreamHandler,
    create_handlers,
)
from rheojax.logging.jax_utils import (
    jax_debug_log,
    jax_safe_log,
    log_array_info,
    log_array_stats,
    log_device_transfer,
    log_jax_config,
    log_numerical_issue,
)
from rheojax.logging.logger import (
    RheoJAXLogger,
    clear_logger_cache,
    get_logger,
)
from rheojax.logging.metrics import (
    ConvergenceTracker,
    IterationLogger,
    log_memory,
    timed,
)
from rheojax.logging.exporters import (
    BatchingExporter,
    CallbackExporter,
    ConsoleExporter,
    ExportingHandler,
    LogEntry,
    LogExporter,
    OpenTelemetryLogExporter,
    create_otel_handler,
)

__all__ = [
    # Configuration
    "LogConfig",
    "LogFormat",
    "configure_logging",
    "get_config",
    "is_configured",
    "reset_config",
    # Logger
    "RheoJAXLogger",
    "get_logger",
    "clear_logger_cache",
    # Context Managers
    "log_operation",
    "log_fit",
    "log_bayesian",
    "log_transform",
    "log_io",
    "log_pipeline_stage",
    # Formatters
    "StandardFormatter",
    "DetailedFormatter",
    "JSONFormatter",
    "ScientificFormatter",
    "get_formatter",
    # Handlers
    "RheoJAXStreamHandler",
    "RheoJAXRotatingFileHandler",
    "RheoJAXMemoryHandler",
    "NullHandler",
    "create_handlers",
    # Metrics
    "timed",
    "log_memory",
    "IterationLogger",
    "ConvergenceTracker",
    # JAX Utilities
    "log_array_info",
    "log_array_stats",
    "jax_safe_log",
    "jax_debug_log",
    "log_jax_config",
    "log_numerical_issue",
    "log_device_transfer",
    # Exporters
    "LogEntry",
    "LogExporter",
    "OpenTelemetryLogExporter",
    "ConsoleExporter",
    "BatchingExporter",
    "CallbackExporter",
    "ExportingHandler",
    "create_otel_handler",
]
