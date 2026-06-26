"""
RheoJAX Logging System.

Structured logging for monitoring and debugging RheoJAX operations.

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
    log_gui_action,
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
from rheojax.logging.logger import (
    RheoJAXLogger,
    clear_logger_cache,
    get_logger,
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
    "log_gui_action",
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
]
