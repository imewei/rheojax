"""
RheoJAX Logging Configuration.

Centralized configuration management for the RheoJAX logging system.
Supports environment variable configuration and programmatic setup.

Environment Variables:
    RHEOJAX_LOG_LEVEL: Global log level (DEBUG, INFO, WARNING, ERROR)
    RHEOJAX_LOG_FILE: Path to log file (enables file logging)
    RHEOJAX_LOG_FORMAT: Output format (standard, detailed, json)
    RHEOJAX_LOG_<SUBSYSTEM>: Per-subsystem level override
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LogFormat(Enum):
    """Available log output formats."""

    STANDARD = "standard"
    DETAILED = "detailed"
    JSON = "json"
    SCIENTIFIC = "scientific"


# Default log levels by subsystem
DEFAULT_SUBSYSTEM_LEVELS: dict[str, str] = {
    # Core modules
    "rheojax.models": "INFO",
    "rheojax.core": "INFO",
    "rheojax.core.bayesian": "INFO",
    "rheojax.transforms": "INFO",
    "rheojax.io": "WARNING",
    "rheojax.pipeline": "INFO",
    "rheojax.utils": "INFO",
    "rheojax.utils.optimization": "INFO",
    "rheojax.visualization": "WARNING",
    # GUI modules - hierarchical for fine-grained control
    "rheojax.gui": "INFO",
    "rheojax.gui.app": "INFO",
    "rheojax.gui.pages": "INFO",
    "rheojax.gui.state": "INFO",
    "rheojax.gui.services": "INFO",
    "rheojax.gui.jobs": "INFO",
    "rheojax.gui.widgets": "WARNING",  # Verbose at DEBUG, quiet by default
    "rheojax.gui.dialogs": "WARNING",  # Verbose at DEBUG, quiet by default
    "rheojax.gui.utils": "WARNING",
}


@dataclass
class LogConfig:
    """RheoJAX logging configuration.

    Attributes:
        level: Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format (standard, detailed, json, scientific)
        console: Enable console output
        file: Path to log file (None disables file logging)
        file_max_bytes: Maximum log file size before rotation (default 10MB)
        file_backup_count: Number of backup files to keep (default 5)
        subsystem_levels: Per-subsystem log level overrides
        lazy_formatting: Enable lazy evaluation of log arguments
        include_timestamps: Include timestamps in log output
        include_thread: Include thread name in log output
        colorize: Enable colored console output
    """

    level: str = "INFO"
    format: LogFormat | str = LogFormat.STANDARD
    console: bool = True
    file: Path | str | None = None
    file_max_bytes: int = 10_000_000  # 10MB
    file_backup_count: int = 5
    subsystem_levels: dict[str, str] = field(
        default_factory=lambda: DEFAULT_SUBSYSTEM_LEVELS.copy()
    )
    lazy_formatting: bool = True
    include_timestamps: bool = True
    include_thread: bool = False
    colorize: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert string format to enum if needed
        if isinstance(self.format, str):
            self.format = LogFormat(self.format.lower())

        # Convert string path to Path if needed
        if isinstance(self.file, str):
            self.file = Path(self.file)

        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log level: {self.level}. "
                f"Must be one of: {', '.join(valid_levels)}"
            )

    @classmethod
    def from_env(cls) -> "LogConfig":
        """Create configuration from environment variables.

        Reads the following environment variables:
            - RHEOJAX_LOG_LEVEL: Global log level
            - RHEOJAX_LOG_FILE: Path to log file
            - RHEOJAX_LOG_FORMAT: Output format
            - RHEOJAX_LOG_COLORIZE: Enable colors (true/false)
            - RHEOJAX_LOG_<SUBSYSTEM>: Per-subsystem levels

        Returns:
            LogConfig instance with environment-based settings.
        """
        # Get subsystem levels from environment
        subsystem_levels = DEFAULT_SUBSYSTEM_LEVELS.copy()
        for key, value in os.environ.items():
            if key.startswith("RHEOJAX_LOG_") and key not in {
                "RHEOJAX_LOG_LEVEL",
                "RHEOJAX_LOG_FILE",
                "RHEOJAX_LOG_FORMAT",
                "RHEOJAX_LOG_COLORIZE",
            }:
                # Convert RHEOJAX_LOG_MODELS to rheojax.models
                subsystem = key.replace("RHEOJAX_LOG_", "").lower()
                subsystem = f"rheojax.{subsystem.replace('_', '.')}"
                subsystem_levels[subsystem] = value.upper()

        # Parse file path
        file_path = os.environ.get("RHEOJAX_LOG_FILE")
        file = Path(file_path) if file_path else None

        # Parse colorize
        colorize_str = os.environ.get("RHEOJAX_LOG_COLORIZE", "true").lower()
        colorize = colorize_str in ("true", "1", "yes")

        # Parse format
        format_str = os.environ.get("RHEOJAX_LOG_FORMAT", "standard").lower()
        try:
            log_format = LogFormat(format_str)
        except ValueError:
            log_format = LogFormat.STANDARD

        return cls(
            level=os.environ.get("RHEOJAX_LOG_LEVEL", "INFO").upper(),
            format=log_format,
            file=file,
            subsystem_levels=subsystem_levels,
            colorize=colorize,
        )

    def get_level(self, logger_name: str) -> int:
        """Get the effective log level for a logger.

        Args:
            logger_name: Full logger name (e.g., "rheojax.models.maxwell")

        Returns:
            Logging level as integer.
        """
        # Check for exact match first
        if logger_name in self.subsystem_levels:
            return getattr(logging, self.subsystem_levels[logger_name].upper())

        # Check for parent matches (most specific first)
        parts = logger_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:i])
            if parent in self.subsystem_levels:
                return getattr(logging, self.subsystem_levels[parent].upper())

        # Fall back to global level
        return getattr(logging, self.level.upper())


# Global configuration instance
_config: LogConfig | None = None
_configured: bool = False


def get_config() -> LogConfig:
    """Get the current logging configuration.

    Returns:
        Current LogConfig instance.
    """
    global _config
    if _config is None:
        _config = LogConfig.from_env()
    return _config


def configure_logging(
    level: str = "INFO",
    format: str = "standard",
    file: str | None = None,
    colorize: bool = True,
    **kwargs,
) -> LogConfig:
    """Configure the RheoJAX logging system.

    This function should be called once at application startup.
    Subsequent calls will reconfigure the logging system.

    Args:
        level: Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format (standard, detailed, json, scientific)
        file: Path to log file (None disables file logging)
        colorize: Enable colored console output
        **kwargs: Additional LogConfig parameters

    Returns:
        The configured LogConfig instance.

    Example:
        >>> from rheojax.logging import configure_logging
        >>> configure_logging(level="DEBUG", file="rheojax.log")
    """
    global _config, _configured

    # Create configuration
    _config = LogConfig(
        level=level,
        format=LogFormat(format.lower()) if isinstance(format, str) else format,
        file=Path(file) if file else None,
        colorize=colorize,
        **kwargs,
    )

    # Apply configuration to logging system
    _apply_config(_config)
    _configured = True

    return _config


def _apply_config(config: LogConfig) -> None:
    """Apply configuration to the Python logging system.

    Args:
        config: LogConfig instance to apply.
    """
    from rheojax.logging.formatters import get_formatter
    from rheojax.logging.handlers import create_handlers

    # Get or create the root rheojax logger
    root_logger = logging.getLogger("rheojax")
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, filter at handler

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create and add handlers
    handlers = create_handlers(config)
    log_format = (
        config.format
        if isinstance(config.format, LogFormat)
        else LogFormat(config.format)
    )
    for handler in handlers:
        formatter = get_formatter(log_format, colorize=config.colorize)
        handler.setFormatter(formatter)
        handler.setLevel(getattr(logging, config.level.upper()))
        root_logger.addHandler(handler)

    # Configure subsystem loggers
    for subsystem, level in config.subsystem_levels.items():
        logger = logging.getLogger(subsystem)
        logger.setLevel(getattr(logging, level.upper()))

    # Prevent propagation to root logger
    root_logger.propagate = False


def is_configured() -> bool:
    """Check if logging has been explicitly configured.

    Returns:
        True if configure_logging() has been called.
    """
    return _configured


def reset_config() -> None:
    """Reset logging configuration to defaults.

    Primarily useful for testing.
    """
    global _config, _configured
    _config = None
    _configured = False

    # Reset root logger
    root_logger = logging.getLogger("rheojax")
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
