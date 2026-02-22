"""
RheoJAX Log Formatters.

Custom formatters for human-readable, detailed, JSON, and scientific output.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from rheojax.logging.config import LogFormat


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Log level colors
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta

    # Component colors
    TIMESTAMP = "\033[90m"  # Gray
    LOGGER = "\033[34m"  # Blue
    MESSAGE = "\033[0m"  # Default


LEVEL_COLORS = {
    "DEBUG": Colors.DEBUG,
    "INFO": Colors.INFO,
    "WARNING": Colors.WARNING,
    "ERROR": Colors.ERROR,
    "CRITICAL": Colors.CRITICAL,
}


class StandardFormatter(logging.Formatter):
    """Human-readable format for console output.

    Format: HH:MM:SS | LEVEL    | logger.name | message

    Supports optional colorization for terminal output.
    """

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    def __init__(self, colorize: bool = True) -> None:
        """Initialize the formatter.

        Args:
            colorize: Enable ANSI color codes in output.
        """
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)
        self.colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record.

        Args:
            record: LogRecord instance to format.

        Returns:
            Formatted log string.
        """
        # Add extra fields to message if present
        message = record.getMessage()
        if hasattr(record, "extra") and record.extra:
            extra_str = " | ".join(
                f"{k}={self._format_value(v)}" for k, v in record.extra.items()
            )
            message = f"{message} | {extra_str}"

        # Create a copy of the record with the modified message
        record = logging.makeLogRecord(record.__dict__)
        record.msg = message
        record.args = ()

        # Format the base message
        formatted = super().format(record)

        if self.colorize:
            # Apply colors
            level_color = LEVEL_COLORS.get(record.levelname, Colors.RESET)
            parts = formatted.split(" | ")
            if len(parts) >= 4:
                formatted = (
                    f"{Colors.TIMESTAMP}{parts[0]}{Colors.RESET} | "
                    f"{level_color}{parts[1]}{Colors.RESET} | "
                    f"{Colors.LOGGER}{parts[2]}{Colors.RESET} | "
                    f"{Colors.MESSAGE}{' | '.join(parts[3:])}{Colors.RESET}"
                )

        return formatted

    def _format_value(self, value: Any) -> str:
        """Format a value for log output.

        Args:
            value: Value to format.

        Returns:
            Formatted string representation.
        """
        if isinstance(value, float):
            if abs(value) < 0.001 or abs(value) > 10000:
                return f"{value:.4e}"
            return f"{value:.4f}"
        if isinstance(value, tuple):
            return str(value)
        return str(value)


class DetailedFormatter(logging.Formatter):
    """Detailed format with file/line info for debugging.

    Format: YYYY-MM-DD HH:MM:SS.ffffff | LEVEL | logger:line | func | message
    """

    FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | "
        "%(funcName)s | %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, colorize: bool = False) -> None:
        """Initialize the formatter.

        Args:
            colorize: Enable ANSI color codes (disabled by default for files).
        """
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)
        self.colorize = colorize

    def formatTime(
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        """Format timestamp with true microsecond precision.

        Args:
            record: LogRecord instance.
            datefmt: Date format string (unused, uses DATE_FORMAT).

        Returns:
            Timestamp string with microseconds.
        """
        ct = datetime.fromtimestamp(record.created)
        base = ct.strftime(self.DATE_FORMAT)
        return f"{base}.{ct.microsecond:06d}"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with microseconds.

        Args:
            record: LogRecord instance to format.

        Returns:
            Formatted log string.
        """
        # Add extra fields
        message = record.getMessage()
        if hasattr(record, "extra") and record.extra:
            extra_str = " | ".join(
                f"{k}={self._format_value(v)}" for k, v in record.extra.items()
            )
            message = f"{message} | {extra_str}"

        record = logging.makeLogRecord(record.__dict__)
        record.msg = message
        record.args = ()

        return super().format(record)

    def _format_value(self, value: Any) -> str:
        """Format a value for log output."""
        if isinstance(value, float):
            return f"{value:.6e}"
        return str(value)


class JSONFormatter(logging.Formatter):
    """JSON format for machine parsing and log aggregation.

    Output: {"timestamp": "...", "level": "...", "logger": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: LogRecord instance to format.

        Returns:
            JSON-formatted log string.
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add thread info if available (thread ID can be 0 on some platforms)
        if record.thread is not None:
            log_data["thread"] = record.thread
            log_data["thread_name"] = record.threadName

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra") and record.extra:
            log_data["extra"] = self._serialize_extra(record.extra)

        try:
            return json.dumps(log_data, default=str, ensure_ascii=False)
        except RecursionError:
            # Circular reference in extra data — fall back to safe serialization
            log_data.pop("extra", None)
            log_data["_serialization_error"] = "circular reference in extra fields"
            return json.dumps(log_data, default=str, ensure_ascii=False)

    def _serialize_extra(self, extra: dict) -> dict:
        """Serialize extra fields for JSON output.

        Args:
            extra: Dictionary of extra fields.

        Returns:
            JSON-serializable dictionary.
        """
        result = {}
        for key, value in extra.items():
            if hasattr(value, "tolist"):  # NumPy/JAX arrays
                result[key] = value.tolist() if value.size < 100 else str(value.shape)
            elif hasattr(value, "__dict__"):
                result[key] = str(value)
            else:
                try:
                    json.dumps(value)
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = str(value)
        return result


class ScientificFormatter(DetailedFormatter):
    """Format optimized for scientific computing output.

    Provides consistent scientific notation for numerical values
    and special handling for array shapes and dtypes.
    """

    def _format_value(self, value: Any) -> str:
        """Format a value with scientific notation.

        Args:
            value: Value to format.

        Returns:
            Formatted string with scientific notation for floats.
        """
        if isinstance(value, float):
            return f"{value:.6e}"
        if isinstance(value, int) and abs(value) > 10000:
            return f"{value:.2e}"
        if hasattr(value, "shape"):  # NumPy/JAX array
            return f"array{value.shape}"
        if isinstance(value, tuple) and len(value) <= 4:
            # Format as shape tuple
            return f"({', '.join(str(v) for v in value)})"
        return str(value)


def get_formatter(format_type: LogFormat, colorize: bool = True) -> logging.Formatter:
    """Get the appropriate formatter for the given format type.

    Args:
        format_type: LogFormat enum value.
        colorize: Enable ANSI color codes.

    Returns:
        Configured logging.Formatter instance.
    """
    formatters = {
        LogFormat.STANDARD: lambda: StandardFormatter(colorize=colorize),
        LogFormat.DETAILED: lambda: DetailedFormatter(colorize=colorize),
        LogFormat.JSON: lambda: JSONFormatter(),
        LogFormat.SCIENTIFIC: lambda: ScientificFormatter(colorize=colorize),
    }

    factory = formatters.get(format_type, formatters[LogFormat.STANDARD])
    return factory()
