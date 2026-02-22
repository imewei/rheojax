"""
RheoJAX Log Handlers.

Custom handlers for console, file, and memory-based logging.
"""

import logging
import sys
from logging.handlers import MemoryHandler, RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rheojax.logging.config import LogConfig


class RheoJAXStreamHandler(logging.StreamHandler):
    """Enhanced stream handler with flush control.

    Provides immediate flushing for interactive use and
    buffered output for batch processing.
    """

    def __init__(self, stream=None, immediate_flush: bool = True) -> None:
        """Initialize the handler.

        Args:
            stream: Output stream (default: sys.stderr)
            immediate_flush: Flush after each log message
        """
        super().__init__(stream or sys.stderr)
        self.immediate_flush = immediate_flush

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record.

        Args:
            record: LogRecord to emit.
        """
        try:
            super().emit(record)
            if self.immediate_flush:
                self.flush()
        except Exception:
            self.handleError(record)


class RheoJAXRotatingFileHandler(RotatingFileHandler):
    """Enhanced rotating file handler with UTF-8 encoding.

    Automatically handles log rotation and maintains backup files.
    """

    def __init__(
        self,
        filename: Path | str,
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
        encoding: str = "utf-8",
    ) -> None:
        """Initialize the rotating file handler.

        Args:
            filename: Path to log file.
            max_bytes: Maximum file size before rotation (default 10MB).
            backup_count: Number of backup files to keep (default 5).
            encoding: File encoding (default UTF-8).
        """
        # Ensure parent directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=str(filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )


class RheoJAXMemoryHandler(MemoryHandler):
    """Memory handler for buffered logging.

    Useful for batch operations where you want to collect logs
    and flush them periodically or at the end of an operation.
    """

    def __init__(
        self,
        capacity: int = 1000,
        flush_level: int = logging.ERROR,
        target: logging.Handler | None = None,
    ) -> None:
        """Initialize the memory handler.

        Args:
            capacity: Number of log records to buffer.
            flush_level: Level that triggers immediate flush.
            target: Target handler to flush to.
        """
        super().__init__(capacity=capacity, flushLevel=flush_level, target=target)

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """Check if buffer should be flushed.

        Extends stdlib behavior: when no target is set, caps the buffer
        at capacity by dropping the oldest records to prevent unbounded
        memory growth. With a target, delegates to stdlib MemoryHandler.

        Args:
            record: Current log record.

        Returns:
            True if buffer should be flushed.
        """
        if self.target is None and len(self.buffer) >= self.capacity:
            self.buffer = self.buffer[-(self.capacity - 1) :]
            return False
        return super().shouldFlush(record)


class NullHandler(logging.NullHandler):
    """Null handler that discards all log records.

    Useful for library mode where the user hasn't configured logging.
    """

    pass


def create_handlers(config: "LogConfig") -> list[logging.Handler]:
    """Create handlers based on configuration.

    Args:
        config: LogConfig instance.

    Returns:
        List of configured handlers.
    """
    handlers: list[logging.Handler] = []

    # Console handler
    if config.console:
        console_handler = RheoJAXStreamHandler(stream=sys.stderr, immediate_flush=True)
        handlers.append(console_handler)

    # File handler
    if config.file:
        file_handler = RheoJAXRotatingFileHandler(
            filename=config.file,
            max_bytes=config.file_max_bytes,
            backup_count=config.file_backup_count,
        )
        handlers.append(file_handler)

    # If no handlers configured, add a null handler
    if not handlers:
        handlers.append(NullHandler())

    return handlers
