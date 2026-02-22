"""
RheoJAX Logger Factory.

Provides the main logger factory function and RheoJAXLogger adapter
for consistent logging across all RheoJAX modules.
"""

import logging
import threading
from collections.abc import MutableMapping
from typing import Any


class RheoJAXLogger(logging.LoggerAdapter):
    """Enhanced logger adapter with context binding.

    Provides structured logging with automatic context injection
    and lazy evaluation support.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Fitting model", model="Maxwell", data_points=1000)
        14:32:05 | INFO     | rheojax.models | Fitting model | model=Maxwell | data_points=1000
    """

    def __init__(
        self, logger: logging.Logger, extra: dict[str, Any] | None = None
    ) -> None:
        """Initialize the logger adapter.

        Args:
            logger: Underlying Python logger.
            extra: Default extra context to include in all log messages.
        """
        super().__init__(logger, extra if extra is not None else {})

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Process the log message and kwargs.

        Merges extra kwargs into the log record for structured output.

        Args:
            msg: Log message.
            kwargs: Keyword arguments passed to the log method.

        Returns:
            Tuple of (message, modified kwargs).
        """
        # Extract extra fields from kwargs
        extra = kwargs.pop("extra", {})

        # Merge with default extra and any additional kwargs
        base_extra = self.extra if self.extra is not None else {}
        merged_extra = {**base_extra, **extra}

        # Move any non-standard kwargs to extra
        standard_keys = {"exc_info", "stack_info", "stacklevel"}
        for key in list(kwargs.keys()):
            if key not in standard_keys:
                merged_extra[key] = kwargs.pop(key)

        # Store merged extra in kwargs
        kwargs["extra"] = {"extra": merged_extra} if merged_extra else {}

        return msg, kwargs

    def bind(self, **context) -> "RheoJAXLogger":
        """Create a new logger with additional bound context.

        Args:
            **context: Key-value pairs to bind to the logger.

        Returns:
            New RheoJAXLogger instance with merged context.

        Example:
            >>> logger = get_logger(__name__).bind(model="Maxwell")
            >>> logger.info("Starting fit")  # model="Maxwell" auto-included
        """
        base_extra = self.extra if self.extra is not None else {}
        merged = {**base_extra, **context}
        return RheoJAXLogger(self.logger, merged)

    # Note: debug, info, warning, error, critical, exception methods
    # are inherited from LoggerAdapter. The process() method handles
    # merging extra context into log records.


# Process-global logger cache. All modules in the same process share this
# cache, so clear_logger_cache() affects all callers. Protected by a lock
# for thread safety.
_logger_cache: dict[tuple, RheoJAXLogger] = {}
_cache_lock = threading.Lock()


def get_logger(name: str, **context) -> RheoJAXLogger:
    """Get a RheoJAX logger for the given name.

    Creates a new logger or returns a cached instance.
    The logger is automatically configured based on the
    current logging configuration.

    Note:
        The cache is process-global: all modules share it, and
        ``clear_logger_cache()`` affects every caller.

    Args:
        name: Logger name (typically __name__).
        **context: Default context to bind to the logger.
            Values must be representable as strings for cache keying.

    Returns:
        RheoJAXLogger instance.

    Example:
        >>> from rheojax.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Model fitted", R2=0.9987)
    """
    # Build a collision-free cache key (no hash() — avoids hash collisions)
    try:
        cache_key = (name, tuple(sorted((k, str(v)) for k, v in context.items())))
    except Exception:
        # Unhashable context values — skip caching
        logger = logging.getLogger(name)
        return RheoJAXLogger(logger, context)

    with _cache_lock:
        if cache_key in _logger_cache:
            return _logger_cache[cache_key]

        # Get underlying logger
        logger = logging.getLogger(name)

        # Create adapter with context
        rheojax_logger = RheoJAXLogger(logger, context)

        # Cache and return
        _logger_cache[cache_key] = rheojax_logger
        return rheojax_logger


def clear_logger_cache() -> None:
    """Clear the logger cache.

    Primarily useful for testing. This is process-global — all
    callers of ``get_logger()`` will get fresh instances afterward.
    """
    with _cache_lock:
        _logger_cache.clear()
