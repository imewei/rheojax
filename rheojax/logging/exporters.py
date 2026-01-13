"""
RheoJAX Logging Exporters.

Structured logging exporters for observability platforms including
OpenTelemetry, Datadog, and custom backends.
"""

import json
import logging
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class LogEntry:
    """Structured log entry for export.

    Attributes:
        timestamp: ISO 8601 timestamp.
        level: Log level name.
        logger: Logger name.
        message: Log message.
        attributes: Additional structured attributes.
        trace_id: Optional trace ID for correlation.
        span_id: Optional span ID for correlation.
        resource: Resource attributes (service name, version, etc.).
    """

    timestamp: str
    level: str
    logger: str
    message: str
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None
    resource: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "timestamp": self.timestamp,
            "level": self.level,
            "logger": self.logger,
            "message": self.message,
        }
        if self.attributes:
            result["attributes"] = self.attributes
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.resource:
            result["resource"] = self.resource
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class LogExporter(ABC):
    """Abstract base class for log exporters.

    Exporters transform and send log entries to external systems
    like OpenTelemetry collectors, Datadog, or custom backends.
    """

    @abstractmethod
    def export(self, entries: list[LogEntry]) -> bool:
        """Export log entries to the backend.

        Args:
            entries: List of LogEntry objects to export.

        Returns:
            True if export succeeded, False otherwise.
        """
        raise NotImplementedError(
            "LogExporter.export must be implemented by subclasses"
        )

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter and flush any pending entries."""
        raise NotImplementedError(
            "LogExporter.shutdown must be implemented by subclasses"
        )


class OpenTelemetryLogExporter(LogExporter):
    """OpenTelemetry-compatible log exporter.

    Exports logs in OTLP (OpenTelemetry Protocol) format to an
    OTLP collector endpoint. Falls back to console output if
    opentelemetry-api is not installed.

    Example:
        >>> from rheojax.logging.exporters import OpenTelemetryLogExporter
        >>> exporter = OpenTelemetryLogExporter(
        ...     endpoint="http://localhost:4317",
        ...     service_name="rheojax-app"
        ... )
        >>> handler = ExportingHandler(exporter)
        >>> logger.addHandler(handler)
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        service_name: str = "rheojax",
        service_version: str | None = None,
        insecure: bool = True,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the OpenTelemetry exporter.

        Args:
            endpoint: OTLP collector endpoint URL.
            service_name: Service name for resource attributes.
            service_version: Service version (auto-detected if None).
            insecure: Use insecure connection (no TLS).
            headers: Additional headers for OTLP requests.
        """
        self.endpoint = endpoint
        self.service_name = service_name
        self.service_version = service_version or self._get_rheojax_version()
        self.insecure = insecure
        self.headers = headers or {}
        self._otel_available = self._check_otel_available()
        self._logger_provider = None
        self._log_emitter = None

        if self._otel_available:
            self._setup_otel()

    def _get_rheojax_version(self) -> str:
        """Get RheoJAX version."""
        try:
            from rheojax import __version__

            return __version__
        except ImportError:
            return "unknown"

    def _check_otel_available(self) -> bool:
        """Check if OpenTelemetry packages are available."""
        try:
            from opentelemetry.sdk._logs import LoggerProvider  # noqa: F401
            from opentelemetry.sdk._logs.export import (
                BatchLogRecordProcessor,  # noqa: F401
            )

            return True
        except ImportError:
            return False

    def _setup_otel(self) -> None:
        """Set up OpenTelemetry logger provider."""
        try:
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter,
            )
            from opentelemetry.sdk._logs import LoggerProvider
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry.sdk.resources import Resource

            # Create resource with service info
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": self.service_version,
                }
            )

            # Create OTLP exporter
            otlp_exporter = OTLPLogExporter(
                endpoint=self.endpoint,
                insecure=self.insecure,
                headers=self.headers,
            )

            # Create logger provider with batch processor
            self._logger_provider = LoggerProvider(resource=resource)
            self._logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(otlp_exporter)
            )

            # Get log emitter
            self._log_emitter = self._logger_provider.get_logger(
                "rheojax", self.service_version
            )

        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to initialize OpenTelemetry: {e}"
            )
            self._otel_available = False

    def export(self, entries: list[LogEntry]) -> bool:
        """Export log entries via OpenTelemetry.

        Args:
            entries: List of LogEntry objects.

        Returns:
            True if export succeeded.
        """
        if not self._otel_available or self._log_emitter is None:
            # Fall back to console output
            for entry in entries:
                print(f"[OTEL-FALLBACK] {entry.to_json()}")
            return True

        try:
            from opentelemetry._logs.severity import SeverityNumber
            from opentelemetry.sdk._logs import LogRecord

            severity_map = {
                "DEBUG": SeverityNumber.DEBUG,
                "INFO": SeverityNumber.INFO,
                "WARNING": SeverityNumber.WARN,
                "ERROR": SeverityNumber.ERROR,
                "CRITICAL": SeverityNumber.FATAL,
            }

            for entry in entries:
                severity = severity_map.get(entry.level, SeverityNumber.INFO)
                self._log_emitter.emit(
                    LogRecord(
                        timestamp=int(
                            datetime.fromisoformat(
                                entry.timestamp.replace("Z", "+00:00")
                            ).timestamp()
                            * 1e9
                        ),
                        severity_number=severity,
                        severity_text=entry.level,
                        body=entry.message,
                        attributes=entry.attributes,
                    )
                )

            return True

        except Exception as e:
            logging.getLogger(__name__).error(f"OTLP export failed: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the OpenTelemetry logger provider."""
        if self._logger_provider is not None:
            self._logger_provider.shutdown()


class ConsoleExporter(LogExporter):
    """Console exporter for structured logs.

    Outputs log entries in structured format (JSON or key-value)
    to stdout/stderr for debugging or piping to other tools.
    """

    def __init__(
        self,
        format: str = "json",
        output: str = "stderr",
        include_resource: bool = False,
    ) -> None:
        """Initialize the console exporter.

        Args:
            format: Output format ("json" or "keyvalue").
            output: Output destination ("stdout" or "stderr").
            include_resource: Include resource attributes in output.
        """
        self.format = format
        self.output = output
        self.include_resource = include_resource
        self._stream = (
            __import__("sys").stderr if output == "stderr" else __import__("sys").stdout
        )

    def export(self, entries: list[LogEntry]) -> bool:
        """Export log entries to console.

        Args:
            entries: List of LogEntry objects.

        Returns:
            Always True.
        """
        for entry in entries:
            if self.format == "json":
                data = entry.to_dict()
                if not self.include_resource:
                    data.pop("resource", None)
                line = json.dumps(data, default=str)
            else:
                # Key-value format
                parts = [
                    f"timestamp={entry.timestamp}",
                    f"level={entry.level}",
                    f"logger={entry.logger}",
                    f'message="{entry.message}"',
                ]
                for key, value in entry.attributes.items():
                    parts.append(f"{key}={value}")
                line = " ".join(parts)

            print(line, file=self._stream)

        return True

    def shutdown(self) -> None:
        """Flush the output stream."""
        self._stream.flush()


class BatchingExporter(LogExporter):
    """Batching wrapper for log exporters.

    Collects log entries and exports them in batches for efficiency.
    Supports configurable batch size and flush intervals.
    """

    def __init__(
        self,
        inner_exporter: LogExporter,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
    ) -> None:
        """Initialize the batching exporter.

        Args:
            inner_exporter: Underlying exporter to use.
            batch_size: Maximum entries per batch.
            flush_interval: Seconds between automatic flushes.
            max_queue_size: Maximum queue size before blocking.
        """
        self._inner = inner_exporter
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: queue.Queue[LogEntry] = queue.Queue(maxsize=max_queue_size)
        self._shutdown_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background thread for periodic flushing."""
        while not self._shutdown_event.is_set():
            self._shutdown_event.wait(timeout=self._flush_interval)
            self._flush()

    def _flush(self) -> None:
        """Flush pending entries to the inner exporter."""
        entries = []
        while not self._queue.empty() and len(entries) < self._batch_size:
            try:
                entries.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if entries:
            try:
                success = self._inner.export(entries)
            except Exception as exc:  # pragma: no cover - defensive
                logging.getLogger(__name__).error(f"Batch export failed: {exc}")
                # best-effort requeue
                for entry in entries:
                    try:
                        self._queue.put_nowait(entry)
                    except queue.Full:
                        break
                return

            if success is False:
                logging.getLogger(__name__).warning(
                    "Inner exporter reported failure during batch flush"
                )

    def export(self, entries: list[LogEntry]) -> bool:
        """Add entries to the batch queue.

        Args:
            entries: List of LogEntry objects.

        Returns:
            True if entries were queued.
        """
        for entry in entries:
            try:
                self._queue.put(entry, block=False)
            except queue.Full:
                # Queue full, force flush
                self._flush()
                try:
                    self._queue.put(entry, block=True, timeout=1.0)
                except queue.Full:
                    logging.getLogger(__name__).error(
                        "BatchingExporter queue is full; dropping entry"
                    )
                    return False

        # Flush if batch size reached
        if self._queue.qsize() >= self._batch_size:
            self._flush()

        return not self._queue.full()

    def shutdown(self) -> None:
        """Shutdown the batching exporter."""
        self._shutdown_event.set()
        self._flush_thread.join(timeout=5.0)
        self._flush()  # Final flush
        if self._flush_thread.is_alive():
            logging.getLogger(__name__).warning(
                "BatchingExporter flush thread did not terminate cleanly"
            )
        self._inner.shutdown()


class ExportingHandler(logging.Handler):
    """Logging handler that exports to structured log backends.

    Bridges Python's logging system with structured log exporters.

    Example:
        >>> from rheojax.logging.exporters import ExportingHandler, ConsoleExporter
        >>> exporter = ConsoleExporter(format="json")
        >>> handler = ExportingHandler(exporter)
        >>> handler.setLevel(logging.INFO)
        >>> logger.addHandler(handler)
    """

    def __init__(
        self,
        exporter: LogExporter,
        service_name: str = "rheojax",
        service_version: str | None = None,
        include_trace_context: bool = True,
    ) -> None:
        """Initialize the exporting handler.

        Args:
            exporter: Log exporter to use.
            service_name: Service name for resource attributes.
            service_version: Service version.
            include_trace_context: Include trace/span IDs if available.
        """
        super().__init__()
        self._exporter = exporter
        self._service_name = service_name
        self._service_version = service_version or self._get_version()
        self._include_trace_context = include_trace_context
        self._resource = {
            "service.name": self._service_name,
            "service.version": self._service_version,
        }

    def _get_version(self) -> str:
        """Get RheoJAX version."""
        try:
            from rheojax import __version__

            return __version__
        except ImportError:
            return "unknown"

    def _get_trace_context(self) -> tuple[str | None, str | None]:
        """Get current trace and span IDs from OpenTelemetry context."""
        if not self._include_trace_context:
            return None, None

        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.is_recording():
                ctx = span.get_span_context()
                return (
                    format(ctx.trace_id, "032x"),
                    format(ctx.span_id, "016x"),
                )
        except ImportError:
            logging.getLogger(__name__).debug(
                "OpenTelemetry not installed; skipping trace context"
            )

        return None, None

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the exporter.

        Args:
            record: Python LogRecord to export.
        """
        try:
            # Build attributes from extra fields (include non-standard record attrs)
            attributes = {}
            if hasattr(record, "extra") and record.extra:
                attributes.update(record.extra)

            for key, value in record.__dict__.items():
                if key.startswith("_"):
                    continue
                if key in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                }:
                    continue
                attributes.setdefault(key, value)

            # Add standard attributes
            attributes["module"] = record.module
            attributes["function"] = record.funcName
            attributes["line"] = record.lineno

            # Get trace context
            trace_id, span_id = self._get_trace_context()

            # Create log entry
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created, tz=UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                level=record.levelname,
                logger=record.name,
                message=record.getMessage(),
                attributes=attributes,
                trace_id=trace_id,
                span_id=span_id,
                resource=self._resource,
            )

            # Export
            self._exporter.export([entry])

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close the handler and shutdown the exporter."""
        self._exporter.shutdown()
        super().close()


class CallbackExporter(LogExporter):
    """Exporter that calls a user-provided callback.

    Useful for custom integrations or testing.
    """

    def __init__(self, callback: Callable[[list[LogEntry]], bool]) -> None:
        """Initialize the callback exporter.

        Args:
            callback: Function to call with log entries.
        """
        self._callback = callback

    def export(self, entries: list[LogEntry]) -> bool:
        """Export entries via callback.

        Args:
            entries: List of LogEntry objects.

        Returns:
            Result of callback.
        """
        return self._callback(entries)

    def shutdown(self) -> None:
        """No-op shutdown."""
        logging.getLogger(__name__).debug("CallbackExporter.shutdown noop")


def create_otel_handler(
    endpoint: str = "http://localhost:4317",
    service_name: str = "rheojax",
    batch_size: int = 100,
    flush_interval: float = 5.0,
) -> ExportingHandler:
    """Create an OpenTelemetry-enabled logging handler.

    Convenience function to create a fully configured OpenTelemetry
    logging handler with batching.

    Args:
        endpoint: OTLP collector endpoint.
        service_name: Service name for resource attributes.
        batch_size: Batch size for export.
        flush_interval: Seconds between automatic flushes.

    Returns:
        Configured ExportingHandler.

    Example:
        >>> from rheojax.logging.exporters import create_otel_handler
        >>> handler = create_otel_handler(
        ...     endpoint="http://localhost:4317",
        ...     service_name="my-rheojax-app"
        ... )
        >>> logging.getLogger("rheojax").addHandler(handler)
    """
    otel_exporter = OpenTelemetryLogExporter(
        endpoint=endpoint,
        service_name=service_name,
    )

    batching_exporter = BatchingExporter(
        inner_exporter=otel_exporter,
        batch_size=batch_size,
        flush_interval=flush_interval,
    )

    return ExportingHandler(
        exporter=batching_exporter,
        service_name=service_name,
    )
