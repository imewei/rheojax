"""
Tests for RheoJAX logging exporters.

Tests the structured logging export functionality including
OpenTelemetry, console, and custom callback exporters.
"""

import json
import logging
import time
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from rheojax.logging.exporters import (
    BatchingExporter,
    CallbackExporter,
    ConsoleExporter,
    ExportingHandler,
    LogEntry,
    OpenTelemetryLogExporter,
    create_otel_handler,
)


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_create_basic_entry(self):
        """Test creating a basic log entry."""
        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="rheojax.models",
            message="Test message",
        )

        assert entry.timestamp == "2024-01-01T12:00:00Z"
        assert entry.level == "INFO"
        assert entry.logger == "rheojax.models"
        assert entry.message == "Test message"

    def test_entry_with_attributes(self):
        """Test creating entry with attributes."""
        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="rheojax.models",
            message="Model fitted",
            attributes={"model": "Maxwell", "R2": 0.9987},
        )

        assert entry.attributes["model"] == "Maxwell"
        assert entry.attributes["R2"] == 0.9987

    def test_entry_with_trace_context(self):
        """Test creating entry with trace context."""
        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="rheojax.models",
            message="Test",
            trace_id="abc123",
            span_id="def456",
        )

        assert entry.trace_id == "abc123"
        assert entry.span_id == "def456"

    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test message",
            attributes={"key": "value"},
        )

        result = entry.to_dict()

        assert result["timestamp"] == "2024-01-01T12:00:00Z"
        assert result["level"] == "INFO"
        assert result["logger"] == "test"
        assert result["message"] == "Test message"
        assert result["attributes"]["key"] == "value"

    def test_to_dict_excludes_none_fields(self):
        """Test that to_dict excludes None trace context."""
        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test",
        )

        result = entry.to_dict()

        assert "trace_id" not in result
        assert "span_id" not in result

    def test_to_json(self):
        """Test JSON serialization."""
        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test",
        )

        json_str = entry.to_json()
        parsed = json.loads(json_str)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test"


class TestConsoleExporter:
    """Tests for ConsoleExporter."""

    def test_json_format(self, capsys):
        """Test JSON format output."""
        exporter = ConsoleExporter(format="json", output="stdout")

        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test message",
        )

        result = exporter.export([entry])

        assert result is True
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["level"] == "INFO"

    def test_keyvalue_format(self, capsys):
        """Test key-value format output."""
        exporter = ConsoleExporter(format="keyvalue", output="stdout")

        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test message",
            attributes={"model": "Maxwell"},
        )

        exporter.export([entry])

        captured = capsys.readouterr()
        assert "level=INFO" in captured.out
        assert "model=Maxwell" in captured.out

    def test_multiple_entries(self, capsys):
        """Test exporting multiple entries."""
        exporter = ConsoleExporter(format="json", output="stdout")

        entries = [
            LogEntry(
                timestamp="2024-01-01T12:00:00Z",
                level="INFO",
                logger="test",
                message=f"Message {i}",
            )
            for i in range(3)
        ]

        exporter.export(entries)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 3

    def test_shutdown(self):
        """Test shutdown flushes stream."""
        exporter = ConsoleExporter()
        exporter.shutdown()  # Should not raise


class TestCallbackExporter:
    """Tests for CallbackExporter."""

    def test_callback_receives_entries(self):
        """Test that callback receives log entries."""
        received = []

        def callback(entries):
            received.extend(entries)
            return True

        exporter = CallbackExporter(callback)

        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test",
        )

        exporter.export([entry])

        assert len(received) == 1
        assert received[0].message == "Test"

    def test_callback_return_value(self):
        """Test callback return value propagates."""
        exporter = CallbackExporter(lambda entries: False)

        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test",
        )

        result = exporter.export([entry])

        assert result is False


class TestBatchingExporter:
    """Tests for BatchingExporter."""

    def test_batches_entries(self):
        """Test that entries are batched."""
        received_batches = []
        inner = CallbackExporter(
            lambda entries: (received_batches.append(entries), True)[1]
        )

        exporter = BatchingExporter(
            inner_exporter=inner,
            batch_size=5,
            flush_interval=10.0,  # Long interval to prevent auto-flush
        )

        try:
            # Add entries one at a time
            for i in range(5):
                entry = LogEntry(
                    timestamp="2024-01-01T12:00:00Z",
                    level="INFO",
                    logger="test",
                    message=f"Message {i}",
                )
                exporter.export([entry])

            # Wait a bit for batching
            time.sleep(0.1)

            # Should have received a batch
            assert len(received_batches) >= 1
        finally:
            exporter.shutdown()

    def test_shutdown_flushes(self):
        """Test that shutdown flushes pending entries."""
        received = []
        inner = CallbackExporter(lambda entries: (received.extend(entries), True)[1])

        exporter = BatchingExporter(
            inner_exporter=inner,
            batch_size=100,  # Large batch to prevent auto-flush
            flush_interval=100.0,
        )

        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test",
        )
        exporter.export([entry])

        # Force shutdown
        exporter.shutdown()

        # Entry should be flushed
        assert len(received) == 1


class TestExportingHandler:
    """Tests for ExportingHandler."""

    def test_handler_exports_records(self):
        """Test that handler exports log records."""
        received = []
        exporter = CallbackExporter(lambda entries: (received.extend(entries), True)[1])

        handler = ExportingHandler(exporter)
        handler.setLevel(logging.INFO)

        logger = logging.getLogger("test.exporting.handler")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            logger.info("Test message")

            assert len(received) == 1
            assert received[0].message == "Test message"
            assert received[0].level == "INFO"
        finally:
            logger.removeHandler(handler)

    def test_handler_includes_extra_fields(self):
        """Test that handler includes extra fields as attributes."""
        received = []
        exporter = CallbackExporter(lambda entries: (received.extend(entries), True)[1])

        handler = ExportingHandler(exporter)
        logger = logging.getLogger("test.exporting.extra")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            # Use extra dict directly (standard Python logging pattern)
            logger.info("Test message", extra={"extra": {"model": "Maxwell"}})

            assert len(received) == 1
            assert received[0].attributes.get("model") == "Maxwell"
        finally:
            logger.removeHandler(handler)

    def test_handler_close_shuts_down_exporter(self):
        """Test that handler.close() shuts down exporter."""
        shutdown_called = []
        exporter = CallbackExporter(lambda entries: True)
        original_shutdown = exporter.shutdown
        exporter.shutdown = lambda: shutdown_called.append(True)

        handler = ExportingHandler(exporter)
        handler.close()

        assert len(shutdown_called) == 1


class TestOpenTelemetryLogExporter:
    """Tests for OpenTelemetryLogExporter."""

    def test_fallback_when_otel_not_available(self, capsys):
        """Test fallback to console when OpenTelemetry not installed."""
        # This test works regardless of whether OTel is installed
        exporter = OpenTelemetryLogExporter(
            endpoint="http://localhost:4317",
            service_name="test-service",
        )

        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            logger="test",
            message="Test message",
        )

        # Should not raise, may fall back to console
        result = exporter.export([entry])
        assert result is True

    def test_service_version_auto_detection(self):
        """Test that service version is auto-detected."""
        exporter = OpenTelemetryLogExporter(service_name="test")

        # Should have a version (either from rheojax or "unknown")
        assert exporter.service_version is not None

    def test_shutdown_does_not_raise(self):
        """Test that shutdown doesn't raise even without OTel."""
        exporter = OpenTelemetryLogExporter(service_name="test")
        exporter.shutdown()  # Should not raise


class TestCreateOtelHandler:
    """Tests for create_otel_handler factory function."""

    def test_creates_handler(self):
        """Test that factory creates a valid handler."""
        handler = create_otel_handler(
            endpoint="http://localhost:4317",
            service_name="test-service",
        )

        assert isinstance(handler, ExportingHandler)

    def test_handler_is_functional(self):
        """Test that created handler can receive log records."""
        handler = create_otel_handler(service_name="test")

        logger = logging.getLogger("test.otel.handler")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            # Should not raise
            logger.info("Test message")
        finally:
            logger.removeHandler(handler)
            handler.close()


class TestExporterIntegration:
    """Integration tests for exporters with RheoJAX logging."""

    def test_with_rheojax_logger(self):
        """Test exporters with RheoJAX logger."""
        from rheojax.logging import get_logger

        received = []
        exporter = CallbackExporter(lambda entries: (received.extend(entries), True)[1])
        handler = ExportingHandler(exporter)

        root_logger = logging.getLogger("rheojax")
        root_logger.addHandler(handler)

        try:
            logger = get_logger("rheojax.models.test")
            logger.info("Model fitted successfully")

            assert len(received) >= 1
            assert any("Model fitted" in e.message for e in received)
        finally:
            root_logger.removeHandler(handler)

    def test_console_exporter_with_log_operation(self, capsys):
        """Test console exporter with log_operation context."""
        from rheojax.logging import log_operation

        exporter = ConsoleExporter(format="json", output="stdout")
        handler = ExportingHandler(exporter)

        logger = logging.getLogger("rheojax.test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            with log_operation(logger, "test_operation"):
                pass

            captured = capsys.readouterr()
            # Should have start and end messages
            assert "test_operation" in captured.out
        finally:
            logger.removeHandler(handler)
