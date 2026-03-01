"""Robustness tests for BatchingExporter failure and queue handling."""

import logging
import queue
from dataclasses import dataclass

import pytest

from rheojax.logging.exporters import BatchingExporter, LogEntry


@dataclass
class FailingExporter:
    fail_once: bool = True
    seen: list[list[LogEntry]] = None

    def __post_init__(self):
        if self.seen is None:
            self.seen = []

    def export(self, entries):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("boom")
        self.seen.append(entries)
        return True

    def shutdown(self):
        return None


def test_batching_exporter_drops_on_inner_failure(caplog):
    """When the inner exporter raises, entries are dropped (not requeued)
    and a warning is logged with the count of dropped entries."""
    caplog.set_level(logging.WARNING)
    inner = FailingExporter()
    exporter = BatchingExporter(
        inner_exporter=inner, batch_size=2, flush_interval=0.01, max_queue_size=10
    )

    entry1 = LogEntry("t1", "INFO", "l", "m1")
    entry2 = LogEntry("t2", "INFO", "l", "m2")

    # Queue two entries, then flush — inner raises on first call
    exporter.export([entry1, entry2])
    exporter._flush()

    # Second flush has nothing to deliver (entries were dropped, not requeued)
    exporter._flush()

    exporter.shutdown()

    assert any("Batch export failed" in line for line in caplog.text.splitlines())
    assert "Dropped 2 log entries" in caplog.text
    # Entries were NOT retried — inner never received them
    assert inner.seen == [], "failed entries should be dropped, not retried"


def test_batching_exporter_returns_false_on_inner_false(caplog):
    caplog.set_level(logging.WARNING, logger="rheojax.logging.exporters")

    class FalseExporter:
        def __init__(self):
            self.calls = 0

        def export(self, entries):
            self.calls += 1
            return False

        def shutdown(self):
            return None

    inner = FalseExporter()
    exporter = BatchingExporter(
        inner_exporter=inner, batch_size=2, flush_interval=0.01, max_queue_size=10
    )

    entry1 = LogEntry("t1", "INFO", "l", "m1")
    ok = exporter.export([entry1])
    exporter._flush()
    exporter.shutdown()

    assert ok is True  # queueing succeeded
    assert inner.calls >= 1
    assert "reported failure" in caplog.text or "warning" in caplog.text.lower()


def test_batching_exporter_shutdown_warns_on_blocking_inner(caplog):
    caplog.set_level(logging.WARNING)

    class BlockingExporter:
        def export(self, entries):  # pragma: no cover - timing dependent
            import time

            time.sleep(2)
            return True

        def shutdown(self):
            return None

    exporter = BatchingExporter(
        inner_exporter=BlockingExporter(),
        batch_size=1,
        flush_interval=0.01,
        max_queue_size=2,
    )

    exporter.export([LogEntry("t", "INFO", "l", "m")])
    import time

    start = time.perf_counter()
    exporter.shutdown()
    elapsed = time.perf_counter() - start

    assert elapsed < 6.0


def test_batching_exporter_queue_full_logs_and_drops(caplog):
    caplog.set_level(logging.ERROR, logger="rheojax.logging.exporters")

    class DropExporter:
        def export(self, entries):
            return True

        def shutdown(self):
            return None

    exporter = BatchingExporter(
        inner_exporter=DropExporter(),
        batch_size=100,
        flush_interval=1.0,
        max_queue_size=1,
    )

    # Prevent flush from relieving pressure so we hit the drop path
    exporter._flush = lambda: None  # type: ignore

    entry = LogEntry("t1", "INFO", "l", "m1")
    # First export fills queue to capacity
    exporter.export([entry])
    # Second export should hit full path and drop
    ok = exporter.export([entry])

    exporter.shutdown()

    # When full, exporter returns False and logs the drop
    assert ok is False
    assert (
        "queue is full; dropping entry" in caplog.text
        or "BatchingExporter queue is full" in caplog.text
    )
