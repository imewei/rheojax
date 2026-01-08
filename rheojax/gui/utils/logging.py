"""
GUI Logging Utilities.

Lightweight logging helpers that bridge the standard library `logging`
framework into the Qt log panel without pulling PySide6 during import
in headless environments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

try:  # Delay Qt dependency for environments without PySide6
    from PySide6.QtCore import QObject, Signal

    class _LogEmitter(QObject):
        message_emitted = Signal(str)

    _HAS_QT = True
except Exception:  # pragma: no cover - PySide6 optional in headless tests
    _HAS_QT = False
    _LogEmitter = None  # type: ignore[misc,assignment]


class GuiLogHandler(logging.Handler):
    """Route log records into the GUI log panel safely."""

    def __init__(self, append_fn: Callable[[str], None]) -> None:
        super().__init__()
        self._append_fn = append_fn
        self._emitter = _LogEmitter() if _HAS_QT and _LogEmitter is not None else None
        if self._emitter is not None:
            self._emitter.message_emitted.connect(self._append_fn)

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        try:
            if self._emitter is not None:
                self._emitter.message_emitted.emit(message)
            else:
                self._append_fn(message)
        except Exception:
            self.handleError(record)


def install_gui_log_handler(
    append_fn: Callable[[str], None],
    level: int = logging.INFO,
    formatter: logging.Formatter | None = None,
) -> GuiLogHandler:
    """Attach GUI handler to root logger and return it."""
    handler = GuiLogHandler(append_fn)
    handler.setLevel(level)
    handler.setFormatter(
        formatter
        or logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler
