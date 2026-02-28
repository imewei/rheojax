"""
Export Worker
=============

Background worker for export operations (R13-GUI-002).

Offloads figure generation and file I/O from the Qt main thread to
prevent UI freezes during large exports.
"""

from __future__ import annotations

from typing import Any

try:
    from rheojax.gui.compat import QObject, QRunnable, Signal

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

    class QObject:  # type: ignore[no-redef]
        pass

    class QRunnable:  # type: ignore[no-redef]
        pass

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args: Any) -> None:
            pass


from rheojax.logging import get_logger

logger = get_logger(__name__)


class ExportWorkerSignals(QObject):
    """Signals for ExportWorker.

    Signals
    -------
    completed : Signal(list)
        Export completed with list of exported file paths.
    failed : Signal(str)
        Export failed with error message.
    progress : Signal(int, str)
        Progress update (percent, label).
    """

    completed = Signal(object)  # list[str] of exported file paths
    failed = Signal(str)  # error message
    progress = Signal(int, str)  # percent, label text


class ExportWorker(QRunnable):
    """Worker for performing exports in a background thread.

    Parameters
    ----------
    export_fn : Callable
        The export function to execute. Must accept no arguments and
        return a list of exported file paths.
    """

    def __init__(self, export_fn: Any) -> None:
        if not HAS_PYSIDE6:
            raise ImportError(
                "PySide6 is required for ExportWorker. "
                "Install with: pip install PySide6"
            )
        super().__init__()
        self.signals = ExportWorkerSignals()
        self._export_fn = export_fn

    def run(self) -> None:
        """Execute export in background thread."""
        try:
            logger.info("Export worker started")
            result = self._export_fn(self.signals.progress.emit)
            logger.info(
                "Export worker completed",
                file_count=len(result) if isinstance(result, list) else 0,
            )
            self.signals.completed.emit(result)
        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Export worker failed",
                exception_type=type(e).__name__,
                error=error_msg,
                exc_info=True,
            )
            self.signals.failed.emit(error_msg)
