"""
Preview Worker
=============

Background worker for data file preview loading.

Offloads file I/O from the Qt main thread to prevent UI freezes
when loading large data files.
"""

from __future__ import annotations

import traceback
from pathlib import Path
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


class PreviewWorkerSignals(QObject):
    """Signals for PreviewWorker.

    Signals
    -------
    completed : Signal(dict)
        Preview loaded successfully with result dict containing
        'data', 'headers', and 'metadata' keys.
    failed : Signal(str)
        Preview loading failed with error message.
    """

    completed = Signal(object)  # dict with preview result
    failed = Signal(str)  # error message


class PreviewWorker(QRunnable):
    """Worker for loading file previews in a background thread.

    Prevents UI freezes when previewing large data files by offloading
    the pandas read operation to a worker thread.

    Example
    -------
    >>> worker = PreviewWorker(  # doctest: +SKIP
    ...     data_service=service,
    ...     file_path=Path("data.csv"),
    ...     max_rows=100,
    ... )
    >>> worker.signals.completed.connect(on_preview_ready)  # doctest: +SKIP
    >>> QThreadPool.globalInstance().start(worker)  # doctest: +SKIP
    """

    def __init__(
        self,
        data_service: Any,
        file_path: Path,
        max_rows: int = 100,
    ) -> None:
        if not HAS_PYSIDE6:
            raise ImportError(
                "PySide6 is required for PreviewWorker. "
                "Install with: pip install PySide6"
            )

        super().__init__()
        self.signals = PreviewWorkerSignals()
        self._data_service = data_service
        self._file_path = file_path
        self._max_rows = max_rows

    def run(self) -> None:
        """Execute file preview in background thread."""
        try:
            logger.debug(
                "Preview worker started",
                filepath=str(self._file_path),
                max_rows=self._max_rows,
            )

            preview_result = self._data_service.preview_file(
                self._file_path, max_rows=self._max_rows
            )

            logger.debug(
                "Preview worker completed",
                filepath=str(self._file_path),
                rows=len(preview_result.get("data", [])),
            )

            self.signals.completed.emit(preview_result)

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Preview worker failed",
                filepath=str(self._file_path),
                error=error_msg,
                exc_info=True,
            )
            self.signals.failed.emit(error_msg)
