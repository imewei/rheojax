"""
Import Worker
=============

Background worker for data file import.

Offloads file I/O and parsing from the Qt main thread to prevent UI
freezes when importing large or multi-segment data files.
"""

from __future__ import annotations

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


class ImportWorkerSignals(QObject):
    """Signals for ImportWorker.

    Signals
    -------
    completed : Signal(object)
        Import succeeded — payload is ``list[RheoData]``.
    failed : Signal(str)
        Import failed — payload is the error message.
    """

    completed = Signal(object)  # list[RheoData]
    failed = Signal(str)  # error message


class ImportWorker(QRunnable):
    """Worker for importing data files in a background thread.

    Wraps ``DataService.load_file_multi`` so that the Qt main thread
    remains responsive during heavy file I/O and parsing.

    Parameters
    ----------
    data_service : DataService
        Service instance used to load data.
    file_path : Path
        Path to the data file.
    x_col, y_col, y2_col : str or None
        Column name overrides.
    test_mode : str or None
        Explicit test mode (``None`` = auto-detect).
    """

    def __init__(
        self,
        data_service: Any,
        file_path: Path,
        x_col: str | None = None,
        y_col: str | None = None,
        y2_col: str | None = None,
        test_mode: str | None = None,
    ) -> None:
        if not HAS_PYSIDE6:
            raise ImportError(
                "PySide6 is required for ImportWorker. "
                "Install with: pip install PySide6"
            )
        super().__init__()
        self.signals = ImportWorkerSignals()
        self._data_service = data_service
        self._file_path = file_path
        self._x_col = x_col
        self._y_col = y_col
        self._y2_col = y2_col
        self._test_mode = test_mode

    def run(self) -> None:
        """Execute file import in background thread."""
        try:
            logger.debug(
                "Import worker started",
                filepath=str(self._file_path),
            )

            datasets = self._data_service.load_file_multi(
                self._file_path,
                x_col=self._x_col,
                y_col=self._y_col,
                y2_col=self._y2_col,
                test_mode=self._test_mode,
            )

            logger.debug(
                "Import worker completed",
                filepath=str(self._file_path),
                num_segments=len(datasets),
            )

            self.signals.completed.emit(datasets)

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Import worker failed",
                filepath=str(self._file_path),
                error=error_msg,
                exc_info=True,
            )
            self.signals.failed.emit(error_msg)
