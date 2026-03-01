"""
Preview Worker
=============

Background worker for data file preview loading.

Offloads file I/O from the Qt main thread to prevent UI freezes
when loading large data files.
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

    def _detect_file_format_background(self) -> str:
        """F-GUI-011 fix: detect file format in the background thread.

        Reads the first 2 KB of the file to identify the format.  Running
        this here avoids blocking the Qt main thread with file I/O.

        Returns
        -------
        str
            Human-readable format name (e.g. 'TA Instruments TRIOS', 'CSV').
        """
        if not self._file_path or not self._file_path.exists():
            return "Unknown"

        suffix = self._file_path.suffix.lower()
        name_lower = self._file_path.name.lower()

        def _read_header() -> str:
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    with open(self._file_path, encoding=enc, errors="strict") as fh:
                        return fh.read(2000)
                except (UnicodeDecodeError, UnicodeError):
                    continue
            # Fallback: replace undecodable bytes
            with open(self._file_path, encoding="utf-8", errors="replace") as fh:
                return fh.read(2000)

        if suffix == ".txt":
            try:
                first_lines = _read_header()
                if "trios" in first_lines.lower() or "[file" in first_lines.lower():
                    return "TA Instruments TRIOS"
                if (
                    "rheometer" in first_lines.lower()
                    or "rheocompass" in first_lines.lower()
                ):
                    return "Anton Paar RheoCompass"
            except Exception:
                pass
            return "Text/CSV"

        if suffix == ".csv":
            try:
                first_lines = _read_header()
                if (
                    "rheometer" in first_lines.lower()
                    or "rheocompass" in first_lines.lower()
                ):
                    return "Anton Paar RheoCompass"
                if "trios" in first_lines.lower():
                    return "TA Instruments TRIOS CSV"
            except Exception:
                pass
            return "CSV"

        if suffix in {".xlsx", ".xls"}:
            if "trios" in name_lower:
                return "TA Instruments TRIOS Excel"
            return "Excel"

        if suffix == ".tri":
            return "TA Instruments TRIOS Binary"

        if suffix in {".rdf", ".dat"}:
            return "Rheological Data"

        return suffix.upper().lstrip(".")

    def run(self) -> None:
        """Execute file preview in background thread."""
        try:
            file_size = (
                self._file_path.stat().st_size if self._file_path.exists() else -1
            )
            logger.info(
                "Preview worker started",
                filepath=str(self._file_path),
                file_size_bytes=file_size,
                max_rows=self._max_rows,
            )

            # NOTE: Cancellation is not implemented for PreviewWorker — the file
            # I/O call below is synchronous and non-interruptible.

            preview_result = self._data_service.preview_file(
                self._file_path, max_rows=self._max_rows
            )

            # F-GUI-011 fix: detect format here (in the worker thread) so that
            # DataPage._on_preview_loaded() does not have to block the main thread
            # on file I/O.  The result is embedded in the preview metadata dict.
            detected_format = self._detect_file_format_background()
            preview_result.setdefault("metadata", {})
            if isinstance(preview_result["metadata"], dict):
                preview_result["metadata"].setdefault("format", detected_format)

            n_rows = len(preview_result.get("data", []))
            headers = preview_result.get("headers", [])
            metadata = preview_result.get("metadata", {})
            warnings_list = preview_result.get("warnings", [])
            detected_cols = (
                metadata.get("detected_columns", {})
                if isinstance(metadata, dict)
                else {}
            )

            logger.info(
                "Preview worker completed",
                filepath=str(self._file_path),
                n_rows_parsed=n_rows,
                n_columns=len(headers),
                detected_format=detected_format,
            )
            logger.debug(
                "Preview column auto-detection result",
                headers=headers,
                detected_columns=detected_cols,
            )
            if warnings_list:
                logger.warning(
                    "Preview parse produced warnings",
                    filepath=str(self._file_path),
                    warnings=warnings_list,
                )

            self.signals.completed.emit(preview_result)

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Preview worker failed",
                filepath=str(self._file_path),
                exception_type=type(e).__name__,
                error=error_msg,
                exc_info=True,
            )
            self.signals.failed.emit(error_msg)
