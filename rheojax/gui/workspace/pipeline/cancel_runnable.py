"""Dispatches a worker's .cancel() off the GUI thread.

ProcessWorkerAdapter.cancel() (and FitWorker.cancel(), where applicable) performs a real
SIGTERM/SIGKILL escalation with blocking process.join(timeout=...) calls (up to ~7 seconds
combined by default). Calling it directly from the GUI thread would freeze the UI -- this
QRunnable exists so callers submit it to QThreadPool.globalInstance() instead.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRunnable


class CancelWorkerRunnable(QRunnable):
    def __init__(self, worker: Any) -> None:
        super().__init__()
        self._worker = worker

    def run(self) -> None:
        self._worker.cancel()
