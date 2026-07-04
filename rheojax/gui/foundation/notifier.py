"""Qt signal carrier for DatasetLibrary mutations.

DatasetLibrary itself stays a plain, Qt-unaware class (required for the project codec's
decode-into-a-temporary-AppState path). This notifier is owned by WorkspaceWindow, constructed
exactly once, and is what call sites emit into after a library mutation -- see
WorkspaceWindow._commit_dataset() (window.py).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class DatasetLibraryNotifier(QObject):
    changed = Signal()
