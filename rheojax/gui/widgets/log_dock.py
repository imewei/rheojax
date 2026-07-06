"""
Shared GUI Log Dock
===================

A dockable, filterable, severity-colored log viewer shared by both the
workspace shell (`WorkspaceWindow`) and the legacy `RheoJAXMainWindow`.
Fed by `rheojax.gui.utils.logging.install_gui_log_handler`.
"""

from __future__ import annotations

import logging
from collections import deque

from rheojax.gui.compat import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_MAX_RECORDS = 2000

_LEVEL_NAMES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_LEVEL_COLORS = {
    logging.DEBUG: "#8a8a8a",
    logging.INFO: None,  # default text color
    logging.WARNING: "#b8860b",
    logging.ERROR: "#cc3333",
    logging.CRITICAL: "#cc3333",
}


class LogDockWidget(QDockWidget):
    """Bottom dock: buffered, filterable, severity-colored log viewer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Log", parent)
        self.setObjectName("LogDock")

        self._records: deque[tuple[int, str]] = deque(maxlen=_MAX_RECORDS)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Level:", container))
        self.level_combo = QComboBox(container)
        self.level_combo.addItems(_LEVEL_NAMES)
        self.level_combo.setCurrentText("INFO")
        self.level_combo.currentTextChanged.connect(self._rerender)
        toolbar.addWidget(self.level_combo)
        toolbar.addStretch(1)
        self.save_button = QPushButton("Save...", container)
        self.save_button.clicked.connect(self._on_save_clicked)
        toolbar.addWidget(self.save_button)
        layout.addLayout(toolbar)

        self.text_edit = QTextEdit(container)
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumHeight(200)
        self.text_edit.setPlaceholderText("Application logs will appear here...")
        layout.addWidget(self.text_edit)

        self.setWidget(container)

    def _current_threshold(self) -> int:
        return getattr(logging, self.level_combo.currentText())

    def append_record(self, levelno: int, message: str) -> None:
        """Buffer a log record and render it if it passes the current filter."""
        self._records.append((levelno, message))
        if levelno >= self._current_threshold():
            self._append_line(levelno, message)

    def _append_line(self, levelno: int, message: str) -> None:
        color = _LEVEL_COLORS.get(levelno)
        if color:
            self.text_edit.append(f'<span style="color:{color};">{message}</span>')
        else:
            self.text_edit.append(message)

    def _rerender(self) -> None:
        self.text_edit.clear()
        threshold = self._current_threshold()
        for levelno, message in self._records:
            if levelno >= threshold:
                self._append_line(levelno, message)

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "rheojax-gui.log", "Log Files (*.log);;All Files (*)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.text_edit.toPlainText())
