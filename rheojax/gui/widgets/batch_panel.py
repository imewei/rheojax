"""
Batch Processing Panel Widget
==============================

Workspace panel for applying a pipeline to multiple files in a directory.
Displayed when the user selects the batch processing workflow.
"""

from __future__ import annotations

import os
from pathlib import Path

from rheojax.gui.compat import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Signal,
)
from rheojax.gui.resources.styles.tokens import Spacing
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Column indices for the file table
_COL_FILENAME = 0
_COL_STATUS = 1
_COL_TIME = 2

_STATUS_PENDING = "PENDING"
_STATUS_RUNNING = "RUNNING"
_STATUS_DONE = "DONE"
_STATUS_FAILED = "FAILED"


class BatchPanel(QWidget):
    """Batch processing workspace panel.

    Allows the user to select a directory, specify a file-name pattern, scan
    for matching files, and execute a batch pipeline run over all found files.

    Signals
    -------
    batch_requested(str, str, list)
        Emitted when the user clicks *Run Batch*.
        Carries: directory path, glob pattern, list of absolute file paths.
    batch_completed(int, int)
        Emitted when the batch run finishes.
        Carries: success count, total count.
    batch_cancelled()
        Emitted when the user cancels an in-progress batch run.

    Example
    -------
    >>> panel = BatchPanel()  # doctest: +SKIP
    >>> panel.batch_requested.connect(my_handler)  # doctest: +SKIP
    """

    batch_requested: Signal = Signal(str, str, list)
    batch_completed: Signal = Signal(int, int)
    batch_cancelled: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self._file_paths: list[str] = []
        self._is_running: bool = False
        self._setup_ui()
        self._update_run_button_state()
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the panel layout."""
        root = QVBoxLayout(self)
        root.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        root.setSpacing(Spacing.SM)

        # Title
        title = QLabel("Batch Processing")
        title.setStyleSheet("font-size: 13pt; font-weight: bold;")
        root.addWidget(title)

        # Directory + pattern group
        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(Spacing.XS)

        # Directory row
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select a directory…")
        self.dir_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        dir_row.addWidget(self.dir_edit)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setFixedWidth(80)
        self.browse_button.clicked.connect(self._on_browse)
        dir_row.addWidget(self.browse_button)
        source_layout.addLayout(dir_row)

        # Pattern row
        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("Pattern:  "))
        self.pattern_edit = QLineEdit("*.csv")
        self.pattern_edit.setMaximumWidth(200)
        pattern_row.addWidget(self.pattern_edit)
        self.scan_button = QPushButton("Scan")
        self.scan_button.setFixedWidth(80)
        self.scan_button.clicked.connect(self._on_scan)
        pattern_row.addWidget(self.scan_button)
        pattern_row.addStretch()
        source_layout.addLayout(pattern_row)

        root.addWidget(source_group)

        # File list group
        files_group = QGroupBox("Files Found")
        files_layout = QVBoxLayout(files_group)

        self.files_count_label = QLabel("Files Found: 0")
        self.files_count_label.setStyleSheet("font-size: 10pt;")
        files_layout.addWidget(self.files_count_label)

        self.file_table = QTableWidget(0, 3)
        self.file_table.setHorizontalHeaderLabels(["Filename", "Status", "Time (s)"])
        self.file_table.horizontalHeader().setSectionResizeMode(
            _COL_FILENAME, QHeaderView.ResizeMode.Stretch
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            _COL_STATUS, QHeaderView.ResizeMode.ResizeToContents
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            _COL_TIME, QHeaderView.ResizeMode.ResizeToContents
        )
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.setMinimumHeight(160)
        files_layout.addWidget(self.file_table)

        root.addWidget(files_group)

        # Progress + actions group
        progress_group = QGroupBox("Execution")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(Spacing.XS)

        # Action buttons row
        actions_row = QHBoxLayout()
        self.run_button = QPushButton("Run Batch")
        self.run_button.setFixedHeight(32)
        self.run_button.clicked.connect(self._on_run)
        actions_row.addWidget(self.run_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedHeight(32)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        actions_row.addWidget(self.cancel_button)
        actions_row.addStretch()
        progress_layout.addLayout(actions_row)

        # Progress bar row
        progress_row = QHBoxLayout()
        progress_row.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_row.addWidget(self.progress_bar)
        progress_layout.addLayout(progress_row)

        # Status label
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("font-size: 10pt; color: gray;")
        progress_layout.addWidget(self.status_label)

        root.addWidget(progress_group)
        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        """Open a directory picker and populate the directory field."""
        current = self.dir_edit.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            current,
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.dir_edit.setText(directory)
            logger.debug(
                "Directory selected",
                widget=self.__class__.__name__,
                directory=directory,
            )
            self._on_scan()

    def _on_scan(self) -> None:
        """Scan the directory for files matching the pattern."""
        directory = self.dir_edit.text().strip()
        pattern = self.pattern_edit.text().strip() or "*.csv"

        if not directory:
            self.status_label.setText("No directory specified.")
            return

        dir_path = Path(directory)
        if not dir_path.is_dir():
            self.status_label.setText(f"Directory not found: {directory}")
            logger.warning(
                "Directory not found",
                widget=self.__class__.__name__,
                directory=directory,
            )
            return

        matched = sorted(dir_path.glob(pattern))
        self._file_paths = [str(p) for p in matched]

        self._populate_table(self._file_paths)
        count = len(self._file_paths)
        self.files_count_label.setText(f"Files Found: {count}")
        self.status_label.setText(
            f"Found {count} file(s) matching '{pattern}'."
            if count
            else f"No files matching '{pattern}'."
        )
        self.progress_bar.setValue(0)
        self._update_run_button_state()

        logger.debug(
            "Scan complete",
            widget=self.__class__.__name__,
            directory=directory,
            pattern=pattern,
            count=count,
        )

    def _on_run(self) -> None:
        """Emit batch_requested with the current directory, pattern, and file list."""
        if not self._file_paths:
            self.status_label.setText("No files to process. Run Scan first.")
            return

        directory = self.dir_edit.text().strip()
        pattern = self.pattern_edit.text().strip() or "*.csv"

        self._is_running = True
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.browse_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting batch run…")

        logger.info(
            "Batch requested",
            widget=self.__class__.__name__,
            directory=directory,
            pattern=pattern,
            file_count=len(self._file_paths),
        )
        self.batch_requested.emit(directory, pattern, list(self._file_paths))

    def _on_cancel(self) -> None:
        """Cancel an in-progress batch run."""
        if not self._is_running:
            return
        self._is_running = False
        self._set_idle_state()
        self.status_label.setText("Cancelled by user.")
        logger.info("Batch cancelled", widget=self.__class__.__name__)
        self.batch_cancelled.emit()

    # ------------------------------------------------------------------
    # Public API — called by the controller/job layer
    # ------------------------------------------------------------------

    def set_file_status(
        self,
        file_path: str,
        status: str,
        elapsed: float | None = None,
    ) -> None:
        """Update the status cell for a single file row.

        Parameters
        ----------
        file_path:
            Absolute path (must be in the current file list).
        status:
            One of ``PENDING``, ``RUNNING``, ``DONE``, ``FAILED``.
        elapsed:
            Elapsed time in seconds, displayed in the Time column.
        """
        # Two-pass lookup: prefer exact full-path match (UserRole), fall back
        # to basename only when no full-path match is found.  This prevents
        # duplicate basenames (e.g. sub1/data.csv and sub2/data.csv) from
        # updating the wrong row.
        target_row = None
        basename = os.path.basename(file_path)
        for row in range(self.file_table.rowCount()):
            item = self.file_table.item(row, _COL_FILENAME)
            if item is None:
                continue
            stored_path = item.data(Qt.ItemDataRole.UserRole)
            if stored_path == file_path:
                target_row = row
                break
            if target_row is None and item.text() == basename:
                target_row = row  # tentative basename match; keep scanning

        if target_row is not None:
            status_item = QTableWidgetItem(status)
            if status == _STATUS_DONE:
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            elif status == _STATUS_FAILED:
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == _STATUS_RUNNING:
                status_item.setForeground(Qt.GlobalColor.darkBlue)
            self.file_table.setItem(target_row, _COL_STATUS, status_item)
            if elapsed is not None:
                self.file_table.setItem(
                    target_row, _COL_TIME, QTableWidgetItem(f"{elapsed:.2f}")
                )

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        """Update the progress bar and status label.

        Parameters
        ----------
        current:
            Number of files processed so far.
        total:
            Total number of files.
        message:
            Optional status message shown below the progress bar.
        """
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))
        if message:
            self.status_label.setText(message)

    def finish_batch(self, success_count: int, total_count: int) -> None:
        """Mark the batch run as complete.

        Parameters
        ----------
        success_count:
            Number of files that completed without error.
        total_count:
            Total files processed.
        """
        self._is_running = False
        self._set_idle_state()
        self.progress_bar.setValue(100)
        self.status_label.setText(
            f"Batch complete: {success_count}/{total_count} succeeded."
        )
        logger.info(
            "Batch finished",
            widget=self.__class__.__name__,
            success=success_count,
            total=total_count,
        )
        self.batch_completed.emit(success_count, total_count)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _populate_table(self, file_paths: list[str]) -> None:
        """Fill the table with filenames, all set to PENDING."""
        self.file_table.setRowCount(0)
        for path in file_paths:
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            name_item = QTableWidgetItem(os.path.basename(path))
            # Store the full path as tooltip so set_file_status can match
            # unambiguously when two files share the same basename (e.g.
            # sub-dir scans with a recursive glob pattern).
            name_item.setToolTip(path)
            # UserRole stores the full absolute path for O(n) exact matching.
            name_item.setData(Qt.ItemDataRole.UserRole, path)
            self.file_table.setItem(row, _COL_FILENAME, name_item)
            self.file_table.setItem(row, _COL_STATUS, QTableWidgetItem(_STATUS_PENDING))
            self.file_table.setItem(row, _COL_TIME, QTableWidgetItem(""))

    def _update_run_button_state(self) -> None:
        """Enable Run Batch only when there are files to process."""
        self.run_button.setEnabled(bool(self._file_paths) and not self._is_running)

    def _set_idle_state(self) -> None:
        """Re-enable controls after a run finishes or is cancelled."""
        self.run_button.setEnabled(bool(self._file_paths))
        self.cancel_button.setEnabled(False)
        self.browse_button.setEnabled(True)
        self.scan_button.setEnabled(True)
