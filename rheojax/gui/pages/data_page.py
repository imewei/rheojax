"""
Data Page
=========

Data loading, visualization, and preprocessing interface.
"""

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.state.store import StateStore


class DataPage(QWidget):
    """Data management page.

    Features:
        - Multi-format data import
        - Dataset tree view
        - Interactive data plotting
        - Data quality validation
        - Preprocessing controls

    Signals
    -------
    file_dropped : Signal(str)
    import_requested : Signal()
    apply_mapping : Signal()

    Example
    -------
    >>> page = DataPage()  # doctest: +SKIP
    >>> page.load_dataset('data.csv')  # doctest: +SKIP
    """

    file_dropped = Signal(str)
    import_requested = Signal()
    apply_mapping = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize data page.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._store = StateStore()
        self._current_file_path: Path | None = None
        self._preview_data: list[list] | None = None
        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left panel: File import
        left_panel = self._create_import_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel: Preview and mapping
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self._create_preview_panel())
        right_splitter.addWidget(self._create_mapper_panel())
        right_splitter.setSizes([400, 200])
        main_layout.addWidget(right_splitter, 2)

    def _create_import_panel(self) -> QWidget:
        """Create file import panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Drop zone
        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self._drop_zone, 1)

        # Browse button
        btn_browse = QPushButton("Browse Files...")
        btn_browse.clicked.connect(self._browse_files)
        layout.addWidget(btn_browse)

        # File info
        info_group = QGroupBox("File Information")
        info_layout = QVBoxLayout(info_group)

        self._file_name_label = QLabel("No file selected")
        self._file_name_label.setWordWrap(True)
        info_layout.addWidget(self._file_name_label)

        self._file_size_label = QLabel("")
        info_layout.addWidget(self._file_size_label)

        layout.addWidget(info_group)

        return panel

    def _create_preview_panel(self) -> QWidget:
        """Create data preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Data Preview (first 100 rows)")
        header.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 5px;")
        layout.addWidget(header)

        # Table
        self._preview_table = QTableWidget()
        self._preview_table.setAlternatingRowColors(True)
        layout.addWidget(self._preview_table)

        return panel

    def _create_mapper_panel(self) -> QWidget:
        """Create column mapper panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Column Mapping")
        header.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 5px;")
        layout.addWidget(header)

        # Mapping controls
        mapper_layout = QHBoxLayout()

        # X column
        x_layout = QVBoxLayout()
        x_layout.addWidget(QLabel("X Column:"))
        self._x_combo = QComboBox()
        x_layout.addWidget(self._x_combo)
        mapper_layout.addLayout(x_layout)

        # Y column
        y_layout = QVBoxLayout()
        y_layout.addWidget(QLabel("Y Column:"))
        self._y_combo = QComboBox()
        y_layout.addWidget(self._y_combo)
        mapper_layout.addLayout(y_layout)

        # Y2 column (optional)
        y2_layout = QVBoxLayout()
        y2_layout.addWidget(QLabel("Y2 Column (optional):"))
        self._y2_combo = QComboBox()
        self._y2_combo.addItem("None")
        y2_layout.addWidget(self._y2_combo)
        mapper_layout.addLayout(y2_layout)

        # Temperature column (optional)
        temp_layout = QVBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self._temp_combo = QComboBox()
        self._temp_combo.addItem("None")
        temp_layout.addWidget(self._temp_combo)
        mapper_layout.addLayout(temp_layout)

        layout.addLayout(mapper_layout)

        # Test mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Test Mode:"))
        self._test_mode_combo = QComboBox()
        self._test_mode_combo.addItems([
            "Auto-detect",
            "oscillation",
            "relaxation",
            "creep",
            "rotation"
        ])
        mode_layout.addWidget(self._test_mode_combo)
        layout.addLayout(mode_layout)

        # Metadata display
        metadata_label = QLabel("Metadata:")
        metadata_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(metadata_label)

        self._metadata_text = QTextEdit()
        self._metadata_text.setMaximumHeight(80)
        self._metadata_text.setReadOnly(True)
        layout.addWidget(self._metadata_text)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._reset_mapping)
        btn_layout.addWidget(btn_reset)

        btn_apply = QPushButton("Apply & Import")
        btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn_apply.clicked.connect(self._apply_import)
        btn_layout.addWidget(btn_apply)

        layout.addLayout(btn_layout)

        return panel

    def _browse_files(self) -> None:
        """Open file browser dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "Data Files (*.csv *.txt *.xlsx *.xls *.tri);;All Files (*)"
        )

        if file_path:
            self._on_file_dropped(file_path)

    def _on_file_dropped(self, file_path: str) -> None:
        """Handle file drop or selection."""
        self._current_file_path = Path(file_path)

        # Update file info
        self._file_name_label.setText(f"File: {self._current_file_path.name}")
        file_size = self._current_file_path.stat().st_size
        self._file_size_label.setText(f"Size: {file_size / 1024:.1f} KB")

        # Load and preview data
        self._load_preview()

        # Emit signal
        self.file_dropped.emit(file_path)

    def _load_preview(self) -> None:
        """Load and display data preview."""
        if not self._current_file_path:
            return

        try:
            # Load data using appropriate loader
            from rheojax.gui.services.data_service import DataService

            service = DataService()
            preview_result = service.preview_file(self._current_file_path, max_rows=100)

            self._preview_data = preview_result["data"]
            headers = preview_result.get("headers", [])
            metadata = preview_result.get("metadata", {})

            # Update table
            self._preview_table.clear()
            self._preview_table.setRowCount(len(self._preview_data))
            self._preview_table.setColumnCount(len(headers) if headers else len(self._preview_data[0]))

            if headers:
                self._preview_table.setHorizontalHeaderLabels(headers)
            else:
                self._preview_table.setHorizontalHeaderLabels([f"Col {i+1}" for i in range(len(self._preview_data[0]))])

            # Populate data
            for row_idx, row_data in enumerate(self._preview_data):
                for col_idx, value in enumerate(row_data):
                    item = QTableWidgetItem(str(value))
                    self._preview_table.setItem(row_idx, col_idx, item)

            # Update column mappers
            self._update_column_mappers(headers or [f"Col {i+1}" for i in range(len(self._preview_data[0]))])

            # Update metadata
            if metadata:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
                self._metadata_text.setText(metadata_text)

        except Exception as e:
            self._file_name_label.setText(f"Error loading file: {str(e)}")

    def _update_column_mappers(self, columns: list[str]) -> None:
        """Update column mapper dropdowns."""
        # Clear and repopulate
        for combo in [self._x_combo, self._y_combo, self._y2_combo, self._temp_combo]:
            combo.blockSignals(True)
            combo.clear()

        self._x_combo.addItems(columns)
        self._y_combo.addItems(columns)
        self._y2_combo.addItem("None")
        self._y2_combo.addItems(columns)
        self._temp_combo.addItem("None")
        self._temp_combo.addItems(columns)

        # Auto-select common names
        for idx, col in enumerate(columns):
            col_lower = col.lower()
            if any(x in col_lower for x in ["time", "freq", "omega", "t", "f", "w"]):
                self._x_combo.setCurrentIndex(idx)
            if any(y in col_lower for y in ["g'", "storage", "stress", "modulus"]):
                self._y_combo.setCurrentIndex(idx)
            if any(y2 in col_lower for y2 in ["g''", "loss"]):
                self._y2_combo.setCurrentIndex(idx + 1)  # +1 for "None" option

        for combo in [self._x_combo, self._y_combo, self._y2_combo, self._temp_combo]:
            combo.blockSignals(False)

    def _reset_mapping(self) -> None:
        """Reset column mapping to defaults."""
        if self._preview_data:
            headers = [self._preview_table.horizontalHeaderItem(i).text()
                      for i in range(self._preview_table.columnCount())]
            self._update_column_mappers(headers)

    def _apply_import(self) -> None:
        """Apply column mapping and import data."""
        if not self._current_file_path:
            return

        # Get mapping
        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()
        y2_col = self._y2_combo.currentText() if self._y2_combo.currentIndex() > 0 else None
        test_mode = self._test_mode_combo.currentText()
        if test_mode == "Auto-detect":
            test_mode = None  # Let service auto-detect

        # Import via service
        try:
            from rheojax.gui.services.data_service import DataService

            service = DataService()
            service.load_file(
                file_path=self._current_file_path,
                x_col=x_col,
                y_col=y_col,
                y2_col=y2_col,
                test_mode=test_mode
            )

            self.apply_mapping.emit()

        except Exception as e:
            self._file_name_label.setText(f"Import error: {str(e)}")

    def load_dataset(self, file_path: str) -> None:
        """Load dataset from file.

        Parameters
        ----------
        file_path : str
            Path to data file
        """
        self._on_file_dropped(file_path)

    def show_dataset(self, dataset_id: str) -> None:
        """Display dataset in plot canvas.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        """
        pass  # Implemented in future iteration

    def validate_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Validate dataset quality.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        dict
            Validation report
        """
        return {}  # Implemented in future iteration

    def apply_preprocessing(
        self,
        dataset_id: str,
        operations: list[str],
    ) -> None:
        """Apply preprocessing operations.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        operations : list[str]
            Preprocessing operation names
        """
        pass  # Implemented in future iteration


class DropZone(QFrame):
    """Drag-and-drop zone for file import."""

    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize drop zone."""
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
            QFrame:hover {
                border-color: #2196F3;
                background-color: #f0f7ff;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon_label = QLabel("Drop Files Here")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #999;")
        layout.addWidget(icon_label)

        text_label = QLabel("Drag and drop data file here\n\nor click Browse to select")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("color: #666; font-size: 12pt;")
        layout.addWidget(text_label)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop event."""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            self.file_dropped.emit(file_path)
            event.acceptProposedAction()
