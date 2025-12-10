"""
Data Page
=========

Data loading, visualization, and preprocessing interface.
"""

from pathlib import Path
from typing import Any
import logging

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
from rheojax.gui.services.data_service import DataService


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
        self._data_service = DataService()
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Drop zone
        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._on_file_dropped)
        self._drop_zone.setToolTip("Drag and drop a data file to load it")
        layout.addWidget(self._drop_zone, 1)

        # Browse button
        btn_browse = QPushButton("Browse Files...")
        btn_browse.clicked.connect(self._browse_files)
        layout.addWidget(btn_browse)

        # Example datasets dropdown
        example_group = QGroupBox("Load Example Dataset")
        example_group.setStyleSheet("QGroupBox { font-size: 10.5pt; font-weight: bold; }")
        example_layout = QHBoxLayout(example_group)
        self._example_combo = QComboBox()
        self._example_combo.setPlaceholderText("Select an example dataset")
        self._example_paths = {
            "Polypropylene Relaxation (basic)": "examples/data/experimental/polypropylene_relaxation.csv",
            "Polystyrene Creep (basic)": "examples/data/experimental/polystyrene_creep.csv",
            "Frequency Sweep TTS (mastercurve)": "examples/data/experimental/frequency_sweep_tts.txt",
            "Multi-technique (advanced)": "examples/data/experimental/multi_technique.txt",
            "OWChirp TTS": "examples/data/experimental/owchirp_tts.txt",
            "OWChirp TCS": "examples/data/experimental/owchirp_tcs.txt",
        }
        for label in self._example_paths:
            self._example_combo.addItem(label)
        example_layout.addWidget(self._example_combo, 1)

        btn_example = QPushButton("Load")
        btn_example.clicked.connect(self._load_example_dataset)
        example_layout.addWidget(btn_example)
        layout.addWidget(example_group)

        # File info
        info_group = QGroupBox("File Information")
        info_group.setStyleSheet("QGroupBox { font-size: 11pt; font-weight: bold; }")
        info_layout = QVBoxLayout(info_group)

        self._file_name_label = QLabel("No file selected")
        self._file_name_label.setWordWrap(True)
        self._file_name_label.setStyleSheet("font-size: 11pt; font-weight: 500;")
        info_layout.addWidget(self._file_name_label)

        self._file_size_label = QLabel("")
        self._file_size_label.setStyleSheet("font-size: 10pt; color: #444;")
        info_layout.addWidget(self._file_size_label)

        layout.addWidget(info_group)

        return panel

    def _load_example_dataset(self) -> None:
        """Load a bundled example dataset quickly."""
        selection = self._example_combo.currentText()
        rel_path = self._example_paths.get(selection)
        if not rel_path:
            return
        path = Path(rel_path)
        if not path.exists():
            # Try resolving from project root
            alt = Path.cwd() / rel_path
            path = alt if alt.exists() else path
        if not path.exists():
            self._file_name_label.setText(f"Example not found: {rel_path}")
            return
        self.load_dataset(str(path))

    def _create_preview_panel(self) -> QWidget:
        """Create data preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Data Preview (first 100 rows)")
        header.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 5px;")
        layout.addWidget(header)

        # Table
        self._preview_table = QTableWidget()
        self._preview_table.setAlternatingRowColors(True)
        self._preview_table.setToolTip("Preview of the loaded dataset. Shows up to 100 rows.")
        layout.addWidget(self._preview_table)

        # Empty state
        empty_label = QLabel("No dataset loaded. Use Browse or Load Example to begin.")
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet("color: #666; padding: 8px;")
        layout.addWidget(empty_label)
        self._empty_label = empty_label

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
        btn_reset.setProperty("variant", "secondary")
        btn_reset.clicked.connect(self._reset_mapping)
        btn_layout.addWidget(btn_reset)

        btn_apply = QPushButton("Apply & Import")
        btn_apply.setProperty("variant", "primary")
        btn_apply.clicked.connect(self._apply_import)
        btn_layout.addWidget(btn_apply)

        layout.addLayout(btn_layout)

        # Empty state placeholder
        self._empty_state = QLabel("Drop a file or browse to import data")
        self._empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state.setStyleSheet("color: #94A3B8; padding: 8px;")
        layout.addWidget(self._empty_state)

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
        path_obj = Path(file_path)

        # Memory guard: warn on large files (>50 MB)
        size_bytes = path_obj.stat().st_size
        if size_bytes > 50 * 1024 * 1024:
            resp = QMessageBox.warning(
                self,
                "Large File",
                "The selected file is larger than 50 MB. Loading may be slow or memory intensive. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp == QMessageBox.StandardButton.No:
                return

        self._current_file_path = path_obj

        # Update file info
        self._file_name_label.setText(f"File: {self._current_file_path.name}")
        self._file_size_label.setText(f"Size: {size_bytes / 1024:.1f} KB")

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

            self._preview_data = preview_result.get("data", [])
            headers = preview_result.get("headers", [])
            metadata = preview_result.get("metadata", {})

            if not self._preview_data:
                self._preview_table.clear()
                self._file_name_label.setText("File loaded but no previewable rows")
                self._metadata_text.clear()
                if hasattr(self, "_empty_state"):
                    self._empty_state.show()
                return

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
            else:
                self._metadata_text.clear()

            if hasattr(self, "_empty_state"):
                self._empty_state.hide()

        except Exception as e:
            self._file_name_label.setText(f"Error loading file: {str(e)}")
            self._file_size_label.setText("")
            self._preview_table.clear()
            self._metadata_text.clear()

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
        import uuid

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
            rheo_data = service.load_file(
                file_path=self._current_file_path,
                x_col=x_col,
                y_col=y_col,
                y2_col=y2_col,
                test_mode=test_mode
            )

            # Auto-detect test mode if not specified
            if test_mode is None:
                test_mode = service.detect_test_mode(rheo_data)

            # Generate dataset_id before dispatch to ensure signal emission
            dataset_id = str(uuid.uuid4())

            # Register dataset in state store
            store = StateStore()
            store.dispatch(
                "IMPORT_DATA_SUCCESS",
                {
                    "dataset_id": dataset_id,
                    "file_path": str(self._current_file_path),
                    "name": self._current_file_path.stem,
                    "test_mode": test_mode or "unknown",
                    "x_data": rheo_data.x,
                    "y_data": rheo_data.y,
                    "y2_data": getattr(rheo_data, "y2", None),
                    "metadata": getattr(rheo_data, "metadata", {}),
                },
            )

            self.apply_mapping.emit()
            self._store.dispatch("SET_TAB", {"tab": "transform"})

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
        from rheojax.gui.state.store import StateStore

        store = StateStore()
        dataset = store.get_dataset(dataset_id)
        if dataset is None:
            return

        # Build a simple table view from stored data
        x_vals = dataset.x_data or []
        y_vals = dataset.y_data or []
        y2_vals = dataset.y2_data or []

        # Determine columns
        headers = ["x", "y"]
        has_y2 = y2_vals is not None and len(y2_vals) > 0
        if has_y2:
            headers.append("y2")

        rows = min(len(x_vals), len(y_vals), 100)

        self._preview_table.clear()
        self._preview_table.setRowCount(rows)
        self._preview_table.setColumnCount(len(headers))
        self._preview_table.setHorizontalHeaderLabels(headers)

        for i in range(rows):
            self._preview_table.setItem(i, 0, QTableWidgetItem(str(x_vals[i])))
            self._preview_table.setItem(i, 1, QTableWidgetItem(str(y_vals[i])))
            if has_y2:
                self._preview_table.setItem(i, 2, QTableWidgetItem(str(y2_vals[i])))

        # Update column mappers to reflect the loaded dataset
        self._update_column_mappers(headers)

        # Update metadata text with validation info
        validation = self.validate_dataset(dataset_id)
        warnings = validation.get("warnings", []) if isinstance(validation, dict) else []
        inferred_mode = validation.get("test_mode") if isinstance(validation, dict) else dataset.test_mode

        metadata_lines = [
            f"rows: {len(x_vals)}",
            f"columns: {len(headers)}",
            f"file: {dataset.file_path.name if dataset.file_path else 'N/A'}",
            f"test_mode: {inferred_mode or dataset.test_mode}",
        ]
        if warnings:
            metadata_lines.append("warnings:")
            metadata_lines.extend([f"- {w}" for w in warnings])
        self._metadata_text.setText("\n".join(metadata_lines))

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
        store = StateStore()
        dataset = store.get_dataset(dataset_id)
        if dataset is None:
            return {"warnings": ["Dataset not found"], "test_mode": "unknown"}

        try:
            rheo_data = self._data_service.to_rheo_data(dataset)
            warnings = self._data_service.validate_data(rheo_data)
            test_mode = self._data_service.detect_test_mode(rheo_data)
        except Exception as exc:  # Defensive: never raise into UI
            logging.getLogger(__name__).warning("Dataset validation failed: %s", exc)
            return {"warnings": [f"Validation error: {exc}"], "test_mode": "unknown"}

        return {"warnings": warnings, "test_mode": test_mode}

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
        store = StateStore()
        dataset = store.get_dataset(dataset_id)
        if dataset is None:
            return

        try:
            rheo_data = self._data_service.to_rheo_data(dataset)

            # Map requested operations to DataService flags
            remove_outliers = "remove_outliers" in operations
            smooth = "smooth" in operations
            kwargs: dict[str, Any] = {}

            processed = self._data_service.preprocess_data(
                rheo_data,
                remove_outliers=remove_outliers,
                smooth=smooth,
                **kwargs,
            )

            # Persist back into state as a modified dataset
            payload = {
                "dataset_id": dataset.id,
                "file_path": str(dataset.file_path) if dataset.file_path else None,
                "name": dataset.name,
                "test_mode": processed.metadata.get("test_mode", dataset.test_mode),
                "x_data": processed.x,
                "y_data": processed.y,
                "y2_data": getattr(processed, "y2", None),
                "metadata": processed.metadata,
            }
            store.dispatch("IMPORT_DATA_SUCCESS", payload)

            # Refresh preview/mapping UI
            self.show_dataset(dataset.id)

        except Exception as exc:
            logging.getLogger(__name__).warning("Preprocessing failed: %s", exc)



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
