"""
Import Wizard Dialog
===================

Step-by-step data import wizard.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from rheojax.logging import get_logger

logger = get_logger(__name__)


class FileSelectionPage(QWizardPage):
    """File selection page."""

    def __init__(self, parent: QWizard | None = None) -> None:
        """Initialize file selection page."""
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self.setTitle("Select Data File")
        self.setSubTitle("Choose a file to import rheological data from")

        layout = QVBoxLayout()

        # File path input
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a file or drag and drop...")
        self.file_path_edit.textChanged.connect(self._on_file_changed)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_file)

        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_button)
        file_group.setLayout(file_layout)

        layout.addWidget(file_group)

        # File info display
        self.info_label = QLabel("No file selected")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        layout.addStretch()
        self.setLayout(layout)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Register field for wizard navigation
        self.registerField("file_path*", self.file_path_edit)

    def _browse_file(self) -> None:
        """Open file browser dialog."""
        logger.debug("Opening file browser", page=self.__class__.__name__)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "All Supported (*.csv *.txt *.xlsx *.xls *.tri);;CSV Files (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx *.xls);;TRIOS Files (*.tri);;All Files (*.*)",
        )
        if file_path:
            self.file_path_edit.setText(file_path)
            logger.debug(
                "Option changed",
                dialog="ImportWizard",
                option="file_path",
                value=file_path,
            )

    def _on_file_changed(self, text: str) -> None:
        """Handle file path change."""
        if text:
            path = Path(text)
            if path.exists() and path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.info_label.setText(
                    f"<b>File:</b> {path.name}<br>"
                    f"<b>Size:</b> {size_mb:.2f} MB<br>"
                    f"<b>Type:</b> {path.suffix.upper()[1:]}"
                )
                self.completeChanged.emit()
                logger.debug(
                    "Option changed",
                    dialog="ImportWizard",
                    option="file_path",
                    value=text,
                )
            else:
                self.info_label.setText("<font color='red'>Invalid file path</font>")
        else:
            self.info_label.setText("No file selected")

    def isComplete(self) -> bool:
        """Check if page is complete."""
        text = self.file_path_edit.text()
        if not text:
            return False
        path = Path(text)
        return path.exists() and path.is_file()

    def dragEnterEvent(self, event) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_path_edit.setText(file_path)
            logger.debug(
                "Option changed",
                dialog="ImportWizard",
                option="file_path",
                value=file_path,
            )


class ColumnMappingPage(QWizardPage):
    """Column mapping page."""

    def __init__(self, parent: QWizard | None = None) -> None:
        """Initialize column mapping page."""
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self.setTitle("Map Columns")
        self.setSubTitle("Assign columns to rheological data types")

        layout = QVBoxLayout()

        # Auto-detect button
        auto_detect_button = QPushButton("Auto-Detect Columns")
        auto_detect_button.clicked.connect(self._auto_detect)
        layout.addWidget(auto_detect_button)

        # Column mapping group
        mapping_group = QGroupBox("Column Assignment")
        mapping_layout = QVBoxLayout()

        # X axis (frequency/time)
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X (Frequency/Time):"))
        self.x_combo = QComboBox()
        # Ignore the emitted text; we only care that completion state may have changed
        self.x_combo.currentTextChanged.connect(self._on_x_column_changed)
        x_layout.addWidget(self.x_combo, 1)
        mapping_layout.addLayout(x_layout)

        # Y axis (modulus/compliance/viscosity)
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y (Modulus/Viscosity):"))
        self.y_combo = QComboBox()
        self.y_combo.currentTextChanged.connect(self._on_y_column_changed)
        y_layout.addWidget(self.y_combo, 1)
        mapping_layout.addLayout(y_layout)

        # Y2 axis (optional, e.g., G'')
        y2_layout = QHBoxLayout()
        y2_layout.addWidget(QLabel("Y2 (Optional, e.g., G''):"))
        self.y2_combo = QComboBox()
        self.y2_combo.currentTextChanged.connect(self._on_y2_column_changed)
        y2_layout.addWidget(self.y2_combo, 1)
        mapping_layout.addLayout(y2_layout)

        # Temperature (optional)
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature (Optional):"))
        self.temp_combo = QComboBox()
        self.temp_combo.currentTextChanged.connect(self._on_temp_column_changed)
        temp_layout.addWidget(self.temp_combo, 1)
        mapping_layout.addLayout(temp_layout)

        mapping_group.setLayout(mapping_layout)
        layout.addWidget(mapping_group)

        # Preview table
        preview_label = QLabel("<b>Preview (First 5 Rows):</b>")
        layout.addWidget(preview_label)

        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(150)
        layout.addWidget(self.preview_table)

        self.setLayout(layout)

        # Register fields
        self.registerField("x_column*", self.x_combo, "currentText")
        self.registerField("y_column*", self.y_combo, "currentText")
        self.registerField("y2_column", self.y2_combo, "currentText")
        self.registerField("temp_column", self.temp_combo, "currentText")

    def _on_x_column_changed(self, text: str) -> None:
        """Handle x column change."""
        self.completeChanged.emit()
        if text:
            logger.debug(
                "Option changed",
                dialog="ImportWizard",
                option="x_column",
                value=text,
            )

    def _on_y_column_changed(self, text: str) -> None:
        """Handle y column change."""
        self.completeChanged.emit()
        if text:
            logger.debug(
                "Option changed",
                dialog="ImportWizard",
                option="y_column",
                value=text,
            )

    def _on_y2_column_changed(self, text: str) -> None:
        """Handle y2 column change."""
        if text:
            logger.debug(
                "Option changed",
                dialog="ImportWizard",
                option="y2_column",
                value=text,
            )

    def _on_temp_column_changed(self, text: str) -> None:
        """Handle temp column change."""
        if text:
            logger.debug(
                "Option changed",
                dialog="ImportWizard",
                option="temp_column",
                value=text,
            )

    def initializePage(self) -> None:
        """Initialize page when shown."""
        file_path = self.field("file_path")
        logger.debug(
            "Loading columns from file",
            page=self.__class__.__name__,
            file_path=file_path,
        )
        self._load_columns(file_path)
        self._auto_detect()

    def _load_columns(self, file_path: str) -> None:
        """Load columns from file."""
        try:
            # Read file to get columns
            path = Path(file_path)
            if path.suffix.lower() in [".csv", ".txt"]:
                from rheojax.io.readers.csv_reader import detect_csv_delimiter

                delimiter = detect_csv_delimiter(file_path)
                df = pd.read_csv(file_path, sep=delimiter, nrows=5)
            elif path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, nrows=5)
            else:
                # For TRIOS or other formats, try CSV first
                from rheojax.io.readers.csv_reader import detect_csv_delimiter

                delimiter = detect_csv_delimiter(file_path)
                df = pd.read_csv(file_path, sep=delimiter, nrows=5)

            columns = list(df.columns)
            logger.debug(
                "Columns loaded",
                page=self.__class__.__name__,
                num_columns=len(columns),
            )

            # Populate combo boxes
            self.x_combo.clear()
            self.y_combo.clear()
            self.y2_combo.clear()
            self.temp_combo.clear()

            self.x_combo.addItems(columns)
            self.y_combo.addItems(columns)

            self.y2_combo.addItem("(None)")
            self.y2_combo.addItems(columns)

            self.temp_combo.addItem("(None)")
            self.temp_combo.addItems(columns)

            # Update preview
            self._update_preview(df)

        except Exception as e:
            logger.error(
                "Failed to load columns",
                page=self.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            QMessageBox.warning(self, "Error", f"Failed to load columns: {e}")

    def _update_preview(self, df: pd.DataFrame) -> None:
        """Update preview table."""
        self.preview_table.setRowCount(len(df))
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels([str(col) for col in df.columns])

        for i in range(len(df)):
            for j in range(len(df.columns)):
                value = str(df.iloc[i, j])
                self.preview_table.setItem(i, j, QTableWidgetItem(value))

        self.preview_table.resizeColumnsToContents()

    def _auto_detect(self) -> None:
        """Auto-detect column mapping."""
        logger.debug("Auto-detecting column mapping", page=self.__class__.__name__)
        # Common column name patterns
        x_patterns = ["freq", "frequency", "omega", "time", "t", "w"]
        y_patterns = ["g'", "gp", "storage", "modulus", "eta", "viscosity", "stress"]
        y2_patterns = ["g''", "gpp", "loss"]
        temp_patterns = ["temp", "temperature", "t"]

        def find_match(combo: QComboBox, patterns: list[str]) -> None:
            """Find best match for column."""
            for i in range(combo.count()):
                text = combo.itemText(i).lower()
                for pattern in patterns:
                    if pattern in text:
                        combo.setCurrentIndex(i)
                        return

        find_match(self.x_combo, x_patterns)
        find_match(self.y_combo, y_patterns)
        find_match(self.y2_combo, y2_patterns)
        find_match(self.temp_combo, temp_patterns)

    def isComplete(self) -> bool:
        """Check if page is complete."""
        return bool(self.x_combo.currentText()) and bool(self.y_combo.currentText())


class TestModeSelectionPage(QWizardPage):
    """Test mode selection page."""

    def __init__(self, parent: QWizard | None = None) -> None:
        """Initialize test mode selection page."""
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self.setTitle("Select Test Mode")
        self.setSubTitle("Specify the type of rheological test")

        layout = QVBoxLayout()

        # Auto-detect checkbox
        self.auto_detect_check = QCheckBox("Auto-detect test mode from data")
        self.auto_detect_check.setChecked(True)
        self.auto_detect_check.stateChanged.connect(self._on_auto_detect_changed)
        layout.addWidget(self.auto_detect_check)

        # Test mode selection
        mode_group = QGroupBox("Test Mode")
        mode_layout = QVBoxLayout()

        self.test_mode_combo = QComboBox()
        self.test_mode_combo.addItems(
            [
                "oscillation",
                "relaxation",
                "creep",
                "flow",
            ]
        )
        self.test_mode_combo.setEnabled(False)
        self.test_mode_combo.currentTextChanged.connect(self._on_test_mode_changed)

        mode_layout.addWidget(self.test_mode_combo)

        # Add descriptions
        description_text = """
<b>Test Modes:</b>
<ul>
<li><b>Oscillation:</b> Frequency sweep (G', G'', ω)</li>
<li><b>Relaxation:</b> Stress relaxation (G(t), time)</li>
<li><b>Creep:</b> Creep compliance (J(t), time)</li>
<li><b>Flow:</b> Steady shear (η, shear rate)</li>
</ul>
        """
        description_label = QLabel(description_text)
        description_label.setWordWrap(True)
        mode_layout.addWidget(description_label)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        layout.addStretch()
        self.setLayout(layout)

        # Register field
        self.registerField("test_mode", self.test_mode_combo, "currentText")
        self.registerField("auto_detect_mode", self.auto_detect_check)

    def _on_auto_detect_changed(self, state: int) -> None:
        """Handle auto-detect checkbox change."""
        enabled = state != Qt.CheckState.Checked.value
        self.test_mode_combo.setEnabled(enabled)
        logger.debug(
            "Option changed",
            dialog="ImportWizard",
            option="auto_detect_mode",
            value=not enabled,
        )

    def _on_test_mode_changed(self, text: str) -> None:
        """Handle test mode change."""
        if text and self.test_mode_combo.isEnabled():
            logger.debug(
                "Option changed",
                dialog="ImportWizard",
                option="test_mode",
                value=text,
            )


class PreviewConfirmPage(QWizardPage):
    """Preview and confirm page."""

    def __init__(self, parent: QWizard | None = None) -> None:
        """Initialize preview page."""
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self.setTitle("Preview and Confirm")
        self.setSubTitle("Review your import settings")

        layout = QVBoxLayout()

        # Summary
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        # Preview table
        preview_label = QLabel("<b>Data Preview:</b>")
        layout.addWidget(preview_label)

        self.preview_table = QTableWidget()
        layout.addWidget(self.preview_table)

        self.setLayout(layout)

    def initializePage(self) -> None:
        """Initialize page when shown."""
        file_path = self.field("file_path")
        x_col = self.field("x_column")
        y_col = self.field("y_column")
        y2_col = self.field("y2_column")
        temp_col = self.field("temp_column")
        test_mode = self.field("test_mode")
        auto_detect = self.field("auto_detect_mode")

        logger.debug(
            "Preparing import preview",
            page=self.__class__.__name__,
            file_path=file_path,
            x_col=x_col,
            y_col=y_col,
        )

        # Build summary
        summary = f"""
<b>Import Configuration:</b><br>
<b>File:</b> {Path(file_path).name}<br>
<b>X Column:</b> {x_col}<br>
<b>Y Column:</b> {y_col}<br>
        """

        if y2_col and y2_col != "(None)":
            summary += f"<b>Y2 Column:</b> {y2_col}<br>"

        if temp_col and temp_col != "(None)":
            summary += f"<b>Temperature Column:</b> {temp_col}<br>"

        if auto_detect:
            summary += "<b>Test Mode:</b> Auto-detect<br>"
        else:
            summary += f"<b>Test Mode:</b> {test_mode}<br>"

        self.summary_label.setText(summary)

        # Load preview data
        self._load_preview(file_path, x_col, y_col, y2_col)

    def _load_preview(
        self, file_path: str, x_col: str, y_col: str, y2_col: str | None = None
    ) -> None:
        """Load preview data."""
        try:
            path = Path(file_path)
            if path.suffix.lower() in [".csv", ".txt"]:
                from rheojax.io.readers.csv_reader import detect_csv_delimiter

                delimiter = detect_csv_delimiter(file_path)
                df = pd.read_csv(file_path, sep=delimiter, nrows=10)
            elif path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, nrows=10)
            else:
                from rheojax.io.readers.csv_reader import detect_csv_delimiter

                delimiter = detect_csv_delimiter(file_path)
                df = pd.read_csv(file_path, sep=delimiter, nrows=10)

            # Select relevant columns
            cols = [x_col, y_col]
            if y2_col and y2_col != "(None)":
                cols.append(y2_col)

            df_preview = df[cols]

            # Update table
            self.preview_table.setRowCount(len(df_preview))
            self.preview_table.setColumnCount(len(df_preview.columns))
            self.preview_table.setHorizontalHeaderLabels(
                [str(col) for col in df_preview.columns]
            )

            for i in range(len(df_preview)):
                for j in range(len(df_preview.columns)):
                    value = str(df_preview.iloc[i, j])
                    self.preview_table.setItem(i, j, QTableWidgetItem(value))

            self.preview_table.resizeColumnsToContents()
            logger.debug(
                "Preview loaded",
                page=self.__class__.__name__,
                num_rows=len(df_preview),
            )

        except Exception as e:
            logger.error(
                "Failed to load preview",
                page=self.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            QMessageBox.warning(self, "Error", f"Failed to load preview: {e}")


class ImportWizard(QWizard):
    """Multi-step data import wizard.

    Steps:
        1. File selection
        2. Column mapping
        3. Test mode selection
        4. Preview and confirm

    Example
    -------
    >>> wizard = ImportWizard()  # doctest: +SKIP
    >>> result = wizard.exec()  # doctest: +SKIP
    >>> if result == QWizard.DialogCode.Accepted:  # doctest: +SKIP
    ...     config = wizard.get_result()  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize import wizard."""
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        self.setWindowTitle("Data Import Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)

        # Add pages
        self.file_page = FileSelectionPage(self)
        self.column_page = ColumnMappingPage(self)
        self.mode_page = TestModeSelectionPage(self)
        self.preview_page = PreviewConfirmPage(self)

        self.addPage(self.file_page)
        self.addPage(self.column_page)
        self.addPage(self.mode_page)
        self.addPage(self.preview_page)

        # Set minimum size
        self.setMinimumSize(700, 500)

        # Connect finish/cancel signals
        self.finished.connect(self._on_finished)

    def _on_finished(self, result: int) -> None:
        """Handle wizard finished."""
        if result == QWizard.DialogCode.Accepted.value:
            logger.debug("Options applied", dialog=self.__class__.__name__)
        else:
            logger.debug("Dialog closed", dialog=self.__class__.__name__)

    def showEvent(self, event) -> None:
        """Handle show event."""
        super().showEvent(event)
        logger.debug("Dialog opened", dialog=self.__class__.__name__)

    def get_result(self) -> dict[str, Any]:
        """Get import configuration.

        Returns
        -------
        dict[str, Any]
            Import configuration with keys:
            - file_path: Path to data file
            - x_column: X-axis column name
            - y_column: Y-axis column name
            - y2_column: Optional Y2-axis column name
            - temp_column: Optional temperature column name
            - test_mode: Test mode or None if auto-detect
            - auto_detect_mode: Whether to auto-detect test mode
        """
        y2_col = self.field("y2_column")
        temp_col = self.field("temp_column")

        config = {
            "file_path": self.field("file_path"),
            "x_column": self.field("x_column"),
            "y_column": self.field("y_column"),
            "y2_column": y2_col if y2_col != "(None)" else None,
            "temp_column": temp_col if temp_col != "(None)" else None,
            "auto_detect_mode": self.field("auto_detect_mode"),
        }

        # Only include test_mode if not auto-detecting
        if not config["auto_detect_mode"]:
            config["test_mode"] = self.field("test_mode")
        else:
            config["test_mode"] = None

        return config
