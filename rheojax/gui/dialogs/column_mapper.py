"""
Column Mapper Dialog
===================

Simple dialog for column reassignment.
"""

from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ColumnMapperDialog(QDialog):
    """Simple dialog for column reassignment.

    Features:
        - List of detected columns from file
        - Dropdowns to assign columns
        - Auto-detect button
        - Preview of first few rows

    Example
    -------
    >>> dialog = ColumnMapperDialog('data.csv')  # doctest: +SKIP
    >>> if dialog.exec() == QDialog.DialogCode.Accepted:  # doctest: +SKIP
    ...     mapping = dialog.get_mapping()  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str | None = None,
        current_mapping: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize column mapper dialog.

        Parameters
        ----------
        file_path : str, optional
            Path to data file
        current_mapping : dict[str, str], optional
            Current column mapping
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self.file_path = file_path
        self.current_mapping = current_mapping or {}
        self.columns: list[str] = []
        self.df_preview: pd.DataFrame | None = None

        self.setWindowTitle("Column Mapper")
        self.setMinimumSize(600, 500)

        self._setup_ui()
        self._load_data()

    def _setup_ui(self) -> None:
        """Set up user interface."""
        layout = QVBoxLayout()

        # File info
        if self.file_path:
            file_label = QLabel(f"<b>File:</b> {Path(self.file_path).name}")
            layout.addWidget(file_label)

        # Auto-detect button
        auto_detect_button = QPushButton("Auto-Detect Columns")
        auto_detect_button.clicked.connect(self._auto_detect)
        layout.addWidget(auto_detect_button)

        # Column mapping group
        mapping_group = QGroupBox("Column Assignment")
        mapping_layout = QVBoxLayout()

        # X axis (frequency/time)
        x_layout = QHBoxLayout()
        x_label = QLabel("X (Frequency/Time):")
        x_label.setMinimumWidth(150)
        x_layout.addWidget(x_label)
        self.x_combo = QComboBox()
        x_layout.addWidget(self.x_combo, 1)
        mapping_layout.addLayout(x_layout)

        # Y axis (modulus/compliance/viscosity)
        y_layout = QHBoxLayout()
        y_label = QLabel("Y (Modulus/Viscosity):")
        y_label.setMinimumWidth(150)
        y_layout.addWidget(y_label)
        self.y_combo = QComboBox()
        y_layout.addWidget(self.y_combo, 1)
        mapping_layout.addLayout(y_layout)

        # Y2 axis (optional, e.g., G'')
        y2_layout = QHBoxLayout()
        y2_label = QLabel("Y2 (Optional, e.g., G''):")
        y2_label.setMinimumWidth(150)
        y2_layout.addWidget(y2_label)
        self.y2_combo = QComboBox()
        y2_layout.addWidget(self.y2_combo, 1)
        mapping_layout.addLayout(y2_layout)

        # Temperature (optional)
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature (Optional):")
        temp_label.setMinimumWidth(150)
        temp_layout.addWidget(temp_label)
        self.temp_combo = QComboBox()
        temp_layout.addWidget(self.temp_combo, 1)
        mapping_layout.addLayout(temp_layout)

        mapping_group.setLayout(mapping_layout)
        layout.addWidget(mapping_group)

        # Preview section
        preview_label = QLabel("<b>Preview (First 5 Rows):</b>")
        layout.addWidget(preview_label)

        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(150)
        layout.addWidget(self.preview_table)

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _load_data(self) -> None:
        """Load data from file."""
        if not self.file_path:
            return

        try:
            path = Path(self.file_path)

            # Read file to get columns
            if path.suffix.lower() in [".csv", ".txt"]:
                self.df_preview = pd.read_csv(self.file_path, nrows=5)
            elif path.suffix.lower() in [".xlsx", ".xls"]:
                self.df_preview = pd.read_excel(self.file_path, nrows=5)
            else:
                # Try CSV as fallback
                self.df_preview = pd.read_csv(self.file_path, nrows=5)

            self.columns = list(self.df_preview.columns)

            # Populate combo boxes
            self._populate_combos()

            # Apply current mapping if provided
            self._apply_current_mapping()

            # Update preview
            self._update_preview()

            # Auto-detect if no current mapping
            if not self.current_mapping:
                self._auto_detect()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file: {e}")

    def _populate_combos(self) -> None:
        """Populate combo boxes with column names."""
        # X and Y are required
        self.x_combo.clear()
        self.y_combo.clear()

        self.x_combo.addItems(self.columns)
        self.y_combo.addItems(self.columns)

        # Y2 and Temperature are optional
        self.y2_combo.clear()
        self.temp_combo.clear()

        self.y2_combo.addItem("(None)")
        self.y2_combo.addItems(self.columns)

        self.temp_combo.addItem("(None)")
        self.temp_combo.addItems(self.columns)

    def _apply_current_mapping(self) -> None:
        """Apply current mapping to combo boxes."""
        if not self.current_mapping:
            return

        # Set X column
        if "x" in self.current_mapping:
            idx = self.x_combo.findText(self.current_mapping["x"])
            if idx >= 0:
                self.x_combo.setCurrentIndex(idx)

        # Set Y column
        if "y" in self.current_mapping:
            idx = self.y_combo.findText(self.current_mapping["y"])
            if idx >= 0:
                self.y_combo.setCurrentIndex(idx)

        # Set Y2 column
        if "y2" in self.current_mapping:
            idx = self.y2_combo.findText(self.current_mapping["y2"])
            if idx >= 0:
                self.y2_combo.setCurrentIndex(idx)

        # Set Temperature column
        if "temperature" in self.current_mapping:
            idx = self.temp_combo.findText(self.current_mapping["temperature"])
            if idx >= 0:
                self.temp_combo.setCurrentIndex(idx)

    def _auto_detect(self) -> None:
        """Auto-detect column mapping based on common patterns."""
        # Common column name patterns
        x_patterns = ["freq", "frequency", "omega", "time", "t", "w", "rate", "shear"]
        y_patterns = ["g'", "gp", "storage", "modulus", "eta", "viscosity", "stress", "g*"]
        y2_patterns = ["g''", "gpp", "loss"]
        temp_patterns = ["temp", "temperature"]

        def find_match(combo: QComboBox, patterns: list[str]) -> bool:
            """Find best match for column."""
            for i in range(combo.count()):
                text = combo.itemText(i).lower()
                if text == "(none)":
                    continue
                for pattern in patterns:
                    if pattern in text:
                        combo.setCurrentIndex(i)
                        return True
            return False

        find_match(self.x_combo, x_patterns)
        find_match(self.y_combo, y_patterns)
        find_match(self.y2_combo, y2_patterns)
        find_match(self.temp_combo, temp_patterns)

    def _update_preview(self) -> None:
        """Update preview table."""
        if self.df_preview is None:
            return

        self.preview_table.setRowCount(len(self.df_preview))
        self.preview_table.setColumnCount(len(self.df_preview.columns))
        self.preview_table.setHorizontalHeaderLabels(
            [str(col) for col in self.df_preview.columns]
        )

        for i in range(len(self.df_preview)):
            for j in range(len(self.df_preview.columns)):
                value = str(self.df_preview.iloc[i, j])
                self.preview_table.setItem(i, j, QTableWidgetItem(value))

        self.preview_table.resizeColumnsToContents()

    def get_mapping(self) -> dict[str, str]:
        """Get column mapping.

        Returns
        -------
        dict[str, str]
            Column mapping with keys:
            - x: X-axis column name
            - y: Y-axis column name
            - y2: Y2-axis column name (optional)
            - temperature: Temperature column name (optional)
        """
        mapping = {
            "x": self.x_combo.currentText(),
            "y": self.y_combo.currentText(),
        }

        # Add optional columns if selected
        y2_col = self.y2_combo.currentText()
        if y2_col and y2_col != "(None)":
            mapping["y2"] = y2_col

        temp_col = self.temp_combo.currentText()
        if temp_col and temp_col != "(None)":
            mapping["temperature"] = temp_col

        return mapping


# Alias for backward compatibility
ColumnMapper = ColumnMapperDialog
