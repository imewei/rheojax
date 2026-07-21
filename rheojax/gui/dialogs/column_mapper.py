"""
Column Mapper Dialog
===================

Simple dialog for column reassignment.
"""

import re
from pathlib import Path

import pandas as pd

from rheojax.gui.compat import (
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
from rheojax.gui.widgets import RheoComboBox
from rheojax.logging import get_logger

logger = get_logger(__name__)


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
        logger.debug("Initializing", class_name=self.__class__.__name__)

        self.file_path = file_path
        self.current_mapping = current_mapping or {}
        self.columns: list[str] = []
        self.df_preview: pd.DataFrame | None = None

        self.setWindowTitle("Column Mapper")
        self.setMinimumSize(600, 500)

        self._setup_ui()
        # Intentional: load data during __init__ before exec_() so the modal
        # dialog is fully populated when it first becomes visible to the user.
        self._load_data()

        logger.debug(
            "Dialog initialized",
            dialog=self.__class__.__name__,
            file_path=file_path,
            has_current_mapping=bool(current_mapping),
        )

    def _setup_ui(self) -> None:
        """Set up user interface."""
        layout = QVBoxLayout()

        # File info
        if self.file_path:
            file_label = QLabel(f"<b>File:</b> {Path(self.file_path).name}")
            layout.addWidget(file_label)

        # Auto-detect button
        auto_detect_button = QPushButton("Auto-Detect Columns")
        auto_detect_button.setAccessibleDescription(
            "Attempts to automatically match file columns to X, Y, Y2, and "
            "Temperature fields based on common naming patterns."
        )
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
        self.x_combo = RheoComboBox(placeholder="Select a column...")
        self.x_combo.setAccessibleName("X axis column")
        self.x_combo.setAccessibleDescription(
            "Required. Select the column containing the independent "
            "variable (frequency or time)."
        )
        self.x_combo.currentTextChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="x_column",
                value=value,
            )
        )
        x_layout.addWidget(self.x_combo, 1)
        mapping_layout.addLayout(x_layout)

        # Y axis (modulus/compliance/viscosity)
        y_layout = QHBoxLayout()
        y_label = QLabel("Y (Modulus/Viscosity):")
        y_label.setMinimumWidth(150)
        y_layout.addWidget(y_label)
        self.y_combo = RheoComboBox(placeholder="Select a column...")
        self.y_combo.setAccessibleName("Y axis column")
        self.y_combo.setAccessibleDescription(
            "Required. Select the column containing the primary response "
            "(modulus, viscosity, or stress)."
        )
        self.y_combo.currentTextChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="y_column",
                value=value,
            )
        )
        y_layout.addWidget(self.y_combo, 1)
        mapping_layout.addLayout(y_layout)

        # Y2 axis (optional, e.g., G'')
        y2_layout = QHBoxLayout()
        y2_label = QLabel("Y2 (Optional, e.g., G''):")
        y2_label.setMinimumWidth(150)
        y2_layout.addWidget(y2_label)
        self.y2_combo = RheoComboBox(placeholder="None (optional)")
        self.y2_combo.setAccessibleName("Y2 axis column (optional)")
        self.y2_combo.setAccessibleDescription(
            "Optional. Select a secondary response column (e.g. loss "
            "modulus G'')."
        )
        self.y2_combo.currentTextChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="y2_column",
                value=value,
            )
        )
        y2_layout.addWidget(self.y2_combo, 1)
        mapping_layout.addLayout(y2_layout)

        # Temperature (optional)
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature (Optional):")
        temp_label.setMinimumWidth(150)
        temp_layout.addWidget(temp_label)
        self.temp_combo = RheoComboBox(placeholder="None (optional)")
        self.temp_combo.setAccessibleName("Temperature column (optional)")
        self.temp_combo.setAccessibleDescription(
            "Optional. Select the column containing sample temperature."
        )
        self.temp_combo.currentTextChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="temperature_column",
                value=value,
            )
        )
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
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self._on_reject)
        layout.addWidget(button_box)

        # Tab order follows the visual top-to-bottom layout of the form.
        self.setTabOrder(auto_detect_button, self.x_combo)
        self.setTabOrder(self.x_combo, self.y_combo)
        self.setTabOrder(self.y_combo, self.y2_combo)
        self.setTabOrder(self.y2_combo, self.temp_combo)

        self.setLayout(layout)

    def _on_accept(self) -> None:
        """Handle dialog accept."""
        if self.x_combo.currentIndex() == -1 or self.y_combo.currentIndex() == -1:
            QMessageBox.warning(
                self,
                "Missing Column Mapping",
                "Please select both an X and a Y column before continuing.",
            )
            return
        logger.debug("Dialog closed", dialog=self.__class__.__name__, result="accepted")
        self.accept()

    def _on_reject(self) -> None:
        """Handle dialog reject."""
        logger.debug("Dialog closed", dialog=self.__class__.__name__, result="rejected")
        self.reject()

    def showEvent(self, event) -> None:
        """Handle dialog show event."""
        super().showEvent(event)
        logger.debug("Dialog opened", dialog=self.__class__.__name__)

    def _load_data(self) -> None:
        """Load data from file."""
        if not self.file_path:
            return

        try:
            path = Path(self.file_path)
            logger.debug(
                "Loading data file",
                dialog=self.__class__.__name__,
                file_path=str(path),
                suffix=path.suffix,
            )

            # Read file to get columns
            if path.suffix.lower() in [".csv", ".txt"]:
                from rheojax.io.readers.csv_reader import detect_csv_delimiter

                delimiter = detect_csv_delimiter(self.file_path)
                self.df_preview = pd.read_csv(self.file_path, sep=delimiter, nrows=5)
            elif path.suffix.lower() in [".xlsx", ".xls"]:
                self.df_preview = pd.read_excel(self.file_path, nrows=5)
            else:
                # Try CSV as fallback
                from rheojax.io.readers.csv_reader import detect_csv_delimiter

                delimiter = detect_csv_delimiter(self.file_path)
                self.df_preview = pd.read_csv(self.file_path, sep=delimiter, nrows=5)

            self.columns = list(self.df_preview.columns)
            logger.debug(
                "Data loaded successfully",
                dialog=self.__class__.__name__,
                num_columns=len(self.columns),
                columns=self.columns,
            )

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
            logger.error(
                "Failed to load file",
                dialog=self.__class__.__name__,
                file_path=self.file_path,
                error=str(e),
                exc_info=True,
            )
            QMessageBox.warning(self, "Error", f"Failed to load file: {e}")

    def _populate_combos(self) -> None:
        """Populate combo boxes with column names."""
        # X and Y are required; Y2 and Temperature are optional (placeholder = unset)
        self.x_combo.set_items_safely(self.columns)
        self.y_combo.set_items_safely(self.columns)
        self.y2_combo.set_items_safely(self.columns)
        self.temp_combo.set_items_safely(self.columns)

    def _apply_current_mapping(self) -> None:
        """Apply current mapping to combo boxes."""
        if not self.current_mapping:
            return

        logger.debug(
            "Applying current mapping",
            dialog=self.__class__.__name__,
            mapping=self.current_mapping,
        )

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
        logger.debug("Auto-detecting columns", dialog=self.__class__.__name__)

        # Common column name patterns
        x_patterns = ["freq", "frequency", "omega", "time", "t", "w", "rate", "shear"]
        y_patterns = [
            "g'",
            "gp",
            "storage",
            "modulus",
            "eta",
            "viscosity",
            "stress",
            "g*",
        ]
        y2_patterns = ["g''", "gpp", "loss"]
        temp_patterns = ["temp", "temperature"]

        claimed: set[str] = set()

        def find_match(combo: QComboBox, patterns: list[str]) -> bool:
            """Find best match for column, skipping columns already claimed
            by another field (e.g. "stress" matches both x and y patterns
            for a "shear stress" column, so a field claimed by an earlier
            call must not be re-matched by a later one)."""
            for i in range(combo.count()):
                text = combo.itemText(i).lower()
                if text in claimed:
                    continue
                for pattern in patterns:
                    # Alphanumeric boundaries (not \b): \b fails after symbolic
                    # modulus patterns like g', g'', g* whose last char is non-word,
                    # while still preventing 'g' from matching inside 'log'/'gpp'.
                    if re.search(
                        r"(?<![a-z0-9])" + re.escape(pattern) + r"(?![a-z0-9])", text
                    ):
                        combo.setCurrentIndex(i)
                        claimed.add(text)
                        return True
            return False

        # y2 claims before y: "g''" is a substring pattern-space neighbor of
        # "g'", so G'' must be claimed by y2 before y is matched against it.
        x_found = find_match(self.x_combo, x_patterns)
        y2_found = find_match(self.y2_combo, y2_patterns)
        y_found = find_match(self.y_combo, y_patterns)
        temp_found = find_match(self.temp_combo, temp_patterns)

        logger.debug(
            "Auto-detect results",
            dialog=self.__class__.__name__,
            x_found=x_found,
            y_found=y_found,
            y2_found=y2_found,
            temp_found=temp_found,
        )

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

        # Add optional columns if selected (empty text = placeholder/unset)
        y2_col = self.y2_combo.currentText()
        if y2_col:
            mapping["y2"] = y2_col

        temp_col = self.temp_combo.currentText()
        if temp_col:
            mapping["temperature"] = temp_col

        logger.debug(
            "Getting mapping",
            dialog=self.__class__.__name__,
            mapping=mapping,
        )

        return mapping


# Alias for backward compatibility
ColumnMapper = ColumnMapperDialog
