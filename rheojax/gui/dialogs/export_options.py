"""
Export Options Dialog
====================

Configure export format and options.
"""

from typing import Any

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheojax.logging import get_logger

logger = get_logger(__name__)


class ExportOptionsDialog(QDialog):
    """Export configuration dialog.

    Options:
        - Data format (CSV, JSON, Excel, HDF5)
        - Figure format (PNG, SVG, PDF)
        - Figure DPI
        - Style preset
        - Metadata inclusion

    Example
    -------
    >>> dialog = ExportOptionsDialog()  # doctest: +SKIP
    >>> if dialog.exec() == QDialog.DialogCode.Accepted:  # doctest: +SKIP
    ...     options = dialog.get_options()  # doctest: +SKIP
    """

    def __init__(
        self,
        current_options: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize export options dialog.

        Parameters
        ----------
        current_options : dict[str, Any], optional
            Current export options
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        self.current_options = current_options or {}

        self.setWindowTitle("Export Options")
        self.setMinimumSize(500, 500)

        self._setup_ui()
        self._load_current_options()

    def _setup_ui(self) -> None:
        """Set up user interface."""
        layout = QVBoxLayout()

        # Data format section
        data_group = QGroupBox("Data Format")
        data_layout = QVBoxLayout()

        self.data_button_group = QButtonGroup()
        self.data_button_group.idClicked.connect(self._on_data_format_changed)

        self.csv_radio = QRadioButton("CSV (Comma-Separated Values)")
        self.csv_radio.setChecked(True)
        self.data_button_group.addButton(self.csv_radio, 0)
        data_layout.addWidget(self.csv_radio)

        self.json_radio = QRadioButton("JSON (JavaScript Object Notation)")
        self.data_button_group.addButton(self.json_radio, 1)
        data_layout.addWidget(self.json_radio)

        self.excel_radio = QRadioButton("Excel (.xlsx)")
        self.data_button_group.addButton(self.excel_radio, 2)
        data_layout.addWidget(self.excel_radio)

        self.hdf5_radio = QRadioButton("HDF5 (Hierarchical Data Format)")
        self.data_button_group.addButton(self.hdf5_radio, 3)
        data_layout.addWidget(self.hdf5_radio)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Figure format section
        figure_group = QGroupBox("Figure Format")
        figure_layout = QVBoxLayout()

        self.figure_button_group = QButtonGroup()
        self.figure_button_group.idClicked.connect(self._on_figure_format_changed)

        self.png_radio = QRadioButton("PNG (Portable Network Graphics)")
        self.png_radio.setChecked(True)
        self.figure_button_group.addButton(self.png_radio, 0)
        figure_layout.addWidget(self.png_radio)

        self.svg_radio = QRadioButton("SVG (Scalable Vector Graphics)")
        self.figure_button_group.addButton(self.svg_radio, 1)
        figure_layout.addWidget(self.svg_radio)

        self.pdf_radio = QRadioButton("PDF (Portable Document Format)")
        self.figure_button_group.addButton(self.pdf_radio, 2)
        figure_layout.addWidget(self.pdf_radio)

        figure_group.setLayout(figure_layout)
        layout.addWidget(figure_group)

        # Figure settings
        settings_group = QGroupBox("Figure Settings")
        settings_layout = QFormLayout()

        # DPI setting
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setSuffix(" dpi")
        self.dpi_spin.valueChanged.connect(
            lambda v: self._on_option_changed("dpi", v)
        )
        settings_layout.addRow("Resolution (DPI):", self.dpi_spin)

        # Style preset
        self.style_combo = QComboBox()
        self.style_combo.addItems(
            [
                "default",
                "publication",
                "presentation",
                "poster",
                "seaborn",
                "ggplot",
            ]
        )
        self.style_combo.currentTextChanged.connect(
            lambda t: self._on_option_changed("style", t)
        )
        settings_layout.addRow("Style Preset:", self.style_combo)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Additional options
        options_group = QGroupBox("Additional Options")
        options_layout = QVBoxLayout()

        # Include metadata
        self.metadata_check = QCheckBox(
            "Include metadata (model parameters, timestamps)"
        )
        self.metadata_check.setChecked(True)
        self.metadata_check.stateChanged.connect(
            lambda s: self._on_option_changed("include_metadata", s != 0)
        )
        options_layout.addWidget(self.metadata_check)

        # Include provenance
        self.provenance_check = QCheckBox(
            "Include provenance information (processing history)"
        )
        self.provenance_check.setChecked(False)
        self.provenance_check.stateChanged.connect(
            lambda s: self._on_option_changed("include_provenance", s != 0)
        )
        options_layout.addWidget(self.provenance_check)

        # Compress data
        self.compress_check = QCheckBox("Compress data (for HDF5 and Excel)")
        self.compress_check.setChecked(True)
        self.compress_check.stateChanged.connect(
            lambda s: self._on_option_changed("compress", s != 0)
        )
        options_layout.addWidget(self.compress_check)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        layout.addStretch()

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self._on_rejected)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _load_current_options(self) -> None:
        """Load current options into UI."""
        if not self.current_options:
            return

        # Data format
        if "data_format" in self.current_options:
            data_format = self.current_options["data_format"].lower()
            if data_format == "csv":
                self.csv_radio.setChecked(True)
            elif data_format == "json":
                self.json_radio.setChecked(True)
            elif data_format == "excel" or data_format == "xlsx":
                self.excel_radio.setChecked(True)
            elif data_format == "hdf5" or data_format == "h5":
                self.hdf5_radio.setChecked(True)

        # Figure format
        if "figure_format" in self.current_options:
            fig_format = self.current_options["figure_format"].lower()
            if fig_format == "png":
                self.png_radio.setChecked(True)
            elif fig_format == "svg":
                self.svg_radio.setChecked(True)
            elif fig_format == "pdf":
                self.pdf_radio.setChecked(True)

        # DPI
        if "dpi" in self.current_options:
            self.dpi_spin.setValue(self.current_options["dpi"])

        # Style
        if "style" in self.current_options:
            style = self.current_options["style"]
            idx = self.style_combo.findText(style)
            if idx >= 0:
                self.style_combo.setCurrentIndex(idx)

        # Metadata
        if "include_metadata" in self.current_options:
            self.metadata_check.setChecked(self.current_options["include_metadata"])

        # Provenance
        if "include_provenance" in self.current_options:
            self.provenance_check.setChecked(self.current_options["include_provenance"])

        # Compression
        if "compress" in self.current_options:
            self.compress_check.setChecked(self.current_options["compress"])

    def _on_data_format_changed(self, button_id: int) -> None:
        """Handle data format radio button change."""
        data_format_map = {0: "csv", 1: "json", 2: "excel", 3: "hdf5"}
        data_format = data_format_map.get(button_id, "csv")
        logger.debug(
            "Option changed",
            dialog=self.__class__.__name__,
            option="data_format",
            value=data_format,
        )

    def _on_figure_format_changed(self, button_id: int) -> None:
        """Handle figure format radio button change."""
        figure_format_map = {0: "png", 1: "svg", 2: "pdf"}
        figure_format = figure_format_map.get(button_id, "png")
        logger.debug(
            "Option changed",
            dialog=self.__class__.__name__,
            option="figure_format",
            value=figure_format,
        )

    def _on_option_changed(self, option: str, value: Any) -> None:
        """Handle option change."""
        logger.debug(
            "Option changed",
            dialog=self.__class__.__name__,
            option=option,
            value=value,
        )

    def _on_accepted(self) -> None:
        """Handle dialog accepted."""
        logger.debug("Options applied", dialog=self.__class__.__name__)
        self.accept()

    def _on_rejected(self) -> None:
        """Handle dialog rejected."""
        logger.debug("Dialog closed", dialog=self.__class__.__name__)
        self.reject()

    def showEvent(self, event) -> None:
        """Handle show event."""
        super().showEvent(event)
        logger.debug("Dialog opened", dialog=self.__class__.__name__)

    def get_options(self) -> dict[str, Any]:
        """Get export options.

        Returns
        -------
        dict[str, Any]
            Export options with keys:
            - data_format: Data format (csv, json, excel, hdf5)
            - figure_format: Figure format (png, svg, pdf)
            - dpi: Figure resolution in DPI
            - style: Style preset name
            - include_metadata: Whether to include metadata
            - include_provenance: Whether to include provenance
            - compress: Whether to compress data
        """
        # Get data format
        data_format_map = {0: "csv", 1: "json", 2: "excel", 3: "hdf5"}
        data_format_id = self.data_button_group.checkedId()
        data_format = data_format_map.get(data_format_id, "csv")

        # Get figure format
        figure_format_map = {0: "png", 1: "svg", 2: "pdf"}
        figure_format_id = self.figure_button_group.checkedId()
        figure_format = figure_format_map.get(figure_format_id, "png")

        options = {
            "data_format": data_format,
            "figure_format": figure_format,
            "dpi": self.dpi_spin.value(),
            "style": self.style_combo.currentText(),
            "include_metadata": self.metadata_check.isChecked(),
            "include_provenance": self.provenance_check.isChecked(),
            "compress": self.compress_check.isChecked(),
        }

        return options


# Alias for backward compatibility
ExportOptions = ExportOptionsDialog
