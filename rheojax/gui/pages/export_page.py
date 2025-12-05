"""
Export Page
==========

Result export interface with format selection and batch operations.
"""

from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.state.store import StateStore


class ExportPage(QWidget):
    """Export and save results page."""

    export_requested = Signal(dict)  # export_config

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = StateStore()
        self.setup_ui()

    def setup_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        # Left: What to export
        left_panel = self._create_content_panel()
        main_layout.addWidget(left_panel, 1)

        # Center: Format and options
        center_panel = self._create_format_panel()
        main_layout.addWidget(center_panel, 1)

        # Right: Preview and export
        right_panel = self._create_export_panel()
        main_layout.addWidget(right_panel, 1)

    def _create_content_panel(self) -> QWidget:
        panel = QGroupBox("Content to Export")
        layout = QVBoxLayout(panel)

        # Checkboxes for content types
        self._check_parameters = QCheckBox("Fit Parameters")
        self._check_parameters.setChecked(True)
        layout.addWidget(self._check_parameters)

        self._check_intervals = QCheckBox("Credible Intervals (95%)")
        self._check_intervals.setChecked(True)
        layout.addWidget(self._check_intervals)

        self._check_posteriors = QCheckBox("Posterior Samples")
        layout.addWidget(self._check_posteriors)

        self._check_figures = QCheckBox("Figures and Plots")
        self._check_figures.setChecked(True)
        layout.addWidget(self._check_figures)

        self._check_diagnostics = QCheckBox("MCMC Diagnostics")
        layout.addWidget(self._check_diagnostics)

        self._check_raw_data = QCheckBox("Raw Data")
        layout.addWidget(self._check_raw_data)

        self._check_metadata = QCheckBox("Metadata and Provenance")
        self._check_metadata.setChecked(True)
        layout.addWidget(self._check_metadata)

        layout.addStretch()

        # Select all/none buttons
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        btn_layout.addWidget(btn_all)

        btn_none = QPushButton("Select None")
        btn_none.clicked.connect(self._select_none)
        btn_layout.addWidget(btn_none)

        layout.addLayout(btn_layout)

        return panel

    def _create_format_panel(self) -> QWidget:
        panel = QGroupBox("Format Options")
        layout = QVBoxLayout(panel)

        # Data format
        layout.addWidget(QLabel("Data Format:", styleSheet="font-weight: bold;"))
        self._data_format_combo = QComboBox()
        self._data_format_combo.addItems(["CSV", "Excel (XLSX)", "HDF5", "JSON"])
        layout.addWidget(self._data_format_combo)

        # Figure format
        layout.addWidget(QLabel("Figure Format:", styleSheet="font-weight: bold; margin-top: 15px;"))
        self._figure_format_combo = QComboBox()
        self._figure_format_combo.addItems(["PNG", "SVG", "PDF", "EPS"])
        layout.addWidget(self._figure_format_combo)

        # Figure DPI
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI:"))
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(72, 600)
        self._dpi_spin.setValue(300)
        self._dpi_spin.setSingleStep(50)
        dpi_layout.addWidget(self._dpi_spin)
        layout.addLayout(dpi_layout)

        # Style preset
        layout.addWidget(QLabel("Plot Style:", styleSheet="font-weight: bold; margin-top: 15px;"))
        self._style_combo = QComboBox()
        self._style_combo.addItems(["Publication", "Presentation", "Default"])
        layout.addWidget(self._style_combo)

        # Figure size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(4, 20)
        self._width_spin.setValue(8)
        size_layout.addWidget(self._width_spin)
        size_layout.addWidget(QLabel("x"))
        self._height_spin = QSpinBox()
        self._height_spin.setRange(3, 16)
        self._height_spin.setValue(6)
        size_layout.addWidget(self._height_spin)
        size_layout.addWidget(QLabel("inches"))
        layout.addLayout(size_layout)

        # Report template
        layout.addWidget(QLabel("Report Template:", styleSheet="font-weight: bold; margin-top: 15px;"))
        self._template_combo = QComboBox()
        self._template_combo.addItems(["None", "Markdown Report", "PDF Report"])
        layout.addWidget(self._template_combo)

        layout.addStretch()

        return panel

    def _create_export_panel(self) -> QWidget:
        panel = QGroupBox("Export")
        layout = QVBoxLayout(panel)

        # Output directory
        layout.addWidget(QLabel("Output Directory:", styleSheet="font-weight: bold;"))

        dir_layout = QHBoxLayout()
        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setPlaceholderText("Select output directory...")
        dir_layout.addWidget(self._output_dir_edit, 1)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(btn_browse)

        layout.addLayout(dir_layout)

        # Preview
        layout.addWidget(QLabel("Export Preview:", styleSheet="font-weight: bold; margin-top: 20px;"))

        self._preview_list = QListWidget()
        self._preview_list.setAlternatingRowColors(True)
        layout.addWidget(self._preview_list)

        btn_preview = QPushButton("Preview Export")
        btn_preview.clicked.connect(self._update_preview)
        layout.addWidget(btn_preview)

        layout.addStretch()

        # Export button
        self._btn_export = QPushButton("Export")
        self._btn_export.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self._btn_export.clicked.connect(self._on_export_clicked)
        layout.addWidget(self._btn_export)

        return panel

    def _select_all(self) -> None:
        for checkbox in [
            self._check_parameters, self._check_intervals, self._check_posteriors,
            self._check_figures, self._check_diagnostics, self._check_raw_data,
            self._check_metadata
        ]:
            checkbox.setChecked(True)

    def _select_none(self) -> None:
        for checkbox in [
            self._check_parameters, self._check_intervals, self._check_posteriors,
            self._check_figures, self._check_diagnostics, self._check_raw_data,
            self._check_metadata
        ]:
            checkbox.setChecked(False)

    def _browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(self._store.get_state().last_export_dir or Path.home())
        )
        if directory:
            self._output_dir_edit.setText(directory)

    def _update_preview(self) -> None:
        self._preview_list.clear()

        output_dir = Path(self._output_dir_edit.text()) if self._output_dir_edit.text() else Path("output")

        # Add preview items based on selections
        if self._check_parameters.isChecked():
            self._preview_list.addItem(f"{output_dir}/parameters.{self._data_format_combo.currentText().lower()}")

        if self._check_intervals.isChecked():
            self._preview_list.addItem(f"{output_dir}/credible_intervals.{self._data_format_combo.currentText().lower()}")

        if self._check_posteriors.isChecked():
            self._preview_list.addItem(f"{output_dir}/posterior_samples.{self._data_format_combo.currentText().lower()}")

        if self._check_figures.isChecked():
            fig_ext = self._figure_format_combo.currentText().lower()
            self._preview_list.addItem(f"{output_dir}/figures/fit_plot.{fig_ext}")
            self._preview_list.addItem(f"{output_dir}/figures/residuals.{fig_ext}")

        if self._check_diagnostics.isChecked():
            self._preview_list.addItem(f"{output_dir}/diagnostics_summary.txt")

        if self._check_raw_data.isChecked():
            self._preview_list.addItem(f"{output_dir}/raw_data.{self._data_format_combo.currentText().lower()}")

        if self._check_metadata.isChecked():
            self._preview_list.addItem(f"{output_dir}/metadata.json")

        if self._template_combo.currentText() == "Markdown Report":
            self._preview_list.addItem(f"{output_dir}/report.md")
        elif self._template_combo.currentText() == "PDF Report":
            self._preview_list.addItem(f"{output_dir}/report.pdf")

    def _on_export_clicked(self) -> None:
        config = {
            "include_parameters": self._check_parameters.isChecked(),
            "include_intervals": self._check_intervals.isChecked(),
            "include_posteriors": self._check_posteriors.isChecked(),
            "include_figures": self._check_figures.isChecked(),
            "include_diagnostics": self._check_diagnostics.isChecked(),
            "include_raw_data": self._check_raw_data.isChecked(),
            "include_metadata": self._check_metadata.isChecked(),
            "data_format": self._data_format_combo.currentText().lower(),
            "figure_format": self._figure_format_combo.currentText().lower(),
            "dpi": self._dpi_spin.value(),
            "style": self._style_combo.currentText().lower(),
            "figure_size": (self._width_spin.value(), self._height_spin.value()),
            "template": self._template_combo.currentText(),
            "output_dir": Path(self._output_dir_edit.text()) if self._output_dir_edit.text() else None,
        }

        self.export_requested.emit(config)

    def export_results(self, item_ids: list[str], file_path: str, format: str | None = None, **kwargs: Any) -> None:
        pass

    def export_plot(self, plot_id: str, file_path: str, dpi: int = 300, **kwargs: Any) -> None:
        pass

    def preview_export(self, item_ids: list[str], format: str) -> dict[str, Any]:
        return {}

    def batch_export(self, exports: list[dict[str, Any]]) -> list[str]:
        return []
