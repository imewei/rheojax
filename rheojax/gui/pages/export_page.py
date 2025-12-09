"""
Export Page
==========

Result export interface with format selection and batch operations.
"""

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.services.export_service import ExportService
from rheojax.gui.state.store import StateStore

logger = logging.getLogger(__name__)


class ExportPage(QWidget):
    """Export and save results page."""

    export_requested = Signal(dict)  # export_config
    export_completed = Signal(str)  # output_path
    export_failed = Signal(str)  # error_message

    # Format extension mapping
    FORMAT_EXTENSIONS = {
        "csv": "csv",
        "excel (xlsx)": "xlsx",
        "hdf5": "hdf5",
        "json": "json",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = StateStore()
        self._export_service = ExportService()
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
        self._data_format_combo.currentTextChanged.connect(self._update_preview)
        layout.addWidget(self._data_format_combo)

        # Figure format
        layout.addWidget(QLabel("Figure Format:", styleSheet="font-weight: bold; margin-top: 15px;"))
        self._figure_format_combo = QComboBox()
        self._figure_format_combo.addItems(["PNG", "SVG", "PDF", "EPS"])
        self._figure_format_combo.currentTextChanged.connect(self._update_preview)
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
        self._template_combo.currentTextChanged.connect(self._update_preview)
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
        self._output_dir_edit.textChanged.connect(self._update_preview)
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

        btn_preview = QPushButton("Refresh Preview")
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
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
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
        self._update_preview()

    def _select_none(self) -> None:
        for checkbox in [
            self._check_parameters, self._check_intervals, self._check_posteriors,
            self._check_figures, self._check_diagnostics, self._check_raw_data,
            self._check_metadata
        ]:
            checkbox.setChecked(False)
        self._update_preview()

    def _browse_output_dir(self) -> None:
        state = self._store.get_state()
        initial_dir = str(state.last_export_dir) if state.last_export_dir else str(Path.home())

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            initial_dir
        )
        if directory:
            self._output_dir_edit.setText(directory)

    def _get_data_extension(self) -> str:
        """Get file extension for current data format selection."""
        format_text = self._data_format_combo.currentText().lower()
        return self.FORMAT_EXTENSIONS.get(format_text, "csv")

    @Slot()
    def _update_preview(self) -> None:
        """Update export preview list with files to be created."""
        self._preview_list.clear()

        output_dir = Path(self._output_dir_edit.text()) if self._output_dir_edit.text() else Path("output")
        data_ext = self._get_data_extension()
        fig_ext = self._figure_format_combo.currentText().lower()

        # Add preview items based on selections
        if self._check_parameters.isChecked():
            self._preview_list.addItem(f"{output_dir}/parameters.{data_ext}")

        if self._check_intervals.isChecked():
            self._preview_list.addItem(f"{output_dir}/credible_intervals.{data_ext}")

        if self._check_posteriors.isChecked():
            self._preview_list.addItem(f"{output_dir}/posterior_samples.{data_ext}")

        if self._check_figures.isChecked():
            self._preview_list.addItem(f"{output_dir}/figures/fit_plot.{fig_ext}")
            self._preview_list.addItem(f"{output_dir}/figures/residuals.{fig_ext}")

        if self._check_diagnostics.isChecked():
            self._preview_list.addItem(f"{output_dir}/diagnostics_summary.{data_ext}")

        if self._check_raw_data.isChecked():
            self._preview_list.addItem(f"{output_dir}/raw_data.{data_ext}")

        if self._check_metadata.isChecked():
            self._preview_list.addItem(f"{output_dir}/metadata.json")

        if self._template_combo.currentText() == "Markdown Report":
            self._preview_list.addItem(f"{output_dir}/report.md")
        elif self._template_combo.currentText() == "PDF Report":
            self._preview_list.addItem(f"{output_dir}/report.pdf")

    def _validate_export(self) -> tuple[bool, str]:
        """Validate export configuration.

        Returns
        -------
        tuple[bool, str]
            (is_valid, error_message)
        """
        # Check output directory
        output_dir = self._output_dir_edit.text().strip()
        if not output_dir:
            return False, "Please select an output directory."

        output_path = Path(output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create output directory: {e}"

        # Check at least one content type is selected
        any_selected = any([
            self._check_parameters.isChecked(),
            self._check_intervals.isChecked(),
            self._check_posteriors.isChecked(),
            self._check_figures.isChecked(),
            self._check_diagnostics.isChecked(),
            self._check_raw_data.isChecked(),
            self._check_metadata.isChecked(),
        ])
        if not any_selected:
            return False, "Please select at least one content type to export."

        # Check data availability
        state = self._store.get_state()

        if self._check_parameters.isChecked() and not state.fit_results:
            return False, "No fit results available. Run fitting first."

        if (self._check_intervals.isChecked() or self._check_posteriors.isChecked() or
                self._check_diagnostics.isChecked()) and not state.bayesian_results:
            return False, "No Bayesian results available. Run Bayesian inference first."

        if self._check_raw_data.isChecked() and not state.datasets:
            return False, "No datasets loaded. Import data first."

        return True, ""

    def _get_export_config(self) -> dict[str, Any]:
        """Build export configuration from UI state."""
        return {
            "include_parameters": self._check_parameters.isChecked(),
            "include_intervals": self._check_intervals.isChecked(),
            "include_posteriors": self._check_posteriors.isChecked(),
            "include_figures": self._check_figures.isChecked(),
            "include_diagnostics": self._check_diagnostics.isChecked(),
            "include_raw_data": self._check_raw_data.isChecked(),
            "include_metadata": self._check_metadata.isChecked(),
            "data_format": self._get_data_extension(),
            "figure_format": self._figure_format_combo.currentText().lower(),
            "dpi": self._dpi_spin.value(),
            "style": self._style_combo.currentText().lower(),
            "figure_size": (self._width_spin.value(), self._height_spin.value()),
            "template": self._template_combo.currentText(),
            "output_dir": Path(self._output_dir_edit.text()),
        }

    @Slot()
    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        # Validate
        is_valid, error_msg = self._validate_export()
        if not is_valid:
            QMessageBox.warning(self, "Export Validation", error_msg)
            return

        config = self._get_export_config()

        # Emit signal for main window to handle
        self.export_requested.emit(config)

        # Also perform export directly
        self._perform_export(config)

    def _perform_export(self, config: dict[str, Any]) -> None:
        """Perform the actual export operation.

        Parameters
        ----------
        config : dict
            Export configuration
        """
        state = self._store.get_state()
        output_dir = config["output_dir"]
        data_format = config["data_format"]

        # Create progress dialog
        progress = QProgressDialog("Exporting...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Export Progress")
        progress.setMinimumDuration(500)
        progress.setValue(0)

        try:
            exported_files = []
            total_steps = sum([
                config["include_parameters"],
                config["include_intervals"],
                config["include_posteriors"],
                config["include_figures"],
                config["include_diagnostics"],
                config["include_raw_data"],
                config["include_metadata"],
                config["template"] != "None",
            ])
            current_step = 0

            # Export parameters
            if config["include_parameters"] and state.fit_results:
                progress.setLabelText("Exporting parameters...")
                for result_id, result in state.fit_results.items():
                    filepath = output_dir / f"parameters_{result_id}.{data_format}"
                    self._export_service.export_parameters(result, filepath, data_format)
                    exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Export credible intervals
            if config["include_intervals"] and state.bayesian_results:
                progress.setLabelText("Exporting credible intervals...")
                for result_id, result in state.bayesian_results.items():
                    filepath = output_dir / f"credible_intervals_{result_id}.{data_format}"
                    # Export intervals as parameters-like structure
                    if result.credible_intervals:
                        import json
                        intervals_dict = {
                            k: {"lower": v[0], "upper": v[1]}
                            for k, v in result.credible_intervals.items()
                        }
                        with open(filepath.with_suffix(".json"), "w") as f:
                            json.dump(intervals_dict, f, indent=2)
                        exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Export posterior samples
            if config["include_posteriors"] and state.bayesian_results:
                progress.setLabelText("Exporting posterior samples...")
                for result_id, result in state.bayesian_results.items():
                    filepath = output_dir / f"posterior_samples_{result_id}.{data_format}"
                    self._export_service.export_posterior(result, filepath, data_format)
                    exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Export figures
            if config["include_figures"]:
                progress.setLabelText("Exporting figures...")
                figures_dir = output_dir / "figures"
                figures_dir.mkdir(exist_ok=True)
                # Figures would be exported from plot service/canvas - placeholder
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Export diagnostics
            if config["include_diagnostics"] and state.bayesian_results:
                progress.setLabelText("Exporting diagnostics...")
                import json
                for result_id, result in state.bayesian_results.items():
                    filepath = output_dir / f"diagnostics_{result_id}.json"
                    diagnostics = {
                        "r_hat": result.r_hat,
                        "ess": result.ess,
                        "divergences": result.divergences,
                        "num_warmup": result.num_warmup,
                        "num_samples": result.num_samples,
                    }
                    with open(filepath, "w") as f:
                        json.dump(diagnostics, f, indent=2)
                    exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Export raw data
            if config["include_raw_data"] and state.datasets:
                progress.setLabelText("Exporting raw data...")
                for dataset_id, dataset in state.datasets.items():
                    if dataset.data is not None:
                        filepath = output_dir / f"data_{dataset_id}.{data_format}"
                        self._export_service.export_data(dataset.data, filepath, data_format)
                        exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Export metadata
            if config["include_metadata"]:
                progress.setLabelText("Exporting metadata...")
                import json
                import datetime
                metadata = {
                    "export_timestamp": datetime.datetime.now().isoformat(),
                    "project_name": state.project_name,
                    "active_model": state.active_model_name,
                    "dataset_count": len(state.datasets),
                    "fit_result_count": len(state.fit_results),
                    "bayesian_result_count": len(state.bayesian_results),
                }
                filepath = output_dir / "metadata.json"
                with open(filepath, "w") as f:
                    json.dump(metadata, f, indent=2)
                exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Generate report
            if config["template"] != "None":
                progress.setLabelText("Generating report...")
                template_type = "summary"
                if "bayesian" in config["template"].lower():
                    template_type = "bayesian"

                report_ext = "md" if "markdown" in config["template"].lower() else "pdf"
                filepath = output_dir / f"report.{report_ext}"

                # Build state dict for report
                report_state = {
                    "model_name": state.active_model_name,
                    "test_mode": state.datasets[list(state.datasets.keys())[0]].metadata.get("test_mode") if state.datasets else None,
                }

                # Add parameters from latest fit
                if state.fit_results:
                    latest_fit = list(state.fit_results.values())[-1]
                    report_state["parameters"] = latest_fit.parameters

                # Add diagnostics from latest Bayesian result
                if state.bayesian_results:
                    latest_bayes = list(state.bayesian_results.values())[-1]
                    report_state["diagnostics"] = {
                        "rhat": latest_bayes.r_hat,
                        "ess": latest_bayes.ess,
                    }

                self._export_service.generate_report(report_state, template_type, filepath)
                exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(100)

            progress.close()

            # Show success message
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(exported_files)} files to:\n{output_dir}"
            )

            self.export_completed.emit(str(output_dir))
            logger.info(f"Export completed: {len(exported_files)} files to {output_dir}")

        except Exception as e:
            progress.close()
            error_msg = f"Export failed: {e}"
            QMessageBox.critical(self, "Export Error", error_msg)
            self.export_failed.emit(error_msg)
            logger.error(error_msg)

    def export_results(
        self,
        item_ids: list[str],
        file_path: str,
        format: str | None = None,
        **kwargs: Any
    ) -> None:
        """Export specific results to file.

        Parameters
        ----------
        item_ids : list[str]
            IDs of items to export (fit_result IDs or bayesian_result IDs)
        file_path : str
            Output file path
        format : str, optional
            Export format (auto-detected from extension if None)
        **kwargs
            Additional export options
        """
        state = self._store.get_state()
        path = Path(file_path)

        if format is None:
            format = path.suffix.lstrip(".")

        for item_id in item_ids:
            # Check fit results
            if item_id in state.fit_results:
                result = state.fit_results[item_id]
                self._export_service.export_parameters(result, path, format)

            # Check bayesian results
            elif item_id in state.bayesian_results:
                result = state.bayesian_results[item_id]
                self._export_service.export_posterior(result, path, format)

    def export_plot(
        self,
        plot_id: str,
        file_path: str,
        dpi: int = 300,
        **kwargs: Any
    ) -> None:
        """Export plot to file.

        Parameters
        ----------
        plot_id : str
            Plot identifier
        file_path : str
            Output file path
        dpi : int
            Resolution in DPI
        **kwargs
            Additional savefig options
        """
        # Get figure from plot service (would need integration with PlotService)
        logger.info(f"Export plot {plot_id} to {file_path} at {dpi} DPI")

    def preview_export(
        self,
        item_ids: list[str],
        format: str
    ) -> dict[str, Any]:
        """Preview export without writing files.

        Parameters
        ----------
        item_ids : list[str]
            IDs of items to preview
        format : str
            Export format

        Returns
        -------
        dict
            Preview information (file paths, sizes, etc.)
        """
        config = self._get_export_config()
        self._update_preview()

        return {
            "files": [self._preview_list.item(i).text() for i in range(self._preview_list.count())],
            "format": format,
            "output_dir": str(config["output_dir"]),
        }

    def batch_export(self, exports: list[dict[str, Any]]) -> list[str]:
        """Batch export multiple items.

        Parameters
        ----------
        exports : list[dict]
            List of export configurations

        Returns
        -------
        list[str]
            Paths to exported files
        """
        exported_files = []

        for export_config in exports:
            try:
                self._perform_export(export_config)
                exported_files.append(str(export_config.get("output_dir", "")))
            except Exception as e:
                logger.error(f"Batch export item failed: {e}")

        return exported_files
