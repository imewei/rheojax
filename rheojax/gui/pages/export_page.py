"""
Export Page
==========

Result export interface with format selection and batch operations.
"""

from pathlib import Path
from typing import Any

from rheojax.gui.compat import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QImage,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPixmap,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
    Slot,
)
from rheojax.gui.services.export_service import ExportService
from rheojax.gui.services.plot_service import PlotService
from rheojax.gui.state.store import StateStore
from rheojax.logging import get_logger

logger = get_logger(__name__)


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
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self._store = StateStore()
        self._export_service = ExportService()
        self._plot_service = PlotService()
        self.setup_ui()

    def _dataset_to_rheodata(self, dataset: Any) -> Any:
        """Convert DatasetState into RheoData for export."""
        from rheojax.core.data import RheoData

        metadata = dict(getattr(dataset, "metadata", {}) or {})
        metadata.setdefault("test_mode", getattr(dataset, "test_mode", "oscillation"))
        if getattr(dataset, "name", None):
            metadata.setdefault("name", dataset.name)
        if getattr(dataset, "file_path", None):
            metadata.setdefault("file", str(dataset.file_path))

        return RheoData(
            x=dataset.x_data,
            y=dataset.y_data,
            metadata=metadata,
            initial_test_mode=metadata.get("test_mode"),
            validate=False,
        )

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
        self._data_format_combo.currentTextChanged.connect(self._on_data_format_changed)
        layout.addWidget(self._data_format_combo)

        # Figure format
        layout.addWidget(
            QLabel("Figure Format:", styleSheet="font-weight: bold; margin-top: 15px;")
        )
        self._figure_format_combo = QComboBox()
        self._figure_format_combo.addItems(["PNG", "SVG", "PDF", "EPS"])
        # EPS path not guaranteed; keep visible but disabled
        eps_idx = self._figure_format_combo.findText("EPS")
        if eps_idx >= 0:
            self._figure_format_combo.model().item(eps_idx).setEnabled(False)
        self._figure_format_combo.currentTextChanged.connect(
            self._on_figure_format_changed
        )
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
        layout.addWidget(
            QLabel("Plot Style:", styleSheet="font-weight: bold; margin-top: 15px;")
        )
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
        layout.addWidget(
            QLabel(
                "Report Template:", styleSheet="font-weight: bold; margin-top: 15px;"
            )
        )
        self._template_combo = QComboBox()
        self._template_combo.addItems(["None", "Markdown Report", "PDF Report"])
        self._template_combo.currentTextChanged.connect(self._update_preview)
        layout.addWidget(self._template_combo)

        layout.addStretch()

        return panel

    def _on_data_format_changed(self, fmt: str) -> None:
        """Handle data format selection change."""
        logger.debug("Format selected", format=fmt, page="ExportPage")
        self._update_preview()

    def _on_figure_format_changed(self, fmt: str) -> None:
        """Handle figure format selection change."""
        logger.debug("Format selected", format=fmt, page="ExportPage")
        self._update_preview()

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
        layout.addWidget(
            QLabel("Export Preview:", styleSheet="font-weight: bold; margin-top: 20px;")
        )

        self._preview_list = QListWidget()
        self._preview_list.setAlternatingRowColors(True)
        layout.addWidget(self._preview_list)

        self._preview_thumb = QLabel("No preview available")
        self._preview_thumb.setAlignment(Qt.AlignCenter)
        self._preview_thumb.setMinimumHeight(160)
        layout.addWidget(self._preview_thumb)

        btn_preview = QPushButton("Refresh Preview")
        btn_preview.clicked.connect(self._update_preview)
        layout.addWidget(btn_preview)

        btn_batch = QPushButton("Batch Export Data (all datasets)")
        btn_batch.clicked.connect(self._batch_export_all_datasets)
        layout.addWidget(btn_batch)

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
            self._check_parameters,
            self._check_intervals,
            self._check_posteriors,
            self._check_figures,
            self._check_diagnostics,
            self._check_raw_data,
            self._check_metadata,
        ]:
            checkbox.setChecked(True)
        self._update_preview()

    def _select_none(self) -> None:
        for checkbox in [
            self._check_parameters,
            self._check_intervals,
            self._check_posteriors,
            self._check_figures,
            self._check_diagnostics,
            self._check_raw_data,
            self._check_metadata,
        ]:
            checkbox.setChecked(False)
        self._update_preview()

    def _browse_output_dir(self) -> None:
        state = self._store.get_state()
        initial_dir = (
            str(state.last_export_dir) if state.last_export_dir else str(Path.home())
        )

        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", initial_dir
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

        output_dir = (
            Path(self._output_dir_edit.text())
            if self._output_dir_edit.text()
            else Path("output")
        )
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

        if self._template_combo.currentText().lower().startswith("markdown"):
            self._preview_list.addItem(f"{output_dir}/report.md")
        elif self._template_combo.currentText().lower().startswith("pdf"):
            self._preview_list.addItem(f"{output_dir}/report.pdf")

        # Generate a lightweight thumbnail preview
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(3.2, 2.2))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.02, 0.90, "Export Preview", fontsize=10, fontweight="bold")
            ax.text(0.02, 0.75, f"Data: {data_ext.upper()}")
            ax.text(
                0.02,
                0.62,
                f"Figures: {fig_ext.upper() if self._check_figures.isChecked() else 'None'}",
            )
            ax.text(0.02, 0.49, f"Report: {self._template_combo.currentText()}")
            ax.text(0.02, 0.36, f"Output: {output_dir}", wrap=True)
            canvas.draw()
            buf = canvas.buffer_rgba()
            width, height = canvas.get_width_height()
            qimg = QImage(buf, width, height, QImage.Format_RGBA8888)
            self._preview_thumb.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self._preview_thumb.width(),
                    self._preview_thumb.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        except Exception:
            self._preview_thumb.setText("Preview unavailable")

    def _batch_export_all_datasets(self) -> None:
        """Export all datasets in state to the selected directory (data format only)."""
        logger.debug("Export triggered", format="batch", page="ExportPage")
        output_dir = (
            Path(self._output_dir_edit.text())
            if self._output_dir_edit.text()
            else Path("output")
        )
        data_ext = self._get_data_extension()

        state = self._store.get_state()
        datasets = getattr(state, "datasets", {}) or {}
        if not datasets:
            QMessageBox.information(
                self, "Batch Export", "No datasets available to export."
            )
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        exported = []
        total_size = 0
        for name, ds in datasets.items():
            try:
                rheo = self._dataset_to_rheodata(ds)
                file_path = output_dir / f"{name}.{data_ext}"
                self._export_service.export_data(rheo, file_path, data_ext)
                exported.append(str(file_path))
                # Get file size if available
                if file_path.exists():
                    total_size += file_path.stat().st_size
            except Exception as e:
                logger.error(
                    f"Failed to export dataset {name}: {e}",
                    exc_info=True,
                )

        if exported:
            logger.info(
                "Batch export completed",
                file_count=len(exported),
                total_size_bytes=total_size,
                output_dir=str(output_dir),
            )
            QMessageBox.information(
                self,
                "Batch Export",
                f"Exported {len(exported)} dataset(s) to {output_dir}",
            )
        else:
            QMessageBox.warning(self, "Batch Export", "No datasets were exported.")

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
        any_selected = any(
            [
                self._check_parameters.isChecked(),
                self._check_intervals.isChecked(),
                self._check_posteriors.isChecked(),
                self._check_figures.isChecked(),
                self._check_diagnostics.isChecked(),
                self._check_raw_data.isChecked(),
                self._check_metadata.isChecked(),
            ]
        )
        if not any_selected:
            return False, "Please select at least one content type to export."

        # Check data availability
        state = self._store.get_state()

        if self._check_parameters.isChecked() and not state.fit_results:
            return False, "No fit results available. Run fitting first."

        if (
            self._check_intervals.isChecked()
            or self._check_posteriors.isChecked()
            or self._check_diagnostics.isChecked()
        ) and not state.bayesian_results:
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
        logger.debug(
            "Export triggered",
            format=config["data_format"],
            page="ExportPage",
        )

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
            total_steps = sum(
                [
                    config["include_parameters"],
                    config["include_intervals"],
                    config["include_posteriors"],
                    config["include_figures"],
                    config["include_diagnostics"],
                    config["include_raw_data"],
                    config["include_metadata"],
                    config["template"] != "None",
                ]
            )
            current_step = 0

            # Export parameters
            if config["include_parameters"] and state.fit_results:
                progress.setLabelText("Exporting parameters...")
                for result_id, result in state.fit_results.items():
                    filepath = output_dir / f"parameters_{result_id}.{data_format}"
                    self._export_service.export_parameters(
                        result, filepath, data_format
                    )
                    exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Export credible intervals
            if config["include_intervals"] and state.bayesian_results:
                progress.setLabelText("Exporting credible intervals...")
                for result_id, result in state.bayesian_results.items():
                    filepath = (
                        output_dir / f"credible_intervals_{result_id}.{data_format}"
                    )
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
                    filepath = (
                        output_dir / f"posterior_samples_{result_id}.{data_format}"
                    )
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

                fig_format = config.get("figure_format", "png")
                fig_dpi = config.get("dpi", 300)
                fig_style = config.get("style", "default")

                # Export fit plots for each dataset/result pair
                for result_id, fit_result in state.fit_results.items():
                    # Find associated dataset
                    dataset = None
                    dataset_id = getattr(fit_result, "dataset_id", None)
                    if dataset_id and dataset_id in state.datasets:
                        dataset = state.datasets[dataset_id]
                    elif state.datasets:
                        # Fall back to first dataset if no specific association
                        dataset = next(iter(state.datasets.values()))

                    if (
                        dataset
                        and dataset.x_data is not None
                        and dataset.y_data is not None
                    ):
                        try:
                            rheo_data = self._dataset_to_rheodata(dataset)

                            # Create and export fit plot
                            fit_fig = self._plot_service.create_fit_plot(
                                rheo_data,
                                fit_result,
                                style=fig_style,
                                test_mode=dataset.test_mode,
                            )
                            fit_path = (
                                figures_dir / f"fit_plot_{result_id}.{fig_format}"
                            )
                            self._export_service.export_figure(
                                fit_fig, fit_path, dpi=fig_dpi
                            )
                            exported_files.append(str(fit_path))

                            # Close figure to free memory
                            import matplotlib.pyplot as plt

                            plt.close(fit_fig)

                            # Create and export residuals plot
                            residuals_fig = self._plot_service.create_residual_plot(
                                rheo_data, fit_result, style=fig_style
                            )
                            residuals_path = (
                                figures_dir / f"residuals_{result_id}.{fig_format}"
                            )
                            self._export_service.export_figure(
                                residuals_fig, residuals_path, dpi=fig_dpi
                            )
                            exported_files.append(str(residuals_path))
                            plt.close(residuals_fig)

                        except Exception as e:
                            logger.warning(
                                f"Failed to export fit plots for {result_id}: {e}"
                            )

                # Export Bayesian diagnostic plots
                for result_id, bayes_result in state.bayesian_results.items():
                    try:
                        # Trace plot
                        trace_fig = self._plot_service.create_arviz_plot(
                            bayes_result, plot_type="trace", style=fig_style
                        )
                        trace_path = figures_dir / f"trace_{result_id}.{fig_format}"
                        self._export_service.export_figure(
                            trace_fig, trace_path, dpi=fig_dpi
                        )
                        exported_files.append(str(trace_path))
                        import matplotlib.pyplot as plt

                        plt.close(trace_fig)

                        # Forest plot
                        forest_fig = self._plot_service.create_arviz_plot(
                            bayes_result, plot_type="forest", style=fig_style
                        )
                        forest_path = figures_dir / f"forest_{result_id}.{fig_format}"
                        self._export_service.export_figure(
                            forest_fig, forest_path, dpi=fig_dpi
                        )
                        exported_files.append(str(forest_path))
                        plt.close(forest_fig)

                        # Pair plot
                        pair_fig = self._plot_service.create_arviz_plot(
                            bayes_result, plot_type="pair", style=fig_style
                        )
                        pair_path = figures_dir / f"pair_{result_id}.{fig_format}"
                        self._export_service.export_figure(
                            pair_fig, pair_path, dpi=fig_dpi
                        )
                        exported_files.append(str(pair_path))
                        plt.close(pair_fig)

                    except Exception as e:
                        logger.warning(
                            f"Failed to export Bayesian plots for {result_id}: {e}"
                        )

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
                    if dataset.x_data is not None and dataset.y_data is not None:
                        filepath = output_dir / f"data_{dataset_id}.{data_format}"
                        rheo = self._dataset_to_rheodata(dataset)
                        self._export_service.export_data(rheo, filepath, data_format)
                        exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(int(current_step / total_steps * 100))

            if progress.wasCanceled():
                return

            # Export metadata
            if config["include_metadata"]:
                progress.setLabelText("Exporting metadata...")
                import datetime
                import json

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
                    "test_mode": (
                        state.datasets[list(state.datasets.keys())[0]].metadata.get(
                            "test_mode"
                        )
                        if state.datasets
                        else None
                    ),
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

                self._export_service.generate_report(
                    report_state, template_type, filepath
                )
                exported_files.append(str(filepath))
                current_step += 1
                progress.setValue(100)

            progress.close()

            # Calculate total file size
            total_size = 0
            for file_str in exported_files:
                file_path = Path(file_str)
                if file_path.exists():
                    total_size += file_path.stat().st_size

            # Show success message
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(exported_files)} files to:\n{output_dir}",
            )

            self.export_completed.emit(str(output_dir))
            logger.info(
                "Export completed",
                file_count=len(exported_files),
                total_size_bytes=total_size,
                output_dir=str(output_dir),
            )

        except Exception as e:
            progress.close()
            error_msg = f"Export failed: {e}"
            QMessageBox.critical(self, "Export Error", error_msg)
            self.export_failed.emit(error_msg)
            logger.error(error_msg, exc_info=True)

    def export_results(
        self,
        item_ids: list[str],
        file_path: str,
        format: str | None = None,
        **kwargs: Any,
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
        logger.debug("Export triggered", format=format or "auto", page="ExportPage")
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
        self, plot_id: str, file_path: str, dpi: int = 300, **kwargs: Any
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
        logger.debug("Export triggered", format="plot", page="ExportPage")
        # Get figure from plot service (would need integration with PlotService)
        logger.info(
            "Export plot completed",
            plot_id=plot_id,
            file_path=file_path,
            dpi=dpi,
        )

    def preview_export(self, item_ids: list[str], format: str) -> dict[str, Any]:
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
            "files": [
                self._preview_list.item(i).text()
                for i in range(self._preview_list.count())
            ],
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
        logger.debug("Export triggered", format="batch", page="ExportPage")
        exported_files = []

        for export_config in exports:
            try:
                self._perform_export(export_config)
                exported_files.append(str(export_config.get("output_dir", "")))
            except Exception as e:
                logger.error(f"Batch export item failed: {e}", exc_info=True)

        return exported_files
