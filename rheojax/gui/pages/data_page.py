"""
Data Page
=========

Data loading, visualization, and preprocessing interface.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rheojax.gui.compat import (
    QComboBox,
    QDragEnterEvent,
    QDropEvent,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QThreadPool,
    QVBoxLayout,
    QWidget,
    Signal,
)
from rheojax.gui.jobs.import_worker import ImportWorker
from rheojax.gui.jobs.preview_worker import PreviewWorker
from rheojax.gui.resources.styles.tokens import ColorPalette, Spacing, Typography
from rheojax.gui.services.data_service import DataService
from rheojax.gui.state.store import StateStore
from rheojax.logging import get_logger

logger = get_logger(__name__)


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
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self._store = StateStore()
        self._data_service = DataService()
        self._current_file_path: Path | None = None
        self._all_file_paths: list[Path] = []
        self._preview_data: list[list] | None = None
        self._active_preview_worker: PreviewWorker | None = None
        self._active_import_worker: ImportWorker | None = None
        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(
            Spacing.PAGE_MARGIN,
            Spacing.PAGE_MARGIN,
            Spacing.PAGE_MARGIN,
            Spacing.PAGE_MARGIN,
        )
        main_layout.setSpacing(Spacing.LG)

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

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
        example_group.setStyleSheet(
            f"QGroupBox {{ font-size: {Typography.SIZE_SM}pt;"
            f" font-weight: {Typography.WEIGHT_BOLD}; }}"
        )
        example_layout = QHBoxLayout(example_group)
        self._example_combo = QComboBox()
        self._example_combo.setPlaceholderText("Select an example dataset")
        _DATA_ROOT = Path(__file__).resolve().parents[3] / "examples" / "data"
        self._example_paths: dict[str, str] = {}
        _sections: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "Relaxation",
                [
                    (
                        "Polypropylene Relaxation (polymer)",
                        "relaxation/polymers/polypropylene_relaxation.csv",
                    ),
                    (
                        "HDPE Relaxation (polymer)",
                        "relaxation/polymers/stressrelaxation_hdpe_data.csv",
                    ),
                    (
                        "Liquid Foam Relaxation",
                        "relaxation/foams/stressrelaxation_liquidfoam_data.csv",
                    ),
                    (
                        "Fish Muscle Relaxation (biological)",
                        "relaxation/biological/stressrelaxation_fishmuscle_data.csv",
                    ),
                    (
                        "Laponite Clay Relaxation",
                        "relaxation/clays/rel_lapo_1200.csv",
                    ),
                ],
            ),
            (
                "Creep",
                [
                    (
                        "Polystyrene Creep (polymer)",
                        "creep/polymers/polystyrene_creep.csv",
                    ),
                    (
                        "Mucus Creep (biological)",
                        "creep/biological/creep_mucus_data.csv",
                    ),
                ],
            ),
            (
                "Oscillation",
                [
                    (
                        "Polystyrene Oscillation 160\u00b0C",
                        "oscillation/polystyrene/oscillation_ps160_data.csv",
                    ),
                    (
                        "Chia Seed Gel Oscillation (food)",
                        "oscillation/foods/oscillation_chia_data.csv",
                    ),
                    (
                        "Metal Network Oscillation",
                        "oscillation/metal_networks/epstein.csv",
                    ),
                ],
            ),
            (
                "Flow",
                [
                    (
                        "Cellulose Hydrogel Flow",
                        "flow/hydrogels/cellulose_hydrogel_flow.csv",
                    ),
                    (
                        "Ethyl Cellulose Solution Flow",
                        "flow/solutions/rotation_ec07_data.csv",
                    ),
                ],
            ),
            (
                "Mastercurve / TTS",
                [
                    (
                        "Frequency Sweep TTS (polymer)",
                        "temperature_sweep/polymers/frequency_sweep_tts.txt",
                    ),
                    (
                        "PS Oscillation Master Curve",
                        "mastercurves/master_curve_ps_oscillation_data.csv",
                    ),
                ],
            ),
            (
                "LAOS / Advanced",
                [
                    ("OWChirp TTS", "laos/owchirp_tts.txt"),
                    ("OWChirp TCS", "laos/owchirp_tcs.txt"),
                    (
                        "Multi-technique (advanced)",
                        "multi_technique/multi_technique.txt",
                    ),
                ],
            ),
            (
                "Transforms",
                [
                    (
                        "FFT: Polypropylene Relaxation",
                        "relaxation/polymers/polypropylene_relaxation.csv",
                    ),
                    (
                        "Mastercurve: Foam DMA Temp Sweep",
                        "temperature_sweep/foams/foam_dma_0C.csv",
                    ),
                    (
                        "SRFS: Emulsion Flow Curves",
                        "flow/emulsions/emulsions_v2.csv",
                    ),
                    (
                        "OWChirp: LAOS TTS Signal",
                        "laos/owchirp_tts.txt",
                    ),
                    (
                        "SPP: LAOS Raw Signal (\u03b3\u2080=10)",
                        "laos/raw_signal_0010.txt",
                    ),
                ],
            ),
        ]
        for section_name, items in _sections:
            self._example_combo.addItem(f"── {section_name} ──")
            sep_idx = self._example_combo.count() - 1
            model = self._example_combo.model()
            model.item(sep_idx).setEnabled(False)
            for label, rel_path in items:
                full_path = str(_DATA_ROOT / rel_path)
                self._example_paths[label] = full_path
                self._example_combo.addItem(label)
        example_layout.addWidget(self._example_combo, 1)

        btn_example = QPushButton("Load")
        btn_example.clicked.connect(self._load_example_dataset)
        example_layout.addWidget(btn_example)
        layout.addWidget(example_group)

        # File info
        info_group = QGroupBox("File Information")
        info_group.setStyleSheet(
            f"QGroupBox {{ font-size: {Typography.SIZE_MD_SM}pt;"
            f" font-weight: {Typography.WEIGHT_BOLD}; }}"
        )
        info_layout = QVBoxLayout(info_group)

        self._file_name_label = QLabel("No file selected")
        self._file_name_label.setWordWrap(True)
        self._file_name_label.setStyleSheet(
            f"font-size: {Typography.SIZE_MD_SM}pt;"
            f" font-weight: {Typography.WEIGHT_MEDIUM};"
        )
        info_layout.addWidget(self._file_name_label)

        self._file_size_label = QLabel("")
        self._file_size_label.setStyleSheet(
            f"font-size: 10pt; color: {ColorPalette.TEXT_SECONDARY};"
        )
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        # Header
        header = QLabel("Data Preview (first 100 rows)")
        header.setStyleSheet(
            f"font-weight: {Typography.WEIGHT_BOLD}; font-size: {Typography.SIZE_MD_SM}pt;"
            f" padding: {Spacing.XS}px;"
        )
        layout.addWidget(header)

        # Table
        self._preview_table = QTableWidget()
        self._preview_table.setAlternatingRowColors(True)
        self._preview_table.setToolTip(
            "Preview of the loaded dataset. Shows up to 100 rows."
        )
        layout.addWidget(self._preview_table)

        # Empty state
        empty_label = QLabel("No dataset loaded. Use Browse or Load Example to begin.")
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_SECONDARY}; padding: {Spacing.SM}px;"
        )
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
        header.setStyleSheet(
            f"font-weight: {Typography.WEIGHT_BOLD}; font-size: {Typography.SIZE_MD_SM}pt;"
            f" padding: {Spacing.XS}px;"
        )
        layout.addWidget(header)

        # Mapping controls
        mapper_layout = QHBoxLayout()

        # X column
        x_layout = QVBoxLayout()
        x_layout.addWidget(QLabel("X Column:"))
        self._x_combo = QComboBox()
        self._x_combo.currentTextChanged.connect(
            lambda text: logger.debug(
                "Column mapping changed", column="x", mapping=text, page="DataPage"
            )
        )
        x_layout.addWidget(self._x_combo)
        mapper_layout.addLayout(x_layout)

        # Y column
        y_layout = QVBoxLayout()
        y_layout.addWidget(QLabel("Y Column:"))
        self._y_combo = QComboBox()
        self._y_combo.currentTextChanged.connect(
            lambda text: logger.debug(
                "Column mapping changed", column="y", mapping=text, page="DataPage"
            )
        )
        y_layout.addWidget(self._y_combo)
        mapper_layout.addLayout(y_layout)

        # Y2 column (optional)
        y2_layout = QVBoxLayout()
        y2_layout.addWidget(QLabel("Y2 Column (optional):"))
        self._y2_combo = QComboBox()
        self._y2_combo.addItem("None")
        self._y2_combo.currentTextChanged.connect(
            lambda text: logger.debug(
                "Column mapping changed", column="y2", mapping=text, page="DataPage"
            )
        )
        y2_layout.addWidget(self._y2_combo)
        mapper_layout.addLayout(y2_layout)

        # Temperature column (optional)
        temp_layout = QVBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self._temp_combo = QComboBox()
        self._temp_combo.addItem("None")
        self._temp_combo.currentTextChanged.connect(
            lambda text: logger.debug(
                "Column mapping changed",
                column="temperature",
                mapping=text,
                page="DataPage",
            )
        )
        temp_layout.addWidget(self._temp_combo)
        mapper_layout.addLayout(temp_layout)

        layout.addLayout(mapper_layout)

        # Test mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Test Mode:"))
        self._test_mode_combo = QComboBox()
        self._test_mode_combo.addItems(
            [
                "Auto-detect",
                "oscillation",
                "relaxation",
                "creep",
                "flow_curve",
                "startup",
                "laos",
            ]
        )
        self._test_mode_combo.currentTextChanged.connect(
            lambda text: logger.debug(
                "Column mapping changed",
                column="test_mode",
                mapping=text,
                page="DataPage",
            )
        )
        mode_layout.addWidget(self._test_mode_combo)
        layout.addLayout(mode_layout)

        # Deformation mode + Poisson ratio (DMTA/DMA support)
        deform_group = QGroupBox("Deformation")
        deform_layout = QHBoxLayout(deform_group)

        deform_mode_layout = QVBoxLayout()
        deform_mode_layout.addWidget(QLabel("Mode:"))
        self._deform_mode_combo = QComboBox()
        self._deform_mode_combo.addItems(["shear", "tension", "bending", "compression"])
        self._deform_mode_combo.setToolTip(
            "Deformation mode used during the experiment. "
            "Required for E* \u2194 G* conversion (tension/bending/compression)."
        )
        self._deform_mode_combo.currentTextChanged.connect(
            self._on_deformation_mode_changed
        )
        deform_mode_layout.addWidget(self._deform_mode_combo)
        deform_layout.addLayout(deform_mode_layout)

        poisson_layout = QVBoxLayout()
        poisson_layout.addWidget(QLabel("Poisson Ratio:"))
        from rheojax.gui.compat import QDoubleSpinBox

        self._poisson_spin = QDoubleSpinBox()
        self._poisson_spin.setRange(0.0, 0.5)
        self._poisson_spin.setValue(0.5)
        self._poisson_spin.setSingleStep(0.05)
        self._poisson_spin.setDecimals(3)
        self._poisson_spin.setToolTip(
            "Poisson ratio of the sample. Used only when deformation mode is not shear."
        )
        self._poisson_spin.valueChanged.connect(self._on_poisson_ratio_changed)
        poisson_layout.addWidget(self._poisson_spin)
        deform_layout.addLayout(poisson_layout)

        layout.addWidget(deform_group)

        # Metadata display
        metadata_label = QLabel("Metadata:")
        metadata_label.setStyleSheet(
            f"font-weight: {Typography.WEIGHT_BOLD}; margin-top: {Spacing.SM}px;"
        )
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
        self._empty_state.setStyleSheet(
            f"color: {ColorPalette.TEXT_MUTED}; padding: {Spacing.SM}px;"
        )
        layout.addWidget(self._empty_state)

        return panel

    def _browse_files(self) -> None:
        """Open file browser dialog."""
        # Build file filter from DataService supported formats
        formats = self._data_service.get_supported_formats()
        ext_list = " ".join(f"*{ext}" for ext in formats)
        file_filter = (
            f"Rheological Data ({ext_list});;"
            "TRIOS Files (*.txt *.csv *.xlsx);;"
            "Anton Paar RheoCompass (*.csv *.txt);;"
            "CSV/TSV Files (*.csv *.txt *.dat *.tsv);;"
            "Excel Files (*.xlsx *.xls);;"
            "All Files (*)"
        )

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Data File(s)", "", file_filter
        )

        if file_paths:
            # Store all selected paths for batch import
            self._all_file_paths = [Path(p) for p in file_paths]
            logger.info(
                "Files selected",
                count=len(file_paths),
                filepaths=file_paths,
                page="DataPage",
            )
            # Preview/validate with the first file
            self._on_file_dropped(file_paths[0])

    def _on_file_dropped(self, file_path: str) -> None:
        """Handle file drop or selection."""
        path_obj = Path(file_path)
        suffix = path_obj.suffix.lower() if path_obj.suffix else "unknown"
        logger.info(
            "File selected for import",
            filepath=file_path,
            extension=suffix,
            size_bytes=path_obj.stat().st_size if path_obj.exists() else 0,
            page="DataPage",
        )

        if not path_obj.exists():
            logger.warning("File not found", filepath=file_path)
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The selected file does not exist:\n{file_path}",
            )
            return

        # Memory guard: hard limit at 500 MB, soft warning at 50 MB
        size_bytes = path_obj.stat().st_size
        if size_bytes > 500 * 1024 * 1024:  # 500 MB hard limit
            QMessageBox.critical(
                self,
                "File Too Large",
                f"File size ({size_bytes / (1024**2):.0f} MB) exceeds the 500 MB limit.\n"
                "Consider splitting the file or using chunked loading.",
            )
            return
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
        # If called from drag-drop (not from _browse_files), reset the list
        if not self._all_file_paths or self._all_file_paths[0] != path_obj:
            self._all_file_paths = [path_obj]

        # Update file info
        n_files = len(self._all_file_paths)
        if n_files > 1:
            self._file_name_label.setText(
                f"Files: {self._current_file_path.name} (+{n_files - 1} more)"
            )
        else:
            self._file_name_label.setText(f"File: {self._current_file_path.name}")
        self._file_size_label.setText(f"Size: {size_bytes / 1024:.1f} KB")

        # Load and preview data
        self._load_preview()

        # Emit signal
        self.file_dropped.emit(file_path)

    def _load_preview(self) -> None:
        """Load and display data preview asynchronously.

        Offloads file I/O to a background thread to prevent UI freezes
        on large files. Results are delivered via signals to
        ``_on_preview_loaded`` / ``_on_preview_failed``.
        """
        if not self._current_file_path:
            return
        logger.debug(
            "Starting async preview load",
            filepath=str(self._current_file_path),
            page="DataPage",
        )

        # Show loading state
        self._file_name_label.setText(f"Loading: {self._current_file_path.name}...")
        self._preview_table.setEnabled(False)

        # R10-DA-001: Increment generation BEFORE cancelling the old worker so
        # that any in-flight result from the old worker sees a stale generation
        # number and is discarded — prevents the cancel/generation race condition
        # where a result arrives between the cancel call and the increment.
        if not hasattr(self, "_preview_generation"):
            self._preview_generation = 0
        self._preview_generation += 1
        gen = self._preview_generation

        if (
            hasattr(self, "_active_preview_worker")
            and self._active_preview_worker is not None
        ):
            # R6-GUI-008: cancel previous preview worker before starting new one
            if (
                hasattr(self._active_preview_worker, "cancel_token")
                and self._active_preview_worker.cancel_token is not None
            ):
                self._active_preview_worker.cancel_token.cancel()
            # Disconnect signals from old worker to prevent stale callbacks
            try:
                self._active_preview_worker.signals.completed.disconnect()
            except (RuntimeError, TypeError):
                pass
            try:
                self._active_preview_worker.signals.failed.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._active_preview_worker = None

        # Sharing self._data_service with PreviewWorker is safe: DataService.preview_file()
        # and DataService.load_file_multi() are stateless with respect to instance variables
        # — they only read self._supported_formats (set once in __init__, never mutated).
        # No per-call state is written back to the service instance, so concurrent workers
        # can safely share a single DataService without synchronization.
        worker = PreviewWorker(
            data_service=self._data_service,
            file_path=self._current_file_path,
            max_rows=100,
        )
        worker.signals.completed.connect(
            lambda result, g=gen: (
                self._on_preview_loaded(result)
                if g == self._preview_generation
                else None
            ),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.signals.failed.connect(
            lambda err, g=gen: (
                self._on_preview_failed(err) if g == self._preview_generation else None
            ),
            Qt.ConnectionType.QueuedConnection,
        )
        self._active_preview_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_preview_loaded(self, preview_result: dict) -> None:
        """Handle successful preview load (called on main thread via signal)."""
        self._preview_table.setEnabled(True)

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

        # Restore file name label
        if self._current_file_path:
            self._file_name_label.setText(f"File: {self._current_file_path.name}")

        # Detect file format for user feedback.
        # F-GUI-011 fix: PreviewWorker now embeds the detected format in the
        # metadata dict so we avoid blocking the main thread with file I/O here.
        # Fall back to the main-thread helper only if the worker didn't set it
        # (e.g. older cached worker instances or test stubs).
        detected_format = metadata.get("format") if isinstance(metadata, dict) else None
        if not detected_format:
            detected_format = self._detect_file_format()
            if isinstance(metadata, dict):
                metadata["format"] = detected_format
        logger.debug(
            "File format detected",
            filepath=str(self._current_file_path),
            detected_format=detected_format,
            page="DataPage",
        )

        # Update table
        self._preview_table.clear()
        self._preview_table.setRowCount(len(self._preview_data))
        self._preview_table.setColumnCount(
            len(headers) if headers else len(self._preview_data[0])
        )

        if headers:
            self._preview_table.setHorizontalHeaderLabels(headers)
        else:
            self._preview_table.setHorizontalHeaderLabels(
                [f"Col {i + 1}" for i in range(len(self._preview_data[0]))]
            )

        # Populate data
        for row_idx, row_data in enumerate(self._preview_data):
            for col_idx, value in enumerate(row_data):
                if isinstance(value, np.ndarray):
                    display_value = f"[{len(value)} values]"
                elif isinstance(value, (list, tuple)) and len(str(value)) > 50:
                    display_value = f"[{len(value)} items]"
                else:
                    display_value = str(value)
                item = QTableWidgetItem(display_value)
                self._preview_table.setItem(row_idx, col_idx, item)

        # Update column mappers
        self._update_column_mappers(
            headers or [f"Col {i + 1}" for i in range(len(self._preview_data[0]))]
        )

        # Update metadata display with format info
        metadata_lines = []
        if detected_format:
            metadata_lines.append(f"format: {detected_format}")
        metadata_lines.extend(
            [f"{k}: {v}" for k, v in metadata.items() if k != "format"]
        )
        self._metadata_text.setText("\n".join(metadata_lines))

        if hasattr(self, "_empty_state"):
            self._empty_state.hide()

        # Log successful data load with record count
        logger.info(
            "Data load completed",
            filepath=str(self._current_file_path),
            record_count=len(self._preview_data),
            column_count=len(headers) if headers else len(self._preview_data[0]),
            page="DataPage",
        )

    def _on_preview_failed(self, error_msg: str) -> None:
        """Handle failed preview load (called on main thread via signal)."""
        self._preview_table.setEnabled(True)
        logger.error(
            "Failed to load data preview",
            filepath=str(self._current_file_path),
            error=error_msg,
            page="DataPage",
        )
        self._file_name_label.setText(f"Error loading file: {error_msg}")
        self._file_size_label.setText("")
        self._preview_table.clear()
        self._metadata_text.clear()
        QMessageBox.critical(
            self, "Preview Failed", f"Failed to preview file:\n\n{error_msg}"
        )

    def _detect_file_format(self) -> str:
        """Detect the file format for user feedback."""
        if not self._current_file_path:
            return "Unknown"

        suffix = self._current_file_path.suffix.lower()
        name_lower = self._current_file_path.name.lower()

        # Check for TRIOS patterns
        if suffix == ".txt":
            try:
                first_lines = ""
                for encoding in ("utf-8", "utf-16", "latin-1"):
                    try:
                        with open(
                            self._current_file_path, encoding=encoding, errors="strict"
                        ) as f:
                            first_lines = f.read(2000)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    with open(
                        self._current_file_path, encoding="utf-8", errors="replace"
                    ) as f:
                        first_lines = f.read(2000)
                if "trios" in first_lines.lower() or "[file" in first_lines.lower():
                    return "TA Instruments TRIOS"
                if (
                    "rheometer" in first_lines.lower()
                    or "rheocompass" in first_lines.lower()
                ):
                    return "Anton Paar RheoCompass"
            except Exception:
                pass
            return "Text/CSV"

        if suffix == ".csv":
            try:
                first_lines = ""
                for encoding in ("utf-8", "utf-16", "latin-1"):
                    try:
                        with open(
                            self._current_file_path, encoding=encoding, errors="strict"
                        ) as f:
                            first_lines = f.read(2000)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    with open(
                        self._current_file_path, encoding="utf-8", errors="replace"
                    ) as f:
                        first_lines = f.read(2000)
                if (
                    "rheometer" in first_lines.lower()
                    or "rheocompass" in first_lines.lower()
                ):
                    return "Anton Paar RheoCompass"
                if "trios" in first_lines.lower():
                    return "TA Instruments TRIOS CSV"
            except Exception:
                pass
            return "CSV"

        if suffix in {".xlsx", ".xls"}:
            if "trios" in name_lower:
                return "TA Instruments TRIOS Excel"
            return "Excel"

        if suffix == ".tri":
            return "TA Instruments TRIOS Binary"

        if suffix in {".rdf", ".dat"}:
            return "Rheological Data"

        return suffix.upper().lstrip(".")

    def _update_column_mappers(self, columns: list[str]) -> None:
        """Update column mapper dropdowns with smart suggestions."""
        if not columns:
            return
        logger.debug(
            "Updating column mappers",
            num_columns=len(columns),
            columns=columns[:10],
            page="DataPage",
        )

        # Clear and repopulate
        for combo in [self._x_combo, self._y_combo, self._y2_combo, self._temp_combo]:
            combo.blockSignals(True)
            combo.clear()

        try:
            self._x_combo.addItems(columns)
            self._y_combo.addItems(columns)
            self._y2_combo.addItem("None")
            self._y2_combo.addItems(columns)
            self._temp_combo.addItem("None")
            self._temp_combo.addItems(columns)

            # Use DataService column suggestions for smarter auto-mapping
            suggestions = {
                "x_suggestions": [],
                "y_suggestions": [],
                "y2_suggestions": [],
            }
            if self._current_file_path:
                try:
                    suggestions = self._data_service.get_column_suggestions(
                        self._current_file_path
                    )
                except Exception:
                    pass  # Fall back to simple matching

            # Apply suggestions or fallback to simple matching
            x_suggestions = suggestions.get("x_suggestions", [])
            y_suggestions = suggestions.get("y_suggestions", [])
            y2_suggestions = suggestions.get("y2_suggestions", [])

            # Select first suggested X column
            if x_suggestions:
                for idx, col in enumerate(columns):
                    if col in x_suggestions:
                        self._x_combo.setCurrentIndex(idx)
                        break
            else:
                # Fallback: simple matching
                for idx, col in enumerate(columns):
                    col_lower = col.lower()
                    if any(
                        x in col_lower for x in ["time", "freq", "omega", "angular"]
                    ):
                        self._x_combo.setCurrentIndex(idx)
                        break

            # Select first suggested Y column
            if y_suggestions:
                for idx, col in enumerate(columns):
                    if col in y_suggestions:
                        self._y_combo.setCurrentIndex(idx)
                        break
            else:
                # Fallback: simple matching (avoid loss modulus)
                for idx, col in enumerate(columns):
                    col_lower = col.lower()
                    if any(
                        y in col_lower for y in ["g'", "storage", "stress", "modulus"]
                    ):
                        if "loss" not in col_lower and "''" not in col:
                            self._y_combo.setCurrentIndex(idx)
                            break

            # Select first suggested Y2 column
            if y2_suggestions:
                for idx, col in enumerate(columns):
                    if col in y2_suggestions:
                        self._y2_combo.setCurrentIndex(idx + 1)  # +1 for "None" option
                        break
            else:
                # Fallback: simple matching for loss modulus
                for idx, col in enumerate(columns):
                    col_lower = col.lower()
                    if any(y2 in col_lower for y2 in ["g''", "loss", "gdoubleprime"]):
                        self._y2_combo.setCurrentIndex(idx + 1)
                        break

            # Temperature column detection
            for idx, col in enumerate(columns):
                col_lower = col.lower()
                if any(t in col_lower for t in ["temp", "temperature"]):
                    self._temp_combo.setCurrentIndex(idx + 1)  # +1 for "None" option
                    break

            logger.debug(
                "Column mapping auto-applied",
                x_selected=self._x_combo.currentText(),
                y_selected=self._y_combo.currentText(),
                y2_selected=self._y2_combo.currentText(),
                used_service_suggestions=bool(x_suggestions or y_suggestions),
                page="DataPage",
            )
        finally:
            for combo in [
                self._x_combo,
                self._y_combo,
                self._y2_combo,
                self._temp_combo,
            ]:
                combo.blockSignals(False)

    def _on_deformation_mode_changed(self, mode: str) -> None:
        """Persist deformation mode selection to the store."""
        logger.debug(
            "Option changed",
            page="DataPage",
            field="deformation_mode",
            value=mode,
        )
        self._store.dispatch("SET_DEFORMATION_MODE", {"deformation_mode": mode})

    def _on_poisson_ratio_changed(self, value: float) -> None:
        """Persist Poisson ratio selection to the store."""
        logger.debug(
            "Option changed",
            page="DataPage",
            field="poisson_ratio",
            value=value,
        )
        self._store.dispatch("SET_POISSON_RATIO", {"poisson_ratio": value})

    def _reset_mapping(self) -> None:
        """Reset column mapping to defaults."""
        if self._preview_data:
            headers = []
            for i in range(self._preview_table.columnCount()):
                item = self._preview_table.horizontalHeaderItem(i)
                headers.append(item.text() if item is not None else f"Column {i}")
            self._update_column_mappers(headers)

    def _apply_import(self) -> None:
        """Apply column mapping and import data via background worker."""
        if not self._current_file_path:
            return

        # Get mapping
        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()
        y2_col = (
            self._y2_combo.currentText() if self._y2_combo.currentIndex() > 0 else None
        )
        test_mode = self._test_mode_combo.currentText()
        if test_mode == "Auto-detect":
            test_mode = None  # Let service auto-detect

        temp_col = self._temp_combo.currentText()
        if temp_col == "None":
            temp_col = None

        logger.info(
            "Import initiated",
            filepath=str(self._current_file_path),
            x_col=x_col,
            y_col=y_col,
            y2_col=y2_col,
            temp_col=temp_col,
            test_mode=test_mode if test_mode is not None else "auto-detect",
            page="DataPage",
        )

        # Validate column selections before launching worker
        if not x_col or not y_col:
            QMessageBox.warning(
                self, "Invalid Selection", "Please select both X and Y columns."
            )
            return
        if x_col == y_col:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "X and Y columns must be different.",
            )
            return

        # Show loading state
        n_files = len(self._all_file_paths)
        if n_files > 1:
            self._file_name_label.setText(f"Importing {n_files} files...")
        else:
            self._file_name_label.setText(
                f"Importing: {self._current_file_path.name}..."
            )

        # Launch background worker
        worker = ImportWorker(
            data_service=self._data_service,
            file_path=self._current_file_path,
            file_paths=self._all_file_paths,
            x_col=x_col or None,
            y_col=y_col or None,
            y2_col=y2_col,
            test_mode=test_mode,
            temp_col=temp_col,
        )
        # R10-STO-002: Use QueuedConnection to guarantee the callback runs on
        # the main thread — ImportWorker emits signals from a worker thread.
        worker.signals.completed.connect(
            lambda datasets: self._on_import_completed(datasets, test_mode),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.signals.failed.connect(
            self._on_import_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._active_import_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_import_completed(self, datasets: list, test_mode: str | None) -> None:
        """Handle successful import (called on main thread via signal)."""
        import uuid

        _VALID_TEST_MODES = {
            "oscillation",
            "relaxation",
            "creep",
            "flow_curve",
            "startup",
            "laos",
            "rotation",
            "unknown",
        }
        _TEST_MODE_ALIASES = {
            "rotation": "flow_curve",
        }

        store = StateStore()
        first_dataset_id: str | None = None

        # Pre-extract source file info before iterating (pop removes tags)
        source_files: list[str | None] = []
        for ds in datasets:
            meta = getattr(ds, "metadata", None) or {}
            source_files.append(meta.pop("_source_file", None))

        # Count datasets per source file for segment numbering
        from collections import Counter

        source_counts = Counter(source_files)

        # Track segment index per source file
        source_seg_idx: dict[str | None, int] = {}

        for idx, rheo_data in enumerate(datasets):
            # Auto-detect test mode if not specified
            detected_mode = test_mode
            if detected_mode is None:
                detected_mode = self._data_service.detect_test_mode(rheo_data)

            # Map legacy test_mode aliases
            if detected_mode in _TEST_MODE_ALIASES:
                logger.debug(
                    "Mapping legacy test_mode",
                    from_mode=detected_mode,
                    to_mode=_TEST_MODE_ALIASES[detected_mode],
                )
                detected_mode = _TEST_MODE_ALIASES[detected_mode]

            # Validate test_mode against known modes
            if detected_mode and detected_mode not in _VALID_TEST_MODES:
                logger.warning(
                    "Unknown test_mode detected, defaulting to 'unknown'",
                    detected=detected_mode,
                )
                detected_mode = "unknown"

            # Generate dataset_id
            dataset_id = str(uuid.uuid4())
            if first_dataset_id is None:
                first_dataset_id = dataset_id

            # Dataset name: use source file stem
            source_file = source_files[idx]
            source_stem = (
                Path(source_file).stem if source_file else self._current_file_path.stem
            )

            # For multi-segment files from the same source, add segment suffix
            if source_counts[source_file] > 1:
                seg_idx = source_seg_idx.get(source_file, 0)
                source_seg_idx[source_file] = seg_idx + 1
                name = f"{source_stem}_segment_{seg_idx + 1}"
            else:
                name = source_stem

            # Register dataset in state store.
            # For complex oscillation data, split into real G' (y_data) and
            # real G'' (y2_data) so both converters reconstruct correctly.
            # F-IO-R3-009: avoids storing redundant complex + real.
            is_complex = np.iscomplexobj(rheo_data.y)
            # Convert JAX arrays to NumPy at the state dispatch boundary
            # to avoid per-element device sync in show_dataset() / str(x[i]).
            x_np = np.asarray(rheo_data.x)
            y_np = np.asarray(rheo_data.y)
            file_path_str = source_file or str(self._current_file_path)
            ds_meta = getattr(rheo_data, "metadata", None) or {}
            store.dispatch(
                "IMPORT_DATA_SUCCESS",
                {
                    "dataset_id": dataset_id,
                    "file_path": file_path_str,
                    "name": name,
                    "test_mode": detected_mode or "unknown",
                    "x_data": x_np,
                    "y_data": (np.real(y_np) if is_complex else y_np),
                    "y2_data": (np.imag(y_np) if is_complex else None),
                    "metadata": {
                        **ds_meta,
                        "x_units": getattr(rheo_data, "x_units", None),
                        "y_units": getattr(rheo_data, "y_units", None),
                    },
                },
            )

        # For multi-file/multi-segment imports, ensure the first dataset is active
        if len(datasets) > 1 and first_dataset_id:
            store.dispatch("SET_ACTIVE_DATASET", {"dataset_id": first_dataset_id})

        # Log successful import
        logger.info(
            "Data import completed",
            filepath=str(self._current_file_path),
            dataset_count=len(datasets),
            record_count=sum(
                getattr(ds.x, "shape", (0,))[0] if ds.x is not None else 0
                for ds in datasets
            ),
            page="DataPage",
        )

        # Notify user about import results
        n_files = len(self._all_file_paths)
        if n_files > 1:
            self._file_name_label.setText(
                f"Imported {len(datasets)} datasets from {n_files} files"
            )
        elif len(datasets) > 1:
            self._file_name_label.setText(
                f"Imported {len(datasets)} segments from {self._current_file_path.name}"
            )
        else:
            self._file_name_label.setText(f"File: {self._current_file_path.name}")

        self.apply_mapping.emit()
        try:
            self._store.dispatch("SET_TAB", {"tab": "transform"})
        except Exception as tab_err:
            logger.warning("Failed to switch tab", error=str(tab_err))

    def _on_import_failed(self, error_msg: str) -> None:
        """Handle failed import (called on main thread via signal)."""
        logger.error(
            "Failed to import data",
            filepath=str(self._current_file_path),
            error=error_msg,
            page="DataPage",
        )
        self._preview_data = None
        self._file_name_label.setText(f"Import error: {error_msg}")
        QMessageBox.critical(
            self, "Import Failed", f"Failed to load file:\n\n{error_msg}"
        )
        store = StateStore()
        store.dispatch(
            "IMPORT_DATA_FAILED",
            {
                "file_path": str(self._current_file_path),
                "error": error_msg,
            },
        )

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
        x_vals = dataset.x_data if dataset.x_data is not None else []
        y_vals = dataset.y_data if dataset.y_data is not None else []
        y2_vals = dataset.y2_data if dataset.y2_data is not None else []

        # If y is complex, extract real/imag for display (F-IO-R3-005)
        y_arr = np.asarray(y_vals) if len(y_vals) > 0 else np.array([])
        if np.iscomplexobj(y_arr):
            y_display = np.real(y_arr)
            y2_display = np.imag(y_arr)
            has_y2 = True
        else:
            y_display = y_arr
            y2_display = (
                np.asarray(y2_vals)
                if y2_vals is not None and len(y2_vals) > 0
                else None
            )
            has_y2 = y2_display is not None and len(y2_display) > 0

        # Determine columns
        headers = ["x", "y"]
        if has_y2:
            headers.append("y2")

        rows = min(len(x_vals), len(y_display), 100)

        self._preview_table.clear()
        self._preview_table.setRowCount(rows)
        self._preview_table.setColumnCount(len(headers))
        self._preview_table.setHorizontalHeaderLabels(headers)

        for i in range(rows):
            self._preview_table.setItem(i, 0, QTableWidgetItem(str(x_vals[i])))
            self._preview_table.setItem(i, 1, QTableWidgetItem(str(y_display[i])))
            if has_y2:
                self._preview_table.setItem(i, 2, QTableWidgetItem(str(y2_display[i])))

        # Update column mappers to reflect the loaded dataset
        self._update_column_mappers(headers)

        # Update metadata text with validation info
        validation = self.validate_dataset(dataset_id)
        warnings = (
            validation.get("warnings", []) if isinstance(validation, dict) else []
        )
        inferred_mode = (
            validation.get("test_mode")
            if isinstance(validation, dict)
            else dataset.test_mode
        )

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
            logger.warning(
                "Dataset validation failed",
                dataset_id=dataset_id,
                error=str(exc),
                page="DataPage",
                exc_info=True,
            )
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

            # Persist back into state as a modified dataset.
            # Same split convention as import: real y_data + real y2_data.
            is_complex = np.iscomplexobj(processed.y)
            payload = {
                "dataset_id": dataset.id,
                "file_path": str(dataset.file_path) if dataset.file_path else None,
                "name": dataset.name,
                "test_mode": processed.metadata.get("test_mode", dataset.test_mode),
                "x_data": processed.x,
                "y_data": (
                    np.real(np.asarray(processed.y)) if is_complex else processed.y
                ),
                "y2_data": (np.imag(np.asarray(processed.y)) if is_complex else None),
                "metadata": processed.metadata,
            }
            store.dispatch("IMPORT_DATA_SUCCESS", payload)

            # Refresh preview/mapping UI
            self.show_dataset(dataset.id)

        except Exception as exc:
            logger.warning(
                "Preprocessing failed",
                dataset_id=dataset_id,
                operations=operations,
                error=str(exc),
                page="DataPage",
                exc_info=True,
            )


class DropZone(QFrame):
    """Drag-and-drop zone for file import."""

    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize drop zone."""
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("File drop zone")
        self.setAccessibleDescription(
            "Drag and drop a data file here, or press Enter to browse"
        )
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {ColorPalette.BG_SURFACE};
                border: 2px dashed {ColorPalette.BORDER_DEFAULT};
                border-radius: 10px;
            }}
            QFrame:hover {{
                border-color: {ColorPalette.PRIMARY};
                background-color: {ColorPalette.PRIMARY_SUBTLE};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon_label = QLabel("Drop Files Here")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(
            f"font-size: {Typography.SIZE_HEADING}pt; font-weight: {Typography.WEIGHT_BOLD};"
            f" color: {ColorPalette.TEXT_MUTED};"
        )
        layout.addWidget(icon_label)

        text_label = QLabel("Drag and drop data file here\n\nor click Browse to select")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_SECONDARY}; font-size: 12pt;"
        )
        layout.addWidget(text_label)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop event."""
        if event.mimeData().hasUrls():
            file_path = None
            for url in event.mimeData().urls():
                candidate = url.toLocalFile()
                if candidate:
                    file_path = candidate
                    break
            if not file_path:
                return
            logger.debug("File selected", filepath=file_path, page="DataPage")
            self.file_dropped.emit(file_path)
            event.acceptProposedAction()
