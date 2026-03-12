"""
Transform Page
=============

Data-driven transform application interface with sidebar selection,
auto-generated parameter forms, and live preview.
"""

from __future__ import annotations

from typing import Any

from rheojax.gui.compat import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QObject,
    QPushButton,
    QRunnable,
    QScrollArea,
    QSplitter,
    Qt,
    QThreadPool,
    QTimer,
    QVBoxLayout,
    QWidget,
    Signal,
    Slot,
)
from rheojax.gui.resources.styles.tokens import (
    ColorPalette,
    Spacing,
    Typography,
    button_style,
    section_header_style,
    themed,
)
from rheojax.gui.services.transform_service import TransformService
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.empty_state import EmptyStateWidget
from rheojax.gui.widgets.parameter_form import ParameterFormBuilder
from rheojax.logging import get_logger

try:
    from rheojax.gui.widgets.pyqtgraph_canvas import (
        PYQTGRAPH_AVAILABLE,
        PyQtGraphCanvas,
    )
except ImportError:
    PYQTGRAPH_AVAILABLE = False

logger = get_logger(__name__)

_MAX_PREVIEW_POINTS = 2000


class _WorkerSignals(QObject):
    """Signals for cross-thread preview result delivery."""

    result_ready = Signal(object)


class _PreviewWorker(QRunnable):
    """QRunnable that computes a transform preview off the main thread."""

    def __init__(
        self,
        service: TransformService,
        transform_key: str,
        data: Any,
        params: dict[str, Any],
        generation: int,
        signals: _WorkerSignals,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._service = service
        self._key = transform_key
        self._data = data
        self._params = params
        self._generation = generation
        self._signals = signals

    def run(self) -> None:
        result = self._service.preview_transform(self._key, self._data, self._params)
        # Attach generation so the page can discard stale results
        result["_generation"] = self._generation
        self._signals.result_ready.emit(result)


class TransformPage(QWidget):
    """Transform application page with sidebar, parameter form, and live preview."""

    transform_selected = Signal(str)
    transform_applied = Signal(str, str)  # display_name, dataset_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        self._store = StateStore()
        self._service = TransformService()
        self._metadata = self._service.get_transform_metadata()

        # Current state
        self._selected_key: str | None = None
        self._selected_display_name: str | None = None
        self._param_form: ParameterFormBuilder | None = None
        self._dataset_checklist: QListWidget | None = None
        self._preview_generation = 0

        # Thread-safe signal bridge for preview results
        self._worker_signals = _WorkerSignals(self)
        self._worker_signals.result_ready.connect(self._update_plot)

        # Debounce timer for live preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(500)
        self._preview_timer.timeout.connect(self._request_preview)

        self._setup_ui()
        logger.debug(
            "Initialization complete",
            class_name=self.__class__.__name__,
            page="TransformPage",
        )

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left sidebar
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(
            Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD
        )

        sidebar_header = QLabel("Transforms")
        sidebar_header.setStyleSheet(section_header_style())
        sidebar_layout.addWidget(sidebar_header)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(200)
        for meta in self._metadata:
            item = QListWidgetItem(meta["name"])
            item.setData(Qt.UserRole, meta["key"])
            self._sidebar.addItem(item)
        self._sidebar.currentRowChanged.connect(self._on_sidebar_changed)
        sidebar_layout.addWidget(self._sidebar)

        hint = QLabel("Select a transform to configure and preview.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {ColorPalette.TEXT_MUTED}; font-size: {Typography.SIZE_SM}pt;"
        )
        sidebar_layout.addWidget(hint)

        splitter.addWidget(sidebar_widget)

        # Right detail panel
        self._detail_container = QWidget()
        self._detail_layout = QVBoxLayout(self._detail_container)
        self._detail_layout.setContentsMargins(
            Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD
        )

        self._empty_state = EmptyStateWidget(
            "No transform selected",
            "Choose a transform from the list on the left.",
        )
        self._detail_layout.addWidget(self._empty_state)

        # Scroll area wrapping the detail content (will be populated on selection)
        self._detail_scroll_content = QWidget()
        self._detail_scroll_layout = QVBoxLayout(self._detail_scroll_content)
        self._detail_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._detail_scroll_layout.setSpacing(Spacing.MD)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._detail_scroll_content)
        self._scroll.setVisible(False)
        self._detail_layout.addWidget(self._scroll, 1)

        # Preview canvas — created lazily on first transform selection to avoid
        # OpenGL context segfaults from QOpenGLWidget init before native window
        # is shown (macOS Metal compatibility layer issue).
        self._preview_canvas: PyQtGraphCanvas | None = None
        self._preview_label: QLabel | None = None
        self._preview_initialized = False

        # Apply button (always visible at bottom of detail panel)
        self._apply_btn = QPushButton("Apply Transform")
        self._apply_btn.setStyleSheet(button_style("success", "lg"))
        self._apply_btn.clicked.connect(self._apply_transform)
        self._apply_btn.setVisible(False)
        self._detail_layout.addWidget(self._apply_btn)

        splitter.addWidget(self._detail_container)
        splitter.setSizes([200, 700])

        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Lazy preview canvas
    # ------------------------------------------------------------------

    def _ensure_preview_canvas(self) -> None:
        """Lazily create the preview canvas on first use.

        Deferred from ``_setup_ui`` to avoid OpenGL context init before the
        native window is shown, which can segfault on macOS ARM64.
        """
        if self._preview_initialized:
            return
        self._preview_initialized = True

        if not PYQTGRAPH_AVAILABLE:
            return

        try:
            self._preview_label = QLabel("Preview")
            self._preview_label.setStyleSheet(
                f"font-weight: {Typography.WEIGHT_SEMIBOLD};"
                f" font-size: {Typography.SIZE_MD_SM}pt;"
                f" margin-top: {Spacing.SM}px;"
            )
            self._preview_label.setVisible(False)
            # Insert before the apply button (last widget in detail_layout)
            idx = self._detail_layout.indexOf(self._apply_btn)
            self._detail_layout.insertWidget(idx, self._preview_label)

            self._preview_canvas = PyQtGraphCanvas()
            self._preview_canvas.setMinimumHeight(250)
            self._preview_canvas.set_labels(x_label="x", y_label="y")
            self._preview_canvas.setVisible(False)
            self._detail_layout.insertWidget(idx + 1, self._preview_canvas, 1)
        except Exception:
            logger.warning(
                "Failed to create preview canvas; preview disabled",
                page="TransformPage",
                exc_info=True,
            )
            if self._preview_label is not None:
                self._detail_layout.removeWidget(self._preview_label)
                self._preview_label.deleteLater()
            self._preview_canvas = None
            self._preview_label = None

    # ------------------------------------------------------------------
    # Sidebar selection
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_sidebar_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._metadata):
            return

        meta = self._metadata[row]
        self._selected_key = meta["key"]
        self._selected_display_name = meta["name"]
        self.transform_selected.emit(meta["name"])
        logger.debug("Transform selected", transform=meta["name"], page="TransformPage")

        self._rebuild_detail(meta)

    def _rebuild_detail(self, meta: dict[str, Any]) -> None:
        """Tear down and rebuild the scrollable detail content."""
        # Disconnect old param form signal before clearing to avoid stale callbacks
        if self._param_form is not None:
            try:
                self._param_form.values_changed.disconnect(self._on_params_changed)
            except RuntimeError:
                pass  # Already disconnected

        # Clear previous content
        self._clear_layout(self._detail_scroll_layout)
        self._param_form = None
        self._dataset_checklist = None

        # Hide empty state, show scroll + apply button
        self._empty_state.setVisible(False)
        self._scroll.setVisible(True)
        self._apply_btn.setVisible(True)

        lay = self._detail_scroll_layout

        # Header
        header = QLabel(meta["name"])
        header.setStyleSheet(section_header_style())
        lay.addWidget(header)

        desc = QLabel(meta["description"])
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {ColorPalette.TEXT_SECONDARY};"
            f" font-size: {Typography.SIZE_MD_SM}pt;"
        )
        lay.addWidget(desc)

        # Parameter form
        specs = self._service.get_transform_params(meta["key"])
        if specs:
            params_label = QLabel("Parameters")
            params_label.setStyleSheet(
                f"font-weight: {Typography.WEIGHT_SEMIBOLD};"
                f" font-size: {Typography.SIZE_MD_SM}pt;"
                f" margin-top: {Spacing.SM}px;"
            )
            lay.addWidget(params_label)

            self._param_form = ParameterFormBuilder(specs)
            self._param_form.values_changed.connect(self._on_params_changed)
            lay.addWidget(self._param_form)

        # Dataset checklist (only for multi-dataset transforms)
        if meta.get("requires_multiple", False):
            ds_label = QLabel("Datasets (select 2+)")
            ds_label.setStyleSheet(
                f"font-weight: {Typography.WEIGHT_SEMIBOLD};"
                f" font-size: {Typography.SIZE_MD_SM}pt;"
                f" margin-top: {Spacing.SM}px;"
            )
            lay.addWidget(ds_label)

            self._dataset_checklist = QListWidget()
            self._dataset_checklist.setMaximumHeight(150)
            state = self._store.get_state()
            for ds_id, ds in state.datasets.items():
                item = QListWidgetItem(ds.name)
                item.setData(Qt.UserRole, ds_id)
                item.setCheckState(Qt.CheckState.Unchecked)
                self._dataset_checklist.addItem(item)
            lay.addWidget(self._dataset_checklist)

        # Lazily create preview canvas on first transform selection to avoid
        # OpenGL context segfaults from QOpenGLWidget init before native window
        # is shown (macOS Metal compatibility layer issue).
        self._ensure_preview_canvas()
        if self._preview_canvas is not None:
            if self._preview_label is not None:
                self._preview_label.setVisible(True)
            self._preview_canvas.setVisible(True)
            self._preview_canvas.clear()

        lay.addStretch()

        # Do NOT auto-trigger preview here: preview fires only when the user
        # explicitly changes a parameter (_on_params_changed).  Auto-triggering
        # on sidebar selection causes full apply_transform calls (expensive and
        # error-prone) every time the user browses available transforms.

    # ------------------------------------------------------------------
    # Parameters changed -> debounced preview
    # ------------------------------------------------------------------

    @Slot()
    def _on_params_changed(self) -> None:
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        self._preview_timer.start()

    @Slot()
    def _request_preview(self) -> None:
        if not self._selected_key or self._preview_canvas is None:
            return

        # Multi-dataset transforms (mastercurve, srfs) can only run once the
        # user clicks Apply with all datasets present.  Running a preview with
        # only the single active dataset would always fail and produce
        # misleading ERROR-level log output.
        meta = next((m for m in self._metadata if m["key"] == self._selected_key), None)
        if meta and meta.get("requires_multiple", False):
            self._preview_canvas.clear()
            return

        dataset = self._store.get_active_dataset()
        if dataset is None:
            return

        # Build RheoData from active dataset (lazy import to avoid circular deps)
        try:
            from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

            data = rheodata_from_dataset_state(dataset)
        except Exception:
            logger.debug(
                "Could not convert dataset for preview",
                page="TransformPage",
                exc_info=True,
            )
            return

        params = self.get_selected_params()
        self._preview_generation += 1
        gen = self._preview_generation

        worker = _PreviewWorker(
            self._service,
            self._selected_key,
            data,
            params,
            gen,
            self._worker_signals,
        )
        QThreadPool.globalInstance().start(worker)

    def _update_plot(self, result: dict[str, Any]) -> None:
        if self._preview_canvas is None:
            return
        # Discard stale results
        if result.get("_generation", 0) < self._preview_generation:
            return

        self._preview_canvas.clear()

        if "error" in result:
            logger.debug(
                "Preview error",
                error=result["error"],
                page="TransformPage",
            )
            return

        x_before = self._downsample(result["x_before"])
        y_before = self._downsample(result["y_before"])
        x_after = self._downsample(result["x_after"])
        y_after = self._downsample(result["y_after"])

        self._preview_canvas.plot_data(
            x_before,
            y_before,
            name="Before",
            color=themed("CHART_1"),
            line_width=2,
        )
        self._preview_canvas.plot_data(
            x_after,
            y_after,
            name="After",
            color=themed("CHART_3"),
            line_width=2,
        )

    # ------------------------------------------------------------------
    # Apply transform
    # ------------------------------------------------------------------

    def _apply_transform(self) -> None:
        if not self._selected_display_name:
            logger.debug(
                "Apply transform called with no transform selected",
                page="TransformPage",
            )
            return

        # Pre-validate dataset requirements so the user gets a helpful message
        # before the background worker is even submitted.
        meta = next((m for m in self._metadata if m["key"] == self._selected_key), None)
        if meta and meta.get("requires_multiple", False):
            state = self._store.get_state()
            n_datasets = len(state.datasets) if state.datasets else 0
            if n_datasets < 2:
                logger.warning(
                    "Insufficient datasets for transform",
                    transform=self._selected_display_name,
                    n_datasets=n_datasets,
                    required=2,
                    page="TransformPage",
                )
                return

        dataset = self._store.get_active_dataset()
        if dataset:
            self.transform_applied.emit(self._selected_display_name, dataset.id)
            logger.info(
                "Transform applied",
                transform=self._selected_display_name,
                dataset_id=dataset.id,
                page="TransformPage",
            )
        else:
            logger.debug(
                "No active dataset for transform",
                transform=self._selected_display_name,
                page="TransformPage",
            )

    # ------------------------------------------------------------------
    # Public API (MainWindow integration contract)
    # ------------------------------------------------------------------

    def get_selected_params(self) -> dict[str, Any]:
        """Return current parameter values for the selected transform."""
        if self._param_form is None:
            return {}
        return self._param_form.get_values()

    def get_available_transforms(self) -> list[dict[str, Any]]:
        """Return list of available transforms with metadata."""
        return self._service.get_transform_metadata()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample(arr: Any) -> Any:
        """Take every Nth element if array exceeds preview limit."""
        if hasattr(arr, "__len__") and len(arr) > _MAX_PREVIEW_POINTS:
            step = len(arr) // _MAX_PREVIEW_POINTS
            return arr[::step]
        return arr

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        """Remove all children from a layout."""
        while layout.count():
            child = layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
