"""
Pipeline Sidebar
================

Sidebar widget for managing visual pipeline steps.

Provides step listing with status indicators, add/remove/reorder controls,
and run buttons. Connects to StateSignals for reactive updates.
"""

from rheojax.gui.compat import (
    QAction,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)
from rheojax.gui.state import actions as pipeline_actions
from rheojax.gui.state.selectors import (
    get_pipeline_name,
    get_selected_pipeline_step,
    get_visual_pipeline_steps,
    is_pipeline_running,
)
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.pipeline_step_delegate import (
    ROLE_STATUS,
    ROLE_STEP_ID,
    ROLE_STEP_TYPE,
    PipelineStepDelegate,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Step type definitions: (display_label, internal step_type key)
_STEP_TYPES: list[tuple[str, str]] = [
    ("Load Data", "load"),
    ("Transform", "transform"),
    ("Fit Model", "fit"),
    ("Bayesian Inference", "bayesian"),
    ("Export Results", "export"),
]


class PipelineSidebar(QWidget):
    """Pipeline management sidebar.

    Displays an editable pipeline name, an "Add Step" button with a dropdown
    menu, a list of pipeline steps with status indicators, and run controls.

    Signals
    -------
    step_selected : Signal(str)
        Emitted when a step is selected (step_id)
    run_all_requested : Signal()
        Emitted when the user clicks "Run All"
    run_step_requested : Signal(str)
        Emitted when the user clicks "Run Step" (step_id)
    step_added : Signal(str)
        Emitted after a step is added to the pipeline (step_id)

    Example
    -------
    >>> sidebar = PipelineSidebar()  # doctest: +SKIP
    >>> sidebar.step_selected.connect(on_step_selected)  # doctest: +SKIP
    """

    step_selected = Signal(str)
    run_all_requested = Signal()
    run_step_requested = Signal(str)
    step_added = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the pipeline sidebar.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Pipeline name ---
        name_label = QLabel("Pipeline")
        name_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: #374151;")
        root.addWidget(name_label)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Untitled Pipeline")
        self._name_edit.setText(get_pipeline_name())
        self._name_edit.setStyleSheet(
            "QLineEdit { border: 1px solid #D1D5DB;"
            " border-radius: 4px; padding: 3px 6px; }"
            "QLineEdit:focus { border-color: #4338CA; }"
        )
        self._name_edit.editingFinished.connect(self._on_name_changed)
        root.addWidget(self._name_edit)

        # --- Add Step button ---
        self._add_btn = QPushButton("+ Add Step")
        self._add_btn.setStyleSheet(
            "QPushButton { background-color: #4338CA; color: white; border: none;"
            "border-radius: 4px; padding: 5px 10px; font-size: 9pt; }"
            "QPushButton:hover { background-color: #4F46E5; }"
            "QPushButton:pressed { background-color: #3730A3; }"
        )
        self._add_btn.clicked.connect(self._show_add_menu)
        root.addWidget(self._add_btn)

        # --- Step list ---
        steps_label = QLabel("Steps")
        steps_label.setStyleSheet("font-size: 8pt; color: #6B7280; margin-top: 4px;")
        root.addWidget(steps_label)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setAlternatingRowColors(False)
        self._list.setStyleSheet(
            "QListWidget { border: 1px solid #E5E7EB;"
            " border-radius: 4px; background: #F9FAFB; }"
            "QListWidget::item { border-bottom: 1px solid #F3F4F6; }"
            "QListWidget::item:selected { background: #EEF2FF; }"
        )
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.currentItemChanged.connect(self._on_item_selected)

        delegate = PipelineStepDelegate(self._list)
        self._list.setItemDelegate(delegate)
        root.addWidget(self._list, stretch=1)

        # --- Run buttons ---
        run_row = QHBoxLayout()
        run_row.setSpacing(6)

        self._run_all_btn = QPushButton("Run All")
        self._run_all_btn.setStyleSheet(
            "QPushButton { background-color: #22C55E; color: white; border: none;"
            "border-radius: 4px; padding: 5px 10px; font-size: 9pt; }"
            "QPushButton:hover { background-color: #16A34A; }"
            "QPushButton:disabled { background-color: #D1D5DB; color: #9CA3AF; }"
        )
        self._run_all_btn.clicked.connect(self._on_run_all)
        run_row.addWidget(self._run_all_btn)

        self._run_step_btn = QPushButton("Run Step")
        self._run_step_btn.setStyleSheet(
            "QPushButton { background-color: #3B82F6; color: white; border: none;"
            "border-radius: 4px; padding: 5px 10px; font-size: 9pt; }"
            "QPushButton:hover { background-color: #2563EB; }"
            "QPushButton:disabled { background-color: #D1D5DB; color: #9CA3AF; }"
        )
        self._run_step_btn.clicked.connect(self._on_run_step)
        run_row.addWidget(self._run_step_btn)

        root.addLayout(run_row)

        # --- Datasets placeholder ---
        datasets_label = QLabel("Datasets")
        datasets_label.setStyleSheet(
            "font-weight: bold; font-size: 9pt; color: #374151; margin-top: 6px;"
        )
        root.addWidget(datasets_label)

        datasets_placeholder = QLabel("(Dataset tree goes here)")
        datasets_placeholder.setStyleSheet(
            "color: #9CA3AF; font-size: 8pt; padding: 6px;"
            "border: 1px dashed #D1D5DB; border-radius: 4px;"
        )
        datasets_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(datasets_placeholder)

        # --- Connect state signals ---
        signals = StateStore().signals
        signals.pipeline_structure_changed.connect(self._refresh_list)
        signals.pipeline_step_status_changed.connect(self._on_step_status_changed)
        signals.pipeline_name_changed.connect(self._on_pipeline_name_changed)
        signals.pipeline_execution_started.connect(self._update_run_buttons)
        signals.pipeline_execution_completed.connect(self._update_run_buttons)
        signals.pipeline_step_status_changed.connect(lambda *_: self._update_run_buttons())

        # Initial populate
        self._refresh_list()
        self._update_run_buttons()
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def select_step(self, step_id: str | None) -> None:
        """Programmatically select a step by ID.

        Parameters
        ----------
        step_id : str | None
            Step ID to select, or None to deselect
        """
        if step_id is None:
            self._list.clearSelection()
            return
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(ROLE_STEP_ID) == step_id:
                self._list.setCurrentItem(item)
                return

    def select_step_by_type(self, step_type: str | None) -> None:
        """Programmatically select the first step matching a step_type.

        List items are UUID-keyed (ROLE_STEP_ID), so ``select_step`` cannot
        match on a step_type string. This method iterates ROLE_STEP_TYPE data
        and selects the first item whose type equals *step_type*.

        Parameters
        ----------
        step_type : str | None
            Step type to match (e.g. ``"load"``, ``"fit"``), or None to
            deselect all items.
        """
        if not step_type:
            self._list.clearSelection()
            return
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(ROLE_STEP_TYPE) == step_type:
                self._list.setCurrentItem(item)
                return

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_name_changed(self) -> None:
        """Dispatch pipeline name update when editing finishes."""
        name = self._name_edit.text().strip() or "Untitled Pipeline"
        pipeline_actions.set_pipeline_name(name)
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="name_changed",
            name=name,
        )

    def _show_add_menu(self) -> None:
        """Show dropdown menu for choosing a new step type."""
        menu = QMenu(self)
        for label, step_type in _STEP_TYPES:
            action = QAction(label, self)
            # Capture step_type and label in default args to avoid late-binding closure
            action.triggered.connect(
                lambda checked=False, st=step_type, lbl=label: self._add_step(st, lbl)
            )
            menu.addAction(action)
        btn_rect = self._add_btn.rect()
        pos = self._add_btn.mapToGlobal(btn_rect.bottomLeft())
        menu.exec(pos)

    def _add_step(self, step_type: str, name: str) -> None:
        """Add a new step to the pipeline.

        Parameters
        ----------
        step_type : str
            Internal step type key
        name : str
            Display name for the step
        """
        step_id = pipeline_actions.add_pipeline_step(step_type=step_type, name=name)
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="add_step",
            step_type=step_type,
            step_id=step_id,
        )
        self.step_added.emit(step_id)

    def _show_context_menu(self, pos) -> None:
        """Show right-click context menu for step list items.

        Parameters
        ----------
        pos : QPoint
            Position of the right-click event (in list coordinates)
        """
        item = self._list.itemAt(pos)
        if item is None:
            return
        step_id: str = item.data(ROLE_STEP_ID)
        row = self._list.row(item)
        total = self._list.count()

        menu = QMenu(self)

        remove_action = QAction("Remove", self)
        remove_action.triggered.connect(lambda: self._remove_step(step_id))
        menu.addAction(remove_action)

        menu.addSeparator()

        if row > 0:
            up_action = QAction("Move Up", self)
            up_action.triggered.connect(lambda: self._move_step(step_id, row - 1))
            menu.addAction(up_action)

        if row < total - 1:
            down_action = QAction("Move Down", self)
            down_action.triggered.connect(lambda: self._move_step(step_id, row + 1))
            menu.addAction(down_action)

        menu.exec(self._list.mapToGlobal(pos))

    def _remove_step(self, step_id: str) -> None:
        """Remove a step from the pipeline.

        Parameters
        ----------
        step_id : str
            ID of the step to remove
        """
        pipeline_actions.remove_pipeline_step(step_id)
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="remove_step",
            step_id=step_id,
        )

    def _move_step(self, step_id: str, new_position: int) -> None:
        """Move a step to a new list position.

        Parameters
        ----------
        step_id : str
            ID of the step to move
        new_position : int
            Zero-based target position
        """
        pipeline_actions.reorder_pipeline_step(step_id, new_position)

    def _on_item_selected(self, current: QListWidgetItem | None, _previous) -> None:
        """Handle list selection change.

        Parameters
        ----------
        current : QListWidgetItem | None
            Newly selected item
        _previous : QListWidgetItem | None
            Previously selected item (unused)
        """
        if current is None:
            pipeline_actions.select_pipeline_step(None)
            return
        step_id: str = current.data(ROLE_STEP_ID)
        pipeline_actions.select_pipeline_step(step_id)
        self._update_run_buttons()
        self.step_selected.emit(step_id)
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="step_selected",
            step_id=step_id,
        )

    def _on_run_all(self) -> None:
        """Emit run_all_requested signal."""
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="run_all_requested",
        )
        self.run_all_requested.emit()

    def _on_run_step(self) -> None:
        """Emit run_step_requested for the currently selected step."""
        current = self._list.currentItem()
        if current is None:
            return
        step_id: str = current.data(ROLE_STEP_ID)
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="run_step_requested",
            step_id=step_id,
        )
        self.run_step_requested.emit(step_id)

    # ------------------------------------------------------------------
    # State-driven refresh
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        """Repopulate the step list from store state."""
        selected_step = get_selected_pipeline_step()
        selected_id = selected_step.id if selected_step else None

        self._list.blockSignals(True)
        self._list.clear()

        steps = get_visual_pipeline_steps()
        for step in steps:
            item = QListWidgetItem(step.name)
            item.setData(ROLE_STEP_ID, step.id)
            item.setData(ROLE_STEP_TYPE, step.step_type)
            item.setData(ROLE_STATUS, step.status)
            self._list.addItem(item)

        self._list.blockSignals(False)

        if selected_id:
            self.select_step(selected_id)
            if self._list.currentItem() is None:
                pipeline_actions.select_pipeline_step(None)
                self.step_selected.emit("")

        self._update_run_buttons()

    def _on_step_status_changed(self, step_id: str, _status_name: str) -> None:
        """Update status icon for a single step without full repopulate.

        Parameters
        ----------
        step_id : str
            ID of the step whose status changed
        _status_name : str
            New status name string (unused — resolved from store)
        """
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(ROLE_STEP_ID) == step_id:
                steps = get_visual_pipeline_steps()
                matching = next((s for s in steps if s.id == step_id), None)
                if matching:
                    item.setData(ROLE_STATUS, matching.status)
                    self._list.viewport().update()
                break

    def _on_pipeline_name_changed(self, new_name: str) -> None:
        """Sync the name QLineEdit when state changes externally.

        Parameters
        ----------
        new_name : str
            New pipeline name from state
        """
        if self._name_edit.text() != new_name:
            self._name_edit.blockSignals(True)
            self._name_edit.setText(new_name)
            self._name_edit.blockSignals(False)

    def _update_run_buttons(self) -> None:
        """Enable/disable run buttons based on pipeline state."""
        running = is_pipeline_running()
        has_steps = self._list.count() > 0
        has_selection = self._list.currentItem() is not None

        self._run_all_btn.setEnabled(not running and has_steps)
        self._run_step_btn.setEnabled(not running and has_selection)
