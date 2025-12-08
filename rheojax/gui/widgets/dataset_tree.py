"""
Dataset Tree Widget
==================

Hierarchical tree view for managing multiple datasets.
"""

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem, QWidget

from rheojax.gui.state.store import DatasetState


class DatasetTree(QTreeWidget):
    """Tree widget for dataset hierarchy with drag-and-drop.

    Features:
        - Three-level hierarchy: Project > Dataset > File
        - Custom icons for each level and status
        - Context menus for each level
        - Drag-and-drop for reordering
        - Status indicators (loaded, fitted, bayesian)

    Signals
    -------
    dataset_selected : Signal(str)
        Emitted when a dataset is selected
    file_selected : Signal(str, Path)
        Emitted when a file is selected (dataset_id, file_path)
    context_menu_requested : Signal(QPoint)
        Emitted when context menu is requested

    Example
    -------
    >>> tree = DatasetTree()  # doctest: +SKIP
    >>> tree.add_dataset(dataset_state)  # doctest: +SKIP
    >>> tree.dataset_selected.connect(on_dataset_selected)  # doctest: +SKIP
    """

    dataset_selected = Signal(str)
    file_selected = Signal(str, Path)
    context_menu_requested = Signal(QPoint)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize dataset tree.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        # Configure tree
        self.setHeaderLabels(["Name", "Type", "Status"])
        self.setStyleSheet("QTreeWidget { font-size: 11pt; } QHeaderView::section { font-size: 11pt; }")
        self.setColumnWidth(0, 250)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 80)

        # Enable drag and drop for reordering
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QTreeWidget.InternalMove)

        # Enable selection
        self.setSelectionMode(QTreeWidget.SingleSelection)

        # Track items
        self._project_item: QTreeWidgetItem | None = None
        self._dataset_items: dict[str, QTreeWidgetItem] = {}

        # Connect signals
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Create root project item
        self._create_project_item()

    def _create_project_item(self) -> None:
        """Create the root project item."""
        self._project_item = QTreeWidgetItem(self, ["Project", "Folder", ""])
        self._project_item.setExpanded(True)
        # Set icon (placeholder - would use actual icon file)
        self._project_item.setForeground(0, QBrush(QColor(100, 100, 100)))

    def add_dataset(self, dataset_state: DatasetState) -> None:
        """Add dataset to tree.

        Parameters
        ----------
        dataset_state : DatasetState
            Dataset state object containing dataset information
        """
        if dataset_state.id in self._dataset_items:
            # Update existing item
            self.update_dataset(dataset_state)
            return

        # Create dataset item
        dataset_item = QTreeWidgetItem(
            self._project_item,
            [dataset_state.name, dataset_state.test_mode, "loaded"]
        )
        dataset_item.setData(0, Qt.UserRole, dataset_state.id)
        dataset_item.setExpanded(False)

        # Set status color
        self._set_status_color(dataset_item, "loaded")

        # Add file sub-item if file_path exists
        if dataset_state.file_path:
            self.add_file(dataset_state.id, dataset_state.file_path)

        self._dataset_items[dataset_state.id] = dataset_item

    def update_dataset(self, dataset_state: DatasetState) -> None:
        """Update existing dataset in tree.

        Parameters
        ----------
        dataset_state : DatasetState
            Updated dataset state
        """
        if dataset_state.id not in self._dataset_items:
            return

        item = self._dataset_items[dataset_state.id]
        item.setText(0, dataset_state.name)
        item.setText(1, dataset_state.test_mode)

    def remove_dataset(self, dataset_id: str) -> None:
        """Remove dataset from tree.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        """
        if dataset_id not in self._dataset_items:
            return

        item = self._dataset_items[dataset_id]
        if self._project_item:
            self._project_item.removeChild(item)

        del self._dataset_items[dataset_id]

    def add_file(self, dataset_id: str, file_path: Path) -> None:
        """Add file as child of dataset.

        Parameters
        ----------
        dataset_id : str
            Parent dataset identifier
        file_path : Path
            File path to add
        """
        if dataset_id not in self._dataset_items:
            return

        dataset_item = self._dataset_items[dataset_id]

        # Create file item
        file_item = QTreeWidgetItem(
            dataset_item,
            [file_path.name, "File", ""]
        )
        file_item.setData(0, Qt.UserRole, str(file_path))
        file_item.setForeground(0, QBrush(QColor(120, 120, 120)))

    def update_status(self, dataset_id: str, status: str) -> None:
        """Update dataset status indicator.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        status : str
            Status string ('loaded', 'fitted', 'bayesian')
        """
        if dataset_id not in self._dataset_items:
            return

        item = self._dataset_items[dataset_id]
        item.setText(2, status)
        self._set_status_color(item, status)

    def _set_status_color(self, item: QTreeWidgetItem, status: str) -> None:
        """Set status column color based on status.

        Parameters
        ----------
        item : QTreeWidgetItem
            Tree item to update
        status : str
            Status string
        """
        color_map = {
            "loaded": QColor(100, 150, 255),    # Blue
            "fitted": QColor(100, 200, 100),    # Green
            "bayesian": QColor(150, 100, 255),  # Purple
        }

        color = color_map.get(status, QColor(150, 150, 150))
        item.setForeground(2, QBrush(color))

    def get_selected_dataset_id(self) -> str | None:
        """Get currently selected dataset ID.

        Returns
        -------
        str | None
            Dataset ID or None if no dataset selected
        """
        selected_items = self.selectedItems()
        if not selected_items:
            return None

        item = selected_items[0]

        # Check if item is a dataset (has dataset_id in UserRole)
        dataset_id = item.data(0, Qt.UserRole)

        # If it's a file item, get parent dataset
        if item.parent() and item.parent() != self._project_item:
            parent_item = item.parent()
            dataset_id = parent_item.data(0, Qt.UserRole)

        return dataset_id if isinstance(dataset_id, str) and dataset_id else None

    def get_selected_file_path(self) -> Path | None:
        """Get currently selected file path.

        Returns
        -------
        Path | None
            File path or None if no file selected
        """
        selected_items = self.selectedItems()
        if not selected_items:
            return None

        item = selected_items[0]

        # Check if item is a file (has path in UserRole and parent is dataset)
        if item.parent() and item.parent() != self._project_item:
            file_path_str = item.data(0, Qt.UserRole)
            if file_path_str:
                return Path(file_path_str)

        return None

    def _on_selection_changed(self) -> None:
        """Handle item selection change."""
        dataset_id = self.get_selected_dataset_id()
        file_path = self.get_selected_file_path()

        if file_path:
            # File selected
            self.file_selected.emit(dataset_id or "", file_path)
        elif dataset_id:
            # Dataset selected
            self.dataset_selected.emit(dataset_id)

    def _show_context_menu(self, position: QPoint) -> None:
        """Show context menu for selected item.

        Parameters
        ----------
        position : QPoint
            Menu position
        """
        item = self.itemAt(position)
        if not item:
            return

        menu = QMenu(self)

        # Determine item type
        is_project = item == self._project_item
        is_dataset = item.parent() == self._project_item
        is_file = item.parent() and item.parent() != self._project_item

        if is_project:
            menu.addAction("Add Dataset...")
            menu.addAction("Import Folder...")
            menu.addSeparator()
            menu.addAction("Expand All")
            menu.addAction("Collapse All")
        elif is_dataset:
            menu.addAction("Rename...")
            menu.addAction("Duplicate")
            menu.addSeparator()
            menu.addAction("Export Data...")
            menu.addAction("Export Parameters...")
            menu.addSeparator()
            menu.addAction("Remove")
        elif is_file:
            menu.addAction("Open in External Editor")
            menu.addAction("Show in Folder")
            menu.addSeparator()
            menu.addAction("Remove from Dataset")

        # Show menu and emit signal
        action = menu.exec(self.viewport().mapToGlobal(position))
        if action:
            self.context_menu_requested.emit(position)
