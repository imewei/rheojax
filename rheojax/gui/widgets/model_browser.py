"""
Model Browser Widget
===================

Searchable model library with categories and model information display.
"""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.utils.icons import IconProvider

# Initialize icon provider (uses ASCII icons, safe on all platforms)
_icon_provider = IconProvider(allow_emoji=False)

# Model category display names and descriptions
# Icons are provided by IconProvider to ensure platform safety
# (emoji causes CoreText/ImageIO crashes on macOS)
CATEGORY_INFO = {
    "classical": {
        "name": "Classical Models",
        "description": "Standard viscoelastic models (Maxwell, Zener, SpringPot)",
    },
    "fractional_maxwell": {
        "name": "Fractional Maxwell",
        "description": "Fractional Maxwell variants for gel and liquid behavior",
    },
    "fractional_zener": {
        "name": "Fractional Zener",
        "description": "Fractional Zener models for solid-liquid transitions",
    },
    "fractional_advanced": {
        "name": "Fractional Advanced",
        "description": "Complex fractional models (Burgers, Poynting-Thomson, Jeffreys)",
    },
    "flow": {
        "name": "Flow Models",
        "description": "Non-Newtonian flow models (Power Law, Carreau, Herschel-Bulkley)",
    },
    "multi_mode": {
        "name": "Multi-Mode",
        "description": "Generalized Maxwell with automatic mode selection",
    },
    "sgr": {
        "name": "Soft Glassy Rheology",
        "description": "SGR models for foams, emulsions, and colloidal suspensions",
    },
    "other": {
        "name": "Other Models",
        "description": "Additional rheological models",
    },
}


def get_category_icon(category: str) -> str:
    """Get platform-safe icon for a category.

    Parameters
    ----------
    category : str
        Category name

    Returns
    -------
    str
        ASCII icon string (safe on all platforms including macOS)
    """
    return _icon_provider.get_category_icon(category)

# Model display names
MODEL_DISPLAY_NAMES = {
    "maxwell": "Maxwell",
    "zener": "Zener (Standard Linear Solid)",
    "springpot": "SpringPot",
    "fractional_maxwell_gel": "Fractional Maxwell Gel",
    "fractional_maxwell_liquid": "Fractional Maxwell Liquid",
    "fractional_maxwell_model": "Fractional Maxwell Model",
    "fractional_kelvin_voigt": "Fractional Kelvin-Voigt",
    "fractional_zener_sl": "Fractional Zener Solid-Liquid",
    "fractional_zener_ss": "Fractional Zener Solid-Solid",
    "fractional_zener_ll": "Fractional Zener Liquid-Liquid",
    "fractional_kv_zener": "Fractional Kelvin-Voigt Zener",
    "fractional_burgers": "Fractional Burgers",
    "fractional_poynting_thomson": "Fractional Poynting-Thomson",
    "fractional_jeffreys": "Fractional Jeffreys",
    "power_law": "Power Law",
    "carreau": "Carreau",
    "carreau_yasuda": "Carreau-Yasuda",
    "cross": "Cross",
    "herschel_bulkley": "Herschel-Bulkley",
    "bingham": "Bingham",
    "generalized_maxwell": "Generalized Maxwell",
    "sgr_conventional": "SGR Conventional",
    "sgr_generic": "SGR GENERIC",
}


class ModelBrowser(QWidget):
    """Model browser with search, filter, and information display.

    Features:
        - Category filtering with tree view
        - Full-text search by name and description
        - Model metadata display (parameters, bounds, equation)
        - Test mode compatibility indicators
        - Selection signal for integration

    Signals
    -------
    model_selected : Signal(str)
        Emitted when a model is selected (model_name)
    model_double_clicked : Signal(str)
        Emitted when a model is double-clicked (model_name)

    Example
    -------
    >>> browser = ModelBrowser()  # doctest: +SKIP
    >>> browser.model_selected.connect(on_model_selected)  # doctest: +SKIP
    >>> browser.set_models(model_service.get_available_models())  # doctest: +SKIP
    """

    model_selected = Signal(str)
    model_double_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize model browser.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._models: dict[str, list[str]] = {}
        self._model_info: dict[str, dict[str, Any]] = {}
        self._selected_model: str | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Filter models...")
        self._search_input.setClearButtonEnabled(True)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self._search_input)
        layout.addLayout(search_layout)

        # Splitter for tree and info panel
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Model tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setAnimated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setMinimumHeight(200)
        splitter.addWidget(self._tree)

        # Info panel
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(8)

        # Model name header
        self._model_name_label = QLabel("Select a model")
        self._model_name_label.setFont(QFont("", -1, QFont.Weight.Bold))
        info_layout.addWidget(self._model_name_label)

        # Description
        self._description_label = QLabel("")
        self._description_label.setWordWrap(True)
        self._description_label.setMaximumHeight(60)
        info_layout.addWidget(self._description_label)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        self._params_text = QTextEdit()
        self._params_text.setReadOnly(True)
        self._params_text.setMaximumHeight(150)
        params_layout.addWidget(self._params_text)
        info_layout.addWidget(params_group)

        # Test modes
        modes_layout = QHBoxLayout()
        modes_layout.addWidget(QLabel("Supported modes:"))
        self._modes_label = QLabel("—")
        self._modes_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        modes_layout.addWidget(self._modes_label)
        info_layout.addLayout(modes_layout)

        # Wrap info in scroll area
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setWidget(info_frame)
        info_scroll.setMinimumHeight(150)
        splitter.addWidget(info_scroll)

        # Set splitter proportions
        splitter.setSizes([300, 200])

        layout.addWidget(splitter, 1)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._search_input.textChanged.connect(self._filter_models)
        self._tree.currentItemChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_double_click)

    def set_models(self, models: dict[str, list[str]]) -> None:
        """Set available models grouped by category.

        Parameters
        ----------
        models : dict[str, list[str]]
            Models grouped by category name
        """
        self._models = models
        self._rebuild_tree()

    def set_model_info_callback(
        self, callback: "Callable[[str], dict[str, Any]]"
    ) -> None:
        """Set callback for fetching model information.

        Parameters
        ----------
        callback : Callable[[str], dict]
            Function that takes model_name and returns model info dict
        """
        self._get_model_info = callback

    def _rebuild_tree(self) -> None:
        """Rebuild the model tree from scratch."""
        self._tree.clear()

        for category, model_names in self._models.items():
            if not model_names:
                continue

            # Create category item
            cat_info = CATEGORY_INFO.get(category, CATEGORY_INFO["other"])
            cat_icon = get_category_icon(category)
            cat_item = QTreeWidgetItem()
            cat_item.setText(0, f"{cat_icon} {cat_info['name']}")
            cat_item.setToolTip(0, cat_info["description"])
            cat_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "category", "name": category})
            cat_item.setFlags(
                cat_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
            )  # Category not selectable

            # Add model items
            for model_name in sorted(model_names):
                display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())
                model_item = QTreeWidgetItem()
                model_item.setText(0, display_name)
                model_item.setData(
                    0, Qt.ItemDataRole.UserRole, {"type": "model", "name": model_name}
                )
                model_item.setToolTip(0, f"Model: {model_name}")
                cat_item.addChild(model_item)

            self._tree.addTopLevelItem(cat_item)
            cat_item.setExpanded(True)

    def _filter_models(self, text: str) -> None:
        """Filter models based on search text.

        Parameters
        ----------
        text : str
            Search text
        """
        text = text.lower().strip()

        for cat_idx in range(self._tree.topLevelItemCount()):
            cat_item = self._tree.topLevelItem(cat_idx)
            cat_visible = False

            for model_idx in range(cat_item.childCount()):
                model_item = cat_item.child(model_idx)
                data = model_item.data(0, Qt.ItemDataRole.UserRole)
                model_name = data.get("name", "")
                display_text = model_item.text(0)

                # Match against model name and display text
                visible = (
                    not text
                    or text in model_name.lower()
                    or text in display_text.lower()
                )
                model_item.setHidden(not visible)

                if visible:
                    cat_visible = True

            # Hide category if no visible children
            cat_item.setHidden(not cat_visible)

    def _on_selection_changed(
        self, current: QTreeWidgetItem | None, previous: QTreeWidgetItem | None
    ) -> None:
        """Handle selection change in tree.

        Parameters
        ----------
        current : QTreeWidgetItem, optional
            Current selected item
        previous : QTreeWidgetItem, optional
            Previous selected item
        """
        if current is None:
            return

        data = current.data(0, Qt.ItemDataRole.UserRole)
        if data is None or data.get("type") != "model":
            return

        model_name = data.get("name")
        self._selected_model = model_name
        self._update_info_panel(model_name)
        self.model_selected.emit(model_name)

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle double-click on tree item.

        Parameters
        ----------
        item : QTreeWidgetItem
            Clicked item
        column : int
            Clicked column
        """
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None or data.get("type") != "model":
            return

        model_name = data.get("name")
        self.model_double_clicked.emit(model_name)

    def _update_info_panel(self, model_name: str) -> None:
        """Update the info panel with model details.

        Parameters
        ----------
        model_name : str
            Name of the model
        """
        # Get display name
        display_name = MODEL_DISPLAY_NAMES.get(
            model_name, model_name.replace("_", " ").title()
        )
        self._model_name_label.setText(display_name)

        # Try to get model info
        info = self._model_info.get(model_name)
        if info is None and hasattr(self, "_get_model_info"):
            try:
                info = self._get_model_info(model_name)
                self._model_info[model_name] = info
            except Exception:
                info = None

        if info:
            # Description
            desc = info.get("description", "")
            if len(desc) > 200:
                desc = desc[:200] + "..."
            self._description_label.setText(desc)

            # Parameters
            params = info.get("parameters", {})
            params_text = ""
            for name, details in params.items():
                default = details.get("default", "?")
                bounds = details.get("bounds", (None, None))
                units = details.get("units", "")
                params_text += f"• {name}: {default}"
                if bounds[0] is not None or bounds[1] is not None:
                    params_text += f" [{bounds[0]}, {bounds[1]}]"
                if units:
                    params_text += f" ({units})"
                params_text += "\n"
            self._params_text.setText(params_text.strip() or "No parameters")

            # Test modes
            modes = info.get("supported_test_modes", [])
            self._modes_label.setText(", ".join(modes) if modes else "Unknown")
        else:
            self._description_label.setText("Model information not available")
            self._params_text.setText("")
            self._modes_label.setText("—")

    def get_selected(self) -> str | None:
        """Get currently selected model name.

        Returns
        -------
        str or None
            Selected model name, or None if no selection
        """
        return self._selected_model

    def select_model(self, model_name: str) -> bool:
        """Programmatically select a model.

        Parameters
        ----------
        model_name : str
            Name of model to select

        Returns
        -------
        bool
            True if model was found and selected
        """
        for cat_idx in range(self._tree.topLevelItemCount()):
            cat_item = self._tree.topLevelItem(cat_idx)

            for model_idx in range(cat_item.childCount()):
                model_item = cat_item.child(model_idx)
                data = model_item.data(0, Qt.ItemDataRole.UserRole)

                if data and data.get("name") == model_name:
                    self._tree.setCurrentItem(model_item)
                    return True

        return False

    def expand_all(self) -> None:
        """Expand all categories."""
        self._tree.expandAll()

    def collapse_all(self) -> None:
        """Collapse all categories."""
        self._tree.collapseAll()

    def clear_selection(self) -> None:
        """Clear current selection."""
        self._tree.clearSelection()
        self._selected_model = None
        self._model_name_label.setText("Select a model")
        self._description_label.setText("")
        self._params_text.setText("")
        self._modes_label.setText("—")
