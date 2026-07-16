"""Reusable theme-aware combo box for RheoJAX."""

from __future__ import annotations

from typing import Any, Literal

from rheojax.gui.compat import QComboBox, QtCore, QtGui, QtWidgets
from rheojax.gui.resources.styles.tokens import themed


class RheoComboBox(QComboBox):
    """QComboBox subclass with safe repopulation, data-keyed selection,
    disabled group headers, and a muted placeholder state.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        placeholder: str = "",
        density: Literal["standard", "compact"] = "standard",
    ) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._setup_placeholder_color()
        if placeholder:
            self.setPlaceholderText(placeholder)
            self.setCurrentIndex(-1)
        self.set_density(density)

    def _setup_placeholder_color(self) -> None:
        palette = self.palette()
        color_role = getattr(QtGui.QPalette.ColorRole, "PlaceholderText", None)
        if color_role is not None:
            palette.setColor(color_role, QtGui.QColor(themed("TEXT_MUTED")))
            self.setPalette(palette)

    def set_items_safely(
        self,
        items: list[str] | list[tuple[str, Any]] | dict[str, Any],
        selected_data: Any = None,
    ) -> None:
        """Clear and repopulate without emitting signals mid-update."""
        was_blocked = self.blockSignals(True)
        try:
            self.clear()
            if isinstance(items, dict):
                pairs = items.items()
            elif isinstance(items, list):
                pairs = (
                    item if isinstance(item, tuple) and len(item) == 2 else (item, item)
                    for item in items
                )
            else:
                raise TypeError("items must be a dict, list of strings, or list of tuples")
            for text, data in pairs:
                self.addItem(str(text), data)

            if selected_data is not None:
                self.set_current_data(selected_data)
            elif self._placeholder:
                self.setCurrentIndex(-1)
        finally:
            self.blockSignals(was_blocked)

    def set_current_data(self, data: Any) -> bool:
        """Select the item whose UserRole data matches. Returns False if not found."""
        if data is None:
            self.setCurrentIndex(-1)
            return True
        for idx in range(self.count()):
            if self.itemData(idx, QtCore.Qt.ItemDataRole.UserRole) == data:
                self.setCurrentIndex(idx)
                return True
        return False

    def current_data(self) -> Any:
        idx = self.currentIndex()
        if idx == -1:
            return None
        return self.itemData(idx, QtCore.Qt.ItemDataRole.UserRole)

    def add_group_header(self, text: str) -> None:
        """Add a bold, muted, disabled category separator that can't be selected."""
        self.addItem(text)
        idx = self.count() - 1
        model = self.model()
        item = model.item(idx) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(False)
            item.setSelectable(False)
            item.setForeground(QtGui.QColor(themed("TEXT_MUTED")))
            font = item.font()
            font.setBold(True)
            item.setFont(font)

    def set_density(self, density: Literal["standard", "compact"]) -> None:
        if density not in ("standard", "compact"):
            raise ValueError("density must be 'standard' or 'compact'")
        self.setProperty("density", density)
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.update()
