"""Style helper functions for property-based widget styling."""

from PySide6.QtWidgets import QWidget


def set_button_variant(button: QWidget, variant: str) -> None:
    """Set a button's visual variant via QSS property.

    Triggers QSS [variant="..."] selectors defined in base.qss.

    Parameters
    ----------
    button : QWidget
        The button widget (QPushButton or QToolButton).
    variant : str
        Visual variant: "primary", "secondary", "success", "error", or "ghost".
    """
    button.setProperty("variant", variant)
    _repolish(button)


def set_density(widget: QWidget, density: str = "normal") -> None:
    """Set density mode on a widget and all its children.

    Triggers QSS [density="compact"] selectors for tighter padding.

    Properties are applied in a single batch with updates disabled to
    avoid per-child unpolish/polish flicker.

    Parameters
    ----------
    widget : QWidget
        The widget (and children) to apply density to.
    density : str
        Density mode: "normal" (default sizing) or "compact" (reduced padding).
    """
    if density not in ("normal", "compact"):
        raise ValueError(f"Unknown density: {density!r}")
    # Batch all property changes with updates disabled to avoid per-child
    # unpolish/polish flicker when panels have many children.
    widget.setUpdatesEnabled(False)
    try:
        widget.setProperty("density", density)
        for child in widget.findChildren(QWidget):
            child.setProperty("density", density)
    finally:
        widget.setUpdatesEnabled(True)
    # Single re-polish pass after all properties are set
    _repolish(widget)
    for child in widget.findChildren(QWidget):
        _repolish(child)


def mark_primary_buttons(parent: QWidget, *object_names: str) -> None:
    """Mark buttons as primary variant by their objectName.

    Convenience function for bulk-setting button variants.

    Parameters
    ----------
    parent : QWidget
        Parent widget containing the buttons.
    *object_names : str
        objectName values of buttons to mark as primary.
    """
    _mark_buttons(parent, "primary", *object_names)


def mark_secondary_buttons(parent: QWidget, *object_names: str) -> None:
    """Mark buttons as secondary variant by their objectName.

    Parameters
    ----------
    parent : QWidget
        Parent widget containing the buttons.
    *object_names : str
        objectName values of buttons to mark as secondary.
    """
    _mark_buttons(parent, "secondary", *object_names)


def _mark_buttons(parent: QWidget, variant: str, *object_names: str) -> None:
    """Internal: find buttons by objectName and set their variant."""
    for name in object_names:
        child = parent.findChild(QWidget, name)
        if child is not None:
            set_button_variant(child, variant)


def _repolish(widget: QWidget) -> None:
    """Force QSS re-evaluation after property change."""
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


__all__ = [
    "set_button_variant",
    "set_density",
    "mark_primary_buttons",
    "mark_secondary_buttons",
]
