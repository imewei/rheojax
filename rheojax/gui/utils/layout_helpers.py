"""Layout helper functions for consistent margins and spacing.

Provides reusable helpers that apply design-token-based margins and spacing
to Qt layouts, widgets, and controls, eliminating magic numbers scattered
across page and widget implementations.

All spacing values are sourced from :class:`rheojax.gui.resources.styles.tokens.Spacing`.
"""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLayout, QSplitter, QWidget

from rheojax.gui.resources.styles.tokens import Spacing

__all__ = [
    "apply_group_box_style",
    "set_compact_margins",
    "set_page_margins",
    "set_panel_margins",
    "set_splitter_sizes_equal",
    "set_toolbar_margins",
    "set_zero_margins",
]


def set_page_margins(widget: QWidget) -> None:
    """Apply standard page-level margins (PAGE_MARGIN on all sides).

    Use for top-level page content layouts.  The widget must already have a
    layout assigned; if no layout is present the call is a no-op.

    Parameters
    ----------
    widget : QWidget
        The page widget whose layout will be configured.
    """
    layout = widget.layout()
    if layout is not None:
        m = Spacing.PAGE_MARGIN
        layout.setContentsMargins(m, m, m, m)


def set_panel_margins(layout: QLayout, margins: int = Spacing.MD) -> None:
    """Apply uniform margins to a panel or section layout.

    Sets identical margins on all four sides and applies SM (8px) item
    spacing, which suits most sidebar panels and grouped-content sections.

    Parameters
    ----------
    layout : QLayout
        The layout to configure.
    margins : int
        Margin size in pixels applied to all four sides.
        Defaults to ``Spacing.MD`` (12 px).
    """
    layout.setContentsMargins(margins, margins, margins, margins)
    layout.setSpacing(Spacing.SM)


def set_compact_margins(layout: QLayout) -> None:
    """Apply compact margins for dense content areas.

    Sets XS (4 px) margins on all sides with XXS (2 px) inter-item spacing.
    Suitable for toolbar contents, property grids, and other dense panels
    where screen real estate is limited.

    Parameters
    ----------
    layout : QLayout
        The layout to configure.
    """
    layout.setContentsMargins(Spacing.XS, Spacing.XS, Spacing.XS, Spacing.XS)
    layout.setSpacing(Spacing.XXS)


def set_toolbar_margins(layout: QLayout) -> None:
    """Apply toolbar-appropriate margins.

    Uses SM (8 px) on the left/right sides and XS (4 px) on the top/bottom
    to give toolbar buttons visual breathing room without excessive vertical
    height.  Inter-item spacing is XS (4 px).

    Parameters
    ----------
    layout : QLayout
        The layout to configure.
    """
    layout.setContentsMargins(Spacing.SM, Spacing.XS, Spacing.SM, Spacing.XS)
    layout.setSpacing(Spacing.XS)


def set_zero_margins(layout: QLayout) -> None:
    """Remove all margins from a layout.

    Use for container layouts where the parent widget already supplies the
    required visual padding, avoiding double-margin artefacts.

    Parameters
    ----------
    layout : QLayout
        The layout to configure.
    """
    layout.setContentsMargins(0, 0, 0, 0)


def apply_group_box_style(group_box: QGroupBox, variant: str = "panel") -> None:
    """Apply a QGroupBox style variant via a QSS property selector.

    Sets the custom ``gbStyle`` property on *group_box* and triggers a
    QSS re-evaluation pass so that stylesheet rules of the form::

        QGroupBox[gbStyle="card"] { ... }

    take effect immediately without reloading the application stylesheet.

    Parameters
    ----------
    group_box : QGroupBox
        The group box widget to style.
    variant : str
        One of the following named variants:

        ``"panel"`` (default)
            Standard bordered panel used for most content groups.
        ``"card"``
            Elevated card appearance with a visible background fill.
        ``"minimal"``
            No border; title only.  Suitable for lightweight grouping.

    Raises
    ------
    ValueError
        If *variant* is not one of the recognised values.
    """
    _VALID_VARIANTS = frozenset(("panel", "card", "minimal"))
    if variant not in _VALID_VARIANTS:
        raise ValueError(
            f"Unknown QGroupBox variant: {variant!r}. "
            f"Valid options are: {sorted(_VALID_VARIANTS)}"
        )
    group_box.setProperty("gbStyle", variant)
    style = group_box.style()
    if style is not None:
        style.unpolish(group_box)
        style.polish(group_box)
    group_box.update()


def set_splitter_sizes_equal(splitter: QSplitter) -> None:
    """Set all splitter panes to equal proportional sizes.

    Uses the ``[10000, 10000, ...]`` pattern so that Qt distributes available
    space evenly across every pane.  If the splitter contains no children the
    call is a no-op.

    Parameters
    ----------
    splitter : QSplitter
        The splitter widget to configure.
    """
    count = splitter.count()
    if count > 0:
        splitter.setSizes([10000] * count)
