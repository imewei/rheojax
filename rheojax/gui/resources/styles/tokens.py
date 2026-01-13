"""
RheoJAX GUI Design Tokens.

Centralized design system constants for consistent styling across the application.
These tokens complement the QSS stylesheets and can be used programmatically.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for the RheoJAX GUI.

    Based on the UI Pro Max recommendations for scientific dashboard applications.
    Uses a professional color scheme with trust blue as primary.
    """

    # Primary colors (JAX-inspired blue)
    PRIMARY: ClassVar[str] = "#2563EB"
    PRIMARY_HOVER: ClassVar[str] = "#1D4ED8"
    PRIMARY_PRESSED: ClassVar[str] = "#1E40AF"
    PRIMARY_LIGHT: ClassVar[str] = "#DBEAFE"

    # Accent colors (Bayesian/scientific purple)
    ACCENT: ClassVar[str] = "#7C3AED"
    ACCENT_HOVER: ClassVar[str] = "#6D28D9"
    ACCENT_PRESSED: ClassVar[str] = "#5B21B6"
    ACCENT_LIGHT: ClassVar[str] = "#EDE9FE"

    # Semantic colors
    SUCCESS: ClassVar[str] = "#10B981"
    SUCCESS_HOVER: ClassVar[str] = "#059669"
    SUCCESS_LIGHT: ClassVar[str] = "#D1FAE5"

    WARNING: ClassVar[str] = "#F59E0B"
    WARNING_HOVER: ClassVar[str] = "#D97706"
    WARNING_LIGHT: ClassVar[str] = "#FEF3C7"

    ERROR: ClassVar[str] = "#EF4444"
    ERROR_HOVER: ClassVar[str] = "#DC2626"
    ERROR_LIGHT: ClassVar[str] = "#FEE2E2"

    INFO: ClassVar[str] = "#3B82F6"
    INFO_LIGHT: ClassVar[str] = "#DBEAFE"

    # Background colors
    BG_BASE: ClassVar[str] = "#FFFFFF"
    BG_SURFACE: ClassVar[str] = "#F8FAFC"
    BG_ELEVATED: ClassVar[str] = "#FFFFFF"
    BG_HOVER: ClassVar[str] = "#F1F5F9"
    BG_ACTIVE: ClassVar[str] = "#E2E8F0"

    # Text colors
    TEXT_PRIMARY: ClassVar[str] = "#1E293B"
    TEXT_SECONDARY: ClassVar[str] = "#475569"
    TEXT_MUTED: ClassVar[str] = "#64748B"
    TEXT_DISABLED: ClassVar[str] = "#94A3B8"
    TEXT_INVERSE: ClassVar[str] = "#FFFFFF"

    # Border colors
    BORDER_DEFAULT: ClassVar[str] = "#E2E8F0"
    BORDER_HOVER: ClassVar[str] = "#CBD5E1"
    BORDER_FOCUS: ClassVar[str] = "#2563EB"

    # Chart/visualization colors
    CHART_1: ClassVar[str] = "#2563EB"  # Blue
    CHART_2: ClassVar[str] = "#7C3AED"  # Purple
    CHART_3: ClassVar[str] = "#10B981"  # Green
    CHART_4: ClassVar[str] = "#F59E0B"  # Amber
    CHART_5: ClassVar[str] = "#EF4444"  # Red
    CHART_6: ClassVar[str] = "#EC4899"  # Pink


@dataclass(frozen=True)
class Spacing:
    """Spacing scale for consistent margins and padding.

    Based on a 4px base unit with common multipliers.
    """

    XXS: ClassVar[int] = 2  # 2px
    XS: ClassVar[int] = 4  # 4px
    SM: ClassVar[int] = 8  # 8px
    MD: ClassVar[int] = 12  # 12px
    LG: ClassVar[int] = 16  # 16px
    XL: ClassVar[int] = 24  # 24px
    XXL: ClassVar[int] = 32  # 32px
    XXXL: ClassVar[int] = 48  # 48px

    # Component-specific spacing
    BUTTON_PADDING_H: ClassVar[int] = 16
    BUTTON_PADDING_V: ClassVar[int] = 8
    INPUT_PADDING_H: ClassVar[int] = 10
    INPUT_PADDING_V: ClassVar[int] = 6
    CARD_PADDING: ClassVar[int] = 16
    SECTION_GAP: ClassVar[int] = 24
    PAGE_MARGIN: ClassVar[int] = 16


@dataclass(frozen=True)
class BorderRadius:
    """Border radius scale for consistent rounding."""

    NONE: ClassVar[int] = 0
    SM: ClassVar[int] = 4  # Buttons, inputs
    MD: ClassVar[int] = 6  # Cards, panels
    LG: ClassVar[int] = 8  # Large cards
    XL: ClassVar[int] = 12  # Dialogs
    FULL: ClassVar[int] = 9999  # Circular/pill


@dataclass(frozen=True)
class Typography:
    """Typography scale for consistent text sizing.

    Uses a system font stack optimized for cross-platform rendering.
    """

    # Font families
    FONT_FAMILY: ClassVar[str] = (
        '"Segoe UI", "SF Pro Text", "SF Pro Display", '
        '"Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"'
    )
    FONT_FAMILY_MONO: ClassVar[str] = (
        '"SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", monospace'
    )

    # Font sizes (in pt)
    SIZE_XS: ClassVar[int] = 9
    SIZE_SM: ClassVar[int] = 10
    SIZE_BASE: ClassVar[int] = 12
    SIZE_MD: ClassVar[int] = 13
    SIZE_LG: ClassVar[int] = 14
    SIZE_XL: ClassVar[int] = 16
    SIZE_XXL: ClassVar[int] = 20
    SIZE_HEADING: ClassVar[int] = 24

    # Font weights
    WEIGHT_NORMAL: ClassVar[int] = 400
    WEIGHT_MEDIUM: ClassVar[int] = 500
    WEIGHT_SEMIBOLD: ClassVar[int] = 600
    WEIGHT_BOLD: ClassVar[int] = 700


@dataclass(frozen=True)
class Shadows:
    """Shadow definitions for elevation effects."""

    # Box shadows (for use with QGraphicsDropShadowEffect)
    SHADOW_SM: ClassVar[tuple[int, int, int, str]] = (0, 1, 3, "rgba(0, 0, 0, 0.1)")
    SHADOW_MD: ClassVar[tuple[int, int, int, str]] = (0, 4, 6, "rgba(0, 0, 0, 0.1)")
    SHADOW_LG: ClassVar[tuple[int, int, int, str]] = (0, 10, 15, "rgba(0, 0, 0, 0.1)")


# Convenience class combining all tokens
class DesignTokens:
    """Combined access to all design tokens."""

    colors = ColorPalette
    spacing = Spacing
    radius = BorderRadius
    typography = Typography
    shadows = Shadows


# Helper functions for generating style strings
def button_style(
    variant: str = "primary",
    size: str = "md",
) -> str:
    """Generate button style string for inline use.

    Parameters
    ----------
    variant : str
        Button variant: primary, secondary, accent, success, warning, error
    size : str
        Button size: sm, md, lg

    Returns
    -------
    str
        QSS style string for the button
    """
    colors = ColorPalette
    spacing = Spacing
    radius = BorderRadius

    # Size mappings
    sizes = {
        "sm": {"padding": f"{spacing.XS}px {spacing.SM}px", "font_size": "10pt"},
        "md": {"padding": f"{spacing.SM}px {spacing.LG}px", "font_size": "11pt"},
        "lg": {"padding": f"{spacing.MD}px {spacing.XL}px", "font_size": "12pt"},
    }

    # Variant mappings
    variants = {
        "primary": {
            "bg": colors.PRIMARY,
            "bg_hover": colors.PRIMARY_HOVER,
            "text": colors.TEXT_INVERSE,
            "border": "none",
        },
        "secondary": {
            "bg": colors.BG_SURFACE,
            "bg_hover": colors.BG_HOVER,
            "text": colors.TEXT_PRIMARY,
            "border": f"1px solid {colors.BORDER_DEFAULT}",
        },
        "accent": {
            "bg": colors.ACCENT,
            "bg_hover": colors.ACCENT_HOVER,
            "text": colors.TEXT_INVERSE,
            "border": "none",
        },
        "success": {
            "bg": colors.SUCCESS,
            "bg_hover": colors.SUCCESS_HOVER,
            "text": colors.TEXT_INVERSE,
            "border": "none",
        },
        "warning": {
            "bg": colors.WARNING,
            "bg_hover": colors.WARNING_HOVER,
            "text": colors.TEXT_PRIMARY,
            "border": "none",
        },
        "error": {
            "bg": colors.ERROR,
            "bg_hover": colors.ERROR_HOVER,
            "text": colors.TEXT_INVERSE,
            "border": "none",
        },
        "ghost": {
            "bg": "transparent",
            "bg_hover": colors.BG_HOVER,
            "text": colors.TEXT_PRIMARY,
            "border": "none",
        },
    }

    s = sizes.get(size, sizes["md"])
    v = variants.get(variant, variants["primary"])

    return f"""
        QPushButton {{
            background-color: {v['bg']};
            color: {v['text']};
            border: {v['border']};
            border-radius: {radius.MD}px;
            padding: {s['padding']};
            font-size: {s['font_size']};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {v['bg_hover']};
        }}
        QPushButton:disabled {{
            background-color: {colors.BG_ACTIVE};
            color: {colors.TEXT_DISABLED};
        }}
    """


def card_style(elevated: bool = False) -> str:
    """Generate card/panel style string.

    Parameters
    ----------
    elevated : bool
        Whether to show elevated shadow effect

    Returns
    -------
    str
        QSS style string for the card
    """
    colors = ColorPalette
    spacing = Spacing
    radius = BorderRadius

    base = f"""
        background-color: {colors.BG_ELEVATED};
        border: 1px solid {colors.BORDER_DEFAULT};
        border-radius: {radius.LG}px;
        padding: {spacing.CARD_PADDING}px;
    """

    return base


def status_badge_style(status: str) -> str:
    """Generate status badge style string.

    Parameters
    ----------
    status : str
        Status type: success, warning, error, info, pending

    Returns
    -------
    str
        QSS style string for the badge
    """
    colors = ColorPalette
    radius = BorderRadius

    status_colors = {
        "success": (colors.SUCCESS, colors.SUCCESS_LIGHT),
        "warning": (colors.WARNING, colors.WARNING_LIGHT),
        "error": (colors.ERROR, colors.ERROR_LIGHT),
        "info": (colors.INFO, colors.INFO_LIGHT),
        "pending": (colors.TEXT_MUTED, colors.BG_HOVER),
    }

    fg, bg = status_colors.get(status, status_colors["pending"])

    return f"""
        background-color: {bg};
        color: {fg};
        border-radius: {radius.SM}px;
        padding: 2px 8px;
        font-size: 10pt;
        font-weight: 500;
    """


def section_header_style() -> str:
    """Generate section header style string."""
    colors = ColorPalette
    typography = Typography

    return f"""
        color: {colors.TEXT_PRIMARY};
        font-size: {typography.SIZE_LG}pt;
        font-weight: {typography.WEIGHT_SEMIBOLD};
        padding-bottom: 8px;
        border-bottom: 1px solid {colors.BORDER_DEFAULT};
        margin-bottom: 12px;
    """


def empty_state_style() -> str:
    """Generate empty state placeholder style string."""
    colors = ColorPalette
    spacing = Spacing

    return f"""
        color: {colors.TEXT_MUTED};
        padding: {spacing.XL}px;
        font-size: 11pt;
    """


# Export all tokens and helpers
__all__ = [
    "ColorPalette",
    "Spacing",
    "BorderRadius",
    "Typography",
    "Shadows",
    "DesignTokens",
    "button_style",
    "card_style",
    "status_badge_style",
    "section_header_style",
    "empty_state_style",
]
