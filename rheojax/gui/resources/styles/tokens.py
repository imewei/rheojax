"""
RheoJAX GUI Design Tokens.

Centralized design system constants for consistent styling across the application.
These tokens complement the QSS stylesheets and can be used programmatically.

Design direction: "Precision Laboratory"
- Warm stone neutrals (not cold slate) evoke lab notebook paper
- Deep, rich primary blue with subtle gradient accents
- Refined typography with geometric sans-serif priority
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for the RheoJAX GUI.

    Uses warm stone neutrals with a deep scientific blue primary.
    The warmth of the neutral tones contrasts with the precision of
    the blue and purple accents — like a well-lit materials lab.
    """

    # Primary colors (deep scientific blue)
    PRIMARY: ClassVar[str] = "#1D4ED8"
    PRIMARY_HOVER: ClassVar[str] = "#1E40AF"
    PRIMARY_PRESSED: ClassVar[str] = "#1E3A8A"
    PRIMARY_LIGHT: ClassVar[str] = "#DBEAFE"
    PRIMARY_SUBTLE: ClassVar[str] = "#EFF6FF"

    # Accent colors (Bayesian/scientific purple)
    ACCENT: ClassVar[str] = "#7C3AED"
    ACCENT_HOVER: ClassVar[str] = "#6D28D9"
    ACCENT_PRESSED: ClassVar[str] = "#5B21B6"
    ACCENT_LIGHT: ClassVar[str] = "#EDE9FE"

    # Semantic colors
    SUCCESS: ClassVar[str] = "#059669"
    SUCCESS_HOVER: ClassVar[str] = "#047857"
    SUCCESS_LIGHT: ClassVar[str] = "#D1FAE5"

    WARNING: ClassVar[str] = "#D97706"
    WARNING_HOVER: ClassVar[str] = "#B45309"
    WARNING_LIGHT: ClassVar[str] = "#FEF3C7"

    ERROR: ClassVar[str] = "#DC2626"
    ERROR_HOVER: ClassVar[str] = "#B91C1C"
    ERROR_LIGHT: ClassVar[str] = "#FEE2E2"

    INFO: ClassVar[str] = "#2563EB"
    INFO_LIGHT: ClassVar[str] = "#DBEAFE"

    # Background colors (warm stone tones)
    BG_BASE: ClassVar[str] = "#FFFFFF"
    BG_SURFACE: ClassVar[str] = "#FAFAF9"
    BG_ELEVATED: ClassVar[str] = "#FFFFFF"
    BG_HOVER: ClassVar[str] = "#F5F5F4"
    BG_ACTIVE: ClassVar[str] = "#E7E5E4"
    BG_CANVAS: ClassVar[str] = "#F5F5F4"

    # Text colors (warm charcoal)
    TEXT_PRIMARY: ClassVar[str] = "#1C1917"
    TEXT_SECONDARY: ClassVar[str] = "#57534E"
    TEXT_MUTED: ClassVar[str] = "#78716C"
    TEXT_DISABLED: ClassVar[str] = "#A8A29E"
    TEXT_INVERSE: ClassVar[str] = "#FFFFFF"

    # Border colors (warm stone)
    BORDER_DEFAULT: ClassVar[str] = "#D6D3D1"
    BORDER_HOVER: ClassVar[str] = "#A8A29E"
    BORDER_FOCUS: ClassVar[str] = "#1D4ED8"
    BORDER_SUBTLE: ClassVar[str] = "#E7E5E4"

    # Chart/visualization colors (refined, high-contrast set)
    CHART_1: ClassVar[str] = "#1D4ED8"  # Deep blue
    CHART_2: ClassVar[str] = "#7C3AED"  # Purple
    CHART_3: ClassVar[str] = "#059669"  # Emerald
    CHART_4: ClassVar[str] = "#D97706"  # Amber
    CHART_5: ClassVar[str] = "#DC2626"  # Red
    CHART_6: ClassVar[str] = "#DB2777"  # Pink


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
    INPUT_PADDING_H: ClassVar[int] = 12
    INPUT_PADDING_V: ClassVar[int] = 8
    CARD_PADDING: ClassVar[int] = 20
    SECTION_GAP: ClassVar[int] = 24
    PAGE_MARGIN: ClassVar[int] = 20


@dataclass(frozen=True)
class BorderRadius:
    """Border radius scale for consistent rounding."""

    NONE: ClassVar[int] = 0
    SM: ClassVar[int] = 4  # Small inputs, badges
    MD: ClassVar[int] = 6  # Buttons, inputs
    LG: ClassVar[int] = 8  # Cards, panels
    XL: ClassVar[int] = 12  # Large cards, dialogs
    FULL: ClassVar[int] = 9999  # Circular/pill


@dataclass(frozen=True)
class Typography:
    """Typography scale for consistent text sizing.

    Prioritizes modern geometric sans-serifs that convey precision,
    falling back to high-quality system fonts on each platform.
    """

    # Font families — geometric sans for scientific precision
    FONT_FAMILY: ClassVar[str] = (
        '"Plus Jakarta Sans", "DM Sans", "Geist", '
        '"SF Pro Display", "SF Pro Text", '
        '"Segoe UI Variable", "Segoe UI", '
        '"Cantarell", "Helvetica Neue", "Helvetica", sans-serif'
    )
    FONT_FAMILY_MONO: ClassVar[str] = (
        '"JetBrains Mono", "Cascadia Code", '
        '"SF Mono", "Menlo", "Consolas", '
        '"DejaVu Sans Mono", monospace'
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
    SHADOW_SM: ClassVar[tuple[int, int, int, str]] = (0, 1, 3, "rgba(28, 25, 23, 0.06)")
    SHADOW_MD: ClassVar[tuple[int, int, int, str]] = (0, 4, 8, "rgba(28, 25, 23, 0.08)")
    SHADOW_LG: ClassVar[tuple[int, int, int, str]] = (
        0,
        10,
        20,
        "rgba(28, 25, 23, 0.10)",
    )


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
            "text": colors.TEXT_INVERSE,
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

    return f"""
        background-color: {colors.BG_ELEVATED};
        border: 1px solid {colors.BORDER_DEFAULT};
        border-radius: {radius.LG}px;
        padding: {spacing.CARD_PADDING}px;
    """


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
        border-bottom: 2px solid {colors.BORDER_SUBTLE};
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
