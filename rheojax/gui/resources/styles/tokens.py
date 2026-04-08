"""
RheoJAX GUI Design Tokens.

Centralized design system constants for consistent styling across the application.
These tokens complement the QSS stylesheets and can be used programmatically.

Design direction: "Precision Laboratory" (v0.6.0+)
- Warm stone neutrals for surfaces and text
- Deep indigo primary (#4338CA) with violet accents
- Inter font family with system fallbacks
- Palette synchronized with light.qss / dark.qss
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for the RheoJAX GUI (light theme).

    Uses warm stone neutrals with deep indigo primary (synced with QSS).
    The warmth of the neutral tones contrasts with the precision of
    the indigo and violet accents.
    """

    # Primary colors (deep indigo, synced with light.qss QPushButton gradient)
    PRIMARY: ClassVar[str] = "#4338CA"  # indigo-700
    PRIMARY_HOVER: ClassVar[str] = "#4F46E5"  # indigo-600
    PRIMARY_PRESSED: ClassVar[str] = "#3730A3"  # indigo-800
    PRIMARY_LIGHT: ClassVar[str] = "#E0E7FF"  # indigo-100
    PRIMARY_SUBTLE: ClassVar[str] = "#EEF2FF"  # indigo-50

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

    # Bright variants (for gradient stops)
    SUCCESS_BRIGHT: ClassVar[str] = "#10B981"  # emerald-500
    WARNING_BRIGHT: ClassVar[str] = "#F59E0B"  # amber-500
    ERROR_BRIGHT: ClassVar[str] = "#EF4444"  # red-500
    ACCENT_BRIGHT: ClassVar[str] = "#8B5CF6"  # violet-500

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
    BORDER_FOCUS: ClassVar[str] = "#6366F1"  # indigo-500, matches QSS focus ring
    BORDER_SUBTLE: ClassVar[str] = "#E7E5E4"

    # Chart/visualization colors (refined, high-contrast set)
    CHART_1: ClassVar[str] = "#4338CA"  # Indigo (matches primary)
    CHART_2: ClassVar[str] = "#7C3AED"  # Purple
    CHART_3: ClassVar[str] = "#059669"  # Emerald
    CHART_4: ClassVar[str] = "#D97706"  # Amber
    CHART_5: ClassVar[str] = "#DC2626"  # Red
    CHART_6: ClassVar[str] = "#DB2777"  # Pink


@dataclass(frozen=True)
class DarkColorPalette:
    """Dark theme color palette (values derived from dark.qss).

    Mirrors the same attribute names as ColorPalette so that
    ThemeManager.get_palette() can be used interchangeably.
    """

    # Primary colors (brighter indigo for dark backgrounds)
    PRIMARY: ClassVar[str] = "#6366F1"  # indigo-500
    PRIMARY_HOVER: ClassVar[str] = "#818CF8"  # indigo-400
    PRIMARY_PRESSED: ClassVar[str] = "#4F46E5"  # indigo-600
    PRIMARY_LIGHT: ClassVar[str] = "#312E81"  # indigo-900
    PRIMARY_SUBTLE: ClassVar[str] = "#1E1B4B"  # indigo-950

    # Accent colors
    ACCENT: ClassVar[str] = "#A78BFA"  # violet-400
    ACCENT_HOVER: ClassVar[str] = "#8B5CF6"
    ACCENT_PRESSED: ClassVar[str] = "#7C3AED"
    ACCENT_LIGHT: ClassVar[str] = "#312E81"

    # Semantic colors (brighter for dark backgrounds)
    SUCCESS: ClassVar[str] = "#34D399"  # emerald-400
    SUCCESS_HOVER: ClassVar[str] = "#10B981"
    SUCCESS_LIGHT: ClassVar[str] = "#064E3B"

    WARNING: ClassVar[str] = "#FBBF24"  # amber-400
    WARNING_HOVER: ClassVar[str] = "#F59E0B"
    WARNING_LIGHT: ClassVar[str] = "#78350F"

    ERROR: ClassVar[str] = "#F87171"  # red-400
    ERROR_HOVER: ClassVar[str] = "#EF4444"
    ERROR_LIGHT: ClassVar[str] = "#7F1D1D"

    INFO: ClassVar[str] = "#60A5FA"
    INFO_LIGHT: ClassVar[str] = "#1E3A5F"

    # Bright variants (for gradient stops — same as light, high-contrast on any bg)
    SUCCESS_BRIGHT: ClassVar[str] = "#10B981"
    WARNING_BRIGHT: ClassVar[str] = "#F59E0B"
    ERROR_BRIGHT: ClassVar[str] = "#EF4444"
    ACCENT_BRIGHT: ClassVar[str] = "#8B5CF6"

    # Background colors (slate dark)
    BG_BASE: ClassVar[str] = "#0F172A"  # slate-900
    BG_SURFACE: ClassVar[str] = "#1E293B"  # slate-800
    BG_ELEVATED: ClassVar[str] = "#1E293B"
    BG_HOVER: ClassVar[str] = "#334155"  # slate-700
    BG_ACTIVE: ClassVar[str] = "#475569"  # slate-600
    BG_CANVAS: ClassVar[str] = "#020617"  # slate-950

    # Text colors (light for dark backgrounds)
    TEXT_PRIMARY: ClassVar[str] = "#F8FAFC"  # slate-50
    TEXT_SECONDARY: ClassVar[str] = "#94A3B8"  # slate-400
    TEXT_MUTED: ClassVar[str] = "#64748B"  # slate-500
    TEXT_DISABLED: ClassVar[str] = "#475569"  # slate-600
    TEXT_INVERSE: ClassVar[str] = "#0F172A"

    # Border colors (slate)
    BORDER_DEFAULT: ClassVar[str] = "#334155"
    BORDER_HOVER: ClassVar[str] = "#475569"
    BORDER_FOCUS: ClassVar[str] = "#6366F1"
    BORDER_SUBTLE: ClassVar[str] = "#1E293B"

    # Chart colors (brighter for dark backgrounds)
    CHART_1: ClassVar[str] = "#818CF8"  # indigo-400
    CHART_2: ClassVar[str] = "#A78BFA"  # violet-400
    CHART_3: ClassVar[str] = "#34D399"  # emerald-400
    CHART_4: ClassVar[str] = "#FBBF24"  # amber-400
    CHART_5: ClassVar[str] = "#F87171"  # red-400
    CHART_6: ClassVar[str] = "#F472B6"  # pink-400


class ThemeManager:
    """Tracks active theme for programmatic style lookups.

    Provides a class-level singleton that style helpers consult
    to return theme-appropriate colors at call time.
    """

    _theme: ClassVar[str] = "light"

    @classmethod
    def set_theme(cls, theme: str) -> None:
        """Set the active theme ('light' or 'dark')."""
        cls._theme = theme

    @classmethod
    def is_dark(cls) -> bool:
        """Return True if the active theme is dark."""
        return cls._theme == "dark"

    @classmethod
    def get_palette(cls) -> type:
        """Return the active color palette class."""
        return DarkColorPalette if cls._theme == "dark" else ColorPalette


def themed(token_name: str) -> str:
    """Return the theme-appropriate value for a ColorPalette token.

    Parameters
    ----------
    token_name : str
        Attribute name on ColorPalette / DarkColorPalette (e.g. "PRIMARY").

    Returns
    -------
    str
        The hex color string for the active theme.  Falls back to
        ``ColorPalette.TEXT_PRIMARY`` if *token_name* is not found.
    """
    palette = ThemeManager.get_palette()
    value = getattr(palette, token_name, None)
    if value is None:
        import logging

        logging.getLogger(
            __name__
        ).warning(  # nosemgrep: python-logger-credential-disclosure
            "Unknown design token '%s', falling back to TEXT_PRIMARY", token_name
        )
        return palette.TEXT_PRIMARY
    return value


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

    Prioritizes Inter (synced with QSS), falling back to high-quality
    system fonts on each platform.
    """

    # Font families — Inter first (synced with light.qss / dark.qss)
    FONT_FAMILY: ClassVar[str] = (
        '"Inter", -apple-system, BlinkMacSystemFont, '
        '"Segoe UI", "Roboto", "Helvetica Neue", sans-serif'
    )
    FONT_FAMILY_MONO: ClassVar[str] = (
        '"JetBrains Mono", "Cascadia Code", '
        '"SF Mono", "Menlo", "Consolas", '
        '"DejaVu Sans Mono", monospace'
    )

    # Font sizes (in pt)
    SIZE_XS: ClassVar[int] = 9
    SIZE_SM: ClassVar[int] = 10
    SIZE_MD_SM: ClassVar[int] = 11  # Between SM and BASE
    SIZE_BASE: ClassVar[int] = 12
    SIZE_MD: ClassVar[int] = 13
    SIZE_LG: ClassVar[int] = 14
    SIZE_XL: ClassVar[int] = 16
    SIZE_XXL: ClassVar[int] = 20
    SIZE_HEADING: ClassVar[int] = 24
    SIZE_HERO: ClassVar[int] = 36  # Hero/display titles

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
    spacing = Spacing
    radius = BorderRadius

    # Size mappings
    sizes = {
        "sm": {"padding": f"{spacing.XS}px {spacing.SM}px", "font_size": "10pt"},
        "md": {"padding": f"{spacing.SM}px {spacing.LG}px", "font_size": "11pt"},
        "lg": {"padding": f"{spacing.MD}px {spacing.XL}px", "font_size": "12pt"},
    }

    # Variant mappings (theme-aware via themed())
    variants = {
        "primary": {
            "bg": themed("PRIMARY"),
            "bg_hover": themed("PRIMARY_HOVER"),
            "text": themed("TEXT_INVERSE"),
            "border": "none",
        },
        "secondary": {
            "bg": themed("BG_SURFACE"),
            "bg_hover": themed("BG_HOVER"),
            "text": themed("TEXT_PRIMARY"),
            "border": f"1px solid {themed('BORDER_DEFAULT')}",
        },
        "accent": {
            "bg": themed("ACCENT"),
            "bg_hover": themed("ACCENT_HOVER"),
            "text": themed("TEXT_INVERSE"),
            "border": "none",
        },
        "success": {
            "bg": themed("SUCCESS"),
            "bg_hover": themed("SUCCESS_HOVER"),
            "text": themed("TEXT_INVERSE"),
            "border": "none",
        },
        "warning": {
            "bg": themed("WARNING"),
            "bg_hover": themed("WARNING_HOVER"),
            "text": themed("TEXT_INVERSE"),
            "border": "none",
        },
        "error": {
            "bg": themed("ERROR"),
            "bg_hover": themed("ERROR_HOVER"),
            "text": themed("TEXT_INVERSE"),
            "border": "none",
        },
        "ghost": {
            "bg": "transparent",
            "bg_hover": themed("BG_HOVER"),
            "text": themed("TEXT_PRIMARY"),
            "border": "none",
        },
    }

    s = sizes.get(size, sizes["md"])
    v = variants.get(variant, variants["primary"])

    return f"""
        QPushButton {{
            background-color: {v["bg"]};
            color: {v["text"]};
            border: {v["border"]};
            border-radius: {radius.MD}px;
            padding: {s["padding"]};
            font-size: {s["font_size"]};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {v["bg_hover"]};
        }}
        QPushButton:disabled {{
            background-color: {themed("BG_ACTIVE")};
            color: {themed("TEXT_DISABLED")};
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
    spacing = Spacing
    radius = BorderRadius

    return f"""
        background-color: {themed("BG_ELEVATED")};
        border: 1px solid {themed("BORDER_DEFAULT")};
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
    radius = BorderRadius

    status_colors = {
        "success": (themed("SUCCESS"), themed("SUCCESS_LIGHT")),
        "warning": (themed("WARNING"), themed("WARNING_LIGHT")),
        "error": (themed("ERROR"), themed("ERROR_LIGHT")),
        "info": (themed("INFO"), themed("INFO_LIGHT")),
        "pending": (themed("TEXT_MUTED"), themed("BG_HOVER")),
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
    typography = Typography

    return f"""
        color: {themed("TEXT_PRIMARY")};
        font-size: {typography.SIZE_LG}pt;
        font-weight: {typography.WEIGHT_SEMIBOLD};
        padding-bottom: 8px;
        border-bottom: 2px solid {themed("BORDER_SUBTLE")};
        margin-bottom: 12px;
    """


def empty_state_style() -> str:
    """Generate empty state placeholder style string."""
    spacing = Spacing

    return f"""
        color: {themed("TEXT_MUTED")};
        padding: {spacing.XL}px;
        font-size: 11pt;
    """


# Python-dict mirrors of the QSS token files (light.qss / dark.qss).
# Keys match the @token_name identifiers used in base.qss (without the @ prefix).
# These dicts enable fully programmatic QSS generation via _generate_qss().

LIGHT_TOKENS: dict[str, str] = {
    # Background
    "bg_main_window": "#F8FAFC",
    "bg_surface": "#FFFFFF",
    "bg_input": "#FFFFFF",
    "bg_hover": "#F1F5F9",
    "bg_pressed": "#E2E8F0",
    "bg_canvas": "#F5F5F4",
    "bg_alt": "#F8FAFC",
    # Text
    "text_primary": "#0F172A",
    "text_secondary": "#64748B",
    "text_disabled": "#94A3B8",
    "text_inverse": "#FFFFFF",
    "text_on_secondary": "#334155",
    # Border
    "border_default": "#E2E8F0",
    "border_input": "#CBD5E1",
    "border_hover": "#94A3B8",
    "border_focus": "#6366F1",
    "border_subtle": "#E7E5E4",
    # Primary
    "primary": "#4338CA",
    "primary_hover": "#4F46E5",
    "primary_pressed": "#312E81",
    "primary_light": "#E0E7FF",
    "primary_dark": "#3730A3",
    "primary_gradient_start": "#4338CA",
    "primary_gradient_end": "#3730A3",
    "primary_hover_grad_start": "#4F46E5",
    "primary_hover_grad_end": "#4338CA",
    # Selection
    "selection_bg": "#E0E7FF",
    "selection_text": "#1E1B4B",
    "input_selection_bg": "#C7D2FE",
    # Form Controls
    "accent_checked": "#1D4ED8",
    "accent_checked_hover": "#1E40AF",
    "control_border": "#D6D3D1",
    "control_border_hover": "#A8A29E",
    "tool_checked_border": "#C7D2FE",
    # Scrollbar
    "scrollbar_handle": "#CBD5E1",
    "scrollbar_hover": "#94A3B8",
    # Tooltip
    "tooltip_bg": "#1C1917",
    "tooltip_text": "#FAFAF9",
    # Button
    "button_border": "1px solid #312E81",
    # Disabled Controls
    "control_disabled_bg": "#FAFAF9",
    "control_disabled_border": "#E7E5E4",
    # Tab
    "tab_hover_bg": "rgba(241, 245, 249, 0.5)",
    # Badges
    "badge_info_bg": "#DBEAFE",
    "badge_info_text": "#1D4ED8",
    # Icon Button Primary
    "icon_btn_primary_bg": "#EFF6FF",
    "icon_btn_primary_hover": "#DBEAFE",
}

DARK_TOKENS: dict[str, str] = {
    # Background
    "bg_main_window": "#0F172A",
    "bg_surface": "#1E293B",
    "bg_input": "#020617",
    "bg_hover": "#334155",
    "bg_pressed": "#0F172A",
    "bg_canvas": "#020617",
    "bg_alt": "#0F172A",
    # Text
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_disabled": "#64748B",
    "text_inverse": "#FFFFFF",
    "text_on_secondary": "#F8FAFC",
    # Border
    "border_default": "#334155",
    "border_input": "#334155",
    "border_hover": "#475569",
    "border_focus": "#6366F1",
    "border_subtle": "#334155",
    # Primary
    "primary": "#6366F1",
    "primary_hover": "#818CF8",
    "primary_pressed": "#4338CA",
    "primary_light": "#312E81",
    "primary_dark": "#4F46E5",
    "primary_gradient_start": "#6366F1",
    "primary_gradient_end": "#4F46E5",
    "primary_hover_grad_start": "#818CF8",
    "primary_hover_grad_end": "#6366F1",
    # Selection
    "selection_bg": "#312E81",
    "selection_text": "#FFFFFF",
    "input_selection_bg": "#4338CA",
    # Form Controls
    "accent_checked": "#6366F1",
    "accent_checked_hover": "#818CF8",
    "control_border": "#475569",
    "control_border_hover": "#64748B",
    "tool_checked_border": "#4338CA",
    # Scrollbar
    "scrollbar_handle": "#475569",
    "scrollbar_hover": "#64748B",
    # Tooltip
    "tooltip_bg": "#F8FAFC",
    "tooltip_text": "#0F172A",
    # Button
    "button_border": "none",
    # Disabled Controls
    "control_disabled_bg": "#1E293B",
    "control_disabled_border": "#334155",
    # Tab
    "tab_hover_bg": "rgba(51, 65, 85, 0.5)",
    # Badges
    "badge_info_bg": "#312E81",
    "badge_info_text": "#818CF8",
    # Icon Button Primary
    "icon_btn_primary_bg": "#312E81",
    "icon_btn_primary_hover": "#3730A3",
}


# Export all tokens and helpers
__all__ = [
    "LIGHT_TOKENS",
    "DARK_TOKENS",
    "ColorPalette",
    "DarkColorPalette",
    "ThemeManager",
    "themed",
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
