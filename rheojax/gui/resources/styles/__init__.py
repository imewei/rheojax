"""
RheoJAX GUI Stylesheets.

Provides light and dark themes for the application, along with design tokens
for consistent programmatic styling.
"""

from pathlib import Path

from rheojax.gui.resources.styles.tokens import (
    BorderRadius,
    ColorPalette,
    DesignTokens,
    Shadows,
    Spacing,
    Typography,
    button_style,
    card_style,
    empty_state_style,
    section_header_style,
    status_badge_style,
)

STYLES_DIR = Path(__file__).parent

__all__ = [
    # Stylesheet loaders
    "get_light_stylesheet",
    "get_dark_stylesheet",
    "get_stylesheet",
    "STYLES_DIR",
    # Design tokens
    "ColorPalette",
    "Spacing",
    "BorderRadius",
    "Typography",
    "Shadows",
    "DesignTokens",
    # Style helpers
    "button_style",
    "card_style",
    "status_badge_style",
    "section_header_style",
    "empty_state_style",
]


def get_light_stylesheet() -> str:
    """
    Load and return the light theme stylesheet.

    Returns
    -------
    str
        Complete QSS stylesheet for light theme.

    Examples
    --------
    >>> stylesheet = get_light_stylesheet()
    >>> app.setStyleSheet(stylesheet)
    """
    return (STYLES_DIR / "light.qss").read_text(encoding="utf-8")


def get_dark_stylesheet() -> str:
    """
    Load and return the dark theme stylesheet.

    Returns
    -------
    str
        Complete QSS stylesheet for dark theme.

    Examples
    --------
    >>> stylesheet = get_dark_stylesheet()
    >>> app.setStyleSheet(stylesheet)
    """
    return (STYLES_DIR / "dark.qss").read_text(encoding="utf-8")


def get_stylesheet(theme: str = "light") -> str:
    """
    Get stylesheet by theme name.

    Parameters
    ----------
    theme : str, optional
        Theme name, either "light" or "dark". Default is "light".

    Returns
    -------
    str
        Complete QSS stylesheet for the requested theme.

    Raises
    ------
    ValueError
        If theme is not "light" or "dark".

    Examples
    --------
    >>> # Get light theme (default)
    >>> stylesheet = get_stylesheet()
    >>> app.setStyleSheet(stylesheet)

    >>> # Get dark theme
    >>> stylesheet = get_stylesheet("dark")
    >>> app.setStyleSheet(stylesheet)
    """
    if theme not in ("light", "dark"):
        raise ValueError(f"Invalid theme '{theme}'. Must be 'light' or 'dark'.")

    if theme == "dark":
        return get_dark_stylesheet()
    return get_light_stylesheet()
