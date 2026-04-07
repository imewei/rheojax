"""
RheoJAX GUI Stylesheets.

Provides light and dark themes for the application, along with design tokens
for consistent programmatic styling.

Architecture: base.qss (structural) + theme token files (color definitions).
Token substitution replaces @token placeholders at load time.
"""

import re
from pathlib import Path

from rheojax.gui.resources.styles.tokens import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    BorderRadius,
    ColorPalette,
    DarkColorPalette,
    DesignTokens,
    Shadows,
    Spacing,
    ThemeManager,
    Typography,
    button_style,
    card_style,
    empty_state_style,
    section_header_style,
    status_badge_style,
    themed,
)

STYLES_DIR = Path(__file__).parent

__all__ = [
    # Stylesheet loaders
    "_generate_qss",
    "get_light_stylesheet",
    "get_dark_stylesheet",
    "get_stylesheet",
    "STYLES_DIR",
    # Design tokens
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
    # Style helpers
    "button_style",
    "card_style",
    "status_badge_style",
    "section_header_style",
    "empty_state_style",
]

# Regex to match @token_name placeholders (lowercase + underscores only)
_TOKEN_RE = re.compile(r"@([a-z][a-z_0-9]*)")


def _parse_tokens(token_text: str) -> dict[str, str]:
    """Parse @token = value definitions from a theme token file.

    Parameters
    ----------
    token_text : str
        Contents of a theme token file (e.g., light.qss or dark.qss).

    Returns
    -------
    dict[str, str]
        Mapping of token names (with @ prefix) to their values.
    """
    tokens: dict[str, str] = {}
    for line in token_text.splitlines():
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("/*") or line.startswith("*"):
            continue
        if line.startswith("@") and "=" in line:
            name, _, value = line.partition("=")
            tokens[name.strip()] = value.strip()
    return tokens


def _substitute_tokens(base_qss: str, tokens: dict[str, str]) -> str:
    """Replace @token placeholders in base QSS with actual values.

    Parameters
    ----------
    base_qss : str
        Base QSS template containing @token placeholders.
    tokens : dict[str, str]
        Token name-to-value mapping from _parse_tokens().

    Returns
    -------
    str
        Fully resolved QSS stylesheet.
    """

    def _replacer(match: re.Match) -> str:
        token = match.group(0)  # includes @ prefix
        return tokens.get(token, token)

    return _TOKEN_RE.sub(_replacer, base_qss)


def _generate_qss(theme: str) -> str:
    """Generate a complete QSS stylesheet by substituting Python token dicts.

    This is the preferred entry-point for programmatic QSS generation.
    It reads ``base.qss`` once and substitutes tokens from ``LIGHT_TOKENS``
    or ``DARK_TOKENS`` defined in ``tokens.py``.  Falls back to the file-based
    token approach (``light.qss`` / ``dark.qss``) when the Python dicts are
    not available or incomplete.

    Parameters
    ----------
    theme : str
        ``"light"`` or ``"dark"``.

    Returns
    -------
    str
        Fully-resolved QSS stylesheet.
    """
    from rheojax.gui.resources.styles.tokens import DARK_TOKENS, LIGHT_TOKENS

    token_dict = DARK_TOKENS if theme == "dark" else LIGHT_TOKENS
    # Convert Python dict keys to @-prefixed form expected by _substitute_tokens
    tokens = {f"@{k}": v for k, v in token_dict.items()}

    base_qss = (STYLES_DIR / "base.qss").read_text(encoding="utf-8")
    return _substitute_tokens(base_qss, tokens)


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
    return get_stylesheet("light")


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
    return get_stylesheet("dark")


def get_stylesheet(theme: str = "light") -> str:
    """
    Get stylesheet by theme name.

    Loads base.qss and substitutes tokens from the theme's token file.

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

    base_qss = (STYLES_DIR / "base.qss").read_text(encoding="utf-8")
    token_text = (STYLES_DIR / f"{theme}.qss").read_text(encoding="utf-8")
    tokens = _parse_tokens(token_text)
    return _substitute_tokens(base_qss, tokens)
