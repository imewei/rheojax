"""
GUI Resources
============

Loader helpers for stylesheets, plot styles, and icons used by the RheoJAX
GUI. These utilities are intentionally lightweight so they can be imported
from the GUI entry point without pulling in Qt.
"""

from pathlib import Path
from typing import Literal

from rheojax.gui.resources.styles import get_stylesheet

RESOURCES_DIR = Path(__file__).parent
ICONS_DIR = RESOURCES_DIR / "icons"
PLOT_STYLES_DIR = RESOURCES_DIR / "styles" / "plot_styles"

__all__ = [
    "load_stylesheet",
    "load_plot_style",
    "get_icon_path",
    "RESOURCES_DIR",
    "ICONS_DIR",
    "PLOT_STYLES_DIR",
]


def load_stylesheet(theme: Literal["light", "dark"] = "light") -> str:
    """Return the QSS stylesheet content for the requested theme.

    Parameters
    ----------
    theme : {"light", "dark"}, default "light"
        Theme name from the GUI specification.

    Returns
    -------
    str
        QSS stylesheet text.
    """

    qss = get_stylesheet(theme)
    # Replace Qt resource prefixes with filesystem paths to avoid missing-icon warnings
    qss = qss.replace(":/icons/", f"{ICONS_DIR.as_posix()}/")
    return qss


def load_plot_style(name: str = "default") -> str:
    """Load a matplotlib style file bundled with the GUI.

    Falls back to the default style when the requested file is missing.

    Parameters
    ----------
    name : str, default "default"
        Style name without extension (e.g., "publication").

    Returns
    -------
    str
        File content of the `.mplstyle` definition.
    """

    style_file = PLOT_STYLES_DIR / f"{name}.mplstyle"
    if not style_file.exists():
        style_file = PLOT_STYLES_DIR / "default.mplstyle"
    return style_file.read_text(encoding="utf-8")


def get_icon_path(name: str) -> Path:
    """Return the path to an SVG icon bundled with the GUI.

    Parameters
    ----------
    name : str
        Icon filename (with or without `.svg`).

    Returns
    -------
    Path
        Filesystem path to the icon (may not exist if not bundled).
    """

    if not name.endswith(".svg"):
        name = f"{name}.svg"
    return ICONS_DIR / name
