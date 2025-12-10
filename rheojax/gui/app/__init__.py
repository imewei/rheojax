"""
Application Core Components
===========================

Main window, menu bar, toolbar, and status bar widgets.
"""

__all__ = [
    "RheoJAXMainWindow",
    "MenuBar",
    "MainToolBar",
    "StatusBar",
]


def __getattr__(name: str):
    """Lazy import for application components."""
    if name == "RheoJAXMainWindow":
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        return RheoJAXMainWindow
    elif name == "MenuBar":
        from rheojax.gui.app.menu_bar import MenuBar

        return MenuBar
    elif name == "MainToolBar":
        from rheojax.gui.app.toolbar import MainToolBar

        return MainToolBar
    elif name == "StatusBar":
        from rheojax.gui.app.status_bar import StatusBar

        return StatusBar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
