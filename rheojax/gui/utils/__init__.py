"""
GUI Utilities
============

Configuration, JAX utilities, provenance, seed management, and platform-safe icons.
"""

__all__ = [
    "Config",
    "IconProvider",
    "JaxUtils",
    "Provenance",
    "SeedManager",
    "StandardIcon",
    "emoji_safe",
    "get_icon_provider",
    "get_standard_icon",
    "is_macos",
]


def __getattr__(name: str):
    """Lazy import for utility components."""
    if name == "Config":
        from rheojax.gui.utils.config import Config

        return Config
    elif name == "JaxUtils":
        from rheojax.gui.utils.jax_utils import JaxUtils

        return JaxUtils
    elif name == "Provenance":
        from rheojax.gui.utils.provenance import Provenance

        return Provenance
    elif name == "SeedManager":
        from rheojax.gui.utils.seeds import SeedManager

        return SeedManager
    elif name == "IconProvider":
        from rheojax.gui.utils.icons import IconProvider

        return IconProvider
    elif name == "StandardIcon":
        from rheojax.gui.utils.icons import StandardIcon

        return StandardIcon
    elif name == "emoji_safe":
        from rheojax.gui.utils.icons import emoji_safe

        return emoji_safe
    elif name == "get_icon_provider":
        from rheojax.gui.utils.icons import get_icon_provider

        return get_icon_provider
    elif name == "get_standard_icon":
        from rheojax.gui.utils.icons import get_standard_icon

        return get_standard_icon
    elif name == "is_macos":
        from rheojax.gui.utils.icons import is_macos

        return is_macos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
