"""
GUI Utilities
============

Configuration, JAX utilities, provenance, and seed management.
"""

__all__ = [
    "Config",
    "JaxUtils",
    "Provenance",
    "SeedManager",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
