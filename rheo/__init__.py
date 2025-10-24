"""Rheo: Unified Rheological Analysis Framework.

A comprehensive rheological analysis package integrating pyRheo's constitutive
models with hermes-rheo's data analysis transforms, providing JAX-accelerated
computations with multiple API styles and full piblin compatibility.
"""

# Runtime version check (must be first)
import sys

if sys.version_info < (3, 9):
    raise ImportError(
        f"Rheo requires Python 3.9 or later. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
    )

__version__ = "1.0.0"
__author__ = "Rheo Development Team"
__email__ = "rheo@example.com"
__license__ = "MIT"

# JAX version information
try:
    import jax

    __jax_version__ = jax.__version__
except ImportError:
    __jax_version__ = "not installed"

# Version information
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable",
    "python_requires": ">=3.9",
}

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__jax_version__",
    "VERSION_INFO",
    # Core modules will be added as they are implemented
    # "core",
    # "models",
    # "transforms",
    # "pipelines",
    # "io",
    # "visualization",
    # "utils",
]

# Optional: Log package loading
import logging

logger = logging.getLogger(__name__)
logger.info(f"Loading rheo version {__version__}")

# Core imports will be added as modules are implemented
# from . import core
# from . import models
# from . import transforms