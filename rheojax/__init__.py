"""RheoJAX: Unified Rheological Analysis Framework.

A comprehensive rheological analysis package integrating pyRheo's constitutive
models with hermes-rheo's data analysis transforms, providing JAX-accelerated
computations with multiple API styles and full piblin compatibility.

Float64 Precision Enforcement:
    This package enforces float64 precision for JAX operations by importing
    NLSQ before JAX. NLSQ automatically configures JAX for float64 when imported.
    All internal modules use safe_import_jax() from rheojax.core.jax_config to ensure
    proper import order.

    Users should NOT import JAX directly in code that uses rheojax. Instead, import
    from rheojax or use safe_import_jax() to maintain float64 precision.
"""

# Runtime version check (must be first)
import sys

# CRITICAL: Import NLSQ before JAX to enable float64 precision
# NLSQ auto-configures JAX for float64 when imported
try:
    import nlsq  # noqa: F401
except ImportError:
    raise ImportError(
        "NLSQ is required for RheoJAX but not installed.\n"
        "Install with: pip install nlsq>=0.1.6\n"
        "NLSQ provides GPU-accelerated optimization and enables float64 precision in JAX."
    )

__version__ = "0.1.0"
__author__ = "RheoJAX Development Team"
__email__ = "rheojax@example.com"
__license__ = "MIT"

# JAX version information (imported AFTER nlsq)
try:
    import jax
    import jax.numpy as jnp

    __jax_version__ = jax.__version__

    # Runtime check: Verify JAX is in float64 mode
    _test_array = jnp.array([1.0])
    if _test_array.dtype != jnp.float64:
        import warnings

        warnings.warn(
            f"JAX is not operating in float64 mode (current dtype: {_test_array.dtype}). "
            f"Float64 precision is required for numerical stability in rheological calculations. "
            f"Ensure NLSQ >= 0.1.6 is installed.",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError:
    __jax_version__ = "not installed"

# Version information
VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "release": "dev",
    "python_requires": ">=3.12",
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
logger.info(f"Loading rheojax version {__version__}")

# Core imports will be added as modules are implemented
# from . import core
# from . import models
# from . import transforms
