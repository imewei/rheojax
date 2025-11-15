"""RheoJAX: Unified Rheological Analysis Framework.

A comprehensive rheological analysis package integrating pyRheo's constitutive
models with hermes-rheo's data analysis transforms, providing JAX-accelerated
computations with multiple API styles and full piblin compatibility.

Float64 Precision Enforcement:
    This package enforces float64 precision for JAX operations by:
    1. Importing NLSQ (required for GPU-accelerated optimization)
    2. Importing JAX
    3. Explicitly enabling float64: jax.config.update("jax_enable_x64", True)

    NLSQ v0.2.1+ uses float32 by default with automatic precision fallback.
    RheoJAX explicitly enables float64 for numerical stability in rheological
    calculations.

    All internal modules use safe_import_jax() from rheojax.core.jax_config to ensure
    proper configuration.

    Users should NOT import JAX directly in code that uses rheojax. Instead, import
    from rheojax or use safe_import_jax() to maintain float64 precision.
"""

# Runtime version check (must be first)
import sys

# CRITICAL: Import NLSQ before JAX
# Required for GPU-accelerated optimization
try:
    import nlsq  # noqa: F401
except ImportError as e:
    raise ImportError(
        "NLSQ is required for RheoJAX but not installed.\n"
        "Install with: pip install nlsq>=0.2.1\n"
        "NLSQ provides GPU-accelerated optimization for rheological models."
    ) from e

__version__ = "0.2.2"
__author__ = "Wei Chen"
__email__ = "wchen@anl.gov"
__license__ = "MIT"

# JAX version information (imported AFTER nlsq)
try:
    import jax
    import jax.numpy as jnp

    # CRITICAL: Explicitly enable float64 precision
    # NLSQ v0.2.1+ uses float32 by default, so we must configure JAX explicitly
    jax.config.update("jax_enable_x64", True)

    __jax_version__ = jax.__version__

    # Runtime check: Verify JAX is in float64 mode
    _test_array = jnp.array([1.0])
    if _test_array.dtype != jnp.float64:
        import warnings

        warnings.warn(
            f"JAX is not operating in float64 mode (current dtype: {_test_array.dtype}). "
            f"Float64 precision is required for numerical stability in rheological calculations. "
            f"This may indicate a JAX configuration issue. "
            f"Ensure JAX 0.8.0 and NLSQ >= 0.2.1 are installed.",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError:
    __jax_version__ = "not installed"

# Version information
VERSION_INFO = {
    "major": 0,
    "minor": 2,
    "patch": 2,
    "release": "stable",
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

# Core imports - models must be imported to register with ModelRegistry
from . import models  # noqa: F401
