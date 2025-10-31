"""JAX configuration and safe import mechanism for float64 precision.

This module provides utilities to ensure JAX operates in float64 mode by
enforcing that NLSQ is imported before JAX. NLSQ automatically configures
JAX for float64 precision when imported.

Critical Import Order:
    1. Import nlsq (auto-configures JAX for float64)
    2. Import JAX (now operates in float64 mode)

Usage:
    ```python
    from rheojax.core.jax_config import safe_import_jax
    jax, jnp = safe_import_jax()
    ```

This replaces direct JAX imports throughout the Rheo codebase.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

# Thread-safe singleton for validation results
_validation_lock = threading.Lock()
_validation_done = False
_jax_module = None
_jnp_module = None


def verify_float64() -> None:
    """Verify that JAX is operating in float64 mode.

    This function checks that JAX's default dtype is float64. It should be
    called after JAX has been imported to validate the configuration.

    Raises:
        RuntimeError: If JAX is not in float64 mode.

    Example:
        >>> import nlsq  # Configure JAX for float64
        >>> import jax.numpy as jnp
        >>> verify_float64()  # Validates float64 mode
    """
    import jax.numpy as jnp

    # Create a test array to check default dtype
    test_array = jnp.array([1.0])

    if test_array.dtype != jnp.float64:
        raise RuntimeError(
            f"JAX is not operating in float64 mode. "
            f"Default dtype is {test_array.dtype}. "
            f"Ensure NLSQ is imported before JAX to enable float64 precision."
        )


def safe_import_jax() -> tuple[Any, Any]:
    """Safely import JAX with float64 precision enforcement.

    This function ensures that NLSQ has been imported before JAX to enable
    float64 precision. It uses a thread-safe singleton pattern to cache
    validation results and avoid repeated checks.

    Returns:
        tuple: A tuple of (jax, jax.numpy) modules for use.

    Raises:
        ImportError: If NLSQ has not been imported before calling this function.

    Example:
        >>> # Correct usage (NLSQ imported first at package level)
        >>> import nlsq
        >>> from rheojax.core.jax_config import safe_import_jax
        >>> jax, jnp = safe_import_jax()
        >>> arr = jnp.array([1.0, 2.0, 3.0])  # Operates in float64

    Note:
        The rheojax package automatically imports NLSQ before JAX in __init__.py,
        so users don't need to worry about import order. This function is for
        internal use by RheoJAX modules.
    """
    global _validation_done, _jax_module, _jnp_module

    # Thread-safe check if validation already done
    with _validation_lock:
        if _validation_done:
            return _jax_module, _jnp_module

        # Check if NLSQ has been imported
        if "nlsq" not in sys.modules:
            raise ImportError(
                "NLSQ must be imported before JAX to enable float64 precision.\n\n"
                "The rheojax package should automatically import NLSQ before JAX.\n"
                "If you are seeing this error, ensure you are importing from rheojax "
                "and not directly importing JAX.\n\n"
                "Correct usage:\n"
                "    import rheojax  # Automatically imports nlsq before JAX\n"
                "    from rheojax.core.jax_config import safe_import_jax\n"
                "    jax, jnp = safe_import_jax()\n\n"
                "Incorrect usage:\n"
                "    import jax  # Direct import bypasses float64 configuration\n\n"
                "For more information, see CLAUDE.md section 'Float64 Precision Enforcement'."
            )

        # Import JAX modules
        import jax
        import jax.numpy as jnp

        # Verify float64 mode
        try:
            verify_float64()
        except RuntimeError as e:
            raise RuntimeError(
                f"Float64 verification failed: {e}\n\n"
                f"Although NLSQ was imported, JAX is not operating in float64 mode. "
                f"This may indicate a version incompatibility or configuration issue.\n"
                f"Please check that NLSQ >= 0.1.6 is installed."
            ) from e

        # Cache the modules for future calls
        _jax_module = jax
        _jnp_module = jnp
        _validation_done = True

        return jax, jnp


def reset_validation() -> None:
    """Reset validation state (for testing purposes only).

    This function is intended for use in tests that need to simulate
    different import scenarios. It should not be used in production code.

    Warning:
        This is not thread-safe and should only be used in single-threaded
        test environments.
    """
    global _validation_done, _jax_module, _jnp_module

    with _validation_lock:
        _validation_done = False
        _jax_module = None
        _jnp_module = None
