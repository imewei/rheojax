"""JAX configuration and safe import mechanism for float64 precision.

This module provides utilities to ensure JAX operates in float64 mode by
enforcing that NLSQ is imported before JAX and explicitly enabling float64.

NLSQ v0.2.1+ uses float32 by default with automatic precision fallback.
RheoJAX explicitly enables float64 for numerical stability in rheological
calculations.

Critical Configuration Steps:
    1. Import nlsq (required for GPU-accelerated optimization)
    2. Import JAX
    3. Enable float64: jax.config.update("jax_enable_x64", True)

Usage:
    ```python
    from rheojax.core.jax_config import safe_import_jax
    jax, jnp = safe_import_jax()
    ```

This replaces direct JAX imports throughout the RheoJAX codebase.
"""

from __future__ import annotations

import sys
import threading
import warnings
from typing import Any

# Thread-safe singleton for validation results
_validation_lock = threading.Lock()
_validation_done = False
_jax_module = None
_jnp_module = None


def suppress_glyph_warnings() -> None:
    """Suppress matplotlib font glyph warnings.

    These warnings are purely cosmetic — plots render correctly, the glyph is
    just displayed as a box or skipped. Common when using Unicode subscripts
    (e.g., σ₀, τ₀) with fonts that lack those glyphs. This is harmless for
    headless batch runs and provides no actionable information to users.

    Call explicitly rather than relying on module-level side effects.
    ``safe_import_jax()`` calls this automatically.
    """
    warnings.filterwarnings(
        "ignore",
        message="Glyph.*missing from.*font",
        category=UserWarning,
    )


def verify_float64() -> None:
    """Verify that JAX is operating in float64 mode.

    This function checks that JAX's default dtype is float64. It should be
    called after JAX has been imported to validate the configuration.

    Raises:
        RuntimeError: If JAX is not in float64 mode.

    Example:
        >>> import nlsq
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> verify_float64()  # Validates float64 mode
    """
    import jax.numpy as jnp

    # Create a test array to check default dtype
    test_array = jnp.array([1.0])

    if test_array.dtype != jnp.float64:
        raise RuntimeError(
            f"JAX is not operating in float64 mode. "
            f"Default dtype is {test_array.dtype}. "
            f"Ensure jax.config.update('jax_enable_x64', True) is called after importing JAX."
        )


def _enable_compilation_cache(jax_module: Any) -> None:
    """Enable JAX persistent compilation cache for cross-session speedup.

    Persists XLA compiled programs to ``~/.cache/rheojax/jax_cache/``.
    Eliminates 764-1552ms cold JIT overhead on subsequent Python sessions.

    Disabled when ``RHEOJAX_NO_JIT_CACHE=1`` is set (useful for debugging
    compilation issues or benchmarking cold-start performance).
    """
    import os
    import pathlib

    if os.environ.get("RHEOJAX_NO_JIT_CACHE", "0") == "1":
        return

    cache_dir = pathlib.Path.home() / ".cache" / "rheojax" / "jax_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # JAX >= 0.4.25: preferred config-based API
        jax_module.config.update("jax_compilation_cache_dir", str(cache_dir))
    except (OSError, RuntimeError, AttributeError, TypeError):
        try:
            # Fallback: older JAX experimental API
            from jax.experimental.compilation_cache import compilation_cache as cc

            cc.set_cache_dir(str(cache_dir))
        except (OSError, RuntimeError, ImportError):
            # Non-fatal: cache is a performance optimization, not required
            pass


def safe_import_jax() -> tuple[Any, Any]:
    """Safely import JAX with float64 precision enforcement.

    This function ensures that NLSQ has been imported before JAX and explicitly
    enables float64 precision. NLSQ v0.2.1+ uses float32 by default, so RheoJAX
    must explicitly configure JAX for float64.

    It uses a thread-safe singleton pattern to cache validation results and
    avoid repeated checks.

    Returns:
        tuple: A tuple of (jax, jax.numpy) modules for use.

    Raises:
        ImportError: If NLSQ has not been imported before calling this function.
        RuntimeError: If float64 mode cannot be enabled.

    Example:
        >>> # Correct usage (NLSQ imported first at package level)
        >>> import nlsq
        >>> from rheojax.core.jax_config import safe_import_jax
        >>> jax, jnp = safe_import_jax()
        >>> arr = jnp.array([1.0, 2.0, 3.0])  # Operates in float64

    Note:
        The rheojax package automatically imports NLSQ and configures JAX in
        __init__.py, so users don't need to worry about configuration. This
        function is for internal use by RheoJAX modules.
    """
    global _validation_done, _jax_module, _jnp_module

    # Thread-safe check if validation already done
    with _validation_lock:
        if _validation_done:
            return _jax_module, _jnp_module

        # Check if NLSQ has been imported
        if "nlsq" not in sys.modules:
            raise ImportError(
                "NLSQ must be imported before using RheoJAX.\n\n"
                "The rheojax package should automatically import NLSQ.\n"
                "If you are seeing this error, ensure you are importing from rheojax "
                "and not directly importing JAX.\n\n"
                "Correct usage:\n"
                "    import rheojax  # Automatically imports nlsq and configures JAX\n"
                "    from rheojax.core.jax_config import safe_import_jax\n"
                "    jax, jnp = safe_import_jax()\n\n"
                "Incorrect usage:\n"
                "    import jax  # Direct import bypasses float64 configuration\n\n"
                "For more information, see CLAUDE.md section 'Float64 Precision Enforcement'."
            )

        # Import JAX modules
        import jax
        import jax.numpy as jnp

        # CRITICAL: Explicitly enable float64 precision
        # NLSQ v0.2.1+ uses float32 by default, so we must configure JAX explicitly
        jax.config.update("jax_enable_x64", True)

        # Enable persistent XLA compilation cache.  This avoids the 764-1552ms
        # cold JIT overhead on subsequent Python sessions by persisting compiled
        # XLA programs to disk.  JAX validates cache entries by HLO hash, so
        # stale entries are automatically ignored after code changes.
        _enable_compilation_cache(jax)

        # Verify float64 mode
        try:
            verify_float64()
        except RuntimeError as e:
            raise RuntimeError(
                f"Float64 verification failed: {e}\n\n"
                f"Although JAX float64 was enabled, verification failed. "
                f"This may indicate a JAX version incompatibility.\n"
                f"Please check that JAX 0.8.0 is installed and NLSQ >= 0.2.1."
            ) from e

        # Suppress cosmetic matplotlib glyph warnings (safe to call repeatedly)
        suppress_glyph_warnings()

        # Cache the modules for future calls
        _jax_module = jax
        _jnp_module = jnp
        _validation_done = True

        return jax, jnp


class _LazyModule:
    """Proxy that defers ``import <name>`` until first attribute access.

    This is used to avoid paying the import cost of heavy optional
    dependencies (e.g. diffrax ~250 ms) at package-load time while
    keeping callsite syntax unchanged (``diffrax.Tsit5()``).
    """

    __slots__ = ("_module_name", "_module")

    def __init__(self, module_name: str) -> None:
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_module", None)

    def _load(self):
        import importlib

        mod = importlib.import_module(self._module_name)
        object.__setattr__(self, "_module", mod)
        return mod

    def __getattr__(self, name: str):
        mod = self._module
        if mod is None:
            mod = self._load()
        return getattr(mod, name)

    def __repr__(self) -> str:
        loaded = self._module is not None
        return f"<LazyModule '{self._module_name}' loaded={loaded}>"


def lazy_import(module_name: str) -> _LazyModule:
    """Return a lazy proxy for *module_name*.

    The real ``import`` is deferred until the first attribute access on the
    returned object.  This is safe for modules that are only used inside
    method bodies (not at module-level scope for decorators, base classes,
    etc.).

    Example::

        diffrax = lazy_import("diffrax")
        # ... later, inside a method ...
        solver = diffrax.Tsit5()   # triggers ``import diffrax`` on first use
    """
    return _LazyModule(module_name)


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
