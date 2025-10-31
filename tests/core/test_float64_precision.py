"""Tests for float64 precision enforcement in JAX.

This test suite validates that NLSQ-before-JAX import order is enforced
and that JAX operates in float64 mode throughout the Rheo package.
"""

import importlib
import sys

import pytest


class TestFloat64ImportOrder:
    """Test import order enforcement for float64 precision."""

    def test_safe_import_jax_without_nlsq_raises_error(self):
        """Test that safe_import_jax() raises error when NLSQ not imported first."""
        # Remove nlsq from sys.modules if present
        if "nlsq" in sys.modules:
            # Save the module for restoration
            nlsq_module = sys.modules.pop("nlsq")
        else:
            nlsq_module = None

        # Remove jax_config from cache to force re-import
        if "rheo.core.jax_config" in sys.modules:
            del sys.modules["rheo.core.jax_config"]

        try:
            # This should raise an ImportError
            from rheojax.core.jax_config import safe_import_jax

            with pytest.raises(ImportError, match="NLSQ must be imported before JAX"):
                jax, jnp = safe_import_jax()
        finally:
            # Restore nlsq module if it was present
            if nlsq_module is not None:
                sys.modules["nlsq"] = nlsq_module

    def test_safe_import_jax_with_nlsq_succeeds(self):
        """Test that safe_import_jax() succeeds when NLSQ imported first."""
        # Ensure nlsq is imported
        import nlsq  # noqa: F401

        # Remove jax_config from cache to force re-import
        if "rheo.core.jax_config" in sys.modules:
            del sys.modules["rheo.core.jax_config"]

        from rheojax.core.jax_config import safe_import_jax

        # This should succeed without raising
        jax, jnp = safe_import_jax()

        assert jax is not None
        assert jnp is not None
        assert hasattr(jax, "numpy")

    def test_jax_operates_in_float64_mode(self):
        """Test that JAX operates in float64 mode after proper import."""
        # Ensure nlsq is imported first
        import nlsq  # noqa: F401

        # Remove jax_config from cache to force re-import
        if "rheo.core.jax_config" in sys.modules:
            del sys.modules["rheo.core.jax_config"]

        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()

        # Create a simple array and check its dtype
        arr = jnp.array([1.0, 2.0, 3.0])

        # JAX should default to float64
        assert arr.dtype == jnp.float64, f"Expected float64, got {arr.dtype}"

    def test_verify_float64_helper_function(self):
        """Test verify_float64() helper function."""
        # Ensure nlsq is imported first
        import nlsq  # noqa: F401

        # Remove jax_config from cache to force re-import
        if "rheo.core.jax_config" in sys.modules:
            del sys.modules["rheo.core.jax_config"]

        from rheojax.core.jax_config import verify_float64

        # This should not raise an exception
        verify_float64()

    def test_import_order_violation_error_message_clear(self):
        """Test that import order violation error messages are clear and actionable."""
        # Remove nlsq from sys.modules if present
        if "nlsq" in sys.modules:
            nlsq_module = sys.modules.pop("nlsq")
        else:
            nlsq_module = None

        # Remove jax_config from cache to force re-import
        if "rheo.core.jax_config" in sys.modules:
            del sys.modules["rheo.core.jax_config"]

        try:
            from rheojax.core.jax_config import safe_import_jax

            with pytest.raises(ImportError) as exc_info:
                jax, jnp = safe_import_jax()

            error_message = str(exc_info.value)

            # Error message should be helpful and actionable
            assert "NLSQ must be imported before JAX" in error_message
            assert (
                "import nlsq" in error_message.lower()
                or "nlsq" in error_message.lower()
            )

        finally:
            # Restore nlsq module if it was present
            if nlsq_module is not None:
                sys.modules["nlsq"] = nlsq_module

    def test_safe_import_jax_is_idempotent(self):
        """Test that calling safe_import_jax() multiple times is safe."""
        # Ensure nlsq is imported first
        import nlsq  # noqa: F401

        # Remove jax_config from cache to force re-import
        if "rheo.core.jax_config" in sys.modules:
            del sys.modules["rheo.core.jax_config"]

        from rheojax.core.jax_config import safe_import_jax

        # Call multiple times
        jax1, jnp1 = safe_import_jax()
        jax2, jnp2 = safe_import_jax()

        # Should return the same modules
        assert jax1 is jax2
        assert jnp1 is jnp2

    def test_float64_precision_maintained_in_operations(self):
        """Test that float64 precision is maintained in JAX operations."""
        # Ensure nlsq is imported first
        import nlsq  # noqa: F401

        # Remove jax_config from cache to force re-import
        if "rheo.core.jax_config" in sys.modules:
            del sys.modules["rheo.core.jax_config"]

        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()

        # Create arrays
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])

        # Perform operations
        c = a + b
        d = a * b
        e = jnp.sin(a)

        # All operations should maintain float64
        assert c.dtype == jnp.float64
        assert d.dtype == jnp.float64
        assert e.dtype == jnp.float64

    def test_rheojax_package_import_order(self):
        """Test that rheojax package itself ensures proper import order."""
        # This test validates that importing rheojax will import nlsq before JAX
        # Save current sys.modules state for rheojax modules
        saved_modules = {k: sys.modules[k] for k in list(sys.modules.keys()) if k.startswith("rheojax")}

        try:
            # Clear sys.modules to force fresh import
            modules_to_clear = [k for k in sys.modules.keys() if k.startswith("rheojax")]
            for mod in modules_to_clear:
                if mod != "rheojax.core.jax_config":  # Don't clear jax_config yet
                    del sys.modules[mod]

            # Import rheojax package
            import rheojax

            # Verify nlsq was imported
            assert "nlsq" in sys.modules, "rheojax package should import nlsq"

            # Verify JAX is in float64 mode
            import jax.numpy as jnp

            arr = jnp.array([1.0])
            assert arr.dtype == jnp.float64
        finally:
            # Restore all rheojax modules to prevent state pollution
            # First, remove any newly imported modules
            current_modules = [k for k in sys.modules.keys() if k.startswith("rheojax")]
            for mod in current_modules:
                if mod not in saved_modules:
                    del sys.modules[mod]

            # Then restore the saved modules
            for mod, obj in saved_modules.items():
                sys.modules[mod] = obj
