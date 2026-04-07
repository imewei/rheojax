"""Tests for safe_import_jax() and float64 configuration."""

import pytest


@pytest.mark.smoke
class TestSafeImportJax:
    """Tests for safe_import_jax()."""

    def test_returns_jax_module(self):
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        assert jax is not None
        assert jnp is not None

    def test_jax_has_expected_attrs(self):
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        assert hasattr(jax, "jit")
        assert hasattr(jax, "grad")
        assert hasattr(jnp, "array")
        assert hasattr(jnp, "float64")

    def test_float64_enabled(self):
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        test_array = jnp.array([1.0])
        assert test_array.dtype == jnp.float64

    def test_verify_float64_passes(self):
        from rheojax.core.jax_config import safe_import_jax, verify_float64

        safe_import_jax()
        # Should not raise
        verify_float64()

    def test_idempotent(self):
        """Calling safe_import_jax() multiple times returns same modules."""
        from rheojax.core.jax_config import safe_import_jax

        jax1, jnp1 = safe_import_jax()
        jax2, jnp2 = safe_import_jax()
        assert jax1 is jax2
        assert jnp1 is jnp2
