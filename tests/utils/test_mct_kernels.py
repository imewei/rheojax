"""Tests for rheojax.utils.mct_kernels (MCT memory kernel utilities)."""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from rheojax.utils.mct_kernels import f12_memory_kernel


@pytest.mark.smoke
class TestF12MemoryKernel:
    """Tests for the F12 schematic model memory kernel."""

    def test_basic_computation(self):
        phi = jnp.array([0.5, 0.8, 1.0])
        v1 = 0.0
        v2 = 4.0
        result = f12_memory_kernel(phi, v1, v2)
        assert result.shape == (3,)
        assert np.all(np.isfinite(np.asarray(result)))

    def test_zero_phi(self):
        """m(0) = 0 for any v1, v2."""
        phi = jnp.array([0.0])
        result = f12_memory_kernel(phi, v1=1.0, v2=4.0)
        assert abs(float(result[0])) < 1e-10

    def test_quadratic_form(self):
        """m(Phi) = v1*Phi + v2*Phi^2."""
        phi = jnp.array([0.5])
        v1 = 2.0
        v2 = 3.0
        result = f12_memory_kernel(phi, v1, v2)
        expected = v1 * 0.5 + v2 * 0.5**2
        np.testing.assert_allclose(float(result[0]), expected, rtol=1e-10)

    def test_glass_transition_value(self):
        """At v2=4, phi=1: m(1) = v2 = 4."""
        phi = jnp.array([1.0])
        result = f12_memory_kernel(phi, v1=0.0, v2=4.0)
        np.testing.assert_allclose(float(result[0]), 4.0, rtol=1e-10)

    def test_array_input(self):
        phi = jnp.linspace(0, 1, 50)
        result = f12_memory_kernel(phi, v1=0.0, v2=4.0)
        assert result.shape == (50,)
        # Should be monotonically increasing for v1=0, v2>0
        diffs = np.diff(np.asarray(result))
        assert np.all(diffs >= -1e-10)
