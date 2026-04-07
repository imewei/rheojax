"""Tests for rheojax.utils.sgr_monte_carlo (SGR MC simulator)."""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from rheojax.utils.sgr_monte_carlo import SGRMCState, initialize_equilibrium


@pytest.mark.smoke
class TestSGRMonteCarlo:
    """Smoke tests for SGR Monte Carlo initialization."""

    def test_initialize_equilibrium(self):
        key = jax.random.PRNGKey(42)
        state = initialize_equilibrium(key, n_particles=100, x=1.5)
        assert isinstance(state, SGRMCState)
        assert state.E.shape == (100,)
        assert state.ell.shape == (100,)

    def test_initial_strain_is_zero(self):
        key = jax.random.PRNGKey(42)
        state = initialize_equilibrium(key, n_particles=50, x=1.5)
        np.testing.assert_allclose(np.asarray(state.ell), 0.0)

    def test_initial_trap_depths_positive(self):
        key = jax.random.PRNGKey(42)
        state = initialize_equilibrium(key, n_particles=200, x=1.5)
        E = np.asarray(state.E)
        assert np.all(E >= 0)

    def test_reproducibility(self):
        key = jax.random.PRNGKey(42)
        state1 = initialize_equilibrium(key, n_particles=50, x=1.5)
        state2 = initialize_equilibrium(key, n_particles=50, x=1.5)
        np.testing.assert_array_equal(np.asarray(state1.E), np.asarray(state2.E))
