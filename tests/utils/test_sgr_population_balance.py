"""Tests for rheojax.utils.sgr_population_balance (SGR PB solver)."""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from rheojax.utils.sgr_population_balance import SGRPBGrid, create_grid


@pytest.mark.smoke
class TestSGRPopulationBalance:
    """Smoke tests for SGR Population Balance grid construction."""

    def test_create_grid_default(self):
        grid = create_grid()
        assert isinstance(grid, SGRPBGrid)
        assert grid.dE > 0
        assert grid.dell > 0

    def test_create_grid_custom(self):
        grid = create_grid(E_max=5.0, ell_max=1.0, N_E=32, N_ell=64)
        assert grid.E_centers.shape == (32,)
        assert grid.ell_centers.shape == (64,)

    def test_grid_edges_consistent(self):
        grid = create_grid(N_E=16, N_ell=32)
        # Edges should be one more than centers
        assert grid.E_edges.shape[0] == grid.E_centers.shape[0] + 1
        assert grid.ell_edges.shape[0] == grid.ell_centers.shape[0] + 1

    def test_grid_centers_within_edges(self):
        grid = create_grid(N_E=16, N_ell=32)
        E_centers = np.asarray(grid.E_centers)
        E_edges = np.asarray(grid.E_edges)
        # All centers should be within edge range
        assert np.all(E_centers >= E_edges[0])
        assert np.all(E_centers <= E_edges[-1])
