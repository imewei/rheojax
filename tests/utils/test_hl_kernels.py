"""Tests for rheojax.utils.hl_kernels (Hebraud-Lequeux kernels)."""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from rheojax.utils.hl_kernels import HLGrid, HLState, make_grid


@pytest.mark.smoke
class TestHLGrid:
    """Tests for HL grid construction."""

    def test_make_grid_default(self):
        grid = make_grid()
        assert isinstance(grid, HLGrid)
        assert grid.n_bins == 501
        assert grid.ds > 0

    def test_make_grid_custom(self):
        grid = make_grid(sigma_max=3.0, n_bins=201)
        assert grid.n_bins == 201
        # Grid should span [-sigma_max, sigma_max]
        assert float(grid.sigma[0]) < 0
        assert float(grid.sigma[-1]) > 0

    def test_grid_symmetry(self):
        grid = make_grid(sigma_max=5.0, n_bins=501)
        # Grid should be approximately symmetric
        assert abs(float(grid.sigma[0]) + float(grid.sigma[-1])) < grid.ds * 2

    def test_grid_spacing_positive(self):
        grid = make_grid()
        assert grid.ds > 0
