"""Tests for Fluidity physics kernels.

Tests cover core kernel functions including local fluidity calculation,
Laplacian with Neumann BCs, and ODE/PDE RHS functions.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled
jax, jnp = safe_import_jax()

from rheojax.models.fluidity._kernels import (
    banding_ratio,
    f_loc_herschel_bulkley,
    fluidity_local_creep_ode_rhs,
    fluidity_local_ode_rhs,
    fluidity_nonlocal_pde_rhs,
    laplacian_1d_neumann,
    shear_banding_cv,
)


@pytest.mark.smoke
class TestFluidityKernelsSmoke:
    """Smoke tests for basic kernel functionality."""

    def test_f_loc_returns_scalar(self):
        """Test f_loc_herschel_bulkley returns scalar output."""
        f_loc = f_loc_herschel_bulkley(sigma=1000.0, tau_y=500.0, K=100.0, n=0.5)
        assert isinstance(float(f_loc), float)
        assert np.isfinite(float(f_loc))

    def test_laplacian_returns_correct_shape(self):
        """Test Laplacian returns same shape as input."""
        f_field = jnp.linspace(0.0, 1.0, 64)
        dy = 1.0 / 63
        lap = laplacian_1d_neumann(f_field, dy)
        assert lap.shape == f_field.shape

    def test_local_ode_rhs_returns_correct_shape(self):
        """Test local ODE RHS returns correct shape."""
        y = jnp.array([1000.0, 1e-6])  # [sigma, f]
        args = {
            "G": 1e6,
            "f_eq": 1e-6,
            "f_inf": 1e-3,
            "theta": 10.0,
            "a": 1.0,
            "n_rejuv": 1.0,
            "gamma_dot": 0.1,
        }
        dy = fluidity_local_ode_rhs(0.0, y, args)
        assert dy.shape == y.shape


@pytest.mark.unit
class TestFluidityKernelsFLoc:
    """Tests for local fluidity calculation."""

    def test_f_loc_below_yield_is_small(self):
        """Test fluidity is very small below yield stress."""
        tau_y = 1000.0
        sigma_below = 500.0  # Below yield stress
        f_loc = f_loc_herschel_bulkley(sigma_below, tau_y, K=100.0, n=0.5)

        # Should be very small (smooth transition, not exactly zero)
        assert float(f_loc) < 0.1

    def test_f_loc_above_yield_is_positive(self):
        """Test fluidity is positive above yield stress."""
        tau_y = 1000.0
        sigma_above = 2000.0  # Above yield stress
        f_loc = f_loc_herschel_bulkley(sigma_above, tau_y, K=100.0, n=0.5)

        # Should be significant
        assert float(f_loc) > 0.01

    def test_f_loc_increases_with_stress(self):
        """Test fluidity increases with stress above yield."""
        tau_y = 1000.0
        K = 100.0
        n = 0.5

        sigma_1 = 1500.0
        sigma_2 = 2000.0
        sigma_3 = 3000.0

        f_1 = float(f_loc_herschel_bulkley(sigma_1, tau_y, K, n))
        f_2 = float(f_loc_herschel_bulkley(sigma_2, tau_y, K, n))
        f_3 = float(f_loc_herschel_bulkley(sigma_3, tau_y, K, n))

        assert f_1 < f_2 < f_3

    def test_f_loc_negative_stress_symmetric(self):
        """Test fluidity is symmetric for positive/negative stress."""
        tau_y = 1000.0
        K = 100.0
        n = 0.5

        f_pos = float(f_loc_herschel_bulkley(2000.0, tau_y, K, n))
        f_neg = float(f_loc_herschel_bulkley(-2000.0, tau_y, K, n))

        np.testing.assert_allclose(f_pos, f_neg, rtol=1e-10)


@pytest.mark.unit
class TestFluidityKernelsLaplacian:
    """Tests for Laplacian with Neumann boundary conditions."""

    def test_laplacian_of_constant_is_zero(self):
        """Test Laplacian of constant field is zero."""
        f_field = jnp.ones(64) * 5.0
        dy = 1.0 / 63
        lap = laplacian_1d_neumann(f_field, dy)

        np.testing.assert_allclose(np.array(lap), 0.0, atol=1e-10)

    def test_laplacian_of_linear_is_zero(self):
        """Test Laplacian of linear field is zero in interior."""
        f_field = jnp.linspace(0.0, 1.0, 64)
        dy = 1.0 / 63
        lap = laplacian_1d_neumann(f_field, dy)

        # Interior points should have zero Laplacian
        interior = np.array(lap)[1:-1]
        np.testing.assert_allclose(interior, 0.0, atol=1e-8)

    def test_laplacian_of_quadratic(self):
        """Test Laplacian of quadratic is constant."""
        x = jnp.linspace(0.0, 1.0, 65)
        f_field = x**2  # d2f/dx2 = 2
        dy = x[1] - x[0]
        lap = laplacian_1d_neumann(f_field, dy)

        # Interior should be approximately 2
        interior = np.array(lap)[2:-2]  # Avoid boundary effects
        np.testing.assert_allclose(interior, 2.0, rtol=0.1)


@pytest.mark.unit
class TestFluidityKernelsODERHS:
    """Tests for ODE right-hand-side functions."""

    def test_local_ode_rhs_stress_increases_with_rate(self):
        """Test stress rate is positive when gamma_dot > sigma*f."""
        y = jnp.array([0.0, 1e-6])  # Zero stress, small fluidity
        args = {
            "G": 1e6,
            "f_eq": 1e-6,
            "f_inf": 1e-3,
            "theta": 10.0,
            "a": 1.0,
            "n_rejuv": 1.0,
            "gamma_dot": 1.0,  # Applied shear rate
        }
        dy = fluidity_local_ode_rhs(0.0, y, args)

        # d_sigma should be positive (stress building)
        d_sigma = float(dy[0])
        assert d_sigma > 0

    def test_local_ode_rhs_fluidity_relaxes_at_rest(self):
        """Test fluidity relaxes toward f_eq at rest."""
        f_init = 1e-3  # Start at high fluidity (just sheared)
        f_eq = 1e-6  # Equilibrium is lower

        y = jnp.array([1000.0, f_init])
        args = {
            "G": 1e6,
            "f_eq": f_eq,
            "f_inf": 1e-3,
            "theta": 10.0,
            "a": 1.0,
            "n_rejuv": 1.0,
            "gamma_dot": 0.0,  # At rest
        }
        dy = fluidity_local_ode_rhs(0.0, y, args)

        # d_f should be negative (fluidity decreasing toward f_eq)
        d_f = float(dy[1])
        assert d_f < 0

    def test_creep_ode_rhs_strain_rate_positive_under_stress(self):
        """Test strain rate is positive under positive stress."""
        y = jnp.array([0.0, 1e-4])  # Zero strain, some fluidity
        args = {
            "sigma_applied": 1000.0,
            "f_eq": 1e-6,
            "f_inf": 1e-3,
            "theta": 10.0,
            "a": 1.0,
            "n_rejuv": 1.0,
        }
        dy = fluidity_local_creep_ode_rhs(0.0, y, args)

        # d_gamma should be positive
        d_gamma = float(dy[0])
        assert d_gamma > 0


@pytest.mark.unit
class TestFluidityKernelsPDERHS:
    """Tests for PDE right-hand-side functions."""

    def test_nonlocal_pde_rhs_returns_correct_shape(self):
        """Test PDE RHS returns correct shape."""
        N_y = 32
        # State: [Sigma, f[0], ..., f[N_y-1]]
        y = jnp.concatenate([jnp.array([0.0]), jnp.ones(N_y) * 1e-6])

        args = {
            "G": 1e6,
            "tau_y": 1000.0,
            "K": 100.0,
            "n_flow": 0.5,
            "theta": 10.0,
            "xi": 1e-5,
            "N_y": N_y,
            "dy": 1e-3 / (N_y - 1),
            "mode": 0,  # rate_controlled
            "gamma_dot": 1.0,
        }
        dy = fluidity_nonlocal_pde_rhs(0.0, y, args)

        assert dy.shape == y.shape

    def test_nonlocal_pde_stress_controlled_no_stress_change(self):
        """Test stress is constant in stress-controlled mode."""
        N_y = 32
        y = jnp.concatenate([jnp.array([1000.0]), jnp.ones(N_y) * 1e-6])

        args = {
            "G": 1e6,
            "tau_y": 500.0,
            "K": 100.0,
            "n_flow": 0.5,
            "theta": 10.0,
            "xi": 1e-5,
            "N_y": N_y,
            "dy": 1e-3 / (N_y - 1),
            "mode": 1,  # stress_controlled
            "sigma_applied": 1000.0,
        }
        dy = fluidity_nonlocal_pde_rhs(0.0, y, args)

        # d_Sigma should be zero
        d_Sigma = float(dy[0])
        assert d_Sigma == 0.0


@pytest.mark.unit
class TestFluidityKernelsBandingMetrics:
    """Tests for shear banding metrics."""

    def test_cv_of_uniform_field_is_zero(self):
        """Test CV of uniform field is zero."""
        f_field = jnp.ones(64) * 1e-4
        cv = shear_banding_cv(f_field)
        np.testing.assert_allclose(float(cv), 0.0, atol=1e-10)

    def test_cv_increases_with_heterogeneity(self):
        """Test CV increases with field heterogeneity."""
        # Uniform field
        f_uniform = jnp.ones(64) * 1e-4
        cv_uniform = float(shear_banding_cv(f_uniform))

        # Slightly heterogeneous
        x = jnp.linspace(0, 1, 64)
        f_het = 1e-4 * (1 + 0.5 * jnp.sin(2 * jnp.pi * x))
        cv_het = float(shear_banding_cv(f_het))

        assert cv_het > cv_uniform

    def test_banding_ratio_for_uniform_is_one(self):
        """Test banding ratio of uniform field is 1."""
        f_field = jnp.ones(64) * 1e-4
        ratio = banding_ratio(f_field)
        np.testing.assert_allclose(float(ratio), 1.0, rtol=1e-10)

    def test_banding_ratio_increases_with_contrast(self):
        """Test banding ratio increases with fluidity contrast."""
        # Low contrast
        f_low = jnp.ones(64) * 1e-4
        f_low = f_low.at[32:].set(2e-4)  # 2x contrast
        ratio_low = float(banding_ratio(f_low))

        # High contrast
        f_high = jnp.ones(64) * 1e-4
        f_high = f_high.at[32:].set(1e-2)  # 100x contrast
        ratio_high = float(banding_ratio(f_high))

        assert ratio_high > ratio_low
        np.testing.assert_allclose(ratio_low, 2.0, rtol=0.1)
        np.testing.assert_allclose(ratio_high, 100.0, rtol=0.1)
