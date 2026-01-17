"""Tests for STZ physics kernels.

Tests cover numerical stability, gradient computation, and correct physical behavior.
Follows the 2-8 test rule per task group.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled
jax, jnp = safe_import_jax()

from rheojax.models.stz._kernels import (
    chi_evolution_langer2008,
    lambda_evolution,
    m_evolution,
    plastic_rate,
    rate_factor_C,
    stz_density,
    stz_ode_rhs,
    transition_T,
)


@pytest.mark.unit
class TestSTZKernels:
    """Test suite for STZ physics kernels."""

    def test_rate_factor_C_stability(self):
        """Test rate_factor_C using log-cosh for numerical stability."""
        sigma_y = 1e6  # 1 MPa yield stress

        # Case 1: Zero stress - C(0) = cosh(0) = 1
        c_zero = rate_factor_C(0.0, sigma_y)
        assert jnp.isfinite(c_zero)
        assert jnp.isclose(c_zero, 1.0, rtol=1e-10)

        # Case 2: Small stress
        c_small = rate_factor_C(1e5, sigma_y)  # 0.1 * sigma_y
        assert jnp.isfinite(c_small)
        assert c_small >= 1.0  # cosh(x) >= 1

        # Case 3: Moderate stress (~ sigma_y)
        c_moderate = rate_factor_C(sigma_y, sigma_y)
        assert jnp.isfinite(c_moderate)
        expected_cosh_1 = jnp.cosh(1.0)
        assert jnp.isclose(c_moderate, expected_cosh_1, rtol=1e-6)

        # Case 4: Large stress - log-cosh should prevent overflow
        # For stress = 50 * sigma_y, cosh(50) ~ 2.6e21 - still manageable
        c_large = rate_factor_C(50.0 * sigma_y, sigma_y)
        assert jnp.isfinite(c_large)

        # Case 5: Gradient exists and is finite
        grad_C = jax.grad(rate_factor_C, argnums=0)(sigma_y, sigma_y)
        assert jnp.isfinite(grad_C)

    def test_transition_T_stability(self):
        """Test transition_T using tanh for stability."""
        sigma_y = 1e6

        # T(s) = tanh(s / sigma_y) should be bounded [-1, 1]
        stresses = jnp.array([-1e10, -sigma_y, 0.0, sigma_y, 1e10])
        for s in stresses:
            t_val = transition_T(s, sigma_y)
            assert jnp.isfinite(t_val)
            assert jnp.abs(t_val) <= 1.0 + 1e-12

        # At zero stress, T = 0
        t_zero = transition_T(0.0, sigma_y)
        assert jnp.isclose(t_zero, 0.0, atol=1e-12)

        # At large positive stress, T -> 1
        t_large_pos = transition_T(100.0 * sigma_y, sigma_y)
        assert jnp.isclose(t_large_pos, 1.0, rtol=1e-6)

        # At large negative stress, T -> -1
        t_large_neg = transition_T(-100.0 * sigma_y, sigma_y)
        assert jnp.isclose(t_large_neg, -1.0, rtol=1e-6)

    def test_stz_density_behavior(self):
        """Test STZ density Lambda = exp(-1/chi)."""
        # For chi = 0.1, Lambda = exp(-10) ~ 4.5e-5 (very small)
        Lambda_low = stz_density(0.1)
        assert jnp.isfinite(Lambda_low)
        assert Lambda_low > 0
        assert jnp.isclose(Lambda_low, jnp.exp(-10.0), rtol=1e-6)

        # For chi = 1.0, Lambda = exp(-1) ~ 0.368
        Lambda_high = stz_density(1.0)
        assert jnp.isfinite(Lambda_high)
        assert jnp.isclose(Lambda_high, jnp.exp(-1.0), rtol=1e-6)

        # Lambda increases with chi
        chi_values = jnp.array([0.05, 0.1, 0.2, 0.5, 1.0])
        Lambda_values = jax.vmap(stz_density)(chi_values)
        assert jnp.all(jnp.diff(Lambda_values) > 0)  # Monotonically increasing

    def test_chi_evolution_saturation(self):
        """Test chi_evolution follows Langer (2008) saturation behavior."""
        chi_inf = 0.2
        sigma_y = 1e6
        c0 = 1.0
        stress = 1e6
        gamma_dot_pl = 1.0

        # If chi < chi_inf, dchi/dt should be positive
        chi_low = 0.1
        dchi_low = chi_evolution_langer2008(
            chi_low, chi_inf, gamma_dot_pl, stress, sigma_y, c0
        )
        assert dchi_low > 0, "Chi should increase when below steady state"

        # If chi > chi_inf, dchi/dt should be negative
        chi_high = 0.3
        dchi_high = chi_evolution_langer2008(
            chi_high, chi_inf, gamma_dot_pl, stress, sigma_y, c0
        )
        assert dchi_high < 0, "Chi should decrease when above steady state"

        # At saturation chi = chi_inf, dchi/dt should be 0
        dchi_eq = chi_evolution_langer2008(
            chi_inf, chi_inf, gamma_dot_pl, stress, sigma_y, c0
        )
        assert jnp.isclose(dchi_eq, 0.0, atol=1e-10)

    def test_lambda_evolution(self):
        """Test Lambda relaxation toward equilibrium."""
        chi = 0.2
        tau_relax = 1.0

        Lambda_eq = stz_density(chi)

        # If Lambda > Lambda_eq, dLambda/dt < 0 (decay)
        Lambda_high = Lambda_eq * 2.0
        dLambda_high = lambda_evolution(Lambda_high, chi, tau_relax)
        assert dLambda_high < 0

        # If Lambda < Lambda_eq, dLambda/dt > 0 (growth)
        Lambda_low = Lambda_eq * 0.5
        dLambda_low = lambda_evolution(Lambda_low, chi, tau_relax)
        assert dLambda_low > 0

        # At equilibrium, dLambda/dt = 0
        dLambda_eq = lambda_evolution(Lambda_eq, chi, tau_relax)
        assert jnp.isclose(dLambda_eq, 0.0, atol=1e-10)

    def test_stz_ode_rhs_shapes(self):
        """Test ODE vector field returns correct shapes for all variants."""
        # Common args
        args = {
            "gamma_dot": 1.0,
            "G0": 1e9,
            "sigma_y": 1e6,
            "tau0": 1e-12,
            "epsilon0": 0.1,
            "chi_inf": 0.2,
            "c0": 1.0,
            "tau_beta": 1.0,
            "m_inf": 0.1,
            "rate_m": 1.0,
        }

        t = 0.0

        # Minimal variant: [stress, chi]
        y_minimal = jnp.array([1e5, 0.1])
        dy_minimal = stz_ode_rhs(t, y_minimal, args)
        assert dy_minimal.shape == (2,)
        assert jnp.all(jnp.isfinite(dy_minimal))

        # Standard variant: [stress, chi, Lambda]
        y_standard = jnp.array([1e5, 0.1, 0.01])
        dy_standard = stz_ode_rhs(t, y_standard, args)
        assert dy_standard.shape == (3,)
        assert jnp.all(jnp.isfinite(dy_standard))

        # Full variant: [stress, chi, Lambda, m]
        y_full = jnp.array([1e5, 0.1, 0.01, 0.0])
        dy_full = stz_ode_rhs(t, y_full, args)
        assert dy_full.shape == (4,)
        assert jnp.all(jnp.isfinite(dy_full))

    def test_plastic_rate_physical(self):
        """Test plastic strain rate has correct physical behavior."""
        sigma_y = 1e6
        tau0 = 1e-12
        epsilon0 = 0.1
        chi = 0.15
        Lambda = stz_density(chi)

        # At zero stress, plastic rate should be zero (no driving force)
        d_pl_zero = plastic_rate(0.0, Lambda, chi, sigma_y, tau0, epsilon0)
        assert jnp.isclose(d_pl_zero, 0.0, atol=1e-20)

        # At positive stress, plastic rate should be positive
        d_pl_pos = plastic_rate(sigma_y, Lambda, chi, sigma_y, tau0, epsilon0)
        assert d_pl_pos > 0

        # At negative stress, plastic rate should be negative
        d_pl_neg = plastic_rate(-sigma_y, Lambda, chi, sigma_y, tau0, epsilon0)
        assert d_pl_neg < 0

        # Rate should increase with Lambda (more zones -> more flow)
        Lambda_high = Lambda * 10
        d_pl_high_lambda = plastic_rate(
            sigma_y, Lambda_high, chi, sigma_y, tau0, epsilon0
        )
        assert d_pl_high_lambda > d_pl_pos
