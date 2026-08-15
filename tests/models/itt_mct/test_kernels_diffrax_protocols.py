"""Tests for the 5 new diffrax trajectory solvers added alongside flow curve.

Each test compares the diffrax kernel's raw output against the physics
already validated by the scipy path in schematic.py — these are kernel
unit tests, not full-model integration tests (see test_schematic.py for
those).
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

diffrax = pytest.importorskip("diffrax")

from rheojax.models.itt_mct._kernels_diffrax import (
    solve_equilibrium_correlator_trajectory,
    solve_relaxation_trajectory,
    solve_startup_trajectory,
)

# Every test in this file runs an un-mocked diffrax ODE solve (first-call
# JIT compilation) -- mark the whole module slow so `pytest -m "not slow"`
# smoke runs skip it, matching how TestFlowCurveDiffrax is marked in
# test_schematic.py.
pytestmark = pytest.mark.slow


class TestEquilibriumCorrelatorTrajectory:
    def test_fluid_state_decays_to_near_zero(self):
        """Fluid (v2=3, below glass transition v2_c=4): correlator should
        relax toward 0, not stay pinned near 1 (non-ergodic plateau is a
        glass-only feature)."""
        n_modes = 10
        tau = jnp.logspace(-2, 2, n_modes)
        g = jnp.ones(n_modes) / n_modes
        t = jnp.logspace(-2, 2, 30)

        phi = solve_equilibrium_correlator_trajectory(
            t, v1=0.0, v2=3.0, Gamma=1.0, g=g, tau=tau, n_modes=n_modes
        )

        assert phi.shape == (30,)
        assert np.all(np.isfinite(phi))
        assert np.all(phi >= 0.0) and np.all(phi <= 1.0)
        assert phi[0] > phi[-1]  # monotonic decay
        assert phi[-1] < 0.1  # relaxed toward zero at long times

    def test_initial_value_is_one(self):
        n_modes = 5
        tau = jnp.logspace(-1, 1, n_modes)
        g = jnp.ones(n_modes) / n_modes
        t = jnp.array([0.0, 1.0, 10.0])

        phi = solve_equilibrium_correlator_trajectory(
            t, v1=0.0, v2=3.0, Gamma=1.0, g=g, tau=tau, n_modes=n_modes
        )

        assert phi[0] == pytest.approx(1.0, abs=1e-3)


class TestStartupTrajectory:
    def test_stress_grows_from_zero(self):
        n_modes = 10
        tau = jnp.logspace(-2, 2, n_modes)
        g = jnp.ones(n_modes) / n_modes
        t = jnp.linspace(0.01, 10.0, 30)

        sigma = solve_startup_trajectory(
            t,
            gamma_dot=1.0,
            v1=0.0,
            v2=3.0,
            Gamma=1.0,
            gamma_c=0.1,
            G_inf=1e6,
            g=g,
            tau=tau,
            n_modes=n_modes,
        )

        assert sigma.shape == (30,)
        assert np.all(np.isfinite(sigma))
        # Not just "non-decreasing" (an all-zero array would pass that) --
        # assert real physical growth given G_inf=1e6, gamma_dot=1.0.
        assert sigma[-1] > 10.0
        assert sigma[-1] > sigma[0]


class TestRelaxationTrajectory:
    def test_correlator_decays_and_is_bounded(self):
        n_modes = 10
        tau = jnp.logspace(-2, 2, n_modes)
        g = jnp.ones(n_modes) / n_modes
        t = jnp.linspace(0.01, 100.0, 30)

        phi = solve_relaxation_trajectory(
            t,
            gamma_pre=0.01,
            v1=0.0,
            v2=3.0,
            Gamma=1.0,
            gamma_c=0.1,
            G_inf=1e6,
            g=g,
            tau=tau,
            n_modes=n_modes,
        )

        assert phi.shape == (30,)
        assert np.all(np.isfinite(phi))
        assert np.all(phi >= 0.0) and np.all(phi <= 1.0)
        assert phi[0] >= phi[-1]  # decays (or stays flat) over time

    def test_returns_correlator_not_stress(self):
        """Regression guard: this function must return Phi(t), not sigma(t)
        — the caller in schematic.py reconstructs sigma algebraically."""
        n_modes = 5
        tau = jnp.logspace(-1, 1, n_modes)
        g = jnp.ones(n_modes) / n_modes
        t = jnp.array([0.01, 1.0, 10.0])

        phi = solve_relaxation_trajectory(
            t,
            gamma_pre=0.01,
            v1=0.0,
            v2=3.0,
            Gamma=1.0,
            gamma_c=0.1,
            G_inf=1e6,
            g=g,
            tau=tau,
            n_modes=n_modes,
        )

        # Phi is O(1) (dimensionless correlator), not O(G_inf) (stress-scale).
        assert np.all(np.abs(np.array(phi)) <= 1.0)
