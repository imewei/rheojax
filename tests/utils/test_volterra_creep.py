"""Tests for the linear-viscoelastic Volterra creep deconvolution solver.

Each test pins the solver against an analytic ``G(t) ↔ J(t)`` interconversion
or against the convolution identity it solves.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.utils.volterra_creep import solve_linear_creep_compliance


def _log_grid(t_min: float, t_max: float, n: int) -> np.ndarray:
    return np.logspace(np.log10(t_min), np.log10(t_max), n)


@pytest.mark.smoke
def test_maxwell_analytic():
    """Single Maxwell mode: G=G0 e^{-t/tau} → J=(1/G0)(1 + t/tau)."""
    G0, tau = 5.0, 2.0
    t = _log_grid(1e-3, 1e2, 400)
    G_t = G0 * np.exp(-t / tau)

    J = solve_linear_creep_compliance(t, G_t, G0=G0)
    J_exact = (1.0 / G0) * (1.0 + t / tau)

    rel = np.abs(J - J_exact) / J_exact
    assert np.max(rel) < 2e-2, f"max rel err {np.max(rel):.3e}"


@pytest.mark.smoke
def test_elastic_solid_constant_modulus():
    """Pure elastic solid: G(t)=G_inf const → J(t)=1/G_inf for all t."""
    G_inf = 3.0
    t = _log_grid(1e-3, 1e3, 200)
    G_t = np.full_like(t, G_inf)

    J = solve_linear_creep_compliance(t, G_t, G0=G_inf)
    assert np.allclose(J, 1.0 / G_inf, rtol=1e-6)


@pytest.mark.smoke
def test_instantaneous_compliance():
    """J(0+) = 1/G(0) is recovered at the earliest time."""
    G0, tau = 10.0, 1.0
    t = _log_grid(1e-4, 1e1, 300)
    G_t = G0 * np.exp(-t / tau)

    J = solve_linear_creep_compliance(t, G_t, G0=G0)
    # First sample is at t≈1e-4 « tau, so J≈1/G0.
    assert J[0] == pytest.approx(1.0 / G0, rel=5e-3)


def test_solid_plateau_limit():
    """Glass-like G(t) = G_inf + (G0-G_inf) e^{-t/tau} → J(∞) = 1/G_inf."""
    G0, G_inf, tau = 20.0, 4.0, 1.0
    t = _log_grid(1e-3, 1e3, 500)
    G_t = G_inf + (G0 - G_inf) * np.exp(-t / tau)

    J = solve_linear_creep_compliance(t, G_t, G0=G0)
    assert J[-1] == pytest.approx(1.0 / G_inf, rel=3e-2)
    assert J[0] == pytest.approx(1.0 / G0, rel=1e-2)
    # Monotone non-decreasing up to integrator noise scaled to compliance range.
    assert np.min(np.diff(J)) >= -1e-6 * (J.max() - J.min())


def test_fluid_viscous_limit():
    """Maxwell fluid: long-time J(t) → t/η with η = G0·tau = ∫G dt."""
    G0, tau = 2.0, 5.0
    eta = G0 * tau
    t = _log_grid(1e-2, 1e4, 600)
    G_t = G0 * np.exp(-t / tau)

    J = solve_linear_creep_compliance(t, G_t, G0=G0)
    # At t » tau the viscous term dominates.
    late = t > 50 * tau
    rel = np.abs(J[late] - t[late] / eta) / (t[late] / eta)
    assert np.max(rel) < 5e-2


def test_two_mode_convolution_identity():
    """Two-mode Prony G(t): solved J must satisfy ∫G(t-s)J(s)ds = t."""
    from rheojax.utils.volterra_creep import creep_compliance_from_prony

    g = np.array([3.0, 1.5])
    tau = np.array([0.5, 8.0])
    t = _log_grid(1e-3, 1e2, 400)

    # Solve exactly from the known Prony modes.
    J = creep_compliance_from_prony(t, g, tau, G_inf=0.0)

    # Verify ∫₀^{t_n} G(t_n - s) J(s) ds = t_n on a dense uniform grid.
    t_dense = np.linspace(0.0, t[-1], 40000)
    J_dense = creep_compliance_from_prony(
        np.concatenate(([0.0], t_dense[1:])), g, tau, G_inf=0.0
    )
    for idx in (150, 250, 350):
        tn = t[idx]
        mask = t_dense <= tn
        lag = tn - t_dense[mask]
        G_lag = (g[None, :] * np.exp(-lag[:, None] / tau[None, :])).sum(axis=1)
        lhs = np.trapezoid(G_lag * J_dense[mask], t_dense[mask])
        assert lhs == pytest.approx(tn, rel=3e-2)
    assert J.shape == t.shape


def test_uniform_grid_starting_at_zero():
    """Solver handles a uniform grid that includes t=0 (G0 inferred)."""
    G0, tau = 4.0, 1.0
    t = np.linspace(0.0, 10.0, 500)
    G_t = G0 * np.exp(-t / tau)

    J = solve_linear_creep_compliance(t, G_t)
    J_exact = (1.0 / G0) * (1.0 + t / tau)
    rel = np.abs(J[1:] - J_exact[1:]) / J_exact[1:]
    assert np.max(rel) < 2e-2


def test_input_validation():
    t = _log_grid(1e-3, 1e2, 50)
    G_t = np.exp(-t)
    with pytest.raises(ValueError):
        solve_linear_creep_compliance(t, G_t[:-1])  # shape mismatch
    with pytest.raises(ValueError):
        solve_linear_creep_compliance(t[::-1], G_t)  # not increasing
    with pytest.raises(ValueError):
        solve_linear_creep_compliance(t, G_t, G0=-1.0)  # bad G0
