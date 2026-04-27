"""Tests for Fluidity identifiability reporting and the structural
degeneracies it documents.

These tests serve two purposes:

1. Lock in the :py:meth:`FluidityBase.identifiability_check` API and the
   partition of parameters it returns for each protocol.

2. Numerically verify the relaxation scale degeneracy the method warns
   about: at ``gamma_dot = 0`` the transformation
   ``(G, f_eq, f_inf) -> (lambda*G, f_eq/lambda, f_inf/lambda)`` leaves
   the stress trajectory sigma(t) invariant to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from rheojax.models.fluidity import FluidityLocal, FluidityNonlocal
from rheojax.models.fluidity._kernels import fluidity_local_ode_rhs

# --- API tests ---------------------------------------------------------------


@pytest.mark.smoke
class TestIdentifiabilityAPI:
    """Locked-in behaviour for identifiability_check()."""

    def test_relaxation_partition(self):
        """Relaxation: theta identifiable, (G, f_eq, f_inf) degenerate,
        the rest inactive.
        """
        rep = FluidityLocal.identifiability_check("relaxation", verbose=False)
        assert rep["identifiable"] == ("theta",)
        assert set(rep["product_degenerate"]) == {"G", "f_eq", "f_inf"}
        assert set(rep["inactive"]) == {"tau_y", "K", "n_flow", "a", "n_rejuv"}
        # All 9 params accounted for, no overlaps.
        union = (
            set(rep["identifiable"])
            | set(rep["product_degenerate"])
            | set(rep["inactive"])
        )
        assert len(union) == 9

    def test_flow_curve_partition(self):
        """Flow curve: only HB params are identifiable; dynamic params inert."""
        rep = FluidityLocal.identifiability_check("flow_curve", verbose=False)
        assert set(rep["identifiable"]) == {"tau_y", "K", "n_flow"}
        assert rep["product_degenerate"] == ()
        assert set(rep["inactive"]) == {
            "G",
            "f_eq",
            "f_inf",
            "theta",
            "a",
            "n_rejuv",
        }

    def test_oscillation_saos_alias(self):
        """'saos' is an alias for 'oscillation'."""
        a = FluidityLocal.identifiability_check("oscillation", verbose=False)
        b = FluidityLocal.identifiability_check("saos", verbose=False)
        assert a == b
        assert set(a["identifiable"]) == {"G", "f_eq"}

    def test_startup_vs_creep_difference(self):
        """Startup: all 6 dynamic params identifiable (gamma_dot != 0).
        Creep: G is not identifiable from strain data (elastic jump usually not
        in the dataset).
        """
        startup = FluidityLocal.identifiability_check("startup", verbose=False)
        creep = FluidityLocal.identifiability_check("creep", verbose=False)
        assert "G" in startup["identifiable"]
        assert "G" in creep["inactive"]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown test_mode"):
            FluidityLocal.identifiability_check("not_a_protocol", verbose=False)

    def test_verbose_does_not_raise(self):
        """verbose=True runs the warning code path without raising."""
        result = FluidityLocal.identifiability_check("relaxation", verbose=True)
        assert "product_degenerate" in result
        assert len(result["product_degenerate"]) > 0


# --- Numerical degeneracy test -----------------------------------------------


def _simulate_relaxation(
    G: float,
    f_eq: float,
    f_inf: float,
    theta: float,
    sigma_0: float = 500.0,
    t_end: float = 100.0,
    n_points: int = 60,
) -> np.ndarray:
    """Integrate the relaxation ODE at gamma_dot=0 and return sigma(t).

    Uses the same kernel as FluidityLocal but with an explicit diffrax call so
    the test is independent of the full fit/predict plumbing.
    """
    import diffrax

    t_jax = jnp.asarray(np.linspace(0.0, t_end, n_points), dtype=jnp.float64)
    args = {
        "G": G,
        "f_eq": f_eq,
        "f_inf": f_inf,
        "theta": theta,
        "a": 0.0,  # silence rejuvenation term entirely
        "n_rejuv": 1.0,
        "gamma_dot": 0.0,
    }
    # State: [sigma, f]; start at [sigma_0, f_inf] as the model does.
    y0 = jnp.array([sigma_0, f_inf])
    term = diffrax.ODETerm(fluidity_local_ode_rhs)
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=float(t_jax[0]),
        t1=float(t_jax[-1]),
        dt0=0.001,
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=t_jax),
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        max_steps=1_000_000,
    )
    return np.asarray(sol.ys[:, 0])


@pytest.mark.smoke
class TestRelaxationScaleDegeneracy:
    """Numerical proof of the (G, f_eq, f_inf) scale degeneracy at gamma_dot=0."""

    # Parameters tuned so all three timescales (tau_fast = 1/(G*f_inf),
    # theta, tau_slow = 1/(G*f_eq)) sit inside the simulation window with
    # stress well above the integrator tolerance throughout.
    _G = 1e3
    _F_EQ = 1e-5
    _F_INF = 1e-4
    _THETA = 30.0
    _T_END = 500.0
    _SIGMA_0 = 500.0

    @pytest.mark.parametrize("lam", [0.1, 0.5, 2.0, 10.0, 100.0])
    def test_sigma_invariant_under_scale(self, lam: float):
        """sigma(t) must be invariant (to 1e-6 relative) under the transform
        (G, f_eq, f_inf) -> (lam*G, f_eq/lam, f_inf/lam) at gamma_dot=0.

        This is the exact mathematical degeneracy that makes single-protocol
        relaxation fits return physically meaningless G and f values while
        still achieving R^2 ~ 0.999.
        """
        sigma_ref = _simulate_relaxation(
            self._G,
            self._F_EQ,
            self._F_INF,
            self._THETA,
            sigma_0=self._SIGMA_0,
            t_end=self._T_END,
        )
        sigma_scaled = _simulate_relaxation(
            lam * self._G,
            self._F_EQ / lam,
            self._F_INF / lam,
            self._THETA,
            sigma_0=self._SIGMA_0,
            t_end=self._T_END,
        )

        # Compare where both are above the integrator's absolute tolerance.
        mask = sigma_ref > 1e-3 * sigma_ref[0]
        assert mask.sum() >= 5, "Not enough points above tolerance floor"
        rel_err = np.abs(sigma_scaled[mask] - sigma_ref[mask]) / np.abs(sigma_ref[mask])
        assert rel_err.max() < 1e-5, (
            f"Scale degeneracy violated at lambda={lam}: "
            f"max rel err = {rel_err.max():.3e}"
        )

    def test_theta_breaks_the_degeneracy(self):
        """Changing theta (at fixed G, f_eq, f_inf) must change sigma(t) —
        theta is the one relaxation parameter that is NOT part of the scale
        degeneracy, so identifiability_check lists it under 'identifiable'.
        """
        sigma_a = _simulate_relaxation(
            self._G,
            self._F_EQ,
            self._F_INF,
            theta=10.0,
            sigma_0=self._SIGMA_0,
            t_end=self._T_END,
        )
        sigma_b = _simulate_relaxation(
            self._G,
            self._F_EQ,
            self._F_INF,
            theta=100.0,
            sigma_0=self._SIGMA_0,
            t_end=self._T_END,
        )
        mask = sigma_a > 1e-3 * sigma_a[0]
        assert mask.sum() >= 5, "Not enough points above tolerance floor"
        rel_err = np.abs(sigma_a[mask] - sigma_b[mask]) / np.abs(sigma_a[mask])
        assert rel_err.max() > 0.05, (
            "theta must shift the trajectory non-trivially; "
            f"max rel diff = {rel_err.max():.3e}"
        )


# --- Nonlocal API + numerical inertness ---------------------------------------


@pytest.mark.smoke
class TestNonlocalIdentifiabilityAPI:
    """FluidityNonlocal overrides _IDENTIFIABILITY because its PDE kernels
    use HB-aging only (no rejuvenation) and carry an extra xi parameter.
    These tests lock in the override so future edits to the PDE RHS that
    change which parameters enter the residual must also update the map.
    """

    def test_creep_partition(self):
        rep = FluidityNonlocal.identifiability_check("creep", verbose=False)
        assert set(rep["identifiable"]) == {"tau_y", "K", "n_flow", "theta"}
        assert rep["product_degenerate"] == ()
        assert set(rep["inactive"]) == {"G", "f_eq", "f_inf", "a", "n_rejuv", "xi"}

    def test_flow_curve_partition(self):
        rep = FluidityNonlocal.identifiability_check("flow_curve", verbose=False)
        assert set(rep["identifiable"]) == {"tau_y", "K", "n_flow"}
        assert set(rep["inactive"]) == {
            "G",
            "f_eq",
            "f_inf",
            "theta",
            "a",
            "n_rejuv",
            "xi",
        }

    def test_startup_partition(self):
        rep = FluidityNonlocal.identifiability_check("startup", verbose=False)
        # Rate-controlled: G enters via dSigma/dt = G(gamma_dot - Sigma*f_avg);
        # HB params via f_loc; theta via relaxation rate.
        assert set(rep["identifiable"]) == {"G", "tau_y", "K", "n_flow", "theta"}
        assert set(rep["inactive"]) == {"f_eq", "f_inf", "a", "n_rejuv", "xi"}

    def test_rejuvenation_params_inactive_across_all_transients(self):
        """a, n_rejuv, f_inf never enter the nonlocal PDE RHS — assert
        they appear in 'inactive' for every transient protocol.
        """
        for mode in ("startup", "relaxation", "creep", "laos"):
            rep = FluidityNonlocal.identifiability_check(mode, verbose=False)
            for p in ("a", "n_rejuv", "f_inf"):
                assert p in rep["inactive"], (
                    f"{p!r} should be inactive in nonlocal {mode}; "
                    f"got identifiable={rep['identifiable']}"
                )


def _simulate_nonlocal_creep(
    model_params: dict,
    t: np.ndarray,
    sigma_applied: float,
    N_y: int = 11,
) -> np.ndarray:
    """Integrate FluidityNonlocal creep and return strain(t).

    Uses a small grid (N_y=11) so the test runs under smoke budget.
    """
    model = FluidityNonlocal(N_y=N_y, gap_width=1e-3)
    model.parameters.update(model_params, strict=True)
    return np.asarray(model.predict(t, test_mode="creep", sigma_applied=sigma_applied))


@pytest.mark.smoke
class TestNonlocalCreepInertParams:
    """Numerical proof that G, f_inf, a, n_rejuv, xi are inert for the
    nonlocal creep PDE.

    This is the regression test guarding the identifiability issue that
    produced a poor-looking fit in examples/fluidity/09_fluidity_nonlocal_creep.ipynb:
    the NLSQ optimizer could not move these parameters because their
    Jacobian columns are numerically zero, and the notebook silently
    presented the stuck values as "fitted".

    If a future RHS change makes any of these parameters active, the
    corresponding test case below will fail AND the _IDENTIFIABILITY
    override in FluidityNonlocal must also be updated.
    """

    _PARAMS_TRUTH = {
        "G": 1e4,
        "tau_y": 100.0,
        "K": 100.0,
        "n_flow": 0.5,
        "f_eq": 1e-4,
        "f_inf": 1e-3,
        "theta": 10.0,
        "a": 2.0,
        "n_rejuv": 1.0,
        "xi": 1e-5,
    }

    # Log-spaced time matching the notebook (t_end=500, n_time=30 is enough
    # to cover the transient and the steady flow regime).
    _T = np.logspace(-2, np.log10(500.0), 30)
    _SIGMA_ABOVE = 1.5 * _PARAMS_TRUTH["tau_y"]  # 150 Pa — above yield

    @pytest.mark.parametrize(
        "inert_param,scale",
        [
            ("G", 10.0),
            ("G", 0.1),
            ("f_inf", 10.0),
            ("f_inf", 0.1),
            ("a", 10.0),
            ("n_rejuv", 1.8),  # bounds (0, 2) -- 1.8x vs truth 1.0
            ("n_rejuv", 0.3),  # factor 0.3 within (0, 2) bounds
            ("xi", 10.0),
            ("xi", 0.1),
        ],
    )
    def test_inert_param_does_not_affect_creep(self, inert_param: str, scale: float):
        """Perturbing an inert parameter must leave creep strain(t) unchanged
        to ~1e-10 relative error (integrator tolerance floor).
        """
        gamma_ref = _simulate_nonlocal_creep(
            self._PARAMS_TRUTH,
            self._T,
            self._SIGMA_ABOVE,
        )

        perturbed = dict(self._PARAMS_TRUTH)
        perturbed[inert_param] = self._PARAMS_TRUTH[inert_param] * scale
        # Respect bounds: f_inf upper bound is 1e-3, clamp the 10× perturbation.
        if inert_param == "f_inf" and perturbed["f_inf"] > 1e-3:
            perturbed["f_inf"] = 1e-3
        # xi lower bound is 1e-9; 0.1× of 1e-5 is still fine.

        gamma_perturbed = _simulate_nonlocal_creep(
            perturbed,
            self._T,
            self._SIGMA_ABOVE,
        )

        # Compare where reference strain is large enough to avoid dividing
        # by near-zero initial samples (γ(t0)=0 by construction).
        mask = gamma_ref > 1e-6
        assert mask.sum() >= 5, "Not enough points above strain floor"
        rel_err = np.abs(gamma_perturbed[mask] - gamma_ref[mask]) / np.abs(
            gamma_ref[mask]
        )

        # PDE integrator tolerance is ~1e-5 rtol / 1e-7 atol, so 1e-4 is a
        # generous margin while still catching any sub-1% leakage.
        assert rel_err.max() < 1e-4, (
            f"Perturbing {inert_param!r} by {scale}× changed creep strain "
            f"by max rel err {rel_err.max():.3e}; expected < 1e-4. "
            f"Either the PDE RHS changed (update _IDENTIFIABILITY) or the "
            f"test fixture is incorrect."
        )

    def test_identifiable_params_do_affect_creep(self):
        """Sanity check: tau_y and theta (both in the identifiable set)
        MUST shift the trajectory non-trivially. If they don't, the test
        above is vacuous because the reference simulation isn't exercising
        the physics we claim it does.
        """
        gamma_ref = _simulate_nonlocal_creep(
            self._PARAMS_TRUTH,
            self._T,
            self._SIGMA_ABOVE,
        )

        # 2× tau_y (100 → 200) moves it above the applied stress (150),
        # so the system transitions from above-yield to below-yield flow.
        tau_perturbed = dict(self._PARAMS_TRUTH, tau_y=200.0)
        gamma_tau = _simulate_nonlocal_creep(
            tau_perturbed,
            self._T,
            self._SIGMA_ABOVE,
        )
        mask = gamma_ref > 1e-4
        rel_tau = np.abs(gamma_tau[mask] - gamma_ref[mask]) / np.abs(gamma_ref[mask])
        assert rel_tau.max() > 0.5, (
            "Doubling tau_y past sigma_applied must change strain substantially; "
            f"max rel diff = {rel_tau.max():.3e}"
        )

        # 5× theta slows the fluidity relaxation → less strain at intermediate t.
        theta_perturbed = dict(self._PARAMS_TRUTH, theta=50.0)
        gamma_theta = _simulate_nonlocal_creep(
            theta_perturbed,
            self._T,
            self._SIGMA_ABOVE,
        )
        rel_theta = np.abs(gamma_theta[mask] - gamma_ref[mask]) / np.abs(
            gamma_ref[mask]
        )
        assert rel_theta.max() > 0.05, (
            "5× theta must shift the trajectory non-trivially; "
            f"max rel diff = {rel_theta.max():.3e}"
        )
