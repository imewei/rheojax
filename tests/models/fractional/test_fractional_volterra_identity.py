"""Volterra convolution-identity regression tests for G(t)<->J(t) interconversion.

The linear-viscoelastic convolution identity
    int_0^t G(tau) J(t-tau) dtau = t
must hold for any model whose relaxation modulus and creep compliance both
correspond to the same underlying linear response. A 2026-08-10 audit found
four fractional models (documented in project memory
project_fractional_creep_heuristic_blends.md) that fabricated one side of the
G/J pair with an ad-hoc heuristic blend, violating this identity by 17%-1574%.
They now recover the fabricated side via a JIT-safe Prony/retardation-series
fit against the model's own exact complex modulus (see rheojax.utils.prony).
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.classical.maxwell import Maxwell
from rheojax.models.classical.springpot import SpringPot
from rheojax.models.fractional.fractional_burgers import FractionalBurgersModel
from rheojax.models.fractional.fractional_jeffreys import FractionalJeffreysModel
from rheojax.models.fractional.fractional_zener_ll import FractionalZenerLiquidLiquid
from rheojax.models.fractional.fractional_zener_sl import FractionalZenerSolidLiquid

jax, jnp = safe_import_jax()

# Times spanning ~2.5 decades either side of each model's tau=1.0 test
# parameter -- the "middle decades" where real rheological data lives.
_MIDDLE_DECADE_TIMES = [0.1, 0.3, 1.0, 3.0, 10.0]


def _convolution_ratio(G_fn, J_fn, t, n=2000, delta_weight=0.0):
    """``int_0^t G(tau) J(t-tau) dtau / t`` (should be ~1.0).

    Uses a ``tau = u**2`` substitution to resolve a ``G(tau) ~ tau^-alpha``
    endpoint singularity (present in all four fractional models under test)
    without needing a singularity-aware quadrature routine.
    """
    u = np.linspace(0.0, np.sqrt(t), n)
    taus = np.clip(u**2, 1e-12, t)
    G_vals = np.asarray(G_fn(jnp.asarray(taus)))
    J_vals = np.asarray(J_fn(jnp.asarray(np.clip(t - taus, 0.0, t))))
    smooth = np.trapezoid(G_vals * J_vals * 2 * u, u)
    delta = delta_weight * float(np.asarray(J_fn(jnp.asarray([t])))[0])
    return (smooth + delta) / t


class TestConvolutionIdentityControls:
    """Known-exact models calibrate the oracle's own quadrature noise floor."""

    def test_springpot_control(self):
        c_alpha, alpha = 5.0, 0.5
        G_fn = lambda t: SpringPot._predict_relaxation(t, c_alpha, alpha)  # noqa: E731
        J_fn = lambda t: SpringPot._predict_creep(t, c_alpha, alpha)  # noqa: E731
        for t in _MIDDLE_DECADE_TIMES:
            ratio = _convolution_ratio(G_fn, J_fn, t)
            assert ratio == pytest.approx(1.0, abs=0.02), f"springpot t={t}: {ratio}"

    def test_maxwell_control(self):
        G0, eta = 3.0, 6.0
        G_fn = lambda t: Maxwell._predict_relaxation(t, G0, eta)  # noqa: E731
        J_fn = lambda t: Maxwell._predict_creep(t, G0, eta)  # noqa: E731
        for t in _MIDDLE_DECADE_TIMES:
            ratio = _convolution_ratio(G_fn, J_fn, t)
            assert ratio == pytest.approx(1.0, abs=0.02), f"maxwell t={t}: {ratio}"


class TestFractionalVolterraIdentity:
    """Regression coverage for the four models fixed in project memory
    project_fractional_creep_heuristic_blends.md. Tolerances are loose
    relative to the controls above because the fitted retardation/Prony
    series is an approximation (not an elementary closed form) -- but far
    tighter than the 17%-1574% the heuristic blends scored.
    """

    def test_zener_sl_creep(self):
        Ge, c_alpha, alpha, tau = 2.0, 5.0, 0.5, 1.0
        G_fn = lambda t: FractionalZenerSolidLiquid._predict_relaxation(  # noqa: E731
            t, Ge, c_alpha, alpha, tau
        )
        J_fn = lambda t: FractionalZenerSolidLiquid._predict_creep(  # noqa: E731
            t, Ge, c_alpha, alpha, tau
        )
        for t in _MIDDLE_DECADE_TIMES:
            ratio = _convolution_ratio(G_fn, J_fn, t)
            assert ratio == pytest.approx(1.0, abs=0.05), f"FZSL t={t}: {ratio}"

    def test_zener_ll_creep(self):
        c1, c2, alpha, beta, gamma, tau = 4.0, 2.0, 0.3, 0.7, 0.4, 1.0
        G_fn = lambda t: FractionalZenerLiquidLiquid._predict_relaxation(  # noqa: E731
            t, c1, c2, alpha, beta, gamma, tau
        )
        J_fn = lambda t: FractionalZenerLiquidLiquid._predict_creep(  # noqa: E731
            t, c1, c2, alpha, beta, gamma, tau
        )
        for t in _MIDDLE_DECADE_TIMES:
            ratio = _convolution_ratio(G_fn, J_fn, t)
            assert ratio == pytest.approx(1.0, abs=0.05), f"FZLL t={t}: {ratio}"

    def test_jeffreys_creep(self):
        eta1, eta2, alpha, tau1 = 10.0, 3.0, 0.5, 1.0
        G_fn = lambda t: FractionalJeffreysModel._predict_relaxation(  # noqa: E731
            t, eta1, eta2, alpha, tau1
        )
        J_fn = lambda t: FractionalJeffreysModel._predict_creep(  # noqa: E731
            t, eta1, eta2, alpha, tau1
        )
        # G(t) here is only the smooth part; the model's own docstring notes
        # an unobservable eta1*r*delta(t) instantaneous-dashpot term that the
        # convolution identity must still include (see the module docstring
        # of fractional_jeffreys.py).
        r = (eta2 / eta1) ** alpha
        delta_weight = eta1 * r
        for t in _MIDDLE_DECADE_TIMES:
            ratio = _convolution_ratio(G_fn, J_fn, t, delta_weight=delta_weight)
            assert ratio == pytest.approx(1.0, abs=0.05), f"Jeffreys t={t}: {ratio}"

    def test_burgers_relaxation(self):
        Jg, eta1, Jk, alpha, tau_k = 1e-3, 50.0, 5e-3, 0.5, 1.0
        G_fn = lambda t: FractionalBurgersModel._predict_relaxation(  # noqa: E731
            t, Jg, eta1, Jk, alpha, tau_k
        )
        J_fn = lambda t: FractionalBurgersModel._predict_creep(  # noqa: E731
            t, Jg, eta1, Jk, alpha, tau_k
        )
        for t in _MIDDLE_DECADE_TIMES:
            ratio = _convolution_ratio(G_fn, J_fn, t)
            assert ratio == pytest.approx(1.0, abs=0.05), f"Burgers t={t}: {ratio}"

    def test_burgers_relaxation_decays_to_zero(self):
        """Burgers is a liquid (creep has an unbounded t/eta1 flow term), so
        G(t) -> 0 as t -> infinity. The Prony fit's equilibrium-modulus term
        must be pinned to 0 (liquid=True in fit_relaxation_prony_series), not
        merely clamped >= 0 -- an unconstrained least-squares solve over the
        fit's finite omega window can otherwise assign a spurious nonzero
        floor that best-fits that window but never decays, which the
        convolution-identity check above (restricted to the tested "middle
        decades") does not directly catch. See PR #110 review."""
        Jg, eta1, Jk, alpha, tau_k = 1e-3, 50.0, 5e-3, 0.5, 1.0
        G_fn = lambda t: FractionalBurgersModel._predict_relaxation(  # noqa: E731
            t, Jg, eta1, Jk, alpha, tau_k
        )
        G_near = float(G_fn(jnp.asarray([10.0]))[0])
        G_far = float(G_fn(jnp.asarray([1e6]))[0])
        assert G_far < 0.01 * G_near, f"G(1e6)={G_far} should be << G(10)={G_near}"
