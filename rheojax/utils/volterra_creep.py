"""Linear-viscoelastic creep compliance via the Volterra history integral.

In linear viscoelasticity the relaxation modulus ``G(t)`` and the creep
compliance ``J(t)`` are tied by the convolution identity

    ∫₀ᵗ G(t - s) · J(s) ds = t,     t ≥ 0,

equivalently ``Ĝ(s)·Ĵ(s) = 1/s²`` in Laplace space. Given ``G(t)`` this is a
Volterra integral equation of the *first kind* for ``J(t)``.

Solving it directly by trapezoidal product integration is only accurate when the
time grid resolves the relaxation kernel everywhere — which fails across the
many decades typical of glassy/MCT moduli. Instead, when ``G`` is a Prony series

    G(t) = G_∞ + Σᵢ gᵢ · exp(-t/τᵢ),

the history integral collapses onto auxiliary memory states. Differentiating the
identity gives the instantaneous modulus ``G(0) = G_∞ + Σ gᵢ`` and

    G(0)·J(t) - Σᵢ (gᵢ/τᵢ)·Hᵢ(t) = 1,      Hᵢ(t) = ∫₀ᵗ e^{-(t-s)/τᵢ} J(s) ds,

where each history state obeys the linear ODE ``dHᵢ/dt = -Hᵢ/τᵢ + J(t)`` with
``Hᵢ(0) = 0``. The states ``Hᵢ`` *are* the history integrals, evaluated exactly
(no decade-spanning quadrature error). The steady state reproduces both limits:
``J(∞)·G_∞ = 1`` for a solid (plateau ``1/G_∞``) and, when ``G_∞ → 0``, viscous
growth ``J(t) → t/η`` with ``η = Σ gᵢ τᵢ = ∫₀^∞ G dt``.

This is the same Prony-mode reduction the ITT-MCT correlator solver uses for its
own Volterra equation, so a single ``G(t)`` yields a creep compliance that is the
exact convolution-inverse of the modulus whose Fourier transform gives ``G*(ω)``
— keeping all linear-response protocols mutually consistent (Kramers-Kronig).

References
----------
Tschoegl N. W. (1989) "The Phenomenological Theory of Linear Viscoelastic
    Behavior", Springer — Ch. 3 (interconversion of G and J).
Hopkins I. L. & Hamming R. W. (1957) J. Appl. Phys. 28, 906.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import nnls

__all__ = [
    "creep_compliance_from_prony",
    "solve_linear_creep_compliance",
]


def creep_compliance_from_prony(
    t: np.ndarray,
    g: np.ndarray,
    tau: np.ndarray,
    G_inf: float = 0.0,
) -> np.ndarray:
    """Creep compliance ``J(t)`` for a Prony-series relaxation modulus.

    Solves the linear-viscoelastic Volterra identity ``∫₀ᵗ G(t-s)J(s)ds = t``
    for ``G(t) = G_∞ + Σ gᵢ e^{-t/τᵢ}`` by integrating the history-state ODE
    system (see module docstring). Exact for exponential kernels.

    Parameters
    ----------
    t : np.ndarray
        Non-negative, strictly increasing output times (s), shape ``(N,)``.
    g : np.ndarray
        Prony weights ``gᵢ`` (Pa), shape ``(M,)``, all ``≥ 0``.
    tau : np.ndarray
        Prony relaxation times ``τᵢ`` (s), shape ``(M,)``, all ``> 0``.
    G_inf : float, default 0.0
        Equilibrium (plateau) modulus ``G_∞`` (Pa), ``≥ 0``. Non-zero ⇒ solid.

    Returns
    -------
    np.ndarray
        Creep compliance ``J(t)`` (1/Pa) on ``t``.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    g = np.asarray(g, dtype=np.float64).ravel()
    tau = np.asarray(tau, dtype=np.float64).ravel()

    if g.shape != tau.shape:
        raise ValueError(
            f"g and tau must have the same shape; got {g.shape} and {tau.shape}"
        )
    if np.any(tau <= 0.0):
        raise ValueError("All tau must be positive.")
    if np.any(t < 0.0):
        raise ValueError("t must be non-negative.")
    if t.size >= 2 and np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing.")

    G0 = float(G_inf) + float(g.sum())
    if not np.isfinite(G0) or G0 <= 0.0:
        raise ValueError(f"G(0) = G_inf + sum(g) must be positive; got {G0}.")

    inv_tau = 1.0 / tau
    coupling = g * inv_tau  # gᵢ/τᵢ, contracts with Hᵢ to form J

    def _rhs(_t: float, H: np.ndarray) -> np.ndarray:
        J = (1.0 + np.dot(coupling, H)) / G0
        return -H * inv_tau + J

    t_max = float(t[-1])
    if t_max <= 0.0:
        # Degenerate single-point request at t=0: elastic compliance only.
        return np.full_like(t, 1.0 / G0)

    sol = solve_ivp(
        _rhs,
        (0.0, t_max),
        y0=np.zeros_like(g),
        t_eval=t,
        method="LSODA",
        rtol=1e-8,
        atol=1e-12,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Creep history ODE failed to integrate: {sol.message}")

    H = sol.y  # shape (M, N)
    J = (1.0 + coupling @ H) / G0
    return np.asarray(J, dtype=np.float64)


def _prony_fit_modulus(
    t: np.ndarray,
    G_t: np.ndarray,
    n_per_decade: int = 8,
    max_modes: int = 60,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Non-negative least-squares Prony fit ``G(t) ≈ G_∞ + Σ gᵢ e^{-t/τᵢ}``.

    Relaxation times are log-spaced across the data window (with a margin on
    both ends). The constant column captures ``G_∞``.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    G_t = np.asarray(G_t, dtype=np.float64).ravel()

    t_pos = t[t > 0.0]
    t_min = float(t_pos.min()) if t_pos.size else float(t[1] if t.size > 1 else 1.0)
    t_max = float(t.max())

    tau_lo = t_min / 3.0
    tau_hi = t_max * 3.0
    n_dec = max(1.0, np.log10(tau_hi / tau_lo))
    n_modes = int(np.clip(np.ceil(n_per_decade * n_dec), 4, max_modes))
    tau = np.logspace(np.log10(tau_lo), np.log10(tau_hi), n_modes)

    # Design matrix: [exp(-t/tau_i) ... | 1] so the last coefficient is G_inf.
    A = np.empty((t.size, n_modes + 1), dtype=np.float64)
    A[:, :n_modes] = np.exp(-t[:, None] / tau[None, :])
    A[:, n_modes] = 1.0

    coeffs, _ = nnls(A, G_t)
    g = coeffs[:n_modes]
    G_inf = float(coeffs[n_modes])

    # Drop numerically dead modes for a leaner ODE system.
    keep = g > (g.max() * 1e-10) if g.size and g.max() > 0 else np.ones_like(g, bool)
    return g[keep], tau[keep], G_inf


def solve_linear_creep_compliance(
    t: np.ndarray,
    G_t: np.ndarray,
    G0: float | None = None,
) -> np.ndarray:
    """Creep compliance ``J(t)`` from a sampled relaxation modulus ``G(t)``.

    Fits ``G_t`` to a non-negative Prony series and inverts the Volterra
    identity exactly via :func:`creep_compliance_from_prony`. Robust across many
    decades because no convolution quadrature is performed on the (possibly
    coarse, log-spaced) input grid.

    Parameters
    ----------
    t : np.ndarray
        Non-negative, strictly increasing time grid (s), shape ``(N,)``.
    G_t : np.ndarray
        Relaxation modulus sampled on ``t`` (Pa), shape ``(N,)``.
    G0 : float, optional
        Ignored; retained for backward compatibility. The instantaneous modulus
        is set by the Prony fit (``G_∞ + Σ gᵢ``).

    Returns
    -------
    np.ndarray
        Creep compliance ``J(t)`` (1/Pa) on ``t``.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    G_t = np.asarray(G_t, dtype=np.float64).ravel()
    if t.shape != G_t.shape:
        raise ValueError(
            f"t and G_t must have the same shape; got {t.shape} and {G_t.shape}"
        )
    if t.size < 2:
        raise ValueError("Need at least two time points to solve for J(t).")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing.")
    if G0 is not None and G0 <= 0.0:
        raise ValueError(f"G0 must be positive when provided; got {G0}.")

    g, tau, G_inf = _prony_fit_modulus(t, G_t)
    if g.size == 0:
        raise ValueError("Prony fit produced no relaxation modes (degenerate G).")
    return creep_compliance_from_prony(t, g, tau, G_inf=G_inf)
