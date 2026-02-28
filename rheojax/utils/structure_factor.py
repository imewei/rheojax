"""Structure factor computation utilities for MCT models.

This module provides functions for computing static structure factors S(k)
needed for full k-resolved MCT/ITT-MCT calculations.

Functions
---------
percus_yevick_sk
    Compute S(k) for hard spheres using Percus-Yevick approximation
interpolate_sk
    Interpolate user-provided S(k) data to required k-grid
sk_derivatives
    Compute derivatives dS/dk needed for MCT vertex functions
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import interpax
import numpy as np
from scipy.interpolate import CubicSpline

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

if TYPE_CHECKING:
    import jax

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


# =============================================================================
# Percus-Yevick Structure Factor
# =============================================================================


def percus_yevick_sk(
    k: np.ndarray,
    phi: float,
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute structure factor S(k) for hard spheres via Percus-Yevick.

    The Percus-Yevick (PY) approximation provides an analytic solution for
    the structure factor of hard spheres:

        S(k) = 1 / [1 - n * c(k)]

    where c(k) is the Fourier transform of the direct correlation function
    and n = 6φ/(πσ³) is the number density.

    Parameters
    ----------
    k : np.ndarray
        Wave vector magnitudes (1/m or dimensionless if σ=1)
    phi : float
        Volume fraction φ ∈ (0, 0.64) for hard spheres
    sigma : float, default 1.0
        Hard sphere diameter (m)

    Returns
    -------
    np.ndarray
        Structure factor S(k) at the given k values

    Notes
    -----
    Valid for φ < φ_rcp ≈ 0.64 (random close packing).
    The glass transition in MCT occurs at φ ≈ 0.516 for hard spheres.

    The PY solution has the analytic form (Wertheim 1963):

        c(k) = -4πσ³ ∫₀^σ r² [c(r)] sin(kr)/(kr) dr

    where c(r) for r < σ is given by polynomial coefficients that depend
    on the volume fraction.

    References
    ----------
    Wertheim M.S. (1963) Phys. Rev. Lett. 10, 321
    Hansen J.P. & McDonald I.R. (2013) "Theory of Simple Liquids", 4th ed.
    """
    k = np.asarray(k, dtype=np.float64)

    # KRN-013: Validate phi to prevent division by zero at phi >= 1
    if phi <= 0 or phi >= 1:
        raise ValueError(
            f"Volume fraction phi must satisfy 0 < phi < 1, got phi={phi}. "
            f"For hard spheres, physically valid range is phi < 0.64."
        )

    q = k * sigma  # Dimensionless wave vector

    # Percus-Yevick coefficients
    eta = phi  # Often use eta = phi for PY

    # Coefficients for direct correlation function c(r)
    # c(r) = -(a + b*r/σ + c*(r/σ)³) for r < σ, 0 otherwise
    denom = (1 - eta) ** 4
    a_coeff = (1 + 2 * eta) ** 2 / denom
    b_coeff = -6 * eta * (1 + eta / 2) ** 2 / denom
    c_coeff = eta * a_coeff / 2

    # Fourier transform of c(r) - analytic result
    # Avoid division by zero at k=0. Use 1.0 (not 1e-10) so q^6 stays finite.
    q_safe = np.where(np.abs(q) < 1e-10, 1.0, q)
    q2 = q_safe * q_safe
    q3 = q2 * q_safe
    q4 = q2 * q2
    q6 = q3 * q3

    sin_q = np.sin(q_safe)
    cos_q = np.cos(q_safe)

    # The Fourier transform involves these terms
    # From Wertheim/Hansen-McDonald derivation
    term1 = a_coeff * (sin_q - q_safe * cos_q) / q3
    term2 = b_coeff * ((2 * q_safe * sin_q + (2 - q2) * cos_q - 2) / q4)
    term3 = c_coeff * (
        (-q4 * cos_q + 4 * ((3 * q2 - 6) * cos_q + (q3 - 6 * q_safe) * sin_q + 6)) / q6
    )

    # Direct correlation function in k-space
    c_k = -24 * eta * (term1 + term2 + term3)

    # Structure factor (guard against divergence when c_k → 1 near MCT transition)
    S_k = 1.0 / np.maximum(1.0 - c_k, 1e-10)

    # Handle k=0 limit: S(0) = (1-η)⁴/(1+2η)² (PY compressibility route)
    S_0 = (1 - eta) ** 4 / (1 + 2 * eta) ** 2
    S_k = np.where(np.abs(q) < 1e-10, S_0, S_k)

    return S_k


@partial(jax.jit, static_argnames=())
def percus_yevick_sk_jax(
    k: jax.Array,
    phi: float,
    sigma: float = 1.0,
) -> jax.Array:
    """JAX-compatible Percus-Yevick S(k) computation.

    Parameters
    ----------
    k : jax.Array
        Wave vector magnitudes
    phi : float
        Volume fraction
    sigma : float, default 1.0
        Particle diameter

    Returns
    -------
    jax.Array
        Structure factor S(k)
    """
    q = k * sigma
    eta = phi

    denom = (1 - eta) ** 4
    a_coeff = (1 + 2 * eta) ** 2 / denom
    b_coeff = -6 * eta * (1 + eta / 2) ** 2 / denom
    c_coeff = eta * a_coeff / 2

    # Safe division: use 1.0 (not 1e-10) so the masked branch stays finite.
    # jnp.where evaluates both branches — q_safe=1e-10 → q6=1e-60 → inf
    # in the unused branch, which corrupts VJP gradients.
    q_safe = jnp.where(jnp.abs(q) < 1e-10, 1.0, q)
    q2 = q_safe * q_safe
    q3 = q2 * q_safe
    q4 = q2 * q2
    q6 = q3 * q3

    sin_q = jnp.sin(q_safe)
    cos_q = jnp.cos(q_safe)

    term1 = a_coeff * (sin_q - q_safe * cos_q) / q3
    term2 = b_coeff * ((2 * q_safe * sin_q + (2 - q2) * cos_q - 2) / q4)
    term3 = c_coeff * (
        (-q4 * cos_q + 4 * ((3 * q2 - 6) * cos_q + (q3 - 6 * q_safe) * sin_q + 6)) / q6
    )

    c_k = -24 * eta * (term1 + term2 + term3)
    S_k = 1.0 / jnp.maximum(1.0 - c_k, 1e-10)

    # k=0 limit: PY compressibility route
    S_0 = (1 - eta) ** 4 / (1 + 2 * eta) ** 2
    S_k = jnp.where(jnp.abs(q) < 1e-10, S_0, S_k)

    return S_k


# =============================================================================
# Structure Factor Interpolation
# =============================================================================


def interpolate_sk(
    k_data: np.ndarray,
    sk_data: np.ndarray,
    k_target: np.ndarray,
    extrapolation: str = "constant",
) -> np.ndarray:
    """Interpolate user-provided S(k) data to target k-grid.

    Parameters
    ----------
    k_data : np.ndarray
        Wave vectors at which S(k) data is provided
    sk_data : np.ndarray
        Structure factor values S(k)
    k_target : np.ndarray
        Wave vectors at which interpolated S(k) is needed
    extrapolation : str, default "constant"
        Extrapolation method: "constant", "linear", or "error"

    Returns
    -------
    np.ndarray
        Interpolated S(k) at target k values

    Raises
    ------
    ValueError
        If extrapolation="error" and k_target extends beyond k_data range
    """
    k_data = np.asarray(k_data)
    sk_data = np.asarray(sk_data)
    k_target = np.asarray(k_target)

    # Sort by k
    sort_idx = np.argsort(k_data)
    k_data = k_data[sort_idx]
    sk_data = sk_data[sort_idx]

    # Check extrapolation bounds
    k_min, k_max = k_data.min(), k_data.max()
    out_of_bounds = (k_target < k_min) | (k_target > k_max)

    if extrapolation == "error" and np.any(out_of_bounds):
        raise ValueError(
            f"k_target has values outside data range [{k_min}, {k_max}]. "
            "Provide more S(k) data or use extrapolation='constant'."
        )

    # Cubic spline interpolation
    spline = CubicSpline(k_data, sk_data, extrapolate=True)
    sk_interp = spline(k_target)

    # Handle extrapolation
    if extrapolation == "constant":
        sk_interp = np.where(k_target < k_min, sk_data[0], sk_interp)
        sk_interp = np.where(k_target > k_max, sk_data[-1], sk_interp)

    # Ensure S(k) > 0
    sk_interp = np.maximum(sk_interp, 1e-10)

    return sk_interp


def create_sk_interpolator(
    k_data: np.ndarray,
    sk_data: np.ndarray,
) -> CubicSpline:
    """Create a cubic spline interpolator for S(k).

    Parameters
    ----------
    k_data : np.ndarray
        Wave vectors at which S(k) is provided
    sk_data : np.ndarray
        Structure factor values

    Returns
    -------
    CubicSpline
        Interpolator object that can be called as sk_spline(k)
    """
    sort_idx = np.argsort(k_data)
    return CubicSpline(k_data[sort_idx], sk_data[sort_idx], extrapolate=True)


def create_sk_interpolator_jax(
    k_data: np.ndarray,
    sk_data: np.ndarray,
) -> interpax.Interpolator1D:
    """Create a JAX-compatible cubic spline interpolator for S(k).

    Uses interpax for JIT-compatible, differentiable interpolation.
    Suitable for use inside jax.jit-decorated functions.

    Parameters
    ----------
    k_data : np.ndarray
        Wave vectors at which S(k) is provided (sorted ascending)
    sk_data : np.ndarray
        Structure factor values

    Returns
    -------
    interpax.Interpolator1D
        JAX-native interpolator callable as sk_interp(k)
    """
    sort_idx = np.argsort(k_data)
    k_sorted = jnp.asarray(k_data[sort_idx])
    sk_sorted = jnp.asarray(sk_data[sort_idx])
    return interpax.Interpolator1D(k_sorted, sk_sorted, method="cubic")


# =============================================================================
# Structure Factor Derivatives
# =============================================================================


def sk_derivatives(
    k: np.ndarray,
    sk: np.ndarray,
    method: str = "finite_diff",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute derivatives dS/dk and d²S/dk² for MCT vertex functions.

    The MCT vertex V(k,q,|k-q|) requires derivatives of S(k) for accurate
    evaluation. These appear in the vertex through the direct correlation
    function c(k) ∝ (1 - 1/S(k)).

    Parameters
    ----------
    k : np.ndarray
        Wave vector array
    sk : np.ndarray
        Structure factor values S(k)
    method : str, default "finite_diff"
        Method: "finite_diff" or "spline"

    Returns
    -------
    dsk_dk : np.ndarray
        First derivative dS/dk
    d2sk_dk2 : np.ndarray
        Second derivative d²S/dk²
    """
    k = np.asarray(k)
    sk = np.asarray(sk)

    if method == "spline":
        spline = CubicSpline(k, sk)
        dsk_dk = spline(k, nu=1)
        d2sk_dk2 = spline(k, nu=2)
    else:
        # Central finite differences (using np.gradient which handles spacing internally)
        # First derivative
        dsk_dk = np.gradient(sk, k)

        # Second derivative
        d2sk_dk2 = np.gradient(dsk_dk, k)

    return dsk_dk, d2sk_dk2


# =============================================================================
# MCT Vertex Functions
# =============================================================================


def mct_vertex_isotropic(
    k: np.ndarray,
    q: np.ndarray,
    phi: float,
    sk_func: Callable | None = None,
    sigma: float = 1.0,
    n_quad: int = 8,
) -> np.ndarray:
    """Compute isotropic MCT vertex V(k,q) after angular integration.

    Integrates the MCT coupling over the angle between k and q using
    Gauss-Legendre quadrature on the triangle-constraint interval for
    p = |k-q_vec|, which runs from p_min = |k-q| to p_max = k+q via
    the law of cosines.

    The coupling vertex at each angle (Götze 2009, Franosch et al. 1997)
    is:

        C(k, q, p) = alpha_q(k,q,p) * c(q) + alpha_p(k,q,p) * c(p)

    where the geometric projection coefficients are:

        alpha_q(k,q,p) = (k² + q² - p²) / (2k)
        alpha_p(k,q,p) = (k² - q² + p²) / (2k)

    and c(k) = (1 - 1/S(k)) / n_density is the direct correlation
    function.  The p-integral contributes to V(k,q) as:

        I(k,q) = ∫_{|k-q|}^{k+q} dp · p · S(p) · C(k,q,p)²

    The full discretised vertex (absorbing S(q), n_density, and the
    q-integration weight q² Δq) is:

        V(k, q) = [n_density * S(k) / (32π²k⁴)] * q² * Δq * S(q) * I(k,q)

    The outer q-sum then gives the MCT memory kernel contribution:

        m(k,t) = ∑_q V(k,q) * Φ(q,t)²

    Parameters
    ----------
    k : np.ndarray
        Wave vector magnitudes (1D array of length n_k)
    q : np.ndarray
        Second wave vector magnitudes (same length as k, uniform spacing)
    phi : float
        Volume fraction
    sk_func : callable, optional
        Function sk_func(k) returning S(k). If None, uses Percus-Yevick.
    sigma : float, default 1.0
        Hard-sphere diameter; used for n_density and the default sk_func.
    n_quad : int, default 8
        Number of Gauss-Legendre quadrature points for the p-integral.

    Returns
    -------
    np.ndarray
        Vertex V(k,q) as 2D array of shape (n_k, n_k).  Non-negative.

    Notes
    -----
    The loop over n_quad quadrature points (8 iterations) is explicit;
    all (n_k, n_k) pairs are handled in a single vectorized NumPy
    operation per iteration.  sk_func is called once per GL point on a
    raveled (n_k * n_k) array — O(n_k²) per quadrature point.

    Degenerate triangles (k=0 or q=0) give p_min = p_max, so
    half_range = 0 and the integral is zero.  The k=0 row is also
    zeroed explicitly via the k⁴ prefactor guard.

    References
    ----------
    Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids", OUP,
      eq. 2.73 (angular integration of MCT vertex).
    Franosch T. et al. (1997) Phys. Rev. E 55, 7153.
    """
    if sk_func is None:

        def sk_func(kk, _sigma=sigma):
            return percus_yevick_sk(kk, phi, sigma=_sigma)

    k = np.asarray(k, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n_k = len(k)

    # Number density: n = 6φ / (π σ³)
    n_density = 6.0 * phi / (np.pi * sigma**3)

    # q-spacing for the uniform integration grid (Δq in the q-sum weight)
    dq = float(q[1] - q[0]) if n_k > 1 else float(q[0])

    # -------------------------------------------------------------------------
    # On-grid structure factors and direct correlation functions
    # -------------------------------------------------------------------------
    sk = sk_func(k)  # (n_k,)
    sq = sk_func(q)  # (n_k,)

    # c(k) = (1 - 1/S(k)) / n_density  (R3-U-006: includes number density)
    cq = (1.0 - 1.0 / np.maximum(sq, 1e-30)) / n_density  # (n_k,)

    # -------------------------------------------------------------------------
    # Gauss-Legendre nodes and weights on [-1, 1]
    # -------------------------------------------------------------------------
    xi, w_gl = np.polynomial.legendre.leggauss(n_quad)
    # xi shape (n_quad,); w_gl shape (n_quad,); sum(w_gl) = 2.0

    # -------------------------------------------------------------------------
    # Broadcast shapes for vectorised (n_k, n_k) computation
    # -------------------------------------------------------------------------
    ki = k[:, None]  # (n_k, 1)
    qj = q[None, :]  # (1, n_k)

    # Triangle-constraint integration limits (Götze 2009 eq. 2.73)
    p_min = np.abs(ki - qj)  # (n_k, n_k)
    p_max = ki + qj  # (n_k, n_k)

    # Change-of-variables p = p_mid + half_range * xi_l, xi_l ∈ [-1, 1]
    half_range = 0.5 * (p_max - p_min)  # (n_k, n_k); = 0 when k=0 or q=0
    p_mid = 0.5 * (p_max + p_min)  # (n_k, n_k)

    # Broadcast S(q) and c(q) — constant across the p-integral for fixed (k,q)
    sq_2d = sq[None, :]  # (1, n_k)
    cq_2d = cq[None, :]  # (1, n_k)

    # Guard 1/(2k) projection denominators; k=0 row is zeroed at the end
    k_safe = np.maximum(ki, 1e-30)  # (n_k, 1)
    ki_sq = ki * ki  # (n_k, 1)
    qj_sq = qj * qj  # (1, n_k)

    # -------------------------------------------------------------------------
    # Gauss-Legendre quadrature over p ∈ [p_min, p_max]:
    #   I(k,q) = ∑_l w_l * half_range * p_l * S(p_l) * C(k,q,p_l)²
    # -------------------------------------------------------------------------
    integral = np.zeros((n_k, n_k), dtype=np.float64)

    for xi_l, w_l in zip(xi, w_gl, strict=True):
        # Quadrature abscissa in physical units: shape (n_k, n_k)
        p_l = p_mid + half_range * xi_l

        # S(p_l): batched sk_func call via ravel/reshape (O(n_k²) per GL point)
        s_pl = sk_func(p_l.ravel()).reshape(n_k, n_k)  # (n_k, n_k)

        # c(p_l) = (1 - 1/S(p_l)) / n_density
        c_pl = (1.0 - 1.0 / np.maximum(s_pl, 1e-30)) / n_density  # (n_k, n_k)

        # Geometric projection coefficients (law of cosines):
        #   alpha_q = (k² + q² - p²) / (2k)   [projection onto q direction]
        #   alpha_p = (k² - q² + p²) / (2k)   [projection onto p direction]
        p_l_sq = p_l * p_l  # (n_k, n_k)
        alpha_q = (ki_sq + qj_sq - p_l_sq) / (2.0 * k_safe)  # (n_k, n_k)
        alpha_p = (ki_sq - qj_sq + p_l_sq) / (2.0 * k_safe)  # (n_k, n_k)

        # MCT coupling vertex: C(k,q,p) = alpha_q * c(q) + alpha_p * c(p)
        C_kqp = alpha_q * cq_2d + alpha_p * c_pl  # (n_k, n_k)

        # Integrand: p * S(q) * S(p) * C(k,q,p)²
        integrand_l = p_l * sq_2d * s_pl * C_kqp * C_kqp  # (n_k, n_k)

        # Accumulate: Jacobian of variable transform is half_range
        integral += w_l * half_range * integrand_l

    # -------------------------------------------------------------------------
    # Assemble the discretised vertex matrix:
    #   V(k,q) = [n_density * S(k) / (32π²k⁴)] * q² * Δq * S(q) * I(k,q)
    #
    # The q² * Δq factor is the q-integration weight so that
    #   m(k,t) = ∑_q V(k,q) * Φ(q,t)²
    # is a plain matrix-vector product (no extra weights needed at call sites).
    # -------------------------------------------------------------------------
    k4_safe = np.maximum(ki_sq * ki_sq, 1e-30)  # (n_k, 1), guards k=0

    V = (
        (n_density * sk[:, None] / (32.0 * np.pi**2 * k4_safe))
        * (qj_sq * dq)  # q² Δq integration weight
        * integral  # p-integral I(k,q), already contains S(q) factor
    )

    # Zero out k=0 row explicitly (no coupling at zero wavevector)
    k_zero_mask = k < 1e-30  # (n_k,)
    V[k_zero_mask, :] = 0.0

    # Numerical guard: C² and S > 0 guarantee non-negativity in exact arithmetic;
    # floating-point noise near S(k) → ∞ (MCT glass transition) can produce
    # tiny negative values — clamp to zero.
    V = np.maximum(V, 0.0)

    return V


# =============================================================================
# Hard Sphere Properties
# =============================================================================


def hard_sphere_properties(phi: float) -> dict:
    """Compute thermodynamic properties for hard spheres at given φ.

    Parameters
    ----------
    phi : float
        Volume fraction

    Returns
    -------
    dict
        Properties including:
        - "phi": volume fraction
        - "eta": packing fraction (same as phi)
        - "S0": S(k=0), related to compressibility
        - "k_max_position": wave vector of S(k) peak
        - "S_max": peak height of S(k)
        - "is_glassy": whether φ > φ_MCT ≈ 0.516

    Notes
    -----
    The MCT glass transition for hard spheres occurs at φ_MCT ≈ 0.516.
    Random close packing is at φ_rcp ≈ 0.64.
    """
    eta = phi

    # Compressibility from PY
    denom = (1 - eta) ** 4
    S0 = denom / ((1 + 2 * eta) ** 2)

    # Peak position roughly at k*σ ≈ 7.0-7.5 for dense systems
    k_peak_approx = 7.2 * (1 + 0.1 * phi)  # Empirical approximation

    # Peak height from S(k)
    S_peak = percus_yevick_sk(np.array([k_peak_approx]), phi)[0]

    return {
        "phi": phi,
        "eta": eta,
        "S0": S0,
        "k_max_position": k_peak_approx,
        "S_max": S_peak,
        "is_glassy": phi > 0.516,
        "phi_mct": 0.516,
        "phi_rcp": 0.64,
    }
