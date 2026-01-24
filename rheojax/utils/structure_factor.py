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

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()
from functools import partial

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
    q = k * sigma  # Dimensionless wave vector

    # Percus-Yevick coefficients
    eta = phi  # Often use eta = phi for PY
    eta2 = eta * eta
    eta3 = eta * eta2
    eta4 = eta2 * eta2

    # Coefficients for direct correlation function c(r)
    # c(r) = -(a + b*r/σ + c*(r/σ)³) for r < σ, 0 otherwise
    denom = (1 - eta) ** 4
    a_coeff = (1 + 2 * eta) ** 2 / denom
    b_coeff = -6 * eta * (1 + eta / 2) ** 2 / denom
    c_coeff = eta * a_coeff / 2

    # Fourier transform of c(r) - analytic result
    # Avoid division by zero at k=0
    q_safe = np.where(np.abs(q) < 1e-10, 1e-10, q)
    q2 = q_safe * q_safe
    q3 = q2 * q_safe
    q4 = q2 * q2
    q6 = q3 * q3

    sin_q = np.sin(q_safe)
    cos_q = np.cos(q_safe)

    # The Fourier transform involves these terms
    # From Wertheim/Hansen-McDonald derivation
    term1 = a_coeff * (sin_q - q_safe * cos_q) / q3
    term2 = b_coeff * (
        (2 * q_safe * sin_q + (2 - q2) * cos_q - 2) / q4
    )
    term3 = c_coeff * (
        (-q4 * cos_q + 4 * ((3 * q2 - 6) * cos_q + (q3 - 6 * q_safe) * sin_q + 6))
        / q6
    )

    # Direct correlation function in k-space
    c_k = -24 * eta * (term1 + term2 + term3)

    # Structure factor
    S_k = 1.0 / (1.0 - c_k)

    # Handle k=0 limit: S(0) = kT χT (compressibility)
    S_0 = (1 - eta) ** 4 / ((1 + 2 * eta) ** 2 + eta3 * (eta - 4))
    S_k = np.where(np.abs(q) < 1e-10, S_0, S_k)

    return S_k


@partial(jax.jit, static_argnames=())
def percus_yevick_sk_jax(
    k: jnp.ndarray,
    phi: float,
    sigma: float = 1.0,
) -> jnp.ndarray:
    """JAX-compatible Percus-Yevick S(k) computation.

    Parameters
    ----------
    k : jnp.ndarray
        Wave vector magnitudes
    phi : float
        Volume fraction
    sigma : float, default 1.0
        Particle diameter

    Returns
    -------
    jnp.ndarray
        Structure factor S(k)
    """
    q = k * sigma
    eta = phi
    eta2 = eta * eta
    eta3 = eta * eta2

    denom = (1 - eta) ** 4
    a_coeff = (1 + 2 * eta) ** 2 / denom
    b_coeff = -6 * eta * (1 + eta / 2) ** 2 / denom
    c_coeff = eta * a_coeff / 2

    # Safe division
    q_safe = jnp.where(jnp.abs(q) < 1e-10, 1e-10, q)
    q2 = q_safe * q_safe
    q3 = q2 * q_safe
    q4 = q2 * q2
    q6 = q3 * q3

    sin_q = jnp.sin(q_safe)
    cos_q = jnp.cos(q_safe)

    term1 = a_coeff * (sin_q - q_safe * cos_q) / q3
    term2 = b_coeff * ((2 * q_safe * sin_q + (2 - q2) * cos_q - 2) / q4)
    term3 = c_coeff * (
        (-q4 * cos_q + 4 * ((3 * q2 - 6) * cos_q + (q3 - 6 * q_safe) * sin_q + 6))
        / q6
    )

    c_k = -24 * eta * (term1 + term2 + term3)
    S_k = 1.0 / (1.0 - c_k)

    # k=0 limit
    S_0 = (1 - eta) ** 4 / ((1 + 2 * eta) ** 2 + eta3 * (eta - 4))
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


# =============================================================================
# Structure Factor Derivatives
# =============================================================================


def sk_derivatives(
    k: np.ndarray,
    sk: np.ndarray,
    method: str = "finite_diff",
) -> Tuple[np.ndarray, np.ndarray]:
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
        # Central finite differences
        dk = np.diff(k)
        dk_avg = np.concatenate([[dk[0]], (dk[:-1] + dk[1:]) / 2, [dk[-1]]])

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
    sk_func: Optional[callable] = None,
) -> np.ndarray:
    """Compute isotropic MCT vertex V(k,q) after angular integration.

    The full MCT vertex is V(k,q,|k-q|) but for isotropic systems, we
    integrate over the angle between k and q to get V(k,q).

    Parameters
    ----------
    k : np.ndarray
        Wave vector magnitudes (1D array of length n_k)
    q : np.ndarray
        Second wave vector magnitudes (same length as k)
    phi : float
        Volume fraction
    sk_func : callable, optional
        Function sk_func(k) returning S(k). If None, uses Percus-Yevick.

    Returns
    -------
    np.ndarray
        Vertex V(k,q) as 2D array of shape (n_k, n_k)

    Notes
    -----
    The vertex function encodes how density fluctuations at different
    length scales couple in the MCT memory kernel. It depends on S(k)
    and the direct correlation function c(k) = 1 - 1/S(k).
    """
    if sk_func is None:
        sk_func = lambda kk: percus_yevick_sk(kk, phi)

    n_k = len(k)
    V = np.zeros((n_k, n_k))

    # Get S(k) and S(q)
    sk = sk_func(k)
    sq = sk_func(q)

    # Direct correlation function
    ck = 1.0 - 1.0 / sk
    cq = 1.0 - 1.0 / sq

    # Simplified vertex for isotropic case
    # V(k,q) ∝ n * S(k) * S(q) * [k·q c(|k-q|) + k c(k) + q c(q)]²
    # This is a simplified form - full calculation requires angular integration

    for i, ki in enumerate(k):
        for j, qj in enumerate(q):
            # Triangle constraint: |k-q| must be realizable
            k_minus_q_min = abs(ki - qj)
            k_minus_q_max = ki + qj

            # Approximate angular average using midpoint
            k_minus_q = (k_minus_q_min + k_minus_q_max) / 2
            s_kmq = sk_func(np.array([k_minus_q]))[0]
            c_kmq = 1.0 - 1.0 / s_kmq

            # Vertex coupling (simplified Verlet-Weis form)
            coupling = ki * ck[i] + qj * cq[j]
            V[i, j] = sk[i] * sq[j] * s_kmq * coupling**2

    # Normalize by density
    n_density = 6 * phi / np.pi  # Number density for unit diameter
    V *= n_density / (16 * np.pi**2)

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
    eta2 = eta * eta
    eta3 = eta * eta2

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
