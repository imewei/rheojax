"""
JAX-compatible SGR (Soft Glassy Rheology) kernel functions.

This module provides efficient, JAX-compatible implementations of the SGR model
kernel functions from Sollich 1998, enabling GPU-accelerated rheological modeling
of soft glassy materials including foams, emulsions, pastes, and colloidal suspensions.

The SGR model describes materials with an exponential distribution of energy traps
rho(E) ~ exp(-E), leading to characteristic power-law rheological responses that
depend on the effective noise temperature x.

Key Functions
-------------
- rho_trap(E): Exponential trap distribution (normalized probability density)
- G0(x): Equilibrium modulus integral (glass transition at x=1)
- Gp(x, z): Frequency-dependent complex modulus for oscillatory response
- Z(x, omega): Partition function for proper normalization

Physical Interpretation
----------------------
x < 1: Glass phase with yield stress (solid-like)
1 < x < 2: Power-law viscoelastic fluid with G' ~ G'' ~ omega^(x-1)
x >= 2: Newtonian viscous liquid with constant viscosity

References
----------
- P. Sollich, Rheological constitutive equation for a model of soft glassy materials,
  Physical Review E, 1998, 58(1), 738-759
- P. Sollich et al., Rheology of Soft Glassy Materials, Physical Review Letters,
  1997, 78(10), 2020-2023
"""

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
# Float64 precision is critical for accurate numerical integration
jax, jnp = safe_import_jax()

logger = get_logger(__name__)

if TYPE_CHECKING:
    from jax import Array


# ============================================================================
# Trap Distribution
# ============================================================================


@jax.jit
def rho_trap(E: "float | Array") -> "float | Array":
    """
    Exponential trap distribution for SGR model.

    The trap energy distribution is given by:
        rho(E) = exp(-E)  for E >= 0

    This is the canonical choice in Sollich 1998, leading to analytical
    simplifications and characteristic power-law scaling.

    Parameters
    ----------
    E : float or jnp.ndarray
        Trap energy (dimensionless). Must be non-negative.

    Returns
    -------
    float or jnp.ndarray
        Probability density rho(E). Normalized such that integral_0^inf rho(E) dE = 1.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.sgr_kernels import rho_trap
    >>>
    >>> # Single energy value
    >>> rho_trap(1.0)  # exp(-1) ≈ 0.368
    >>>
    >>> # Array of energies
    >>> E = jnp.linspace(0, 5, 50)
    >>> rho_E = rho_trap(E)

    Notes
    -----
    - Normalized: integral_0^inf exp(-E) dE = 1
    - This distribution leads to power-law rheology for 1 < x < 2
    - For E < 0, returns 0 (unphysical)
    """
    E_arr = jnp.asarray(E, dtype=jnp.float64)
    # JIT-safe: no Python-level branch on ndim; let callers handle shape
    # Ref: Sollich 1998, Eq. (3): rho(E) = exp(-E) for E >= 0
    # R8-SGR-003: clamp E for safe exp to avoid inf in jnp.where VJP
    E_safe = jnp.where(E_arr >= 0, E_arr, 0.0)
    return jnp.where(E_arr >= 0, jnp.exp(-E_safe), 0.0)


# ============================================================================
# Equilibrium Modulus G0(x)
# ============================================================================


def _G0_integrand(E: float, x: float) -> float:
    """
    Integrand for G0(x) equilibrium modulus integral.

    G0(x) = integral_0^inf rho(E) * E * (1 - exp(-E/x)) dE

    Parameters
    ----------
    E : float
        Trap energy
    x : float
        Effective noise temperature

    Returns
    -------
    float
        Integrand value at energy E
    """
    return rho_trap(E) * E * (1.0 - jnp.exp(-E / x))


@partial(jax.jit, static_argnums=(1, 2))
def _G0_compute(x: float, n_points: int = 128, E_max: float = 20.0) -> float:
    """
    Compute G0(x) using numerical quadrature (internal).

    Parameters
    ----------
    x : float
        Effective noise temperature
    n_points : int, optional
        Number of quadrature points
    E_max : float, optional
        Upper integration limit

    Returns
    -------
    float
        G0(x) equilibrium modulus
    """
    # Pre-compute energy grid outside of JAX tracing using NumPy (static args
    # guarantee n_points and E_max are concrete at trace time).
    E_lin = np.linspace(0, E_max, n_points // 2)
    E_log = np.logspace(-3, np.log10(E_max), n_points // 2)
    E_grid = jnp.asarray(np.sort(np.concatenate([E_lin, E_log])))

    # Compute integrand values
    integrand_vals = jax.vmap(lambda E: _G0_integrand(E, x))(E_grid)

    # Trapezoidal integration
    integral = jnp.trapezoid(integrand_vals, E_grid)

    return integral


def G0(x: "float | Array") -> "float | Array":
    """
    Equilibrium modulus for SGR model.

    Computes the dimensionless equilibrium modulus:
        G0(x) = integral_0^inf rho(E) * E * (1 - exp(-E/x)) dE

    With rho(E) = exp(-E), this is evaluated numerically using adaptive quadrature.

    Parameters
    ----------
    x : float or jnp.ndarray
        Effective noise temperature. Must be positive.

    Returns
    -------
    float or jnp.ndarray
        Equilibrium modulus G0(x) (dimensionless)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.sgr_kernels import G0
    >>>
    >>> # Glass transition region (x ≈ 1)
    >>> G0(1.0)  # Returns equilibrium modulus at glass transition
    >>>
    >>> # Power-law fluid (1 < x < 2)
    >>> G0(1.5)  # Power-law fluid region
    >>>
    >>> # Newtonian limit (x >= 2)
    >>> G0(2.0)  # Near Newtonian limit
    >>>
    >>> # Array evaluation
    >>> x_vals = jnp.array([1.2, 1.5, 1.8, 2.0])
    >>> G0_vals = G0(x_vals)

    Notes
    -----
    - Glass transition at x=1: Below this, material behaves as yield-stress solid
    - For x << 1: G0 → 1 (deeply glassy, all particles stuck in traps)
    - For x >> 1: G0 decreases (more fluid-like, easier trap hopping)
    - G0 decreases monotonically with increasing x
    - Numerical integration used for accuracy
    - Float64 precision critical for stability near x=1

    Physical Interpretation
    -----------------------
    G0(x) represents the fraction of elastic energy stored in the material.
    At low noise temperature (x << 1), particles are stuck in deep traps,
    giving maximum elasticity (G0 → 1). As x increases, thermal fluctuations
    enable trap hopping, reducing the elastic response (G0 decreases). The
    transition at x=1 marks the glass transition where the material changes
    from solid-like (x<1) to fluid-like (x>1) behavior.
    """
    x_arr = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
    x_safe = jnp.maximum(x_arr, 1e-10)
    result = jax.vmap(_G0_compute)(x_safe)
    # R5-JAX-001: Replace Python `if jnp.ndim(x) == 0` with unconditional
    # jnp.squeeze().  When x is scalar, atleast_1d wraps it to shape (1,) so
    # result has shape (1,); squeeze removes that dimension to restore scalar
    # output.  When x is already an array of shape (N,), result is (N,) and
    # squeeze is a no-op.  This removes the Python branch entirely, making G0()
    # safe to call inside JIT regardless of whether x is a concrete or traced value.
    return jnp.squeeze(result)


# ============================================================================
# Frequency-Dependent Modulus Gp(x, z)
# ============================================================================


def _Gp_integrand_real(E: float, x: float, omega_tau0: float) -> float:
    """
    Real part of Gp integrand for storage modulus G'(omega).

    Parameters
    ----------
    E : float
        Trap energy
    x : float
        Effective noise temperature
    omega_tau0 : float
        Dimensionless frequency omega * tau0 (= z = omega * tau0)

    Returns
    -------
    float
        Real part of integrand

    Notes
    -----
    Correct SGR physics (Sollich 1998): tau_E = tau0 * exp(E/x), so
    omega*tau_E = omega_tau0 * exp(E/x).

    G'(omega) = integral rho(E) * E * (omega*tau_E)^2 / (1 + (omega*tau_E)^2) dE
    """
    # KRN-008 / R10-SGR-KRN-001: tau_E = tau0 * exp(E/x), so omega*tau_E = z*exp(E/x)
    # Cap E/x to avoid overflow; omega_tau0 * exp(E/x) is the dimensionless product.
    exp_arg = jnp.minimum(E / x, 709.0)
    tau_E_z = omega_tau0 * jnp.exp(exp_arg)
    numerator = E * tau_E_z**2
    denominator = 1.0 + tau_E_z**2
    return rho_trap(E) * numerator / denominator


def _Gp_integrand_imag(E: float, x: float, omega_tau0: float) -> float:
    """
    Imaginary part of Gp integrand for loss modulus G''(omega).

    Parameters
    ----------
    E : float
        Trap energy
    x : float
        Effective noise temperature
    omega_tau0 : float
        Dimensionless frequency omega * tau0 (= z = omega * tau0)

    Returns
    -------
    float
        Imaginary part of integrand

    Notes
    -----
    Correct SGR physics (Sollich 1998): tau_E = tau0 * exp(E/x), so
    omega*tau_E = omega_tau0 * exp(E/x).

    G''(omega) = integral rho(E) * E * (omega*tau_E) / (1 + (omega*tau_E)^2) dE
    """
    # KRN-008 / R10-SGR-KRN-001: tau_E = tau0 * exp(E/x), so omega*tau_E = z*exp(E/x)
    exp_arg = jnp.minimum(E / x, 709.0)
    tau_E_z = omega_tau0 * jnp.exp(exp_arg)
    numerator = E * tau_E_z
    denominator = 1.0 + tau_E_z**2
    return rho_trap(E) * numerator / denominator


@partial(jax.jit, static_argnums=(2, 3))
def _Gp_quadrature(
    x: float, omega_tau0: float, n_points: int = 128, E_max: float = 20.0
) -> tuple[float, float]:
    """
    Compute Gp(x, z) using numerical quadrature.

    Parameters
    ----------
    x : float
        Effective noise temperature
    omega_tau0 : float
        Dimensionless frequency omega * tau0
    n_points : int, optional
        Number of quadrature points
    E_max : float, optional
        Upper integration limit

    Returns
    -------
    tuple[float, float]
        (G_prime, G_double_prime) - real and imaginary parts
    """
    # Pre-compute energy grid outside of JAX tracing using NumPy (static args
    # guarantee n_points and E_max are concrete at trace time).
    E_lin = np.linspace(0, E_max, n_points // 2)
    E_log = np.logspace(-3, np.log10(E_max), n_points // 2)
    E_grid = jnp.asarray(np.sort(np.concatenate([E_lin, E_log])))

    # Compute integrands
    integrand_real = jax.vmap(lambda E: _Gp_integrand_real(E, x, omega_tau0))(E_grid)
    integrand_imag = jax.vmap(lambda E: _Gp_integrand_imag(E, x, omega_tau0))(E_grid)

    # Trapezoidal integration
    G_prime = jnp.trapezoid(integrand_real, E_grid)
    G_double_prime = jnp.trapezoid(integrand_imag, E_grid)

    return G_prime, G_double_prime


def Gp(x: "float | Array", omega_tau0: "float | Array") -> "tuple[Array, Array]":
    """
    Frequency-dependent complex modulus for SGR model.

    Computes the dimensionless complex modulus G*(omega) = G'(omega) + i*G''(omega):

        G'(omega)  = integral_0^inf rho(E) * E * (z*exp(E/x))^2 / (1 + (z*exp(E/x))^2) dE
        G''(omega) = integral_0^inf rho(E) * E * (z*exp(E/x))   / (1 + (z*exp(E/x))^2) dE

    where z = omega * tau0 is the dimensionless frequency and tau_E = tau0 * exp(E/x)
    is the trap-energy-dependent relaxation time (Sollich 1998, Eq. 19).

    Parameters
    ----------
    x : float or jnp.ndarray
        Effective noise temperature. Must be positive.
    omega_tau0 : float or jnp.ndarray
        Dimensionless frequency omega * tau0. Must be positive.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (G_prime, G_double_prime) where:
        - G_prime: Storage modulus G'(omega) (dimensionless)
        - G_double_prime: Loss modulus G''(omega) (dimensionless)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.sgr_kernels import Gp
    >>>
    >>> # Single frequency
    >>> x = 1.5
    >>> omega_tau0 = 1.0
    >>> G_prime, G_double_prime = Gp(x, omega_tau0)
    >>>
    >>> # Frequency sweep (vectorized)
    >>> omega_tau0_vals = jnp.logspace(-2, 2, 50)
    >>> G_prime, G_double_prime = jax.vmap(lambda w: Gp(x, w))(omega_tau0_vals)
    >>>
    >>> # Multiple x values
    >>> x_vals = jnp.array([1.2, 1.5, 1.8, 2.0])
    >>> omega_tau0 = 1.0
    >>> results = jax.vmap(lambda x_val: Gp(x_val, omega_tau0))(x_vals)

    Notes
    -----
    - Power-law scaling for 1 < x < 2: G' ~ G'' ~ omega^(x-1)
    - At low frequencies (omega → 0): G'' dominates (viscous)
    - At high frequencies (omega → ∞): G' → G0(x) (elastic plateau)
    - For x < 1: Solid-like with yield stress
    - For x >= 2: Newtonian with G' ~ omega^2, G'' ~ omega
    - Numerical integration uses adaptive quadrature

    Physical Interpretation
    -----------------------
    The frequency-dependent modulus captures the transition from elastic (high freq)
    to viscous (low freq) response. The characteristic power-law scaling omega^(x-1)
    is the signature of SGR dynamics, arising from the broad distribution of
    relaxation times due to the exponential trap distribution.
    """
    # Ensure arrays
    x_arr = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
    omega_arr = jnp.atleast_1d(jnp.asarray(omega_tau0, dtype=jnp.float64))

    # R8-OPT-004: validate shapes before JIT-traced dispatch
    if not (jnp.ndim(x) == 0 or jnp.ndim(omega_tau0) == 0) and x_arr.shape != omega_arr.shape:
        raise ValueError(f"Shape mismatch: x {x_arr.shape} vs omega {omega_arr.shape}")

    # Validate inputs
    x_safe = jnp.maximum(x_arr, 1e-10)
    omega_safe = jnp.maximum(omega_arr, 1e-10)

    # R5-JAX-001: Replace Python if/elif on jnp.ndim() with a single unified
    # vmap path.  Python-if on ndim() freezes the branch at trace time, so
    # calling Gp() inside a JIT function with a scalar-vs-array input depending
    # on a traced value would silently produce wrong output shapes.
    #
    # Unified strategy: always vmap over omega_safe (length >= 1).
    # • If x is scalar: broadcast to length-1 and let vmap walk omega.
    # • If x and omega are arrays: must be the same length (existing contract).
    # • Scalar squeeze is deferred to the caller via jnp.squeeze on 0-d inputs.
    is_x_scalar = jnp.ndim(x) == 0  # Python int — safe at eager / closure time
    is_omega_scalar = jnp.ndim(omega_tau0) == 0

    if is_x_scalar:
        # x scalar — broadcast x across all omega values
        results = jax.vmap(lambda w: _Gp_quadrature(x_safe[0], w))(omega_safe)
    elif is_omega_scalar:
        # omega scalar, x array — vectorize over x
        results = jax.vmap(lambda x_val: _Gp_quadrature(x_val, omega_safe[0]))(x_safe)
    else:
        # x array and omega array — shape already validated above
        results = jax.vmap(_Gp_quadrature)(x_safe, omega_safe)

    # Squeeze the length-1 axis when both inputs were scalars so callers
    # that pass scalar (x, omega) get a scalar back, matching prior behaviour.
    if is_x_scalar and is_omega_scalar:
        return results[0][0], results[1][0]
    return results[0], results[1]


# ============================================================================
# Partition Function Z(x, omega)
# ============================================================================


@jax.jit
def Z(x: float, omega_tau0: "float | Array") -> "float | Array":
    """
    Partition function for SGR model normalization.

    Computes the normalization integral:
        Z(x, omega) = integral_0^inf rho(E) * exp(-E/x) dE

    For exponential trap distribution rho(E) = exp(-E), this simplifies to:
        Z(x) = x / (x + 1)

    Parameters
    ----------
    x : float
        Effective noise temperature. Must be positive.
    omega_tau0 : float or jnp.ndarray
        Dimensionless frequency omega * tau0 (included for API consistency,
        but Z is independent of frequency for equilibrium SGR).

    Returns
    -------
    float or jnp.ndarray
        Partition function Z(x) (dimensionless)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.sgr_kernels import Z
    >>>
    >>> # Single value
    >>> Z(1.5, 1.0)  # ≈ 0.6
    >>>
    >>> # Array of frequencies (Z independent of omega for equilibrium)
    >>> omega_vals = jnp.logspace(-2, 2, 50)
    >>> Z_vals = Z(1.5, omega_vals)  # All same value

    Notes
    -----
    - For exponential trap distribution: Z(x) = x / (x + 1)
    - Z → 0 as x → 0 (deeply glassy, all traps occupied)
    - Z → 1 as x → ∞ (high temperature, traps unoccupied)
    - In equilibrium SGR, Z is frequency-independent
    - Required for proper probability normalization in GENERIC formulation
    """
    x_safe = jnp.maximum(x, 1e-10)

    # Analytical formula for exponential distribution
    # Z = integral_0^inf exp(-E) * exp(-E/x) dE
    #   = integral_0^inf exp(-E(1 + 1/x)) dE
    #   = 1 / (1 + 1/x)
    #   = x / (x + 1)

    Z_val = x_safe / (x_safe + 1.0)

    # R5-JAX-001: Replace Python if on jnp.ndim() with NumPy-style broadcasting.
    # The original code dispatched via `if jnp.ndim(omega_tau0) == 0` which
    # freezes one branch inside JIT when omega_tau0 is a traced value.
    #
    # Z(x) is frequency-independent; omega_tau0 is included only for API
    # consistency (e.g., callers that pass (x, omega) pairs uniformly).
    # The output shape should follow standard broadcast rules between x and
    # omega_tau0:
    #   - Z(scalar x, scalar omega)  → scalar
    #   - Z(scalar x, array omega)   → array of omega.shape (all same value)
    #   - Z(array x, scalar omega)   → array of x.shape
    #   - Z(array x, array omega)    → broadcast(x.shape, omega.shape)
    #
    # jnp.broadcast_shapes() + broadcast_to handles all four cases without
    # any Python branch on ndim, making this safe inside JIT.
    out_shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(omega_tau0))
    return jnp.broadcast_to(Z_val, out_shape)


# ============================================================================
# Power-Law Scaling Verification
# ============================================================================


@jax.jit
def power_law_exponent(x: float) -> float:
    """
    Theoretical power-law exponent for SGR model.

    For 1 < x < 2, the SGR model predicts:
        G'(omega) ~ omega^(x-1)
        G''(omega) ~ omega^(x-1)

    Parameters
    ----------
    x : float
        Effective noise temperature

    Returns
    -------
    float
        Power-law exponent (x - 1)

    Examples
    --------
    >>> from rheojax.utils.sgr_kernels import power_law_exponent
    >>> power_law_exponent(1.5)  # 0.5
    >>> power_law_exponent(1.8)  # 0.8

    Notes
    -----
    - Valid for 1 < x < 2 (power-law fluid regime)
    - For x < 1: Solid-like with yield stress (no power-law)
    - For x >= 2: Newtonian with G' ~ omega^2, G'' ~ omega
    """
    return x - 1.0


# ============================================================================
# Convenience Exports
# ============================================================================

__all__ = [
    "rho_trap",
    "G0",
    "Gp",
    "Z",
    "power_law_exponent",
]
