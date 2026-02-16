"""JAX-accelerated physics kernels for VLB transient network models.

This module provides JIT-compiled functions for:
1. Distribution tensor evolution (ODE RHS for shear and uniaxial)
2. Stress functions (shear stress, normal stress differences)
3. Analytical solutions for single-network protocols
4. Multi-network analytical solutions (superposition)
5. Uniaxial extension solutions
6. Creep helpers for multi-network implicit solve

All functions are designed for float64 precision and JAX compatibility.

Key Equation
------------
The VLB constitutive equation for the distribution tensor mu::

    dmu/dt = k_d * (I - mu) + D·mu + mu·D

where:
- mu is the distribution tensor (dimensionless, equilibrium mu = I)
- D is the rate-of-deformation tensor (symmetric part of velocity gradient)
- k_d is the dissociation rate (1/s)

Stress is computed from the distribution tensor::

    sigma = G0 * (mu - I) + p*I

State Vector Convention
-----------------------
For 3D incompressible simple shear, we use a 4-component state::

    state = [mu_xx, mu_yy, mu_zz, mu_xy]

Equilibrium: mu = I -> [1, 1, 1, 0]

Reference
---------
Vernerey, Long, & Brighenti (2017). JMPS 107, 1-20.
"""

from __future__ import annotations

from functools import partial

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


# =============================================================================
# Distribution Tensor ODE Right-Hand Sides
# =============================================================================


@jax.jit
def vlb_mu_rhs_shear(
    mu_xx: float,
    mu_yy: float,
    mu_zz: float,
    mu_xy: float,
    gamma_dot: float,
    k_d: float,
) -> tuple[float, float, float, float]:
    """ODE RHS for distribution tensor under simple shear.

    Simple shear velocity gradient: L_xy = gamma_dot, all others zero.
    Rate of deformation: D_xy = D_yx = gamma_dot / 2.

    dmu_xx/dt = k_d*(1 - mu_xx) + gamma_dot * mu_xy
    dmu_yy/dt = k_d*(1 - mu_yy) + gamma_dot * mu_xy  (correction: 0 from D for yy)
    dmu_zz/dt = k_d*(1 - mu_zz)
    dmu_xy/dt = k_d*(0 - mu_xy) + gamma_dot/2 * (mu_xx + mu_yy)

    Note: In simple shear, L = [[0, gamma_dot, 0], [0, 0, 0], [0, 0, 0]].
    D = (L + L^T)/2 = [[0, gdot/2, 0], [gdot/2, 0, 0], [0, 0, 0]].
    D.mu + mu.D contributes:
      (D.mu)_xx = D_xy * mu_yx = (gdot/2)*mu_xy
      (mu.D)_xx = mu_xy * D_yx = mu_xy*(gdot/2)
      -> d(mu_xx) from deformation = gamma_dot * mu_xy

      (D.mu)_yy = D_yx * mu_xy = (gdot/2)*mu_xy
      (mu.D)_yy = mu_yx * D_xy = mu_xy*(gdot/2)
      -> d(mu_yy) from deformation = gamma_dot * mu_xy

    Wait, that's wrong for shear. Let me re-derive carefully.

    Actually for the VLB model with velocity gradient L (not just D):
    dmu/dt = k_d(I - mu) + L.mu + mu.L^T

    With L_12 = gamma_dot (1-2 index = xy):
    (L.mu)_11 = L_12 * mu_21 = gamma_dot * mu_xy
    (mu.L^T)_11 = mu_12 * L^T_21 = mu_xy * gamma_dot
    -> contribution to mu_xx: 2 * gamma_dot * mu_xy

    (L.mu)_22 = 0, (mu.L^T)_22 = 0
    -> contribution to mu_yy: 0

    (L.mu)_12 = L_12 * mu_22 = gamma_dot * mu_yy
    (mu.L^T)_12 = mu_11 * L^T_12 = 0  (L^T_12 = L_21 = 0)
    Wait, L^T_12 = L_21 = 0 in simple shear.
    Actually (mu.L^T)_12 = mu_11 * (L^T)_12 + mu_12 * (L^T)_22 + mu_13 * (L^T)_32
    L^T = [[0,0,0],[gdot,0,0],[0,0,0]]
    (mu.L^T)_12 = mu_11 * 0 + mu_12 * 0 + mu_13 * 0 = 0

    So for the FULL velocity gradient (not symmetric D):
    dmu_xx/dt = k_d*(1 - mu_xx) + 2*gamma_dot*mu_xy
    dmu_yy/dt = k_d*(1 - mu_yy)
    dmu_zz/dt = k_d*(1 - mu_zz)
    dmu_xy/dt = -k_d*mu_xy + gamma_dot*mu_yy

    This matches the TNT pattern exactly (upper-convected Maxwell).

    Parameters
    ----------
    mu_xx, mu_yy, mu_zz, mu_xy : float
        Distribution tensor components
    gamma_dot : float
        Shear rate (1/s)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    tuple of 4 floats
        Time derivatives (dmu_xx/dt, dmu_yy/dt, dmu_zz/dt, dmu_xy/dt)
    """
    dmu_xx = k_d * (1.0 - mu_xx) + 2.0 * gamma_dot * mu_xy
    dmu_yy = k_d * (1.0 - mu_yy)
    dmu_zz = k_d * (1.0 - mu_zz)
    dmu_xy = -k_d * mu_xy + gamma_dot * mu_yy

    return dmu_xx, dmu_yy, dmu_zz, dmu_xy


@jax.jit
def vlb_mu_rhs_uniaxial(
    mu_11: float,
    mu_22: float,
    eps_dot: float,
    k_d: float,
) -> tuple[float, float]:
    """ODE RHS for distribution tensor under uniaxial extension.

    Uniaxial extension: L = diag(eps_dot, -eps_dot/2, -eps_dot/2).
    D = L (symmetric). Off-diagonal mu components remain zero.

    dmu_11/dt = k_d*(1 - mu_11) + 2*eps_dot*mu_11
    dmu_22/dt = k_d*(1 - mu_22) - eps_dot*mu_22

    Parameters
    ----------
    mu_11, mu_22 : float
        Distribution tensor axial and transverse components
    eps_dot : float
        Extensional strain rate (1/s)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    tuple of 2 floats
        Time derivatives (dmu_11/dt, dmu_22/dt)
    """
    dmu_11 = k_d * (1.0 - mu_11) + 2.0 * eps_dot * mu_11
    dmu_22 = k_d * (1.0 - mu_22) - eps_dot * mu_22

    return dmu_11, dmu_22


# =============================================================================
# Stress Functions
# =============================================================================


@jax.jit
def vlb_shear_stress(mu_xy: float, G0: float) -> float:
    """Compute shear stress from distribution tensor.

    sigma_12 = G0 * mu_xy

    (Since sigma = G0*(mu - I), and I_xy = 0.)

    Parameters
    ----------
    mu_xy : float
        Off-diagonal distribution tensor component
    G0 : float
        Network modulus (Pa)

    Returns
    -------
    float
        Shear stress (Pa)
    """
    return G0 * mu_xy


@jax.jit
def vlb_normal_stress_1(mu_xx: float, mu_yy: float, G0: float) -> float:
    """Compute first normal stress difference N1.

    N1 = sigma_xx - sigma_yy = G0 * (mu_xx - mu_yy)

    Parameters
    ----------
    mu_xx, mu_yy : float
        Diagonal distribution tensor components
    G0 : float
        Network modulus (Pa)

    Returns
    -------
    float
        First normal stress difference N1 (Pa)
    """
    return G0 * (mu_xx - mu_yy)


# =============================================================================
# Single-Network Analytical Solutions
# =============================================================================


@jax.jit
def vlb_steady_shear(gamma_dot: float, G0: float, k_d: float) -> float:
    """Steady-state shear stress for single VLB network.

    At steady state, dmu/dt = 0:
    mu_xy_ss = gamma_dot * mu_yy_ss / k_d
    mu_yy_ss = 1 (from dmu_yy = k_d*(1-mu_yy) = 0)

    sigma = G0 * gamma_dot / k_d

    This is Newtonian: sigma = eta * gamma_dot, with eta = G0/k_d.

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        Steady-state shear stress (Pa)
    """
    return G0 * gamma_dot / k_d


# Vectorized version
vlb_steady_shear_vec = jax.jit(jax.vmap(vlb_steady_shear, in_axes=(0, None, None)))


@jax.jit
def vlb_steady_n1(gamma_dot: float, G0: float, k_d: float) -> float:
    """Steady-state first normal stress difference for single VLB network.

    At steady state:
    mu_xx_ss = 1 + 2*(gamma_dot/k_d)^2
    mu_yy_ss = 1

    N1 = G0 * (mu_xx - mu_yy) = 2*G0*(gamma_dot/k_d)^2

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        Steady N1 (Pa)
    """
    Wi = gamma_dot / k_d
    return 2.0 * G0 * Wi * Wi


vlb_steady_n1_vec = jax.jit(jax.vmap(vlb_steady_n1, in_axes=(0, None, None)))


@jax.jit
def vlb_startup_stress(
    t: float, gamma_dot: float, G0: float, k_d: float
) -> float:
    """Transient shear stress during startup flow.

    sigma(t) = G0 * (gamma_dot / k_d) * (1 - exp(-k_d * t))

    Derived from the exact solution of the mu_xy ODE with mu_xy(0) = 0
    and mu_yy = 1 (quasi-steady, which is exact since mu_yy relaxes to 1
    independently of the shear).

    Parameters
    ----------
    t : float
        Time (s)
    gamma_dot : float
        Applied shear rate (1/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        Shear stress at time t (Pa)
    """
    t_R = 1.0 / k_d
    return G0 * gamma_dot * t_R * (1.0 - jnp.exp(-t / t_R))


vlb_startup_stress_vec = jax.jit(
    jax.vmap(vlb_startup_stress, in_axes=(0, None, None, None))
)


@jax.jit
def vlb_startup_n1(
    t: float, gamma_dot: float, G0: float, k_d: float
) -> float:
    """Transient first normal stress difference during startup.

    N1(t) = 2*G0*(gamma_dot/k_d)^2 * [1 - (1 + k_d*t)*exp(-k_d*t)]

    Derived from mu_xx(t) = 1 + 2*(gdot/k_d)^2 * [1 - (1+k_d*t)*exp(-k_d*t)]
    and mu_yy(t) = 1 (equilibrium for yy).

    Parameters
    ----------
    t : float
        Time (s)
    gamma_dot : float
        Applied shear rate (1/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        N1 at time t (Pa)
    """
    Wi = gamma_dot / k_d
    kd_t = k_d * t
    return 2.0 * G0 * Wi * Wi * (1.0 - (1.0 + kd_t) * jnp.exp(-kd_t))


vlb_startup_n1_vec = jax.jit(
    jax.vmap(vlb_startup_n1, in_axes=(0, None, None, None))
)


@jax.jit
def vlb_relaxation_modulus(t: float, G0: float, k_d: float) -> float:
    """Stress relaxation modulus G(t).

    G(t) = G0 * exp(-k_d * t) = G0 * exp(-t / t_R)

    Single exponential decay (Maxwell model).

    Parameters
    ----------
    t : float
        Time after cessation of flow (s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        Relaxation modulus G(t) (Pa)
    """
    return G0 * jnp.exp(-k_d * t)


vlb_relaxation_modulus_vec = jax.jit(
    jax.vmap(vlb_relaxation_modulus, in_axes=(0, None, None))
)


@jax.jit
def vlb_saos_moduli(
    omega: float, G0: float, k_d: float
) -> tuple[float, float]:
    """SAOS storage and loss moduli for single VLB network.

    G'(omega) = G0 * omega^2 * t_R^2 / (1 + omega^2 * t_R^2)
    G''(omega) = G0 * omega * t_R / (1 + omega^2 * t_R^2)

    Maxwell model form with t_R = 1/k_d.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    tuple of (float, float)
        (G', G'') storage and loss moduli (Pa)
    """
    t_R = 1.0 / k_d
    wt = omega * t_R
    wt2 = wt * wt
    denom = 1.0 + wt2
    G_prime = G0 * wt2 / denom
    G_double_prime = G0 * wt / denom
    return G_prime, G_double_prime


def _vlb_saos_G_prime(omega: float, G0: float, k_d: float) -> float:
    """Helper returning only G' for vmap."""
    G_p, _ = vlb_saos_moduli(omega, G0, k_d)
    return G_p


def _vlb_saos_G_double_prime(omega: float, G0: float, k_d: float) -> float:
    """Helper returning only G'' for vmap."""
    _, G_pp = vlb_saos_moduli(omega, G0, k_d)
    return G_pp


def vlb_saos_moduli_vec(
    omega_arr: jnp.ndarray, G0: float, k_d: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized SAOS moduli over frequency array.

    Parameters
    ----------
    omega_arr : jnp.ndarray
        Angular frequency array (rad/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        (G', G'') arrays (Pa)
    """
    t_R = 1.0 / k_d
    wt = omega_arr * t_R
    wt2 = wt * wt
    denom = 1.0 + wt2
    G_prime = G0 * wt2 / denom
    G_double_prime = G0 * wt / denom
    return G_prime, G_double_prime


vlb_saos_moduli_vec = jax.jit(vlb_saos_moduli_vec)


@jax.jit
def vlb_creep_compliance_single(t: float, G0: float, k_d: float) -> float:
    """Creep compliance for single VLB network (Maxwell).

    J(t) = (1 + k_d * t) / G0 = 1/G0 + t/(G0/k_d)

    Elastic jump J(0) = 1/G0 followed by viscous flow with slope k_d/G0 = 1/eta.

    Parameters
    ----------
    t : float
        Time (s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        Creep compliance J(t) (1/Pa)
    """
    return (1.0 + k_d * t) / G0


vlb_creep_compliance_single_vec = jax.jit(
    jax.vmap(vlb_creep_compliance_single, in_axes=(0, None, None))
)


# =============================================================================
# Multi-Network Analytical Solutions
# =============================================================================


@jax.jit
def vlb_multi_saos(
    omega: float,
    G_modes: jnp.ndarray,
    kd_modes: jnp.ndarray,
    G_e: float,
    eta_s: float,
) -> tuple[float, float]:
    """SAOS moduli for multi-network VLB.

    G'(omega) = G_e + sum_i G_i * omega^2 * t_R_i^2 / (1 + omega^2 * t_R_i^2)
    G''(omega) = sum_i G_i * omega * t_R_i / (1 + omega^2 * t_R_i^2) + eta_s * omega

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    G_modes : jnp.ndarray
        Mode moduli array (Pa)
    kd_modes : jnp.ndarray
        Mode dissociation rates array (1/s)
    G_e : float
        Permanent network modulus (Pa), 0 if no permanent network
    eta_s : float
        Solvent viscosity (Pa*s)

    Returns
    -------
    tuple of (float, float)
        (G', G'') (Pa)
    """
    t_R_modes = 1.0 / kd_modes
    wt = omega * t_R_modes
    wt2 = wt * wt
    denom = 1.0 + wt2

    G_prime = G_e + jnp.sum(G_modes * wt2 / denom)
    G_double_prime = jnp.sum(G_modes * wt / denom) + eta_s * omega

    return G_prime, G_double_prime


@partial(jax.jit, static_argnums=())
def vlb_multi_saos_vec(
    omega_arr: jnp.ndarray,
    G_modes: jnp.ndarray,
    kd_modes: jnp.ndarray,
    G_e: float,
    eta_s: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized multi-network SAOS moduli.

    Parameters
    ----------
    omega_arr : jnp.ndarray
        Frequency array (rad/s), shape (N,)
    G_modes : jnp.ndarray
        Mode moduli, shape (M,)
    kd_modes : jnp.ndarray
        Mode dissociation rates, shape (M,)
    G_e : float
        Permanent network modulus (Pa)
    eta_s : float
        Solvent viscosity (Pa*s)

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        (G', G'') arrays, each shape (N,)
    """
    # omega: (N,), t_R_modes: (M,) -> wt: (N, M)
    t_R_modes = 1.0 / kd_modes
    wt = omega_arr[:, None] * t_R_modes[None, :]
    wt2 = wt * wt
    denom = 1.0 + wt2

    # G_modes: (M,) broadcast -> sum over modes axis
    G_prime = G_e + jnp.sum(G_modes[None, :] * wt2 / denom, axis=1)
    G_double_prime = (
        jnp.sum(G_modes[None, :] * wt / denom, axis=1) + eta_s * omega_arr
    )

    return G_prime, G_double_prime


@jax.jit
def vlb_multi_relaxation(
    t: float,
    G_modes: jnp.ndarray,
    kd_modes: jnp.ndarray,
    G_e: float,
) -> float:
    """Relaxation modulus for multi-network VLB (Prony series).

    G(t) = G_e + sum_i G_i * exp(-k_d_i * t)

    Parameters
    ----------
    t : float
        Time (s)
    G_modes : jnp.ndarray
        Mode moduli (Pa)
    kd_modes : jnp.ndarray
        Mode dissociation rates (1/s)
    G_e : float
        Permanent network modulus (Pa)

    Returns
    -------
    float
        Relaxation modulus G(t) (Pa)
    """
    return G_e + jnp.sum(G_modes * jnp.exp(-kd_modes * t))


vlb_multi_relaxation_vec = jax.jit(
    jax.vmap(vlb_multi_relaxation, in_axes=(0, None, None, None))
)


@jax.jit
def vlb_multi_startup_stress(
    t: float,
    gdot: float,
    G_modes: jnp.ndarray,
    kd_modes: jnp.ndarray,
    G_e: float,
    eta_s: float,
) -> float:
    """Startup stress for multi-network VLB (superposition).

    sigma(t) = sum_i G_i * (gdot/k_d_i) * (1 - exp(-k_d_i * t))
               + G_e * gdot * t  (if permanent)
               + eta_s * gdot

    Note: The permanent network contributes a linear growth G_e * gdot * t
    because it has no relaxation. However, for practical fitting we cap
    this at a reasonable value since real permanent networks are elastic:
    sigma_permanent = G_e * gamma where gamma = gdot * t, i.e., neo-Hookean.
    For NLSQ fitting, this is correct (no relaxation).

    Parameters
    ----------
    t : float
        Time (s)
    gdot : float
        Shear rate (1/s)
    G_modes : jnp.ndarray
        Mode moduli (Pa)
    kd_modes : jnp.ndarray
        Mode dissociation rates (1/s)
    G_e : float
        Permanent network modulus (Pa)
    eta_s : float
        Solvent viscosity (Pa*s)

    Returns
    -------
    float
        Shear stress at time t (Pa)
    """
    t_R_modes = 1.0 / kd_modes
    transient = jnp.sum(
        G_modes * gdot * t_R_modes * (1.0 - jnp.exp(-kd_modes * t))
    )
    permanent = G_e * gdot * t
    solvent = eta_s * gdot
    return transient + permanent + solvent


vlb_multi_startup_stress_vec = jax.jit(
    jax.vmap(vlb_multi_startup_stress, in_axes=(0, None, None, None, None, None))
)


@jax.jit
def vlb_multi_steady_viscosity(
    G_modes: jnp.ndarray,
    kd_modes: jnp.ndarray,
    eta_s: float,
) -> float:
    """Zero-shear viscosity for multi-network VLB.

    eta_0 = sum_i G_i / k_d_i + eta_s

    Parameters
    ----------
    G_modes : jnp.ndarray
        Mode moduli (Pa)
    kd_modes : jnp.ndarray
        Mode dissociation rates (1/s)
    eta_s : float
        Solvent viscosity (Pa*s)

    Returns
    -------
    float
        Zero-shear viscosity (Pa*s)
    """
    return jnp.sum(G_modes / kd_modes) + eta_s


# =============================================================================
# Uniaxial Extension Solutions
# =============================================================================


@jax.jit
def vlb_uniaxial_steady(eps_dot: float, G0: float, k_d: float) -> float:
    """Steady-state extensional stress for single VLB network.

    At steady state:
    mu_11_ss = k_d / (k_d - 2*eps_dot)
    mu_22_ss = k_d / (k_d + eps_dot)

    sigma_E = G0 * (mu_11 - mu_22)

    Singularity at eps_dot = k_d/2 (extensional catastrophe).
    We regularize by clamping the denominator.

    Parameters
    ----------
    eps_dot : float
        Extensional strain rate (1/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        Steady extensional stress (Pa)
    """
    # Regularize near singularity
    denom_11 = jnp.maximum(k_d - 2.0 * eps_dot, 1e-10)
    denom_22 = k_d + eps_dot

    mu_11 = k_d / denom_11
    mu_22 = k_d / denom_22

    return G0 * (mu_11 - mu_22)


vlb_uniaxial_steady_vec = jax.jit(
    jax.vmap(vlb_uniaxial_steady, in_axes=(0, None, None))
)


@jax.jit
def vlb_uniaxial_transient(
    t: float, eps_dot: float, G0: float, k_d: float
) -> float:
    """Transient extensional stress during startup extension.

    sigma_11(t) = G0 * [mu_11(t) - mu_22(t)]

    mu_11(t) = k_d/(k_d-2*edot) * [1 - exp(-(k_d-2*edot)*t)] + exp(-(k_d-2*edot)*t)
             = 1 + [k_d/(k_d-2*edot) - 1] * [1 - exp(-(k_d-2*edot)*t)]

    For |k_d - 2*edot| << k_d, use linearized form to avoid numerical issues.

    Parameters
    ----------
    t : float
        Time (s)
    eps_dot : float
        Extensional strain rate (1/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        Extensional stress sigma_E(t) (Pa)
    """
    lambda_11 = k_d - 2.0 * eps_dot
    lambda_22 = k_d + eps_dot

    # Regularize: avoid division by very small numbers
    safe_lambda_11 = jnp.where(jnp.abs(lambda_11) > 1e-10, lambda_11, 1e-10)

    mu_11 = k_d / safe_lambda_11 + (1.0 - k_d / safe_lambda_11) * jnp.exp(
        -safe_lambda_11 * t
    )
    mu_22 = k_d / lambda_22 + (1.0 - k_d / lambda_22) * jnp.exp(-lambda_22 * t)

    return G0 * (mu_11 - mu_22)


vlb_uniaxial_transient_vec = jax.jit(
    jax.vmap(vlb_uniaxial_transient, in_axes=(0, None, None, None))
)


@jax.jit
def vlb_trouton_ratio(eps_dot: float, G0: float, k_d: float) -> float:
    """Trouton ratio eta_E / eta_0.

    eta_E = sigma_E / (3 * eps_dot)  [engineering convention for uniaxial]
    Actually, eta_E = sigma_E / eps_dot for uniaxial.
    eta_0 = G0 / k_d (zero-shear viscosity)

    Trouton ratio Tr = eta_E / eta_0

    At low eps_dot: Tr -> 3 (Trouton's rule)
    At high eps_dot (approaching k_d/2): Tr -> infinity (extensional hardening)

    Parameters
    ----------
    eps_dot : float
        Extensional strain rate (1/s)
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)

    Returns
    -------
    float
        Trouton ratio (dimensionless)
    """
    sigma_E = vlb_uniaxial_steady(eps_dot, G0, k_d)
    eta_E = sigma_E / jnp.maximum(eps_dot, 1e-20)
    eta_0 = G0 / k_d
    return eta_E / eta_0


vlb_trouton_ratio_vec = jax.jit(
    jax.vmap(vlb_trouton_ratio, in_axes=(0, None, None))
)


# =============================================================================
# Creep Helpers
# =============================================================================


@jax.jit
def vlb_creep_compliance_dual(
    t: float, G0: float, k_d: float, G_e: float
) -> float:
    """Creep compliance for VLB network + permanent elastic network.

    This is the standard linear solid (SLS) / Zener compliance:
    J(t) = 1/(G0+G_e) + [G0/(G_e*(G0+G_e))] * (1 - exp(-t/tau_ret))

    where tau_ret = (G0 + G_e) / (k_d * G_e) is the retardation time.

    At t=0: J = 1/(G0+G_e) (instantaneous elastic)
    At t->inf: J = 1/G_e (equilibrium elastic, bounded)

    Parameters
    ----------
    t : float
        Time (s)
    G0 : float
        Transient network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)
    G_e : float
        Permanent network modulus (Pa)

    Returns
    -------
    float
        Creep compliance J(t) (1/Pa)
    """
    G_total = G0 + G_e
    # Avoid division by zero if G_e = 0
    G_e_safe = jnp.maximum(G_e, 1e-30)
    tau_ret = G_total / (k_d * G_e_safe)

    J_inst = 1.0 / G_total
    J_retard = G0 / (G_e_safe * G_total) * (1.0 - jnp.exp(-t / tau_ret))

    return J_inst + J_retard


vlb_creep_compliance_dual_vec = jax.jit(
    jax.vmap(vlb_creep_compliance_dual, in_axes=(0, None, None, None))
)


@jax.jit
def vlb_solve_creep_gamma_dot(
    sigma_0: float,
    mu_xy_modes: jnp.ndarray,
    G_modes: jnp.ndarray,
    kd_modes: jnp.ndarray,
    G_e: float,
    gamma: float,
    eta_s: float,
) -> float:
    """Solve for instantaneous shear rate during multi-network creep.

    Stress balance: sigma_0 = sum_i G_i * mu_xy_i + G_e * gamma + eta_s * gamma_dot

    Solve for gamma_dot. This is used within a jax.lax.scan loop.

    Parameters
    ----------
    sigma_0 : float
        Applied constant stress (Pa)
    mu_xy_modes : jnp.ndarray
        Off-diagonal distribution tensor components for each mode
    G_modes : jnp.ndarray
        Mode moduli (Pa)
    kd_modes : jnp.ndarray
        Mode dissociation rates (1/s), unused here but kept for interface
    G_e : float
        Permanent network modulus (Pa)
    gamma : float
        Current total strain
    eta_s : float
        Solvent viscosity (Pa*s)

    Returns
    -------
    float
        Instantaneous shear rate (1/s)
    """
    elastic_stress = jnp.sum(G_modes * mu_xy_modes) + G_e * gamma
    remaining_stress = sigma_0 - elastic_stress

    # eta_s * gamma_dot = remaining_stress
    eta_s_safe = jnp.maximum(eta_s, 1e-30)
    gamma_dot = remaining_stress / eta_s_safe

    return gamma_dot


# =============================================================================
# LAOS ODE RHS (single network, for jax.lax.scan)
# =============================================================================


@jax.jit
def vlb_laos_step(
    carry: tuple,
    t_val: float,
    G0: float,
    k_d: float,
    gamma_0: float,
    omega: float,
    dt: float,
) -> tuple:
    """Single Euler step for LAOS simulation (single network).

    Applies oscillatory shear gamma(t) = gamma_0 * sin(omega * t),
    so gamma_dot(t) = gamma_0 * omega * cos(omega * t).

    Uses explicit Euler for simplicity in jax.lax.scan.

    Parameters
    ----------
    carry : tuple
        (mu_xx, mu_yy, mu_zz, mu_xy) current state
    t_val : float
        Current time
    G0 : float
        Network modulus (Pa)
    k_d : float
        Dissociation rate (1/s)
    gamma_0 : float
        Strain amplitude
    omega : float
        Angular frequency (rad/s)
    dt : float
        Time step (s)

    Returns
    -------
    tuple
        (new_carry, output_tuple) where output_tuple = (sigma, N1)
    """
    mu_xx, mu_yy, mu_zz, mu_xy = carry

    # Instantaneous shear rate
    gamma_dot = gamma_0 * omega * jnp.cos(omega * t_val)

    # ODE RHS
    dmu_xx, dmu_yy, dmu_zz, dmu_xy = vlb_mu_rhs_shear(
        mu_xx, mu_yy, mu_zz, mu_xy, gamma_dot, k_d
    )

    # Euler step
    mu_xx_new = mu_xx + dt * dmu_xx
    mu_yy_new = mu_yy + dt * dmu_yy
    mu_zz_new = mu_zz + dt * dmu_zz
    mu_xy_new = mu_xy + dt * dmu_xy

    # Compute stress and N1 at current state
    sigma = vlb_shear_stress(mu_xy, G0)
    N1 = vlb_normal_stress_1(mu_xx, mu_yy, G0)

    new_carry = (mu_xx_new, mu_yy_new, mu_zz_new, mu_xy_new)
    return new_carry, (sigma, N1)


@jax.jit
def vlb_multi_laos_step(
    carry: tuple,
    t_val: float,
    G_modes: jnp.ndarray,
    kd_modes: jnp.ndarray,
    eta_s: float,
    gamma_0: float,
    omega: float,
    dt: float,
) -> tuple:
    """Single Euler step for multi-network LAOS.

    State: (mu_xy_modes, mu_xx_modes, mu_yy_modes) where each is shape (M,).

    Parameters
    ----------
    carry : tuple
        (mu_xy_modes, mu_xx_modes, mu_yy_modes) current state arrays
    t_val : float
        Current time
    G_modes : jnp.ndarray
        Mode moduli (Pa), shape (M,)
    kd_modes : jnp.ndarray
        Mode dissociation rates (1/s), shape (M,)
    eta_s : float
        Solvent viscosity (Pa*s)
    gamma_0 : float
        Strain amplitude
    omega : float
        Angular frequency (rad/s)
    dt : float
        Time step (s)

    Returns
    -------
    tuple
        (new_carry, (sigma, N1))
    """
    mu_xy_modes, mu_xx_modes, mu_yy_modes = carry

    gamma_dot = gamma_0 * omega * jnp.cos(omega * t_val)

    # ODE RHS for each mode (vectorized)
    dmu_xx = kd_modes * (1.0 - mu_xx_modes) + 2.0 * gamma_dot * mu_xy_modes
    dmu_yy = kd_modes * (1.0 - mu_yy_modes)
    dmu_xy = -kd_modes * mu_xy_modes + gamma_dot * mu_yy_modes

    # Euler step
    mu_xx_new = mu_xx_modes + dt * dmu_xx
    mu_yy_new = mu_yy_modes + dt * dmu_yy
    mu_xy_new = mu_xy_modes + dt * dmu_xy

    # Stress = sum G_i * mu_xy_i + eta_s * gamma_dot
    sigma = jnp.sum(G_modes * mu_xy_modes) + eta_s * gamma_dot
    N1 = jnp.sum(G_modes * (mu_xx_modes - mu_yy_modes))

    new_carry = (mu_xy_new, mu_xx_new, mu_yy_new)
    return new_carry, (sigma, N1)


# =============================================================================
# Bell Breakage Rate Functions
# =============================================================================


@jax.jit
def vlb_breakage_bell(
    mu_xx: float, mu_yy: float, mu_zz: float, k_d_0: float, nu: float
) -> float:
    """Bell model force-dependent dissociation rate.

    k_d(mu) = k_d_0 · exp(nu · (stretch - 1))

    where stretch = sqrt(tr(mu)/3). At equilibrium (tr(mu)=3),
    stretch=1 and k_d = k_d_0.

    Parameters
    ----------
    mu_xx, mu_yy, mu_zz : float
        Distribution tensor diagonal components
    k_d_0 : float
        Unstressed dissociation rate (1/s)
    nu : float
        Force sensitivity parameter (dimensionless)

    Returns
    -------
    float
        Effective dissociation rate (1/s)
    """
    tr_mu = mu_xx + mu_yy + mu_zz
    stretch = jnp.sqrt(jnp.maximum(tr_mu / 3.0, 1e-30))
    return k_d_0 * jnp.exp(nu * (stretch - 1.0))


# =============================================================================
# FENE-P Stress Amplification Functions
# =============================================================================


@jax.jit
def vlb_fene_factor(
    mu_xx: float, mu_yy: float, mu_zz: float, L_max: float
) -> float:
    """FENE-P Peterlin spring factor.

    f = L²/(L² - tr(mu) + 3)

    Note: We use (L² - tr(mu) + 3) because tr(mu_eq) = 3 (not 0).
    The FENE-P divergence occurs when tr(mu) → L² + 3, i.e., when the
    excess trace tr(mu) - 3 approaches L².

    At equilibrium: f(tr=3) = L²/L² = 1.

    Parameters
    ----------
    mu_xx, mu_yy, mu_zz : float
        Distribution tensor diagonal components
    L_max : float
        Maximum extensibility (dimensionless)

    Returns
    -------
    float
        FENE-P factor (dimensionless, >= 1)
    """
    tr_mu = mu_xx + mu_yy + mu_zz
    L2 = L_max * L_max
    return L2 / jnp.maximum(L2 - tr_mu + 3.0, 1e-10)


@jax.jit
def vlb_stress_fene_xy(
    mu_xx: float,
    mu_yy: float,
    mu_zz: float,
    mu_xy: float,
    G0: float,
    L_max: float,
) -> float:
    """FENE-P shear stress: sigma_xy = G0 · f(tr(mu)) · mu_xy.

    Parameters
    ----------
    mu_xx, mu_yy, mu_zz : float
        Diagonal distribution tensor components
    mu_xy : float
        Off-diagonal component
    G0 : float
        Network modulus (Pa)
    L_max : float
        Maximum extensibility

    Returns
    -------
    float
        FENE-P shear stress (Pa)
    """
    f = vlb_fene_factor(mu_xx, mu_yy, mu_zz, L_max)
    return G0 * f * mu_xy


@jax.jit
def vlb_stress_fene_n1(
    mu_xx: float,
    mu_yy: float,
    mu_zz: float,
    G0: float,
    L_max: float,
) -> float:
    """FENE-P first normal stress difference: N1 = G0·f·(mu_xx - mu_yy).

    Parameters
    ----------
    mu_xx, mu_yy, mu_zz : float
        Diagonal distribution tensor components
    G0 : float
        Network modulus (Pa)
    L_max : float
        Maximum extensibility

    Returns
    -------
    float
        N1 (Pa)
    """
    f = vlb_fene_factor(mu_xx, mu_yy, mu_zz, L_max)
    return G0 * f * (mu_xx - mu_yy)


# =============================================================================
# Arrhenius Temperature Shift Functions
# =============================================================================


@jax.jit
def vlb_arrhenius_shift(
    k_d_0: float, E_a: float, T: float, T_ref: float
) -> float:
    """Arrhenius temperature shift for dissociation rate.

    k_d(T) = k_d_0 · exp(-E_a/R · (1/T - 1/T_ref))

    Parameters
    ----------
    k_d_0 : float
        Reference dissociation rate at T_ref (1/s)
    E_a : float
        Activation energy (J/mol)
    T : float
        Temperature (K)
    T_ref : float
        Reference temperature (K)

    Returns
    -------
    float
        Temperature-shifted dissociation rate (1/s)
    """
    R = 8.314  # J/(mol·K)
    return k_d_0 * jnp.exp(-E_a / R * (1.0 / T - 1.0 / T_ref))


@jax.jit
def vlb_thermal_modulus(G0: float, T: float, T_ref: float) -> float:
    """Temperature scaling of network modulus.

    G0(T) = G0_ref · (T/T_ref)

    Based on rubber elasticity: G = c · k_B · T.

    Parameters
    ----------
    G0 : float
        Reference modulus at T_ref (Pa)
    T : float
        Temperature (K)
    T_ref : float
        Reference temperature (K)

    Returns
    -------
    float
        Temperature-scaled modulus (Pa)
    """
    return G0 * T / T_ref


# =============================================================================
# Variant-Aware ODE RHS Builders
# =============================================================================
#
# Factory functions that return JIT-compiled ODE RHS functions for specific
# VLB variant combinations. Python-level if/else on breakage_type and
# stress_type is resolved at JAX trace time, so each combination compiles
# to an optimized function with only the relevant code path.


def build_vlb_ode_rhs(breakage_type="constant", stress_type="linear"):
    """Build variant-specific ODE RHS for VLB distribution tensor.

    Parameters
    ----------
    breakage_type : str
        "constant" or "bell"
    stress_type : str
        "linear" or "fene" (FENE-P only affects stress, not the mu ODE,
        but we include it here for consistency)

    Returns
    -------
    callable
        JIT-compiled f(t, state, gamma_dot, G0, k_d_0, nu, L_max) -> dstate
        where state = [mu_xx, mu_yy, mu_zz, mu_xy].
    """

    @jax.jit
    def ode_rhs(t, state, gamma_dot, G0, k_d_0, nu, L_max):
        mu_xx, mu_yy, mu_zz, mu_xy = state

        # Dissociation rate
        if breakage_type == "bell":
            k_d = vlb_breakage_bell(mu_xx, mu_yy, mu_zz, k_d_0, nu)
        else:
            k_d = k_d_0

        # Distribution tensor evolution (upper-convected derivative)
        dmu_xx = k_d * (1.0 - mu_xx) + 2.0 * gamma_dot * mu_xy
        dmu_yy = k_d * (1.0 - mu_yy)
        dmu_zz = k_d * (1.0 - mu_zz)
        dmu_xy = -k_d * mu_xy + gamma_dot * mu_yy

        return jnp.array([dmu_xx, dmu_yy, dmu_zz, dmu_xy])

    return ode_rhs


def build_vlb_relaxation_ode_rhs(breakage_type="constant", stress_type="linear"):
    """Build variant-specific relaxation ODE RHS (gamma_dot = 0).

    Returns
    -------
    callable
        JIT-compiled f(t, state, G0, k_d_0, nu, L_max) -> dstate.
    """
    rate_ode = build_vlb_ode_rhs(breakage_type, stress_type)

    @jax.jit
    def ode_rhs(t, state, G0, k_d_0, nu, L_max):
        return rate_ode(t, state, 0.0, G0, k_d_0, nu, L_max)

    return ode_rhs


def build_vlb_creep_ode_rhs(breakage_type="constant", stress_type="linear"):
    """Build variant-specific creep ODE RHS (5-state: mu + gamma).

    Stress is held constant; shear rate derived from the constitutive
    equation by enforcing d(sigma_elastic)/dt such that total stress
    remains sigma_applied.

    For Maxwell-type models (eta_s ≈ 0), shear rate is derived from
    the mu_xy evolution equation at constant stress:
        gamma_dot = (k_d * mu_xy + d(mu_xy)/dt_target) / mu_yy
    where d(mu_xy)/dt_target = 0 at constant stress → gamma_dot = k_d * mu_xy / mu_yy.

    For Jeffreys-type (eta_s > 0): sigma_applied = sigma_elastic + eta_s * gamma_dot.

    Returns
    -------
    callable
        JIT-compiled f(t, state, sigma_applied, G0, k_d_0, eta_s,
        nu, L_max) -> dstate where state = [mu_xx, mu_yy, mu_zz, mu_xy, gamma].
    """

    @jax.jit
    def ode_rhs(t, state, sigma_applied, G0, k_d_0, eta_s, nu, L_max):
        mu_xx, mu_yy, mu_zz, mu_xy, gamma = state

        # Dissociation rate
        if breakage_type == "bell":
            k_d = vlb_breakage_bell(mu_xx, mu_yy, mu_zz, k_d_0, nu)
        else:
            k_d = k_d_0

        # Elastic stress from distribution tensor
        if stress_type == "fene":
            sigma_elastic = vlb_stress_fene_xy(
                mu_xx, mu_yy, mu_zz, mu_xy, G0, L_max
            )
        else:
            sigma_elastic = G0 * mu_xy

        # Shear rate from stress balance
        # Two regimes: if eta_s is significant, use Jeffreys approach;
        # otherwise derive from constitutive equation
        # sigma_applied = sigma_elastic + eta_s * gamma_dot
        # For pure Maxwell (eta_s → 0), d(sigma_elastic)/dt = 0 at steady creep
        # → gamma_dot = k_d * mu_xy / max(mu_yy, eps)
        mu_yy_safe = jnp.maximum(mu_yy, 1e-10)
        gamma_dot_maxwell = k_d * mu_xy / mu_yy_safe
        gamma_dot_jeffreys = (sigma_applied - sigma_elastic) / jnp.maximum(eta_s, 1e-20)
        # Use Jeffreys when eta_s is appreciable, Maxwell otherwise
        eta_threshold = 1e-6 * G0 / k_d_0
        gamma_dot = jnp.where(
            eta_s > eta_threshold,
            gamma_dot_jeffreys,
            gamma_dot_maxwell,
        )

        # Distribution tensor evolution (upper-convected)
        dmu_xx = k_d * (1.0 - mu_xx) + 2.0 * gamma_dot * mu_xy
        dmu_yy = k_d * (1.0 - mu_yy)
        dmu_zz = k_d * (1.0 - mu_zz)
        dmu_xy = -k_d * mu_xy + gamma_dot * mu_yy

        return jnp.array([dmu_xx, dmu_yy, dmu_zz, dmu_xy, gamma_dot])

    return ode_rhs


def build_vlb_laos_ode_rhs(breakage_type="constant", stress_type="linear"):
    """Build variant-specific LAOS ODE RHS.

    Oscillatory shear: gamma_dot(t) = gamma_0 * omega * cos(omega * t).

    Returns
    -------
    callable
        JIT-compiled f(t, state, gamma_0, omega, G0, k_d_0, nu, L_max) -> dstate.
    """
    rate_ode = build_vlb_ode_rhs(breakage_type, stress_type)

    @jax.jit
    def ode_rhs(t, state, gamma_0, omega, G0, k_d_0, nu, L_max):
        gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
        return rate_ode(t, state, gamma_dot, G0, k_d_0, nu, L_max)

    return ode_rhs


# =============================================================================
# Stress Computation Dispatch
# =============================================================================


def build_vlb_stress_fn(stress_type="linear"):
    """Build stress computation function for given stress type.

    Parameters
    ----------
    stress_type : str
        "linear" or "fene"

    Returns
    -------
    callable
        JIT-compiled f(mu_xx, mu_yy, mu_zz, mu_xy, G0, L_max) -> sigma_xy
    """
    if stress_type == "fene":

        @jax.jit
        def stress_fn(mu_xx, mu_yy, mu_zz, mu_xy, G0, L_max):
            return vlb_stress_fene_xy(mu_xx, mu_yy, mu_zz, mu_xy, G0, L_max)

    else:

        @jax.jit
        def stress_fn(mu_xx, mu_yy, mu_zz, mu_xy, G0, L_max):
            return G0 * mu_xy

    return stress_fn


# =============================================================================
# Nonlocal Kernels (Spatial PDE)
# =============================================================================


@jax.jit
def laplacian_1d_neumann_vlb(
    field: jnp.ndarray, dy: float
) -> jnp.ndarray:
    """1D Laplacian with Neumann (zero-flux) boundary conditions.

    ∂²f/∂y² using second-order central differences with ghost points
    for zero-flux BC: df/dy = 0 at y=0 and y=H.

    Parameters
    ----------
    field : jnp.ndarray
        Field values on spatial grid, shape (N_y,)
    dy : float
        Grid spacing (m)

    Returns
    -------
    jnp.ndarray
        Laplacian of field, shape (N_y,)
    """
    # Pad with ghost points for Neumann BC
    padded = jnp.concatenate([field[0:1], field, field[-1:]])
    return (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / (dy * dy)


def build_vlb_nonlocal_pde_rhs(breakage_type="constant", stress_type="linear"):
    """Build PDE RHS for nonlocal VLB with tensor diffusion.

    State vector: [Sigma, mu_xx[0:N_y], mu_yy[0:N_y], mu_zz[0:N_y],
                   mu_xy[0:N_y]]
    where Sigma is the uniform wall stress and mu fields are spatial.

    PDE: dmu/dt = k_d(I - mu) + L·mu + mu·L^T + D_mu * nabla^2(mu)

    Parameters
    ----------
    breakage_type : str
        "constant" or "bell"
    stress_type : str
        "linear" or "fene"

    Returns
    -------
    callable
        JIT-compiled f(t, state, gamma_dot_avg, G0, k_d_0, nu, L_max,
        D_mu, dy, n_points) -> dstate.
    """

    @jax.jit
    def pde_rhs(t, state, gamma_dot_avg, G0, k_d_0, nu, L_max, D_mu, dy, n_points):
        n = n_points

        # Unpack state: Sigma (scalar) + 4 spatial fields
        Sigma = state[0]
        mu_xx = state[1 : 1 + n]
        mu_yy = state[1 + n : 1 + 2 * n]
        mu_zz = state[1 + 2 * n : 1 + 3 * n]
        mu_xy = state[1 + 3 * n : 1 + 4 * n]

        # Local dissociation rate at each spatial point
        if breakage_type == "bell":
            k_d = jax.vmap(
                lambda xx, yy, zz: vlb_breakage_bell(xx, yy, zz, k_d_0, nu)
            )(mu_xx, mu_yy, mu_zz)
        else:
            k_d = jnp.full(n, k_d_0)

        # Local stress → local shear rate
        if stress_type == "fene":
            sigma_elastic = jax.vmap(
                lambda xx, yy, zz, xy: vlb_stress_fene_xy(xx, yy, zz, xy, G0, L_max)
            )(mu_xx, mu_yy, mu_zz, mu_xy)
        else:
            sigma_elastic = G0 * mu_xy

        # gamma_dot(y) from stress balance: Sigma = sigma_elastic(y) + eta_visc * gamma_dot(y)
        # For viscoelastic models, we use: gamma_dot = (Sigma - sigma_elastic) / eta_reg
        # where eta_reg is a small regularization viscosity
        eta_reg = 1e-6 * G0 / k_d_0
        gamma_dot = (Sigma - sigma_elastic) / eta_reg

        # Local mu evolution (vectorized over space)
        dmu_xx = k_d * (1.0 - mu_xx) + 2.0 * gamma_dot * mu_xy
        dmu_yy = k_d * (1.0 - mu_yy)
        dmu_zz = k_d * (1.0 - mu_zz)
        dmu_xy = -k_d * mu_xy + gamma_dot * mu_yy

        # Add diffusion terms
        dmu_xx = dmu_xx + D_mu * laplacian_1d_neumann_vlb(mu_xx, dy)
        dmu_yy = dmu_yy + D_mu * laplacian_1d_neumann_vlb(mu_yy, dy)
        dmu_zz = dmu_zz + D_mu * laplacian_1d_neumann_vlb(mu_zz, dy)
        dmu_xy = dmu_xy + D_mu * laplacian_1d_neumann_vlb(mu_xy, dy)

        # Stress evolution: enforce average shear rate constraint
        # integral(gamma_dot) * dy / H = gamma_dot_avg
        # dSigma/dt = K * (gamma_dot_avg - mean(gamma_dot))
        # K is a large feedback gain for quasi-static enforcement
        K = 10.0 * G0
        mean_gamma_dot = jnp.mean(gamma_dot)
        dSigma = K * (gamma_dot_avg - mean_gamma_dot)

        return jnp.concatenate([
            jnp.array([dSigma]),
            dmu_xx, dmu_yy, dmu_zz, dmu_xy,
        ])

    return pde_rhs
