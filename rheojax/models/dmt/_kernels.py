"""JIT-compiled kernels for DMT (de Souza Mendes-Thompson) models.

This module provides JAX-compiled functions for:
1. Material functions (viscosity, elastic modulus, yield stress)
2. Structure parameter evolution
3. Stress evolution (Maxwell backbone)
4. Full ODE right-hand-side functions for different protocols

All functions use `safe_import_jax()` for float64 enforcement.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

# Safe JAX import
jax, jnp = safe_import_jax()


# =============================================================================
# Material Functions
# =============================================================================


@jax.jit
def viscosity_exponential(
    lam: jnp.ndarray,
    eta_0: float,
    eta_inf: float,
) -> jnp.ndarray:
    """Exponential viscosity closure (original DMT 2013).

    η(λ) = η_∞ · (η_0/η_∞)^λ

    This provides smooth interpolation:
    - λ = 1 (fully structured): η = η_0
    - λ = 0 (fully broken): η = η_∞

    Parameters
    ----------
    lam : array
        Structure parameter λ ∈ [0, 1]
    eta_0 : float
        Zero-shear viscosity (λ=1)
    eta_inf : float
        Infinite-shear viscosity (λ=0)

    Returns
    -------
    array
        Viscosity η(λ)
    """
    # Ensure positive ratio for power
    ratio = jnp.maximum(eta_0 / eta_inf, 1e-10)
    return eta_inf * jnp.power(ratio, lam)


@jax.jit
def viscosity_herschel_bulkley_regularized(
    lam: jnp.ndarray,
    gamma_dot: jnp.ndarray,
    tau_y0: float,
    K0: float,
    n_flow: float,
    eta_inf: float,
    m1: float,
    m2: float,
    m_reg: float = 1e6,
) -> jnp.ndarray:
    """Herschel-Bulkley viscosity with Papanastasiou regularization.

    η = τ_y(λ)·(1 - exp(-m·|γ̇|))/|γ̇| + K(λ)·|γ̇|^(n-1) + η_∞

    Structure-dependent parameters:
    - τ_y(λ) = τ_y0 · λ^m1
    - K(λ) = K0 · λ^m2

    The Papanastasiou regularization avoids the 1/|γ̇| singularity:
    - As γ̇ → 0: term → τ_y · m (finite slope)
    - As γ̇ → ∞: term → τ_y / |γ̇| (standard HB)

    Parameters
    ----------
    lam : array
        Structure parameter λ ∈ [0, 1]
    gamma_dot : array
        Shear rate γ̇
    tau_y0 : float
        Fully-structured yield stress
    K0 : float
        Fully-structured consistency
    n_flow : float
        Flow index
    eta_inf : float
        Infinite-shear viscosity
    m1 : float
        Yield stress exponent
    m2 : float
        Consistency exponent
    m_reg : float
        Papanastasiou regularization parameter (larger = sharper yield)

    Returns
    -------
    array
        Effective viscosity η(λ, γ̇)
    """
    gamma_dot_abs = jnp.maximum(jnp.abs(gamma_dot), 1e-12)

    # Structure-dependent parameters
    lam_safe = jnp.maximum(lam, 1e-10)
    tau_y = tau_y0 * jnp.power(lam_safe, m1)
    K = K0 * jnp.power(lam_safe, m2)

    # Papanastasiou regularization for yield term
    yield_term = tau_y * (1.0 - jnp.exp(-m_reg * gamma_dot_abs)) / gamma_dot_abs

    # Power-law term
    power_term = K * jnp.power(gamma_dot_abs, n_flow - 1.0)

    return yield_term + power_term + eta_inf


@jax.jit
def yield_stress(lam: jnp.ndarray, tau_y0: float, m1: float) -> jnp.ndarray:
    """Structure-dependent yield stress.

    τ_y(λ) = τ_y0 · λ^m1

    Parameters
    ----------
    lam : array
        Structure parameter λ ∈ [0, 1]
    tau_y0 : float
        Fully-structured yield stress
    m1 : float
        Yield stress exponent

    Returns
    -------
    array
        Yield stress τ_y(λ)
    """
    lam_safe = jnp.maximum(lam, 1e-10)
    return tau_y0 * jnp.power(lam_safe, m1)


@jax.jit
def consistency(lam: jnp.ndarray, K0: float, m2: float) -> jnp.ndarray:
    """Structure-dependent consistency.

    K(λ) = K0 · λ^m2

    Parameters
    ----------
    lam : array
        Structure parameter λ ∈ [0, 1]
    K0 : float
        Fully-structured consistency
    m2 : float
        Consistency exponent

    Returns
    -------
    array
        Consistency K(λ)
    """
    lam_safe = jnp.maximum(lam, 1e-10)
    return K0 * jnp.power(lam_safe, m2)


@jax.jit
def elastic_modulus(lam: jnp.ndarray, G0: float, m_G: float) -> jnp.ndarray:
    """Structure-dependent elastic modulus.

    G(λ) = G0 · λ^m_G

    Parameters
    ----------
    lam : array
        Structure parameter λ ∈ [0, 1]
    G0 : float
        Fully-structured elastic modulus
    m_G : float
        Modulus exponent

    Returns
    -------
    array
        Elastic modulus G(λ)
    """
    lam_safe = jnp.maximum(lam, 1e-10)
    return G0 * jnp.power(lam_safe, m_G)


# =============================================================================
# Equilibrium Structure
# =============================================================================


@jax.jit
def equilibrium_structure(
    gamma_dot: jnp.ndarray,
    a: float,
    c: float,
) -> jnp.ndarray:
    """Equilibrium structure parameter at given shear rate.

    At steady state (dλ/dt = 0):

    λ_eq(γ̇) = 1 / (1 + a·|γ̇|^c)

    Limiting behaviors:
    - γ̇ → 0: λ_eq → 1 (fully structured)
    - γ̇ → ∞: λ_eq → 0 (fully broken)

    Parameters
    ----------
    gamma_dot : array
        Shear rate γ̇
    a : float
        Breakdown rate coefficient
    c : float
        Breakdown rate exponent

    Returns
    -------
    array
        Equilibrium structure λ_eq(γ̇)
    """
    gamma_dot_abs = jnp.abs(gamma_dot)
    return 1.0 / (1.0 + a * jnp.power(gamma_dot_abs + 1e-12, c))


# =============================================================================
# Structure Evolution
# =============================================================================


@jax.jit
def structure_evolution(
    lam: jnp.ndarray,
    gamma_dot: jnp.ndarray,
    t_eq: float,
    a: float,
    c: float,
) -> jnp.ndarray:
    """DMT structure kinetics.

    dλ/dt = (1 - λ)/t_eq - a·λ·|γ̇|^c / t_eq

    First term: Buildup (aging) toward λ = 1
    Second term: Breakdown (rejuvenation) toward λ = 0

    This can also be written as:
    dλ/dt = (λ_eq - λ) · r
    where r = 1/t_eq + a·|γ̇|^c/t_eq is the total rate.

    Parameters
    ----------
    lam : array
        Current structure parameter
    gamma_dot : array
        Current shear rate
    t_eq : float
        Equilibrium (buildup) timescale
    a : float
        Breakdown rate coefficient
    c : float
        Breakdown rate exponent

    Returns
    -------
    array
        Time derivative dλ/dt
    """
    gamma_dot_abs = jnp.abs(gamma_dot)

    # Buildup term (toward λ = 1)
    buildup = (1.0 - lam) / t_eq

    # Breakdown term (toward λ = 0)
    breakdown = a * lam * jnp.power(gamma_dot_abs + 1e-12, c) / t_eq

    return buildup - breakdown


# =============================================================================
# Maxwell Stress Evolution
# =============================================================================


@jax.jit
def maxwell_stress_evolution(
    sigma: jnp.ndarray,
    gamma_dot: jnp.ndarray,
    G: jnp.ndarray,
    theta_1: jnp.ndarray,
) -> jnp.ndarray:
    """Maxwell stress evolution equation.

    dσ/dt = G·γ̇ - σ/θ₁

    where θ₁ = η/G is the relaxation time.

    Parameters
    ----------
    sigma : array
        Current stress
    gamma_dot : array
        Current shear rate
    G : array
        Current elastic modulus G(λ)
    theta_1 : array
        Current relaxation time θ₁(λ)

    Returns
    -------
    array
        Time derivative dσ/dt
    """
    return G * gamma_dot - sigma / jnp.maximum(theta_1, 1e-12)


# =============================================================================
# ODE Right-Hand-Side Functions
# =============================================================================


def rhs_viscous_rate_control(t, state, args):
    """RHS for DMT-Viscous under rate control (shear rate prescribed).

    State vector: [λ, γ]
    - λ: Structure parameter
    - γ: Total accumulated strain

    Parameters
    ----------
    t : float
        Current time
    state : array
        [λ, γ]
    args : dict
        Parameters including gamma_dot, t_eq, a, c

    Returns
    -------
    array
        [dλ/dt, dγ/dt]
    """
    lam = state[0]
    gamma_dot = args["gamma_dot"]
    t_eq = args["t_eq"]
    a = args["a"]
    c = args["c"]

    # Structure evolution
    dlam_dt = structure_evolution(lam, gamma_dot, t_eq, a, c)

    # Strain accumulation
    dgamma_dt = gamma_dot

    return jnp.array([dlam_dt, dgamma_dt])


def rhs_maxwell_rate_control(t, state, args):
    """RHS for DMT-Maxwell under rate control (shear rate prescribed).

    State vector: [σ, λ, γ]
    - σ: Shear stress
    - λ: Structure parameter
    - γ: Total accumulated strain

    Parameters
    ----------
    t : float
        Current time
    state : array
        [σ, λ, γ]
    args : dict
        Parameters including gamma_dot, closure type, and material parameters

    Returns
    -------
    array
        [dσ/dt, dλ/dt, dγ/dt]
    """
    sigma, lam, gamma = state
    gamma_dot = args["gamma_dot"]

    # Structure evolution
    dlam_dt = structure_evolution(lam, gamma_dot, args["t_eq"], args["a"], args["c"])

    # Elastic modulus
    G = elastic_modulus(lam, args["G0"], args["m_G"])

    # Viscosity depends on closure
    if args["closure"] == "exponential":
        eta = viscosity_exponential(lam, args["eta_0"], args["eta_inf"])
    else:
        eta = viscosity_herschel_bulkley_regularized(
            lam,
            gamma_dot,
            args["tau_y0"],
            args["K0"],
            args["n_flow"],
            args["eta_inf"],
            args["m1"],
            args["m2"],
        )

    # Relaxation time
    theta_1 = eta / jnp.maximum(G, 1e-10)

    # Stress evolution
    dsigma_dt = maxwell_stress_evolution(sigma, gamma_dot, G, theta_1)

    # Strain accumulation
    dgamma_dt = gamma_dot

    return jnp.array([dsigma_dt, dlam_dt, dgamma_dt])


def rhs_maxwell_relaxation(t, state, args):
    """RHS for DMT-Maxwell stress relaxation (γ̇ = 0 after step strain).

    State vector: [σ, λ]
    - σ: Shear stress (relaxing)
    - λ: Structure parameter (recovering toward λ=1)

    Parameters
    ----------
    t : float
        Current time
    state : array
        [σ, λ]
    args : dict
        Parameters

    Returns
    -------
    array
        [dσ/dt, dλ/dt]
    """
    sigma, lam = state

    # Structure recovery (no breakdown, γ̇ = 0)
    # dλ/dt = (1 - λ)/t_eq
    dlam_dt = (1.0 - lam) / args["t_eq"]

    # Elastic modulus
    G = elastic_modulus(lam, args["G0"], args["m_G"])

    # For relaxation, use zero shear rate in viscosity
    if args["closure"] == "exponential":
        eta = viscosity_exponential(lam, args["eta_0"], args["eta_inf"])
    else:
        # HB at zero shear rate: just eta_inf (regularized)
        eta = viscosity_herschel_bulkley_regularized(
            lam,
            0.0,
            args["tau_y0"],
            args["K0"],
            args["n_flow"],
            args["eta_inf"],
            args["m1"],
            args["m2"],
        )

    # Relaxation time
    theta_1 = eta / jnp.maximum(G, 1e-10)

    # Stress relaxation (no driving term)
    dsigma_dt = -sigma / jnp.maximum(theta_1, 1e-12)

    return jnp.array([dsigma_dt, dlam_dt])


# =============================================================================
# Steady-State Flow Curve Functions
# =============================================================================


@jax.jit
def steady_stress_exponential(
    gamma_dot: jnp.ndarray,
    eta_0: float,
    eta_inf: float,
    a: float,
    c: float,
) -> jnp.ndarray:
    """Steady-state stress for exponential closure.

    σ(γ̇) = η(λ_eq(γ̇)) · γ̇

    Parameters
    ----------
    gamma_dot : array
        Shear rate
    eta_0, eta_inf : float
        Viscosity bounds
    a, c : float
        Breakdown kinetics parameters

    Returns
    -------
    array
        Steady-state stress
    """
    lam_eq = equilibrium_structure(gamma_dot, a, c)
    eta = viscosity_exponential(lam_eq, eta_0, eta_inf)
    return eta * gamma_dot


@jax.jit
def steady_stress_herschel_bulkley(
    gamma_dot: jnp.ndarray,
    tau_y0: float,
    K0: float,
    n_flow: float,
    eta_inf: float,
    a: float,
    c: float,
    m1: float,
    m2: float,
) -> jnp.ndarray:
    """Steady-state stress for Herschel-Bulkley closure.

    σ(γ̇) = τ_y(λ_eq) + K(λ_eq)·|γ̇|^n + η_∞·|γ̇|

    Parameters
    ----------
    gamma_dot : array
        Shear rate
    tau_y0, K0 : float
        Yield stress and consistency at λ=1
    n_flow : float
        Flow index
    eta_inf : float
        Infinite-shear viscosity
    a, c : float
        Breakdown kinetics parameters
    m1, m2 : float
        Structure exponents

    Returns
    -------
    array
        Steady-state stress
    """
    lam_eq = equilibrium_structure(gamma_dot, a, c)
    gamma_dot_abs = jnp.maximum(jnp.abs(gamma_dot), 1e-12)

    # Structure-dependent parameters
    tau_y = yield_stress(lam_eq, tau_y0, m1)
    K = consistency(lam_eq, K0, m2)

    # HB stress (without Papanastasiou regularization for steady state)
    return tau_y + K * jnp.power(gamma_dot_abs, n_flow) + eta_inf * gamma_dot_abs


# =============================================================================
# SAOS Functions (Maxwell variant only)
# =============================================================================


@jax.jit
def saos_moduli_maxwell(
    omega: jnp.ndarray,
    G: float,
    theta_1: float,
    eta_inf: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """SAOS moduli for linearized Maxwell model.

    G'(ω) = G · (ωθ₁)² / (1 + (ωθ₁)²)
    G''(ω) = G · ωθ₁ / (1 + (ωθ₁)²) + η_∞·ω

    This assumes small amplitude so λ ≈ λ₀ (constant structure).

    Parameters
    ----------
    omega : array
        Angular frequency
    G : float
        Elastic modulus G(λ₀)
    theta_1 : float
        Relaxation time θ₁(λ₀)
    eta_inf : float
        Solvent/infinite-shear viscosity

    Returns
    -------
    G_prime : array
        Storage modulus
    G_double_prime : array
        Loss modulus
    """
    omega_theta = omega * theta_1
    omega_theta_sq = omega_theta**2
    denom = 1.0 + omega_theta_sq

    G_prime = G * omega_theta_sq / denom
    G_double_prime = G * omega_theta / denom + eta_inf * omega

    return G_prime, G_double_prime


# =============================================================================
# Creep Inversion Helper
# =============================================================================


@jax.jit
def invert_stress_for_gamma_dot_exponential(
    sigma_0: float,
    lam: float,
    eta_0: float,
    eta_inf: float,
) -> float:
    """Solve for γ̇ given σ₀ = η(λ)·γ̇ (exponential closure).

    For exponential closure, this is a direct inversion:
    γ̇ = σ₀ / η(λ)

    Parameters
    ----------
    sigma_0 : float
        Applied stress
    lam : float
        Current structure parameter
    eta_0, eta_inf : float
        Viscosity bounds

    Returns
    -------
    float
        Shear rate γ̇
    """
    eta = viscosity_exponential(lam, eta_0, eta_inf)
    return sigma_0 / jnp.maximum(eta, 1e-12)


def invert_stress_for_gamma_dot_hb(
    sigma_0: float,
    lam: float,
    tau_y0: float,
    K0: float,
    n_flow: float,
    eta_inf: float,
    m1: float,
    m2: float,
    n_iter: int = 10,
) -> float:
    """Solve for γ̇ given σ₀ = τ_y(λ) + K(λ)|γ̇|^n + η_∞|γ̇| (HB closure).

    Uses fixed-point iteration starting from power-law estimate.

    Parameters
    ----------
    sigma_0 : float
        Applied stress
    lam : float
        Current structure parameter
    tau_y0, K0, n_flow, eta_inf, m1, m2 : float
        HB closure parameters
    n_iter : int
        Number of fixed-point iterations

    Returns
    -------
    float
        Shear rate γ̇
    """
    tau_y = yield_stress(lam, tau_y0, m1)
    K = consistency(lam, K0, m2)

    # Below yield: no flow
    below_yield = sigma_0 <= tau_y

    # Initial guess (power-law approximation)
    sigma_excess = jnp.maximum(sigma_0 - tau_y, 1e-12)
    gamma_dot_init = jnp.power(sigma_excess / K, 1.0 / n_flow)

    # Fixed-point iteration
    def iterate(gamma_dot):
        sigma_pred = (
            tau_y + K * jnp.power(gamma_dot + 1e-12, n_flow) + eta_inf * gamma_dot
        )
        # Newton-like update
        gamma_dot_new = gamma_dot * sigma_0 / jnp.maximum(sigma_pred, 1e-12)
        return jnp.clip(gamma_dot_new, 1e-12, 1e8)

    gamma_dot = gamma_dot_init
    for _ in range(n_iter):
        gamma_dot = iterate(gamma_dot)

    # If below yield, return near-zero
    return jnp.where(below_yield, 1e-12, gamma_dot)
