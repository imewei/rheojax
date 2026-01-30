"""JAX-accelerated physics kernels for Transient Network Theory (TNT) models.

This module provides JIT-compiled functions for:
1. Breakage rate functions (constant, Bell, power-law)
2. Stress formulas (linear, FENE-P)
3. Convective derivative terms (upper-convected, Gordon-Schowalter)
4. Analytical solutions for steady and oscillatory shear
5. ODE right-hand sides for transient protocols

All functions are designed for float64 precision and JAX compatibility.

Key Equation
------------
The TNT constitutive equation for the conformation tensor S::

    dS/dt = L·S + S·L^T + g₀·I - β(S)·S

where:
- S is the conformation tensor (dimensionless, equilibrium S = I)
- L is the velocity gradient tensor
- g₀ is the creation rate (= β_eq for equilibrium)
- β(S) is the destruction (breakage) rate

Stress is computed from the conformation tensor::

    σ = G·stress_fn(S) + η_s·γ̇

State Vector Convention
-----------------------
For 3D incompressible simple shear, we use a 4-component state::

    state = [S_xx, S_yy, S_zz, S_xy]

Equilibrium: S = I → [1, 1, 1, 0]
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


# =============================================================================
# Breakage Rate Functions
# =============================================================================


@jax.jit
def breakage_constant(S_xx: float, S_yy: float, S_zz: float, tau_b: float) -> float:
    """Constant breakage rate β = 1/τ_b.

    State-independent destruction rate. At equilibrium (S=I),
    creation g₀ = β = 1/τ_b balances destruction.

    Parameters
    ----------
    S_xx, S_yy, S_zz : float
        Conformation tensor diagonal components (dimensionless)
    tau_b : float
        Bond lifetime (s)

    Returns
    -------
    float
        Breakage rate β (1/s)
    """
    return 1.0 / tau_b


@jax.jit
def breakage_bell(
    S_xx: float, S_yy: float, S_zz: float, tau_b: float, nu: float
) -> float:
    """Bell model force-dependent breakage rate.

    β = (1/τ_b) · exp(ν · (stretch - 1))

    where stretch = sqrt(tr(S)/3). At equilibrium (tr(S)=3),
    stretch=1 and β = 1/τ_b.

    Based on Bell's equation for slip-bond kinetics (Bell, 1978).

    Parameters
    ----------
    S_xx, S_yy, S_zz : float
        Conformation tensor diagonal components
    tau_b : float
        Equilibrium bond lifetime (s)
    nu : float
        Force sensitivity parameter (dimensionless)

    Returns
    -------
    float
        Breakage rate β (1/s)
    """
    tr_S = S_xx + S_yy + S_zz
    stretch = jnp.sqrt(jnp.maximum(tr_S / 3.0, 0.0))
    return (1.0 / tau_b) * jnp.exp(nu * (stretch - 1.0))


@jax.jit
def breakage_power_law(
    S_xx: float, S_yy: float, S_zz: float, tau_b: float, m_break: float
) -> float:
    """Power-law breakage rate.

    β = (1/τ_b) · stretch^m

    where stretch = sqrt(tr(S)/3).

    Parameters
    ----------
    S_xx, S_yy, S_zz : float
        Conformation tensor diagonal components
    tau_b : float
        Equilibrium bond lifetime (s)
    m_break : float
        Breakage power-law exponent

    Returns
    -------
    float
        Breakage rate β (1/s)
    """
    tr_S = S_xx + S_yy + S_zz
    stretch = jnp.sqrt(jnp.maximum(tr_S / 3.0, 1e-30))
    return (1.0 / tau_b) * jnp.power(stretch, m_break)


# =============================================================================
# Stress Functions
# =============================================================================


@jax.jit
def stress_linear_xy(S_xy: float, G: float) -> float:
    """Linear (Gaussian) stress: σ_xy = G · S_xy.

    Parameters
    ----------
    S_xy : float
        Off-diagonal conformation tensor component
    G : float
        Network modulus (Pa)

    Returns
    -------
    float
        Shear stress contribution (Pa)
    """
    return G * S_xy


@jax.jit
def stress_fene_xy(
    S_xx: float, S_yy: float, S_zz: float, S_xy: float, G: float, L_max: float
) -> float:
    """FENE-P stress: σ_xy = G · f(tr(S)) · S_xy.

    f = L²/(L² - tr(S)) is the Peterlin spring factor.
    Diverges as tr(S) → L² (full extension).

    Parameters
    ----------
    S_xx, S_yy, S_zz : float
        Diagonal conformation tensor components
    S_xy : float
        Off-diagonal component
    G : float
        Network modulus (Pa)
    L_max : float
        Maximum extensibility (dimensionless)

    Returns
    -------
    float
        FENE-P shear stress contribution (Pa)
    """
    tr_S = S_xx + S_yy + S_zz
    L2 = L_max * L_max
    f = L2 / jnp.maximum(L2 - tr_S, 1e-10)
    return G * f * S_xy


@jax.jit
def stress_fene_n1(
    S_xx: float, S_yy: float, S_zz: float, G: float, L_max: float
) -> float:
    """FENE-P first normal stress difference: N1 = G·f·(S_xx - S_yy).

    Parameters
    ----------
    S_xx, S_yy, S_zz : float
        Diagonal conformation tensor components
    G : float
        Network modulus (Pa)
    L_max : float
        Maximum extensibility

    Returns
    -------
    float
        N1 (Pa)
    """
    tr_S = S_xx + S_yy + S_zz
    L2 = L_max * L_max
    f = L2 / jnp.maximum(L2 - tr_S, 1e-10)
    return G * f * (S_xx - S_yy)


# =============================================================================
# Convective Derivative Terms
# =============================================================================


@jax.jit
def upper_convected_2d(
    S_xx: float, S_yy: float, S_xy: float, gamma_dot: float
) -> tuple[float, float, float]:
    """Upper-convected derivative terms in simple shear.

    For simple shear with velocity gradient L = [[0, γ̇, 0], [0, 0, 0], [0, 0, 0]]::

        (L·S + S·L^T)_xx = 2γ̇·S_xy
        (L·S + S·L^T)_yy = 0
        (L·S + S·L^T)_xy = γ̇·S_yy

    Note: S_zz has no convective contribution in simple shear.

    Parameters
    ----------
    S_xx : float
        Normal conformation component S_xx
    S_yy : float
        Normal conformation component S_yy
    S_xy : float
        Shear conformation component S_xy
    gamma_dot : float
        Shear rate (1/s)

    Returns
    -------
    tuple[float, float, float]
        Convective terms: (conv_xx, conv_yy, conv_xy)
    """
    conv_xx = 2.0 * gamma_dot * S_xy
    conv_yy = 0.0
    conv_xy = gamma_dot * S_yy
    return conv_xx, conv_yy, conv_xy


@jax.jit
def gordon_schowalter_2d(
    S_xx: float,
    S_yy: float,
    S_zz: float,
    S_xy: float,
    gamma_dot: float,
    xi: float,
) -> tuple[float, float, float]:
    """Gordon-Schowalter derivative terms for non-affine motion.

    Effective velocity gradient L_eff = L - (ξ/2)·D gives::

        conv_xx = (2-ξ)·γ̇·S_xy
        conv_yy = -ξ·γ̇·S_xy
        conv_xy = γ̇·S_yy - (ξ/2)·γ̇·(S_xx + S_yy)

    When ξ=0 recovers upper-convected; ξ=1 gives corotational.
    Non-zero ξ produces N₂ ≠ 0.

    Parameters
    ----------
    S_xx, S_yy, S_zz : float
        Diagonal conformation components
    S_xy : float
        Shear conformation component
    gamma_dot : float
        Shear rate (1/s)
    xi : float
        Slip parameter (0=upper-convected, 1=corotational)

    Returns
    -------
    tuple[float, float, float]
        Convective terms: (conv_xx, conv_yy, conv_xy)
    """
    conv_xx = (2.0 - xi) * gamma_dot * S_xy
    conv_yy = -xi * gamma_dot * S_xy
    conv_xy = gamma_dot * S_yy - 0.5 * xi * gamma_dot * (S_xx + S_yy)
    return conv_xx, conv_yy, conv_xy


# =============================================================================
# Analytical Steady-State Solutions (constant breakage, linear stress)
# =============================================================================


@jax.jit
def tnt_base_steady_conformation(
    gamma_dot: float, tau_b: float
) -> tuple[float, float, float, float]:
    """Steady-state conformation tensor for basic TNT (UCM-like).

    At steady state dS/dt = 0 with constant breakage β = 1/τ_b::

        S_xx = 1 + 2(τ_b·γ̇)²/(1 + (τ_b·γ̇)²)     ... wait, let me derive correctly.

    Actually for UCM network with upper-convected derivative and constant
    breakage/creation (g₀ = β = 1/τ_b):

        conv + g₀·I - β·S = 0

    Components in simple shear:
        2γ̇·S_xy + 1/τ_b - S_xx/τ_b = 0  →  S_xx = 1 + 2τ_b·γ̇·S_xy
        0 + 1/τ_b - S_yy/τ_b = 0          →  S_yy = 1
        0 + 1/τ_b - S_zz/τ_b = 0          →  S_zz = 1
        γ̇·S_yy - S_xy/τ_b = 0             →  S_xy = τ_b·γ̇

    So:
        S_xy = τ_b·γ̇
        S_xx = 1 + 2(τ_b·γ̇)²
        S_yy = 1
        S_zz = 1

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    tau_b : float
        Bond lifetime (s)

    Returns
    -------
    tuple[float, float, float, float]
        (S_xx, S_yy, S_zz, S_xy)
    """
    wi = tau_b * gamma_dot
    S_xy = wi
    S_xx = 1.0 + 2.0 * wi * wi
    S_yy = 1.0
    S_zz = 1.0
    return S_xx, S_yy, S_zz, S_xy


@jax.jit
def tnt_base_steady_stress(
    gamma_dot: float, G: float, tau_b: float, eta_s: float
) -> float:
    """Steady shear stress for basic TNT model.

    σ = G·S_xy + η_s·γ̇ = G·τ_b·γ̇ + η_s·γ̇

    This is the UCM steady-state result (no shear thinning for constant breakage).

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    G : float
        Network modulus (Pa)
    tau_b : float
        Bond lifetime (s)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    float
        Steady shear stress (Pa)
    """
    return G * tau_b * gamma_dot + eta_s * gamma_dot


# Vectorized version
tnt_base_steady_stress_vec = jax.jit(
    jax.vmap(tnt_base_steady_stress, in_axes=(0, None, None, None))
)


@jax.jit
def tnt_base_steady_n1(gamma_dot: float, G: float, tau_b: float) -> float:
    """First normal stress difference for basic TNT.

    N₁ = G·(S_xx - S_yy) = G·2·(τ_b·γ̇)² = 2G·τ_b²·γ̇²

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    G : float
        Network modulus (Pa)
    tau_b : float
        Bond lifetime (s)

    Returns
    -------
    float
        First normal stress difference N₁ (Pa)
    """
    wi = tau_b * gamma_dot
    return 2.0 * G * wi * wi


tnt_base_steady_n1_vec = jax.jit(jax.vmap(tnt_base_steady_n1, in_axes=(0, None, None)))


# =============================================================================
# SAOS (Small-Amplitude Oscillatory Shear)
# =============================================================================


@jax.jit
def tnt_saos_moduli(
    omega: float, G: float, tau_b: float, eta_s: float
) -> tuple[float, float]:
    """Storage and loss moduli for single-mode TNT (Maxwell-like).

    In the linear regime, all TNT variants reduce to Maxwell behavior::

        G'(ω) = G·(ωτ_b)² / (1 + (ωτ_b)²)
        G''(ω) = G·(ωτ_b) / (1 + (ωτ_b)²) + η_s·ω

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    G : float
        Network modulus (Pa)
    tau_b : float
        Bond lifetime (s)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    tuple[float, float]
        (G', G'') in Pa
    """
    wt = omega * tau_b
    wt2 = wt * wt
    denom = 1.0 + wt2
    G_prime = G * wt2 / denom
    G_double_prime = G * wt / denom + eta_s * omega
    return G_prime, G_double_prime


tnt_saos_moduli_vec = jax.jit(jax.vmap(tnt_saos_moduli, in_axes=(0, None, None, None)))


@jax.jit
def tnt_multimode_saos_moduli(
    omega: float, G_modes: jnp.ndarray, tau_modes: jnp.ndarray, eta_s: float
) -> tuple[float, float]:
    """Multi-mode SAOS moduli by superposition.

    G'(ω) = Σ G_k·(ωτ_k)² / (1 + (ωτ_k)²)
    G''(ω) = Σ G_k·(ωτ_k) / (1 + (ωτ_k)²) + η_s·ω

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    G_modes : jnp.ndarray
        Mode moduli (Pa), shape (N,)
    tau_modes : jnp.ndarray
        Mode relaxation times (s), shape (N,)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    tuple[float, float]
        (G', G'') in Pa
    """
    wt = omega * tau_modes
    wt2 = wt * wt
    denom = 1.0 + wt2
    G_prime = jnp.sum(G_modes * wt2 / denom)
    G_double_prime = jnp.sum(G_modes * wt / denom) + eta_s * omega
    return G_prime, G_double_prime


tnt_multimode_saos_moduli_vec = jax.jit(
    jax.vmap(tnt_multimode_saos_moduli, in_axes=(0, None, None, None))
)


# =============================================================================
# Relaxation
# =============================================================================


@jax.jit
def tnt_base_relaxation(t: float, sigma_0: float, tau_b: float) -> float:
    """Single-exponential stress relaxation for basic TNT.

    σ(t) = σ₀ · exp(-t/τ_b)

    Parameters
    ----------
    t : float
        Time since cessation (s)
    sigma_0 : float
        Initial stress at t=0 (Pa)
    tau_b : float
        Bond lifetime / relaxation time (s)

    Returns
    -------
    float
        Relaxing stress (Pa)
    """
    return sigma_0 * jnp.exp(-t / tau_b)


tnt_base_relaxation_vec = jax.jit(
    jax.vmap(tnt_base_relaxation, in_axes=(0, None, None))
)


@jax.jit
def tnt_multimode_relaxation(
    t: float, sigma_0_modes: jnp.ndarray, tau_modes: jnp.ndarray
) -> float:
    """Multi-mode exponential stress relaxation.

    σ(t) = Σ σ₀_k · exp(-t/τ_k)

    Parameters
    ----------
    t : float
        Time (s)
    sigma_0_modes : jnp.ndarray
        Initial stress per mode (Pa), shape (N,)
    tau_modes : jnp.ndarray
        Relaxation times (s), shape (N,)

    Returns
    -------
    float
        Total relaxing stress (Pa)
    """
    return jnp.sum(sigma_0_modes * jnp.exp(-t / tau_modes))


tnt_multimode_relaxation_vec = jax.jit(
    jax.vmap(tnt_multimode_relaxation, in_axes=(0, None, None))
)


# =============================================================================
# Cates Living Polymer
# =============================================================================


@jax.jit
def tnt_cates_effective_tau(tau_rep: float, tau_break: float) -> float:
    """Effective relaxation time for Cates living polymers.

    In the fast-breaking limit (τ_break << τ_rep)::

        τ_d = √(τ_rep · τ_break)

    Parameters
    ----------
    tau_rep : float
        Reptation time (s)
    tau_break : float
        Average breaking time (s)

    Returns
    -------
    float
        Effective relaxation time τ_d (s)
    """
    return jnp.sqrt(tau_rep * tau_break)


# =============================================================================
# ODE Right-Hand Sides for Transient Protocols
# =============================================================================


@jax.jit
def tnt_single_mode_ode_rhs(
    t: float,
    state: jnp.ndarray,
    gamma_dot: float,
    G: float,
    tau_b: float,
) -> jnp.ndarray:
    """ODE right-hand side for single-mode TNT (constant breakage).

    State: [S_xx, S_yy, S_zz, S_xy]

    dS/dt = conv(S, γ̇) + (1/τ_b)·I - (1/τ_b)·S

    where conv = L·S + S·L^T (upper-convected).

    Parameters
    ----------
    t : float
        Time (s), unused
    state : jnp.ndarray
        State vector [S_xx, S_yy, S_zz, S_xy]
    gamma_dot : float
        Applied shear rate (1/s)
    G : float
        Network modulus (Pa), unused in ODE but kept for interface consistency
    tau_b : float
        Bond lifetime (s)

    Returns
    -------
    jnp.ndarray
        Time derivatives [dS_xx/dt, dS_yy/dt, dS_zz/dt, dS_xy/dt]
    """
    S_xx, S_yy, S_zz, S_xy = state

    # Upper-convected derivative terms
    conv_xx, conv_yy, conv_xy = upper_convected_2d(S_xx, S_yy, S_xy, gamma_dot)

    # Breakage rate
    beta = 1.0 / tau_b

    # Creation rate (equilibrium balance: g₀ = β at S=I)
    g0 = beta

    # dS/dt = conv + g₀·I - β·S
    dS_xx = conv_xx + g0 - beta * S_xx
    dS_yy = conv_yy + g0 - beta * S_yy
    dS_zz = g0 - beta * S_zz  # No convective term for zz in simple shear
    dS_xy = conv_xy - beta * S_xy  # No identity contribution for off-diagonal

    return jnp.array([dS_xx, dS_yy, dS_zz, dS_xy])


@jax.jit
def tnt_single_mode_ode_rhs_laos(
    t: float,
    state: jnp.ndarray,
    gamma_0: float,
    omega: float,
    G: float,
    tau_b: float,
) -> jnp.ndarray:
    """ODE right-hand side for LAOS.

    γ̇(t) = γ₀·ω·cos(ωt)

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [S_xx, S_yy, S_zz, S_xy]
    gamma_0 : float
        Strain amplitude
    omega : float
        Angular frequency (rad/s)
    G : float
        Network modulus (Pa)
    tau_b : float
        Bond lifetime (s)

    Returns
    -------
    jnp.ndarray
        Time derivatives
    """
    gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
    return tnt_single_mode_ode_rhs(t, state, gamma_dot, G, tau_b)


@jax.jit
def tnt_single_mode_creep_ode_rhs(
    t: float,
    state: jnp.ndarray,
    sigma_applied: float,
    G: float,
    tau_b: float,
    eta_s: float,
) -> jnp.ndarray:
    """ODE right-hand side for creep (stress-controlled).

    State: [S_xx, S_yy, S_zz, S_xy, γ]

    The applied stress is held constant::

        σ = G·S_xy + η_s·γ̇
        γ̇ = (σ - G·S_xy) / η_s

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [S_xx, S_yy, S_zz, S_xy, γ]
    sigma_applied : float
        Applied constant stress (Pa)
    G : float
        Network modulus (Pa)
    tau_b : float
        Bond lifetime (s)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    jnp.ndarray
        Time derivatives [dS_xx/dt, dS_yy/dt, dS_zz/dt, dS_xy/dt, dγ/dt]
    """
    S_xx, S_yy, S_zz, S_xy, gamma = state

    # Compute shear rate from stress constraint
    eta_s_reg = jnp.maximum(eta_s, 1e-10 * G * tau_b)
    gamma_dot = (sigma_applied - G * S_xy) / eta_s_reg

    # Conformation evolution (reuse rate-controlled RHS)
    conf_state = jnp.array([S_xx, S_yy, S_zz, S_xy])
    d_conf = tnt_single_mode_ode_rhs(t, conf_state, gamma_dot, G, tau_b)

    # Strain evolution
    d_gamma = gamma_dot

    return jnp.concatenate([d_conf, jnp.array([d_gamma])])


@jax.jit
def tnt_single_mode_relaxation_ode_rhs(
    t: float,
    state: jnp.ndarray,
    G: float,
    tau_b: float,
) -> jnp.ndarray:
    """ODE right-hand side for stress relaxation (γ̇ = 0).

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [S_xx, S_yy, S_zz, S_xy]
    G : float
        Network modulus (Pa)
    tau_b : float
        Bond lifetime (s)

    Returns
    -------
    jnp.ndarray
        Time derivatives
    """
    return tnt_single_mode_ode_rhs(t, state, 0.0, G, tau_b)


# =============================================================================
# Multi-Mode ODE (for StickyRouse, MultiSpecies)
# =============================================================================


@jax.jit
def tnt_multimode_ode_rhs(
    t: float,
    state: jnp.ndarray,
    gamma_dot: float,
    G_modes: jnp.ndarray,
    tau_modes: jnp.ndarray,
) -> jnp.ndarray:
    """ODE right-hand side for multi-mode TNT dynamics.

    State: [S_xx_0, S_yy_0, S_zz_0, S_xy_0, ..., S_xy_{N-1}]
    Total length: 4*N

    Each mode evolves independently with its own (G_k, τ_k).

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        Concatenated state vector, shape (4*N,)
    gamma_dot : float
        Applied shear rate (1/s)
    G_modes : jnp.ndarray
        Mode moduli (Pa), shape (N,)
    tau_modes : jnp.ndarray
        Mode relaxation times (s), shape (N,)

    Returns
    -------
    jnp.ndarray
        Time derivatives, shape (4*N,)
    """
    n_modes = G_modes.shape[0]
    state_reshaped = state.reshape((n_modes, 4))

    def single_mode_rhs(mode_state, mode_params):
        G_k, tau_k = mode_params
        return tnt_single_mode_ode_rhs(t, mode_state, gamma_dot, G_k, tau_k)

    mode_params = jnp.stack([G_modes, tau_modes], axis=1)
    d_states = jax.vmap(single_mode_rhs)(state_reshaped, mode_params)
    return d_states.flatten()


# =============================================================================
# Variant-Aware ODE RHS Builders (Python-level dispatch before trace)
# =============================================================================
#
# Factory functions that return JIT-compiled ODE RHS functions for specific
# variant combinations. Python-level if/else on (breakage_type, use_fene,
# use_gs) is resolved at JAX trace time, so each combination compiles to an
# optimized function that executes only the relevant code path.
#
# All returned functions accept the full set of variant parameters. Parameters
# irrelevant to the selected variant are passed but ignored by the trace.


def build_tnt_ode_rhs(breakage_type="constant", use_fene=False, use_gs=False):
    """Build a variant-specific ODE RHS for single-mode TNT.

    Parameters
    ----------
    breakage_type : str
        "constant", "bell", "power_law", or "stretch_creation"
    use_fene : bool
        If True, FENE-P stress type (affects creation normalization)
    use_gs : bool
        If True, use Gordon-Schowalter convective derivative

    Returns
    -------
    callable
        JIT-compiled ``f(t, state, gamma_dot, G, tau_b, nu, m_break,
        kappa, L_max, xi) -> dstate`` with state = [S_xx, S_yy, S_zz, S_xy].
    """

    @jax.jit
    def ode_rhs(t, state, gamma_dot, G, tau_b, nu, m_break, kappa, L_max, xi):
        S_xx, S_yy, S_zz, S_xy = state

        # --- Breakage rate beta(S) and creation rate g0 ---
        if breakage_type == "constant":
            beta = 1.0 / tau_b
            g0 = beta
        elif breakage_type == "bell":
            beta = breakage_bell(S_xx, S_yy, S_zz, tau_b, nu)
            g0 = 1.0 / tau_b
        elif breakage_type == "power_law":
            beta = breakage_power_law(S_xx, S_yy, S_zz, tau_b, m_break)
            g0 = 1.0 / tau_b
        elif breakage_type == "stretch_creation":
            beta = 1.0 / tau_b
            tr_S = S_xx + S_yy + S_zz
            stretch = jnp.sqrt(jnp.maximum(tr_S / 3.0, 1e-30))
            g0 = (1.0 + kappa * (stretch - 1.0)) / tau_b
        else:
            beta = 1.0 / tau_b
            g0 = beta

        # --- Convective derivative ---
        if use_gs:
            conv_xx, conv_yy, conv_xy = gordon_schowalter_2d(
                S_xx, S_yy, S_zz, S_xy, gamma_dot, xi
            )
        else:
            conv_xx, conv_yy, conv_xy = upper_convected_2d(S_xx, S_yy, S_xy, gamma_dot)

        # --- dS/dt = conv + g0*I - beta*S ---
        dS_xx = conv_xx + g0 - beta * S_xx
        dS_yy = conv_yy + g0 - beta * S_yy
        dS_zz = g0 - beta * S_zz
        dS_xy = conv_xy - beta * S_xy

        return jnp.array([dS_xx, dS_yy, dS_zz, dS_xy])

    return ode_rhs


def build_tnt_creep_ode_rhs(breakage_type="constant", use_fene=False, use_gs=False):
    """Build variant-specific creep ODE RHS (5-state: S + gamma).

    Stress is held constant; shear rate derived from stress constraint.

    Returns
    -------
    callable
        JIT-compiled ``f(t, state, sigma_applied, G, tau_b, eta_s, nu,
        m_break, kappa, L_max, xi) -> dstate`` with
        state = [S_xx, S_yy, S_zz, S_xy, gamma].
    """
    rate_ode = build_tnt_ode_rhs(breakage_type, use_fene, use_gs)

    @jax.jit
    def ode_rhs(
        t,
        state,
        sigma_applied,
        G,
        tau_b,
        eta_s,
        nu,
        m_break,
        kappa,
        L_max,
        xi,
    ):
        S_xx, S_yy, S_zz, S_xy, gamma = state

        # Elastic stress from conformation
        if use_fene:
            sigma_elastic = stress_fene_xy(S_xx, S_yy, S_zz, S_xy, G, L_max)
        else:
            sigma_elastic = stress_linear_xy(S_xy, G)

        # Shear rate from stress constraint
        eta_s_reg = jnp.maximum(eta_s, 1e-10 * G * tau_b)
        gamma_dot = (sigma_applied - sigma_elastic) / eta_s_reg

        # Conformation evolution
        conf_state = jnp.array([S_xx, S_yy, S_zz, S_xy])
        d_conf = rate_ode(
            t,
            conf_state,
            gamma_dot,
            G,
            tau_b,
            nu,
            m_break,
            kappa,
            L_max,
            xi,
        )

        return jnp.concatenate([d_conf, jnp.array([gamma_dot])])

    return ode_rhs


def build_tnt_laos_ode_rhs(breakage_type="constant", use_fene=False, use_gs=False):
    """Build variant-specific LAOS ODE RHS.

    Oscillatory shear: gamma_dot(t) = gamma_0 * omega * cos(omega * t).

    Returns
    -------
    callable
        JIT-compiled ``f(t, state, gamma_0, omega, G, tau_b, nu,
        m_break, kappa, L_max, xi) -> dstate``.
    """
    rate_ode = build_tnt_ode_rhs(breakage_type, use_fene, use_gs)

    @jax.jit
    def ode_rhs(
        t,
        state,
        gamma_0,
        omega,
        G,
        tau_b,
        nu,
        m_break,
        kappa,
        L_max,
        xi,
    ):
        gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
        return rate_ode(
            t,
            state,
            gamma_dot,
            G,
            tau_b,
            nu,
            m_break,
            kappa,
            L_max,
            xi,
        )

    return ode_rhs


def build_tnt_relaxation_ode_rhs(
    breakage_type="constant", use_fene=False, use_gs=False
):
    """Build variant-specific relaxation ODE RHS (gamma_dot = 0).

    Returns
    -------
    callable
        JIT-compiled ``f(t, state, G, tau_b, nu, m_break, kappa,
        L_max, xi) -> dstate``.
    """
    rate_ode = build_tnt_ode_rhs(breakage_type, use_fene, use_gs)

    @jax.jit
    def ode_rhs(t, state, G, tau_b, nu, m_break, kappa, L_max, xi):
        return rate_ode(
            t,
            state,
            0.0,
            G,
            tau_b,
            nu,
            m_break,
            kappa,
            L_max,
            xi,
        )

    return ode_rhs
