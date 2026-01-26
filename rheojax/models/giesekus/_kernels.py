"""JAX-accelerated physics kernels for the Giesekus model.

This module provides JIT-compiled functions for:
1. Stress tensor operations (quadratic term τ·τ)
2. Upper-convected derivative in simple shear
3. Analytical solutions for steady and oscillatory shear
4. ODE right-hand sides for transient protocols

All functions are designed for float64 precision and JAX compatibility.

Key Equation
------------
The Giesekus constitutive equation is::

    τ + λ∇̂τ + (αλ/η_p)τ·τ = 2η_p D

where:
- τ is the polymer extra stress tensor
- λ is the relaxation time
- α is the mobility factor (0 ≤ α ≤ 0.5)
- η_p is the polymer viscosity
- D is the rate-of-deformation tensor
- ∇̂ denotes the upper-convected derivative

State Vector Convention
-----------------------
For 3D incompressible flow, we use a 4-component state::

    state = [τ_xx, τ_yy, τ_xy, τ_zz]

where τ_zz is needed for the trace in tr(τ·τ) calculations.
For simple shear: τ_zz = 0 at steady state.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


# =============================================================================
# Tensor Operations
# =============================================================================


@jax.jit
def stress_tensor_product_2d(
    tau_xx: float,
    tau_yy: float,
    tau_xy: float,
    tau_zz: float,
) -> tuple[float, float, float, float]:
    """Compute τ·τ components for the Giesekus quadratic term.

    For a symmetric stress tensor in simple shear::

        τ = [[τ_xx, τ_xy, 0   ],
             [τ_xy, τ_yy, 0   ],
             [0,    0,    τ_zz]]

    The product τ·τ gives::

        (τ·τ)_xx = τ_xx² + τ_xy²
        (τ·τ)_yy = τ_xy² + τ_yy²
        (τ·τ)_xy = τ_xy(τ_xx + τ_yy)
        (τ·τ)_zz = τ_zz²

    Parameters
    ----------
    tau_xx : float
        Normal stress component τ_xx (Pa)
    tau_yy : float
        Normal stress component τ_yy (Pa)
    tau_xy : float
        Shear stress component τ_xy (Pa)
    tau_zz : float
        Normal stress component τ_zz (Pa)

    Returns
    -------
    tuple[float, float, float, float]
        Components (τ·τ)_xx, (τ·τ)_yy, (τ·τ)_xy, (τ·τ)_zz
    """
    tt_xx = tau_xx * tau_xx + tau_xy * tau_xy
    tt_yy = tau_xy * tau_xy + tau_yy * tau_yy
    tt_xy = tau_xy * (tau_xx + tau_yy)
    tt_zz = tau_zz * tau_zz

    return tt_xx, tt_yy, tt_xy, tt_zz


@jax.jit
def upper_convected_derivative_2d(
    tau_xx: float,
    tau_yy: float,
    tau_xy: float,
    gamma_dot: float,
) -> tuple[float, float, float]:
    """Compute upper-convected derivative terms in simple shear.

    For simple shear flow with velocity gradient::

        L = [[0, γ̇, 0],
             [0, 0, 0],
             [0, 0, 0]]

    The upper-convected derivative is::

        ∇̂τ = dτ/dt - L·τ - τ·L^T

    The convective terms (L·τ + τ·L^T) are::

        (L·τ + τ·L^T)_xx = 2γ̇τ_xy
        (L·τ + τ·L^T)_yy = 0
        (L·τ + τ·L^T)_xy = γ̇τ_yy

    Note: τ_zz has no convective contribution in simple shear.

    Parameters
    ----------
    tau_xx : float
        Normal stress component τ_xx (Pa)
    tau_yy : float
        Normal stress component τ_yy (Pa)
    tau_xy : float
        Shear stress component τ_xy (Pa)
    gamma_dot : float
        Shear rate (1/s)

    Returns
    -------
    tuple[float, float, float]
        Convective terms: (conv_xx, conv_yy, conv_xy)
        These are subtracted from dτ/dt to get ∇̂τ components.
    """
    conv_xx = 2.0 * gamma_dot * tau_xy
    conv_yy = 0.0
    conv_xy = gamma_dot * tau_yy

    return conv_xx, conv_yy, conv_xy


# =============================================================================
# Steady-State Analytical Solutions
# =============================================================================


@jax.jit
def _solve_giesekus_f_quartic(
    wi: float,
    alpha: float,
) -> float:
    """Solve for the viscosity reduction factor f(Wi, α).

    The steady-state polymer viscosity is η_p(γ̇) = η_p·f where f ∈ (0, 1]
    satisfies the Giesekus steady-state momentum balance.

    Uses Newton iteration on the reduced shear stress balance with
    τ_xx and τ_yy eliminated analytically from the normal stress balances.
    At steady state in simple shear, the dimensionless component equations
    (s_ij = τ_ij·λ/η_p, Wi = λγ̇) are::

        (1) s_xx - 2Wi·s_xy + α(s_xx² + s_xy²) = 0
        (2) s_yy + α(s_xy² + s_yy²) = 0
        (3) s_xy - Wi·s_yy + α·s_xy·(s_xx + s_yy) = Wi

    With s_xy = Wi·f, equations (1)-(2) give s_xx(f) and s_yy(f) in
    closed form via quadratic formula. The residual is (3) / Wi.

    Parameters
    ----------
    wi : float
        Weissenberg number Wi = λγ̇
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)

    Returns
    -------
    float
        Viscosity reduction factor f (dimensionless, 0 < f ≤ 1)
    """
    # Limiting cases where f = 1 exactly
    is_trivial = (alpha < 1e-12) | (wi < 1e-12)

    wi2 = wi * wi
    # Guard against division by zero in the α > 0 branch
    alpha_safe = jnp.maximum(alpha, 1e-30)

    # Physical upper bound: disc_yy ≥ 0 requires |2α·Wi·f| ≤ 1
    f_max = jnp.minimum(
        0.999 / (2.0 * alpha_safe * jnp.maximum(wi, 1e-10)),
        1.0 - 1e-10,
    )

    def residual(f_val):
        s_xy = wi * f_val
        s_xy2 = s_xy * s_xy

        # s_yy from Eq(2): quadratic α·s_yy² + s_yy + α·s_xy² = 0
        disc_yy = jnp.maximum(1.0 - 4.0 * alpha_safe * alpha_safe * s_xy2, 0.0)
        s_yy = (-1.0 + jnp.sqrt(disc_yy + 1e-30)) / (2.0 * alpha_safe)

        # s_xx from Eq(1): quadratic α·s_xx² + s_xx - (2Wi·s_xy - α·s_xy²) = 0
        q_xx = 2.0 * wi * s_xy - alpha_safe * s_xy2
        disc_xx = jnp.maximum(1.0 + 4.0 * alpha_safe * q_xx, 0.0)
        s_xx = (-1.0 + jnp.sqrt(disc_xx + 1e-30)) / (2.0 * alpha_safe)

        # Residual from Eq(3) divided by Wi
        return f_val - s_yy + alpha_safe * f_val * (s_xx + s_yy) - 1.0

    def d_residual(f_val):
        s_xy = wi * f_val
        s_xy2 = s_xy * s_xy

        disc_yy = jnp.maximum(1.0 - 4.0 * alpha_safe * alpha_safe * s_xy2, 0.0)
        sqrt_disc_yy = jnp.sqrt(disc_yy + 1e-30)
        s_yy = (-1.0 + sqrt_disc_yy) / (2.0 * alpha_safe)

        q_xx = 2.0 * wi * s_xy - alpha_safe * s_xy2
        disc_xx = jnp.maximum(1.0 + 4.0 * alpha_safe * q_xx, 0.0)
        sqrt_disc_xx = jnp.sqrt(disc_xx + 1e-30)
        s_xx = (-1.0 + sqrt_disc_xx) / (2.0 * alpha_safe)

        # ds_yy/df: chain rule through s_xy = wi·f
        # ds_yy/ds_xy = -4α²·s_xy / (2α·√disc_yy) = -2α·s_xy / √disc_yy
        ds_yy_df = -2.0 * alpha_safe * s_xy * wi / (sqrt_disc_yy + 1e-30)

        # ds_xx/df: d(disc_xx)/ds_xy = 4α(2Wi - 2α·s_xy)
        # ds_xx/ds_xy = 4α(2Wi - 2α·s_xy) / (4α·√disc_xx) = (2Wi - 2α·s_xy)/√disc_xx
        ds_xx_df = (
            2.0 * (wi - alpha_safe * s_xy) * wi / (sqrt_disc_xx + 1e-30)
        )

        # dR/df = 1 - ds_yy/df + α(s_xx+s_yy) + α·f·(ds_xx/df + ds_yy/df)
        return (
            1.0
            - ds_yy_df
            + alpha_safe * (s_xx + s_yy)
            + alpha_safe * f_val * (ds_xx_df + ds_yy_df)
        )

    # Initial guess interpolating between asymptotic limits
    # Low Wi: f ≈ 1 - α·Wi² (first-order perturbation)
    f_low = jnp.maximum(1.0 - alpha_safe * wi2, 0.3)
    # High Wi: f approaches f_max = 1/(2α·Wi) from below
    f_high = 0.9 * f_max
    f_init = jnp.where(wi < 1.0, jnp.minimum(f_low, f_max), f_high)

    # Newton iteration with damping
    def newton_step(f_val, _):
        res = residual(f_val)
        dres = d_residual(f_val)
        delta = res / jnp.where(jnp.abs(dres) > 1e-20, dres, 1e-20)
        f_new = f_val - 0.7 * delta
        return jnp.clip(f_new, 1e-10, f_max), None

    f_final, _ = jax.lax.scan(newton_step, f_init, None, length=20)

    return jnp.where(is_trivial, 1.0, f_final)


@jax.jit
def giesekus_steady_shear_viscosity(
    gamma_dot: float,
    eta_p: float,
    lambda_1: float,
    alpha: float,
    eta_s: float,
) -> float:
    """Compute steady shear viscosity η(γ̇) for Giesekus model.

    The viscosity is::

        η(γ̇) = η_s + η_p·f(Wi)

    where f is the auxiliary function solving the Giesekus implicit equation.

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    float
        Steady shear viscosity η (Pa·s)
    """
    wi = lambda_1 * jnp.abs(gamma_dot)
    f = _solve_giesekus_f_quartic(wi, alpha)
    return eta_s + eta_p * f


@jax.jit
def giesekus_steady_shear_stress(
    gamma_dot: float,
    eta_p: float,
    lambda_1: float,
    alpha: float,
    eta_s: float,
) -> float:
    """Compute steady shear stress σ(γ̇) for Giesekus model.

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    float
        Steady shear stress σ = η·γ̇ (Pa)
    """
    eta = giesekus_steady_shear_viscosity(gamma_dot, eta_p, lambda_1, alpha, eta_s)
    return eta * gamma_dot


# Vectorized version for arrays
giesekus_steady_shear_stress_vec = jax.jit(
    jax.vmap(
        giesekus_steady_shear_stress,
        in_axes=(0, None, None, None, None),
    )
)


@jax.jit
def giesekus_steady_normal_stresses(
    gamma_dot: float,
    eta_p: float,
    lambda_1: float,
    alpha: float,
) -> tuple[float, float]:
    """Compute first and second normal stress differences.

    For the Giesekus model::

        N₁ = τ_xx - τ_yy = Ψ₁·γ̇²
        N₂ = τ_yy - τ_zz = Ψ₂·γ̇² = -(α/2)·N₁

    The first normal stress coefficient is::

        Ψ₁ = 2η_p·λ·f² / [1 + (1-2α)·Wi²·f²]

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)

    Returns
    -------
    tuple[float, float]
        (N₁, N₂) in Pa
    """
    wi = lambda_1 * jnp.abs(gamma_dot)
    f = _solve_giesekus_f_quartic(wi, alpha)

    # First normal stress coefficient
    f2 = f * f
    wi2f2 = wi * wi * f2
    psi1 = 2.0 * eta_p * lambda_1 * f2 / (1.0 + (1.0 - 2.0 * alpha) * wi2f2)

    # Normal stress differences
    gd2 = gamma_dot * gamma_dot
    N1 = psi1 * gd2
    N2 = -alpha * N1 / 2.0  # Giesekus prediction: N2/N1 = -α/2

    return N1, N2


# Vectorized version
giesekus_steady_normal_stresses_vec = jax.jit(
    jax.vmap(
        giesekus_steady_normal_stresses,
        in_axes=(0, None, None, None),
    )
)


@jax.jit
def giesekus_steady_stress_components(
    gamma_dot: float,
    eta_p: float,
    lambda_1: float,
    alpha: float,
    eta_s: float,
) -> tuple[float, float, float, float]:
    """Compute all steady-state stress tensor components.

    Returns the full stress state [τ_xx, τ_yy, τ_xy, τ_zz] at steady state.

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    tuple[float, float, float, float]
        (τ_xx, τ_yy, τ_xy, τ_zz) in Pa
    """
    # Get shear stress
    tau_xy = giesekus_steady_shear_stress(gamma_dot, eta_p, lambda_1, alpha, eta_s)

    # Get normal stresses (polymer contribution only)
    N1, N2 = giesekus_steady_normal_stresses(gamma_dot, eta_p, lambda_1, alpha)

    # From definitions: N1 = τ_xx - τ_yy, N2 = τ_yy - τ_zz
    # At steady state in simple shear, τ_zz = 0 (no flow in z)
    tau_zz = 0.0
    tau_yy = N2 + tau_zz  # τ_yy = N2 (since τ_zz = 0)
    tau_xx = N1 + tau_yy  # τ_xx = N1 + τ_yy

    return tau_xx, tau_yy, tau_xy, tau_zz


# Vectorized version
giesekus_steady_stress_components_vec = jax.jit(
    jax.vmap(
        giesekus_steady_stress_components,
        in_axes=(0, None, None, None, None),
    )
)


# =============================================================================
# SAOS (Small-Amplitude Oscillatory Shear) - Linear Regime
# =============================================================================


@jax.jit
def giesekus_saos_moduli(
    omega: float,
    eta_p: float,
    lambda_1: float,
    eta_s: float,
) -> tuple[float, float]:
    """Compute storage and loss moduli G'(ω) and G''(ω).

    In the linear viscoelastic regime (small strain amplitude), the Giesekus
    model reduces to Maxwell behavior (α-independent)::

        G'(ω) = G·(ωλ)² / (1 + (ωλ)²)
        G''(ω) = G·(ωλ) / (1 + (ωλ)²) + η_s·ω

    where G = η_p/λ is the elastic modulus.

    Note: α does not appear because the quadratic τ·τ term is O(γ₀²)
    and vanishes in the linear limit.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    tuple[float, float]
        (G', G'') in Pa
    """
    G = eta_p / lambda_1  # Elastic modulus
    omega_lambda = omega * lambda_1
    omega_lambda_sq = omega_lambda * omega_lambda
    denom = 1.0 + omega_lambda_sq

    G_prime = G * omega_lambda_sq / denom
    G_double_prime = G * omega_lambda / denom + eta_s * omega

    return G_prime, G_double_prime


# Vectorized version for frequency sweep
giesekus_saos_moduli_vec = jax.jit(
    jax.vmap(
        giesekus_saos_moduli,
        in_axes=(0, None, None, None),
    )
)


@jax.jit
def giesekus_complex_viscosity(
    omega: float,
    eta_p: float,
    lambda_1: float,
    eta_s: float,
) -> tuple[float, float]:
    """Compute complex viscosity magnitude and phase.

    The complex viscosity is::

        η*(ω) = η' - iη'' = (G'' - iG')/ω

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    tuple[float, float]
        (|η*|, δ) where δ is the phase angle in radians
    """
    G_prime, G_double_prime = giesekus_saos_moduli(omega, eta_p, lambda_1, eta_s)

    # |η*| = |G*|/ω = sqrt(G'² + G''²)/ω
    G_star_mag = jnp.sqrt(G_prime * G_prime + G_double_prime * G_double_prime)
    eta_star_mag = G_star_mag / omega

    # Phase angle: tan(δ) = G''/G'
    delta = jnp.arctan2(G_double_prime, G_prime)

    return eta_star_mag, delta


# =============================================================================
# ODE Right-Hand Sides for Transient Protocols
# =============================================================================


@jax.jit
def giesekus_ode_rhs(
    t: float,
    state: jnp.ndarray,
    gamma_dot: float,
    eta_p: float,
    lambda_1: float,
    alpha: float,
) -> jnp.ndarray:
    """ODE right-hand side for rate-controlled Giesekus dynamics.

    State vector: [τ_xx, τ_yy, τ_xy, τ_zz]

    The evolution equations come from rearranging::

        τ + λ∇̂τ + (αλ/η_p)τ·τ = 2η_p D

    to::

        dτ/dt = (2η_p D - τ - (αλ/η_p)τ·τ)/λ + (L·τ + τ·L^T)

    For simple shear with D_xy = γ̇/2, the rate-of-deformation tensor gives
    a source term only in the xy component.

    Parameters
    ----------
    t : float
        Time (s), unused but required by ODE solvers
    state : jnp.ndarray
        State vector [τ_xx, τ_yy, τ_xy, τ_zz] (Pa)
    gamma_dot : float
        Applied shear rate (1/s)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)

    Returns
    -------
    jnp.ndarray
        Time derivatives [dτ_xx/dt, dτ_yy/dt, dτ_xy/dt, dτ_zz/dt]
    """
    tau_xx, tau_yy, tau_xy, tau_zz = state

    # Upper-convected derivative terms (L·τ + τ·L^T)
    conv_xx, conv_yy, conv_xy = upper_convected_derivative_2d(
        tau_xx, tau_yy, tau_xy, gamma_dot
    )

    # Quadratic stress term τ·τ
    tt_xx, tt_yy, tt_xy, tt_zz = stress_tensor_product_2d(
        tau_xx, tau_yy, tau_xy, tau_zz
    )

    # Nonlinear coefficient
    alpha_lambda_over_eta = alpha * lambda_1 / eta_p

    # Rate of deformation contribution (only xy component for simple shear)
    # D_xy = γ̇/2, so 2η_p·D_xy = η_p·γ̇
    source_xy = eta_p * gamma_dot

    # Time derivatives: dτ/dt = (source - τ - α_coeff·τ·τ)/λ + convective
    inv_lambda = 1.0 / lambda_1

    d_tau_xx = (-tau_xx - alpha_lambda_over_eta * tt_xx) * inv_lambda + conv_xx
    d_tau_yy = (-tau_yy - alpha_lambda_over_eta * tt_yy) * inv_lambda + conv_yy
    d_tau_xy = (
        source_xy - tau_xy - alpha_lambda_over_eta * tt_xy
    ) * inv_lambda + conv_xy
    d_tau_zz = (
        -tau_zz - alpha_lambda_over_eta * tt_zz
    ) * inv_lambda  # No convective term

    return jnp.array([d_tau_xx, d_tau_yy, d_tau_xy, d_tau_zz])


@jax.jit
def giesekus_ode_rhs_laos(
    t: float,
    state: jnp.ndarray,
    gamma_0: float,
    omega: float,
    eta_p: float,
    lambda_1: float,
    alpha: float,
) -> jnp.ndarray:
    """ODE right-hand side for LAOS (Large-Amplitude Oscillatory Shear).

    The applied strain is γ(t) = γ₀·sin(ωt), so::

        γ̇(t) = γ₀·ω·cos(ωt)

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [τ_xx, τ_yy, τ_xy, τ_zz] (Pa)
    gamma_0 : float
        Strain amplitude (dimensionless)
    omega : float
        Angular frequency (rad/s)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)

    Returns
    -------
    jnp.ndarray
        Time derivatives [dτ_xx/dt, dτ_yy/dt, dτ_xy/dt, dτ_zz/dt]
    """
    gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
    return giesekus_ode_rhs(t, state, gamma_dot, eta_p, lambda_1, alpha)


@jax.jit
def giesekus_creep_ode_rhs(
    t: float,
    state: jnp.ndarray,
    sigma_applied: float,
    eta_p: float,
    lambda_1: float,
    alpha: float,
    eta_s: float,
) -> jnp.ndarray:
    """ODE right-hand side for stress-controlled (creep) deformation.

    State vector: [τ_xx, τ_yy, τ_xy, τ_zz, γ]
    where γ is the accumulated strain.

    In creep, the total stress σ = τ_xy + η_s·γ̇ is held constant,
    so γ̇ must be computed implicitly::

        γ̇ = (σ - τ_xy) / η_s

    For η_s = 0, creep is not well-defined (instantaneous jump to
    steady state), so a small regularization is applied.

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [τ_xx, τ_yy, τ_xy, τ_zz, γ] (Pa, Pa, Pa, Pa, -)
    sigma_applied : float
        Applied constant stress (Pa)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    jnp.ndarray
        Time derivatives [dτ_xx/dt, dτ_yy/dt, dτ_xy/dt, dτ_zz/dt, dγ/dt]
    """
    tau_xx, tau_yy, tau_xy, tau_zz, gamma = state

    # Compute shear rate from stress constraint
    # σ = τ_xy + η_s·γ̇  =>  γ̇ = (σ - τ_xy) / η_s
    # Regularize for small η_s
    eta_s_reg = jnp.maximum(eta_s, 1e-10 * eta_p)
    gamma_dot = (sigma_applied - tau_xy) / eta_s_reg

    # Stress evolution (same as rate-controlled)
    stress_state = jnp.array([tau_xx, tau_yy, tau_xy, tau_zz])
    d_stress = giesekus_ode_rhs(t, stress_state, gamma_dot, eta_p, lambda_1, alpha)

    # Strain evolution
    d_gamma = gamma_dot

    return jnp.concatenate([d_stress, jnp.array([d_gamma])])


@jax.jit
def giesekus_relaxation_ode_rhs(
    t: float,
    state: jnp.ndarray,
    eta_p: float,
    lambda_1: float,
    alpha: float,
) -> jnp.ndarray:
    """ODE right-hand side for stress relaxation (γ̇ = 0).

    After cessation of flow, the stress relaxes with γ̇ = 0.
    The Giesekus model predicts faster-than-exponential decay
    due to the quadratic τ·τ term.

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [τ_xx, τ_yy, τ_xy, τ_zz] (Pa)
    eta_p : float
        Polymer viscosity (Pa·s)
    lambda_1 : float
        Relaxation time (s)
    alpha : float
        Mobility factor (0 ≤ α ≤ 0.5)

    Returns
    -------
    jnp.ndarray
        Time derivatives [dτ_xx/dt, dτ_yy/dt, dτ_xy/dt, dτ_zz/dt]
    """
    return giesekus_ode_rhs(t, state, 0.0, eta_p, lambda_1, alpha)


# =============================================================================
# Multi-Mode Extensions
# =============================================================================


@jax.jit
def giesekus_multimode_saos_moduli(
    omega: float,
    eta_p_modes: jnp.ndarray,
    lambda_modes: jnp.ndarray,
    eta_s: float,
) -> tuple[float, float]:
    """Compute multi-mode SAOS moduli by superposition.

    For N modes with individual (η_p,i, λ_i)::

        G'(ω) = Σ G_i·(ωλ_i)² / (1 + (ωλ_i)²)
        G''(ω) = Σ G_i·(ωλ_i) / (1 + (ωλ_i)²) + η_s·ω

    where G_i = η_p,i / λ_i.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    eta_p_modes : jnp.ndarray
        Polymer viscosities for each mode (Pa·s), shape (N,)
    lambda_modes : jnp.ndarray
        Relaxation times for each mode (s), shape (N,)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    tuple[float, float]
        (G', G'') in Pa
    """
    # Compute per-mode contributions
    G_modes = eta_p_modes / lambda_modes
    omega_lambda = omega * lambda_modes
    omega_lambda_sq = omega_lambda * omega_lambda
    denom = 1.0 + omega_lambda_sq

    G_prime_contributions = G_modes * omega_lambda_sq / denom
    G_double_prime_contributions = G_modes * omega_lambda / denom

    # Sum over modes
    G_prime = jnp.sum(G_prime_contributions)
    G_double_prime = jnp.sum(G_double_prime_contributions) + eta_s * omega

    return G_prime, G_double_prime


@jax.jit
def giesekus_multimode_ode_rhs(
    t: float,
    state: jnp.ndarray,
    gamma_dot: float,
    eta_p_modes: jnp.ndarray,
    lambda_modes: jnp.ndarray,
    alpha_modes: jnp.ndarray,
) -> jnp.ndarray:
    """ODE right-hand side for multi-mode Giesekus dynamics.

    State vector: [τ_xx^1, τ_yy^1, τ_xy^1, τ_zz^1, ..., τ_xx^N, ..., τ_zz^N]
    Total length: 4*N

    Each mode evolves independently with its own (η_p, λ, α).

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        Concatenated state vector, shape (4*N,)
    gamma_dot : float
        Applied shear rate (1/s)
    eta_p_modes : jnp.ndarray
        Polymer viscosities for each mode (Pa·s), shape (N,)
    lambda_modes : jnp.ndarray
        Relaxation times for each mode (s), shape (N,)
    alpha_modes : jnp.ndarray
        Mobility factors for each mode, shape (N,)

    Returns
    -------
    jnp.ndarray
        Time derivatives, shape (4*N,)
    """
    # Reshape state: (N, 4) — n_modes derived from array shape (static at trace time)
    n_modes = eta_p_modes.shape[0]
    state_reshaped = state.reshape((n_modes, 4))

    def single_mode_rhs(mode_state, mode_params):
        eta_p, lambda_1, alpha = mode_params
        return giesekus_ode_rhs(t, mode_state, gamma_dot, eta_p, lambda_1, alpha)

    mode_params = jnp.stack([eta_p_modes, lambda_modes, alpha_modes], axis=1)

    # vmap over modes
    d_states = jax.vmap(single_mode_rhs)(state_reshaped, mode_params)

    return d_states.flatten()
