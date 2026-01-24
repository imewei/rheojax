"""JAX-accelerated physics kernels for Fluidity-Saramito EVP models.

This module implements the core physics equations for the Saramito
Elastoviscoplastic model combined with thixotropic fluidity evolution.
The model captures:

1. **Viscoelasticity**: Upper-convected Maxwell with elastic recoil
2. **Viscoplasticity**: Von Mises yield criterion with Herschel-Bulkley flow
3. **Thixotropy**: Time-dependent aging and shear rejuvenation via fluidity

Key functions:
- saramito_plasticity_alpha: Von Mises plasticity activation (α)
- upper_convected_2d: Upper-convected derivative in simple shear
- fluidity_evolution_saramito: df/dt for Saramito-fluidity coupling
- yield_stress_from_fluidity: Dynamic τ_y(f) for full coupling
- saramito_local_ode_rhs: ODE RHS for rate-controlled protocols
- saramito_local_creep_ode_rhs: ODE RHS for stress-controlled (creep)

References:
    - Saramito, P. (2007). JNNFM 145, 1-14.
    - Saramito, P. (2009). JNNFM 158, 154-161.
    - Coussot, P. et al. (2002). J. Rheol. 46(3), 573-589.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

# Safe import ensures float64
jax, jnp = safe_import_jax()


# ============================================================================
# Von Mises Plasticity Functions
# ============================================================================


@jax.jit
def von_mises_stress_2d(tau_xx: float, tau_yy: float, tau_xy: float) -> float:
    """Compute Von Mises equivalent stress for 2D stress tensor.

    For plane stress (σ_zz = 0):
    |τ| = √(τ_xx² + τ_yy² - τ_xx*τ_yy + 3*τ_xy²)

    For traceless deviatoric tensor (τ_xx + τ_yy = 0 in simple shear):
    |τ| = √(τ_xx² + τ_yy² + 2*τ_xy²)

    We use the second invariant formulation:
    |τ| = √(0.5 * τ:τ) where τ:τ = τ_ij * τ_ij

    Args:
        tau_xx: Normal stress in flow direction (Pa)
        tau_yy: Normal stress in gradient direction (Pa)
        tau_xy: Shear stress component (Pa)

    Returns:
        Von Mises equivalent stress (Pa)
    """
    # Second invariant: τ:τ = τ_xx² + τ_yy² + τ_zz² + 2(τ_xy² + τ_xz² + τ_yz²)
    # For 2D with τ_zz = -(τ_xx + τ_yy) (traceless), and τ_xz = τ_yz = 0:
    tau_zz = -(tau_xx + tau_yy)  # Traceless condition
    tau_sq = tau_xx**2 + tau_yy**2 + tau_zz**2 + 2.0 * tau_xy**2

    # Von Mises: |τ| = √(0.5 * τ:τ) for deviatoric tensor
    # Note: Some formulations use √(3/2 * τ':τ') for deviatoric part
    # We use the direct second invariant form: √((3/2) * τ_ij * τ_ij)
    # For simple shear where τ_xy dominates: |τ| ≈ √3 * |τ_xy|
    tau_mag = jnp.sqrt(0.5 * tau_sq + 1e-30)

    return tau_mag


@jax.jit
def saramito_plasticity_alpha(
    tau_xx: float,
    tau_yy: float,
    tau_xy: float,
    tau_y: float,
) -> float:
    """Compute Saramito plasticity function α.

    α = max(0, 1 - τ_y / |τ|)

    This determines the fraction of elastic strain that converts to plastic flow:
    - α = 0: Below yield, purely elastic (|τ| < τ_y)
    - α > 0: Above yield, elastic + plastic flow (|τ| > τ_y)
    - α → 1: Far above yield, mostly plastic (|τ| >> τ_y)

    The smooth implementation uses a softplus approximation for differentiability.

    Args:
        tau_xx: Normal stress in flow direction (Pa)
        tau_yy: Normal stress in gradient direction (Pa)
        tau_xy: Shear stress component (Pa)
        tau_y: Yield stress (Pa)

    Returns:
        Plasticity parameter α (dimensionless, 0 ≤ α ≤ 1)
    """
    tau_mag = von_mises_stress_2d(tau_xx, tau_yy, tau_xy)

    # Smooth max(0, x) for differentiability
    # α = max(0, 1 - τ_y/|τ|)
    # Use softplus: softplus(x/s)*s ≈ max(0, x) with smooth transition

    # Compute (1 - τ_y/|τ|) safely
    ratio = tau_y / (tau_mag + 1e-20)
    raw_alpha = 1.0 - ratio

    # Smooth max(0, raw_alpha) using softplus
    # Scale for smooth transition around yield
    scale = 0.01  # 1% smoothing region

    # Numerically stable softplus
    x = raw_alpha / scale
    alpha_smooth = scale * jnp.where(
        x > 20.0,
        x,  # For large x, softplus ≈ x
        jnp.log1p(jnp.exp(x)),
    )

    # Clip to [0, 1] for numerical safety
    alpha = jnp.clip(alpha_smooth, 0.0, 1.0)

    return alpha


# ============================================================================
# Upper-Convected Derivative
# ============================================================================


@jax.jit
def upper_convected_2d(
    tau_xx: float,
    tau_yy: float,
    tau_xy: float,
    gamma_dot: float,
) -> tuple[float, float, float]:
    """Compute upper-convected derivative components in simple shear.

    For simple shear flow with velocity gradient:
    L = [[0, γ̇, 0], [0, 0, 0], [0, 0, 0]]

    The upper-convected derivative is:
    ∇̂τ = dτ/dt - L·τ - τ·L^T

    Components in simple shear:
    ∇̂τ_xx = dτ_xx/dt - 2*γ̇*τ_xy
    ∇̂τ_yy = dτ_yy/dt
    ∇̂τ_xy = dτ_xy/dt - γ̇*τ_yy

    Note: The -γ̇*τ_yy term for τ_xy is what generates normal stress
    differences (Weissenberg effect).

    Args:
        tau_xx: Current τ_xx component (Pa)
        tau_yy: Current τ_yy component (Pa)
        tau_xy: Current τ_xy component (Pa)
        gamma_dot: Shear rate (1/s)

    Returns:
        Tuple of (L·τ + τ·L^T) terms to subtract from dτ/dt:
        (convective_xx, convective_yy, convective_xy)
    """
    # L·τ + τ·L^T contributions (to be subtracted from dτ/dt)
    # For upper-convected: dτ/dt = ∇̂τ + L·τ + τ·L^T
    # So: L·τ + τ·L^T = (terms from velocity gradient coupling)

    # In simple shear:
    # L = [[0, γ̇], [0, 0]] (2D version)
    # L·τ = [[γ̇*τ_xy, γ̇*τ_yy], [0, 0]]
    # τ·L^T = [[0, 0], [γ̇*τ_xy, γ̇*τ_yy]]
    # Sum = [[γ̇*τ_xy, γ̇*τ_yy], [γ̇*τ_xy, γ̇*τ_yy]] -- not quite right

    # Correct computation for 2D:
    # (L·τ)_ij = L_ik * τ_kj
    # (τ·L^T)_ij = τ_ik * L_jk

    # For L = [[0, γ̇], [0, 0]]:
    # (L·τ)_xx = L_xy * τ_yx = γ̇ * τ_xy
    # (L·τ)_xy = L_xy * τ_yy = γ̇ * τ_yy
    # (L·τ)_yy = 0
    # (τ·L^T)_xx = τ_xy * L_xy = τ_xy * γ̇
    # (τ·L^T)_xy = 0
    # (τ·L^T)_yy = 0

    # So L·τ + τ·L^T:
    # xx: 2*γ̇*τ_xy
    # yy: 0
    # xy: γ̇*τ_yy

    convective_xx = 2.0 * gamma_dot * tau_xy
    convective_yy = 0.0
    convective_xy = gamma_dot * tau_yy

    return convective_xx, convective_yy, convective_xy


# ============================================================================
# Fluidity Evolution for Saramito Model
# ============================================================================


@jax.jit
def fluidity_evolution_saramito(
    f: float,
    driving_rate: float,
    f_age: float,
    f_flow: float,
    t_a: float,
    b: float,
    n_rej: float,
) -> float:
    """Compute fluidity evolution rate for Saramito-fluidity model.

    df/dt = (f_age - f)/t_a + b * |driving|^n_rej * (f_flow - f)

    where:
    - f_age: Equilibrium fluidity under aging (structural build-up)
    - f_flow: High-shear fluidity limit (flow-induced breakdown)
    - t_a: Aging timescale
    - b: Rejuvenation amplitude
    - n_rej: Rejuvenation rate exponent
    - driving_rate: γ̇ for rate-controlled, σf for stress-controlled

    For Saramito coupling, the driving rate can be:
    - Rate-controlled: driving = |γ̇|
    - Stress-controlled: driving = |σ|*f (plastic strain rate)

    Args:
        f: Current fluidity (1/(Pa·s))
        driving_rate: Driving rate for rejuvenation (1/s or equivalent)
        f_age: Aging fluidity limit (1/(Pa·s))
        f_flow: Flow fluidity limit (1/(Pa·s))
        t_a: Aging timescale (s)
        b: Rejuvenation amplitude (dimensionless or s^n_rej)
        n_rej: Rejuvenation exponent (dimensionless)

    Returns:
        df/dt: Fluidity evolution rate (1/(Pa·s²))
    """
    # Ensure fluidity stays positive
    f_safe = jnp.maximum(f, 1e-20)

    # Aging: relaxation toward f_age
    aging_rate = (f_age - f_safe) / t_a

    # Rejuvenation: flow-induced increase toward f_flow
    driving_abs = jnp.abs(driving_rate)
    rejuv_rate = b * jnp.power(driving_abs + 1e-20, n_rej) * (f_flow - f_safe)

    df_dt = aging_rate + rejuv_rate

    return df_dt


# ============================================================================
# Dynamic Yield Stress Coupling
# ============================================================================


@jax.jit
def yield_stress_from_fluidity(
    f: float,
    tau_y0: float,
    tau_y_coupling: float,
    m_yield: float,
) -> float:
    """Compute fluidity-dependent yield stress for full coupling mode.

    τ_y(f) = τ_y0 + a_y / f^m

    This captures:
    - Aged state (f → f_age small): High yield stress (strong structure)
    - Rejuvenated state (f → f_flow large): Low yield stress (weak structure)

    For minimal coupling, use tau_y_coupling = 0 to get τ_y = τ_y0.

    Args:
        f: Current fluidity (1/(Pa·s))
        tau_y0: Base yield stress (Pa) - minimum value
        tau_y_coupling: Yield stress coupling coefficient (Pa / (1/(Pa·s))^m)
        m_yield: Yield stress fluidity exponent (dimensionless)

    Returns:
        Effective yield stress τ_y(f) (Pa)
    """
    f_safe = jnp.maximum(f, 1e-20)

    # Full coupling: τ_y increases as f decreases (aged structure)
    tau_y_dynamic = tau_y0 + tau_y_coupling / jnp.power(f_safe, m_yield)

    return tau_y_dynamic


# ============================================================================
# Herschel-Bulkley Viscosity
# ============================================================================


@jax.jit
def herschel_bulkley_viscosity(
    gamma_dot: float,
    tau_y: float,
    K: float,
    n: float,
    eta_s: float = 0.0,
) -> float:
    """Compute Herschel-Bulkley apparent viscosity.

    η_HB(γ̇) = τ_y/|γ̇| + K*|γ̇|^(n-1) + η_s

    For Saramito model, this is used when α > 0 to determine
    plastic flow contribution.

    Args:
        gamma_dot: Shear rate (1/s)
        tau_y: Yield stress (Pa)
        K: Flow consistency (Pa·s^n)
        n: Flow exponent (dimensionless)
        eta_s: Solvent viscosity (Pa·s)

    Returns:
        Apparent viscosity (Pa·s)
    """
    gamma_dot_abs = jnp.abs(gamma_dot) + 1e-20

    # Herschel-Bulkley viscosity
    eta_yield = tau_y / gamma_dot_abs
    eta_power = K * jnp.power(gamma_dot_abs, n - 1.0)

    eta_HB = eta_yield + eta_power + eta_s

    return eta_HB


# ============================================================================
# ODE Right-Hand Sides
# ============================================================================


def saramito_local_ode_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """ODE vector field for local Fluidity-Saramito model (rate-controlled).

    State vector: y = [τ_xx, τ_yy, τ_xy, f, γ]
    - τ_xx: Normal stress in flow direction (Pa)
    - τ_yy: Normal stress in gradient direction (Pa)
    - τ_xy: Shear stress (Pa)
    - f: Fluidity (1/(Pa·s))
    - γ: Accumulated strain (dimensionless)

    Constitutive equations (Saramito 2009):
    λ(dτ/dt - L·τ - τ·L^T) + α(τ)τ = 2η_p D

    where:
    - λ = 1/f (fluidity-dependent relaxation time)
    - α = max(0, 1 - τ_y/|τ|) (Von Mises plasticity)
    - η_p = G*λ = G/f (polymeric viscosity)
    - D = 0.5*(L + L^T) (rate of deformation, D_xy = γ̇/2)

    Rearranging for dτ/dt:
    dτ/dt = L·τ + τ·L^T + (2η_p D - α*τ) / λ
          = L·τ + τ·L^T + (2G D - α*f*τ)
          = L·τ + τ·L^T + G*γ̇*δ_xy - α*f*τ  (for shear)

    Component equations (simple shear, D_xy = γ̇/2):
    dτ_xx/dt = 2γ̇τ_xy - αfτ_xx
    dτ_yy/dt = -αfτ_yy
    dτ_xy/dt = γ̇τ_yy + Gγ̇ - αfτ_xy

    Note: The Gγ̇ term comes from 2η_p*D_xy/λ = 2*(G/f)*(γ̇/2)/λ = G*γ̇
    when λ = 1/f (so λ/f = 1).

    Args:
        t: Time (s)
        y: State vector [τ_xx, τ_yy, τ_xy, f, γ]
        args: Dictionary of parameters:
            - gamma_dot: Applied shear rate (1/s)
            - G: Elastic modulus (Pa)
            - tau_y0: Base yield stress (Pa)
            - tau_y_coupling: Yield stress coupling (Pa/(1/(Pa·s))^m)
            - m_yield: Yield stress exponent
            - f_age: Aging fluidity (1/(Pa·s))
            - f_flow: Flow fluidity (1/(Pa·s))
            - t_a: Aging timescale (s)
            - b: Rejuvenation amplitude
            - n_rej: Rejuvenation exponent
            - coupling_mode: "minimal" or "full"

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Unpack state
    tau_xx = y[0]
    tau_yy = y[1]
    tau_xy = y[2]
    f = y[3]
    gamma = y[4]  # noqa: F841 - tracked for output

    # Get parameters
    gamma_dot = args.get("gamma_dot", 0.0)
    G = args["G"]
    tau_y0 = args["tau_y0"]
    f_age = args["f_age"]
    f_flow = args["f_flow"]
    t_a = args["t_a"]
    b = args["b"]
    n_rej = args["n_rej"]

    # Coupling mode determines yield stress
    coupling_mode = args.get("coupling_mode", "minimal")
    if coupling_mode == "full":
        tau_y_coupling = args.get("tau_y_coupling", 0.0)
        m_yield = args.get("m_yield", 1.0)
        tau_y = yield_stress_from_fluidity(f, tau_y0, tau_y_coupling, m_yield)
    else:
        tau_y = tau_y0

    # Ensure fluidity is positive
    f_safe = jnp.maximum(f, 1e-20)

    # 1. Compute plasticity parameter α
    alpha = saramito_plasticity_alpha(tau_xx, tau_yy, tau_xy, tau_y)

    # 2. Upper-convected derivative terms
    conv_xx, conv_yy, conv_xy = upper_convected_2d(tau_xx, tau_yy, tau_xy, gamma_dot)

    # 3. Stress evolution (Saramito equations)
    # dτ/dt = L·τ + τ·L^T + G*γ̇*δ - α*f*τ
    # δ_xy = 1 for shear stress (actually the rate contribution is Gγ̇)

    d_tau_xx = conv_xx - alpha * f_safe * tau_xx
    d_tau_yy = conv_yy - alpha * f_safe * tau_yy
    d_tau_xy = conv_xy + G * gamma_dot - alpha * f_safe * tau_xy

    # 4. Fluidity evolution
    # For rate-controlled: driving = |γ̇|
    d_f = fluidity_evolution_saramito(
        f, jnp.abs(gamma_dot), f_age, f_flow, t_a, b, n_rej
    )

    # 5. Strain evolution
    d_gamma = gamma_dot

    return jnp.array([d_tau_xx, d_tau_yy, d_tau_xy, d_f, d_gamma])


def saramito_local_creep_ode_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """ODE vector field for local Fluidity-Saramito model (stress-controlled creep).

    State vector: y = [γ, f]
    - γ: Strain (dimensionless)
    - f: Fluidity (1/(Pa·s))

    For constant applied stress σ_applied:
    - If σ_applied < τ_y: Elastic response only (bounded strain)
    - If σ_applied > τ_y: Creep flow (unbounded strain growth)

    The creep bifurcation is a key signature of thixotropic yield-stress fluids.

    Equations:
    dγ/dt = γ̇ = α(σ) * f * σ  (plastic strain rate)
    df/dt = aging + rejuvenation (with driving = |γ̇|)

    Args:
        t: Time (s)
        y: State vector [γ, f]
        args: Dictionary of parameters:
            - sigma_applied: Applied stress (Pa)
            - G: Elastic modulus (Pa)
            - tau_y0: Base yield stress (Pa)
            - tau_y_coupling: Yield stress coupling (Pa/(1/(Pa·s))^m)
            - m_yield: Yield stress exponent
            - f_age: Aging fluidity (1/(Pa·s))
            - f_flow: Flow fluidity (1/(Pa·s))
            - t_a: Aging timescale (s)
            - b: Rejuvenation amplitude
            - n_rej: Rejuvenation exponent
            - coupling_mode: "minimal" or "full"

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Unpack state
    gamma = y[0]  # noqa: F841 - tracked for output
    f = y[1]

    # Get parameters
    sigma_applied = args["sigma_applied"]
    tau_y0 = args["tau_y0"]
    f_age = args["f_age"]
    f_flow = args["f_flow"]
    t_a = args["t_a"]
    b = args["b"]
    n_rej = args["n_rej"]

    # Coupling mode determines yield stress
    coupling_mode = args.get("coupling_mode", "minimal")
    if coupling_mode == "full":
        tau_y_coupling = args.get("tau_y_coupling", 0.0)
        m_yield = args.get("m_yield", 1.0)
        tau_y = yield_stress_from_fluidity(f, tau_y0, tau_y_coupling, m_yield)
    else:
        tau_y = tau_y0

    # Ensure fluidity is positive
    f_safe = jnp.maximum(f, 1e-20)

    # For creep, stress tensor is approximately:
    # τ_xy ≈ σ_applied (shear stress)
    # τ_xx, τ_yy ≈ 0 (normal stresses relax to zero at constant stress)
    # This simplification is valid for steady creep

    # Plasticity parameter (simplified for scalar stress)
    sigma_abs = jnp.abs(sigma_applied)

    # α = max(0, 1 - τ_y/|σ|)
    ratio = tau_y / (sigma_abs + 1e-20)
    raw_alpha = 1.0 - ratio
    scale = 0.01
    x = raw_alpha / scale
    alpha = scale * jnp.where(x > 20.0, x, jnp.log1p(jnp.exp(x)))
    alpha = jnp.clip(alpha, 0.0, 1.0)

    # 1. Strain evolution
    # γ̇ = α * f * σ (plastic strain rate)
    gamma_dot = alpha * f_safe * sigma_applied
    d_gamma = gamma_dot

    # 2. Fluidity evolution
    # For stress-controlled: driving = |γ̇| = |α * f * σ|
    driving_rate = jnp.abs(gamma_dot)
    d_f = fluidity_evolution_saramito(f, driving_rate, f_age, f_flow, t_a, b, n_rej)

    return jnp.array([d_gamma, d_f])


def saramito_local_relaxation_ode_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """ODE vector field for stress relaxation (zero rate after step strain).

    State vector: y = [τ_xx, τ_yy, τ_xy, f]

    After step strain (γ̇ = 0), stress relaxes via:
    dτ/dt = -α*f*τ (no convective terms, no elastic loading)

    If stress falls below yield: α → 0, stress freezes (solid-like)
    If stress stays above yield: continues to relax (viscoplastic)

    Args:
        t: Time (s)
        y: State vector [τ_xx, τ_yy, τ_xy, f]
        args: Dictionary of parameters

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Unpack state
    tau_xx = y[0]
    tau_yy = y[1]
    tau_xy = y[2]
    f = y[3]

    # Get parameters
    tau_y0 = args["tau_y0"]
    f_age = args["f_age"]
    f_flow = args["f_flow"]
    t_a = args["t_a"]
    b = args["b"]
    n_rej = args["n_rej"]

    # Coupling mode determines yield stress
    coupling_mode = args.get("coupling_mode", "minimal")
    if coupling_mode == "full":
        tau_y_coupling = args.get("tau_y_coupling", 0.0)
        m_yield = args.get("m_yield", 1.0)
        tau_y = yield_stress_from_fluidity(f, tau_y0, tau_y_coupling, m_yield)
    else:
        tau_y = tau_y0

    # Ensure fluidity is positive
    f_safe = jnp.maximum(f, 1e-20)

    # Plasticity parameter
    alpha = saramito_plasticity_alpha(tau_xx, tau_yy, tau_xy, tau_y)

    # Stress evolution with γ̇ = 0 (no convective terms, no elastic loading)
    d_tau_xx = -alpha * f_safe * tau_xx
    d_tau_yy = -alpha * f_safe * tau_yy
    d_tau_xy = -alpha * f_safe * tau_xy

    # Fluidity evolution with γ̇ = 0 (pure aging)
    # But there's still plastic flow if α > 0, so driving = |τ|*f*α ~ stress decay rate
    # Use zero driving for pure aging after cessation
    d_f = fluidity_evolution_saramito(f, 0.0, f_age, f_flow, t_a, b, n_rej)

    return jnp.array([d_tau_xx, d_tau_yy, d_tau_xy, d_f])


# ============================================================================
# Steady-State Flow Curve
# ============================================================================


@jax.jit
def _saramito_flow_curve_steady_minimal(
    gamma_dot: jnp.ndarray,
    tau_y0: float,
    K_HB: float,
    n_HB: float,
    f_age: float,
    f_flow: float,
    t_a: float,
    b: float,
    n_rej: float,
) -> jnp.ndarray:
    """Compute steady-state flow curve with minimal coupling (constant τ_y)."""
    gamma_dot_abs = jnp.abs(gamma_dot) + 1e-20

    # Steady-state fluidity
    flow_term = b * jnp.power(gamma_dot_abs, n_rej)
    numerator = f_age / t_a + flow_term * f_flow
    denominator = 1.0 / t_a + flow_term
    f_ss = numerator / (denominator + 1e-20)  # noqa: F841 - computed for consistency

    # Constant yield stress
    tau_y = tau_y0

    # Herschel-Bulkley flow curve
    sigma_HB = tau_y + K_HB * jnp.power(gamma_dot_abs, n_HB)
    sigma_ss = sigma_HB * jnp.sign(gamma_dot + 1e-20)

    return sigma_ss


@jax.jit
def _saramito_flow_curve_steady_full(
    gamma_dot: jnp.ndarray,
    tau_y0: float,
    K_HB: float,
    n_HB: float,
    f_age: float,
    f_flow: float,
    t_a: float,
    b: float,
    n_rej: float,
    tau_y_coupling: float,
    m_yield: float,
) -> jnp.ndarray:
    """Compute steady-state flow curve with full coupling (τ_y depends on f)."""
    gamma_dot_abs = jnp.abs(gamma_dot) + 1e-20

    # Steady-state fluidity
    flow_term = b * jnp.power(gamma_dot_abs, n_rej)
    numerator = f_age / t_a + flow_term * f_flow
    denominator = 1.0 / t_a + flow_term
    f_ss = numerator / (denominator + 1e-20)

    # Full coupling: yield stress depends on fluidity
    tau_y = tau_y0 + tau_y_coupling / jnp.power(f_ss + 1e-20, m_yield)

    # Herschel-Bulkley flow curve
    sigma_HB = tau_y + K_HB * jnp.power(gamma_dot_abs, n_HB)
    sigma_ss = sigma_HB * jnp.sign(gamma_dot + 1e-20)

    return sigma_ss


def saramito_flow_curve_steady(
    gamma_dot: jnp.ndarray,
    G: float,
    tau_y0: float,
    K_HB: float,
    n_HB: float,
    f_age: float,
    f_flow: float,
    t_a: float,
    b: float,
    n_rej: float,
    coupling_mode: str = "minimal",
    tau_y_coupling: float = 0.0,
    m_yield: float = 1.0,
) -> jnp.ndarray:
    """Compute steady-state flow curve for Saramito model.

    At steady state with constant γ̇:
    1. Fluidity reaches f_ss = (f_age/t_a + b|γ̇|^n_rej * f_flow) / (1/t_a + b|γ̇|^n_rej)
    2. Stress reaches σ_ss from force balance

    For fully plastic flow (α → 1 at high rates):
    σ_ss ≈ τ_y + K_HB * |γ̇|^n_HB  (Herschel-Bulkley)

    The transition from viscoelastic-solid to viscoplastic-flow
    occurs around γ̇ where σ ≈ τ_y.

    Args:
        gamma_dot: Shear rate array (1/s)
        G: Elastic modulus (Pa) - not used in steady state, kept for API consistency
        tau_y0: Base yield stress (Pa)
        K_HB: HB consistency (Pa·s^n)
        n_HB: HB exponent (dimensionless)
        f_age: Aging fluidity (1/(Pa·s))
        f_flow: Flow fluidity (1/(Pa·s))
        t_a: Aging timescale (s)
        b: Rejuvenation amplitude
        n_rej: Rejuvenation exponent
        coupling_mode: "minimal" or "full"
        tau_y_coupling: Yield stress coupling coefficient
        m_yield: Yield stress fluidity exponent

    Returns:
        Steady-state stress array (Pa)
    """
    # Dispatch to JIT-compiled function based on coupling mode
    if coupling_mode == "full":
        return _saramito_flow_curve_steady_full(
            gamma_dot,
            tau_y0,
            K_HB,
            n_HB,
            f_age,
            f_flow,
            t_a,
            b,
            n_rej,
            tau_y_coupling,
            m_yield,
        )
    else:
        return _saramito_flow_curve_steady_minimal(
            gamma_dot, tau_y0, K_HB, n_HB, f_age, f_flow, t_a, b, n_rej
        )


@jax.jit
def saramito_steady_state_full(
    gamma_dot: jnp.ndarray,
    G: float,
    tau_y0: float,
    K_HB: float,
    n_HB: float,
    f_age: float,
    f_flow: float,
    t_a: float,
    b: float,
    n_rej: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute steady-state stress tensor components.

    At steady state in simple shear:
    - τ_xy = σ_ss (shear stress from HB)
    - τ_xx = first normal stress (from viscoelasticity)
    - τ_yy ≈ -τ_xx/2 (approximately, for traceless)

    The first normal stress difference N₁ = τ_xx - τ_yy is related
    to the elastic nature of the material.

    For upper-convected Maxwell at steady shear:
    N₁ = 2 * λ * γ̇ * τ_xy  (where λ = 1/f)

    Args:
        gamma_dot: Shear rate array (1/s)
        G, tau_y0, ...: Model parameters

    Returns:
        (tau_xy, tau_xx, N1) arrays
    """
    gamma_dot_abs = jnp.abs(gamma_dot) + 1e-20

    # Shear stress (HB form)
    sigma_ss = saramito_flow_curve_steady(
        gamma_dot, G, tau_y0, K_HB, n_HB, f_age, f_flow, t_a, b, n_rej
    )
    tau_xy = sigma_ss

    # Steady-state fluidity
    flow_term = b * jnp.power(gamma_dot_abs, n_rej)
    numerator = f_age / t_a + flow_term * f_flow
    denominator = 1.0 / t_a + flow_term
    f_ss = numerator / (denominator + 1e-20)

    # Relaxation time λ = 1/f
    lam = 1.0 / f_ss

    # First normal stress from UCM steady shear:
    # At steady state: 2γ̇τ_xy = α*f*τ_xx
    # So τ_xx = 2γ̇τ_xy / (α*f) ≈ 2*λ*γ̇*τ_xy (for α ≈ 1)
    tau_xx = 2.0 * lam * gamma_dot * tau_xy

    # First normal stress difference
    # N₁ = τ_xx - τ_yy, and at steady state in simple shear τ_yy ≈ 0
    N1 = tau_xx  # τ_yy ≈ 0 at steady state

    return tau_xy, tau_xx, N1


# ============================================================================
# Nonlocal Diffusion Terms
# ============================================================================


@jax.jit
def laplacian_1d_neumann(f_field: jnp.ndarray, dy: float) -> jnp.ndarray:
    """Compute 1D Laplacian with Neumann (zero-flux) boundary conditions.

    ∂²f/∂y² ≈ (f[i+1] - 2f[i] + f[i-1]) / dy²

    Neumann BCs: ∂f/∂y = 0 at boundaries
    Implemented via ghost points: f[-1] = f[0], f[N] = f[N-1]

    Args:
        f_field: Field array, shape (N_y,)
        dy: Grid spacing (m)

    Returns:
        Laplacian array, shape (N_y,)
    """
    # Interior points
    lap = (jnp.roll(f_field, -1) - 2.0 * f_field + jnp.roll(f_field, 1)) / (dy**2)

    # Neumann boundary conditions: zero-flux
    lap_bc0 = 2.0 * (f_field[1] - f_field[0]) / (dy**2)
    lap_bcN = 2.0 * (f_field[-2] - f_field[-1]) / (dy**2)

    lap = lap.at[0].set(lap_bc0)
    lap = lap.at[-1].set(lap_bcN)

    return lap


def saramito_nonlocal_pde_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """PDE vector field for nonlocal Fluidity-Saramito model.

    State vector: y = [τ_xy_bulk, f[0], f[1], ..., f[N_y-1]]
    - τ_xy_bulk: Bulk (average) shear stress (Pa)
    - f[i]: Fluidity at grid point i (1/(Pa·s))

    Includes nonlocal fluidity diffusion:
    ∂f/∂t = (f_loc - f)/θ + D_f * ∇²f

    where D_f = ξ²/θ is the fluidity diffusivity and ξ is
    the cooperativity length.

    This enables shear banding: localized high-fluidity bands
    coexisting with low-fluidity (jammed) regions.

    Args:
        t: Time (s)
        y: State vector [τ_xy_bulk, f[0], ..., f[N_y-1]]
        args: Dictionary of parameters including:
            - xi: Cooperativity length (m)
            - dy: Grid spacing (m)
            - N_y: Number of grid points
            - gamma_dot: Applied shear rate (1/s)
            - Other Saramito parameters

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Get grid info
    N_y = args["N_y"]

    # Unpack state
    tau_xy_bulk = y[0]
    f_field = y[1 : N_y + 1]

    # Get parameters
    G = args["G"]
    tau_y0 = args["tau_y0"]
    f_age = args["f_age"]
    f_flow = args["f_flow"]
    t_a = args["t_a"]
    b = args["b"]
    n_rej = args["n_rej"]
    xi = args["xi"]
    dy = args["dy"]
    gamma_dot = args.get("gamma_dot", 0.0)

    # Coupling mode
    coupling_mode = args.get("coupling_mode", "minimal")
    if coupling_mode == "full":
        tau_y_coupling = args.get("tau_y_coupling", 0.0)
        m_yield = args.get("m_yield", 1.0)
    else:
        tau_y_coupling = 0.0
        m_yield = 1.0

    # Ensure fluidity is positive
    f_field_safe = jnp.maximum(f_field, 1e-20)

    # Average fluidity for bulk response
    f_avg = jnp.mean(f_field_safe)

    # Bulk stress evolution (average over gap)
    # Assuming homogeneous stress in Couette geometry
    d_tau_bulk = G * (gamma_dot - tau_xy_bulk * f_avg)

    # Local yield stress (may depend on local fluidity)
    if coupling_mode == "full":
        tau_y_local = tau_y0 + tau_y_coupling / jnp.power(f_field_safe, m_yield)
    else:
        tau_y_local = tau_y0 * jnp.ones_like(f_field_safe)

    # Local plasticity (assuming uniform stress in gap)
    alpha_local = jnp.clip(1.0 - tau_y_local / (jnp.abs(tau_xy_bulk) + 1e-20), 0.0, 1.0)

    # Local shear rate (from force balance: σ = const across gap)
    # γ̇_local = α * f * σ
    gamma_dot_local = alpha_local * f_field_safe * tau_xy_bulk

    # Fluidity field evolution
    # 1. Local driving
    d_f_local = jax.vmap(
        lambda f_i, gd_i: fluidity_evolution_saramito(
            f_i, jnp.abs(gd_i), f_age, f_flow, t_a, b, n_rej
        )
    )(f_field_safe, gamma_dot_local)

    # 2. Nonlocal diffusion: D_f * ∇²f where D_f = ξ²/t_a
    D_f = xi**2 / t_a
    lap_f = laplacian_1d_neumann(f_field_safe, dy)
    d_f_diffusion = D_f * lap_f

    # Total fluidity evolution
    d_f_field = d_f_local + d_f_diffusion

    # Assemble output
    d_y = jnp.concatenate([jnp.array([d_tau_bulk]), d_f_field])

    return d_y


# ============================================================================
# Shear Banding Detection
# ============================================================================


@jax.jit
def shear_banding_cv(f_field: jnp.ndarray) -> float:
    """Compute coefficient of variation as shear banding metric.

    CV = std(f) / mean(f)

    CV > 0.3 typically indicates significant shear banding.

    Args:
        f_field: Fluidity field across gap, shape (N_y,)

    Returns:
        Coefficient of variation (dimensionless)
    """
    f_mean = jnp.mean(f_field)
    f_std = jnp.std(f_field)
    cv = f_std / (f_mean + 1e-20)
    return cv


@jax.jit
def banding_ratio(f_field: jnp.ndarray) -> float:
    """Compute ratio of max to min fluidity as banding metric.

    ratio = f_max / f_min

    ratio > 10 indicates strong localization.

    Args:
        f_field: Fluidity field across gap, shape (N_y,)

    Returns:
        Banding ratio (dimensionless)
    """
    f_max = jnp.max(f_field)
    f_min = jnp.min(f_field)
    ratio = f_max / (f_min + 1e-20)
    return ratio


@jax.jit
def detect_shear_bands(
    f_field: jnp.ndarray,
    threshold: float = 0.3,
) -> tuple[bool, float, float]:
    """Detect shear banding from fluidity profile.

    Args:
        f_field: Fluidity field across gap, shape (N_y,)
        threshold: CV threshold for banding (default 0.3)

    Returns:
        (is_banded, cv, ratio): Detection result and metrics
    """
    cv = shear_banding_cv(f_field)
    ratio = banding_ratio(f_field)

    is_banded = cv > threshold

    return is_banded, cv, ratio
