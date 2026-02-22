"""JAX-accelerated physics kernels for Fluidity models.

This module implements the core physics equations for the Fluidity model
for yield-stress fluids using JAX. Supports both Local (0D homogeneous)
and Non-Local (1D Coussot-Ovarlez with spatial diffusion) variants.

Key functions:
- f_loc_herschel_bulkley: Local fluidity from HB flow curve
- laplacian_1d_neumann: Finite difference Laplacian with Neumann BCs
- fluidity_local_ode_rhs: ODE RHS for Local model
- fluidity_nonlocal_pde_rhs: PDE RHS for Non-Local model

References:
    - Coussot et al., Phys. Rev. Lett. 88, 175501 (2002)
    - Goyon et al., Nature 454, 84-87 (2008)
    - Ovarlez et al., J. Non-Newtonian Fluid Mech. 177-178, 19-28 (2012)
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

# Safe import ensures float64
jax, jnp = safe_import_jax()


# ============================================================================
# Local Fluidity Kernel
# ============================================================================


@jax.jit
def f_loc_herschel_bulkley(
    sigma: float,
    tau_y: float,
    K: float,
    n: float,
    eps: float = 1e-12,
) -> float:
    """Compute local equilibrium fluidity from Herschel-Bulkley flow curve.

    f_loc = max(0, |σ| - τ_y)^n / K

    This is the inverse of the HB flow curve σ = τ_y + K*γ̇^n.

    Uses smooth approximation to avoid discontinuity at yield stress:
    f_loc = softplus((|σ| - τ_y) / σ_scale)^n / K

    Args:
        sigma: Deviatoric stress (Pa)
        tau_y: Yield stress (Pa)
        K: Flow consistency (Pa·s^n)
        n: Flow exponent (dimensionless)
        eps: Small value for numerical stability

    Returns:
        Local fluidity f_loc (1/(Pa·s))
    """
    # Smooth max(0, x) using softplus for differentiability
    # softplus(x) = log(1 + exp(x)), approaches x for x >> 0, approaches 0 for x << 0
    # Scale factor to control transition sharpness
    scale = tau_y * 0.01 + eps  # 1% of yield stress for smooth transition

    delta_sigma = jnp.abs(sigma) - tau_y

    # Smooth approximation: softplus(delta_sigma / scale) * scale
    # This is equivalent to max(0, delta_sigma) but smooth
    x = delta_sigma / scale
    # Use jax.nn.softplus for numerically stable implementation
    # (handles gradient overflow that manual jnp.where pattern misses)
    delta_smooth = scale * jax.nn.softplus(x)

    # Fluidity: f = γ̇ / σ, where γ̇ = ((σ - τ_y) / K)^(1/n)
    # So f = ((σ - τ_y)^(1/n) / K^(1/n)) / σ
    # Simplify: f_loc = (δσ)^(1/n) / (K^(1/n) * |σ|) when σ > τ_y
    # But for the evolution equation, we use f_loc = (δσ/K)^(1/n)
    # which has units of 1/(Pa·s) when n=1

    # Actually the standard form is:
    # f_loc = (max(0, |σ| - τ_y) / K)^(1/n)  [strain rate form]
    # Then γ̇ = f_loc [this is shear rate, not fluidity]

    # For fluidity f = γ̇/σ:
    # f = ((|σ| - τ_y) / K)^(1/n) / |σ|

    # Let's use the simpler form where f_loc directly gives shear rate
    # when multiplied by stress in evolution equations
    f_loc = jnp.power(delta_smooth / (K + eps), 1.0 / n)

    return f_loc


# ============================================================================
# Spatial Discretization for Non-Local Model
# ============================================================================


@jax.jit
def laplacian_1d_neumann(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    """Compute 1D Laplacian with Neumann (zero-flux) boundary conditions.

    ∂²f/∂y² ≈ (f[i+1] - 2f[i] + f[i-1]) / dy²

    Neumann BCs: ∂f/∂y = 0 at boundaries
    Implemented via ghost points: f[-1] = f[0], f[N] = f[N-1]

    Args:
        f: Field array, shape (N_y,)
        dy: Grid spacing

    Returns:
        Laplacian array, shape (N_y,)
    """
    # Interior points
    lap = (jnp.roll(f, -1) - 2.0 * f + jnp.roll(f, 1)) / (dy**2)

    # Neumann boundary conditions: zero-flux
    # At y=0: ∂f/∂y = 0 → f[-1] = f[1] (ghost point)
    # At y=L: ∂f/∂y = 0 → f[N] = f[N-2] (ghost point)
    # Interior Laplacian formula: (f[i+1] - 2f[i] + f[i-1]) / dy²
    # At i=0: (f[1]-2f[0]+f[-1])/dy² = 2(f[1]-f[0])/dy² (via ghost f[-1]=f[1])
    # At i=N-1: (f[N]-2f[N-1]+f[N-2])/dy² = 2(f[N-2]-f[N-1])/dy² (via ghost)

    lap_bc0 = 2.0 * (f[1] - f[0]) / (dy**2)
    lap_bcN = 2.0 * (f[-2] - f[-1]) / (dy**2)

    lap = lap.at[0].set(lap_bc0)
    lap = lap.at[-1].set(lap_bcN)

    return lap


# ============================================================================
# ODE/PDE Right-Hand Sides
# ============================================================================


def fluidity_local_ode_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """ODE vector field for Local Fluidity model.

    State vector: y = [σ, f]
    - σ: stress (Pa)
    - f: fluidity (1/(Pa·s))

    Evolution equations:
    dσ/dt = G(γ̇ - σf)                           [Maxwell + plastic flow]
    df/dt = (f_eq - f)/θ + a|γ̇|^n(f_inf - f)    [aging + rejuvenation]

    where:
    - γ̇ is the applied shear rate (input)
    - f_eq is equilibrium (low-shear) fluidity
    - f_inf is high-shear fluidity
    - θ is relaxation time (aging timescale)
    - a is rejuvenation amplitude
    - n is rejuvenation exponent

    Args:
        t: Time (s)
        y: State vector [σ, f]
        args: Dictionary of parameters and inputs:
            - gamma_dot: Total shear rate (1/s)
            - G: Elastic modulus (Pa)
            - f_eq: Equilibrium fluidity (1/(Pa·s))
            - f_inf: High-shear fluidity (1/(Pa·s))
            - theta: Relaxation time (s)
            - a: Rejuvenation amplitude (dimensionless)
            - n_rejuv: Rejuvenation exponent (dimensionless)

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Unpack state
    sigma = y[0]
    f = y[1]

    # Get parameters
    G = args["G"]
    f_eq = args["f_eq"]
    f_inf = args["f_inf"]
    theta = args["theta"]
    a = args["a"]
    n_rejuv = args["n_rejuv"]

    # Get forcing
    gamma_dot = args.get("gamma_dot", 0.0)

    # Ensure fluidity is positive
    f_safe = jnp.maximum(f, 1e-20)

    # 1. Stress evolution: Maxwell + plastic
    # dσ/dt = G(γ̇ - σf)
    d_sigma = G * (gamma_dot - sigma * f_safe)

    # 2. Fluidity evolution: aging + rejuvenation
    # df/dt = (f_eq - f)/θ + a|γ̇|^n(f_inf - f)
    # Aging term: relaxation toward f_eq
    aging_rate = (f_eq - f_safe) / theta

    # Rejuvenation term: flow-induced increase toward f_inf
    gamma_dot_abs = jnp.abs(gamma_dot)
    rejuv_rate = a * jnp.power(gamma_dot_abs + 1e-20, n_rejuv) * (f_inf - f_safe)

    d_f = aging_rate + rejuv_rate

    return jnp.array([d_sigma, d_f])


def fluidity_local_creep_ode_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """ODE vector field for Local Fluidity model in creep mode.

    State vector: y = [γ, f]
    - γ: strain (dimensionless)
    - f: fluidity (1/(Pa·s))

    For creep (constant stress σ_applied):
    dγ/dt = σ_applied * f                        [strain rate from fluidity]
    df/dt = (f_eq - f)/θ + a|γ̇|^n(f_inf - f)    [aging + rejuvenation]

    Args:
        t: Time (s)
        y: State vector [γ, f]
        args: Dictionary of parameters and inputs:
            - sigma_applied: Applied stress (Pa)
            - f_eq: Equilibrium fluidity (1/(Pa·s))
            - f_inf: High-shear fluidity (1/(Pa·s))
            - theta: Relaxation time (s)
            - a: Rejuvenation amplitude
            - n_rejuv: Rejuvenation exponent

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Unpack state
    gamma = y[0]  # noqa: F841 - kept for clarity
    f = y[1]

    # Get parameters
    sigma_applied = args["sigma_applied"]
    f_eq = args["f_eq"]
    f_inf = args["f_inf"]
    theta = args["theta"]
    a = args["a"]
    n_rejuv = args["n_rejuv"]

    # Ensure fluidity is positive
    f_safe = jnp.maximum(f, 1e-20)

    # 1. Strain evolution
    # γ̇ = σ * f (strain rate is stress times fluidity)
    gamma_dot = sigma_applied * f_safe
    d_gamma = gamma_dot

    # 2. Fluidity evolution
    aging_rate = (f_eq - f_safe) / theta
    rejuv_rate = a * jnp.power(jnp.abs(gamma_dot) + 1e-20, n_rejuv) * (f_inf - f_safe)
    d_f = aging_rate + rejuv_rate

    return jnp.array([d_gamma, d_f])


def fluidity_nonlocal_pde_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """PDE vector field for Non-Local Fluidity model.

    State vector: y = [Σ, f[0], f[1], ..., f[N_y-1]]
    - Σ: macroscopic stress (Pa) - bulk average
    - f[i]: fluidity at grid point i (1/(Pa·s))

    Couette geometry assumed: γ̇(y) ≈ V/H (homogeneous in gap)
    For stress-controlled: use Σ directly
    For rate-controlled: Σ determined by force balance

    Evolution equations:
    ∂f/∂t = (f_loc(σ) - f)/θ + ξ²∂²f/∂y²

    where:
    - f_loc(σ) is local equilibrium fluidity from HB
    - θ is relaxation time
    - ξ is cooperativity length
    - ∂²f/∂y² is diffusion term (Laplacian)

    The local stress σ(y) is assumed uniform = Σ in Couette
    (or could be computed from γ̇(y) in more complex geometries)

    Args:
        t: Time (s)
        y: State vector [Σ, f[0], ..., f[N_y-1]]
        args: Dictionary of parameters and inputs:
            - gamma_dot: Applied shear rate (1/s) for rate-controlled
            - sigma_applied: Applied stress (Pa) for stress-controlled (optional)
            - mode: "rate_controlled" or "stress_controlled"
            - G: Elastic modulus (Pa)
            - tau_y: Yield stress (Pa)
            - K: Flow consistency (Pa·s^n)
            - n_flow: Flow exponent
            - theta: Relaxation time (s)
            - xi: Cooperativity length (m)
            - dy: Grid spacing (m)
            - N_y: Number of grid points

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Unpack state — infer N_y from state vector shape to avoid
    # non-static integer in args (required for jax.checkpoint compatibility)
    Sigma = y[0]
    f_field = y[1:]

    # Get parameters
    G = args["G"]
    tau_y = args["tau_y"]
    K = args["K"]
    n_flow = args["n_flow"]
    theta = args["theta"]
    xi = args["xi"]
    dy = args["dy"]

    # Ensure fluidity is positive
    f_field_safe = jnp.maximum(f_field, 1e-20)

    # Average fluidity (for bulk response)
    f_avg = jnp.mean(f_field_safe)

    # Mode determines stress evolution (0=rate_controlled, 1=stress_controlled)
    mode = args.get("mode", 0)
    gamma_dot = args.get("gamma_dot", 0.0)
    sigma_applied = args.get("sigma_applied", 0.0)

    # Rate controlled: dΣ/dt = G(γ̇ - Σ*f_avg)
    d_Sigma_rate = G * (gamma_dot - Sigma * f_avg)
    # Stress controlled: dΣ/dt = 0
    d_Sigma = jnp.where(mode == 1, 0.0, d_Sigma_rate)
    sigma_local = jnp.where(mode == 1, sigma_applied, Sigma)

    # Fluidity field evolution
    # 1. Local equilibrium fluidity from HB
    f_loc = f_loc_herschel_bulkley(sigma_local, tau_y, K, n_flow)

    # 2. Relaxation toward local equilibrium
    relax_rate = (f_loc - f_field_safe) / theta

    # 3. Non-local diffusion: ξ²∂²f/∂y²
    lap_f = laplacian_1d_neumann(f_field_safe, dy)
    diffusion_rate = xi**2 * lap_f

    # Total fluidity evolution
    d_f_field = relax_rate + diffusion_rate

    # Assemble output
    d_y = jnp.concatenate([jnp.array([d_Sigma]), d_f_field])

    return d_y


def fluidity_nonlocal_creep_pde_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """PDE vector field for Non-Local Fluidity model in creep mode.

    State vector: y = [γ, f[0], f[1], ..., f[N_y-1]]
    - γ: bulk strain (dimensionless)
    - f[i]: fluidity at grid point i (1/(Pa·s))

    For creep, stress is constant and we track strain:
    dγ/dt = σ_applied * f_avg
    ∂f/∂t = (f_loc - f)/θ + ξ²∂²f/∂y²

    Args:
        t: Time (s)
        y: State vector [γ, f[0], ..., f[N_y-1]]
        args: Dictionary of parameters:
            - sigma_applied: Applied stress (Pa)
            - tau_y: Yield stress (Pa)
            - K: Flow consistency (Pa·s^n)
            - n_flow: Flow exponent
            - theta: Relaxation time (s)
            - xi: Cooperativity length (m)
            - dy: Grid spacing (m)
            - N_y: Number of grid points

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Unpack state — infer N_y from state vector shape to avoid
    # non-static integer in args (required for jax.checkpoint compatibility)
    gamma = y[0]  # noqa: F841 - kept for clarity
    f_field = y[1:]

    # Get parameters
    sigma_applied = args["sigma_applied"]
    tau_y = args["tau_y"]
    K = args["K"]
    n_flow = args["n_flow"]
    theta = args["theta"]
    xi = args["xi"]
    dy = args["dy"]

    # Ensure fluidity is positive
    f_field_safe = jnp.maximum(f_field, 1e-20)

    # Average fluidity
    f_avg = jnp.mean(f_field_safe)

    # 1. Strain evolution: dγ/dt = σ * f_avg
    d_gamma = sigma_applied * f_avg

    # 2. Fluidity field evolution
    f_loc = f_loc_herschel_bulkley(sigma_applied, tau_y, K, n_flow)
    relax_rate = (f_loc - f_field_safe) / theta
    lap_f = laplacian_1d_neumann(f_field_safe, dy)
    diffusion_rate = xi**2 * lap_f
    d_f_field = relax_rate + diffusion_rate

    # Assemble output
    d_y = jnp.concatenate([jnp.array([d_gamma]), d_f_field])

    return d_y


# ============================================================================
# Steady-State Flow Curve (Algebraic Solutions)
# ============================================================================


@jax.jit
def fluidity_local_steady_state(
    gamma_dot: jnp.ndarray,
    G: float,
    tau_y: float,
    K: float,
    n_flow: float,
    f_eq: float,
    f_inf: float,
    theta: float,
    a: float,
    n_rejuv: float,
) -> jnp.ndarray:
    """Compute steady-state flow curve for Local Fluidity model.

    At steady state:
    f_ss = (f_eq/θ + a|γ̇|^n * f_inf) / (1/θ + a|γ̇|^n)
    σ_ss = γ̇ / f_ss

    This is derived from df/dt = 0:
    0 = (f_eq - f)/θ + a|γ̇|^n(f_inf - f)
    Solving for f gives f_ss.

    Args:
        gamma_dot: Shear rate array (1/s)
        G: Elastic modulus (Pa) - not used for steady state
        tau_y: Yield stress (Pa) - additive yield contribution
        K: Flow consistency (Pa·s^n) - used in HB contribution
        n_flow: Flow exponent - used in HB contribution
        f_eq: Equilibrium fluidity (1/(Pa·s))
        f_inf: High-shear fluidity (1/(Pa·s))
        theta: Relaxation time (s)
        a: Rejuvenation amplitude
        n_rejuv: Rejuvenation exponent

    Returns:
        Steady-state stress array (Pa)
    """
    gamma_dot_abs = jnp.abs(gamma_dot)

    # Flow term contribution
    flow_term = a * jnp.power(gamma_dot_abs + 1e-20, n_rejuv)

    # Steady-state fluidity
    # f_ss = (f_eq/θ + flow_term * f_inf) / (1/θ + flow_term)
    numerator = f_eq / theta + flow_term * f_inf
    denominator = 1.0 / theta + flow_term

    f_ss = numerator / (denominator + 1e-20)

    # Stress: σ = τ_y + γ̇ / f_ss
    # The yield stress τ_y provides the finite stress at γ̇→0,
    # ensuring yield-stress fluid behavior. K and n_flow contribute
    # indirectly through the fluidity evolution (f_eq, f_inf dependence).
    sigma_ss = tau_y + gamma_dot_abs / (f_ss + 1e-20)

    # Preserve sign of gamma_dot
    sigma_ss = sigma_ss * jnp.sign(gamma_dot + 1e-20)

    return sigma_ss


@jax.jit
def fluidity_nonlocal_steady_state(
    gamma_dot: jnp.ndarray,
    G: float,
    tau_y: float,
    K: float,
    n_flow: float,
    f_eq: float,
    f_inf: float,
    theta: float,
) -> jnp.ndarray:
    """Compute steady-state flow curve for Non-Local Fluidity model.

    For homogeneous steady state (no shear banding):
    σ_ss comes from inverting f_loc(σ) = f_ss where f_ss = γ̇/σ

    For Herschel-Bulkley:
    σ = τ_y + K*γ̇^n

    Args:
        gamma_dot: Shear rate array (1/s)
        G: Elastic modulus (Pa) - not used for steady state
        tau_y: Yield stress (Pa)
        K: Flow consistency (Pa·s^n)
        n_flow: Flow exponent
        f_eq: Equilibrium fluidity - not used for HB steady state
        f_inf: High-shear fluidity - not used for HB steady state
        theta: Relaxation time - not used for steady state

    Returns:
        Steady-state stress array (Pa)
    """
    # HB flow curve: σ = τ_y + K*|γ̇|^n * sign(γ̇)
    sigma_ss = tau_y + K * jnp.power(jnp.abs(gamma_dot) + 1e-20, n_flow)
    sigma_ss = sigma_ss * jnp.sign(gamma_dot + 1e-20)

    return sigma_ss


# ============================================================================
# Shear Banding Metrics
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
