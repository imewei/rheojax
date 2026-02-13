"""JAX-JIT kernels for FIKH (Fractional IKH) models.

This module provides the core computational kernels for FIKH models:
- Fractional structure evolution with Caputo derivative
- Temperature-coupled viscoplastic stress evolution
- Return mapping algorithm with fractional memory correction
- Scan kernels for efficient time-series processing

The FIKH model extends IKH with:
1. Caputo fractional derivative for structure evolution (power-law memory)
2. Full thermokinematic coupling (Arrhenius + viscous heating)

State vector for thermal FIKH: [σ, α, R, T, γᵖ, λ]
    - σ: deviatoric stress
    - α: backstress (kinematic hardening)
    - R: isotropic hardening (optional)
    - T: temperature
    - γᵖ: accumulated plastic strain
    - λ: structure parameter

State vector for isothermal FIKH: [σ, α, γᵖ, λ]
"""

from functools import partial
from typing import Any

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from rheojax.models.fikh._caputo import (
    compute_l1_coefficients,
    update_history_buffer,
)
from rheojax.models.fikh._thermal import (
    arrhenius_viscosity,
    temperature_evolution_rate,
    thermal_yield_stress,
)


@jax.jit
def macaulay(x: jnp.ndarray) -> jnp.ndarray:
    """Macaulay bracket: <x> = max(0, x)."""
    return jnp.maximum(0.0, x)


@jax.jit
def sign_safe(x: jnp.ndarray, eps: float = 1e-20) -> jnp.ndarray:
    """Sign function with regularization to avoid zero."""
    return jnp.sign(x + eps)


# =============================================================================
# Fractional Structure Evolution
# =============================================================================


@jax.jit
def fractional_structure_rhs(
    lam: jnp.ndarray,
    gamma_dot_p_abs: jnp.ndarray,
    tau_thix: float,
    Gamma: float,
) -> jnp.ndarray:
    """Right-hand side of fractional structure evolution.

    For the equation: D^α λ = f(λ, γ̇ᵖ)
    where f(λ, γ̇ᵖ) = (1-λ)/τ_thix - Γ·λ·|γ̇ᵖ|

    This is the RHS that the Caputo derivative equals.

    Args:
        lam: Current structure parameter.
        gamma_dot_p_abs: Absolute plastic shear rate.
        tau_thix: Thixotropic rebuilding time scale.
        Gamma: Structural breakdown coefficient.

    Returns:
        RHS value f(λ, γ̇ᵖ).
    """
    # Build-up term (Brownian/chemical recovery)
    tau_safe = jnp.maximum(tau_thix, 1e-12)
    build_up = (1.0 - lam) / tau_safe

    # Breakdown term (shear-induced)
    break_down = Gamma * lam * gamma_dot_p_abs

    return build_up - break_down


@jax.jit
def solve_fractional_lambda_implicit(
    lam_n: jnp.ndarray,
    lam_history: jnp.ndarray,
    gamma_dot_p_abs: jnp.ndarray,
    dt: float,
    alpha: float,
    tau_thix: float,
    Gamma: float,
    b_coeffs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve for λ_{n+1} using implicit L1 scheme for fractional ODE.

    The fractional ODE: D^α λ = (1-λ)/τ - Γλ|γ̇ᵖ|

    Using L1 implicit discretization:
        (λ_{n+1} - Σ w_k λ_{n-k}) / (Γ(2-α)·dt^α) = f(λ_{n+1})

    Rearranging for the linear case where f is affine in λ:
        λ_{n+1} = (history_term + dt^α·Γ(2-α)·(1/τ)) /
                  (1 + dt^α·Γ(2-α)·(1/τ + Γ|γ̇ᵖ|))

    Args:
        lam_n: Current structure parameter.
        lam_history: History buffer [λ_{n-N}, ..., λ_{n-1}].
        gamma_dot_p_abs: Absolute plastic shear rate.
        dt: Time step.
        alpha: Fractional order.
        tau_thix: Thixotropic time scale.
        Gamma: Breakdown coefficient.
        b_coeffs: Pre-computed L1 coefficients.

    Returns:
        Updated structure parameter λ_{n+1}.
    """
    # Normalization factor for L1 scheme
    gamma_factor = jax.scipy.special.gamma(2.0 - alpha)
    dt_alpha = jnp.power(dt, alpha)
    c_alpha = gamma_factor * dt_alpha

    # Effective rates
    tau_safe = jnp.maximum(tau_thix, 1e-12)
    k_build = 1.0 / tau_safe
    k_break = Gamma * gamma_dot_p_abs

    # Compute history contribution using L1 weights
    # Full history: [λ_{n-N}, ..., λ_{n-1}, λ_n]
    if lam_history.ndim == 0:
        # Scalar case - no history
        history_term = lam_n
    else:
        full_history = jnp.concatenate([lam_history, lam_n[None]])
        n_hist = full_history.shape[0]

        # L1 scheme: Σ b_k (λ_{n-k+1} - λ_{n-k}) for k=0,...,n-1
        # Simplify: b_0·λ_n + Σ_{k=1}^{n-1} (b_k - b_{k-1})·λ_{n-k} - b_{n-1}·λ_0
        # For short memory, we use a weighted sum approximation
        b_trunc = b_coeffs[: n_hist - 1]
        diffs = full_history[1:] - full_history[:-1]
        diffs_reversed = jnp.flip(diffs, axis=0)
        history_term = lam_n - jnp.dot(b_trunc, diffs_reversed) / (
            gamma_factor * jnp.power(dt, alpha)
        )

    # Implicit solve: λ_{n+1} = (history + c_α·k_build) / (1 + c_α·(k_build + k_break))
    numerator = history_term + c_alpha * k_build
    denominator = 1.0 + c_alpha * (k_build + k_break)

    lam_next = numerator / jnp.maximum(denominator, 1e-12)

    # Clip to valid range
    return jnp.clip(lam_next, 0.0, 1.0)


@jax.jit
def solve_fractional_lambda_explicit(
    lam_n: jnp.ndarray,
    lam_history: jnp.ndarray,
    gamma_dot_p_abs: jnp.ndarray,
    dt: float,
    alpha: float,
    tau_thix: float,
    Gamma: float,
    b_coeffs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve for λ_{n+1} using explicit predictor-corrector for fractional ODE.

    Uses Adams-Bashforth-Moulton predictor-corrector adapted for fractional ODEs.

    For simplicity, we use a first-order approximation:
        λ_{n+1} ≈ λ_n + dt^α · Γ(1+α) · f(λ_n, γ̇ᵖ_n) + O(dt^{2α})

    This maintains the fractional scaling while being explicit and stable
    for small dt.

    Args:
        lam_n: Current structure parameter.
        lam_history: History buffer (used for memory correction).
        gamma_dot_p_abs: Absolute plastic shear rate.
        dt: Time step.
        alpha: Fractional order.
        tau_thix: Thixotropic time scale.
        Gamma: Breakdown coefficient.
        b_coeffs: Pre-computed L1 coefficients.

    Returns:
        Updated structure parameter λ_{n+1}.
    """
    # Handle zero time step (initial condition)
    dt_safe = jnp.maximum(dt, 1e-15)

    # RHS at current state
    f_n = fractional_structure_rhs(lam_n, gamma_dot_p_abs, tau_thix, Gamma)

    # Memory correction from history
    if lam_history.ndim == 0 or lam_history.shape[0] < 2:
        memory_correction = 0.0
    else:
        # Compute fractional derivative of history
        full_history = jnp.concatenate([lam_history, lam_n[None]])
        n_hist = full_history.shape[0]
        b_trunc = b_coeffs[: n_hist - 1]

        diffs = full_history[1:] - full_history[:-1]
        diffs_reversed = jnp.flip(diffs, axis=0)

        gamma_factor = jax.scipy.special.gamma(2.0 - alpha)
        dt_alpha = jnp.power(dt_safe, alpha)

        # Past contribution (memory term)
        if len(b_trunc) > 1:
            memory_correction = jnp.dot(b_trunc[1:], diffs_reversed[1:]) / (
                gamma_factor * dt_alpha
            )
        else:
            memory_correction = 0.0

    # Explicit update with fractional scaling
    gamma_1_plus_alpha = jax.scipy.special.gamma(1.0 + alpha)
    dt_alpha = jnp.power(dt_safe, alpha)

    # For dt ≈ 0, no update (keep initial value)
    update = dt_alpha * gamma_1_plus_alpha * (f_n - memory_correction)
    lam_next = jnp.where(dt > 1e-14, lam_n + update, lam_n)

    # Clip to valid range
    return jnp.clip(lam_next, 0.0, 1.0)


# =============================================================================
# FIKH Return Mapping (Incremental Formulation)
# =============================================================================


@jax.jit
def fikh_return_step_isothermal(
    state: tuple[jnp.ndarray, ...],
    inputs: tuple[jnp.ndarray, ...],
    params: dict[str, Any],
    lam_history: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    alpha: float,
) -> tuple[tuple, tuple, jnp.ndarray]:
    """Radial return mapping step for isothermal FIKH model.

    State: (σ, α, γᵖ, λ) - stress, backstress, plastic strain, structure

    Args:
        state: Current state tuple (sigma, alpha, gamma_p, lam).
        inputs: Input tuple (dt, d_gamma).
        params: Model parameters dictionary.
        lam_history: History buffer for λ.
        b_coeffs: Pre-computed L1 coefficients.
        alpha: Fractional order.

    Returns:
        new_state: Updated state tuple.
        outputs: (sigma_total, d_gamma_p) - total stress and plastic increment.
        lam_history_new: Updated history buffer.
    """
    sigma_n, alpha_n, gamma_p_n, lam_n = state
    dt, d_gamma = inputs

    # Extract parameters
    G = params["G"]
    eta = params.get("eta", 1e6)
    C = params.get("C", 0.0)
    gamma_dyn = params.get("gamma_dyn", 1.0)
    m = params.get("m", 1.0)
    sigma_y0 = params["sigma_y0"]
    delta_sigma_y = params.get("delta_sigma_y", 0.0)
    tau_thix = params.get("tau_thix", 1.0)
    Gamma_thix = params.get("Gamma", 0.5)
    eta_inf = params.get("eta_inf", 0.0)

    # Shear rate
    dt_safe = jnp.maximum(dt, 1e-15)
    gamma_dot = d_gamma / dt_safe

    # Current yield stress
    sigma_y_current = sigma_y0 + delta_sigma_y * lam_n

    # 1. Elastic Predictor (with Maxwell relaxation)
    tau_relax = eta / jnp.maximum(G, 1e-12)
    relax_factor = jnp.exp(-dt / tau_relax)
    sigma_trial = sigma_n * relax_factor + G * d_gamma

    # Effective stress (relative to backstress)
    xi_trial = sigma_trial - alpha_n
    xi_abs = jnp.abs(xi_trial)

    # 2. Yield condition
    f_yield = xi_abs - sigma_y_current
    is_plastic = f_yield > 0

    # 3. Plastic corrector
    sign_xi = sign_safe(xi_trial)

    # AF dynamic recovery contribution
    alpha_abs = jnp.abs(alpha_n)
    af_term = gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * sign_xi * alpha_n

    # Denominator for plastic multiplier
    denom = G + C - af_term
    denom_safe = jnp.maximum(denom, G / 10.0)

    # Plastic multiplier
    d_gamma_p = macaulay(f_yield) / denom_safe

    # Stress update
    sigma_next = sigma_trial - G * d_gamma_p * sign_xi

    # Backstress update (Armstrong-Frederick)
    d_alpha = (
        C * d_gamma_p * sign_xi
        - gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha_n * d_gamma_p
    )
    alpha_next = alpha_n + d_alpha

    # Plastic strain accumulation
    gamma_p_next = gamma_p_n + d_gamma_p

    # 4. Fractional structure evolution
    gamma_dot_p_abs = d_gamma_p / dt_safe

    # Use explicit scheme for stability in scan
    lam_next = solve_fractional_lambda_explicit(
        jnp.atleast_1d(lam_n)[0] if lam_n.ndim == 0 else lam_n,
        lam_history,
        gamma_dot_p_abs,
        dt,
        alpha,
        tau_thix,
        Gamma_thix,
        b_coeffs,
    )

    # Select elastic or plastic values
    sigma_final = jnp.where(is_plastic, sigma_next, sigma_trial)
    alpha_final = jnp.where(is_plastic, alpha_next, alpha_n)
    gamma_p_final = jnp.where(is_plastic, gamma_p_next, gamma_p_n)
    d_gamma_p_final = jnp.where(is_plastic, d_gamma_p, 0.0)

    # Lambda evolves even in elastic regime (buildup only when elastic)
    lam_elastic = solve_fractional_lambda_explicit(
        jnp.atleast_1d(lam_n)[0] if lam_n.ndim == 0 else lam_n,
        lam_history,
        0.0,  # No plastic flow
        dt,
        alpha,
        tau_thix,
        Gamma_thix,
        b_coeffs,
    )
    lam_final = jnp.where(is_plastic, lam_next, lam_elastic)

    # Update history buffer (lam_final is scalar)
    lam_history_new = update_history_buffer(lam_history, lam_final)

    # 5. Total stress (with viscous background)
    sigma_total = sigma_final + eta_inf * gamma_dot

    new_state = (sigma_final, alpha_final, gamma_p_final, lam_final)
    outputs = (sigma_total, d_gamma_p_final)

    return new_state, outputs, lam_history_new


@jax.jit
def fikh_return_step_thermal(
    state: tuple[jnp.ndarray, ...],
    inputs: tuple[jnp.ndarray, ...],
    params: dict[str, Any],
    lam_history: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    alpha: float,
) -> tuple[tuple, tuple, jnp.ndarray]:
    """Radial return mapping step for thermal FIKH model.

    State: (σ, α, R, T, γᵖ, λ) - stress, backstress, isotropic hardening,
           temperature, plastic strain, structure

    Args:
        state: Current state tuple.
        inputs: Input tuple (dt, d_gamma).
        params: Model parameters dictionary.
        lam_history: History buffer for λ.
        b_coeffs: Pre-computed L1 coefficients.
        alpha: Fractional order.

    Returns:
        new_state: Updated state tuple.
        outputs: (sigma_total, d_gamma_p, T) - stress, plastic increment, temperature.
        lam_history_new: Updated history buffer.
    """
    sigma_n, alpha_n, R_n, T_n, gamma_p_n, lam_n = state
    dt, d_gamma = inputs

    # Extract parameters
    G = params["G"]
    eta = params.get("eta", 1e6)
    C = params.get("C", 0.0)
    gamma_dyn = params.get("gamma_dyn", 1.0)
    m = params.get("m", 1.0)
    sigma_y0 = params["sigma_y0"]
    delta_sigma_y = params.get("delta_sigma_y", 0.0)
    tau_thix = params.get("tau_thix", 1.0)
    Gamma_thix = params.get("Gamma", 0.5)
    eta_inf = params.get("eta_inf", 0.0)

    # Thermal parameters
    T_ref = params.get("T_ref", 298.15)
    E_a = params.get("E_a", 5e4)
    E_y = params.get("E_y", 3e4)
    m_y = params.get("m_y", 1.0)
    rho_cp = params.get("rho_cp", 4e6)
    chi = params.get("chi", 0.9)
    h = params.get("h", 100.0)
    T_env = params.get("T_env", 298.15)

    # Temperature-dependent viscosity
    eta_T = arrhenius_viscosity(eta, T_n, T_ref, E_a)

    # Temperature and structure dependent yield stress
    sigma_y_base = sigma_y0 + delta_sigma_y * lam_n
    sigma_y_current = thermal_yield_stress(sigma_y_base, lam_n, m_y, T_n, T_ref, E_y)

    # Shear rate
    dt_safe = jnp.maximum(dt, 1e-15)
    gamma_dot = d_gamma / dt_safe

    # 1. Elastic Predictor (with temperature-dependent relaxation)
    tau_relax = eta_T / jnp.maximum(G, 1e-12)
    relax_factor = jnp.exp(-dt / tau_relax)
    sigma_trial = sigma_n * relax_factor + G * d_gamma

    # Effective stress
    xi_trial = sigma_trial - alpha_n
    xi_abs = jnp.abs(xi_trial)

    # 2. Yield condition (with isotropic hardening R)
    f_yield = xi_abs - (sigma_y_current + R_n)
    is_plastic = f_yield > 0

    # 3. Plastic corrector
    sign_xi = sign_safe(xi_trial)

    # AF contribution
    alpha_abs = jnp.abs(alpha_n)
    af_term = gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * sign_xi * alpha_n

    denom = G + C - af_term
    denom_safe = jnp.maximum(denom, G / 10.0)

    d_gamma_p = macaulay(f_yield) / denom_safe

    # Stress update
    sigma_next = sigma_trial - G * d_gamma_p * sign_xi

    # Backstress update
    d_alpha = (
        C * d_gamma_p * sign_xi
        - gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha_n * d_gamma_p
    )
    alpha_next = alpha_n + d_alpha

    # Isotropic hardening update (optional)
    Q_iso = params.get("Q_iso", 0.0)
    d_R = Q_iso * (1.0 - R_n / jnp.maximum(Q_iso, 1e-12)) * d_gamma_p
    R_next = R_n + d_R

    # Plastic strain
    gamma_p_next = gamma_p_n + d_gamma_p

    # 4. Temperature evolution
    gamma_dot_p = d_gamma_p / dt_safe
    dT_dt = temperature_evolution_rate(
        T_n, sigma_next, gamma_dot_p, T_env, rho_cp, chi, h
    )
    T_next = T_n + dT_dt * dt

    # 5. Fractional structure evolution
    gamma_dot_p_abs = jnp.abs(gamma_dot_p)
    lam_next = solve_fractional_lambda_explicit(
        jnp.atleast_1d(lam_n)[0] if lam_n.ndim == 0 else lam_n,
        lam_history,
        gamma_dot_p_abs,
        dt,
        alpha,
        tau_thix,
        Gamma_thix,
        b_coeffs,
    )

    # Select values
    sigma_final = jnp.where(is_plastic, sigma_next, sigma_trial)
    alpha_final = jnp.where(is_plastic, alpha_next, alpha_n)
    R_final = jnp.where(is_plastic, R_next, R_n)
    T_final = T_next  # Temperature always evolves
    gamma_p_final = jnp.where(is_plastic, gamma_p_next, gamma_p_n)
    d_gamma_p_final = jnp.where(is_plastic, d_gamma_p, 0.0)

    # Lambda evolution
    lam_elastic = solve_fractional_lambda_explicit(
        jnp.atleast_1d(lam_n)[0] if lam_n.ndim == 0 else lam_n,
        lam_history,
        0.0,
        dt,
        alpha,
        tau_thix,
        Gamma_thix,
        b_coeffs,
    )
    lam_final = jnp.where(is_plastic, lam_next, lam_elastic)

    # Update history
    lam_history_new = update_history_buffer(lam_history, jnp.atleast_1d(lam_final))

    # Total stress
    sigma_total = sigma_final + eta_inf * gamma_dot

    new_state = (sigma_final, alpha_final, R_final, T_final, gamma_p_final, lam_final)
    outputs = (sigma_total, d_gamma_p_final, T_final)

    return new_state, outputs, lam_history_new


# =============================================================================
# ODE Formulations (for Diffrax integration)
# =============================================================================


def fikh_maxwell_ode_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict[str, Any],
) -> jnp.ndarray:
    """Maxwell ODE RHS for rate-controlled FIKH protocols.

    State vector y = [σ, α, T, γᵖ, λ] (5 components, thermal)
    or y = [σ, α, γᵖ, λ] (4 components, isothermal)

    Used for: startup, relaxation protocols.

    Args:
        t: Current time.
        y: State vector.
        args: Parameter dictionary including 'gamma_dot' for rate.

    Returns:
        dy/dt: Rate of change of state.
    """
    include_thermal = args.get("include_thermal", True)

    if include_thermal:
        sigma, alpha, T, _gamma_p, lam = y[0], y[1], y[2], y[3], y[4]
    else:
        sigma, alpha, _gamma_p, lam = y[0], y[1], y[2], y[3]
        T = args.get("T_ref", 298.15)

    # Parameters
    G = args["G"]
    eta = args.get("eta", 1e6)
    C = args.get("C", 0.0)
    gamma_dyn = args.get("gamma_dyn", 1.0)
    m = args.get("m", 1.0)
    sigma_y0 = args["sigma_y0"]
    delta_sigma_y = args.get("delta_sigma_y", 0.0)
    tau_thix = args.get("tau_thix", 1.0)
    Gamma_thix = args.get("Gamma", 0.5)
    alpha_frac = args.get("alpha_structure", 0.5)
    gamma_dot = args.get("gamma_dot", 0.0)

    # Thermal parameters
    T_ref = args.get("T_ref", 298.15)
    E_a = args.get("E_a", 5e4)
    E_y = args.get("E_y", 3e4)
    m_y = args.get("m_y", 1.0)
    rho_cp = args.get("rho_cp", 4e6)
    chi = args.get("chi", 0.9)
    h = args.get("h", 100.0)
    T_env = args.get("T_env", 298.15)

    # Temperature-dependent viscosity
    eta_T = arrhenius_viscosity(eta, T, T_ref, E_a) if include_thermal else eta

    # Yield stress
    sigma_y_base = sigma_y0 + delta_sigma_y * lam
    sigma_y = (
        thermal_yield_stress(sigma_y_base, lam, m_y, T, T_ref, E_y)
        if include_thermal
        else sigma_y_base
    )

    # Effective stress
    xi = sigma - alpha
    xi_abs = jnp.abs(xi)
    sign_xi = sign_safe(xi)

    # Plastic flow (Perzyna-type overstress)
    mu_p = args.get("mu_p", 1e-3)
    f_yield = xi_abs - sigma_y
    gamma_dot_p = macaulay(f_yield) / jnp.maximum(mu_p, 1e-12) * sign_xi

    # Stress rate: dσ/dt = G(γ̇ - γ̇ᵖ) - σ/τ
    tau_relax = eta_T / jnp.maximum(G, 1e-12)
    d_sigma = G * (gamma_dot - gamma_dot_p) - sigma / tau_relax

    # Backstress rate (Armstrong-Frederick)
    alpha_abs = jnp.abs(alpha)
    d_alpha = C * jnp.abs(gamma_dot_p) * sign_xi - gamma_dyn * jnp.power(
        alpha_abs + 1e-20, m - 1
    ) * alpha * jnp.abs(gamma_dot_p)

    # Plastic strain rate
    d_gamma_p = jnp.abs(gamma_dot_p)

    # Structure evolution (integer-order approximation in ODE)
    # For proper fractional: D^α λ = RHS, we use α → 1 limit in ODE
    # The fractional behavior is captured in the scan kernel
    gamma_1_alpha = jax.scipy.special.gamma(1.0 + alpha_frac)
    tau_eff = tau_thix / gamma_1_alpha  # Effective time scale adjustment
    d_lam = fractional_structure_rhs(lam, jnp.abs(gamma_dot_p), tau_eff, Gamma_thix)

    if include_thermal:
        # Temperature rate
        d_T = temperature_evolution_rate(
            T, sigma, jnp.abs(gamma_dot_p), T_env, rho_cp, chi, h
        )
        return jnp.array([d_sigma, d_alpha, d_T, d_gamma_p, d_lam])
    else:
        return jnp.array([d_sigma, d_alpha, d_gamma_p, d_lam])


def fikh_creep_ode_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict[str, Any],
) -> jnp.ndarray:
    """Creep ODE RHS for stress-controlled FIKH protocol.

    State vector y = [γ, α, T, γᵖ, λ] (5 components, thermal)
    or y = [γ, α, γᵖ, λ] (4 components, isothermal)

    Used for: creep protocol with constant applied stress.

    Args:
        t: Current time.
        y: State vector.
        args: Parameter dictionary including 'sigma_applied'.

    Returns:
        dy/dt: Rate of change of state.
    """
    include_thermal = args.get("include_thermal", True)

    if include_thermal:
        _gamma, alpha, T, _gamma_p, lam = y[0], y[1], y[2], y[3], y[4]
    else:
        _gamma, alpha, _gamma_p, lam = y[0], y[1], y[2], y[3]
        T = args.get("T_ref", 298.15)

    # Applied stress
    sigma_applied = args.get("sigma_applied", 100.0)
    sigma = sigma_applied

    # Parameters
    eta = args.get("eta", 1e6)
    C = args.get("C", 0.0)
    gamma_dyn = args.get("gamma_dyn", 1.0)
    m = args.get("m", 1.0)
    sigma_y0 = args["sigma_y0"]
    delta_sigma_y = args.get("delta_sigma_y", 0.0)
    tau_thix = args.get("tau_thix", 1.0)
    Gamma_thix = args.get("Gamma", 0.5)
    alpha_frac = args.get("alpha_structure", 0.5)
    eta_inf = args.get("eta_inf", 0.0)

    # Thermal parameters
    T_ref = args.get("T_ref", 298.15)
    E_a = args.get("E_a", 5e4)
    E_y = args.get("E_y", 3e4)
    m_y = args.get("m_y", 1.0)
    rho_cp = args.get("rho_cp", 4e6)
    chi = args.get("chi", 0.9)
    h = args.get("h", 100.0)
    T_env = args.get("T_env", 298.15)

    # Temperature-dependent viscosity
    eta_T = arrhenius_viscosity(eta, T, T_ref, E_a) if include_thermal else eta

    # Yield stress
    sigma_y_base = sigma_y0 + delta_sigma_y * lam
    sigma_y = (
        thermal_yield_stress(sigma_y_base, lam, m_y, T, T_ref, E_y)
        if include_thermal
        else sigma_y_base
    )

    # Effective stress
    xi = sigma - alpha
    xi_abs = jnp.abs(xi)
    sign_xi = sign_safe(xi)

    # Plastic flow
    mu_p = args.get("mu_p", 1e-3)
    f_yield = xi_abs - sigma_y
    gamma_dot_p = macaulay(f_yield) / jnp.maximum(mu_p, 1e-12) * sign_xi

    # Total strain rate from stress equilibrium: σ = G·(γ - γᵖ) + η_inf·γ̇
    # For creep with constant σ: γ̇ = γ̇ᵖ + σ/(η + η_inf) (Maxwell-like)
    # More rigorously: σ = σ_elastic + σ_viscous = G·(γ-γᵖ-γᵛ) + η·γ̇ᵛ
    # Simplification for EVP creep:
    d_gamma = gamma_dot_p + sigma / eta_T + eta_inf * gamma_dot_p / eta_T

    # Backstress rate
    alpha_abs = jnp.abs(alpha)
    d_alpha = C * jnp.abs(gamma_dot_p) * sign_xi - gamma_dyn * jnp.power(
        alpha_abs + 1e-20, m - 1
    ) * alpha * jnp.abs(gamma_dot_p)

    # Plastic strain rate
    d_gamma_p = jnp.abs(gamma_dot_p)

    # Structure evolution
    gamma_1_alpha = jax.scipy.special.gamma(1.0 + alpha_frac)
    tau_eff = tau_thix / gamma_1_alpha
    d_lam = fractional_structure_rhs(lam, jnp.abs(gamma_dot_p), tau_eff, Gamma_thix)

    if include_thermal:
        d_T = temperature_evolution_rate(
            T, sigma, jnp.abs(gamma_dot_p), T_env, rho_cp, chi, h
        )
        return jnp.array([d_gamma, d_alpha, d_T, d_gamma_p, d_lam])
    else:
        return jnp.array([d_gamma, d_alpha, d_gamma_p, d_lam])


# =============================================================================
# Scan Kernels for Time-Series Processing
# =============================================================================


@partial(jax.jit, static_argnums=(2, 4))
def fikh_scan_kernel_isothermal(
    times: jnp.ndarray,
    strains: jnp.ndarray,
    n_history: int = 100,
    alpha: float = 0.5,
    use_viscosity: bool = True,
    **params,
) -> jnp.ndarray:
    """Scan kernel for isothermal FIKH model.

    Processes full time series using JAX scan with history buffer.

    Args:
        times: Array of time points.
        strains: Array of strain points.
        n_history: Number of history points for fractional derivative.
        alpha: Fractional order for structure evolution.
        use_viscosity: Whether to include η_inf term.
        **params: Model parameters.

    Returns:
        sigma_total_series: Array of stress values.
    """
    # Pre-compute L1 coefficients
    b_coeffs = compute_l1_coefficients(alpha, n_history + 1)

    # Calculate increments
    dts = jnp.diff(times, prepend=times[0])
    d_gammas = jnp.diff(strains, prepend=strains[0])

    # Initial state: (sigma=0, alpha=0, gamma_p=0, lambda=1)
    init_state = (
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(1.0),
    )

    # Initial history buffer (λ=1 throughout)
    init_history = jnp.ones(n_history)

    # Prepare params dict
    params_local = dict(params)
    if not use_viscosity:
        params_local["eta_inf"] = 0.0

    def scan_fn(carry, inputs):
        state, history = carry
        dt, d_gamma = inputs

        new_state, (stress, _), new_history = fikh_return_step_isothermal(
            state, (dt, d_gamma), params_local, history, b_coeffs, alpha
        )
        return (new_state, new_history), stress

    # Run scan
    inputs = (dts, d_gammas)
    _, stress_series = jax.lax.scan(scan_fn, (init_state, init_history), inputs)

    return stress_series


@partial(jax.jit, static_argnums=(2, 4))
def fikh_scan_kernel_thermal(
    times: jnp.ndarray,
    strains: jnp.ndarray,
    n_history: int = 100,
    alpha: float = 0.5,
    use_viscosity: bool = True,
    T_init: float = 298.15,
    **params,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scan kernel for thermal FIKH model.

    Args:
        times: Array of time points.
        strains: Array of strain points.
        n_history: Number of history points.
        alpha: Fractional order.
        use_viscosity: Whether to include η_inf term.
        T_init: Initial temperature [K].
        **params: Model parameters.

    Returns:
        sigma_total_series: Array of stress values.
        T_series: Array of temperature values.
    """
    b_coeffs = compute_l1_coefficients(alpha, n_history + 1)

    dts = jnp.diff(times, prepend=times[0])
    d_gammas = jnp.diff(strains, prepend=strains[0])

    # Initial state: (sigma, alpha, R, T, gamma_p, lambda)
    init_state = (
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(T_init),
        jnp.array(0.0),
        jnp.array(1.0),
    )

    init_history = jnp.ones(n_history)

    params_local = dict(params)
    if not use_viscosity:
        params_local["eta_inf"] = 0.0

    def scan_fn(carry, inputs):
        state, history = carry
        dt, d_gamma = inputs

        new_state, (stress, _, T), new_history = fikh_return_step_thermal(
            state, (dt, d_gamma), params_local, history, b_coeffs, alpha
        )
        return (new_state, new_history), (stress, T)

    _, (stress_series, T_series) = jax.lax.scan(
        scan_fn, (init_state, init_history), (dts, d_gammas)
    )

    return stress_series, T_series


# =============================================================================
# Steady-State Flow Curve
# =============================================================================


@jax.jit
def fikh_flow_curve_steady_state(
    gamma_dot: jnp.ndarray,
    include_thermal: bool = False,
    **params,
) -> jnp.ndarray:
    """Compute steady-state flow curve for FIKH model.

    At steady state:
    - λ_ss = 1 / (1 + Γ·τ_thix·|γ̇|)  (integer-order limit)
    - σ_y = σ_y0 + δσ_y·λ_ss
    - σ = σ_y + η_eff·|γ̇|

    For the fractional case, λ_ss depends on α but the qualitative
    behavior is similar.

    Args:
        gamma_dot: Array of shear rates.
        include_thermal: Whether to include thermal effects.
        **params: Model parameters.

    Returns:
        Steady-state stress values.
    """
    sigma_y0 = params.get("sigma_y0", 10.0)
    delta_sigma_y = params.get("delta_sigma_y", 50.0)
    tau_thix = params.get("tau_thix", 1.0)
    Gamma_thix = params.get("Gamma", 0.5)
    eta_inf = params.get("eta_inf", 0.1)
    mu_p = params.get("mu_p", 1e-3)
    gamma_dot_abs = jnp.abs(gamma_dot)

    # Steady-state structure parameter
    # For fractional: λ_ss ~ (Γ·τ·|γ̇|)^(-α) at high rates
    # For simplicity, use integer-order formula (α→1 limit)
    denominator = 1.0 + Gamma_thix * tau_thix * gamma_dot_abs
    lam_ss = 1.0 / jnp.maximum(denominator, 1e-12)

    # Steady-state yield stress
    sigma_y_ss = sigma_y0 + delta_sigma_y * lam_ss

    # Effective viscosity (Perzyna + background)
    eta_eff = mu_p + eta_inf

    # Herschel-Bulkley type flow curve
    # σ = σ_y + η_eff·|γ̇| for |γ̇| > 0
    sigma = sigma_y_ss + eta_eff * gamma_dot_abs

    return sigma * jnp.sign(gamma_dot + 1e-20)
