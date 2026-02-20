"""IKH model kernels: ODE formulations and return mapping algorithms.

This module provides JAX-accelerated kernels for the MIKH and ML-IKH models:

1. Maxwell ODE formulation (for creep/relaxation protocols)
2. Radial return mapping (for startup/LAOS protocols)
3. Multi-mode kernels (for ML-IKH)

Key corrections over previous implementation:
- Lambda timing: Structure updated AFTER stress calculation
- AF dynamic recovery: Included in plastic multiplier denominator
- Maxwell relaxation term: -(G/η)σ for proper viscoelastic response
"""

from functools import partial

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


# =============================================================================
# Utility Functions
# =============================================================================


def macaulay(x):
    """Macaulay brackets: max(x, 0)."""
    return jnp.maximum(x, 0.0)


# =============================================================================
# Structure Evolution
# =============================================================================


@jax.jit
def evolution_lambda(lam, gamma_dot_p_abs, params):
    """Evolution of structural parameter lambda.

    d(lambda)/dt = k1*(1 - lambda) - k2*lambda*|γ̇ᵖ|

    Using the original parameterization:
    d(lambda)/dt = (1 - lambda)/tau_thix - Gamma*lambda*|γ̇ᵖ|

    Args:
        lam: Current structural parameter (0 to 1).
        gamma_dot_p_abs: Absolute plastic shear rate.
        params: Dictionary containing 'tau_thix' (or 'k1') and 'Gamma' (or 'k2').

    Returns:
        Rate of change of lambda.
    """
    # Support both parameterizations
    if "k1" in params:
        k1 = params["k1"]
        k2 = params["k2"]
    else:
        tau_thix = params["tau_thix"]
        Gamma = params["Gamma"]
        k1 = 1.0 / jnp.maximum(tau_thix, 1e-12)
        k2 = Gamma

    # Build-up term (Brownian/chemical recovery)
    build_up = k1 * (1.0 - lam)

    # Breakdown term (shear induced)
    break_down = k2 * lam * gamma_dot_p_abs

    return build_up - break_down


# =============================================================================
# Maxwell ODE Formulation (for creep/relaxation)
# =============================================================================


@jax.jit
def ikh_maxwell_ode_rhs(t, y, args):
    """ODE RHS for Maxwell formulation (stress relaxation / startup).

    State: y = [σ, α, λ]
    - σ: Deviatoric stress
    - α: Backstress (kinematic hardening)
    - λ: Structural parameter (0 to 1)

    Equations:
        dσ/dt = G(γ̇ - γ̇ᵖ) - (G/η)σ      (Maxwell + plastic)
        dα/dt = C·γ̇ᵖ - γ_dyn·|α|^(m-1)·α·|γ̇ᵖ|   (Armstrong-Frederick)
        dλ/dt = k1(1-λ) - k2·λ·|γ̇ᵖ|     (thixotropy)

    Plastic flow rule (Perzyna-type):
        γ̇ᵖ = <f>/μ_p · sign(ξ)
        f = |ξ| - σ_y(λ)
        ξ = σ - α

    Args:
        t: Time
        y: State vector [σ, α, λ]
        args: Parameter dictionary

    Returns:
        dy/dt: Time derivative of state vector
    """
    sigma, alpha, lam = y[0], y[1], y[2]

    # Extract parameters
    G = args["G"]
    eta = args.get("eta", 1e12)  # Maxwell viscosity (large = elastic)
    C = args["C"]
    gamma_dyn = args.get("gamma_dyn", args.get("q", 1.0))
    m = args.get("m", 1.0)  # AF exponent
    sigma_y0 = args["sigma_y0"]
    delta_sigma_y = args.get("delta_sigma_y", args.get("k3", 0.0))
    mu_p = args.get("mu_p", 1e-6)  # Plastic viscosity (regularization)

    # Applied shear rate
    gamma_dot = args.get("gamma_dot", 0.0)

    # Current yield stress
    sigma_y = sigma_y0 + delta_sigma_y * lam

    # Relative stress (shifted by backstress)
    xi = sigma - alpha
    xi_abs = jnp.abs(xi)
    sign_xi = jnp.sign(xi + 1e-20)  # Regularized sign

    # Yield function
    f_yield = xi_abs - sigma_y

    # Plastic strain rate (Perzyna regularization)
    # γ̇ᵖ = <f>/μ_p · sign(ξ)
    gamma_dot_p = macaulay(f_yield) / jnp.maximum(mu_p, 1e-12) * sign_xi
    gamma_dot_p_abs = jnp.abs(gamma_dot_p)

    # Elastic strain rate
    gamma_dot_e = gamma_dot - gamma_dot_p

    # Stress evolution: Maxwell + plasticity
    # dσ/dt = G·γ̇ᵉ - (G/η)·σ = G(γ̇ - γ̇ᵖ) - (G/η)σ
    d_sigma = G * gamma_dot_e - (G / eta) * sigma

    # Backstress evolution: Armstrong-Frederick
    # dα/dt = C·γ̇ᵖ - γ_dyn·|α|^(m-1)·α·|γ̇ᵖ|
    alpha_abs = jnp.abs(alpha)
    recovery_term = (
        gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha * gamma_dot_p_abs
    )
    d_alpha = C * gamma_dot_p - recovery_term

    # Structure evolution
    d_lambda = evolution_lambda(lam, gamma_dot_p_abs, args)

    return jnp.array([d_sigma, d_alpha, d_lambda])


@jax.jit
def ikh_creep_ode_rhs(t, y, args):
    """ODE RHS for stress-controlled creep.

    State: y = [γ, α, λ]
    - γ: Total strain
    - α: Backstress
    - λ: Structural parameter

    For creep, stress is constant (σ = σ_applied), so we track strain.

    Equations:
        dγ/dt = γ̇ = (σ - G·γ_e)/η + γ̇ᵖ   (simplified, quasi-static)

        More precisely, we use:
        dγ/dt = σ·f + γ̇ᵖ    where f = 1/η (fluidity-like)

        Or for EVP materials:
        γ̇ = (1/η)·σ + γ̇ᵖ   (viscous + plastic)

    Args:
        t: Time
        y: State vector [γ, α, λ]
        args: Parameter dictionary

    Returns:
        dy/dt: Time derivative of state vector
    """
    # y[0] is gamma (total strain) - not needed for derivative computation
    alpha, lam = y[1], y[2]

    # Extract parameters
    eta = args.get("eta", 1e12)
    C = args["C"]
    gamma_dyn = args.get("gamma_dyn", args.get("q", 1.0))
    m = args.get("m", 1.0)
    sigma_y0 = args["sigma_y0"]
    delta_sigma_y = args.get("delta_sigma_y", args.get("k3", 0.0))
    mu_p = args.get("mu_p", 1e-6)

    # Applied stress (constant in creep)
    sigma_applied = args["sigma_applied"]

    # Current yield stress
    sigma_y = sigma_y0 + delta_sigma_y * lam

    # For creep, we need to track the elastic/plastic decomposition
    # In quasi-static: σ ≈ G·γ_e where γ = γ_e + γ_p
    # But σ is fixed, so elastic strain is: γ_e = σ/G

    # Relative stress for yield check
    xi = sigma_applied - alpha
    xi_abs = jnp.abs(xi)
    sign_xi = jnp.sign(xi + 1e-20)

    # Yield function
    f_yield = xi_abs - sigma_y

    # Plastic strain rate
    gamma_dot_p = macaulay(f_yield) / jnp.maximum(mu_p, 1e-12) * sign_xi
    gamma_dot_p_abs = jnp.abs(gamma_dot_p)

    # Total strain rate
    # In creep: γ̇ = γ̇ᵖ + (σ/η)  (viscous + plastic contributions)
    # The elastic part doesn't contribute rate since σ is constant
    gamma_dot_viscous = sigma_applied / jnp.maximum(eta, 1e-12)
    d_gamma = gamma_dot_p + gamma_dot_viscous

    # Backstress evolution
    alpha_abs = jnp.abs(alpha)
    recovery_term = (
        gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha * gamma_dot_p_abs
    )
    d_alpha = C * gamma_dot_p - recovery_term

    # Structure evolution
    d_lambda = evolution_lambda(lam, gamma_dot_p_abs, args)

    return jnp.array([d_gamma, d_alpha, d_lambda])


# -----------------------------------------------------------------------------
# Multi-Mode ODE Kernels (ML-IKH)
# -----------------------------------------------------------------------------


def make_ml_ikh_maxwell_ode_rhs_per_mode(n_modes):
    """Factory that returns an ODE RHS with n_modes captured statically.

    This avoids JAX tracing issues with dynamic slicing (y[:n_modes])
    by closing over n_modes as a Python int.
    """

    def _ode_rhs(t, y, args):
        # Extract state components using static n_modes
        sigmas = y[:n_modes]
        alphas = y[n_modes : 2 * n_modes]
        lambdas = y[2 * n_modes : 3 * n_modes]

        # Applied shear rate
        gamma_dot = args.get("gamma_dot", 0.0)

        # Per-mode parameter arrays
        G_arr = args["G"]  # Shape (n_modes,)
        C_arr = args["C"]  # Shape (n_modes,)
        gamma_dyn_arr = args["gamma_dyn"]
        sigma_y0_arr = args["sigma_y0"]
        delta_sigma_y_arr = args["delta_sigma_y"]
        tau_thix_arr = args["tau_thix"]
        Gamma_arr = args["Gamma"]
        eta_arr = args.get("eta", jnp.full(n_modes, 1e12))
        mu_p_arr = args.get("mu_p", jnp.full(n_modes, 1e-6))
        m_arr = args.get("m", jnp.ones(n_modes))

        def mode_evolution(
            sigma_i,
            alpha_i,
            lam_i,
            G_i,
            C_i,
            gamma_dyn_i,
            sigma_y0_i,
            delta_sigma_y_i,
            tau_thix_i,
            Gamma_i,
            eta_i,
            mu_p_i,
            m_i,
        ):
            """Compute derivatives for a single mode."""
            # Current yield stress for this mode
            sigma_y_i = sigma_y0_i + delta_sigma_y_i * lam_i

            # Relative stress
            xi_i = sigma_i - alpha_i
            xi_abs = jnp.abs(xi_i)
            sign_xi = jnp.sign(xi_i + 1e-20)

            # Yield function
            f_yield = xi_abs - sigma_y_i

            # Plastic strain rate (Perzyna regularization)
            gamma_dot_p = macaulay(f_yield) / jnp.maximum(mu_p_i, 1e-12) * sign_xi
            gamma_dot_p_abs = jnp.abs(gamma_dot_p)

            # Elastic strain rate for this mode
            gamma_dot_e = gamma_dot - gamma_dot_p

            # Stress evolution: Maxwell + plasticity
            d_sigma = G_i * gamma_dot_e - (G_i / eta_i) * sigma_i

            # Backstress evolution: Armstrong-Frederick
            alpha_abs = jnp.abs(alpha_i)
            recovery = (
                gamma_dyn_i
                * jnp.power(alpha_abs + 1e-20, m_i - 1)
                * alpha_i
                * gamma_dot_p_abs
            )
            d_alpha = C_i * gamma_dot_p - recovery

            # Structure evolution
            k1_i = 1.0 / jnp.maximum(tau_thix_i, 1e-12)
            k2_i = Gamma_i
            d_lambda = k1_i * (1.0 - lam_i) - k2_i * lam_i * gamma_dot_p_abs

            return d_sigma, d_alpha, d_lambda

        # Vectorize over modes
        d_sigmas, d_alphas, d_lambdas = jax.vmap(mode_evolution)(
            sigmas,
            alphas,
            lambdas,
            G_arr,
            C_arr,
            gamma_dyn_arr,
            sigma_y0_arr,
            delta_sigma_y_arr,
            tau_thix_arr,
            Gamma_arr,
            eta_arr,
            mu_p_arr,
            m_arr,
        )

        return jnp.concatenate([d_sigmas, d_alphas, d_lambdas])

    return _ode_rhs


def ml_ikh_maxwell_ode_rhs_per_mode(t, y, args):
    """ODE RHS for multi-mode Maxwell formulation (per-mode yield surfaces).

    State: y = [σ_1, ..., σ_N, α_1, ..., α_N, λ_1, ..., λ_N]  (3N states)
    - σ_i: Deviatoric stress for mode i
    - α_i: Backstress for mode i
    - λ_i: Structural parameter for mode i

    Each mode evolves independently. Total stress is sum of mode stresses.

    NOTE: This legacy wrapper reads n_modes from args at runtime.
    For JIT-traced contexts (e.g. diffrax), use make_ml_ikh_maxwell_ode_rhs_per_mode(n_modes)
    to capture n_modes statically in a closure.

    Args:
        t: Time
        y: State vector of length 3*n_modes
        args: Parameter dictionary including 'n_modes' and per-mode parameters

    Returns:
        dy/dt: Time derivative of state vector
    """
    n_modes = args["n_modes"]
    return make_ml_ikh_maxwell_ode_rhs_per_mode(n_modes)(t, y, args)


def make_ml_ikh_maxwell_ode_rhs_weighted_sum(n_modes):
    """Factory that returns an ODE RHS with n_modes captured statically.

    This avoids JAX tracing issues with dynamic slicing (y[2:2+n_modes])
    by closing over n_modes as a Python int.
    """

    def _ode_rhs(t, y, args):
        # Extract state using static n_modes
        sigma = y[0]
        alpha = y[1]
        lambdas = y[2 : 2 + n_modes]

        # Global parameters
        G = args["G"]
        C = args["C"]
        gamma_dyn = args.get("gamma_dyn", 1.0)
        m = args.get("m", 1.0)
        sigma_y0 = args["sigma_y0"]
        k3 = args.get("k3", 0.0)
        eta = args.get("eta", 1e12)
        mu_p = args.get("mu_p", 1e-6)
        gamma_dot = args.get("gamma_dot", 0.0)

        # Per-mode parameters
        tau_thix_arr = args["tau_thix"]
        Gamma_arr = args["Gamma"]
        w_arr = args.get("w", jnp.ones(n_modes) / n_modes)

        # Global yield stress from weighted sum of lambdas
        sigma_y = sigma_y0 + k3 * jnp.sum(w_arr * lambdas)

        # Relative stress
        xi = sigma - alpha
        xi_abs = jnp.abs(xi)
        sign_xi = jnp.sign(xi + 1e-20)

        # Yield function and plastic flow
        f_yield = xi_abs - sigma_y
        gamma_dot_p = macaulay(f_yield) / jnp.maximum(mu_p, 1e-12) * sign_xi
        gamma_dot_p_abs = jnp.abs(gamma_dot_p)

        # Elastic strain rate
        gamma_dot_e = gamma_dot - gamma_dot_p

        # Stress evolution
        d_sigma = G * gamma_dot_e - (G / eta) * sigma

        # Backstress evolution
        alpha_abs = jnp.abs(alpha)
        recovery = gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha * gamma_dot_p_abs
        d_alpha = C * gamma_dot_p - recovery

        # Lambda evolution for each mode (share same plastic rate)
        def evolve_lambda(lam_i, tau_i, Gamma_i):
            k1_i = 1.0 / jnp.maximum(tau_i, 1e-12)
            k2_i = Gamma_i
            return k1_i * (1.0 - lam_i) - k2_i * lam_i * gamma_dot_p_abs

        d_lambdas = jax.vmap(evolve_lambda)(lambdas, tau_thix_arr, Gamma_arr)

        return jnp.concatenate([jnp.array([d_sigma, d_alpha]), d_lambdas])

    return _ode_rhs


def ml_ikh_maxwell_ode_rhs_weighted_sum(t, y, args):
    """ODE RHS for multi-mode Maxwell formulation (weighted-sum yield surface).

    State: y = [σ, α, λ_1, ..., λ_N]  (2+N states)
    - σ: Global deviatoric stress
    - α: Global backstress
    - λ_i: Structural parameter for mode i

    Single yield surface with weighted structure contribution.

    NOTE: This legacy wrapper reads n_modes from args at runtime.
    For JIT-traced contexts (e.g. diffrax), use make_ml_ikh_maxwell_ode_rhs_weighted_sum(n_modes)
    to capture n_modes statically in a closure.

    Args:
        t: Time
        y: State vector of length 2+n_modes
        args: Parameter dictionary

    Returns:
        dy/dt: Time derivative of state vector
    """
    n_modes = args["n_modes"]
    return make_ml_ikh_maxwell_ode_rhs_weighted_sum(n_modes)(t, y, args)


def make_ml_ikh_creep_ode_rhs_per_mode(n_modes):
    """Factory that returns an ODE RHS with n_modes captured statically.

    This avoids JAX tracing issues with dynamic slicing (y[1:1+n_modes])
    by closing over n_modes as a Python int.
    """

    def _ode_rhs(t, y, args):
        # Extract state using static n_modes
        # y[0] is gamma (total strain) - not needed for derivative computation
        alphas = y[1 : 1 + n_modes]
        lambdas = y[1 + n_modes : 1 + 2 * n_modes]

        # Applied stress (constant in creep)
        sigma_applied = args["sigma_applied"]
        eta_inf = args.get("eta_inf", 0.0)

        # Per-mode parameters
        G_arr = args["G"]
        C_arr = args["C"]
        gamma_dyn_arr = args["gamma_dyn"]
        sigma_y0_arr = args["sigma_y0"]
        delta_sigma_y_arr = args["delta_sigma_y"]
        tau_thix_arr = args["tau_thix"]
        Gamma_arr = args["Gamma"]
        eta_arr = args.get("eta", jnp.full(n_modes, 1e12))
        mu_p_arr = args.get("mu_p", jnp.full(n_modes, 1e-6))
        m_arr = args.get("m", jnp.ones(n_modes))

        def mode_creep(
            alpha_i,
            lam_i,
            G_i,
            C_i,
            gamma_dyn_i,
            sigma_y0_i,
            delta_sigma_y_i,
            tau_thix_i,
            Gamma_i,
            eta_i,
            mu_p_i,
            m_i,
        ):
            """Compute creep derivatives for a single mode."""
            # Mode yield stress
            sigma_y_i = sigma_y0_i + delta_sigma_y_i * lam_i

            # Relative stress (stress distributed across modes)
            xi_i = sigma_applied - alpha_i
            xi_abs = jnp.abs(xi_i)
            sign_xi = jnp.sign(xi_i + 1e-20)

            # Yield function
            f_yield = xi_abs - sigma_y_i

            # Plastic strain rate
            gamma_dot_p = macaulay(f_yield) / jnp.maximum(mu_p_i, 1e-12) * sign_xi
            gamma_dot_p_abs = jnp.abs(gamma_dot_p)

            # Backstress evolution
            alpha_abs = jnp.abs(alpha_i)
            recovery = (
                gamma_dyn_i
                * jnp.power(alpha_abs + 1e-20, m_i - 1)
                * alpha_i
                * gamma_dot_p_abs
            )
            d_alpha = C_i * gamma_dot_p - recovery

            # Lambda evolution
            k1_i = 1.0 / jnp.maximum(tau_thix_i, 1e-12)
            k2_i = Gamma_i
            d_lambda = k1_i * (1.0 - lam_i) - k2_i * lam_i * gamma_dot_p_abs

            # Viscous contribution from mode
            gamma_dot_viscous_i = sigma_applied / jnp.maximum(eta_i, 1e-12)

            return gamma_dot_p + gamma_dot_viscous_i, d_alpha, d_lambda

        # Vectorize over modes
        gamma_dot_modes, d_alphas, d_lambdas = jax.vmap(mode_creep)(
            alphas,
            lambdas,
            G_arr,
            C_arr,
            gamma_dyn_arr,
            sigma_y0_arr,
            delta_sigma_y_arr,
            tau_thix_arr,
            Gamma_arr,
            eta_arr,
            mu_p_arr,
            m_arr,
        )

        # Total strain rate (sum of mode contributions + global viscous)
        d_gamma = jnp.sum(gamma_dot_modes) / n_modes  # Average contribution
        # Add global viscous contribution (safe for eta_inf=0)
        d_gamma = d_gamma + sigma_applied / jnp.maximum(eta_inf, 1e-30)

        return jnp.concatenate([jnp.array([d_gamma]), d_alphas, d_lambdas])

    return _ode_rhs


def ml_ikh_creep_ode_rhs_per_mode(t, y, args):
    """ODE RHS for multi-mode creep (per-mode yield surfaces).

    State: y = [γ, α_1, ..., α_N, λ_1, ..., λ_N]  (1+2N states)
    - γ: Total strain (shared, since material is in series for stress)
    - α_i: Backstress for mode i
    - λ_i: Structural parameter for mode i

    For creep, stress is constant. Each mode contributes plastic strain rate.
    Total strain rate = Σᵢ γ̇ᵖ_i + σ/η_total

    NOTE: This legacy wrapper reads n_modes from args at runtime.
    For JIT-traced contexts (e.g. diffrax), use make_ml_ikh_creep_ode_rhs_per_mode(n_modes)
    to capture n_modes statically in a closure.

    Args:
        t: Time
        y: State vector
        args: Parameter dictionary

    Returns:
        dy/dt: Time derivative of state vector
    """
    n_modes = args["n_modes"]
    return make_ml_ikh_creep_ode_rhs_per_mode(n_modes)(t, y, args)


def make_ml_ikh_creep_ode_rhs_weighted_sum(n_modes):
    """Factory that returns an ODE RHS with n_modes captured statically.

    This avoids JAX tracing issues with dynamic slicing (y[2:2+n_modes])
    by closing over n_modes as a Python int.
    """

    def _ode_rhs(t, y, args):
        # Extract state using static n_modes
        # y[0] is gamma (total strain) - not needed for derivative computation
        alpha = y[1]
        lambdas = y[2 : 2 + n_modes]

        # Global parameters
        C = args["C"]
        gamma_dyn = args.get("gamma_dyn", 1.0)
        m = args.get("m", 1.0)
        sigma_y0 = args["sigma_y0"]
        k3 = args.get("k3", 0.0)
        eta = args.get("eta", 1e12)
        mu_p = args.get("mu_p", 1e-6)
        eta_inf = args.get("eta_inf", 0.0)

        sigma_applied = args["sigma_applied"]

        # Per-mode parameters
        tau_thix_arr = args["tau_thix"]
        Gamma_arr = args["Gamma"]
        w_arr = args.get("w", jnp.ones(n_modes) / n_modes)

        # Global yield stress
        sigma_y = sigma_y0 + k3 * jnp.sum(w_arr * lambdas)

        # Relative stress
        xi = sigma_applied - alpha
        xi_abs = jnp.abs(xi)
        sign_xi = jnp.sign(xi + 1e-20)

        # Yield function and plastic flow
        f_yield = xi_abs - sigma_y
        gamma_dot_p = macaulay(f_yield) / jnp.maximum(mu_p, 1e-12) * sign_xi
        gamma_dot_p_abs = jnp.abs(gamma_dot_p)

        # Strain rate
        gamma_dot_viscous = sigma_applied / jnp.maximum(eta, 1e-12)
        d_gamma = gamma_dot_p + gamma_dot_viscous
        # Add global viscous contribution (safe for eta_inf=0)
        d_gamma = d_gamma + sigma_applied / jnp.maximum(eta_inf, 1e-30)

        # Backstress evolution
        alpha_abs = jnp.abs(alpha)
        recovery = gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha * gamma_dot_p_abs
        d_alpha = C * gamma_dot_p - recovery

        # Lambda evolution for each mode
        def evolve_lambda(lam_i, tau_i, Gamma_i):
            k1_i = 1.0 / jnp.maximum(tau_i, 1e-12)
            k2_i = Gamma_i
            return k1_i * (1.0 - lam_i) - k2_i * lam_i * gamma_dot_p_abs

        d_lambdas = jax.vmap(evolve_lambda)(lambdas, tau_thix_arr, Gamma_arr)

        return jnp.concatenate([jnp.array([d_gamma, d_alpha]), d_lambdas])

    return _ode_rhs


def ml_ikh_creep_ode_rhs_weighted_sum(t, y, args):
    """ODE RHS for multi-mode creep (weighted-sum yield surface).

    State: y = [γ, α, λ_1, ..., λ_N]  (2+N states)
    - γ: Total strain
    - α: Global backstress
    - λ_i: Structural parameter for mode i

    Single yield surface with weighted structure contribution.

    NOTE: This legacy wrapper reads n_modes from args at runtime.
    For JIT-traced contexts (e.g. diffrax), use make_ml_ikh_creep_ode_rhs_weighted_sum(n_modes)
    to capture n_modes statically in a closure.

    Args:
        t: Time
        y: State vector
        args: Parameter dictionary

    Returns:
        dy/dt: Time derivative of state vector
    """
    n_modes = args["n_modes"]
    return make_ml_ikh_creep_ode_rhs_weighted_sum(n_modes)(t, y, args)


# =============================================================================
# Radial Return Mapping (for startup/LAOS)
# =============================================================================


@jax.jit
def radial_return_step_corrected(state, inputs, params):
    """Corrected radial return mapping step for MIKH model.

    FIXES over original implementation:
    1. Lambda updated AFTER stress (timing fix)
    2. AF dynamic recovery in plastic multiplier denominator
    3. Proper plastic strain rate computation

    Args:
        state: Tuple (sigma, alpha, lam)
        inputs: Tuple (dt, d_gamma)
        params: Dictionary of model parameters

    Returns:
        new_state: (sigma_next, alpha_next, lambda_next)
        output: (sigma_total, d_gamma_p) - total stress and plastic strain increment
    """
    sigma_n, alpha_n, lambda_n = state
    dt, d_gamma = inputs

    # Extract parameters
    G = params["G"]
    C = params["C"]
    gamma_dyn = params.get("gamma_dyn", params.get("q", 1.0))
    m = params.get("m", 1.0)
    sigma_y0 = params["sigma_y0"]
    delta_sigma_y = params.get("delta_sigma_y", params.get("k3", 0.0))
    eta_inf = params.get("eta_inf", 0.0)

    # Shear rate for lambda evolution
    gamma_dot = d_gamma / jnp.maximum(dt, 1e-12)

    # Current yield stress (using lambda_n, NOT updated value)
    sigma_y_current = sigma_y0 + delta_sigma_y * lambda_n

    # 1. Elastic Predictor
    sigma_trial = sigma_n + G * d_gamma
    xi_trial = sigma_trial - alpha_n
    xi_abs = jnp.abs(xi_trial)

    # 2. Yield Condition
    f_yield = xi_abs - sigma_y_current
    is_plastic = f_yield > 0

    # 3. Plastic Corrector with AF correction in denominator
    sign_xi = jnp.sign(xi_trial + 1e-20)

    # Effective tangent modulus (includes AF dynamic recovery contribution)
    # For AF: dα = C·dγ_p·sign(ξ) - γ_dyn·|α|^(m-1)·α·|dγ_p|
    # The second term modifies the effective hardening
    alpha_abs = jnp.abs(alpha_n)
    af_correction = gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * sign_xi * alpha_n

    # Denominator with regularization to prevent singularity
    # denom = G + C - γ_dyn·|α|^(m-1)·sign(ξ)·α
    denom_full = G + C - af_correction
    denom = jnp.maximum(denom_full, G / 10.0)  # Regularization

    # Plastic multiplier
    d_gamma_p = macaulay(f_yield) / denom

    # Stress update
    sigma_next = sigma_trial - G * d_gamma_p * sign_xi

    # Backstress update (Armstrong-Frederick)
    d_alpha = (
        C * d_gamma_p * sign_xi
        - gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha_n * d_gamma_p
    )
    alpha_next = alpha_n + d_alpha

    # 4. Lambda update AFTER stress (timing fix)
    # Use plastic strain rate for lambda evolution
    gamma_dot_p_abs = d_gamma_p / jnp.maximum(dt, 1e-12)
    d_lambda = evolution_lambda(lambda_n, gamma_dot_p_abs, params) * dt
    lambda_next = jnp.clip(lambda_n + d_lambda, 0.0, 1.0)

    # Select elastic or plastic values
    sigma_final = jnp.where(is_plastic, sigma_next, sigma_trial)
    alpha_final = jnp.where(is_plastic, alpha_next, alpha_n)
    d_gamma_p_final = jnp.where(is_plastic, d_gamma_p, 0.0)

    # Lambda always evolves (even elastically, just with zero plastic rate in that case)
    # Actually, if elastic, plastic rate = 0, so lambda evolves only via buildup
    d_lambda_elastic = evolution_lambda(lambda_n, 0.0, params) * dt
    lambda_final = jnp.where(
        is_plastic, lambda_next, jnp.clip(lambda_n + d_lambda_elastic, 0.0, 1.0)
    )

    # 5. Total stress (with viscous background)
    sigma_total = sigma_final + eta_inf * gamma_dot

    return (sigma_final, alpha_final, lambda_final), (sigma_total, d_gamma_p_final)


# Legacy wrapper for backward compatibility
@jax.jit
def radial_return_step(state, inputs, params):
    """Legacy wrapper - calls corrected version and unpacks output."""
    new_state, (sigma_total, _) = radial_return_step_corrected(state, inputs, params)
    return new_state, sigma_total


# =============================================================================
# Scan Kernels
# =============================================================================


@partial(jax.jit, static_argnums=(2,))
def ikh_scan_kernel(times, strains, use_viscosity=True, **params):
    """Scan kernel for IKH model processing a full time series.

    Args:
        times: Array of time points.
        strains: Array of strain points.
        use_viscosity: Whether to include eta_inf term (static arg).
        params: Model parameters.

    Returns:
        sigma_total_series: Array of stress values.
    """
    # Calculate increments
    dts = jnp.diff(times, prepend=times[0])
    d_gammas = jnp.diff(strains, prepend=strains[0])

    # Initial state: (sigma=0, alpha=0, lambda=1)
    init_state = (0.0, 0.0, 1.0)

    # Temporarily modify params if viscosity disabled
    params_local = dict(params)
    if not use_viscosity:
        params_local["eta_inf"] = 0.0

    def scan_fn(state, inputs):
        dt, d_gamma = inputs
        new_state, (stress, _) = radial_return_step_corrected(
            state, (dt, d_gamma), params_local
        )
        return new_state, stress

    scan_fn = jax.checkpoint(scan_fn)

    # Run scan
    inputs = (dts, d_gammas)
    _, stress_series = jax.lax.scan(scan_fn, init_state, inputs)

    return stress_series


# =============================================================================
# ML-IKH Multi-Mode Kernels
# =============================================================================


@partial(jax.jit, static_argnums=(2, 3))
def ml_ikh_scan_kernel(
    times, strains, num_modes, use_viscosity=True, eta_inf=0.0, **params
):
    """Multi-Lambda IKH scan kernel (per-mode yield surfaces).

    Each mode has independent yield surface. Total stress is sum of mode stresses.

    Args:
        times: Array of time points.
        strains: Array of strains.
        num_modes: Number of modes.
        use_viscosity: Static boolean.
        eta_inf: Global high-shear viscosity (scalar).
        params: Dictionary where keys like 'G', 'C' map to arrays of shape (num_modes,).

    Returns:
        sigma_total_series: Sum of stresses from all modes.
    """
    dts = jnp.diff(times, prepend=times[0])
    d_gammas = jnp.diff(strains, prepend=strains[0])

    # Vectorize step over modes
    # State shape: (num_modes,) for each variable
    init_state = (jnp.zeros(num_modes), jnp.zeros(num_modes), jnp.ones(num_modes))

    # Per-mode params (no global viscosity at mode level)
    mode_params = dict(params)
    mode_params["eta_inf"] = jnp.zeros(num_modes)

    def vmap_return_step(states, dt_dgamma, mode_params):
        """Apply return step to each mode."""
        sigma_vec, alpha_vec, lambda_vec = states
        dt, d_gamma = dt_dgamma

        def single_mode_step(
            sigma, alpha, lam, G, C, gamma_dyn, sigma_y0, delta_sigma_y, tau_thix, Gamma
        ):
            p = {
                "G": G,
                "C": C,
                "gamma_dyn": gamma_dyn,
                "sigma_y0": sigma_y0,
                "delta_sigma_y": delta_sigma_y,
                "tau_thix": tau_thix,
                "Gamma": Gamma,
                "eta_inf": 0.0,
            }
            new_state, (stress, _) = radial_return_step_corrected(
                (sigma, alpha, lam), (dt, d_gamma), p
            )
            return new_state, stress

        new_states, stresses = jax.vmap(single_mode_step)(
            sigma_vec,
            alpha_vec,
            lambda_vec,
            mode_params["G"],
            mode_params["C"],
            mode_params["gamma_dyn"],
            mode_params["sigma_y0"],
            mode_params["delta_sigma_y"],
            mode_params["tau_thix"],
            mode_params["Gamma"],
        )
        return new_states, stresses

    def step_fn(state, inputs):
        dt, d_gamma = inputs
        new_state_tuple, stresses = vmap_return_step(state, (dt, d_gamma), mode_params)

        # Sum stresses from all modes
        total_stress = jnp.sum(stresses)

        # Add global viscosity
        if use_viscosity:
            gamma_dot = d_gamma / jnp.maximum(dt, 1e-12)
            total_stress = total_stress + eta_inf * gamma_dot

        return new_state_tuple, total_stress

    step_fn = jax.checkpoint(step_fn)

    _, stress_series = jax.lax.scan(step_fn, init_state, (dts, d_gammas))

    return stress_series


@partial(jax.jit, static_argnums=(2, 3))
def ml_ikh_weighted_sum_kernel(times, strains, num_modes, use_viscosity=True, **params):
    """ML-IKH with weighted-sum yield surface.

    Single global yield surface: σ_y = σ_y0 + k3·Σ(w_i·λ_i)
    All modes share the elastic/plastic response.

    Args:
        times: Array of time points.
        strains: Array of strains.
        num_modes: Number of structural modes.
        use_viscosity: Static boolean.
        params: Dictionary containing:
            - G: Global shear modulus
            - C: Global hardening modulus
            - gamma_dyn: Global dynamic recovery
            - sigma_y0: Base yield stress
            - k3: Structure-yield coupling
            - eta_inf: High-shear viscosity
            - tau_thix_i, Gamma_i, w_i: Per-mode structure parameters

    Returns:
        sigma_total_series: Stress time series.
    """
    dts = jnp.diff(times, prepend=times[0])
    d_gammas = jnp.diff(strains, prepend=strains[0])

    # Extract global parameters
    G = params["G"]
    C = params["C"]
    gamma_dyn = params.get("gamma_dyn", 1.0)
    m = params.get("m", 1.0)
    sigma_y0 = params["sigma_y0"]
    k3 = params.get("k3", 0.0)
    eta_inf = params.get("eta_inf", 0.0) if use_viscosity else 0.0

    # Extract per-mode parameters (arrays of length num_modes)
    tau_thix_arr = params["tau_thix"]  # Array
    Gamma_arr = params["Gamma"]  # Array
    w_arr = params.get("w", jnp.ones(num_modes) / num_modes)  # Weights, default uniform

    # Initial state: (sigma=0, alpha=0, lambda_1=1, ..., lambda_N=1)
    init_sigma = 0.0
    init_alpha = 0.0
    init_lambdas = jnp.ones(num_modes)

    def step_fn(state, inputs):
        sigma_n, alpha_n, lambdas_n = state
        dt, d_gamma = inputs

        gamma_dot = d_gamma / jnp.maximum(dt, 1e-12)

        # Weighted sum yield stress
        sigma_y = sigma_y0 + k3 * jnp.sum(w_arr * lambdas_n)

        # Elastic predictor
        sigma_trial = sigma_n + G * d_gamma
        xi_trial = sigma_trial - alpha_n
        xi_abs = jnp.abs(xi_trial)
        sign_xi = jnp.sign(xi_trial + 1e-20)

        # Yield condition
        f_yield = xi_abs - sigma_y
        is_plastic = f_yield > 0

        # Plastic corrector with AF
        alpha_abs = jnp.abs(alpha_n)
        af_correction = (
            gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * sign_xi * alpha_n
        )
        denom = jnp.maximum(G + C - af_correction, G / 10.0)
        d_gamma_p = macaulay(f_yield) / denom

        # Updates
        sigma_next = sigma_trial - G * d_gamma_p * sign_xi
        d_alpha = (
            C * d_gamma_p * sign_xi
            - gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha_n * d_gamma_p
        )
        alpha_next = alpha_n + d_alpha

        # Lambda evolution for each mode
        gamma_dot_p_abs = d_gamma_p / jnp.maximum(dt, 1e-12)

        def evolve_lambda_i(lam_i, tau_i, Gamma_i):
            k1_i = 1.0 / jnp.maximum(tau_i, 1e-12)
            k2_i = Gamma_i
            d_lam = (k1_i * (1.0 - lam_i) - k2_i * lam_i * gamma_dot_p_abs) * dt
            return jnp.clip(lam_i + d_lam, 0.0, 1.0)

        lambdas_next = jax.vmap(evolve_lambda_i)(lambdas_n, tau_thix_arr, Gamma_arr)

        # Elastic path
        d_lambda_elastic = jax.vmap(
            lambda lam, tau: (1.0 / jnp.maximum(tau, 1e-12)) * (1.0 - lam) * dt
        )(lambdas_n, tau_thix_arr)
        lambdas_elastic = jnp.clip(lambdas_n + d_lambda_elastic, 0.0, 1.0)

        # Select
        sigma_final = jnp.where(is_plastic, sigma_next, sigma_trial)
        alpha_final = jnp.where(is_plastic, alpha_next, alpha_n)
        lambdas_final = jnp.where(is_plastic, lambdas_next, lambdas_elastic)

        # Total stress
        sigma_total = sigma_final + eta_inf * gamma_dot

        new_state = (sigma_final, alpha_final, lambdas_final)
        return new_state, sigma_total

    step_fn = jax.checkpoint(step_fn)

    init_state = (init_sigma, init_alpha, init_lambdas)
    _, stress_series = jax.lax.scan(step_fn, init_state, (dts, d_gammas))

    return stress_series


# =============================================================================
# Flow Curve (Steady State)
# =============================================================================


@jax.jit
def ikh_flow_curve_steady_state(gamma_dot, **params):
    """Compute steady-state flow curve for IKH model.

    At steady state:
    - λ_ss = k1/(k1 + k2·|γ̇|)  [structure balance]
    - σ_y = σ_y0 + Δσ_y·λ_ss  [yield stress]
    - σ = σ_y + η_inf·|γ̇|    [stress above yield]

    Args:
        gamma_dot: Array of shear rates
        **params: Model parameters

    Returns:
        sigma: Array of steady-state stresses
    """
    # Extract parameters
    sigma_y0 = params["sigma_y0"]
    delta_sigma_y = params.get("delta_sigma_y", params.get("k3", 0.0))
    eta_inf = params.get("eta_inf", 0.0)

    # Structure kinetics
    if "k1" in params:
        k1 = params["k1"]
        k2 = params["k2"]
    else:
        tau_thix = params["tau_thix"]
        Gamma = params["Gamma"]
        k1 = 1.0 / jnp.maximum(tau_thix, 1e-12)
        k2 = Gamma

    gamma_dot_abs = jnp.abs(gamma_dot)

    # Steady-state lambda
    lambda_ss = k1 / (k1 + k2 * gamma_dot_abs + 1e-20)

    # Steady-state yield stress
    sigma_y_ss = sigma_y0 + delta_sigma_y * lambda_ss

    # Total stress (yield + viscous)
    sigma = sigma_y_ss + eta_inf * gamma_dot_abs

    return sigma


@partial(jax.jit, static_argnums=(1,))
def ml_ikh_flow_curve_steady_state_per_mode(gamma_dot, n_modes, **params):
    """Multi-mode steady-state flow curve (per-mode yield surfaces).

    At steady state for each mode i:
    - λ_ss_i = k1_i/(k1_i + k2_i·|γ̇|)  [structure balance]
    - σ_y_i = σ_y0_i + Δσ_y_i·λ_ss_i    [mode yield stress]

    Total: σ = Σᵢ σ_y_i + η_inf·|γ̇|

    Args:
        gamma_dot: Array of shear rates
        n_modes: Number of modes (static for JIT)
        **params: Mode parameters with keys G_i, sigma_y0_i, delta_sigma_y_i,
                  tau_thix_i, Gamma_i for i=1..n_modes, plus eta_inf

    Returns:
        sigma: Array of steady-state stresses
    """
    gamma_dot_abs = jnp.abs(gamma_dot)
    eta_inf = params.get("eta_inf", 0.0)

    # Sum yield stress contributions from all modes
    sigma_total = jnp.zeros_like(gamma_dot)

    for i in range(1, n_modes + 1):
        sigma_y0_i = params[f"sigma_y0_{i}"]
        delta_sigma_y_i = params.get(f"delta_sigma_y_{i}", 0.0)

        # Structure kinetics for mode i
        if f"k1_{i}" in params:
            k1_i = params[f"k1_{i}"]
            k2_i = params[f"k2_{i}"]
        else:
            tau_thix_i = params[f"tau_thix_{i}"]
            Gamma_i = params[f"Gamma_{i}"]
            k1_i = 1.0 / jnp.maximum(tau_thix_i, 1e-12)
            k2_i = Gamma_i

        # Steady-state lambda for mode i
        lambda_ss_i = k1_i / (k1_i + k2_i * gamma_dot_abs + 1e-20)

        # Mode yield stress contribution
        sigma_y_i = sigma_y0_i + delta_sigma_y_i * lambda_ss_i
        sigma_total = sigma_total + sigma_y_i

    # Add viscous contribution
    sigma_total = sigma_total + eta_inf * gamma_dot_abs

    return sigma_total


@partial(jax.jit, static_argnums=(1,))
def ml_ikh_flow_curve_steady_state_weighted_sum(gamma_dot, n_modes, **params):
    """Multi-mode steady-state flow curve (weighted-sum yield surface).

    At steady state:
    - λ_ss_i = k1_i/(k1_i + k2_i·|γ̇|)  [structure balance per mode]
    - σ_y = σ_y0 + k3·Σᵢ(w_i·λ_ss_i)    [global yield stress]

    Total: σ = σ_y + η_inf·|γ̇|

    Args:
        gamma_dot: Array of shear rates
        n_modes: Number of modes (static for JIT)
        **params: Contains sigma_y0, k3, w_i, tau_thix_i, Gamma_i, eta_inf

    Returns:
        sigma: Array of steady-state stresses
    """
    gamma_dot_abs = jnp.abs(gamma_dot)

    sigma_y0 = params["sigma_y0"]
    k3 = params.get("k3", 0.0)
    eta_inf = params.get("eta_inf", 0.0)

    # Compute weighted sum of steady-state lambdas
    weighted_lambda_sum = jnp.zeros_like(gamma_dot)

    for i in range(1, n_modes + 1):
        w_i = params.get(f"w_{i}", 1.0 / n_modes)

        # Structure kinetics for mode i
        if f"k1_{i}" in params:
            k1_i = params[f"k1_{i}"]
            k2_i = params[f"k2_{i}"]
        else:
            tau_thix_i = params[f"tau_thix_{i}"]
            Gamma_i = params[f"Gamma_{i}"]
            k1_i = 1.0 / jnp.maximum(tau_thix_i, 1e-12)
            k2_i = Gamma_i

        # Steady-state lambda for mode i
        lambda_ss_i = k1_i / (k1_i + k2_i * gamma_dot_abs + 1e-20)
        weighted_lambda_sum = weighted_lambda_sum + w_i * lambda_ss_i

    # Global yield stress
    sigma_y = sigma_y0 + k3 * weighted_lambda_sum

    # Total stress
    sigma = sigma_y + eta_inf * gamma_dot_abs

    return sigma
