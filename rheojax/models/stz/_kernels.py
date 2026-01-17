"""JAX-accelerated physics kernels for STZ model.

This module implements the core physics equations for the Shear Transformation Zone
(STZ) model using JAX. It follows the Langer (2008) formulation.

Key functions:
- rate_factor_C: Activation rate factor with log-cosh stability
- transition_T: Bias factor using tanh
- stz_density: Density of zones (Lambda)
- plastic_rate: Plastic strain rate calculation
- chi_evolution_langer2008: Effective temperature evolution (Langer 2008)
- lambda_evolution: STZ density evolution (for Standard/Full variants)
- m_evolution: Orientational bias evolution (for Full variant)
- stz_ode_rhs: Vector field for Diffrax
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

# Safe import ensures float64
jax, jnp = safe_import_jax()


@jax.jit
def rate_factor_C(stress: float, sigma_y: float) -> float:
    """Compute rate factor C(sigma) using overflow-safe log-cosh.

    C(sigma) = cosh(sigma / sigma_y)

    For large arguments, cosh(x) ~ exp(|x|) / 2.
    We compute this in log space to avoid overflow, then exp().

    Args:
        stress: Deviatoric stress (Pa)
        sigma_y: Yield stress scale (Pa)

    Returns:
        Rate factor C (dimensionless, >= 1)
    """
    x = stress / sigma_y

    # Safe computation of cosh(x)
    # log(cosh(x)) = |x| - log(2) + log(1 + exp(-2|x|))
    abs_x = jnp.abs(x)
    log_cosh = abs_x - jnp.log(2.0) + jnp.log1p(jnp.exp(-2.0 * abs_x))

    return jnp.exp(log_cosh)


@jax.jit
def transition_T(stress: float, sigma_y: float) -> float:
    """Compute transition bias T(sigma) using tanh.

    T(sigma) = tanh(sigma / sigma_y)

    Represents the bias of forward vs backward transformations.

    Args:
        stress: Deviatoric stress (Pa)
        sigma_y: Yield stress scale (Pa)

    Returns:
        Transition bias T (dimensionless, [-1, 1])
    """
    return jnp.tanh(stress / sigma_y)


@jax.jit
def stz_density(chi: float) -> float:
    """Compute equilibrium STZ density Lambda(chi).

    Lambda = exp(-1 / chi)

    Args:
        chi: Effective temperature (dimensionless, scaled by ez/kB)

    Returns:
        STZ density Lambda (dimensionless)
    """
    # Avoid division by zero
    chi_safe = jnp.maximum(chi, 1e-6)
    return jnp.exp(-1.0 / chi_safe)


@jax.jit
def plastic_rate(
    stress: float,
    Lambda: float,
    chi: float,  # Used if Lambda is not state var, but here we pass Lambda
    sigma_y: float,
    tau0: float,
    epsilon0: float,
) -> float:
    """Compute plastic strain rate.

    gamma_dot_pl = (2 * epsilon0 / tau0) * Lambda * C(sigma) * T(sigma)

    The exp(-1/chi) is typically inside Lambda (if Lambda is equilibrium).
    Here Lambda is passed explicitly (either state var or computed).

    Args:
        stress: Deviatoric stress (Pa)
        Lambda: STZ density
        chi: Effective temperature (unused if Lambda passed directly, kept for API consistency)
        sigma_y: Yield stress scale (Pa)
        tau0: Molecular attempt time (s)
        epsilon0: Characteristic strain increment (dimensionless)

    Returns:
        Plastic strain rate (1/s)
    """
    C = rate_factor_C(stress, sigma_y)
    T = transition_T(stress, sigma_y)

    prefactor = (2.0 * epsilon0) / tau0

    return prefactor * Lambda * C * T


@jax.jit
def chi_evolution_langer2008(
    chi: float,
    chi_inf: float,
    gamma_dot_pl: float,
    stress: float,
    sigma_y: float,
    c0: float,
) -> float:
    """Compute effective temperature evolution d(chi)/dt.

    Follows Langer (2008) formulation:
    d(chi)/dt = (gamma_dot_pl * stress / (c0 * sigma_y)) * (chi_inf - chi)

    Usually the rate is proportional to plastic work W_pl = stress * gamma_dot_pl.
    Scaling factor c0 represents specific heat.

    Args:
        chi: Current effective temperature
        chi_inf: Steady-state effective temperature
        gamma_dot_pl: Plastic strain rate (1/s)
        stress: Deviatoric stress (Pa)
        sigma_y: Yield stress scale (Pa) (for normalization)
        c0: Effective specific heat parameter (dimensionless or energy units)

    Returns:
        d(chi)/dt
    """
    # Plastic work rate
    work_rate = stress * gamma_dot_pl

    # Normalized driving force
    # We use absolute work rate to ensure aging drives to chi_inf correctly
    # Rate factor kappa * W_pl / sigma_y
    # c0 handles the scaling
    rate = jnp.abs(work_rate) / (c0 * sigma_y)

    # Evolution
    dchi = rate * (chi_inf - chi)

    return dchi


@jax.jit
def lambda_evolution(
    Lambda: float,
    chi: float,
    tau_relax: float,
) -> float:
    """Compute STZ density evolution d(Lambda)/dt.

    For Standard/Full variants where Lambda is a state variable.
    Relaxation toward equilibrium Lambda_eq(chi) = exp(-1/chi).

    d(Lambda)/dt = -(Lambda - exp(-1/chi)) / tau_relax

    Args:
        Lambda: Current STZ density
        chi: Current effective temperature
        tau_relax: Relaxation timescale (s)

    Returns:
        d(Lambda)/dt
    """
    Lambda_eq = stz_density(chi)
    return -(Lambda - Lambda_eq) / tau_relax


@jax.jit
def m_evolution(
    m: float,
    stress: float,
    gamma_dot_pl: float,
    sigma_y: float,
    m_inf: float,
    rate_m: float,
) -> float:
    """Compute orientational bias m evolution.

    For Full variant.
    dm/dt = rate_m * gamma_dot_pl * (m_inf * sign(stress) - m)

    Args:
        m: Current orientational bias
        stress: Current stress
        gamma_dot_pl: Plastic strain rate
        sigma_y: Yield stress
        m_inf: Saturation value for m
        rate_m: Rate coefficient

    Returns:
        dm/dt
    """
    target = m_inf * jnp.sign(stress)
    return rate_m * jnp.abs(gamma_dot_pl) * (target - m)


def stz_ode_rhs(
    t: float,
    y: jnp.ndarray,
    args: dict,
) -> jnp.ndarray:
    """ODE vector field for STZ model.

    Computes dy/dt for state vector y.

    State vector y depends on variant:
    - Minimal: [stress, chi]
    - Standard: [stress, chi, Lambda]
    - Full: [stress, chi, Lambda, m]

    Args:
        t: Time (s)
        y: State vector
        args: Dictionary of parameters and inputs:
            - gamma_dot: Current total strain rate (1/s)
            - G0: Shear modulus (Pa)
            - sigma_y: Yield stress (Pa)
            - tau0: Attempt time (s)
            - epsilon0: Strain increment
            - chi_inf: Steady state chi
            - c0: Specific heat
            - variant: 0 (Minimal), 1 (Standard), 2 (Full)
            - tau_beta: Relaxation time for Lambda (Standard/Full)
            - m_inf: m saturation (Full)
            - rate_m: m rate (Full)

    Returns:
        dy/dt: Time derivative of state vector
    """
    # Unpack variant - assume passed as integer
    # We must handle JAX tracing, so variant should be static or handled carefully
    # In diffrax, args are passed through.
    # To support JIT, we should avoid conditional branching on non-static args if possible,
    # or ensure variant is passed structurally.
    # However, if 'args' is a dict, it might be treated as pytree.
    # We'll assume variant is consistent with the y shape.

    # We can infer variant from y.shape
    n_states = y.shape[0]

    # Unpack state
    stress = y[0]
    chi = y[1]

    # Default values
    Lambda = 0.0
    m = 0.0

    # Logic based on shape/variant
    if n_states == 2:  # Minimal
        Lambda = stz_density(chi)
    elif n_states == 3:  # Standard
        Lambda = y[2]
    elif n_states == 4:  # Full
        Lambda = y[2]
        m = y[3]
    else:
        # Fallback for scalar/other shapes
        Lambda = stz_density(chi)

    # Get parameters from args
    # Note: dictionary access in JIT requires args to be a registered PyTree or use loose dict
    # Diffrax passes args as is.
    G0 = args["G0"]
    sigma_y = args["sigma_y"]
    tau0 = args["tau0"]
    epsilon0 = args["epsilon0"]
    chi_inf = args["chi_inf"]
    c0 = args["c0"]

    # Get forcing
    gamma_dot_tot = args.get("gamma_dot", 0.0)

    # Compute plastic rate
    d_gamma_pl = plastic_rate(stress, Lambda, chi, sigma_y, tau0, epsilon0)

    # 1. Stress evolution: d_sigma/dt = G0 * (gamma_dot_tot - gamma_dot_pl)
    d_sigma = G0 * (gamma_dot_tot - d_gamma_pl)

    # 2. Chi evolution
    d_chi = chi_evolution_langer2008(chi, chi_inf, d_gamma_pl, stress, sigma_y, c0)

    derivatives = [d_sigma, d_chi]

    # 3. Lambda evolution
    if n_states >= 3:
        tau_beta = args.get("tau_beta", tau0 * 100.0)
        d_lambda = lambda_evolution(Lambda, chi, tau_beta)
        derivatives.append(d_lambda)

    # 4. m evolution
    if n_states >= 4:
        m_inf = args.get("m_inf", 0.1)
        rate_m = args.get("rate_m", 1.0)
        d_m = m_evolution(m, stress, d_gamma_pl, sigma_y, m_inf, rate_m)
        derivatives.append(d_m)

    return jnp.stack(derivatives)
