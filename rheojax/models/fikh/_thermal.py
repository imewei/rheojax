"""Thermal coupling utilities for FIKH models.

This module provides JAX-compatible implementations of thermokinematic
coupling for the FIKH (Fractional IKH) model family, including:
- Arrhenius viscosity temperature dependence
- Thermal yield stress evolution
- Temperature evolution ODE (viscous dissipation + convective cooling)

The thermal coupling follows:
    η(T) = η_ref · exp(E_a/R · (1/T - 1/T_ref))     [Arrhenius viscosity]
    σ_y(λ,T) = σ_y0 · λ^m_y · exp(E_y/R · (1/T - 1/T_ref))  [Thermal yield]
    ρc_p·dT/dt = χ·σ·γ̇ᵖ - h·(T - T_env)            [Temperature ODE]

References:
    - Saramito, P. (2007). A new elastoviscoplastic model based on the
      Herschel-Bulkley viscoplastic model. JNNFM.
    - Dimitriou, C. J., & McKinley, G. H. (2019). A canonical framework for
      modeling elasto-viscoplasticity in complex fluids. JNNFM.
"""

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

# Universal gas constant [J/(mol·K)]
R_GAS = 8.314462618


@jax.jit
def arrhenius_viscosity(
    eta_ref: float,
    T: jnp.ndarray,
    T_ref: float,
    E_a: float,
) -> jnp.ndarray:
    """Compute Arrhenius temperature-dependent viscosity.

    η(T) = η_ref · exp(E_a/R · (1/T - 1/T_ref))

    For E_a > 0, viscosity decreases with increasing temperature.

    Args:
        eta_ref: Reference viscosity at T_ref [Pa·s].
        T: Current temperature [K].
        T_ref: Reference temperature [K].
        E_a: Activation energy [J/mol].

    Returns:
        Temperature-dependent viscosity [Pa·s].
    """
    # Avoid division by zero
    T_safe = jnp.maximum(T, 1.0)
    T_ref_safe = jnp.maximum(T_ref, 1.0)

    exponent = (E_a / R_GAS) * (1.0 / T_safe - 1.0 / T_ref_safe)

    # Clip exponent to prevent overflow
    exponent_clipped = jnp.clip(exponent, -50.0, 50.0)

    return eta_ref * jnp.exp(exponent_clipped)


@jax.jit
def arrhenius_modulus(
    G_ref: float,
    T: jnp.ndarray,
    T_ref: float,
    E_G: float,
) -> jnp.ndarray:
    """Compute Arrhenius temperature-dependent modulus.

    G(T) = G_ref · exp(E_G/R · (1/T - 1/T_ref))

    For rubber-like materials, E_G may be negative (modulus increases with T).

    Args:
        G_ref: Reference modulus at T_ref [Pa].
        T: Current temperature [K].
        T_ref: Reference temperature [K].
        E_G: Activation energy for modulus [J/mol].

    Returns:
        Temperature-dependent modulus [Pa].
    """
    T_safe = jnp.maximum(T, 1.0)
    T_ref_safe = jnp.maximum(T_ref, 1.0)

    exponent = (E_G / R_GAS) * (1.0 / T_safe - 1.0 / T_ref_safe)
    exponent_clipped = jnp.clip(exponent, -50.0, 50.0)

    return G_ref * jnp.exp(exponent_clipped)


@jax.jit
def thermal_yield_stress(
    sigma_y0: float,
    lam: jnp.ndarray,
    m_y: float,
    T: jnp.ndarray,
    T_ref: float,
    E_y: float,
) -> jnp.ndarray:
    """Compute structure- and temperature-dependent yield stress.

    σ_y(λ,T) = σ_y0 · λ^m_y · exp(E_y/R · (1/T - 1/T_ref))

    The yield stress:
    - Increases with structure parameter λ (thixotropic hardening)
    - Follows Arrhenius dependence on temperature

    Args:
        sigma_y0: Base yield stress at T_ref with λ=1 [Pa].
        lam: Structure parameter (0 ≤ λ ≤ 1).
        m_y: Structure exponent (typically ~1).
        T: Current temperature [K].
        T_ref: Reference temperature [K].
        E_y: Yield stress activation energy [J/mol].

    Returns:
        Temperature- and structure-dependent yield stress [Pa].
    """
    T_safe = jnp.maximum(T, 1.0)
    T_ref_safe = jnp.maximum(T_ref, 1.0)

    # Structure contribution
    lam_safe = jnp.maximum(lam, 1e-10)
    structure_factor = jnp.power(lam_safe, m_y)

    # Temperature contribution
    exponent = (E_y / R_GAS) * (1.0 / T_safe - 1.0 / T_ref_safe)
    exponent_clipped = jnp.clip(exponent, -50.0, 50.0)
    temp_factor = jnp.exp(exponent_clipped)

    return sigma_y0 * structure_factor * temp_factor


@jax.jit
def temperature_evolution_rate(
    T: jnp.ndarray,
    sigma: jnp.ndarray,
    gamma_dot_p: jnp.ndarray,
    T_env: float,
    rho_cp: float,
    chi: float,
    h: float,
) -> jnp.ndarray:
    """Compute rate of temperature change from viscous dissipation and cooling.

    dT/dt = (χ·σ·γ̇ᵖ - h·(T - T_env)) / (ρ·c_p)

    The temperature evolves due to:
    - Viscous heating: χ·σ·γ̇ᵖ (Taylor-Quinney coefficient χ ≈ 0.9)
    - Convective cooling: h·(T - T_env)

    Args:
        T: Current temperature [K].
        sigma: Current stress magnitude [Pa].
        gamma_dot_p: Plastic shear rate magnitude [1/s].
        T_env: Environmental temperature [K].
        rho_cp: Volumetric heat capacity [J/(m³·K)].
        chi: Taylor-Quinney coefficient (fraction of plastic work → heat).
        h: Heat transfer coefficient [W/(m²·K)] / characteristic length [m].
            For bulk: h ~ k/L² where k is thermal conductivity, L is length.

    Returns:
        Rate of temperature change dT/dt [K/s].
    """
    # Viscous heating (plastic work → heat)
    heating = chi * jnp.abs(sigma) * jnp.abs(gamma_dot_p)

    # Convective cooling
    cooling = h * (T - T_env)

    # Net rate
    rho_cp_safe = jnp.maximum(rho_cp, 1e-6)
    dT_dt = (heating - cooling) / rho_cp_safe

    return dT_dt


@jax.jit
def steady_state_temperature(
    sigma: jnp.ndarray,
    gamma_dot_p: jnp.ndarray,
    T_env: float,
    chi: float,
    h: float,
) -> jnp.ndarray:
    """Compute steady-state temperature from heat balance.

    At steady state: χ·σ·γ̇ᵖ = h·(T_ss - T_env)
    Therefore: T_ss = T_env + χ·σ·γ̇ᵖ / h

    Args:
        sigma: Stress magnitude [Pa].
        gamma_dot_p: Plastic shear rate magnitude [1/s].
        T_env: Environmental temperature [K].
        chi: Taylor-Quinney coefficient.
        h: Heat transfer coefficient.

    Returns:
        Steady-state temperature [K].
    """
    h_safe = jnp.maximum(h, 1e-12)
    T_rise = chi * jnp.abs(sigma) * jnp.abs(gamma_dot_p) / h_safe
    return T_env + T_rise


@jax.jit
def update_thermal_parameters(
    params: dict,
    T: jnp.ndarray,
) -> dict:
    """Update all temperature-dependent parameters.

    This is a convenience function that computes temperature-dependent
    viscosity, modulus, and yield stress from a parameter dictionary.

    Args:
        params: Parameter dictionary containing:
            - eta: Reference viscosity [Pa·s]
            - G: Reference modulus [Pa]
            - sigma_y0: Base yield stress [Pa]
            - T_ref: Reference temperature [K]
            - E_a: Viscosity activation energy [J/mol]
            - E_y: Yield stress activation energy [J/mol]
            - m_y: Structure exponent for yield stress
            And optionally:
            - lam: Current structure parameter
        T: Current temperature [K].

    Returns:
        Updated parameter dictionary with temperature-corrected values:
            - eta_T: Temperature-dependent viscosity
            - G_T: Temperature-dependent modulus (if E_G provided)
            - sigma_y_T: Temperature-dependent yield stress base
    """
    T_ref = params.get("T_ref", 298.15)
    E_a = params.get("E_a", 0.0)
    E_y = params.get("E_y", 0.0)
    E_G = params.get("E_G", 0.0)

    # Temperature-dependent viscosity
    eta_ref = params.get("eta", 1e6)
    eta_T = arrhenius_viscosity(eta_ref, T, T_ref, E_a)

    # Temperature-dependent modulus (optional)
    G_ref = params.get("G", 1e3)
    G_T = arrhenius_modulus(G_ref, T, T_ref, E_G) if E_G != 0.0 else G_ref

    # Temperature-dependent yield stress base
    sigma_y0 = params.get("sigma_y0", 10.0)
    m_y = params.get("m_y", 1.0)
    lam = params.get("lam", 1.0)
    sigma_y_T = thermal_yield_stress(sigma_y0, lam, m_y, T, T_ref, E_y)

    return {
        **params,
        "eta_T": eta_T,
        "G_T": G_T,
        "sigma_y_T": sigma_y_T,
    }


@jax.jit
def compute_adiabatic_temperature_rise(
    gamma_total: jnp.ndarray,
    sigma_avg: jnp.ndarray,
    rho_cp: float,
    chi: float,
) -> jnp.ndarray:
    """Compute adiabatic temperature rise from total deformation work.

    ΔT_adiabatic = χ·∫σ·dγ / (ρ·c_p) ≈ χ·σ_avg·γ_total / (ρ·c_p)

    This is an upper bound on temperature rise (no cooling).

    Args:
        gamma_total: Total accumulated strain.
        sigma_avg: Average stress during deformation [Pa].
        rho_cp: Volumetric heat capacity [J/(m³·K)].
        chi: Taylor-Quinney coefficient.

    Returns:
        Adiabatic temperature rise [K].
    """
    rho_cp_safe = jnp.maximum(rho_cp, 1e-6)
    return chi * jnp.abs(sigma_avg) * jnp.abs(gamma_total) / rho_cp_safe


# Default thermal parameters for typical polymer melts
DEFAULT_THERMAL_PARAMS = {
    "T_ref": 298.15,  # Room temperature [K]
    "E_a": 5e4,  # Viscosity activation energy [J/mol] (typical for polymers)
    "E_y": 3e4,  # Yield stress activation energy [J/mol]
    "E_G": 0.0,  # Modulus activation energy [J/mol] (often negligible)
    "m_y": 1.0,  # Structure exponent
    "rho_cp": 4e6,  # Volumetric heat capacity [J/(m³·K)] (water-like)
    "chi": 0.9,  # Taylor-Quinney coefficient
    "h": 100.0,  # Heat transfer coefficient [W/(m²·K)]
    "T_env": 298.15,  # Environmental temperature [K]
}
