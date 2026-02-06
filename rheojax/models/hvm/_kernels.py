"""JAX-accelerated physics kernels for HVM (Hybrid Vitrimer Model).

This module provides JIT-compiled functions for:
1. TST (Transition State Theory) bond exchange rate functions
2. Stress functions for 3-subnetwork architecture (P + E + D)
3. Evolution ODEs for exchangeable network with natural-state tensor
4. Analytical solutions for linear viscoelastic regime (constant k_BER)
5. Damage evolution with cooperative shielding

The HVM extends VLB transient network theory to vitrimers — polymers with
permanent covalent crosslinks (P), associative exchangeable bonds (E),
and optionally dissociative reversible bonds (D).

Key Physics
-----------
The vitrimer hallmark is the **evolving natural-state tensor** mu^E_nat.
Unlike standard transient networks where broken bonds reform at equilibrium
(mu -> I), exchangeable bonds undergo topology rearrangement so the
stress-free reference state evolves toward the deformed state:

    dmu^E/dt = L.mu^E + mu^E.L^T + k_BER*(mu^E_nat - mu^E)
    dmu^E_nat/dt = k_BER*(mu^E - mu^E_nat)   [vitrimer hallmark]

Stress depends on the *difference*: sigma_E = G_E*(mu^E - mu^E_nat).
Both tensors relax toward each other at rate k_BER, so the difference
relaxes at rate 2*k_BER, giving tau_E_eff = 1/(2*k_BER).

TST Kinetics
-------------
Bond exchange rate depends on mechanical state via Transition State Theory:

    k_BER = nu_0 * exp(-E_a/(R*T)) * cosh(V_act * sigma_VM / (R*T))

where sigma_VM is the von Mises stress on the E-network.

State Vector Convention (simple shear, 11 components)
-----------------------------------------------------
    [mu_E_xx, mu_E_yy, mu_E_xy,           # E-network distribution (3)
     mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy, # E-network natural state (3)
     mu_D_xx, mu_D_yy, mu_D_xy,           # D-network distribution (3)
     gamma,                                 # accumulated strain (1)
     D]                                     # damage variable (1)

References
----------
- Vernerey, Long, & Brighenti (2017). JMPS 107, 1-20.
- Meng, Simon, Niu, McKenna, & Hallinan (2019). Macromolecules 52, 8.
- Stukalin et al. (2013). Macromolecules 46, 7525-7541.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

# Gas constant (J/(mol*K))
_R_GAS = 8.314462618


# =============================================================================
# TST Bond Exchange Rate Functions
# =============================================================================


@jax.jit
def hvm_ber_rate_constant(nu_0: float, E_a: float, T: float) -> float:
    """Thermal BER rate at zero stress (Arrhenius).

    k_BER_0 = nu_0 * exp(-E_a / (R*T))

    Parameters
    ----------
    nu_0 : float
        Attempt frequency (1/s)
    E_a : float
        Activation energy (J/mol)
    T : float
        Temperature (K)

    Returns
    -------
    float
        Zero-stress BER rate (1/s)
    """
    return nu_0 * jnp.exp(-E_a / (_R_GAS * T))


@jax.jit
def hvm_ber_rate_stress(
    sigma_E_xx: float,
    sigma_E_yy: float,
    sigma_E_xy: float,
    nu_0: float,
    E_a: float,
    V_act: float,
    T: float,
) -> float:
    """BER rate with von Mises stress coupling (TST).

    k_BER = nu_0 * exp(-E_a/(R*T)) * cosh(V_act * sigma_VM / (R*T))

    where sigma_VM = sqrt(sigma_xx^2 + sigma_yy^2 - sigma_xx*sigma_yy + 3*sigma_xy^2)
    is the 2D von Mises equivalent stress on the E-network.

    Parameters
    ----------
    sigma_E_xx, sigma_E_yy, sigma_E_xy : float
        E-network stress components (Pa)
    nu_0 : float
        Attempt frequency (1/s)
    E_a : float
        Activation energy (J/mol)
    V_act : float
        Activation volume (m^3/mol)
    T : float
        Temperature (K)

    Returns
    -------
    float
        Stress-enhanced BER rate (1/s)
    """
    # 2D von Mises stress
    sigma_vm = jnp.sqrt(
        jnp.maximum(
            sigma_E_xx**2 + sigma_E_yy**2
            - sigma_E_xx * sigma_E_yy
            + 3.0 * sigma_E_xy**2,
            0.0,
        )
    )
    RT = _R_GAS * T
    k0 = nu_0 * jnp.exp(-E_a / RT)
    # cosh coupling: symmetric acceleration under tension/compression
    return k0 * jnp.cosh(V_act * sigma_vm / RT)


@jax.jit
def hvm_ber_rate_stretch(
    mu_E_xx: float,
    mu_E_yy: float,
    mu_E_nat_xx: float,
    mu_E_nat_yy: float,
    G_E: float,
    nu_0: float,
    E_a: float,
    V_act: float,
    T: float,
) -> float:
    """BER rate with chain stretch coupling (TST).

    k_BER = nu_0 * exp(-E_a/(R*T)) * cosh(V_act * G_E * delta_stretch / (R*T))

    where delta_stretch = sqrt(tr(mu^E - mu^E_nat) / 3) quantifies
    how far the chains are from their natural configuration.

    Parameters
    ----------
    mu_E_xx, mu_E_yy : float
        E-network distribution tensor diagonal components
    mu_E_nat_xx, mu_E_nat_yy : float
        E-network natural state diagonal components
    G_E : float
        Exchangeable network modulus (Pa)
    nu_0 : float
        Attempt frequency (1/s)
    E_a : float
        Activation energy (J/mol)
    V_act : float
        Activation volume (m^3/mol)
    T : float
        Temperature (K)

    Returns
    -------
    float
        Stretch-enhanced BER rate (1/s)
    """
    # mu_zz = mu_nat_zz for simple shear (both equal to their respective values)
    # In simple shear, mu_E_zz ≈ mu_E_nat_zz, so we use the 2D trace difference
    delta_trace = (mu_E_xx - mu_E_nat_xx) + (mu_E_yy - mu_E_nat_yy)
    delta_stretch = jnp.sqrt(jnp.maximum(jnp.abs(delta_trace) / 2.0, 0.0))

    RT = _R_GAS * T
    k0 = nu_0 * jnp.exp(-E_a / RT)
    return k0 * jnp.cosh(V_act * G_E * delta_stretch / RT)


# =============================================================================
# Stress Functions (Simple Shear)
# =============================================================================


@jax.jit
def hvm_permanent_stress_shear(
    gamma: float, G_P: float, D: float
) -> float:
    """Permanent network (P) shear stress with damage.

    sigma_P = (1 - D) * G_P * gamma

    Parameters
    ----------
    gamma : float
        Accumulated shear strain
    G_P : float
        Permanent network modulus (Pa)
    D : float
        Damage variable [0, 1]

    Returns
    -------
    float
        Permanent network shear stress (Pa)
    """
    return (1.0 - D) * G_P * gamma


@jax.jit
def hvm_exchangeable_stress(
    mu_E_xy: float, mu_E_nat_xy: float, G_E: float
) -> float:
    """Exchangeable network (E) shear stress.

    sigma_E = G_E * (mu^E_xy - mu^E_nat_xy)

    Stress depends on difference between current and natural state.
    At equilibrium (mu^E = mu^E_nat): sigma_E = 0.

    Parameters
    ----------
    mu_E_xy : float
        E-network off-diagonal distribution component
    mu_E_nat_xy : float
        E-network off-diagonal natural state component
    G_E : float
        Exchangeable network modulus (Pa)

    Returns
    -------
    float
        Exchangeable network shear stress (Pa)
    """
    return G_E * (mu_E_xy - mu_E_nat_xy)


@jax.jit
def hvm_dissociative_stress(mu_D_xy: float, G_D: float) -> float:
    """Dissociative network (D) shear stress.

    sigma_D = G_D * mu^D_xy

    Standard VLB form — stress relative to fixed equilibrium (I).

    Parameters
    ----------
    mu_D_xy : float
        D-network off-diagonal distribution component
    G_D : float
        Dissociative network modulus (Pa)

    Returns
    -------
    float
        Dissociative network shear stress (Pa)
    """
    return G_D * mu_D_xy


@jax.jit
def hvm_total_stress_shear(
    gamma: float,
    mu_E_xy: float,
    mu_E_nat_xy: float,
    mu_D_xy: float,
    G_P: float,
    G_E: float,
    G_D: float,
    D: float,
) -> float:
    """Total shear stress from all three subnetworks.

    sigma = sigma_P + sigma_E + sigma_D
          = (1-D)*G_P*gamma + G_E*(mu^E_xy - mu^E_nat_xy) + G_D*mu^D_xy

    Parameters
    ----------
    gamma : float
        Accumulated shear strain
    mu_E_xy : float
        E-network off-diagonal distribution component
    mu_E_nat_xy : float
        E-network off-diagonal natural state component
    mu_D_xy : float
        D-network off-diagonal distribution component
    G_P, G_E, G_D : float
        Subnetwork moduli (Pa)
    D : float
        Damage variable [0, 1]

    Returns
    -------
    float
        Total shear stress (Pa)
    """
    sigma_P = (1.0 - D) * G_P * gamma
    sigma_E = G_E * (mu_E_xy - mu_E_nat_xy)
    sigma_D = G_D * mu_D_xy
    return sigma_P + sigma_E + sigma_D


@jax.jit
def hvm_normal_stress_1(
    mu_E_xx: float,
    mu_E_yy: float,
    mu_E_nat_xx: float,
    mu_E_nat_yy: float,
    mu_D_xx: float,
    mu_D_yy: float,
    G_P: float,
    G_E: float,
    G_D: float,
) -> float:
    """First normal stress difference N1 from E and D networks.

    N1 = G_E*[(mu^E_xx - mu^E_nat_xx) - (mu^E_yy - mu^E_nat_yy)]
         + G_D*(mu^D_xx - mu^D_yy)

    Note: P-network contributes no normal stress in simple shear
    (neo-Hookean with gamma only in xy-component).

    Parameters
    ----------
    mu_E_xx, mu_E_yy : float
        E-network diagonal distribution components
    mu_E_nat_xx, mu_E_nat_yy : float
        E-network diagonal natural state components
    mu_D_xx, mu_D_yy : float
        D-network diagonal distribution components
    G_P : float
        Permanent network modulus (unused, kept for interface consistency)
    G_E, G_D : float
        Exchangeable and dissociative network moduli (Pa)

    Returns
    -------
    float
        First normal stress difference N1 (Pa)
    """
    N1_E = G_E * ((mu_E_xx - mu_E_nat_xx) - (mu_E_yy - mu_E_nat_yy))
    N1_D = G_D * (mu_D_xx - mu_D_yy)
    return N1_E + N1_D


# =============================================================================
# Evolution ODEs (Simple Shear)
# =============================================================================


@jax.jit
def hvm_exchangeable_rhs_shear(
    mu_E_xx: float,
    mu_E_yy: float,
    mu_E_xy: float,
    mu_E_nat_xx: float,
    mu_E_nat_yy: float,
    mu_E_nat_xy: float,
    gamma_dot: float,
    k_BER: float,
) -> tuple[float, float, float, float, float, float]:
    """ODE RHS for exchangeable network distribution + natural state.

    E-network distribution (upper-convected derivative + BER relaxation):
        dmu^E_xx/dt = 2*gamma_dot*mu^E_xy + k_BER*(mu^E_nat_xx - mu^E_xx)
        dmu^E_yy/dt = k_BER*(mu^E_nat_yy - mu^E_yy)
        dmu^E_xy/dt = gamma_dot*mu^E_yy + k_BER*(mu^E_nat_xy - mu^E_xy)

    Natural state evolution (vitrimer hallmark):
        dmu^E_nat_xx/dt = k_BER*(mu^E_xx - mu^E_nat_xx)
        dmu^E_nat_yy/dt = k_BER*(mu^E_yy - mu^E_nat_yy)
        dmu^E_nat_xy/dt = k_BER*(mu^E_xy - mu^E_nat_xy)

    The natural state evolves toward the current state at rate k_BER.
    The distribution relaxes toward the natural state at rate k_BER.
    Their *difference* (which determines stress) decays at rate 2*k_BER.

    Parameters
    ----------
    mu_E_xx, mu_E_yy, mu_E_xy : float
        E-network distribution tensor components
    mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy : float
        E-network natural state tensor components
    gamma_dot : float
        Shear rate (1/s)
    k_BER : float
        Bond exchange rate (1/s)

    Returns
    -------
    tuple of 6 floats
        (dmu_E_xx, dmu_E_yy, dmu_E_xy,
         dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy)
    """
    # E-network distribution evolution
    dmu_E_xx = 2.0 * gamma_dot * mu_E_xy + k_BER * (mu_E_nat_xx - mu_E_xx)
    dmu_E_yy = k_BER * (mu_E_nat_yy - mu_E_yy)
    dmu_E_xy = gamma_dot * mu_E_yy + k_BER * (mu_E_nat_xy - mu_E_xy)

    # Natural state evolution (vitrimer hallmark)
    dmu_E_nat_xx = k_BER * (mu_E_xx - mu_E_nat_xx)
    dmu_E_nat_yy = k_BER * (mu_E_yy - mu_E_nat_yy)
    dmu_E_nat_xy = k_BER * (mu_E_xy - mu_E_nat_xy)

    return dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy


@jax.jit
def hvm_damage_rhs(
    mu_E_xx: float,
    mu_E_yy: float,
    mu_E_nat_xx: float,
    mu_E_nat_yy: float,
    mu_D_xx: float,
    mu_D_yy: float,
    D: float,
    G_P: float,
    G_E: float,
    G_D: float,
    Gamma_0: float,
    lambda_crit: float,
) -> float:
    """Damage evolution with cooperative shielding.

    dD/dt = Gamma_0 * (1 - D) * max(0, lambda_eff - lambda_crit)

    where lambda_eff is the effective network stretch, computed from
    the weighted average of E and D network stretches.

    Parameters
    ----------
    mu_E_xx, mu_E_yy : float
        E-network diagonal distribution components
    mu_E_nat_xx, mu_E_nat_yy : float
        E-network diagonal natural state components
    mu_D_xx, mu_D_yy : float
        D-network diagonal distribution components
    D : float
        Current damage variable [0, 1]
    G_P, G_E, G_D : float
        Subnetwork moduli (Pa)
    Gamma_0 : float
        Damage rate coefficient (1/s)
    lambda_crit : float
        Critical stretch for damage onset

    Returns
    -------
    float
        dD/dt (1/s)
    """
    # Effective stretch from E-network (relative to natural state)
    trace_E_diff = (mu_E_xx - mu_E_nat_xx) + (mu_E_yy - mu_E_nat_yy)
    # D-network stretch (relative to equilibrium I)
    trace_D = mu_D_xx + mu_D_yy - 2.0

    # Weighted effective stretch
    G_tot_transient = jnp.maximum(G_E + G_D, 1e-30)
    trace_eff = (G_E * trace_E_diff + G_D * trace_D) / G_tot_transient
    lambda_eff = jnp.sqrt(jnp.maximum(1.0 + trace_eff / 2.0, 0.0))

    # Damage evolution with cooperative shielding
    driving = jnp.maximum(lambda_eff - lambda_crit, 0.0)
    dD = Gamma_0 * (1.0 - D) * driving

    return dD


# =============================================================================
# Analytical Solutions (constant k_BER, linear regime)
# =============================================================================


@jax.jit
def hvm_saos_moduli(
    omega: float,
    G_P: float,
    G_E: float,
    G_D: float,
    k_BER_0: float,
    k_d_D: float,
) -> tuple[float, float]:
    """SAOS storage and loss moduli for full HVM.

    Three contributions:
    1. P-network: G_P (elastic plateau, no loss)
    2. E-network: Maxwell mode with tau_E_eff = 1/(2*k_BER_0)
       - Factor-of-2: both mu^E and mu^E_nat relax, so stress
         (proportional to their difference) decays at 2*k_BER_0
    3. D-network: Maxwell mode with tau_D = 1/k_d_D

    G'(omega) = G_P + G_E*wt_E^2/(1+wt_E^2) + G_D*wt_D^2/(1+wt_D^2)
    G''(omega) = G_E*wt_E/(1+wt_E^2) + G_D*wt_D/(1+wt_D^2)

    where wt_E = omega * tau_E_eff, wt_D = omega * tau_D.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    G_P : float
        Permanent network modulus (Pa)
    G_E : float
        Exchangeable network modulus (Pa)
    G_D : float
        Dissociative network modulus (Pa)
    k_BER_0 : float
        Zero-stress BER rate (1/s)
    k_d_D : float
        Dissociative rate (1/s)

    Returns
    -------
    tuple of (float, float)
        (G', G'') storage and loss moduli (Pa)
    """
    # E-network: effective relaxation time with factor-of-2
    tau_E_eff = 1.0 / (2.0 * jnp.maximum(k_BER_0, 1e-30))
    wt_E = omega * tau_E_eff
    wt_E2 = wt_E * wt_E
    denom_E = 1.0 + wt_E2

    # D-network: standard Maxwell
    tau_D = 1.0 / jnp.maximum(k_d_D, 1e-30)
    wt_D = omega * tau_D
    wt_D2 = wt_D * wt_D
    denom_D = 1.0 + wt_D2

    G_prime = G_P + G_E * wt_E2 / denom_E + G_D * wt_D2 / denom_D
    G_double_prime = G_E * wt_E / denom_E + G_D * wt_D / denom_D

    return G_prime, G_double_prime


def _hvm_saos_G_prime(
    omega: float, G_P: float, G_E: float, G_D: float,
    k_BER_0: float, k_d_D: float,
) -> float:
    """Helper returning only G' for vmap."""
    G_p, _ = hvm_saos_moduli(omega, G_P, G_E, G_D, k_BER_0, k_d_D)
    return G_p


def _hvm_saos_G_double_prime(
    omega: float, G_P: float, G_E: float, G_D: float,
    k_BER_0: float, k_d_D: float,
) -> float:
    """Helper returning only G'' for vmap."""
    _, G_pp = hvm_saos_moduli(omega, G_P, G_E, G_D, k_BER_0, k_d_D)
    return G_pp


def hvm_saos_moduli_vec(
    omega_arr: jnp.ndarray,
    G_P: float,
    G_E: float,
    G_D: float,
    k_BER_0: float,
    k_d_D: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized SAOS moduli over frequency array.

    Parameters
    ----------
    omega_arr : jnp.ndarray
        Angular frequency array (rad/s)
    G_P, G_E, G_D : float
        Subnetwork moduli (Pa)
    k_BER_0 : float
        Zero-stress BER rate (1/s)
    k_d_D : float
        Dissociative rate (1/s)

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        (G', G'') arrays (Pa)
    """
    tau_E_eff = 1.0 / (2.0 * jnp.maximum(k_BER_0, 1e-30))
    wt_E = omega_arr * tau_E_eff
    wt_E2 = wt_E * wt_E
    denom_E = 1.0 + wt_E2

    tau_D = 1.0 / jnp.maximum(k_d_D, 1e-30)
    wt_D = omega_arr * tau_D
    wt_D2 = wt_D * wt_D
    denom_D = 1.0 + wt_D2

    G_prime = G_P + G_E * wt_E2 / denom_E + G_D * wt_D2 / denom_D
    G_double_prime = G_E * wt_E / denom_E + G_D * wt_D / denom_D

    return G_prime, G_double_prime


hvm_saos_moduli_vec = jax.jit(hvm_saos_moduli_vec)


@jax.jit
def hvm_relaxation_modulus(
    t: float,
    G_P: float,
    G_E: float,
    G_D: float,
    k_BER_0: float,
    k_d_D: float,
    D: float,
) -> float:
    """Stress relaxation modulus G(t) for HVM (constant k_BER).

    G(t) = (1-D)*G_P + G_E*exp(-2*k_BER_0*t) + G_D*exp(-k_d_D*t)

    Three terms:
    - Permanent plateau (1-D)*G_P (infinite relaxation time)
    - E-network exponential with tau_E_eff = 1/(2*k_BER_0)
    - D-network exponential with tau_D = 1/k_d_D

    Parameters
    ----------
    t : float
        Time after step strain (s)
    G_P, G_E, G_D : float
        Subnetwork moduli (Pa)
    k_BER_0 : float
        Zero-stress BER rate (1/s)
    k_d_D : float
        Dissociative rate (1/s)
    D : float
        Damage variable [0, 1]

    Returns
    -------
    float
        Relaxation modulus G(t) (Pa)
    """
    G_perm = (1.0 - D) * G_P
    G_exch = G_E * jnp.exp(-2.0 * k_BER_0 * t)
    G_diss = G_D * jnp.exp(-k_d_D * t)
    return G_perm + G_exch + G_diss


hvm_relaxation_modulus_vec = jax.jit(
    jax.vmap(hvm_relaxation_modulus, in_axes=(0, None, None, None, None, None, None))
)


@jax.jit
def hvm_startup_stress_linear(
    t: float,
    gamma_dot: float,
    G_P: float,
    G_E: float,
    G_D: float,
    k_BER_0: float,
    k_d_D: float,
) -> float:
    """Startup stress in linear regime (constant k_BER, no TST feedback).

    3-term analytical solution:
    sigma(t) = G_P*gamma_dot*t
             + G_E*gamma_dot*tau_E_eff*(1 - exp(-t/tau_E_eff))
             + G_D*gamma_dot*tau_D*(1 - exp(-t/tau_D))

    where tau_E_eff = 1/(2*k_BER_0), tau_D = 1/k_d_D.

    Parameters
    ----------
    t : float
        Time (s)
    gamma_dot : float
        Applied shear rate (1/s)
    G_P, G_E, G_D : float
        Subnetwork moduli (Pa)
    k_BER_0 : float
        Zero-stress BER rate (1/s)
    k_d_D : float
        Dissociative rate (1/s)

    Returns
    -------
    float
        Shear stress at time t (Pa)
    """
    # Permanent: linear growth (neo-Hookean)
    sigma_P = G_P * gamma_dot * t

    # Exchangeable: Maxwell with tau_E_eff = 1/(2*k_BER_0)
    tau_E = 1.0 / (2.0 * jnp.maximum(k_BER_0, 1e-30))
    sigma_E = G_E * gamma_dot * tau_E * (1.0 - jnp.exp(-t / tau_E))

    # Dissociative: standard Maxwell with tau_D = 1/k_d_D
    tau_D = 1.0 / jnp.maximum(k_d_D, 1e-30)
    sigma_D = G_D * gamma_dot * tau_D * (1.0 - jnp.exp(-t / tau_D))

    return sigma_P + sigma_E + sigma_D


hvm_startup_stress_linear_vec = jax.jit(
    jax.vmap(
        hvm_startup_stress_linear,
        in_axes=(0, None, None, None, None, None, None),
    )
)


@jax.jit
def hvm_steady_shear_stress(
    gamma_dot: float,
    G_P: float,
    G_D: float,
    k_d_D: float,
) -> float:
    """Steady-state shear stress (E-network contributes zero).

    At steady state, mu^E -> mu^E_nat (bond exchange fully relaxes
    the exchangeable network), so sigma_E -> 0. Only P and D contribute:

    sigma_ss = sigma_D = G_D * gamma_dot / k_d_D

    Note: sigma_P = G_P * gamma grows unbounded at steady state under
    constant shear rate. For flow curve prediction, we report only the
    viscous contributions. The P-network provides a static elastic
    contribution that is relevant for startup but not for steady flow.

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
    G_P : float
        Permanent network modulus (Pa, unused for steady viscous stress)
    G_D : float
        Dissociative network modulus (Pa)
    k_d_D : float
        Dissociative rate (1/s)

    Returns
    -------
    float
        Steady-state viscous shear stress (Pa)
    """
    eta_D = G_D / jnp.maximum(k_d_D, 1e-30)
    return eta_D * gamma_dot


hvm_steady_shear_stress_vec = jax.jit(
    jax.vmap(hvm_steady_shear_stress, in_axes=(0, None, None, None))
)


@jax.jit
def hvm_creep_compliance_linear(
    t: float,
    G_P: float,
    G_E: float,
    G_D: float,
    k_BER_0: float,
    k_d_D: float,
) -> float:
    """Creep compliance J(t) for HVM in linear regime.

    With a permanent network (G_P > 0), compliance approaches 1/G_P
    at long times. Two retardation modes from E and D networks.

    For the case G_P > 0 (SLS-like per mode):
    J(t) ≈ 1/G_tot + contribution from E-mode retardation
            + contribution from D-mode retardation

    where G_tot = G_P + G_E + G_D and retardation times depend on
    the coupling between networks.

    Simplified form valid when G_P >> G_E, G_D:
    J(t) ≈ 1/G_tot + (G_E/(G_P*G_tot)) * (1 - exp(-t/tau_ret_E))
            + (G_D/(G_P*G_tot)) * (1 - exp(-t/tau_ret_D))

    Parameters
    ----------
    t : float
        Time (s)
    G_P, G_E, G_D : float
        Subnetwork moduli (Pa)
    k_BER_0 : float
        Zero-stress BER rate (1/s)
    k_d_D : float
        Dissociative rate (1/s)

    Returns
    -------
    float
        Creep compliance J(t) (1/Pa)
    """
    G_tot = G_P + G_E + G_D
    G_P_safe = jnp.maximum(G_P, 1e-30)

    # Instantaneous elastic compliance
    J_inst = 1.0 / G_tot

    # E-network retardation
    tau_E_eff = 1.0 / (2.0 * jnp.maximum(k_BER_0, 1e-30))
    tau_ret_E = G_tot / (G_P_safe * 2.0 * jnp.maximum(k_BER_0, 1e-30))
    J_E = G_E / (G_P_safe * G_tot) * (1.0 - jnp.exp(-t / tau_ret_E))

    # D-network retardation
    tau_ret_D = G_tot / (G_P_safe * jnp.maximum(k_d_D, 1e-30))
    J_D = G_D / (G_P_safe * G_tot) * (1.0 - jnp.exp(-t / tau_ret_D))

    return J_inst + J_E + J_D


hvm_creep_compliance_linear_vec = jax.jit(
    jax.vmap(hvm_creep_compliance_linear, in_axes=(0, None, None, None, None, None))
)
