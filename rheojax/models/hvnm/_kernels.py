"""JAX-accelerated physics kernels for HVNM (Hybrid Vitrimer Nanocomposite Model).

This module provides JIT-compiled functions for:
1. NP geometry: Guth-Gold amplification, interphase fraction, effective phi
2. Dual TST bond exchange rates (matrix and interphase, independent)
3. Stress functions for 4-subnetwork architecture (P + E + D + I)
4. Evolution ODEs for interphase network with strain amplification
5. Interfacial damage with self-healing
6. Analytical solutions for linear viscoelastic regime (constant rates)

The HVNM extends the HVM by adding an interphase (I) subnetwork around
nanoparticle surfaces with distinct kinetics and strain amplification.

Key Physics
-----------
- Dual TST: k_BER^mat and k_BER^int are independent (Li et al. 2024)
- Strain amplification: Guth-Gold X(phi) = 1 + 2.5*phi + 14.1*phi^2
- Interphase affine term uses amplified strain rate: X_I * gamma_dot
- Factor-of-2: tau_E_eff = 1/(2*k_BER^mat_0), tau_I_eff = 1/(2*k_BER^int_0)
- sigma_E -> 0, sigma_I -> 0 at steady state (both have evolving natural states)

State Vector Convention (simple shear, 17 components without D_int)
-------------------------------------------------------------------
    y[0:3]   = [mu_E_xx, mu_E_yy, mu_E_xy]           # E-network distribution (3)
    y[3:6]   = [mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy] # E-network natural state (3)
    y[6:9]   = [mu_D_xx, mu_D_yy, mu_D_xy]           # D-network distribution (3)
    y[9]     = gamma                                   # accumulated strain (1)
    y[10]    = D                                       # matrix damage (1)
    y[11:14] = [mu_I_xx, mu_I_yy, mu_I_xy]           # I-network distribution (3)
    y[14:17] = [mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy] # I-network natural state (3)
    y[17]    = D_int                                   # interfacial damage (1, optional)

References
----------
- Vernerey, Long, & Brighenti (2017). JMPS 107, 1-20.
- Li, Zhao, Duan, Zhang, Liu (2024). Langmuir 40, 7550-7560.
- Karim, Vernerey, Sain (2025). Macromolecules 58, 4899-4912.
- Papon, Montes et al. (2012). Soft Matter 8, 4090-4096.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

# Constants
_R_GAS = 8.314462618  # J/(mol*K)
_K_B = 1.380649e-23  # J/K (Boltzmann constant)


# =============================================================================
# NP Geometry Functions
# =============================================================================


@jax.jit
def hvnm_guth_gold(phi: float) -> float:
    """Guth-Gold strain amplification factor for spherical particles.

    X(phi) = 1 + 2.5*phi + 14.1*phi^2

    Parameters
    ----------
    phi : float
        NP volume fraction [0, 0.5]

    Returns
    -------
    float
        Strain amplification factor (>= 1)
    """
    return 1.0 + 2.5 * phi + 14.1 * phi**2


@jax.jit
def hvnm_effective_phi(phi: float, R_NP: float, delta_g: float) -> float:
    """Effective NP volume fraction including glassy layer.

    phi_eff = phi * (1 + delta_g / R_NP)^3

    The glassy layer (~1-2 nm) is effectively rigid and contributes
    to the hard-sphere volume fraction.

    Parameters
    ----------
    phi : float
        NP volume fraction
    R_NP : float
        NP radius (m)
    delta_g : float
        Glassy layer thickness (m), default ~1e-9

    Returns
    -------
    float
        Effective NP volume fraction
    """
    R_safe = jnp.maximum(R_NP, 1e-15)
    return phi * (1.0 + delta_g / R_safe) ** 3


@jax.jit
def hvnm_interphase_fraction(
    phi: float, R_NP: float, delta_g: float, delta_m: float
) -> float:
    """Mobile interphase volume fraction from NP geometry.

    phi_I = phi_eff * [(1 + delta_m / (R_NP + delta_g))^3 - 1]

    Parameters
    ----------
    phi : float
        NP volume fraction
    R_NP : float
        NP radius (m)
    delta_g : float
        Glassy layer thickness (m)
    delta_m : float
        Mobile interphase thickness (m)

    Returns
    -------
    float
        Interphase volume fraction
    """
    phi_eff = hvnm_effective_phi(phi, R_NP, delta_g)
    R_eff = jnp.maximum(R_NP + delta_g, 1e-15)
    return phi_eff * ((1.0 + delta_m / R_eff) ** 3 - 1.0)


@jax.jit
def hvnm_interphase_modulus(
    G_E: float, beta_I: float, phi_I: float
) -> float:
    """Effective interphase modulus.

    G_I_eff = beta_I * G_E * phi_I

    Parameters
    ----------
    G_E : float
        Exchangeable network modulus (Pa)
    beta_I : float
        Interphase reinforcement ratio G_I / G_E
    phi_I : float
        Interphase volume fraction

    Returns
    -------
    float
        Effective interphase modulus (Pa)
    """
    return beta_I * G_E * phi_I


# =============================================================================
# Dual TST Rate Functions
# =============================================================================


@jax.jit
def hvnm_ber_rate_matrix_stress(
    sigma_E_xx: float,
    sigma_E_yy: float,
    sigma_E_xy: float,
    nu_0: float,
    E_a: float,
    V_act: float,
    T: float,
) -> float:
    """Matrix BER rate with von Mises stress coupling (TST).

    k_BER^mat = nu_0 * exp(-E_a/(R*T)) * cosh(V_act * sigma_VM / (R*T))

    Parameters
    ----------
    sigma_E_xx, sigma_E_yy, sigma_E_xy : float
        E-network stress components (Pa)
    nu_0 : float
        Matrix attempt frequency (1/s)
    E_a : float
        Matrix activation energy (J/mol)
    V_act : float
        Matrix activation volume (m^3/mol)
    T : float
        Temperature (K)

    Returns
    -------
    float
        Stress-enhanced matrix BER rate (1/s)
    """
    # Import HVM kernel to avoid duplication
    from rheojax.models.hvm._kernels import hvm_ber_rate_stress
    return hvm_ber_rate_stress(
        sigma_E_xx, sigma_E_yy, sigma_E_xy, nu_0, E_a, V_act, T
    )


@jax.jit
def hvnm_ber_rate_interphase_stress(
    sigma_I_xx: float,
    sigma_I_yy: float,
    sigma_I_xy: float,
    nu_0_int: float,
    E_a_int: float,
    V_act_int: float,
    T: float,
) -> float:
    """Interfacial BER rate with von Mises stress coupling (TST).

    k_BER^int = nu_0_int * exp(-E_a_int/(R*T)) * cosh(V_act_int * sigma_VM_I / (R*T))

    Parameters
    ----------
    sigma_I_xx, sigma_I_yy, sigma_I_xy : float
        I-network stress components (Pa)
    nu_0_int : float
        Interfacial attempt frequency (1/s)
    E_a_int : float
        Interfacial activation energy (J/mol)
    V_act_int : float
        Interfacial activation volume (m^3/mol)
    T : float
        Temperature (K)

    Returns
    -------
    float
        Stress-enhanced interfacial BER rate (1/s)
    """
    sigma_vm = jnp.sqrt(
        jnp.maximum(
            sigma_I_xx**2 + sigma_I_yy**2
            - sigma_I_xx * sigma_I_yy
            + 3.0 * sigma_I_xy**2,
            0.0,
        )
    )
    RT = _R_GAS * T
    k0 = nu_0_int * jnp.exp(-E_a_int / RT)
    return k0 * jnp.cosh(V_act_int * sigma_vm / RT)


@jax.jit
def hvnm_ber_rate_matrix_stretch(
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
    """Matrix BER rate with chain stretch coupling (TST).

    Delegates to HVM kernel.
    """
    from rheojax.models.hvm._kernels import hvm_ber_rate_stretch
    return hvm_ber_rate_stretch(
        mu_E_xx, mu_E_yy, mu_E_nat_xx, mu_E_nat_yy,
        G_E, nu_0, E_a, V_act, T,
    )


@jax.jit
def hvnm_ber_rate_interphase_stretch(
    mu_I_xx: float,
    mu_I_yy: float,
    mu_I_nat_xx: float,
    mu_I_nat_yy: float,
    G_I_eff: float,
    nu_0_int: float,
    E_a_int: float,
    V_act_int: float,
    T: float,
) -> float:
    """Interfacial BER rate with chain stretch coupling (TST).

    k_BER^int = nu_0_int * exp(-E_a_int/(R*T)) * cosh(V_act_int * G_I_eff * delta_stretch / (R*T))

    Parameters
    ----------
    mu_I_xx, mu_I_yy : float
        I-network distribution tensor diagonal components
    mu_I_nat_xx, mu_I_nat_yy : float
        I-network natural state diagonal components
    G_I_eff : float
        Effective interphase modulus (Pa)
    nu_0_int, E_a_int, V_act_int, T : float
        Interfacial TST parameters

    Returns
    -------
    float
        Stretch-enhanced interfacial BER rate (1/s)
    """
    delta_trace = (mu_I_xx - mu_I_nat_xx) + (mu_I_yy - mu_I_nat_yy)
    delta_stretch = jnp.sqrt(jnp.maximum(jnp.abs(delta_trace) / 2.0, 0.0))

    RT = _R_GAS * T
    k0 = nu_0_int * jnp.exp(-E_a_int / RT)
    return k0 * jnp.cosh(V_act_int * G_I_eff * delta_stretch / RT)


@jax.jit
def hvnm_ber_rate_constant_matrix(
    nu_0: float, E_a: float, T: float
) -> float:
    """Thermal matrix BER rate at zero stress.

    k_BER^mat_0 = nu_0 * exp(-E_a / (R*T))

    Parameters
    ----------
    nu_0 : float
        Matrix attempt frequency (1/s)
    E_a : float
        Matrix activation energy (J/mol)
    T : float
        Temperature (K)

    Returns
    -------
    float
        Zero-stress matrix BER rate (1/s)
    """
    return nu_0 * jnp.exp(-E_a / (_R_GAS * T))


@jax.jit
def hvnm_ber_rate_constant_interphase(
    nu_0_int: float, E_a_int: float, T: float
) -> float:
    """Thermal interfacial BER rate at zero stress.

    k_BER^int_0 = nu_0_int * exp(-E_a_int / (R*T))

    Parameters
    ----------
    nu_0_int : float
        Interfacial attempt frequency (1/s)
    E_a_int : float
        Interfacial activation energy (J/mol)
    T : float
        Temperature (K)

    Returns
    -------
    float
        Zero-stress interfacial BER rate (1/s)
    """
    return nu_0_int * jnp.exp(-E_a_int / (_R_GAS * T))


# =============================================================================
# Stress Functions (Simple Shear)
# =============================================================================


@jax.jit
def hvnm_permanent_stress_shear(
    gamma: float, G_P: float, X_phi: float, D: float
) -> float:
    """Amplified permanent network (P) shear stress with damage.

    sigma_P = (1 - D) * G_P * X(phi) * gamma

    Parameters
    ----------
    gamma : float
        Accumulated shear strain
    G_P : float
        Permanent network modulus (Pa)
    X_phi : float
        Guth-Gold amplification factor
    D : float
        Matrix damage variable [0, 1]

    Returns
    -------
    float
        Permanent network shear stress (Pa)
    """
    return (1.0 - D) * G_P * X_phi * gamma


@jax.jit
def hvnm_interphase_stress(
    mu_I_xy: float,
    mu_I_nat_xy: float,
    G_I_eff: float,
    X_I: float,
    D_int: float,
) -> float:
    """Interphase network (I) shear stress with damage and amplification.

    sigma_I = (1 - D_int) * G_I_eff * X_I * (mu^I_xy - mu^I_nat_xy)

    Note: X_I appears in the stress because the interphase modulus is
    volume-fraction weighted and amplified by the local strain field.

    Parameters
    ----------
    mu_I_xy : float
        I-network off-diagonal distribution component
    mu_I_nat_xy : float
        I-network off-diagonal natural state component
    G_I_eff : float
        Effective interphase modulus (Pa)
    X_I : float
        Interphase strain amplification factor
    D_int : float
        Interfacial damage variable [0, 1]

    Returns
    -------
    float
        Interphase network shear stress (Pa)
    """
    return (1.0 - D_int) * G_I_eff * X_I * (mu_I_xy - mu_I_nat_xy)


@jax.jit
def hvnm_total_stress_shear(
    gamma: float,
    mu_E_xy: float,
    mu_E_nat_xy: float,
    mu_D_xy: float,
    mu_I_xy: float,
    mu_I_nat_xy: float,
    G_P: float,
    G_E: float,
    G_D: float,
    G_I_eff: float,
    X_phi: float,
    X_I: float,
    D: float,
    D_int: float,
) -> float:
    """Total shear stress from all four subnetworks.

    sigma = (1-D)*G_P*X*gamma + G_E*(mu^E_xy - mu^E_nat_xy)
            + G_D*mu^D_xy + (1-D_int)*G_I_eff*X_I*(mu^I_xy - mu^I_nat_xy)

    Parameters
    ----------
    gamma : float
        Accumulated shear strain
    mu_E_xy, mu_E_nat_xy : float
        E-network distribution and natural state
    mu_D_xy : float
        D-network distribution
    mu_I_xy, mu_I_nat_xy : float
        I-network distribution and natural state
    G_P, G_E, G_D, G_I_eff : float
        Subnetwork moduli (Pa)
    X_phi : float
        Guth-Gold amplification for P-network
    X_I : float
        Strain amplification for I-network
    D : float
        Matrix damage [0, 1]
    D_int : float
        Interfacial damage [0, 1]

    Returns
    -------
    float
        Total shear stress (Pa)
    """
    sigma_P = (1.0 - D) * G_P * X_phi * gamma
    sigma_E = G_E * (mu_E_xy - mu_E_nat_xy)
    sigma_D = G_D * mu_D_xy
    sigma_I = (1.0 - D_int) * G_I_eff * X_I * (mu_I_xy - mu_I_nat_xy)
    return sigma_P + sigma_E + sigma_D + sigma_I


@jax.jit
def hvnm_total_normal_stress_1(
    mu_E_xx: float,
    mu_E_yy: float,
    mu_E_nat_xx: float,
    mu_E_nat_yy: float,
    mu_D_xx: float,
    mu_D_yy: float,
    mu_I_xx: float,
    mu_I_yy: float,
    mu_I_nat_xx: float,
    mu_I_nat_yy: float,
    G_E: float,
    G_D: float,
    G_I_eff: float,
    X_I: float,
    D_int: float,
) -> float:
    """First normal stress difference N1 with interphase contribution.

    N1 = G_E*[(mu^E_xx - mu^E_nat_xx) - (mu^E_yy - mu^E_nat_yy)]
         + G_D*(mu^D_xx - mu^D_yy)
         + (1-D_int)*G_I_eff*X_I*[(mu^I_xx - mu^I_nat_xx) - (mu^I_yy - mu^I_nat_yy)]

    Parameters
    ----------
    [tensor components and moduli]

    Returns
    -------
    float
        First normal stress difference N1 (Pa)
    """
    N1_E = G_E * ((mu_E_xx - mu_E_nat_xx) - (mu_E_yy - mu_E_nat_yy))
    N1_D = G_D * (mu_D_xx - mu_D_yy)
    N1_I = (1.0 - D_int) * G_I_eff * X_I * (
        (mu_I_xx - mu_I_nat_xx) - (mu_I_yy - mu_I_nat_yy)
    )
    return N1_E + N1_D + N1_I


# =============================================================================
# Evolution ODEs (Simple Shear)
# =============================================================================


@jax.jit
def hvnm_interphase_rhs_shear(
    mu_I_xx: float,
    mu_I_yy: float,
    mu_I_xy: float,
    mu_I_nat_xx: float,
    mu_I_nat_yy: float,
    mu_I_nat_xy: float,
    gamma_dot: float,
    X_I: float,
    k_BER_int: float,
) -> tuple[float, float, float, float, float, float]:
    """ODE RHS for interphase network distribution + natural state.

    I-network distribution (upper-convected + BER relaxation + amplification):
        dmu^I_xx/dt = 2*X_I*gamma_dot*mu^I_xy + k_BER^int*(mu^I_nat_xx - mu^I_xx)
        dmu^I_yy/dt = k_BER^int*(mu^I_nat_yy - mu^I_yy)
        dmu^I_xy/dt = X_I*gamma_dot*mu^I_yy + k_BER^int*(mu^I_nat_xy - mu^I_xy)

    Natural state (vitrimer hallmark, no amplification):
        dmu^I_nat_xx/dt = k_BER^int*(mu^I_xx - mu^I_nat_xx)
        dmu^I_nat_yy/dt = k_BER^int*(mu^I_yy - mu^I_nat_yy)
        dmu^I_nat_xy/dt = k_BER^int*(mu^I_xy - mu^I_nat_xy)

    Note: Strain amplification X_I appears in the affine terms (L*mu + mu*L^T)
    but NOT in the BER terms. The BER relaxation is local to the interphase.

    Parameters
    ----------
    mu_I_xx, mu_I_yy, mu_I_xy : float
        I-network distribution tensor components
    mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy : float
        I-network natural state tensor components
    gamma_dot : float
        Macroscopic shear rate (1/s) — amplified by X_I internally
    X_I : float
        Interphase strain amplification factor
    k_BER_int : float
        Interfacial bond exchange rate (1/s)

    Returns
    -------
    tuple of 6 floats
        (dmu_I_xx, dmu_I_yy, dmu_I_xy,
         dmu_I_nat_xx, dmu_I_nat_yy, dmu_I_nat_xy)
    """
    # Amplified strain rate for interphase
    gdot_amp = X_I * gamma_dot

    # I-network distribution evolution
    dmu_I_xx = 2.0 * gdot_amp * mu_I_xy + k_BER_int * (mu_I_nat_xx - mu_I_xx)
    dmu_I_yy = k_BER_int * (mu_I_nat_yy - mu_I_yy)
    dmu_I_xy = gdot_amp * mu_I_yy + k_BER_int * (mu_I_nat_xy - mu_I_xy)

    # Natural state evolution (vitrimer hallmark)
    dmu_I_nat_xx = k_BER_int * (mu_I_xx - mu_I_nat_xx)
    dmu_I_nat_yy = k_BER_int * (mu_I_yy - mu_I_nat_yy)
    dmu_I_nat_xy = k_BER_int * (mu_I_xy - mu_I_nat_xy)

    return dmu_I_xx, dmu_I_yy, dmu_I_xy, dmu_I_nat_xx, dmu_I_nat_yy, dmu_I_nat_xy


@jax.jit
def hvnm_interfacial_damage_rhs(
    mu_I_xx: float,
    mu_I_yy: float,
    D_int: float,
    Gamma_0_int: float,
    lambda_crit_int: float,
    h_0: float,
    E_a_heal: float,
    n_h: float,
    T: float,
) -> float:
    """Interfacial damage evolution with self-healing.

    dD_int/dt = Gamma_0_int * (lambda_chain^I - lambda_crit^int)^+ * (1 - D_int)
                - h_int(T) * D_int^n_h

    where lambda_chain^I = sqrt(tr(mu^I) / 3), h_int = h_0 * exp(-E_a_heal / (R*T))

    Parameters
    ----------
    mu_I_xx, mu_I_yy : float
        I-network distribution diagonal components
    D_int : float
        Current interfacial damage [0, 1]
    Gamma_0_int : float
        Interfacial damage rate (1/s)
    lambda_crit_int : float
        Critical interfacial stretch
    h_0 : float
        Self-healing pre-exponential (1/s)
    E_a_heal : float
        Healing activation energy (J/mol)
    n_h : float
        Healing exponent
    T : float
        Temperature (K)

    Returns
    -------
    float
        dD_int/dt (1/s)
    """
    # mu_I_zz ≈ 1 for simple shear
    tr_mu_I = mu_I_xx + mu_I_yy + 1.0
    lambda_chain = jnp.sqrt(jnp.maximum(tr_mu_I / 3.0, 0.0))

    # Damage creation
    driving = jnp.maximum(lambda_chain - lambda_crit_int, 0.0)
    creation = Gamma_0_int * (1.0 - D_int) * driving

    # Self-healing (TST, active above T_v^int)
    h_int = h_0 * jnp.exp(-E_a_heal / (_R_GAS * T))
    healing = h_int * jnp.power(jnp.maximum(D_int, 0.0), n_h)

    return creation - healing


# =============================================================================
# Analytical Solutions (constant rates, linear regime)
# =============================================================================


@jax.jit
def hvnm_saos_moduli(
    omega: float,
    G_P: float,
    G_E: float,
    G_D: float,
    G_I_eff: float,
    X_phi: float,
    X_I: float,
    k_BER_mat_0: float,
    k_d_D: float,
    k_BER_int_0: float,
    D: float,
    D_int: float,
) -> tuple[float, float]:
    """SAOS storage and loss moduli for full HVNM.

    Four contributions:
    1. P-network: (1-D)*G_P*X(phi) (elastic plateau, no loss)
    2. E-network: Maxwell mode with tau_m/2 = 1/(2*k_BER^mat_0)
    3. D-network: Maxwell mode with tau_D = 1/k_d_D
    4. I-network: Maxwell mode with tau_I/2 = 1/(2*k_BER^int_0)

    G'(omega) = (1-D)*G_P*X
                + G_E*wt_E^2/(1+wt_E^2) + G_D*wt_D^2/(1+wt_D^2)
                + (1-D_int)*G_I_eff*X_I*wt_I^2/(1+wt_I^2)

    G''(omega) = G_E*wt_E/(1+wt_E^2) + G_D*wt_D/(1+wt_D^2)
                 + (1-D_int)*G_I_eff*X_I*wt_I/(1+wt_I^2)

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    G_P, G_E, G_D, G_I_eff : float
        Subnetwork moduli (Pa)
    X_phi : float
        Guth-Gold amplification for P-network
    X_I : float
        Strain amplification for I-network
    k_BER_mat_0 : float
        Zero-stress matrix BER rate (1/s)
    k_d_D : float
        Dissociative rate (1/s)
    k_BER_int_0 : float
        Zero-stress interfacial BER rate (1/s)
    D : float
        Matrix damage [0, 1]
    D_int : float
        Interfacial damage [0, 1]

    Returns
    -------
    tuple of (float, float)
        (G', G'') storage and loss moduli (Pa)
    """
    # E-network: effective relaxation time with factor-of-2
    tau_E = 1.0 / (2.0 * jnp.maximum(k_BER_mat_0, 1e-30))
    wt_E = omega * tau_E
    wt_E2 = wt_E * wt_E
    denom_E = 1.0 + wt_E2

    # D-network: standard Maxwell
    tau_D = 1.0 / jnp.maximum(k_d_D, 1e-30)
    wt_D = omega * tau_D
    wt_D2 = wt_D * wt_D
    denom_D = 1.0 + wt_D2

    # I-network: factor-of-2
    tau_I = 1.0 / (2.0 * jnp.maximum(k_BER_int_0, 1e-30))
    wt_I = omega * tau_I
    wt_I2 = wt_I * wt_I
    denom_I = 1.0 + wt_I2

    G_I_contrib = (1.0 - D_int) * G_I_eff * X_I

    G_prime = (
        (1.0 - D) * G_P * X_phi
        + G_E * wt_E2 / denom_E
        + G_D * wt_D2 / denom_D
        + G_I_contrib * wt_I2 / denom_I
    )
    G_double_prime = (
        G_E * wt_E / denom_E
        + G_D * wt_D / denom_D
        + G_I_contrib * wt_I / denom_I
    )

    return G_prime, G_double_prime


def hvnm_saos_moduli_vec(
    omega_arr: jnp.ndarray,
    G_P: float,
    G_E: float,
    G_D: float,
    G_I_eff: float,
    X_phi: float,
    X_I: float,
    k_BER_mat_0: float,
    k_d_D: float,
    k_BER_int_0: float,
    D: float,
    D_int: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized SAOS moduli over frequency array.

    Uses array broadcasting for performance (no vmap overhead).
    """
    tau_E = 1.0 / (2.0 * jnp.maximum(k_BER_mat_0, 1e-30))
    wt_E = omega_arr * tau_E
    wt_E2 = wt_E * wt_E
    denom_E = 1.0 + wt_E2

    tau_D = 1.0 / jnp.maximum(k_d_D, 1e-30)
    wt_D = omega_arr * tau_D
    wt_D2 = wt_D * wt_D
    denom_D = 1.0 + wt_D2

    tau_I = 1.0 / (2.0 * jnp.maximum(k_BER_int_0, 1e-30))
    wt_I = omega_arr * tau_I
    wt_I2 = wt_I * wt_I
    denom_I = 1.0 + wt_I2

    G_I_contrib = (1.0 - D_int) * G_I_eff * X_I

    G_prime = (
        (1.0 - D) * G_P * X_phi
        + G_E * wt_E2 / denom_E
        + G_D * wt_D2 / denom_D
        + G_I_contrib * wt_I2 / denom_I
    )
    G_double_prime = (
        G_E * wt_E / denom_E
        + G_D * wt_D / denom_D
        + G_I_contrib * wt_I / denom_I
    )

    return G_prime, G_double_prime


hvnm_saos_moduli_vec = jax.jit(hvnm_saos_moduli_vec)


@jax.jit
def hvnm_relaxation_modulus(
    t: float,
    G_P: float,
    G_E: float,
    G_D: float,
    G_I_eff: float,
    X_phi: float,
    X_I: float,
    k_BER_mat_0: float,
    k_d_D: float,
    k_BER_int_0: float,
    D: float,
    D_int: float,
) -> float:
    """Stress relaxation modulus G(t) for HVNM (constant rates).

    G(t) = (1-D)*G_P*X + G_E*exp(-2*k_m*t) + G_D*exp(-k_D*t)
           + (1-D_int)*G_I_eff*X_I*exp(-2*k_I*t)

    Four terms: permanent plateau, E-mode, D-mode, I-mode (NEW).

    Parameters
    ----------
    t : float
        Time after step strain (s)
    [moduli, amplification, rates, damage]

    Returns
    -------
    float
        Relaxation modulus G(t) (Pa)
    """
    G_perm = (1.0 - D) * G_P * X_phi
    G_exch = G_E * jnp.exp(-2.0 * k_BER_mat_0 * t)
    G_diss = G_D * jnp.exp(-k_d_D * t)
    G_inter = (1.0 - D_int) * G_I_eff * X_I * jnp.exp(-2.0 * k_BER_int_0 * t)
    return G_perm + G_exch + G_diss + G_inter


hvnm_relaxation_modulus_vec = jax.jit(
    jax.vmap(
        hvnm_relaxation_modulus,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
    )
)


@jax.jit
def hvnm_startup_stress_linear(
    t: float,
    gamma_dot: float,
    G_P: float,
    G_E: float,
    G_D: float,
    G_I_eff: float,
    X_phi: float,
    X_I: float,
    k_BER_mat_0: float,
    k_d_D: float,
    k_BER_int_0: float,
    D_int: float,
) -> float:
    """Startup stress in linear regime (constant rates, no TST feedback).

    4-term analytical solution:
    sigma(t) = G_P*X*gamma_dot*t
             + G_E*gamma_dot*tau_E*(1 - exp(-t/tau_E))
             + G_D*gamma_dot*tau_D*(1 - exp(-t/tau_D))
             + (1-D_int)*G_I_eff*X_I*gamma_dot*tau_I*(1 - exp(-t/tau_I))

    Parameters
    ----------
    t : float
        Time (s)
    gamma_dot : float
        Applied shear rate (1/s)
    [moduli, amplification, rates, damage]

    Returns
    -------
    float
        Shear stress at time t (Pa)
    """
    # Permanent: linear growth with amplification
    sigma_P = G_P * X_phi * gamma_dot * t

    # Exchangeable: Maxwell with factor-of-2
    tau_E = 1.0 / (2.0 * jnp.maximum(k_BER_mat_0, 1e-30))
    sigma_E = G_E * gamma_dot * tau_E * (1.0 - jnp.exp(-t / tau_E))

    # Dissociative: standard Maxwell
    tau_D = 1.0 / jnp.maximum(k_d_D, 1e-30)
    sigma_D = G_D * gamma_dot * tau_D * (1.0 - jnp.exp(-t / tau_D))

    # Interphase: Maxwell with factor-of-2 and amplification
    tau_I = 1.0 / (2.0 * jnp.maximum(k_BER_int_0, 1e-30))
    G_I_contrib = (1.0 - D_int) * G_I_eff * X_I
    sigma_I = G_I_contrib * gamma_dot * tau_I * (1.0 - jnp.exp(-t / tau_I))

    return sigma_P + sigma_E + sigma_D + sigma_I


hvnm_startup_stress_linear_vec = jax.jit(
    jax.vmap(
        hvnm_startup_stress_linear,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
    )
)


@jax.jit
def hvnm_steady_shear_stress(
    gamma_dot: float,
    G_D: float,
    k_d_D: float,
) -> float:
    """Steady-state shear stress (E and I networks contribute zero).

    At steady state:
    - mu^E -> mu^E_nat => sigma_E -> 0
    - mu^I -> mu^I_nat => sigma_I -> 0
    - sigma_P grows linearly (elastic, not viscous)
    Only D-network contributes viscous: sigma_D = G_D * gamma_dot / k_d_D

    Parameters
    ----------
    gamma_dot : float
        Shear rate (1/s)
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


hvnm_steady_shear_stress_vec = jax.jit(
    jax.vmap(hvnm_steady_shear_stress, in_axes=(0, None, None))
)


@jax.jit
def hvnm_creep_compliance_linear(
    t: float,
    G_P: float,
    G_E: float,
    G_D: float,
    G_I_eff: float,
    X_phi: float,
    X_I: float,
    k_BER_mat_0: float,
    k_d_D: float,
    k_BER_int_0: float,
) -> float:
    """Creep compliance J(t) for HVNM in linear regime.

    Three retardation modes (E, D, I) plus instantaneous elastic response.

    J(0+) = 1 / G_tot^NC
    J(inf) = 1 / (G_P * X)

    Parameters
    ----------
    t : float
        Time (s)
    [moduli, amplification, rates]

    Returns
    -------
    float
        Creep compliance J(t) (1/Pa)
    """
    G_P_amp = G_P * X_phi
    G_I_amp = G_I_eff * X_I
    G_tot = G_P_amp + G_E + G_D + G_I_amp
    G_P_safe = jnp.maximum(G_P_amp, 1e-30)

    # Instantaneous elastic compliance
    J_inst = 1.0 / jnp.maximum(G_tot, 1e-30)

    # E-network retardation
    tau_E_eff = 1.0 / (2.0 * jnp.maximum(k_BER_mat_0, 1e-30))
    tau_ret_E = G_tot / (G_P_safe * 2.0 * jnp.maximum(k_BER_mat_0, 1e-30))
    J_E = G_E / (G_P_safe * G_tot) * (1.0 - jnp.exp(-t / tau_ret_E))

    # D-network retardation
    tau_ret_D = G_tot / (G_P_safe * jnp.maximum(k_d_D, 1e-30))
    J_D = G_D / (G_P_safe * G_tot) * (1.0 - jnp.exp(-t / tau_ret_D))

    # I-network retardation (NEW)
    tau_I_eff = 1.0 / (2.0 * jnp.maximum(k_BER_int_0, 1e-30))
    tau_ret_I = G_tot / (G_P_safe * 2.0 * jnp.maximum(k_BER_int_0, 1e-30))
    J_I = G_I_amp / (G_P_safe * G_tot) * (1.0 - jnp.exp(-t / tau_ret_I))

    return J_inst + J_E + J_D + J_I


hvnm_creep_compliance_linear_vec = jax.jit(
    jax.vmap(
        hvnm_creep_compliance_linear,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )
)


# =============================================================================
# Relaxation with Diffusion (optional slow mode)
# =============================================================================


@jax.jit
def hvnm_relaxation_modulus_with_diffusion(
    t: float,
    G_P: float,
    G_E: float,
    G_D: float,
    G_I_eff: float,
    X_phi: float,
    X_I: float,
    k_BER_mat_0: float,
    k_d_D: float,
    k_BER_int_0: float,
    k_diff_mat: float,
    k_diff_int: float,
    D: float,
    D_int: float,
) -> float:
    """Relaxation modulus with diffusion-limited slow modes.

    G(t) = (1-D)*G_P*X
           + G_E*exp(-2*k_m*t)*exp(-k_diff_mat*t)
           + G_D*exp(-k_D*t)
           + (1-D_int)*G_I_eff*X_I*exp(-2*k_I*t)*exp(-k_diff_int*t)

    The additional exp(-k_diff*t) factors produce long-time tails.
    k_diff << k_BER, so these only matter at t >> 1/k_BER.

    Parameters
    ----------
    [same as hvnm_relaxation_modulus plus diffusion rates]
    k_diff_mat : float
        Matrix diffusion rate (1/s)
    k_diff_int : float
        Interfacial diffusion rate (1/s)

    Returns
    -------
    float
        Relaxation modulus with diffusion tails (Pa)
    """
    G_perm = (1.0 - D) * G_P * X_phi
    G_exch = G_E * jnp.exp(-(2.0 * k_BER_mat_0 + k_diff_mat) * t)
    G_diss = G_D * jnp.exp(-k_d_D * t)
    G_inter = (
        (1.0 - D_int) * G_I_eff * X_I
        * jnp.exp(-(2.0 * k_BER_int_0 + k_diff_int) * t)
    )
    return G_perm + G_exch + G_diss + G_inter


hvnm_relaxation_modulus_with_diffusion_vec = jax.jit(
    jax.vmap(
        hvnm_relaxation_modulus_with_diffusion,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None),
    )
)
