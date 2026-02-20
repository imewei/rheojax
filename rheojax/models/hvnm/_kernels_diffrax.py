"""diffrax ODE solvers for HVNM (Hybrid Vitrimer Nanocomposite Model).

Provides adaptive ODE integration for protocols that require numerical
solution: startup, relaxation, creep, and LAOS. Uses diffrax with
adaptive step size control.

State Vector (simple shear, 17 components without D_int, 18 with)
------------------------------------------------------------------
    y[0:3]   = [mu_E_xx, mu_E_yy, mu_E_xy]           # E-network distribution (3)
    y[3:6]   = [mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy] # E-network natural state (3)
    y[6:9]   = [mu_D_xx, mu_D_yy, mu_D_xy]           # D-network distribution (3)
    y[9]     = gamma                                   # accumulated strain (1)
    y[10]    = D                                       # matrix damage (1)
    y[11:14] = [mu_I_xx, mu_I_yy, mu_I_xy]           # I-network distribution (3)
    y[14:17] = [mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy] # I-network natural state (3)
    y[17]    = D_int                                   # interfacial damage (1, optional)

Key Differences from HVM ODE
-----------------------------
1. State vector 11 → 17/18 components
2. Interphase affine term uses amplified strain rate: X_I * gamma_dot
3. Dual k_BER computation: separate matrix and interphase rates at each step
4. Interfacial damage with self-healing: dD_int/dt has creation + healing terms
5. Amplified permanent stress: sigma_P = (1-D)*G_P*X*gamma
"""

from __future__ import annotations

from rheojax.core.jax_config import lazy_import, safe_import_jax

diffrax = lazy_import("diffrax")
from rheojax.models.hvm._kernels import (
    hvm_ber_rate_stress,
    hvm_ber_rate_stretch,
    hvm_damage_rhs,
    hvm_exchangeable_rhs_shear,
)
from rheojax.models.hvnm._kernels import (
    hvnm_ber_rate_interphase_stress,
    hvnm_ber_rate_interphase_stretch,
    hvnm_interfacial_damage_rhs,
    hvnm_interphase_rhs_shear,
    hvnm_total_stress_shear,
)
from rheojax.models.vlb._kernels import vlb_mu_rhs_shear

jax, jnp = safe_import_jax()


# =============================================================================
# Equilibrium Initial Condition
# =============================================================================


def _hvnm_initial_state(include_interfacial_damage: bool) -> jnp.ndarray:
    """Create equilibrium initial state vector for HVNM.

    At equilibrium:
    - mu^E = mu^E_nat = I  (exchangeable at natural state)
    - mu^D = I              (dissociative at equilibrium)
    - gamma = 0             (no strain)
    - D = 0                 (no matrix damage)
    - mu^I = mu^I_nat = I  (interphase at natural state)
    - D_int = 0             (no interfacial damage)

    Parameters
    ----------
    include_interfacial_damage : bool
        Whether interfacial damage is active (D_int slot always present)

    Returns
    -------
    jnp.ndarray
        Initial state vector (always 18 components)
    """
    # E-network: [mu_E_xx=1, mu_E_yy=1, mu_E_xy=0]
    # E-network natural state: [mu_E_nat_xx=1, mu_E_nat_yy=1, mu_E_nat_xy=0]
    # D-network: [mu_D_xx=1, mu_D_yy=1, mu_D_xy=0]
    # gamma=0, D=0
    # I-network: [mu_I_xx=1, mu_I_yy=1, mu_I_xy=0]
    # I-network natural state: [mu_I_nat_xx=1, mu_I_nat_yy=1, mu_I_nat_xy=0]
    # D_int=0 (interfacial damage, zero when not included)
    components = [
        1.0,
        1.0,
        0.0,  # E-network
        1.0,
        1.0,
        0.0,  # E-network natural state
        1.0,
        1.0,
        0.0,  # D-network
        0.0,  # gamma
        0.0,  # D (matrix damage)
        1.0,
        1.0,
        0.0,  # I-network
        1.0,
        1.0,
        0.0,  # I-network natural state
        0.0,  # D_int (interfacial damage, zero when not included)
    ]

    return jnp.array(components, dtype=jnp.float64)


def _hvnm_relaxation_initial_state(
    gamma_step: float, include_interfacial_damage: bool
) -> jnp.ndarray:
    """Create initial state after instantaneous step strain.

    Both E, D, and I networks are affinely deformed by gamma_step.
    Natural states remain at equilibrium (no exchange during step).

    Parameters
    ----------
    gamma_step : float
        Applied step strain
    include_interfacial_damage : bool
        Whether interfacial damage is active (D_int slot always present)

    Returns
    -------
    jnp.ndarray
        Deformed initial state vector (always 18 components)
    """
    mu_xx = 1.0 + 2.0 * gamma_step**2
    mu_yy = 1.0
    mu_xy = gamma_step

    components = [
        mu_xx,
        mu_yy,
        mu_xy,  # E-network (deformed)
        1.0,
        1.0,
        0.0,  # E-network natural state (equilibrium)
        mu_xx,
        mu_yy,
        mu_xy,  # D-network (deformed)
        gamma_step,  # gamma
        0.0,  # D
        mu_xx,
        mu_yy,
        mu_xy,  # I-network (deformed, with amplification applied)
        1.0,
        1.0,
        0.0,  # I-network natural state (equilibrium)
        0.0,  # D_int (interfacial damage, zero when not included)
    ]

    return jnp.array(components, dtype=jnp.float64)


# =============================================================================
# BER Rate Computation Helpers
# =============================================================================


def _compute_k_ber_matrix(
    y: jnp.ndarray,
    G_E: float,
    nu_0: float,
    E_a: float,
    V_act: float,
    T: float,
    kinetics: str,
) -> float:
    """Compute matrix BER rate from state vector."""
    mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
    mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]

    if kinetics == "stress":
        sigma_E_xx = G_E * (mu_E_xx - mu_E_nat_xx)
        sigma_E_yy = G_E * (mu_E_yy - mu_E_nat_yy)
        sigma_E_xy = G_E * (mu_E_xy - mu_E_nat_xy)
        return hvm_ber_rate_stress(
            sigma_E_xx, sigma_E_yy, sigma_E_xy, nu_0, E_a, V_act, T
        )
    else:  # stretch
        return hvm_ber_rate_stretch(
            mu_E_xx,
            mu_E_yy,
            mu_E_nat_xx,
            mu_E_nat_yy,
            G_E,
            nu_0,
            E_a,
            V_act,
            T,
        )


def _compute_k_ber_interphase(
    y: jnp.ndarray,
    G_I_eff: float,
    X_I: float,
    nu_0_int: float,
    E_a_int: float,
    V_act_int: float,
    T: float,
    kinetics: str,
) -> float:
    """Compute interfacial BER rate from state vector."""
    mu_I_xx, mu_I_yy, mu_I_xy = y[11], y[12], y[13]
    mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy = y[14], y[15], y[16]

    if kinetics == "stress":
        # I-network stress components
        sigma_I_xx = G_I_eff * X_I * (mu_I_xx - mu_I_nat_xx)
        sigma_I_yy = G_I_eff * X_I * (mu_I_yy - mu_I_nat_yy)
        sigma_I_xy = G_I_eff * X_I * (mu_I_xy - mu_I_nat_xy)
        return hvnm_ber_rate_interphase_stress(
            sigma_I_xx,
            sigma_I_yy,
            sigma_I_xy,
            nu_0_int,
            E_a_int,
            V_act_int,
            T,
        )
    else:  # stretch
        return hvnm_ber_rate_interphase_stretch(
            mu_I_xx,
            mu_I_yy,
            mu_I_nat_xx,
            mu_I_nat_yy,
            G_I_eff,
            nu_0_int,
            E_a_int,
            V_act_int,
            T,
        )


# =============================================================================
# ODE Vector Fields
# =============================================================================


def _make_hvnm_startup_vector_field(
    kinetics: str,
    include_damage: bool,
    include_dissociative: bool,
    include_interfacial_damage: bool,
):
    """Create ODE vector field for startup shear.

    gamma_dot is constant, passed via args dict.
    """

    def vector_field(t, y, args):
        gamma_dot = args["gamma_dot"]
        G_P = args["G_P"]
        G_E = args["G_E"]
        G_D = args["G_D"]
        k_d_D = args["k_d_D"]
        nu_0 = args["nu_0"]
        E_a = args["E_a"]
        V_act = args["V_act"]
        T = args["T"]
        Gamma_0 = args["Gamma_0"]
        lambda_crit = args["lambda_crit"]
        # NP params
        G_I_eff = args["G_I_eff"]
        X_I = args["X_I"]
        nu_0_int = args["nu_0_int"]
        E_a_int = args["E_a_int"]
        V_act_int = args["V_act_int"]
        # Interfacial damage params
        Gamma_0_int = args["Gamma_0_int"]
        lambda_crit_int = args["lambda_crit_int"]
        h_0 = args["h_0"]
        E_a_heal = args["E_a_heal"]
        n_h = args["n_h"]

        # Unpack state (always 18 components)
        mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
        mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]
        mu_D_xx, mu_D_yy, mu_D_xy = y[6], y[7], y[8]
        D_val = y[10]
        mu_I_xx, mu_I_yy, mu_I_xy = y[11], y[12], y[13]
        mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy = y[14], y[15], y[16]
        D_int_val = y[17]

        # Compute dual BER rates
        k_BER_mat = _compute_k_ber_matrix(y, G_E, nu_0, E_a, V_act, T, kinetics)
        k_BER_int = _compute_k_ber_interphase(
            y, G_I_eff, X_I, nu_0_int, E_a_int, V_act_int, T, kinetics
        )

        # E-network evolution (6 equations, reusing HVM kernel)
        dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy = (
            hvm_exchangeable_rhs_shear(
                mu_E_xx,
                mu_E_yy,
                mu_E_xy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_E_nat_xy,
                gamma_dot,
                k_BER_mat,
            )
        )

        # D-network evolution (3 equations, reusing VLB kernel)
        dmu_D_xx, dmu_D_yy, _, dmu_D_xy = vlb_mu_rhs_shear(
            mu_D_xx,
            mu_D_yy,
            1.0,
            mu_D_xy,
            gamma_dot,
            k_d_D,
        )
        dmu_D_xx = jnp.where(include_dissociative, dmu_D_xx, 0.0)
        dmu_D_yy = jnp.where(include_dissociative, dmu_D_yy, 0.0)
        dmu_D_xy = jnp.where(include_dissociative, dmu_D_xy, 0.0)

        # I-network evolution (6 equations, with strain amplification)
        dmu_I_xx, dmu_I_yy, dmu_I_xy, dmu_I_nat_xx, dmu_I_nat_yy, dmu_I_nat_xy = (
            hvnm_interphase_rhs_shear(
                mu_I_xx,
                mu_I_yy,
                mu_I_xy,
                mu_I_nat_xx,
                mu_I_nat_yy,
                mu_I_nat_xy,
                gamma_dot,
                X_I,
                k_BER_int,
            )
        )

        # Strain accumulation
        dgamma = gamma_dot

        # Matrix damage evolution
        dD = jnp.where(
            include_damage,
            hvm_damage_rhs(
                mu_E_xx,
                mu_E_yy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_D_xx,
                mu_D_yy,
                D_val,
                G_P,
                G_E,
                G_D,
                Gamma_0,
                lambda_crit,
            ),
            0.0,
        )

        # Interfacial damage (zero when not included)
        dD_int = jnp.where(
            include_interfacial_damage,
            hvnm_interfacial_damage_rhs(
                mu_I_xx,
                mu_I_yy,
                D_int_val,
                Gamma_0_int,
                lambda_crit_int,
                h_0,
                E_a_heal,
                n_h,
                T,
            ),
            0.0,
        )

        return jnp.array([
            dmu_E_xx,
            dmu_E_yy,
            dmu_E_xy,
            dmu_E_nat_xx,
            dmu_E_nat_yy,
            dmu_E_nat_xy,
            dmu_D_xx,
            dmu_D_yy,
            dmu_D_xy,
            dgamma,
            dD,
            dmu_I_xx,
            dmu_I_yy,
            dmu_I_xy,
            dmu_I_nat_xx,
            dmu_I_nat_yy,
            dmu_I_nat_xy,
            dD_int,
        ])

    return jax.checkpoint(vector_field)


def _make_hvnm_relaxation_vector_field(
    kinetics: str,
    include_damage: bool,
    include_dissociative: bool,
    include_interfacial_damage: bool,
):
    """Create ODE vector field for stress relaxation (gamma_dot = 0)."""

    def vector_field(t, y, args):
        G_E = args["G_E"]
        k_d_D = args["k_d_D"]
        nu_0 = args["nu_0"]
        E_a = args["E_a"]
        V_act = args["V_act"]
        T = args["T"]
        G_I_eff = args["G_I_eff"]
        X_I = args["X_I"]
        nu_0_int = args["nu_0_int"]
        E_a_int = args["E_a_int"]
        V_act_int = args["V_act_int"]

        mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
        mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]
        mu_D_xx, mu_D_yy, mu_D_xy = y[6], y[7], y[8]
        mu_I_xx, mu_I_yy, mu_I_xy = y[11], y[12], y[13]
        mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy = y[14], y[15], y[16]

        gamma_dot = 0.0

        k_BER_mat = _compute_k_ber_matrix(y, G_E, nu_0, E_a, V_act, T, kinetics)
        k_BER_int = _compute_k_ber_interphase(
            y, G_I_eff, X_I, nu_0_int, E_a_int, V_act_int, T, kinetics
        )

        dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy = (
            hvm_exchangeable_rhs_shear(
                mu_E_xx,
                mu_E_yy,
                mu_E_xy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_E_nat_xy,
                gamma_dot,
                k_BER_mat,
            )
        )

        dmu_D_xx, dmu_D_yy, _, dmu_D_xy = vlb_mu_rhs_shear(
            mu_D_xx,
            mu_D_yy,
            1.0,
            mu_D_xy,
            gamma_dot,
            k_d_D,
        )
        dmu_D_xx = jnp.where(include_dissociative, dmu_D_xx, 0.0)
        dmu_D_yy = jnp.where(include_dissociative, dmu_D_yy, 0.0)
        dmu_D_xy = jnp.where(include_dissociative, dmu_D_xy, 0.0)

        dmu_I_xx, dmu_I_yy, dmu_I_xy, dmu_I_nat_xx, dmu_I_nat_yy, dmu_I_nat_xy = (
            hvnm_interphase_rhs_shear(
                mu_I_xx,
                mu_I_yy,
                mu_I_xy,
                mu_I_nat_xx,
                mu_I_nat_yy,
                mu_I_nat_xy,
                gamma_dot,
                X_I,
                k_BER_int,
            )
        )

        return jnp.array([
            dmu_E_xx,
            dmu_E_yy,
            dmu_E_xy,
            dmu_E_nat_xx,
            dmu_E_nat_yy,
            dmu_E_nat_xy,
            dmu_D_xx,
            dmu_D_yy,
            dmu_D_xy,
            0.0,
            0.0,  # dgamma = 0, dD = 0 during relaxation
            dmu_I_xx,
            dmu_I_yy,
            dmu_I_xy,
            dmu_I_nat_xx,
            dmu_I_nat_yy,
            dmu_I_nat_xy,
            0.0,  # dD_int = 0 during relaxation
        ])

    return jax.checkpoint(vector_field)


def _make_hvnm_laos_vector_field(
    kinetics: str,
    include_damage: bool,
    include_dissociative: bool,
    include_interfacial_damage: bool,
):
    """Create ODE vector field for LAOS.

    Oscillatory shear: gamma(t) = gamma_0 * sin(omega * t)
                       gamma_dot(t) = gamma_0 * omega * cos(omega * t)
    """

    def vector_field(t, y, args):
        gamma_0 = args["gamma_0"]
        omega = args["omega"]
        G_P = args["G_P"]
        G_E = args["G_E"]
        G_D = args["G_D"]
        k_d_D = args["k_d_D"]
        nu_0 = args["nu_0"]
        E_a = args["E_a"]
        V_act = args["V_act"]
        T = args["T"]
        Gamma_0 = args["Gamma_0"]
        lambda_crit = args["lambda_crit"]
        G_I_eff = args["G_I_eff"]
        X_I = args["X_I"]
        nu_0_int = args["nu_0_int"]
        E_a_int = args["E_a_int"]
        V_act_int = args["V_act_int"]
        Gamma_0_int = args["Gamma_0_int"]
        lambda_crit_int = args["lambda_crit_int"]
        h_0 = args["h_0"]
        E_a_heal = args["E_a_heal"]
        n_h = args["n_h"]

        # Unpack state (always 18 components)
        mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
        mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]
        mu_D_xx, mu_D_yy, mu_D_xy = y[6], y[7], y[8]
        D_val = y[10]
        mu_I_xx, mu_I_yy, mu_I_xy = y[11], y[12], y[13]
        mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy = y[14], y[15], y[16]
        D_int_val = y[17]

        gamma_dot = gamma_0 * omega * jnp.cos(omega * t)

        k_BER_mat = _compute_k_ber_matrix(y, G_E, nu_0, E_a, V_act, T, kinetics)
        k_BER_int = _compute_k_ber_interphase(
            y, G_I_eff, X_I, nu_0_int, E_a_int, V_act_int, T, kinetics
        )

        dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy = (
            hvm_exchangeable_rhs_shear(
                mu_E_xx,
                mu_E_yy,
                mu_E_xy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_E_nat_xy,
                gamma_dot,
                k_BER_mat,
            )
        )

        dmu_D_xx, dmu_D_yy, _, dmu_D_xy = vlb_mu_rhs_shear(
            mu_D_xx,
            mu_D_yy,
            1.0,
            mu_D_xy,
            gamma_dot,
            k_d_D,
        )
        dmu_D_xx = jnp.where(include_dissociative, dmu_D_xx, 0.0)
        dmu_D_yy = jnp.where(include_dissociative, dmu_D_yy, 0.0)
        dmu_D_xy = jnp.where(include_dissociative, dmu_D_xy, 0.0)

        dmu_I_xx, dmu_I_yy, dmu_I_xy, dmu_I_nat_xx, dmu_I_nat_yy, dmu_I_nat_xy = (
            hvnm_interphase_rhs_shear(
                mu_I_xx,
                mu_I_yy,
                mu_I_xy,
                mu_I_nat_xx,
                mu_I_nat_yy,
                mu_I_nat_xy,
                gamma_dot,
                X_I,
                k_BER_int,
            )
        )

        dgamma = gamma_dot

        dD = jnp.where(
            include_damage,
            hvm_damage_rhs(
                mu_E_xx,
                mu_E_yy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_D_xx,
                mu_D_yy,
                D_val,
                G_P,
                G_E,
                G_D,
                Gamma_0,
                lambda_crit,
            ),
            0.0,
        )

        # Interfacial damage (zero when not included)
        dD_int = jnp.where(
            include_interfacial_damage,
            hvnm_interfacial_damage_rhs(
                mu_I_xx,
                mu_I_yy,
                D_int_val,
                Gamma_0_int,
                lambda_crit_int,
                h_0,
                E_a_heal,
                n_h,
                T,
            ),
            0.0,
        )

        return jnp.array([
            dmu_E_xx,
            dmu_E_yy,
            dmu_E_xy,
            dmu_E_nat_xx,
            dmu_E_nat_yy,
            dmu_E_nat_xy,
            dmu_D_xx,
            dmu_D_yy,
            dmu_D_xy,
            dgamma,
            dD,
            dmu_I_xx,
            dmu_I_yy,
            dmu_I_xy,
            dmu_I_nat_xx,
            dmu_I_nat_yy,
            dmu_I_nat_xy,
            dD_int,
        ])

    return jax.checkpoint(vector_field)


def _make_hvnm_creep_vector_field(
    kinetics: str,
    include_damage: bool,
    include_dissociative: bool,
    include_interfacial_damage: bool,
):
    """Create ODE vector field for creep.

    Constant applied stress sigma_0. At each step, solve for gamma_dot
    from the 4-network stress balance.
    """

    def vector_field(t, y, args):
        sigma_0 = args["sigma_0"]
        G_P = args["G_P"]
        G_E = args["G_E"]
        G_D = args["G_D"]
        k_d_D = args["k_d_D"]
        nu_0 = args["nu_0"]
        E_a = args["E_a"]
        V_act = args["V_act"]
        T = args["T"]
        Gamma_0 = args["Gamma_0"]
        lambda_crit = args["lambda_crit"]
        G_I_eff = args["G_I_eff"]
        X_phi = args["X_phi"]
        X_I = args["X_I"]
        nu_0_int = args["nu_0_int"]
        E_a_int = args["E_a_int"]
        V_act_int = args["V_act_int"]
        Gamma_0_int = args["Gamma_0_int"]
        lambda_crit_int = args["lambda_crit_int"]
        h_0 = args["h_0"]
        E_a_heal = args["E_a_heal"]
        n_h = args["n_h"]

        # Unpack state (always 18 components)
        mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
        mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]
        mu_D_xx, mu_D_yy, mu_D_xy = y[6], y[7], y[8]
        gamma = y[9]
        D_val = y[10]
        mu_I_xx, mu_I_yy, mu_I_xy = y[11], y[12], y[13]
        mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy = y[14], y[15], y[16]
        D_int_val = jnp.where(include_interfacial_damage, y[17], 0.0)

        # Current total stress
        sigma_elastic = hvnm_total_stress_shear(
            gamma,
            mu_E_xy,
            mu_E_nat_xy,
            mu_D_xy,
            mu_I_xy,
            mu_I_nat_xy,
            G_P,
            G_E,
            G_D,
            G_I_eff,
            X_phi,
            X_I,
            D_val,
            D_int_val,
        )
        sigma_residual = sigma_0 - sigma_elastic

        # Compute dual BER rates for viscosity
        k_BER_mat = _compute_k_ber_matrix(y, G_E, nu_0, E_a, V_act, T, kinetics)
        k_BER_int = _compute_k_ber_interphase(
            y, G_I_eff, X_I, nu_0_int, E_a_int, V_act_int, T, kinetics
        )
        k_BER_mat_safe = jnp.maximum(k_BER_mat, 1e-30)
        k_BER_int_safe = jnp.maximum(k_BER_int, 1e-30)
        k_d_D_safe = jnp.maximum(k_d_D, 1e-30)

        # Effective viscosity from all relaxing networks
        eta_E = G_E / (2.0 * k_BER_mat_safe)
        eta_D = jnp.where(include_dissociative, G_D / k_d_D_safe, 0.0)
        eta_I = G_I_eff * X_I / (2.0 * k_BER_int_safe)
        # Small regularization from P-network elastic stiffness
        eta_eff = eta_E + eta_D + eta_I + G_P * X_phi * 1e-6

        gamma_dot = sigma_residual / jnp.maximum(eta_eff, 1e-30)
        gamma_dot = jnp.clip(gamma_dot, -1e10, 1e10)

        # E-network evolution
        dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy = (
            hvm_exchangeable_rhs_shear(
                mu_E_xx,
                mu_E_yy,
                mu_E_xy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_E_nat_xy,
                gamma_dot,
                k_BER_mat,
            )
        )

        # D-network evolution
        dmu_D_xx, dmu_D_yy, _, dmu_D_xy = vlb_mu_rhs_shear(
            mu_D_xx,
            mu_D_yy,
            1.0,
            mu_D_xy,
            gamma_dot,
            k_d_D,
        )
        dmu_D_xx = jnp.where(include_dissociative, dmu_D_xx, 0.0)
        dmu_D_yy = jnp.where(include_dissociative, dmu_D_yy, 0.0)
        dmu_D_xy = jnp.where(include_dissociative, dmu_D_xy, 0.0)

        # I-network evolution
        dmu_I_xx, dmu_I_yy, dmu_I_xy, dmu_I_nat_xx, dmu_I_nat_yy, dmu_I_nat_xy = (
            hvnm_interphase_rhs_shear(
                mu_I_xx,
                mu_I_yy,
                mu_I_xy,
                mu_I_nat_xx,
                mu_I_nat_yy,
                mu_I_nat_xy,
                gamma_dot,
                X_I,
                k_BER_int,
            )
        )

        dgamma = gamma_dot

        dD = jnp.where(
            include_damage,
            hvm_damage_rhs(
                mu_E_xx,
                mu_E_yy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_D_xx,
                mu_D_yy,
                D_val,
                G_P,
                G_E,
                G_D,
                Gamma_0,
                lambda_crit,
            ),
            0.0,
        )

        # Interfacial damage (zero when not included)
        dD_int = jnp.where(
            include_interfacial_damage,
            hvnm_interfacial_damage_rhs(
                mu_I_xx,
                mu_I_yy,
                D_int_val,
                Gamma_0_int,
                lambda_crit_int,
                h_0,
                E_a_heal,
                n_h,
                T,
            ),
            0.0,
        )

        return jnp.array([
            dmu_E_xx,
            dmu_E_yy,
            dmu_E_xy,
            dmu_E_nat_xx,
            dmu_E_nat_yy,
            dmu_E_nat_xy,
            dmu_D_xx,
            dmu_D_yy,
            dmu_D_xy,
            dgamma,
            dD,
            dmu_I_xx,
            dmu_I_yy,
            dmu_I_xy,
            dmu_I_nat_xx,
            dmu_I_nat_yy,
            dmu_I_nat_xy,
            dD_int,
        ])

    return jax.checkpoint(vector_field)


# =============================================================================
# Solver Dispatcher
# =============================================================================


def _solve_hvnm_ode(
    t: jnp.ndarray,
    vector_field,
    y0: jnp.ndarray,
    args: dict,
    max_steps: int = 500_000,
) -> diffrax.Solution:
    """Solve HVNM ODE system with Tsit5 solver.

    Uses Tsit5 (explicit) to avoid Kvaerno5 TracerBoolConversionError
    with JAX tracing (known issue with lineax LU).

    Parameters
    ----------
    t : jnp.ndarray
        Time array for output
    vector_field : callable
        ODE vector field function
    y0 : jnp.ndarray
        Initial state vector (17 or 18 components)
    args : dict
        Parameters for the vector field
    max_steps : int
        Maximum ODE steps

    Returns
    -------
    diffrax.Solution
        ODE solution
    """
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)

    t0 = t[0]
    t1 = t[-1]
    dt0 = (t1 - t0) / jnp.maximum(jnp.float64(len(t)), 1000.0)

    saveat = diffrax.SaveAt(ts=t)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=False,
    )

    return sol


def _mask_failed_solution_ys(sol) -> jnp.ndarray:
    """Extract sol.ys, masking with NaN if the solver failed.

    diffrax throw=False returns garbage ys when the solver hits max_steps
    or diverges. sol.result == 0 means success; nonzero means failure.
    """
    failed = sol.result != 0
    return jnp.where(failed, jnp.nan, sol.ys)


# =============================================================================
# Public Solver Functions
# =============================================================================


def _default_hvnm_args(params: dict) -> dict:
    """Set defaults for optional HVNM parameters."""
    args = {**params}
    args.setdefault("G_D", 0.0)
    args.setdefault("k_d_D", 1.0)
    args.setdefault("Gamma_0", 0.0)
    args.setdefault("lambda_crit", 10.0)
    args.setdefault("G_I_eff", 0.0)
    args.setdefault("X_phi", 1.0)
    args.setdefault("X_I", 1.0)
    args.setdefault("nu_0_int", 1e10)
    args.setdefault("E_a_int", 90e3)
    args.setdefault("V_act_int", 5e-6)
    args.setdefault("Gamma_0_int", 0.0)
    args.setdefault("lambda_crit_int", 10.0)
    args.setdefault("h_0", 0.0)
    args.setdefault("E_a_heal", 100e3)
    args.setdefault("n_h", 1.0)
    return args


def hvnm_solve_startup(
    t: jnp.ndarray,
    gamma_dot: float,
    params: dict,
    kinetics: str = "stress",
    include_damage: bool = False,
    include_dissociative: bool = True,
    include_interfacial_damage: bool = False,
) -> diffrax.Solution:
    """Solve HVNM startup shear ODE.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (s)
    gamma_dot : float
        Constant applied shear rate (1/s)
    params : dict
        Model parameters including NP geometry
    kinetics : str
        TST coupling type
    include_damage : bool
        Whether matrix damage is active
    include_dissociative : bool
        Whether D-network is present
    include_interfacial_damage : bool
        Whether interfacial damage is active

    Returns
    -------
    diffrax.Solution
        ODE solution with y shape (n_times, 17 or 18)
    """
    vf = _make_hvnm_startup_vector_field(
        kinetics, include_damage, include_dissociative, include_interfacial_damage
    )
    y0 = _hvnm_initial_state(include_interfacial_damage)
    args = _default_hvnm_args(params)
    args["gamma_dot"] = gamma_dot
    return _solve_hvnm_ode(t, vf, y0, args)


def hvnm_solve_relaxation(
    t: jnp.ndarray,
    gamma_step: float,
    params: dict,
    kinetics: str = "stress",
    include_damage: bool = False,
    include_dissociative: bool = True,
    include_interfacial_damage: bool = False,
) -> diffrax.Solution:
    """Solve HVNM stress relaxation ODE after step strain.

    Parameters
    ----------
    t : jnp.ndarray
        Time array after step (s)
    gamma_step : float
        Applied step strain
    params : dict
        Model parameters
    [other args same as hvnm_solve_startup]

    Returns
    -------
    diffrax.Solution
        ODE solution
    """
    vf = _make_hvnm_relaxation_vector_field(
        kinetics, include_damage, include_dissociative, include_interfacial_damage
    )
    y0 = _hvnm_relaxation_initial_state(gamma_step, include_interfacial_damage)
    args = _default_hvnm_args(params)
    return _solve_hvnm_ode(t, vf, y0, args)


def hvnm_solve_creep(
    t: jnp.ndarray,
    sigma_0: float,
    params: dict,
    kinetics: str = "stress",
    include_damage: bool = False,
    include_dissociative: bool = True,
    include_interfacial_damage: bool = False,
) -> diffrax.Solution:
    """Solve HVNM creep ODE under constant stress.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (s)
    sigma_0 : float
        Applied constant stress (Pa)
    params : dict
        Model parameters
    [other args same as hvnm_solve_startup]

    Returns
    -------
    diffrax.Solution
        ODE solution
    """
    vf = _make_hvnm_creep_vector_field(
        kinetics, include_damage, include_dissociative, include_interfacial_damage
    )
    y0 = _hvnm_initial_state(include_interfacial_damage)
    args = _default_hvnm_args(params)
    args["sigma_0"] = sigma_0
    return _solve_hvnm_ode(t, vf, y0, args)


def hvnm_solve_laos(
    t: jnp.ndarray,
    gamma_0: float,
    omega: float,
    params: dict,
    kinetics: str = "stress",
    include_damage: bool = False,
    include_dissociative: bool = True,
    include_interfacial_damage: bool = False,
) -> diffrax.Solution:
    """Solve HVNM LAOS ODE under oscillatory shear.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (s)
    gamma_0 : float
        Strain amplitude
    omega : float
        Angular frequency (rad/s)
    params : dict
        Model parameters
    [other args same as hvnm_solve_startup]

    Returns
    -------
    diffrax.Solution
        ODE solution
    """
    vf = _make_hvnm_laos_vector_field(
        kinetics, include_damage, include_dissociative, include_interfacial_damage
    )
    y0 = _hvnm_initial_state(include_interfacial_damage)
    args = _default_hvnm_args(params)
    args["gamma_0"] = gamma_0
    args["omega"] = omega
    return _solve_hvnm_ode(t, vf, y0, args)
