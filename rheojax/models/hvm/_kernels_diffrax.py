"""diffrax ODE solvers for HVM (Hybrid Vitrimer Model).

Provides adaptive ODE integration for protocols that require numerical
solution: startup, relaxation, creep, and LAOS. Uses diffrax with
adaptive step size control.

State Vector (simple shear, 11 components)
-------------------------------------------
    y[0:3]  = [mu_E_xx, mu_E_yy, mu_E_xy]       # E-network distribution
    y[3:6]  = [mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy]  # E-network natural state
    y[6:9]  = [mu_D_xx, mu_D_yy, mu_D_xy]        # D-network distribution
    y[9]    = gamma                                 # accumulated strain
    y[10]   = D                                     # damage variable

Solver Selection
-----------------
- TST active (kinetics='stress'/'stretch'): Tsit5 with tighter tolerances
  with PIDController(rtol=1e-8, atol=1e-10)
- Constant rates / SAOS regime: Tsit5 (explicit, faster)
  with PIDController(rtol=1e-6, atol=1e-8)
"""

from __future__ import annotations

from rheojax.core.jax_config import lazy_import, safe_import_jax

diffrax = lazy_import("diffrax")
from rheojax.models.hvm._kernels import (
    hvm_ber_rate_stress,
    hvm_ber_rate_stretch,
    hvm_damage_rhs,
    hvm_exchangeable_rhs_shear,
    hvm_total_stress_shear,
)
from rheojax.models.vlb._kernels import vlb_mu_rhs_shear

jax, jnp = safe_import_jax()


# =============================================================================
# Equilibrium Initial Condition
# =============================================================================


def _hvm_initial_state(include_dissociative: bool, include_damage: bool) -> jnp.ndarray:
    """Create equilibrium initial state vector.

    At equilibrium:
    - mu^E = mu^E_nat = I  (exchangeable at natural state)
    - mu^D = I              (dissociative at equilibrium)
    - gamma = 0             (no strain)
    - D = 0                 (no damage)
    """
    # E-network: mu_E_xx=1, mu_E_yy=1, mu_E_xy=0
    # E-network natural state: mu_E_nat_xx=1, mu_E_nat_yy=1, mu_E_nat_xy=0
    y0 = jnp.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

    # D-network: mu_D_xx=1, mu_D_yy=1, mu_D_xy=0 (always included in state, zeroed if not active)
    y0 = jnp.concatenate([y0, jnp.array([1.0, 1.0, 0.0], dtype=jnp.float64)])

    # gamma = 0
    y0 = jnp.concatenate([y0, jnp.array([0.0], dtype=jnp.float64)])

    # D = 0 (damage)
    y0 = jnp.concatenate([y0, jnp.array([0.0], dtype=jnp.float64)])

    return y0


# =============================================================================
# BER Rate Computation Helper
# =============================================================================


def _compute_k_ber(
    y: jnp.ndarray,
    G_E: float,
    nu_0: float,
    E_a: float,
    V_act: float,
    T: float,
    kinetics: str,
) -> float:
    """Compute BER rate based on kinetics type.

    Parameters
    ----------
    y : jnp.ndarray
        State vector (11 components)
    G_E : float
        Exchangeable network modulus (Pa)
    nu_0, E_a, V_act, T : float
        TST parameters
    kinetics : str
        'stress' or 'stretch'

    Returns
    -------
    float
        BER rate k_BER (1/s)
    """
    mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
    mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]

    # Always compute both BER rate pathways
    sigma_E_xx = G_E * (mu_E_xx - mu_E_nat_xx)
    sigma_E_yy = G_E * (mu_E_yy - mu_E_nat_yy)
    sigma_E_xy = G_E * (mu_E_xy - mu_E_nat_xy)

    k_ber_stress = hvm_ber_rate_stress(
        sigma_E_xx, sigma_E_yy, sigma_E_xy, nu_0, E_a, V_act, T
    )
    k_ber_stretch = hvm_ber_rate_stretch(
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

    # Select based on kinetics type (closure-captured Python bool)
    is_stress = (kinetics == "stress")
    return jnp.where(is_stress, k_ber_stress, k_ber_stretch)


# =============================================================================
# ODE Vector Fields
# =============================================================================


def _make_startup_vector_field(
    kinetics: str,
    include_damage: bool,
    include_dissociative: bool,
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

        # Unpack state
        mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
        mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]
        mu_D_xx, mu_D_yy, mu_D_xy = y[6], y[7], y[8]
        D_val = y[10]

        # Compute BER rate
        k_BER = _compute_k_ber(y, G_E, nu_0, E_a, V_act, T, kinetics)

        # E-network evolution (6 equations)
        dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy = (
            hvm_exchangeable_rhs_shear(
                mu_E_xx,
                mu_E_yy,
                mu_E_xy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_E_nat_xy,
                gamma_dot,
                k_BER,
            )
        )

        # D-network evolution (3 equations, reusing VLB kernel)
        dmu_D_xx, dmu_D_yy, _dmu_D_zz, dmu_D_xy = vlb_mu_rhs_shear(
            mu_D_xx,
            mu_D_yy,
            1.0,
            mu_D_xy,
            gamma_dot,
            k_d_D,
        )

        # Zero out D-network if not included
        dmu_D_xx = jnp.where(include_dissociative, dmu_D_xx, 0.0)
        dmu_D_yy = jnp.where(include_dissociative, dmu_D_yy, 0.0)
        dmu_D_xy = jnp.where(include_dissociative, dmu_D_xy, 0.0)

        # Strain accumulation
        dgamma = gamma_dot

        # Damage evolution
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

        return jnp.array(
            [
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
            ]
        )

    # Wrap with checkpoint to trade compute for memory during NUTS reverse-mode AD
    return jax.checkpoint(vector_field)


def _make_relaxation_vector_field(
    kinetics: str,
    include_damage: bool,
    include_dissociative: bool,
):
    """Create ODE vector field for stress relaxation.

    After step strain, gamma_dot = 0. State starts from deformed
    configuration with gamma = gamma_step.
    """

    def vector_field(t, y, args):
        G_E = args["G_E"]
        k_d_D = args["k_d_D"]
        nu_0 = args["nu_0"]
        E_a = args["E_a"]
        V_act = args["V_act"]
        T = args["T"]

        mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
        mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]
        mu_D_xx, mu_D_yy, mu_D_xy = y[6], y[7], y[8]

        # gamma_dot = 0 for relaxation
        gamma_dot = 0.0

        k_BER = _compute_k_ber(y, G_E, nu_0, E_a, V_act, T, kinetics)

        dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy = (
            hvm_exchangeable_rhs_shear(
                mu_E_xx,
                mu_E_yy,
                mu_E_xy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_E_nat_xy,
                gamma_dot,
                k_BER,
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

        return jnp.array(
            [
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
            ]
        )

    return jax.checkpoint(vector_field)


def _make_laos_vector_field(
    kinetics: str,
    include_damage: bool,
    include_dissociative: bool,
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

        mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
        mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]
        mu_D_xx, mu_D_yy, mu_D_xy = y[6], y[7], y[8]
        D_val = y[10]

        gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
        k_BER = _compute_k_ber(y, G_E, nu_0, E_a, V_act, T, kinetics)

        dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy = (
            hvm_exchangeable_rhs_shear(
                mu_E_xx,
                mu_E_yy,
                mu_E_xy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_E_nat_xy,
                gamma_dot,
                k_BER,
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

        return jnp.array(
            [
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
            ]
        )

    return jax.checkpoint(vector_field)


def _make_creep_vector_field(
    kinetics: str,
    include_damage: bool,
    include_dissociative: bool,
):
    """Create ODE vector field for creep.

    Constant applied stress sigma_0. At each step, solve for gamma_dot
    from the stress balance:
    sigma_0 = (1-D)*G_P*gamma + G_E*(mu^E_xy - mu^E_nat_xy) + G_D*mu^D_xy

    The P-network acts as a spring, so the "viscous" contribution comes
    from E and D network relaxation allowing strain accumulation.
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

        mu_E_xx, mu_E_yy, mu_E_xy = y[0], y[1], y[2]
        mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy = y[3], y[4], y[5]
        mu_D_xx, mu_D_yy, mu_D_xy = y[6], y[7], y[8]
        gamma = y[9]
        D_val = y[10]

        # Solve stress balance for gamma_dot
        # sigma_0 = (1-D)*G_P*gamma + G_E*(mu_E_xy - mu_E_nat_xy) + G_D*mu_D_xy
        # + eta_eff * gamma_dot
        #
        # For a network with no explicit solvent viscosity, we need to
        # construct an effective viscosity from the network dynamics.
        # The total stress rate must equal zero (constant sigma_0), so:
        # dsigma/dt = 0 => G_P*gamma_dot + G_E*d(mu_E_xy - mu_E_nat_xy)/dt + G_D*dmu_D_xy/dt = 0
        #
        # For stability, we use the quasi-static approximation:
        # gamma_dot = sigma_residual / eta_eff
        sigma_elastic = hvm_total_stress_shear(
            gamma,
            mu_E_xy,
            mu_E_nat_xy,
            mu_D_xy,
            G_P,
            G_E,
            G_D,
            D_val,
        )
        sigma_residual = sigma_0 - sigma_elastic

        # Effective viscosity = sum of network viscosities
        # eta_E = G_E / (2*k_BER), eta_D = G_D / k_d_D
        k_BER = _compute_k_ber(y, G_E, nu_0, E_a, V_act, T, kinetics)
        k_BER_safe = jnp.maximum(k_BER, 1e-30)
        k_d_D_safe = jnp.maximum(k_d_D, 1e-30)

        eta_E = G_E / (2.0 * k_BER_safe)
        eta_D = jnp.where(include_dissociative, G_D / k_d_D_safe, 0.0)
        # Add P-network elastic stiffness as a regularization
        # (prevents unbounded gamma_dot at t=0)
        eta_eff = eta_E + eta_D + G_P * 1e-6  # Small regularization

        gamma_dot = sigma_residual / jnp.maximum(eta_eff, 1e-30)
        # Clamp to prevent numerical blowup
        gamma_dot = jnp.clip(gamma_dot, -1e10, 1e10)

        dmu_E_xx, dmu_E_yy, dmu_E_xy, dmu_E_nat_xx, dmu_E_nat_yy, dmu_E_nat_xy = (
            hvm_exchangeable_rhs_shear(
                mu_E_xx,
                mu_E_yy,
                mu_E_xy,
                mu_E_nat_xx,
                mu_E_nat_yy,
                mu_E_nat_xy,
                gamma_dot,
                k_BER,
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

        return jnp.array(
            [
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
            ]
        )

    return jax.checkpoint(vector_field)


# =============================================================================
# Solver Dispatcher
# =============================================================================


def _solve_hvm_ode(
    t: jnp.ndarray,
    vector_field,
    y0: jnp.ndarray,
    args: dict,
    use_stiff_solver: bool = True,
    max_steps: int = 500_000,
) -> diffrax.Solution:
    """Solve HVM ODE system with appropriate solver.

    Parameters
    ----------
    t : jnp.ndarray
        Time array for output
    vector_field : callable
        ODE vector field function
    y0 : jnp.ndarray
        Initial state vector
    args : dict
        Parameters for the vector field
    use_stiff_solver : bool
        If True, use tighter tolerances (1e-8/1e-10); otherwise standard (1e-6/1e-8)
    max_steps : int
        Maximum ODE steps

    Returns
    -------
    diffrax.Solution
        ODE solution
    """
    term = diffrax.ODETerm(vector_field)

    if use_stiff_solver:
        # Use Tsit5 instead of Kvaerno5 to avoid lineax LU transpose
        # TracerBoolConversionError during JAX tracing (NUTS/JIT)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)
    else:
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    t0 = t[0]
    t1 = t[-1]
    dt0 = (t1 - t0) / jnp.maximum(jnp.float64(t.shape[0]), 1000.0)

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


def _mask_failed_solution_ys(sol: diffrax.Solution) -> jnp.ndarray:
    """Extract sol.ys, masking with NaN if the solver failed.

    diffrax throw=False returns garbage ys when the solver hits max_steps
    or diverges. sol.result == 0 means success; nonzero means failure.
    The BayesianMixin NaN guard then rejects these parameter regions.
    """
    failed = sol.result != diffrax.RESULTS.successful
    return jnp.where(failed, jnp.nan, sol.ys)


# =============================================================================
# Public Solver Functions
# =============================================================================


def hvm_solve_startup(
    t: jnp.ndarray,
    gamma_dot: float,
    params: dict,
    kinetics: str = "stress",
    include_damage: bool = False,
    include_dissociative: bool = True,
) -> diffrax.Solution:
    """Solve startup shear ODE.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (s)
    gamma_dot : float
        Constant applied shear rate (1/s)
    params : dict
        Model parameters (G_P, G_E, G_D, k_d_D, nu_0, E_a, V_act, T,
        Gamma_0, lambda_crit)
    kinetics : str
        TST coupling type
    include_damage : bool
        Whether damage is active
    include_dissociative : bool
        Whether D-network is present

    Returns
    -------
    diffrax.Solution
        ODE solution with y shape (n_times, 11)
    """
    vf = _make_startup_vector_field(kinetics, include_damage, include_dissociative)
    y0 = _hvm_initial_state(include_dissociative, include_damage)

    args = {**params, "gamma_dot": gamma_dot}
    # Set defaults for optional params
    args.setdefault("G_D", 0.0)
    args.setdefault("k_d_D", 1.0)
    args.setdefault("Gamma_0", 0.0)
    args.setdefault("lambda_crit", 10.0)

    return _solve_hvm_ode(t, vf, y0, args, use_stiff_solver=False)


def hvm_solve_relaxation(
    t: jnp.ndarray,
    gamma_step: float,
    params: dict,
    kinetics: str = "stress",
    include_damage: bool = False,
    include_dissociative: bool = True,
) -> diffrax.Solution:
    """Solve stress relaxation ODE after step strain.

    The initial condition is set by applying an instantaneous step strain
    gamma_step to the equilibrium state. For small gamma_step (linear regime),
    the distribution tensors are:
    mu^E_xy(0) = gamma_step, mu^E_xx(0) ≈ 1 + 2*gamma_step^2
    mu^D_xy(0) = gamma_step, mu^D_xx(0) ≈ 1 + 2*gamma_step^2

    Parameters
    ----------
    t : jnp.ndarray
        Time array after step (s)
    gamma_step : float
        Applied step strain
    params : dict
        Model parameters
    kinetics : str
        TST coupling type
    include_damage : bool
        Whether damage is active
    include_dissociative : bool
        Whether D-network is present

    Returns
    -------
    diffrax.Solution
        ODE solution
    """
    vf = _make_relaxation_vector_field(kinetics, include_damage, include_dissociative)

    # Initial state: step strain applied to E and D networks
    # mu_xy = gamma_step (from affine deformation)
    # mu_xx ≈ 1 + 2*gamma_step^2 (from shear coupling)
    mu_xx_init = 1.0 + 2.0 * gamma_step**2
    mu_yy_init = 1.0
    mu_xy_init = gamma_step

    y0 = jnp.array(
        [
            mu_xx_init,
            mu_yy_init,
            mu_xy_init,  # E-network (deformed)
            1.0,
            1.0,
            0.0,  # E-network natural state (at equilibrium)
            mu_xx_init,
            mu_yy_init,
            mu_xy_init,  # D-network (deformed)
            gamma_step,  # gamma = gamma_step
            0.0,  # D = 0
        ],
        dtype=jnp.float64,
    )

    args = {**params}
    args.setdefault("G_D", 0.0)
    args.setdefault("k_d_D", 1.0)

    return _solve_hvm_ode(t, vf, y0, args, use_stiff_solver=False)


def hvm_solve_creep(
    t: jnp.ndarray,
    sigma_0: float,
    params: dict,
    kinetics: str = "stress",
    include_damage: bool = False,
    include_dissociative: bool = True,
) -> diffrax.Solution:
    """Solve creep ODE under constant stress.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (s)
    sigma_0 : float
        Applied constant stress (Pa)
    params : dict
        Model parameters
    kinetics : str
        TST coupling type
    include_damage : bool
        Whether damage is active
    include_dissociative : bool
        Whether D-network is present

    Returns
    -------
    diffrax.Solution
        ODE solution
    """
    vf = _make_creep_vector_field(kinetics, include_damage, include_dissociative)
    y0 = _hvm_initial_state(include_dissociative, include_damage)

    args = {**params, "sigma_0": sigma_0}
    args.setdefault("G_D", 0.0)
    args.setdefault("k_d_D", 1.0)
    args.setdefault("Gamma_0", 0.0)
    args.setdefault("lambda_crit", 10.0)

    return _solve_hvm_ode(t, vf, y0, args, use_stiff_solver=False)


def hvm_solve_laos(
    t: jnp.ndarray,
    gamma_0: float,
    omega: float,
    params: dict,
    kinetics: str = "stress",
    include_damage: bool = False,
    include_dissociative: bool = True,
) -> diffrax.Solution:
    """Solve LAOS ODE under oscillatory shear.

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
    kinetics : str
        TST coupling type
    include_damage : bool
        Whether damage is active
    include_dissociative : bool
        Whether D-network is present

    Returns
    -------
    diffrax.Solution
        ODE solution
    """
    vf = _make_laos_vector_field(kinetics, include_damage, include_dissociative)
    y0 = _hvm_initial_state(include_dissociative, include_damage)

    args = {**params, "gamma_0": gamma_0, "omega": omega}
    args.setdefault("G_D", 0.0)
    args.setdefault("k_d_D", 1.0)
    args.setdefault("Gamma_0", 0.0)
    args.setdefault("lambda_crit", 10.0)

    return _solve_hvm_ode(t, vf, y0, args, use_stiff_solver=False)
