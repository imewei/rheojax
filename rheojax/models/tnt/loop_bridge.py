"""TNT Loop-Bridge model for telechelic polymer networks.

This module implements `TNTLoopBridge`, a constitutive model for telechelic
polymers with reversible loop-bridge kinetics. Telechelic chains have
associating end-groups that form reversible junctions, and chains can exist
as either bridges (both ends attached to different junctions, load-bearing)
or loops (both ends on the same junction, non-load-bearing).

Key Physics
-----------
The bridge fraction f_B evolves dynamically via:

- **Association**: loops → bridges (rate: 1/τ_a)
- **Force-activated dissociation**: bridges → loops (rate: β(stretch))

The bridge fraction dynamics are coupled to the conformation tensor S
(tracking bridge configurations) via force-dependent breakage:

    df_B/dt = (1 - f_B)/τ_a - f_B·β(stretch)
    β(stretch) = (1/τ_b)·exp(ν·(stretch - 1))

where stretch = sqrt(tr(S)/3) represents the average bridge extension.

Only bridges contribute to stress: σ = f_B·G·S_xy + η_s·γ̇

Supported Protocols
-------------------
- FLOW_CURVE: Steady shear (ODE-to-steady-state)
- OSCILLATION: Small-amplitude oscillatory shear (linearized analytical)
- STARTUP: Transient stress growth (ODE)
- RELAXATION: Stress decay after cessation (ODE)
- CREEP: Strain evolution under constant stress (ODE)
- LAOS: Large-amplitude oscillatory shear (ODE)

Example
-------
>>> from rheojax.models.tnt import TNTLoopBridge
>>> import numpy as np
>>>
>>> # Create model with loop-bridge kinetics
>>> model = TNTLoopBridge()
>>>
>>> # Flow curve with shear thinning from force-dependent unbinding
>>> gamma_dot = np.logspace(-2, 2, 50)
>>> sigma = model.predict(gamma_dot, test_mode='flow_curve')
>>>
>>> # Fit to data
>>> model.fit(gamma_dot, sigma_data, test_mode='flow_curve')
>>> print(f"Bridge fraction: {model.f_B_eq}")
>>>
>>> # Startup with bridge fraction evolution
>>> t = np.linspace(0, 100, 500)
>>> stress, f_B = model.simulate_startup(t, gamma_dot=10.0, return_bridge_fraction=True)

References
----------
- Leibler, L., Rubinstein, M., & Colby, R.H. (1991). Macromolecules 24, 4701-4707.
- Tanaka, F. & Edwards, S.F. (1992). Macromolecules 25, 1516-1523.
- Bell, G.I. (1978). Science 200, 618-627.
"""

from __future__ import annotations

import logging
from typing import Literal

import diffrax
import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.models.tnt._base import TNTBase
from rheojax.models.tnt._kernels import (
    breakage_bell,
    tnt_saos_moduli,
    tnt_saos_moduli_vec,
    upper_convected_2d,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


# =============================================================================
# Loop-Bridge ODE Kernels
# =============================================================================


@jax.jit
def _loop_bridge_ode_rhs(
    t: float,
    state: jnp.ndarray,
    gamma_dot: float,
    G: float,
    tau_b: float,
    tau_a: float,
    nu: float,
) -> jnp.ndarray:
    """ODE right-hand side for loop-bridge dynamics.

    State: [f_B, S_xx, S_yy, S_zz, S_xy] (5 components)

    Equations:
        df_B/dt = (1 - f_B)/τ_a - f_B·β(stretch)
        dS/dt = L·S + S·L^T + g_0·I - β(stretch)·S

    where β(stretch) = (1/τ_b)·exp(ν·(stretch - 1))
    and g_0 = 1/τ_b (creation rate).

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [f_B, S_xx, S_yy, S_zz, S_xy]
    gamma_dot : float
        Applied shear rate (1/s)
    G : float
        Network modulus (Pa)
    tau_b : float
        Bridge detachment time (s)
    tau_a : float
        Loop attachment time (s)
    nu : float
        Bell force sensitivity (dimensionless)

    Returns
    -------
    jnp.ndarray
        Time derivatives [df_B/dt, dS_xx/dt, dS_yy/dt, dS_zz/dt, dS_xy/dt]
    """
    f_B = state[0]
    S_xx, S_yy, S_zz, S_xy = state[1], state[2], state[3], state[4]

    # Compute stretch from conformation tensor
    tr_S = S_xx + S_yy + S_zz
    stretch = jnp.sqrt(jnp.maximum(tr_S / 3.0, 0.0))

    # Bell force-dependent breakage rate
    beta = (1.0 / tau_b) * jnp.exp(nu * (stretch - 1.0))

    # Bridge fraction kinetics: attachment - force-activated dissociation
    df_B = (1.0 - f_B) / tau_a - f_B * beta

    # Conformation tensor evolution (bridges only)
    # Upper-convected derivative
    conv_xx, conv_yy, conv_xy = upper_convected_2d(S_xx, S_yy, S_xy, gamma_dot)

    # Creation rate (assumes equilibrium recovery: g_0 = 1/τ_b)
    g_0 = 1.0 / tau_b

    # dS/dt = conv + g_0·I - β·S
    dS_xx = conv_xx + g_0 - beta * S_xx
    dS_yy = conv_yy + g_0 - beta * S_yy
    dS_zz = g_0 - beta * S_zz  # No convective term for zz in simple shear
    dS_xy = conv_xy - beta * S_xy

    return jnp.array([df_B, dS_xx, dS_yy, dS_zz, dS_xy])


@jax.jit
def _loop_bridge_creep_ode_rhs(
    t: float,
    state: jnp.ndarray,
    sigma_applied: float,
    G: float,
    tau_b: float,
    tau_a: float,
    nu: float,
    f_B_eq: float,
    eta_s: float,
) -> jnp.ndarray:
    """ODE right-hand side for loop-bridge creep (stress-controlled).

    State: [f_B, S_xx, S_yy, S_zz, S_xy, gamma] (6 components)

    The applied stress is held constant:
        σ = f_B·G·S_xy + η_s·γ̇
        γ̇ = (σ - f_B·G·S_xy) / η_s

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [f_B, S_xx, S_yy, S_zz, S_xy, gamma]
    sigma_applied : float
        Applied constant stress (Pa)
    G : float
        Network modulus (Pa)
    tau_b : float
        Bridge detachment time (s)
    tau_a : float
        Loop attachment time (s)
    nu : float
        Bell force sensitivity
    f_B_eq : float
        Equilibrium bridge fraction (unused in dynamics, for reference)
    eta_s : float
        Solvent viscosity (Pa·s)

    Returns
    -------
    jnp.ndarray
        Time derivatives [df_B/dt, dS_xx/dt, dS_yy/dt, dS_zz/dt, dS_xy/dt, dγ/dt]
    """
    f_B = state[0]
    S_xx, S_yy, S_zz, S_xy = state[1], state[2], state[3], state[4]
    gamma = state[5]

    # Elastic stress from bridges
    sigma_elastic = f_B * G * S_xy

    # Shear rate from stress constraint
    eta_s_reg = jnp.maximum(eta_s, 1e-10 * G * tau_b)
    gamma_dot = (sigma_applied - sigma_elastic) / eta_s_reg

    # Conformation and bridge fraction evolution (reuse rate-controlled RHS)
    conf_state = jnp.array([f_B, S_xx, S_yy, S_zz, S_xy])
    d_conf = _loop_bridge_ode_rhs(t, conf_state, gamma_dot, G, tau_b, tau_a, nu)

    # Strain evolution
    d_gamma = gamma_dot

    return jnp.concatenate([d_conf, jnp.array([d_gamma])])


@jax.jit
def _loop_bridge_laos_ode_rhs(
    t: float,
    state: jnp.ndarray,
    gamma_0: float,
    omega: float,
    G: float,
    tau_b: float,
    tau_a: float,
    nu: float,
) -> jnp.ndarray:
    """ODE right-hand side for loop-bridge LAOS.

    Oscillatory shear: γ̇(t) = γ₀·ω·cos(ωt)

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [f_B, S_xx, S_yy, S_zz, S_xy]
    gamma_0 : float
        Strain amplitude
    omega : float
        Angular frequency (rad/s)
    G : float
        Network modulus (Pa)
    tau_b : float
        Bridge detachment time (s)
    tau_a : float
        Loop attachment time (s)
    nu : float
        Bell force sensitivity

    Returns
    -------
    jnp.ndarray
        Time derivatives
    """
    gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
    return _loop_bridge_ode_rhs(t, state, gamma_dot, G, tau_b, tau_a, nu)


@jax.jit
def _loop_bridge_relaxation_ode_rhs(
    t: float,
    state: jnp.ndarray,
    G: float,
    tau_b: float,
    tau_a: float,
    nu: float,
) -> jnp.ndarray:
    """ODE right-hand side for loop-bridge relaxation (γ̇ = 0).

    Parameters
    ----------
    t : float
        Time (s)
    state : jnp.ndarray
        State vector [f_B, S_xx, S_yy, S_zz, S_xy]
    G : float
        Network modulus (Pa)
    tau_b : float
        Bridge detachment time (s)
    tau_a : float
        Loop attachment time (s)
    nu : float
        Bell force sensitivity

    Returns
    -------
    jnp.ndarray
        Time derivatives
    """
    return _loop_bridge_ode_rhs(t, state, 0.0, G, tau_b, tau_a, nu)


# =============================================================================
# TNTLoopBridge Model Class
# =============================================================================


@ModelRegistry.register(
    "tnt_loop_bridge",
    protocols=["flow_curve", "oscillation", "startup", "relaxation", "creep", "laos"],
)
class TNTLoopBridge(TNTBase):
    """Loop-bridge kinetics model for telechelic polymer networks.

    Implements reversible loop-bridge kinetics for telechelic polymers where
    chains can exist as load-bearing bridges (both ends on different junctions)
    or non-load-bearing loops (both ends on same junction).

    The bridge fraction f_B evolves dynamically via attachment (loops → bridges)
    and force-activated dissociation (bridges → loops via Bell kinetics).

    State Variables
    ---------------
    - f_B: Bridge fraction (0 to 1)
    - S: Conformation tensor of bridges [S_xx, S_yy, S_zz, S_xy]

    Key Equations
    -------------
    Bridge fraction kinetics::

        df_B/dt = (1 - f_B)/τ_a - f_B·β(stretch)
        β(stretch) = (1/τ_b)·exp(ν·(stretch - 1))

    Conformation tensor (bridges only)::

        dS/dt = L·S + S·L^T + g_0·I - β(stretch)·S

    Stress (only bridges carry load)::

        σ = f_B·G·S_xy + η_s·γ̇

    Parameters
    ----------
    G : float, default 1e3
        Network modulus (fully bridged, Pa), bounds (1e0, 1e8)
    tau_b : float, default 1.0
        Bridge detachment time (s), bounds (1e-6, 1e4)
    tau_a : float, default 5.0
        Loop attachment time (s), bounds (1e-6, 1e4)
    nu : float, default 1.0
        Bell force sensitivity (dimensionless), bounds (0.01, 20)
    f_B_eq : float, default 0.5
        Equilibrium bridge fraction, bounds (0.01, 0.99)
    eta_s : float, default 0.0
        Solvent viscosity (Pa·s), bounds (0.0, 1e4)

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters
    fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    Basic usage:

    >>> model = TNTLoopBridge()
    >>> gamma_dot = np.logspace(-2, 2, 50)
    >>> sigma = model.predict(gamma_dot, test_mode='flow_curve')

    Startup with bridge fraction tracking:

    >>> t = np.linspace(0, 100, 500)
    >>> stress, f_B = model.simulate_startup(
    ...     t, gamma_dot=10.0, return_bridge_fraction=True
    ... )

    See Also
    --------
    TNTSingleMode : Basic single-mode TNT (constant breakage)
    TNTCates : Living polymer (wormlike micelle) model
    """

    def __init__(self):
        """Initialize TNT loop-bridge model."""
        super().__init__()
        self._setup_parameters()
        self._test_mode = None

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with loop-bridge parameters.

        Parameters:
        - G: Network modulus (Pa)
        - tau_b: Bridge detachment time (s)
        - tau_a: Loop attachment time (s)
        - nu: Bell force sensitivity (dimensionless)
        - f_B_eq: Equilibrium bridge fraction (dimensionless)
        - eta_s: Solvent viscosity (Pa·s)
        """
        self.parameters = ParameterSet()

        self.parameters.add(
            name="G",
            value=1e3,
            bounds=(1e0, 1e8),
            units="Pa",
            description="Network modulus (fully bridged state)",
        )
        self.parameters.add(
            name="tau_b",
            value=1.0,
            bounds=(1e-6, 1e4),
            units="s",
            description="Bridge detachment time (mean lifetime of bridge bond)",
        )
        self.parameters.add(
            name="tau_a",
            value=5.0,
            bounds=(1e-6, 1e4),
            units="s",
            description="Loop attachment time (mean time for loop-to-bridge conversion)",
        )
        self.parameters.add(
            name="nu",
            value=1.0,
            bounds=(0.01, 20.0),
            units="dimensionless",
            description="Bell force sensitivity (higher = more shear-thinning)",
        )
        self.parameters.add(
            name="f_B_eq",
            value=0.5,
            bounds=(0.01, 0.99),
            units="dimensionless",
            description="Equilibrium bridge fraction at rest",
        )
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e4),
            units="Pa·s",
            description="Solvent viscosity (Newtonian background contribution)",
        )

    # =========================================================================
    # Property Accessors
    # =========================================================================

    @property
    def G(self) -> float:
        """Get network modulus G (Pa)."""
        return float(self.parameters.get_value("G"))

    @property
    def tau_b(self) -> float:
        """Get bridge detachment time τ_b (s)."""
        return float(self.parameters.get_value("tau_b"))

    @property
    def tau_a(self) -> float:
        """Get loop attachment time τ_a (s)."""
        return float(self.parameters.get_value("tau_a"))

    @property
    def nu(self) -> float:
        """Get Bell force sensitivity ν (dimensionless)."""
        return float(self.parameters.get_value("nu"))

    @property
    def f_B_eq(self) -> float:
        """Get equilibrium bridge fraction f_B_eq (dimensionless)."""
        return float(self.parameters.get_value("f_B_eq"))

    @property
    def eta_s(self) -> float:
        """Get solvent viscosity η_s (Pa·s)."""
        return float(self.parameters.get_value("eta_s"))

    @property
    def G_eff(self) -> float:
        """Get effective modulus G_eff = f_B_eq·G (Pa).

        This is the linearized modulus at equilibrium.
        """
        return self.f_B_eq * self.G

    @property
    def eta_0(self) -> float:
        """Get zero-shear viscosity η₀ = f_B_eq·G·τ_b + η_s (Pa·s)."""
        return self.f_B_eq * self.G * self.tau_b + self.eta_s

    # =========================================================================
    # Equilibrium State
    # =========================================================================

    def get_equilibrium_state(self) -> jnp.ndarray:
        """Return equilibrium state [f_B_eq, 1, 1, 1, 0].

        At rest: f_B = f_B_eq, S = I (unstretched, isotropic)

        Returns
        -------
        jnp.ndarray
            Equilibrium state [f_B, S_xx, S_yy, S_zz, S_xy]
        """
        return jnp.array(
            [self.f_B_eq, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float64
        )

    # =========================================================================
    # Core Interface Methods
    # =========================================================================

    def _fit(self, x, y, **kwargs):
        """Fit model to data using protocol-aware optimization.

        Parameters
        ----------
        x : array-like
            Independent variable (shear rate, frequency, or time)
        y : array-like
            Dependent variable (stress, modulus, or strain)
        **kwargs
            Additional arguments including test_mode

        Returns
        -------
        self
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        self._test_mode = test_mode

        x_jax = jnp.asarray(x, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        # Store protocol-specific inputs
        self._gamma_dot_applied = kwargs.get("gamma_dot")
        self._sigma_applied = kwargs.get("sigma_applied")
        self._gamma_0 = kwargs.get("gamma_0")
        self._omega_laos = kwargs.get("omega")

        # Smart initialization based on protocol
        if test_mode in ["flow_curve", "steady_shear", "rotation"]:
            self.initialize_from_flow_curve(
                np.asarray(x), np.asarray(y)
            )
        elif test_mode == "oscillation":
            self.initialize_from_saos(
                np.asarray(x), np.real(np.asarray(y)), np.imag(np.asarray(y))
            )

        # Define model function for fitting
        def model_fn(x_fit, params):
            return self.model_function(x_fit, params, test_mode=test_mode)

        # Create objective and optimize
        objective = create_least_squares_objective(
            model_fn,
            x_jax,
            y_jax,
            use_log_residuals=kwargs.get(
                "use_log_residuals", test_mode == "flow_curve"
            ),
        )

        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            method=kwargs.get("method", "auto"),
            max_iter=kwargs.get("max_iter", 2000),
        )

        self.fitted_ = True
        self._nlsq_result = result

        logger.info(
            f"Fitted TNTLoopBridge: G={self.G:.2e}, τ_b={self.tau_b:.2e}, "
            f"τ_a={self.tau_a:.2e}, f_B_eq={self.f_B_eq:.3f}"
        )

        return self

    def _predict(self, x, **kwargs):
        """Predict response using protocol-aware dispatch.

        Parameters
        ----------
        x : array-like
            Independent variable
        **kwargs
            Additional arguments including test_mode

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        x_jax = jnp.asarray(x, dtype=jnp.float64)

        # Build parameter array from ParameterSet (ordering matters)
        param_values = [
            float(self.parameters.get_value(name))
            for name in self.parameters.keys()
        ]
        params = jnp.array(param_values)
        return self.model_function(x_jax, params, test_mode=test_mode)

    def model_function(self, X, params, test_mode=None):
        """NumPyro/BayesianMixin model function.

        Routes to appropriate prediction based on test_mode. This is the
        stateless function used for both NLSQ optimization and Bayesian
        inference.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values: [G, tau_b, tau_a, nu, f_B_eq, eta_s]
        test_mode : str, optional
            Override stored test mode

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        # Unpack parameters
        G = params[0]
        tau_b = params[1]
        tau_a = params[2]
        nu = params[3]
        f_B_eq = params[4]
        eta_s = params[5]

        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return self._flow_curve_internal(X_jax, G, tau_b, tau_a, nu, f_B_eq, eta_s)

        elif mode == "oscillation":
            # Linearized SAOS: effective Maxwell with G_eff = f_B_eq·G, τ_eff = τ_b
            G_prime, G_double_prime = tnt_saos_moduli_vec(
                X_jax, f_B_eq * G, tau_b, eta_s
            )
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            gamma_dot = self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return self._simulate_startup_internal(
                X_jax, G, tau_b, tau_a, nu, f_B_eq, eta_s, gamma_dot
            )

        elif mode == "relaxation":
            gamma_dot = self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError(
                    "relaxation mode requires gamma_dot (pre-shear rate)"
                )
            return self._simulate_relaxation_internal(
                X_jax, G, tau_b, tau_a, nu, f_B_eq, eta_s, gamma_dot
            )

        elif mode == "creep":
            sigma_applied = self._sigma_applied
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            return self._simulate_creep_internal(
                X_jax, G, tau_b, tau_a, nu, f_B_eq, eta_s, sigma_applied
            )

        elif mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, G, tau_b, tau_a, nu, f_B_eq, eta_s,
                self._gamma_0, self._omega_laos
            )
            return stress

        else:
            logger.warning(
                f"Unknown test_mode '{mode}', defaulting to flow_curve"
            )
            return self._flow_curve_internal(X_jax, G, tau_b, tau_a, nu, f_B_eq, eta_s)

    # =========================================================================
    # Flow Curve (ODE-to-steady-state)
    # =========================================================================

    def _flow_curve_internal(
        self,
        gamma_dot_arr: jnp.ndarray,
        G: float,
        tau_b: float,
        tau_a: float,
        nu: float,
        f_B_eq: float,
        eta_s: float,
    ) -> jnp.ndarray:
        """Compute flow curve by running ODE to steady state.

        For each shear rate, integrate ODE for t_end = 50·max(τ_a, τ_b)
        starting from equilibrium [f_B_eq, 1, 1, 1, 0].

        Parameters
        ----------
        gamma_dot_arr : jnp.ndarray
            Shear rate array (1/s)
        G : float
            Network modulus (Pa)
        tau_b : float
            Bridge detachment time (s)
        tau_a : float
            Loop attachment time (s)
        nu : float
            Bell force sensitivity
        f_B_eq : float
            Equilibrium bridge fraction
        eta_s : float
            Solvent viscosity (Pa·s)

        Returns
        -------
        jnp.ndarray
            Steady shear stress array (Pa)
        """

        def solve_single(gdot):
            """Solve for steady-state stress at a single shear rate."""

            def ode_fn(ti, yi, args):
                return _loop_bridge_ode_rhs(
                    ti, yi, args["gdot"], args["G"], args["tau_b"],
                    args["tau_a"], args["nu"]
                )

            args = {
                "gdot": gdot,
                "G": G,
                "tau_b": tau_b,
                "tau_a": tau_a,
                "nu": nu,
            }
            y0 = jnp.array([f_B_eq, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

            # Run for 50·max(τ_a, τ_b) to reach steady state
            t_end = 50.0 * jnp.maximum(tau_a, tau_b)
            dt0 = jnp.maximum(tau_a, tau_b) / 20.0

            term = diffrax.ODETerm(ode_fn)
            solver = diffrax.Tsit5()  # Explicit solver for vmap compatibility
            controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
            saveat = diffrax.SaveAt(ts=jnp.array([t_end]))

            sol = diffrax.diffeqsolve(
                term,
                solver,
                0.0,
                t_end,
                dt0,
                y0,
                args=args,
                saveat=saveat,
                stepsize_controller=controller,
                max_steps=500_000,
            )

            # Extract final state
            state_final = sol.ys[0]
            f_B_final = state_final[0]
            S_xy_final = state_final[4]

            # Stress: σ = f_B·G·S_xy + η_s·γ̇
            sigma = f_B_final * G * S_xy_final + eta_s * gdot

            return sigma

        return jax.vmap(solve_single)(gamma_dot_arr)

    def _steady_state_conformation(
        self,
        gamma_dot_arr: jnp.ndarray,
        G: float,
        tau_b: float,
        tau_a: float,
        nu: float,
        f_B_eq: float,
    ) -> jnp.ndarray:
        """Compute steady-state [f_B, S] via ODE.

        Returns array of shape (N, 5) with [f_B, S_xx, S_yy, S_zz, S_xy]
        for each shear rate.

        Parameters
        ----------
        gamma_dot_arr : jnp.ndarray
            Shear rate array (1/s)
        G : float
            Network modulus (Pa)
        tau_b : float
            Bridge detachment time (s)
        tau_a : float
            Loop attachment time (s)
        nu : float
            Bell force sensitivity
        f_B_eq : float
            Equilibrium bridge fraction

        Returns
        -------
        jnp.ndarray
            Steady-state array, shape (N, 5)
        """

        def solve_single(gdot):
            """Solve for steady-state conformation at a single shear rate."""

            def ode_fn(ti, yi, args):
                return _loop_bridge_ode_rhs(
                    ti, yi, args["gdot"], args["G"], args["tau_b"],
                    args["tau_a"], args["nu"]
                )

            args = {
                "gdot": gdot,
                "G": G,
                "tau_b": tau_b,
                "tau_a": tau_a,
                "nu": nu,
            }
            y0 = jnp.array([f_B_eq, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

            t_end = 50.0 * jnp.maximum(tau_a, tau_b)
            dt0 = jnp.maximum(tau_a, tau_b) / 20.0

            term = diffrax.ODETerm(ode_fn)
            solver = diffrax.Tsit5()
            controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
            saveat = diffrax.SaveAt(ts=jnp.array([t_end]))

            sol = diffrax.diffeqsolve(
                term,
                solver,
                0.0,
                t_end,
                dt0,
                y0,
                args=args,
                saveat=saveat,
                stepsize_controller=controller,
                max_steps=500_000,
            )

            return sol.ys[0]

        return jax.vmap(solve_single)(gamma_dot_arr)

    # =========================================================================
    # ODE-Based Internal Simulations (for model_function)
    # =========================================================================

    def _simulate_startup_internal(
        self,
        t: jnp.ndarray,
        G: float,
        tau_b: float,
        tau_a: float,
        nu: float,
        f_B_eq: float,
        eta_s: float,
        gamma_dot: float,
    ) -> jnp.ndarray:
        """Internal startup simulation for model_function.

        Returns total shear stress σ(t).
        """

        def ode_fn(ti, yi, args):
            return _loop_bridge_ode_rhs(
                ti, yi, args["gamma_dot"], args["G"], args["tau_b"],
                args["tau_a"], args["nu"]
            )

        args = {
            "gamma_dot": gamma_dot,
            "G": G,
            "tau_b": tau_b,
            "tau_a": tau_a,
            "nu": nu,
        }
        y0 = jnp.array([f_B_eq, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t[0])
        t1 = float(t[-1])
        dt0 = (t1 - t0) / max(len(t), 1000)

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
            max_steps=500_000,
        )

        # Stress: σ = f_B·G·S_xy + η_s·γ̇
        f_B_t = sol.ys[:, 0]
        S_xy_t = sol.ys[:, 4]
        sigma = f_B_t * G * S_xy_t + eta_s * gamma_dot

        return sigma

    def _simulate_relaxation_internal(
        self,
        t: jnp.ndarray,
        G: float,
        tau_b: float,
        tau_a: float,
        nu: float,
        f_B_eq: float,
        eta_s: float,
        gamma_dot_preshear: float,
    ) -> jnp.ndarray:
        """Internal relaxation simulation for model_function.

        Returns relaxing stress σ(t).
        """
        # First find steady-state conformation from pre-shear
        ss = self._steady_state_conformation(
            jnp.array([gamma_dot_preshear]), G, tau_b, tau_a, nu, f_B_eq
        )
        y0 = ss[0]  # [f_B_0, S_xx_0, S_yy_0, S_zz_0, S_xy_0]

        def ode_fn(ti, yi, args):
            return _loop_bridge_relaxation_ode_rhs(
                ti, yi, args["G"], args["tau_b"], args["tau_a"], args["nu"]
            )

        args = {"G": G, "tau_b": tau_b, "tau_a": tau_a, "nu": nu}

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t[0])
        t1 = float(t[-1])
        dt0 = (t1 - t0) / max(len(t), 1000)

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
            max_steps=500_000,
        )

        # Stress: σ = f_B·G·S_xy (γ̇=0 in relaxation)
        f_B_t = sol.ys[:, 0]
        S_xy_t = sol.ys[:, 4]
        sigma = f_B_t * G * S_xy_t

        return sigma

    def _simulate_creep_internal(
        self,
        t: jnp.ndarray,
        G: float,
        tau_b: float,
        tau_a: float,
        nu: float,
        f_B_eq: float,
        eta_s: float,
        sigma_applied: float,
    ) -> jnp.ndarray:
        """Internal creep simulation for model_function.

        Returns accumulated strain γ(t).
        """

        def ode_fn(ti, yi, args):
            return _loop_bridge_creep_ode_rhs(
                ti,
                yi,
                args["sigma_applied"],
                args["G"],
                args["tau_b"],
                args["tau_a"],
                args["nu"],
                args["f_B_eq"],
                args["eta_s"],
            )

        args = {
            "sigma_applied": sigma_applied,
            "G": G,
            "tau_b": tau_b,
            "tau_a": tau_a,
            "nu": nu,
            "f_B_eq": f_B_eq,
            "eta_s": eta_s,
        }
        y0 = jnp.array([f_B_eq, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)

        t0 = float(t[0])
        t1 = float(t[-1])
        dt0 = (t1 - t0) / max(len(t), 10000)  # Smaller dt for creep

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
            max_steps=1_000_000,
        )

        return sol.ys[:, 5]  # γ (strain)

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        G: float,
        tau_b: float,
        tau_a: float,
        nu: float,
        f_B_eq: float,
        eta_s: float,
        gamma_0: float,
        omega: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal LAOS simulation for model_function.

        Returns (strain, stress) arrays.
        """

        def ode_fn(ti, yi, args):
            return _loop_bridge_laos_ode_rhs(
                ti,
                yi,
                args["gamma_0"],
                args["omega"],
                args["G"],
                args["tau_b"],
                args["tau_a"],
                args["nu"],
            )

        args = {
            "gamma_0": gamma_0,
            "omega": omega,
            "G": G,
            "tau_b": tau_b,
            "tau_a": tau_a,
            "nu": nu,
        }
        y0 = jnp.array([f_B_eq, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t[0])
        t1 = float(t[-1])
        dt0 = (t1 - t0) / max(len(t), 1000)

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
            max_steps=500_000,
        )

        strain = gamma_0 * jnp.sin(omega * t)
        gamma_dot_t = gamma_0 * omega * jnp.cos(omega * t)

        # Stress: σ = f_B·G·S_xy + η_s·γ̇(t)
        f_B_t = sol.ys[:, 0]
        S_xy_t = sol.ys[:, 4]
        stress = f_B_t * G * S_xy_t + eta_s * gamma_dot_t

        return strain, stress

    # =========================================================================
    # Public Prediction Methods (return numpy arrays)
    # =========================================================================

    def predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict steady shear stress and viscosity.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        return_components : bool, default False
            If True, return (sigma, eta, N1)

        Returns
        -------
        np.ndarray or tuple
            Shear stress σ (Pa), or (σ, η, N₁) if return_components=True
        """
        gd = jnp.asarray(gamma_dot, dtype=jnp.float64)

        sigma = self._flow_curve_internal(
            gd, self.G, self.tau_b, self.tau_a, self.nu, self.f_B_eq, self.eta_s
        )

        if return_components:
            eta = sigma / jnp.maximum(gd, 1e-20)

            # N1 from steady-state conformation
            ss = self._steady_state_conformation(
                gd, self.G, self.tau_b, self.tau_a, self.nu, self.f_B_eq
            )
            f_B_ss = ss[:, 0]
            S_xx_ss = ss[:, 1]
            S_yy_ss = ss[:, 2]
            N1 = f_B_ss * self.G * (S_xx_ss - S_yy_ss)

            return np.asarray(sigma), np.asarray(eta), np.asarray(N1)

        return np.asarray(sigma)

    def predict_saos(
        self,
        omega: np.ndarray,
        return_components: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

        In the linear regime, loop-bridge model reduces to effective Maxwell:
        G_eff = f_B_eq·G, τ_eff = τ_b

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency array (rad/s)
        return_components : bool, default True
            If True, return (G', G'')

        Returns
        -------
        tuple or np.ndarray
            (G', G'') if return_components=True, else |G*|
        """
        w = jnp.asarray(omega, dtype=jnp.float64)
        G_prime, G_double_prime = tnt_saos_moduli_vec(
            w, self.G_eff, self.tau_b, self.eta_s
        )

        if return_components:
            return np.asarray(G_prime), np.asarray(G_double_prime)

        G_star_mag = jnp.sqrt(G_prime**2 + G_double_prime**2)
        return np.asarray(G_star_mag)

    def predict_normal_stresses(
        self,
        gamma_dot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict first and second normal stress differences.

        N₁ = f_B·G·(S_xx - S_yy)
        N₂ = 0 (upper-convected derivative)

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (N₁, N₂) in Pa
        """
        gd = jnp.asarray(gamma_dot, dtype=jnp.float64)

        # Compute from steady-state conformation
        ss = self._steady_state_conformation(
            gd, self.G, self.tau_b, self.tau_a, self.nu, self.f_B_eq
        )

        f_B_ss = ss[:, 0]
        S_xx_ss = ss[:, 1]
        S_yy_ss = ss[:, 2]
        S_zz_ss = ss[:, 3]

        N1 = f_B_ss * self.G * (S_xx_ss - S_yy_ss)
        N2 = f_B_ss * self.G * (S_yy_ss - S_zz_ss)  # Typically ~0 for UCM

        return np.asarray(N1), np.asarray(N2)

    # =========================================================================
    # Public Simulation Methods (return numpy arrays + trajectories)
    # =========================================================================

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_bridge_fraction: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Simulate startup flow at constant shear rate.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_bridge_fraction : bool, default False
            If True, also return f_B(t)

        Returns
        -------
        np.ndarray or tuple
            Shear stress σ(t), or (σ, f_B) if return_bridge_fraction=True
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return _loop_bridge_ode_rhs(
                ti, yi, args["gamma_dot"], args["G"], args["tau_b"],
                args["tau_a"], args["nu"]
            )

        args = {
            "gamma_dot": gamma_dot,
            "G": self.G,
            "tau_b": self.tau_b,
            "tau_a": self.tau_a,
            "nu": self.nu,
        }
        y0 = jnp.array(
            [self.f_B_eq, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float64
        )

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t_jax[0])
        t1 = float(t_jax[-1])
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t_jax)

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
            max_steps=500_000,
        )

        self._trajectory = {
            "t": np.asarray(t_jax),
            "f_B": np.asarray(sol.ys[:, 0]),
            "S_xx": np.asarray(sol.ys[:, 1]),
            "S_yy": np.asarray(sol.ys[:, 2]),
            "S_zz": np.asarray(sol.ys[:, 3]),
            "S_xy": np.asarray(sol.ys[:, 4]),
        }

        # Stress: σ = f_B·G·S_xy + η_s·γ̇
        f_B_t = sol.ys[:, 0]
        S_xy_t = sol.ys[:, 4]
        sigma = f_B_t * self.G * S_xy_t + self.eta_s * gamma_dot

        if return_bridge_fraction:
            return np.asarray(sigma), np.asarray(f_B_t)

        return np.asarray(sigma)

    def simulate_relaxation(
        self,
        t: np.ndarray,
        gamma_dot_preshear: float,
        return_bridge_fraction: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Simulate stress relaxation after cessation of steady shear.

        Parameters
        ----------
        t : np.ndarray
            Time array (s), starting from t=0 (cessation)
        gamma_dot_preshear : float
            Shear rate before cessation (1/s)
        return_bridge_fraction : bool, default False
            If True, also return f_B(t)

        Returns
        -------
        np.ndarray or tuple
            Relaxing stress σ(t), or (σ, f_B) if return_bridge_fraction=True
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)

        # Find steady-state conformation from pre-shear
        ss = self._steady_state_conformation(
            jnp.array([gamma_dot_preshear]),
            self.G,
            self.tau_b,
            self.tau_a,
            self.nu,
            self.f_B_eq,
        )
        y0 = ss[0]

        def ode_fn(ti, yi, args):
            return _loop_bridge_relaxation_ode_rhs(
                ti, yi, args["G"], args["tau_b"], args["tau_a"], args["nu"]
            )

        args = {
            "G": self.G,
            "tau_b": self.tau_b,
            "tau_a": self.tau_a,
            "nu": self.nu,
        }

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t_jax[0])
        t1 = float(t_jax[-1])
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t_jax)

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
            max_steps=500_000,
        )

        self._trajectory = {
            "t": np.asarray(t_jax),
            "f_B": np.asarray(sol.ys[:, 0]),
            "S_xx": np.asarray(sol.ys[:, 1]),
            "S_yy": np.asarray(sol.ys[:, 2]),
            "S_zz": np.asarray(sol.ys[:, 3]),
            "S_xy": np.asarray(sol.ys[:, 4]),
        }

        # Stress: σ = f_B·G·S_xy (γ̇=0 in relaxation)
        f_B_t = sol.ys[:, 0]
        S_xy_t = sol.ys[:, 4]
        sigma = f_B_t * self.G * S_xy_t

        if return_bridge_fraction:
            return np.asarray(sigma), np.asarray(f_B_t)

        return np.asarray(sigma)

    def simulate_creep(
        self,
        t: np.ndarray,
        sigma_applied: float,
        return_rate: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Simulate creep deformation under constant stress.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma_applied : float
            Applied constant stress (Pa)
        return_rate : bool, default False
            If True, also return shear rate γ̇(t)

        Returns
        -------
        np.ndarray or tuple
            Strain γ(t), or (γ, γ̇) if return_rate=True
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return _loop_bridge_creep_ode_rhs(
                ti,
                yi,
                args["sigma_applied"],
                args["G"],
                args["tau_b"],
                args["tau_a"],
                args["nu"],
                args["f_B_eq"],
                args["eta_s"],
            )

        args = {
            "sigma_applied": sigma_applied,
            "G": self.G,
            "tau_b": self.tau_b,
            "tau_a": self.tau_a,
            "nu": self.nu,
            "f_B_eq": self.f_B_eq,
            "eta_s": self.eta_s,
        }
        y0 = jnp.array(
            [self.f_B_eq, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float64
        )

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)

        t0 = float(t_jax[0])
        t1 = float(t_jax[-1])
        dt0 = (t1 - t0) / max(len(t), 10000)

        saveat = diffrax.SaveAt(ts=t_jax)

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
            max_steps=1_000_000,
        )

        gamma = np.asarray(sol.ys[:, 5])
        f_B_t = np.asarray(sol.ys[:, 0])
        S_xy_t = np.asarray(sol.ys[:, 4])

        self._trajectory = {
            "t": np.asarray(t_jax),
            "gamma": gamma,
            "f_B": f_B_t,
            "S_xy": S_xy_t,
        }

        if return_rate:
            eta_s_reg = max(self.eta_s, 1e-10 * self.G * self.tau_b)
            sigma_elastic = f_B_t * self.G * S_xy_t
            gamma_dot = (sigma_applied - sigma_elastic) / eta_s_reg
            return gamma, gamma_dot

        return gamma

    def simulate_laos(
        self,
        t: np.ndarray,
        gamma_0: float,
        omega: float,
        n_cycles: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Simulate Large-Amplitude Oscillatory Shear (LAOS).

        Parameters
        ----------
        t : np.ndarray
            Time array (s), or None to auto-generate
        gamma_0 : float
            Strain amplitude (dimensionless)
        omega : float
            Angular frequency (rad/s)
        n_cycles : int, optional
            Number of oscillation cycles (overrides t)

        Returns
        -------
        dict
            Dictionary with keys: 't', 'strain', 'stress', 'strain_rate', 'f_B'
        """
        if n_cycles is not None:
            T = 2 * np.pi / omega
            t = np.linspace(0, n_cycles * T, n_cycles * 200)

        t_jax = jnp.asarray(t, dtype=jnp.float64)

        strain, stress = self._simulate_laos_internal(
            t_jax,
            self.G,
            self.tau_b,
            self.tau_a,
            self.nu,
            self.f_B_eq,
            self.eta_s,
            gamma_0,
            omega,
        )

        strain_rate = gamma_0 * omega * jnp.cos(omega * t_jax)

        # Re-run for trajectory storage
        def ode_fn(ti, yi, args):
            return _loop_bridge_laos_ode_rhs(
                ti,
                yi,
                args["gamma_0"],
                args["omega"],
                args["G"],
                args["tau_b"],
                args["tau_a"],
                args["nu"],
            )

        args = {
            "gamma_0": gamma_0,
            "omega": omega,
            "G": self.G,
            "tau_b": self.tau_b,
            "tau_a": self.tau_a,
            "nu": self.nu,
        }
        y0 = jnp.array(
            [self.f_B_eq, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float64
        )

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t_jax[0])
        t1 = float(t_jax[-1])
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t_jax)

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
            max_steps=500_000,
        )

        self._trajectory = {
            "t": np.asarray(t_jax),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "strain_rate": np.asarray(strain_rate),
            "f_B": np.asarray(sol.ys[:, 0]),
        }

        return {
            "t": np.asarray(t_jax),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "strain_rate": np.asarray(strain_rate),
            "f_B": np.asarray(sol.ys[:, 0]),
        }

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_bridge_fraction_vs_rate(
        self,
        gamma_dot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute steady-state bridge fraction vs shear rate.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (gamma_dot, f_B_steady)
        """
        gd = jnp.asarray(gamma_dot, dtype=jnp.float64)

        ss = self._steady_state_conformation(
            gd, self.G, self.tau_b, self.tau_a, self.nu, self.f_B_eq
        )
        f_B_ss = ss[:, 0]

        return np.asarray(gamma_dot), np.asarray(f_B_ss)

    def extract_laos_harmonics(
        self,
        laos_result: dict[str, np.ndarray],
        n_harmonics: int = 5,
    ) -> dict[str, np.ndarray]:
        """Extract Fourier harmonics from LAOS stress response.

        Parameters
        ----------
        laos_result : dict
            Result from simulate_laos()
        n_harmonics : int, default 5
            Number of harmonics to extract

        Returns
        -------
        dict
            Dictionary with 'n', 'sigma_prime', 'sigma_double_prime',
            'intensity', 'I3_I1'
        """
        t = laos_result["t"]
        stress = laos_result["stress"]
        strain = laos_result["strain"]

        fft_strain = np.fft.fft(strain)
        freqs = np.fft.fftfreq(len(t), t[1] - t[0])
        omega = 2 * np.pi * np.abs(
            freqs[np.argmax(np.abs(fft_strain[1:])) + 1]
        )

        harmonics = [2 * i + 1 for i in range(n_harmonics)]
        sigma_prime = []
        sigma_double_prime = []

        for n in harmonics:
            sin_basis = np.sin(n * omega * t)
            cos_basis = np.cos(n * omega * t)

            dt = t[1] - t[0]
            sigma_n_prime = (
                2 * np.trapezoid(stress * sin_basis, dx=dt) / (t[-1] - t[0])
            )
            sigma_n_double_prime = (
                2 * np.trapezoid(stress * cos_basis, dx=dt) / (t[-1] - t[0])
            )

            sigma_prime.append(sigma_n_prime)
            sigma_double_prime.append(sigma_n_double_prime)

        sigma_prime = np.array(sigma_prime)
        sigma_double_prime = np.array(sigma_double_prime)
        intensity = np.sqrt(sigma_prime**2 + sigma_double_prime**2)

        return {
            "n": np.array(harmonics),
            "sigma_prime": sigma_prime,
            "sigma_double_prime": sigma_double_prime,
            "intensity": intensity,
            "I3_I1": (
                intensity[1] / intensity[0] if intensity[0] > 0 else 0.0
            ),
        }
