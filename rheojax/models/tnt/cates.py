"""TNT model for living polymers (Cates, wormlike micelles).

This module implements `TNTCates`, a constitutive model for living polymers
(wormlike micelles) that undergo reversible scission and recombination.

Physics
-------
Cates living polymers combine two timescales:
- τ_rep: Reptation time (curvilinear diffusion along the tube)
- τ_break: Average breaking time (scission events)

In the fast-breaking limit (τ_break << τ_rep), the system behaves as a
single Maxwell mode with effective relaxation time:

    τ_d = √(τ_rep · τ_break)

This "geometric mean" time emerges from the interplay of reptation and
scission: breaking accelerates stress relaxation by shortening chains,
yielding single-exponential stress decay characteristic of fast-breaking
wormlike micelles.

Key Predictions
---------------
1. **SAOS**: Single Maxwell mode with τ_d (G' ~ G'' crossover at ω = 1/τ_d)
2. **Flow curve**: UCM-like (no shear thinning for constant breakage)
3. **Startup**: Monotonic rise to steady state (no overshoot)
4. **Relaxation**: Single exponential with time constant τ_d

Supported Protocols
-------------------
- FLOW_CURVE: Steady shear stress σ = G₀·τ_d·γ̇ / (1 + (τ_d·γ̇)²) + η_s·γ̇
- OSCILLATION: SAOS moduli with effective τ_d
- STARTUP: Transient stress growth (ODE)
- RELAXATION: Exponential decay σ(t) = σ₀·exp(-t/τ_d)
- CREEP: Strain evolution under constant stress (ODE)
- LAOS: Large-amplitude oscillatory shear (ODE)

Example
-------
>>> from rheojax.models.tnt import TNTCates
>>> import numpy as np
>>>
>>> # Create Cates model
>>> model = TNTCates()
>>>
>>> # Flow curve (analytical)
>>> gamma_dot = np.logspace(-2, 2, 50)
>>> sigma = model.predict(gamma_dot, test_mode='flow_curve')
>>>
>>> # SAOS with effective τ_d
>>> omega = np.logspace(-2, 2, 50)
>>> G_prime, G_double_prime = model.predict_saos(omega)
>>>
>>> # Startup flow (ODE)
>>> t = np.linspace(0, 10, 500)
>>> sigma_t = model.simulate_startup(t, gamma_dot=10.0)

References
----------
- Cates, M.E. (1987). Macromolecules 20, 2289-2296.
  "Reptation of Living Polymers: Dynamics of Entangled Polymers in the
  Presence of Reversible Chain-Scission Reactions."
- Cates, M.E. (1990). J. Phys. Chem. 94, 371-375.
  "Nonlinear viscoelasticity of wormlike micelles."
"""

from __future__ import annotations

import logging

import diffrax
import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.models.tnt._base import TNTBase
from rheojax.models.tnt._kernels import (
    tnt_base_relaxation_vec,
    tnt_saos_moduli_vec,
    tnt_single_mode_creep_ode_rhs,
    tnt_single_mode_ode_rhs,
    tnt_single_mode_ode_rhs_laos,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "tnt_cates",
    protocols=["flow_curve", "oscillation", "startup", "relaxation", "creep", "laos"],
)
class TNTCates(TNTBase):
    """Cates living polymer (wormlike micelle) model.

    Implements the Cates theory for living polymers with reversible scission.
    In the fast-breaking limit, the system behaves as a single Maxwell mode
    with effective relaxation time τ_d = √(τ_rep · τ_break).

    The constitutive equation is identical to the basic TNT model (constant
    breakage, linear stress, upper-convected derivative), but with τ_d replacing
    the single bond lifetime τ_b:

        dS/dt = L·S + S·L^T + (1/τ_d)·I - (1/τ_d)·S
        σ = G₀·S_xy + η_s·γ̇

    Parameters
    ----------
    G_0 : float, default 1e3
        Plateau modulus (Pa). Network elastic modulus.
    tau_rep : float, default 10.0
        Reptation time (s). Curvilinear diffusion time along the tube.
    tau_break : float, default 0.1
        Average breaking time (s). Mean time between scission events.
    eta_s : float, default 0.0
        Solvent viscosity (Pa·s). Newtonian background contribution.

    Derived
    -------
    tau_d : float
        Effective relaxation time τ_d = √(τ_rep · τ_break)
    eta_0 : float
        Zero-shear viscosity η₀ = G₀·τ_d + η_s

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters for fitting
    fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    Basic usage with default parameters:

    >>> model = TNTCates()
    >>> print(model.tau_d)  # Effective time
    1.0

    Fit to SAOS data:

    >>> omega = np.logspace(-2, 2, 50)
    >>> G_star = generate_synthetic_data(omega)
    >>> model.fit(omega, G_star, test_mode='oscillation')

    Predict flow curve:

    >>> gamma_dot = np.logspace(-2, 2, 50)
    >>> sigma = model.predict_flow_curve(gamma_dot)

    See Also
    --------
    TNTSingleMode : Single-mode TNT with variants
    TNTLoopBridge : Two-species loop-bridge kinetics
    """

    def __init__(self):
        """Initialize Cates living polymer model."""
        super().__init__()
        self._setup_parameters()
        self._test_mode = None

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with Cates parameters.

        Parameters:
        - G_0: Plateau modulus (Pa)
        - tau_rep: Reptation time (s)
        - tau_break: Average breaking time (s)
        - eta_s: Solvent viscosity (Pa·s)
        """
        self.parameters = ParameterSet()

        self.parameters.add(
            name="G_0",
            value=1e3,
            bounds=(1e0, 1e8),
            units="Pa",
            description="Plateau modulus (elastic contribution from network)",
        )
        self.parameters.add(
            name="tau_rep",
            value=10.0,
            bounds=(1e-4, 1e6),
            units="s",
            description="Reptation time (curvilinear diffusion along tube)",
        )
        self.parameters.add(
            name="tau_break",
            value=0.1,
            bounds=(1e-6, 1e4),
            units="s",
            description="Average breaking time (mean time between scission events)",
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
    def G_0(self) -> float:
        """Get plateau modulus G₀ (Pa)."""
        val = self.parameters.get_value("G_0")
        return float(val) if val is not None else 0.0

    @property
    def tau_rep(self) -> float:
        """Get reptation time τ_rep (s)."""
        val = self.parameters.get_value("tau_rep")
        return float(val) if val is not None else 0.0

    @property
    def tau_break(self) -> float:
        """Get breaking time τ_break (s)."""
        val = self.parameters.get_value("tau_break")
        return float(val) if val is not None else 0.0

    @property
    def eta_s(self) -> float:
        """Get solvent viscosity η_s (Pa·s)."""
        val = self.parameters.get_value("eta_s")
        return float(val) if val is not None else 0.0

    @property
    def tau_d(self) -> float:
        """Get effective relaxation time τ_d = √(τ_rep · τ_break) (s)."""
        return float(jnp.sqrt(self.tau_rep * self.tau_break))

    @property
    def eta_0(self) -> float:
        """Get zero-shear viscosity η₀ = G₀·τ_d + η_s (Pa·s)."""
        return self.G_0 * self.tau_d + self.eta_s

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
            f"Fitted TNTCates: G₀={self.G_0:.2e}, τ_rep={self.tau_rep:.2e}, "
            f"τ_break={self.tau_break:.2e}, τ_d={self.tau_d:.2e}, η_s={self.eta_s:.2e}"
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

        # Build parameter array from ParameterSet (ordering: G_0, tau_rep, tau_break, eta_s)
        param_values = [
            float(self.parameters.get_value(name))
            for name in ["G_0", "tau_rep", "tau_break", "eta_s"]
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
            Parameter values: [G_0, tau_rep, tau_break, eta_s]
        test_mode : str, optional
            Override stored test mode

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        # Unpack parameters
        G_0 = params[0]
        tau_rep = params[1]
        tau_break = params[2]
        eta_s = params[3]

        # Compute effective relaxation time
        tau_d = jnp.sqrt(tau_rep * tau_break)

        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            # Analytical steady-state: σ = G₀·τ_d·γ̇ + η_s·γ̇
            return G_0 * tau_d * X_jax + eta_s * X_jax

        elif mode == "oscillation":
            # SAOS with effective τ_d
            G_prime, G_double_prime = tnt_saos_moduli_vec(
                X_jax, G_0, tau_d, eta_s
            )
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            gamma_dot = self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return self._simulate_startup_internal(
                X_jax, G_0, tau_d, eta_s, gamma_dot
            )

        elif mode == "relaxation":
            gamma_dot = self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError(
                    "relaxation mode requires gamma_dot (pre-shear rate)"
                )
            return self._simulate_relaxation_internal(
                X_jax, G_0, tau_d, eta_s, gamma_dot
            )

        elif mode == "creep":
            sigma_applied = self._sigma_applied
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            return self._simulate_creep_internal(
                X_jax, G_0, tau_d, eta_s, sigma_applied
            )

        elif mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, G_0, tau_d, eta_s, self._gamma_0, self._omega_laos
            )
            return stress

        else:
            logger.warning(
                f"Unknown test_mode '{mode}', defaulting to flow_curve"
            )
            return G_0 * tau_d * X_jax + eta_s * X_jax

    # =========================================================================
    # Analytical Predictions
    # =========================================================================

    def predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict steady shear stress and viscosity.

        For Cates model with constant breakage:
        σ = G₀·τ_d·γ̇ + η_s·γ̇ (UCM-like, no shear thinning)

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
        tau_d = self.tau_d

        # Analytical steady-state stress
        sigma = self.G_0 * tau_d * gd + self.eta_s * gd

        if return_components:
            eta = sigma / jnp.maximum(gd, 1e-20)
            # N₁ = 2G₀·(τ_d·γ̇)² (UCM formula with τ_d)
            wi = tau_d * gd
            N1 = 2.0 * self.G_0 * wi * wi
            return np.asarray(sigma), np.asarray(eta), np.asarray(N1)

        return np.asarray(sigma)

    def predict_saos(
        self,
        omega: np.ndarray,
        return_components: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

        Cates model reduces to single-mode Maxwell with τ_d:
        G'(ω) = G₀·(ωτ_d)²/(1+(ωτ_d)²)
        G''(ω) = G₀·(ωτ_d)/(1+(ωτ_d)²) + η_s·ω

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
        tau_d = self.tau_d

        G_prime, G_double_prime = tnt_saos_moduli_vec(
            w, self.G_0, tau_d, self.eta_s
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

        Cates model (UCM-like):
        N₁ = 2G₀·(τ_d·γ̇)²
        N₂ = 0

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
        tau_d = self.tau_d

        wi = tau_d * gd
        N1 = 2.0 * self.G_0 * wi * wi
        N2 = jnp.zeros_like(N1)

        return np.asarray(N1), np.asarray(N2)

    # =========================================================================
    # ODE-Based Internal Simulations (for model_function)
    # =========================================================================

    def _simulate_startup_internal(
        self,
        t: jnp.ndarray,
        G_0: float,
        tau_d: float,
        eta_s: float,
        gamma_dot: float,
    ) -> jnp.ndarray:
        """Internal startup simulation for model_function.

        Returns total shear stress σ_xy(t).
        """

        def ode_fn(ti, yi, args):
            return tnt_single_mode_ode_rhs(
                ti, yi, args["gamma_dot"], args["G_0"], args["tau_d"]
            )

        args = {"gamma_dot": gamma_dot, "G_0": G_0, "tau_d": tau_d}
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t[0]
        t1 = t[-1]
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

        # Total stress: σ = G₀·S_xy + η_s·γ̇
        sigma = G_0 * sol.ys[:, 3] + eta_s * gamma_dot
        return sigma

    def _simulate_relaxation_internal(
        self,
        t: jnp.ndarray,
        G_0: float,
        tau_d: float,
        eta_s: float,
        gamma_dot_preshear: float,
    ) -> jnp.ndarray:
        """Internal relaxation simulation for model_function.

        Analytical: σ(t) = G₀·τ_d·γ̇·exp(-t/τ_d)
        """
        # Analytical single-exponential relaxation
        sigma_0 = G_0 * tau_d * gamma_dot_preshear
        return tnt_base_relaxation_vec(t, sigma_0, tau_d)

    def _simulate_creep_internal(
        self,
        t: jnp.ndarray,
        G_0: float,
        tau_d: float,
        eta_s: float,
        sigma_applied: float,
    ) -> jnp.ndarray:
        """Internal creep simulation for model_function.

        Returns accumulated strain γ(t).
        """

        def ode_fn(ti, yi, args):
            return tnt_single_mode_creep_ode_rhs(
                ti,
                yi,
                args["sigma_applied"],
                args["G_0"],
                args["tau_d"],
                args["eta_s"],
            )

        args = {
            "sigma_applied": sigma_applied,
            "G_0": G_0,
            "tau_d": tau_d,
            "eta_s": eta_s,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t[0]
        t1 = t[-1]
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

        return sol.ys[:, 4]  # γ (strain)

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        G_0: float,
        tau_d: float,
        eta_s: float,
        gamma_0: float,
        omega: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal LAOS simulation for model_function.

        Returns (strain, stress) arrays.
        """

        def ode_fn(ti, yi, args):
            return tnt_single_mode_ode_rhs_laos(
                ti,
                yi,
                args["gamma_0"],
                args["omega"],
                args["G_0"],
                args["tau_d"],
            )

        args = {
            "gamma_0": gamma_0,
            "omega": omega,
            "G_0": G_0,
            "tau_d": tau_d,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t[0]
        t1 = t[-1]
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
        stress = G_0 * sol.ys[:, 3] + eta_s * gamma_dot_t

        return strain, stress

    # =========================================================================
    # Public Simulation Methods (return numpy arrays)
    # =========================================================================

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup flow at constant shear rate.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_full : bool, default False
            If True, return full conformation tensor components

        Returns
        -------
        np.ndarray or tuple
            Shear stress σ(t), or (S_xx, S_yy, S_xy, S_zz) if return_full
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        tau_d = self.tau_d

        def ode_fn(ti, yi, args):
            return tnt_single_mode_ode_rhs(
                ti, yi, args["gamma_dot"], args["G_0"], args["tau_d"]
            )

        args = {"gamma_dot": gamma_dot, "G_0": self.G_0, "tau_d": tau_d}
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t_jax[0]
        t1 = t_jax[-1]
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
            "S_xx": np.asarray(sol.ys[:, 0]),
            "S_yy": np.asarray(sol.ys[:, 1]),
            "S_zz": np.asarray(sol.ys[:, 2]),
            "S_xy": np.asarray(sol.ys[:, 3]),
        }

        if return_full:
            return (
                np.asarray(sol.ys[:, 0]),
                np.asarray(sol.ys[:, 1]),
                np.asarray(sol.ys[:, 3]),
                np.asarray(sol.ys[:, 2]),
            )

        # Total stress: σ = G₀·S_xy + η_s·γ̇
        sigma = self.G_0 * sol.ys[:, 3] + self.eta_s * gamma_dot
        return np.asarray(sigma)

    def simulate_relaxation(
        self,
        t: np.ndarray,
        gamma_dot_preshear: float,
    ) -> np.ndarray:
        """Simulate stress relaxation after cessation of steady shear.

        Analytical single-exponential decay:
        σ(t) = G₀·τ_d·γ̇·exp(-t/τ_d)

        Parameters
        ----------
        t : np.ndarray
            Time array (s), starting from t=0 (cessation)
        gamma_dot_preshear : float
            Shear rate before cessation (1/s)

        Returns
        -------
        np.ndarray
            Relaxing stress σ(t)
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        tau_d = self.tau_d

        # Analytical relaxation
        sigma_0 = self.G_0 * tau_d * gamma_dot_preshear
        sigma = tnt_base_relaxation_vec(t_jax, sigma_0, tau_d)

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
        tau_d = self.tau_d

        def ode_fn(ti, yi, args):
            return tnt_single_mode_creep_ode_rhs(
                ti,
                yi,
                args["sigma_applied"],
                args["G_0"],
                args["tau_d"],
                args["eta_s"],
            )

        args = {
            "sigma_applied": sigma_applied,
            "G_0": self.G_0,
            "tau_d": tau_d,
            "eta_s": self.eta_s,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t_jax[0]
        t1 = t_jax[-1]
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

        gamma = np.asarray(sol.ys[:, 4])
        S_xy = np.asarray(sol.ys[:, 3])

        self._trajectory = {
            "t": np.asarray(t_jax),
            "gamma": gamma,
            "S_xy": S_xy,
        }

        if return_rate:
            eta_s_reg = max(self.eta_s, 1e-10 * self.G_0 * tau_d)
            sigma_elastic = self.G_0 * S_xy
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
            Dictionary with keys: 't', 'strain', 'stress', 'strain_rate'
        """
        if n_cycles is not None:
            T = 2 * np.pi / omega
            t = np.linspace(0, n_cycles * T, n_cycles * 200)

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        tau_d = self.tau_d

        strain, stress = self._simulate_laos_internal(
            t_jax, self.G_0, tau_d, self.eta_s, gamma_0, omega
        )

        strain_rate = gamma_0 * omega * jnp.cos(omega * t_jax)

        self._trajectory = {
            "t": np.asarray(t_jax),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "strain_rate": np.asarray(strain_rate),
        }

        return {
            "t": np.asarray(t_jax),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "strain_rate": np.asarray(strain_rate),
        }

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_relaxation_spectrum(
        self,
        t: np.ndarray | None = None,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get relaxation modulus G(t).

        For Cates model: G(t) = G₀·exp(-t/τ_d)

        Parameters
        ----------
        t : np.ndarray, optional
            Time array (default: logspace from 0.01·τ_d to 100·τ_d)
        n_points : int, default 100
            Number of points if t not provided

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t, G(t))
        """
        tau_d = self.tau_d

        if t is None:
            t = np.logspace(
                np.log10(0.01 * tau_d),
                np.log10(100 * tau_d),
                n_points,
            )

        G_t = self.G_0 * np.exp(-t / tau_d)
        return t, G_t

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
        sigma_prime_list: list[float] = []
        sigma_double_prime_list: list[float] = []

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

            sigma_prime_list.append(sigma_n_prime)
            sigma_double_prime_list.append(sigma_n_double_prime)

        sigma_prime = np.array(sigma_prime_list)
        sigma_double_prime = np.array(sigma_double_prime_list)
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

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TNTCates(G_0={self.G_0:.3e}, tau_rep={self.tau_rep:.3e}, "
            f"tau_break={self.tau_break:.3e}, tau_d={self.tau_d:.3e}, "
            f"eta_s={self.eta_s:.3e})"
        )
