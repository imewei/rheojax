"""Local (0D) Fluidity-Saramito Elastoviscoplastic Model.

This module implements `FluiditySaramitoLocal`, a homogeneous model for
elastoviscoplastic materials with thixotropic fluidity evolution.

The model captures:
- Viscoelastic stress overshoot in startup
- Creep bifurcation at the yield stress
- Non-exponential stress relaxation
- Normal stress differences (N₁ from UCM)
- Nonlinear LAOS response

Coupling Modes
--------------
- "minimal": λ = 1/f only, τ_y = τ_y0 constant
- "full": λ = 1/f + τ_y(f) aging yield stress

Supported Protocols
-------------------
All six standard protocols: FLOW_CURVE, CREEP, RELAXATION, STARTUP,
OSCILLATION, LAOS.

References
----------
- Saramito, P. (2007). JNNFM 145, 1-14.
- Saramito, P. (2009). JNNFM 158, 154-161.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import diffrax
import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.logging import log_fit
from rheojax.models.fluidity.saramito._base import FluiditySaramitoBase
from rheojax.models.fluidity.saramito._kernels import (
    saramito_flow_curve_steady,
    saramito_local_creep_ode_rhs,
    saramito_local_ode_rhs,
    saramito_local_relaxation_ode_rhs,
)

# Safe import ensures float64
jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "fluidity_saramito_local",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.STARTUP,
        Protocol.OSCILLATION,
        Protocol.LAOS,
    ],
)
class FluiditySaramitoLocal(FluiditySaramitoBase):
    r"""Local (0D) Fluidity-Saramito Model for elastoviscoplastic fluids.

    Implements a homogeneous Saramito EVP model where the material state
    is characterized by tensorial stress [τ_xx, τ_yy, τ_xy] and scalar
    fluidity f(t) that evolve via coupled ODEs.

    The constitutive equation (Saramito 2009):
    λ(∇̂τ) + α(τ)τ = 2η_p D

    where:

    - λ = 1/f: Fluidity-dependent relaxation time
    - ∇̂τ: Upper-convected derivative
    - α = max(0, 1 - τ_y/\|τ\|): Von Mises plasticity
    - η_p = G/f: Polymeric viscosity

    Fluidity evolves via:
    df/dt = (f_age - f)/t_a + b\|γ̇\|^n(f_flow - f)

    Parameters
    ----------
    coupling : {"minimal", "full"}, default "minimal"
        Coupling mode for yield stress evolution:
        - "minimal": τ_y = τ_y0 (constant)
        - "full": τ_y(f) = τ_y0 + a_y/f^m (aging yield stress)

    Examples
    --------
    Basic flow curve fitting:

    >>> model = FluiditySaramitoLocal(coupling="minimal")
    >>> model.fit(gamma_dot, sigma, test_mode="flow_curve")
    >>> sigma_pred = model.predict(gamma_dot)

    Startup with stress overshoot:

    >>> model = FluiditySaramitoLocal()
    >>> model.fit(t, sigma, test_mode="startup", gamma_dot=1.0)
    >>> strain, stress, f = model.simulate_startup(t, gamma_dot=1.0)

    Bayesian inference:

    >>> result = model.fit_bayesian(gamma_dot, sigma, test_mode="flow_curve",
    ...                             num_warmup=1000, num_samples=2000)
    >>> intervals = model.get_credible_intervals(result.posterior_samples)
    """

    def __init__(
        self,
        coupling: Literal["minimal", "full"] = "minimal",
    ):
        """Initialize Local Fluidity-Saramito Model.

        Parameters
        ----------
        coupling : {"minimal", "full"}, default "minimal"
            Coupling mode for yield stress evolution
        """
        super().__init__(coupling=coupling)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> FluiditySaramitoLocal:
        """Fit Saramito model to data.

        Parameters
        ----------
        X : np.ndarray
            Independent variable (time, frequency, or shear rate)
        y : np.ndarray
            Dependent variable (stress, modulus, or strain)
        **kwargs
            Optimizer options. Must include 'test_mode'.

        Returns
        -------
        self
            Fitted model instance
        """
        test_mode = kwargs.get("test_mode")
        if test_mode is None:
            if hasattr(self, "_test_mode") and self._test_mode:
                test_mode = self._test_mode
            else:
                raise ValueError("test_mode must be specified for Saramito fitting")

        with log_fit(logger, model="FluiditySaramitoLocal", data_shape=X.shape) as ctx:
            self._test_mode = cast(str, test_mode)
            ctx["test_mode"] = test_mode
            ctx["coupling"] = self.coupling

            if test_mode in ["steady_shear", "rotation", "flow_curve"]:
                self._fit_flow_curve(X, y, **kwargs)
            elif test_mode == "startup":
                self._fit_transient(X, y, mode="startup", **kwargs)
            elif test_mode == "relaxation":
                self._fit_transient(X, y, mode="relaxation", **kwargs)
            elif test_mode == "creep":
                self._fit_transient(X, y, mode="creep", **kwargs)
            elif test_mode in ["oscillation", "saos"]:
                self._fit_oscillation(X, y, **kwargs)
            elif test_mode == "laos":
                self._fit_laos(X, y, **kwargs)
            else:
                raise ValueError(f"Unsupported test_mode: {test_mode}")

            self.fitted_ = True

        return self

    # =========================================================================
    # Flow Curve (Steady State)
    # =========================================================================

    def _fit_flow_curve(
        self, gamma_dot: np.ndarray, stress: np.ndarray, **kwargs
    ) -> None:
        """Fit steady-state flow curve.

        At steady state, the model reduces to Herschel-Bulkley form
        with fluidity-dependent yield stress in full coupling mode.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        stress : np.ndarray
            Shear stress array (Pa)
        **kwargs
            Optimizer options
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        stress_jax = jnp.asarray(stress, dtype=jnp.float64)

        # Smart initialization
        if kwargs.pop("smart_init", True):
            self.initialize_from_flow_curve(gamma_dot, stress)

        coupling = self.coupling

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))

            tau_y_coupling = (
                p_map.get("tau_y_coupling", 0.0) if coupling == "full" else 0.0
            )
            m_yield = p_map.get("m_yield", 1.0) if coupling == "full" else 1.0

            return saramito_flow_curve_steady(
                x_data,
                p_map["G"],
                p_map["tau_y0"],
                p_map["K_HB"],
                p_map["n_HB"],
                p_map["f_age"],
                p_map["f_flow"],
                p_map["t_a"],
                p_map["b"],
                p_map["n_rej"],
                coupling,
                tau_y_coupling,
                m_yield,
            )

        objective = create_least_squares_objective(
            model_fn,
            gamma_dot_jax,
            stress_jax,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Saramito flow curve fit warning: {result.message}")

    def _predict_flow_curve(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict steady-state flow curve.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        np.ndarray
            Steady-state shear stress (Pa)
        """
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        p = self.get_parameter_dict()

        tau_y_coupling = (
            p.get("tau_y_coupling", 0.0) if self.coupling == "full" else 0.0
        )
        m_yield = p.get("m_yield", 1.0) if self.coupling == "full" else 1.0

        result = saramito_flow_curve_steady(
            gamma_dot_jax,
            p["G"],
            p["tau_y0"],
            p["K_HB"],
            p["n_HB"],
            p["f_age"],
            p["f_flow"],
            p["t_a"],
            p["b"],
            p["n_rej"],
            self.coupling,
            tau_y_coupling,
            m_yield,
        )
        return np.array(result)

    # =========================================================================
    # Transient Protocols (Startup, Relaxation, Creep)
    # =========================================================================

    def _fit_transient(self, t: np.ndarray, y: np.ndarray, mode: str, **kwargs) -> None:
        """Fit transient response.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        y : np.ndarray
            Response data (stress for startup/relaxation, strain for creep)
        mode : str
            Protocol: 'startup', 'relaxation', or 'creep'
        **kwargs
            Protocol-specific inputs and optimizer options
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        # Extract protocol-specific inputs
        gamma_dot = kwargs.pop("gamma_dot", None)
        sigma_applied = kwargs.pop("sigma_applied", None)
        sigma_0 = kwargs.pop("sigma_0", None)
        t_wait = kwargs.pop("t_wait", 0.0)

        if mode == "startup" and gamma_dot is None:
            raise ValueError("startup mode requires gamma_dot in kwargs")
        if mode == "creep" and sigma_applied is None:
            raise ValueError("creep mode requires sigma_applied in kwargs")

        # Store for prediction
        self._gamma_dot_applied = gamma_dot
        self._sigma_applied = sigma_applied
        self._t_wait = t_wait

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            result = self._simulate_transient(
                x_data, p_map, mode, gamma_dot, sigma_applied, sigma_0, t_wait
            )
            return result

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            y_jax,
            use_log_residuals=kwargs.get("use_log_residuals", False),
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Saramito transient fit warning: {result.message}")

    def _simulate_transient(
        self,
        t: jnp.ndarray,
        params: dict,
        mode: str,
        gamma_dot: float | None,
        sigma_applied: float | None,
        sigma_0: float | None,
        t_wait: float = 0.0,
    ) -> jnp.ndarray:
        """Simulate transient response using Diffrax ODE integration.

        Parameters
        ----------
        t : jnp.ndarray
            Time array
        params : dict
            Parameter dictionary
        mode : str
            Protocol: 'startup', 'relaxation', or 'creep'
        gamma_dot : float, optional
            Applied shear rate (for startup)
        sigma_applied : float, optional
            Applied stress (for creep)
        sigma_0 : float, optional
            Initial stress (for relaxation)
        t_wait : float, default 0.0
            Waiting time for initial fluidity

        Returns
        -------
        jnp.ndarray
            Stress (startup/relaxation) or strain (creep)
        """
        # Build args for ODE RHS
        args = self._get_saramito_ode_args(params)

        # Initial fluidity based on waiting time
        f_age = params["f_age"]
        f_flow = params["f_flow"]
        t_a = params["t_a"]

        if t_wait > 0:
            f_init = f_age + (f_flow - f_age) * jnp.exp(-t_wait / t_a)
        else:
            f_init = f_age  # Well-aged sample

        # Mode-specific setup
        if mode == "creep":
            # Creep: constant stress, track strain
            ode_fn = saramito_local_creep_ode_rhs
            args["sigma_applied"] = sigma_applied if sigma_applied is not None else 0.0
            # State: [γ, f]
            y0 = jnp.array([0.0, f_init])
        elif mode == "startup":
            # Startup: constant rate, track tensorial stress
            ode_fn = saramito_local_ode_rhs
            args["gamma_dot"] = gamma_dot if gamma_dot is not None else 0.0
            # State: [τ_xx, τ_yy, τ_xy, f, γ]
            y0 = jnp.array([0.0, 0.0, 0.0, f_init, 0.0])
        else:  # relaxation
            # Relaxation: rate = 0, stress decays
            ode_fn = saramito_local_relaxation_ode_rhs
            sigma_init = sigma_0 if sigma_0 is not None else params["tau_y0"]
            # Start with elevated fluidity (just flowed) and initial stress
            f_init_relax = f_flow
            # State: [τ_xx, τ_yy, τ_xy, f]
            # Initial stress is in shear component
            y0 = jnp.array([0.0, 0.0, sigma_init, f_init_relax])

        # Diffrax setup
        term = diffrax.ODETerm(
            lambda ti, yi, args_i: ode_fn(cast(float, ti), yi, args_i)
        )
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

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
            max_steps=10_000_000,
            throw=False,  # Return partial result on failure (for optimization)
        )

        # Extract primary variable
        if mode == "creep":
            # Return strain (index 0)
            result = sol.ys[:, 0]
        elif mode == "startup":
            # Return shear stress τ_xy (index 2)
            result = sol.ys[:, 2]
        else:  # relaxation
            # Return shear stress τ_xy (index 2)
            result = sol.ys[:, 2]

        # Handle solver failure by returning NaN (optimization will avoid this)
        result = jnp.where(sol.result == diffrax.RESULTS.successful, result, jnp.nan)

        return result

    def _predict_transient(self, t: np.ndarray, mode: str | None = None) -> np.ndarray:
        """Predict transient response.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        mode : str, optional
            Protocol mode. If None, uses stored mode.

        Returns
        -------
        np.ndarray
            Predicted response
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        p = self.get_parameter_dict()

        mode = mode or self._test_mode
        if mode is None:
            raise ValueError("Test mode not specified for prediction")

        result = self._simulate_transient(
            t_jax,
            p,
            mode,
            self._gamma_dot_applied,
            self._sigma_applied,
            None,
            self._t_wait,
        )
        return np.array(result)

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        t_wait: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup response with full trajectory.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        t_wait : float, default 0.0
            Waiting time before startup (s)

        Returns
        -------
        strain : np.ndarray
            Accumulated strain γ(t)
        stress : np.ndarray
            Shear stress τ_xy(t) (Pa)
        fluidity : np.ndarray
            Fluidity f(t) (1/(Pa·s))
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        p = self.get_parameter_dict()
        args = self._get_saramito_ode_args(p)
        args["gamma_dot"] = gamma_dot

        # Initial fluidity
        f_age = p["f_age"]
        f_flow = p["f_flow"]
        t_a = p["t_a"]

        if t_wait > 0:
            f_init = f_age + (f_flow - f_age) * np.exp(-t_wait / t_a)
        else:
            f_init = f_age

        # State: [τ_xx, τ_yy, τ_xy, f, γ]
        y0 = jnp.array([0.0, 0.0, 0.0, f_init, 0.0])

        term = diffrax.ODETerm(
            lambda ti, yi, args_i: saramito_local_ode_rhs(cast(float, ti), yi, args_i)
        )
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

        t0, t1 = t[0], t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=t_jax),
            stepsize_controller=stepsize_controller,
            max_steps=10_000_000,
            throw=False,  # Return partial result on failure (for optimization)
        )

        # Handle solver failure
        sol_ys = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys,
            jnp.nan * jnp.ones_like(sol.ys),
        )

        stress = np.array(sol_ys[:, 2])  # τ_xy
        fluidity = np.array(sol_ys[:, 3])  # f
        strain = np.array(sol_ys[:, 4])  # γ

        # Store trajectory
        self._trajectory = {
            "t": np.array(t),
            "tau_xx": np.array(sol_ys[:, 0]),
            "tau_yy": np.array(sol_ys[:, 1]),
            "tau_xy": stress,
            "fluidity": fluidity,
            "strain": strain,
        }

        return strain, stress, fluidity

    def simulate_creep(
        self,
        t: np.ndarray,
        sigma_applied: float,
        t_wait: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate creep response.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma_applied : float
            Applied stress (Pa)
        t_wait : float, default 0.0
            Waiting time before creep (s)

        Returns
        -------
        strain : np.ndarray
            Accumulated strain γ(t)
        fluidity : np.ndarray
            Fluidity f(t) (1/(Pa·s))
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        p = self.get_parameter_dict()
        args = self._get_saramito_ode_args(p)
        args["sigma_applied"] = sigma_applied

        # Initial fluidity
        f_age = p["f_age"]
        f_flow = p["f_flow"]
        t_a = p["t_a"]

        if t_wait > 0:
            f_init = f_age + (f_flow - f_age) * np.exp(-t_wait / t_a)
        else:
            f_init = f_age

        # State: [γ, f]
        y0 = jnp.array([0.0, f_init])

        term = diffrax.ODETerm(
            lambda ti, yi, args_i: saramito_local_creep_ode_rhs(
                cast(float, ti), yi, args_i
            )
        )
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

        t0, t1 = t[0], t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=t_jax),
            stepsize_controller=stepsize_controller,
            max_steps=10_000_000,
            throw=False,  # Return partial result on failure (for optimization)
        )

        # Handle solver failure
        sol_ys = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys,
            jnp.nan * jnp.ones_like(sol.ys),
        )

        strain = np.array(sol_ys[:, 0])
        fluidity = np.array(sol_ys[:, 1])

        # Store trajectory
        self._trajectory = {
            "t": np.array(t),
            "strain": strain,
            "fluidity": fluidity,
        }

        return strain, fluidity

    # =========================================================================
    # Oscillatory Protocols (SAOS, LAOS)
    # =========================================================================

    def _fit_oscillation(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit SAOS data.

        For small amplitude, uses linear viscoelastic approximation
        based on the Maxwell relaxation time λ = 1/(G*f_eq).

        Parameters
        ----------
        X : np.ndarray
            Frequency array (rad/s)
        y : np.ndarray
            Complex modulus [G', G''] or complex G*
        **kwargs
            Optimizer options
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        omega_jax = jnp.asarray(X, dtype=jnp.float64)

        # Handle G_star format
        G_star_np = np.asarray(y)
        if np.iscomplexobj(G_star_np):
            G_star_2d = np.column_stack([np.real(G_star_np), np.imag(G_star_np)])
        elif G_star_np.ndim == 2 and G_star_np.shape[1] == 2:
            G_star_2d = G_star_np
        else:
            raise ValueError(f"G_star must be complex or (M, 2), got {G_star_np.shape}")

        G_star_jax = jnp.asarray(G_star_2d, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            return self._predict_saos_jit(
                x_data,
                p_map["G"],
                p_map["f_age"],
            )

        objective = create_least_squares_objective(
            model_fn,
            omega_jax,
            G_star_jax,
            normalize=True,
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Saramito SAOS fit warning: {result.message}")

    @staticmethod
    @jax.jit
    def _predict_saos_jit(
        omega: jnp.ndarray,
        G: float,
        f_eq: float,
    ) -> jnp.ndarray:
        """SAOS prediction using linear viscoelastic approximation.

        In the linear limit (small strain, σ < τ_y), the model behaves
        like a Maxwell model with relaxation time τ = 1/(G*f_eq).

        G*(ω) = G * (iωτ) / (1 + iωτ)

        Parameters
        ----------
        omega : jnp.ndarray
            Angular frequency (rad/s)
        G : float
            Elastic modulus (Pa)
        f_eq : float
            Equilibrium fluidity (1/(Pa·s))

        Returns
        -------
        jnp.ndarray
            Shape (N, 2) with [G', G'']
        """
        # Effective relaxation time
        tau_eff = 1.0 / (G * f_eq + 1e-30)

        omega_tau = omega * tau_eff
        denom = 1.0 + omega_tau**2

        G_prime = G * omega_tau**2 / denom
        G_double_prime = G * omega_tau / denom

        return jnp.stack([G_prime, G_double_prime], axis=1)

    def _fit_laos(self, t: np.ndarray, sigma: np.ndarray, **kwargs) -> None:
        """Fit LAOS data using full ODE integration.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma : np.ndarray
            Stress response (Pa)
        **kwargs
            Must include gamma_0 and omega
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        gamma_0 = kwargs.pop("gamma_0", None)
        omega = kwargs.pop("omega", None)

        if gamma_0 is None or omega is None:
            raise ValueError("LAOS fitting requires gamma_0 and omega")

        self._gamma_0 = gamma_0
        self._omega_laos = omega

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        sigma_jax = jnp.asarray(sigma, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            _, stress = self._simulate_laos_internal(x_data, p_map, gamma_0, omega)
            return stress

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            sigma_jax,
            normalize=True,
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Saramito LAOS fit warning: {result.message}")

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        params: dict,
        gamma_0: float,
        omega: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate LAOS response using Diffrax.

        Parameters
        ----------
        t : jnp.ndarray
            Time array
        params : dict
            Parameter dictionary
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        strain : jnp.ndarray
            Strain array γ(t) = γ_0 sin(ωt)
        stress : jnp.ndarray
            Stress array τ_xy(t)
        """
        args = self._get_saramito_ode_args(params)

        # Initial conditions (aged state)
        f_init = params["f_age"]
        y0 = jnp.array([0.0, 0.0, 0.0, f_init, 0.0])  # [τ_xx, τ_yy, τ_xy, f, γ]

        # ODE with time-varying gamma_dot
        def laos_ode(ti, yi, args_i):
            gamma_dot_t = gamma_0 * omega * jnp.cos(omega * ti)
            args_with_rate = {**args_i, "gamma_dot": gamma_dot_t}
            return saramito_local_ode_rhs(ti, yi, args_with_rate)

        term = diffrax.ODETerm(laos_ode)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

        t0, t1 = t[0], t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=t),
            stepsize_controller=stepsize_controller,
            max_steps=10_000_000,
            throw=False,  # Return partial result on failure (for optimization)
        )

        stress = sol.ys[:, 2]  # τ_xy
        strain = gamma_0 * jnp.sin(omega * t)

        # Handle solver failure by returning NaN
        stress = jnp.where(sol.result == diffrax.RESULTS.successful, stress, jnp.nan)

        return strain, stress

    def simulate_laos(
        self,
        gamma_0: float,
        omega: float,
        n_cycles: int = 3,
        n_points_per_cycle: int = 256,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate LAOS response.

        Parameters
        ----------
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)
        n_cycles : int, default 3
            Number of oscillation cycles
        n_points_per_cycle : int, default 256
            Points per cycle

        Returns
        -------
        t : np.ndarray
            Time array (s)
        strain : np.ndarray
            Strain array
        stress : np.ndarray
            Stress array (Pa)
        """
        self._gamma_0 = gamma_0
        self._omega_laos = omega

        period = 2.0 * np.pi / omega
        t_max = n_cycles * period
        n_points = n_cycles * n_points_per_cycle
        t = np.linspace(0, t_max, n_points, endpoint=False)
        t_jax = jnp.asarray(t, dtype=jnp.float64)

        p = self.get_parameter_dict()

        strain, stress = self._simulate_laos_internal(t_jax, p, gamma_0, omega)

        return t, np.array(strain), np.array(stress)

    def extract_harmonics(
        self,
        stress: np.ndarray,
        n_points_per_cycle: int = 256,
    ) -> dict:
        """Extract Fourier harmonics from LAOS stress response.

        Parameters
        ----------
        stress : np.ndarray
            Stress array from simulate_laos
        n_points_per_cycle : int, default 256
            Points per cycle

        Returns
        -------
        dict
            Dictionary with I_1, I_3, I_5 amplitudes and ratios
        """
        # Use last complete cycle
        stress_cycle = stress[-n_points_per_cycle:]

        # FFT
        stress_fft = np.fft.fft(stress_cycle)
        n = len(stress_cycle)

        harmonics = {}

        # Fundamental
        I_1 = 2.0 * np.abs(stress_fft[1]) / n
        harmonics["I_1"] = I_1

        # Third harmonic
        if 3 < n // 2:
            I_3 = 2.0 * np.abs(stress_fft[3]) / n
        else:
            I_3 = 0.0
        harmonics["I_3"] = I_3

        # Fifth harmonic
        if 5 < n // 2:
            I_5 = 2.0 * np.abs(stress_fft[5]) / n
        else:
            I_5 = 0.0
        harmonics["I_5"] = I_5

        # Ratios (nonlinearity measures)
        if I_1 > 0:
            harmonics["I_3_I_1"] = I_3 / I_1
            harmonics["I_5_I_1"] = I_5 / I_1
        else:
            harmonics["I_3_I_1"] = 0.0
            harmonics["I_5_I_1"] = 0.0

        return harmonics

    # =========================================================================
    # Bayesian / Model Function Interface
    # =========================================================================

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function.

        Routes to appropriate prediction based on test_mode.
        Accepts protocol-specific kwargs (gamma_dot, sigma_applied, etc.).

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values in order
        test_mode : str, optional
            Override stored test mode
        **kwargs
            Protocol-specific arguments (gamma_dot, sigma_applied, gamma_0, omega)

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        p_values = dict(zip(self.parameters.keys(), params, strict=True))
        mode = test_mode or self._test_mode
        if mode is None:
            mode = "flow_curve"

        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # Extract protocol-specific args from kwargs or fall back to instance attrs
        gamma_dot = kwargs.get("gamma_dot", self._gamma_dot_applied)
        sigma_applied = kwargs.get("sigma_applied", self._sigma_applied)
        gamma_0 = kwargs.get("gamma_0", self._gamma_0)
        omega = kwargs.get("omega", self._omega_laos)
        t_wait = kwargs.get("t_wait", self._t_wait)

        if mode in ["steady_shear", "rotation", "flow_curve"]:
            tau_y_coupling = (
                p_values.get("tau_y_coupling", 0.0) if self.coupling == "full" else 0.0
            )
            m_yield = p_values.get("m_yield", 1.0) if self.coupling == "full" else 1.0

            return saramito_flow_curve_steady(
                X_jax,
                p_values["G"],
                p_values["tau_y0"],
                p_values["K_HB"],
                p_values["n_HB"],
                p_values["f_age"],
                p_values["f_flow"],
                p_values["t_a"],
                p_values["b"],
                p_values["n_rej"],
                self.coupling,
                tau_y_coupling,
                m_yield,
            )
        elif mode == "oscillation":
            return self._predict_saos_jit(
                X_jax,
                p_values["G"],
                p_values["f_age"],
            )
        elif mode in ["startup", "relaxation", "creep"]:
            return self._simulate_transient(
                X_jax,
                p_values,
                mode,
                gamma_dot,
                sigma_applied,
                None,
                t_wait,
            )
        elif mode == "laos":
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, p_values, gamma_0, omega
            )
            return stress

        return jnp.zeros_like(X_jax)

    # =========================================================================
    # Prediction Interface
    # =========================================================================

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict based on fitted state.

        Parameters
        ----------
        X : np.ndarray
            Independent variable
        **kwargs
            Additional options

        Returns
        -------
        np.ndarray
            Predicted response
        """
        X_jax = jnp.asarray(X, dtype=jnp.float64)
        p = self.get_parameter_dict()

        # Get test_mode from kwargs or instance attribute
        test_mode = kwargs.get("test_mode") or getattr(self, "_test_mode", None)
        if test_mode is None:
            raise ValueError("test_mode must be specified for prediction")

        if test_mode in ["steady_shear", "rotation", "flow_curve"]:
            return self._predict_flow_curve(X)

        elif test_mode == "oscillation":
            result = self._predict_saos_jit(
                X_jax,
                p["G"],
                p["f_age"],
            )
            return np.array(result)

        elif test_mode in ["startup", "relaxation", "creep"]:
            return self._predict_transient(X, mode=test_mode)

        elif test_mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS prediction requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, p, self._gamma_0, self._omega_laos
            )
            return np.array(stress)

        return np.zeros_like(X)

    # =========================================================================
    # Additional Analysis Methods
    # =========================================================================

    def get_overshoot_ratio(
        self,
        gamma_dot: float,
        t_max: float = 100.0,
        n_points: int = 1000,
    ) -> float:
        """Compute stress overshoot ratio σ_max / σ_ss.

        Parameters
        ----------
        gamma_dot : float
            Shear rate (1/s)
        t_max : float, default 100.0
            Maximum simulation time (s)
        n_points : int, default 1000
            Number of time points

        Returns
        -------
        float
            Overshoot ratio (1.0 = no overshoot)
        """
        t = np.linspace(0, t_max, n_points)
        _, stress, _ = self.simulate_startup(t, gamma_dot)

        sigma_max = np.max(stress)
        sigma_ss = stress[-1]

        return sigma_max / sigma_ss if sigma_ss > 0 else 1.0

    def get_critical_stress(self) -> float:
        """Get critical stress for creep bifurcation.

        Returns
        -------
        float
            Critical stress σ_c ≈ τ_y (Pa)
        """
        return self.parameters.get_value("tau_y0")
