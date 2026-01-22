"""Local Fluidity Model Implementation.

This module implements the Local (0D, homogeneous) Fluidity model for
yield-stress fluids, supporting multiple protocols via JAX and Diffrax.
"""

from __future__ import annotations

from typing import cast

import diffrax
import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.logging import get_logger, log_fit
from rheojax.models.fluidity._base import FluidityBase
from rheojax.models.fluidity._kernels import (
    fluidity_local_creep_ode_rhs,
    fluidity_local_ode_rhs,
    fluidity_local_steady_state,
)

# Safe JAX import
jax, jnp = safe_import_jax()

# Logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "fluidity_local",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.STARTUP,
        Protocol.OSCILLATION,
        Protocol.LAOS,
    ],
)
class FluidityLocal(FluidityBase):
    """Local (0D) Fluidity Model for yield-stress fluids.

    Implements a homogeneous fluidity model where the material state
    is characterized by a single fluidity value f(t) that evolves via:

    df/dt = (f_eq - f)/θ + a|γ̇|^n(f_inf - f)

    This captures:
    - Aging: structural build-up at rest, f → f_eq
    - Rejuvenation: flow-induced breakdown, f → f_inf

    The stress evolves as a viscoelastic solid with plastic flow:
    dσ/dt = G(γ̇ - σf)

    Protocols:
    - Flow Curve: Algebraic steady-state solution
    - Startup: ODE integration with constant γ̇
    - Relaxation: ODE integration with γ̇=0, stress decays
    - Creep: ODE integration with constant σ
    - SAOS/LAOS: ODE integration + FFT for moduli
    """

    def __init__(self):
        """Initialize Local Fluidity Model."""
        super().__init__()

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> FluidityLocal:
        """Fit Fluidity model to data.

        Args:
            X: Independent variable (time, frequency, or shear rate)
            y: Dependent variable (stress, modulus, viscosity)
            **kwargs: Optimizer options. Must include 'test_mode'.
        """
        test_mode = kwargs.get("test_mode")
        if test_mode is None:
            if hasattr(self, "_test_mode") and self._test_mode:
                test_mode = self._test_mode
            else:
                raise ValueError("test_mode must be specified for Fluidity fitting")

        with log_fit(logger, model="FluidityLocal", data_shape=X.shape) as ctx:
            self._test_mode = cast(str, test_mode)
            ctx["test_mode"] = test_mode

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

        At steady state:
        f_ss = (f_eq/θ + a|γ̇|^n * f_inf) / (1/θ + a|γ̇|^n)
        σ_ss = γ̇ / f_ss

        Args:
            gamma_dot: Shear rate array (1/s)
            stress: Shear stress array (Pa)
            **kwargs: Optimizer options
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        stress_jax = jnp.asarray(stress, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            return fluidity_local_steady_state(
                x_data,
                p_map["G"],
                p_map["tau_y"],
                p_map["K"],
                p_map["n_flow"],
                p_map["f_eq"],
                p_map["f_inf"],
                p_map["theta"],
                p_map["a"],
                p_map["n_rejuv"],
            )

        objective = create_least_squares_objective(
            model_fn,
            gamma_dot_jax,
            stress_jax,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Fluidity flow curve fit warning: {result.message}")

    def _predict_flow_curve(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict steady-state flow curve."""
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        p = self.get_parameter_dict()

        result = fluidity_local_steady_state(
            gamma_dot_jax,
            p["G"],
            p["tau_y"],
            p["K"],
            p["n_flow"],
            p["f_eq"],
            p["f_inf"],
            p["theta"],
            p["a"],
            p["n_rejuv"],
        )
        return np.array(result)

    # =========================================================================
    # Transient Protocols (Startup, Relaxation, Creep)
    # =========================================================================

    def _fit_transient(
        self, t: np.ndarray, y: np.ndarray, mode: str, **kwargs
    ) -> None:
        """Fit transient response.

        Args:
            t: Time array (s)
            y: Response data (stress for startup/relaxation, strain for creep)
            mode: 'startup', 'relaxation', or 'creep'
            **kwargs: Protocol-specific inputs and optimizer options
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

        if mode == "startup" and gamma_dot is None:
            raise ValueError("startup mode requires gamma_dot in kwargs")
        if mode == "creep" and sigma_applied is None:
            raise ValueError("creep mode requires sigma_applied in kwargs")

        # Store for prediction
        self._gamma_dot_applied = gamma_dot
        self._sigma_applied = sigma_applied

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            return self._simulate_transient(
                x_data, p_map, mode, gamma_dot, sigma_applied, sigma_0
            )

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            y_jax,
            use_log_residuals=kwargs.get("use_log_residuals", False),
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Fluidity transient fit warning: {result.message}")

    def _simulate_transient(
        self,
        t: jnp.ndarray,
        params: dict,
        mode: str,
        gamma_dot: float | None,
        sigma_applied: float | None,
        sigma_0: float | None,
    ) -> jnp.ndarray:
        """Simulate transient response using Diffrax ODE integration.

        Args:
            t: Time array
            params: Parameter dictionary
            mode: 'startup', 'relaxation', or 'creep'
            gamma_dot: Applied shear rate (for startup)
            sigma_applied: Applied stress (for creep)
            sigma_0: Initial stress (for relaxation)

        Returns:
            Stress (for startup/relaxation) or strain (for creep)
        """
        # Build args for ODE RHS
        args = {
            "G": params["G"],
            "f_eq": params["f_eq"],
            "f_inf": params["f_inf"],
            "theta": params["theta"],
            "a": params["a"],
            "n_rejuv": params["n_rejuv"],
        }

        # Initial fluidity (equilibrium state)
        f_init = params["f_eq"]

        # Mode-specific setup
        if mode == "creep":
            # Creep: constant stress, track strain
            ode_fn = fluidity_local_creep_ode_rhs
            args["sigma_applied"] = sigma_applied if sigma_applied is not None else 0.0
            # State: [strain, f]
            y0 = jnp.array([0.0, f_init])
        elif mode == "startup":
            # Startup: constant rate, track stress
            ode_fn = fluidity_local_ode_rhs
            args["gamma_dot"] = gamma_dot if gamma_dot is not None else 0.0
            # State: [sigma, f]
            y0 = jnp.array([0.0, f_init])
        else:  # relaxation
            # Relaxation: rate = 0, stress decays
            ode_fn = fluidity_local_ode_rhs
            args["gamma_dot"] = 0.0
            sigma_init = sigma_0 if sigma_0 is not None else params["tau_y"]
            # Start with elevated fluidity (just flowed)
            f_init_relax = params["f_inf"]
            y0 = jnp.array([sigma_init, f_init_relax])

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
        )

        # Extract primary variable (index 0)
        # For creep: strain; for startup/relaxation: stress
        result = sol.ys[:, 0]

        return result

    def _predict_transient(self, t: np.ndarray, mode: str | None = None) -> np.ndarray:
        """Predict transient response."""
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
        )
        return np.array(result)

    # =========================================================================
    # Oscillatory Protocols (SAOS, LAOS)
    # =========================================================================

    def _fit_oscillation(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit SAOS data.

        For small amplitude, uses linear viscoelastic approximation.

        Args:
            X: Frequency array (rad/s)
            y: Complex modulus [G', G'']
            **kwargs: Optimizer options
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
                p_map["f_eq"],
                p_map["theta"],
            )

        objective = create_least_squares_objective(
            model_fn,
            omega_jax,
            G_star_jax,
            normalize=True,
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Fluidity SAOS fit warning: {result.message}")

    @staticmethod
    @jax.jit
    def _predict_saos_jit(
        omega: jnp.ndarray,
        G: float,
        f_eq: float,
        theta: float,
    ) -> jnp.ndarray:
        """SAOS prediction using linear viscoelastic approximation.

        In the linear limit (small strain), the model behaves like a Maxwell
        model with effective relaxation time tau_eff = 1/(G*f_eq).

        G*(ω) = G * (iωτ) / (1 + iωτ)
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

        Args:
            t: Time array (s)
            sigma: Stress response (Pa)
            **kwargs: Must include gamma_0 and omega
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
            logger.warning(f"Fluidity LAOS fit warning: {result.message}")

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        params: dict,
        gamma_0: float,
        omega: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate LAOS response using Diffrax.

        Args:
            t: Time array
            params: Parameter dictionary
            gamma_0: Strain amplitude
            omega: Angular frequency

        Returns:
            (strain, stress) arrays
        """
        # Base args
        base_args = {
            "G": params["G"],
            "f_eq": params["f_eq"],
            "f_inf": params["f_inf"],
            "theta": params["theta"],
            "a": params["a"],
            "n_rejuv": params["n_rejuv"],
        }

        # Initial conditions (steady state at rest)
        f_init = params["f_eq"]
        y0 = jnp.array([0.0, f_init])  # [sigma, f]

        # ODE with time-varying gamma_dot
        def laos_ode(ti, yi, args_i):
            gamma_dot_t = gamma_0 * omega * jnp.cos(omega * ti)
            args_with_rate = {**args_i, "gamma_dot": gamma_dot_t}
            return fluidity_local_ode_rhs(ti, yi, args_with_rate)

        term = diffrax.ODETerm(laos_ode)
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
            args=base_args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=10_000_000,
        )

        stress = sol.ys[:, 0]
        strain = gamma_0 * jnp.sin(omega * t)

        return strain, stress

    def simulate_laos(
        self,
        gamma_0: float,
        omega: float,
        n_cycles: int = 2,
        n_points_per_cycle: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate LAOS response.

        Args:
            gamma_0: Strain amplitude
            omega: Angular frequency (rad/s)
            n_cycles: Number of oscillation cycles
            n_points_per_cycle: Points per cycle

        Returns:
            (strain, stress) arrays
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

        return np.array(strain), np.array(stress)

    def extract_harmonics(
        self,
        stress: np.ndarray,
        n_points_per_cycle: int = 256,
    ) -> dict:
        """Extract Fourier harmonics from LAOS stress response.

        Args:
            stress: Stress array from simulate_laos
            n_points_per_cycle: Points per cycle

        Returns:
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

        # Ratios
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

    def model_function(self, X, params, test_mode=None):
        """NumPyro/BayesianMixin model function.

        Routes to appropriate prediction based on test_mode.
        """
        p_values = dict(zip(self.parameters.keys(), params, strict=True))
        mode = test_mode or self._test_mode
        if mode is None:
            mode = "oscillation"

        X_jax = jnp.asarray(X, dtype=jnp.float64)

        if mode in ["steady_shear", "rotation", "flow_curve"]:
            return fluidity_local_steady_state(
                X_jax,
                p_values["G"],
                p_values["tau_y"],
                p_values["K"],
                p_values["n_flow"],
                p_values["f_eq"],
                p_values["f_inf"],
                p_values["theta"],
                p_values["a"],
                p_values["n_rejuv"],
            )
        elif mode == "oscillation":
            return self._predict_saos_jit(
                X_jax,
                p_values["G"],
                p_values["f_eq"],
                p_values["theta"],
            )
        elif mode in ["startup", "relaxation", "creep"]:
            return self._simulate_transient(
                X_jax,
                p_values,
                mode,
                self._gamma_dot_applied,
                self._sigma_applied,
                None,
            )
        elif mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, p_values, self._gamma_0, self._omega_laos
            )
            return stress

        return jnp.zeros_like(X_jax)

    # =========================================================================
    # Prediction Interface
    # =========================================================================

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict based on fitted state."""
        X_jax = jnp.asarray(X, dtype=jnp.float64)
        p = self.get_parameter_dict()

        if self._test_mode in ["steady_shear", "rotation", "flow_curve"]:
            result = fluidity_local_steady_state(
                X_jax,
                p["G"],
                p["tau_y"],
                p["K"],
                p["n_flow"],
                p["f_eq"],
                p["f_inf"],
                p["theta"],
                p["a"],
                p["n_rejuv"],
            )
            return np.array(result)

        elif self._test_mode == "oscillation":
            result = self._predict_saos_jit(
                X_jax,
                p["G"],
                p["f_eq"],
                p["theta"],
            )
            return np.array(result)

        elif self._test_mode in ["startup", "relaxation", "creep"]:
            return self._predict_transient(X)

        elif self._test_mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS prediction requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, p, self._gamma_0, self._omega_laos
            )
            return np.array(stress)

        return np.zeros_like(X)
