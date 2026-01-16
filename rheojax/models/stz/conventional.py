"""STZ Conventional Model Implementation.

This module implements the concrete Shear Transformation Zone (STZ) model,
supporting multiple protocols (Flow, Transient, SAOS, LAOS) via JAX and Diffrax.
"""

from __future__ import annotations

import diffrax
import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.logging import get_logger, log_fit
from rheojax.models.stz._base import STZBase
from rheojax.models.stz._kernels import (
    stz_ode_rhs,
)

# Safe JAX import
jax, jnp = safe_import_jax()

# Logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "stz_conventional",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.STARTUP,
        Protocol.OSCILLATION,
        Protocol.LAOS,
    ],
)
class STZConventional(STZBase):
    """Conventional Shear Transformation Zone (STZ) Model.

    Implements STZ plasticity with Langer (2008) formulation.
    Supports Minimal, Standard, and Full complexity variants.

    Protocols:
    - Steady-State Flow: Algebraic solution for flow curve
    - Transient: Diffrax ODE integration for creep/relaxation/startup
    - SAOS/LAOS: Diffrax ODE integration + FFT for harmonic analysis
    """

    def __init__(self, variant: str = "standard"):
        """Initialize STZ Conventional Model.

        Args:
            variant: Model variant ('minimal', 'standard', 'full')
        """
        super().__init__(variant=variant)
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None
        self._gamma_dot_applied: float | None = None

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_mode: str | None = None,
        **kwargs,
    ) -> None:
        """Fit STZ model to data.

        Args:
            X: Independent variable (time, frequency, or shear rate)
            y: Dependent variable (stress, modulus, viscosity)
            test_mode: Protocol mode ('steady_shear', 'relaxation', 'creep',
                       'startup', 'laos', 'oscillation')
            **kwargs: Optimizer options
        """
        if test_mode is None:
            raise ValueError("test_mode must be specified for STZ fitting")

        with log_fit(logger, model="STZConventional", data_shape=X.shape) as ctx:
            self._test_mode = test_mode
            ctx["test_mode"] = test_mode
            ctx["variant"] = self.variant

            if test_mode == "steady_shear":
                self._fit_steady_shear(X, y, **kwargs)
            elif test_mode in ["relaxation", "creep", "startup"]:
                self._fit_transient(X, y, mode=test_mode, **kwargs)
            elif test_mode in ["laos", "oscillation"]:
                self._fit_oscillation(X, y, **kwargs)
            else:
                raise ValueError(f"Unsupported test_mode: {test_mode}")

            self.fitted_ = True

    # =========================================================================
    # Steady State Flow
    # =========================================================================

    def _fit_steady_shear(
        self, gamma_dot: np.ndarray, stress: np.ndarray, **kwargs
    ) -> None:
        """Fit steady-state flow curve (stress vs shear rate)."""
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        stress_jax = jnp.asarray(stress, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            return self._predict_steady_shear_jit(
                x_data,
                p_map["sigma_y"],
                p_map["chi_inf"],
                p_map["tau0"],
                p_map["ez"],
            )

        objective = create_least_squares_objective(
            model_fn,
            gamma_dot_jax,
            stress_jax,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            raise RuntimeError(f"STZ steady shear fit failed: {result.message}")

    @staticmethod
    @jax.jit
    def _predict_steady_shear_jit(gamma_dot, sigma_y, chi_inf, tau0, ez):
        """Analytical steady-state flow curve prediction."""
        # At steady state, chi -> chi_inf
        # Lambda -> exp(-ez / chi_inf)
        # Solve for sigma given gamma_dot
        term = jnp.exp(-(1.0 + ez) / chi_inf)
        arg = (gamma_dot * tau0) / (term + 1e-30)
        arg_clamped = jnp.clip(arg, -0.999999, 0.999999)
        sigma = sigma_y * jnp.arctanh(arg_clamped)
        return sigma

    # =========================================================================
    # Transient (ODE) - Startup, Relaxation, Creep
    # =========================================================================

    def _fit_transient(
        self, t: np.ndarray, y: np.ndarray, mode: str, **kwargs
    ) -> None:
        """Fit transient response (Stress Growth / Relaxation / Creep).

        Args:
            t: Time array (s)
            y: Response data (stress for startup/relaxation, strain for creep)
            mode: 'startup', 'relaxation', or 'creep'
            **kwargs: Optimizer options including:
                - gamma_dot: Applied shear rate for startup (required)
                - sigma_0: Initial stress for relaxation (optional)
                - sigma_applied: Applied stress for creep (required)
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

        # Build model function that uses ODE integration
        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            return self._simulate_transient_jit(
                x_data, p_map, mode, gamma_dot, sigma_applied, sigma_0, self.variant
            )

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            y_jax,
            use_log_residuals=kwargs.get("use_log_residuals", False),
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"STZ transient fit warning: {result.message}")

    def _simulate_transient_jit(
        self,
        t: jnp.ndarray,
        params: dict,
        mode: str,
        gamma_dot: float | None,
        sigma_applied: float | None,
        sigma_0: float | None,
        variant: str,
    ) -> jnp.ndarray:
        """Simulate transient response using Diffrax ODE integration.

        Args:
            t: Time array
            params: Parameter dictionary
            mode: 'startup', 'relaxation', or 'creep'
            gamma_dot: Applied shear rate (for startup)
            sigma_applied: Applied stress (for creep)
            sigma_0: Initial stress (for relaxation)
            variant: Model variant

        Returns:
            Stress (for startup/relaxation) or strain (for creep)
        """
        # Build args dict for stz_ode_rhs
        args = {
            "G0": params["G0"],
            "sigma_y": params["sigma_y"],
            "tau0": params["tau0"],
            "epsilon0": params["epsilon0"],
            "chi_inf": params["chi_inf"],
            "c0": params["c0"],
        }

        # Add variant-specific parameters
        if variant in ["standard", "full"]:
            args["tau_beta"] = params.get("tau_beta", params["tau0"] * 100)
        if variant == "full":
            args["m_inf"] = params.get("m_inf", 0.1)
            args["rate_m"] = params.get("rate_m", 1.0)

        # Set up initial conditions based on mode
        chi_init = 0.05  # Annealed state
        ez = params.get("ez", 1.0)
        Lambda_init = jnp.exp(-ez / chi_init)

        if mode == "startup":
            # Strain-controlled: apply constant gamma_dot, measure stress
            args["gamma_dot"] = gamma_dot
            sigma_init = 0.0
        elif mode == "relaxation":
            # Strain-controlled: gamma_dot = 0, initial stress decays
            args["gamma_dot"] = 0.0
            sigma_init = sigma_0 if sigma_0 is not None else params["sigma_y"]
            chi_init = params["chi_inf"]  # Start at steady-state chi
            Lambda_init = jnp.exp(-ez / chi_init)
        elif mode == "creep":
            # Stress-controlled: constant stress, measure strain
            # This requires a modified ODE (stress is input, not state)
            # For now, we approximate with strain-controlled approach
            args["gamma_dot"] = sigma_applied / (params["G0"] * params["tau0"])
            sigma_init = 0.0

        # Build initial state based on variant
        if variant == "minimal":
            y0 = jnp.array([sigma_init, chi_init])
        elif variant == "standard":
            y0 = jnp.array([sigma_init, chi_init, Lambda_init])
        else:  # full
            y0 = jnp.array([sigma_init, chi_init, Lambda_init, 0.0])

        # Set up Diffrax solver
        term = diffrax.ODETerm(lambda ti, yi, args_i: stz_ode_rhs(ti, yi, args_i))
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)

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

        # Extract stress (index 0)
        stress = sol.ys[:, 0]

        if mode == "creep":
            # For creep, return strain instead
            # Strain = integral of gamma_dot * dt
            # Approximate from plastic rate history
            return jnp.cumsum(jnp.ones_like(stress)) * args["gamma_dot"] * dt0

        return stress

    def _predict_transient(
        self, t: np.ndarray, mode: str | None = None
    ) -> np.ndarray:
        """Predict transient response."""
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        p_values = {k: self.parameters.get_value(k) for k in self.parameters.keys()}

        mode = mode or self._test_mode

        result = self._simulate_transient_jit(
            t_jax,
            p_values,
            mode,
            self._gamma_dot_applied,
            None,  # sigma_applied
            None,  # sigma_0
            self.variant,
        )
        return np.array(result)

    # =========================================================================
    # SAOS / LAOS (ODE + FFT)
    # =========================================================================

    def _fit_oscillation(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit oscillation data (SAOS or LAOS).

        Args:
            X: Frequency array (rad/s) for SAOS, or time array for LAOS
            y: Complex modulus [G', G''] for SAOS, or stress for LAOS
            **kwargs: Required for LAOS: gamma_0, omega
        """

        gamma_0 = kwargs.pop("gamma_0", None)
        omega = kwargs.pop("omega", None)

        # Store for prediction
        self._gamma_0 = gamma_0
        self._omega_laos = omega

        if gamma_0 is not None and gamma_0 > 0.01:
            # LAOS mode - full ODE integration
            self._fit_laos_mode(X, y, gamma_0, omega, **kwargs)
        else:
            # SAOS mode - linear viscoelastic approximation
            self._fit_saos_mode(X, y, **kwargs)

    def _fit_saos_mode(
        self, omega: np.ndarray, G_star: np.ndarray, **kwargs
    ) -> None:
        """Fit SAOS data using linear viscoelastic approximation.

        In SAOS limit, STZ behaves like a Maxwell-like viscoelastic solid.
        G*(omega) approximated from steady-state chi and Lambda.
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        omega_jax = jnp.asarray(omega, dtype=jnp.float64)

        # Handle G_star format
        G_star_np = np.asarray(G_star)
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
                p_map["G0"],
                p_map["sigma_y"],
                p_map["chi_inf"],
                p_map["tau0"],
                p_map["epsilon0"],
            )

        objective = create_least_squares_objective(
            model_fn,
            omega_jax,
            G_star_jax,
            normalize=True,
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"STZ SAOS fit warning: {result.message}")

    @staticmethod
    @jax.jit
    def _predict_saos_jit(omega, G0, sigma_y, chi_inf, tau0, epsilon0):
        """SAOS prediction using linear viscoelastic approximation.

        In the linear limit (small strain), STZ behaves like a Maxwell model
        with effective relaxation time tau_eff.
        """
        # At steady state chi -> chi_inf
        Lambda_ss = jnp.exp(-1.0 / chi_inf)

        # Effective Maxwell relaxation time
        # tau_eff ~ tau0 / (epsilon0 * Lambda_ss)
        tau_eff = tau0 / (2.0 * epsilon0 * Lambda_ss + 1e-30)

        # Maxwell model: G* = G0 * (i * omega * tau) / (1 + i * omega * tau)
        omega_tau = omega * tau_eff
        denom = 1.0 + omega_tau**2

        G_prime = G0 * omega_tau**2 / denom
        G_double_prime = G0 * omega_tau / denom

        return jnp.stack([G_prime, G_double_prime], axis=1)

    def _fit_laos_mode(
        self,
        t: np.ndarray,
        sigma: np.ndarray,
        gamma_0: float,
        omega: float,
        **kwargs,
    ) -> None:
        """Fit LAOS data using full ODE integration + FFT."""
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        sigma_jax = jnp.asarray(sigma, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            _, stress = self._simulate_laos_internal(
                x_data, p_map, gamma_0, omega, self.variant
            )
            return stress

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            sigma_jax,
            normalize=True,
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"STZ LAOS fit warning: {result.message}")

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        params: dict,
        gamma_0: float,
        omega: float,
        variant: str,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate LAOS response using Diffrax.

        Args:
            t: Time array
            params: Parameter dictionary
            gamma_0: Strain amplitude
            omega: Angular frequency
            variant: Model variant

        Returns:
            (strain, stress) arrays
        """
        # Strain input: gamma(t) = gamma_0 * sin(omega * t)
        # Strain rate: gamma_dot(t) = gamma_0 * omega * cos(omega * t)

        # Build args with time-varying gamma_dot
        # We need to pass a function for gamma_dot, but stz_ode_rhs expects scalar
        # Solution: use a wrapper that interpolates

        base_args = {
            "G0": params["G0"],
            "sigma_y": params["sigma_y"],
            "tau0": params["tau0"],
            "epsilon0": params["epsilon0"],
            "chi_inf": params["chi_inf"],
            "c0": params["c0"],
        }

        if variant in ["standard", "full"]:
            base_args["tau_beta"] = params.get("tau_beta", params["tau0"] * 100)
        if variant == "full":
            base_args["m_inf"] = params.get("m_inf", 0.1)
            base_args["rate_m"] = params.get("rate_m", 1.0)

        # Initial conditions
        chi_init = params["chi_inf"]  # Start at steady state for LAOS
        ez = params.get("ez", 1.0)
        Lambda_init = jnp.exp(-ez / chi_init)
        sigma_init = 0.0

        if variant == "minimal":
            y0 = jnp.array([sigma_init, chi_init])
        elif variant == "standard":
            y0 = jnp.array([sigma_init, chi_init, Lambda_init])
        else:
            y0 = jnp.array([sigma_init, chi_init, Lambda_init, 0.0])

        # Define ODE term with time-varying gamma_dot
        def laos_ode(ti, yi, args_i):
            gamma_dot_t = gamma_0 * omega * jnp.cos(omega * ti)
            args_with_rate = {**args_i, "gamma_dot": gamma_dot_t}
            return stz_ode_rhs(ti, yi, args_with_rate)

        term = diffrax.ODETerm(laos_ode)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)

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

        # Extract stress
        stress = sol.ys[:, 0]

        # Compute strain
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

        p_values = {k: self.parameters.get_value(k) for k in self.parameters.keys()}

        strain, stress = self._simulate_laos_internal(
            t_jax, p_values, gamma_0, omega, self.variant
        )

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

        # Fundamental (n=1)
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
    # Bayesian Mixin Interface
    # =========================================================================

    def model_function(self, X, params, test_mode=None):
        """NumPyro/BayesianMixin model function.

        Routes to appropriate prediction based on test_mode.
        """
        p_values = dict(zip(self.parameters.keys(), params, strict=True))
        mode = test_mode or self._test_mode

        X_jax = jnp.asarray(X, dtype=jnp.float64)

        if mode == "steady_shear":
            return self._predict_steady_shear_jit(
                X_jax,
                p_values["sigma_y"],
                p_values["chi_inf"],
                p_values["tau0"],
                p_values["ez"],
            )
        elif mode == "oscillation":
            return self._predict_saos_jit(
                X_jax,
                p_values["G0"],
                p_values["sigma_y"],
                p_values["chi_inf"],
                p_values["tau0"],
                p_values["epsilon0"],
            )
        elif mode in ["startup", "relaxation", "creep"]:
            return self._simulate_transient_jit(
                X_jax,
                p_values,
                mode,
                self._gamma_dot_applied,
                None,
                None,
                self.variant,
            )
        elif mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, p_values, self._gamma_0, self._omega_laos, self.variant
            )
            return stress

        return jnp.zeros_like(X_jax)

    # =========================================================================
    # Prediction Interface
    # =========================================================================

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict based on fitted state."""
        X_jax = jnp.asarray(X, dtype=jnp.float64)
        p_values = {k: self.parameters.get_value(k) for k in self.parameters.keys()}

        if self._test_mode == "steady_shear":
            result = self._predict_steady_shear_jit(
                X_jax,
                p_values["sigma_y"],
                p_values["chi_inf"],
                p_values["tau0"],
                p_values["ez"],
            )
            return np.array(result)

        elif self._test_mode == "oscillation":
            result = self._predict_saos_jit(
                X_jax,
                p_values["G0"],
                p_values["sigma_y"],
                p_values["chi_inf"],
                p_values["tau0"],
                p_values["epsilon0"],
            )
            return np.array(result)

        elif self._test_mode in ["startup", "relaxation", "creep"]:
            return self._predict_transient(X)

        elif self._test_mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS prediction requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, p_values, self._gamma_0, self._omega_laos, self.variant
            )
            return np.array(stress)

        return np.zeros_like(X)
