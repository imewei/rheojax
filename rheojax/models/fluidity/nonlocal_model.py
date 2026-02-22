"""Non-Local Fluidity Model Implementation.

This module implements the Non-Local (1D PDE, Coussot-Ovarlez) Fluidity model
for yield-stress fluids with spatial diffusion, supporting shear banding analysis.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import lazy_import, safe_import_jax

diffrax = lazy_import("diffrax")
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import get_logger, log_fit
from rheojax.models.fluidity._base import FluidityBase
from rheojax.models.fluidity._kernels import (
    banding_ratio,
    fluidity_nonlocal_creep_pde_rhs,
    fluidity_nonlocal_pde_rhs,
    fluidity_nonlocal_steady_state,
    shear_banding_cv,
)

# Safe JAX import
jax, jnp = safe_import_jax()

# Logger
logger = get_logger(__name__)

# Sentinel for distinguishing "not provided" from falsy values (FL-009)
_MISSING = object()

# FL-006: kwargs to pop before forwarding to nlsq_optimize
_NLSQ_RESERVED = {
    "test_mode", "use_log_residuals", "smart_init", "use_multi_start",
    "n_starts", "perturb_factor", "gamma_dot", "sigma_applied",
    "gamma_0", "omega", "omega_laos", "t_wait", "n_cycles",
    "points_per_cycle", "deformation_mode", "poisson_ratio",
    "method", "callback",
}


@ModelRegistry.register(
    "fluidity_nonlocal",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.STARTUP,
        Protocol.OSCILLATION,
        Protocol.LAOS,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class FluidityNonlocal(FluidityBase):
    """Non-Local (1D PDE) Fluidity Model for yield-stress fluids.

    Implements the Coussot-Ovarlez non-local fluidity model where the
    fluidity field f(y,t) evolves across the gap (y-direction) via:

    ∂f/∂t = (f_loc(σ) - f)/θ + ξ²∂²f/∂y²

    where:
    - f_loc(σ) is the local equilibrium fluidity from HB flow curve
    - θ is the relaxation time
    - ξ is the cooperativity length (non-local diffusion)

    This captures shear banding: localized flow in yield-stress fluids
    where the cooperativity length ξ determines band width.

    Key features:
    - 1D Couette gap discretization (N_y points)
    - Neumann (zero-flux) boundary conditions at walls
    - Diffrax Dopri5 solver (explicit, robust) for PDE
    - Shear banding metrics: CV and max/min ratio

    Attributes:
        N_y: Number of grid points across gap
        gap_width: Physical gap width (m)
    """

    def __init__(self, N_y: int = 64, gap_width: float = 1e-3):
        """Initialize Non-Local Fluidity Model.

        Args:
            N_y: Number of spatial grid points (default 64)
            gap_width: Physical gap width in meters (default 1 mm)
        """
        super().__init__()
        # FL-011: Guard against N_y < 2 which causes ZeroDivisionError in dy
        if N_y < 2:
            raise ValueError(
                f"N_y must be >= 2 for spatial discretization, got {N_y}"
            )
        self.N_y = N_y
        self.gap_width = gap_width
        self.dy = gap_width / (N_y - 1)

        # Add non-local specific parameter
        self._add_nonlocal_parameters()

        # Storage for fluidity field trajectory
        self._f_field_trajectory: np.ndarray | None = None

    def _add_nonlocal_parameters(self):
        """Add non-local specific parameters."""
        # xi: Cooperativity length (m)
        self.parameters.add(
            name="xi",
            value=1e-5,
            bounds=(1e-9, 1e-3),
            units="m",
            description="Cooperativity length (non-local diffusion scale)",
        )

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> FluidityNonlocal:
        """Fit Non-Local Fluidity model to data.

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

        # FL-001: Normalize aliases early so self._test_mode is canonical
        if test_mode == "saos":
            test_mode = "oscillation"

        with log_fit(logger, model="FluidityNonlocal", data_shape=X.shape) as ctx:
            self._test_mode = cast(str, test_mode)
            ctx["test_mode"] = test_mode
            ctx["N_y"] = self.N_y

            if test_mode in ["steady_shear", "rotation", "flow_curve"]:
                self._fit_flow_curve(X, y, **kwargs)
            elif test_mode == "startup":
                self._fit_transient(X, y, mode="startup", **kwargs)
            elif test_mode == "relaxation":
                self._fit_transient(X, y, mode="relaxation", **kwargs)
            elif test_mode == "creep":
                self._fit_transient(X, y, mode="creep", **kwargs)
            elif test_mode == "oscillation":
                self._fit_oscillation(X, y, **kwargs)
            elif test_mode == "laos":
                self._fit_laos(X, y, **kwargs)
            else:
                raise ValueError(f"Unsupported test_mode: {test_mode}")

            self.fitted_ = True

        return self

    # =========================================================================
    # Grid and Initial Conditions
    # =========================================================================

    def _get_grid_args(self, params: dict | None = None) -> dict:
        """Get grid-related arguments for PDE solver.

        Args:
            params: Optional parameter dictionary

        Returns:
            Dictionary with grid parameters
        """
        if params is None:
            params = self.get_parameter_dict()

        return {
            "N_y": self.N_y,
            "dy": self.dy,
            "xi": params.get("xi", 1e-5),
        }

    def _get_initial_f_field(
        self, f_init: float | None = None, N_y: int | None = None
    ) -> jnp.ndarray:
        """Get initial fluidity field (uniform across gap).

        Args:
            f_init: Initial fluidity value. If None, uses f_eq.
            N_y: Number of grid points override. If None, uses self.N_y.

        Returns:
            Fluidity field array, shape (N_y,)
        """
        if f_init is None:
            f_init = self.get_initial_fluidity()
        n = N_y if N_y is not None else self.N_y
        return jnp.ones(n) * f_init

    def _get_initial_state(
        self,
        mode: str,
        params: dict,
        sigma_0: float | None = None,
        N_y: int | None = None,
    ) -> jnp.ndarray:
        """Get initial state vector for PDE solver.

        State vector: [Σ (or γ for creep), f[0], f[1], ..., f[N_y-1]]

        Args:
            mode: 'startup', 'relaxation', 'creep', or 'laos'
            params: Parameter dictionary
            sigma_0: Initial stress for relaxation

        Returns:
            Initial state vector
        """
        f_eq = params["f_eq"]
        f_inf = params["f_inf"]

        if mode == "creep":
            # State: [γ, f_field] - strain starts at 0
            f_field = self._get_initial_f_field(f_eq, N_y=N_y)
            return jnp.concatenate([jnp.array([0.0]), f_field])

        elif mode == "relaxation":
            # State: [Σ, f_field] - stress starts at sigma_0, f at f_inf
            sigma_init = sigma_0 if sigma_0 is not None else params["tau_y"]
            f_field = self._get_initial_f_field(f_inf, N_y=N_y)  # Just flowed
            return jnp.concatenate([jnp.array([sigma_init]), f_field])

        else:  # startup or laos
            # State: [Σ, f_field] - stress at 0, f at f_eq
            f_field = self._get_initial_f_field(f_eq, N_y=N_y)
            return jnp.concatenate([jnp.array([0.0]), f_field])

    # =========================================================================
    # Flow Curve (Steady State)
    # =========================================================================

    def _fit_flow_curve(
        self, gamma_dot: np.ndarray, stress: np.ndarray, **kwargs
    ) -> None:
        """Fit steady-state flow curve.

        For homogeneous (non-banding) steady state, uses HB:
        σ = τ_y + K*|γ̇|^n

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
            return fluidity_nonlocal_steady_state(
                x_data,
                p_map["G"],
                p_map["tau_y"],
                p_map["K"],
                p_map["n_flow"],
                p_map["f_eq"],
                p_map["f_inf"],
                p_map["theta"],
            )

        objective = create_least_squares_objective(
            model_fn,
            gamma_dot_jax,
            stress_jax,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        # FL-006: Pop protocol/meta kwargs before forwarding to nlsq_optimize
        nlsq_kwargs = {k: v for k, v in kwargs.items() if k not in _NLSQ_RESERVED}
        result = nlsq_optimize(objective, self.parameters, **nlsq_kwargs)
        if not result.success:
            logger.warning(f"Fluidity flow curve fit warning: {result.message}")

    # FL-013: _predict_flow_curve is not used by _predict() or model_function()
    # (flow curve routing goes through fluidity_nonlocal_steady_state directly).
    # Kept as a thin compatibility wrapper for external callers.
    def _predict_flow_curve(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict steady-state flow curve (compatibility wrapper)."""
        return np.array(self._predict(gamma_dot, test_mode="flow_curve"))

    # =========================================================================
    # Transient Protocols (Startup, Relaxation, Creep)
    # =========================================================================

    def _fit_transient(self, t: np.ndarray, y: np.ndarray, mode: str, **kwargs) -> None:
        """Fit transient response using PDE solver.

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

        # FL-003: Use local variables for coarser grid instead of mutating self
        # This avoids thread-safety issues where concurrent access could corrupt
        # self.N_y and self.dy during fitting
        fit_N_y = kwargs.pop("fit_N_y", min(self.N_y, 32))
        fit_dy = self.gap_width / (fit_N_y - 1)

        if mode == "startup" and gamma_dot is None:
            raise ValueError("startup mode requires gamma_dot in kwargs")
        if mode == "creep" and sigma_applied is None:
            raise ValueError("creep mode requires sigma_applied in kwargs")

        # Store for prediction
        self._gamma_dot_applied = gamma_dot
        self._sigma_applied = sigma_applied

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            return self._simulate_pde(
                x_data, p_map, mode, gamma_dot, sigma_applied, sigma_0,
                N_y=fit_N_y, dy=fit_dy,
            )

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            y_jax,
            use_log_residuals=kwargs.get("use_log_residuals", False),
        )

        # FL-006: Pop protocol/meta kwargs before forwarding to nlsq_optimize
        nlsq_kwargs = {k: v for k, v in kwargs.items() if k not in _NLSQ_RESERVED}
        result = nlsq_optimize(objective, self.parameters, **nlsq_kwargs)
        if not result.success:
            logger.warning(f"Fluidity transient fit warning: {result.message}")

    def _simulate_pde(
        self,
        t: jnp.ndarray,
        params: dict,
        mode: str,
        gamma_dot: float | None,
        sigma_applied: float | None,
        sigma_0: float | None,
        N_y: int | None = None,
        dy: float | None = None,
    ) -> jnp.ndarray:
        """Simulate PDE response using Diffrax.

        Args:
            t: Time array
            params: Parameter dictionary
            mode: 'startup', 'relaxation', or 'creep'
            gamma_dot: Applied shear rate (for startup)
            sigma_applied: Applied stress (for creep)
            sigma_0: Initial stress (for relaxation)
            N_y: Grid points override (FL-003 thread safety). If None, uses self.N_y.
            dy: Grid spacing override (FL-003 thread safety). If None, uses self.dy.

        Returns:
            Primary output (stress for startup/relaxation, strain for creep)
        """
        # FL-003: Use local variables instead of self.N_y/self.dy for thread safety
        N_y_local = N_y if N_y is not None else self.N_y
        dy_local = dy if dy is not None else self.dy

        # Build args for PDE RHS
        # FL-012: Removed dead "N_y" key — PDE kernels infer N_y from state vector shape
        args = {
            "G": params["G"],
            "tau_y": params["tau_y"],
            "K": params["K"],
            "n_flow": params["n_flow"],
            "theta": params["theta"],
            "xi": params.get("xi", 1e-5),
            "dy": dy_local,
        }

        # Mode-specific setup
        if mode == "creep":
            pde_fn = fluidity_nonlocal_creep_pde_rhs
            args["sigma_applied"] = sigma_applied if sigma_applied is not None else 0.0
        else:
            pde_fn = fluidity_nonlocal_pde_rhs
            if mode == "startup":
                args["mode"] = 0  # rate_controlled
                args["gamma_dot"] = gamma_dot if gamma_dot is not None else 0.0
            else:  # relaxation
                args["mode"] = 0  # rate_controlled
                args["gamma_dot"] = 0.0

        # Initial state (uses N_y_local for grid size)
        y0 = self._get_initial_state(mode, params, sigma_0, N_y=N_y_local)

        # Diffrax setup - use Dopri5 for stiff PDEs (explicit, avoids tracer issues)
        term = diffrax.ODETerm(
            jax.checkpoint(lambda ti, yi, args_i: pde_fn(cast(float, ti), yi, args_i))
        )
        solver = diffrax.Dopri5()
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
            throw=False,  # Return partial result on failure (for optimization)
        )

        # Store trajectory for analysis (skip during JAX tracing, e.g. NUTS)
        # FL-007: Log exceptions instead of silently swallowing
        try:
            self._f_field_trajectory = np.array(sol.ys[:, 1:])
        except Exception as e:
            logger.warning("Could not store fluidity field trajectory: %s", e)

        # Extract primary variable (index 0)
        # For creep: strain; for startup/relaxation: stress
        result = sol.ys[:, 0]

        # Handle solver failure by returning NaN (optimization will avoid this)
        result = jnp.where(sol.result == diffrax.RESULTS.successful, result, jnp.nan)

        return result

    def _predict_transient(self, t: np.ndarray, mode: str | None = None) -> np.ndarray:
        """Predict transient response."""
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        p = self.get_parameter_dict()

        mode = mode or self._test_mode
        if mode is None:
            raise ValueError("Test mode not specified for prediction")

        result = self._simulate_pde(
            t_jax,
            p,
            mode,
            self._gamma_dot_applied,
            self._sigma_applied,
            None,
        )
        return np.array(result)

    # =========================================================================
    # Shear Banding Analysis
    # =========================================================================

    def get_fluidity_profile(self, time_idx: int = -1) -> np.ndarray:
        """Get fluidity profile at specified time index.

        Args:
            time_idx: Time index (default -1 for final time)

        Returns:
            Fluidity field across gap, shape (N_y,)
        """
        if self._f_field_trajectory is None:
            raise ValueError("No trajectory available. Run simulation first.")
        return self._f_field_trajectory[time_idx]

    def get_shear_banding_metric(self, f_field: np.ndarray | None = None) -> float:
        """Compute coefficient of variation as shear banding metric.

        CV = std(f) / mean(f)
        CV > 0.3 typically indicates significant shear banding.

        Args:
            f_field: Fluidity field. If None, uses final simulation state.

        Returns:
            Coefficient of variation (dimensionless)
        """
        if f_field is None:
            f_field = self.get_fluidity_profile(-1)
        f_jax = jnp.asarray(f_field, dtype=jnp.float64)
        return float(shear_banding_cv(f_jax))

    def get_banding_ratio(self, f_field: np.ndarray | None = None) -> float:
        """Compute max/min fluidity ratio as banding metric.

        ratio > 10 indicates strong localization.

        Args:
            f_field: Fluidity field. If None, uses final simulation state.

        Returns:
            Banding ratio (dimensionless)
        """
        if f_field is None:
            f_field = self.get_fluidity_profile(-1)
        f_jax = jnp.asarray(f_field, dtype=jnp.float64)
        return float(banding_ratio(f_jax))

    def is_banding(
        self, f_field: np.ndarray | None = None, cv_threshold: float = 0.3
    ) -> bool:
        """Check if shear banding is occurring.

        Args:
            f_field: Fluidity field. If None, uses final simulation state.
            cv_threshold: CV threshold for banding (default 0.3)

        Returns:
            True if CV > threshold
        """
        return self.get_shear_banding_metric(f_field) > cv_threshold

    # =========================================================================
    # Oscillatory Protocols
    # =========================================================================

    def _fit_oscillation(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit SAOS data using linear viscoelastic approximation.

        For small amplitude, bulk response approximates Local model.
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
            )

        objective = create_least_squares_objective(
            model_fn,
            omega_jax,
            G_star_jax,
            normalize=True,
        )

        # FL-006: Pop protocol/meta kwargs before forwarding to nlsq_optimize
        nlsq_kwargs = {k: v for k, v in kwargs.items() if k not in _NLSQ_RESERVED}
        result = nlsq_optimize(objective, self.parameters, **nlsq_kwargs)
        if not result.success:
            logger.warning(f"Fluidity SAOS fit warning: {result.message}")

    # TODO (FL-010): _predict_saos_jit is duplicated in FluidityLocal.
    # Consider extracting to a shared module-level function or into _base.py.
    @staticmethod
    @jax.jit
    def _predict_saos_jit(
        omega: jnp.ndarray,
        G: float,
        f_eq: float,
        theta: float = 0.0,  # FL-005: dead parameter, kept for backward compatibility
    ) -> jnp.ndarray:
        """SAOS prediction using linear viscoelastic approximation.

        Note:
            theta parameter is unused (FL-005) but kept for backward
            compatibility with external callers.
        """
        del theta  # FL-005: explicitly unused
        tau_eff = 1.0 / (G * f_eq + 1e-30)
        omega_tau = omega * tau_eff
        denom = 1.0 + omega_tau**2

        G_prime = G * omega_tau**2 / denom
        G_double_prime = G * omega_tau / denom

        return jnp.stack([G_prime, G_double_prime], axis=1)

    def _fit_laos(self, t: np.ndarray, sigma: np.ndarray, **kwargs) -> None:
        """Fit LAOS data using full PDE integration."""
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

        # FL-003: Use local variables for coarser grid instead of mutating self
        fit_N_y = kwargs.pop("fit_N_y", min(self.N_y, 32))
        fit_dy = self.gap_width / (fit_N_y - 1)

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        sigma_jax = jnp.asarray(sigma, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            _, stress = self._simulate_laos_internal(
                x_data, p_map, gamma_0, omega, N_y=fit_N_y, dy=fit_dy,
            )
            return stress

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            sigma_jax,
            normalize=True,
        )

        # FL-006: Pop protocol/meta kwargs before forwarding to nlsq_optimize
        nlsq_kwargs = {k: v for k, v in kwargs.items() if k not in _NLSQ_RESERVED}
        result = nlsq_optimize(objective, self.parameters, **nlsq_kwargs)
        if not result.success:
            logger.warning(f"Fluidity LAOS fit warning: {result.message}")

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        params: dict,
        gamma_0: float,
        omega: float,
        N_y: int | None = None,
        dy: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate LAOS response using PDE solver.

        Args:
            t: Time array
            params: Parameter dictionary
            gamma_0: Strain amplitude
            omega: Angular frequency
            N_y: Grid points override (FL-003 thread safety). If None, uses self.N_y.
            dy: Grid spacing override (FL-003 thread safety). If None, uses self.dy.
        """
        # FL-003: Use local variables instead of self.N_y/self.dy for thread safety
        N_y_local = N_y if N_y is not None else self.N_y
        dy_local = dy if dy is not None else self.dy

        # Base args
        # FL-012: Removed dead "N_y" key — PDE kernels infer N_y from state vector shape
        base_args = {
            "G": params["G"],
            "tau_y": params["tau_y"],
            "K": params["K"],
            "n_flow": params["n_flow"],
            "theta": params["theta"],
            "xi": params.get("xi", 1e-5),
            "dy": dy_local,
            "mode": 0,  # rate_controlled
        }

        # Initial state (uses N_y_local for grid size)
        y0 = self._get_initial_state("laos", params, N_y=N_y_local)

        # PDE with time-varying gamma_dot
        def laos_pde(ti, yi, args_i):
            gamma_dot_t = gamma_0 * omega * jnp.cos(omega * ti)
            args_with_rate = {**args_i, "gamma_dot": gamma_dot_t}
            return fluidity_nonlocal_pde_rhs(ti, yi, args_with_rate)

        term = diffrax.ODETerm(jax.checkpoint(laos_pde))
        solver = diffrax.Dopri5()
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
            max_steps=16_000_000,
            throw=False,  # Return partial result on failure (for optimization)
        )

        stress = sol.ys[:, 0]
        strain = gamma_0 * jnp.sin(omega * t)

        # Handle solver failure by returning NaN
        stress = jnp.where(sol.result == diffrax.RESULTS.successful, stress, jnp.nan)

        # Store trajectory only when not in JIT context (concrete arrays)
        # FL-008: Use ConcretizationTypeError (modern) instead of deprecated
        # TracerArrayConversionError
        try:
            # This will fail during JIT tracing
            self._f_field_trajectory = np.asarray(sol.ys[:, 1:])
        except (TypeError, jax.errors.ConcretizationTypeError):
            # During JIT tracing, skip storage
            pass

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

    # =========================================================================
    # Bayesian / Model Function Interface
    # =========================================================================

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function.

        Accepts protocol-specific kwargs (gamma_dot, sigma_applied, etc.).
        """
        p_values = dict(zip(self.parameters.keys(), params, strict=True))
        mode = test_mode or self._test_mode
        if mode is None:
            mode = "oscillation"

        # FL-001: Normalize aliases
        if mode == "saos":
            mode = "oscillation"

        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # FL-009: Use sentinel pattern to avoid swallowing falsy values (e.g. 0.0)
        gamma_dot = kwargs.get("gamma_dot", _MISSING)
        if gamma_dot is _MISSING:
            gamma_dot = getattr(self, "_gamma_dot_applied", None)
        sigma_applied = kwargs.get("sigma_applied", _MISSING)
        if sigma_applied is _MISSING:
            sigma_applied = getattr(self, "_sigma_applied", None)
        gamma_0 = kwargs.get("gamma_0", _MISSING)
        if gamma_0 is _MISSING:
            gamma_0 = getattr(self, "_gamma_0", None)
        omega = kwargs.get("omega", _MISSING)
        if omega is _MISSING:
            omega = getattr(self, "_omega_laos", None)

        if mode in ["steady_shear", "rotation", "flow_curve"]:
            return fluidity_nonlocal_steady_state(
                X_jax,
                p_values["G"],
                p_values["tau_y"],
                p_values["K"],
                p_values["n_flow"],
                p_values["f_eq"],
                p_values["f_inf"],
                p_values["theta"],
            )
        elif mode == "oscillation":
            return self._predict_saos_jit(
                X_jax,
                p_values["G"],
                p_values["f_eq"],
            )
        elif mode in ["startup", "relaxation", "creep"]:
            return self._simulate_pde(
                X_jax,
                p_values,
                mode,
                gamma_dot,
                sigma_applied,
                None,
            )
        elif mode == "laos":
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(X_jax, p_values, gamma_0, omega)
            return stress

        return jnp.zeros_like(X_jax)

    # =========================================================================
    # Prediction Interface
    # =========================================================================

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict based on fitted state."""
        X_jax = jnp.asarray(X, dtype=jnp.float64)
        p = self.get_parameter_dict()

        # Get test_mode from kwargs or instance attribute
        test_mode = kwargs.get("test_mode") or getattr(self, "_test_mode", None)
        if test_mode is None:
            raise ValueError("test_mode must be specified for prediction")

        # FL-001: Normalize aliases
        if test_mode == "saos":
            test_mode = "oscillation"

        if test_mode in ["steady_shear", "rotation", "flow_curve"]:
            result = fluidity_nonlocal_steady_state(
                X_jax,
                p["G"],
                p["tau_y"],
                p["K"],
                p["n_flow"],
                p["f_eq"],
                p["f_inf"],
                p["theta"],
            )
            return np.array(result)

        elif test_mode == "oscillation":
            result = self._predict_saos_jit(
                X_jax,
                p["G"],
                p["f_eq"],
            )
            return np.array(result)

        elif test_mode in ["startup", "relaxation", "creep"]:
            return self._predict_transient(X, mode=test_mode)

        elif test_mode == "laos":
            # Get gamma_0 and omega from kwargs or instance attributes
            gamma_0 = kwargs.get("gamma_0", self._gamma_0)
            omega = kwargs.get("omega", self._omega_laos)
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS prediction requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(X_jax, p, gamma_0, omega)
            return np.array(stress)

        return np.zeros_like(X)
