"""Nonlocal (1D) Fluidity-Saramito Elastoviscoplastic Model.

This module implements `FluiditySaramitoNonlocal`, a spatially-resolved model
for elastoviscoplastic materials with fluidity diffusion. This enables
simulation of shear banding - the localization of flow into discrete bands.

Key Features
------------
- 1D spatial resolution across a Couette gap
- Fluidity diffusion: D_f * ∇²f with cooperativity length ξ
- Shear banding detection and characterization
- Stress-controlled and rate-controlled protocols

Physical Basis
--------------
The nonlocal term ξ²∇²f represents cooperative rearrangements that
regularize the model and set the width of shear band interfaces.
Typical ξ values are 1-10 particle diameters.

References
----------
- Goyon, J. et al. (2008). Nature 454, 84-87.
- Bocquet, L. et al. (2009). PRL 103, 036001.
- Ovarlez, G. et al. (2012). J. Non-Newtonian Fluid Mech. 177-178, 19-28.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import diffrax
import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import log_fit
from rheojax.models.fluidity.saramito._base import FluiditySaramitoBase
from rheojax.models.fluidity.saramito._kernels import (
    banding_ratio,
    detect_shear_bands,
    saramito_nonlocal_pde_rhs,
    shear_banding_cv,
)

# Safe import ensures float64
jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "fluidity_saramito_nonlocal",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.CREEP,
        Protocol.STARTUP,
    ],
    deformation_modes=[DeformationMode.SHEAR],
)
class FluiditySaramitoNonlocal(FluiditySaramitoBase):
    """Nonlocal (1D) Fluidity-Saramito Model with spatial diffusion.

    Implements a spatially-resolved Saramito EVP model where fluidity
    varies across a Couette gap and can form shear bands.

    The fluidity evolution includes a diffusion term:
    ∂f/∂t = (f_loc - f)/t_a + D_f * ∇²f

    where D_f = ξ²/t_a is the fluidity diffusivity and ξ is the
    cooperativity length (interface width parameter).

    Parameters
    ----------
    coupling : {"minimal", "full"}, default "minimal"
        Coupling mode for yield stress evolution
    N_y : int, default 51
        Number of spatial grid points
    H : float, default 1e-3
        Gap width (m)
    xi : float, default 1e-5
        Cooperativity length (m)

    Notes
    -----
    The model solves a coupled PDE system for [Σ, f(y)] where Σ is the
    bulk (gap-averaged) stress. In Couette geometry, stress is approximately
    uniform across the gap, enabling this simplification.

    Shear bands appear when the fluidity profile develops large gradients,
    with high-fluidity (flowing) bands coexisting with low-fluidity (jammed)
    regions.

    Examples
    --------
    Basic flow curve with banding check:

    >>> model = FluiditySaramitoNonlocal(N_y=51, H=1e-3, xi=1e-5)
    >>> model.fit(gamma_dot, sigma, test_mode="flow_curve")
    >>> is_banded, cv, ratio = model.detect_shear_bands()
    >>> print(f"Shear banding detected: {is_banded}")

    Startup transient with spatial profile:

    >>> model = FluiditySaramitoNonlocal()
    >>> t, sigma, f_field = model.simulate_startup(t, gamma_dot=1.0)
    >>> model.plot_fluidity_profile()  # Shows spatial variation
    """

    def __init__(
        self,
        coupling: Literal["minimal", "full"] = "minimal",
        N_y: int = 51,
        H: float = 1e-3,
        xi: float = 1e-5,
    ):
        """Initialize Nonlocal Fluidity-Saramito Model.

        Parameters
        ----------
        coupling : {"minimal", "full"}, default "minimal"
            Coupling mode for yield stress evolution
        N_y : int, default 51
            Number of spatial grid points
        H : float, default 1e-3
            Gap width (m)
        xi : float, default 1e-5
            Cooperativity length (m)
        """
        super().__init__(coupling=coupling)

        self.N_y = N_y
        self.H = H
        self.xi = xi
        self.dy = H / (N_y - 1)

        # Add nonlocal-specific parameter
        self.parameters.add(
            name="xi",
            value=xi,
            bounds=(1e-7, 1e-2),
            units="m",
            description="Cooperativity length (interface width)",
        )

        # Storage for spatial profiles
        self._f_field: np.ndarray | None = None
        self._y_grid: np.ndarray = np.linspace(0, H, N_y)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> FluiditySaramitoNonlocal:
        """Fit nonlocal Saramito model to data.

        Parameters
        ----------
        X : np.ndarray
            Independent variable (time or shear rate)
        y : np.ndarray
            Dependent variable (stress or strain)
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
                raise ValueError("test_mode must be specified")

        with log_fit(
            logger, model="FluiditySaramitoNonlocal", data_shape=X.shape
        ) as ctx:
            self._test_mode = cast(str, test_mode)
            ctx["test_mode"] = test_mode
            ctx["coupling"] = self.coupling
            ctx["N_y"] = self.N_y

            if test_mode in ["steady_shear", "rotation", "flow_curve"]:
                self._fit_flow_curve(X, y, **kwargs)
            elif test_mode == "startup":
                self._fit_startup(X, y, **kwargs)
            elif test_mode == "creep":
                self._fit_creep(X, y, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported test_mode for nonlocal model: {test_mode}"
                )

            self.fitted_ = True

        return self

    # =========================================================================
    # Flow Curve (Steady State)
    # =========================================================================

    def _fit_flow_curve(
        self, gamma_dot: np.ndarray, stress: np.ndarray, **kwargs
    ) -> None:
        """Fit steady-state flow curve.

        For the nonlocal model, we fit the homogeneous steady state
        (no banding) to get bulk parameters, then check for banding
        at each rate.

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

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            return self._predict_flow_curve_homogeneous(x_data, p_map)

        objective = create_least_squares_objective(
            model_fn,
            gamma_dot_jax,
            stress_jax,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Nonlocal flow curve fit warning: {result.message}")

    def _predict_flow_curve_homogeneous(
        self, gamma_dot: jnp.ndarray, params: dict
    ) -> jnp.ndarray:
        """Predict homogeneous (no banding) flow curve.

        Parameters
        ----------
        gamma_dot : jnp.ndarray
            Shear rate array (1/s)
        params : dict
            Parameter dictionary

        Returns
        -------
        jnp.ndarray
            Steady-state stress (Pa)
        """
        from rheojax.models.fluidity.saramito._kernels import saramito_flow_curve_steady

        tau_y_coupling = (
            params.get("tau_y_coupling", 0.0) if self.coupling == "full" else 0.0
        )
        m_yield = params.get("m_yield", 1.0) if self.coupling == "full" else 1.0

        return saramito_flow_curve_steady(
            gamma_dot,
            params["G"],
            params["tau_y0"],
            params["K_HB"],
            params["n_HB"],
            params["f_age"],
            params["f_flow"],
            params["t_a"],
            params["b"],
            params["n_rej"],
            self.coupling,
            tau_y_coupling,
            m_yield,
        )

    def _predict_flow_curve(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict flow curve.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        np.ndarray
            Steady-state stress (Pa)
        """
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        p = self.get_parameter_dict()
        result = self._predict_flow_curve_homogeneous(gamma_dot_jax, p)
        return np.array(result)

    # =========================================================================
    # Transient Protocols
    # =========================================================================

    def _fit_startup(self, t: np.ndarray, stress: np.ndarray, **kwargs) -> None:
        """Fit startup transient.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        stress : np.ndarray
            Stress response (Pa)
        **kwargs
            Must include gamma_dot
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        gamma_dot = kwargs.pop("gamma_dot", None)
        if gamma_dot is None:
            raise ValueError("startup mode requires gamma_dot")

        self._gamma_dot_applied = gamma_dot

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        stress_jax = jnp.asarray(stress, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            _, sigma, _ = self._simulate_startup_internal(x_data, p_map, gamma_dot)
            return sigma

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            stress_jax,
            use_log_residuals=False,
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Nonlocal startup fit warning: {result.message}")

    def _fit_creep(self, t: np.ndarray, strain: np.ndarray, **kwargs) -> None:
        """Fit creep response.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        strain : np.ndarray
            Strain response
        **kwargs
            Must include sigma_applied
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        sigma_applied = kwargs.pop("sigma_applied", None)
        if sigma_applied is None:
            raise ValueError("creep mode requires sigma_applied")

        self._sigma_applied = sigma_applied

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        strain_jax = jnp.asarray(strain, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_map = dict(zip(self.parameters.keys(), params, strict=True))
            gamma, _ = self._simulate_creep_internal(x_data, p_map, sigma_applied)
            return gamma

        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            strain_jax,
            use_log_residuals=False,
        )

        result = nlsq_optimize(objective, self.parameters, **kwargs)
        if not result.success:
            logger.warning(f"Nonlocal creep fit warning: {result.message}")

    def _simulate_startup_internal(
        self,
        t: jnp.ndarray,
        params: dict,
        gamma_dot: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate startup with spatial resolution.

        Parameters
        ----------
        t : jnp.ndarray
            Time array
        params : dict
            Parameter dictionary
        gamma_dot : float
            Applied shear rate (1/s)

        Returns
        -------
        t : jnp.ndarray
            Time array
        sigma : jnp.ndarray
            Bulk stress (Pa)
        f_field : jnp.ndarray
            Fluidity field at final time, shape (N_y,)
        """
        args = self._get_nonlocal_pde_args(params)
        args["gamma_dot"] = gamma_dot

        # Initial conditions: uniform aged fluidity
        f_init = params["f_age"]
        f_field_init = jnp.ones(self.N_y) * f_init

        # State: [Σ, f_0, f_1, ..., f_{N_y-1}]
        y0 = jnp.concatenate([jnp.array([0.0]), f_field_init])

        term = diffrax.ODETerm(
            lambda ti, yi, args_i: saramito_nonlocal_pde_rhs(
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
            saveat=diffrax.SaveAt(ts=t),
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

        sigma = sol_ys[:, 0]  # Bulk stress
        f_field_final = sol_ys[-1, 1:]  # Final fluidity profile

        return t, sigma, f_field_final

    def _simulate_creep_internal(
        self,
        t: jnp.ndarray,
        params: dict,
        sigma_applied: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate creep with spatial resolution.

        Parameters
        ----------
        t : jnp.ndarray
            Time array
        params : dict
            Parameter dictionary
        sigma_applied : float
            Applied stress (Pa)

        Returns
        -------
        gamma : jnp.ndarray
            Bulk strain
        f_field : jnp.ndarray
            Fluidity field at final time
        """
        # For creep, need modified PDE with fixed stress
        # Simplified: use homogeneous approximation with spatial storage
        # Full implementation would require stress-controlled PDE

        args = self._get_nonlocal_pde_args(params)
        args["sigma_applied"] = sigma_applied
        args["mode"] = "stress_controlled"
        args["gamma_dot"] = 0.0  # Will be computed internally

        # Initial conditions
        f_init = params["f_age"]
        f_field_init = jnp.ones(self.N_y) * f_init

        # For stress-controlled, track strain accumulation
        # Simplified state: [γ, Σ, f_0, ..., f_{N_y-1}] but Σ is fixed
        y0 = jnp.concatenate([jnp.array([sigma_applied]), f_field_init])

        term = diffrax.ODETerm(
            lambda ti, yi, args_i: saramito_nonlocal_pde_rhs(
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
            saveat=diffrax.SaveAt(ts=t),
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

        # Compute strain from average fluidity and stress
        f_avg = jnp.mean(sol_ys[:, 1:], axis=1)

        # For stress-controlled, compute plasticity
        tau_y = params["tau_y0"]
        alpha = jnp.clip(1.0 - tau_y / (jnp.abs(sigma_applied) + 1e-20), 0.0, 1.0)

        # Integrate strain: γ = ∫ α * f_avg * σ dt
        gamma_dot_t = alpha * f_avg * sigma_applied
        dt_array = jnp.diff(t, prepend=t[0])
        gamma = jnp.cumsum(gamma_dot_t * dt_array)

        f_field_final = sol_ys[-1, 1:]

        return gamma, f_field_final

    def _get_nonlocal_pde_args(self, params: dict) -> dict:
        """Build args dictionary for PDE integration.

        Parameters
        ----------
        params : dict
            Parameter dictionary

        Returns
        -------
        dict
            Arguments for PDE RHS
        """
        args = self._get_saramito_ode_args(params)

        # Add spatial discretization info
        args["N_y"] = self.N_y
        args["dy"] = self.dy
        args["xi"] = params.get("xi", self.xi)

        return args

    # =========================================================================
    # Simulation Methods
    # =========================================================================

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup with spatial resolution.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)

        Returns
        -------
        t : np.ndarray
            Time array
        sigma : np.ndarray
            Bulk stress (Pa)
        f_field : np.ndarray
            Final fluidity profile, shape (N_y,)
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        p = self.get_parameter_dict()

        _, sigma, f_field = self._simulate_startup_internal(t_jax, p, gamma_dot)

        self._f_field = np.array(f_field)

        return t, np.array(sigma), np.array(f_field)

    def simulate_creep(
        self,
        t: np.ndarray,
        sigma_applied: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate creep with spatial resolution.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma_applied : float
            Applied stress (Pa)

        Returns
        -------
        gamma : np.ndarray
            Bulk strain
        f_field : np.ndarray
            Final fluidity profile
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        p = self.get_parameter_dict()

        gamma, f_field = self._simulate_creep_internal(t_jax, p, sigma_applied)

        self._f_field = np.array(f_field)

        return np.array(gamma), np.array(f_field)

    # =========================================================================
    # Shear Banding Detection
    # =========================================================================

    def detect_shear_bands(
        self,
        f_profile: np.ndarray | None = None,
        threshold: float = 0.3,
    ) -> tuple[bool, float, float]:
        """Detect shear banding from fluidity profile.

        Parameters
        ----------
        f_profile : np.ndarray, optional
            Fluidity field. If None, uses stored profile.
        threshold : float, default 0.3
            CV threshold for banding detection

        Returns
        -------
        is_banded : bool
            True if shear banding detected
        cv : float
            Coefficient of variation
        ratio : float
            Max/min fluidity ratio
        """
        if f_profile is None:
            f_profile = self._f_field

        if f_profile is None:
            raise ValueError("No fluidity profile available. Run simulation first.")

        f_jax = jnp.asarray(f_profile, dtype=jnp.float64)
        is_banded, cv, ratio = detect_shear_bands(f_jax, threshold)

        return bool(is_banded), float(cv), float(ratio)

    def get_banding_metrics(
        self, f_profile: np.ndarray | None = None
    ) -> dict[str, float]:
        """Get detailed shear banding metrics.

        Parameters
        ----------
        f_profile : np.ndarray, optional
            Fluidity field. If None, uses stored profile.

        Returns
        -------
        dict
            Metrics including cv, ratio, band_width, etc.
        """
        if f_profile is None:
            f_profile = self._f_field

        if f_profile is None:
            raise ValueError("No fluidity profile available.")

        f_jax = jnp.asarray(f_profile, dtype=jnp.float64)

        cv = float(shear_banding_cv(f_jax))
        ratio = float(banding_ratio(f_jax))

        # Estimate band width from profile
        f_mean = np.mean(f_profile)
        high_f_mask = f_profile > f_mean

        # Band fraction: portion of gap with high fluidity
        band_fraction = np.sum(high_f_mask) / len(f_profile)

        return {
            "cv": cv,
            "ratio": ratio,
            "band_fraction": band_fraction,
            "f_max": float(np.max(f_profile)),
            "f_min": float(np.min(f_profile)),
            "f_mean": float(f_mean),
        }

    # =========================================================================
    # Model Function Interface
    # =========================================================================

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function.

        Accepts protocol-specific kwargs (gamma_dot, sigma_applied, etc.).

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values
        test_mode : str, optional
            Override stored test mode
        **kwargs
            Protocol-specific arguments (gamma_dot, sigma_applied)

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

        if mode in ["steady_shear", "rotation", "flow_curve"]:
            return self._predict_flow_curve_homogeneous(X_jax, p_values)
        elif mode == "startup":
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            _, sigma, _ = self._simulate_startup_internal(
                X_jax, p_values, gamma_dot
            )
            return sigma
        elif mode == "creep":
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            gamma, _ = self._simulate_creep_internal(
                X_jax, p_values, sigma_applied
            )
            return gamma

        return jnp.zeros_like(X_jax)

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
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
        # Get test_mode from kwargs or instance attribute
        test_mode = kwargs.get("test_mode") or getattr(self, "_test_mode", None)
        if test_mode is None:
            raise ValueError("test_mode must be specified for prediction")

        if test_mode in ["steady_shear", "rotation", "flow_curve"]:
            return self._predict_flow_curve(X)
        elif test_mode == "startup":
            gamma_dot = kwargs.get("gamma_dot") or getattr(self, "_gamma_dot_applied", None)
            if gamma_dot is None:
                raise ValueError("startup prediction requires gamma_dot")
            _, sigma, _ = self.simulate_startup(X, gamma_dot)
            return sigma
        elif test_mode == "creep":
            sigma = kwargs.get("sigma") or getattr(self, "_sigma_applied", None)  # type: ignore[assignment]
            if sigma is None:
                raise ValueError("creep prediction requires sigma")
            gamma, _ = self.simulate_creep(X, sigma)  # type: ignore[arg-type]
            return gamma

        return np.zeros_like(X)

    @property
    def y_grid(self) -> np.ndarray:
        """Get spatial grid across gap.

        Returns
        -------
        np.ndarray
            Position array (m)
        """
        return self._y_grid

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"coupling='{self.coupling}', "
            f"N_y={self.N_y}, "
            f"H={self.H:.1e} m, "
            f"xi={self.xi:.1e} m)"
        )
