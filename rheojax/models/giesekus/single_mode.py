"""Single-mode Giesekus nonlinear viscoelastic model.

This module implements `GiesekusSingleMode`, a differential constitutive
model for polymer melts and solutions with shear-thinning behavior.

Supported Protocols
-------------------
- FLOW_CURVE: Steady shear viscometry (analytical)
- OSCILLATION: Small-amplitude oscillatory shear (analytical)
- STARTUP: Transient stress at constant rate (ODE)
- RELAXATION: Stress decay after cessation of flow (ODE)
- CREEP: Strain evolution under constant stress (ODE)
- LAOS: Large-amplitude oscillatory shear (ODE + FFT)

Example
-------
>>> from rheojax.models.giesekus import GiesekusSingleMode
>>> import numpy as np
>>>
>>> # Create model
>>> model = GiesekusSingleMode()
>>>
>>> # Fit to flow curve data
>>> gamma_dot = np.logspace(-2, 2, 50)
>>> sigma = model.predict(gamma_dot, test_mode='flow_curve')
>>>
>>> # Fit to SAOS data
>>> omega = np.logspace(-1, 3, 50)
>>> G_star = model.predict(omega, test_mode='oscillation')
>>>
>>> # Simulate startup with stress overshoot
>>> t = np.linspace(0, 10, 500)
>>> sigma_t = model.simulate_startup(t, gamma_dot=10.0)

References
----------
- Giesekus, H. (1982). J. Non-Newtonian Fluid Mech. 11, 69-109.
- Bird, R.B. et al. (1987). Dynamics of Polymeric Liquids, Vol. 1.
"""

from __future__ import annotations

import logging

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import lazy_import, safe_import_jax

diffrax = lazy_import("diffrax")
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.giesekus._base import GiesekusBase
from rheojax.models.giesekus._kernels import (
    giesekus_creep_ode_rhs,
    giesekus_ode_rhs,
    giesekus_ode_rhs_laos,
    giesekus_relaxation_ode_rhs,
    giesekus_saos_moduli_vec,
    giesekus_steady_normal_stresses_vec,
    giesekus_steady_shear_stress_vec,
    giesekus_steady_stress_components,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "giesekus",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.LAOS,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
@ModelRegistry.register(
    "giesekus_single",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.LAOS,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class GiesekusSingleMode(GiesekusBase):
    """Single-mode Giesekus nonlinear viscoelastic model.

    The Giesekus model extends the Upper-Convected Maxwell framework with
    a quadratic stress term representing anisotropic molecular mobility::

        τ + λ∇̂τ + (αλ/η_p)τ·τ = 2η_p D

    This captures:

    1. **Shear-thinning**: Viscosity decreases with increasing shear rate
    2. **Normal stresses**: Both N₁ > 0 and N₂ < 0 (with N₂/N₁ = -α/2)
    3. **Stress overshoot**: Peak stress in startup flow
    4. **Nonlinear LAOS**: Higher harmonics in large-amplitude oscillation

    Parameters
    ----------
    eta_p : float
        Polymer viscosity η_p (Pa·s). Default: 100.0
    lambda_1 : float
        Relaxation time λ (s). Default: 1.0
    alpha : float
        Mobility factor α (dimensionless, 0 ≤ α ≤ 0.5). Default: 0.3
    eta_s : float
        Solvent viscosity η_s (Pa·s). Default: 0.0

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters for fitting
    fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    Basic fitting and prediction:

    >>> model = GiesekusSingleMode()
    >>> # Generate synthetic data
    >>> gamma_dot = np.logspace(-2, 2, 50)
    >>> sigma_data = model.predict(gamma_dot, test_mode='flow_curve')
    >>> # Fit to data
    >>> model.fit(gamma_dot, sigma_data, test_mode='flow_curve')

    Bayesian inference:

    >>> result = model.fit_bayesian(gamma_dot, sigma_data, test_mode='flow_curve')
    >>> print(result.diagnostics)

    See Also
    --------
    GiesekusMultiMode : Multi-mode extension with N relaxation times
    """

    def __init__(self):
        """Initialize single-mode Giesekus model."""
        super().__init__()
        self._test_mode = None

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
            # Estimate from flow curve shape
            eta_est = y_jax / jnp.maximum(x_jax, 1e-10)
            self.parameters.set_value("eta_p", float(jnp.mean(eta_est[:3])))
        elif test_mode == "oscillation":
            # Estimate from SAOS crossover
            # Handle both complex and 2-column format
            if y_jax.ndim == 2:
                G_prime = np.asarray(y_jax[:, 0])
                G_double_prime = np.asarray(y_jax[:, 1])
            elif np.iscomplexobj(y):
                G_prime = np.real(np.asarray(y))
                G_double_prime = np.imag(np.asarray(y))
            else:
                # Assume magnitude was passed, can't separate components
                G_prime = np.asarray(y) * 0.7  # Rough estimate
                G_double_prime = np.asarray(y) * 0.7
            self.initialize_from_saos(np.asarray(x), G_prime, G_double_prime)

        # Define model function for fitting (follows ParameterSet ordering)
        def model_fn(x_fit, params):
            """Stateless model function for optimization."""
            return self.model_function(x_fit, params, test_mode=test_mode)

        # Create objective and optimize using ParameterSet
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
            f"Fitted Giesekus: η_p={self.eta_p:.2e}, λ={self.lambda_1:.2e}, "
            f"α={self.alpha:.3f}, η_s={self.eta_s:.2e}"
        )

        return self

    def _predict(self, x, **kwargs):
        """Predict response using protocol-aware dispatch.

        Parameters
        ----------
        x : array-like
            Independent variable
        **kwargs
            Additional arguments including test_mode, sigma_applied, etc.

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        test_mode = kwargs.pop("test_mode", self._test_mode or "flow_curve")
        x_jax = jnp.asarray(x, dtype=jnp.float64)

        params = jnp.array([self.eta_p, self.lambda_1, self.alpha, self.eta_s])
        # Filter out BaseModel kwargs that model_function doesn't expect
        fwd_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("deformation_mode", "poisson_ratio")
        }
        return self.model_function(x_jax, params, test_mode=test_mode, **fwd_kwargs)

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function.

        Routes to appropriate prediction based on test_mode.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values [eta_p, lambda_1, alpha, eta_s]
        test_mode : str, optional
            Override stored test mode
        **kwargs : dict
            Protocol-specific arguments (gamma_dot, sigma_applied, gamma_0, etc.)

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        eta_p, lambda_1, alpha, eta_s = params
        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return giesekus_steady_shear_stress_vec(
                X_jax, eta_p, lambda_1, alpha, eta_s
            )

        elif mode == "oscillation":
            G_prime, G_double_prime = giesekus_saos_moduli_vec(
                X_jax, eta_p, lambda_1, eta_s
            )
            # Return components if requested or if we're in a context that expects them
            # We'll use a heuristic: if we're fitting and the target has 2 columns, return 2 columns
            return jnp.column_stack([G_prime, G_double_prime])

        elif mode == "startup":
            # Get gamma_dot from kwargs or instance attribute
            gamma_dot = kwargs.get("gamma_dot") or self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return self._simulate_startup_internal(
                X_jax, eta_p, lambda_1, alpha, gamma_dot
            )

        elif mode == "relaxation":
            # Get gamma_dot from kwargs or instance attribute
            gamma_dot = kwargs.get("gamma_dot") or self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError("relaxation mode requires gamma_dot (pre-shear rate)")
            return self._simulate_relaxation_internal(
                X_jax, eta_p, lambda_1, alpha, eta_s, gamma_dot
            )

        elif mode == "creep":
            # Get sigma_applied from kwargs or instance attribute
            sigma_applied = kwargs.get("sigma_applied") or self._sigma_applied
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            return self._simulate_creep_internal(
                X_jax, eta_p, lambda_1, alpha, eta_s, sigma_applied
            )

        elif mode == "laos":
            # Get gamma_0 and omega from kwargs or instance attributes
            gamma_0 = kwargs.get("gamma_0") or self._gamma_0
            omega = kwargs.get("omega") or self._omega_laos
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, eta_p, lambda_1, alpha, gamma_0, omega
            )
            return stress

        else:
            logger.warning(f"Unknown test_mode '{mode}', defaulting to flow_curve")
            return giesekus_steady_shear_stress_vec(
                X_jax, eta_p, lambda_1, alpha, eta_s
            )

    # =========================================================================
    # Analytical Predictions
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
        sigma = giesekus_steady_shear_stress_vec(
            gd, self.eta_p, self.lambda_1, self.alpha, self.eta_s
        )

        if return_components:
            eta = sigma / jnp.maximum(gd, 1e-20)
            N1, _ = giesekus_steady_normal_stresses_vec(
                gd, self.eta_p, self.lambda_1, self.alpha
            )
            return np.asarray(sigma), np.asarray(eta), np.asarray(N1)

        return np.asarray(sigma)

    def predict_saos(
        self,
        omega: np.ndarray,
        return_components: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

        In the linear viscoelastic regime, the Giesekus model reduces
        to Maxwell behavior (α-independent).

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
        G_prime, G_double_prime = giesekus_saos_moduli_vec(
            w, self.eta_p, self.lambda_1, self.eta_s
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

        The Giesekus model predicts::

            N₁ = τ_xx - τ_yy > 0  (first normal stress difference)
            N₂ = τ_yy - τ_zz < 0  (second normal stress difference)

        with the diagnostic ratio N₂/N₁ = -α/2.

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
        N1, N2 = giesekus_steady_normal_stresses_vec(
            gd, self.eta_p, self.lambda_1, self.alpha
        )
        return np.asarray(N1), np.asarray(N2)

    # =========================================================================
    # ODE-Based Simulations
    # =========================================================================

    def _simulate_startup_internal(
        self,
        t: jnp.ndarray,
        eta_p: float,
        lambda_1: float,
        alpha: float,
        gamma_dot: float,
    ) -> jnp.ndarray:
        """Internal startup simulation for model_function.

        Returns shear stress τ_xy(t).
        """

        # ODE setup
        def ode_fn(ti, yi, args):
            return giesekus_ode_rhs(
                ti,
                yi,
                args["gamma_dot"],
                args["eta_p"],
                args["lambda_1"],
                args["alpha"],
            )

        args = {
            "gamma_dot": gamma_dot,
            "eta_p": eta_p,
            "lambda_1": lambda_1,
            "alpha": alpha,
        }
        y0 = jnp.zeros(4, dtype=jnp.float64)

        term = diffrax.ODETerm(jax.checkpoint(ode_fn))
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
            max_steps=100_000,
            throw=False,
        )

        # Return τ_xy (index 2)
        result = sol.ys[:, 2]
        result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            result,
            jnp.nan * jnp.ones_like(result),
        )
        return result

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup flow at constant shear rate.

        The Giesekus model predicts stress overshoot in startup flow,
        where the stress first exceeds then relaxes to its steady-state
        value.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_full : bool, default False
            If True, return full stress tensor components

        Returns
        -------
        np.ndarray or tuple
            Shear stress τ_xy(t), or (τ_xx, τ_yy, τ_xy, τ_zz) if return_full=True
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return giesekus_ode_rhs(
                ti,
                yi,
                args["gamma_dot"],
                args["eta_p"],
                args["lambda_1"],
                args["alpha"],
            )

        args = {
            "gamma_dot": gamma_dot,
            "eta_p": self.eta_p,
            "lambda_1": self.lambda_1,
            "alpha": self.alpha,
        }
        y0 = jnp.zeros(4, dtype=jnp.float64)

        term = diffrax.ODETerm(jax.checkpoint(ode_fn))
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
            max_steps=100_000,
            throw=False,
        )

        # Handle solver failures
        tau_xx = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 0],
            jnp.nan * jnp.ones_like(sol.ys[:, 0]),
        )
        tau_yy = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 1],
            jnp.nan * jnp.ones_like(sol.ys[:, 1]),
        )
        tau_xy = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 2],
            jnp.nan * jnp.ones_like(sol.ys[:, 2]),
        )
        tau_zz = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 3],
            jnp.nan * jnp.ones_like(sol.ys[:, 3]),
        )

        # Store trajectory for debugging
        self._trajectory = {
            "t": np.asarray(t_jax),
            "tau_xx": np.asarray(tau_xx),
            "tau_yy": np.asarray(tau_yy),
            "tau_xy": np.asarray(tau_xy),
            "tau_zz": np.asarray(tau_zz),
        }

        if return_full:
            return (
                np.asarray(tau_xx),
                np.asarray(tau_yy),
                np.asarray(tau_xy),
                np.asarray(tau_zz),
            )

        return np.asarray(tau_xy)

    def _simulate_relaxation_internal(
        self,
        t: jnp.ndarray,
        eta_p: float,
        lambda_1: float,
        alpha: float,
        eta_s: float,
        gamma_dot_preshear: float,
    ) -> jnp.ndarray:
        """Internal relaxation simulation for model_function."""
        # Initial condition: steady state at pre-shear rate
        tau_xx, tau_yy, tau_xy, tau_zz = giesekus_steady_stress_components(
            gamma_dot_preshear, eta_p, lambda_1, alpha, eta_s
        )
        y0 = jnp.array([tau_xx, tau_yy, tau_xy, tau_zz], dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return giesekus_relaxation_ode_rhs(
                ti, yi, args["eta_p"], args["lambda_1"], args["alpha"]
            )

        args = {"eta_p": eta_p, "lambda_1": lambda_1, "alpha": alpha}

        term = diffrax.ODETerm(jax.checkpoint(ode_fn))
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
            max_steps=100_000,
            throw=False,
        )

        result = sol.ys[:, 2]  # τ_xy
        result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            result,
            jnp.nan * jnp.ones_like(result),
        )
        return result

    def simulate_relaxation(
        self,
        t: np.ndarray,
        gamma_dot_preshear: float,
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate stress relaxation after cessation of steady shear.

        The Giesekus model predicts faster-than-exponential relaxation
        due to the quadratic τ·τ term.

        Parameters
        ----------
        t : np.ndarray
            Time array (s), starting from t=0 (cessation)
        gamma_dot_preshear : float
            Shear rate before cessation (1/s)
        return_full : bool, default False
            If True, return full stress tensor components

        Returns
        -------
        np.ndarray or tuple
            Relaxing stress τ_xy(t), or (τ_xx, τ_yy, τ_xy, τ_zz) if return_full
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)

        # Initial condition: steady state
        tau_xx, tau_yy, tau_xy, tau_zz = giesekus_steady_stress_components(
            gamma_dot_preshear, self.eta_p, self.lambda_1, self.alpha, self.eta_s
        )
        y0 = jnp.array([tau_xx, tau_yy, tau_xy, tau_zz], dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return giesekus_relaxation_ode_rhs(
                ti, yi, args["eta_p"], args["lambda_1"], args["alpha"]
            )

        args = {"eta_p": self.eta_p, "lambda_1": self.lambda_1, "alpha": self.alpha}

        term = diffrax.ODETerm(jax.checkpoint(ode_fn))
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
            max_steps=100_000,
            throw=False,
        )

        # Handle solver failures
        tau_xx = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 0],
            jnp.nan * jnp.ones_like(sol.ys[:, 0]),
        )
        tau_yy = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 1],
            jnp.nan * jnp.ones_like(sol.ys[:, 1]),
        )
        tau_xy = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 2],
            jnp.nan * jnp.ones_like(sol.ys[:, 2]),
        )
        tau_zz = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 3],
            jnp.nan * jnp.ones_like(sol.ys[:, 3]),
        )

        self._trajectory = {
            "t": np.asarray(t_jax),
            "tau_xx": np.asarray(tau_xx),
            "tau_yy": np.asarray(tau_yy),
            "tau_xy": np.asarray(tau_xy),
            "tau_zz": np.asarray(tau_zz),
        }

        if return_full:
            return (
                np.asarray(tau_xx),
                np.asarray(tau_yy),
                np.asarray(tau_xy),
                np.asarray(tau_zz),
            )

        return np.asarray(tau_xy)

    def _simulate_creep_internal(
        self,
        t: jnp.ndarray,
        eta_p: float,
        lambda_1: float,
        alpha: float,
        eta_s: float,
        sigma_applied: float,
    ) -> jnp.ndarray:
        """Internal creep simulation for model_function."""
        # State: [τ_xx, τ_yy, τ_xy, τ_zz, γ]
        y0 = jnp.zeros(5, dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return giesekus_creep_ode_rhs(
                ti,
                yi,
                args["sigma"],
                args["eta_p"],
                args["lambda_1"],
                args["alpha"],
                args["eta_s"],
            )

        args = {
            "sigma": sigma_applied,
            "eta_p": eta_p,
            "lambda_1": lambda_1,
            "alpha": alpha,
            "eta_s": eta_s,
        }

        term = diffrax.ODETerm(jax.checkpoint(ode_fn))
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
            max_steps=100_000,
            throw=False,
        )

        result = sol.ys[:, 4]  # γ (strain)
        result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            result,
            jnp.nan * jnp.ones_like(result),
        )
        return result

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

        # State: [τ_xx, τ_yy, τ_xy, τ_zz, γ]
        y0 = jnp.zeros(5, dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return giesekus_creep_ode_rhs(
                ti,
                yi,
                args["sigma"],
                args["eta_p"],
                args["lambda_1"],
                args["alpha"],
                args["eta_s"],
            )

        args = {
            "sigma": sigma_applied,
            "eta_p": self.eta_p,
            "lambda_1": self.lambda_1,
            "alpha": self.alpha,
            "eta_s": self.eta_s,
        }

        term = diffrax.ODETerm(jax.checkpoint(ode_fn))
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
            max_steps=100_000,
            throw=False,
        )

        # Handle solver failures
        gamma_result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 4],
            jnp.nan * jnp.ones_like(sol.ys[:, 4]),
        )
        tau_xy_result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 2],
            jnp.nan * jnp.ones_like(sol.ys[:, 2]),
        )

        gamma = np.asarray(gamma_result)
        tau_xy = np.asarray(tau_xy_result)

        self._trajectory = {
            "t": np.asarray(t_jax),
            "gamma": gamma,
            "tau_xy": tau_xy,
        }

        if return_rate:
            # Compute γ̇ = (σ - τ_xy) / η_s
            eta_s_reg = max(self.eta_s, 1e-10 * self.eta_p)
            gamma_dot = (sigma_applied - tau_xy) / eta_s_reg
            return gamma, gamma_dot

        return gamma

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        eta_p: float,
        lambda_1: float,
        alpha: float,
        gamma_0: float,
        omega: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal LAOS simulation for model_function."""
        y0 = jnp.zeros(4, dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return giesekus_ode_rhs_laos(
                ti,
                yi,
                args["gamma_0"],
                args["omega"],
                args["eta_p"],
                args["lambda_1"],
                args["alpha"],
            )

        args = {
            "gamma_0": gamma_0,
            "omega": omega,
            "eta_p": eta_p,
            "lambda_1": lambda_1,
            "alpha": alpha,
        }

        term = diffrax.ODETerm(jax.checkpoint(ode_fn))
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
            max_steps=100_000,
            throw=False,
        )

        # Handle solver failures
        stress_result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            sol.ys[:, 2],
            jnp.nan * jnp.ones_like(sol.ys[:, 2]),
        )

        # Strain and stress
        strain = gamma_0 * jnp.sin(omega * t)
        stress = stress_result  # τ_xy

        return strain, stress

    def simulate_laos(
        self,
        t: np.ndarray,
        gamma_0: float,
        omega: float,
        n_cycles: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Simulate Large-Amplitude Oscillatory Shear (LAOS).

        The Giesekus model produces nonlinear stress response in LAOS,
        characterized by higher harmonics (I₃, I₅, ...).

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
            Dictionary with keys:
            - 't': Time array
            - 'strain': γ(t) = γ₀·sin(ωt)
            - 'stress': τ_xy(t)
            - 'strain_rate': γ̇(t) = γ₀·ω·cos(ωt)
        """
        if n_cycles is not None:
            T = 2 * np.pi / omega
            t = np.linspace(0, n_cycles * T, n_cycles * 200)

        t_jax = jnp.asarray(t, dtype=jnp.float64)

        strain, stress = self._simulate_laos_internal(
            t_jax, self.eta_p, self.lambda_1, self.alpha, gamma_0, omega
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

    def extract_laos_harmonics(
        self,
        laos_result: dict[str, np.ndarray],
        n_harmonics: int = 5,
    ) -> dict[str, np.ndarray]:
        """Extract Fourier harmonics from LAOS stress response.

        The nonlinear stress response can be decomposed as::

            σ(t) = Σ [σ'_n·sin(nωt) + σ''_n·cos(nωt)]

        Only odd harmonics (n=1,3,5,...) are expected for symmetric response.

        Parameters
        ----------
        laos_result : dict
            Result from simulate_laos()
        n_harmonics : int, default 5
            Number of harmonics to extract

        Returns
        -------
        dict
            Dictionary with:
            - 'n': Harmonic indices [1, 3, 5, ...]
            - 'sigma_prime': In-phase (elastic) components
            - 'sigma_double_prime': Out-of-phase (viscous) components
            - 'intensity': |σ_n| = sqrt(σ'_n² + σ''_n²)
        """
        t = laos_result["t"]
        stress = laos_result["stress"]
        strain = laos_result["strain"]

        # Estimate omega from zero crossings or FFT
        fft_strain = np.fft.fft(strain)
        freqs = np.fft.fftfreq(len(t), t[1] - t[0])
        omega = 2 * np.pi * np.abs(freqs[np.argmax(np.abs(fft_strain[1:])) + 1])

        # Extract harmonics
        harmonics = [2 * i + 1 for i in range(n_harmonics)]  # 1, 3, 5, ...
        sigma_prime = []
        sigma_double_prime = []

        for n in harmonics:
            # Project onto sin(nωt) and cos(nωt)
            sin_basis = np.sin(n * omega * t)
            cos_basis = np.cos(n * omega * t)

            # Numerical integration (trapezoidal)
            dt = t[1] - t[0]
            sigma_n_prime = 2 * np.trapezoid(stress * sin_basis, dx=dt) / (t[-1] - t[0])
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
            ),  # I₃/I₁ ratio
        }

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_overshoot_ratio(
        self,
        gamma_dot: float,
        t_max: float | None = None,
    ) -> tuple[float, float]:
        """Compute stress overshoot ratio in startup flow.

        The overshoot ratio is defined as::

            overshoot_ratio = σ_max / σ_ss

        where σ_max is the peak stress and σ_ss is the steady-state stress.

        Parameters
        ----------
        gamma_dot : float
            Shear rate (1/s)
        t_max : float, optional
            Maximum simulation time (default: 20λ)

        Returns
        -------
        tuple[float, float]
            (overshoot_ratio, strain_at_overshoot)
        """
        if t_max is None:
            t_max = 20 * self.lambda_1

        t = np.linspace(0, t_max, 1000)
        sigma = self.simulate_startup(t, gamma_dot)

        # Find peak
        peak_idx = np.argmax(sigma)
        sigma_max = sigma[peak_idx]
        strain_at_peak = gamma_dot * t[peak_idx]

        # Steady state
        sigma_ss = sigma[-1]

        overshoot_ratio = sigma_max / sigma_ss if sigma_ss > 0 else 1.0

        return overshoot_ratio, strain_at_peak

    def get_relaxation_spectrum(
        self,
        t: np.ndarray | None = None,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get effective relaxation spectrum G(t).

        Note: For single-mode Giesekus, this is single-exponential in
        the linear regime but deviates due to nonlinearity.

        Parameters
        ----------
        t : np.ndarray, optional
            Time array (default: logspace from 0.01λ to 100λ)
        n_points : int, default 100
            Number of points if t not provided

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t, G(t))
        """
        if t is None:
            t = np.logspace(
                np.log10(0.01 * self.lambda_1),
                np.log10(100 * self.lambda_1),
                n_points,
            )

        # Linear Maxwell relaxation
        G = self.eta_p / self.lambda_1
        G_t = G * np.exp(-t / self.lambda_1)

        return t, G_t
