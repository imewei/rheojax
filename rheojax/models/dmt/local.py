"""Local (0D) de Souza Mendes-Thompson (DMT) model.

Implements the homogeneous DMT model for thixotropic yield-stress fluids
with optional Maxwell viscoelastic backbone.

Supports all standard rheological protocols:
- Flow curve (steady shear)
- Start-up shear (stress overshoot)
- Stress relaxation (Maxwell only)
- Creep (delayed yielding)
- SAOS (Maxwell only)
- LAOS (nonlinear oscillatory)
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.logging import get_logger
from rheojax.models.dmt._base import DMTBase
from rheojax.models.dmt._kernels import (
    elastic_modulus,
    equilibrium_structure,
    invert_stress_for_gamma_dot_exponential,
    invert_stress_for_gamma_dot_hb,
    maxwell_stress_evolution,
    saos_moduli_maxwell,
    steady_stress_exponential,
    steady_stress_herschel_bulkley,
    structure_evolution,
    viscosity_exponential,
    viscosity_herschel_bulkley_regularized,
)

# Safe JAX import
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "dmt_local",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.OSCILLATION,
        Protocol.LAOS,
    ],
)
class DMTLocal(DMTBase):
    """Local (0D) DMT model for homogeneous thixotropic flow.

    This model assumes spatially homogeneous flow (no shear banding).
    For shear banding analysis, use DMTNonlocal.

    The model captures:
    - **Yielding**: Stress plateau at low shear rates (HB closure)
    - **Thixotropy**: Time-dependent viscosity via structure kinetics
    - **Viscoelasticity**: Optional Maxwell backbone for overshoot/SAOS

    Two viscosity closures:
    - "exponential": η(λ) = η_∞·(η_0/η_∞)^λ (smooth, original DMT)
    - "herschel_bulkley": Explicit yield stress τ_y(λ) + K(λ)|γ̇|^n

    Parameters
    ----------
    closure : {"exponential", "herschel_bulkley"}, default "exponential"
        Viscosity closure type.
    include_elasticity : bool, default True
        Include Maxwell viscoelastic backbone for stress overshoot and SAOS.

    Examples
    --------
    >>> from rheojax.models.dmt import DMTLocal
    >>>
    >>> # Create model with Herschel-Bulkley closure
    >>> model = DMTLocal(closure="herschel_bulkley", include_elasticity=True)
    >>>
    >>> # Fit to flow curve data
    >>> model.fit(gamma_dot, stress, test_mode="flow_curve")
    >>>
    >>> # Simulate startup shear
    >>> t, stress, lam = model.simulate_startup(gamma_dot=10.0, t_end=100.0)

    See Also
    --------
    DMTNonlocal : Nonlocal (1D) variant with shear banding
    FluidityLocal : Simpler fluidity-based thixotropic model

    References
    ----------
    de Souza Mendes, P.R. & Thompson, R.L. (2013).
        "A unified approach to model elasto-viscoplastic thixotropic
        yield-stress materials and apparent yield-stress fluids."
        Rheol. Acta 52, 673-694.
    """

    def __init__(
        self,
        closure: Literal["exponential", "herschel_bulkley"] = "exponential",
        include_elasticity: bool = True,
    ):
        """Initialize DMTLocal model."""
        super().__init__(closure=closure, include_elasticity=include_elasticity)
        logger.info(
            "DMTLocal initialized",
            closure=closure,
            include_elasticity=include_elasticity,
        )

    # =========================================================================
    # Required Abstract Methods
    # =========================================================================

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "DMTLocal":
        """Fit model to data.

        Dispatches to protocol-specific fitting method based on test_mode.

        Parameters
        ----------
        X : array
            Independent variable (γ̇ for flow_curve, t for transients)
        y : array
            Dependent variable (σ for flow_curve/startup, γ for creep)
        **kwargs
            Additional arguments including test_mode

        Returns
        -------
        self
            Fitted model instance
        """
        test_mode = kwargs.get("test_mode", "flow_curve")

        if test_mode in ("flow_curve", "rotation"):
            return self._fit_flow_curve(X, y, **kwargs)
        elif test_mode == "startup":
            return self._fit_transient(X, y, **kwargs)
        elif test_mode == "relaxation":
            if not self.include_elasticity:
                raise ValueError(
                    "Relaxation requires include_elasticity=True (DMT-Maxwell)"
                )
            return self._fit_relaxation(X, y, **kwargs)
        elif test_mode == "creep":
            return self._fit_creep(X, y, **kwargs)
        elif test_mode == "oscillation":
            if not self.include_elasticity:
                raise ValueError(
                    "SAOS requires include_elasticity=True (DMT-Maxwell)"
                )
            return self._fit_oscillation(X, y, **kwargs)
        elif test_mode == "laos":
            return self._fit_laos(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict model response.

        Dispatches to protocol-specific prediction method based on test_mode.

        Parameters
        ----------
        X : array
            Independent variable
        **kwargs
            Additional arguments including test_mode

        Returns
        -------
        array
            Predicted response
        """
        test_mode = kwargs.get("test_mode", "flow_curve")

        if test_mode in ("flow_curve", "rotation"):
            return self._predict_flow_curve(X)
        elif test_mode == "startup":
            return self._predict_startup(X, **kwargs)
        elif test_mode == "relaxation":
            return self._predict_relaxation(X, **kwargs)
        elif test_mode == "creep":
            return self._predict_creep(X, **kwargs)
        elif test_mode == "oscillation":
            return self._predict_oscillation(X, **kwargs)
        else:
            raise ValueError(f"Unknown test_mode for prediction: {test_mode}")

    # =========================================================================
    # Flow Curve (Steady Shear)
    # =========================================================================

    def _fit_flow_curve(
        self, gamma_dot: np.ndarray, stress: np.ndarray, **kwargs
    ) -> "DMTLocal":
        """Fit to steady-state flow curve σ(γ̇).

        Uses NLSQ to optimize parameters to match equilibrium stress-rate curve.

        Parameters
        ----------
        gamma_dot : array
            Shear rate array [1/s]
        stress : array
            Stress array [Pa]
        **kwargs
            Fitting options

        Returns
        -------
        self
            Fitted model
        """
        from rheojax.core.parameters import ParameterSet
        from rheojax.utils.optimization import nlsq_curve_fit

        # Convert to numpy
        gamma_dot_np = np.asarray(gamma_dot, dtype=np.float64)
        stress_np = np.asarray(stress, dtype=np.float64)

        # Create a ParameterSet with only flow curve parameters
        if self.closure == "exponential":
            param_names = ["eta_0", "eta_inf", "a", "c"]
        else:
            param_names = ["tau_y0", "K0", "n_flow", "eta_inf", "a", "c", "m1", "m2"]

        fit_params = ParameterSet()
        for name in param_names:
            param = self.parameters[name]
            fit_params.add(
                name=name,
                value=param.value,
                bounds=param.bounds,
                units=param.units,
                description=param.description,
            )

        # Define model function f(x, params_array) -> y_pred
        def model_fn(x, params_array):
            if self.closure == "exponential":
                eta_0, eta_inf, a, c = params_array[:4]
                return np.array(steady_stress_exponential(
                    jnp.array(x), eta_0, eta_inf, a, c
                ))
            else:
                tau_y0, K0, n_flow, eta_inf, a, c, m1, m2 = params_array[:8]
                return np.array(steady_stress_herschel_bulkley(
                    jnp.array(x), tau_y0, K0, n_flow, eta_inf, a, c, m1, m2
                ))

        # Fit using nlsq_curve_fit
        result = nlsq_curve_fit(
            model_fn, gamma_dot_np, stress_np, fit_params, **kwargs
        )

        # Update main parameters with fitted values
        for name in param_names:
            self.parameters[name].value = fit_params[name].value

        self._fitted = True
        self._fit_result = result
        logger.info(
            "DMTLocal flow curve fit complete",
            r_squared=result.r_squared,
            rmse=result.rmse,
        )

        return self

    def _predict_flow_curve(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict steady-state stress from flow curve.

        Parameters
        ----------
        gamma_dot : array
            Shear rate [1/s]

        Returns
        -------
        array
            Predicted stress [Pa]
        """
        gamma_dot_jax = jnp.array(gamma_dot)
        params = self.get_parameter_dict()

        if self.closure == "exponential":
            stress = steady_stress_exponential(
                gamma_dot_jax,
                params["eta_0"],
                params["eta_inf"],
                params["a"],
                params["c"],
            )
        else:
            stress = steady_stress_herschel_bulkley(
                gamma_dot_jax,
                params["tau_y0"],
                params["K0"],
                params["n_flow"],
                params["eta_inf"],
                params["a"],
                params["c"],
                params["m1"],
                params["m2"],
            )

        return np.array(stress)

    # =========================================================================
    # Startup Shear
    # =========================================================================

    def simulate_startup(
        self,
        gamma_dot: float,
        t_end: float,
        dt: float = 0.01,
        lam_init: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup of steady shear from rest.

        Parameters
        ----------
        gamma_dot : float
            Applied constant shear rate [1/s]
        t_end : float
            Simulation end time [s]
        dt : float
            Time step [s]
        lam_init : float
            Initial structure parameter (default: 1.0, fully structured)

        Returns
        -------
        t : array
            Time array [s]
        stress : array
            Stress response [Pa]
        lam : array
            Structure parameter evolution
        """
        n_steps = int(t_end / dt)
        t = jnp.linspace(0, t_end, n_steps)
        params = self.get_parameter_dict()

        if self.include_elasticity:
            return self._simulate_startup_maxwell(t, dt, gamma_dot, lam_init, params)
        else:
            return self._simulate_startup_viscous(t, dt, gamma_dot, lam_init, params)

    def _simulate_startup_viscous(
        self,
        t: jnp.ndarray,
        dt: float,
        gamma_dot: float,
        lam_init: float,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup for DMT-Viscous (no elasticity)."""

        def step(lam, _):
            # Structure evolution
            dlam = structure_evolution(lam, gamma_dot, params["t_eq"], params["a"], params["c"])
            lam_new = jnp.clip(lam + dt * dlam, 0.0, 1.0)

            # Viscosity
            if self.closure == "exponential":
                eta = viscosity_exponential(lam_new, params["eta_0"], params["eta_inf"])
            else:
                eta = viscosity_herschel_bulkley_regularized(
                    lam_new, gamma_dot, params["tau_y0"], params["K0"],
                    params["n_flow"], params["eta_inf"], params["m1"], params["m2"]
                )

            stress = eta * gamma_dot
            return lam_new, (stress, lam_new)

        _, (stress, lam) = jax.lax.scan(step, lam_init, None, length=len(t))

        return np.array(t), np.array(stress), np.array(lam)

    def _simulate_startup_maxwell(
        self,
        t: jnp.ndarray,
        dt: float,
        gamma_dot: float,
        lam_init: float,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup for DMT-Maxwell (with elasticity)."""

        def step(state, _):
            sigma, lam = state

            # Structure evolution
            dlam = structure_evolution(lam, gamma_dot, params["t_eq"], params["a"], params["c"])
            lam_new = jnp.clip(lam + dt * dlam, 0.0, 1.0)

            # Elastic modulus
            G = elastic_modulus(lam_new, params["G0"], params["m_G"])

            # Viscosity
            if self.closure == "exponential":
                eta = viscosity_exponential(lam_new, params["eta_0"], params["eta_inf"])
            else:
                eta = viscosity_herschel_bulkley_regularized(
                    lam_new, gamma_dot, params["tau_y0"], params["K0"],
                    params["n_flow"], params["eta_inf"], params["m1"], params["m2"]
                )

            # Relaxation time
            theta_1 = eta / jnp.maximum(G, 1e-10)

            # Stress evolution
            dsigma = maxwell_stress_evolution(sigma, gamma_dot, G, theta_1)
            sigma_new = sigma + dt * dsigma

            return (sigma_new, lam_new), (sigma_new, lam_new)

        init_state = (0.0, lam_init)  # Zero initial stress
        _, (stress, lam) = jax.lax.scan(step, init_state, None, length=len(t))

        return np.array(t), np.array(stress), np.array(lam)

    def _fit_transient(
        self, t: np.ndarray, stress: np.ndarray, **kwargs
    ) -> "DMTLocal":
        """Fit to transient startup data."""
        # Extract gamma_dot from kwargs
        gamma_dot = kwargs.get("gamma_dot", 1.0)
        lam_init = kwargs.get("lam_init", 1.0)

        from rheojax.utils.optimization import fit_with_nlsq

        t_jax = jnp.array(t)
        stress_jax = jnp.array(stress)
        dt = float(t[1] - t[0])

        def residual_fn(params_array):
            # Reconstruct parameter dict
            param_dict = self._params_array_to_dict(params_array)
            _, stress_pred, _ = self._simulate_with_params(
                t_jax, dt, gamma_dot, lam_init, param_dict
            )
            return stress_pred - stress_jax

        params_array, bounds = self._get_params_for_optimization()

        result = fit_with_nlsq(
            residual_fn, params_array, bounds=bounds, **kwargs
        )

        self._set_params_from_array(result.x)
        self._fitted = True

        return self

    def _predict_startup(self, t: np.ndarray, **kwargs) -> np.ndarray:
        """Predict startup stress."""
        gamma_dot = kwargs.get("gamma_dot", 1.0)
        lam_init = kwargs.get("lam_init", 1.0)
        _, stress, _ = self.simulate_startup(gamma_dot, float(t[-1]), float(t[1] - t[0]), lam_init)
        return stress

    # =========================================================================
    # Stress Relaxation (Maxwell only)
    # =========================================================================

    def simulate_relaxation(
        self,
        t_end: float,
        dt: float = 0.01,
        sigma_init: float = 100.0,
        lam_init: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate stress relaxation after cessation of shear.

        Requires include_elasticity=True.

        Parameters
        ----------
        t_end : float
            Simulation end time [s]
        dt : float
            Time step [s]
        sigma_init : float
            Initial stress at cessation [Pa]
        lam_init : float
            Initial structure at cessation

        Returns
        -------
        t : array
            Time array [s]
        stress : array
            Relaxing stress [Pa]
        lam : array
            Recovering structure
        """
        if not self.include_elasticity:
            raise ValueError("Stress relaxation requires include_elasticity=True")

        n_steps = int(t_end / dt)
        t = jnp.linspace(0, t_end, n_steps)
        params = self.get_parameter_dict()

        def step(state, _):
            sigma, lam = state

            # Structure recovery (no breakdown, γ̇ = 0)
            dlam = (1.0 - lam) / params["t_eq"]
            lam_new = jnp.clip(lam + dt * dlam, 0.0, 1.0)

            # Elastic modulus
            G = elastic_modulus(lam_new, params["G0"], params["m_G"])

            # Viscosity at zero shear rate
            if self.closure == "exponential":
                eta = viscosity_exponential(lam_new, params["eta_0"], params["eta_inf"])
            else:
                eta = params["eta_inf"]  # HB at zero shear rate

            # Relaxation time
            theta_1 = eta / jnp.maximum(G, 1e-10)

            # Stress relaxation
            dsigma = -sigma / jnp.maximum(theta_1, 1e-12)
            sigma_new = sigma + dt * dsigma

            return (sigma_new, lam_new), (sigma_new, lam_new)

        init_state = (sigma_init, lam_init)
        _, (stress, lam) = jax.lax.scan(step, init_state, None, length=n_steps)

        return np.array(t), np.array(stress), np.array(lam)

    def _fit_relaxation(self, t: np.ndarray, stress: np.ndarray, **kwargs) -> "DMTLocal":
        """Fit to relaxation data."""
        # Implementation similar to _fit_transient
        raise NotImplementedError("Relaxation fitting not yet implemented")

    def _predict_relaxation(self, t: np.ndarray, **kwargs) -> np.ndarray:
        """Predict relaxation stress."""
        sigma_init = kwargs.get("sigma_init", 100.0)
        lam_init = kwargs.get("lam_init", 0.5)
        _, stress, _ = self.simulate_relaxation(
            float(t[-1]), float(t[1] - t[0]), sigma_init, lam_init
        )
        return stress

    # =========================================================================
    # Creep
    # =========================================================================

    def simulate_creep(
        self,
        sigma_0: float,
        t_end: float,
        dt: float = 0.01,
        lam_init: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate creep under constant applied stress.

        For the Maxwell variant (include_elasticity=True), the total strain
        includes both elastic and viscous contributions:

        γ(t) = γ_e(t) + γ_v(t)

        where:
        - γ_e(t) = σ₀/G(λ(t)) is the elastic strain (changes with structure)
        - γ_v(t) = ∫₀ᵗ σ₀/η(λ(s)) ds is the viscous strain

        This correctly captures:
        - Initial elastic jump: γ(0⁺) = σ₀/G(λ_init)
        - Elastic strain recovery/growth as structure evolves
        - Viscous flow accumulation

        Parameters
        ----------
        sigma_0 : float
            Applied constant stress [Pa]
        t_end : float
            Simulation end time [s]
        dt : float
            Time step [s]
        lam_init : float
            Initial structure parameter

        Returns
        -------
        t : array
            Time array [s]
        gamma : array
            Total accumulated strain (elastic + viscous for Maxwell variant)
        gamma_dot : array
            Total shear rate evolution [1/s]
        lam : array
            Structure parameter evolution
        """
        n_steps = int(t_end / dt)
        t = jnp.linspace(0, t_end, n_steps)
        params = self.get_parameter_dict()

        if self.include_elasticity:
            # Maxwell variant: track elastic and viscous strain separately
            return self._simulate_creep_maxwell(
                t, dt, n_steps, sigma_0, lam_init, params
            )
        else:
            # Viscous variant: purely viscous flow
            return self._simulate_creep_viscous(
                t, dt, n_steps, sigma_0, lam_init, params
            )

    def _simulate_creep_viscous(
        self,
        t: jnp.ndarray,
        dt: float,
        n_steps: int,
        sigma_0: float,
        lam_init: float,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate creep for DMT-Viscous (no elasticity).

        Pure viscous flow: γ̇ = σ₀/η(λ)
        """

        def step(state, _):
            lam, gamma = state

            # Viscous flow rate: γ̇ = σ₀/η(λ)
            if self.closure == "exponential":
                gamma_dot = invert_stress_for_gamma_dot_exponential(
                    sigma_0, lam, params["eta_0"], params["eta_inf"]
                )
            else:
                gamma_dot = invert_stress_for_gamma_dot_hb(
                    sigma_0, lam, params["tau_y0"], params["K0"],
                    params["n_flow"], params["eta_inf"], params["m1"], params["m2"]
                )

            # Structure evolution (driven by viscous flow rate)
            dlam = structure_evolution(lam, gamma_dot, params["t_eq"], params["a"], params["c"])
            lam_new = jnp.clip(lam + dt * dlam, 0.0, 1.0)

            # Strain accumulation
            gamma_new = gamma + dt * gamma_dot

            return (lam_new, gamma_new), (gamma_new, gamma_dot, lam_new)

        init_state = (lam_init, 0.0)
        _, (gamma, gamma_dot, lam) = jax.lax.scan(step, init_state, None, length=n_steps)

        return np.array(t), np.array(gamma), np.array(gamma_dot), np.array(lam)

    def _simulate_creep_maxwell(
        self,
        t: jnp.ndarray,
        dt: float,
        n_steps: int,
        sigma_0: float,
        lam_init: float,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate creep for DMT-Maxwell (with elasticity).

        Total strain: γ = γ_e + γ_v
        - Elastic: γ_e = σ₀/G(λ)
        - Viscous: dγ_v/dt = σ₀/η(λ)

        The total shear rate includes both viscous flow and elastic strain change:
        γ̇ = dγ_v/dt + dγ_e/dt = σ₀/η + d(σ₀/G)/dt

        Structure evolution uses the viscous flow rate as the driving deformation.
        """

        def step(state, _):
            lam, gamma_v, lam_prev = state

            # Compute elastic modulus and strain
            G = elastic_modulus(lam, params["G0"], params["m_G"])
            gamma_e = sigma_0 / jnp.maximum(G, 1e-10)

            # Previous elastic strain (for rate calculation)
            G_prev = elastic_modulus(lam_prev, params["G0"], params["m_G"])
            gamma_e_prev = sigma_0 / jnp.maximum(G_prev, 1e-10)

            # Viscous flow rate: γ̇_v = σ₀/η(λ)
            if self.closure == "exponential":
                gamma_dot_v = invert_stress_for_gamma_dot_exponential(
                    sigma_0, lam, params["eta_0"], params["eta_inf"]
                )
            else:
                gamma_dot_v = invert_stress_for_gamma_dot_hb(
                    sigma_0, lam, params["tau_y0"], params["K0"],
                    params["n_flow"], params["eta_inf"], params["m1"], params["m2"]
                )

            # Elastic strain rate (from structure change)
            gamma_dot_e = (gamma_e - gamma_e_prev) / dt

            # Total shear rate
            gamma_dot_total = gamma_dot_v + gamma_dot_e

            # Structure evolution (driven by viscous flow rate)
            # Use viscous rate since that represents actual material deformation
            dlam = structure_evolution(lam, gamma_dot_v, params["t_eq"], params["a"], params["c"])
            lam_new = jnp.clip(lam + dt * dlam, 0.0, 1.0)

            # Viscous strain accumulation
            gamma_v_new = gamma_v + dt * gamma_dot_v

            # Total strain = elastic + viscous
            gamma_total = gamma_e + gamma_v_new

            return (lam_new, gamma_v_new, lam), (gamma_total, gamma_dot_total, lam_new)

        # Initial elastic strain (instantaneous response to applied stress)
        G_init = elastic_modulus(lam_init, params["G0"], params["m_G"])
        gamma_e_init = sigma_0 / jnp.maximum(G_init, 1e-10)

        # State: (λ, γ_v, λ_prev)
        init_state = (lam_init, 0.0, lam_init)
        _, (gamma, gamma_dot, lam) = jax.lax.scan(step, init_state, None, length=n_steps)

        return np.array(t), np.array(gamma), np.array(gamma_dot), np.array(lam)

    def _fit_creep(self, t: np.ndarray, gamma: np.ndarray, **kwargs) -> "DMTLocal":
        """Fit to creep data."""
        raise NotImplementedError("Creep fitting not yet implemented")

    def _predict_creep(self, t: np.ndarray, **kwargs) -> np.ndarray:
        """Predict creep strain."""
        sigma_0 = kwargs.get("sigma_0", 10.0)
        lam_init = kwargs.get("lam_init", 1.0)
        _, gamma, _, _ = self.simulate_creep(
            sigma_0, float(t[-1]), float(t[1] - t[0]), lam_init
        )
        return gamma

    # =========================================================================
    # SAOS (Maxwell only)
    # =========================================================================

    def predict_saos(
        self, omega: np.ndarray, lam_0: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict SAOS moduli G'(ω) and G''(ω).

        Requires include_elasticity=True.
        Assumes small amplitude so structure remains at λ₀.

        Parameters
        ----------
        omega : array
            Angular frequency [rad/s]
        lam_0 : float
            Reference structure level (default: 1.0, fully structured)

        Returns
        -------
        G_prime : array
            Storage modulus G'(ω) [Pa]
        G_double_prime : array
            Loss modulus G''(ω) [Pa]
        """
        if not self.include_elasticity:
            raise ValueError("SAOS requires include_elasticity=True")

        omega_jax = jnp.array(omega)
        params = self.get_parameter_dict()

        # Get G and η at reference structure
        G = elastic_modulus(lam_0, params["G0"], params["m_G"])

        if self.closure == "exponential":
            eta = viscosity_exponential(lam_0, params["eta_0"], params["eta_inf"])
        else:
            # HB at low shear rate
            eta = viscosity_herschel_bulkley_regularized(
                lam_0, 1e-6, params["tau_y0"], params["K0"],
                params["n_flow"], params["eta_inf"], params["m1"], params["m2"]
            )

        theta_1 = eta / jnp.maximum(G, 1e-10)

        G_prime, G_double_prime = saos_moduli_maxwell(
            omega_jax, float(G), float(theta_1), params["eta_inf"]
        )

        return np.array(G_prime), np.array(G_double_prime)

    def _fit_oscillation(self, omega: np.ndarray, G_star: np.ndarray, **kwargs) -> "DMTLocal":
        """Fit to SAOS data."""
        raise NotImplementedError("SAOS fitting not yet implemented")

    def _predict_oscillation(self, omega: np.ndarray, **kwargs) -> np.ndarray:
        """Predict complex modulus."""
        lam_0 = kwargs.get("lam_0", 1.0)
        G_prime, G_double_prime = self.predict_saos(omega, lam_0)
        return G_prime + 1j * G_double_prime

    # =========================================================================
    # LAOS
    # =========================================================================

    def simulate_laos(
        self,
        gamma_0: float,
        omega: float,
        n_cycles: int = 10,
        points_per_cycle: int = 128,
        lam_init: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Simulate LAOS (Large Amplitude Oscillatory Shear).

        Parameters
        ----------
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency [rad/s]
        n_cycles : int
            Number of cycles to simulate
        points_per_cycle : int
            Points per cycle for discretization
        lam_init : float
            Initial structure parameter

        Returns
        -------
        dict
            't': time array
            'strain': strain γ(t)
            'strain_rate': strain rate γ̇(t)
            'stress': stress σ(t)
            'lam': structure λ(t)
        """
        period = 2 * np.pi / omega
        t_total = n_cycles * period
        n_points = n_cycles * points_per_cycle
        dt = t_total / n_points

        t = jnp.linspace(0, t_total, n_points)
        strain = gamma_0 * jnp.sin(omega * t)
        strain_rate = gamma_0 * omega * jnp.cos(omega * t)

        params = self.get_parameter_dict()

        if self.include_elasticity:
            # Maxwell LAOS
            def step(state, sr):
                sigma, lam = state

                # Structure evolution
                dlam = structure_evolution(lam, sr, params["t_eq"], params["a"], params["c"])
                lam_new = jnp.clip(lam + dt * dlam, 0.0, 1.0)

                # Elastic modulus
                G = elastic_modulus(lam_new, params["G0"], params["m_G"])

                # Viscosity
                if self.closure == "exponential":
                    eta = viscosity_exponential(lam_new, params["eta_0"], params["eta_inf"])
                else:
                    eta = viscosity_herschel_bulkley_regularized(
                        lam_new, sr, params["tau_y0"], params["K0"],
                        params["n_flow"], params["eta_inf"], params["m1"], params["m2"]
                    )

                # Relaxation time
                theta_1 = eta / jnp.maximum(G, 1e-10)

                # Stress evolution
                dsigma = maxwell_stress_evolution(sigma, sr, G, theta_1)
                sigma_new = sigma + dt * dsigma

                return (sigma_new, lam_new), (sigma_new, lam_new)

            init_state = (0.0, lam_init)
            _, (stress, lam) = jax.lax.scan(step, init_state, strain_rate)
        else:
            # Viscous LAOS
            def step(lam, sr):
                dlam = structure_evolution(lam, sr, params["t_eq"], params["a"], params["c"])
                lam_new = jnp.clip(lam + dt * dlam, 0.0, 1.0)

                if self.closure == "exponential":
                    eta = viscosity_exponential(lam_new, params["eta_0"], params["eta_inf"])
                else:
                    eta = viscosity_herschel_bulkley_regularized(
                        lam_new, sr, params["tau_y0"], params["K0"],
                        params["n_flow"], params["eta_inf"], params["m1"], params["m2"]
                    )

                stress = eta * sr
                return lam_new, (stress, lam_new)

            _, (stress, lam) = jax.lax.scan(step, lam_init, strain_rate)

        return {
            "t": np.array(t),
            "strain": np.array(strain),
            "strain_rate": np.array(strain_rate),
            "stress": np.array(stress),
            "lam": np.array(lam),
        }

    def extract_harmonics(
        self,
        laos_result: dict,
        n_harmonics: int = 5,
    ) -> dict[str, np.ndarray]:
        """Extract Fourier harmonics from LAOS stress response.

        Parameters
        ----------
        laos_result : dict
            Result from simulate_laos()
        n_harmonics : int
            Number of harmonics to extract

        Returns
        -------
        dict
            'sigma_prime': in-phase coefficients (odd harmonics)
            'sigma_double_prime': out-of-phase coefficients
            'I_n_1': normalized harmonic intensities
        """
        from scipy.integrate import trapezoid

        t = laos_result["t"]
        stress = laos_result["stress"]
        omega = laos_result["strain_rate"].max() / laos_result["strain"].max()

        # Use last cycle for steady-state analysis
        period = 2 * np.pi / omega
        dt = t[1] - t[0]
        points_per_cycle = int(period / dt)

        t_cycle = t[-points_per_cycle:]
        t_cycle = t_cycle - t_cycle[0]  # Reset to 0
        stress_cycle = stress[-points_per_cycle:]

        sigma_prime = []
        sigma_double_prime = []

        for n in range(1, 2 * n_harmonics, 2):  # Odd harmonics
            # In-phase (sin)
            sp = 2 * trapezoid(stress_cycle * np.sin(n * omega * t_cycle), t_cycle) / period
            # Out-of-phase (cos)
            spp = 2 * trapezoid(stress_cycle * np.cos(n * omega * t_cycle), t_cycle) / period

            sigma_prime.append(sp)
            sigma_double_prime.append(spp)

        sigma_prime = np.array(sigma_prime)
        sigma_double_prime = np.array(sigma_double_prime)

        # Normalized intensities I_n/I_1
        I_1 = np.sqrt(sigma_prime[0]**2 + sigma_double_prime[0]**2)
        I_n_1 = np.array([
            np.sqrt(sp**2 + spp**2) / I_1
            for sp, spp in zip(sigma_prime, sigma_double_prime)
        ])

        return {
            "sigma_prime": sigma_prime,
            "sigma_double_prime": sigma_double_prime,
            "I_n_1": I_n_1,
        }

    def _fit_laos(self, t: np.ndarray, stress: np.ndarray, **kwargs) -> "DMTLocal":
        """Fit to LAOS data."""
        raise NotImplementedError("LAOS fitting not yet implemented")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _compute_r2(self, y_true: jnp.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² coefficient of determination."""
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot)

    def _get_params_for_optimization(self) -> tuple[jnp.ndarray, tuple]:
        """Get parameter array and bounds for optimization."""
        param_names = list(self.parameters.keys())
        params = jnp.array([self.parameters.get_value(n) for n in param_names])
        bounds_lower = jnp.array([self.parameters.get_bounds(n)[0] for n in param_names])
        bounds_upper = jnp.array([self.parameters.get_bounds(n)[1] for n in param_names])
        return params, (bounds_lower, bounds_upper)

    def _params_array_to_dict(self, params_array: jnp.ndarray) -> dict:
        """Convert parameter array to dictionary."""
        param_names = list(self.parameters.keys())
        return {name: float(params_array[i]) for i, name in enumerate(param_names)}

    def _set_params_from_array(self, params_array: jnp.ndarray) -> None:
        """Set parameters from array."""
        param_names = list(self.parameters.keys())
        for i, name in enumerate(param_names):
            self.parameters.set_value(name, float(params_array[i]))

    def _simulate_with_params(
        self,
        t: jnp.ndarray,
        dt: float,
        gamma_dot: float,
        lam_init: float,
        params: dict,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate with given parameters (for fitting)."""
        # Temporarily set parameters
        original_params = self.get_parameter_dict()
        for name, value in params.items():
            if name in self.parameters.keys():
                self.parameters.set_value(name, value)

        # Simulate
        _, stress, lam = self.simulate_startup(gamma_dot, float(t[-1]), dt, lam_init)

        # Restore
        for name, value in original_params.items():
            self.parameters.set_value(name, value)

        return t, jnp.array(stress), jnp.array(lam)
