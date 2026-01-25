"""Maxwell-Isotropic-Kinematic Hardening (MIKH) Model.

A thixotropic elasto-viscoplastic model combining:
1. Maxwell viscoelastic element
2. Armstrong-Frederick kinematic hardening (backstress evolution)
3. Isotropic hardening/softening via structural parameter lambda (thixotropy)
4. Viscous background solvent

Supports all 6 experimental protocols:
- Flow curve (steady state)
- Startup shear
- Stress relaxation
- Creep
- SAOS (small amplitude oscillatory shear)
- LAOS (large amplitude oscillatory shear)
"""

from typing import cast

import diffrax
import numpy as np

from rheojax.core.base import ArrayLike
from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.models.ikh._base import IKHBase
from rheojax.models.ikh._kernels import (
    ikh_creep_ode_rhs,
    ikh_flow_curve_steady_state,
    ikh_maxwell_ode_rhs,
    ikh_scan_kernel,
)

jax, jnp = safe_import_jax()


@ModelRegistry.register(
    "mikh",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.OSCILLATION,
        Protocol.LAOS,
    ],
)
class MIKH(IKHBase):
    r"""Maxwell-Isotropic-Kinematic Hardening (MIKH) Model.

    A thixotropic elasto-viscoplastic model combining:
    1. Armstrong-Frederick kinematic hardening (backstress evolution).
    2. Isotropic hardening/softening via structural parameter lambda (thixotropy).
    3. Maxwell viscoelastic element for proper relaxation behavior.
    4. Viscous background solvent.

    Two Formulations:
        - **Maxwell ODE** (via Diffrax): For creep/relaxation protocols
        - **Return Mapping**: For startup/LAOS protocols (incremental)

    Governing Equations:
        σ_total = σ + η_inf * γ̇

        Stress Evolution (ODE formulation):
        dσ/dt = G(γ̇ - γ̇ᵖ) - (G/η)σ

        Yield Surface: \|σ - α\| ≤ σ_y(λ)
        σ_y(λ) = σ_y0 + Δσ_y * λ

        Structure Evolution:
        dλ/dt = (1-λ)/τ_thix - Γ*λ*\|γ̇ᵖ\|

        Backstress Evolution (Armstrong-Frederick):
        dα = C*dγ_p - γ_dyn*\|α\|^(m-1)*α*\|dγ_p\|

    Parameters:
        G: Shear modulus [Pa]
        eta: Maxwell viscosity [Pa·s] (controls relaxation time τ = η/G)
        C: Kinematic hardening modulus [Pa]
        gamma_dyn: Dynamic recovery parameter for backstress [-]
        m: AF recovery exponent [-] (typically 1.0)
        sigma_y0: Minimal (destructured) yield stress [Pa]
        delta_sigma_y: Yield stress increment (structured - destructured) [Pa]
        tau_thix: Thixotropic rebuilding time scale [s]
        Gamma: Structural breakdown coefficient [-]
        eta_inf: High-shear viscosity [Pa·s]
        mu_p: Plastic viscosity for Perzyna regularization [Pa·s]
    """

    def __init__(self):
        super().__init__()
        self._test_mode = None  # Store test mode for Bayesian

        # Elasticity
        self.parameters.add(
            "G", value=1e3, bounds=(1e-1, 1e9), units="Pa", description="Shear modulus"
        )
        self.parameters.add(
            "eta",
            value=1e6,
            bounds=(1e-3, 1e12),
            units="Pa s",
            description="Maxwell viscosity (relaxation time = eta/G)",
        )

        # Kinematic Hardening (Armstrong-Frederick)
        self.parameters.add(
            "C",
            value=5e2,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Kinematic hardening modulus",
        )
        self.parameters.add(
            "gamma_dyn",
            value=1.0,
            bounds=(0.0, 1e4),
            units="-",
            description="Dynamic recovery parameter",
        )
        self.parameters.add(
            "m",
            value=1.0,
            bounds=(0.5, 3.0),
            units="-",
            description="AF recovery exponent",
        )

        # Yield Stress & Thixotropy
        self.parameters.add(
            "sigma_y0",
            value=10.0,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Minimal yield stress (destructured)",
        )
        self.parameters.add(
            "delta_sigma_y",
            value=50.0,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Structural yield stress contribution",
        )
        self.parameters.add(
            "tau_thix",
            value=1.0,
            bounds=(1e-6, 1e12),
            units="s",
            description="Rebuilding time scale",
        )
        self.parameters.add(
            "Gamma",
            value=0.5,
            bounds=(0.0, 1e4),
            units="-",
            description="Breakdown coefficient",
        )

        # Viscosity
        self.parameters.add(
            "eta_inf",
            value=0.1,
            bounds=(0.0, 1e9),
            units="Pa s",
            description="High-shear viscosity (solvent)",
        )
        self.parameters.add(
            "mu_p",
            value=1e-3,
            bounds=(1e-9, 1e3),
            units="Pa s",
            description="Plastic viscosity (Perzyna regularization)",
        )

    def _fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MIKH":
        """Fit model parameters to data using protocol-aware optimization.

        Args:
            X: Input data (time/strain array or RheoData)
            y: Target data (stress or strain depending on protocol)
            **kwargs: Options including:
                - test_mode: Protocol ('flow_curve', 'startup', 'relaxation',
                             'creep', 'oscillation', 'laos')
                - gamma_dot: Shear rate (for startup)
                - sigma_applied: Applied stress (for creep)
                - sigma_0: Initial stress (for relaxation)
        """
        test_mode = kwargs.get("test_mode", "startup")
        self._test_mode = test_mode

        if test_mode == "flow_curve":
            return self._fit_flow_curve(X, y, **kwargs)
        elif test_mode in ["creep", "relaxation"]:
            return self._fit_ode_formulation(X, y, **kwargs)
        elif test_mode in ["startup", "laos"]:
            return self._fit_return_mapping(X, y, **kwargs)
        elif test_mode in ["oscillation", "saos"]:
            return self._fit_oscillation(X, y, **kwargs)
        else:
            # Default to return mapping for strain-driven protocols
            return self._fit_return_mapping(X, y, **kwargs)

    def _fit_flow_curve(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MIKH":
        """Fit to steady-state flow curve data."""
        from rheojax.utils.optimization import nlsq_optimize

        gamma_dot = jnp.asarray(X)
        sigma_target = jnp.asarray(y)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            sigma_pred = ikh_flow_curve_steady_state(gamma_dot, **p_dict)
            return sigma_pred - sigma_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_ode_formulation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MIKH":
        """Fit using ODE formulation (for creep/relaxation)."""
        from rheojax.utils.optimization import nlsq_optimize

        t = jnp.asarray(X)
        y_target = jnp.asarray(y)
        test_mode = kwargs.get("test_mode", "relaxation")
        gamma_dot = kwargs.get("gamma_dot", 0.0)
        sigma_applied = kwargs.get("sigma_applied", 100.0)
        sigma_0 = kwargs.get("sigma_0", 100.0)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            y_pred = self._simulate_transient(
                t, p_dict, test_mode, gamma_dot, sigma_applied, sigma_0
            )
            return y_pred - y_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_return_mapping(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MIKH":
        """Fit using return mapping formulation (for startup/LAOS)."""
        from rheojax.utils.optimization import nlsq_optimize

        times, strains = self._extract_time_strain(X, **kwargs)
        sigma_target = jnp.asarray(y)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            sigma_pred = self._predict_from_params(times, strains, p_dict)
            return sigma_pred - sigma_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_oscillation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MIKH":
        """Fit to oscillatory data (SAOS)."""
        # For SAOS, use linearized response or small-amplitude LAOS
        return self._fit_return_mapping(X, y, **kwargs)

    def _simulate_transient(
        self,
        t: jnp.ndarray,
        params: dict,
        mode: str,
        gamma_dot: float | None = None,
        sigma_applied: float | None = None,
        sigma_0: float | None = None,
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
        args = {k: params[k] for k in params}

        # Initial state based on mode
        lambda_init = 1.0  # Fully structured initially

        if mode == "creep":
            # Creep: constant stress, track strain
            ode_fn = ikh_creep_ode_rhs
            args["sigma_applied"] = (
                sigma_applied if sigma_applied is not None else 100.0
            )
            # State: [strain, alpha, lambda]
            y0 = jnp.array([0.0, 0.0, lambda_init])
        elif mode == "startup":
            # Startup: constant rate, track stress
            ode_fn = ikh_maxwell_ode_rhs
            args["gamma_dot"] = gamma_dot if gamma_dot is not None else 1.0
            # State: [sigma, alpha, lambda]
            y0 = jnp.array([0.0, 0.0, lambda_init])
        else:  # relaxation
            # Relaxation: rate = 0, stress decays
            ode_fn = ikh_maxwell_ode_rhs
            args["gamma_dot"] = 0.0
            sigma_init = (
                sigma_0
                if sigma_0 is not None
                else params.get("sigma_y0", 10.0) + params.get("delta_sigma_y", 50.0)
            )
            # Start partially destructured
            lambda_init_relax = 0.5
            y0 = jnp.array([sigma_init, 0.0, lambda_init_relax])

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

        # Add viscous contribution for startup
        if mode == "startup" and params.get("eta_inf", 0.0) > 0:
            result = result + params["eta_inf"] * args["gamma_dot"]

        return result

    def _predict_from_params(self, times, strains, params):
        """Predict using parameter dictionary (for NLSQ/Bayesian)."""
        return ikh_scan_kernel(times, strains, use_viscosity=True, **params)

    def _predict(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Predict stress from time/strain history or based on test_mode.

        Args:
            X: Input data. Shape depends on test_mode:
                - flow_curve: shear rates
                - startup/laos: (2, N) array of [time, strain] or RheoData
                - relaxation: time array (requires sigma_0)
                - creep: time array (requires sigma_applied)
            **kwargs: Additional parameters (test_mode, gamma_dot, etc.)
        """
        test_mode = kwargs.get("test_mode", self._test_mode or "startup")
        params = self.parameters.get_values()
        param_dict = dict(zip(self.parameters.keys(), params, strict=False))

        if test_mode == "flow_curve":
            gamma_dot = jnp.asarray(X)
            return ikh_flow_curve_steady_state(gamma_dot, **param_dict)

        elif test_mode in ["creep", "relaxation"]:
            t = jnp.asarray(X)
            gamma_dot = kwargs.get("gamma_dot", 0.0)
            sigma_applied = kwargs.get("sigma_applied", 100.0)
            sigma_0 = kwargs.get("sigma_0", 60.0)
            return self._simulate_transient(
                t, param_dict, test_mode, gamma_dot, sigma_applied, sigma_0
            )

        else:
            # Strain-driven protocols (startup, laos, oscillation)
            times, strains = self._extract_time_strain(X, **kwargs)
            return self._predict_from_params(times, strains, param_dict)

    def predict_flow_curve(self, gamma_dot: ArrayLike) -> ArrayLike:
        """Predict steady-state flow curve."""
        return self._predict(gamma_dot, test_mode="flow_curve")

    def predict_startup(self, t: ArrayLike, gamma_dot: float = 1.0) -> ArrayLike:
        """Predict startup shear response.

        Args:
            t: Time array
            gamma_dot: Constant shear rate

        Returns:
            Stress vs time
        """
        params = dict(
            zip(self.parameters.keys(), self.parameters.get_values(), strict=False)
        )
        return self._simulate_transient(
            jnp.asarray(t), params, "startup", gamma_dot=gamma_dot
        )

    def predict_relaxation(self, t: ArrayLike, sigma_0: float = 100.0) -> ArrayLike:
        """Predict stress relaxation.

        Args:
            t: Time array
            sigma_0: Initial stress

        Returns:
            Stress vs time
        """
        params = dict(
            zip(self.parameters.keys(), self.parameters.get_values(), strict=False)
        )
        return self._simulate_transient(
            jnp.asarray(t), params, "relaxation", sigma_0=sigma_0
        )

    def predict_creep(self, t: ArrayLike, sigma_applied: float = 50.0) -> ArrayLike:
        """Predict creep response.

        Args:
            t: Time array
            sigma_applied: Applied constant stress

        Returns:
            Strain vs time
        """
        params = dict(
            zip(self.parameters.keys(), self.parameters.get_values(), strict=False)
        )
        return self._simulate_transient(
            jnp.asarray(t), params, "creep", sigma_applied=sigma_applied
        )

    def predict_laos(
        self, t: ArrayLike, gamma_0: float = 1.0, omega: float = 1.0
    ) -> ArrayLike:
        """Predict LAOS response.

        Args:
            t: Time array
            gamma_0: Strain amplitude
            omega: Angular frequency

        Returns:
            Stress vs time
        """
        t_arr = jnp.asarray(t)
        strain = gamma_0 * jnp.sin(omega * t_arr)
        return self._predict_from_params(
            t_arr,
            strain,
            dict(
                zip(self.parameters.keys(), self.parameters.get_values(), strict=False)
            ),
        )

    def model_function(self, X, params, test_mode=None):
        """NumPyro model function for Bayesian inference."""
        # Use stored test_mode if not provided
        mode = test_mode or self._test_mode or "startup"

        # Convert array to dict for kernel
        if isinstance(params, (np.ndarray, jnp.ndarray)):
            param_names = list(self.parameters.keys())
            param_dict = dict(zip(param_names, params, strict=False))
        else:
            param_dict = params

        if mode == "flow_curve":
            gamma_dot = jnp.asarray(X)
            return ikh_flow_curve_steady_state(gamma_dot, **param_dict)

        elif mode in ["creep", "relaxation"]:
            # For Bayesian with ODE, we need to handle this carefully
            # X should contain time array
            t = jnp.asarray(X)
            # Default values - these should be passed via kwargs in fit_bayesian
            gamma_dot = param_dict.get("_gamma_dot", 0.0)
            sigma_applied = param_dict.get("_sigma_applied", 100.0)
            sigma_0 = param_dict.get("_sigma_0", 60.0)
            return self._simulate_transient(
                t, param_dict, mode, gamma_dot, sigma_applied, sigma_0
            )

        else:
            times, strains = self._extract_time_strain(X)
            return self._predict_from_params(times, strains, param_dict)
