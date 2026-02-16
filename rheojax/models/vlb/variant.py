"""VLB Variant model with Bell breakage, FENE-P stress, and temperature dependence.

This module implements `VLBVariant`, a composable constitutive model that extends
the basic VLB framework (constant k_d, linear stress) with:

1. **Bell breakage**: Force-dependent dissociation rate
   k_d(mu) = k_d_0 * exp(nu * (stretch - 1))
   → Shear thinning, stress overshoot, nonlinear LAOS

2. **FENE-P stress**: Finite extensibility
   sigma = G0 * f(tr(mu)) * (mu - I), f = L²/(L² - tr(mu) + 3)
   → Bounded extensional stress, strain hardening

3. **Temperature**: Arrhenius kinetics
   k_d(T) = k_d_0 * exp(-E_a/R * (1/T - 1/T_ref))
   G0(T) = G0_ref * T/T_ref
   → Time-temperature superposition

All three extensions can be combined independently via constructor flags,
following the TNT composable pattern (TNTSingleMode).

Parameters
----------
breakage : str, default "constant"
    "constant" (Newtonian) or "bell" (force-dependent, shear thinning)
stress_type : str, default "linear"
    "linear" (Gaussian) or "fene" (finite extensibility)
temperature : bool, default False
    If True, adds Arrhenius temperature dependence

Example
-------
>>> from rheojax.models.vlb import VLBVariant
>>> import numpy as np
>>>
>>> # Bell model: shear-thinning VLB
>>> model = VLBVariant(breakage="bell")
>>> model.parameters.set_value("G0", 1000.0)
>>> model.parameters.set_value("k_d_0", 1.0)
>>> model.parameters.set_value("nu", 3.0)
>>>
>>> gamma_dot = np.logspace(-2, 2, 50)
>>> sigma = model.predict(gamma_dot, test_mode='flow_curve')

References
----------
- Vernerey, F.J., Long, R. & Brighenti, R. (2017). JMPS 107, 1-20.
- Bell, G.I. (1978). Science 200(4342), 618-627.
- Bird, R.B. et al. (1987). Dynamics of Polymeric Liquids, Vol. 2.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import lazy_import, safe_import_jax
diffrax = lazy_import("diffrax")
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.vlb._base import VLBBase
from rheojax.models.vlb._kernels import (
    build_vlb_creep_ode_rhs,
    build_vlb_laos_ode_rhs,
    build_vlb_ode_rhs,
    build_vlb_relaxation_ode_rhs,
    vlb_arrhenius_shift,
    vlb_breakage_bell,
    vlb_fene_factor,
    vlb_saos_moduli_vec,
    vlb_stress_fene_n1,
    vlb_thermal_modulus,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)

BreakageType = Literal["constant", "bell"]
StressType = Literal["linear", "fene"]


@ModelRegistry.register(
    "vlb_variant",
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
class VLBVariant(VLBBase):
    """VLB with Bell breakage, FENE-P stress, and/or temperature dependence.

    This is the composable variant class for VLB models. It supports all
    6 protocols via ODE integration (required when k_d depends on state).

    When breakage="constant" and stress_type="linear", the model matches
    VLBLocal exactly (regression verified).

    Parameters
    ----------
    breakage : str, default "constant"
        Breakage rate function: "constant" or "bell"
    stress_type : str, default "linear"
        Stress formula: "linear" or "fene"
    temperature : bool, default False
        If True, enable Arrhenius temperature dependence
    """

    def __init__(
        self,
        breakage: BreakageType = "constant",
        stress_type: StressType = "linear",
        temperature: bool = False,
    ):
        """Initialize VLBVariant model."""
        # Store flags before calling super().__init__
        self._breakage = breakage
        self._stress_type = stress_type
        self._temperature = temperature

        super().__init__()
        self._setup_parameters()
        self._build_variant_ode_functions()
        self._test_mode = None

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with variant-dependent parameters.

        Core parameters (always present):
        - G0: Network modulus (Pa)
        - k_d_0: Unstressed dissociation rate (1/s)
        - eta_s: Solvent viscosity (Pa·s)

        Conditional parameters:
        - nu: Force sensitivity (Bell breakage)
        - L_max: Maximum extensibility (FENE stress)
        - E_a: Activation energy (temperature)
        - T_ref: Reference temperature (temperature)
        """
        self.parameters = ParameterSet()

        # Core parameters
        self.parameters.add(
            name="G0",
            value=1e3,
            bounds=(1e0, 1e8),
            units="Pa",
            description="Network modulus (elastic contribution from active chains)",
        )
        self.parameters.add(
            name="k_d_0",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="1/s",
            description="Unstressed dissociation rate",
        )
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e4),
            units="Pa·s",
            description="Solvent viscosity (Newtonian background)",
        )

        # Bell breakage
        if self._breakage == "bell":
            self.parameters.add(
                name="nu",
                value=1.0,
                bounds=(0.0, 20.0),
                units="dimensionless",
                description="Force sensitivity (Bell model, higher = more shear-thinning)",
            )

        # FENE-P stress
        if self._stress_type == "fene":
            self.parameters.add(
                name="L_max",
                value=10.0,
                bounds=(1.5, 1000.0),
                units="dimensionless",
                description="Maximum chain extensibility (FENE-P spring)",
            )

        # Temperature dependence
        if self._temperature:
            self.parameters.add(
                name="E_a",
                value=50e3,
                bounds=(1e3, 500e3),
                units="J/mol",
                description="Activation energy for bond dissociation",
            )
            self.parameters.add(
                name="T_ref",
                value=298.15,
                bounds=(200.0, 500.0),
                units="K",
                description="Reference temperature",
            )

    # =========================================================================
    # ODE Function Builders
    # =========================================================================

    def _build_variant_ode_functions(self):
        """Build and cache variant-specific ODE RHS functions.

        Called once in __init__. Each variant combination traces to a
        separate JAX-compiled function.
        """
        self._variant_ode = build_vlb_ode_rhs(self._breakage, self._stress_type)
        self._variant_creep_ode = build_vlb_creep_ode_rhs(
            self._breakage, self._stress_type
        )
        self._variant_laos_ode = build_vlb_laos_ode_rhs(
            self._breakage, self._stress_type
        )
        self._variant_relax_ode = build_vlb_relaxation_ode_rhs(
            self._breakage, self._stress_type
        )

    # =========================================================================
    # Parameter Unpacking
    # =========================================================================

    def _unpack_variant_params(self, params) -> dict:
        """Unpack variant parameters from a JAX params array.

        Returns dict with all variant param values (dummy values for
        inactive variants).
        """
        result = {
            "nu": 0.0,
            "L_max": 10.0,
        }
        idx = 3  # After G0, k_d_0, eta_s
        if self._breakage == "bell":
            result["nu"] = params[idx]
            idx += 1
        if self._stress_type == "fene":
            result["L_max"] = params[idx]
            idx += 1
        if self._temperature:
            result["E_a"] = params[idx]
            result["T_ref"] = params[idx + 1]
            idx += 2
        return result

    # =========================================================================
    # Parameter Array Builder
    # =========================================================================

    def _build_params_array(self) -> jnp.ndarray:
        """Build JAX parameter array from ParameterSet.

        Returns params in ParameterSet order: [G0, k_d_0, eta_s, (nu), (L_max), (E_a, T_ref)].
        """
        param_values = [
            float(self.parameters.get_value(name)) # type: ignore[arg-type]
            for name in self.parameters.keys()
        ]
        return jnp.array(param_values, dtype=jnp.float64)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def G0(self) -> float:
        """Network modulus (Pa)."""
        val = self.parameters.get_value("G0")
        return float(val) if val is not None else 1e3

    @property
    def k_d_0(self) -> float:
        """Unstressed dissociation rate (1/s)."""
        val = self.parameters.get_value("k_d_0")
        return float(val) if val is not None else 1.0

    @property
    def nu(self) -> float | None:
        """Force sensitivity parameter (Bell only)."""
        if self._breakage != "bell":
            return None
        val = self.parameters.get_value("nu")
        return float(val) if val is not None else 1.0

    @property
    def L_max(self) -> float | None:
        """Maximum extensibility (FENE only)."""
        if self._stress_type != "fene":
            return None
        val = self.parameters.get_value("L_max")
        return float(val) if val is not None else 10.0

    @property
    def relaxation_time(self) -> float:
        """Equilibrium relaxation time t_R = 1/k_d_0 (s)."""
        return 1.0 / self.k_d_0

    @property
    def viscosity(self) -> float:
        """Zero-shear viscosity eta_0 = G0/k_d_0 (Pa·s)."""
        return self.G0 / self.k_d_0

    # =========================================================================
    # Stress Computation Helper
    # =========================================================================

    def _compute_stress_from_mu(
        self, mu_xx, mu_yy, mu_zz, mu_xy, G0, eta_s, gamma_dot, vp
    ):
        """Compute total shear stress from distribution tensor components."""
        if self._stress_type == "fene":
            f = vlb_fene_factor(mu_xx, mu_yy, mu_zz, vp["L_max"])
            sigma_el = G0 * f * mu_xy
        else:
            sigma_el = G0 * mu_xy
        return sigma_el + eta_s * gamma_dot

    # =========================================================================
    # ODE Solver Helper
    # =========================================================================

    @staticmethod
    def _get_ode_solver():
        """Return standard diffrax solver and controller."""
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
        return solver, controller

    # =========================================================================
    # Core Fit/Predict
    # =========================================================================

    def _fit(self, x, y, **kwargs):
        """Fit model to data using protocol-aware optimization."""
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
            self.initialize_from_flow_curve(np.asarray(x), np.asarray(y))
        elif test_mode == "oscillation":
            self.initialize_from_saos(
                np.asarray(x), np.real(np.asarray(y)), np.imag(np.asarray(y))
            )

        # Filter kwargs for model_function
        fwd_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "test_mode",
                "use_log_residuals",
                "use_jax",
                "method",
                "max_iter",
                "use_multi_start",
                "n_starts",
                "perturb_factor",
            )
        }

        def model_fn(x_fit, params):
            return self.model_function(x_fit, params, test_mode=test_mode, **fwd_kwargs)

        objective = create_least_squares_objective(
            model_fn,
            x_jax,
            y_jax,
            use_log_residuals=kwargs.get(
                "use_log_residuals", test_mode == "flow_curve"
            ),
        )

        # Force method="scipy" for VLBVariant: diffrax ODE solvers use custom_vjp
        # which is incompatible with NLSQ's forward-mode autodiff (jvp).
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            method="scipy",
            max_iter=kwargs.get("max_iter", 2000),
        )

        self.fitted_ = True
        self._nlsq_result = result

        logger.info(f"Fitted VLBVariant: G0={self.G0:.2e}, k_d_0={self.k_d_0:.2e}")
        return self

    def _predict(self, X, **kwargs):
        """Predict response from fitted model."""
        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        x_jax = jnp.asarray(X, dtype=jnp.float64)

        # Store protocol-specific parameters from kwargs
        if "gamma_dot" in kwargs:
            self._gamma_dot_applied = kwargs["gamma_dot"]
        if "sigma_applied" in kwargs:
            self._sigma_applied = kwargs["sigma_applied"]
        if "gamma_0" in kwargs:
            self._gamma_0 = kwargs["gamma_0"]
        if "omega" in kwargs:
            self._omega_laos = kwargs["omega"]

        # Build parameter array from ParameterSet
        param_values = [
            float(self.parameters.get_value(name))
            for name in self.parameters.keys()
        ]
        params = jnp.array(param_values)

        fwd_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("test_mode", "deformation_mode", "poisson_ratio")
        }
        return self.model_function(x_jax, params, test_mode=test_mode, **fwd_kwargs)

    # =========================================================================
    # Model Function (Stateless, for NLSQ/NumPyro)
    # =========================================================================

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function.

        Routes to appropriate prediction based on test_mode.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values in ParameterSet order:
            [G0, k_d_0, eta_s, (nu), (L_max), (E_a, T_ref)]
        test_mode : str, optional
            Override stored test mode
        **kwargs
            Protocol-specific: gamma_dot, sigma_applied, gamma_0, omega, T

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        # Core parameters
        G0 = params[0]
        k_d_0 = params[1]
        eta_s = params[2]

        # Variant parameters
        vp = self._unpack_variant_params(params)

        # Temperature scaling
        if self._temperature:
            T = kwargs.get("T", vp.get("T_ref", 298.15))
            T_ref = vp["T_ref"]
            E_a = vp["E_a"]
            k_d_0 = vlb_arrhenius_shift(k_d_0, E_a, T, T_ref)
            G0 = vlb_thermal_modulus(G0, T, T_ref)

        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # Protocol parameters
        gamma_dot = kwargs.get("gamma_dot", self._gamma_dot_applied)
        sigma_applied = kwargs.get("sigma_applied", self._sigma_applied)
        gamma_0 = kwargs.get("gamma_0", self._gamma_0)
        omega = kwargs.get("omega", self._omega_laos)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return self._variant_flow_curve_internal(
                X_jax, G0, k_d_0, eta_s, vp
            )

        elif mode == "oscillation":
            # All VLB variants linearize to Maxwell in SAOS
            # (at equilibrium stretch=1, Bell gives k_d = k_d_0)
            G_prime, G_double_prime = vlb_saos_moduli_vec(X_jax, G0, k_d_0)
            # Add solvent contribution
            G_double_prime = G_double_prime + eta_s * X_jax
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return self._simulate_startup_internal(
                X_jax, G0, k_d_0, eta_s, gamma_dot, vp
            )

        elif mode == "relaxation":
            # Return relaxation modulus G(t) for consistency with VLBLocal
            # For constant k_d: G(t) = G0*exp(-k_d*t) (single exponential)
            # For Bell: integrate from small step strain (linear regime)
            if self._breakage == "constant":
                # Analytical single-exponential decay
                return G0 * jnp.exp(-k_d_0 * X_jax)
            else:
                # ODE from small-strain pre-shear (linear regime)
                gamma_dot_ps = gamma_dot if gamma_dot is not None else 0.01 * k_d_0
                stress = self._simulate_relaxation_internal(
                    X_jax, G0, k_d_0, eta_s, gamma_dot_ps, vp
                )
                # Normalize to modulus: G(t) = sigma(t) / gamma_step
                # For small pre-shear, sigma_0 ≈ G0 * Wi, so G(t) ≈ sigma(t) * k_d / gamma_dot
                return stress * k_d_0 / gamma_dot_ps

        elif mode == "creep":
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            return self._simulate_creep_internal(
                X_jax, G0, k_d_0, eta_s, sigma_applied, vp
            )

        elif mode == "laos":
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, G0, k_d_0, eta_s, gamma_0, omega, vp
            )
            return stress

        else:
            logger.warning(f"Unknown test_mode '{mode}', defaulting to flow_curve")
            return self._variant_flow_curve_internal(
                X_jax, G0, k_d_0, eta_s, vp
            )

    # =========================================================================
    # Flow Curve (ODE to Steady State)
    # =========================================================================

    def _variant_flow_curve_internal(
        self,
        gamma_dot_arr: jnp.ndarray,
        G0: float,
        k_d_0: float,
        eta_s: float,
        vp: dict,
    ) -> jnp.ndarray:
        """Compute variant flow curve by running ODE to steady state.

        For Bell breakage, the steady-state conformation cannot be solved
        analytically. We integrate for ~50/k_d_0 at each shear rate.
        """
        variant_ode = self._variant_ode
        is_fene = self._stress_type == "fene"

        def solve_single(gdot):
            def ode_fn(ti, yi, args):
                return variant_ode(
                    ti, yi,
                    args["gdot"], args["G0"], args["k_d_0"],
                    args["nu"], args["L_max"],
                )

            args = {"gdot": gdot, "G0": G0, "k_d_0": k_d_0, **vp}
            y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
            tau_b = 1.0 / k_d_0
            t_end = 50.0 * tau_b
            dt0 = tau_b / 20.0

            term = diffrax.ODETerm(ode_fn)
            solver = diffrax.Tsit5()
            controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
            saveat = diffrax.SaveAt(ts=jnp.array([t_end]))

            sol = diffrax.diffeqsolve(
                term, solver, 0.0, t_end, dt0, y0,
                args=args, saveat=saveat,
                stepsize_controller=controller,
                max_steps=500_000, throw=False,
            )

            mu_final = sol.ys[0]
            if is_fene:
                f = vlb_fene_factor(mu_final[0], mu_final[1], mu_final[2], vp["L_max"])
                sigma_el = G0 * f * mu_final[3]
            else:
                sigma_el = G0 * mu_final[3]

            result = sigma_el + eta_s * gdot
            result = jnp.where(
                sol.result == diffrax.RESULTS.successful, result, jnp.nan
            )
            return result

        return jax.vmap(solve_single)(gamma_dot_arr)

    # =========================================================================
    # Startup Shear
    # =========================================================================

    def _simulate_startup_internal(
        self,
        t: jnp.ndarray,
        G0: float,
        k_d_0: float,
        eta_s: float,
        gamma_dot: float,
        vp: dict,
    ) -> jnp.ndarray:
        """Internal startup simulation. Returns total shear stress sigma_xy(t)."""
        variant_ode = self._variant_ode

        def ode_fn(ti, yi, args):
            return variant_ode(
                ti, yi,
                args["gamma_dot"], args["G0"], args["k_d_0"],
                args["nu"], args["L_max"],
            )

        args = {"gamma_dot": gamma_dot, "G0": G0, "k_d_0": k_d_0, **vp}
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver, controller = self._get_ode_solver()

        # Always start from t=0 so saveat points get properly integrated
        t1 = t[-1]
        dt0 = t1 / max(len(t), 1000)
        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term, solver, 0.0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=controller,
            max_steps=500_000, throw=False,
        )

        result = self._compute_stress_from_mu(
            sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3],
            G0, eta_s, gamma_dot, vp,
        )
        result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            result, jnp.nan * jnp.ones_like(result),
        )
        return result

    # =========================================================================
    # Stress Relaxation
    # =========================================================================

    def _simulate_relaxation_internal(
        self,
        t: jnp.ndarray,
        G0: float,
        k_d_0: float,
        eta_s: float,
        gamma_dot_preshear: float,
        vp: dict,
    ) -> jnp.ndarray:
        """Internal relaxation simulation.

        Computes steady-state pre-shear conformation, then relaxes with
        gamma_dot = 0.
        """
        # Steady-state pre-shear conformation (constant k_d approximation for IC)
        tau_b = 1.0 / k_d_0
        Wi = gamma_dot_preshear * tau_b
        mu_xx_0 = 1.0 + 2.0 * Wi * Wi
        mu_yy_0 = 1.0
        mu_zz_0 = 1.0
        mu_xy_0 = Wi
        y0 = jnp.array([mu_xx_0, mu_yy_0, mu_zz_0, mu_xy_0], dtype=jnp.float64)

        variant_relax_ode = self._variant_relax_ode

        def ode_fn(ti, yi, args):
            return variant_relax_ode(
                ti, yi,
                args["G0"], args["k_d_0"],
                args["nu"], args["L_max"],
            )

        args = {"G0": G0, "k_d_0": k_d_0, **vp}

        term = diffrax.ODETerm(ode_fn)
        solver, controller = self._get_ode_solver()

        # Start from t=0 so saveat points get properly integrated
        t1 = t[-1]
        dt0 = t1 / max(len(t), 1000)
        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term, solver, 0.0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=controller,
            max_steps=500_000, throw=False,
        )

        result = self._compute_stress_from_mu(
            sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3],
            G0, eta_s, 0.0, vp,
        )
        result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            result, jnp.nan * jnp.ones_like(result),
        )
        return result

    # =========================================================================
    # Creep
    # =========================================================================

    def _simulate_creep_internal(
        self,
        t: jnp.ndarray,
        G0: float,
        k_d_0: float,
        eta_s: float,
        sigma_applied: float,
        vp: dict,
    ) -> jnp.ndarray:
        """Internal creep simulation. Returns accumulated strain gamma(t)."""
        variant_creep_ode = self._variant_creep_ode

        def ode_fn(ti, yi, args):
            return variant_creep_ode(
                ti, yi,
                args["sigma_applied"], args["G0"], args["k_d_0"],
                args["eta_s"], args["nu"], args["L_max"],
            )

        args = {
            "sigma_applied": sigma_applied,
            "G0": G0, "k_d_0": k_d_0, "eta_s": eta_s,
            **vp,
        }
        # Initial condition: elastic jump at t=0+
        # For Maxwell model: mu_xy(0+) = sigma/G0, gamma(0+) = sigma/G0
        mu_xy_0 = sigma_applied / G0
        gamma_0 = sigma_applied / G0
        y0 = jnp.array([1.0, 1.0, 1.0, mu_xy_0, gamma_0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver, controller = self._get_ode_solver()

        # Start from t=0 so saveat points get properly integrated
        t1 = t[-1]
        dt0 = t1 / max(len(t), 1000)
        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term, solver, 0.0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=controller,
            max_steps=500_000, throw=False,
        )

        result = sol.ys[:, 4]  # gamma (strain)
        result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            result, jnp.nan * jnp.ones_like(result),
        )
        return result

    # =========================================================================
    # LAOS
    # =========================================================================

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        G0: float,
        k_d_0: float,
        eta_s: float,
        gamma_0: float,
        omega: float,
        vp: dict,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal LAOS simulation. Returns (strain, stress) arrays."""
        variant_laos_ode = self._variant_laos_ode

        def ode_fn(ti, yi, args):
            return variant_laos_ode(
                ti, yi,
                args["gamma_0"], args["omega"],
                args["G0"], args["k_d_0"],
                args["nu"], args["L_max"],
            )

        args = {
            "gamma_0": gamma_0, "omega": omega,
            "G0": G0, "k_d_0": k_d_0,
            **vp,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver, controller = self._get_ode_solver()

        # Start from t[0] to match VLBLocal behavior (IC at equilibrium)
        t0, t1 = t[0], t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)
        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=controller,
            max_steps=500_000, throw=False,
        )

        strain = gamma_0 * jnp.sin(omega * t)
        gamma_dot_t = gamma_0 * omega * jnp.cos(omega * t)
        stress = self._compute_stress_from_mu(
            sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3],
            G0, eta_s, gamma_dot_t, vp,
        )
        stress = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            stress, jnp.nan * jnp.ones_like(stress),
        )

        return strain, stress

    # =========================================================================
    # Public Convenience Methods
    # =========================================================================

    def predict_flow_curve(
        self, gamma_dot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict steady-state flow curve.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        sigma : np.ndarray
            Steady-state shear stress (Pa)
        eta : np.ndarray
            Apparent viscosity (Pa·s)
        """
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        params = self._build_params_array()
        G0, k_d_0, eta_s = params[0], params[1], params[2]
        vp = self._unpack_variant_params(params)

        if self._temperature:
            T_ref = vp["T_ref"]
            E_a = vp["E_a"]
            k_d_0 = vlb_arrhenius_shift(k_d_0, E_a, T_ref, T_ref)

        sigma = self._variant_flow_curve_internal(gamma_dot_jax, G0, k_d_0, eta_s, vp)
        sigma = np.asarray(sigma)
        eta = sigma / np.maximum(np.asarray(gamma_dot), 1e-20)
        return sigma, eta

    def predict_saos(
        self, omega: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict SAOS moduli (Maxwell, analytical).

        In the linear regime, Bell reduces to constant k_d = k_d_0.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency array (rad/s)

        Returns
        -------
        G_prime : np.ndarray
            Storage modulus G' (Pa)
        G_double_prime : np.ndarray
            Loss modulus G'' (Pa)
        """
        omega_jax = jnp.asarray(omega, dtype=jnp.float64)
        G_prime, G_double_prime = vlb_saos_moduli_vec(omega_jax, self.G0, self.k_d_0)
        G_double_prime = G_double_prime + float(self.parameters.get_value("eta_s") or 0.0) * omega_jax
        return np.asarray(G_prime), np.asarray(G_double_prime)

    def predict_normal_stresses(
        self, gamma_dot: np.ndarray
    ) -> np.ndarray:
        """Predict steady-state first normal stress difference N1.

        For Bell breakage, this requires ODE integration.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        np.ndarray
            N1 values (Pa)
        """
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        params = self._build_params_array()
        G0, k_d_0 = params[0], params[1]
        vp = self._unpack_variant_params(params)
        is_fene = self._stress_type == "fene"
        variant_ode = self._variant_ode

        def solve_n1(gdot):
            def ode_fn(ti, yi, args):
                return variant_ode(
                    ti, yi,
                    args["gdot"], args["G0"], args["k_d_0"],
                    args["nu"], args["L_max"],
                )

            args = {"gdot": gdot, "G0": G0, "k_d_0": k_d_0, **vp}
            y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
            tau_b = 1.0 / k_d_0
            t_end = 50.0 * tau_b
            dt0 = tau_b / 20.0

            term = diffrax.ODETerm(ode_fn)
            solver = diffrax.Tsit5()
            controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
            saveat = diffrax.SaveAt(ts=jnp.array([t_end]))

            sol = diffrax.diffeqsolve(
                term, solver, 0.0, t_end, dt0, y0,
                args=args, saveat=saveat,
                stepsize_controller=controller,
                max_steps=500_000, throw=False,
            )

            mu_f = sol.ys[0]
            if is_fene:
                n1 = vlb_stress_fene_n1(mu_f[0], mu_f[1], mu_f[2], G0, vp["L_max"])
            else:
                n1 = G0 * (mu_f[0] - mu_f[1])
            return jnp.where(sol.result == diffrax.RESULTS.successful, n1, jnp.nan)

        return np.asarray(jax.vmap(solve_n1)(gamma_dot_jax))

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_full: bool = False,
    ) -> np.ndarray | dict:
        """Simulate startup shear.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_full : bool
            If True, return dict with stress, N1, strain

        Returns
        -------
        np.ndarray or dict
            Shear stress sigma(t), or dict with full trajectory
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        params = self._build_params_array()
        G0, k_d_0, eta_s = params[0], params[1], params[2]
        vp = self._unpack_variant_params(params)

        stress = self._simulate_startup_internal(
            t_jax, G0, k_d_0, eta_s, gamma_dot, vp
        )

        if return_full:
            return {
                "t": np.asarray(t),
                "stress": np.asarray(stress),
                "strain": np.asarray(t) * gamma_dot,
            }
        return np.asarray(stress)

    def simulate_relaxation(
        self,
        t: np.ndarray,
        gamma_dot_preshear: float = 10.0,
    ) -> np.ndarray:
        """Simulate stress relaxation after cessation of flow.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot_preshear : float
            Pre-shear rate (1/s)

        Returns
        -------
        np.ndarray
            Relaxing stress sigma(t)
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        params = self._build_params_array()
        G0, k_d_0, eta_s = params[0], params[1], params[2]
        vp = self._unpack_variant_params(params)

        stress = self._simulate_relaxation_internal(
            t_jax, G0, k_d_0, eta_s, gamma_dot_preshear, vp
        )
        return np.asarray(stress)

    def simulate_creep(
        self,
        t: np.ndarray,
        sigma_applied: float,
    ) -> np.ndarray:
        """Simulate creep (stress-controlled).

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma_applied : float
            Applied stress (Pa)

        Returns
        -------
        np.ndarray
            Strain gamma(t)
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        params = self._build_params_array()
        G0, k_d_0, eta_s = params[0], params[1], params[2]
        vp = self._unpack_variant_params(params)

        strain = self._simulate_creep_internal(
            t_jax, G0, k_d_0, eta_s, sigma_applied, vp
        )
        return np.asarray(strain)

    def simulate_laos(
        self,
        t: np.ndarray,
        gamma_0: float,
        omega: float,
        n_cycles: int = 10,
    ) -> dict:
        """Simulate Large Amplitude Oscillatory Shear (LAOS).

        Parameters
        ----------
        t : np.ndarray or None
            Time array (if None, auto-generated from n_cycles)
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)
        n_cycles : int
            Number of cycles (if t is None)

        Returns
        -------
        dict
            't', 'strain', 'stress', 'gamma_dot'
        """
        if t is None:
            period = 2 * np.pi / omega # type: ignore[unreachable]
            t = np.linspace(0, n_cycles * period, n_cycles * 200)

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        params = self._build_params_array()
        G0, k_d_0, eta_s = params[0], params[1], params[2]
        vp = self._unpack_variant_params(params)

        strain, stress = self._simulate_laos_internal(
            t_jax, G0, k_d_0, eta_s, gamma_0, omega, vp
        )

        self._trajectory = {
            "t": np.asarray(t),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "gamma_dot": np.asarray(gamma_0 * omega * jnp.cos(omega * t_jax)),
        }
        return self._trajectory

    def predict_uniaxial_extension(
        self, eps_dot: np.ndarray
    ) -> np.ndarray:
        """Predict steady-state extensional stress.

        For FENE-P, extensional stress is bounded (no singularity).

        Parameters
        ----------
        eps_dot : np.ndarray
            Extension rate array (1/s)

        Returns
        -------
        np.ndarray
            Extensional stress sigma_E (Pa)
        """
        eps_dot_jax = jnp.asarray(eps_dot, dtype=jnp.float64)
        params = self._build_params_array()
        G0, k_d_0 = params[0], params[1]
        vp = self._unpack_variant_params(params)
        is_fene = self._stress_type == "fene"

        def solve_ext(ed):
            # Uniaxial extension: L = diag(ed, -ed/2, -ed/2)
            # mu evolution: 2-component (mu_11, mu_22), no off-diagonal
            if self._breakage == "bell":
                # Need ODE for Bell
                def ode_fn(ti, yi, args):
                    mu_11, mu_22 = yi[0], yi[1]
                    mu_zz = yi[1]  # mu_22 = mu_33 by symmetry
                    k_d = vlb_breakage_bell(mu_11, mu_22, mu_zz, k_d_0, vp["nu"])
                    dmu_11 = k_d * (1.0 - mu_11) + 2.0 * ed * mu_11
                    dmu_22 = k_d * (1.0 - mu_22) - ed * mu_22
                    return jnp.array([dmu_11, dmu_22])

                y0 = jnp.array([1.0, 1.0], dtype=jnp.float64)
                tau_b = 1.0 / k_d_0
                t_end = 50.0 * tau_b
                dt0 = tau_b / 20.0

                term = diffrax.ODETerm(ode_fn)
                solver = diffrax.Tsit5()
                controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
                saveat = diffrax.SaveAt(ts=jnp.array([t_end]))

                sol = diffrax.diffeqsolve(
                    term, solver, 0.0, t_end, dt0, y0,
                    args=None, saveat=saveat,
                    stepsize_controller=controller,
                    max_steps=500_000, throw=False,
                )

                mu_f = sol.ys[0]
                mu_11, mu_22 = mu_f[0], mu_f[1]
            else:
                # Analytical steady state for constant k_d
                denom_11 = jnp.maximum(k_d_0 - 2.0 * ed, 1e-10)
                denom_22 = k_d_0 + ed
                mu_11 = k_d_0 / denom_11
                mu_22 = k_d_0 / denom_22

            if is_fene:
                f = vlb_fene_factor(mu_11, mu_22, mu_22, vp["L_max"])
                return G0 * f * (mu_11 - mu_22)
            else:
                return G0 * (mu_11 - mu_22)

        return np.asarray(jax.vmap(solve_ext)(eps_dot_jax))

    def extract_laos_harmonics(
        self,
        result: dict | None = None,
        n_harmonics: int = 5,
    ) -> dict:
        """Extract Fourier harmonics from LAOS data.

        Parameters
        ----------
        result : dict, optional
            LAOS result dict (uses stored _trajectory if None)
        n_harmonics : int
            Number of harmonics to extract

        Returns
        -------
        dict
            'harmonics': array of harmonic intensities
            'I3_I1': third-to-first harmonic ratio
        """
        if result is None:
            result = self._trajectory
        if result is None:
            raise ValueError("No LAOS data. Run simulate_laos() first.")

        stress = result["stress"]
        t = result["t"]

        # Use last cycle for steady-state harmonics
        omega = self._omega_laos or 1.0
        period = 2 * np.pi / omega
        t_last_cycle = t[-1] - period
        mask = t >= t_last_cycle
        stress_cycle = stress[mask]

        # FFT
        n = len(stress_cycle)
        fft_vals = np.fft.rfft(stress_cycle)
        magnitudes = 2.0 * np.abs(fft_vals) / n

        # Harmonics: I_1, I_3, I_5, ...
        harmonics = magnitudes[1 : 2 * n_harmonics : 2] if len(magnitudes) > 2 * n_harmonics else magnitudes[1:]

        I1 = harmonics[0] if len(harmonics) > 0 else 1e-30
        I3 = harmonics[1] if len(harmonics) > 1 else 0.0

        return {
            "harmonics": harmonics,
            "I3_I1": float(I3 / max(I1, 1e-30)),
        }

    # =========================================================================
    # Dimensionless Numbers Override
    # =========================================================================

    def weissenberg_number(self, gamma_dot: float) -> float:
        """Compute Weissenberg number Wi = t_R * gamma_dot."""
        return self.relaxation_time * abs(gamma_dot)

    def deborah_number(self, omega: float) -> float:
        """Compute Deborah number De = t_R * omega."""
        return self.relaxation_time * omega

    # =========================================================================
    # Repr
    # =========================================================================

    def __repr__(self) -> str:
        flags = []
        if self._breakage != "constant":
            flags.append(f"breakage={self._breakage!r}")
        if self._stress_type != "linear":
            flags.append(f"stress={self._stress_type!r}")
        if self._temperature:
            flags.append("temperature=True")
        flag_str = ", ".join(flags) if flags else "constant/linear"
        return f"VLBVariant({flag_str}, G0={self.G0:.2e}, k_d_0={self.k_d_0:.2e})"
