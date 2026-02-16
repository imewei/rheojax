"""Multi-Lambda Isotropic-Kinematic Hardening (ML-IKH) Model.

Extends MIKH to N modes for capturing distributed thixotropic timescales.
Supports two yield surface formulations:

1. **Per-Mode Yield** (default): Each mode has independent yield surface
   - Total stress = Σ σᵢ (parallel connection)
   - Parameters: 7 per mode + 1 global

2. **Weighted-Sum Yield**: Single global yield surface
   - σ_y = σ_y0 + k3·Σ(wᵢ·λᵢ)
   - All modes share elastic/plastic response
   - Parameters: 5 global + 3 per mode
"""

from typing import Literal

import numpy as np

from rheojax.core.base import ArrayLike
from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import lazy_import, safe_import_jax
diffrax = lazy_import("diffrax")
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.ikh._base import IKHBase
from rheojax.models.ikh._kernels import (
    ml_ikh_creep_ode_rhs_per_mode,
    ml_ikh_creep_ode_rhs_weighted_sum,
    ml_ikh_flow_curve_steady_state_per_mode,
    ml_ikh_flow_curve_steady_state_weighted_sum,
    ml_ikh_maxwell_ode_rhs_per_mode,
    ml_ikh_maxwell_ode_rhs_weighted_sum,
    ml_ikh_scan_kernel,
    ml_ikh_weighted_sum_kernel,
)

jax, jnp = safe_import_jax()


@ModelRegistry.register(
    "ml_ikh",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
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
class MLIKH(IKHBase):
    """Multi-Lambda Isotropic-Kinematic Hardening (ML-IKH) Model.

    Extends MIKH to N modes connected in parallel. Each mode evolves its own
    internal variables (stress, backstress, structural lambda) with distinct
    timescales (tau_thix_i) and properties.

    Two Yield Mode Options:
        - **per_mode** (default): Each mode has independent yield surface.
          Total stress is sum of mode stresses.
        - **weighted_sum**: Single global yield surface with structure
          contribution from all modes: σ_y = σ_y0 + k3·Σ(wᵢ·λᵢ)

    Per-Mode Parameters (for each mode i=1..N):
        G_i: Shear modulus
        C_i: Backstress modulus
        gamma_dyn_i: Dynamic recovery
        sigma_y0_i: Minimal yield stress
        delta_sigma_y_i: Structural yield stress
        tau_thix_i: Rebuilding timescale
        Gamma_i: Breakdown coefficient

    Weighted-Sum Parameters:
        G: Global shear modulus
        C: Global hardening modulus
        gamma_dyn: Global dynamic recovery
        sigma_y0: Base yield stress
        k3: Structure-yield coupling
        tau_thix_i: Per-mode rebuilding timescales
        Gamma_i: Per-mode breakdown coefficients
        w_i: Per-mode structure weights

    Global Parameters:
        eta_inf: High-shear viscosity (both modes)

    Supported Protocols:
        - FLOW_CURVE: Steady-state stress vs shear rate (analytical solution)
        - STARTUP: Transient stress growth at constant shear rate (return mapping)
        - RELAXATION: Stress decay at constant strain (ODE formulation via Diffrax)
        - CREEP: Strain evolution at constant stress (ODE formulation via Diffrax)
        - OSCILLATION: Small amplitude oscillatory shear response
        - LAOS: Large amplitude oscillatory shear (return mapping with sinusoidal strain)

    Note:
        Both yield modes (per_mode, weighted_sum) support all protocols.
        ODE protocols (creep, relaxation) use Diffrax for numerical integration.
        Return mapping protocols (startup, LAOS) use JAX scan for time stepping.

    Args:
        n_modes: Number of structural modes (default: 2)
        yield_mode: Yield formulation ('per_mode' or 'weighted_sum')
    """

    def __init__(
        self,
        n_modes: int = 2,
        yield_mode: Literal["per_mode", "weighted_sum"] = "per_mode",
    ):
        super().__init__()
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        if yield_mode not in ("per_mode", "weighted_sum"):
            raise ValueError(
                f"yield_mode must be 'per_mode' or 'weighted_sum', got {yield_mode}"
            )

        self._n_modes = n_modes
        self._yield_mode = yield_mode
        self._test_mode = None
        self._create_parameters()

    def _create_parameters(self):
        """Initialize parameters based on yield_mode."""
        self.parameters = ParameterSet()

        if self._yield_mode == "per_mode":
            self._create_per_mode_parameters()
        else:
            self._create_weighted_sum_parameters()

    def _create_per_mode_parameters(self):
        """Create parameters for per-mode yield formulation."""
        for i in range(1, self._n_modes + 1):
            # Elasticity & Hardening
            self.parameters.add(
                f"G_{i}",
                value=1e3 / self._n_modes,
                bounds=(0.0, 1e9),
                units="Pa",
                description=f"Mode {i} Shear modulus",
            )
            self.parameters.add(
                f"C_{i}",
                value=5e2 / self._n_modes,
                bounds=(0.0, 1e9),
                units="Pa",
                description=f"Mode {i} Kinematic hardening modulus",
            )
            self.parameters.add(
                f"gamma_dyn_{i}",
                value=1.0,
                bounds=(0.0, 1e4),
                units="-",
                description=f"Mode {i} Dynamic recovery",
            )

            # Yield Stress & Thixotropy
            self.parameters.add(
                f"sigma_y0_{i}",
                value=10.0 / self._n_modes,
                bounds=(0.0, 1e9),
                units="Pa",
                description=f"Mode {i} Minimal yield stress",
            )
            self.parameters.add(
                f"delta_sigma_y_{i}",
                value=50.0 / self._n_modes,
                bounds=(0.0, 1e9),
                units="Pa",
                description=f"Mode {i} Structural yield stress",
            )

            # Timescales distributed logarithmically
            tau_val = 10.0 ** (i - 1 - self._n_modes / 2)
            self.parameters.add(
                f"tau_thix_{i}",
                value=tau_val,
                bounds=(1e-6, 1e12),
                units="s",
                description=f"Mode {i} Rebuilding time scale",
            )

            self.parameters.add(
                f"Gamma_{i}",
                value=0.5,
                bounds=(0.0, 1e4),
                units="-",
                description=f"Mode {i} Breakdown coefficient",
            )

        # Global Viscosity
        self.parameters.add(
            "eta_inf",
            value=0.1,
            bounds=(0.0, 1e9),
            units="Pa s",
            description="High-shear viscosity",
        )

    def _create_weighted_sum_parameters(self):
        """Create parameters for weighted-sum yield formulation."""
        # Global mechanical parameters
        self.parameters.add(
            "G",
            value=1e3,
            bounds=(1e-1, 1e9),
            units="Pa",
            description="Global shear modulus",
        )
        self.parameters.add(
            "C",
            value=5e2,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Global kinematic hardening modulus",
        )
        self.parameters.add(
            "gamma_dyn",
            value=1.0,
            bounds=(0.0, 1e4),
            units="-",
            description="Global dynamic recovery",
        )
        self.parameters.add(
            "m",
            value=1.0,
            bounds=(0.5, 3.0),
            units="-",
            description="AF recovery exponent",
        )

        # Yield stress
        self.parameters.add(
            "sigma_y0",
            value=10.0,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Base yield stress",
        )
        self.parameters.add(
            "k3",
            value=50.0,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Structure-yield coupling",
        )

        # Per-mode structure parameters
        for i in range(1, self._n_modes + 1):
            tau_val = 10.0 ** (i - 1 - self._n_modes / 2)
            self.parameters.add(
                f"tau_thix_{i}",
                value=tau_val,
                bounds=(1e-6, 1e12),
                units="s",
                description=f"Mode {i} Rebuilding time scale",
            )
            self.parameters.add(
                f"Gamma_{i}",
                value=0.5,
                bounds=(0.0, 1e4),
                units="-",
                description=f"Mode {i} Breakdown coefficient",
            )
            self.parameters.add(
                f"w_{i}",
                value=1.0 / self._n_modes,
                bounds=(0.0, 1.0),
                units="-",
                description=f"Mode {i} structure weight",
            )

        # Global Viscosity
        self.parameters.add(
            "eta_inf",
            value=0.1,
            bounds=(0.0, 1e9),
            units="Pa s",
            description="High-shear viscosity",
        )

    def _stack_mode_params(self, params, names=None):
        """Stack per-mode parameters in a single pass.

        Reduces repeated dict lookups + jnp.stack calls from O(N*names)
        to O(1) per prediction, which matters during Bayesian inference
        (4000-8000 evaluations).
        """
        n = self._n_modes
        if names is None:
            names = ["G", "C", "gamma_dyn", "sigma_y0",
                     "delta_sigma_y", "tau_thix", "Gamma"]
        return {
            name: jnp.stack([params[f"{name}_{i}"] for i in range(1, n + 1)])
            for name in names
        }

    def _predict_from_params(self, times, strains, params):
        """Predict using parameter dictionary (for NLSQ/Bayesian)."""
        if self._yield_mode == "per_mode":
            return self._predict_per_mode(times, strains, params)
        else:
            return self._predict_weighted_sum(times, strains, params)

    def _predict_per_mode(self, times, strains, params):
        """Predict with per-mode yield surfaces."""
        # Stack all per-mode parameters in a single pass
        kernel_params = self._stack_mode_params(params)
        eta_inf = params["eta_inf"]

        return ml_ikh_scan_kernel(
            times,
            strains,
            num_modes=self._n_modes,
            use_viscosity=True,
            eta_inf=eta_inf,
            **kernel_params,
        )

    def _predict_weighted_sum(self, times, strains, params):
        """Predict with weighted-sum yield surface."""
        stacked = self._stack_mode_params(params, names=["tau_thix", "Gamma", "w"])

        kernel_params = {
            "G": params["G"],
            "C": params["C"],
            "gamma_dyn": params["gamma_dyn"],
            "m": params.get("m", 1.0),
            "sigma_y0": params["sigma_y0"],
            "k3": params["k3"],
            "eta_inf": params["eta_inf"],
            **stacked,
        }

        return ml_ikh_weighted_sum_kernel(
            times, strains, num_modes=self._n_modes, use_viscosity=True, **kernel_params
        )

    def _predict_flow_curve_from_params(self, gamma_dot, params):
        """Predict steady-state flow curve from parameter dictionary."""
        if self._yield_mode == "per_mode":
            return ml_ikh_flow_curve_steady_state_per_mode(
                gamma_dot, self._n_modes, **params
            )
        else:
            return ml_ikh_flow_curve_steady_state_weighted_sum(
                gamma_dot, self._n_modes, **params
            )

    def _build_ode_args(self, params, **kwargs):
        """Build args dictionary for ODE integration."""
        args = {"n_modes": self._n_modes}

        if self._yield_mode == "per_mode":
            # Stack all per-mode parameters in a single pass
            args.update(self._stack_mode_params(params))
            # Default arrays for optional parameters
            args["eta"] = jnp.full(self._n_modes, 1e12)
            args["mu_p"] = jnp.full(self._n_modes, 1e-6)
            args["m"] = jnp.ones(self._n_modes)
        else:
            # Global parameters
            args["G"] = params["G"]
            args["C"] = params["C"]
            args["gamma_dyn"] = params["gamma_dyn"]
            args["m"] = params.get("m", 1.0)
            args["sigma_y0"] = params["sigma_y0"]
            args["k3"] = params.get("k3", 0.0)
            # Per-mode structure parameters
            stacked = self._stack_mode_params(params, names=["tau_thix", "Gamma", "w"])
            args.update(stacked)
            args["eta"] = 1e12
            args["mu_p"] = 1e-6

        args["eta_inf"] = params.get("eta_inf", 0.0)

        # Add protocol-specific args
        for key in ["gamma_dot", "sigma_applied"]:
            if key in kwargs:
                args[key] = kwargs[key]

        return args

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
        n = self._n_modes
        args = self._build_ode_args(params)

        # Initial lambda (fully structured)
        lambda_init = 1.0

        if self._yield_mode == "per_mode":
            if mode == "creep":
                # State: [γ, α_1..α_N, λ_1..λ_N] (1+2N)
                ode_fn = ml_ikh_creep_ode_rhs_per_mode
                args["sigma_applied"] = (
                    sigma_applied if sigma_applied is not None else 100.0
                )
                y0 = jnp.concatenate(
                    [
                        jnp.array([0.0]),  # gamma
                        jnp.zeros(n),  # alphas
                        jnp.full(n, lambda_init),  # lambdas
                    ]
                )
            elif mode == "startup":
                # State: [σ_1..σ_N, α_1..α_N, λ_1..λ_N] (3N)
                ode_fn = ml_ikh_maxwell_ode_rhs_per_mode
                args["gamma_dot"] = gamma_dot if gamma_dot is not None else 1.0
                y0 = jnp.concatenate(
                    [
                        jnp.zeros(n),  # sigmas
                        jnp.zeros(n),  # alphas
                        jnp.full(n, lambda_init),  # lambdas
                    ]
                )
            else:  # relaxation
                ode_fn = ml_ikh_maxwell_ode_rhs_per_mode
                args["gamma_dot"] = 0.0
                # Initial stress distributed across modes
                sigma_init = (
                    sigma_0
                    if sigma_0 is not None
                    else (jnp.sum(args["sigma_y0"]) + jnp.sum(args["delta_sigma_y"]))
                )
                lambda_init_relax = 0.5
                y0 = jnp.concatenate(
                    [
                        jnp.full(n, sigma_init / n),  # sigmas (distributed)
                        jnp.zeros(n),  # alphas
                        jnp.full(n, lambda_init_relax),  # lambdas
                    ]
                )
        else:  # weighted_sum
            if mode == "creep":
                # State: [γ, α, λ_1..λ_N] (2+N)
                ode_fn = ml_ikh_creep_ode_rhs_weighted_sum
                args["sigma_applied"] = (
                    sigma_applied if sigma_applied is not None else 100.0
                )
                y0 = jnp.concatenate(
                    [
                        jnp.array([0.0, 0.0]),  # gamma, alpha
                        jnp.full(n, lambda_init),  # lambdas
                    ]
                )
            elif mode == "startup":
                # State: [σ, α, λ_1..λ_N] (2+N)
                ode_fn = ml_ikh_maxwell_ode_rhs_weighted_sum
                args["gamma_dot"] = gamma_dot if gamma_dot is not None else 1.0
                y0 = jnp.concatenate(
                    [
                        jnp.array([0.0, 0.0]),  # sigma, alpha
                        jnp.full(n, lambda_init),  # lambdas
                    ]
                )
            else:  # relaxation
                ode_fn = ml_ikh_maxwell_ode_rhs_weighted_sum
                args["gamma_dot"] = 0.0
                sigma_init = (
                    sigma_0 if sigma_0 is not None else (args["sigma_y0"] + args["k3"])
                )
                lambda_init_relax = 0.5
                y0 = jnp.concatenate(
                    [
                        jnp.array([sigma_init, 0.0]),
                        jnp.full(n, lambda_init_relax),
                    ]
                )

        # Diffrax setup
        term = diffrax.ODETerm(lambda ti, yi, args_i: ode_fn(ti, yi, args_i))
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
            throw=False,
        )

        # Extract primary variable
        if mode == "creep":
            # Return strain (first component)
            result = sol.ys[:, 0]
        else:
            # Return stress
            if self._yield_mode == "per_mode":
                # Sum mode stresses (first n components)
                result = jnp.sum(sol.ys[:, :n], axis=1)
            else:
                # Single global stress (first component)
                result = sol.ys[:, 0]

        # Handle solver failures
        result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            result,
            jnp.nan * jnp.ones_like(result)
        )

        # Add viscous contribution for startup
        if mode == "startup" and params.get("eta_inf", 0.0) > 0:
            result = result + params["eta_inf"] * args["gamma_dot"]

        return result

    def _predict(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Predict response with protocol-aware dispatch.

        Args:
            X: Input data (shear rates for flow_curve, time for others)
            **kwargs: Options including test_mode, gamma_dot, sigma_applied, etc.

        Returns:
            Predicted stress or strain depending on protocol
        """
        test_mode = kwargs.get("test_mode", self._test_mode or "startup")

        # Get parameters as dict
        params = self.parameters.get_values()
        param_dict = dict(zip(self.parameters.keys(), params, strict=False))

        if test_mode == "flow_curve":
            return self._predict_flow_curve_from_params(jnp.asarray(X), param_dict)
        elif test_mode in ["creep", "relaxation"]:
            return self._simulate_transient(
                jnp.asarray(X),
                param_dict,
                test_mode,
                gamma_dot=kwargs.get("gamma_dot"),
                sigma_applied=kwargs.get("sigma_applied"),
                sigma_0=kwargs.get("sigma_0"),
            )
        else:  # startup, laos, oscillation
            times, strains = self._extract_time_strain(X, **kwargs)
            return self._predict_from_params(times, strains, param_dict)

    def _fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MLIKH":
        """Fit model parameters using protocol-aware optimization.

        Args:
            X: Input data (shear rates, time array, or time/strain)
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

    def _fit_flow_curve(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MLIKH":
        """Fit to steady-state flow curve data."""
        from rheojax.utils.optimization import nlsq_optimize

        gamma_dot = jnp.asarray(X)
        sigma_target = jnp.asarray(y)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            sigma_pred = self._predict_flow_curve_from_params(gamma_dot, p_dict)
            return sigma_pred - sigma_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_ode_formulation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MLIKH":
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

        # Force method="scipy": diffrax ODE solvers use custom_vjp which is
        # incompatible with NLSQ's forward-mode autodiff (jvp).
        kwargs["method"] = "scipy"
        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_return_mapping(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MLIKH":
        """Fit using return-mapping algorithm (for startup/LAOS)."""
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

    def _fit_oscillation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MLIKH":
        """Fit to oscillation data (SAOS/MAOS).

        Supports two modes:
        1. Frequency-domain: X=omega, y=|G*| or complex G* (uses Maxwell analytical solution)
        2. Time-domain: X=time, y=stress (uses return mapping with sinusoidal strain)
        """
        X_arr = jnp.asarray(X)

        # Detect if this is frequency-domain or time-domain
        is_time_domain = len(X_arr) > 100

        if is_time_domain:
            return self._fit_return_mapping(X, y, **kwargs)
        else:
            return self._fit_saos_frequency_domain(X, y, **kwargs)

    def _fit_saos_frequency_domain(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "MLIKH":
        """Fit to frequency-domain SAOS data using Maxwell analytical expressions.

        Args:
            X: Angular frequency array (omega)
            y: Complex modulus G* = G' + i*G'' or magnitude |G*|
        """
        from rheojax.utils.optimization import nlsq_optimize

        omega = jnp.asarray(X)

        # Handle different y formats
        y_arr = jnp.asarray(y)
        if jnp.iscomplexobj(y_arr):
            target_magnitude = jnp.abs(y_arr)
        else:
            target_magnitude = y_arr

        def objective(param_values):
            """Compute residual using Maxwell analytical SAOS expressions."""
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))

            # Extract G and eta based on yield_mode
            if self._yield_mode == "per_mode":
                # Sum contributions from all modes (parallel Maxwell elements)
                G_prime_total = jnp.zeros_like(omega)
                G_double_prime_total = jnp.zeros_like(omega)

                for i in range(1, self._n_modes + 1):
                    G_i = p_dict[f"G_{i}"]
                    # Estimate eta from bounds or use large value for elastic behavior
                    eta_i = 1e12  # Effectively infinite for SAOS
                    tau_i = eta_i / G_i

                    wt_i = omega * tau_i
                    G_prime_i = G_i * wt_i**2 / (1 + wt_i**2)
                    G_double_prime_i = G_i * wt_i / (1 + wt_i**2)

                    G_prime_total += G_prime_i
                    G_double_prime_total += G_double_prime_i
            else:
                # Weighted-sum mode: use global G
                G = p_dict["G"]
                eta = 1e12  # Effectively infinite
                tau = eta / G

                wt = omega * tau
                G_prime_total = G * wt**2 / (1 + wt**2)
                G_double_prime_total = G * wt / (1 + wt**2)

            G_star_magnitude = jnp.sqrt(G_prime_total**2 + G_double_prime_total**2)

            return G_star_magnitude - target_magnitude

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro model function with protocol-aware dispatch.

        Accepts protocol-specific kwargs (gamma_dot, sigma_applied, sigma_0).

        Args:
            X: Input data
            params: Parameter array or dict from NumPyro
            test_mode: Optional protocol override
            **kwargs: Protocol-specific arguments

        Returns:
            Predicted response
        """
        # Convert params to dict if array
        if isinstance(params, (np.ndarray, jnp.ndarray)):
            param_names = list(self.parameters.keys())
            param_dict = dict(zip(param_names, params, strict=False))
        else:
            param_dict = params

        mode = test_mode or self._test_mode or "startup"

        # Extract protocol-specific args from kwargs or fall back to instance attrs
        gamma_dot = kwargs.get("gamma_dot", getattr(self, "_fit_gamma_dot", None))
        sigma_applied = kwargs.get("sigma_applied", getattr(self, "_fit_sigma_applied", None))
        sigma_0 = kwargs.get("sigma_0", getattr(self, "_fit_sigma_0", None))

        if mode == "flow_curve":
            return self._predict_flow_curve_from_params(jnp.asarray(X), param_dict)
        elif mode in ["creep", "relaxation"]:
            return self._simulate_transient(
                jnp.asarray(X),
                param_dict,
                mode,
                gamma_dot=gamma_dot,
                sigma_applied=sigma_applied,
                sigma_0=sigma_0,
            )
        elif mode == "oscillation":
            # Frequency-domain SAOS using multi-Maxwell analytical expressions
            omega = jnp.asarray(X)

            if self._yield_mode == "per_mode":
                # Sum contributions from all modes (parallel Maxwell elements)
                G_prime_total = jnp.zeros_like(omega)
                G_double_prime_total = jnp.zeros_like(omega)

                for i in range(1, self._n_modes + 1):
                    G_i = param_dict[f"G_{i}"]
                    eta_i = param_dict.get(f"eta_{i}", 1e12)  # High viscosity if not specified
                    tau_i = eta_i / G_i

                    wt_i = omega * tau_i
                    G_prime_total += G_i * wt_i**2 / (1 + wt_i**2)
                    G_double_prime_total += G_i * wt_i / (1 + wt_i**2)
            else:
                # Weighted-sum mode: use global G
                G = param_dict["G"]
                eta = param_dict.get("eta", 1e12)  # High viscosity if not specified
                tau = eta / G

                wt = omega * tau
                G_prime_total = G * wt**2 / (1 + wt**2)
                G_double_prime_total = G * wt / (1 + wt**2)

            return jnp.sqrt(G_prime_total**2 + G_double_prime_total**2)
        else:  # startup, laos
            # startup/laos modes need strain computed from kwargs
            times, strains = self._extract_time_strain(X, **kwargs)
            return self._predict_from_params(times, strains, param_dict)

    @property
    def n_modes(self) -> int:
        """Number of structural modes."""
        return self._n_modes

    @property
    def yield_mode(self) -> str:
        """Yield formulation mode ('per_mode' or 'weighted_sum')."""
        return self._yield_mode

    # -------------------------------------------------------------------------
    # Convenience Methods for Protocol-Specific Predictions
    # -------------------------------------------------------------------------

    def predict_flow_curve(self, gamma_dot: ArrayLike) -> ArrayLike:
        """Predict steady-state flow curve.

        Args:
            gamma_dot: Array of shear rates

        Returns:
            Array of steady-state stresses
        """
        return self._predict(gamma_dot, test_mode="flow_curve")

    def predict_startup(
        self, t: ArrayLike, gamma_dot: float = 1.0, strain: ArrayLike | None = None
    ) -> ArrayLike:
        """Predict startup shear response.

        Args:
            t: Time array
            gamma_dot: Applied shear rate (default: 1.0)
            strain: Optional strain array (if None, uses gamma_dot * t)

        Returns:
            Array of stresses
        """
        t_arr = jnp.asarray(t)
        if strain is None:
            strain = gamma_dot * t_arr
        return self._predict(jnp.stack([t_arr, strain]), test_mode="startup")

    def predict_relaxation(self, t: ArrayLike, sigma_0: float = 100.0) -> ArrayLike:
        """Predict stress relaxation after step strain.

        Args:
            t: Time array
            sigma_0: Initial stress (default: 100.0)

        Returns:
            Array of decaying stresses
        """
        return self._predict(t, test_mode="relaxation", sigma_0=sigma_0)

    def predict_creep(self, t: ArrayLike, sigma_applied: float = 50.0) -> ArrayLike:
        """Predict creep response under constant stress.

        Args:
            t: Time array
            sigma_applied: Applied constant stress (default: 50.0)

        Returns:
            Array of strains
        """
        return self._predict(t, test_mode="creep", sigma_applied=sigma_applied)

    def predict_laos(
        self, t: ArrayLike, gamma_0: float = 1.0, omega: float = 1.0
    ) -> ArrayLike:
        """Predict large amplitude oscillatory shear response.

        Args:
            t: Time array
            gamma_0: Strain amplitude (default: 1.0)
            omega: Angular frequency in rad/s (default: 1.0)

        Returns:
            Array of stresses
        """
        t_arr = jnp.asarray(t)
        strain = gamma_0 * jnp.sin(omega * t_arr)
        return self._predict(jnp.stack([t_arr, strain]), test_mode="laos")
