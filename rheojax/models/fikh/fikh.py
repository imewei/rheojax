"""FIKH (Fractional Isotropic-Kinematic Hardening) Model.

This module implements the FIKH model, a thixotropic elasto-viscoplastic
model with Caputo fractional derivative for structure evolution and
optional thermokinematic coupling.

Key Features:
    - Power-law memory in structure evolution (Caputo derivative)
    - Temperature-dependent viscosity and yield stress (Arrhenius)
    - Viscous heating with convective cooling
    - Armstrong-Frederick kinematic hardening

Mathematical Framework:
    Stress: σ_total = σ + η_inf·γ̇
    Maxwell relaxation: dσ/dt = G(γ̇ - γ̇ᵖ) - σ/τ
    Yield: |σ - α| ≤ σ_y(λ, T)
    Backstress: dα = C·dγᵖ - γ_dyn·|α|^(m-1)·α·|dγᵖ|
    Structure: D^α_C λ = (1-λ)/τ_thix - Γ·λ·|γ̇ᵖ|
    Temperature: ρc_p·dT/dt = χ·σ·γ̇ᵖ - h·(T-T_env)

Example:
    >>> from rheojax.models.fikh import FIKH
    >>> model = FIKH(include_thermal=True, alpha_structure=0.5)
    >>> model.fit(t, stress, test_mode='startup', strain=strain)
    >>> sigma_pred = model.predict(t_new, strain=strain_new)
"""

from typing import Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode, Protocol, TestMode
from rheojax.logging import get_logger
from rheojax.models.fikh._base import FIKHBase
from rheojax.utils.optimization import nlsq_optimize

jax, jnp = safe_import_jax()

logger = get_logger(__name__)

# Type alias
ArrayLike = np.ndarray | jnp.ndarray | list | tuple


@ModelRegistry.register(
    "fikh",
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
class FIKH(FIKHBase):
    r"""Fractional Isotropic-Kinematic Hardening (FIKH) Model.

    A thixotropic elasto-viscoplastic model extending MIKH with:
    1. Caputo fractional derivative for structure evolution (power-law memory).
    2. Full thermokinematic coupling (Arrhenius + viscous heating).

    The fractional derivative captures memory effects in thixotropic recovery,
    where the structure remembers its history via a power-law kernel rather
    than simple exponential decay.

    Governing Equations:
        σ_total = σ + η_inf·γ̇

        Stress Evolution (ODE):
            dσ/dt = G(γ̇ - γ̇ᵖ) - (G/η)σ

        Yield Surface:
            |σ - α| ≤ σ_y(λ, T)
            σ_y = σ_y0 + Δσ_y·λ^m_y · exp(E_y/R·(1/T - 1/T_ref))

        Fractional Structure Evolution (Caputo):
            D^α_C λ = (1-λ)/τ_thix - Γ·λ·|γ̇ᵖ|

        Backstress (Armstrong-Frederick):
            dα = C·dγᵖ - γ_dyn·|α|^(m-1)·α·|dγᵖ|

        Temperature:
            ρc_p·dT/dt = χ·σ·γ̇ᵖ - h·(T - T_env)

    Parameters (22 with thermal):
        G: Shear modulus [Pa]
        eta: Maxwell viscosity [Pa·s]
        C: Kinematic hardening modulus [Pa]
        gamma_dyn: Dynamic recovery parameter [-]
        m: AF recovery exponent [-]
        sigma_y0: Minimal yield stress [Pa]
        delta_sigma_y: Structural yield contribution [Pa]
        tau_thix: Thixotropic time scale [s]
        Gamma: Breakdown coefficient [-]
        alpha_structure: Fractional order (0 < α < 1) [-]
        eta_inf: High-shear viscosity [Pa·s]
        mu_p: Plastic viscosity [Pa·s]
        T_ref: Reference temperature [K]
        E_a: Viscosity activation energy [J/mol]
        E_y: Yield stress activation energy [J/mol]
        m_y: Structure exponent for yield [-]
        rho_cp: Volumetric heat capacity [J/(m³·K)]
        chi: Taylor-Quinney coefficient [-]
        h: Heat transfer coefficient [W/(m²·K)]
        T_env: Environmental temperature [K]

    Limiting Behavior:
        α → 1: Recovers classical IKH/MIKH (exponential structure relaxation)
        E_a = E_y = 0: Isothermal behavior (temperature-independent)

    Example:
        >>> # Isothermal FIKH
        >>> model = FIKH(include_thermal=False, alpha_structure=0.7)
        >>> model.fit(omega, G_star, test_mode='oscillation')

        >>> # Thermal FIKH with startup
        >>> model = FIKH(include_thermal=True)
        >>> result = model.fit(t, stress, test_mode='startup', strain=strain)
        >>> sigma_pred = model.predict_startup(t_new, gamma_dot=1.0)
    """

    def __init__(
        self,
        include_thermal: bool = True,
        include_isotropic_hardening: bool = False,
        alpha_structure: float = 0.5,
        n_history: int = 100,
    ):
        """Initialize FIKH model.

        Args:
            include_thermal: Enable thermokinematic coupling (Arrhenius + heating).
            include_isotropic_hardening: Enable isotropic hardening R.
            alpha_structure: Fractional order for structure (0 < α < 1).
                - α → 0: Strong memory (slow recovery)
                - α → 1: Weak memory (fast, exponential recovery)
            n_history: History buffer size for Caputo derivative.
        """
        super().__init__(
            include_thermal=include_thermal,
            include_isotropic_hardening=include_isotropic_hardening,
            alpha_structure=alpha_structure,
            n_history=n_history,
        )
        logger.debug(
            "Initialized FIKH model",
            include_thermal=include_thermal,
            alpha_structure=alpha_structure,
        )

    # =========================================================================
    # Fitting Methods
    # =========================================================================

    def _fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "FIKH":
        """Fit model parameters using protocol-aware optimization.

        Args:
            X: Input data (depends on test_mode).
            y: Target data (stress or strain).
            **kwargs: Options including:
                - test_mode: Protocol type
                - gamma_dot: Shear rate (startup)
                - sigma_applied: Applied stress (creep)
                - sigma_0: Initial stress (relaxation)
                - strain: Strain array (if X is time only)

        Returns:
            Self with fitted parameters.
        """
        test_mode = kwargs.get("test_mode", "startup")
        self._test_mode = test_mode

        mode = self._validate_test_mode(test_mode)

        if mode == TestMode.FLOW_CURVE:
            return self._fit_flow_curve(X, y, **kwargs)
        elif mode in (TestMode.CREEP, TestMode.RELAXATION):
            return self._fit_ode_formulation(X, y, **kwargs)
        elif mode == TestMode.STARTUP:
            # STARTUP and LAOS both use return mapping
            return self._fit_return_mapping(X, y, **kwargs)
        elif mode == TestMode.OSCILLATION:
            return self._fit_oscillation(X, y, **kwargs)
        else:
            return self._fit_return_mapping(X, y, **kwargs)

    def _fit_flow_curve(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "FIKH":
        """Fit to steady-state flow curve data."""
        from rheojax.models.fikh._kernels import fikh_flow_curve_steady_state

        gamma_dot = jnp.asarray(X)
        sigma_target = jnp.asarray(y)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            sigma_pred = fikh_flow_curve_steady_state(
                gamma_dot, include_thermal=self.include_thermal, **p_dict
            )
            return sigma_pred - sigma_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_ode_formulation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "FIKH":
        """Fit using ODE formulation (creep/relaxation)."""
        t = jnp.asarray(X)
        y_target = jnp.asarray(y)
        test_mode = kwargs.get("test_mode", "relaxation")
        gamma_dot = kwargs.get("gamma_dot", 0.0)
        sigma_applied = kwargs.get("sigma_applied", 100.0)
        sigma_0 = kwargs.get("sigma_0", 100.0)
        T_init = kwargs.get("T_init", None)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            y_pred = self._simulate_transient(
                t, p_dict, test_mode, gamma_dot, sigma_applied, sigma_0, T_init
            )
            return y_pred - y_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_return_mapping(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "FIKH":
        """Fit using return mapping (startup/LAOS)."""
        times, strains = self._extract_time_strain(X, **kwargs)
        sigma_target = jnp.asarray(y)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            sigma_pred = self._predict_from_params(times, strains, p_dict)
            return sigma_pred - sigma_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_oscillation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "FIKH":
        """Fit to oscillatory data (SAOS).

        This method fits to frequency-domain SAOS data by internally simulating
        time-domain oscillations at each frequency and extracting G* via Fourier.

        Args:
            X: Angular frequency array (omega) [rad/s].
            y: Target modulus data - can be:
                - Complex G* = G' + i·G'' (uses both components)
                - Real |G*| magnitude (fits to magnitude)
            **kwargs: Options including:
                - gamma_0: Strain amplitude (default 0.01)
                - n_cycles: Number of cycles per frequency (default 5)

        Returns:
            Self with fitted parameters.
        """
        omega = jnp.asarray(X)
        y_arr = jnp.asarray(y)
        gamma_0 = kwargs.get("gamma_0", 0.01)
        n_cycles = kwargs.get("n_cycles", 5)

        # Determine if fitting to complex or magnitude
        is_complex = jnp.iscomplexobj(y_arr)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))

            # Predict G* at each frequency using time-domain simulation
            G_star_pred = self._predict_oscillation_from_params(
                omega, p_dict, gamma_0, n_cycles
            )

            if is_complex:
                # Fit both G' and G'' by stacking residuals
                residuals = jnp.concatenate([
                    jnp.real(G_star_pred) - jnp.real(y_arr),
                    jnp.imag(G_star_pred) - jnp.imag(y_arr),
                ])
            else:
                # Fit to magnitude |G*|
                residuals = jnp.abs(G_star_pred) - jnp.abs(y_arr)

            return residuals

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _predict_oscillation_from_params(
        self,
        omega: jnp.ndarray,
        params: dict[str, Any],
        gamma_0: float = 0.01,
        n_cycles: int = 5,
    ) -> jnp.ndarray:
        """Predict complex modulus G* from parameter dictionary.

        Internal method used by both NLSQ fitting and Bayesian inference.

        Args:
            omega: Angular frequency array.
            params: Parameter dictionary.
            gamma_0: Strain amplitude.
            n_cycles: Number of cycles to simulate.

        Returns:
            Complex modulus G* = G' + i·G'' for each frequency.
        """
        from rheojax.models.fikh._kernels import (
            fikh_scan_kernel_isothermal,
            fikh_scan_kernel_thermal,
        )

        alpha = params.get("alpha_structure", self.alpha_structure)

        G_star = []
        for w in omega:
            # Time array for n_cycles
            period = 2 * jnp.pi / w
            t = jnp.linspace(0, n_cycles * period, int(100 * n_cycles))

            # Strain signal
            strain = gamma_0 * jnp.sin(w * t)

            # Predict stress using appropriate kernel
            if self.include_thermal:
                T_init = params.get("T_env", params.get("T_ref", 298.15))
                stress, _ = fikh_scan_kernel_thermal(
                    t,
                    strain,
                    n_history=self.n_history,
                    alpha=alpha,
                    use_viscosity=True,
                    T_init=T_init,
                    **params,
                )
            else:
                stress = fikh_scan_kernel_isothermal(
                    t,
                    strain,
                    n_history=self.n_history,
                    alpha=alpha,
                    use_viscosity=True,
                    **params,
                )

            # Extract last cycle for Fourier analysis
            last_cycle_start = int(len(t) * (n_cycles - 1) / n_cycles)
            t_last = t[last_cycle_start:]
            stress_last = stress[last_cycle_start:]

            # Fourier decomposition (first harmonic)
            T_cycle = 2 * jnp.pi / w
            dt = t_last[1] - t_last[0]

            G_prime = (2 / (gamma_0 * T_cycle)) * jnp.trapezoid(
                stress_last * jnp.sin(w * t_last), dx=dt
            )
            G_double_prime = (2 / (gamma_0 * T_cycle)) * jnp.trapezoid(
                stress_last * jnp.cos(w * t_last), dx=dt
            )

            G_star.append(G_prime + 1j * G_double_prime)

        return jnp.array(G_star)

    # =========================================================================
    # Prediction Methods
    # =========================================================================

    def _predict_from_params(
        self,
        times: jnp.ndarray,
        strains: jnp.ndarray,
        params: dict[str, Any],
    ) -> jnp.ndarray:
        """Predict stress using parameter dictionary.

        This is the core prediction method used by both NLSQ fitting and
        Bayesian inference.

        Args:
            times: Time array.
            strains: Strain array.
            params: Parameter dictionary.

        Returns:
            Predicted stress array.
        """
        from rheojax.models.fikh._kernels import (
            fikh_scan_kernel_isothermal,
            fikh_scan_kernel_thermal,
        )

        # Extract alpha (can now be a traced value since it's not in static_argnums)
        alpha = params.get("alpha_structure", self.alpha_structure)

        if self.include_thermal:
            T_init = params.get("T_env", params.get("T_ref", 298.15))
            sigma_series, _ = fikh_scan_kernel_thermal(
                times,
                strains,
                n_history=self.n_history,
                alpha=alpha,
                use_viscosity=True,
                T_init=T_init,
                **params,
            )
        else:
            sigma_series = fikh_scan_kernel_isothermal(
                times,
                strains,
                n_history=self.n_history,
                alpha=alpha,
                use_viscosity=True,
                **params,
            )

        return sigma_series

    def _predict(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Predict based on test_mode.

        Args:
            X: Input data (shape depends on test_mode).
            **kwargs: Additional parameters.

        Returns:
            Predicted values.
        """
        test_mode = kwargs.get("test_mode", self._test_mode or "startup")
        mode = self._validate_test_mode(test_mode)
        params = self._get_params_dict()

        if mode == TestMode.FLOW_CURVE:
            from rheojax.models.fikh._kernels import fikh_flow_curve_steady_state

            gamma_dot = jnp.asarray(X)
            return fikh_flow_curve_steady_state(
                gamma_dot, include_thermal=self.include_thermal, **params
            )

        elif mode in (TestMode.CREEP, TestMode.RELAXATION):
            t = jnp.asarray(X)
            gamma_dot = kwargs.get("gamma_dot", 0.0)
            sigma_applied = kwargs.get("sigma_applied", 100.0)
            sigma_0 = kwargs.get("sigma_0", 60.0)
            T_init = kwargs.get("T_init", None)
            return self._simulate_transient(
                t, params, mode.value, gamma_dot, sigma_applied, sigma_0, T_init
            )

        elif mode == TestMode.OSCILLATION:
            # Frequency-domain SAOS: X is omega, return G*
            omega = jnp.asarray(X)
            gamma_0 = kwargs.get("gamma_0", 0.01)
            n_cycles = kwargs.get("n_cycles", 5)
            return self._predict_oscillation_from_params(omega, params, gamma_0, n_cycles)

        else:
            # Strain-driven protocols (startup, laos)
            times, strains = self._extract_time_strain(X, **kwargs)
            return self._predict_from_params(times, strains, params)

    # =========================================================================
    # Protocol-Specific Prediction Methods
    # =========================================================================

    def predict_flow_curve(
        self, gamma_dot: ArrayLike, T: float | None = None
    ) -> ArrayLike:
        """Predict steady-state flow curve.

        Args:
            gamma_dot: Shear rate array.
            T: Temperature (if thermal enabled).

        Returns:
            Stress array.
        """
        return self._predict(gamma_dot, test_mode="flow_curve")

    def predict_startup(
        self,
        t: ArrayLike,
        gamma_dot: float = 1.0,
        T_init: float | None = None,
    ) -> ArrayLike:
        """Predict startup shear response.

        Args:
            t: Time array.
            gamma_dot: Constant shear rate.
            T_init: Initial temperature.

        Returns:
            Stress vs time.
        """
        params = self._get_params_dict()
        return self._simulate_transient(
            jnp.asarray(t), params, "startup", gamma_dot=gamma_dot, T_init=T_init
        )

    def predict_relaxation(
        self,
        t: ArrayLike,
        sigma_0: float = 100.0,
        T_init: float | None = None,
    ) -> ArrayLike:
        """Predict stress relaxation.

        Args:
            t: Time array.
            sigma_0: Initial stress.
            T_init: Initial temperature.

        Returns:
            Stress vs time.
        """
        params = self._get_params_dict()
        return self._simulate_transient(
            jnp.asarray(t), params, "relaxation", sigma_0=sigma_0, T_init=T_init
        )

    def predict_creep(
        self,
        t: ArrayLike,
        sigma_applied: float = 50.0,
        T_init: float | None = None,
    ) -> ArrayLike:
        """Predict creep response.

        Args:
            t: Time array.
            sigma_applied: Applied constant stress.
            T_init: Initial temperature.

        Returns:
            Strain vs time.
        """
        params = self._get_params_dict()
        return self._simulate_transient(
            jnp.asarray(t), params, "creep", sigma_applied=sigma_applied, T_init=T_init
        )

    def predict_oscillation(
        self,
        omega: ArrayLike,
        gamma_0: float = 0.01,
        n_cycles: int = 5,
    ) -> jnp.ndarray:
        """Predict oscillatory response (SAOS G*, G', G'').

        For small amplitudes, this uses the linearized response.
        For accurate nonlinear response, use predict_laos().

        Args:
            omega: Angular frequency array.
            gamma_0: Strain amplitude (should be small).
            n_cycles: Number of cycles to simulate.

        Returns:
            Complex modulus G* = G' + i·G'' for each frequency.
        """
        omega_arr = jnp.asarray(omega)
        params = self._get_params_dict()

        # Simulate each frequency
        G_star = []
        for w in omega_arr:
            # Time array for n_cycles
            period = 2 * jnp.pi / w
            t = jnp.linspace(0, n_cycles * period, int(100 * n_cycles))

            # Strain signal
            strain = gamma_0 * jnp.sin(w * t)

            # Predict stress
            stress = self._predict_from_params(t, strain, params)

            # Extract last cycle for Fourier analysis
            last_cycle_start = int(len(t) * (n_cycles - 1) / n_cycles)
            t_last = t[last_cycle_start:]
            stress_last = stress[last_cycle_start:]

            # Fourier decomposition (first harmonic)
            # G' = (1/γ₀) · (2/T) ∫ σ·sin(ωt) dt
            # G'' = (1/γ₀) · (2/T) ∫ σ·cos(ωt) dt
            T_cycle = 2 * jnp.pi / w
            dt = t_last[1] - t_last[0]

            G_prime = (2 / (gamma_0 * T_cycle)) * jnp.trapezoid(
                stress_last * jnp.sin(w * t_last), dx=dt
            )
            G_double_prime = (2 / (gamma_0 * T_cycle)) * jnp.trapezoid(
                stress_last * jnp.cos(w * t_last), dx=dt
            )

            G_star.append(G_prime + 1j * G_double_prime)

        return jnp.array(G_star)

    def predict_laos(
        self,
        t: ArrayLike,
        gamma_0: float = 1.0,
        omega: float = 1.0,
        T_init: float | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Predict LAOS (Large Amplitude Oscillatory Shear) response.

        Args:
            t: Time array.
            gamma_0: Strain amplitude.
            omega: Angular frequency.
            T_init: Initial temperature.

        Returns:
            Dictionary with 'time', 'strain', 'stress', and optionally 'temperature'.
        """
        t_arr = jnp.asarray(t)
        strain = gamma_0 * jnp.sin(omega * t_arr)
        params = self._get_params_dict()

        from rheojax.models.fikh._kernels import (
            fikh_scan_kernel_isothermal,
            fikh_scan_kernel_thermal,
        )

        alpha = params.get("alpha_structure", self.alpha_structure)

        if self.include_thermal:
            T_0 = T_init if T_init is not None else params.get("T_env", 298.15)
            stress, temperature = fikh_scan_kernel_thermal(
                t_arr,
                strain,
                n_history=self.n_history,
                alpha=alpha,
                use_viscosity=True,
                T_init=T_0,
                **params,
            )
            return {
                "time": t_arr,
                "strain": strain,
                "stress": stress,
                "temperature": temperature,
            }
        else:
            stress = fikh_scan_kernel_isothermal(
                t_arr,
                strain,
                n_history=self.n_history,
                alpha=alpha,
                use_viscosity=True,
                **params,
            )
            return {
                "time": t_arr,
                "strain": strain,
                "stress": stress,
            }

    # =========================================================================
    # Bayesian Interface
    # =========================================================================

    def model_function(
        self,
        X: ArrayLike,
        params: ArrayLike | dict[str, Any],
        test_mode: str | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Model function for NumPyro Bayesian inference.

        This method provides a pure function interface for Bayesian sampling,
        capturing the test_mode via closure for correct mode-aware inference.

        Args:
            X: Input data.
            params: Parameter array or dictionary.
            test_mode: Protocol (uses stored _test_mode if None).
            **kwargs: Protocol-specific arguments (e.g., strain, sigma_0).

        Returns:
            Predicted values.
        """
        mode = test_mode or self._test_mode or "startup"

        # Convert array to dict if needed
        if isinstance(params, (np.ndarray, jnp.ndarray)):
            param_names = list(self.parameters.keys())
            param_dict = dict(zip(param_names, params, strict=False))
        else:
            param_dict = dict(params)

        mode_enum = self._validate_test_mode(mode)

        if mode_enum == TestMode.FLOW_CURVE:
            from rheojax.models.fikh._kernels import fikh_flow_curve_steady_state

            gamma_dot = jnp.asarray(X)
            return fikh_flow_curve_steady_state(
                gamma_dot, include_thermal=self.include_thermal, **param_dict
            )

        elif mode_enum in (TestMode.CREEP, TestMode.RELAXATION):
            t = jnp.asarray(X)
            gamma_dot = kwargs.get("gamma_dot", param_dict.pop("_gamma_dot", 0.0))
            sigma_applied = kwargs.get(
                "sigma_applied", param_dict.pop("_sigma_applied", 100.0)
            )
            sigma_0 = kwargs.get("sigma_0", param_dict.pop("_sigma_0", 60.0))
            return self._simulate_transient(
                t, param_dict, mode_enum.value, gamma_dot, sigma_applied, sigma_0
            )

        elif mode_enum == TestMode.OSCILLATION:
            # Frequency-domain SAOS: X is omega, return |G*| for Bayesian fitting
            omega = jnp.asarray(X)
            gamma_0 = kwargs.get("gamma_0", param_dict.pop("_gamma_0", 0.01))
            n_cycles = kwargs.get("n_cycles", param_dict.pop("_n_cycles", 5))
            G_star = self._predict_oscillation_from_params(
                omega, param_dict, gamma_0, n_cycles
            )
            # Return magnitude for comparison with |G*| target
            return jnp.abs(G_star)

        else:
            # Strain-driven protocols (startup, laos)
            times, strains = self._extract_time_strain(X, **kwargs)
            return self._predict_from_params(times, strains, param_dict)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_limiting_behavior(self) -> dict[str, Any]:
        """Get limiting behavior diagnostics.

        Returns:
            Dictionary with limiting cases and expected behavior.
        """
        alpha = self.parameters.get_value("alpha_structure")
        E_a = self.parameters.get_value("E_a") if self.include_thermal else 0.0

        return {
            "fractional_order": alpha,
            "is_near_integer": alpha > 0.95,
            "memory_type": (
                "weak (near exponential)" if alpha > 0.7 else "strong (power-law)"
            ),
            "thermal_coupling": self.include_thermal,
            "arrhenius_enabled": E_a > 0 if self.include_thermal else False,
            "limiting_case_alpha_1": "Classical MIKH behavior",
            "limiting_case_E_a_0": "Isothermal FIKH behavior",
        }

    def precompile(self, n_points: int = 100, verbose: bool = True) -> float:
        """Precompile JIT kernels for faster subsequent predictions.

        Triggers JAX JIT compilation of the core FIKH kernels by running
        a small dummy prediction. This is useful when you want to avoid
        the compilation overhead on first real prediction.

        Args:
            n_points: Number of time points for dummy data.
            verbose: Print compilation time if True.

        Returns:
            Compilation time in seconds.

        Example:
            >>> model = FIKH(include_thermal=True)
            >>> compile_time = model.precompile()  # Triggers JIT
            >>> # Now predictions will be fast
            >>> sigma = model.predict_startup(t_real, gamma_dot=1.0)
        """
        import time as time_module

        # Create dummy data
        t_dummy = jnp.linspace(0, 10, n_points)
        strain_dummy = 0.1 * t_dummy  # Linear ramp

        params = self._get_params_dict()

        start = time_module.perf_counter()

        # Trigger isothermal kernel compilation
        from rheojax.models.fikh._kernels import (
            fikh_scan_kernel_isothermal,
            fikh_scan_kernel_thermal,
        )

        alpha = params.get("alpha_structure", self.alpha_structure)

        # Always compile isothermal kernel
        _ = fikh_scan_kernel_isothermal(
            t_dummy,
            strain_dummy,
            n_history=self.n_history,
            alpha=alpha,
            use_viscosity=True,
            **params,
        )

        # Compile thermal kernel if enabled
        if self.include_thermal:
            T_init = params.get("T_env", params.get("T_ref", 298.15))
            _ = fikh_scan_kernel_thermal(
                t_dummy,
                strain_dummy,
                n_history=self.n_history,
                alpha=alpha,
                use_viscosity=True,
                T_init=T_init,
                **params,
            )

        elapsed = time_module.perf_counter() - start

        if verbose:
            logger.info(
                "FIKH kernels precompiled",
                compile_time_s=f"{elapsed:.2f}",
                include_thermal=self.include_thermal,
            )

        return elapsed

    def __repr__(self) -> str:
        """String representation."""
        alpha = self.parameters.get_value("alpha_structure")
        return (
            f"FIKH(include_thermal={self.include_thermal}, "
            f"alpha_structure={alpha:.3f}, n_history={self.n_history})"
        )
