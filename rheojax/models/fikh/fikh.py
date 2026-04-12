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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode, Protocol, TestMode
from rheojax.logging import get_logger
from rheojax.models.fikh._base import FIKHBase
from rheojax.utils.optimization import create_least_squares_objective, nlsq_optimize

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


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
        stable_dt: float = 0.01,
    ):
        """Initialize FIKH model.

        Args:
            include_thermal: Enable thermokinematic coupling (Arrhenius + heating).
            include_isotropic_hardening: Enable isotropic hardening R.
            alpha_structure: Fractional order for structure (0 < α < 1).
                - α → 0: Strong memory (slow recovery)
                - α → 1: Weak memory (fast, exponential recovery)
            n_history: History buffer size for Caputo derivative.
            stable_dt: Internal integration substep (seconds) for startup / LAOS.
                See ``FIKHBase`` for the full explanation. Coarse user grids
                are densified to this step before the explicit return-mapping
                kernel runs. Set to 0 to disable. Default 0.02 s.
        """
        super().__init__(
            include_thermal=include_thermal,
            include_isotropic_hardening=include_isotropic_hardening,
            alpha_structure=alpha_structure,
            n_history=n_history,
            stable_dt=stable_dt,
        )
        logger.debug(
            "Initialized FIKH model",
            include_thermal=include_thermal,
            alpha_structure=alpha_structure,
        )

    # =========================================================================
    # Fitting Methods
    # =========================================================================

    def _fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FIKH:
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

    def _fit_flow_curve(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FIKH:
        """Fit to steady-state flow curve data."""
        from rheojax.models.fikh._kernels import fikh_flow_curve_steady_state

        gamma_dot = jnp.asarray(X)
        sigma_target = jnp.asarray(y, dtype=jnp.float64)

        def model_fn(x_data, params):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, params, strict=False))
            return fikh_flow_curve_steady_state(
                x_data, include_thermal=self.include_thermal, **p_dict
            )

        # Flow curves span decades — log residuals give equal weight to
        # low and high shear rate regions.
        objective = create_least_squares_objective(
            model_fn,
            gamma_dot,
            sigma_target,
            use_log_residuals=kwargs.pop("use_log_residuals", True),
        )

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_ode_formulation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FIKH:
        """Fit using ODE formulation (creep/relaxation)."""
        t = jnp.asarray(X)
        y_target = jnp.asarray(y, dtype=jnp.float64)
        test_mode = kwargs.get("test_mode", "relaxation")
        gamma_dot = kwargs.get("gamma_dot", 0.0)
        sigma_applied = kwargs.get("sigma_applied", 100.0)
        sigma_0 = kwargs.get("sigma_0", 100.0)
        T_init = kwargs.get("T_init", None)

        # Cache protocol kwargs so model_function() can retrieve them during NUTS
        self._fit_gamma_dot = gamma_dot
        self._fit_sigma_applied = sigma_applied
        self._fit_sigma_0 = sigma_0

        def model_fn(x_data, params):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, params, strict=False))
            return self._simulate_transient(
                x_data, p_dict, test_mode, gamma_dot, sigma_applied, sigma_0, T_init
            )

        # Transient data (creep/relaxation) often starts at zero — normalize=False
        # avoids division by ~0 at early time points (same rationale as fluidity).
        objective = create_least_squares_objective(
            model_fn,
            t,
            y_target,
            normalize=False,
            use_log_residuals=kwargs.pop("use_log_residuals", False),
        )

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_return_mapping(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FIKH:
        """Fit using return mapping (startup/LAOS)."""
        times, strains = self._extract_time_strain(X, **kwargs)
        sigma_target = jnp.asarray(y, dtype=jnp.float64)

        # Pre-compute the stable-dt substep count from concrete times and cache
        # it so the subsequent NUTS trace reuses the same integration grid.
        # See FIKHBase._compute_n_sub / _densify_grid_for_return_mapping for
        # why this is necessary (explicit return mapping is only stable when
        # G·Δγ per step is small relative to the yield stress).
        self._n_sub_cached = self._compute_n_sub(times)

        def model_fn(x_data, params):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, params, strict=False))
            return self._predict_from_params(x_data, strains, p_dict)

        # Startup/LAOS stress crosses zero — normalize=False avoids
        # division by ~0 at the zero crossings.
        objective = create_least_squares_objective(
            model_fn,
            times,
            sigma_target,
            normalize=False,
            use_log_residuals=kwargs.pop("use_log_residuals", False),
        )

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_oscillation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FIKH:
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

        # Cache protocol kwargs so model_function() can retrieve them during NUTS
        self._fit_gamma_0 = gamma_0
        self._fit_n_cycles = n_cycles

        # Determine if fitting to complex, (N, 2) [G', G''], or magnitude
        is_complex = jnp.iscomplexobj(y_arr)
        is_stacked = y_arr.ndim == 2 and y_arr.shape[1] == 2

        # Pre-compute normalization denominators for consistent residual weighting.
        # FIKH oscillation uses time-domain FFT (not analytical), so we handle
        # the complex dispatch manually rather than through create_least_squares_objective.
        _norm_floor = jnp.float64(1e-10)
        if is_complex:
            _norm_Gp = jnp.maximum(jnp.abs(jnp.real(y_arr)), _norm_floor)
            _norm_Gpp = jnp.maximum(jnp.abs(jnp.imag(y_arr)), _norm_floor)
        elif is_stacked:
            _norm_Gp = jnp.maximum(jnp.abs(y_arr[:, 0]), _norm_floor)
            _norm_Gpp = jnp.maximum(jnp.abs(y_arr[:, 1]), _norm_floor)
        else:
            _norm_mag = jnp.maximum(jnp.abs(y_arr), _norm_floor)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))

            # Predict G* at each frequency using time-domain simulation
            G_star_pred = self._predict_oscillation_from_params(
                omega, p_dict, gamma_0, n_cycles
            )

            if is_complex:
                # Fit both G' and G'' by stacking normalized residuals
                residuals = jnp.concatenate(
                    [
                        (jnp.real(G_star_pred) - jnp.real(y_arr)) / _norm_Gp,
                        (jnp.imag(G_star_pred) - jnp.imag(y_arr)) / _norm_Gpp,
                    ]
                )
            elif is_stacked:
                # (N, 2) array - [G', G''] format
                residuals = jnp.concatenate(
                    [
                        (jnp.real(G_star_pred) - y_arr[:, 0]) / _norm_Gp,
                        (jnp.imag(G_star_pred) - y_arr[:, 1]) / _norm_Gpp,
                    ]
                )
            else:
                # Fit to magnitude |G*|
                residuals = (jnp.abs(G_star_pred) - jnp.abs(y_arr)) / _norm_mag

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
        F-004/F-024: Vectorized via jax.vmap over frequencies (replaces Python loop).

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
        n_pts = 100 * n_cycles
        # Static slice index for last cycle extraction
        last_cycle_start = n_pts * (n_cycles - 1) // n_cycles
        n_last = n_pts - last_cycle_start

        # Close over params/options so only omega varies
        include_thermal = self.include_thermal
        n_history = self.n_history

        def predict_single_omega(w):
            """Compute G* at a single frequency (vmappable)."""
            period = 2 * jnp.pi / w
            t = jnp.linspace(0.0, n_cycles * period, n_pts)
            strain = gamma_0 * jnp.sin(w * t)

            if include_thermal:
                T_init = params.get("T_env", params.get("T_ref", 298.15))
                stress, _ = fikh_scan_kernel_thermal(
                    t,
                    strain,
                    n_history=n_history,
                    alpha=alpha,
                    use_viscosity=True,
                    T_init=T_init,
                    **params,
                )
            else:
                stress = fikh_scan_kernel_isothermal(
                    t,
                    strain,
                    n_history=n_history,
                    alpha=alpha,
                    use_viscosity=True,
                    **params,
                )

            # Extract last cycle via dynamic_slice (trace-safe)
            t_last = jax.lax.dynamic_slice(t, [last_cycle_start], [n_last])
            stress_last = jax.lax.dynamic_slice(stress, [last_cycle_start], [n_last])

            # Fourier decomposition (first harmonic)
            # F-034: use dt from actual time points (not T_cycle / n_last)
            dt = t_last[1] - t_last[0]
            T_cycle = t_last[-1] - t_last[0] + dt  # exact integration span

            G_prime = (2 / (gamma_0 * T_cycle)) * jnp.trapezoid(
                stress_last * jnp.sin(w * t_last), dx=dt
            )
            G_double_prime = (2 / (gamma_0 * T_cycle)) * jnp.trapezoid(
                stress_last * jnp.cos(w * t_last), dx=dt
            )

            return jnp.array([G_prime, G_double_prime])

        # Vectorize over all frequencies at once
        results = jax.vmap(predict_single_omega)(omega)  # (N_omega, 2)
        return results[:, 0] + 1j * results[:, 1]

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
        Bayesian inference. The user's (times, strains) grid is densified to
        the base-class ``stable_dt`` before the scan kernel runs so that the
        explicit return mapping stays well inside its linearization regime,
        then the result is subsampled back to the user's time points.

        Args:
            times: Time array.
            strains: Strain array.
            params: Parameter dictionary.

        Returns:
            Predicted stress array at the user's time points.
        """
        from rheojax.models.fikh._kernels import (
            fikh_scan_kernel_isothermal,
            fikh_scan_kernel_thermal,
        )

        # Extract alpha (can now be a traced value since it's not in static_argnums)
        alpha = params.get("alpha_structure", self.alpha_structure)

        t_dense, strain_dense, n_sub = self._densify_grid_for_return_mapping(
            times, strains
        )

        if self.include_thermal:
            T_init = params.get("T_env", params.get("T_ref", 298.15))
            sigma_dense, _ = fikh_scan_kernel_thermal(
                t_dense,
                strain_dense,
                n_history=self.n_history,
                alpha=alpha,
                use_viscosity=True,
                T_init=T_init,
                **params,
            )
        else:
            sigma_dense = fikh_scan_kernel_isothermal(
                t_dense,
                strain_dense,
                n_history=self.n_history,
                alpha=alpha,
                use_viscosity=True,
                **params,
            )

        if n_sub > 1:
            return sigma_dense[::n_sub]
        return sigma_dense

    def _predict(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Predict based on test_mode.

        Args:
            X: Input data (shape depends on test_mode).
            **kwargs: Additional parameters.

        Returns:
            Predicted values.
        """
        _kw_mode = kwargs.get("test_mode")
        test_mode = (
            _kw_mode
            if _kw_mode is not None
            else (
                getattr(self, "_test_mode", None)
                if getattr(self, "_test_mode", None) is not None
                else "startup"
            )
        )
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
            return self._predict_oscillation_from_params(
                omega, params, gamma_0, n_cycles
            )

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

        # Reuse the vectorized implementation from _predict_oscillation_from_params
        return self._predict_oscillation_from_params(
            omega_arr, params, gamma_0, n_cycles
        )

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
        # Prefer explicit test_mode; fall back to _last_fit_kwargs
        # (set by fit()) over stale self._test_mode to avoid wrong NUTS likelihood
        if test_mode is not None:
            mode = test_mode
        elif getattr(self, "_last_fit_kwargs", {}).get("test_mode") is not None:
            mode = self._last_fit_kwargs["test_mode"]
        elif self._test_mode is not None:
            mode = self._test_mode
        else:
            mode = "startup"

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
            gamma_dot = kwargs.get("gamma_dot", getattr(self, "_fit_gamma_dot", 0.0))
            sigma_applied = kwargs.get(
                "sigma_applied", getattr(self, "_fit_sigma_applied", 100.0)
            )
            sigma_0 = kwargs.get("sigma_0", getattr(self, "_fit_sigma_0", 60.0))
            return self._simulate_transient(
                t, param_dict, mode_enum.value, gamma_dot, sigma_applied, sigma_0
            )

        elif mode_enum == TestMode.OSCILLATION:
            # Frequency-domain SAOS: X is omega, return |G*| for Bayesian fitting
            omega = jnp.asarray(X)
            gamma_0 = kwargs.get("gamma_0", getattr(self, "_fit_gamma_0", 0.01))
            n_cycles = kwargs.get("n_cycles", getattr(self, "_fit_n_cycles", 5))
            G_star = self._predict_oscillation_from_params(
                omega, param_dict, gamma_0, n_cycles
            )
            return jnp.column_stack([jnp.real(G_star), jnp.imag(G_star)])

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

    def precompile(
        self,
        test_mode: str = "relaxation",
        X=None,
        y=None,
        *,
        n_points: int = 100,
        verbose: bool = True,
    ) -> float:
        """Precompile JIT kernels for faster subsequent predictions.

        Triggers JAX JIT compilation of the core FIKH kernels by running
        a small dummy prediction. This is useful when you want to avoid
        the compilation overhead on first real prediction.

        Args:
            test_mode: Accepted for parent compatibility (unused).
            X: Accepted for parent compatibility (unused).
            y: Accepted for parent compatibility (unused).
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
