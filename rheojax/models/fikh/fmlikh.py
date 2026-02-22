"""FMLIKH (Fractional Multi-Layer IKH) Model.

This module implements the multi-layer variant of FIKH, allowing multiple
Maxwell-like modes with independent elastic and kinematic hardening
parameters while sharing yield and thixotropy behavior.

Key Features:
    - Multiple viscoelastic modes (Prony-series-like)
    - Per-mode: G_i, eta_i, C_i, gamma_dyn_i
    - Shared: sigma_y0, delta_sigma_y, tau_thix, Gamma, thermal params
    - Optional per-mode or shared fractional order

Use Cases:
    - Materials with broad relaxation spectra
    - Complex rheological signatures requiring multiple time scales
    - Fitting to wide-frequency SAOS data

Example:
    >>> from rheojax.models.fikh import FMLIKH
    >>> model = FMLIKH(n_modes=3, include_thermal=False)
    >>> model.fit(omega, G_star, test_mode='oscillation')
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode, Protocol, TestMode
from rheojax.logging import get_logger
from rheojax.models.fikh._base import FIKHBase
from rheojax.models.fractional.fractional_mixin import FRACTIONAL_ORDER_BOUNDS
from rheojax.utils.optimization import nlsq_optimize

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


@ModelRegistry.register(
    "fmlikh",
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
class FMLIKH(FIKHBase):
    r"""Fractional Multi-Layer Isotropic-Kinematic Hardening (FMLIKH) Model.

    A multi-mode extension of FIKH supporting multiple viscoelastic relaxation
    processes with shared yield and thixotropy behavior.

    The total stress is the sum of contributions from N modes:
        σ_total = Σᵢ σᵢ + η_inf·γ̇

    Each mode i has its own:
        - Gᵢ: Shear modulus
        - ηᵢ: Viscosity (defines τᵢ = ηᵢ/Gᵢ)
        - Cᵢ: Kinematic hardening modulus
        - γ_dyn,ᵢ: Dynamic recovery parameter

    Shared across all modes:
        - σ_y0, δσ_y: Yield stress parameters
        - τ_thix, Γ: Thixotropy parameters
        - α: Fractional order (or per-mode if shared_alpha=False)
        - Thermal parameters

    This structure allows capturing materials with:
        - Multiple relaxation time scales
        - Complex SAOS signatures (wide frequency range)
        - Non-trivial startup overshoot dynamics

    Parameters:
        n_modes: Number of viscoelastic modes.
        shared_alpha: If True, use single α for all modes. If False, α_i per mode.
        Other parameters inherited from FIKHBase.

    Example:
        >>> model = FMLIKH(n_modes=3, include_thermal=True, shared_alpha=True)
        >>> model.fit(t, stress, test_mode='startup', strain=strain)

        >>> # Access per-mode parameters
        >>> G_values = [model.parameters.get_value(f'G_{i}') for i in range(3)]
    """

    def __init__(
        self,
        n_modes: int = 3,
        include_thermal: bool = True,
        include_isotropic_hardening: bool = False,
        shared_alpha: bool = True,
        alpha_structure: float = 0.5,
        n_history: int = 100,
    ):
        """Initialize FMLIKH model.

        Args:
            n_modes: Number of viscoelastic modes (≥1).
            include_thermal: Enable thermokinematic coupling.
            include_isotropic_hardening: Enable isotropic hardening R.
            shared_alpha: Use single fractional order (True) or per-mode (False).
            alpha_structure: Fractional order (used if shared_alpha=True).
            n_history: History buffer size.
        """
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")

        self._n_modes = n_modes
        self.shared_alpha = shared_alpha

        # Initialize base (this sets up shared parameters)
        super().__init__(
            include_thermal=include_thermal,
            include_isotropic_hardening=include_isotropic_hardening,
            alpha_structure=alpha_structure,
            n_history=n_history,
        )

        # Setup multi-mode parameters (overrides single G, eta, C, gamma_dyn)
        self._setup_per_mode_parameters()

        logger.debug(
            "Initialized FMLIKH model",
            n_modes=n_modes,
            shared_alpha=shared_alpha,
            include_thermal=include_thermal,
        )

    def _setup_per_mode_parameters(self) -> None:
        """Setup per-mode parameters, replacing single-mode defaults."""
        # Remove single-mode parameters
        for param in ["G", "eta", "C", "gamma_dyn"]:
            if param in self.parameters:
                del self.parameters._parameters[param]
                if param in self.parameters._order:
                    self.parameters._order.remove(param)

        # Also remove single alpha_structure if using per-mode
        if not self.shared_alpha:
            if "alpha_structure" in self.parameters:
                del self.parameters._parameters["alpha_structure"]
                if "alpha_structure" in self.parameters._order:
                    self.parameters._order.remove("alpha_structure")

        # Add per-mode parameters
        for i in range(self._n_modes):
            # Modulus - logarithmically spaced defaults
            G_default = 1e3 / (10**i)
            self.parameters.add(
                f"G_{i}",
                value=G_default,
                bounds=(1e-3, 1e9),
                units="Pa",
                description=f"Shear modulus for mode {i}",
            )

            # Viscosity - also logarithmically spaced
            eta_default = 1e6 / (10**i)
            self.parameters.add(
                f"eta_{i}",
                value=eta_default,
                bounds=(1e-3, 1e12),
                units="Pa s",
                description=f"Viscosity for mode {i}",
            )

            # Kinematic hardening
            C_default = 5e2 / (10**i)
            self.parameters.add(
                f"C_{i}",
                value=C_default,
                bounds=(0.0, 1e9),
                units="Pa",
                description=f"Kinematic hardening modulus for mode {i}",
            )

            # Dynamic recovery
            self.parameters.add(
                f"gamma_dyn_{i}",
                value=1.0,
                bounds=(0.0, 1e4),
                units="-",
                description=f"Dynamic recovery parameter for mode {i}",
            )

            # Per-mode fractional order (if not shared)
            if not self.shared_alpha:
                self.parameters.add(
                    f"alpha_{i}",
                    value=self.alpha_structure,
                    bounds=FRACTIONAL_ORDER_BOUNDS,
                    units="-",
                    description=f"Fractional order for mode {i}",
                )

    @property
    def n_modes(self) -> int:
        """Number of viscoelastic modes."""
        return self._n_modes

    def _get_mode_params(self, params: dict[str, Any], mode_idx: int) -> dict[str, Any]:
        """Extract parameters for a single mode.

        Args:
            params: Full parameter dictionary.
            mode_idx: Mode index (0 to n_modes-1).

        Returns:
            Dictionary with mode-specific parameters (G, eta, C, gamma_dyn)
            plus all shared parameters.
        """
        # F-030: build a clean dict with only the keys the kernel expects,
        # instead of copying all params (which leaks G_0, G_1, etc.)
        mode_params = {
            k: v for k, v in params.items()
            if not any(k.startswith(p) and k[-1].isdigit()
                       for p in ("G_", "eta_", "C_", "gamma_dyn_", "alpha_"))
        }

        # Set modal parameters
        mode_params["G"] = params.get(f"G_{mode_idx}", 1e3)
        mode_params["eta"] = params.get(f"eta_{mode_idx}", 1e6)
        mode_params["C"] = params.get(f"C_{mode_idx}", 5e2)
        mode_params["gamma_dyn"] = params.get(f"gamma_dyn_{mode_idx}", 1.0)

        # Fractional order
        if self.shared_alpha:
            mode_params["alpha_structure"] = params.get(
                "alpha_structure", self.alpha_structure
            )
        else:
            mode_params["alpha_structure"] = params.get(
                f"alpha_{mode_idx}", self.alpha_structure
            )

        return mode_params

    def _predict_from_params(
        self,
        times: jnp.ndarray,
        strains: jnp.ndarray,
        params: dict[str, Any],
    ) -> jnp.ndarray:
        """Predict stress as sum of all modes.

        Args:
            times: Time array.
            strains: Strain array.
            params: Full parameter dictionary.

        Returns:
            Total predicted stress.
        """
        from rheojax.models.fikh._kernels import (
            fikh_scan_kernel_isothermal,
            fikh_scan_kernel_thermal,
        )

        total_stress = jnp.zeros_like(times)

        for i in range(self._n_modes):
            mode_params = self._get_mode_params(params, i)
            alpha = mode_params.get("alpha_structure", self.alpha_structure)

            # Set eta_inf to 0 for intermediate modes (add only once at end)
            if i < self._n_modes - 1:
                mode_params["eta_inf"] = 0.0

            if self.include_thermal:
                T_init = mode_params.get("T_env", mode_params.get("T_ref", 298.15))
                stress_i, _ = fikh_scan_kernel_thermal(
                    times,
                    strains,
                    n_history=self.n_history,
                    alpha=alpha,
                    use_viscosity=(i == self._n_modes - 1),
                    T_init=T_init,
                    **mode_params,
                )
            else:
                stress_i = fikh_scan_kernel_isothermal(
                    times,
                    strains,
                    n_history=self.n_history,
                    alpha=alpha,
                    use_viscosity=(i == self._n_modes - 1),
                    **mode_params,
                )

            total_stress = total_stress + stress_i

        return total_stress

    def _fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FMLIKH:
        """Fit multi-layer model."""
        test_mode = kwargs.get("test_mode", "startup")
        self._test_mode = test_mode

        mode = self._validate_test_mode(test_mode)

        if mode == TestMode.FLOW_CURVE:
            return self._fit_flow_curve(X, y, **kwargs)
        elif mode in (TestMode.CREEP, TestMode.RELAXATION):
            return self._fit_ode_formulation(X, y, **kwargs)
        elif mode == TestMode.OSCILLATION:
            return self._fit_oscillation(X, y, **kwargs)
        else:
            return self._fit_return_mapping(X, y, **kwargs)

    def _fit_flow_curve(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FMLIKH:
        """Fit to flow curve data."""
        gamma_dot = jnp.asarray(X)
        sigma_target = jnp.asarray(y)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            sigma_pred = self._predict_flow_curve_from_params(gamma_dot, p_dict)
            return sigma_pred - sigma_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _predict_flow_curve_from_params(
        self,
        gamma_dot: jnp.ndarray,
        params: dict[str, Any],
    ) -> jnp.ndarray:
        """Predict steady-state flow curve from all modes."""
        from rheojax.models.fikh._kernels import fikh_flow_curve_steady_state

        total_stress = jnp.zeros_like(gamma_dot)

        for i in range(self._n_modes):
            mode_params = self._get_mode_params(params, i)

            # Only last mode contributes eta_inf
            if i < self._n_modes - 1:
                mode_params["eta_inf"] = 0.0

            stress_i = fikh_flow_curve_steady_state(
                gamma_dot,
                include_thermal=self.include_thermal,
                **mode_params,
            )
            total_stress = total_stress + stress_i

        return total_stress

    def _fit_ode_formulation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FMLIKH:
        """Fit using ODE formulation for relaxation/creep.

        Uses per-mode _simulate_transient and sums the contributions.
        """
        t = jnp.asarray(X)
        y_target = jnp.asarray(y)
        mode = self._validate_test_mode(
            kwargs.get("test_mode", self._test_mode or "startup")
        )

        # Extract kwargs relevant to transient simulation
        sim_kwargs = {
            k: kwargs[k]
            for k in ("sigma_0", "sigma_applied", "gamma_dot", "T_init")
            if k in kwargs
        }

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            y_pred = self._predict_transient_multimode(t, p_dict, mode, **sim_kwargs)
            return y_pred - y_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_return_mapping(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FMLIKH:
        """Fit using return mapping."""
        times, strains = self._extract_time_strain(X, **kwargs)
        sigma_target = jnp.asarray(y)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            sigma_pred = self._predict_from_params(times, strains, p_dict)
            return sigma_pred - sigma_target

        nlsq_optimize(objective, self.parameters, **kwargs)
        return self

    def _fit_oscillation(self, X: ArrayLike, y: ArrayLike, **kwargs) -> FMLIKH:
        """Fit to oscillatory data (SAOS).

        Fits multi-mode FMLIKH to frequency-domain SAOS data.

        Args:
            X: Angular frequency array (omega) [rad/s].
            y: Target modulus data (complex G* or |G*|).
            **kwargs: Options including gamma_0, n_cycles.

        Returns:
            Self with fitted parameters.
        """
        omega = jnp.asarray(X)
        y_arr = jnp.asarray(y)
        gamma_0 = kwargs.get("gamma_0", 0.01)
        n_cycles = kwargs.get("n_cycles", 5)
        is_complex = jnp.iscomplexobj(y_arr)

        def objective(param_values):
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))
            G_star_pred = self._predict_oscillation_from_params(
                omega, p_dict, gamma_0, n_cycles
            )

            if is_complex:
                residuals = jnp.concatenate(
                    [
                        jnp.real(G_star_pred) - jnp.real(y_arr),
                        jnp.imag(G_star_pred) - jnp.imag(y_arr),
                    ]
                )
            else:
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

        Sum contributions from all modes.
        F-004: Vectorized via jax.vmap over frequencies (replaces Python loop).

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
        last_cycle_start = n_pts * (n_cycles - 1) // n_cycles
        n_last = n_pts - last_cycle_start

        # Close over constants for vmap
        include_thermal = self.include_thermal
        n_history = self.n_history
        n_modes = self._n_modes

        # Pre-compute per-mode params (Python loop — static, not over frequencies)
        mode_params_list = [self._get_mode_params(params, i) for i in range(n_modes)]
        mode_alphas = [mp.get("alpha_structure", alpha) for mp in mode_params_list]

        def predict_single_omega(w):
            """Compute G* at a single frequency (vmappable)."""
            period = 2 * jnp.pi / w
            t = jnp.linspace(0.0, n_cycles * period, n_pts)
            strain = gamma_0 * jnp.sin(w * t)

            total_stress = jnp.zeros(n_pts)

            # Sum stress from all modes (Python loop over modes — static count)
            for i in range(n_modes):
                mode_params = mode_params_list[i]
                mode_alpha = mode_alphas[i]

                if include_thermal:
                    T_init = params.get("T_env", params.get("T_ref", 298.15))
                    stress_i, _ = fikh_scan_kernel_thermal(
                        t, strain,
                        n_history=n_history, alpha=mode_alpha,
                        use_viscosity=(i == n_modes - 1),
                        T_init=T_init, **mode_params,
                    )
                else:
                    stress_i = fikh_scan_kernel_isothermal(
                        t, strain,
                        n_history=n_history, alpha=mode_alpha,
                        use_viscosity=(i == n_modes - 1),
                        **mode_params,
                    )

                total_stress = total_stress + stress_i

            # Extract last cycle via dynamic_slice (trace-safe)
            t_last = jax.lax.dynamic_slice(t, [last_cycle_start], [n_last])
            stress_last = jax.lax.dynamic_slice(total_stress, [last_cycle_start], [n_last])

            # Fourier decomposition (first harmonic)
            dt = t_last[1] - t_last[0]
            T_cycle = t_last[-1] - t_last[0] + dt

            G_prime = (2 / (gamma_0 * T_cycle)) * jnp.trapezoid(
                stress_last * jnp.sin(w * t_last), dx=dt
            )
            G_double_prime = (2 / (gamma_0 * T_cycle)) * jnp.trapezoid(
                stress_last * jnp.cos(w * t_last), dx=dt
            )

            return jnp.array([G_prime, G_double_prime])

        results = jax.vmap(predict_single_omega)(omega)  # (N_omega, 2)
        return results[:, 0] + 1j * results[:, 1]

    def predict_oscillation(
        self,
        omega: ArrayLike,
        gamma_0: float = 0.01,
        n_cycles: int = 5,
    ) -> jnp.ndarray:
        """Predict oscillatory response (SAOS G*) for multi-mode model.

        Args:
            omega: Angular frequency array.
            gamma_0: Strain amplitude.
            n_cycles: Number of cycles to simulate.

        Returns:
            Complex modulus G* = G' + i·G'' for each frequency.
        """
        omega_arr = jnp.asarray(omega)
        params = self._get_params_dict()
        return self._predict_oscillation_from_params(
            omega_arr, params, gamma_0, n_cycles
        )

    def _predict(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Predict based on test_mode."""
        test_mode = kwargs.get("test_mode", self._test_mode or "startup")
        mode = self._validate_test_mode(test_mode)
        params = self._get_params_dict()

        if mode == TestMode.FLOW_CURVE:
            gamma_dot = jnp.asarray(X)
            return self._predict_flow_curve_from_params(gamma_dot, params)
        elif mode == TestMode.OSCILLATION:
            omega = jnp.asarray(X)
            gamma_0 = kwargs.get("gamma_0", 0.01)
            n_cycles = kwargs.get("n_cycles", 5)
            return self._predict_oscillation_from_params(
                omega, params, gamma_0, n_cycles
            )
        elif mode in (TestMode.CREEP, TestMode.RELAXATION):
            return self._predict_transient_multimode(X, params, mode, **kwargs)
        else:
            times, strains = self._extract_time_strain(X, **kwargs)
            return self._predict_from_params(times, strains, params)

    def _predict_transient_multimode(
        self,
        X: ArrayLike,
        params: dict[str, Any],
        mode: TestMode,
        **kwargs,
    ) -> jnp.ndarray:
        """Multi-mode transient prediction for relaxation and creep.

        Runs _simulate_transient per mode and sums the result.
        For relaxation, initial stress is distributed proportional to G_i.
        """
        t = jnp.asarray(X)
        sigma_0 = kwargs.get("sigma_0", 60.0)
        sigma_applied = kwargs.get("sigma_applied", 100.0)
        T_init = kwargs.get("T_init", None)

        total_result = jnp.zeros_like(t)

        # Compute total G for distributing sigma_0 across modes
        # F-016: use jnp.sum for trace-safe summation of potentially traced values
        G_values = jnp.array([params.get(f"G_{i}", 1e3) for i in range(self._n_modes)])
        G_total = jnp.sum(G_values)

        for i in range(self._n_modes):
            mode_params = self._get_mode_params(params, i)

            # Only add eta_inf contribution on last mode
            if i < self._n_modes - 1:
                mode_params["eta_inf"] = 0.0

            if mode == TestMode.RELAXATION:
                # Distribute sigma_0 proportional to mode stiffness
                G_i = mode_params.get("G", 1e3)
                mode_sigma_0 = sigma_0 * G_i / jnp.maximum(G_total, 1e-10)
                result_i = self._simulate_transient(
                    t,
                    mode_params,
                    mode.value,
                    gamma_dot=0.0,
                    sigma_0=mode_sigma_0,
                    T_init=T_init,
                )
            else:  # CREEP
                result_i = self._simulate_transient(
                    t,
                    mode_params,
                    mode.value,
                    sigma_applied=sigma_applied,
                    T_init=T_init,
                )

            total_result = total_result + result_i

        return total_result

    def model_function(
        self,
        X: ArrayLike,
        params: ArrayLike | dict[str, Any],
        test_mode: str | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Model function for Bayesian inference."""
        # F-012: prefer explicit test_mode; fall back to _last_fit_kwargs
        mode = (
            test_mode
            or getattr(self, "_last_fit_kwargs", {}).get("test_mode")
            or self._test_mode
            or "startup"
        )

        if isinstance(params, (np.ndarray, jnp.ndarray)):
            param_names = list(self.parameters.keys())
            param_dict = dict(zip(param_names, params, strict=False))
        else:
            param_dict = dict(params)

        mode_enum = self._validate_test_mode(mode)

        if mode_enum == TestMode.FLOW_CURVE:
            gamma_dot = jnp.asarray(X)
            return self._predict_flow_curve_from_params(gamma_dot, param_dict)
        elif mode_enum == TestMode.OSCILLATION:
            omega = jnp.asarray(X)
            gamma_0 = kwargs.get("gamma_0", param_dict.pop("_gamma_0", 0.01))
            n_cycles = kwargs.get("n_cycles", param_dict.pop("_n_cycles", 5))
            G_star = self._predict_oscillation_from_params(
                omega, param_dict, gamma_0, n_cycles
            )
            return jnp.column_stack([jnp.real(G_star), jnp.imag(G_star)])
        elif mode_enum in (TestMode.CREEP, TestMode.RELAXATION):
            return self._predict_transient_multimode(X, param_dict, mode_enum, **kwargs)
        else:
            times, strains = self._extract_time_strain(X, **kwargs)
            return self._predict_from_params(times, strains, param_dict)

    def get_mode_info(self) -> dict[str, Any]:
        """Get information about each mode.

        Returns:
            Dictionary with per-mode parameters and derived quantities.
        """
        info = {
            "n_modes": self._n_modes,
            "shared_alpha": self.shared_alpha,
            "modes": [],
        }

        params = self._get_params_dict()

        for i in range(self._n_modes):
            G_i = params.get(f"G_{i}", 1e3)
            eta_i = params.get(f"eta_{i}", 1e6)
            C_i = params.get(f"C_{i}", 5e2)
            gamma_dyn_i = params.get(f"gamma_dyn_{i}", 1.0)

            tau_i = eta_i / max(G_i, 1e-12)

            mode_info = {
                "mode": i,
                "G": G_i,
                "eta": eta_i,
                "tau": tau_i,
                "C": C_i,
                "gamma_dyn": gamma_dyn_i,
            }

            if not self.shared_alpha:
                mode_info["alpha"] = params.get(f"alpha_{i}", self.alpha_structure)

            info["modes"].append(mode_info)  # type: ignore[attr-defined]

        if self.shared_alpha:
            info["alpha_shared"] = params.get("alpha_structure", self.alpha_structure)

        return info

    def __repr__(self) -> str:
        """String representation."""
        alpha = (
            self.parameters.get_value("alpha_structure")
            if self.shared_alpha
            else f"[per-mode x{self._n_modes}]"
        )
        return (
            f"FMLIKH(n_modes={self._n_modes}, include_thermal={self.include_thermal}, "
            f"shared_alpha={self.shared_alpha}, alpha_structure={alpha})"
        )
