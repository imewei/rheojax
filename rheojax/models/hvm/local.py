"""HVMLocal: Local (0D) Hybrid Vitrimer Model.

Single-point constitutive model for vitrimers with three subnetworks:
1. Permanent (P): covalent crosslinks, neo-Hookean elastic
2. Exchangeable (E): associative vitrimer bonds with BER kinetics
3. Dissociative (D): physical reversible bonds, standard Maxwell

Supports 6 rheological protocols:
- Flow curve (analytical: sigma_E → 0 at steady state)
- SAOS (analytical: two Maxwell modes + permanent plateau)
- Startup shear (analytical or ODE with TST feedback)
- Stress relaxation (analytical or ODE with TST feedback)
- Creep (ODE: implicit gamma_dot solve)
- LAOS (ODE: nonlinear oscillatory response)

Limiting Cases
--------------
- G_E=0, G_D=0: Neo-Hookean elastic solid
- G_P=0, G_E=0: Single Maxwell fluid (VLB)
- G_E=0: Zener/SLS (spring + dashpot)
- G_D=0, G_P=0: Pure vitrimer
- G_D=0: Partial vitrimer (Meng et al. 2019)
- Full: 3-network HVM

References
----------
- Vernerey, Long, & Brighenti (2017). JMPS 107, 1-20.
- Meng, Simon, Niu, McKenna, & Hallinan (2019). Macromolecules 52, 8.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import lazy_import, safe_import_jax
diffrax = lazy_import("diffrax")
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.hvm._base import HVMBase
from rheojax.models.hvm._kernels import (
    hvm_ber_rate_constant,
    hvm_creep_compliance_linear_vec,
    hvm_normal_stress_1,
    hvm_relaxation_modulus_vec,
    hvm_saos_moduli_vec,
    hvm_startup_stress_linear_vec,
    hvm_steady_shear_stress_vec,
    hvm_total_stress_shear,
)
from rheojax.models.hvm._kernels_diffrax import (
    hvm_solve_creep,
    hvm_solve_laos,
    hvm_solve_relaxation,
    hvm_solve_startup,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "hvm_local",
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
    "hvm",
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
class HVMLocal(HVMBase):
    """Local (0D) Hybrid Vitrimer Model.

    A constitutive model for vitrimers combining:
    - Permanent network (P): covalent crosslinks, elastic
    - Exchangeable network (E): vitrimer bonds with TST kinetics
    - Dissociative network (D): physical bonds, Maxwell relaxation

    Parameters
    ----------
    kinetics : {'stress', 'stretch'}, default 'stress'
        TST coupling mechanism for bond exchange rate
    include_damage : bool, default False
        Enable cooperative damage shielding
    include_dissociative : bool, default True
        Include dissociative (D) network

    Examples
    --------
    >>> from rheojax.models import HVMLocal
    >>> model = HVMLocal()
    >>> omega = np.logspace(-2, 2, 50)
    >>> G_prime, G_double_prime = model.predict_saos(omega)

    >>> # Partial vitrimer (Meng 2019)
    >>> model = HVMLocal(include_dissociative=False)

    >>> # With TST stress feedback
    >>> model = HVMLocal(kinetics='stress')
    >>> t = np.linspace(0, 10, 200)
    >>> result = model.simulate_startup(t, gamma_dot=1.0, return_full=True)
    """

    def __init__(
        self,
        kinetics: Literal["stress", "stretch"] = "stress",
        include_damage: bool = False,
        include_dissociative: bool = True,
    ):
        super().__init__(
            kinetics=kinetics,
            include_damage=include_damage,
            include_dissociative=include_dissociative,
        )
        self._setup_parameters()
        self._test_mode = None
        logger.info(
            "HVMLocal initialized",
            extra={
                "kinetics": kinetics,
                "include_damage": include_damage,
                "include_dissociative": include_dissociative,
            },
        )

    # =========================================================================
    # Parameter Helpers
    # =========================================================================

    def _get_params_dict(self) -> dict[str, float]:
        """Get parameters as dict with defaults for optional params."""
        d = self.get_parameter_dict()
        d.setdefault("G_D", 0.0)
        d.setdefault("k_d_D", 1.0)
        d.setdefault("Gamma_0", 0.0)
        d.setdefault("lambda_crit", 10.0)
        return d

    def _get_k_ber_0(self) -> float:
        """Compute zero-stress BER rate from current parameters."""
        return float(hvm_ber_rate_constant(self.nu_0, self.E_a, self.T))

    # =========================================================================
    # Flow Curve (Analytical)
    # =========================================================================

    def predict_flow_curve(
        self, gamma_dot: np.ndarray, return_components: bool = False
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict steady-state flow curve.

        At steady state, mu^E → mu^E_nat, so sigma_E → 0.
        Only the D-network contributes viscous stress.

        Parameters
        ----------
        gamma_dot : array-like
            Shear rate array (1/s)
        return_components : bool, default False
            If True, return dict with subnetwork contributions

        Returns
        -------
        np.ndarray or dict
            Steady-state stress (Pa) or component dict
        """
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        sigma = hvm_steady_shear_stress_vec(
            gamma_dot_jax, self.G_P, self.G_D, self.k_d_D
        )

        if return_components:
            eta_D = self.G_D / jnp.maximum(self.k_d_D, 1e-30)
            sigma_D = eta_D * gamma_dot_jax
            return {
                "stress": np.asarray(sigma),
                "sigma_P": np.zeros_like(np.asarray(gamma_dot)),  # Elastic, not viscous
                "sigma_E": np.zeros_like(np.asarray(gamma_dot)),  # Relaxed at SS
                "sigma_D": np.asarray(sigma_D),
                "eta_eff": np.asarray(sigma / jnp.maximum(gamma_dot_jax, 1e-30)),
            }
        return np.asarray(sigma)

    # =========================================================================
    # SAOS (Analytical)
    # =========================================================================

    def predict_saos(
        self, omega: np.ndarray, return_components: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

        Two Maxwell modes (E, D) plus permanent plateau (P).

        Parameters
        ----------
        omega : array-like
            Angular frequency array (rad/s)
        return_components : bool, default True
            If True, return (G', G''); if False, return |G*|

        Returns
        -------
        tuple of (np.ndarray, np.ndarray) or np.ndarray
            (G', G'') or |G*|
        """
        omega_jax = jnp.asarray(omega, dtype=jnp.float64)
        k_ber_0 = self._get_k_ber_0()

        G_prime, G_double_prime = hvm_saos_moduli_vec(
            omega_jax, self.G_P, self.G_E, self.G_D, k_ber_0, self.k_d_D
        )

        if return_components:
            return np.asarray(G_prime), np.asarray(G_double_prime)
        return np.asarray(jnp.sqrt(G_prime**2 + G_double_prime**2))

    # =========================================================================
    # Startup Shear
    # =========================================================================

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_full: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Simulate startup shear flow.

        Uses analytical solution for constant-rate case, or ODE with
        TST feedback for nonlinear regime.

        Parameters
        ----------
        t : array-like
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_full : bool, default False
            If True, return dict with all trajectories

        Returns
        -------
        np.ndarray or dict
            Stress array or full trajectory dict
        """
        self._gamma_dot_applied = gamma_dot
        t_jax = jnp.asarray(t, dtype=jnp.float64)

        # Use ODE solver for TST feedback
        params = self._get_params_dict()
        assert params is not None
        sol = hvm_solve_startup(
            t_jax, gamma_dot, params,
            kinetics=self._kinetics,
            include_damage=self._include_damage,
            include_dissociative=self._include_dissociative,
        )

        ys = sol.ys  # (n_times, 11)
        assert ys is not None

        # Compute stress from state
        stress = jax.vmap(
            lambda y: hvm_total_stress_shear(
                y[9], y[2], y[5], y[8],
                params["G_P"], params["G_E"],
                params.get("G_D", 0.0), y[10],
            )
        )(ys)

        # Handle solver failure
        stress = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            stress,
            jnp.nan * jnp.ones_like(stress),
        )

        if return_full:
            return {
                "time": np.asarray(t),
                "stress": np.asarray(stress),
                "strain": np.asarray(ys[:, 9]),
                "mu_E_xx": np.asarray(ys[:, 0]),
                "mu_E_yy": np.asarray(ys[:, 1]),
                "mu_E_xy": np.asarray(ys[:, 2]),
                "mu_E_nat_xx": np.asarray(ys[:, 3]),
                "mu_E_nat_yy": np.asarray(ys[:, 4]),
                "mu_E_nat_xy": np.asarray(ys[:, 5]),
                "mu_D_xx": np.asarray(ys[:, 6]),
                "mu_D_yy": np.asarray(ys[:, 7]),
                "mu_D_xy": np.asarray(ys[:, 8]),
                "damage": np.asarray(ys[:, 10]),
                "N1": np.asarray(
                    jax.vmap(
                        lambda y: hvm_normal_stress_1(
                            y[0], y[1], y[3], y[4], y[6], y[7],
                            params["G_P"], params["G_E"],
                            params.get("G_D", 0.0),
                        )
                    )(ys)
                ),
            }
        return np.asarray(stress)

    # =========================================================================
    # Stress Relaxation
    # =========================================================================

    def simulate_relaxation(
        self,
        t: np.ndarray,
        gamma_step: float = 1.0,
        return_full: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Simulate stress relaxation after step strain.

        Parameters
        ----------
        t : array-like
            Time array after step (s)
        gamma_step : float, default 1.0
            Applied step strain
        return_full : bool, default False
            If True, return full trajectory dict

        Returns
        -------
        np.ndarray or dict
            G(t) relaxation modulus or trajectory dict
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        params = self._get_params_dict()
        assert params is not None

        sol = hvm_solve_relaxation(
            t_jax, gamma_step, params,
            kinetics=self._kinetics,
            include_damage=self._include_damage,
            include_dissociative=self._include_dissociative,
        )

        ys = sol.ys
        assert ys is not None

        # G(t) = sigma(t) / gamma_step
        stress = jax.vmap(
            lambda y: hvm_total_stress_shear(
                y[9], y[2], y[5], y[8],
                params["G_P"], params["G_E"],
                params.get("G_D", 0.0), y[10],
            )
        )(ys)

        G_t = stress / jnp.maximum(jnp.abs(gamma_step), 1e-30)
        G_t = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            G_t, jnp.nan * jnp.ones_like(G_t),
        )

        if return_full:
            return {
                "time": np.asarray(t),
                "G_t": np.asarray(G_t),
                "stress": np.asarray(stress),
                "mu_E_xy": np.asarray(ys[:, 2]),
                "mu_E_nat_xy": np.asarray(ys[:, 5]),
                "mu_D_xy": np.asarray(ys[:, 8]),
                "damage": np.asarray(ys[:, 10]),
            }
        return np.asarray(G_t)

    # =========================================================================
    # Creep
    # =========================================================================

    def simulate_creep(
        self,
        t: np.ndarray,
        sigma_0: float,
        return_full: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Simulate creep under constant stress.

        Parameters
        ----------
        t : array-like
            Time array (s)
        sigma_0 : float
            Applied constant stress (Pa)
        return_full : bool, default False
            If True, return full trajectory dict

        Returns
        -------
        np.ndarray or dict
            Strain gamma(t) or trajectory dict
        """
        self._sigma_applied = sigma_0
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        params = self._get_params_dict()
        assert params is not None

        sol = hvm_solve_creep(
            t_jax, sigma_0, params,
            kinetics=self._kinetics,
            include_damage=self._include_damage,
            include_dissociative=self._include_dissociative,
        )

        ys = sol.ys
        assert ys is not None
        gamma = ys[:, 9]
        gamma = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            gamma, jnp.nan * jnp.ones_like(gamma),
        )

        if return_full:
            J_t = gamma / jnp.maximum(jnp.abs(sigma_0), 1e-30)
            return {
                "time": np.asarray(t),
                "strain": np.asarray(gamma),
                "compliance": np.asarray(J_t),
                "mu_E_xy": np.asarray(ys[:, 2]),
                "mu_E_nat_xy": np.asarray(ys[:, 5]),
                "mu_D_xy": np.asarray(ys[:, 8]),
                "damage": np.asarray(ys[:, 10]),
            }
        return np.asarray(gamma)

    # =========================================================================
    # LAOS
    # =========================================================================

    def simulate_laos(
        self,
        t: np.ndarray,
        gamma_0: float,
        omega: float,
    ) -> dict[str, np.ndarray]:
        """Simulate LAOS (Large Amplitude Oscillatory Shear).

        Parameters
        ----------
        t : array-like
            Time array (s)
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        dict
            Keys: time, strain, stress, gamma_dot, N1,
            mu_E_xy, mu_E_nat_xy, mu_D_xy, damage
        """
        self._gamma_0 = gamma_0
        self._omega_laos = omega
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        params = self._get_params_dict()
        assert params is not None

        sol = hvm_solve_laos(
            t_jax, gamma_0, omega, params,
            kinetics=self._kinetics,
            include_damage=self._include_damage,
            include_dissociative=self._include_dissociative,
        )

        ys = sol.ys
        assert ys is not None
        strain = gamma_0 * jnp.sin(omega * t_jax)
        gamma_dot_arr = gamma_0 * omega * jnp.cos(omega * t_jax)

        stress = jax.vmap(
            lambda y: hvm_total_stress_shear(
                y[9], y[2], y[5], y[8],
                params["G_P"], params["G_E"],
                params.get("G_D", 0.0), y[10],
            )
        )(ys)

        N1 = jax.vmap(
            lambda y: hvm_normal_stress_1(
                y[0], y[1], y[3], y[4], y[6], y[7],
                params["G_P"], params["G_E"],
                params.get("G_D", 0.0),
            )
        )(ys)

        # Handle solver failure
        failed = sol.result != diffrax.RESULTS.successful
        stress = jnp.where(failed, jnp.nan, stress)
        N1 = jnp.where(failed, jnp.nan, N1)

        return {
            "time": np.asarray(t),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "gamma_dot": np.asarray(gamma_dot_arr),
            "N1": np.asarray(N1),
            "mu_E_xy": np.asarray(ys[:, 2]),
            "mu_E_nat_xy": np.asarray(ys[:, 5]),
            "mu_D_xy": np.asarray(ys[:, 8]),
            "damage": np.asarray(ys[:, 10]),
        }

    # =========================================================================
    # Normal Stresses
    # =========================================================================

    def predict_normal_stresses(
        self, gamma_dot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict steady-state normal stress differences.

        At steady state, E-network contributes zero normal stress
        (mu^E → mu^E_nat). Only D-network contributes N1.

        N1 = 2 * G_D * (gamma_dot / k_d_D)^2
        N2 = 0 (upper-convected Maxwell)

        Parameters
        ----------
        gamma_dot : array-like
            Shear rate array (1/s)

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (N1, N2) arrays (Pa)
        """
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        Wi_D = gamma_dot_jax / jnp.maximum(self.k_d_D, 1e-30)
        N1 = 2.0 * self.G_D * Wi_D**2
        N2 = jnp.zeros_like(N1)
        return np.asarray(N1), np.asarray(N2)

    # =========================================================================
    # LAOS Harmonic Extraction
    # =========================================================================

    def extract_laos_harmonics(
        self, laos_result: dict[str, np.ndarray], n_harmonics: int = 5
    ) -> dict[str, np.ndarray]:
        """Extract Fourier harmonics from LAOS simulation.

        Parameters
        ----------
        laos_result : dict
            Output from simulate_laos()
        n_harmonics : int, default 5
            Number of harmonics to extract

        Returns
        -------
        dict
            Keys: harmonic_index (1, 3, 5, ...), sigma_harmonics,
            N1_harmonics
        """
        t = laos_result["time"]
        stress = laos_result["stress"]
        omega = self._omega_laos or 1.0

        # Use last complete cycle(s) for FFT
        period = 2.0 * np.pi / omega
        n_periods = int((t[-1] - t[0]) / period)
        if n_periods < 1:
            n_periods = 1
        t_start = t[-1] - n_periods * period
        mask = t >= t_start

        t_cycle = t[mask]
        stress_cycle = stress[mask]

        # FFT
        n_pts = len(t_cycle)
        dt = (t_cycle[-1] - t_cycle[0]) / max(n_pts - 1, 1)
        freqs = np.fft.rfftfreq(n_pts, d=dt)
        fft_stress = np.fft.rfft(stress_cycle)

        # Extract odd harmonics (1, 3, 5, ...)
        f_fundamental = omega / (2.0 * np.pi)
        harmonics_idx = np.arange(1, 2 * n_harmonics, 2)
        amplitudes = np.zeros(n_harmonics)

        for i, n in enumerate(harmonics_idx):
            target_freq = n * f_fundamental
            idx = np.argmin(np.abs(freqs - target_freq))
            amplitudes[i] = 2.0 * np.abs(fft_stress[idx]) / n_pts

        return {
            "harmonic_index": harmonics_idx,
            "sigma_harmonics": amplitudes,
        }

    # =========================================================================
    # Fitting (NLSQ)
    # =========================================================================

    def _fit(self, x, y, **kwargs):
        """Fit model to data using protocol-aware optimization.

        Parameters
        ----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        **kwargs
            test_mode, gamma_dot, sigma_applied, gamma_0, omega, etc.

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

        # Filter out fitting-specific and BaseModel kwargs
        fwd_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in (
                "test_mode", "deformation_mode", "poisson_ratio",
                "use_log_residuals", "use_jax", "method",
                "max_iter", "use_multi_start", "n_starts", "perturb_factor",
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
            f"Fitted HVMLocal: G_P={self.G_P:.2e}, G_E={self.G_E:.2e}, "
            f"G_D={self.G_D:.2e}"
        )

        return self

    def _predict(self, X, **kwargs):
        """Predict response using fitted parameters.

        Parameters
        ----------
        X : array-like
            Independent variable
        **kwargs
            test_mode and protocol-specific parameters

        Returns
        -------
        np.ndarray
            Predicted response
        """
        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        param_values = jnp.array(
            [self.parameters.get_value(n) for n in self.parameters.keys()],
            dtype=jnp.float64,
        )
        fwd_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("test_mode", "deformation_mode", "poisson_ratio")
        }
        return np.asarray(
            self.model_function(X, param_values, test_mode=test_mode, **fwd_kwargs)
        )

    # =========================================================================
    # Model Function (NLSQ / NumPyro)
    # =========================================================================

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function for HVM.

        Routes to appropriate JAX-traceable prediction based on test_mode.
        Required by BayesianMixin for NumPyro NUTS sampling.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values in ParameterSet order
        test_mode : str, optional
            Override stored test mode
        **kwargs
            Protocol-specific: gamma_dot, sigma_applied, gamma_0, omega

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        # Unpack parameters by position
        p_names = list(self.parameters.keys())
        p_dict = dict(zip(p_names, params, strict=True))

        G_P = p_dict["G_P"]
        G_E = p_dict["G_E"]
        nu_0 = p_dict["nu_0"]
        E_a = p_dict["E_a"]
        V_act = p_dict["V_act"]
        T = p_dict["T"]
        G_D = p_dict.get("G_D", 0.0)
        k_d_D = p_dict.get("k_d_D", 1.0)

        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        gamma_dot = kwargs.get("gamma_dot", self._gamma_dot_applied)
        sigma_applied = kwargs.get("sigma_applied", self._sigma_applied)
        gamma_0 = kwargs.get("gamma_0", self._gamma_0)
        omega = kwargs.get("omega", self._omega_laos)

        k_ber_0 = hvm_ber_rate_constant(nu_0, E_a, T)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return hvm_steady_shear_stress_vec(X_jax, G_P, G_D, k_d_D)

        elif mode == "oscillation":
            G_prime, G_double_prime = hvm_saos_moduli_vec(
                X_jax, G_P, G_E, G_D, k_ber_0, k_d_D
            )
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return hvm_startup_stress_linear_vec(
                X_jax, gamma_dot, G_P, G_E, G_D, k_ber_0, k_d_D
            )

        elif mode == "relaxation":
            D_val = 0.0  # No damage in linear model_function
            return hvm_relaxation_modulus_vec(
                X_jax, G_P, G_E, G_D, k_ber_0, k_d_D, D_val
            )

        elif mode == "creep":
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            J = hvm_creep_compliance_linear_vec(
                X_jax, G_P, G_E, G_D, k_ber_0, k_d_D
            )
            return sigma_applied * J

        elif mode == "laos":
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            # Use ODE solver for LAOS (cannot use analytical)
            params_dict = {
                "G_P": G_P, "G_E": G_E, "G_D": G_D, "k_d_D": k_d_D,
                "nu_0": nu_0, "E_a": E_a, "V_act": V_act, "T": T,
                "Gamma_0": 0.0, "lambda_crit": 10.0,
            }
            sol = hvm_solve_laos(
                X_jax, gamma_0, omega, params_dict,
                kinetics=self._kinetics,
                include_damage=False,
                include_dissociative=self._include_dissociative,
            )
            stress = jax.vmap(
                lambda y: hvm_total_stress_shear(
                    y[9], y[2], y[5], y[8], G_P, G_E, G_D, y[10],
                )
            )(sol.ys)
            return stress

        else:
            logger.warning(f"Unknown test_mode '{mode}', defaulting to flow_curve")
            return hvm_steady_shear_stress_vec(X_jax, G_P, G_D, k_d_D)

    # =========================================================================
    # Factory Methods (Limiting Cases)
    # =========================================================================

    @classmethod
    def neo_hookean(cls, G_P: float = 1e4) -> HVMLocal:
        """Create neo-Hookean elastic solid (G_E=0, G_D=0).

        Parameters
        ----------
        G_P : float
            Permanent network modulus (Pa)

        Returns
        -------
        HVMLocal
            Model with only P-network active
        """
        model = cls(include_dissociative=False)
        model.parameters.set_value("G_P", G_P)
        model.parameters.set_value("G_E", 0.0)
        return model

    @classmethod
    def maxwell(cls, G_D: float = 1e4, k_d_D: float = 1.0) -> HVMLocal:
        """Create single Maxwell fluid (G_P=0, G_E=0).

        Parameters
        ----------
        G_D : float
            Network modulus (Pa)
        k_d_D : float
            Dissociation rate (1/s)

        Returns
        -------
        HVMLocal
            Model with only D-network active
        """
        model = cls(include_dissociative=True)
        model.parameters.set_value("G_P", 0.0)
        model.parameters.set_value("G_E", 0.0)
        model.parameters.set_value("G_D", G_D)
        model.parameters.set_value("k_d_D", k_d_D)
        return model

    @classmethod
    def zener(
        cls, G_P: float = 1e4, G_D: float = 1e4, k_d_D: float = 1.0
    ) -> HVMLocal:
        """Create Zener/SLS model (G_E=0).

        Parameters
        ----------
        G_P : float
            Permanent network modulus (Pa)
        G_D : float
            Dissociative network modulus (Pa)
        k_d_D : float
            Dissociation rate (1/s)

        Returns
        -------
        HVMLocal
            Model with P + D networks (no vitrimer exchange)
        """
        model = cls(include_dissociative=True)
        model.parameters.set_value("G_P", G_P)
        model.parameters.set_value("G_E", 0.0)
        model.parameters.set_value("G_D", G_D)
        model.parameters.set_value("k_d_D", k_d_D)
        return model

    @classmethod
    def pure_vitrimer(
        cls,
        G_E: float = 1e4,
        nu_0: float = 1e10,
        E_a: float = 80e3,
        V_act: float = 1e-5,
        T: float = 300.0,
    ) -> HVMLocal:
        """Create pure vitrimer (G_P=0, G_D=0).

        Parameters
        ----------
        G_E : float
            Exchangeable network modulus (Pa)
        nu_0, E_a, V_act, T : float
            TST parameters

        Returns
        -------
        HVMLocal
            Model with only E-network active
        """
        model = cls(include_dissociative=False)
        model.parameters.set_value("G_P", 0.0)
        model.parameters.set_value("G_E", G_E)
        model.parameters.set_value("nu_0", nu_0)
        model.parameters.set_value("E_a", E_a)
        model.parameters.set_value("V_act", V_act)
        model.parameters.set_value("T", T)
        return model

    @classmethod
    def partial_vitrimer(
        cls,
        G_P: float = 1e4,
        G_E: float = 1e4,
        nu_0: float = 1e10,
        E_a: float = 80e3,
        V_act: float = 1e-5,
        T: float = 300.0,
    ) -> HVMLocal:
        """Create partial vitrimer (G_D=0, Meng 2019).

        Parameters
        ----------
        G_P : float
            Permanent network modulus (Pa)
        G_E : float
            Exchangeable network modulus (Pa)
        nu_0, E_a, V_act, T : float
            TST parameters

        Returns
        -------
        HVMLocal
            Model with P + E networks (no dissociative bonds)
        """
        model = cls(include_dissociative=False)
        model.parameters.set_value("G_P", G_P)
        model.parameters.set_value("G_E", G_E)
        model.parameters.set_value("nu_0", nu_0)
        model.parameters.set_value("E_a", E_a)
        model.parameters.set_value("V_act", V_act)
        model.parameters.set_value("T", T)
        return model
