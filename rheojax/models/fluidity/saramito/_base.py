"""Base class for Fluidity-Saramito Elastoviscoplastic Models.

This module provides `FluiditySaramitoBase`, the shared foundation for
Local (0D) and Nonlocal (1D) Saramito EVP models.

Key Features
------------
- Tensorial stress state: [τ_xx, τ_yy, τ_xy] for normal stress predictions
- Fluidity-dependent relaxation: λ = 1/f
- Optional dynamic yield stress: τ_y(f) = τ_y0 + a_y/f^m
- Herschel-Bulkley plastic flow: σ = τ_y + K*γ̇^n

Coupling Modes
--------------
- "minimal": Only λ = 1/f, τ_y = τ_y0 (constant)
- "full": λ = 1/f + τ_y(f) increases as structure ages

References
----------
- Saramito, P. (2007). JNNFM 145, 1-14.
- Saramito, P. (2009). JNNFM 158, 154-161.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet

# Safe import ensures float64
jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


class FluiditySaramitoBase(BaseModel):
    """Base class for Fluidity-Saramito EVP models.

    Implements shared parameter management, coupling logic, and
    trajectory storage for Local (0D) and Nonlocal (1D) variants.

    The Saramito model describes elastoviscoplastic materials through:
    1. Upper-convected Maxwell viscoelasticity with λ = 1/f
    2. Von Mises yield criterion with Herschel-Bulkley plastic flow
    3. Thixotropic fluidity evolution (aging + rejuvenation)

    Attributes
    ----------
    coupling : {"minimal", "full"}
        Coupling mode for yield stress:
        - "minimal": τ_y = τ_y0 (constant)
        - "full": τ_y(f) = τ_y0 + a_y/f^m (aging yield stress)
    parameters : ParameterSet
        Model parameters for fitting
    """

    def __init__(
        self,
        coupling: Literal["minimal", "full"] = "minimal",
    ):
        """Initialize Fluidity-Saramito Base Model.

        Parameters
        ----------
        coupling : {"minimal", "full"}, default "minimal"
            Coupling mode for yield stress evolution
        """
        super().__init__()

        self.coupling = coupling
        self._setup_parameters()

        # Internal storage for trajectories (for plotting/debugging)
        self._trajectory: dict[str, np.ndarray] | None = None

        # Protocol-specific inputs
        self._gamma_dot_applied: float | None = None
        self._sigma_applied: float | None = None
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None
        self._t_wait: float = 0.0  # Waiting time before test

    def _setup_parameters(self):
        """Initialize ParameterSet with Saramito-fluidity parameters."""
        self.parameters = ParameterSet()

        # =====================================================================
        # Elastic / Viscoelastic Parameters
        # =====================================================================

        self.parameters.add(
            name="G",
            value=1e4,
            bounds=(1e1, 1e8),
            units="Pa",
            description="Elastic modulus",
        )

        # Solvent viscosity - typically zero for concentrated yield-stress fluids
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e3),
            units="Pa·s",
            description="Solvent viscosity (typically zero for yield-stress fluids)",
        )

        # =====================================================================
        # Yield Stress / Plasticity Parameters (Herschel-Bulkley)
        # =====================================================================

        self.parameters.add(
            name="tau_y0",
            value=100.0,
            bounds=(1e-1, 1e5),
            units="Pa",
            description="Base yield stress (Von Mises threshold)",
        )

        self.parameters.add(
            name="K_HB",
            value=50.0,
            bounds=(1e-2, 1e5),
            units="Pa·s^n",
            description="Herschel-Bulkley consistency index",
        )

        self.parameters.add(
            name="n_HB",
            value=0.5,
            bounds=(0.1, 1.5),
            units="dimensionless",
            description="Herschel-Bulkley flow exponent (n < 1: shear-thinning)",
        )

        # =====================================================================
        # Fluidity Dynamics Parameters
        # =====================================================================

        self.parameters.add(
            name="f_age",
            value=1e-6,
            bounds=(1e-12, 1e-2),
            units="1/(Pa·s)",
            description="Aging fluidity limit (equilibrium at rest)",
        )

        self.parameters.add(
            name="f_flow",
            value=1e-2,
            bounds=(1e-6, 1.0),
            units="1/(Pa·s)",
            description="Flow fluidity limit (rejuvenation at high rates)",
        )

        self.parameters.add(
            name="t_a",
            value=10.0,
            bounds=(1e-2, 1e5),
            units="s",
            description="Aging timescale (structural build-up)",
        )

        self.parameters.add(
            name="b",
            value=1.0,
            bounds=(0.0, 1e3),
            units="dimensionless",
            description="Rejuvenation amplitude",
        )

        self.parameters.add(
            name="n_rej",
            value=1.0,
            bounds=(0.1, 3.0),
            units="dimensionless",
            description="Rejuvenation rate exponent",
        )

        # =====================================================================
        # Full Coupling Parameters (only active when coupling="full")
        # =====================================================================

        if self.coupling == "full":
            self.parameters.add(
                name="tau_y_coupling",
                value=1.0,
                bounds=(0.0, 1e4),
                units="Pa·(Pa·s)^m",
                description="Yield stress fluidity coupling coefficient",
            )

            self.parameters.add(
                name="m_yield",
                value=0.5,
                bounds=(0.1, 2.0),
                units="dimensionless",
                description="Yield stress fluidity exponent",
            )

    def get_initial_fluidity(self, t_wait: float = 0.0) -> float:
        """Get initial fluidity value based on waiting time.

        For aged samples (t_wait >> t_a): f → f_age
        For fresh samples (t_wait = 0): f can start at f_flow

        Parameters
        ----------
        t_wait : float, default 0.0
            Waiting time before measurement (s)

        Returns
        -------
        float
            Initial fluidity f_0 (1/(Pa·s))
        """
        f_age = self.parameters.get_value("f_age")
        f_flow = self.parameters.get_value("f_flow")
        t_a = self.parameters.get_value("t_a")

        if f_age is None or f_flow is None or t_a is None:
            return 1e-6

        # Exponential aging from f_flow toward f_age
        # f(t_wait) = f_age + (f_flow - f_age) * exp(-t_wait/t_a)
        if t_wait > 0:
            f_init = f_age + (f_flow - f_age) * np.exp(-t_wait / t_a)
        else:
            # Fresh sample starts between f_age and f_flow
            # Default: start at f_age (well-aged)
            f_init = f_age

        return float(f_init)

    def get_parameter_dict(self) -> dict:
        """Get all parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary of parameter name → value
        """
        return {k: self.parameters.get_value(k) for k in self.parameters.keys()}

    def get_effective_yield_stress(self, f: float) -> float:
        """Get effective yield stress based on coupling mode.

        Parameters
        ----------
        f : float
            Current fluidity (1/(Pa·s))

        Returns
        -------
        float
            Effective yield stress τ_y(f) (Pa)
        """
        tau_y0 = self.parameters.get_value("tau_y0")

        if self.coupling == "full":
            tau_y_coupling = self.parameters.get_value("tau_y_coupling")
            m_yield = self.parameters.get_value("m_yield")

            if tau_y_coupling is None or m_yield is None:
                return float(tau_y0)

            f_safe = max(f, 1e-20)
            tau_y = tau_y0 + tau_y_coupling / (f_safe**m_yield)
            return float(tau_y)
        else:
            return float(tau_y0)

    def _get_saramito_ode_args(self, params: dict | None = None) -> dict:
        """Build args dictionary for ODE integration.

        Parameters
        ----------
        params : dict, optional
            Parameter dictionary. If None, uses stored values.

        Returns
        -------
        dict
            Dictionary with all parameters needed for ODE RHS functions.
        """
        if params is None:
            params = self.get_parameter_dict()

        args = {
            # Elastic
            "G": params["G"],
            "eta_s": params.get("eta_s", 0.0),
            # Yield/plastic
            "tau_y0": params["tau_y0"],
            "K_HB": params["K_HB"],
            "n_HB": params["n_HB"],
            # Fluidity
            "f_age": params["f_age"],
            "f_flow": params["f_flow"],
            "t_a": params["t_a"],
            "b": params["b"],
            "n_rej": params["n_rej"],
            # Coupling mode (int for JAX compatibility: 0=minimal, 1=full)
            "coupling_mode": 1 if self.coupling == "full" else 0,
        }

        # Add full coupling parameters if active
        if self.coupling == "full":
            args["tau_y_coupling"] = params.get("tau_y_coupling", 0.0)
            args["m_yield"] = params.get("m_yield", 1.0)

        return args

    def initialize_from_flow_curve(
        self,
        gamma_dot: np.ndarray,
        sigma: np.ndarray,
    ) -> None:
        """Initialize parameters from flow curve data.

        Smart initialization strategy:
        1. τ_y0 from low-rate plateau (stress intercept)
        2. K_HB, n_HB from high-rate power-law region
        3. G estimated from stress overshoot if available
        4. Fluidity parameters from viscosity scaling

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate data (1/s)
        sigma : np.ndarray
            Shear stress data (Pa)
        """
        gamma_dot = np.asarray(gamma_dot)
        sigma = np.asarray(sigma)

        # Sort by shear rate
        sort_idx = np.argsort(gamma_dot)
        gamma_dot = gamma_dot[sort_idx]
        sigma = sigma[sort_idx]

        # 1. Estimate yield stress from low-rate extrapolation
        # Use lowest 3-5 points for linear extrapolation
        n_low = min(5, len(gamma_dot) // 3)
        if n_low >= 2:
            # Linear fit to low-rate region
            coeffs = np.polyfit(gamma_dot[:n_low], sigma[:n_low], 1)
            tau_y_est = max(coeffs[1], sigma.min() * 0.9)  # Intercept
        else:
            tau_y_est = sigma.min() * 0.9

        self.parameters.set_value("tau_y0", tau_y_est)
        logger.debug(f"Initialized tau_y0 = {tau_y_est:.2f} Pa")

        # 2. Fit HB parameters to high-rate region
        # σ - τ_y = K * γ̇^n → log(σ - τ_y) = log(K) + n*log(γ̇)
        sigma_excess = sigma - tau_y_est
        valid_mask = sigma_excess > 0.1 * tau_y_est  # Above noise floor

        if valid_mask.sum() >= 3:
            log_gd = np.log(gamma_dot[valid_mask])
            log_sigma_ex = np.log(sigma_excess[valid_mask])

            # Linear fit in log-log space
            coeffs_hb = np.polyfit(log_gd, log_sigma_ex, 1)
            n_est = coeffs_hb[0]
            K_est = np.exp(coeffs_hb[1])

            # Constrain to reasonable range
            n_est = np.clip(n_est, 0.1, 1.5)
            K_est = np.clip(K_est, 1e-2, 1e5)

            self.parameters.set_value("n_HB", n_est)
            self.parameters.set_value("K_HB", K_est)
            logger.debug(f"Initialized n_HB = {n_est:.3f}, K_HB = {K_est:.2f}")

        # 3. Estimate elastic modulus from high-rate viscosity
        # At high rates: η ≈ K*γ̇^(n-1), and G ≈ η/λ with λ ~ t_a
        eta_high = sigma[-1] / gamma_dot[-1] if gamma_dot[-1] > 0 else 1e3
        t_a_default = self.parameters.get_value("t_a")
        G_est = eta_high / t_a_default if t_a_default else 1e4
        G_est = np.clip(G_est, 1e1, 1e8)

        self.parameters.set_value("G", G_est)
        logger.debug(f"Initialized G = {G_est:.2e} Pa")

        # 4. Estimate fluidity parameters
        # f_flow ≈ γ̇_max / σ_max (high-rate fluidity)
        f_flow_est = gamma_dot[-1] / sigma[-1] if sigma[-1] > 0 else 1e-2
        f_flow_est = np.clip(f_flow_est, 1e-6, 1.0)
        self.parameters.set_value("f_flow", f_flow_est)

        # f_age << f_flow (typically 3-4 orders of magnitude lower)
        f_age_est = f_flow_est * 1e-4
        f_age_est = np.clip(f_age_est, 1e-12, 1e-2)
        self.parameters.set_value("f_age", f_age_est)

        logger.debug(f"Initialized f_age = {f_age_est:.2e}, f_flow = {f_flow_est:.2e}")

    def predict_normal_stresses(
        self,
        gamma_dot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict first and second normal stress differences.

        The Saramito model predicts non-zero N₁ from its
        upper-convected Maxwell foundation.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        N1 : np.ndarray
            First normal stress difference τ_xx - τ_yy (Pa)
        N2 : np.ndarray
            Second normal stress difference τ_yy - τ_zz (Pa)
            Note: N2 = 0 for simple UCM models
        """
        from rheojax.models.fluidity.saramito._kernels import saramito_steady_state_full

        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        p = self.get_parameter_dict()

        tau_xy, tau_xx, N1 = saramito_steady_state_full(
            gamma_dot_jax,
            p["G"],
            p["tau_y0"],
            p["K_HB"],
            p["n_HB"],
            p["f_age"],
            p["f_flow"],
            p["t_a"],
            p["b"],
            p["n_rej"],
        )

        # N2 = 0 for upper-convected Maxwell
        N2 = np.zeros_like(gamma_dot)

        return np.array(N1), N2

    @property
    def relaxation_time(self) -> float:
        """Get characteristic relaxation time λ = 1/(G*f_age).

        This is the relaxation time in the aged (rest) state.

        Returns
        -------
        float
            Relaxation time (s)
        """
        G = self.parameters.get_value("G")
        f_age = self.parameters.get_value("f_age")

        if G is None or f_age is None:
            return 1.0

        return 1.0 / (G * f_age)

    @property
    def deborah_number(self) -> float | None:
        """Get Deborah number De = λ * ω (for oscillatory tests).

        Returns
        -------
        float or None
            Deborah number if omega is set, else None
        """
        if self._omega_laos is None:
            return None

        return self.relaxation_time * self._omega_laos

    @property
    def weissenberg_number(self) -> float | None:
        """Get Weissenberg number Wi = λ * γ̇ (for steady/startup tests).

        Returns
        -------
        float or None
            Weissenberg number if gamma_dot is set, else None
        """
        if self._gamma_dot_applied is None:
            return None

        return self.relaxation_time * self._gamma_dot_applied

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"coupling='{self.coupling}', "
            f"G={self.parameters.get_value('G'):.2e} Pa, "
            f"tau_y0={self.parameters.get_value('tau_y0'):.2f} Pa)"
        )
