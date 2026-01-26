"""Base class for Giesekus nonlinear viscoelastic models.

This module provides `GiesekusBase`, the shared foundation for single-mode
and multi-mode Giesekus implementations.

Key Features
------------
- Parameter management with physical bounds
- Protocol dispatch for 6 experimental modes
- ODE integration infrastructure via diffrax
- Diagnostic methods for model validation

The Giesekus Constitutive Equation
----------------------------------
The model relates stress τ to deformation rate D through::

    τ + λ∇̂τ + (αλ/η_p)τ·τ = 2η_p D

where:
- λ is the relaxation time (s)
- α is the mobility factor (0 ≤ α ≤ 0.5)
- η_p is the polymer viscosity (Pa·s)
- ∇̂ denotes the upper-convected derivative

Physical Interpretation
-----------------------
- α = 0: Recovers Upper-Convected Maxwell (no shear-thinning)
- α > 0: Introduces anisotropic mobility → shear-thinning
- α = 0.5: Maximum physical anisotropy

References
----------
- Giesekus, H. (1982). J. Non-Newtonian Fluid Mech. 11, 69-109.
- Bird, R.B. et al. (1987). Dynamics of Polymeric Liquids, Vol. 1.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Literal

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)

# Protocol type alias
GiesekusProtocol = Literal[
    "flow_curve",
    "oscillation",
    "startup",
    "relaxation",
    "creep",
    "laos",
]


class GiesekusBase(BaseModel):
    """Base class for Giesekus viscoelastic models.

    Implements shared parameter management, protocol dispatch, and
    utility methods for single-mode and multi-mode variants.

    The Giesekus model is a nonlinear differential constitutive equation
    that extends the Upper-Convected Maxwell model with a quadratic
    stress term representing anisotropic molecular mobility.

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters for fitting
    _test_mode : str or None
        Current test mode/protocol
    _trajectory : dict or None
        Internal storage for ODE trajectories (debugging)

    Notes
    -----
    The mobility factor α controls shear-thinning behavior:

    - α = 0: No shear-thinning (UCM limit)
    - α = 0.5: Maximum shear-thinning

    The diagnostic ratio N₂/N₁ = -α/2 provides a direct experimental
    route to determine α from normal stress measurements.
    """

    def __init__(self):
        """Initialize Giesekus base model."""
        super().__init__()

        self._setup_parameters()

        # Internal storage for trajectories
        self._trajectory: dict[str, np.ndarray] | None = None

        # Protocol-specific inputs
        self._gamma_dot_applied: float | None = None
        self._sigma_applied: float | None = None
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None

    def _setup_parameters(self):
        """Initialize ParameterSet with Giesekus parameters.

        The Giesekus model has 4 parameters:

        1. η_p (eta_p): Polymer viscosity
           - Units: Pa·s
           - Typical range: 10 - 10,000 Pa·s for polymer melts
           - Physical meaning: Zero-shear polymer contribution to viscosity

        2. λ (lambda_1): Relaxation time
           - Units: s
           - Typical range: 0.01 - 100 s
           - Physical meaning: Characteristic time for stress relaxation

        3. α (alpha): Mobility factor
           - Units: dimensionless
           - Range: 0 ≤ α ≤ 0.5
           - Physical meaning: Degree of anisotropic molecular drag

        4. η_s (eta_s): Solvent viscosity
           - Units: Pa·s
           - Typical range: 0 - 100 Pa·s
           - Physical meaning: Newtonian solvent contribution
        """
        self.parameters = ParameterSet()

        # Polymer viscosity
        self.parameters.add(
            name="eta_p",
            value=100.0,
            bounds=(1e-3, 1e6),
            units="Pa·s",
            description="Polymer viscosity (zero-shear polymer contribution)",
        )

        # Relaxation time
        self.parameters.add(
            name="lambda_1",
            value=1.0,
            bounds=(1e-6, 1e4),
            units="s",
            description="Relaxation time",
        )

        # Mobility factor
        self.parameters.add(
            name="alpha",
            value=0.3,
            bounds=(0.0, 0.5),
            units="dimensionless",
            description="Mobility factor (0=UCM, 0.5=max anisotropy)",
        )

        # Solvent viscosity
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e4),
            units="Pa·s",
            description="Solvent viscosity (Newtonian contribution)",
        )

    # =========================================================================
    # Property Accessors
    # =========================================================================

    @property
    def eta_p(self) -> float:
        """Get polymer viscosity η_p (Pa·s)."""
        return float(self.parameters.get_value("eta_p"))

    @property
    def lambda_1(self) -> float:
        """Get relaxation time λ (s)."""
        return float(self.parameters.get_value("lambda_1"))

    @property
    def alpha(self) -> float:
        """Get mobility factor α (dimensionless)."""
        return float(self.parameters.get_value("alpha"))

    @property
    def eta_s(self) -> float:
        """Get solvent viscosity η_s (Pa·s)."""
        return float(self.parameters.get_value("eta_s"))

    @property
    def eta_0(self) -> float:
        """Get zero-shear viscosity η₀ = η_p + η_s (Pa·s)."""
        return self.eta_p + self.eta_s

    @property
    def G(self) -> float:
        """Get elastic modulus G = η_p/λ (Pa)."""
        return self.eta_p / self.lambda_1

    @property
    def relaxation_time(self) -> float:
        """Get relaxation time λ (s). Alias for lambda_1."""
        return self.lambda_1

    # =========================================================================
    # Dimensionless Numbers
    # =========================================================================

    def weissenberg_number(self, gamma_dot: float) -> float:
        """Compute Weissenberg number Wi = λ·γ̇.

        The Weissenberg number characterizes the degree of nonlinearity
        in steady shear flow.

        Parameters
        ----------
        gamma_dot : float
            Shear rate (1/s)

        Returns
        -------
        float
            Weissenberg number (dimensionless)
        """
        return self.lambda_1 * abs(gamma_dot)

    def deborah_number(self, omega: float) -> float:
        """Compute Deborah number De = λ·ω.

        The Deborah number characterizes the relative importance of
        elastic vs viscous effects in oscillatory flow.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        float
            Deborah number (dimensionless)
        """
        return self.lambda_1 * omega

    # =========================================================================
    # Diagnostic Methods
    # =========================================================================

    def get_normal_stress_ratio(self) -> float:
        """Get theoretical N₂/N₁ ratio.

        For the Giesekus model, the ratio of second to first normal
        stress difference is exactly::

            N₂/N₁ = -α/2

        This provides a direct experimental route to determine α.

        Returns
        -------
        float
            N₂/N₁ ratio (negative, between -0.25 and 0)
        """
        return -self.alpha / 2.0

    def get_critical_weissenberg(self) -> float:
        """Get characteristic Weissenberg number for onset of shear-thinning.

        The transition from Newtonian to shear-thinning behavior occurs
        roughly when Wi ≈ 1. For the Giesekus model, the precise onset
        depends on α.

        Returns
        -------
        float
            Approximate Wi for shear-thinning onset
        """
        # Onset scales as 1/sqrt(α) for small α
        if self.alpha > 1e-6:
            return 1.0 / jnp.sqrt(self.alpha)
        else:
            return jnp.inf  # No shear-thinning for α=0

    def is_ucm_limit(self, tol: float = 1e-6) -> bool:
        """Check if model is effectively at UCM limit (α ≈ 0).

        Parameters
        ----------
        tol : float, default 1e-6
            Tolerance for α comparison

        Returns
        -------
        bool
            True if α < tol (effectively UCM)
        """
        return self.alpha < tol

    # =========================================================================
    # Parameter Dictionary Methods
    # =========================================================================

    def get_parameter_dict(self) -> dict[str, float]:
        """Get all parameters as a dictionary.

        Returns
        -------
        dict[str, float]
            Dictionary of parameter name → value
        """
        return {
            "eta_p": self.eta_p,
            "lambda_1": self.lambda_1,
            "alpha": self.alpha,
            "eta_s": self.eta_s,
        }

    def set_parameter_dict(self, params: dict[str, float]) -> None:
        """Set parameters from a dictionary.

        Parameters
        ----------
        params : dict[str, float]
            Dictionary of parameter name → value
        """
        for name, value in params.items():
            if name in self.parameters.keys():
                self.parameters.set_value(name, value)

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def initialize_from_saos(
        self,
        omega: np.ndarray,
        G_prime: np.ndarray,
        G_double_prime: np.ndarray,
    ) -> None:
        """Initialize parameters from SAOS data.

        Uses the crossover frequency and modulus to estimate λ and η_p.

        In the linear regime (SAOS), the Giesekus model reduces to Maxwell:
        - Crossover: G' = G'' at ω_c, where ω_c·λ = 1
        - Plateau: G' → G = η_p/λ as ω → ∞

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency array (rad/s)
        G_prime : np.ndarray
            Storage modulus G' (Pa)
        G_double_prime : np.ndarray
            Loss modulus G'' (Pa)
        """
        omega = np.asarray(omega)
        G_prime = np.asarray(G_prime)
        G_double_prime = np.asarray(G_double_prime)

        # Find crossover (G' ≈ G'')
        diff = np.abs(G_prime - G_double_prime)
        crossover_idx = np.argmin(diff)

        omega_c = omega[crossover_idx]
        G_c = (G_prime[crossover_idx] + G_double_prime[crossover_idx]) / 2

        # At crossover: ω_c·λ = 1
        lambda_est = 1.0 / omega_c

        # At crossover: G_c ≈ G/2 = η_p/(2λ)
        eta_p_est = 2.0 * G_c * lambda_est

        # Estimate solvent viscosity from high-frequency G''
        # G'' → η_s·ω as ω → ∞
        if len(omega) > 1:
            high_freq_idx = np.argmax(omega)
            eta_s_est = G_double_prime[high_freq_idx] / omega[high_freq_idx]
            eta_s_est = max(
                0.0, eta_s_est - eta_p_est / (1 + omega[-1] ** 2 * lambda_est**2)
            )
        else:
            eta_s_est = 0.0

        # Set parameters
        self.parameters.set_value("lambda_1", np.clip(lambda_est, 1e-6, 1e4))
        self.parameters.set_value("eta_p", np.clip(eta_p_est, 1e-3, 1e6))
        self.parameters.set_value("eta_s", np.clip(eta_s_est, 0.0, 1e4))

        logger.debug(
            f"SAOS initialization: λ={lambda_est:.3e} s, "
            f"η_p={eta_p_est:.3e} Pa·s, η_s={eta_s_est:.3e} Pa·s"
        )

    def initialize_from_flow_curve(
        self,
        gamma_dot: np.ndarray,
        eta: np.ndarray,
    ) -> None:
        """Initialize parameters from viscosity vs shear rate data.

        Uses the zero-shear plateau and shear-thinning onset to estimate
        parameters.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        eta : np.ndarray
            Viscosity array (Pa·s)
        """
        gamma_dot = np.asarray(gamma_dot)
        eta = np.asarray(eta)

        # Sort by shear rate
        sort_idx = np.argsort(gamma_dot)
        gamma_dot = gamma_dot[sort_idx]
        eta = eta[sort_idx]

        # Zero-shear viscosity from low-rate plateau
        eta_0_est = eta[0]

        # Find shear-thinning onset (where η drops to 90% of η₀)
        thinning_idx = np.where(eta < 0.9 * eta_0_est)[0]
        if len(thinning_idx) > 0:
            gamma_dot_c = gamma_dot[thinning_idx[0]]
            lambda_est = 1.0 / gamma_dot_c
        else:
            # No thinning observed, estimate from highest rate
            lambda_est = 0.1 / gamma_dot[-1]

        # Estimate α from degree of shear-thinning
        # At high Wi: η/η₀ ≈ 1/(α·Wi) for α > 0
        if len(gamma_dot) > 3:
            # Use slope in log-log space at high rates
            log_gd = np.log(gamma_dot[-3:])
            log_eta = np.log(eta[-3:])
            slope = (log_eta[-1] - log_eta[0]) / (log_gd[-1] - log_gd[0])
            # Power-law index n ≈ 1 + slope, and n relates to α
            # For Giesekus: n → 0.5 as Wi → ∞, so α ≈ (1-2n)/2
            n_est = max(0.1, min(1.0, 1.0 + slope))
            alpha_est = np.clip((1.0 - 2.0 * n_est) / 2.0, 0.0, 0.5)
        else:
            alpha_est = 0.3  # Default

        # Set parameters
        self.parameters.set_value("lambda_1", np.clip(lambda_est, 1e-6, 1e4))
        self.parameters.set_value("eta_p", np.clip(eta_0_est, 1e-3, 1e6))
        self.parameters.set_value("alpha", alpha_est)

        logger.debug(
            f"Flow curve initialization: λ={lambda_est:.3e} s, "
            f"η_p={eta_0_est:.3e} Pa·s, α={alpha_est:.3f}"
        )

    # =========================================================================
    # ODE Integration Utilities
    # =========================================================================

    def _get_ode_solver_settings(
        self,
        t_span: tuple[float, float],
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 10000,
    ) -> dict:
        """Get default ODE solver settings for diffrax.

        Parameters
        ----------
        t_span : tuple[float, float]
            Time span (t_start, t_end)
        rtol : float, default 1e-6
            Relative tolerance
        atol : float, default 1e-8
            Absolute tolerance
        max_steps : int, default 10000
            Maximum number of steps

        Returns
        -------
        dict
            Settings dictionary for diffrax solver
        """
        return {
            "t0": t_span[0],
            "t1": t_span[1],
            "rtol": rtol,
            "atol": atol,
            "max_steps": max_steps,
        }

    def _get_initial_stress_state(
        self,
        protocol: str,
        gamma_dot: float | None = None,
    ) -> jnp.ndarray:
        """Get initial stress state for ODE integration.

        Parameters
        ----------
        protocol : str
            Protocol name ('startup', 'relaxation', 'creep', 'laos')
        gamma_dot : float, optional
            Shear rate for relaxation (to set steady-state initial condition)

        Returns
        -------
        jnp.ndarray
            Initial state vector [τ_xx, τ_yy, τ_xy, τ_zz]
        """
        if protocol == "startup" or protocol == "laos":
            # Start from rest
            return jnp.zeros(4, dtype=jnp.float64)

        elif protocol == "relaxation" and gamma_dot is not None:
            # Start from steady state at given shear rate
            from rheojax.models.giesekus._kernels import (
                giesekus_steady_stress_components,
            )

            tau_xx, tau_yy, tau_xy, tau_zz = giesekus_steady_stress_components(
                gamma_dot, self.eta_p, self.lambda_1, self.alpha, self.eta_s
            )
            return jnp.array([tau_xx, tau_yy, tau_xy, tau_zz], dtype=jnp.float64)

        elif protocol == "creep":
            # Start from rest (5-element state with strain)
            return jnp.zeros(5, dtype=jnp.float64)

        else:
            return jnp.zeros(4, dtype=jnp.float64)

    # =========================================================================
    # Abstract Methods (to be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def _fit(self, x, y, **kwargs):
        """Fit model to data. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _predict(self, x, **kwargs):
        """Predict response. Must be implemented by subclasses."""
        raise NotImplementedError

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"η_p={self.eta_p:.2e} Pa·s, "
            f"λ={self.lambda_1:.2e} s, "
            f"α={self.alpha:.3f}, "
            f"η_s={self.eta_s:.2e} Pa·s)"
        )
