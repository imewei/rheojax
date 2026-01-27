"""Base class for Transient Network Theory (TNT) models.

This module provides `TNTBase`, the shared foundation for all TNT model
variants (single-mode, loop-bridge, sticky Rouse, Cates, multi-species).

Key Physics
-----------
TNT models describe polymer networks with reversible (physical) crosslinks:

- Chains form temporary junctions with lifetime τ_b
- Stress arises from chain stretch between active junctions
- Network evolves via creation (attachment) and destruction (detachment)

The conformation tensor S tracks the average chain configuration:

    dS/dt = L·S + S·L^T + g₀·I - β(S)·S

where g₀ is the creation rate and β is the destruction (breakage) rate.
Stress is computed from S via σ = G·(S - I) + η_s·γ̇ (linear) or
σ = G·f(S)·(S - I) + η_s·γ̇ (FENE-P).

References
----------
- Green, M.S. & Tobolsky, A.V. (1946). J. Chem. Phys. 14, 80-92.
- Tanaka, F. & Edwards, S.F. (1992). Macromolecules 25, 1516-1523.
- Lodge, A.S. (1956). Trans. Faraday Soc. 52, 120-130.
"""

from __future__ import annotations

import logging
from abc import abstractmethod

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


class TNTBase(BaseModel):
    """Base class for Transient Network Theory models.

    Provides shared infrastructure for all TNT variants:

    - Parameter management with physical bounds
    - Protocol dispatch for 6 experimental modes
    - Conformation tensor utilities (equilibrium, stretch)
    - ODE integration infrastructure via diffrax

    The TNT model family uses the conformation tensor S as the primary
    state variable. At equilibrium, S = I (identity). Stress is derived
    from S via the stress formula (linear or FENE-P).

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters for fitting
    _test_mode : str or None
        Current test mode/protocol
    _trajectory : dict or None
        Internal storage for ODE trajectories

    Notes
    -----
    The conformation tensor S differs from the stress tensor τ used in
    Giesekus models. S is dimensionless with equilibrium S = I, while
    τ has units of Pa with equilibrium τ = 0.
    """

    def __init__(self):
        """Initialize TNT base model."""
        super().__init__()

        # Internal storage for trajectories
        self._trajectory: dict[str, np.ndarray] | None = None

        # Protocol-specific inputs (stored between fit and predict)
        self._gamma_dot_applied: float | None = None
        self._sigma_applied: float | None = None
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None

    # =========================================================================
    # Conformation Tensor Utilities
    # =========================================================================

    @staticmethod
    def get_equilibrium_conformation() -> jnp.ndarray:
        """Return equilibrium conformation tensor S_eq = I.

        In the absence of flow, chains adopt their equilibrium
        configuration: S = I (isotropic, unstretched).

        Returns
        -------
        jnp.ndarray
            Equilibrium state [S_xx, S_yy, S_zz, S_xy] = [1, 1, 1, 0]
        """
        return jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

    @staticmethod
    def compute_stretch(S_xx: float, S_yy: float, S_zz: float) -> float:
        """Compute stretch ratio from conformation tensor.

        stretch = sqrt(tr(S)/3)

        At equilibrium (S=I): stretch = 1.
        stretch > 1 indicates chains are extended beyond equilibrium.

        Parameters
        ----------
        S_xx, S_yy, S_zz : float
            Diagonal conformation tensor components

        Returns
        -------
        float
            Stretch ratio (dimensionless, ≥ 0)
        """
        tr_S = S_xx + S_yy + S_zz
        return float(jnp.sqrt(jnp.maximum(tr_S / 3.0, 0.0)))

    # =========================================================================
    # Dimensionless Numbers
    # =========================================================================

    def weissenberg_number(self, gamma_dot: float) -> float:
        """Compute Weissenberg number Wi = τ_b·γ̇.

        Parameters
        ----------
        gamma_dot : float
            Shear rate (1/s)

        Returns
        -------
        float
            Weissenberg number (dimensionless)
        """
        tau_b = float(self.parameters.get_value("tau_b"))
        return tau_b * abs(gamma_dot)

    def deborah_number(self, omega: float) -> float:
        """Compute Deborah number De = τ_b·ω.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        float
            Deborah number (dimensionless)
        """
        tau_b = float(self.parameters.get_value("tau_b"))
        return tau_b * omega

    # =========================================================================
    # Parameter Dictionary Methods
    # =========================================================================

    def get_parameter_dict(self) -> dict[str, float]:
        """Get all parameters as a dictionary.

        Returns
        -------
        dict[str, float]
            Dictionary of parameter name -> value
        """
        return {
            name: float(self.parameters.get_value(name))
            for name in self.parameters.keys()
        }

    def set_parameter_dict(self, params: dict[str, float]) -> None:
        """Set parameters from a dictionary.

        Parameters
        ----------
        params : dict[str, float]
            Dictionary of parameter name -> value
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

        Uses the crossover frequency and modulus to estimate τ_b and G.
        In the linear regime, TNT reduces to Maxwell: crossover at ω_c·τ_b = 1.

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

        # At crossover: ω_c·τ_b = 1
        tau_b_est = 1.0 / omega_c

        # At crossover: G_c ≈ G/2
        G_est = 2.0 * G_c

        # Estimate solvent viscosity from high-frequency G''
        if len(omega) > 1:
            high_freq_idx = np.argmax(omega)
            eta_s_est = G_double_prime[high_freq_idx] / omega[high_freq_idx]
            # Subtract Maxwell contribution
            wt = omega[-1] * tau_b_est
            eta_s_est = max(0.0, eta_s_est - G_est * tau_b_est / (1 + wt * wt))
        else:
            eta_s_est = 0.0

        # Set parameters
        if "tau_b" in self.parameters.keys():
            self.parameters.set_value("tau_b", np.clip(tau_b_est, 1e-6, 1e4))
        if "G" in self.parameters.keys():
            self.parameters.set_value("G", np.clip(G_est, 1e0, 1e8))
        if "eta_s" in self.parameters.keys():
            self.parameters.set_value("eta_s", np.clip(eta_s_est, 0.0, 1e4))

        logger.debug(
            f"SAOS initialization: τ_b={tau_b_est:.3e} s, "
            f"G={G_est:.3e} Pa, η_s={eta_s_est:.3e} Pa·s"
        )

    def initialize_from_flow_curve(
        self,
        gamma_dot: np.ndarray,
        sigma: np.ndarray,
    ) -> None:
        """Initialize parameters from flow curve data.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        sigma : np.ndarray
            Stress array (Pa)
        """
        gamma_dot = np.asarray(gamma_dot)
        sigma = np.asarray(sigma)

        # Sort by shear rate
        sort_idx = np.argsort(gamma_dot)
        gamma_dot = gamma_dot[sort_idx]
        sigma = sigma[sort_idx]

        # Estimate zero-shear viscosity from low-rate data
        eta_0_est = sigma[0] / max(gamma_dot[0], 1e-10)

        # Estimate τ_b from onset of shear thinning (if present)
        eta = sigma / np.maximum(gamma_dot, 1e-10)
        thinning_idx = np.where(eta < 0.9 * eta[0])[0]
        if len(thinning_idx) > 0:
            gamma_dot_c = gamma_dot[thinning_idx[0]]
            tau_b_est = 1.0 / gamma_dot_c
        else:
            tau_b_est = 1.0 / gamma_dot[-1]

        # G = η₀/τ_b
        G_est = eta_0_est / tau_b_est

        if "tau_b" in self.parameters.keys():
            self.parameters.set_value("tau_b", np.clip(tau_b_est, 1e-6, 1e4))
        if "G" in self.parameters.keys():
            self.parameters.set_value("G", np.clip(G_est, 1e0, 1e8))

        logger.debug(
            f"Flow curve initialization: τ_b={tau_b_est:.3e} s, G={G_est:.3e} Pa"
        )

    # =========================================================================
    # ODE Integration Utilities
    # =========================================================================

    def _get_ode_solver_settings(
        self,
        t_span: tuple[float, float],
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 100_000,
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
        max_steps : int, default 100_000
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

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def _setup_parameters(self):
        """Initialize ParameterSet. Must be implemented by subclasses."""
        raise NotImplementedError

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
        params_str = ", ".join(
            f"{name}={float(self.parameters.get_value(name)):.3e}"
            for name in list(self.parameters.keys())[:4]
        )
        return f"{self.__class__.__name__}({params_str})"
