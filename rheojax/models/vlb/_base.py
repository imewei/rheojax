"""Base class for VLB (Vernerey-Long-Brighenti) transient network models.

This module provides `VLBBase`, the shared foundation for all VLB model
variants (single-network, multi-network).

Key Physics
-----------
VLB models describe polymers with dynamic crosslinks via a distribution
tensor mu that captures the chain-level state:

- mu evolves via bond kinetics (creation/destruction at rate k_d)
- Stress arises from mu deviating from equilibrium (I)
- The framework has a molecular-statistical foundation

The distribution tensor mu tracks the second moment of the chain
end-to-end vector distribution phi(r,t):

    dmu/dt = k_d*(I - mu) + L·mu + mu·L^T

where k_d is the dissociation rate and L is the velocity gradient.
Stress is computed from mu via sigma = G0*(mu - I) + p*I.

References
----------
- Vernerey, F.J., Long, R. & Brighenti, R. (2017). JMPS 107, 1-20.
"""

from __future__ import annotations

import logging
from abc import abstractmethod

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


class VLBBase(BaseModel):
    """Base class for VLB transient network models.

    Provides shared infrastructure for all VLB variants:

    - Parameter management with physical bounds
    - Protocol dispatch for 6 experimental modes
    - Distribution tensor utilities (equilibrium, stretch)
    - Initialization helpers from SAOS and flow curve data

    The VLB model family uses the distribution tensor mu as the primary
    state variable. At equilibrium, mu = I (identity). Stress is derived
    from mu via sigma = G0*(mu - I).

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
    The distribution tensor mu has the same mathematical structure as the
    conformation tensor S used in TNT models. The distinction is physical:
    mu is derived from the chain distribution function phi(r,t), while S
    is derived from end-to-end vector averaging. For constant k_d, both
    give the Maxwell model.
    """

    def __init__(self):
        """Initialize VLB base model."""
        super().__init__()

        # Internal storage for trajectories
        self._trajectory: dict[str, np.ndarray] | None = None

        # Protocol-specific inputs (stored between fit and predict)
        self._gamma_dot_applied: float | None = None
        self._sigma_applied: float | None = None
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None

    # =========================================================================
    # Distribution Tensor Utilities
    # =========================================================================

    @staticmethod
    def get_equilibrium_distribution() -> jnp.ndarray:
        """Return equilibrium distribution tensor mu_eq = I.

        In the absence of flow, chains adopt their equilibrium
        configuration: mu = I (isotropic, unstretched).

        Returns
        -------
        jnp.ndarray
            Equilibrium state [mu_xx, mu_yy, mu_zz, mu_xy] = [1, 1, 1, 0]
        """
        return jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

    @staticmethod
    def compute_stretch(mu_xx: float, mu_yy: float, mu_zz: float) -> float:
        """Compute stretch ratio from distribution tensor.

        stretch = sqrt(tr(mu)/3)

        At equilibrium (mu=I): stretch = 1.
        stretch > 1 indicates chains are extended beyond equilibrium.

        Parameters
        ----------
        mu_xx, mu_yy, mu_zz : float
            Diagonal distribution tensor components

        Returns
        -------
        float
            Stretch ratio (dimensionless, >= 0)
        """
        tr_mu = mu_xx + mu_yy + mu_zz
        return float(jnp.sqrt(jnp.maximum(tr_mu / 3.0, 0.0)))

    # =========================================================================
    # Dimensionless Numbers
    # =========================================================================

    def weissenberg_number(self, gamma_dot: float) -> float:
        """Compute Weissenberg number Wi = t_R * gamma_dot.

        Parameters
        ----------
        gamma_dot : float
            Shear rate (1/s)

        Returns
        -------
        float
            Weissenberg number (dimensionless)
        """
        val = self.parameters.get_value("k_d")
        k_d = float(val) if val is not None else 1.0
        t_R = 1.0 / k_d
        return t_R * abs(gamma_dot)

    def deborah_number(self, omega: float) -> float:
        """Compute Deborah number De = t_R * omega.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        float
            Deborah number (dimensionless)
        """
        val = self.parameters.get_value("k_d")
        k_d = float(val) if val is not None else 1.0
        t_R = 1.0 / k_d
        return t_R * omega

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
        result: dict[str, float] = {}
        for name in self.parameters.keys():
            val = self.parameters.get_value(name)
            result[name] = float(val) if val is not None else 0.0
        return result

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

        Uses the crossover frequency and modulus to estimate k_d and G0.
        In the linear regime, VLB reduces to Maxwell: crossover at omega_c = k_d.

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

        # Find crossover (G' ~ G'')
        diff = np.abs(G_prime - G_double_prime)
        crossover_idx = np.argmin(diff)

        omega_c = omega[crossover_idx]
        G_c = (G_prime[crossover_idx] + G_double_prime[crossover_idx]) / 2

        # At crossover: omega_c = k_d (for Maxwell)
        k_d_est = omega_c

        # At crossover: G_c ~ G0/2
        G0_est = 2.0 * G_c

        # Estimate solvent viscosity from high-frequency G''
        if len(omega) > 1:
            high_freq_idx = np.argmax(omega)
            t_R = 1.0 / k_d_est
            eta_s_est = G_double_prime[high_freq_idx] / omega[high_freq_idx]
            wt = omega[-1] * t_R
            eta_s_est = max(0.0, eta_s_est - G0_est * t_R / (1 + wt * wt))
        else:
            eta_s_est = 0.0

        # Set parameters
        if "k_d" in self.parameters.keys():
            self.parameters.set_value("k_d", np.clip(k_d_est, 1e-6, 1e6))
        if "G0" in self.parameters.keys():
            self.parameters.set_value("G0", np.clip(G0_est, 1e0, 1e8))
        if "eta_s" in self.parameters.keys():
            self.parameters.set_value("eta_s", np.clip(eta_s_est, 0.0, 1e4))

        logger.debug(
            f"SAOS initialization: k_d={k_d_est:.3e} 1/s, "
            f"G0={G0_est:.3e} Pa, eta_s={eta_s_est:.3e} Pa*s"
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

        # For constant k_d, VLB is Newtonian: eta_0 = G0/k_d
        # Use an arbitrary split: k_d ~ 1, G0 ~ eta_0
        k_d_est = 1.0
        G0_est = eta_0_est * k_d_est

        if "k_d" in self.parameters.keys():
            self.parameters.set_value("k_d", np.clip(k_d_est, 1e-6, 1e6))
        if "G0" in self.parameters.keys():
            self.parameters.set_value("G0", np.clip(G0_est, 1e0, 1e8))

        logger.debug(
            f"Flow curve initialization: k_d={k_d_est:.3e} 1/s, G0={G0_est:.3e} Pa"
        )

    # =========================================================================
    # Virtual Method for Phase 2 Extensions
    # =========================================================================

    def dissociation_rate(self, mu_xx: float, mu_yy: float, mu_zz: float) -> float:
        """Compute dissociation rate (possibly state-dependent).

        Base implementation returns constant k_d. Subclasses (e.g., VLBBell)
        can override this for force-dependent kinetics.

        Parameters
        ----------
        mu_xx, mu_yy, mu_zz : float
            Distribution tensor diagonal components

        Returns
        -------
        float
            Dissociation rate (1/s)
        """
        val = self.parameters.get_value("k_d")
        return float(val) if val is not None else 1.0

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def _setup_parameters(self):
        """Initialize ParameterSet. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _fit(self, X, y, **kwargs):
        """Fit model to data. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _predict(self, X, **kwargs):
        """Predict response. Must be implemented by subclasses."""
        raise NotImplementedError

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""

        def _fmt(name: str) -> str:
            val = self.parameters.get_value(name)
            return f"{name}={float(val) if val is not None else 0.0:.3e}"

        params_str = ", ".join(
            _fmt(name) for name in list(self.parameters.keys())[:4]
        )
        return f"{self.__class__.__name__}({params_str})"
