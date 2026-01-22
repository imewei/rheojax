"""Lattice-based Elasto-Plastic Model (EPM) implementation."""

from functools import partial
from typing import Dict, Tuple, Optional, Any

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.models.epm.base import EPMBase
from rheojax.utils.epm_kernels import (
    make_propagator_q,
    epm_step,
    compute_plastic_strain_rate
)

jax, jnp = safe_import_jax()


@ModelRegistry.register("lattice_epm")
class LatticeEPM(EPMBase):
    """2D Lattice Elasto-Plastic Model (EPM).

    A mesoscopic model for amorphous solids (glasses, gels) that explicitly resolves
    spatial heterogeneity, plastic avalanches, and stress redistribution.

    Physics:
        - Lattice of elastoplastic blocks.
        - Elastic loading (affine).
        - Local yielding when stress > threshold.
        - Long-range stress redistribution via quadrupolar Eshelby propagator.
        - Structural renewal (disorder).

    Parameters:
        mu (float): Shear modulus. Default 1.0.
        tau_pl (float): Plastic relaxation timescale. Default 1.0.
        sigma_c_mean (float): Mean yield threshold. Default 1.0.
        sigma_c_std (float): Disorder strength (std dev of thresholds). Default 0.1.
        smoothing_width (float): Width for smooth yielding approx (inference only). Default 0.1.

    Configuration:
        L (int): Lattice size (LxL). Default 64.
        dt (float): Time step. Default 0.01.
    """

    def __init__(
        self,
        L: int = 64,
        dt: float = 0.01,
        mu: float = 1.0,
        tau_pl: float = 1.0,
        sigma_c_mean: float = 1.0,
        sigma_c_std: float = 0.1,
    ):
        """Initialize the Lattice EPM."""
        # Initialize base class with common parameters
        super().__init__(
            L=L,
            dt=dt,
            mu=mu,
            tau_pl=tau_pl,
            sigma_c_mean=sigma_c_mean,
            sigma_c_std=sigma_c_std,
        )

        # Precompute Propagator (Cached)
        # Using 1.0 as shear_modulus here, will scale by mu during execution
        self._propagator_q_norm = make_propagator_q(L, L, shear_modulus=1.0)

    def _init_stress(self, key: jax.Array) -> jax.Array:
        """Initialize scalar stress field.

        Args:
            key: PRNG key (unused for zero initialization).

        Returns:
            Zero-initialized stress array of shape (L, L).
        """
        # Start relaxed (zero stress)
        return jnp.zeros((self.L, self.L))

    def _epm_step(
        self,
        state: Tuple[jax.Array, jax.Array, float, jax.Array],
        propagator_q: jax.Array,
        shear_rate: float,
        dt: float,
        params: dict,
        smooth: bool,
    ) -> Tuple[jax.Array, jax.Array, float, jax.Array]:
        """Perform one scalar EPM time step.

        Delegates to epm_step kernel from epm_kernels module.

        Args:
            state: Current state (stress, thresholds, strain, key).
            propagator_q: Precomputed propagator.
            shear_rate: Imposed shear rate.
            dt: Time step size.
            params: Model parameters.
            smooth: Use smooth yielding.

        Returns:
            Updated state tuple.
        """
        return epm_step(state, propagator_q, shear_rate, dt, params, smooth)

    def _predict(self, rheo_data: RheoData, **kwargs) -> RheoData:
        """Simulate the model for the given protocol.

        Args:
            rheo_data: Input data defining the protocol (t, gamma_dot, stress, etc.).
            kwargs:
                test_mode (str): 'flow_curve', 'startup', 'relaxation', 'creep', 'oscillation'.
                smooth (bool): Use smooth yielding (default False for simulation, True for fitting).
                seed (int): Random seed (default 0).

        Returns:
            RheoData with simulation results (stress or strain).
        """
        test_mode = kwargs.get("test_mode", rheo_data.test_mode)
        smooth = kwargs.get("smooth", False)
        seed = kwargs.get("seed", 0)
        key = jax.random.PRNGKey(seed)

        # Extract Parameters
        # Scale propagator by current mu
        mu = self.params.get_value("mu")
        propagator_q = self._propagator_q_norm * mu

        # Use base class method for parameter extraction
        param_dict = self._get_param_dict()

        if test_mode == "flow_curve":
            return self._run_flow_curve(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "startup":
            return self._run_startup(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "relaxation":
            return self._run_relaxation(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "creep":
            return self._run_creep(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "oscillation":
            return self._run_oscillation(rheo_data, key, propagator_q, param_dict, smooth)
        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")

