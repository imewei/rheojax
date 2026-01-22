"""Base class for Fluidity models.

Provides shared infrastructure for:
1. Parameter initialization for Local and Non-Local variants
2. Common parameter definitions for yield-stress fluid behavior
3. Shared methods for ODE/PDE integration with Diffrax
"""

from __future__ import annotations

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.logging import get_logger

# Safe JAX import
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)


class FluidityBase(BaseModel):
    """Base class for Fluidity models for yield-stress fluids.

    Implements shared parameter management and trajectory storage
    for Local (0D) and Non-Local (1D) variants.

    The Fluidity model describes yield-stress fluids through a scalar
    fluidity field f(t) or f(y,t) that evolves via:
    - Aging (structural build-up) toward equilibrium fluidity f_eq
    - Rejuvenation (flow-induced breakdown) toward high-shear fluidity f_inf

    Attributes:
        parameters: ParameterSet containing model constants
    """

    def __init__(self):
        """Initialize Fluidity Base Model."""
        super().__init__()
        self._setup_parameters()

        # Internal storage for trajectories (for plotting/debugging)
        self._trajectory: dict[str, np.ndarray] | None = None

        # Protocol-specific inputs
        self._gamma_dot_applied: float | None = None
        self._sigma_applied: float | None = None
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None

    def _setup_parameters(self):
        """Initialize ParameterSet with shared parameters."""
        self.parameters = ParameterSet()

        # --- Elastic/Viscoelastic Parameters ---

        # G: Elastic modulus (Pa)
        self.parameters.add(
            name="G",
            value=1e6,
            bounds=(1e3, 1e9),
            units="Pa",
            description="Elastic modulus",
        )

        # --- Yield Stress / Flow Curve Parameters (Herschel-Bulkley) ---

        # tau_y: Yield stress (Pa)
        self.parameters.add(
            name="tau_y",
            value=1e3,
            bounds=(1e1, 1e6),
            units="Pa",
            description="Yield stress",
        )

        # K: Flow consistency (Pa·s^n)
        self.parameters.add(
            name="K",
            value=1e3,
            bounds=(1e0, 1e6),
            units="Pa·s^n",
            description="Flow consistency (HB K parameter)",
        )

        # n_flow: Flow exponent (dimensionless)
        self.parameters.add(
            name="n_flow",
            value=0.5,
            bounds=(0.1, 2.0),
            units="dimensionless",
            description="Flow exponent (HB n parameter)",
        )

        # --- Fluidity Dynamics Parameters ---

        # f_eq: Equilibrium (low-shear) fluidity (1/(Pa·s))
        self.parameters.add(
            name="f_eq",
            value=1e-6,
            bounds=(1e-12, 1e-3),
            units="1/(Pa·s)",
            description="Equilibrium fluidity (aging limit)",
        )

        # f_inf: High-shear fluidity (1/(Pa·s))
        self.parameters.add(
            name="f_inf",
            value=1e-3,
            bounds=(1e-6, 1.0),
            units="1/(Pa·s)",
            description="High-shear fluidity (rejuvenation limit)",
        )

        # theta: Relaxation time / aging timescale (s)
        self.parameters.add(
            name="theta",
            value=10.0,
            bounds=(0.1, 1e4),
            units="s",
            description="Structural relaxation time (aging timescale)",
        )

        # a: Rejuvenation amplitude (dimensionless)
        self.parameters.add(
            name="a",
            value=1.0,
            bounds=(0.0, 100.0),
            units="dimensionless",
            description="Rejuvenation amplitude",
        )

        # n_rejuv: Rejuvenation exponent (dimensionless)
        self.parameters.add(
            name="n_rejuv",
            value=1.0,
            bounds=(0.0, 2.0),
            units="dimensionless",
            description="Rejuvenation exponent",
        )

    def get_initial_fluidity(self) -> float:
        """Get initial fluidity value (equilibrium).

        Returns:
            Initial fluidity f_0 = f_eq
        """
        f_eq = self.parameters.get_value("f_eq")
        return f_eq if f_eq is not None else 1e-6

    def get_parameter_dict(self) -> dict:
        """Get all parameters as a dictionary.

        Returns:
            Dictionary of parameter name -> value
        """
        return {k: self.parameters.get_value(k) for k in self.parameters.keys()}

    def _get_base_ode_args(self, params: dict | None = None) -> dict:
        """Build base args dictionary for ODE integration.

        Args:
            params: Optional parameter dictionary. If None, uses stored values.

        Returns:
            Dictionary with all parameters needed for ODE RHS.
        """
        if params is None:
            params = self.get_parameter_dict()

        return {
            "G": params["G"],
            "tau_y": params["tau_y"],
            "K": params["K"],
            "n_flow": params["n_flow"],
            "f_eq": params["f_eq"],
            "f_inf": params["f_inf"],
            "theta": params["theta"],
            "a": params["a"],
            "n_rejuv": params["n_rejuv"],
        }
