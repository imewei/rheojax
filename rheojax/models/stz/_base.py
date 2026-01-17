"""Base class for STZ models.

Provides shared infrastructure for:
1. Parameter initialization based on complexity variants
2. JAX-based ODE system definitions (Flow, Transient, LAOS)
3. Integration with Diffrax for time-stepping
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.logging import get_logger
from rheojax.models.stz._kernels import stz_ode_rhs

# Safe JAX import
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)

# Type definitions
VariantType = Literal["minimal", "standard", "full"]


class STZBase(BaseModel):
    """Base class for Shear Transformation Zone (STZ) models.

    Implements the core state evolution logic and parameter management
    for different model variants.

    Attributes:
        variant: Model complexity variant ('minimal', 'standard', 'full')
        parameters: ParameterSet containing model constants
    """

    def __init__(self, variant: VariantType = "standard"):
        """Initialize STZ Base Model.

        Args:
            variant: Complexity variant.
                - 'minimal': chi only (2 state vars: stress, chi)
                - 'standard': chi + Lambda (3 state vars) [Default]
                - 'full': chi + Lambda + m (4 state vars)
        """
        super().__init__()
        self.variant = variant
        self._setup_parameters()

        # Internal storage for trajectories (for plotting/debugging)
        self._trajectory: dict[str, np.ndarray] | None = None

    def _setup_parameters(self):
        """Initialize ParameterSet based on selected variant."""
        self.parameters = ParameterSet()

        # --- Common Parameters (All Variants) ---

        # G0: Elastic modulus (Pa) - renamed from G_inf to match kernel args
        self.parameters.add(
            name="G0",
            value=1e9,
            bounds=(1e6, 1e12),
            units="Pa",
            description="High-frequency elastic modulus",
        )

        # sigma_y: Yield stress (Pa) - scales the activation barrier
        self.parameters.add(
            name="sigma_y",
            value=1e6,
            bounds=(1e3, 1e9),
            units="Pa",
            description="Yield stress (characteristic stress scale)",
        )

        # chi_inf: Steady-state effective temperature (dimensionless)
        # Represents the structural disorder at high shear rates
        self.parameters.add(
            name="chi_inf",
            value=0.1,
            bounds=(0.01, 0.5),
            units="dimensionless",
            description="Steady-state effective temperature",
        )

        # tau0: Molecular attempt time (s)
        self.parameters.add(
            name="tau0",
            value=1e-12,
            bounds=(1e-14, 1e-9),
            units="s",
            description="Molecular vibration timescale",
        )

        # epsilon0: Characteristic strain increment
        self.parameters.add(
            name="epsilon0",
            value=0.1,
            bounds=(0.01, 1.0),
            units="dimensionless",
            description="Characteristic strain increment per STZ event",
        )

        # c0: Specific heat (dimensionless) - controls chi evolution rate
        self.parameters.add(
            name="c0",
            value=1.0,
            bounds=(0.1, 100.0),
            units="dimensionless",
            description="Specific heat parameter (controls chi rate)",
        )

        # Activation energy barrier (dimensionless, scaled by chi)
        # Often fixed to 1.0 in theoretical treatments, but can be fit
        self.parameters.add(
            name="ez",
            value=1.0,
            bounds=(0.1, 5.0),
            units="dimensionless",
            description="STZ formation energy (normalized)",
        )

        # --- Variant Specific Parameters ---

        if self.variant in ["standard", "full"]:
            # Lambda dynamics included
            # Relaxation time for Lambda
            self.parameters.add(
                name="tau_beta",
                value=1.0,
                bounds=(0.01, 100.0),
                units="s",
                description="Relaxation timescale for STZ density",
            )

        if self.variant == "full":
            # Back stress / orientation parameters
            self.parameters.add(
                name="m_inf",
                value=0.1,
                bounds=(0.0, 0.5),
                units="dimensionless",
                description="Saturation value for orientational bias",
            )
            self.parameters.add(
                name="rate_m",
                value=1.0,
                bounds=(0.1, 100.0),
                units="dimensionless",
                description="Rate coefficient for orientational bias",
            )

    def get_initial_state(self, stress_init: float = 0.0) -> jnp.ndarray:
        """Get initial state vector based on variant.

        Args:
            stress_init: Initial stress value.

        Returns:
            Initial state vector y0.
        """
        # Default initial conditions for internal variables
        # chi_init: start at annealed state (low chi)
        chi_init = 0.05

        # Lambda_init: Equilibrium at chi_init
        ez = self.parameters.get_value("ez")
        # Avoid div by zero if chi_init is 0 (unlikely)
        safe_chi = max(chi_init, 1e-6)
        lambda_init = jnp.exp(-ez / safe_chi)

        if self.variant == "minimal":
            # State: [stress, chi]
            return jnp.array([stress_init, chi_init])
        elif self.variant == "standard":
            # State: [stress, chi, Lambda]
            return jnp.array([stress_init, chi_init, lambda_init])
        elif self.variant == "full":
            # State: [stress, chi, Lambda, m]
            m_init = 0.0
            return jnp.array([stress_init, chi_init, lambda_init, m_init])
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _ode_term_strain_controlled(
        self, t: float, y: jnp.ndarray, args: dict
    ) -> jnp.ndarray:
        """ODE vector field wrapper for Diffrax.

        Delegates to the JAX-compiled kernel stz_ode_rhs.

        Args:
            t: Time
            y: State vector
            args: Dictionary of parameters and inputs

        Returns:
            dy/dt
        """
        return stz_ode_rhs(t, y, args)
