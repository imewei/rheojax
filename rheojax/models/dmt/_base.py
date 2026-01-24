"""Base class for de Souza Mendes-Thompson (DMT) models.

Provides shared infrastructure for:
1. Parameter initialization for Local and Nonlocal variants
2. Common parameter definitions for thixotropic yield-stress fluid behavior
3. Shared methods for ODE/PDE integration

The DMT model captures yielding, thixotropy, and optional viscoelasticity
through a structure parameter λ ∈ [0, 1] that evolves via buildup/breakdown
kinetics.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.logging import get_logger

# Safe JAX import
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)


class DMTBase(BaseModel):
    """Base class for DMT (de Souza Mendes-Thompson) thixotropic models.

    Implements shared parameter management and trajectory storage
    for Local (0D) and Nonlocal (1D) variants.

    The DMT model describes thixotropic yield-stress fluids through a
    scalar structure parameter λ(t) that evolves via:
    - Buildup (aging) toward fully-structured state (λ → 1)
    - Breakdown (rejuvenation) toward fully-broken state (λ → 0)

    Two viscosity closures are supported:
    - "exponential": η(λ) = η_∞ · (η_0/η_∞)^λ (original DMT 2013)
    - "herschel_bulkley": η = τ_y(λ)/|γ̇| + K(λ)|γ̇|^(n-1) + η_∞

    Optionally includes Maxwell viscoelastic backbone for:
    - Stress overshoot in startup shear
    - True stress relaxation
    - SAOS moduli G'(ω), G''(ω)

    Parameters
    ----------
    closure : {"exponential", "herschel_bulkley"}, default "exponential"
        Viscosity closure type.
        - "exponential": Smooth interpolation between η_0 and η_∞
        - "herschel_bulkley": Explicit yield stress + consistency

    include_elasticity : bool, default True
        Whether to include Maxwell viscoelastic backbone.
        - True: DMT-Maxwell with stress overshoot and SAOS
        - False: DMT-Viscous (pure generalized Newtonian)

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters with bounds, units, and descriptions
    closure : str
        Active viscosity closure type
    include_elasticity : bool
        Whether elasticity is active

    References
    ----------
    de Souza Mendes, P.R. & Thompson, R.L. (2012).
        "A critical overview of elasto-viscoplastic thixotropic modeling."
        J. Non-Newtonian Fluid Mech. 187-188, 8-15.

    de Souza Mendes, P.R. & Thompson, R.L. (2013).
        "A unified approach to model elasto-viscoplastic thixotropic
        yield-stress materials and apparent yield-stress fluids."
        Rheol. Acta 52, 673-694.
    """

    def __init__(
        self,
        closure: Literal["exponential", "herschel_bulkley"] = "exponential",
        include_elasticity: bool = True,
    ):
        """Initialize DMT Base Model.

        Parameters
        ----------
        closure : {"exponential", "herschel_bulkley"}, default "exponential"
            Viscosity closure type.
        include_elasticity : bool, default True
            Include Maxwell viscoelastic backbone.
        """
        self.closure = closure
        self.include_elasticity = include_elasticity

        super().__init__()
        self._setup_parameters()

        # Internal storage for trajectories (for plotting/debugging)
        self._trajectory: dict[str, np.ndarray] | None = None

        # Protocol-specific inputs
        self._gamma_dot_applied: float | None = None
        self._sigma_applied: float | None = None
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None

        logger.debug(
            "DMTBase initialized",
            closure=self.closure,
            include_elasticity=self.include_elasticity,
        )

    def _setup_parameters(self):
        """Initialize ParameterSet with model parameters.

        Parameters are added conditionally based on closure type
        and include_elasticity setting.
        """
        self.parameters = ParameterSet()

        # --- Core Viscosity Parameters (all closures) ---

        # η₀: Zero-shear viscosity (fully structured, λ=1)
        self.parameters.add(
            name="eta_0",
            value=1e5,
            bounds=(1e2, 1e8),
            units="Pa·s",
            description="Zero-shear viscosity (λ=1, fully structured)",
        )

        # η_∞: Infinite-shear viscosity (fully broken, λ=0)
        self.parameters.add(
            name="eta_inf",
            value=0.1,
            bounds=(1e-3, 1e2),
            units="Pa·s",
            description="Infinite-shear viscosity (λ=0, fully broken)",
        )

        # --- Herschel-Bulkley Closure Parameters ---

        if self.closure == "herschel_bulkley":
            # τ_y0: Fully-structured yield stress
            self.parameters.add(
                name="tau_y0",
                value=10.0,
                bounds=(0.1, 1e4),
                units="Pa",
                description="Fully-structured yield stress",
            )

            # K₀: Fully-structured consistency
            self.parameters.add(
                name="K0",
                value=5.0,
                bounds=(0.1, 1e3),
                units="Pa·s^n",
                description="Fully-structured consistency",
            )

            # n: Flow index
            self.parameters.add(
                name="n_flow",
                value=0.5,
                bounds=(0.1, 1.0),
                units="dimensionless",
                description="Flow index (shear-thinning exponent)",
            )

            # m₁: Yield stress exponent
            self.parameters.add(
                name="m1",
                value=1.0,
                bounds=(0.5, 2.0),
                units="dimensionless",
                description="Yield stress structure exponent: τ_y = τ_y0·λ^m1",
            )

            # m₂: Consistency exponent
            self.parameters.add(
                name="m2",
                value=1.0,
                bounds=(0.5, 2.0),
                units="dimensionless",
                description="Consistency structure exponent: K = K0·λ^m2",
            )

        # --- Elastic Parameters (Maxwell backbone) ---

        if self.include_elasticity:
            # G₀: Elastic modulus at λ=1
            self.parameters.add(
                name="G0",
                value=100.0,
                bounds=(1e0, 1e6),
                units="Pa",
                description="Elastic modulus at λ=1 (fully structured)",
            )

            # m_G: Modulus structure exponent
            self.parameters.add(
                name="m_G",
                value=1.0,
                bounds=(0.5, 2.0),
                units="dimensionless",
                description="Modulus structure exponent: G = G0·λ^m_G",
            )

        # --- Structure Kinetics Parameters ---

        # t_eq: Equilibrium/buildup timescale
        self.parameters.add(
            name="t_eq",
            value=100.0,
            bounds=(0.1, 1e4),
            units="s",
            description="Structural equilibrium (buildup) timescale",
        )

        # a: Breakdown rate coefficient
        self.parameters.add(
            name="a",
            value=1.0,
            bounds=(1e-3, 1e2),
            units="dimensionless",
            description="Breakdown rate coefficient",
        )

        # c: Breakdown rate exponent
        self.parameters.add(
            name="c",
            value=1.0,
            bounds=(0.1, 2.0),
            units="dimensionless",
            description="Breakdown rate exponent (shear rate sensitivity)",
        )

    def get_initial_structure(self) -> float:
        """Get initial structure parameter value (fully structured).

        Returns
        -------
        float
            Initial λ = 1.0 (fully structured state)
        """
        return 1.0

    def get_parameter_dict(self) -> dict:
        """Get all parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary of parameter name -> value
        """
        return {k: self.parameters.get_value(k) for k in self.parameters.keys()}

    def _get_base_ode_args(self, params: dict | None = None) -> dict:
        """Build base args dictionary for ODE integration.

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
            "eta_0": params["eta_0"],
            "eta_inf": params["eta_inf"],
            "t_eq": params["t_eq"],
            "a": params["a"],
            "c": params["c"],
            "closure": self.closure,
            "include_elasticity": self.include_elasticity,
        }

        # Add HB-specific parameters
        if self.closure == "herschel_bulkley":
            args.update(
                {
                    "tau_y0": params["tau_y0"],
                    "K0": params["K0"],
                    "n_flow": params["n_flow"],
                    "m1": params["m1"],
                    "m2": params["m2"],
                }
            )

        # Add elasticity parameters
        if self.include_elasticity:
            args.update(
                {
                    "G0": params["G0"],
                    "m_G": params["m_G"],
                }
            )

        return args

    def get_closure_info(self) -> str:
        """Get human-readable description of the closure.

        Returns
        -------
        str
            Description of the viscosity closure in use.
        """
        if self.closure == "exponential":
            return "Exponential: η(λ) = η_∞·(η_0/η_∞)^λ"
        else:
            return "Herschel-Bulkley: σ = τ_y(λ) + K(λ)|γ̇|^n + η_∞|γ̇|"

    def get_elasticity_info(self) -> str:
        """Get human-readable description of elasticity setting.

        Returns
        -------
        str
            Description of the elasticity configuration.
        """
        if self.include_elasticity:
            return "DMT-Maxwell: Viscoelastic with stress overshoot, SAOS"
        else:
            return "DMT-Viscous: Pure generalized Newtonian (no SAOS)"

    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        return (
            f"{class_name}(closure='{self.closure}', "
            f"include_elasticity={self.include_elasticity})"
        )
