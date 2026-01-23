"""Fluidity Models for yield-stress fluids.

This package provides fluidity-based constitutive models for yield-stress
fluids, implementing both Local (0D homogeneous) and Non-Local (1D Couette
with spatial diffusion) variants.

Models:
    FluidityLocal: Local (0D) model with aging/rejuvenation dynamics
    FluidityNonlocal: Non-Local (1D PDE) model with cooperativity length

The fluidity framework describes yield-stress behavior through a scalar
fluidity field f(t) or f(y,t) that evolves via competing mechanisms:
- Aging: structural build-up at rest (f → f_eq)
- Rejuvenation: flow-induced breakdown (f → f_inf)

For the Non-Local model, spatial diffusion with cooperativity length ξ
enables shear banding predictions in Couette geometry.

Example:
    >>> from rheojax.models.fluidity import FluidityLocal, FluidityNonlocal
    >>>
    >>> # Local model - simple yield stress behavior
    >>> model = FluidityLocal()
    >>> model.fit(gamma_dot, stress, test_mode='flow_curve')
    >>>
    >>> # Non-Local model with shear banding
    >>> model_nl = FluidityNonlocal(N_y=64, gap_width=1e-3)
    >>> model_nl.fit(t, stress, test_mode='startup', gamma_dot=1.0)
    >>> cv = model_nl.get_shear_banding_metric()  # CV > 0.3 indicates banding

References:
    - Coussot et al., Phys. Rev. Lett. 88, 175501 (2002)
    - Goyon et al., Nature 454, 84-87 (2008)
    - Ovarlez et al., J. Non-Newtonian Fluid Mech. 177-178, 19-28 (2012)
"""

from rheojax.models.fluidity.local import FluidityLocal
from rheojax.models.fluidity.nonlocal_model import FluidityNonlocal
from rheojax.models.fluidity.saramito import (
    FluiditySaramitoLocal,
    FluiditySaramitoNonlocal,
)

__all__ = [
    "FluidityLocal",
    "FluidityNonlocal",
    "FluiditySaramitoLocal",
    "FluiditySaramitoNonlocal",
]
