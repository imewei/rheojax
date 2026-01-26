"""Giesekus Nonlinear Viscoelastic Models.

This package implements the Giesekus constitutive model (1982) for polymer
melts and solutions. The model extends the Upper-Convected Maxwell framework
with a quadratic stress term representing anisotropic molecular mobility.

Model Variants
--------------
- `GiesekusSingleMode`: Single relaxation time model
- `GiesekusMultiMode`: Multi-mode extension with N relaxation times

Key Physics
-----------
The Giesekus model introduces a mobility factor α (0 ≤ α ≤ 0.5) that
captures:

1. **Shear-thinning**: Viscosity decreases with increasing shear rate
2. **Normal stress differences**: Predicts both N₁ > 0 and N₂ < 0
3. **Stress overshoot**: Transient behavior in startup flow
4. **Nonlinear LAOS**: Higher harmonics in large-amplitude oscillation

The constitutive equation is::

    τ + λ∇̂τ + (αλ/η_p)τ·τ = 2η_p D

where ∇̂ denotes the upper-convected derivative.

Supported Protocols
-------------------
- FLOW_CURVE: Steady shear viscometry (analytical)
- OSCILLATION: Small-amplitude oscillatory shear (analytical)
- STARTUP: Transient stress at constant rate (ODE)
- RELAXATION: Stress decay after step strain (ODE)
- CREEP: Strain evolution under constant stress (ODE)
- LAOS: Large-amplitude oscillatory shear (ODE + FFT)

Special Cases
-------------
- α = 0: Recovers Upper-Convected Maxwell (UCM) model
- α = 0.5: Maximum anisotropy, approaches Oldroyd-B behavior

Diagnostic Relationship
-----------------------
The ratio of normal stress differences is directly related to α::

    N₂/N₁ = -α/2

This provides a direct experimental route to determine α.

References
----------
- Giesekus, H. (1982). J. Non-Newtonian Fluid Mech. 11, 69-109.
- Bird, R.B. et al. (1987). Dynamics of Polymeric Liquids, Vol. 1.
- Larson, R.G. (1988). Constitutive Equations for Polymer Melts.
"""

from rheojax.models.giesekus.multi_mode import GiesekusMultiMode
from rheojax.models.giesekus.single_mode import GiesekusSingleMode

__all__ = [
    "GiesekusSingleMode",
    "GiesekusMultiMode",
]
