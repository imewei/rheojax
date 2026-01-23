"""Fluidity-Saramito Elastoviscoplastic (EVP) Models.

This package implements the Saramito constitutive model for elastoviscoplastic
materials with thixotropic fluidity evolution. The model combines:

1. **Tensorial Viscoelasticity**: Upper-convected Maxwell framework
2. **Viscoplasticity**: Von Mises yield criterion with Herschel-Bulkley flow
3. **Thixotropy**: Time-dependent aging and shear rejuvenation via fluidity

Model Variants
--------------
- `FluiditySaramitoLocal`: Homogeneous (0D) model for bulk rheometry
- `FluiditySaramitoNonlocal`: Spatially-resolved (1D) model with diffusion

Coupling Modes
--------------
- "minimal": Relaxation time λ = 1/f only
- "full": λ = 1/f + τ_y(f) aging yield stress

Supported Protocols
-------------------
- FLOW_CURVE: Steady shear viscometry
- CREEP: Stress-controlled strain evolution (bifurcation)
- RELAXATION: Stress decay after step strain
- STARTUP: Transient stress at constant rate (overshoot)
- OSCILLATION: SAOS (G', G'')
- LAOS: Large amplitude nonlinear response

References
----------
- Saramito, P. (2007). JNNFM 145, 1-14.
- Saramito, P. (2009). JNNFM 158, 154-161.
- Coussot, P. et al. (2002). J. Rheol. 46(3), 573-589.
"""

from rheojax.models.fluidity.saramito.local import FluiditySaramitoLocal
from rheojax.models.fluidity.saramito.nonlocal_model import FluiditySaramitoNonlocal

__all__ = [
    "FluiditySaramitoLocal",
    "FluiditySaramitoNonlocal",
]
