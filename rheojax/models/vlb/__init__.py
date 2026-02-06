"""Vernerey-Long-Brighenti (VLB) Transient Network Models.

Statistically-based continuum models for polymers with dynamic cross-links.
The distribution tensor mu captures the chain-level state and evolves via
a tensorial ODE coupled to bond kinetics.

Model Variants
--------------
- `VLBLocal`: Single transient network (G0, k_d -> Maxwell with molecular basis)
- `VLBMultiNetwork`: M transient + permanent + solvent (generalized Maxwell)
- `VLBVariant`: Composable Bell + FENE-P + Temperature extensions
- `VLBNonlocal`: Spatial PDE with tensor diffusion for shear banding

Supported Protocols
-------------------
FLOW_CURVE, STARTUP, RELAXATION, CREEP, OSCILLATION, LAOS

Reference
---------
Vernerey, Long, & Brighenti (2017). JMPS 107, 1-20.
"""

from rheojax.models.vlb.local import VLBLocal
from rheojax.models.vlb.multi_network import VLBMultiNetwork
from rheojax.models.vlb.nonlocal_model import VLBNonlocal
from rheojax.models.vlb.variant import VLBVariant

__all__ = ["VLBLocal", "VLBMultiNetwork", "VLBNonlocal", "VLBVariant"]
