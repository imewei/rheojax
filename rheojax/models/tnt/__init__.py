"""Transient Network Theory (TNT) Models.

This package implements constitutive models based on transient network theory
for associative polymers, physical gels, and reversibly crosslinked networks.

Model Variants
--------------
- `TNTSingleMode`: Composable single-mode model (basic, Bell, FENE-P, NonAffine)
- `TNTLoopBridge`: Two-species loop-bridge kinetics for telechelic polymers
- `TNTStickyRouse`: Multi-mode with sticker kinetics for ionomers
- `TNTCates`: Living polymers (wormlike micelles) with reptation + breakage
- `TNTMultiSpecies`: Multiple bond types with different lifetimes

Key Physics
-----------
TNT models describe polymer networks with reversible (physical) crosslinks:

1. **Network kinetics**: Chains attach/detach with rates g₀ and β(S)
2. **Rubber elasticity**: Stress from chain stretch, σ = G·(S - I)
3. **Maxwell-like SAOS**: Single relaxation time in linear regime
4. **Nonlinear flow**: Stress overshoot, shear thinning (variant-dependent)

The conformation tensor S evolves via::

    dS/dt = L·S + S·L^T + g₀·I - β(S)·S

Supported Protocols
-------------------
- FLOW_CURVE: Steady shear (analytical for basic, numerical for Bell/FENE)
- OSCILLATION: Small-amplitude oscillatory shear (Maxwell-like, analytical)
- STARTUP: Transient stress growth at constant rate (ODE)
- RELAXATION: Stress decay after cessation (analytical or ODE)
- CREEP: Strain evolution under constant stress (ODE)
- LAOS: Large-amplitude oscillatory shear (ODE)

References
----------
- Green, M.S. & Tobolsky, A.V. (1946). J. Chem. Phys. 14, 80-92.
- Tanaka, F. & Edwards, S.F. (1992). Macromolecules 25, 1516-1523.
- Bell, G.I. (1978). Science 200, 618-627.
- Cates, M.E. (1987). Macromolecules 20, 2289-2296.
- Annable, T. et al. (1993). J. Rheol. 37, 695-726.
"""

from rheojax.models.tnt.cates import TNTCates
from rheojax.models.tnt.loop_bridge import TNTLoopBridge
from rheojax.models.tnt.multi_species import TNTMultiSpecies
from rheojax.models.tnt.single_mode import TNTSingleMode
from rheojax.models.tnt.sticky_rouse import TNTStickyRouse

__all__ = [
    "TNTSingleMode",
    "TNTLoopBridge",
    "TNTStickyRouse",
    "TNTCates",
    "TNTMultiSpecies",
]
