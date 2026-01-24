"""de Souza Mendes-Thompson (DMT) thixotropic models.

This module provides DMT model implementations for thixotropic
yield-stress fluids with structural kinetics.

The DMT model family captures:
- **Yielding**: Explicit yield stress dependent on microstructure
- **Thixotropy**: Time-dependent structure via buildup/breakdown kinetics
- **Viscoelasticity**: Optional Maxwell backbone for stress overshoot and SAOS

Models:
    DMTLocal: Local (0D) DMT model for homogeneous flow
    DMTNonlocal: Nonlocal (1D) DMT model for shear banding analysis

Supported Protocols:
    - Flow curve (steady shear)
    - Start-up shear (stress overshoot)
    - Stress relaxation (Maxwell variant only)
    - Creep (delayed yielding)
    - SAOS (Maxwell variant only)
    - LAOS (nonlinear oscillatory)

References:
    de Souza Mendes, P.R. & Thompson, R.L. (2012).
        "A critical overview of elasto-viscoplastic thixotropic modeling."
        J. Non-Newtonian Fluid Mech. 187-188, 8-15.

    de Souza Mendes, P.R. & Thompson, R.L. (2013).
        "A unified approach to model elasto-viscoplastic thixotropic
        yield-stress materials and apparent yield-stress fluids."
        Rheol. Acta 52, 673-694.

Example:
    >>> from rheojax.models.dmt import DMTLocal
    >>>
    >>> # Create model with Herschel-Bulkley closure and elasticity
    >>> model = DMTLocal(closure="herschel_bulkley", include_elasticity=True)
    >>>
    >>> # Fit to flow curve data
    >>> model.fit(gamma_dot, stress, test_mode="flow_curve")
    >>>
    >>> # Simulate startup shear
    >>> t, stress, lam = model.simulate_startup(gamma_dot=10.0, t_end=100.0)
"""

from rheojax.models.dmt.local import DMTLocal
from rheojax.models.dmt.nonlocal_model import DMTNonlocal

__all__ = ["DMTLocal", "DMTNonlocal"]
