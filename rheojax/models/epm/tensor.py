"""Tensorial Elasto-Plastic Model (EPM) scaffolding.

This module reserves the API space for a future implementation of a full Tensorial EPM,
which will track the full stress tensor ($\sigma_{xx}, \sigma_{xy}, \sigma_{yy}$) and
use a tensorial Eshelby propagator to capture normal stress differences and
complex flow geometries.

Currently not implemented.
"""

from rheojax.core.base import BaseModel
from rheojax.core.registry import ModelRegistry


@ModelRegistry.register("tensorial_epm")
class TensorialEPM(BaseModel):
    """3-Component Tensorial Lattice EPM (Scaffold).

    Planned Features:
        - Full stress tensor state per site.
        - Rank-4 Eshelby propagator $\mathcal{G}_{ijkl}$.
        - Von Mises yield criterion ($\sqrt{J_2} > \sigma_c$).
        - Prediction of Normal Stress Differences ($N_1, N_2$).
    """

    def __init__(self, L: int = 64, **kwargs):
        super().__init__()
        self.L = L

    def _fit(self, X, y, **kwargs):
        raise NotImplementedError(
            "TensorialEPM is not yet implemented. Use LatticeEPM for scalar simulations."
        )

    def _predict(self, rheo_data, **kwargs):
        raise NotImplementedError(
            "TensorialEPM is not yet implemented. Use LatticeEPM for scalar simulations."
        )
