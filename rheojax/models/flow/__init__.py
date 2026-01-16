"""Non-Newtonian flow models.

Contains viscosity models for ROTATION test mode:
- PowerLaw: Simple power-law model (K*gamma_dot^n)
- Carreau: Smooth transition from Newtonian to power-law
- CarreauYasuda: Extended Carreau with transition parameter
- Cross: Alternative to Carreau for polymer solutions
- HerschelBulkley: Power-law with yield stress
- Bingham: Linear viscoplastic (yield stress + constant viscosity)
"""

from rheojax.models.flow.bingham import Bingham
from rheojax.models.flow.carreau import Carreau
from rheojax.models.flow.carreau_yasuda import CarreauYasuda
from rheojax.models.flow.cross import Cross
from rheojax.models.flow.herschel_bulkley import HerschelBulkley
from rheojax.models.flow.power_law import PowerLaw

__all__ = [
    "PowerLaw",
    "Carreau",
    "CarreauYasuda",
    "Cross",
    "HerschelBulkley",
    "Bingham",
]
