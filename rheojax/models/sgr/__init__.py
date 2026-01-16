"""Soft Glassy Rheology (SGR) models.

Contains statistical mechanics models for soft glassy materials
(foams, emulsions, pastes, colloidal suspensions):
- SGRConventional: Soft Glassy Rheology model (Sollich 1998)
- SGRGeneric: GENERIC framework SGR (thermodynamically consistent, Fuereder & Ilg 2013)
"""

from rheojax.models.sgr.sgr_conventional import SGRConventional
from rheojax.models.sgr.sgr_generic import SGRGeneric

__all__ = ["SGRConventional", "SGRGeneric"]
