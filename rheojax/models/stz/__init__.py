"""Shear Transformation Zone (STZ) Models.

This package implements the STZ plasticity model based on the formulation by
Langer (2008), featuring:
- Effective temperature (chi) dynamics
- STZ density (Lambda) evolution
- Orientational bias (m) for Bauschinger effect
- JAX-accelerated kernels and Diffrax integration
"""

from rheojax.models.stz._base import STZBase
from rheojax.models.stz.conventional import STZConventional

__all__ = ["STZBase", "STZConventional"]
