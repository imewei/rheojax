"""Classical rheological models.

Contains fundamental spring-dashpot models:
- Maxwell: Spring and dashpot in series
- Zener: Standard Linear Solid (SLS)
- SpringPot: Fractional power-law element
"""

from rheojax.models.classical.maxwell import Maxwell
from rheojax.models.classical.springpot import SpringPot
from rheojax.models.classical.zener import Zener

__all__ = ["Maxwell", "Zener", "SpringPot"]
