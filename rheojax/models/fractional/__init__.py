"""Fractional rheological models.

Contains models based on fractional calculus (SpringPot elements):

Fractional Maxwell Family (4):
- FractionalMaxwellGel: SpringPot in series with dashpot
- FractionalMaxwellLiquid: Spring in series with SpringPot
- FractionalMaxwellModel: Two SpringPots in series (most general)
- FractionalKelvinVoigt: Spring and SpringPot in parallel

Fractional Zener Family (4):
- FractionalZenerSolidLiquid (FZSL): Fractional Maxwell + spring in parallel
- FractionalZenerSolidSolid (FZSS): Two springs + SpringPot
- FractionalZenerLiquidLiquid (FZLL): Most general fractional Zener
- FractionalKelvinVoigtZener (FKVZ): FKV + spring in series

Advanced Fractional Models (3):
- FractionalBurgersModel (FBM): Maxwell + FKV in series
- FractionalPoyntingThomson (FPT): FKV + spring in series (alternate formulation)
- FractionalJeffreysModel (FJM): Two dashpots + SpringPot
"""

from rheojax.models.fractional.fractional_burgers import FBM, FractionalBurgersModel
from rheojax.models.fractional.fractional_jeffreys import FJM, FractionalJeffreysModel
from rheojax.models.fractional.fractional_kelvin_voigt import FractionalKelvinVoigt
from rheojax.models.fractional.fractional_kv_zener import FKVZ, FractionalKelvinVoigtZener
from rheojax.models.fractional.fractional_maxwell_gel import FractionalMaxwellGel
from rheojax.models.fractional.fractional_maxwell_liquid import FractionalMaxwellLiquid
from rheojax.models.fractional.fractional_maxwell_model import FractionalMaxwellModel
from rheojax.models.fractional.fractional_mixin import FractionalModelMixin
from rheojax.models.fractional.fractional_poynting_thomson import (
    FPT,
    FractionalPoyntingThomson,
)
from rheojax.models.fractional.fractional_zener_ll import FZLL, FractionalZenerLiquidLiquid
from rheojax.models.fractional.fractional_zener_sl import FZSL, FractionalZenerSolidLiquid
from rheojax.models.fractional.fractional_zener_ss import FZSS, FractionalZenerSolidSolid

__all__ = [
    # Mixin
    "FractionalModelMixin",
    # Fractional Maxwell family
    "FractionalMaxwellGel",
    "FractionalMaxwellLiquid",
    "FractionalMaxwellModel",
    "FractionalKelvinVoigt",
    # Fractional Zener Family
    "FractionalZenerSolidLiquid",
    "FZSL",
    "FractionalZenerSolidSolid",
    "FZSS",
    "FractionalZenerLiquidLiquid",
    "FZLL",
    "FractionalKelvinVoigtZener",
    "FKVZ",
    # Advanced Fractional Models
    "FractionalBurgersModel",
    "FBM",
    "FractionalPoyntingThomson",
    "FPT",
    "FractionalJeffreysModel",
    "FJM",
]
