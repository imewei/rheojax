"""Rheological models package.

This package contains implementations of classical, fractional, and flow rheological models.

Classical Models:
    - Maxwell: Spring and dashpot in series
    - Zener: Standard Linear Solid (SLS)
    - SpringPot: Fractional power-law element

Fractional Maxwell Family:
    - FractionalMaxwellGel: SpringPot in series with dashpot
    - FractionalMaxwellLiquid: Spring in series with SpringPot
    - FractionalMaxwellModel: Two SpringPots in series (most general)
    - FractionalKelvinVoigt: Spring and SpringPot in parallel

Fractional Zener Family (Task Group 12):
    - FractionalZenerSolidLiquid: Fractional Maxwell + spring in parallel
    - FractionalZenerSolidSolid: Two springs + SpringPot
    - FractionalZenerLiquidLiquid: Most general fractional Zener
    - FractionalKelvinVoigtZener: FKV + spring in series

Advanced Fractional Models (Task Group 12):
    - FractionalBurgersModel: Maxwell + FKV in series
    - FractionalPoyntingThomson: FKV + spring in series (alternate formulation)
    - FractionalJeffreysModel: Two dashpots + SpringPot

Non-Newtonian Flow Models (ROTATION test mode):
    - PowerLaw: Simple power-law model (K*γ̇^n)
    - Carreau: Smooth transition from Newtonian to power-law
    - CarreauYasuda: Extended Carreau with transition parameter
    - Cross: Alternative to Carreau for polymer solutions
    - HerschelBulkley: Power-law with yield stress
    - Bingham: Linear viscoplastic (yield stress + constant viscosity)

Usage:
    >>> from rheo.models import Maxwell, Zener, SpringPot
    >>> from rheo.models import FractionalMaxwellGel, FractionalMaxwellLiquid
    >>> from rheo.models import FractionalZenerSolidLiquid, FractionalBurgersModel
    >>> from rheo.models import PowerLaw, Carreau, HerschelBulkley
    >>> from rheo.core.registry import ModelRegistry
    >>>
    >>> # Direct instantiation
    >>> model = Maxwell()
    >>> fzsl_model = FractionalZenerSolidLiquid()
    >>> flow_model = PowerLaw()
    >>>
    >>> # Factory pattern
    >>> model = ModelRegistry.create('maxwell')
    >>> fzsl_model = ModelRegistry.create('fractional_zener_sl')
    >>> flow_model = ModelRegistry.create('power_law')
    >>>
    >>> # List available models
    >>> models = ModelRegistry.list_models()
"""

# Classical models
from rheo.models.maxwell import Maxwell
from rheo.models.zener import Zener
from rheo.models.springpot import SpringPot

# Fractional Maxwell family
from rheo.models.fractional_maxwell_gel import FractionalMaxwellGel
from rheo.models.fractional_maxwell_liquid import FractionalMaxwellLiquid
from rheo.models.fractional_maxwell_model import FractionalMaxwellModel
from rheo.models.fractional_kelvin_voigt import FractionalKelvinVoigt

# Fractional Zener Family (Task Group 12)
from rheo.models.fractional_zener_sl import FractionalZenerSolidLiquid, FZSL
from rheo.models.fractional_zener_ss import FractionalZenerSolidSolid, FZSS
from rheo.models.fractional_zener_ll import FractionalZenerLiquidLiquid, FZLL
from rheo.models.fractional_kv_zener import FractionalKelvinVoigtZener, FKVZ

# Advanced Fractional Models (Task Group 12)
from rheo.models.fractional_burgers import FractionalBurgersModel, FBM
from rheo.models.fractional_poynting_thomson import FractionalPoyntingThomson, FPT
from rheo.models.fractional_jeffreys import FractionalJeffreysModel, FJM

# Non-Newtonian flow models
from rheo.models.power_law import PowerLaw
from rheo.models.carreau import Carreau
from rheo.models.carreau_yasuda import CarreauYasuda
from rheo.models.cross import Cross
from rheo.models.herschel_bulkley import HerschelBulkley
from rheo.models.bingham import Bingham

__all__ = [
    # Classical models
    'Maxwell',
    'Zener',
    'SpringPot',
    # Fractional Maxwell family
    'FractionalMaxwellGel',
    'FractionalMaxwellLiquid',
    'FractionalMaxwellModel',
    'FractionalKelvinVoigt',
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
    # Non-Newtonian flow models
    'PowerLaw',
    'Carreau',
    'CarreauYasuda',
    'Cross',
    'HerschelBulkley',
    'Bingham',
]
