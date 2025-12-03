"""Rheological models package.

This package contains implementations of classical, fractional, and flow rheological models.

Classical Models:
    - Maxwell: Spring and dashpot in series
    - Zener: Standard Linear Solid (SLS)
    - SpringPot: Fractional power-law element

Multi-Mode Models:
    - GeneralizedMaxwell: Generalized Maxwell Model (Prony series, N modes)

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

Soft Glassy Rheology Models:
    - SGRConventional: Soft Glassy Rheology model (Sollich 1998)

Usage:
    >>> from rheojax.models import Maxwell, Zener, SpringPot
    >>> from rheojax.models import FractionalMaxwellGel, FractionalMaxwellLiquid
    >>> from rheojax.models import FractionalZenerSolidLiquid, FractionalBurgersModel
    >>> from rheojax.models import PowerLaw, Carreau, HerschelBulkley
    >>> from rheojax.models import SGRConventional
    >>> from rheojax.core.registry import ModelRegistry
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
from rheojax.models.bingham import Bingham
from rheojax.models.carreau import Carreau
from rheojax.models.carreau_yasuda import CarreauYasuda
from rheojax.models.cross import Cross

# Advanced Fractional Models (Task Group 12)
from rheojax.models.fractional_burgers import FBM, FractionalBurgersModel
from rheojax.models.fractional_jeffreys import FJM, FractionalJeffreysModel
from rheojax.models.fractional_kelvin_voigt import FractionalKelvinVoigt
from rheojax.models.fractional_kv_zener import FKVZ, FractionalKelvinVoigtZener

# Fractional Maxwell family
from rheojax.models.fractional_maxwell_gel import FractionalMaxwellGel
from rheojax.models.fractional_maxwell_liquid import FractionalMaxwellLiquid
from rheojax.models.fractional_maxwell_model import FractionalMaxwellModel
from rheojax.models.fractional_poynting_thomson import FPT, FractionalPoyntingThomson
from rheojax.models.fractional_zener_ll import FZLL, FractionalZenerLiquidLiquid

# Fractional Zener Family (Task Group 12)
from rheojax.models.fractional_zener_sl import FZSL, FractionalZenerSolidLiquid
from rheojax.models.fractional_zener_ss import FZSS, FractionalZenerSolidSolid

# Multi-Mode models
from rheojax.models.generalized_maxwell import GeneralizedMaxwell
from rheojax.models.herschel_bulkley import HerschelBulkley
from rheojax.models.maxwell import Maxwell

# Non-Newtonian flow models
from rheojax.models.power_law import PowerLaw

# Soft Glassy Rheology models
from rheojax.models.sgr_conventional import SGRConventional
from rheojax.models.sgr_generic import SGRGeneric

# SPP Yield Stress model (LAOS analysis)
from rheojax.models.spp_yield_stress import SPPYieldStress
from rheojax.models.springpot import SpringPot
from rheojax.models.zener import Zener

__all__ = [
    # Classical models
    "Maxwell",
    "Zener",
    "SpringPot",
    # Multi-Mode models
    "GeneralizedMaxwell",
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
    # Non-Newtonian flow models
    "PowerLaw",
    "Carreau",
    "CarreauYasuda",
    "Cross",
    "HerschelBulkley",
    "Bingham",
    # Soft Glassy Rheology models
    "SGRConventional",
    "SGRGeneric",
    # SPP Yield Stress model
    "SPPYieldStress",
]
