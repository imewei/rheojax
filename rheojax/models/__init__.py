"""Rheological models package.

This package contains 27 rheological models organized into 10 categories.

Classical Models (3):
    - Maxwell: Spring and dashpot in series
    - Zener: Standard Linear Solid (SLS)
    - SpringPot: Fractional power-law element

Fractional Maxwell Family (4):
    - FractionalMaxwellGel: SpringPot in series with dashpot
    - FractionalMaxwellLiquid: Spring in series with SpringPot
    - FractionalMaxwellModel: Two SpringPots in series (most general)
    - FractionalKelvinVoigt: Spring and SpringPot in parallel

Fractional Zener Family (4):
    - FractionalZenerSolidLiquid: Fractional Maxwell + spring in parallel
    - FractionalZenerSolidSolid: Two springs + SpringPot
    - FractionalZenerLiquidLiquid: Most general fractional Zener
    - FractionalKelvinVoigtZener: FKV + spring in series

Fractional Advanced Models (3):
    - FractionalBurgersModel: Maxwell + FKV in series
    - FractionalPoyntingThomson: FKV + spring in series (alternate formulation)
    - FractionalJeffreysModel: Two dashpots + SpringPot

Non-Newtonian Flow Models (6, ROTATION test mode):
    - PowerLaw: Simple power-law model (K*gamma_dot^n)
    - Carreau: Smooth transition from Newtonian to power-law
    - CarreauYasuda: Extended Carreau with transition parameter
    - Cross: Alternative to Carreau for polymer solutions
    - HerschelBulkley: Power-law with yield stress
    - Bingham: Linear viscoplastic (yield stress + constant viscosity)

Multi-Mode Models (1):
    - GeneralizedMaxwell: Generalized Maxwell Model (Prony series, N modes)

Soft Glassy Rheology Models (2):
    - SGRConventional: Soft Glassy Rheology model (Sollich 1998)
    - SGRGeneric: GENERIC framework SGR (thermodynamically consistent)

STZ Models (1):
    - STZConventional: Shear Transformation Zone model (Langer 2008)

Fluidity Models (2):
    - FluidityLocal: Local (0D) fluidity model with aging/rejuvenation
    - FluidityNonlocal: Non-local (1D PDE) fluidity model with shear banding

Fluidity-Saramito EVP Models (2):
    - FluiditySaramitoLocal: Local (0D) tensorial EVP with thixotropic fluidity
    - FluiditySaramitoNonlocal: Nonlocal (1D) EVP for shear banding

SPP LAOS Models (1):
    - SPPYieldStress: Yield stress model for SPP LAOS analysis

Usage:
    >>> from rheojax.models import Maxwell, Zener, SpringPot
    >>> from rheojax.models import FractionalMaxwellGel, FractionalMaxwellLiquid
    >>> from rheojax.models import FractionalZenerSolidLiquid, FractionalBurgersModel
    >>> from rheojax.models import PowerLaw, Carreau, HerschelBulkley
    >>> from rheojax.models import SGRConventional, STZConventional
    >>> from rheojax.models import FluidityLocal, FluidityNonlocal
    >>> from rheojax.core.registry import ModelRegistry
    >>>
    >>> # Direct instantiation
    >>> model = Maxwell()
    >>> fzsl_model = FractionalZenerSolidLiquid()
    >>> flow_model = PowerLaw()
    >>> fluidity_model = FluidityLocal()
    >>>
    >>> # Factory pattern
    >>> model = ModelRegistry.create('maxwell')
    >>> fzsl_model = ModelRegistry.create('fractional_zener_sl')
    >>> flow_model = ModelRegistry.create('power_law')
    >>> fluidity_model = ModelRegistry.create('fluidity_local')
    >>>
    >>> # List available models
    >>> models = ModelRegistry.list_models()
"""

# Classical models
from rheojax.models.classical import Maxwell, SpringPot, Zener

# EPM models
from rheojax.models.epm import LatticeEPM, TensorialEPM

# Flow models
from rheojax.models.flow import (
    Bingham,
    Carreau,
    CarreauYasuda,
    Cross,
    HerschelBulkley,
    PowerLaw,
)

# Fluidity models
from rheojax.models.fluidity import FluidityLocal, FluidityNonlocal

# Fluidity-Saramito EVP models
from rheojax.models.fluidity.saramito import (
    FluiditySaramitoLocal,
    FluiditySaramitoNonlocal,
)

# Fractional models
from rheojax.models.fractional import (
    FBM,
    FJM,
    FKVZ,
    FPT,
    FZLL,
    FZSL,
    FZSS,
    FractionalBurgersModel,
    FractionalJeffreysModel,
    FractionalKelvinVoigt,
    FractionalKelvinVoigtZener,
    FractionalMaxwellGel,
    FractionalMaxwellLiquid,
    FractionalMaxwellModel,
    FractionalPoyntingThomson,
    FractionalZenerLiquidLiquid,
    FractionalZenerSolidLiquid,
    FractionalZenerSolidSolid,
)

# HL models
from rheojax.models.hl import HebraudLequeux

# IKH models
from rheojax.models.ikh import MIKH, MLIKH

# Multi-mode models
from rheojax.models.multimode import GeneralizedMaxwell

# SGR models
from rheojax.models.sgr import SGRConventional, SGRGeneric

# SPP models
from rheojax.models.spp import SPPYieldStress

# STZ models
from rheojax.models.stz import STZConventional

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
    # STZ models
    "STZConventional",
    # HL models
    "HebraudLequeux",
    # SPP Yield Stress model
    "SPPYieldStress",
    # EPM models
    "LatticeEPM",
    "TensorialEPM",
    # Fluidity models
    "FluidityLocal",
    "FluidityNonlocal",
    # Fluidity-Saramito EVP models
    "FluiditySaramitoLocal",
    "FluiditySaramitoNonlocal",
    # IKH models
    "MIKH",
    "MLIKH",
]
