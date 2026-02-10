"""Rheological models package.

This package contains 53 rheological models organized into 22 categories.

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

DMT Thixotropic Models (2):
    - DMTLocal: Local (0D) de Souza Mendes-Thompson thixotropic model
    - DMTNonlocal: Nonlocal (1D) DMT model for shear banding

SPP LAOS Models (1):
    - SPPYieldStress: Yield stress model for SPP LAOS analysis

ITT-MCT Models (2):
    - ITTMCTSchematic: F₁₂ schematic model (scalar correlator)
    - ITTMCTIsotropic: Isotropically sheared model with k-resolved S(k)

FIKH (Fractional IKH) Models (2):
    - FIKH: Fractional IKH with Caputo derivative and thermal coupling
    - FMLIKH: Multi-layer variant with per-mode parameters

Giesekus Nonlinear Viscoelastic Models (2):
    - GiesekusSingleMode: Single-mode Giesekus with shear-thinning and normal stresses
    - GiesekusMultiMode: Multi-mode extension with N relaxation times

Transient Network Theory (TNT) Models (5):
    - TNTSingleMode: Composable single-mode network model (basic, Bell, FENE, NonAffine)
    - TNTLoopBridge: Two-species loop-bridge kinetics for telechelic polymers
    - TNTStickyRouse: Multi-mode with sticker kinetics for ionomers
    - TNTCates: Living polymers (wormlike micelles) with reptation + breakage
    - TNTMultiSpecies: Multiple bond types with different lifetimes

VLB (Vernerey-Long-Brighenti) Transient Network Models (4):
    - VLBLocal: Single transient network with molecular-statistical foundation
    - VLBMultiNetwork: Multi-network VLB with N transient + permanent + solvent
    - VLBVariant: Composable Bell + FENE-P + Temperature extensions
    - VLBNonlocal: Spatial PDE with tensor diffusion for shear banding

IKH (Isotropic Kinematic Hardening) Models (2):
    - MIKH: Modified IKH model for thixotropic yield stress fluids
    - MLIKH: Multi-layer IKH with per-layer structure parameters

HVM (Hybrid Vitrimer Model) (1):
    - HVMLocal: Local (0D) hybrid vitrimer model (P + E + D networks)

HVNM (Hybrid Vitrimer Nanocomposite Model) (1):
    - HVNMLocal: Local (0D) NP-filled vitrimer model (P + E + D + I networks)

Hébraud-Lequeux Model (1):
    - HebraudLequeux: Stochastic trap model for soft glassy materials

Elasto-Plastic Mesoscopic (EPM) Models (2):
    - LatticeEPM: Lattice-based elasto-plastic model for amorphous solids
    - TensorialEPM: Tensorial EPM with full stress tensor evolution

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

# DMT thixotropic models
from rheojax.models.dmt import DMTLocal, DMTNonlocal

# EPM models
from rheojax.models.epm import LatticeEPM, TensorialEPM

# FIKH (Fractional IKH) models
from rheojax.models.fikh import FIKH, FMLIKH

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

# Giesekus nonlinear viscoelastic models
from rheojax.models.giesekus import GiesekusMultiMode, GiesekusSingleMode

# HL models
from rheojax.models.hl import HebraudLequeux

# HVM (Hybrid Vitrimer Model) models
from rheojax.models.hvm import HVMLocal

# HVNM (Hybrid Vitrimer Nanocomposite Model) models
from rheojax.models.hvnm import HVNMLocal

# IKH models
from rheojax.models.ikh import MIKH, MLIKH

# ITT-MCT models
from rheojax.models.itt_mct import ITTMCTIsotropic, ITTMCTSchematic

# Multi-mode models
from rheojax.models.multimode import GeneralizedMaxwell

# SGR models
from rheojax.models.sgr import SGRConventional, SGRGeneric

# SPP models
from rheojax.models.spp import SPPYieldStress

# STZ models
from rheojax.models.stz import STZConventional

# TNT (Transient Network Theory) models
from rheojax.models.tnt import (
    TNTCates,
    TNTLoopBridge,
    TNTMultiSpecies,
    TNTSingleMode,
    TNTStickyRouse,
)

# VLB (Vernerey-Long-Brighenti) transient network models
from rheojax.models.vlb import VLBLocal, VLBMultiNetwork, VLBNonlocal, VLBVariant

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
    # HVM (Hybrid Vitrimer Model) models
    "HVMLocal",
    # HVNM (Hybrid Vitrimer Nanocomposite Model) models
    "HVNMLocal",
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
    # FIKH (Fractional IKH) models
    "FIKH",
    "FMLIKH",
    # ITT-MCT models
    "ITTMCTSchematic",
    "ITTMCTIsotropic",
    # DMT thixotropic models
    "DMTLocal",
    "DMTNonlocal",
    # Giesekus nonlinear viscoelastic models
    "GiesekusSingleMode",
    "GiesekusMultiMode",
    # TNT (Transient Network Theory) models
    "TNTSingleMode",
    "TNTLoopBridge",
    "TNTStickyRouse",
    "TNTCates",
    "TNTMultiSpecies",
    # VLB (Vernerey-Long-Brighenti) transient network models
    "VLBLocal",
    "VLBMultiNetwork",
    "VLBVariant",
    "VLBNonlocal",
]
