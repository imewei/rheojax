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

# Lazy import map: public name -> (submodule, attribute_name)
# Each entry triggers the minimum imports needed on first access.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Classical models
    "Maxwell": ("rheojax.models.classical", "Maxwell"),
    "Zener": ("rheojax.models.classical", "Zener"),
    "SpringPot": ("rheojax.models.classical", "SpringPot"),
    # DMT thixotropic models
    "DMTLocal": ("rheojax.models.dmt", "DMTLocal"),
    "DMTNonlocal": ("rheojax.models.dmt", "DMTNonlocal"),
    # EPM models
    "LatticeEPM": ("rheojax.models.epm", "LatticeEPM"),
    "TensorialEPM": ("rheojax.models.epm", "TensorialEPM"),
    # FIKH (Fractional IKH) models
    "FIKH": ("rheojax.models.fikh", "FIKH"),
    "FMLIKH": ("rheojax.models.fikh", "FMLIKH"),
    # Flow models
    "Bingham": ("rheojax.models.flow", "Bingham"),
    "Carreau": ("rheojax.models.flow", "Carreau"),
    "CarreauYasuda": ("rheojax.models.flow", "CarreauYasuda"),
    "Cross": ("rheojax.models.flow", "Cross"),
    "HerschelBulkley": ("rheojax.models.flow", "HerschelBulkley"),
    "PowerLaw": ("rheojax.models.flow", "PowerLaw"),
    # Fluidity models
    "FluidityLocal": ("rheojax.models.fluidity", "FluidityLocal"),
    "FluidityNonlocal": ("rheojax.models.fluidity", "FluidityNonlocal"),
    # Fluidity-Saramito EVP models
    "FluiditySaramitoLocal": ("rheojax.models.fluidity.saramito", "FluiditySaramitoLocal"),
    "FluiditySaramitoNonlocal": ("rheojax.models.fluidity.saramito", "FluiditySaramitoNonlocal"),
    # Fractional models
    "FBM": ("rheojax.models.fractional", "FBM"),
    "FJM": ("rheojax.models.fractional", "FJM"),
    "FKVZ": ("rheojax.models.fractional", "FKVZ"),
    "FPT": ("rheojax.models.fractional", "FPT"),
    "FZLL": ("rheojax.models.fractional", "FZLL"),
    "FZSL": ("rheojax.models.fractional", "FZSL"),
    "FZSS": ("rheojax.models.fractional", "FZSS"),
    "FractionalBurgersModel": ("rheojax.models.fractional", "FractionalBurgersModel"),
    "FractionalJeffreysModel": ("rheojax.models.fractional", "FractionalJeffreysModel"),
    "FractionalKelvinVoigt": ("rheojax.models.fractional", "FractionalKelvinVoigt"),
    "FractionalKelvinVoigtZener": ("rheojax.models.fractional", "FractionalKelvinVoigtZener"),
    "FractionalMaxwellGel": ("rheojax.models.fractional", "FractionalMaxwellGel"),
    "FractionalMaxwellLiquid": ("rheojax.models.fractional", "FractionalMaxwellLiquid"),
    "FractionalMaxwellModel": ("rheojax.models.fractional", "FractionalMaxwellModel"),
    "FractionalPoyntingThomson": ("rheojax.models.fractional", "FractionalPoyntingThomson"),
    "FractionalZenerLiquidLiquid": ("rheojax.models.fractional", "FractionalZenerLiquidLiquid"),
    "FractionalZenerSolidLiquid": ("rheojax.models.fractional", "FractionalZenerSolidLiquid"),
    "FractionalZenerSolidSolid": ("rheojax.models.fractional", "FractionalZenerSolidSolid"),
    # Giesekus nonlinear viscoelastic models
    "GiesekusMultiMode": ("rheojax.models.giesekus", "GiesekusMultiMode"),
    "GiesekusSingleMode": ("rheojax.models.giesekus", "GiesekusSingleMode"),
    # HL models
    "HebraudLequeux": ("rheojax.models.hl", "HebraudLequeux"),
    # HVM (Hybrid Vitrimer Model) models
    "HVMLocal": ("rheojax.models.hvm", "HVMLocal"),
    # HVNM (Hybrid Vitrimer Nanocomposite Model) models
    "HVNMLocal": ("rheojax.models.hvnm", "HVNMLocal"),
    # IKH models
    "MIKH": ("rheojax.models.ikh", "MIKH"),
    "MLIKH": ("rheojax.models.ikh", "MLIKH"),
    # ITT-MCT models
    "ITTMCTIsotropic": ("rheojax.models.itt_mct", "ITTMCTIsotropic"),
    "ITTMCTSchematic": ("rheojax.models.itt_mct", "ITTMCTSchematic"),
    # Multi-mode models
    "GeneralizedMaxwell": ("rheojax.models.multimode", "GeneralizedMaxwell"),
    # SGR models
    "SGRConventional": ("rheojax.models.sgr", "SGRConventional"),
    "SGRGeneric": ("rheojax.models.sgr", "SGRGeneric"),
    # SPP models
    "SPPYieldStress": ("rheojax.models.spp", "SPPYieldStress"),
    # STZ models
    "STZConventional": ("rheojax.models.stz", "STZConventional"),
    # TNT (Transient Network Theory) models
    "TNTCates": ("rheojax.models.tnt", "TNTCates"),
    "TNTLoopBridge": ("rheojax.models.tnt", "TNTLoopBridge"),
    "TNTMultiSpecies": ("rheojax.models.tnt", "TNTMultiSpecies"),
    "TNTSingleMode": ("rheojax.models.tnt", "TNTSingleMode"),
    "TNTStickyRouse": ("rheojax.models.tnt", "TNTStickyRouse"),
    # VLB (Vernerey-Long-Brighenti) transient network models
    "VLBLocal": ("rheojax.models.vlb", "VLBLocal"),
    "VLBMultiNetwork": ("rheojax.models.vlb", "VLBMultiNetwork"),
    "VLBNonlocal": ("rheojax.models.vlb", "VLBNonlocal"),
    "VLBVariant": ("rheojax.models.vlb", "VLBVariant"),
}


def __getattr__(name: str):
    """Lazy-load model classes on first access.

    Model registration decorators (@ModelRegistry.register) fire when the
    model's submodule is first imported here, so the registry is populated
    on demand rather than all at startup.
    """
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        obj = getattr(module, attr)
        # Cache in module globals so subsequent access is a direct dict lookup
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
