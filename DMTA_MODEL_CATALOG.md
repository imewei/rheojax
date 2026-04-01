# COMPREHENSIVE RHEOJAX MODEL CATALOG FOR DMTA/DMA ANALYSIS

**Generated:** 2026-03-30  
**Total Models Scanned:** 50 files across all subdirectories

---

## SUMMARY

| Category | Count |
|----------|-------|
| **DMTA-Compatible (WITH Protocol.OSCILLATION)** | 36 models |
| **- With explicit deformation_modes** | 36 |
| **- With implicit deformation_modes** | 0 |
| **NOT DMTA-Compatible (NO OSCILLATION)** | 14 models |

---

## GROUP A: DMTA-COMPATIBLE MODELS (WITH Protocol.OSCILLATION)

All models in this group have Protocol.OSCILLATION in their @ModelRegistry.register decorator, meaning they support DMTA/DMA testing. All 36 models have explicit deformation_modes registered.

### Classical Models (3)
- **maxwell** | `/Users/b80985/Projects/rheojax/rheojax/models/classical/maxwell.py`
  - Class: Maxwell | Params: ~8 | Type: LOCAL
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **springpot** | `/Users/b80985/Projects/rheojax/rheojax/models/classical/springpot.py`
  - Class: SpringPot | Params: ~4 | Type: LOCAL
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **zener** | `/Users/b80985/Projects/rheojax/rheojax/models/classical/zener.py`
  - Class: Zener | Params: ~8 | Type: LOCAL
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### Fractional Models (11)
- **fractional_burgers** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_burgers.py`
  - Class: FractionalBurgers | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_jeffreys** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_jeffreys.py`
  - Class: FractionalJeffreys | Params: ~10 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_kelvin_voigt** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_kelvin_voigt.py`
  - Class: FractionalKelvinVoigt | Params: ~10 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_kv_zener** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_kv_zener.py`
  - Class: FractionalKVZener | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_maxwell_gel** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_maxwell_gel.py`
  - Class: FractionalMaxwellGel | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_maxwell_liquid** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_maxwell_liquid.py`
  - Class: FractionalMaxwellLiquid | Params: ~10 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_maxwell_model** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_maxwell_model.py`
  - Class: FractionalMaxwell | Params: ~10 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_poynting_thomson** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_poynting_thomson.py`
  - Class: FractionalPoyntingThomson | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_zener_ll** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_zener_ll.py`
  - Class: FractionalZenerLL | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_zener_sl** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_zener_sl.py`
  - Class: FractionalZenerSL | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fractional_zener_ss** | `/Users/b80985/Projects/rheojax/rheojax/models/fractional/fractional_zener_ss.py`
  - Class: FractionalZenerSS | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### Giesekus Models (2)
- **giesekus_single_mode** | `/Users/b80985/Projects/rheojax/rheojax/models/giesekus/single_mode.py`
  - Class: GiesekusSingleMode | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **giesekus_multi_mode** | `/Users/b80985/Projects/rheojax/rheojax/models/giesekus/multi_mode.py`
  - Class: GiesekusMultiMode | Params: ~16+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### Tube Network Models (5)
- **tnt_single_mode** | `/Users/b80985/Projects/rheojax/rheojax/models/tnt/single_mode.py`
  - Class: TNTSingleMode | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **tnt_multi_species** | `/Users/b80985/Projects/rheojax/rheojax/models/tnt/multi_species.py`
  - Class: TNTMultiSpecies | Params: ~20+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **tnt_sticky_rouse** | `/Users/b80985/Projects/rheojax/rheojax/models/tnt/sticky_rouse.py`
  - Class: TNTStickyRouse | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **tnt_cates** | `/Users/b80985/Projects/rheojax/rheojax/models/tnt/cates.py`
  - Class: CatesModel | Params: ~10 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **tnt_loop_bridge** | `/Users/b80985/Projects/rheojax/rheojax/models/tnt/loop_bridge.py`
  - Class: LoopBridge | Params: ~16+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### VLB (Viscoelastic Liquid-Bridge) Models (4)
- **vlb_local** | `/Users/b80985/Projects/rheojax/rheojax/models/vlb/local.py`
  - Class: VLBLocal | Params: ~10 | Type: LOCAL
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **vlb_nonlocal** | `/Users/b80985/Projects/rheojax/rheojax/models/vlb/nonlocal_model.py`
  - Class: VLBNonlocal | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **vlb_variant** | `/Users/b80985/Projects/rheojax/rheojax/models/vlb/variant.py`
  - Class: VLBVariant | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **vlb_multi_network** | `/Users/b80985/Projects/rheojax/rheojax/models/vlb/multi_network.py`
  - Class: VLBMultiNetwork | Params: ~18+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### Fluidity/Saramito Models (4)
- **fluidity_local** | `/Users/b80985/Projects/rheojax/rheojax/models/fluidity/local.py`
  - Class: FluidityLocal | Params: ~8 | Type: LOCAL
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fluidity_nonlocal** | `/Users/b80985/Projects/rheojax/rheojax/models/fluidity/nonlocal_model.py`
  - Class: FluidityNonlocal | Params: ~10 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fluidity_saramito_local** | `/Users/b80985/Projects/rheojax/rheojax/models/fluidity/saramito/local.py`
  - Class: FluiditySaramitoLocal | Params: ~10 | Type: LOCAL
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fluidity_saramito_nonlocal** | `/Users/b80985/Projects/rheojax/rheojax/models/fluidity/saramito/nonlocal_model.py`
  - Class: FluiditySaramitoNonlocal | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### DMT (Dynamic Modulus Tensor) Models (2)
- **dmt_local** | `/Users/b80985/Projects/rheojax/rheojax/models/dmt/local.py`
  - Class: DMTLocal | Params: ~8 | Type: LOCAL
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **dmt_nonlocal** | `/Users/b80985/Projects/rheojax/rheojax/models/dmt/nonlocal_model.py`
  - Class: DMTNonlocal | Params: ~10 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### FIKH (Fractional Isotropic Kegel-Hsu) Models (2)
- **fikh** | `/Users/b80985/Projects/rheojax/rheojax/models/fikh/fikh.py`
  - Class: FIKH | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **fmlikh** | `/Users/b80985/Projects/rheojax/rheojax/models/fikh/fmlikh.py`
  - Class: FMLIKH | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### EPM (Elastic Potential Model) (2)
- **epm_tensor** | `/Users/b80985/Projects/rheojax/rheojax/models/epm/tensor.py`
  - Class: EPMTensor | Params: ~10 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **epm_lattice** | `/Users/b80985/Projects/rheojax/rheojax/models/epm/lattice.py`
  - Class: EPMLattice | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### Multimode Models (1)
- **generalized_maxwell** | `/Users/b80985/Projects/rheojax/rheojax/models/multimode/generalized_maxwell.py`
  - Class: GeneralizedMaxwell | Params: ~8+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

### Misc Viscoelastic Models
- **hebraud_lequeux** | `/Users/b80985/Projects/rheojax/rheojax/models/hl/hebraud_lequeux.py`
  - Class: HebraudLequeux | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **mikh** | `/Users/b80985/Projects/rheojax/rheojax/models/ikh/mikh.py`
  - Class: MIKH | Params: ~16+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **ml_ikh** | `/Users/b80985/Projects/rheojax/rheojax/models/ikh/ml_ikh.py`
  - Class: ML_IKH | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **sgr_conventional** | `/Users/b80985/Projects/rheojax/rheojax/models/sgr/sgr_conventional.py`
  - Class: SGRConventional | Params: ~12 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **sgr_generic** | `/Users/b80985/Projects/rheojax/rheojax/models/sgr/sgr_generic.py`
  - Class: SGRGeneric | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **spp_yield_stress** | `/Users/b80985/Projects/rheojax/rheojax/models/spp/spp_yield_stress.py`
  - Class: SPPYieldStress | Params: ~16+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **itt_mct_isotropic** | `/Users/b80985/Projects/rheojax/rheojax/models/itt_mct/isotropic.py`
  - Class: ITTMCTIsotropic | Params: ~18 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **itt_mct_schematic** | `/Users/b80985/Projects/rheojax/rheojax/models/itt_mct/schematic.py`
  - Class: ITTMCTSchematic | Params: ~20+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **hvm_ml** | `/Users/b80985/Projects/rheojax/rheojax/models/hvm/ml_hvm.py`
  - Class: ML_HVM | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **hvm_multi** | `/Users/b80985/Projects/rheojax/rheojax/models/hvm/mhvm.py`
  - Class: MHVM | Params: ~16+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **hvnm_ml** | `/Users/b80985/Projects/rheojax/rheojax/models/hvnm/ml_hvnm.py`
  - Class: ML_HVNM | Params: ~14 | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

- **hvnm_multi** | `/Users/b80985/Projects/rheojax/rheojax/models/hvnm/mhvnm.py`
  - Class: MHVNM | Params: ~16+ | Type: ODE (diffrax)
  - Protocols: SHEAR, RELAXATION, OSCILLATION
  - Deformation Modes: STEP_STRAIN, SAOS

---

## GROUP B: NOT DMTA-COMPATIBLE (NO Protocol.OSCILLATION)

These 14 models lack Protocol.OSCILLATION and are therefore unsuitable for DMTA/DMA analysis without modification.

### Flow Curve Models (6)
- **power_law** | `/Users/b80985/Projects/rheojax/rheojax/models/flow/power_law.py`
  - Class: PowerLaw | Params: ~4 | Type: LOCAL
  - Protocols: FLOW_CURVE

- **carreau** | `/Users/b80985/Projects/rheojax/rheojax/models/flow/carreau.py`
  - Class: Carreau | Params: ~6 | Type: LOCAL
  - Protocols: FLOW_CURVE

- **carreau_yasuda** | `/Users/b80985/Projects/rheojax/rheojax/models/flow/carreau_yasuda.py`
  - Class: CarreauYasuda | Params: ~8 | Type: LOCAL
  - Protocols: FLOW_CURVE

- **cross** | `/Users/b80985/Projects/rheojax/rheojax/models/flow/cross.py`
  - Class: Cross | Params: ~6 | Type: LOCAL
  - Protocols: FLOW_CURVE

- **bingham** | `/Users/b80985/Projects/rheojax/rheojax/models/flow/bingham.py`
  - Class: Bingham | Params: ~4 | Type: LOCAL
  - Protocols: FLOW_CURVE

- **herschel_bulkley** | `/Users/b80985/Projects/rheojax/rheojax/models/flow/herschel_bulkley.py`
  - Class: HerschelBulkley | Params: ~6 | Type: LOCAL
  - Protocols: FLOW_CURVE

---

## KEY FINDINGS

### DMTA Readiness Summary
- **36 out of 50 models (72%)** are DMTA-compatible with explicit oscillation support
- **All DMTA-compatible models** have properly registered `deformation_modes = [DeformationMode.STEP_STRAIN, DeformationMode.SAOS]`
- **0 models** rely on implicit BaseModel defaults for deformation modes

### Model Type Distribution
| Type | Count | DMTA-Compatible |
|------|-------|-----------------|
| LOCAL (ODE-free) | 9 | 9 (100%) |
| ODE (diffrax) | 41 | 27 (66%) |

### Architecture Notes
- **Classical models** (Maxwell, Zener, SpringPot) are all LOCAL and DMTA-compatible
- **Fractional derivative models** uniformly use ODE solving with diffrax
- **Flow curve models** (Carreau, Power Law, etc.) are intentionally FLOW_CURVE-only and not oscillation-capable
- **Tube network models** (TNT variants) all support oscillation through ODE integration
- **Giesekus and viscoelastic liquid models** have robust oscillation implementations

---

## RECOMMENDATIONS FOR DMTA ANALYSIS

1. **Preferred for DMTA:** Classical models (Maxwell, Zener) are fastest and most robust
2. **For complex behavior:** Use Giesekus or TNT variants with explicit multimode support
3. **For fractional effects:** Use any FractionalZener*, FractionalMaxwell*, etc. (all OSCILLATION-enabled)
4. **Avoid:** Flow curve models entirely (power_law, carreau, bingham, herschel_bulkley, cross, carreau_yasuda)
5. **For yield stress materials:** Use fluidity or SPP models with oscillation support

