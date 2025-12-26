# RheoJAX Model Test Mode Compatibility

This document provides a comprehensive overview of which rheological models support each test mode in RheoJAX.

## Test Modes Overview

| Test Mode | Domain | Description |
|-----------|--------|-------------|
| **relaxation** | Time | Stress relaxation under constant strain (decreasing response) |
| **creep** | Time | Strain development under constant stress (increasing response) |
| **oscillation** | Frequency | Small amplitude oscillatory shear (SAOS) |
| **rotation** | Steady | Steady shear flow |

---

## Compatibility by Test Mode

### Relaxation (Time domain - stress relaxation under constant strain)

| Category | Models |
|----------|--------|
| Classical | Maxwell, Zener, SpringPot |
| Multi-Mode | GeneralizedMaxwell |
| Frac. Maxwell | FractionalMaxwellModel, FractionalMaxwellLiquid, FractionalMaxwellGel, FractionalKelvinVoigt |
| Frac. Zener | FractionalZenerSolidSolid, FractionalZenerSolidLiquid, FractionalZenerLiquidLiquid, FractionalKelvinVoigtZener, FractionalPoyntingThomson |
| Frac. Advanced | FractionalBurgersModel, FractionalJeffreysModel |
| **Total** | **15 models** |

### Creep (Time domain - strain under constant stress)

| Category | Models |
|----------|--------|
| Classical | Maxwell, Zener, SpringPot |
| Multi-Mode | GeneralizedMaxwell (numerical) |
| Frac. Maxwell | FractionalMaxwellModel, FractionalMaxwellLiquid, FractionalMaxwellGel, FractionalKelvinVoigt |
| Frac. Zener | FractionalZenerSolidSolid, FractionalZenerSolidLiquid, FractionalZenerLiquidLiquid, FractionalKelvinVoigtZener, FractionalPoyntingThomson |
| Frac. Advanced | FractionalBurgersModel, FractionalJeffreysModel |
| **Total** | **15 models** |

### Oscillation (Frequency domain - SAOS)

| Category | Models |
|----------|--------|
| Classical | Maxwell, Zener, SpringPot |
| Multi-Mode | GeneralizedMaxwell |
| Frac. Maxwell | FractionalMaxwellModel, FractionalMaxwellLiquid, FractionalMaxwellGel, FractionalKelvinVoigt |
| Frac. Zener | FractionalZenerSolidSolid, FractionalZenerSolidLiquid, FractionalZenerLiquidLiquid, FractionalKelvinVoigtZener, FractionalPoyntingThomson |
| Frac. Advanced | FractionalBurgersModel, FractionalJeffreysModel |
| SGR | SGRConventional, SGRGeneric |
| SPP | SPPYieldStress |
| **Total** | **18 models** |

### Rotation (Steady shear flow)

| Category | Models |
|----------|--------|
| Classical | Maxwell, Zener |
| Flow (dedicated) | PowerLaw, Bingham, HerschelBulkley, Carreau, Cross, CarreauYasuda |
| Frac. Advanced | FractionalJeffreysModel |
| SPP | SPPYieldStress |
| **Partial** | FractionalZenerLiquidLiquid, FractionalBurgersModel (power-law at high rates) |
| **Total** | **8-10 models** |

---

## Quick Reference Matrix

| Model | Relaxation | Creep | Oscillation | Rotation |
|-------|:----------:|:-----:|:-----------:|:--------:|
| Maxwell | ✓ | ✓ | ✓ | ✓ |
| Zener | ✓ | ✓ | ✓ | ✓ |
| SpringPot | ✓ | ✓ | ✓ | ✗ |
| PowerLaw | ✗ | ✗ | ✗ | ✓ |
| Bingham | ✗ | ✗ | ✗ | ✓ |
| HerschelBulkley | ✗ | ✗ | ✗ | ✓ |
| Carreau | ✗ | ✗ | ✗ | ✓ |
| Cross | ✗ | ✗ | ✗ | ✓ |
| CarreauYasuda | ✗ | ✗ | ✗ | ✓ |
| GeneralizedMaxwell | ✓ | ✓ | ✓ | ✗ |
| FractionalMaxwellModel | ✓ | ✓ | ✓ | ✗ |
| FractionalMaxwellLiquid | ✓ | ✓ | ✓ | ✗ |
| FractionalMaxwellGel | ✓ | ✓ | ✓ | ✗ |
| FractionalKelvinVoigt | ✓ | ✓ | ✓ | ✗ |
| FractionalZenerSolidSolid | ✓ | ✓ | ✓ | ✗ |
| FractionalZenerSolidLiquid | ✓ | ✓ | ✓ | ✗ |
| FractionalZenerLiquidLiquid | ✓ | ✓ | ✓ | ~ |
| FractionalKelvinVoigtZener | ✓ | ✓ | ✓ | ✗ |
| FractionalPoyntingThomson | ✓ | ✓ | ✓ | ✗ |
| FractionalBurgersModel | ✓ | ✓ | ✓ | ~ |
| FractionalJeffreysModel | ✓ | ✓ | ✓ | ✓ |
| SGRConventional | ✗ | ✗ | ✓ | ✗ |
| SGRGeneric | ✗ | ✗ | ✓ | ✗ |
| SPPYieldStress | ✗ | ✗ | ✓ | ✓ |

**Legend:** ✓ = Full support | ~ = Partial support | ✗ = Not supported

---

## Model Categories

### Classical Viscoelastic (3 models)
- **Maxwell**: Basic 2-parameter viscoelastic element (G0, eta)
- **Zener**: Standard Linear Solid with equilibrium spring (Ge, Gm, eta)
- **SpringPot**: Fractional power-law element (c_alpha, alpha)

### Flow Models (6 models)
All rotation-only models for steady shear characterization:
- **PowerLaw**: σ = K|γ̇|^n
- **Bingham**: Linear viscoplastic with yield stress
- **HerschelBulkley**: Power-law viscoplastic
- **Carreau**: Smooth Newtonian-to-power-law transition
- **Cross**: Alternative to Carreau functional form
- **CarreauYasuda**: Extended Carreau with transition parameter

### Multi-Mode (1 model)
- **GeneralizedMaxwell**: Prony series with N modes (E_inf, E_i, tau_i)

### Fractional Maxwell Family (4 models)
- **FractionalMaxwellModel**: Two SpringPots in series (most general)
- **FractionalMaxwellLiquid**: Spring + SpringPot in series
- **FractionalMaxwellGel**: SpringPot + dashpot in series
- **FractionalKelvinVoigt**: Spring + SpringPot in parallel

### Fractional Zener Family (5 models)
- **FractionalZenerSolidSolid (FZSS)**: Two springs + SpringPot
- **FractionalZenerSolidLiquid (FZSL)**: Spring + Fractional Maxwell
- **FractionalZenerLiquidLiquid (FZLL)**: Most general fractional Zener
- **FractionalKelvinVoigtZener (FKVZ)**: Spring in series with Fractional KV
- **FractionalPoyntingThomson (FPT)**: Identical math to FKVZ, different interpretation

### Fractional Advanced (2 models)
- **FractionalBurgersModel**: Four-parameter with instantaneous, viscous, and retardation
- **FractionalJeffreysModel**: Fractional viscoelastic liquid (only fractional with full rotation)

### Soft Glassy Rheology (2 models)
- **SGRConventional**: Sollich 1998 statistical mechanics (foams, emulsions, pastes)
- **SGRGeneric**: GENERIC thermodynamic framework (Fuereder & Ilg 2013)

### SPP Yield Stress (1 model)
- **SPPYieldStress**: Sequence of Physical Processes for LAOS amplitude sweeps

---

## Implementation Notes

All models follow a consistent architecture:
1. Store `_test_mode` attribute during `_fit()`
2. Implement separate `_predict_*()` static methods for each supported test mode
3. Dispatch in `_predict()` and `model_function()` based on test mode
4. Handle RheoData auto-detection via `detect_test_mode(rheo_data)`
5. Validate test mode compatibility (unsupported modes raise `ValueError`)

The Bayesian inference layer receives an explicit `test_mode` parameter to ensure closure-based capture of the correct mode (v0.4.0 critical fix).
