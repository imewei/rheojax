# ITT-MCT Tutorial Notebooks

Integration Through Transients Mode-Coupling Theory (ITT-MCT) models for dense colloidal suspensions and glassy materials.

## Overview

This tutorial series covers 12 notebooks demonstrating ITT-MCT models across 6 rheological protocols:
- **Schematic F₁₂ Model** (NB01-NB06): Fast, qualitative model with single-mode correlator
- **Isotropic ISM Model** (NB07-NB12): Full k-resolved model with Percus-Yevick S(k)

## Quick Start

```python
from rheojax.models.itt_mct import ITTMCTSchematic, ITTMCTIsotropic

# Schematic F₁₂ model
model = ITTMCTSchematic(epsilon=0.05)  # Glass state (ε > 0)
model.fit(gamma_dot, stress, test_mode='flow_curve')

# Isotropic model with structure factor
model_ism = ITTMCTIsotropic(phi=0.55)  # Above φ_MCT = 0.516
compile_time = model_ism.precompile()  # First-time JIT compilation
model_ism.fit(gamma_dot, stress, test_mode='flow_curve')
```

## Notebook Dependencies

```
NB01 (Schematic Flow Curve) ──┬──► NB03 (Relaxation, synthetic)
                              ├──► NB04 (Creep, synthetic)
                              └──► NB05 (SAOS, synthetic)

NB02 (Schematic Startup) ────► Independent (real data)
NB06 (Schematic LAOS) ───────► Independent (real data)

NB07 (ISM Flow Curve) ───────┬──► NB09 (Relaxation, synthetic)
                             ├──► NB10 (Creep, synthetic)
                             └──► NB11 (SAOS, synthetic)

NB08 (ISM Startup) ──────────► Independent (real data)
NB12 (ISM LAOS) ─────────────► Independent (real data)
```

## Notebooks

### Schematic F₁₂ Model (Fast, Qualitative)

| Notebook | Protocol | Data Source | Runtime |
|----------|----------|-------------|---------|
| **01_schematic_flow_curve** | Flow curve | Real (ML-IKH) | ~2-3 min |
| **02_schematic_startup_shear** | Startup | Real (PNAS) | ~2-3 min |
| **03_schematic_stress_relaxation** | Relaxation | Synthetic | ~2-3 min |
| **04_schematic_creep** | Creep | Synthetic | ~2-3 min |
| **05_schematic_saos** | SAOS | Synthetic | ~2-3 min |
| **06_schematic_laos** | LAOS | Real (PNAS) | ~3-5 min |

### Isotropic ISM Model (Full k-Resolved)

| Notebook | Protocol | Data Source | Runtime |
|----------|----------|-------------|---------|
| **07_isotropic_flow_curve** | Flow curve | Real (ML-IKH) | ~3-5 min |
| **08_isotropic_startup_shear** | Startup | Real (PNAS) | ~3-5 min |
| **09_isotropic_stress_relaxation** | Relaxation | Synthetic | ~3-5 min |
| **10_isotropic_creep** | Creep | Synthetic | ~3-5 min |
| **11_isotropic_saos** | SAOS | Synthetic | ~3-5 min |
| **12_isotropic_laos** | LAOS | Real (PNAS) | ~5-8 min |

## Model Comparison

| Aspect | Schematic F₁₂ | Isotropic ISM |
|--------|---------------|---------------|
| Correlator | Single Φ(t) | k-resolved Φ(k,t) |
| Control parameter | v₂ vertex | φ volume fraction |
| Glass transition | v₂ = 4 | φ_MCT ≈ 0.516 |
| Structure factor | None | Percus-Yevick S(k) |
| Stress integral | G∞ × single mode | k_BT/σ³ × k-integral |
| Accuracy | Qualitative | Quantitative |
| Speed | Fast | Slower (precompile helps) |

## Key Physics

### Glass Transition

- **Fluid state**: Complete relaxation, σ(∞) = 0
- **Glass state**: Arrested dynamics, σ(∞) > 0

### Non-Ergodicity Parameter f

The long-time limit of the correlator in the glass state:
- f = 0: Fluid (ergodic)
- 0 < f < 1: Glass (non-ergodic)

### Two-Step Relaxation

1. **β-process** (fast): Cage vibration, time scale τ_β ~ 1/Γ
2. **α-process** (slow): Cage rearrangement, τ_α → ∞ in glass

### Strain Decorrelation

Cage breaking under strain: h(γ) = exp(-(γ/γc)²)

## Parameters

### Schematic F₁₂

| Parameter | Symbol | Physical Meaning | Typical Range |
|-----------|--------|------------------|---------------|
| v1 | v₁ | Linear vertex (usually 0) | 0 |
| v2 | v₂ | Quadratic vertex | 3.5 - 5.0 |
| Gamma | Γ | Bare relaxation rate | 0.1 - 10 s⁻¹ |
| gamma_c | γc | Critical strain | 0.05 - 0.3 |
| G_inf | G∞ | High-frequency modulus | 10 - 10000 Pa |

### Isotropic ISM

| Parameter | Symbol | Physical Meaning | Typical Range |
|-----------|--------|------------------|---------------|
| phi | φ | Volume fraction | 0.3 - 0.64 |
| sigma_d | σ | Particle diameter | 10⁻⁹ - 10⁻³ m |
| D0 | D₀ | Bare diffusion | 10⁻¹⁸ - 10⁻⁶ m²/s |
| kBT | k_BT | Thermal energy | ~4×10⁻²¹ J (300K) |
| gamma_c | γc | Critical strain | 0.05 - 0.3 |

## Configuration

### Fast Demo (Default)

```python
NUM_WARMUP = 200
NUM_SAMPLES = 500
NUM_CHAINS = 1
```

Runtime: 2-5 minutes per notebook

### Production Quality

```python
NUM_WARMUP = 1000
NUM_SAMPLES = 2000
NUM_CHAINS = 4
```

Runtime: 10-20 minutes per notebook

## Troubleshooting

### Precompilation Takes Long

First-time JIT compilation for ISM can take 30-90 seconds. Use `model.precompile()` to trigger this before fitting.

### Glass State Not Detected

Ensure v₂ > 4 (Schematic) or φ > 0.516 (ISM) for glass state. Check with `model.get_glass_transition_info()`.

### Convergence Issues

Use NLSQ warm-start for Bayesian inference:

```python
# Fit NLSQ first
model.fit(x, y, test_mode='flow_curve')

# Use fitted values as initial conditions
initial_values = {name: model.parameters.get_value(name) for name in param_names}
result = model.fit_bayesian(..., initial_values=initial_values)
```

## References

1. Götze, W. (2009). Complex Dynamics of Glass-Forming Liquids. Oxford University Press.
2. Brader, J. M. et al. (2008). First-principles constitutive equation for suspension rheology. Phys. Rev. E.
3. Fuchs, M. & Cates, M. E. (2002). Theory of nonlinear rheology and yielding of dense colloidal suspensions. Phys. Rev. Lett.

## Learning Path

1. **Start here**: `01_schematic_flow_curve.ipynb` - Calibrate parameters
2. **Transient protocols**: `02_startup`, `03_relaxation`, `04_creep`
3. **Oscillatory**: `05_saos`, `06_laos`
4. **Full k-resolved**: `07_isotropic_flow_curve.ipynb` and continue
