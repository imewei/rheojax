# Model Selection Guide

This guide helps you choose the appropriate rheological model based on your experimental data characteristics and material type.

## Quick Selection Flowchart

```
Data Type?
├─ Time Domain (Relaxation/Creep)
│  ├─ Decay Type?
│  │  ├─ Exponential decay → Maxwell, Zener
│  │  ├─ Power-law decay → FractionalMaxwellGel, FZSS
│  │  └─ Finite equilibrium modulus → Zener, FZSS, FKV
│  └─ Material Type?
│     ├─ Liquid-like (flows) → Maxwell, FractionalMaxwellLiquid
│     ├─ Solid-like (elastic) → Zener, FZSS, FractionalKelvinVoigt
│     └─ Gel-like → FractionalMaxwellGel
└─ Frequency Domain (Oscillation)
   ├─ Low-frequency behavior?
   │  ├─ G' > G" → Solid-like models (FZSS, FKV, Zener)
   │  └─ G" > G' → Liquid-like models (Maxwell, FML)
   └─ Slope in log-log plot?
      ├─ ~2 (liquid) → Maxwell, FML
      ├─ ~0.5 (gel) → FractionalMaxwellGel
      └─ plateau (solid) → FZSS, Zener, FKV
```

## Model Categories

### Classical Viscoelastic Models

Models using integer-order derivatives (ideal for simple materials).

#### Maxwell Model
**When to use:**
- Simple liquid-like materials that fully relax
- Exponential stress relaxation: G(t) = G₀ exp(-t/τ)
- Polymer solutions, simple fluids
- G" > G' at low frequencies

**Characteristics:**
- **Decay type**: Exponential
- **Equilibrium modulus**: Zero (flows under constant stress)
- **Parameters**: G₀ (modulus), η (viscosity)
- **Typical τ range**: 0.001-100 s

**Don't use when:**
- Material has finite equilibrium modulus
- Relaxation shows power-law behavior
- Multiple relaxation modes present

---

#### Zener (Standard Linear Solid)
**When to use:**
- Viscoelastic solids with finite equilibrium modulus
- Exponential relaxation with plateau
- Filled polymers, soft solids
- G' > G" at low frequencies

**Characteristics:**
- **Decay type**: Exponential with plateau
- **Equilibrium modulus**: Finite (Ge > 0)
- **Parameters**: Ge (equilibrium), Gm (Maxwell arm), τ (relaxation time)
- **Typical applications**: Rubber, gels with crosslinks

**Don't use when:**
- Material shows power-law relaxation
- No clear equilibrium modulus
- Complex multi-mode behavior

### Fractional Viscoelastic Models

Models using fractional derivatives (captures power-law behavior).

#### FractionalMaxwellGel (FMG)
**When to use:**
- Gel-like materials with power-law relaxation
- Critical gels, physical gels near gel point
- G(t) ∝ t^(-α) decay
- Slope ≈ α in log(G') vs log(ω)

**Characteristics:**
- **Decay type**: Power-law
- **Equilibrium modulus**: Typically very small
- **Parameters**: c_α (gel strength), α (fractional order 0-1), η (viscosity)
- **Typical α range**: 0.3-0.7

**Physical interpretation:**
- α ≈ 0.5: Critical gel (gel point)
- α < 0.5: Pre-gel (liquid-like)
- α > 0.5: Post-gel (solid-like)

**Don't use when:**
- Exponential relaxation observed
- Strong equilibrium modulus present
- Simple liquid or solid behavior

---

#### FractionalZenerSolidSolid (FZSS)
**When to use:**
- Solid-like materials with fractional dynamics
- Power-law relaxation with finite equilibrium modulus
- Filled polymers, soft solids with complex microstructure
- G(t) = Ge + Gm·Eα(-(t/τα)^α)

**Characteristics:**
- **Decay type**: Mittag-Leffler (generalized power-law)
- **Equilibrium modulus**: Finite (Ge > 0)
- **Parameters**: Ge (equilibrium), Gm (Maxwell arm), α (fractional order), τα (relaxation time)
- **Typical applications**: Filled rubbers, nanocomposites

**Don't use when:**
- Simple exponential decay (use Zener instead)
- Material flows (use FML instead)
- Pure power-law without plateau (use FMG instead)

---

#### FractionalMaxwellLiquid (FML)
**When to use:**
- Liquid-like materials with fractional relaxation
- Polymer melts, complex fluids
- No equilibrium modulus (flows)
- Power-law frequency dependence

**Characteristics:**
- **Decay type**: Mittag-Leffler
- **Equilibrium modulus**: Zero
- **Parameters**: Gm (modulus), α (fractional order), τα (relaxation time)
- **Typical applications**: Polymer melts, wormlike micelles

**Don't use when:**
- Finite equilibrium modulus (use FZSS)
- Simple exponential behavior (use Maxwell)
- Gel-like behavior (use FMG)

---

#### FractionalKelvinVoigt (FKV)
**When to use:**
- Solid-like materials with power-law creep
- Filled polymers under constant stress
- Creep compliance: J(t) = 1/Ge · (1 - Eα(-(t/τε)^α))
- Primary mode: Creep testing

**Characteristics:**
- **Decay type**: Power-law creep
- **Equilibrium modulus**: Finite (Ge > 0)
- **Parameters**: Ge (equilibrium), c_α (SpringPot constant), α (fractional order)
- **Best used in**: Creep mode

**Don't use when:**
- Material flows (zero equilibrium modulus)
- Simple exponential behavior
- Primarily relaxation experiments

---

#### FractionalKelvinVoigtZener (FKVZ)
**When to use:**
- Retardation-dominated behavior
- Creep experiments with finite equilibrium compliance
- Series combination: spring + fractional KV element

**Characteristics:**
- **Decay type**: Mittag-Leffler (retardation)
- **Equilibrium compliance**: Finite
- **Parameters**: Ge (series spring), Gk (KV modulus), α, τ (retardation time)
- **Best used in**: Creep mode

**Don't use when:**
- Relaxation experiments
- Simple exponential behavior
- Liquid-like materials

### Advanced Fractional Models

#### FractionalMaxwellModel (FMM)
**When to use:**
- Maximum flexibility needed
- Two independent fractional orders (α, β)
- Complex materials with multi-scale dynamics
- G(t) = c₁·t^(-α)·E₁₋α(-(t/τ)^β)

**Characteristics:**
- **Most general** fractional Maxwell model
- **Parameters**: c₁, α, β, τ
- **Use when**: Simpler models fail

**Don't use when:**
- Data can be fit with simpler models (prefer parsimony)
- Physical interpretation is priority

---

#### FractionalPoyntingThomson (FPT)
**When to use:**
- Complex relaxation AND retardation behavior
- Materials showing both liquid and solid characteristics
- Spring in series with fractional Maxwell element

**Characteristics:**
- **Decay type**: Complex (relaxation + retardation)
- **Parameters**: Ge, Gm, α, τ
- **Applications**: Complex polymer systems

---

#### FractionalJeffreys
**When to use:**
- Two independent relaxation and retardation times
- Emulsions, suspensions
- Modified Maxwell with additional SpringPot

**Characteristics:**
- **Parameters**: η₁, η₂, α, τ₁
- **Applications**: Viscoelastic fluids

---

#### FractionalBurgers
**When to use:**
- Combination of Maxwell and Kelvin-Voigt behavior
- Materials with both immediate and delayed elastic response
- Four-parameter model

**Characteristics:**
- **Most complex** fractional model
- **Parameters**: Ge, Gm, α, η, τ
- **Use when**: Simpler models inadequate

### Flow Models (Steady Shear)

#### PowerLaw
**When to use:**
- Shear-thinning or shear-thickening fluids
- Viscosity vs shear rate: η(γ̇) = K·γ̇^(n-1)
- n < 1: Shear-thinning
- n > 1: Shear-thickening

**Don't use when:**
- Viscosity plateaus at low/high shear rates
- Yield stress present

---

#### Carreau / Carreau-Yasuda
**When to use:**
- Polymer solutions/melts with plateaus
- Zero-shear and infinite-shear viscosity plateaus
- Smooth transition between Newtonian regions

**Characteristics:**
- **Parameters**: η₀ (zero-shear), η∞ (infinite-shear), λ (time constant), n (power-law index)
- **Carreau-Yasuda**: Additional parameter a for transition

**Don't use when:**
- Simple power-law sufficient
- Yield stress present

---

#### Cross
**When to use:**
- Alternative to Carreau for plateau behavior
- Different mathematical form, similar physics
- Sometimes better numerical stability

---

#### Herschel-Bulkley / Bingham
**When to use:**
- Materials with yield stress (τ₀)
- No flow below yield stress
- Muds, pastes, food products

**Characteristics:**
- **Herschel-Bulkley**: τ = τ₀ + K·γ̇^n (generalized)
- **Bingham**: τ = τ₀ + η·γ̇ (Newtonian above yield)

## Data-Driven Selection

### Based on Relaxation Modulus G(t)

**Linear in log(G) vs t** (Semi-log plot)
→ Exponential decay → **Maxwell** or **Zener**

**Linear in log(G) vs log(t)** (Log-log plot)
→ Power-law decay → **FractionalMaxwellGel** or **FZSS**

**Plateau at long times**
→ Finite equilibrium modulus → **Zener**, **FZSS**, or **FKV**

**No plateau (G → 0)**
→ Liquid-like → **Maxwell** or **FractionalMaxwellLiquid**

### Based on Complex Modulus G*(ω)

**Low-frequency slope in log(G') vs log(ω)**
- Slope ≈ 2: Liquid → **Maxwell**, **FML**
- Slope ≈ 0: Solid → **Zener**, **FZSS**, **FKV**
- Slope ≈ α (0 < α < 1): Gel → **FractionalMaxwellGel**

**G'/G" crossover**
- Present: Relaxation time identifiable → **Maxwell**, **Zener**
- Absent (G' > G" always): Strong solid → **FKV**, **FZSS**
- Absent (G" > G' always): Strong liquid → **Maxwell**, **FML**

### Based on Creep Compliance J(t)

**Linear in log(J) vs t**
→ Exponential creep → Classical models

**Linear in log(J) vs log(t)**
→ Power-law creep → **FractionalKelvinVoigt**, **FKVZ**

**Finite equilibrium compliance**
→ Solid-like → **FKVZ**, **Zener**

**Unbounded compliance**
→ Liquid-like → **Maxwell**, **FML**

## Automatic Model Selection

RheoJAX provides automatic compatibility checking to help identify inappropriate models:

```python
from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid
from rheojax.utils.compatibility import check_model_compatibility, format_compatibility_message

# Check before fitting
model = FractionalZenerSolidSolid()
compat = check_model_compatibility(
    model,
    t=time_data,
    G_t=modulus_data,
    test_mode='relaxation'
)

# Print compatibility report
print(format_compatibility_message(compat))

# Or enable automatic checking during fit
model.fit(time_data, modulus_data, check_compatibility=True)
```

The system will:
- Detect decay type (exponential, power-law, etc.)
- Identify material type (solid, liquid, gel)
- Warn about incompatibilities
- Suggest alternative models

## Model Comparison Strategy

When uncertain, fit multiple models and compare:

```python
from rheojax.pipeline.workflows import ModelComparisonPipeline

# Automatically fit and compare multiple models
pipeline = ModelComparisonPipeline()
pipeline.load('data.csv', x_col='time', y_col='modulus')
pipeline.add_models([
    'maxwell',
    'zener',
    'fractional_maxwell_gel',
    'fractional_zener_ss'
])
pipeline.fit_all()

# Get results sorted by quality
results = pipeline.get_results()  # Sorted by R² or AIC/BIC
best_model = results[0]['model']
```

## Common Pitfalls

### 1. Over-parameterization
**Problem**: Using complex models when simple ones suffice
**Solution**: Start simple (Maxwell, Zener), add complexity only if needed

### 2. Wrong test mode
**Problem**: Fitting oscillation data in relaxation mode
**Solution**: Always specify `test_mode='oscillation'` for frequency-domain data

### 3. Ignoring physics
**Problem**: Forcing a solid model on liquid data
**Solution**: Use compatibility checking, respect material behavior

### 4. Poor initial guesses
**Problem**: Optimization fails due to bad starting point
**Solution**: Smart initialization is automatic in oscillation mode (Issue #9)

### 5. Insufficient data range
**Problem**: Can't identify behavior from narrow data range
**Solution**: Collect data over 3+ decades in time or frequency

## Parameter Bounds

All models have physically reasonable default bounds:

- **Moduli (G, E)**: 1 Pa to 1 GPa
- **Viscosity (η)**: 1 mPa·s to 1 MPa·s
- **Time constants (τ)**: 1 μs to 1 Ms
- **Fractional orders (α)**: 0 to 1

Adjust bounds if your material is outside these ranges.

## Further Reading

- **Mainardi (2010)**: Fractional Calculus and Waves in Linear Viscoelasticity
- **Ferry (1980)**: Viscoelastic Properties of Polymers
- **Barnes et al. (1989)**: An Introduction to Rheology
- **Tschoegl (1989)**: The Phenomenological Theory of Linear Viscoelastic Behavior

## Getting Help

If you're unsure which model to use:

1. Run compatibility check first
2. Fit 3-4 candidate models
3. Compare R², AIC, BIC
4. Check if parameters are physically reasonable
5. Validate predictions on held-out data

For advanced cases, consider:
- Bayesian model selection with `fit_bayesian()`
- Cross-validation
- Physical constraints from material science
