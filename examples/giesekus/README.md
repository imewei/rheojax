# Giesekus Viscoelastic Model Tutorials

## Overview

The Giesekus model is a nonlinear viscoelastic constitutive equation that extends the Upper-Convected Maxwell (UCM) model with a quadratic stress term. It captures key polymer rheology phenomena including:

- **Shear-thinning** at high shear rates
- **Non-zero second normal stress difference** (N₂ ≠ 0)
- **Stress overshoot** in startup flow
- **Faster-than-exponential relaxation**

### Constitutive Equation

$$
\boldsymbol{\tau} + \lambda \stackrel{\nabla}{\boldsymbol{\tau}} + \frac{\alpha \lambda}{\eta_p} \boldsymbol{\tau} \cdot \boldsymbol{\tau} = 2\eta_p \mathbf{D}
$$

where:
- $\boldsymbol{\tau}$ is the polymer stress tensor
- $\lambda$ is the relaxation time
- $\alpha$ is the mobility parameter (0 ≤ α ≤ 0.5)
- $\eta_p$ is the polymer viscosity
- $\stackrel{\nabla}{\boldsymbol{\tau}}$ is the upper-convected derivative

### Key Physics

| Parameter | Effect |
|-----------|--------|
| α = 0 | Recovers UCM (Maxwell) model |
| α > 0 | Enables shear-thinning, N₂ < 0 |
| α → 0.5 | Maximum shear-thinning |

**Critical Relation:** N₂/N₁ = -α/2 (exact, independent of shear rate)

---

## Model Parameters

| Parameter | Symbol | Units | Typical Range | Description |
|-----------|--------|-------|---------------|-------------|
| Polymer viscosity | η_p | Pa·s | 1 - 10⁵ | Polymer contribution to viscosity |
| Relaxation time | λ | s | 10⁻³ - 10³ | Characteristic relaxation time |
| Mobility parameter | α | - | 0 - 0.5 | Controls nonlinearity (0 = UCM) |
| Solvent viscosity | η_s | Pa·s | 0 - 10³ | Newtonian solvent contribution |

**Derived Quantities:**
- Zero-shear viscosity: η₀ = η_p + η_s
- Plateau modulus: G₀ = η_p/λ
- Critical Weissenberg: Wi_c ≈ 1/√α

---

## Tutorial Notebooks

| NB | Protocol | Data Source | Runtime | Key Topics |
|----|----------|-------------|---------|------------|
| [01](01_giesekus_flow_curve.ipynb) | Flow Curve | Real emulsion (φ=0.80) | ~30 min | Shear-thinning, α from curvature |
| [02](02_giesekus_saos.ipynb) | SAOS | Real polystyrene (T=145°C) | ~35 min | G'(ω), G''(ω), Cole-Cole plot |
| [03](03_giesekus_startup.ipynb) | Startup Shear | Synthetic (3% noise) | ~25 min | Stress overshoot, Wi effect |
| [04](04_giesekus_normal_stresses.ipynb) | Normal Stresses | Synthetic | ~25 min | N₂/N₁ = -α/2 extraction |
| [05](05_giesekus_creep.ipynb) | Creep | Real mucus | ~30 min | J(t) compliance, retardation |
| [06](06_giesekus_relaxation.ipynb) | Relaxation | Real polymer | ~30 min | G(t), faster-than-Maxwell decay |
| [07](07_giesekus_laos.ipynb) | LAOS | Synthetic | ~35 min | Lissajous curves, harmonics I₃/I₁ |

**Fast Demo Mode:** Each notebook includes fast demo settings (~5 min) with `NUM_CHAINS=1, NUM_SAMPLES=500`.

---

## Recommended Learning Path

### Beginner Path
1. **NB01 (Flow Curve)** - Start here for calibration
2. **NB02 (SAOS)** - Linear viscoelasticity
3. **NB04 (Normal Stresses)** - Understand the α parameter

### Intermediate Path
4. **NB03 (Startup)** - Transient nonlinearity
5. **NB05 (Creep)** or **NB06 (Relaxation)** - Complementary tests

### Advanced Path
6. **NB07 (LAOS)** - Full nonlinear oscillatory response

---

## Prerequisites

Before starting these tutorials, complete:

1. **Phase 1: Basic Tutorials** (`examples/basic/`)
   - `01-maxwell-fitting.ipynb` - Model fitting basics
   - `02-bayesian-basics.ipynb` - NUTS inference

2. **Phase 3: Bayesian Inference** (`examples/bayesian/`)
   - ArviZ diagnostics (trace, pair, forest plots)
   - Convergence criteria (R-hat, ESS)

---

## Data Sources

### Real Experimental Data
| Dataset | Location | Description |
|---------|----------|-------------|
| Emulsion φ=0.80 | `data/flow/emulsions/0.80.csv` | Shear stress vs shear rate |
| Polystyrene SAOS | `data/oscillation/polystyrene/oscillation_ps145_data.csv` | G', G'' at T=145°C |
| Mucus creep | `data/creep/biological/creep_mucus_data.csv` | Creep compliance J(t) |
| Polymer relaxation | `data/relaxation/polymers/stressrelaxation_ps145_data.csv` | Relaxation modulus G(t) |

### Synthetic Data
Generated from calibrated NB01 parameters with 3% Gaussian noise:
- Startup stress σ(t)
- Normal stresses N₁(γ̇), N₂(γ̇)
- LAOS time series σ(t), γ(t)

---

## Inference Pipeline

All notebooks follow the NLSQ → NUTS workflow:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Load Data     │ ──▶ │   NLSQ Fit      │ ──▶ │   NUTS Bayesian │
│   (RheoData)    │     │   (warm-start)  │     │   (4 chains)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┘
                        ▼
              ┌─────────────────┐     ┌─────────────────┐
              │ ArviZ Diagnostics│ ──▶ │   Save Results  │
              │ (trace, forest) │     │   (JSON, HDF5)  │
              └─────────────────┘     └─────────────────┘
```

### Convergence Criteria
- R-hat < 1.05 (all parameters)
- ESS > 100 (effective sample size)
- No divergences

---

## Output Files

Results are saved to `examples/outputs/giesekus/<protocol>/`:

| File | Contents |
|------|----------|
| `nlsq_params_<protocol>.json` | NLSQ point estimates |
| `posterior_<protocol>.json` | Full posterior samples |
| `<protocol>_harmonics.json` | LAOS-specific: I₁, I₃, I₅ |
| `alpha_extraction.json` | NB04: α from N₂/N₁ |

---

## Physical Applications

The Giesekus model is widely used for:

| Application | Relevant Notebooks |
|-------------|-------------------|
| Polymer processing | NB01 (flow), NB03 (startup) |
| Extrusion/die swell | NB04 (normal stresses) |
| Oscillatory rheometry | NB02 (SAOS), NB07 (LAOS) |
| Creep/recovery | NB05 (creep), NB06 (relaxation) |
| Polymer solutions | All notebooks |
| Emulsions | NB01 (flow curve) |

---

## Model Comparison

| Property | UCM (α=0) | Giesekus (α>0) | Oldroyd-B |
|----------|-----------|----------------|-----------|
| Shear-thinning | No | Yes | No |
| N₂ | 0 | -αN₁/2 | 0 |
| Stress overshoot | Mild | Pronounced | Mild |
| Extensional singularity | Yes | No (α>0) | Yes |

---

## Troubleshooting

**NLSQ doesn't converge:**
- Check parameter bounds (α ∈ [0, 0.5])
- Increase `max_iter=5000`
- Try different initial guesses

**Bayesian divergences:**
- Increase `num_warmup=2000`
- Use NLSQ warm-start (critical)
- Check for parameter correlations (pair plot)

**Low ESS:**
- Increase `num_samples`
- Check for multimodality (trace plot)

**Float64 errors:**
- Always use `safe_import_jax()` before importing JAX

---

## References

1. Giesekus, H. (1982). "A simple constitutive equation for polymer fluids based on the concept of deformation-dependent tensorial mobility." *J. Non-Newtonian Fluid Mech.* 11, 69-109.

2. Bird, R.B., Armstrong, R.C., Hassager, O. (1987). *Dynamics of Polymeric Liquids, Vol. 1: Fluid Mechanics*. Wiley.

3. Larson, R.G. (1988). *Constitutive Equations for Polymer Melts and Solutions*. Butterworths.
