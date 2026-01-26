# SGR Soft Glassy Rheology Tutorials

Comprehensive tutorials for the Soft Glassy Rheology (SGR) model across all 6 rheological protocols with NLSQ → NUTS Bayesian inference pipelines.

## Overview

The SGR model (Sollich 1998) describes soft glassy materials — foams, emulsions, pastes, and colloidal suspensions — using a single **noise temperature** parameter `x` that controls the glass transition:

- **x < 1**: Glass phase (yield stress, aging, solid-like)
- **1 < x < 2**: Power-law viscoelastic fluid (G' ~ G'' ~ ω^(x-1))
- **x ≥ 2**: Newtonian liquid

## Notebooks

| # | Notebook | Protocol | Data | Key Topics |
|---|----------|----------|------|------------|
| 01 | [Flow Curve](01_sgr_flow_curve.ipynb) | Viscosity η(γ̇) | Real emulsions (6 φ) | Volume fraction sweep, phase regime, shear banding, SGRGeneric comparison |
| 02 | [Stress Relaxation](02_sgr_stress_relaxation.ipynb) | G(t) | Real laponite clay (5 aging times) | Power-law G(t)~t^(x-2), aging sweep, thermodynamic consistency |
| 03 | [SAOS](03_sgr_saos.ipynb) | G'(ω), G''(ω) | Real chia seed gel | Phase regime exploration, power-law scaling, Cole-Cole, SGRGeneric |
| 04 | [Creep](04_sgr_creep.ipynb) | J(t) | Real mucus | Power-law creep J(t)~t^(2-x), limited-data Bayesian inference |
| 05 | [Startup](05_sgr_startup.ipynb) | η⁺(t) | Synthetic (from NB 01) | Stress growth, dynamic_x thixotropy, evolve_x() visualization |
| 06 | [LAOS](06_sgr_laos.ipynb) | σ(t) | Synthetic (from NB 01) | Lissajous curves, Fourier harmonics, Chebyshev decomposition |

## Prerequisites

- Phase 1 basic/ notebooks (model fitting fundamentals)
- Phase 3 bayesian/ notebooks (Bayesian inference with NUTS)

## Recommended Order

1. **Start with 01-flow-curve** — calibrates parameters for Notebooks 05 and 06
2. **02-relaxation**, **03-saos**, and **04-creep** use independent real data (any order)
3. **05-startup** and **06-laos** use synthetic data from NB 01 calibrated parameters

## Data Sources

- **Real data (NB 01-04):** Emulsion flow curves, laponite clay relaxation, chia seed gel oscillation, mucus creep compliance
- **Synthetic data (NB 05-06):** Generated from NB 01 calibrated emulsion parameters (φ=0.80) with 3% Gaussian noise

## Models

- **SGRConventional** (Sollich 1998): Primary model in all 6 notebooks, supports all protocols
- **SGRGeneric** (Fuereder & Ilg 2013): Thermodynamically consistent GENERIC framework variant, compared in NB 01-03

## Parameters

| Parameter | Symbol | Bounds | Description |
|-----------|--------|--------|-------------|
| Noise temperature | x | (0.5, 3.0) | Controls glass transition |
| Modulus scale | G0 | (1e-3, 1e9) Pa | Absolute elastic magnitude |
| Attempt time | τ₀ | (1e-9, 1e3) s | Microscopic relaxation timescale |

## Runtime

- **Fast demo** (1 chain): ~2-3 min per notebook
- **Production** (4 chains): ~5-10 min per notebook
