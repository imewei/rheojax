# Rheo Examples and Tutorials

Welcome to the Rheo example notebooks! This directory contains comprehensive, hands-on tutorials for learning the Rheo package.

## Overview

Rheo provides a unified framework for analyzing experimental rheology data with modern computational patterns. These examples progress from basic model fitting through advanced analysis techniques, all leveraging JAX acceleration and Bayesian inference.

**Key Features Demonstrated:**
- JAX-accelerated numerical optimization (5-270x speedup)
- Bayesian inference with full convergence diagnostics
- Flexible API design (Pipeline, Modular, Core layers)
- All 23 rheological models with examples (including SGR)
- Advanced transforms for experimental data analysis (including SRFS)
- **Google Colab compatible** - all notebooks run directly on https://colab.google/

**59 Tutorial Notebooks** (all complete ✅) organized into 9 learning paths:
- **Basic Model Fitting** (5 notebooks ✅) - Fundamental rheological models
- **Transform Workflows** (7 notebooks ✅) - Data analysis techniques (+ SRFS)
- **Bayesian Inference** (7 notebooks ✅) - Uncertainty quantification
- **Advanced Workflows** (9 notebooks ✅) - Production patterns (+ SGR)
- **DMT Thixotropic Models** (6 notebooks ✅) - Structural-kinetics thixotropy across all 6 protocols
- **EPM Mesoscopic Models** (6 notebooks ✅) - Elasto-Plastic Model for amorphous solids across 5 protocols + visualization
- **SGR Soft Glassy Models** (6 notebooks ✅) - Soft Glassy Rheology across all 6 protocols
- **STZ Amorphous Solid Models** (6 notebooks ✅) - Shear Transformation Zones across all 6 protocols
- **HL Hébraud-Lequeux Models** (6 notebooks ✅) - Mean-field elastoplastic model across all 6 protocols

## Prerequisites

**Required Knowledge:**
- Basic Python (variables, functions, lists, dictionaries)
- NumPy arrays and NumPy operations
- Matplotlib for basic plotting
- General understanding of rheology (viscosity, modulus, etc.)

**Required Installation:**
```bash
pip install rheojax
# For GPU acceleration (Linux + CUDA only):
pip install jax[cuda12-local]==0.8.0
```

See [CLAUDE.md](../CLAUDE.md) for detailed installation instructions.

## Learning Progression

We recommend following this learning path:

### Phase 1: Basic Model Fitting (Start Here!)

Learn fundamental model fitting with classical rheological models through hands-on examples that demonstrate the complete workflow: NLSQ optimization → Bayesian inference → ArviZ diagnostics.

| Notebook | Model | Test Mode | Time | Key Topics |
|----------|-------|-----------|------|------------|
| **[01-maxwell-fitting.ipynb](basic/01-maxwell-fitting.ipynb)** | Maxwell | Relaxation | 35 min | Pipeline API, NLSQ vs SciPy, NUTS warm-start, all 6 ArviZ plots |
| **[02-zener-fitting.ipynb](basic/02-zener-fitting.ipynb)** | Zener/SLS | Oscillation | 40 min | G'/G" frequency-domain fitting, finite equilibrium modulus |
| **[03-springpot-fitting.ipynb](basic/03-springpot-fitting.ipynb)** | SpringPot | Relaxation | 35 min | Fractional calculus, power-law decay, parameter correlations |
| **[04-bingham-fitting.ipynb](basic/04-bingham-fitting.ipynb)** | Bingham | Rotation | 30 min | Yield stress detection, flow curves, shear stress vs rate |
| **[05-power-law-fitting.ipynb](basic/05-power-law-fitting.ipynb)** | Power-Law | Rotation | 30 min | Shear-thinning/thickening, viscosity curves, non-Newtonian flow |

**Recommended Learning Path:**
1. Start with **01-maxwell-fitting** - establishes all core patterns (NLSQ, Bayesian, diagnostics)
2. Continue with **02-zener-fitting** - extends to frequency-domain and oscillation data
3. Choose **03-springpot** (fractional models) OR **04-bingham**/**05-power-law** (flow behavior) based on interest

**Prerequisites:**
- Python basics (variables, functions, imports)
- NumPy arrays and operations
- Basic matplotlib plotting
- Rheology fundamentals (stress, strain, viscosity, modulus)

**Total Time Investment:** 2.5-3 hours for all 5 notebooks

**Learning Outcomes:**
- Master Pipeline and Modular API patterns
- Understand NLSQ optimization advantages (5-270x speedup)
- Perform Bayesian inference with warm-start workflow
- Interpret all 6 ArviZ diagnostic plots (pair, forest, energy, autocorr, rank, ESS)
- Extract physically meaningful parameters with uncertainty quantification
- Recognize when Maxwell, Zener, SpringPot, Bingham, and Power-Law models are appropriate

### Phase 2: Transform Workflows

Apply advanced data transforms to experimental rheological data for enhanced analysis capabilities. These transforms extend basic model fitting with sophisticated signal processing, superposition principles, and material characterization metrics.

Learn how to preprocess data, extract frequency-domain information, construct mastercurves, quantify viscoelastic character, and analyze nonlinear responses.

| Notebook | Transform | Application | Time | Key Topics |
|----------|-----------|-------------|------|------------|
| **[01-fft-analysis.ipynb](transforms/01-fft-analysis.ipynb)** | FFT | Time→Frequency | 30 min | FFT principles, G'/G" extraction, Cole-Cole plots, Kramers-Kronig |
| **[02-mastercurve-tts.ipynb](transforms/02-mastercurve-tts.ipynb)** | Mastercurve | TTS with Real Data | 35 min | WLF equation (TRIOS data), shift factors, T_g estimation, quality metrics |
| **[02b-mastercurve-wlf-validation.ipynb](transforms/02b-mastercurve-wlf-validation.ipynb)** | WLF Validation | Synthetic TTS Validation | 25 min | Known WLF parameters, fractional model fitting, temperature predictions |
| **[03-mutation-number.ipynb](transforms/03-mutation-number.ipynb)** | Mutation Number | Material Classification | 30 min | Viscoelastic character Δ, gel point detection, integration methods |
| **[04-owchirp-laos-analysis.ipynb](transforms/04-owchirp-laos-analysis.ipynb)** | OWChirp/LAOS | Nonlinear Analysis | 40 min | Harmonic extraction, Lissajous curves, nonlinearity quantification |
| **[05-smooth-derivative.ipynb](transforms/05-smooth-derivative.ipynb)** | Savitzky-Golay | Noise-Robust Differentiation | 25 min | Local polynomial fitting, noise amplification, parameter optimization |
| **[06-mastercurve_auto_shift.ipynb](transforms/06-mastercurve_auto_shift.ipynb)** | Mastercurve | Automatic Shift Factors | 30 min | Power-law intersection, no WLF parameters required |
| **[07-srfs-strain-rate-superposition.ipynb](transforms/07-srfs-strain-rate-superposition.ipynb)** | SRFS | Strain-Rate Superposition | 35 min | Flow curve collapse, shift factors, shear banding detection, thixotropy |

**Recommended Learning Path:**
1. Start with **01-fft-analysis** - fundamental time↔frequency conversion for all rheologists
2. Continue with **02-mastercurve-tts** - essential for polymer melt characterization with real data
3. Try **02b-mastercurve-wlf-validation** - validate WLF extraction with known parameters
4. Choose **03-mutation-number** (gelation studies) OR **04-owchirp-laos** (nonlinear behavior) based on application
5. Complete with **05-smooth-derivative** - practical tool for noisy data analysis
6. Try **07-srfs-strain-rate-superposition** for flow curve analysis with SGR connection

**Prerequisites:**
- Phase 1 basic/ notebooks (understand model fitting workflow)
- Fourier analysis basics (for FFT and OWChirp notebooks)
- Complex modulus concepts (G', G", tan δ)
- Basic understanding of viscoelastic materials

**Total Time Investment:** 4-4.5 hours for all 8 notebooks

**Learning Outcomes:**
- Master FFT-based frequency-domain analysis from time-domain data
- Construct mastercurves via time-temperature superposition (WLF equation)
- Quantify material character using mutation number (solid vs fluid classification)
- Analyze nonlinear viscoelasticity with LAOS/OWChirp protocols
- Apply noise-robust differentiation to experimental data
- Chain multiple transforms in analysis workflows
- Interpret Cole-Cole plots, Lissajous curves, and shift factor plots
- Validate transform quality with diagnostic metrics (R², overlap error, harmonics)
- Apply SRFS to collapse flow curves and extract SGR noise temperature
- Detect shear banding and thixotropy in complex fluids

### Phase 3: Bayesian Inference Focus

Master Bayesian uncertainty quantification and advanced diagnostics with comprehensive ArviZ integration. This learning path builds on Phase 1 basics to provide complete Bayesian workflow capabilities.

| Notebook | Focus | Time | Key Topics |
|----------|-------|------|------------|
| **[01-bayesian-basics.ipynb](bayesian/01-bayesian-basics.ipynb)** | NUTS Workflow | 40 min | Warm-start from NLSQ, posterior sampling, credible intervals, R-hat/ESS diagnostics |
| **[02-prior-selection.ipynb](bayesian/02-prior-selection.ipynb)** | Prior Elicitation | 35 min | Transform parameter bounds to priors, weakly informative priors, prior sensitivity |
| **[03-convergence-diagnostics.ipynb](bayesian/03-convergence-diagnostics.ipynb)** | MCMC Diagnostics | 45 min | All 6 ArviZ plots (pair, forest, energy, autocorr, rank, ESS), divergence troubleshooting |
| **[04-model-comparison.ipynb](bayesian/04-model-comparison.ipynb)** | Model Selection | 40 min | WAIC, LOO, az.compare, Bayes factors, posterior predictive checks |
| **[05-uncertainty-propagation.ipynb](bayesian/05-uncertainty-propagation.ipynb)** | Uncertainty Quantification | 45 min | Derived quantities, parameter correlations, prediction bands, reporting guidelines |

**Recommended Learning Path:**
1. Start with **01-bayesian-basics** - complete NLSQ → NUTS workflow with Maxwell model
2. Continue with **02-prior-selection** - transform parameter bounds to informative Bayesian priors
3. Master **03-convergence-diagnostics** - all 6 ArviZ diagnostic plots for MCMC quality assessment
4. Learn **04-model-comparison** - select best model using WAIC/LOO with ArviZ compare
5. Complete with **05-uncertainty-propagation** - propagate uncertainty to derived quantities and predictions

**Prerequisites:**
- Phase 1 basic/ notebooks (understand NLSQ fitting workflow)
- Basic probability (Bayes' theorem, conditional probability)
- Familiarity with Maxwell, Zener, and SpringPot models
- Understanding of credible intervals vs confidence intervals

**Total Time Investment:** 3-3.5 hours for all 5 notebooks

**Learning Outcomes:**
- Master NLSQ → NUTS warm-start workflow (5-10x faster convergence)
- Transform parameter bounds to weakly informative priors
- Interpret all 6 ArviZ diagnostic plots:
  - **Pair plot**: Parameter correlations and identifiability
  - **Forest plot**: Credible intervals comparison across parameters
  - **Energy plot**: Detect problematic posterior geometry
  - **Autocorrelation plot**: Assess MCMC mixing quality
  - **Rank plot**: Modern convergence diagnostic (alternative to trace plots)
  - **ESS plot**: Quantify sampling efficiency per parameter
- Perform Bayesian model comparison with WAIC and LOO
- Propagate parameter uncertainty to derived quantities and predictions
- Generate prediction uncertainty bands with credible intervals
- Diagnose parameter identifiability via correlation analysis
- Report Bayesian results following scientific best practices

**Key Concepts:**
- **R-hat < 1.01**: Convergence diagnostic (between-chain vs within-chain variance)
- **ESS > 400**: Effective Sample Size (accounting for autocorrelation)
- **Divergences < 1%**: NUTS sampler quality metric
- **WAIC/LOO**: Information criteria for model comparison (lower is better)
- **Warm-start**: Initialize NUTS from NLSQ fit for faster convergence
- **Credible interval**: Bayesian uncertainty quantification (e.g., 95% CI)
- **Posterior predictive**: Predictions with full parameter uncertainty
- **Parameter correlation**: Identifies identifiability issues (|ρ| > 0.7)

### Phase 4: Advanced Workflows

Tackle complex, production-ready analysis patterns for advanced rheological characterization.

| Notebook | Focus | Time | Status | Key Topics |
|----------|-------|------|--------|------------|
| **[01-multi-technique-fitting.ipynb](advanced/01-multi-technique-fitting.ipynb)** | Constrained Fitting | 45 min | ✅ Complete | Simultaneous oscillation + relaxation, parameter consistency validation, cross-domain predictions, uncertainty reduction |
| **[02-batch-processing.ipynb](advanced/02-batch-processing.ipynb)** | Parallel Execution | 45-50 min | ✅ Complete | BatchPipeline API, 20-dataset processing, statistical aggregation, parameter distributions, HDF5 export, performance comparison |
| **[03-custom-models.ipynb](advanced/03-custom-models.ipynb)** | Model Development | 50-55 min | ✅ Complete | Complete 4-parameter Burgers model, BaseModel inheritance, registry integration, Bayesian capabilities, JIT optimization, testing |
| **[04-fractional-models-deep-dive.ipynb](advanced/04-fractional-models-deep-dive.ipynb)** | Fractional Calculus | 55-60 min | ✅ Complete | All 11 fractional models, Mittag-Leffler functions, model comparison (AIC/BIC), parameter interpretation, validation |
| **[05-performance-optimization.ipynb](advanced/05-performance-optimization.ipynb)** | GPU & JIT | 55-60 min | ✅ Complete | JAX JIT compilation, CPU vs GPU benchmarks, memory optimization, NLSQ vs scipy (5-270x), vmap vectorization, real case study |
| **[06-frequentist-model-selection.ipynb](advanced/06-frequentist-model-selection.ipynb)** | Frequentist Selection | 45 min | ✅ Complete | ModelComparisonPipeline, AIC/BIC information criteria, AIC weights, evidence ratios, complexity trade-offs |
| **[07-trios_chunked_reading_example.ipynb](advanced/07-trios_chunked_reading_example.ipynb)** | Large File I/O | 30 min | ✅ Complete | TRIOS auto-chunking, memory optimization, progress callbacks |
| **[08-generalized_maxwell_fitting.ipynb](advanced/08-generalized_maxwell_fitting.ipynb)** | Generalized Maxwell | 45 min | ✅ Complete | Multi-mode Prony series, element minimization, GMM optimization |
| **[09-sgr-soft-glassy-rheology.ipynb](advanced/09-sgr-soft-glassy-rheology.ipynb)** | SGR Models | 45 min | ✅ Complete | SGRConventional, SGRGeneric, noise temperature x, material classification, Bayesian inference |

**Recommended Learning Path:**
1. Start with **01-multi-technique-fitting** - builds on Phase 1 Maxwell/Zener knowledge
2. Continue with **02-batch-processing** - applies multi-technique patterns to multiple datasets
3. Explore **03-custom-models** - learn model development workflow
4. Study **04-fractional-models-deep-dive** - comprehensive fractional model coverage
5. Try **06-frequentist-model-selection** - compare with Bayesian model comparison from Phase 3
6. Learn **09-sgr-soft-glassy-rheology** - SGR models for soft glassy materials
7. Complete with **05-performance-optimization** - GPU acceleration and scaling

**Prerequisites:**
- Phase 1 basic/ notebooks (model fitting fundamentals)
- Phase 2 transforms/ notebooks (data processing workflows)
- Phase 3 bayesian/ notebooks (uncertainty quantification; especially for comparing model selection)
- Understanding of JAX acceleration concepts
- For notebook 05: GPU installation (Linux + CUDA only, optional)

**Total Time Investment:** ~7-8 hours for all 9 notebooks

**Learning Outcomes:**
- Master multi-technique constrained fitting for parameter consistency
- Process large batches of datasets efficiently (5-10x speedup via vmap)
- Create custom rheological models with automatic Bayesian capabilities
- Select appropriate fractional models based on data characteristics
- Optimize computational performance using GPU and JIT compilation
- Scale analyses to 10K+ data points efficiently
- Integrate Rheo into production workflows
- Apply SGR models to soft glassy materials (foams, emulsions, pastes)
- Interpret noise temperature x and classify material regimes

**GPU Requirements (Notebook 05):**
- **Linux + CUDA 12.1-12.9 only** (GPU acceleration not available on macOS/Windows)
- Installation: `make install-jax-gpu` or see [CLAUDE.md](../CLAUDE.md)
- GPU tests marked with `@pytest.mark.gpu` and skip gracefully if unavailable
- CPU-only execution provided as fallback in all notebooks

### Phase 5: DMT Thixotropic Models

Comprehensive tutorials for the de Souza Mendes-Thompson (DMT) structural-kinetics thixotropic model across all 6 rheological protocols with NLSQ → NUTS Bayesian inference pipelines.

| Notebook | Protocol | Data | Key Topics |
|----------|----------|------|------------|
| **[01-dmt-flow-curve.ipynb](dmt/01_dmt_flow_curve.ipynb)** | Flow curve | Real emulsion φ=0.80 | Exponential vs HB closure, equilibrium structure λ_eq(γ̇), closure comparison |
| **[02-dmt-startup-shear.ipynb](dmt/02_dmt_startup_shear.ipynb)** | Startup shear | Synthetic (calibrated) | Stress overshoot, Maxwell backbone, structure breakdown λ(t) |
| **[03-dmt-stress-relaxation.ipynb](dmt/03_dmt_stress_relaxation.ipynb)** | Relaxation | Real laponite clay | Structure recovery, multi-aging-time analysis, accelerating relaxation |
| **[04-dmt-creep.ipynb](dmt/04_dmt_creep.ipynb)** | Creep | Real mucus | Viscosity bifurcation, delayed yielding, Maxwell elastic jump |
| **[05-dmt-saos.ipynb](dmt/05_dmt_saos.ipynb)** | SAOS | Synthetic (calibrated) | G'/G'' crossover, preshear effects, identifiability limits, Cole-Cole |
| **[06-dmt-laos.ipynb](dmt/06_dmt_laos.ipynb)** | LAOS | Synthetic (calibrated) | Lissajous-Bowditch, Fourier harmonics, intra-cycle structure |

**Recommended Order:**
1. Start with **01-flow-curve** — calibrates parameters for Notebooks 02, 05, 06
2. **03-relaxation** and **04-creep** use independent real data and can be done in any order
3. Complete with **05-saos** and **06-laos** for linear/nonlinear oscillatory analysis

**Prerequisites:**
- Phase 1 basic/ notebooks (model fitting)
- Phase 3 bayesian/ notebooks (Bayesian inference fundamentals)

**Data sources:**
- Real: emulsion flow curve, laponite clay relaxation (5 aging times), mucus creep compliance
- Synthetic: startup, SAOS, and LAOS from parameters calibrated to the real emulsion data (3% noise)

### Phase 6: EPM Mesoscopic Models

Comprehensive tutorials for the Elasto-Plastic Model (EPM) — a mesoscopic approach for amorphous solids (glasses, gels, dense emulsions) with lattice-based plastic avalanche dynamics.

| Notebook | Protocol | Data | Key Topics |
|----------|----------|------|------------|
| **[01-epm-flow-curve.ipynb](epm/01_epm_flow_curve.ipynb)** | Flow curve | Real emulsion φ=0.80 | NLSQ→NUTS, Eshelby propagator, disorder distribution, TensorialEPM N₁ |
| **[02-epm-saos.ipynb](epm/02_epm_saos.ipynb)** | SAOS | Real polystyrene | G'/G'' from disorder, Cole-Cole, crossover frequency ω_c |
| **[03-epm-startup.ipynb](epm/03_epm_startup.ipynb)** | Startup shear | Synthetic (calibrated) | Stress overshoot, avalanche onset, N₁(t) evolution, parameter recovery |
| **[04-epm-creep.ipynb](epm/04_epm_creep.ipynb)** | Creep | Real mucus | Bounded vs unbounded creep, yield stress estimation |
| **[05-epm-relaxation.ipynb](epm/05_epm_relaxation.ipynb)** | Relaxation | Real polystyrene | Disorder-induced multi-relaxation, relaxation spectrum, SAOS comparison |
| **[06-epm-visualization.ipynb](epm/06_epm_visualization.ipynb)** | All | Synthetic | Visualization gallery, tensorial fields, von Mises, animations |

**Recommended Order:**
1. Start with **01-flow-curve** — establishes EPM physics and calibrates parameters for Notebook 03
2. **02-saos**, **04-creep**, and **05-relaxation** use independent real data (any order)
3. Complete with **06-visualization** for lattice stress field analysis

**Models:**
- **LatticeEPM**: Scalar stress field, full NLSQ + Bayesian fitting
- **TensorialEPM**: 3-component stress tensor [σ_xx, σ_yy, σ_xy], forward predictions only (N₁)

**Prerequisites:**
- Phase 1 basic/ notebooks (model fitting)
- Phase 3 bayesian/ notebooks (Bayesian inference fundamentals)

**Data sources:**
- Real: emulsion flow curve, polystyrene SAOS, mucus creep, polystyrene relaxation
- Synthetic: startup from parameters calibrated to the real emulsion data (3% noise)

### Phase 7: SGR Soft Glassy Rheology

Comprehensive tutorials for the Soft Glassy Rheology (SGR) model — a statistical mechanics framework for soft glassy materials (foams, emulsions, pastes, colloidal suspensions) controlled by noise temperature x.

| Notebook | Protocol | Data | Key Topics |
|----------|----------|------|------------|
| **[01-sgr-flow-curve.ipynb](sgr/01_sgr_flow_curve.ipynb)** | Flow curve | Real emulsions (6 φ) | Volume fraction sweep x(φ), phase regime, shear banding, SGRGeneric |
| **[02-sgr-stress-relaxation.ipynb](sgr/02_sgr_stress_relaxation.ipynb)** | Relaxation | Real laponite clay (5 aging times) | Power-law G(t)~t^(x-2), aging sweep x(t_wait), thermodynamic consistency |
| **[03-sgr-saos.ipynb](sgr/03_sgr_saos.ipynb)** | SAOS | Real chia seed gel | Phase regime exploration, power-law scaling, Cole-Cole, SGRGeneric |
| **[04-sgr-creep.ipynb](sgr/04_sgr_creep.ipynb)** | Creep | Real mucus | Power-law J(t)~t^(2-x), limited-data Bayesian inference |
| **[05-sgr-startup.ipynb](sgr/05_sgr_startup.ipynb)** | Startup | Synthetic (calibrated) | Stress growth η⁺(t), dynamic_x thixotropy, evolve_x() |
| **[06-sgr-laos.ipynb](sgr/06_sgr_laos.ipynb)** | LAOS | Synthetic (calibrated) | Lissajous curves, Fourier harmonics, Chebyshev decomposition |

**Recommended Order:**
1. Start with **01-flow-curve** — calibrates parameters for Notebooks 05 and 06
2. **02-relaxation**, **03-saos**, and **04-creep** use independent real data (any order)
3. Complete with **05-startup** and **06-laos** for nonlinear protocols

**Models:**
- **SGRConventional** (Sollich 1998): Primary model in all 6 notebooks, all protocols
- **SGRGeneric** (Fuereder & Ilg 2013): Thermodynamic GENERIC framework, compared in NB 01-03

**Prerequisites:**
- Phase 1 basic/ notebooks (model fitting)
- Phase 3 bayesian/ notebooks (Bayesian inference fundamentals)

**Data sources:**
- Real: emulsion flow curves (6 volume fractions), laponite clay relaxation (5 aging times), chia seed gel SAOS, mucus creep
- Synthetic: startup and LAOS from parameters calibrated to real emulsion data (3% noise)

### Phase 8: STZ Shear Transformation Zone Models

Comprehensive tutorials for the Shear Transformation Zone (STZ) model — a microscopic theory for plastic flow in amorphous solids (metallic glasses, colloidal glasses, dense emulsions) based on effective temperature dynamics.

| Notebook | Protocol | Data | Key Topics |
|----------|----------|------|------------|
| **[01-stz-flow-curve.ipynb](stz/01_stz_flow_curve.ipynb)** | Flow curve | Synthetic (arctanh) | Arctanh formula, C/T kernels, chi_inf activation, 4-parameter steady state |
| **[02-stz-startup-shear.ipynb](stz/02_stz_startup_shear.ipynb)** | Startup shear | Synthetic (calibrated) | Stress overshoot, chi evolution, Lambda dynamics, variant comparison |
| **[03-stz-stress-relaxation.ipynb](stz/03_stz_stress_relaxation.ipynb)** | Relaxation | Real laponite clay (5 ages) | Physical aging, chi_inf(t_wait), STZ vs SGR comparison |
| **[04-stz-creep.ipynb](stz/04_stz_creep.ipynb)** | Creep | Real mucus | Yield bifurcation, sub-yield vs super-yield, chi trajectory |
| **[05-stz-saos.ipynb](stz/05_stz_saos.ipynb)** | SAOS | Real polystyrene (ps145) | Maxwell approximation, tau_eff, crossover, Cole-Cole |
| **[06-stz-laos.ipynb](stz/06_stz_laos.ipynb)** | LAOS | Synthetic (calibrated) | Lissajous curves, Fourier harmonics I3/I1, variant comparison |

**Recommended Order:**
1. Start with **01-flow-curve** — calibrates parameters for Notebooks 02 and 06
2. **03-relaxation**, **04-creep**, and **05-saos** use independent real data (any order)
3. Complete with **06-laos** for nonlinear oscillatory analysis

**Models:**
- **STZConventional**: 3 variants (minimal/standard/full) controlling state variable complexity

**Prerequisites:**
- Phase 1 basic/ notebooks (model fitting)
- Phase 3 bayesian/ notebooks (Bayesian inference fundamentals)

**Data sources:**
- Synthetic: flow curve (NB01, arctanh formula with known params), startup (NB02) and LAOS (NB06) from NB01 params (3% noise)
- Real: laponite clay relaxation (5 aging times), mucus creep, polystyrene SAOS

### Phase 9: HL Hébraud-Lequeux Models

Comprehensive tutorials for the Hébraud-Lequeux (HL) mean-field elastoplastic model — a mesoscopic approach for yield-stress fluids (emulsions, foams, pastes, colloidal glasses) based on stress block dynamics with stochastic yielding.

| Notebook | Protocol | Data | Key Topics |
|----------|----------|------|------------|
| **[01-hl-flow-curve.ipynb](hl/01_hl_flow_curve.ipynb)** | Flow curve | Real emulsion (6 φ) | Yield stress extraction, volume fraction sweep α(φ), P(glass) |
| **[02-hl-relaxation.ipynb](hl/02_hl_relaxation.ipynb)** | Relaxation | Real laponite clay (5 t_w) | Incomplete relaxation, aging sweep α(t_w), glass signature |
| **[03-hl-creep.ipynb](hl/03_hl_creep.ipynb)** | Creep | Real polystyrene (5 T) | Delayed yielding, temperature sweep α(T), identifiability |
| **[04-hl-saos.ipynb](hl/04_hl_saos.ipynb)** | SAOS | Synthetic (calibrated) | G'/G'' analysis, glass vs fluid, Cole-Cole plots |
| **[05-hl-startup.ipynb](hl/05_hl_startup.ipynb)** | Startup | Synthetic (calibrated) | Stress overshoot, τ from transients, shear rate effects |
| **[06-hl-laos.ipynb](hl/06_hl_laos.ipynb)** | LAOS | Synthetic (calibrated) | Lissajous curves, Fourier harmonics I₃/I₁, nonlinearity |

**Recommended Order:**
1. Start with **01-flow-curve** — calibrates parameters for Notebooks 04-06
2. **02-relaxation** and **03-creep** use independent real data (any order)
3. Complete with **04-saos**, **05-startup**, and **06-laos** for oscillatory/transient analysis

**Key Physics:**
- **Coupling parameter α**: Controls glass (α < 0.5) vs fluid (α ≥ 0.5) behavior
- **P(glass) = P(α < 0.5)**: Bayesian phase classification from posteriors
- **Incomplete relaxation**: Glass phase signature (G(t→∞) > 0)

**Prerequisites:**
- Phase 1 basic/ notebooks (model fitting)
- Phase 3 bayesian/ notebooks (Bayesian inference fundamentals)

**Data sources:**
- Real: emulsion flow curves (6 volume fractions), laponite clay relaxation (5 aging times), polystyrene creep (5 temperatures)
- Synthetic: SAOS, startup, and LAOS from parameters calibrated to real emulsion data (3% noise)

## Quick Reference

### By Topic

| Topic | Notebook | Category | Status |
|-------|----------|----------|--------|
| Maxwell Model | [basic/01-maxwell-fitting](basic/01-maxwell-fitting.ipynb) | Basic | ✅ Complete |
| Zener (SLS) Model | [basic/02-zener-fitting](basic/02-zener-fitting.ipynb) | Basic | ✅ Complete |
| SpringPot Fractional Element | [basic/03-springpot-fitting](basic/03-springpot-fitting.ipynb) | Basic | ✅ Complete |
| Bingham Plastic (Yield Stress) | [basic/04-bingham-fitting](basic/04-bingham-fitting.ipynb) | Basic | ✅ Complete |
| Power-Law Fluids | [basic/05-power-law-fitting](basic/05-power-law-fitting.ipynb) | Basic | ✅ Complete |
| FFT Analysis | [transforms/01-fft-analysis](transforms/01-fft-analysis.ipynb) | Transforms | ✅ Complete |
| Mastercurve TTS | [transforms/02-mastercurve-tts](transforms/02-mastercurve-tts.ipynb) | Transforms | ✅ Complete |
| Mutation Number | [transforms/03-mutation-number](transforms/03-mutation-number.ipynb) | Transforms | ✅ Complete |
| OWChirp LAOS | [transforms/04-owchirp-laos-analysis](transforms/04-owchirp-laos-analysis.ipynb) | Transforms | ✅ Complete |
| Smooth Derivative | [transforms/05-smooth-derivative](transforms/05-smooth-derivative.ipynb) | Transforms | ✅ Complete |
| Auto Shift Factors | [transforms/06-mastercurve_auto_shift](transforms/06-mastercurve_auto_shift.ipynb) | Transforms | ✅ Complete |
| SRFS Transform | [transforms/07-srfs-strain-rate-superposition](transforms/07-srfs-strain-rate-superposition.ipynb) | Transforms | ✅ Complete |
| Bayesian Basics | [bayesian/01-bayesian-basics](bayesian/01-bayesian-basics.ipynb) | Bayesian | ✅ Complete |
| Prior Selection | [bayesian/02-prior-selection](bayesian/02-prior-selection.ipynb) | Bayesian | ✅ Complete |
| Diagnostics | [bayesian/03-convergence-diagnostics](bayesian/03-convergence-diagnostics.ipynb) | Bayesian | ✅ Complete |
| Model Comparison | [bayesian/04-model-comparison](bayesian/04-model-comparison.ipynb) | Bayesian | ✅ Complete |
| Uncertainty Propagation | [bayesian/05-uncertainty-propagation](bayesian/05-uncertainty-propagation.ipynb) | Bayesian | ✅ Complete |
| Bayesian Workflow Demo | [bayesian/06-bayesian_workflow_demo](bayesian/06-bayesian_workflow_demo.ipynb) | Bayesian | ✅ Complete |
| GMM Bayesian | [bayesian/07-gmm_bayesian_workflow](bayesian/07-gmm_bayesian_workflow.ipynb) | Bayesian | ✅ Complete |
| Multi-Technique | [advanced/01-multi-technique-fitting](advanced/01-multi-technique-fitting.ipynb) | Advanced | ✅ Complete |
| Batch Processing | [advanced/02-batch-processing](advanced/02-batch-processing.ipynb) | Advanced | ✅ Complete |
| Custom Models | [advanced/03-custom-models](advanced/03-custom-models.ipynb) | Advanced | ✅ Complete |
| Fractional Models | [advanced/04-fractional-models-deep-dive](advanced/04-fractional-models-deep-dive.ipynb) | Advanced | ✅ Complete |
| Performance | [advanced/05-performance-optimization](advanced/05-performance-optimization.ipynb) | Advanced | ✅ Complete |
| Model Selection | [advanced/06-frequentist-model-selection](advanced/06-frequentist-model-selection.ipynb) | Advanced | ✅ Complete |
| TRIOS Chunking | [advanced/07-trios_chunked_reading_example](advanced/07-trios_chunked_reading_example.ipynb) | Advanced | ✅ Complete |
| Generalized Maxwell | [advanced/08-generalized_maxwell_fitting](advanced/08-generalized_maxwell_fitting.ipynb) | Advanced | ✅ Complete |
| SGR Models | [advanced/09-sgr-soft-glassy-rheology](advanced/09-sgr-soft-glassy-rheology.ipynb) | Advanced | ✅ Complete |
| DMT Flow Curve | [dmt/01-dmt-flow-curve](dmt/01_dmt_flow_curve.ipynb) | DMT | ✅ Complete |
| DMT Startup Shear | [dmt/02-dmt-startup-shear](dmt/02_dmt_startup_shear.ipynb) | DMT | ✅ Complete |
| DMT Stress Relaxation | [dmt/03-dmt-stress-relaxation](dmt/03_dmt_stress_relaxation.ipynb) | DMT | ✅ Complete |
| DMT Creep | [dmt/04-dmt-creep](dmt/04_dmt_creep.ipynb) | DMT | ✅ Complete |
| DMT SAOS | [dmt/05-dmt-saos](dmt/05_dmt_saos.ipynb) | DMT | ✅ Complete |
| DMT LAOS | [dmt/06-dmt-laos](dmt/06_dmt_laos.ipynb) | DMT | ✅ Complete |
| EPM Flow Curve | [epm/01-epm-flow-curve](epm/01_epm_flow_curve.ipynb) | EPM | ✅ Complete |
| EPM SAOS | [epm/02-epm-saos](epm/02_epm_saos.ipynb) | EPM | ✅ Complete |
| EPM Startup | [epm/03-epm-startup](epm/03_epm_startup.ipynb) | EPM | ✅ Complete |
| EPM Creep | [epm/04-epm-creep](epm/04_epm_creep.ipynb) | EPM | ✅ Complete |
| EPM Relaxation | [epm/05-epm-relaxation](epm/05_epm_relaxation.ipynb) | EPM | ✅ Complete |
| EPM Visualization | [epm/06-epm-visualization](epm/06_epm_visualization.ipynb) | EPM | ✅ Complete |
| SGR Flow Curve | [sgr/01-sgr-flow-curve](sgr/01_sgr_flow_curve.ipynb) | SGR | ✅ Complete |
| SGR Stress Relaxation | [sgr/02-sgr-stress-relaxation](sgr/02_sgr_stress_relaxation.ipynb) | SGR | ✅ Complete |
| SGR SAOS | [sgr/03-sgr-saos](sgr/03_sgr_saos.ipynb) | SGR | ✅ Complete |
| SGR Creep | [sgr/04-sgr-creep](sgr/04_sgr_creep.ipynb) | SGR | ✅ Complete |
| SGR Startup | [sgr/05-sgr-startup](sgr/05_sgr_startup.ipynb) | SGR | ✅ Complete |
| SGR LAOS | [sgr/06-sgr-laos](sgr/06_sgr_laos.ipynb) | SGR | ✅ Complete |
| STZ Flow Curve | [stz/01-stz-flow-curve](stz/01_stz_flow_curve.ipynb) | STZ | ✅ Complete |
| STZ Startup Shear | [stz/02-stz-startup-shear](stz/02_stz_startup_shear.ipynb) | STZ | ✅ Complete |
| STZ Stress Relaxation | [stz/03-stz-stress-relaxation](stz/03_stz_stress_relaxation.ipynb) | STZ | ✅ Complete |
| STZ Creep | [stz/04-stz-creep](stz/04_stz_creep.ipynb) | STZ | ✅ Complete |
| STZ SAOS | [stz/05-stz-saos](stz/05_stz_saos.ipynb) | STZ | ✅ Complete |
| STZ LAOS | [stz/06-stz-laos](stz/06_stz_laos.ipynb) | STZ | ✅ Complete |
| HL Flow Curve | [hl/01-hl-flow-curve](hl/01_hl_flow_curve.ipynb) | HL | ✅ Complete |
| HL Relaxation | [hl/02-hl-relaxation](hl/02_hl_relaxation.ipynb) | HL | ✅ Complete |
| HL Creep | [hl/03-hl-creep](hl/03_hl_creep.ipynb) | HL | ✅ Complete |
| HL SAOS | [hl/04-hl-saos](hl/04_hl_saos.ipynb) | HL | ✅ Complete |
| HL Startup | [hl/05-hl-startup](hl/05_hl_startup.ipynb) | HL | ✅ Complete |
| HL LAOS | [hl/06-hl-laos](hl/06_hl_laos.ipynb) | HL | ✅ Complete |

### By API Level

- **Pipeline API** (Easiest): `pipeline.load().fit().plot().save()` - Start here
- **Modular API** (Medium): `model.fit()` with explicit control
- **Core API** (Advanced): Direct JAX operations for custom workflows

## How to Run Notebooks

### Option 1: Google Colab (Easiest - No Installation Required!)

All notebooks are Colab-compatible and include automatic setup:

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook (or use GitHub integration)
3. Run the first code cell (Colab setup - installs rheojax automatically)
4. Continue with the rest of the notebook

The setup cell automatically:
- Installs rheojax and all dependencies
- Enables float64 precision for numerical stability
- Skips when running locally

### Option 2: Jupyter Lab (Recommended for Local)

```bash
cd /Users/b80985/Projects/rheojax/examples
jupyter lab
```

Then open any `.ipynb` file from the browser interface.

### Option 2: Jupyter Notebook

```bash
cd /Users/b80985/Projects/rheojax/examples
jupyter notebook
```

### Option 3: VSCode with Jupyter Extension

- Install "Jupyter" extension in VSCode
- Open any `.ipynb` file
- Click "Run All" or execute cells individually

### Option 4: Command Line Execution (for testing)

```bash
# Run a single notebook
jupyter nbconvert --to notebook --execute example.ipynb --output example_executed.ipynb

# Run all notebooks in a directory
for nb in basic/*.ipynb; do
  jupyter nbconvert --to notebook --execute "$nb"
done
```

## Troubleshooting

### "Module not found" errors

**Problem:** `ImportError: No module named 'rheojax'`

**Solution:** Install the package in development mode:
```bash
cd /Users/b80985/Projects/rheojax
pip install -e .
```

### "JAX is float32 instead of float64"

**Problem:** Numerical results are inaccurate

**Solution:** JAX requires safe import order. The notebook template includes:
```python
from rheojax.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()  # Must be called before any JAX operations
```

If you see this error, ensure you're using this pattern, not direct imports.

### Bayesian examples won't converge (R-hat > 1.01)

**Problem:** MCMC sampling hasn't converged

**Solutions:**
1. Increase num_warmup: `fit_bayesian(num_warmup=2000)`
2. Increase num_samples: `fit_bayesian(num_samples=5000)`
3. Check data quality and model appropriateness
4. Use warm-start from NLSQ fit (done automatically)

### Optimization is too slow

**Problem:** Fitting takes too long

**Solutions:**
1. Enable GPU: Install JAX with CUDA support
2. Check data size and consider resampling
3. Use better initial parameter guesses
4. Consider model simplification

### GPU not detected

**Problem:** `jax.devices()` shows CPU only

**Solution:** GPU requires Linux + CUDA 12.1-12.9. See [CLAUDE.md](../CLAUDE.md) for GPU installation:
```bash
make install-jax-gpu
```

## Data Sources

Example datasets are organized in the `data/` directory:
- **Synthetic data** - Generated programmatically within notebooks (reproducible, known parameters)
- **data/experimental/** - Real instrument files (TRIOS format, CSV)

See [data/README.md](data/README.md) for detailed dataset descriptions.

## Contributing

Found an error or have a suggestion? Please:

1. Open an issue on GitHub: https://github.com/imewei/rheojax/issues
2. Include the notebook name and specific section
3. Provide error message or unexpected behavior
4. Suggest improvements for clarity or pedagogy

## Next Steps After Examples

Once you've completed these examples:

1. **Read the Documentation:** https://rheojax.readthedocs.io/
2. **Explore Source Code:** `/Users/b80985/Projects/rheojax/rheojax/`
3. **Run Tests:** `make test` to verify your installation
4. **Contribute:** Help improve Rheo or add new examples

## Key Concepts Reference

### Notation and Units

- **G** = Modulus (Pa)
- **G0** = Initial/Storage modulus (Pa)
- **Gp** = Loss modulus (Pa)
- **eta** = Viscosity (Pa·s)
- **tau** = Relaxation time (s)
- **omega** = Frequency (rad/s)
- **gamma** = Strain (dimensionless)
- **gamma_dot** = Shear rate (1/s)
- **sigma** = Stress (Pa)

### Test Modes

- **Oscillation** (SAOS): Small-amplitude oscillatory shear - frequency domain
- **Relaxation** (SR): Stress relaxation after step strain
- **Creep** (SC): Strain evolution under constant stress
- **Rotation** (SS): Steady shear (viscosity) at various shear rates

### Modern Rheo Advantages

- **NLSQ Optimization:** 5-270x faster than scipy
- **JAX Acceleration:** 20-100x speedup on GPU
- **Bayesian Inference:** Full uncertainty quantification
- **ArviZ Integration:** Comprehensive MCMC diagnostics
- **Safe JAX Imports:** Guaranteed float64 precision

---

**Last Updated:** [Date]
**Examples Version:** 1.0
**RheoJAX Version:** [Check with `import rheojax; print(rheojax.__version__)`]
