# Rheo Examples and Tutorials

Welcome to the Rheo example notebooks! This directory contains comprehensive, hands-on tutorials for learning the Rheo package.

## Overview

Rheo provides a unified framework for analyzing experimental rheology data with modern computational patterns. These examples progress from basic model fitting through advanced analysis techniques, all leveraging JAX acceleration and Bayesian inference.

**Key Features Demonstrated:**
- JAX-accelerated numerical optimization (5-270x speedup)
- Bayesian inference with full convergence diagnostics
- Flexible API design (Pipeline, Modular, Core layers)
- All 20 rheological models with examples
- Advanced transforms for experimental data analysis

**22 Tutorial Notebooks** (18 complete ✅, 4 stubs 🚧) organized into 4 learning paths:
- **Basic Model Fitting** (5 notebooks ✅) - Fundamental rheological models
- **Transform Workflows** (6 notebooks ✅) - Data analysis techniques
- **Bayesian Inference** (5 notebooks ✅) - Uncertainty quantification
- **Advanced Workflows** (2 complete ✅, 4 stubs 🚧) - Production patterns

## Prerequisites

**Required Knowledge:**
- Basic Python (variables, functions, lists, dictionaries)
- NumPy arrays and NumPy operations
- Matplotlib for basic plotting
- General understanding of rheology (viscosity, modulus, etc.)

**Required Installation:**
```bash
pip install rheo
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

**Recommended Learning Path:**
1. Start with **01-fft-analysis** - fundamental time↔frequency conversion for all rheologists
2. Continue with **02-mastercurve-tts** - essential for polymer melt characterization with real data
3. Try **02b-mastercurve-wlf-validation** - validate WLF extraction with known parameters
4. Choose **03-mutation-number** (gelation studies) OR **04-owchirp-laos** (nonlinear behavior) based on application
5. Complete with **05-smooth-derivative** - practical tool for noisy data analysis

**Prerequisites:**
- Phase 1 basic/ notebooks (understand model fitting workflow)
- Fourier analysis basics (for FFT and OWChirp notebooks)
- Complex modulus concepts (G', G", tan δ)
- Basic understanding of viscoelastic materials

**Total Time Investment:** 3-3.5 hours for all 6 notebooks

**Learning Outcomes:**
- Master FFT-based frequency-domain analysis from time-domain data
- Construct mastercurves via time-temperature superposition (WLF equation)
- Quantify material character using mutation number (solid vs fluid classification)
- Analyze nonlinear viscoelasticity with LAOS/OWChirp protocols
- Apply noise-robust differentiation to experimental data
- Chain multiple transforms in analysis workflows
- Interpret Cole-Cole plots, Lissajous curves, and shift factor plots
- Validate transform quality with diagnostic metrics (R², overlap error, harmonics)

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
| **[02-batch-processing.ipynb](advanced/02-batch-processing.ipynb)** | Parallel Execution | ~15 min | 🚧 Stub | BatchPipeline basics demonstrated; full version with vmap, aggregation, HDF5 export coming soon |
| **[03-custom-models.ipynb](advanced/03-custom-models.ipynb)** | Model Development | ~20 min | 🚧 Stub | Simplified Burgers model shown; full version with advanced features coming soon |
| **[04-fractional-models-deep-dive.ipynb](advanced/04-fractional-models-deep-dive.ipynb)** | Fractional Calculus | ~10 min | 🚧 Stub | Basic introduction; comprehensive coverage of 11 models coming soon |
| **[05-performance-optimization.ipynb](advanced/05-performance-optimization.ipynb)** | GPU & JIT | ~10 min | 🚧 Stub | Basic concepts; full benchmarking and optimization guide coming soon |
| **[06-frequentist-model-selection.ipynb](advanced/06-frequentist-model-selection.ipynb)** | Frequentist Selection | 45 min | ✅ Complete | ModelComparisonPipeline, AIC/BIC information criteria, AIC weights, evidence ratios, complexity trade-offs |

**Recommended Learning Path:**
1. Start with **01-multi-technique-fitting** - builds on Phase 1 Maxwell/Zener knowledge
2. Continue with **02-batch-processing** - applies multi-technique patterns to multiple datasets
3. Explore **03-custom-models** - learn model development workflow
4. Study **04-fractional-models-deep-dive** - comprehensive fractional model coverage
5. Try **06-frequentist-model-selection** - compare with Bayesian model comparison from Phase 3
6. Complete with **05-performance-optimization** - GPU acceleration and scaling

**Prerequisites:**
- Phase 1 basic/ notebooks (model fitting fundamentals)
- Phase 2 transforms/ notebooks (data processing workflows)
- Phase 3 bayesian/ notebooks (uncertainty quantification; especially for comparing model selection)
- Understanding of JAX acceleration concepts
- For notebook 05: GPU installation (Linux + CUDA only, optional)

**Total Time Investment:** ~2-2.5 hours for complete notebooks (01, 06); ~1 hour for stubs (02-05)

**Learning Outcomes:**
- Master multi-technique constrained fitting for parameter consistency
- Process large batches of datasets efficiently (5-10x speedup via vmap)
- Create custom rheological models with automatic Bayesian capabilities
- Select appropriate fractional models based on data characteristics
- Optimize computational performance using GPU and JIT compilation
- Scale analyses to 10K+ data points efficiently
- Integrate Rheo into production workflows

**GPU Requirements (Notebook 05):**
- **Linux + CUDA 12.1-12.9 only** (GPU acceleration not available on macOS/Windows)
- Installation: `make install-jax-gpu` or see [CLAUDE.md](../CLAUDE.md)
- GPU tests marked with `@pytest.mark.gpu` and skip gracefully if unavailable
- CPU-only execution provided as fallback in all notebooks

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
| Bayesian Basics | [bayesian/01-bayesian-basics](bayesian/01-bayesian-basics.ipynb) | Bayesian | ✅ Complete |
| Prior Selection | [bayesian/02-prior-selection](bayesian/02-prior-selection.ipynb) | Bayesian | ✅ Complete |
| Diagnostics | [bayesian/03-convergence-diagnostics](bayesian/03-convergence-diagnostics.ipynb) | Bayesian | ✅ Complete |
| Model Comparison | [bayesian/04-model-comparison](bayesian/04-model-comparison.ipynb) | Bayesian | ✅ Complete |
| Uncertainty Propagation | [bayesian/05-uncertainty-propagation](bayesian/05-uncertainty-propagation.ipynb) | Bayesian | ✅ Complete |
| Multi-Technique | [advanced/01-multi-technique-fitting](advanced/01-multi-technique-fitting.ipynb) | Advanced | ✅ Complete |
| Batch Processing | [advanced/02-batch-processing](advanced/02-batch-processing.ipynb) | Advanced | ✅ Complete |
| Custom Models | [advanced/03-custom-models](advanced/03-custom-models.ipynb) | Advanced | ✅ Complete |
| Fractional Models | [advanced/04-fractional-models-deep-dive](advanced/04-fractional-models-deep-dive.ipynb) | Advanced | ✅ Complete |
| Performance | [advanced/05-performance-optimization](advanced/05-performance-optimization.ipynb) | Advanced | ✅ Complete |

### By API Level

- **Pipeline API** (Easiest): `pipeline.load().fit().plot().save()` - Start here
- **Modular API** (Medium): `model.fit()` with explicit control
- **Core API** (Advanced): Direct JAX operations for custom workflows

## How to Run Notebooks

### Option 1: Jupyter Lab (Recommended)

```bash
cd /Users/b80985/Projects/Rheo/examples
jupyter lab
```

Then open any `.ipynb` file from the browser interface.

### Option 2: Jupyter Notebook

```bash
cd /Users/b80985/Projects/Rheo/examples
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

**Problem:** `ImportError: No module named 'rheo'`

**Solution:** Install the package in development mode:
```bash
cd /Users/b80985/Projects/Rheo
pip install -e .
```

### "JAX is float32 instead of float64"

**Problem:** Numerical results are inaccurate

**Solution:** JAX requires safe import order. The notebook template includes:
```python
from rheo.core.jax_config import safe_import_jax
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

1. Open an issue on GitHub: https://github.com/rheo/rheo/issues
2. Include the notebook name and specific section
3. Provide error message or unexpected behavior
4. Suggest improvements for clarity or pedagogy

## Next Steps After Examples

Once you've completed these examples:

1. **Read the Documentation:** https://rheo.readthedocs.io/
2. **Explore Source Code:** `/Users/b80985/Projects/Rheo/rheo/`
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
**Rheo Version:** [Check with `import rheo; print(rheo.__version__)`]
