==================
Tutorial Notebooks
==================

Rheo includes 23 comprehensive tutorial notebooks organized into four learning paths:
**Basic Model Fitting** (5), **Transform Workflows** (6), **Bayesian Inference** (6), and **Advanced Patterns** (6).

All notebooks are located in the ``examples/`` directory and demonstrate best practices with
real-world datasets and synthetic data generation patterns.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Navigation
================

.. list-table:: Learning Paths
   :header-rows: 1
   :widths: 25 50 25

   * - Category
     - Description
     - Notebooks
   * - **Basic**
     - Fundamental rheological model fitting
     - 5 notebooks
   * - **Transforms**
     - Data analysis workflows (FFT, TTS, LAOS)
     - 5 notebooks
   * - **Bayesian**
     - Bayesian inference and uncertainty quantification
     - 6 notebooks
   * - **Advanced**
     - Production patterns and custom implementations
     - 5 notebooks

Basic Model Fitting
===================

Foundation tutorials for fundamental rheological models. Each notebook demonstrates:

- Synthetic data generation with known parameters
- NLSQ optimization with JAX acceleration
- Parameter validation (relative error < 1e-6)
- Publication-quality visualization

1. Maxwell Model Fitting
------------------------

**File**: ``examples/basic/01-maxwell-fitting.ipynb``

**Model**: Standard Linear Viscoelastic (SLV) / Maxwell model

**Content**:
   - Stress relaxation data (synthetic, 50 points)
   - Two-parameter fitting: G_0 (elastic modulus), eta (viscosity)
   - Analytical solution validation
   - Parameter bounds and optimization

**Key Concepts**:
   - BaseModel API: ``.fit()``, ``.predict()``
   - Parameter system: ``ParameterSet``
   - Automatic test mode detection
   - NLSQ optimization (5-270x speedup)

**Learning Objectives**:
   - Understand basic model fitting workflow
   - Validate fitted parameters against ground truth
   - Visualize model predictions vs. data

2. Zener Model Fitting
----------------------

**File**: ``examples/basic/02-zener-fitting.ipynb``

**Model**: Standard Linear Solid (SLS) / Zener model

**Content**:
   - Frequency-domain oscillation data (complex modulus)
   - Three-parameter fitting: G_0, G_inf, tau
   - Storage (G') and loss (G'') modulus
   - Real and imaginary components

**Key Concepts**:
   - Complex-valued predictions
   - Frequency-domain fitting
   - Multiple parameter estimation
   - Model selection criteria

**Learning Objectives**:
   - Work with oscillatory rheology data
   - Understand complex modulus interpretation
   - Fit multi-parameter models

3. SpringPot Fitting (Fractional Calculus)
------------------------------------------

**File**: ``examples/basic/03-springpot-fitting.ipynb``

**Model**: Fractional SpringPot element

**Content**:
   - Power-law relaxation (fractional derivative behavior)
   - Two-parameter fitting: alpha (fractional order), tau (relaxation time)
   - Mittag-Leffler function evaluation
   - Fractional calculus introduction

**Key Concepts**:
   - Fractional calculus models
   - Power-law relaxation and creep
   - Mittag-Leffler functions (1-parameter and 2-parameter)
   - Subdiffusive vs. superdiffusive behavior

**Learning Objectives**:
   - Understand fractional rheological models
   - Fit power-law materials
   - Interpret fractional order alpha

4. Bingham Model Fitting (Yield Stress)
---------------------------------------

**File**: ``examples/basic/04-bingham-fitting.ipynb``

**Model**: Bingham plastic model

**Content**:
   - Steady shear flow curves (viscosity vs. shear rate)
   - Two-parameter fitting: tau_0 (yield stress), eta_p (plastic viscosity)
   - Yield stress materials (pastes, gels, slurries)
   - Flow curve analysis

**Key Concepts**:
   - Yield stress materials
   - Flow curve fitting
   - Rotation test mode
   - Non-Newtonian fluids

**Learning Objectives**:
   - Identify yield stress from flow data
   - Fit Bingham plastic model
   - Understand shear-thinning behavior

5. Power-Law Fitting (Shear-Thinning)
-------------------------------------

**File**: ``examples/basic/05-power-law-fitting.ipynb``

**Model**: Ostwald-de Waele power-law model

**Content**:
   - Shear-thinning fluid flow curves
   - Two-parameter fitting: K (consistency index), n (flow index)
   - Pseudoplastic behavior (n < 1)
   - Viscosity-shear rate relationship

**Key Concepts**:
   - Power-law fluids
   - Shear-thinning and shear-thickening
   - Flow consistency index
   - Non-Newtonian viscosity

**Learning Objectives**:
   - Fit power-law models to flow data
   - Interpret flow index n
   - Understand pseudoplastic behavior

Transform Workflows
===================

Data analysis techniques for advanced rheological characterization.

6. FFT Analysis
---------------

**File**: ``examples/transforms/01-fft-analysis.ipynb``

**Transform**: Fast Fourier Transform (time <-> frequency domain)

**Content**:
   - Time-domain relaxation -> frequency-domain G'(omega), G''(omega)
   - FFT validation with Maxwell analytical solution
   - Nyquist frequency and sampling considerations
   - Inverse FFT: frequency -> time

**Key Concepts**:
   - FFT for rheological interconversion
   - Complex modulus calculation from time data
   - Sampling theory and aliasing
   - Validation against analytical solutions

**Learning Objectives**:
   - Convert time-domain to frequency-domain data
   - Understand FFT limitations and artifacts
   - Validate FFT accuracy

7. Mastercurve Construction (Time-Temperature Superposition)
------------------------------------------------------------

**File**: ``examples/transforms/02-mastercurve-tts.ipynb``

**Transform**: Time-Temperature Superposition (TTS)

**Content**:
   - Multi-temperature frequency sweeps
   - Horizontal shift factor (a_T) calculation
   - WLF equation fitting (Williams-Landel-Ferry)
   - Reference temperature selection

**Key Concepts**:
   - Time-temperature equivalence
   - Horizontal shifting
   - WLF parameters (C_1, C_2, T_ref)
   - Master curve construction

**Dataset**: ``data/experimental/frequency_sweep_tts.txt`` (TRIOS format)

**Learning Objectives**:
   - Construct master curves from multi-temp data
   - Fit WLF equation
   - Understand thermorheological simplicity

7b. WLF Parameter Validation (Synthetic TTS)
--------------------------------------------

**File**: ``examples/transforms/02b-mastercurve-wlf-validation.ipynb``

**Transform**: WLF parameter extraction and validation

**Content**:
   - Synthetic multi-temperature data with **known WLF parameters**
   - WLF parameter extraction accuracy validation (C_1=17.44, C_2=51.6K)
   - Fractional Maxwell liquid fitting to mastercurve
   - Temperature-by-temperature prediction validation
   - Shift factor visualization and WLF linearization checks

**Key Concepts**:
   - WLF equation accuracy assessment
   - Ground truth parameter recovery
   - Model fitting to extended frequency range
   - Temperature-dependent predictions

**Learning Objectives**:
   - Validate WLF extraction workflow
   - Understand parameter error propagation
   - Compare fitted vs true parameters

8. Mutation Number (Material Classification)
--------------------------------------------

**File**: ``examples/transforms/03-mutation-number.ipynb``

**Transform**: Mutation number calculation

**Content**:
   - Material classification: solid, viscoelastic, fluid
   - Mutation number from G'(omega) and G''(omega)
   - Three synthetic materials demonstration
   - Gelation point detection

**Key Concepts**:
   - Mutation number theory
   - Solid-like vs. fluid-like behavior
   - Viscoelastic character quantification
   - Time-evolving materials (gelation)

**Learning Objectives**:
   - Calculate mutation number from oscillatory data
   - Classify materials by viscoelastic character
   - Detect gelation transitions

9. OWChirp LAOS Analysis
------------------------

**File**: ``examples/transforms/04-owchirp-laos-analysis.ipynb``

**Transform**: Optimally Windowed Chirp (OWChirp) protocol

**Content**:
   - Large Amplitude Oscillatory Shear (LAOS) analysis
   - Harmonic extraction from time-domain waveforms
   - Nonlinear viscoelasticity quantification
   - Fourier decomposition

**Key Concepts**:
   - LAOS (Large Amplitude Oscillatory Shear)
   - OWChirp protocol
   - Harmonic analysis
   - Nonlinear rheology

**Learning Objectives**:
   - Analyze LAOS data with OWChirp
   - Extract higher harmonics
   - Quantify nonlinear viscoelastic response

10. Smooth Derivative Calculation
---------------------------------

**File**: ``examples/transforms/05-smooth-derivative.ipynb``

**Transform**: Noise-robust derivative calculation

**Content**:
   - Numerical differentiation of noisy data
   - Savitzky-Golay filter
   - Comparison: finite differences vs. smoothing methods
   - Derivative accuracy validation

**Key Concepts**:
   - Noise amplification in derivatives
   - Savitzky-Golay smoothing
   - Filter window length selection
   - Accuracy vs. smoothness trade-off

**Learning Objectives**:
   - Compute derivatives from noisy rheological data
   - Choose appropriate smoothing parameters
   - Validate derivative accuracy

Bayesian Inference
==================

Bayesian parameter estimation, uncertainty quantification, and model comparison.
All notebooks use NLSQ -> NUTS warm-start workflow (2-5x faster convergence).

11. Bayesian Basics
-------------------

**File**: ``examples/bayesian/01-bayesian-basics.ipynb``

**Content**:
   - NLSQ point estimation (fast optimization)
   - NumPyro NUTS sampling (Bayesian inference)
   - Warm-start workflow demonstration
   - Posterior distribution visualization

**Key Concepts**:
   - Two-stage workflow: NLSQ -> NUTS
   - Warm-start initialization
   - Posterior samples
   - Credible intervals (95%, 68%)

**Learning Objectives**:
   - Understand Bayesian workflow in Rheo
   - Compare point estimates vs. posterior distributions
   - Interpret credible intervals

12. Prior Selection and Sensitivity
-----------------------------------

**File**: ``examples/bayesian/02-prior-selection.ipynb``

**Content**:
   - Prior distribution choices (uniform, normal, log-normal)
   - Prior sensitivity analysis
   - Informative vs. uninformative priors
   - Prior-posterior comparison

**Key Concepts**:
   - Prior elicitation
   - Prior impact on posterior
   - Informative priors from literature
   - Weakly informative priors

**Learning Objectives**:
   - Choose appropriate priors
   - Assess prior influence on results
   - Use domain knowledge in priors

13. Convergence Diagnostics
---------------------------

**File**: ``examples/bayesian/03-convergence-diagnostics.ipynb``

**Content**:
   - R-hat (Gelman-Rubin statistic)
   - Effective Sample Size (ESS)
   - Divergent transitions analysis
   - ArviZ diagnostic plots (6 types)

**Key Concepts**:
   - MCMC convergence assessment
   - R-hat < 1.01 criterion
   - ESS > 400 recommendation
   - Divergence troubleshooting

**ArviZ Plots Demonstrated**:
   1. **Pair plot**: Parameter correlations, divergences
   2. **Forest plot**: Credible intervals comparison
   3. **Energy plot**: NUTS sampler diagnostic
   4. **Autocorrelation plot**: Mixing quality
   5. **Rank plot**: Convergence diagnostic
   6. **ESS plot**: Effective sample size

**Learning Objectives**:
   - Check MCMC convergence with R-hat and ESS
   - Use ArviZ diagnostic suite
   - Troubleshoot divergent transitions

14. Bayesian Model Comparison
-----------------------------

**File**: ``examples/bayesian/04-model-comparison.ipynb``

**Content**:
   - WAIC (Widely Applicable Information Criterion)
   - LOO (Leave-One-Out Cross-Validation)
   - Model selection demonstration
   - Predictive performance comparison

**Key Concepts**:
   - Bayesian model selection
   - Information criteria (WAIC, LOO)
   - Model comparison workflow
   - Overfitting detection

**Learning Objectives**:
   - Compare multiple models with WAIC/LOO
   - Select best model for data
   - Understand model complexity trade-offs

15. Uncertainty Propagation
---------------------------

**File**: ``examples/bayesian/05-uncertainty-propagation.ipynb``

**Content**:
   - Credible intervals for predictions
   - Posterior predictive distributions
   - Parameter uncertainty visualization
   - Prediction bands (95%, 68%)

**Key Concepts**:
   - Predictive uncertainty
   - Credible bands
   - Posterior predictive checks
   - Uncertainty quantification

**Learning Objectives**:
   - Propagate parameter uncertainty to predictions
   - Visualize prediction uncertainty
   - Understand sources of uncertainty

16. SPP Analysis for Yield-Stress LAOS
--------------------------------------

**File**: ``examples/bayesian/08-spp-laos.ipynb``

**Content**:
   - Sequence of Physical Processes (SPP) framework
   - Yield-stress material LAOS analysis
   - Cage modulus and yield stress extraction
   - Bayesian inference with NLSQ warm-start

**Key Concepts**:
   - Time-domain LAOS analysis (no Fourier)
   - Cage modulus G_cage
   - Static and dynamic yield stress
   - SPP vs Fourier comparison

**Learning Objectives**:
   - Apply SPP analysis to yield-stress LAOS data
   - Extract physical parameters (cage modulus, yield stress)
   - Quantify uncertainty in SPP parameters
   - Understand SPP limitations and best practices

Advanced Workflows
==================

Production patterns, custom model development, and performance optimization.

17. Multi-Technique Fitting
---------------------------

**File**: ``examples/advanced/01-multi-technique-fitting.ipynb``

**Content**:
   - Simultaneous fitting of multiple test modes
   - Combined relaxation + oscillation data
   - Shared parameters across datasets
   - Global optimization strategy

**Key Concepts**:
   - Multi-objective fitting
   - Data fusion from multiple techniques
   - Shared parameter constraints
   - Weighted residuals

**Learning Objectives**:
   - Fit models to multiple datasets simultaneously
   - Combine different rheological test modes
   - Improve parameter identifiability

18. Batch Processing
--------------------

**File**: ``examples/advanced/02-batch-processing.ipynb``

**Content**:
   - Process multiple datasets in parallel
   - BatchPipeline API demonstration
   - Automated report generation
   - Result aggregation

**Key Concepts**:
   - Batch processing workflows
   - Pipeline automation
   - Parallel processing
   - Result consolidation

**Learning Objectives**:
   - Process multiple samples efficiently
   - Automate repetitive analysis tasks
   - Generate batch reports

19. Custom Model Development
----------------------------

**File**: ``examples/advanced/03-custom-models.ipynb``

**Content**:
   - Implement custom rheological model
   - Inherit from BaseModel
   - Register custom model
   - Integration with Pipeline API

**Key Concepts**:
   - Custom model interface
   - Model registry system
   - ``_fit()`` and ``_predict()`` implementation
   - Plugin architecture

**Learning Objectives**:
   - Create custom rheological models
   - Integrate models into Rheo ecosystem
   - Use model registry for discovery

20. Fractional Models Deep Dive
-------------------------------

**File**: ``examples/advanced/04-fractional-models-deep-dive.ipynb``

**Content**:
   - 11 fractional model variants
   - Mittag-Leffler functions (1-param and 2-param)
   - Fractional derivatives in rheology
   - Model comparison for power-law materials

**Key Concepts**:
   - Fractional calculus theory
   - Mittag-Leffler special functions
   - Fractional Maxwell, Zener, Kelvin-Voigt models
   - Power-law behavior modeling

**Learning Objectives**:
   - Understand fractional rheological models
   - Apply Mittag-Leffler functions
   - Select appropriate fractional model

21. Performance Optimization
----------------------------

**File**: ``examples/advanced/05-performance-optimization.ipynb``

**Content**:
   - JAX JIT compilation
   - GPU acceleration (CUDA)
   - Performance benchmarking
   - NLSQ vs. scipy comparison (5-270x speedup)

**Key Concepts**:
   - JAX acceleration
   - GPU computing
   - JIT compilation
   - Performance profiling

**Learning Objectives**:
   - Enable GPU acceleration
   - Use JAX for performance
   - Benchmark optimization methods

22. Frequentist Model Selection
-------------------------------

**File**: ``examples/advanced/06-frequentist-model-selection.ipynb``

**Content**:
   - `ModelComparisonPipeline` API for automated comparison
   - Systematic comparison of 5 models (Maxwell, Zener, fractional variants)
   - AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)
   - AIC weights and evidence ratios for model selection
   - Complexity vs performance trade-off analysis
   - Residual analysis for all models

**Key Concepts**:
   - Frequentist information criteria (AIC/BIC)
   - Model complexity penalty
   - Evidence ratios (relative model likelihood)
   - Comparison with Bayesian WAIC/LOO (see bayesian/04-model-comparison.ipynb)

**Learning Objectives**:
   - Use `ModelComparisonPipeline` for batch model fitting
   - Interpret AIC/BIC information criteria
   - Calculate and interpret AIC weights
   - Understand DeltaAIC thresholds (2, 4, 7, 10)
   - Compare frequentist vs Bayesian model selection

Running the Notebooks
=====================

Prerequisites
-------------

Install Rheo with all dependencies:

.. code-block:: bash

   pip install rheojax[all]

Or install with specific extras:

.. code-block:: bash

   pip install rheojax[bayesian]  # NumPyro + ArviZ
   pip install rheojax[gpu]       # JAX with CUDA (Linux only)

Jupyter Setup
-------------

Launch Jupyter from the examples directory:

.. code-block:: bash

   cd examples/
   jupyter notebook

Or use JupyterLab:

.. code-block:: bash

   cd examples/
   jupyter lab

Executing Notebooks
-------------------

All notebooks are designed to run independently with:

- **Synthetic data generation** (no external data files needed for basic/bayesian/advanced)
- **Fixed random seed** (42) for reproducibility
- **Known ground truth** for validation
- **Self-contained code** (all imports and data generation included)

Some transform notebooks require experimental data files from ``examples/data/experimental/``.

GPU Acceleration (Optional)
---------------------------

For GPU acceleration (Linux + CUDA 12.1-12.9):

.. code-block:: bash

   make install-jax-gpu

   # Verify GPU detection
   python -c "import jax; print('Devices:', jax.devices())"
   # Expected: [cuda(id=0)]

**Note**: CPU-only JAX works on all platforms (Linux, macOS, Windows). GPU provides 20-100x speedup for large datasets.

Data Files
==========

Dataset Organization
--------------------

.. code-block:: text

   examples/data/
   |-- experimental/    # 8 real instrument files
   |   |-- polypropylene_relaxation.csv
   |   |-- polystyrene_creep.csv
   |   |-- cellulose_hydrogel_flow.csv
   |   |-- frequency_sweep_tts.txt (TRIOS)
   |   |-- owchirp_tts.txt (TRIOS, 80 MB)
   |   |-- owchirp_tcs.txt (TRIOS, 66 MB)
   |   |-- creep_experiment.txt (TRIOS)
   |   \-- multi_technique.txt (TRIOS)
   \-- synthetic/
       \-- ...

Synthetic Data Pattern
----------------------

Most notebooks generate synthetic data in-notebook:

.. code-block:: python

   import numpy as np
   from rheojax.models.maxwell import Maxwell

   # Set seed for reproducibility
   np.random.seed(42)

   # Generate time array
   t = np.logspace(-2, 2, 50)  # 0.01 to 100 s

   # Known parameters for validation
   G0_true = 1e5  # Pa
   eta_true = 1e3  # Pa*s

   # Generate clean data
   model = Maxwell()
   model.parameters.set_value('G0', G0_true)
   model.parameters.set_value('eta', eta_true)
   G_clean = model.predict(t)

   # Add realistic noise (1.5%)
   noise = np.random.normal(0, 0.015 * G_clean)
   G_data = G_clean + noise

**Advantages**:
   - Known ground truth (validate fitted parameters)
   - Reproducible (fixed seed)
   - Educational (see generation code)
   - No external dependencies

Learning Paths
==============

For Beginners
-------------

Start with basic model fitting to understand fundamentals:

1. ``basic/01-maxwell-fitting.ipynb``
2. ``basic/02-zener-fitting.ipynb``
3. ``transforms/01-fft-analysis.ipynb``
4. ``bayesian/01-bayesian-basics.ipynb``

For Intermediate Users
----------------------

Explore transforms and Bayesian workflows:

1. ``transforms/02-mastercurve-tts.ipynb``
2. ``bayesian/03-convergence-diagnostics.ipynb``
3. ``bayesian/04-model-comparison.ipynb``
4. ``advanced/01-multi-technique-fitting.ipynb``

For Advanced Users
------------------

Deep dive into fractional models and custom development:

1. ``basic/03-springpot-fitting.ipynb``
2. ``advanced/04-fractional-models-deep-dive.ipynb``
3. ``advanced/03-custom-models.ipynb``
4. ``advanced/05-performance-optimization.ipynb``

For Production Workflows
------------------------

Focus on automation and best practices:

1. ``advanced/02-batch-processing.ipynb``
2. ``bayesian/05-uncertainty-propagation.ipynb``
3. ``advanced/01-multi-technique-fitting.ipynb``
4. ``advanced/05-performance-optimization.ipynb``

Additional Resources
====================

**Related Documentation**
   - :doc:`../user_guide/bayesian_inference` - Complete Bayesian inference guide
   - :doc:`../user_guide/pipeline_api` - Pipeline API reference
   - :doc:`../user_guide/transforms` - Transform workflows
   - :doc:`../user_guide/modular_api` - Modular API patterns

**Example Data**
   - ``examples/data/README.md`` - Dataset catalog and loading instructions
   - ``examples/README.md`` - Overview of all 22 notebooks

**External Resources**
   - JAX Documentation: https://jax.readthedocs.io/
   - NumPyro Documentation: https://num.pyro.ai/
   - ArviZ Documentation: https://arviz-devs.github.io/arviz/

Contributing
============

To contribute new tutorial notebooks:

1. Follow the ``.notebook_template.ipynb`` structure
2. Use synthetic data generation with fixed seed (42)
3. Include validation against ground truth
4. Add comprehensive markdown explanations
5. Test notebook execution (``jupyter nbconvert --execute``)
6. Update this documentation page
7. Submit pull request

See ``CONTRIBUTING.md`` for detailed guidelines.
