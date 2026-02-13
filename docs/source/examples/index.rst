==================
Tutorial Notebooks
==================

RheoJAX includes 244 comprehensive tutorial notebooks organized into 21 categories:
**Basic** (5), **Transforms** (8), **Bayesian** (9), **Advanced** (10), **I/O** (1),
plus 14 model family tutorial suites and a 31-notebook verification suite.

All notebooks are located in the ``examples/`` directory and demonstrate best practices with
real-world datasets and synthetic data generation patterns.

Quick Navigation
================

.. list-table:: General Tutorials
   :header-rows: 1
   :widths: 25 50 25

   * - Category
     - Description
     - Notebooks
   * - **Basic**
     - Fundamental rheological model fitting
     - 5 notebooks
   * - **Transforms**
     - Data analysis workflows (FFT, TTS, LAOS, SRFS)
     - 8 notebooks
   * - **Bayesian**
     - Bayesian inference and uncertainty quantification (including SPP)
     - 9 notebooks
   * - **Advanced**
     - Production patterns, SGR, and SPP implementations
     - 10 notebooks
   * - **I/O**
     - Data I/O demonstrations
     - 1 notebook

.. list-table:: Model Family Tutorials
   :header-rows: 1
   :widths: 25 50 25

   * - Family
     - Description
     - Notebooks
   * - **DMT**
     - de Souza Mendes-Thompson thixotropic models
     - 6 notebooks
   * - **DMTA**
     - Dynamic Mechanical Thermal Analysis (E* ↔ G* workflows)
     - 8 notebooks
   * - **EPM**
     - Elasto-plastic models (Lattice + Tensorial)
     - 6 notebooks
   * - **FIKH**
     - Fredrickson-IKH + FMLIKH models
     - 12 notebooks
   * - **Fluidity**
     - Fluidity local/nonlocal + Saramito EVP local/nonlocal
     - 24 notebooks
   * - **Giesekus**
     - Giesekus constitutive model (single-mode)
     - 7 notebooks
   * - **HL**
     - Hébraud-Lequeux stochastic model
     - 6 notebooks
   * - **HVM**
     - Hybrid Vitrimer Model (basic + advanced tutorials)
     - 13 notebooks
   * - **HVNM**
     - Hybrid Vitrimer Nanocomposite Model (basic + NLSQ/NUTS)
     - 15 notebooks
   * - **IKH**
     - Isotropic kinematic hardening (MIKH + MLIKH)
     - 12 notebooks
   * - **ITT-MCT**
     - Integration Through Transients Mode-Coupling Theory
     - 12 notebooks
   * - **SGR**
     - Soft Glassy Rheology (Conventional + Generic)
     - 6 notebooks
   * - **STZ**
     - Shear Transformation Zone theory
     - 6 notebooks
   * - **TNT**
     - Transient Network Theory (5 sub-models)
     - 30 notebooks
   * - **VLB**
     - Vasquez-Cook-McKinley transient network (basic + NLSQ/NUTS)
     - 16 notebooks

.. list-table:: Verification Suite
   :header-rows: 1
   :widths: 25 50 25

   * - Category
     - Description
     - Notebooks
   * - **Verification**
     - Cross-model protocol validation + material-specific benchmarks
     - 31 notebooks

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
   - Two-parameter fitting: :math:`G_0` (elastic modulus), :math:`\eta` (viscosity)
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
   - Three-parameter fitting: :math:`G_0`, :math:`G_\infty`, :math:`\tau`
   - Storage (:math:`G'`) and loss (:math:`G''`) modulus
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
   - Two-parameter fitting: :math:`\alpha` (fractional order), :math:`\tau` (relaxation time)
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
   - Two-parameter fitting: :math:`\tau_0` (yield stress), :math:`\eta_p` (plastic viscosity)
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
   - Time-domain relaxation -> frequency-domain :math:`G'(\omega)`, :math:`G''(\omega)`
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
   - WLF parameters (:math:`C_1`, :math:`C_2`, :math:`T_\text{ref}`)
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
   - WLF parameter extraction accuracy validation (:math:`C_1 = 17.44`, :math:`C_2 = 51.6\,\text{K}`)
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
   - Mutation number from :math:`G'(\omega)` and :math:`G''(\omega)`
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
   - Understand Bayesian workflow in RheoJAX
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
   - Cage modulus :math:`G_\text{cage}`
   - Static and dynamic yield stress
   - SPP vs Fourier comparison

**Learning Objectives**:
   - Apply SPP analysis to yield-stress LAOS data
   - Extract physical parameters (cage modulus, yield stress)
   - Quantify uncertainty in SPP parameters
   - Understand SPP limitations and best practices

17. SPP LAOS Workflow (Rogers Defaults, NLSQ→NUTS)
--------------------------------------------------

**File**: ``examples/bayesian/09-spp-rheojax-workflow.ipynb``

**Content**:
   - Rogers-parity defaults (M=39, k=8, num_mode=2, wrapped rate)
   - SPPDecomposer on synthetic LAOS amplitude sweep
   - NLSQ warm-start and NumPyro NUTS posterior diagnostics

**Key Concepts**:
   - Phase-aligned time-domain SPP extraction
   - Warm-started Bayesian inference for yield parameters
   - Practical parameter defaults and when to override

**Learning Objectives**:
   - Run the end-to-end SPP pipeline with recommended defaults
   - Interpret posterior means/credible intervals for σ_sy and exponents
   - Compare SPP outputs against Fourier assumptions

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
   - Integrate models into RheoJAX ecosystem
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

23. TRIOS Chunked Reading
-------------------------

**File**: ``examples/advanced/07-trios_chunked_reading_example.ipynb``

**Content**:
   - Large TRIOS file handling with auto-chunking
   - Memory-efficient loading for files > 5 MB
   - Progress callbacks and memory monitoring

**Key Concepts**:
   - Auto-chunking for large datasets
   - Memory optimization (50-70% reduction)
   - TRIOS file format handling

**Learning Objectives**:
   - Load large TRIOS files efficiently
   - Monitor memory usage during loading
   - Configure chunking parameters

24. Generalized Maxwell Fitting
-------------------------------

**File**: ``examples/advanced/08-generalized_maxwell_fitting.ipynb``

**Content**:
   - Multi-mode Maxwell model fitting
   - Automatic element minimization (R²-based)
   - Prony series representation

**Key Concepts**:
   - Generalized Maxwell Model (GMM)
   - Element search with warm-start optimization
   - Model complexity selection

**Learning Objectives**:
   - Fit multi-mode relaxation spectra
   - Understand R²-based element selection
   - Interpret Prony series parameters

25. Soft Glassy Rheology (SGR)
------------------------------

**File**: ``examples/advanced/09-sgr-soft-glassy-rheology.ipynb``

**Content**:
   - SGRConventional model for soft glassy materials
   - Effective noise temperature (x) interpretation
   - SGRGeneric thermodynamic framework
   - Material phase classification (glass vs fluid)

**Key Concepts**:
   - Soft Glassy Rheology (SGR) theory
   - Effective noise temperature x
   - Power-law rheology from trap dynamics
   - Thixotropy and aging

**Learning Objectives**:
   - Fit SGR models to soft glassy materials
   - Interpret effective temperature x for phase behavior
   - Distinguish glass, power-law fluid, and Newtonian regimes

26. SPP LAOS Tutorial
---------------------

**File**: ``examples/advanced/10-spp-laos-tutorial.ipynb``

**Content**:
   - Sequence of Physical Processes (SPP) framework
   - Time-domain LAOS analysis (no Fourier)
   - Cage modulus and yield stress extraction
   - Comparison with Fourier/Chebyshev methods

**Key Concepts**:
   - SPP framework (Rogers 2012)
   - Cage modulus :math:`G_\text{cage}`
   - Static and dynamic yield stress
   - Phase angle evolution in LAOS

**Learning Objectives**:
   - Apply SPP analysis to yield-stress LAOS
   - Extract physical parameters from LAOS cycles
   - Understand SPP vs Fourier trade-offs

I/O Demonstrations
===================

27. TRIOS Complex Modulus Plot
------------------------------

**File**: ``examples/io/plot_trios_complex_modulus.ipynb``

**Content**:
   - TRIOS file format loading and visualization
   - Complex modulus (:math:`G'`, :math:`G''`) plotting

Model Family Tutorials
======================

Each model family provides 6 protocol-specific notebooks covering: flow curve, startup shear,
stress relaxation, creep, SAOS, and LAOS. Models with multiple sub-models (e.g., FIKH/FMLIKH,
MIKH/MLIKH) have 12 notebooks (6 per sub-model). All notebooks follow a consistent pattern:
synthetic data generation, NLSQ fitting, Bayesian inference (FAST_MODE-aware), and visualization.

DMT — Thixotropic Models (6 notebooks)
---------------------------------------

**Directory**: ``examples/dmt/``

**Model**: de Souza Mendes-Thompson structural-kinetics model with scalar structure parameter.

**Notebooks**:
   1. ``01_dmt_flow_curve.ipynb`` — Steady-state flow curve
   2. ``02_dmt_startup_shear.ipynb`` — Stress overshoot in startup
   3. ``03_dmt_stress_relaxation.ipynb`` — Structural relaxation
   4. ``04_dmt_creep.ipynb`` — Delayed yielding under constant stress
   5. ``05_dmt_saos.ipynb`` — Small-amplitude oscillatory shear
   6. ``06_dmt_laos.ipynb`` — Large-amplitude oscillatory shear

DMTA — Dynamic Mechanical Thermal Analysis (8 notebooks)
---------------------------------------------------------

**Directory**: ``examples/dmta/``

**Feature**: Automatic E* ↔ G* modulus conversion for tensile, bending, and compression oscillatory data.

**Notebooks**:
   1. ``01_dmta_basics.ipynb`` — DMTA fundamentals and deformation modes
   2. ``02_dmta_master_curve.ipynb`` — Temperature-frequency mastercurves for E*
   3. ``03_dmta_fractional_models.ipynb`` — Fractional models for broad relaxation spectra
   4. ``04_dmta_relaxation.ipynb`` — Relaxation modulus from DMTA data
   5. ``05_dmta_vitrimer.ipynb`` — Vitrimer DMTA with HVM model
   6. ``06_dmta_model_selection.ipynb`` — Systematic model selection for E* data
   7. ``07_dmta_tts_pipeline.ipynb`` — TTS pipeline with tensile modulus
   8. ``08_dmta_cross_domain.ipynb`` — Cross-domain fitting (tension + shear)

EPM — Elasto-Plastic Models (6 notebooks)
------------------------------------------

**Directory**: ``examples/epm/``

**Models**: Lattice EPM (mesoscale) and Tensorial EPM (continuum).

**Notebooks**:
   1. ``01_epm_flow_curve.ipynb`` — Flow curve with yield stress
   2. ``02_epm_saos.ipynb`` — Linear viscoelastic response
   3. ``03_epm_startup.ipynb`` — Startup shear and overshoot
   4. ``04_epm_creep.ipynb`` — Creep compliance
   5. ``05_epm_relaxation.ipynb`` — Stress relaxation
   6. ``06_epm_visualization.ipynb`` — Spatial visualization of plastic events

FIKH — Fredrickson-IKH Models (12 notebooks)
---------------------------------------------

**Directory**: ``examples/fikh/``

**Models**: FIKH (Fredrickson IKH, 6 notebooks) and FMLIKH (multi-lambda variant, 6 notebooks).

**Notebooks**:
   1–6. ``01–06_fikh_*.ipynb`` — FIKH: flow curve, startup, relaxation, creep, SAOS, LAOS
   7–12. ``07–12_fmlikh_*.ipynb`` — FMLIKH: flow curve, startup, relaxation, creep, SAOS, LAOS

Fluidity — Fluidity & Saramito EVP Models (24 notebooks)
---------------------------------------------------------

**Directory**: ``examples/fluidity/``

**Models**: Four model variants × 6 protocols each:

- **Fluidity Local** (01–06): Homogeneous thixotropic fluidity model
- **Fluidity Nonlocal** (07–12): Shear-banding capable with diffusion
- **Saramito Local** (13–18): Elastoviscoplastic with tensorial stress
- **Saramito Nonlocal** (19–24): EVP with spatial coupling

Each set of 6 covers: flow curve, startup, creep, relaxation, SAOS, LAOS.

Giesekus — Constitutive Model (7 notebooks)
--------------------------------------------

**Directory**: ``examples/giesekus/``

**Model**: Giesekus model with anisotropic drag (mobility factor α).

**Notebooks**:
   1. ``01_giesekus_flow_curve.ipynb`` — Shear-thinning flow curve
   2. ``02_giesekus_saos.ipynb`` — Linear viscoelastic response
   3. ``03_giesekus_startup.ipynb`` — Stress overshoot dynamics
   4. ``04_giesekus_normal_stresses.ipynb`` — N₁, N₂ predictions
   5. ``05_giesekus_creep.ipynb`` — Creep compliance
   6. ``06_giesekus_relaxation.ipynb`` — Stress relaxation
   7. ``07_giesekus_laos.ipynb`` — Nonlinear LAOS response

HL — Hébraud-Lequeux Model (6 notebooks)
-----------------------------------------

**Directory**: ``examples/hl/``

**Model**: Stochastic mean-field model for soft glassy materials (PDE-based).

**Notebooks**:
   1. ``01_hl_flow_curve.ipynb`` — Yield stress and flow curve
   2. ``02_hl_relaxation.ipynb`` — Stress relaxation
   3. ``03_hl_creep.ipynb`` — Creep with viscosity bifurcation
   4. ``04_hl_saos.ipynb`` — Linear viscoelastic moduli
   5. ``05_hl_startup.ipynb`` — Startup shear
   6. ``06_hl_laos.ipynb`` — LAOS nonlinear response

HVM — Hybrid Vitrimer Model (13 notebooks)
-------------------------------------------

**Directory**: ``examples/hvm/``

**Model**: Constitutive model for vitrimers with permanent, exchangeable (BER/TST), and dissociative subnetworks.

**Notebooks** (Basic, 01–06):
   1. ``01_hvm_saos.ipynb`` — SAOS with dual-Maxwell modes + plateau
   2. ``02_hvm_stress_relaxation.ipynb`` — Multi-timescale relaxation
   3. ``03_hvm_startup_shear.ipynb`` — TST-driven stress overshoot
   4. ``04_hvm_creep.ipynb`` — Creep with evolving natural state
   5. ``05_hvm_flow_curve.ipynb`` — Steady-state flow curve
   6. ``06_hvm_laos.ipynb`` — Nonlinear LAOS

**Notebooks** (Advanced tutorials, 07–13):
   7. ``07_hvm_overview.ipynb`` — Model overview and parameter guide
   8–13. Advanced flow curve, creep, relaxation, startup, SAOS, LAOS tutorials

HVNM — Hybrid Vitrimer Nanocomposite Model (15 notebooks)
----------------------------------------------------------

**Directory**: ``examples/hvnm/``

**Model**: Extends HVM with interphase subnetwork around nanoparticles (Guth-Gold amplification).

**Notebooks** (Basic, 01–07):
   1–6. ``01–06_hvnm_*.ipynb`` — SAOS, relaxation, startup, creep, flow curve, LAOS
   7. ``07_hvnm_limiting_cases.ipynb`` — phi=0 recovers HVM exactly

**Notebooks** (NLSQ→NUTS workflows, 08–15):
   8. ``08_data_intake_and_qc.ipynb`` — Data intake and quality control
   9. ``09_flow_curve_nlsq_nuts.ipynb`` — Flow curve NLSQ → NUTS
   10. ``10_creep_compliance_nlsq_nuts.ipynb`` — Creep NLSQ → NUTS
   11. ``11_stress_relaxation_nlsq_nuts.ipynb`` — Relaxation NLSQ → NUTS
   12. ``12_startup_shear_nlsq_nuts.ipynb`` — Startup NLSQ → NUTS
   13. ``13_saos_nlsq_nuts.ipynb`` — SAOS NLSQ → NUTS
   14. ``14_laos_nlsq_nuts.ipynb`` — LAOS NLSQ → NUTS
   15. ``15_global_multi_protocol.ipynb`` — Multi-protocol global fitting

IKH — Isotropic Kinematic Hardening (12 notebooks)
---------------------------------------------------

**Directory**: ``examples/ikh/``

**Models**: MIKH (modified IKH, 6 notebooks) and MLIKH (multi-lambda IKH, 6 notebooks).

**Notebooks**:
   1–6. ``01–06_mikh_*.ipynb`` — MIKH: flow curve, startup, relaxation, creep, SAOS, LAOS
   7–12. ``07–12_mlikh_*.ipynb`` — MLIKH: flow curve, startup, relaxation, creep, SAOS, LAOS

ITT-MCT — Mode-Coupling Theory (12 notebooks)
----------------------------------------------

**Directory**: ``examples/itt_mct/``

**Models**: F₁₂ Schematic (6 notebooks) and Isotropic with S(k) input (6 notebooks).

**Notebooks**:
   1–6. ``01–06_schematic_*.ipynb`` — Schematic: flow curve, startup, relaxation, creep, SAOS, LAOS
   7–12. ``07–12_isotropic_*.ipynb`` — Isotropic: flow curve, startup, relaxation, creep, SAOS, LAOS

SGR — Soft Glassy Rheology (6 notebooks)
-----------------------------------------

**Directory**: ``examples/sgr/``

**Models**: SGRConventional (Sollich 1998) and SGRGeneric (Fuereder & Ilg 2013).

**Notebooks**:
   1. ``01_sgr_flow_curve.ipynb`` — Flow curve with noise temperature x
   2. ``02_sgr_stress_relaxation.ipynb`` — Power-law relaxation
   3. ``03_sgr_saos.ipynb`` — Linear viscoelastic moduli
   4. ``04_sgr_creep.ipynb`` — Creep compliance
   5. ``05_sgr_startup.ipynb`` — Startup shear
   6. ``06_sgr_laos.ipynb`` — Nonlinear LAOS

STZ — Shear Transformation Zone (6 notebooks)
----------------------------------------------

**Directory**: ``examples/stz/``

**Model**: STZ theory for amorphous solids (Falk & Langer).

**Notebooks**:
   1. ``01_stz_flow_curve.ipynb`` — Flow curve with yield stress
   2. ``02_stz_startup_shear.ipynb`` — Startup transient
   3. ``03_stz_stress_relaxation.ipynb`` — Stress relaxation
   4. ``04_stz_creep.ipynb`` — Creep compliance
   5. ``05_stz_saos.ipynb`` — Linear viscoelastic response
   6. ``06_stz_laos.ipynb`` — LAOS nonlinear response

TNT — Transient Network Theory (30 notebooks)
----------------------------------------------

**Directory**: ``examples/tnt/``

**Models**: 5 sub-models × 6 protocols each:

- **SingleMode** (01–06): Single relaxation mode
- **Cates** (07–12): Living polymer reptation-reaction model
- **LoopBridge** (13–18): Loop-bridge topology switching
- **MultiSpecies** (19–24): Multi-species reaction network
- **StickyRouse** (25–30): Sticky Rouse dynamics

Each set covers: flow curve, startup, relaxation, creep, SAOS, LAOS.

VLB — Transient Network Models (16 notebooks)
----------------------------------------------

**Directory**: ``examples/vlb/``

**Models**: Vasquez-Cook-McKinley (VLB) transient network with Bell, FENE, and Nonlocal extensions.

**Notebooks** (Basic, 01–10):
   1–6. ``01–06_vlb_*.ipynb`` — Flow curve, startup, relaxation, creep, SAOS, LAOS
   7. ``07_vlb_bayesian_workflow.ipynb`` — Bayesian inference workflow
   8. ``08_vlb_bell_shear_thinning.ipynb`` — Bell model (force-enhanced breakage)
   9. ``09_vlb_fene_extensional.ipynb`` — FENE finite extensibility
   10. ``10_vlb_nonlocal_banding.ipynb`` — Shear banding PDE

**Notebooks** (NLSQ→NUTS workflows, 11–16):
   11–16. ``11–16_vlb_*_nlsq_to_nuts.ipynb`` — NLSQ → NUTS for 6 protocols

Verification Suite (31 notebooks)
=================================

**Directory**: ``examples/verification/``

Cross-model validation notebooks that verify protocol implementations against known analytical
solutions and experimental data.

**Protocol validators** (7 notebooks):
   - ``00_verification_index.ipynb`` — Verification suite overview
   - ``01–06_validate_*.ipynb`` — Flow curve, creep, relaxation, startup, SAOS, LAOS

**Material-specific benchmarks** (24 notebooks):
   - ``creep/`` — 3 notebooks (mucus, perihepatic abscess, polystyrene)
   - ``oscillation/`` — 13 notebooks (mastercurves, model evaluation, 11 material-specific)
   - ``relaxation/`` — 7 notebooks (fish muscle, laponite, foams, polyethylene, polypropylene, polystyrene, time master)
   - ``rotation/`` — 1 notebook (emulsion flow curve)

Running the Notebooks
=====================

Prerequisites
-------------

Install RheoJAX with all dependencies:

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

For GPU acceleration (Linux + CUDA 12+ or 13+):

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
   from rheojax.models import Maxwell

   # Set seed for reproducibility
   np.random.seed(42)

   # Generate time array
   t = np.logspace(-2, 2, 50)  # 0.01 to 100 s

   # Known parameters for validation
   G0_true = 1e5  # Pa
   eta_true = 1e3  # Pa·s

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

For Constitutive ODE Models
---------------------------

Models based on ODE/PDE integration with diffrax:

1. ``giesekus/01_giesekus_flow_curve.ipynb`` — Start with a single ODE model
2. ``vlb/01_vlb_flow_curve.ipynb`` — Transient network theory
3. ``hvm/07_hvm_overview.ipynb`` — Vitrimer model overview
4. ``hvnm/08_data_intake_and_qc.ipynb`` — Full NLSQ→NUTS pipeline

For Thixotropy & Yielding
--------------------------

Models for thixotropic, yield-stress, and glassy materials:

1. ``dmt/01_dmt_flow_curve.ipynb`` — Thixotropic structure kinetics
2. ``fluidity/01_fluidity_local_flow_curve.ipynb`` — Fluidity model
3. ``ikh/01_mikh_flow_curve.ipynb`` — Kinematic hardening
4. ``stz/01_stz_flow_curve.ipynb`` — Amorphous solids

For Dense Suspensions & Glasses
-------------------------------

Microscopic and mesoscale models:

1. ``sgr/01_sgr_flow_curve.ipynb`` — Soft Glassy Rheology
2. ``itt_mct/01_schematic_flow_curve.ipynb`` — MCT schematic model
3. ``hl/01_hl_flow_curve.ipynb`` — Hébraud-Lequeux stochastic model
4. ``epm/01_epm_flow_curve.ipynb`` — Elasto-plastic models

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
   - ``examples/README.md`` - Overview of all 240+ notebooks

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
