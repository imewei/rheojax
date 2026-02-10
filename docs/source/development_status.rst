.. _development-status:

Development Status & Performance
================================

This page tracks the development history of RheoJAX across 10 completed phases
and provides performance benchmarks.

Development Phases
------------------

**Phase 1 (Complete)**: Core Infrastructure
   - Base abstractions (BaseModel, BaseTransform, RheoData, Parameters)
   - Test mode detection system
   - Mittag-Leffler functions for fractional calculus
   - Optimization integration with JAX gradients (NLSQ 5-270x speedup)
   - Model and Transform registry system
   - File I/O (TRIOS, CSV, Excel, Anton Paar readers; HDF5, Excel writers)
   - Visualization with matplotlib

**Phase 2 (Complete)**: Models and Transforms
   - 53 rheological models across 22 families
   - 7 data transforms (FFT, Mastercurve/TTS with auto-shift, Mutation Number, OWChirp/LAOS, Smooth Derivative, SRFS, SPP Decomposer)
   - Pipeline API for fluent workflows
   - 56 tutorial notebooks (basic, transforms, bayesian, advanced, vlb, hvm, hvnm)

**Phase 3 (Complete)**: Bayesian Inference
   - NumPyro NUTS sampling with NLSQ warm-start (2-5x faster convergence)
   - Uncertainty quantification via credible intervals and posterior distributions
   - ArviZ integration (6 diagnostic plot types: pair, forest, energy, autocorr, rank, ESS)
   - Model comparison (WAIC/LOO)
   - BayesianMixin: All 53 models support Bayesian inference
   - BayesianPipeline with fluent API for NLSQ -> NUTS workflows
   - 9 Bayesian inference tutorial notebooks (including SPP LAOS)

**Phase 4 (Complete)**: Performance & Correctness (v0.3.0-v0.4.0)
   - **v0.3.0**: Generalized Maxwell Model with transparent element minimization (R-based)
   - **v0.3.0**: Automatic shift factors for time-temperature superposition (TTS)
   - **v0.3.0**: Bayesian prior safety mechanism (tiered hard_failure/suspicious/good)
   - **v0.3.1**: JAX-native foundation (30-45% improvement) - 5 foundational optimizations
   - **v0.3.2**: Vectorization optimizations (50-75% cumulative) - intelligent convergence & batching
   - **v0.4.0**: Mode-aware Bayesian inference (CRITICAL correctness fix for creep/oscillation)
   - **v0.4.0**: GMM element search optimization (2-5x speedup with warm-start)
   - **v0.4.0**: TRIOS auto-chunking for large files (50-70% memory reduction)

**Phase 5 (Complete)**: Soft Glassy Rheology & SPP (v0.5.0)
   - SGRConventional model (Sollich 1998) for soft glassy materials
   - SGRGeneric model (GENERIC thermodynamic framework, Fuereder & Ilg 2013)
   - SGR kernel functions (Fourier transforms, yield stress, aging dynamics)
   - SRFS transform (Strain-Rate Frequency Superposition for flow curves)
   - Shear banding detection and coexistence analysis utilities
   - SPPDecomposer transform (Sequence of Physical Processes for LAOS)
   - SPPYieldStress model with Bayesian inference support
   - SPPAmplitudeSweepPipeline for amplitude sweep workflows

**Phase 6 (Complete)**: Shear Transformation Zone (v0.6.0)
   - STZConventional model (Langer 2008 effective temperature formulation)
   - Three complexity variants: minimal, standard, full
   - Multi-protocol support: steady shear, transient, SAOS, LAOS
   - State evolution: effective temperature, STZ density, orientation

**Phase 7 (Complete)**: Elasto-Plastic Models (v0.6.0)
   - LatticeEPM: Mesoscopic lattice model with FFT-based stress redistribution
   - TensorialEPM: Scaffolding for full tensor implementation
   - EPM Kernels: JAX-accelerated Eshelby propagator and plastic event logic

**Phase 8 (Complete)**: Advanced Constitutive Models (v0.6.0)
   - Fluidity-Saramito EVP: Tensorial viscoelasticity with thixotropic fluidity
   - IKH/FIKH: Isotropic-kinematic hardening with fractional variants (4 models)
   - Hebraud-Lequeux: Mean-field model for concentrated emulsions
   - Giesekus: Single-mode and multi-mode nonlinear viscoelastic models
   - DMT: de Souza Mendes-Thompson thixotropic models (local + nonlocal)
   - ITT-MCT: Mode-Coupling Theory for dense colloids (schematic + isotropic)

**Phase 9 (Complete)**: Transient Network Models (v0.6.0)
   - TNT: 5 transient network variants (SingleMode, LoopBridge, StickyRouse, Cates, MultiSpecies)
   - VLB: 4 Vernerey-Long-Brighenti models (Local, MultiNetwork, Variant with Bell/FENE, Nonlocal PDE)
   - Distribution-tensor formulation with analytical SAOS + diffrax ODE for transients

**Phase 10 (Complete)**: Vitrimer and Nanocomposite Models (v0.6.0)
   - HVM: Hybrid Vitrimer Model with 3-subnetwork architecture (P + E + D)
   - HVNM: Hybrid Vitrimer Nanocomposite Model extending HVM with interphase network
   - TST kinetics with Arrhenius temperature dependence
   - Factor-of-2 relaxation, Guth-Gold strain amplification for HVNM
   - 5 factory methods each for limiting cases

----

Performance Benchmarks
----------------------

RheoJAX delivers exceptional performance through JAX acceleration and systematic optimizations (v0.3.1-v0.3.2):

.. list-table:: Performance Benchmarks (v0.6.0)
   :header-rows: 1
   :widths: 40 20 20 20

   * - Operation
     - Baseline (NumPy/SciPy)
     - v0.4.0 (JAX + Optimizations)
     - Speedup
   * - Mittag-Leffler (1000 points)
     - 45 ms
     - 0.8 ms
     - 56x
   * - NLSQ Parameter Optimization
     - 2.5 s (scipy)
     - 0.08 s (NLSQ)
     - 30x
   * - Mastercurve Transform (10 datasets)
     - 8.5 s
     - 1.7 s (v0.3.2 vectorization)
     - 5x
   * - GMM Element Minimization (N=10)
     - 50 s (cold-start)
     - 10 s (v0.4.0 warm-start)
     - 5x
   * - TRIOS Large File Load (50k points)
     - 35 MB peak memory
     - 11 MB (v0.4.0 auto-chunk)
     - 68% reduction

**Cumulative Improvements v0.3.1-v0.3.2**: 50-75% end-to-end latency reduction through JAX-native foundation, JIT compilation, vectorization, and intelligent convergence.

Technology Stack
----------------

**Core Dependencies**
   - Python 3.12+
   - JAX 0.8.0+ for acceleration and automatic differentiation
   - NLSQ 0.6.6+ for GPU-accelerated optimization
   - NumPyro for Bayesian inference (MCMC NUTS sampling)
   - ArviZ 0.15.0+ for Bayesian visualization and diagnostics
   - NumPy, SciPy for numerical operations
   - Matplotlib for visualization
   - h5py, pandas, openpyxl for I/O

**Optional Dependencies**
   - CUDA 12+ or 13+ for GPU acceleration (Linux only)
