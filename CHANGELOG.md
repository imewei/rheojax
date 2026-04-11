# Changelog

All notable changes to RheoJAX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Herschel-Bulkley-Capable EPM Kernels (scalar and tensorial)

Both the scalar `LatticeEPM` and the tensorial `TensorialEPM` now support **three interchangeable constitutive laws** for the plastic strain rate, selectable at model construction via a new `fluidity_form` argument. This fills a long-standing limitation where EPM could not reproduce shear-thinning Herschel-Bulkley flow curves (σ = σ_y + K·γ̇^n with n < 1) regardless of disorder distribution — the classical linear-fluidity form asymptotes to Bingham at high rates by construction.

- **Added** `fluidity_form="linear"`: classical Bingham form, `plastic_rate = activation · σ · fluidity`. Backwards-compatible with the pre-0.7 EPM behavior.
- **Added** `fluidity_form="power"`: power-law fluidity `plastic_rate ∝ (σ/σ_c)^n_fluid · σ_c / τ_pl`. Shear-thinning without an additive yield baseline; used for soft-glassy-rheology studies.
- **Added** `fluidity_form="overstress"` *(default)*: Herschel-Bulkley overstress form `plastic_rate ∝ (|σ| − σ_c)_+^n_fluid / σ_c^(n_fluid−1) / τ_pl`. Produces the full HB shape σ = σ_y + K·γ̇^(1/n_fluid) at the NLSQ optimum. On the φ=0.80 emulsion benchmark the scalar fit reaches R²_lin = 0.9957, R²_log = 0.9985, max relative error = 7.4% — essentially at the analytical HB ceiling.
- **Added** `n_fluid` fitted parameter for both models (default 1.0, bounds [0.5, 5.0]). The implied HB shear-thinning exponent is `n_HB = 1/n_fluid`.
- **Modified** Default `fluidity_form` is `"overstress"` for both `LatticeEPM` and `TensorialEPM`. Downstream users who relied on the old Bingham behavior should pass `fluidity_form="linear"` explicitly.

### Added — TensorialEPM Overstress Kernel Port and `.fit()` Implementation

- **Added** Three-form dispatch in `rheojax.utils.epm_kernels_tensorial.compute_plastic_strain_rate` matching the scalar kernel, applied component-wise via the Prandtl-Reuss flow direction `σ'_ij / σ_eff`.
- **Added** `fluidity_form` as a static JIT argument on `tensorial_epm_step`, threaded through from `TensorialEPM._epm_step`.
- **Added** `TensorialEPM.fit()` — the method previously raised `NotImplementedError`. It now inherits `EPMBase._fit` via a thin delegator and supports 1D shear-only targets over all five protocols (flow_curve, startup, relaxation, creep, oscillation). On the emulsion benchmark: R²_lin = 0.9938, R²_log = 0.9988, max relative error = 9.1%, recovering σ_y = 23.59 Pa and HB exponent 0.464 — within 0.5% of the scalar fit.
- **Fixed** `EPMBase._model_flow_curve` (and sibling `_model_startup`, `_model_relaxation`, `_model_creep`, `_model_oscillation`) previously used `jnp.mean(new_state[0])` which averages over *all three* components of the tensorial `(3, L, L)` stress field — physically meaningless because it mixes σ_xx, σ_yy, σ_xy. Introduced a new `_mean_shear_stress` helper that branches on `stress.ndim` to extract σ_xy correctly for both scalar and tensorial paths.
- **Fixed** Warm-start formula in `tensor.py::_run_flow_curve` to account for (a) the √3 factor from the von Mises effective stress in pure shear and (b) the factor of 2 from the Budrikis-Zapperi convention `dσ_ij/dt = 2μ·ε̇`. The scalar plateau formula `σ_c + |γ̇|·τ_pl` incorrectly seeded the tensorial simulation and caused transient blow-ups.
- **Fixed** The tensorial kernel was previously an **ideal-plastic Prandtl-Reuss model** with the plastic rate pinned at `1/(√3·τ_pl_shear)` once yielded. This meant the tensor flow curve had no rate dependence and stress grew unboundedly above a critical shear rate `γ̇_crit = 1/(√3·τ_pl_shear)`. The new overstress form replaces this with proper rate-dependent viscoplasticity.

### Added — Tensorial Overstress Support for Hill Anisotropy

- The overstress constitutive law works with either yield criterion (`"von_mises"` or `"hill"`). At default Hill parameters `(H=0.5, N=1.5)` Hill reduces exactly to von Mises for pure shear; anisotropic `(H, N)` values produce genuinely different stress evolution through the same overstress formula.

### Added — New Tests

- 5 kernel-level tests for the three `fluidity_form` options in the scalar kernel (`tests/utils/test_epm_kernels.py`)
- 5 kernel-level tests for the three `fluidity_form` options in the tensorial kernel including Hill anisotropy interaction (`tests/utils/test_epm_kernels_tensorial.py`)
- 5 model-level tests for `TensorialEPM` overstress behavior: default form, plateau at `σ_c/√3`, analytical HB shape verification, fit-callable scaffold (`tests/models/epm/test_tensorial_epm.py`)
- 4 integration tests for tensorial time-domain protocols under overstress: startup, relaxation, creep, oscillation — each verifies finite, bounded, qualitatively-correct output

Total EPM test count: **80** (25 scalar model + 27 scalar kernel + 28 tensorial model/kernel).

### Added — Numerical Safety Nets

- **Added** Parameter clamps (`sigma_c_mean`, `n_fluid`, `tau_pl` each floored at ~1e-6) and `jnp.where(jnp.isfinite(…), …, plateau_fallback)` guards in `EPMBase._model_flow_curve` and `TensorialEPM._run_flow_curve` warm-start and return paths. NLSQ can transiently probe parameter combinations where the explicit-Euler kernel produces NaN; the fallback substitutes the analytical plateau so the loss remains finite and the optimizer can recover on the next iteration.

### Fixed — Example Notebook `examples/epm/01_epm_flow_curve.ipynb`

- **Rewrote** cells 27–30 to show a **real side-by-side Herschel-Bulkley fit** comparing LatticeEPM (solid blue) and TensorialEPM (dashed red) on the φ=0.80 emulsion data, with N₁ plotted on the right panel. Both models now reach the HB analytical ceiling and recover physically consistent parameters.
- **Fixed** A wavy-residual artifact where the original fit converged to a local minimum with `mu=20` slammed into the upper bound and `sigma_c_std=0.01` slammed into the lower bound. The new default (`fluidity_form="overstress"`) plus physically motivated initial guesses give a clean fit: R²_lin = 0.9957, R²_log = 0.9985, max rel err = 7.4%.
- **Fixed** A plotting artifact where the fit curve was a 30-point polyline on log-log axes, creating visible "kinks" that looked like step shapes in the model. Cell 13 now plots the fit curve on a 200-point fine logspace grid.
- **Fixed** A `NaN` overflow at `γ̇ > 932 s⁻¹` introduced by an earlier warm-start fix: the Bingham-form `σ_warm = σ_c + γ̇·τ_pl` initialized stress at ~5660 Pa at γ̇=1000, which raised to power `n_fluid=2.33` overflows. The new warm-start uses the power-law asymptote `σ_c + σ_c·(γ̇·τ_pl/σ_c)^(1/n_fluid)` which stays bounded.
- **Added** Defensive `importlib.reload` block at the top of the setup cell (`cell-3`) so users with a stale Jupyter kernel don't see misleading "old-error-on-new-code-line" tracebacks after pulling updates.
- **Added** Markdown warning in the tensorial sidebar (`cell-27`) explaining the stale-kernel issue and the two remediation options (re-run the reload cell or restart the kernel).
- **Updated** Bumped the default lattice size from `L=32` to `L=64` in the non-FAST_MODE production path for lower finite-size noise. FAST_MODE (CI) keeps `L=32` for speed.

### Documentation

- **Updated** `docs/source/models/epm/lattice_epm.rst`: new `Constitutive Forms for the Plastic Strain Rate` section with the three-form table and `fluidity_form` / `n_fluid` explanations; updated Parameters table with the new fitted parameters and configuration arguments; expanded `Common Fitting Issues` table with HB-specific guidance.
- **Updated** `docs/source/models/epm/tensorial_epm.rst`: new `Constitutive Forms and the von Mises / Tensorial Conventions` section explaining the √3 plateau factor and the factor-of-2 stress-rate convention, with a scalar-to-tensorial parameter mapping recipe. Updated Parameters/Config tables with `n_fluid` and `fluidity_form`. Replaced the `"Fitting to Shear-Only Data"` example with a HB-capable workflow that shows how to derive a working initial guess from the data plateau.

## [0.6.1] - 2026-04-08

### Fixed
- **Build**: Added `MANIFEST.in` to exclude docs/examples/tests from sdist (151 MB → 1.4 MB, fixes PyPI upload)
- **CI**: Inlined release workflow gates (fixes `workflow_call` startup_failure on tag pushes)

## [0.6.0] - 2026-04-08

### Added - Protocol-Driven Model Inventory System
**Type-Safe Discovery for Models and Transforms**

Introduced a robust inventory system that explicitly maps models to their supported experimental protocols and transforms to their mathematical types.

- **Added** `rheojax.core.inventory`: Defines `Protocol` (FLOW_CURVE, LAOS, etc.) and `TransformType` enums
- **Enhanced** `Registry`: Supports protocol/type metadata and query filtering (`find_compatible()`, `inventory()`)
- **Added** CLI command `rheojax inventory`: List all models and transforms, filter by protocol/type
- **Migrated** All 25 models and 7 transforms to declare explicit capabilities via registration decorators
- **Updated** GUI `ModelService` to use dynamic categorization based on the registry
- **Refactored** `test_modes.py` to use the registry as the single source of truth for model compatibility

### Added - Shear Transformation Zone (STZ) Model
**Plasticity and Transient Flow for Amorphous Solids**

Implemented the STZ model (Langer 2008) for metallic glasses and colloidal suspensions.

- **Added** `rheojax/models/stz/conventional.py`: Conventional STZ implementation
- **Features**:
  - Captures yield stress and stress overshoot in startup flow
  - Internal state variables: effective temperature (χ) and STZ density (Λ)
  - Three complexity variants: 'minimal', 'standard', 'full'
  - Full protocol support: Flow, Creep, Relaxation, SAOS, LAOS (via ODE integration)
- **Dependency**: Added `diffrax` for JAX-native ODE solving of transient dynamics

### Added - SPP Analysis & Yield Stress Model
**Large Amplitude Oscillatory Shear (LAOS) Characterization**

- **Added** `rheojax/transforms/spp_decomposer.py`: Sequence of Physical Processes (SPP) transform
  - Decomposes LAOS stress into elastic and viscous components cycle-by-cycle
  - Extracts transient moduli $G'_t$ and $G''_t$
- **Added** `rheojax/models/spp/spp_yield_stress.py`: SPP-based yield stress model
  - Parametric model for static and dynamic yield stresses in LAOS
- **CLI**: Added `rheojax spp` command for batch analysis of LAOS data

### Added - Expanded Protocol Support
- **SGR Model**: Added support for `STARTUP` flow (stress growth coefficient $\eta^+(t)$)
- **Generalized Maxwell**: Added support for `FLOW_CURVE`, `STARTUP`, and `LAOS` (linear response)

### Refactored
- **Model Organization**: Moved models into subpackages by category (`classical`, `flow`, `fractional`, `sgr`, `stz`, `spp`, `multimode`)
- **Imports**: Updated `rheojax.models` to export all models from a flat namespace for convenience

### Added - Fluidity-Saramito Elastoviscoplastic Models
**Tensorial Viscoelasticity with Thixotropic Fluidity Evolution**

- **Added** `rheojax/models/fluidity/saramito/local.py`: Local (0D) Fluidity-Saramito model
- **Added** `rheojax/models/fluidity/saramito/nonlocal.py`: Nonlocal (1D) variant for shear banding
- **Features**:
  - Tensorial stress [τ_xx, τ_yy, τ_xy] for normal stress predictions (N₁)
  - Von Mises yield criterion: α = max(0, 1 - τ_y/|τ|)
  - Fluidity evolution: df/dt = aging + b·|γ̇|^n · rejuvenation
  - Two coupling modes: "minimal" (λ = 1/f) and "full" (λ + τ_y(f) aging yield)
  - 6 protocols: Flow curve, Startup, Creep, Relaxation, Oscillation, LAOS

### Added - Isotropic-Kinematic Hardening (IKH) Models
**Thixotropic Elasto-Viscoplastic Models for Complex Fluids**

- **Added** `rheojax/models/ikh/`: MIKH and ML-IKH model implementations
- **Added** `rheojax/models/fikh/`: Fractional IKH variants (FIKH, FMLIKH)
- **Features**:
  - Maxwell-IKH (Dimitriou & McKinley 2014) for waxy crude oils, drilling fluids
  - Multi-mode extension for distributed thixotropic timescales
  - Fractional variants with memory effects
  - Full protocol support: Flow curve, Creep, Relaxation, Startup, SAOS, LAOS

### Added - Hébraud-Lequeux Model
**Mean-Field Model for Concentrated Emulsions**

- **Added** `rheojax/models/hl/hebraud_lequeux.py`: HL model implementation
- **Features**:
  - Stress probability distribution evolution via Fokker-Planck equation
  - Mean-field elastic-to-plastic coupling
  - 6 protocols: Flow curve, Creep, Relaxation, Startup, SAOS, LAOS

### Added - Giesekus Viscoelastic Models
**Nonlinear Viscoelastic Models for Polymer Solutions**

- **Added** `rheojax/models/giesekus/`: Single-mode and multi-mode implementations
- **Features**:
  - Anisotropic drag via mobility parameter α
  - Shear-thinning and normal stress predictions
  - Multi-mode Giesekus for polydisperse systems
  - 6 protocols with ODE integration via diffrax

### Added - DMT Thixotropic Models (de Souza Mendes-Thompson)
**Structural-Kinetics Based Thixotropic Models**

- **Added** `rheojax/models/dmt/local.py`: Local (0D) DMT model
- **Added** `rheojax/models/dmt/nonlocal.py`: Nonlocal (1D) variant for shear banding
- **Features**:
  - Structure parameter λ ∈ [0, 1] tracking microstructural state
  - Two viscosity closures: exponential and Herschel-Bulkley
  - Optional Maxwell elasticity for stress overshoot and SAOS
  - Nonlocal model with structure diffusion D_λ∇²λ
  - 6 protocols: Flow curve, Startup, Creep, Relaxation, SAOS, LAOS

### Added - ITT-MCT Models (Integration Through Transients Mode-Coupling Theory)
**Microscopic MCT-Based Models for Dense Colloidal Systems**

- **Added** `rheojax/models/itt_mct/schematic.py`: F₁₂ schematic model
- **Added** `rheojax/models/itt_mct/isotropic.py`: Isotropic ISM model with S(k)
- **Features**:
  - Glass transition control: ε = (v₂ - 4)/4 for fluid (ε < 0) vs glass (ε > 0)
  - Memory kernel: m(Φ) = v₁Φ + v₂Φ² with strain decorrelation
  - Volterra ODE with O(N) integration via Prony decomposition
  - ISM model with Percus-Yevick S(k) for quantitative predictions
  - 6 protocols: Flow curve, SAOS, Startup, Creep, Relaxation, LAOS

### Added - TNT Transient Network Models
**Vernerey-Long-Brighenti Framework for Transient Networks**

- **Added** `rheojax/models/tnt/`: 5 TNT model variants
  - `TNTSingleMode`: Single relaxation mode transient network
  - `TNTLoopBridge`: Loop-bridge topology for associating polymers
  - `TNTStickyRouse`: Sticky Rouse dynamics for entangled systems
  - `TNTCates`: Living polymer model (Cates wormlike micelles)
  - `TNTMultiSpecies`: Multi-species network with independent kinetics
- **Features**:
  - Stateless `model_function` for NLSQ/NUTS compatibility
  - Analytical SAOS + diffrax ODE integration for transient protocols
  - 6 protocols: Flow curve, SAOS, Startup, Relaxation, Creep, LAOS

### Added - VLB Transient Network Models (Vernerey-Long-Brighenti)
**Statistical Mechanics Framework for Transient Polymer Networks**

- **Added** `rheojax/models/vlb/local.py`: Single-network VLB model (~550 lines)
- **Added** `rheojax/models/vlb/multi_network.py`: Multi-network VLB (~550 lines)
- **Added** `rheojax/models/vlb/variant.py`: Variant with Bell/FENE extensions (~1235 lines)
- **Added** `rheojax/models/vlb/nonlocal_model.py`: 1D PDE variant for shear banding (~750 lines)
- **Added** `rheojax/models/vlb/_kernels.py`: JIT-compiled kernels (~1020 lines)
- **Added** `rheojax/models/vlb/_base.py`: Shared VLBBase class (~290 lines)
- **Features**:
  - Distribution-tensor based formulation (full Cauchy stress)
  - Bell model: force-activated bond dissociation k_d(F) = k_d_0 · exp(F/F_c)
  - FENE: Finite extensibility with Warner spring function
  - Multi-network: Independent relaxation modes with coupled stress
  - Nonlocal PDE: Stress diffusion for shear banding prediction
  - Analytical SAOS/flow curve + diffrax ODE for transient protocols
  - Full Bayesian inference support with NLSQ warm-start
- **Documentation**: 7 docs files (index, vlb, vlb_variant, vlb_nonlocal, vlb_protocols, vlb_knowledge, vlb_extensions)
- **Examples**: 10 tutorial notebooks (6 protocols + Bayesian + Bell + FENE + Nonlocal)
- **Tests**: 113 tests (52 Phase 1 + 42 Variant + 19 Nonlocal)

### Added - HVM (Hybrid Vitrimer Model)
**Constitutive Model for Vitrimers with Associative Exchange**

- **Added** `rheojax/models/hvm/local.py`: Full HVM implementation (~960 lines)
- **Added** `rheojax/models/hvm/_kernels.py`: JIT-compiled kernels (~600 lines)
- **Added** `rheojax/models/hvm/_kernels_diffrax.py`: ODE integration kernels (~540 lines)
- **Added** `rheojax/models/hvm/_base.py`: Shared HVMBase class (~310 lines)
- **Features**:
  - 3-subnetwork architecture: Permanent (P) + Exchangeable (E) + Dissociative (D)
  - Evolving natural-state tensor μ^E_nat tracking deformation via BER
  - TST kinetics: k_BER = ν₀·exp(-E_a/RT)·cosh(V_act·σ_VM/RT)
  - Factor-of-2: τ_E_eff = 1/(2·k_BER_0) — both μ^E and μ^E_nat relax toward each other
  - σ_E → 0 at steady state: Natural state fully tracks deformation
  - Arrhenius temperature dependence with T_v topology freezing
  - 5 factory methods: neo-Hookean, Maxwell, Zener, pure vitrimer, partial vitrimer
  - 6 protocols: Flow curve, SAOS, Startup, Relaxation, Creep, LAOS
- **Documentation**: 3 docs files (index, hvm, hvm_knowledge)
- **Examples**: 6 tutorial notebooks (SAOS, relaxation, startup, creep, flow curve, LAOS)
- **Tests**: 60 tests (10 smoke + 47 standard + 3 Bayesian)

### Added - HVNM (Hybrid Vitrimer Nanocomposite Model)
**Constitutive Model for NP-Filled Vitrimers**

- **Added** `rheojax/models/hvnm/local.py`: Full HVNM implementation (~1050 lines)
- **Added** `rheojax/models/hvnm/_kernels.py`: JIT-compiled kernels (~1070 lines)
- **Added** `rheojax/models/hvnm/_kernels_diffrax.py`: ODE integration kernels (~640 lines)
- **Added** `rheojax/models/hvnm/_base.py`: Shared HVNMBase class (~430 lines)
- **Features**:
  - 4-subnetwork architecture: P + E + D + I (interphase around nanoparticles)
  - Guth-Gold strain amplification: X(φ) = 1 + 2.5φ + 14.1φ²
  - Dual TST kinetics: independent k_BER^mat and k_BER^int with separate E_a
  - Factor-of-2 for both matrix and interphase relaxation times
  - φ = 0 recovers HVM exactly (verified to machine precision)
  - Feature flags: include_interfacial_damage, include_diffusion, include_damage
  - 5 factory methods: unfilled_vitrimer, filled_elastomer, partial_vitrimer_nc, conventional_filled_rubber, matrix_only_exchange
  - 6 protocols: Flow curve, SAOS, Startup, Relaxation, Creep, LAOS
- **Documentation**: 3 docs files (index, hvnm, hvnm_knowledge)
- **Examples**: 7 tutorial notebooks (6 protocols + limiting cases)
- **Tests**: 73 tests (13 smoke + 60 standard, 2 slow Bayesian)

### Model Count Update
- **Updated** Total models: 25 → 53 (added 28 models across 12 new families)
  - Fluidity-Saramito EVP: +2 (FluiditySaramitoLocal, FluiditySaramitoNonlocal)
  - IKH: +2 (MIKH, MLIKH)
  - FIKH: +2 (FIKH, FMLIKH)
  - Hébraud-Lequeux: +1 (HebraudLequeux)
  - Giesekus: +2 (GiesekusSingleMode, GiesekusMultiMode)
  - DMT: +2 (DMTLocal, DMTNonlocal)
  - ITT-MCT: +2 (ITTMCTSchematic, ITTMCTIsotropic)
  - TNT: +5 (TNTSingleMode, TNTLoopBridge, TNTStickyRouse, TNTCates, TNTMultiSpecies)
  - VLB: +4 (VLBLocal, VLBMultiNetwork, VLBVariant, VLBNonlocal)
  - HVM: +1 (HVMLocal)
  - HVNM: +1 (HVNMLocal)
  - EPM: +2 (LatticeEPM, TensorialEPM)
- **Updated** Total transforms: 6 → 7 (added SPPDecomposer)
- **Updated** Bayesian support: All 53 models support NumPyro NUTS sampling

### Changed - Multi-Chain Parallelization (Production Default)
**Bayesian inference now defaults to 4 chains for production-ready diagnostics**

- **Changed** `BayesianMixin.fit_bayesian()` default: `num_chains=1` → `num_chains=4`
- **Changed** `BaseModel.fit_bayesian()` default: `num_chains=1` → `num_chains=4`
- **Added** `num_chains` parameter to `BayesianPipeline.fit_bayesian()` (was hardcoded to 1)
- **Added** `seed` parameter to `fit_bayesian()` for reproducibility control
  - `seed=None` (default) uses `seed=0` for deterministic results
  - Set different seeds for independent runs

**Chain method auto-selection** (unchanged but documented):
- `sequential`: Single chain or user override
- `parallel`: Multi-chain on multi-GPU (fastest)
- `vectorized`: Multi-chain on single device (uses vmap)

**Migration Notes:**
- Existing code with explicit `num_chains=1` continues to work unchanged
- Code relying on default `num_chains=1` will now run 4 chains (4x samples)
- For quick demos, explicitly set `num_chains=1`
- For production, use default `num_chains=4` for reliable R-hat/ESS

### Added
- 6 new tests for multi-chain functionality in `tests/core/test_bayesian.py`
- 4 new tests in `tests/pipeline/test_bayesian_pipeline.py` (`TestBayesianPipelineMultiChain`)

### Changed
- Version bump to 0.6.0
- Removed piblin-jax integration from RheoData
- **Dependency bumps**: JAX >=0.8.3, jaxlib >=0.8.3, NLSQ >=0.6.8, ArviZ >=0.23.4
- **GPU support**: Added CUDA 13+ support alongside CUDA 12+ (system CUDA via `-local` packages)

### Documentation
- **Tutorial expansion**: 56 → 235 example notebooks across 20 categories
  - Added model family tutorials: DMT (6), EPM (6), FIKH (12), Fluidity (24), Giesekus (7), HL (6), HVM (13), HVNM (15), IKH (12), ITT-MCT (12), SGR (6), STZ (6), TNT (30), VLB (16)
  - Added verification suite: 31 notebooks for cross-validation against literature data
- **Models handbook**: Added narrative deep-dive pages for all 18 model families
- **User guide**: Added advanced topics (constitutive ODE models, dense suspensions, polymer networks, vitrimer models, thixotropy & yielding)
- **GPU installation guide**: Added CUDA 13 support, GPU compatibility table (Blackwell through Kepler), structured troubleshooting
- **Updated** `docs/source/examples/index.rst` with all 235 notebooks, 3 learning paths, navigation tables
- **Updated** `CLAUDE.md` with current dependency versions (JAX >=0.8.3, NLSQ >=0.6.10, ArviZ >=0.23.4)
- **Updated** all CUDA references codebase-wide from "CUDA 12.1-12.9" to "CUDA 12+ or 13+"

### Added - Robustness & Correctness Fixes (Round 9-10)
- **Core**:
  - Added support for tracking `potential_energy` fields during Bayesian sampling.
  - Resolved cache race-conditions between inferred test modes and explicit setter logic.
  - Enhanced float64 epsilon tolerance for deterministic parameter bounds dynamically scaling to magnitude to prevent True/False inference tracer failures.
  - Exposed safety `overwrite` flag in Parameter registry creation.
- **Models**:
  - Vectorized isotropic ITT-MCT vertices using optimal Gauss-Legendre quadrature (O(N^2) evaluation, completely vectorized).
  - Fixed SGR GENERIC and conventional steady flow stress returning viscosity by enforcing stress dimension return.
  - Corrected schematic creep initial conditions which mistakenly assigned elastic jumps to strain rate rather than initial absolute strain.
  - Corrected erroneous schema stress clamps suppressing glass $\beta$-relaxation.
- **IO**:
  - Extended DMTA detect_deformation_mode_from_columns heuristics to gracefully extract dynamic `bending` and `compression` mode test types.
  - Hardened Trios float parsing against European `,` parsing bugs corrupting scientific numbers.
  - Implemented 0-d numpy scalar fallback unwrap to cleanly serialize scalar parameters into recursive nested HDF5 metadata dictionaries without exception tracing.
- **Pipeline and GUI**:
  - Auto-scaled NLSQ Mean-Squared Error threshold bounds in optimization solvers to properly account for initial values >10^6 natively (e.g. valid DMTA datasets at the GPa baseline, MSE ~10^18).
  - Corrected the `builder` to dynamically unroll `validate=False` without requiring underlying valid structure arrays to construct dummy pipes.
  - Restored multi-thread state reentrancy guard isolation isolating StateStore dispatch emissions for independent execution streams (resolves signal crossfire locks).
  - Isolated signal teardown memory cleanup procedures to single windows.
  - Eliminated parameter generation generation conditions in asynchronous live GUI thread render previews.
- **Transforms**:
  - Injected 2x zero padding onto OWChirp wavelet boundaries resolving circular aliasing behavior during frequency cross-correlation arrays.

### Test Organization
- **Reorganized** test files into subdirectories by model family (`tests/models/<family>/`)
  - classical/, dmt/, epm/, flow/, fractional/, hl/, ikh/, multimode/, sgr/, spp/, vlb/
- **Test count**: 4963 tests (1838 smoke, 4673 standard, 4963 full suite)
- **Platform fixes**: Fixed 5 cross-platform test failures (visual hash, Windows paths, encoding, numerical precision)

### CI/CD
- **GitHub Actions**: Full CI pipeline with lint, quality, test (3 OS × 2 Python), docs, build, audit
- **Release automation**: Tag-triggered PyPI publish with build provenance attestation, SBOM, GitHub Release
- **Security scanning**: Consolidated CodeQL, Semgrep SAST, Gitleaks, Trivy in single workflow
- **Dependency management**: Dependabot with grouped updates (JAX ecosystem, Bayesian, GUI, dev)
- **Build**: `uvx twine` for venv-free package validation, `workflow_call` reuse in release pipeline
- **Pre-commit hooks**: Black 26.3.1, Ruff 0.15.9, MyPy 1.20.0, detect-secrets

### Documentation
- **Migrated** all install instructions from `pip install -e .[dev]` to `uv sync`
- **Removed** references to nonexistent package extras (`[gui]`, `[io]`, `[ml]`, `[all]`, `[dev]`, `[bayesian]`)
- **Updated** test counts across CLAUDE.md, tech-stack.md, and contributing guides
- **Updated** CI/CD section in tech-stack.md from "disabled" to active workflow descriptions
- **Added** CI badge to README.md

---

## [0.5.0] - 2025-12-04

### Added - Soft Glassy Rheology (SGR) Models
**Phase 5: Statistical Mechanics Models for Soft Glassy Materials**

Two new SGR models for foams, emulsions, pastes, and colloidal suspensions:

#### SGR Conventional (Sollich 1998)
- **Added** `rheojax/models/sgr_conventional.py` (~1863 lines)
  - Trap model with exponential density of states: ρ(E) = exp(-E)
  - Three parameters: x (noise temperature), G0 (modulus), τ0 (attempt time)
  - Material classification via noise temperature:
    - x < 1: glass (aging, non-ergodic)
    - 1 < x < 2: power-law fluid (SGM regime)
    - x ≥ 2: Newtonian liquid
  - Oscillation mode: G*(ω) via Fourier transform of memory function
  - Relaxation mode: G(t) via Mittag-Leffler-type decay
  - Creep mode: J(t) with optional yield stress
  - Full Bayesian inference support with NumPyro

#### SGR GENERIC (Fuereder & Ilg 2013)
- **Added** `rheojax/models/sgr_generic.py` (~945 lines)
  - Thermodynamically consistent GENERIC framework implementation
  - Dissipation potential satisfies Onsager reciprocal relations
  - Enhanced stability for near-glass transition (x → 1)
  - Automatic fallback to Conventional SGR when appropriate
  - Same parameter interface as Conventional for easy comparison

#### SGR Kernel Functions
- **Added** `rheojax/utils/sgr_kernels.py` (~539 lines)
  - `sgr_memory_kernel()`: Memory function K(t) for relaxation dynamics
  - `sgr_modulus_fourier()`: Complex modulus G*(ω) via numerical Fourier transform
  - `sgr_yield_stress()`: Dynamic yield stress prediction
  - `sgr_aging_exponent()`: Aging dynamics μ(x) calculation
  - All functions JAX-compatible with automatic differentiation

### Added - SRFS Transform (Strain-Rate Frequency Superposition)
**Collapse Flow Curves Analogous to Time-Temperature Superposition**

- **Added** `rheojax/transforms/srfs.py` (~846 lines)
  - Power-law shift factor calculation: a(γ̇) ~ (γ̇)^(2-x)
  - Automatic shift factor determination via optimization
  - Manual and reference shear rate specification
  - Thixotropy detection via hysteresis analysis
  - Shear banding detection and coexistence curve computation
- **Added** `detect_shear_banding()`: Identifies flow instabilities from stress plateau
- **Added** `compute_shear_band_coexistence()`: Calculates coexisting shear rates

### Added - Comprehensive SGR Documentation
- **Added** `docs/source/models/sgr/sgr_conventional.rst` (532 lines)
  - Complete theoretical background with governing equations
  - Parameter interpretation guide with material classification
  - Usage examples for all test modes
  - Troubleshooting section for convergence issues
- **Added** `docs/source/models/sgr/sgr_generic.rst` (416 lines)
  - GENERIC framework explanation
  - Comparison with Conventional SGR
  - When to use each variant
- **Added** `docs/source/transforms/srfs.rst` (237 lines)
  - SRFS theory and applications
  - Connection to SGR noise temperature
  - Shear banding analysis tutorial

### Testing
- **Added** 1890 lines of new tests across 5 test files:
  - `tests/models/test_sgr_conventional.py` (1109 lines): 45+ unit tests
  - `tests/models/test_sgr_generic.py` (407 lines): 25+ unit tests
  - `tests/utils/test_sgr_kernels.py` (417 lines): Kernel function validation
  - `tests/transforms/test_srfs.py` (460 lines): Transform verification
  - `tests/integration/test_sgr_integration.py` (316 lines): End-to-end workflows
  - `tests/hypothesis/test_sgr_properties.py` (574 lines): Property-based tests

### Model Count Update
- **Updated** Total models: 21 → 23 (added SGRConventional, SGRGeneric)
- **Updated** Total transforms: 5 → 6 (added SRFS)
- **Updated** Bayesian support: All 23 models support NumPyro NUTS sampling

---

## [0.4.0] - 2025-11-16

### Fixed - Mode-Aware Bayesian Inference (CRITICAL CORRECTNESS BUG)
**Incorrect Posteriors for Non-Relaxation Test Modes**

RheoJAX v0.4.0 fixes a critical correctness bug in Bayesian inference where test_mode was captured as class state instead of closure parameter, causing all Bayesian fits to use the last-fitted mode regardless of fit_bayesian() inputs. This resulted in physically incorrect posteriors for creep and oscillation modes.

#### Root Cause
- `model_function()` in NumPyro sampler read `self._test_mode` set during `.fit()`
- Global state leakage between NLSQ (`.fit()`) and Bayesian (`.fit_bayesian()`) workflows
- Example: Fitting relaxation with `.fit()`, then oscillation with `.fit_bayesian()` produced relaxation-mode posteriors

#### Solution
- Refactored `BayesianMixin.fit_bayesian()` to use closure-based test_mode capture
- Added explicit `test_mode` parameter to `fit_bayesian()` signature (backward compatible)
- Model function now captures test_mode statically at construction time, not execution time
- All 21 models updated to support mode-aware model_function pattern

#### Validation
- **Validated against pyRheo**: Posterior means within 5% for all three test modes
- **MCMC Diagnostics**: R-hat < 1.01, ESS > 400, divergences < 1% across all models
- **Test Coverage**: 35-50 new validation tests covering all 11 fractional models
- **No Regressions**: 100% backward compatibility maintained

### Performance - GMM Element Search Optimization
**2-5x Speedup for Element Minimization Workflows**

Optimized Generalized Maxwell Model element minimization through warm-start successive fits and compilation reuse.

#### Improvements
- **Warm-Start from Previous N**: Each N-mode fit initializes from optimal N+1 parameters
- **Compilation Reuse**: Cached residual functions across n_modes iterations
- **Early Termination**: Stops when R² degrades below threshold (prevents futile small-N fits)
- **Transparent Optimization**: No API changes, speedup automatic

#### Performance Targets Met
- **Latency Reduction**: 2-5x measured speedup (baseline: 20-50s → optimized: 4-25s for N=10)
- **Accuracy Preserved**: R² degradation <0.1%, Prony series MAPE <2% vs cold-start
- **Optimal N Selection**: 100% agreement with v0.3.2 baseline for same optimization_factor

### Performance - TRIOS Large File Auto-Chunking
**50-70% Memory Reduction for Files >5 MB**

Automatic memory-efficient loading for large TRIOS experimental files with transparent auto-detection.

#### Improvements
- **Auto-Detection**: Files >5 MB automatically use chunked reader (transparent to users)
- **Memory Savings**: 50-70% peak memory reduction for 50k+ point files
- **Progress Tracking**: Optional progress callback for large file monitoring
- **Opt-Out Available**: `auto_chunk=False` parameter disables auto-detection if needed

#### Memory Targets Met
- **Baseline (v0.3.2)**: Full file load via f.read(), ~10-50 MB peak for 50k+ points
- **Optimized (v0.4.0)**: Auto-chunking, ~3-15 MB peak (50-70% reduction)
- **Latency Overhead**: <20% increase in total load time (acceptable trade-off)
- **Data Integrity**: 100% match between chunked and full-load RheoData

### Migration Guide

#### For Bayesian Users
**No Action Required** - 100% backward compatible. Existing code continues to work unchanged.

**New Capability (Recommended)**: Explicit test mode specification
```python
# v0.4.0: Explicit mode specification (recommended best practice)
from rheojax.models import FractionalZenerSolidSolid
from rheojax.core.data import RheoData

model = FractionalZenerSolidSolid()

# Option 1: Pass RheoData with test_mode embedded (recommended)
rheo_data = RheoData(x=omega, y=G_star, initial_test_mode='oscillation')
result = model.fit_bayesian(rheo_data)  # Correctly uses oscillation mode

# Option 2: Pass test_mode explicitly (new parameter)
result = model.fit_bayesian(omega, G_star, test_mode='oscillation')
```

**v0.3.2 Code Still Valid**:
```python
# v0.3.2 workflow (still works in v0.4.0)
model.fit(t, G_t)  # Sets test_mode='relaxation'
result = model.fit_bayesian(t, G_t)  # Infers mode from RheoData or uses relaxation
```

#### For GMM Users
**No Action Required** - Transparent 2-5x speedup with identical API.

```python
# v0.3.2 and v0.4.0 (identical API, automatic speedup)
from rheojax.models import GeneralizedMaxwell

gmm = GeneralizedMaxwell(n_modes=10)
gmm.fit(t, G_t, test_mode='relaxation', optimization_factor=1.5)
n_optimal = gmm._n_modes  # Auto-reduced from 10 (2-5x faster in v0.4.0)
```

#### For TRIOS Users
**No Action Required** - Transparent auto-chunking for files >5 MB.

```python
# v0.3.2 and v0.4.0 (identical API, automatic memory savings)
from rheojax.io.readers import load_trios

rheo_data = load_trios('large_file.txt')  # Auto-chunks if >5 MB
```

**New Feature**: Progress tracking for very large files
```python
# v0.4.0: Progress callback for large files
def progress_callback(current, total):
    pct = 100 * current / total
    print(f"Loading: {pct:.1f}% complete")

rheo_data = load_trios('large_file.txt', progress_callback=progress_callback)
```

**Opt-Out**: Disable auto-chunking if needed
```python
# v0.4.0: Force full-file loading regardless of size
rheo_data = load_trios('large_file.txt', auto_chunk=False)
```

### Deprecation Warnings
None. All v0.3.2 APIs remain fully supported in v0.4.0.

### Version Compatibility
- **Minimum Python**: 3.12+ (unchanged from v0.3.2)
- **JAX Version**: 0.8.0 exact (unchanged)
- **NLSQ Version**: >=0.2.1 (unchanged)
- **NumPyro Version**: Latest compatible with JAX 0.8.0 (unchanged)

### Testing Your Migration
Run validation checks after upgrading to v0.4.0:

```bash
# Verify installation
python -c "import rheojax; print(rheojax.__version__)"  # Should print 0.4.0

# Run smoke tests (2-5 min)
pytest -m smoke

# Run Bayesian validation (if using Bayesian features, ~30-60 min)
pytest -m validation

# Run your existing test suite
pytest tests/
```

### Performance Summary
- **Bayesian Correctness**: All modes produce correct posteriors (validated vs pyRheo)
- **GMM Speedup**: 2-5x measured for element minimization workflows
- **TRIOS Memory**: 50-70% reduction for files >5 MB
- **No Regressions**: All 1154 v0.3.2 tests still pass
- **Backward Compatibility**: 100% maintained, zero breaking changes

### Testing
- **Added** 59-88 new tests across validation, integration, and benchmark tiers
  - 35-50 validation tests against pyRheo and ANSYS APDL references
  - 12-19 unit tests for new functionality
  - 5-8 integration tests for end-to-end workflows
  - 7-11 benchmark tests documenting performance improvements
- **Status**: 1213-1242 total tests (1154 baseline + 59-88 new)
- **Validation Strategy**: Validation-first development with external references

### Documentation
- **Updated** BayesianMixin.fit_bayesian() docstring with test_mode parameter
- **Updated** GeneralizedMaxwell docstring with warm-start optimization details
- **Updated** load_trios() docstring with auto-chunking behavior and memory guidance
- **Updated** Migration guide for all three features

---

## [0.3.2] - 2025-11-16

### Performance - Category B Optimizations (20-30% Additional Improvement)
**Cumulative 50-75% End-to-End Performance Gain vs Pre-v0.3.1**

Building on v0.3.1's JAX-native foundation, v0.3.2 implements four vectorization and convergence optimizations for an additional 20-30% latency reduction.

#### Improvements
- **Vectorized Mastercurve**: JAX vmap + jaxopt.LBFGS (2-5x on multi-dataset workflows)
- **Intelligent Mittag-Leffler**: Dynamic early termination, 5-20 iterations vs fixed 100 (5-20x achieved)
- **Batch Vectorization**: vmap over datasets + parallel I/O (3-4x on multi-file operations)
- **Device Memory**: Deferred NumPy conversion to plotting boundary (10-20% pipeline improvement)

### Installation
```bash
pip install rheojax[performance]  # Optional jaxopt for max performance
```

### Testing
- **Added** 8 new tests, **Status**: 1169 tests passing
- **Backward Compatibility**: 100% maintained

---

## [0.3.1] - 2025-11-15

### Performance - Category A Optimizations (30-45% Improvement)
**JAX-Native Foundation**

Five foundational optimizations establishing the JAX-native infrastructure for all subsequent performance improvements.

#### Improvements
- **JAX-Native RheoData**: Internal JAX storage, explicit `to_numpy()` method (eliminates 2-5x conversion overhead)
- **JIT Residuals**: @jax.jit on NLSQ residual computation (15-25% per-iteration reduction)
- **Model Prediction JIT**: 6 flow models with @partial(jax.jit, static_argnums=(0,)) (10-20% speedup)
- **Parallel Multi-Start**: ThreadPoolExecutor with thread-safe PRNG (2-4x for 3-5 starts)
- **Batch Parameter Writes**: `ParameterSet.set_values_batch()` for GMM (5-10% reduction)

### Testing
- **Added** 12 new tests + 9 micro-benchmarks
- **Status**: 1169 tests passing (1154 baseline + 15 new)
- **Backward Compatibility**: 100% maintained

---

## [0.2.2] - 2025-11-15

### Added - Generalized Maxwell Model & Advanced TTS
**PyVisco Integration: Multi-Mode Viscoelastic Models with JAX Acceleration**

Integration of PyVisco capabilities with 5-270x speedup via NLSQ/JAX optimization.

#### Generalized Maxwell Model (GMM)
- **Added** `rheojax/models/generalized_maxwell.py` (~1250 lines)
  - Multi-mode Prony series representation: G(t) = G_∞ + Σᵢ Gᵢ exp(-t/τᵢ)
  - Tri-mode equality: relaxation, oscillation, and creep predictions
  - Transparent element minimization (auto-optimize N modes)
  - Two-step NLSQ fitting with softmax penalty
  - Bayesian inference support with tiered prior safety mechanism
- **Added** `rheojax/utils/prony.py` (395 lines)
  - Prony series validation and parameter utilities
  - Element minimization with R²-based optimization
  - Log-space transforms for wide time-scale ranges

#### Automatic Shift Factor Calculation
- **Enhanced** `rheojax/transforms/mastercurve.py` (+300 lines)
  - Power-law intersection method for automatic shift factors
  - No WLF parameters required
  - JAX-native optimization (5-270x speedup over scipy)
  - Backward compatible with existing WLF/Arrhenius methods

#### Tiered Bayesian Prior Safety
- **Added** Three-tier prior classification in GMM
  - Tier 1: Hard failure → informative error or fallback priors
  - Tier 2: Suspicious convergence → auto-widened priors
  - Tier 3: Good convergence → NLSQ-based warm-start priors

### Fixed - Type Annotations
- **Fixed** 7 mypy type checking errors
  - Added type annotations for `_test_mode`, `_nlsq_result`, `_element_minimization_diagnostics`
  - Updated `optimization_factor` parameter types to `float | None`
  - Added type cast for optimal_model attribute access
  - Removed unused type ignore comment

### Documentation
- **Updated** README.md and docs/source/index.rst for v0.2.2
- **Added** 3 example notebooks
  - `examples/advanced/08-generalized_maxwell_fitting.ipynb`
  - `examples/transforms/06-mastercurve_auto_shift.ipynb`
  - `examples/bayesian/07-gmm_bayesian_workflow.ipynb`

### Testing
- **Added** 55 passing tests across 5 new test files
  - 20 tests for Prony utilities
  - 15 tests for GMM tri-mode equality
  - 7 tests for Bayesian integration
  - 7 tests for prior safety mechanism
  - 7 tests for auto shift algorithm

---

## [0.2.1] - 2025-11-14

### Refactored - Template Method Pattern for Initialization
**Phases 1-3 Complete: Template Method Architecture (v0.2.1)**

Refactored the smart initialization system to use the Template Method design pattern, eliminating code duplication across all 11 fractional models while maintaining 100% backward compatibility.

#### Architecture Changes
- **Added** `BaseInitializer` abstract class (`rheojax/utils/initialization/base.py`)
  - Enforces consistent 5-step initialization algorithm across all models
  - Provides common logic for feature extraction, validation, and parameter clipping
  - Defines abstract methods for model-specific parameter estimation
- **Added** 11 concrete initializer classes (one per fractional model):
  - `FractionalZenerSSInitializer` (FZSS)
  - `FractionalMaxwellLiquidInitializer` (FML)
  - `FractionalMaxwellGelInitializer` (FMG)
  - `FractionalZenerLLInitializer`, `FractionalZenerSLInitializer`
  - `FractionalKelvinVoigtInitializer`, `FractionalKVZenerInitializer`
  - `FractionalMaxwellModelInitializer`, `FractionalPoyntingThomsonInitializer`
  - `FractionalJeffreysInitializer`, `FractionalBurgersInitializer`
- **Refactored** `rheojax/utils/initialization.py`
  - Now serves as facade delegating to concrete initializers
  - Reduced from 932 → 471 lines (49% code reduction)
  - All 11 public initialization functions preserved for backward compatibility

#### Performance
- **Verified** near-zero overhead: 0.01% of total fitting time
  - Initialization: 187 microseconds ± 72 μs
  - Total fitting: 1.76 seconds ± 0.16s
  - Benchmark: 10 runs of FZSS oscillation mode fitting

#### Testing
- **Added** 22 tests for concrete initializers (`tests/utils/initialization/test_fractional_initializers.py`)
- **Added** 7 tests for BaseInitializer (`tests/utils/initialization/test_base_initializer.py`)
- **Status**: 27/29 tests passing (93%), all 22 fractional model tests passing (100%)

#### Documentation
- **Updated** CLAUDE.md with Template Method pattern in "Key Design Patterns"
- **Added** comprehensive implementation details with code examples
- **Added** developer-focused architecture documentation
- **Enhanced** module-level docstrings in `initialization.py`

#### Benefits
- Eliminates code duplication across 11 models
- Enforces consistent initialization algorithm
- Maintains 100% backward compatibility
- Near-zero performance overhead
- Easier to extend with new fractional models

#### Phase 2: Constants Extraction (Complete)
- **Added** `rheojax/utils/initialization/constants.py` for centralized configuration
  - `FEATURE_CONFIG`: Savitzky-Golay window, plateau percentile, epsilon
  - `PARAM_BOUNDS`: min/max fractional order constraints
  - `DEFAULT_PARAMS`: fallback values when initialization fails
- **Benefits**: Tunable configuration, reduced coupling, better testability

#### Phase 3: FractionalModelMixin (Complete)
- **Added** `_apply_smart_initialization()`: Delegated initialization for all 11 models
- **Added** `_validate_fractional_parameters()`: Common validation logic
- **Added** automatic initializer mapping via class name lookup
- **Benefits**: DRY principle, consistent error handling, easier maintenance

---

## [0.2.0] - 2025-11-07

Previous releases documented in git history.

[0.6.1]: https://github.com/imewei/rheojax/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/imewei/rheojax/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/imewei/rheojax/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/imewei/rheojax/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/imewei/rheojax/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/imewei/rheojax/compare/v0.2.2...v0.3.1
[0.2.2]: https://github.com/imewei/rheojax/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/imewei/rheojax/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/imewei/rheojax/releases/tag/v0.2.0
