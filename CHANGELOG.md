# Changelog

All notable changes to the Rheo project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation
- **Tutorial Consolidation**: Migrated all tutorial notebooks to unified `examples/` directory
  - 22 notebooks organized into 4 learning paths (basic, transforms, bayesian, advanced)
  - Added `examples/advanced/06-frequentist-model-selection.ipynb` (AIC/BIC model comparison)
  - Added `examples/transforms/02b-mastercurve-wlf-validation.ipynb` (WLF parameter validation with synthetic data)
  - Deprecated `docs/examples/` directory (legacy notebooks removed or superseded)
  - Updated all Sphinx documentation references (`user_guide.rst`, `bayesian_inference.rst`, `examples/index.rst`)
  - Total learning time: 13-16 hours across all tutorials
  - See `examples/README.md` for complete tutorial guide

### Changed
- **Examples directory structure**: Moved from `docs/examples/` to `examples/` with improved organization
- **Data anonymization**: Completed full anonymization of experimental data (serial numbers, operators, dates)
- **Test configuration**: Added `notebook_smoke` pytest marker for notebook validation framework

### Fixed
- Fixed hardcoded absolute path in `tests/validation/test_migrated_notebooks.py`
- Documented Bayesian convergence threshold rationale (R-hat, ESS, divergences)

## [0.1.0] - TBD (Initial Development)

### Added

#### Core Infrastructure
- `BaseModel` abstract class with scikit-learn compatible API
- `BaseTransform` abstract class for data preprocessing
- `RheoData` container with metadata and JAX compatibility
- `ParameterSet` system with bounds and constraints
- `BayesianMixin` class providing Bayesian capabilities to all models

#### Models (20 total)
- 3 classical models: Maxwell, Zener, SpringPot
- 11 fractional models: Fractional Maxwell (4 variants), Fractional Kelvin-Voigt (4 variants), Fractional Zener (3 variants)
- 6 non-Newtonian flow models: Power Law, Carreau, Carreau-Yasuda, Cross, Herschel-Bulkley, Bingham

#### Transforms (5 total)
- FFT Analysis: Time → frequency domain conversion
- Mastercurve: Time-temperature superposition (WLF, Arrhenius)
- Mutation Number: Quantify viscoelastic character
- OWChirp: Optimal waveform analysis for LAOS
- Smooth Derivative: Noise-robust differentiation

#### Bayesian Inference
- Complete NLSQ → NUTS workflow with NumPyro
- `BayesianResult` dataclass storing posterior samples and diagnostics
- `BayesianPipeline` for fluent API Bayesian workflows
- Warm-start support from NLSQ point estimates (2-5x faster convergence)
- Convergence diagnostics: R-hat, ESS, divergences
- Credible interval computation (HDI)

#### ArviZ Integration
- `plot_pair()`: Parameter correlation analysis with divergence highlighting
- `plot_forest()`: Credible interval comparison plots
- `plot_energy()`: NUTS-specific energy diagnostic
- `plot_autocorr()`: Mixing quality assessment
- `plot_rank()`: Convergence diagnostic (rank plots)
- `plot_ess()`: Effective sample size visualization
- `to_inference_data()`: Convert BayesianResult to ArviZ InferenceData format

#### Pipeline API
- High-level fluent interface for workflows
- Method chaining: `.load().transform().fit().plot().save()`
- Pre-configured pipelines: Mastercurve, Model Comparison, Batch, Bayesian
- Programmatic construction via PipelineBuilder

#### I/O System
- TRIOS format reader
- CSV reader with auto-detection
- Excel reader (.xlsx, .xls)
- Anton Paar format support
- HDF5 writer (full fidelity)
- Excel writer (results export)

#### Visualization
- Publication-quality plotting
- Automatic plot type selection based on test mode
- Three built-in styles: default, publication, presentation
- Matplotlib-based with customization support

#### Test Mode Detection
- Automatic experimental technique identification
- Relaxation (stress decay)
- Creep (strain increase)
- Oscillation (frequency-domain)
- Rotation (flow curves)

#### Multi-Technique Fitting
- Shared parameters across experiments
- Combine relaxation + creep with shared modulus
- Global optimization across multiple datasets
- Uncertainty propagation

#### Plugin System
- Registry-based model/transform discovery
- `@ModelRegistry.register()` decorator
- `@TransformRegistry.register()` decorator
- Dynamic instantiation by name

#### JAX Integration
- Float64 precision enforcement
- Safe import mechanism (`safe_import_jax()`)
- Automatic CPU/GPU dispatch
- Gradient-based optimization
- JAX-accelerated model fitting: 10-100x speedup
- GPU support for all 20 models
- JIT compilation for repeated operations

#### Documentation
- 700+ line comprehensive Bayesian inference user guide
- Complete NLSQ → NUTS workflow documentation
- All 6 ArviZ diagnostic methods fully documented
- Best practices and common pitfalls
- Multiple complete examples
- Convergence criteria and troubleshooting
- Modernized Sphinx configuration with optional extensions

### Technical Requirements
- Python 3.12+ requirement
- JAX 0.8.0 pinned version
- Type hints throughout
- Comprehensive test suite

### Dependencies
- Core: `jax==0.8.0`, `jaxlib==0.8.0`
- Optimization: `nlsq>=0.1.6` for GPU-accelerated NLSQ
- Bayesian: `numpyro>=0.13.0,<1.0.0` for MCMC sampling
- Diagnostics: `arviz>=0.15.0,<1.0.0` for Bayesian diagnostics
- Scientific: `numpy`, `scipy`, `pandas`
- I/O: `h5py`, `openpyxl`
- Visualization: `matplotlib>=3.5.0`

## Versioning Strategy

- **Major version** (X.0.0): Breaking API changes, major feature additions
- **Minor version** (0.X.0): New features, backward-compatible
- **Patch version** (0.0.X): Bug fixes, documentation updates

## Links

- **Documentation**: https://rheo.readthedocs.io
- **Repository**: https://github.com/username/rheo
- **Issue Tracker**: https://github.com/username/rheo/issues

## Contributors

- Rheo Development Team
- Community contributors (see GitHub)

## License

MIT License - see LICENSE file for details
