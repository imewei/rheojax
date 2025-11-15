# Changelog

All notable changes to RheoJAX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/imewei/rheojax/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/imewei/rheojax/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/imewei/rheojax/releases/tag/v0.2.0
