# Changelog

All notable changes to RheoJAX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Refactored - Template Method Pattern for Initialization
**Phase 1 of 2: Template Method Architecture (v0.2.0)**

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

#### Next Phase
**Phase 2 (Planned):** Extract model-specific constants into dedicated modules to further reduce coupling and improve maintainability.

---

## [0.2.0] - 2025-11-07

Previous releases documented in git history.

[Unreleased]: https://github.com/username/rheojax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/username/rheojax/releases/tag/v0.2.0
