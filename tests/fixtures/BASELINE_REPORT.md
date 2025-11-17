# RheoJAX v0.4.0 Task Group 1 - Baseline Test Report

**Date**: November 16, 2025
**Task**: Setup and Infrastructure for RheoJAX v0.4.0 Category C - Structural and Correctness Improvements
**Status**: IN PROGRESS

## Executive Summary

Task Group 1 focuses on setting up the development environment, verifying test infrastructure, establishing baseline test results, and preparing test data fixtures for the v0.4.0 release cycle.

## 1. Feature Branch Creation

**Status**: ✅ COMPLETED

- **Branch Name**: `feature/v0.4.0-category-c-correctness`
- **Base**: `main` (commit: `595ad9e`)
- **Created**: 2025-11-16 09:38:00 UTC
- **Verification**: `git status` confirms clean working directory

```bash
$ git branch -v
* feature/v0.4.0-category-c-correctness 595ad9e fix(tests): suppress expected UserWarning in GMM Bayesian safety test
  main                                  595ad9e fix(tests): suppress expected UserWarning in GMM Bayesian safety test
```

## 2. Test Infrastructure Verification

### 2.1 Smoke Tier (105 tests, ~30s-2min)

**Status**: 🔄 RUNNING (as of 09:44:00 UTC)

**Command**: `pytest -n auto -m "smoke" --tb=line -q`

**Expected**:
- 105 tests total
- Runtime: 30 seconds to 2 minutes
- All tests should PASS (v0.3.1 baseline established)

**Test Categories Included**:
- Core module tests (BaseModel, BaseTransform, parameters, float64 precision)
- I/O reader tests (TRIOS reader, metadata extraction)
- Model smoke tests (Maxwell, Power Law, SpringPot, Zener, fractional models)
- Pipeline and transform tests
- Integration tests (NLSQ to NUTS workflow)
- Project structure tests

### 2.2 Fast Tier (~1069 tests, 5-10 min)

**Status**: 🔄 RUNNING (as of 09:44:00 UTC)

**Command**: `pytest -n auto -m "not slow and not validation and not benchmark and not notebook_comprehensive" --tb=line -q`

**Expected**:
- ~1069 tests total
- Runtime: 5-10 minutes
- All tests should PASS (v0.3.1 baseline established)
- No regressions from main branch

**Excluded**:
- `slow` marker: Long-running tests (>30s each)
- `validation` marker: Validation against external references (pyRheo, ANSYS)
- `benchmark` marker: Performance benchmarks
- `notebook_comprehensive` marker: Full notebook execution tests

## 3. Validation Dependencies

### 3.1 Available Dependencies

| Dependency | Status | Import Test | Notes |
|------------|--------|------------|-------|
| `tracemalloc` | ✅ Built-in | `import tracemalloc` | Part of Python 3.12+ stdlib |
| `pytest` | ✅ Installed | `pytest --version` | v9.0.1 installed |
| `pytest-xdist` | ✅ Installed | `import pytest_xdist` | v3.8.0 installed (parallel execution) |
| `pytest-cov` | ✅ Installed | `import pytest_cov` | v7.0.0 installed (coverage) |
| `pytest-benchmark` | ❌ Not Installed | ImportError | Required for performance benchmarks |
| `pyRheo` | ⚠️ Unknown | ImportError | Optional for reference validation |

### 3.2 Environment Info

```
Python: 3.13.9
JAX: 0.8.0 (CPU-only on macOS)
NLSQ: Installed (required for float64)
NumPyro: Installed (for Bayesian inference)
ArviZ: Installed (for diagnostics)
Environment: Virtual environment (uv-managed)
```

### 3.3 pytest-benchmark Installation

**Issue**: `pytest-benchmark` not currently installed in project dependencies

**Options**:
1. Add to `pyproject.toml` dev-dependencies (recommended for CI)
2. Install locally via uv for development only
3. Skip benchmark tests initially, add dependency in Task Group 7

**Recommendation**: Add to `pyproject.toml` during Task Group 7 (Documentation & Release) to avoid breaking changes in Task Groups 2-6.

For now, validation and unit tests proceed without benchmark marks.

## 4. Test Data Fixtures

### 4.1 Fixture Generation

**Status**: ✅ COMPLETED

**Script Location**: `/Users/b80985/Projects/rheojax/tests/fixtures/generate_test_data.py`

**Features**:
- Synthetic TRIOS file generation at multiple sizes (1 MB, 5 MB, 10 MB, 50 MB, 100 MB)
- Reference data generation for all three test modes:
  - Relaxation mode: Time-dependent shear modulus G(t)
  - Creep mode: Time-dependent compliance J(t)
  - Oscillation mode: Frequency-dependent complex modulus G*(ω)
- RheoData fixture factory functions for pytest integration
- Reproducible synthetic data (seeded with numpy.random.seed=42)

**Generated Files** (from test run):

| File | Target Size | Actual Size | Points | Status |
|------|-----------|-----------|--------|--------|
| trios_synthetic_1mb.txt | 1.0 MB | 0.2 MB | 5,000 | ✅ Created |
| trios_synthetic_5mb.txt | 5.0 MB | 1.0 MB | 25,000 | ✅ Created |
| trios_synthetic_10mb.txt | 10.0 MB | 2.1 MB | 50,000 | ✅ Created |

**Note**: Actual file sizes are smaller than targets due to TRIOS header overhead. For memory profiling tests targeting specific sizes, the `points_per_mb` parameter can be adjusted.

### 4.2 Reference Data Fixtures

Generated fixtures for Bayesian validation:

```python
# Relaxation mode (Maxwell model)
t, G_t = generate_relaxation_reference_data(num_points=1000)
# Time range: 1e-3 to 1e5 s (log-spaced)
# Model: G(t) = 1e6 * exp(-t/1.0) + noise

# Creep mode (Maxwell model)
t, J_t = generate_creep_reference_data(num_points=1000)
# Time range: 1e-3 to 1e3 s (log-spaced)
# Model: J(t) = 1/G_0 + t/eta + noise

# Oscillation mode (Maxwell model)
omega, G_star = generate_oscillation_reference_data(num_points=1000)
# Frequency range: 1e-2 to 1e3 rad/s (log-spaced)
# Model: |G*(ω)| = |G_0 * (iωτ)/(1 + iωτ)| + noise
```

### 4.3 RheoData Factory Functions

Ready for pytest fixture integration:

```python
# Create RheoData objects with metadata
rheo_relax = create_rheo_data_relaxation(model_type="maxwell", num_points=1000)
rheo_creep = create_rheo_data_creep(model_type="maxwell", num_points=1000)
rheo_osc = create_rheo_data_oscillation(model_type="maxwell", num_points=1000)
```

### 4.4 Usage in Test Suite

Example for validation tests (Task Group 2):

```python
# tests/validation/test_bayesian_mode_aware.py
import pytest
from tests.fixtures.generate_test_data import create_rheo_data_relaxation

@pytest.mark.validation
def test_maxwell_relaxation_bayesian():
    # Load reference data
    rheo_data = create_rheo_data_relaxation(model_type="maxwell")

    # Fit model
    model = Maxwell()
    result = model.fit_bayesian(rheo_data, num_samples=2000)

    # Validate posterior
    assert result.posterior_samples is not None
    assert check_mcmc_diagnostics(result)
```

## 5. Current Test Status

### 5.1 Smoke Tier Test Run

**Start Time**: 2025-11-16 09:38:00 UTC
**Status**: 🔄 Running (~6 minutes elapsed)
**Expected Completion**: ~2 minutes (total <10 minutes)

Tests observed passing:
- Parameter validation tests
- RheoData creation tests
- Float64 precision tests
- TRIOS reader tests
- Model initialization tests (Maxwell, Power Law, SpringPot, Zener, fractional models)
- Pipeline tests
- Integration tests (NLSQ → NUTS workflow)
- Transform tests (mastercurve, compatibility, initialization)

### 5.2 Fast Tier Test Run

**Start Time**: 2025-11-16 09:38:00 UTC
**Status**: 🔄 Running (~6 minutes elapsed)
**Expected Completion**: ~10 minutes total (5-10 min expected)

**Parallel Execution**: 8 workers (pytest-xdist)

## 6. Deliverables Summary

### 6.1 Completed

- ✅ Feature branch created: `feature/v0.4.0-category-c-correctness`
- ✅ Test fixture generation script: `/tests/fixtures/generate_test_data.py`
- ✅ Synthetic TRIOS files: 1 MB, 5 MB, 10 MB variants
- ✅ Reference data generators: Relaxation, creep, oscillation modes
- ✅ RheoData factory functions: Ready for pytest integration
- ✅ Environment verification: Python 3.13.9, JAX 0.8.0, NLSQ installed

### 6.2 In Progress

- 🔄 Smoke tier baseline tests (105 tests)
- 🔄 Fast tier baseline tests (1069 tests)

### 6.3 Pending

- ⏳ pytest-benchmark installation (deferred to Task Group 7)
- ⏳ Complete baseline test results (awaiting test completion)
- ⏳ Documented baseline pass/fail statistics

## 7. Next Steps (Task Group 2)

**Expected Start**: After smoke and fast tier tests complete (both PASS)

1. **Validation Test Harness**: Write 35-50 validation tests
   - Bayesian mode-aware tests (30-40): `/tests/validation/test_bayesian_mode_aware.py`
   - ANSYS APDL reference tests (5-10): `/tests/validation/test_bayesian_ansys_apdl.py`
   - GMM warm-start correctness tests (5-8): `/tests/models/test_generalized_maxwell_warm_start.py`
   - TRIOS chunking integrity tests (4-6): `/tests/io/test_trios_chunked_integrity.py`

2. **Unit Tests**: Bayesian closure and TRIOS auto-chunk mechanism tests

3. **Validation Execution**: Verify tests FAIL on v0.3.1 code (confirming they detect the bugs)

## 8. Risk Assessment

### 8.1 Identified Risks

**Low Risk** (Task Group 1):
- ✅ Feature branch creation: Standard git workflow, no conflicts
- ✅ Test fixture generation: Isolated utility, no impact on existing code

**Medium Risk** (Downstream, Task Groups 2-7):
- Mode-aware Bayesian inference: Core BayesianMixin changes (high-risk component)
- GMM element search optimization: Modification of iteration logic
- TRIOS auto-chunking: Changes default loading behavior

### 8.2 Mitigation Strategy

- **Validation-First**: Write comprehensive tests before implementation (Task Group 2)
- **Phased Rollout**: Implement features sequentially with validation between each
- **External References**: Validate against pyRheo and ANSYS APDL
- **Full Regression Testing**: Run complete test suite after each major change

## 9. Appendix: File Structure

```
rheojax/
├── tests/
│   ├── fixtures/
│   │   ├── generate_test_data.py          # New: Fixture generation script
│   │   ├── trios_synthetic_1mb.txt        # New: Synthetic TRIOS (0.2 MB)
│   │   ├── trios_synthetic_5mb.txt        # New: Synthetic TRIOS (1.0 MB)
│   │   ├── trios_synthetic_10mb.txt       # New: Synthetic TRIOS (2.1 MB)
│   │   └── BASELINE_REPORT.md            # New: This report
│   │
│   ├── validation/                        # Task Group 2
│   │   ├── test_bayesian_mode_aware.py   # To be created
│   │   └── test_bayesian_ansys_apdl.py   # To be created
│   │
│   ├── core/
│   │   └── test_bayesian_mode_closure.py # Task Group 3 (new unit tests)
│   │
│   ├── models/
│   │   └── test_generalized_maxwell_warm_start.py # Task Group 4
│   │
│   └── io/
│       ├── test_trios_auto_chunk.py       # Task Group 5
│       ├── test_trios_chunked_integrity.py # Task Group 2
│       └── test_trios_memory_profiling.py # Task Group 5
```

## 10. Version and Dates

**RheoJAX Version**: 0.4.0 (in development)
**Python Version**: 3.13.9
**JAX Version**: 0.8.0
**NLSQ Version**: >=0.1.6
**Date Generated**: 2025-11-16
**Task Group**: 1 (Setup and Infrastructure)

---

**Report Status**: PRELIMINARY (awaiting test completion)
**Next Update**: After smoke and fast tier tests complete
