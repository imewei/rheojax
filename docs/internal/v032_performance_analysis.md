# v0.3.2 Performance Integration Tests - Root Cause Analysis

## Executive Summary

The v0.3.2 performance integration test suite contains **parameter naming errors** that prevent tests from running, along with some performance thresholds that may need relaxation for CI/CD environments. The primary failure is **not due to algorithmic regressions**, but rather incorrect parameter names used in the test setup.

---

## Failing Test Details

### 1. **CRITICAL: `test_cumulative_performance_vs_baseline` - KeyError on Parameter Names**

**Status**: FAILED
**Error**: `KeyError: "Parameter 'tau' not found"` at line 381
**Severity**: Blocker (prevents test execution)

#### Root Cause Analysis

The test attempts to set FractionalZenerSolidSolid model parameters using **incorrect parameter names**:

```python
# WRONG (line 381-383)
model_gen.parameters.set_value("tau", tau)          # ❌ Parameter doesn't exist
model_gen.parameters.set_value("E_0", E0)           # ❌ Parameter doesn't exist
model_gen.parameters.set_value("E_inf", Einf)       # ❌ Parameter doesn't exist
```

#### Actual FractionalZenerSolidSolid Parameters

The model defines exactly **4 parameters** (verified in `rheojax/models/fractional_zener_ss.py:91-118`):

| Parameter | Type | Default | Bounds | Units | Description |
|-----------|------|---------|--------|-------|-------------|
| **Ge** | float | 1000.0 | [1e-3, 1e9] | Pa | Equilibrium modulus |
| **Gm** | float | 1000.0 | [1e-3, 1e9] | Pa | Maxwell arm modulus |
| **alpha** | float | 0.5 | [0.0, 1.0] | (none) | Fractional order |
| **tau_alpha** | float | 1.0 | [1e-6, 1e6] | s^α | Relaxation time |

**What the test should use instead**:

```python
# CORRECT (suggested fix)
model_gen.parameters.set_value("alpha", alpha)        # ✓ Correct
model_gen.parameters.set_value("tau_alpha", tau)      # ✓ Use tau_alpha instead of tau
model_gen.parameters.set_value("Ge", Ge_true)         # ✓ Use Ge instead of E_0
model_gen.parameters.set_value("Gm", Gm_true)         # ✓ Use Gm instead of E_inf
```

#### Evidence of Parameter Naming Convention

- **tau → tau_alpha**: The model uses underscore notation following mathematical convention: τ_α (relaxation time with fractional order alpha)
- **E_0 → Ge**: The code uses shear moduli (G prefix), not Young's moduli (E prefix). Ge = G_e (equilibrium modulus)
- **E_inf → Gm**: Similarly, Gm = G_m (Maxwell arm modulus)

#### When Was This Test Created?

The test file was created in **commit cfe65f6** (Nov 16, 2025) with these incorrect parameter names already present. This suggests the test was written without validating against the actual model implementation.

#### Note on Contrasting Usage

Interestingly, the **same test method** correctly uses parameters on **lines 109-112**:
```python
model_gen.parameters.set_value("alpha", alpha_true)      # ✓ Correct
model_gen.parameters.set_value("tau_alpha", tau_alpha_true)  # ✓ Correct
model_gen.parameters.set_value("Ge", Ge_true)            # ✓ Correct
model_gen.parameters.set_value("Gm", Gm_true)            # ✓ Correct
```

This inconsistency within the same test class indicates a copy-paste error or incomplete refactoring.

---

### 2. **RMSE Threshold Tests - Status Check**

**Tests Analyzed**:
- `test_end_to_end_pipeline_performance` - **PASSED** ✓
- `test_fractional_model_convergence_speedup` - Uses try/except, may skip
- `test_mastercurve_multi_dataset_speedup` - Uses try/except, may skip
- `test_batch_pipeline_multi_file_processing` - **PASSED** ✓
- `test_device_memory_efficiency` - **PASSED** ✓
- `test_backward_compatibility_api` - **PASSED** ✓

#### RMSE Analysis

**Test Result**: End-to-end pipeline test passes with:
- Absolute RMSE: 2.899e+04
- Relative RMSE: 0.066051 (6.6%)
- Threshold: 0.1 (10%)
- Status: ✓ **Within acceptable bounds**

**Conclusion**: RMSE thresholds are appropriately set and not causing failures in the main tests. The thresholds align with realistic noise levels (1% Gaussian noise added to synthetic data).

---

## Summary of Issues

| Issue | Type | Location | Severity | Root Cause |
|-------|------|----------|----------|-----------|
| **Parameter 'tau' not found** | KeyError | Line 381 | **CRITICAL** | Test uses wrong param name; should be `tau_alpha` |
| **Parameter 'E_0' not found** | Would fail after tau fix | Line 382 | **CRITICAL** | Test uses Young's modulus naming; should be `Ge` (shear) |
| **Parameter 'E_inf' not found** | Would fail after E_0 fix | Line 383 | **CRITICAL** | Test uses Young's modulus naming; should be `Gm` (shear) |
| **No RMSE threshold failure** | N/A | N/A | None | Thresholds pass validation |
| **No algorithmic regression** | N/A | N/A | None | RMSE and performance metrics acceptable |

---

## Root Cause Determination

### What This Is NOT

- ❌ **NOT a genuine algorithmic regression**: The Maxwell model produces acceptable RMSE (6.6% vs 10% threshold)
- ❌ **NOT an over-strict threshold**: Thresholds are relaxed for CI/CD (see comments on lines 76, 243, 291)
- ❌ **NOT a JAX version mismatch**: All JIT compilation and array operations working correctly
- ❌ **NOT a parameter validation issue**: The ParameterSet correctly enforces the 4-parameter model structure

### What This IS

✓ **A test authoring error**: The test uses parameter names that never existed in the FractionalZenerSolidSolid model
✓ **A parameter naming convention mismatch**: Test author used energy modulus (E) conventions instead of shear modulus (G) conventions
✓ **An inconsistency within the same test class**: Lines 109-112 use correct names; lines 381-383 use incorrect names

---

## Recommendations

### **Recommendation 1: FIX THE TEST (Highest Priority)**

**Action**: Correct parameter names in `test_cumulative_performance_vs_baseline` at lines 381-383:

```python
# Before (WRONG)
model_gen.parameters.set_value("tau", tau)
model_gen.parameters.set_value("E_0", E0)
model_gen.parameters.set_value("E_inf", Einf)

# After (CORRECT)
model_gen.parameters.set_value("tau_alpha", tau)
model_gen.parameters.set_value("Ge", E0)  # Map E_0 concept to Ge
model_gen.parameters.set_value("Gm", Einf)  # Map E_inf concept to Gm
```

**Rationale**:
- **Minimal change**: Only parameter names need updating; no model code changes required
- **No side effects**: The test logic remains unchanged; only variable mappings updated
- **Restores intended validation**: The test will now correctly validate fractional model performance
- **Aligns with model contract**: Uses the actual parameter names defined in the FractionalZenerSolidSolid class

**Expected Outcome**: Test will proceed to parameter fitting and performance validation (no RMSE threshold failures anticipated based on analysis).

---

### **Recommendation 2: Validate Against Implementation (For Code Review)**

**Action**: Add a simple validation step during test setup to ensure parameter names match the model:

```python
# Add to test class setup or as a fixture
model = FractionalZenerSolidSolid()
expected_params = {"Ge", "Gm", "alpha", "tau_alpha"}
actual_params = set(model.parameters.names)
assert expected_params == actual_params, f"Parameter mismatch: {actual_params}"
```

**Rationale**: Prevents similar mismatches in future test development

---

### **Recommendation 3: Threshold Review (Low Priority)**

**Action**: Current RMSE and timing thresholds are appropriate:
- 0.1 (10%) relative RMSE allows 1% noise with 10x margin
- 7.5s batch threshold relaxed from 5.0s for CI/CD variability
- 3.0s device efficiency threshold relaxed from 2.0s for CI/CD variability

**No changes recommended** - thresholds are well-calibrated and tests pass.

---

## Testing Verification Checklist

After implementing Recommendation 1, verify:

- [ ] `test_cumulative_performance_vs_baseline` executes without KeyError
- [ ] Fractional model parameters are set correctly
- [ ] Model predictions complete successfully
- [ ] RMSE validation passes (should be similar to end-to-end test ~6-7%)
- [ ] All 8 test methods in TestV032PerformanceIntegration pass or skip gracefully

---

## Conclusion

**The v0.3.2 performance integration tests contain no genuine algorithmic regressions or overly strict thresholds.** The primary blocker is incorrect parameter names in the test setup. This is a straightforward test authoring error requiring only parameter name corrections (3-line change). Once fixed, the test suite should validate v0.3.2 performance improvements correctly.

**Implementation Status**:
- **READY TO FIX** ✓ All issues are well-understood
- **LOW RISK** ✓ Changes are isolated to test code only
- **HIGH CONFIDENCE** ✓ Fixes align with verified model contracts
