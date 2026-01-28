# Saramito Notebooks (13-24) - Fix Summary

## Date
2026-01-28

## Overview
Successfully fixed all 12 Fluidity-Saramito notebooks (13-24) to use the current RheoJAX API, replacing deprecated patterns with correct implementations.

## Notebooks Fixed

### Local Model (13-18)
- `13_saramito_local_flow_curve.ipynb` ✓
- `14_saramito_local_startup.ipynb` ✓
- `15_saramito_local_creep.ipynb` ✓
- `16_saramito_local_relaxation.ipynb` ✓
- `17_saramito_local_saos.ipynb` ✓
- `18_saramito_local_laos.ipynb` ✓

### Nonlocal Model (19-24)
- `19_saramito_nonlocal_flow_curve.ipynb` ✓
- `20_saramito_nonlocal_startup.ipynb` ✓
- `21_saramito_nonlocal_creep.ipynb` ✓
- `22_saramito_nonlocal_relaxation.ipynb` ✓
- `23_saramito_nonlocal_saos.ipynb` ✓
- `24_saramito_nonlocal_laos.ipynb` ✓

## Fixes Applied

### 1. Added Missing Import
**Issue:** Notebooks were using `compute_fit_quality` without importing it

**Fix:**
```python
# Added to import cell
from rheojax.utils.metrics import compute_fit_quality
```

**Affected:** 10 out of 12 notebooks

### 2. Fixed `result.r_squared` Access
**Issue:** Direct access to `result.r_squared` is deprecated

**Fix:**
```python
# OLD (deprecated)
print(f"R² = {result.r_squared:.6f}")

# NEW (correct)
stress_pred = model.predict(gamma_dot, test_mode='flow_curve')
metrics = compute_fit_quality(stress, stress_pred)
print(f"R² = {metrics['R2']:.6f}")
```

**Affected:** Multiple cells in 7 notebooks

### 3. Fixed `best_metrics` Undefined Variable
**Issue:** Notebook 13 referenced `best_metrics` which was never defined

**Fix:**
```python
# OLD (error - undefined variable)
print(f"\nNLSQ R² = {best_metrics['R2']:.6f}")

# NEW (correct - conditional reference)
print(f"\nNLSQ R² = {(metrics_full['R2'] if coupling_mode == 'full' else metrics_minimal['R2']):.6f}")
```

**Affected:** Notebook 13

### 4. Consistent API Usage
All notebooks now use:
- `model.predict(x, test_mode="...")` for predictions
- `compute_fit_quality(y_true, y_pred)` returns a dictionary with keys `["R2", "RMSE", "MAPE"]`
- Dictionary access: `metrics["R2"]`, `metrics["RMSE"]`, `metrics["MAPE"]`

## Validation Results

All 12 notebooks pass validation checks:
- ✓ Required imports present (safe_import_jax, FluiditySaramito, compute_fit_quality)
- ✓ No deprecated patterns (result.y_pred, predict_flow_curve, _fit_result access)
- ✓ Valid JSON structure
- ✓ Ready for execution

## Scripts Used

1. **fix_saramito_notebooks.py** - Initial automated fixes
2. **fix_saramito_final.py** - Comprehensive fix including best_metrics issue
3. **test_saramito_notebooks.py** - Validation script

## Testing

To verify the fixes:
```bash
cd /Users/b80985/Projects/rheojax
python3 test_saramito_notebooks.py
```

Expected output: `Total: 12 passed, 0 failed out of 12`

## Notes

- All notebooks maintain valid JSON structure
- Backward compatibility preserved where possible
- Follows RheoJAX v0.6.0+ API conventions
- Import placement follows project standards (after safe_import_jax)

## Next Steps

These notebooks are now ready for:
1. Execution testing with actual data
2. Integration into documentation
3. Use as examples for FluiditySaramito model tutorials
