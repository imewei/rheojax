# Fluidity Saramito Nonlocal Notebooks: API Deprecation Fixes

## Summary

Fixed deprecated API usage in notebooks 19-24 (FluiditySaramitoNonlocal series) to align with current RheoJAX API where `fit()` returns `self` rather than a result object.

## Problem Patterns Fixed

### 1. Incorrect: `result.r_squared` and `result.rmse` access
**Issue**: `fit()` returns `self`, not a result object with `r_squared` and `rmse` attributes.

**Old (broken) code:**
```python
result_nlsq = model.fit(rheo_data, test_mode='flow_curve')
print(f"R²: {result_nlsq.r_squared:.6f}")
print(f"RMSE: {result_nlsq.rmse:.4f}")
```

**New (correct) code:**
```python
def compute_fit_quality(y_true, y_pred):
    """Compute R² and RMSE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    if y_true.ndim > 1:
        residuals = residuals.ravel()
        y_true = y_true.ravel()
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))
    return {"R2": r2, "RMSE": rmse}

model.fit(rheo_data, test_mode='flow_curve')
y_pred = model.predict(X, test_mode='flow_curve')
metrics = compute_fit_quality(y, y_pred)
print(f"R²: {metrics['R2']:.6f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
```

### 2. Incorrect: `model._fit_result` access
**Issue**: Internal `_fit_result` attribute doesn't exist.

**Fixed by**: Using the `compute_fit_quality()` helper function instead.

## Files Modified

### 19_saramito_nonlocal_flow_curve.ipynb
- **Cell 9**: Added `compute_fit_quality()` helper, replaced `result_nlsq.r_squared/rmse` with `metrics['R2']/metrics['RMSE']`
- **Cell 18**: Fixed local model comparison to use `compute_fit_quality()`
- **Cell 22**: Updated save results to use `metrics` dict

### 20_saramito_nonlocal_startup.ipynb
- **Cell 17**: Added `compute_fit_quality()` helper, fixed NLSQ fitting metrics
- **Cell 19**: Fixed fit quality visualization to use computed metrics
- **Cell 27**: Updated save results to reference `metrics['R2']`

### 21_saramito_nonlocal_creep.ipynb
- **Status**: Already uses utility functions from `fluidity_tutorial_utils.py` - no changes needed
- **Note**: This notebook imports `compute_fit_quality()` from utilities

### 22_saramito_nonlocal_relaxation.ipynb
- **Cell 10**: Added `compute_fit_quality()` helper, fixed NLSQ fitting
- **Cell 11**: Fixed parameter recovery display (cosmetic, no API issue)

### 23_saramito_nonlocal_saos.ipynb
- **Cell 10**: Added `compute_fit_quality()` helper, fixed NLSQ fitting metrics
- **Cell 11**: Removed references to `result_nlsq` in plot titles
- **Cell 25**: Fixed local model comparison to use `compute_fit_quality()`

### 24_saramito_nonlocal_laos.ipynb
- **Cell 12**: Added `compute_fit_quality()` helper, fixed NLSQ fitting metrics

## Key Changes

1. **Added helper function** to all affected notebooks:
   ```python
   def compute_fit_quality(y_true, y_pred):
       """Compute R² and RMSE."""
       # ... implementation
       return {"R2": r2, "RMSE": rmse}
   ```

2. **Replaced all instances** of:
   - `result.r_squared` → `metrics['R2']`
   - `result.rmse` → `metrics['RMSE']`
   - `result_nlsq.r_squared` → `metrics['R2']`
   - `result_nlsq.rmse` → `metrics['RMSE']`

3. **Updated workflow** from:
   ```python
   result = model.fit(data)
   print(result.r_squared)
   ```

   To:
   ```python
   model.fit(data)
   y_pred = model.predict(X, test_mode=...)
   metrics = compute_fit_quality(y_true, y_pred)
   print(metrics['R2'])
   ```

## Test Mode Mapping

Ensured correct test modes for each notebook:
- **19 (flow_curve)**: `test_mode="flow_curve"`
- **20 (startup)**: `test_mode="startup"`, `gamma_dot=...`
- **21 (creep)**: `test_mode="creep"`, `sigma_0=...` or `stress=...`
- **22 (relaxation)**: `test_mode="relaxation"`, `gamma_0=...`, `f_init=...`
- **23 (saos)**: `test_mode="oscillation"`, `gamma_0=0.01`
- **24 (laos)**: `test_mode="laos"`, `gamma_0=...`, `omega=...`

## Verification

All fixes ensure:
1. ✅ No access to non-existent `result.r_squared` or `result.rmse`
2. ✅ No access to non-existent `model._fit_result`
3. ✅ Proper use of `model.predict()` with correct `test_mode` and protocol kwargs
4. ✅ Consistent metric computation across all notebooks
5. ✅ Backward compatibility with existing code patterns

## Notes

- Notebook 21 (creep) was already using utility functions and didn't need fixes
- All other notebooks now use the same `compute_fit_quality()` pattern
- The helper function handles both 1D and multi-dimensional arrays correctly
- Metrics are computed after calling `predict()` to ensure consistency
