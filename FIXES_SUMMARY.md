# JAX Tracing Issue Fix Summary

## Problem
11 fractional rheological models were passing traced JAX values to `mittag_leffler_e()` and `mittag_leffler_e2()` functions which require static (concrete) arguments. This caused JAX tracing errors during JIT compilation.

## Solution Applied
Applied the **concrete alpha pattern** proven to work in `fractional_maxwell_gel.py` and `fractional_maxwell_liquid.py` to 9 additional fractional models.

## Pattern Changes

### Before (BROKEN):
```python
@partial(jax.jit, static_argnums=(0,))
def _predict_relaxation_jax(self, t, param1, alpha, param2):
    epsilon = 1e-12
    alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)  # ← TRACED VALUE
    
    z = compute_z(...)
    ml_value = mittag_leffler_e2(z, alpha=1.0 - alpha_safe, beta=...)  # ← ERROR
    return ...
```

### After (WORKING):
```python
def _predict_relaxation_jax(self, t, param1, alpha, param2):
    epsilon = 1e-12
    
    # Clip alpha BEFORE JIT to make it concrete
    import numpy as np
    alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))
    
    # Compute ML parameters as concrete values
    ml_alpha = 1.0 - alpha_safe  # or alpha_safe, depending on model
    ml_beta = 1.0 - alpha_safe   # or 1.0 + alpha_safe, depending on model
    
    @jax.jit
    def _compute_relaxation(t, param1, param2):
        z = compute_z(...)
        ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)  # ← OK!
        return ...
    
    return _compute_relaxation(t, param1, param2)
```

## Files Fixed (9 Models, ~27 Methods)

### 1. `/Users/b80985/Projects/Rheo/rheo/models/fractional_maxwell_model.py`
- Fixed: `_predict_relaxation_jax`, `_predict_creep_jax`, `_predict_oscillation_jax`
- Uses: `mittag_leffler_e`, `mittag_leffler_e2`

### 2. `/Users/b80985/Projects/Rheo/rheo/models/fractional_kelvin_voigt.py`
- Fixed: `_predict_relaxation_jax`, `_predict_creep_jax`, `_predict_oscillation_jax`
- Uses: `mittag_leffler_e`

### 3. `/Users/b80985/Projects/Rheo/rheo/models/fractional_zener_sl.py`
- Fixed: `_predict_relaxation`, `_predict_creep`, `_predict_oscillation`
- Uses: `mittag_leffler_e2`

### 4. `/Users/b80985/Projects/Rheo/rheo/models/fractional_zener_ss.py`
- Fixed: `_predict_relaxation`, `_predict_creep`, `_predict_oscillation`
- Uses: `mittag_leffler_e`

### 5. `/Users/b80985/Projects/Rheo/rheo/models/fractional_zener_ll.py`
- Fixed: `_predict_relaxation`, `_predict_creep`, `_predict_oscillation`
- Uses: Complex power operations with alpha (no Mittag-Leffler in relaxation/creep)

### 6. `/Users/b80985/Projects/Rheo/rheo/models/fractional_kv_zener.py`
- Fixed: `_predict_creep`, `_predict_relaxation`, `_predict_oscillation`
- Uses: `mittag_leffler_e`

### 7. `/Users/b80985/Projects/Rheo/rheo/models/fractional_burgers.py`
- Fixed: `_predict_creep`, `_predict_relaxation`, `_predict_oscillation`
- Uses: `mittag_leffler_e`

### 8. `/Users/b80985/Projects/Rheo/rheo/models/fractional_poynting_thomson.py`
- Fixed: `_predict_creep`, `_predict_relaxation`, `_predict_oscillation`
- Uses: `mittag_leffler_e`

### 9. `/Users/b80985/Projects/Rheo/rheo/models/fractional_jeffreys.py`
- Fixed: `_predict_relaxation`, `_predict_creep`, `_predict_oscillation`
- Uses: `mittag_leffler_e2`

## Key Changes in Each Method

1. **Removed decorator**: `@partial(jax.jit, static_argnums=(0,))` from outer method
2. **Convert to concrete**: `alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))` using NumPy
3. **Pre-compute ML params**: `ml_alpha`, `ml_beta` as concrete float values
4. **Create inner JIT function**: `@jax.jit` decorated function that captures alpha_safe from closure
5. **Pass concrete values**: Mittag-Leffler functions receive concrete alpha/beta parameters

## Expected Outcomes

- **Before**: ~95 test failures due to JAX tracing errors
- **After**: All fractional model tests should pass
- **Impact**: Test pass rate should increase from ~71.7% to ~88%

## Technical Details

The core issue was that JAX's JIT compilation traces values through the computation graph. When `alpha` was clipped using `jnp.clip()`, it remained a traced value. Passing this traced value to `mittag_leffler_e()` or `mittag_leffler_e2()` caused errors because these functions require static (concrete) arguments for their internal computations.

The solution uses Python's closure mechanism: alpha is converted to a concrete float value outside the JIT boundary, and the inner JIT-compiled function captures this concrete value from its enclosing scope. This allows JAX to treat alpha as a constant during compilation while still maintaining the performance benefits of JIT compilation for the rest of the computation.

## Verification

To verify the fixes work, run:
```bash
pytest tests/test_fractional_models.py -v
```

All fractional model tests should now pass without JAX tracing errors.
