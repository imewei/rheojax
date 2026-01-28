# IKH Notebooks (01-06) - Required API Fixes

## Summary

All 6 MIKH notebooks (01_mikh_flow_curve through 06_mikh_laos) require updates to align with the RheoJAX v0.4.0+ IKH model API. The core issue is that IKH models use a **return mapping formulation** for strain-driven protocols, requiring both time AND strain arrays.

## Issues Identified

### 1. NB02: Startup Shear - CRITICAL FIX NEEDED

**Current code (Cell 9):**
```python
model.fit(t_data, stress_data, test_mode="startup", gamma_dot=ref_rate)
```

**Problem:** For startup shear (constant shear rate), IKH models require explicit strain history, not just `gamma_dot`.

**Fix:**
```python
# Compute strain from constant shear rate
strain_data = ref_rate * t_data

# Option 1: Pass strain as kwarg
model.fit(t_data, stress_data, test_mode="startup", strain=strain_data)

# Option 2: Pass as (2, N) array
X = np.vstack([t_data, strain_data])
model.fit(X, stress_data, test_mode="startup")
```

**Cells to update:** 9, 10, 11, 13, 15, 18

### 2. NB03: Stress Relaxation - CRITICAL FIX NEEDED

**Current code (Cell 12):**
```python
model_fit.fit(t, stress, test_mode='relaxation', sigma_0=100.0)
```

**Problem:** JAX autodiff incompatibility with Diffrax ODE solver when using NLSQ jacobian computation.

**Fix:** Use the `method='scipy'` fallback or adjust NLSQ settings:
```python
# Option 1: Use scipy fallback
model_fit.fit(t, stress, test_mode='relaxation', sigma_0=100.0, method='scipy')

# Option 2: Disable NLSQ jacobian (use finite differences)
from rheojax.utils.optimization import nlsq_optimize
# This is already handled by the fallback in optimization.py
```

**Cells to update:** 12

### 3. NB04: Creep - CRITICAL FIX NEEDED

**Current code (Cell 9):**
```python
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied)
```

**Problem:** Same as NB03 - ODE formulation with autodiff issues.

**Fix:**
```python
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied, method='scipy')
```

**Cells to update:** 9

### 4. NB05: SAOS - WORKING (minor improvement)

**Current status:** Functional, but could be improved.

**Optional improvement (Cell 12):**
```python
# Current: fits to magnitude only
model_fit.fit(omega, np.abs(G_star), test_mode='oscillation')

# Better: fit to complex modulus (if supported)
# For now, keep as-is since MIKH SAOS uses Maxwell approximation
```

### 5. NB06: LAOS - CRITICAL FIX NEEDED

**Current code (Cell 9):**
```python
model.fit(t_data, stress_data, test_mode="laos", gamma_0=gamma_0, omega=omega)
```

**Problem:** Missing strain data - LAOS uses return mapping and needs explicit strain.

**Fix:**
```python
# Use the loaded strain data from load_pnas_laos
# The data loader already provides strain!
model.fit(t_data, stress_data, test_mode="laos", strain=strain_data)

# Or compute from gamma_0 and omega if needed
# strain_computed = gamma_0 * np.sin(omega * t_data)
```

**Cells to update:** 9, 10, 11, 17, 18, 20

## Implementation Priority

### HIGH PRIORITY (Blocking execution):
1. **NB02** - Fix startup strain passing
2. **NB04** - Fix creep ODE fitting
3. **NB06** - Fix LAOS strain passing

### MEDIUM PRIORITY (Fallback available):
1. **NB03** - Relaxation ODE fitting (scipy fallback works)

### LOW PRIORITY (Working):
1. **NB01** - Flow curve (no issues)
2. **NB05** - SAOS (working as-is)

## Detailed Cell-by-Cell Fixes

### NB02: 02_mikh_startup_shear.ipynb

**Cell 9 (NLSQ fit):**
```python
# OLD:
model.fit(t_data, stress_data, test_mode="startup", gamma_dot=ref_rate)

# NEW:
strain_data = ref_rate * t_data
model.fit(t_data, stress_data, test_mode="startup", strain=strain_data)
```

**Cell 15 (Bayesian fit):**
```python
# OLD:
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="startup",
    gamma_dot=ref_rate,  # This kwarg is passed through but not used correctly
    ...
)

# NEW:
strain_data = ref_rate * t_data
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="startup",
    strain=strain_data,
    ...
)
```

### NB03: 03_mikh_stress_relaxation.ipynb

**Cell 12 (NLSQ fit):**
```python
# OLD:
model_fit.fit(t_data, stress_data, test_mode='relaxation', sigma_0=sigma_0)

# NEW (use scipy fallback):
model_fit.fit(t_data, stress_data, test_mode='relaxation', sigma_0=sigma_0, method='scipy')
```

**Cell 16 (Bayesian fit):**
```python
# Keep as-is - Bayesian uses different code path that works
```

### NB04: 04_mikh_creep.ipynb

**Cell 9 (NLSQ fit):**
```python
# OLD:
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied)

# NEW:
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied, method='scipy')
```

### NB06: 06_mikh_laos.ipynb

**Cell 9 (NLSQ fit):**
```python
# OLD:
model.fit(t_data, stress_data, test_mode="laos", gamma_0=gamma_0, omega=omega)

# NEW (strain_data is already loaded!):
model.fit(t_data, stress_data, test_mode="laos", strain=strain_data)
```

**Cell 20 (Bayesian fit):**
```python
# OLD:
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="laos",
    gamma_0=gamma_0,
    omega=omega,
    ...
)

# NEW:
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="laos",
    strain=strain_data,
    ...
)
```

## Testing After Fixes

Run the validation script:
```bash
cd /Users/b80985/Projects/rheojax/examples/ikh
python test_mikh_notebooks.py
```

Expected output after fixes:
```
NB01_flow_curve           ✓ PASS
NB02_startup_shear        ✓ PASS
NB03_relaxation           ✓ PASS
NB04_creep                ✓ PASS
NB05_saos                 ✓ PASS
NB06_laos                 ✓ PASS

Total: 6/6 passed
```

## Notes

1. **Flow curve (NB01)** is already working correctly because it uses the specialized `predict_flow_curve` which doesn't require strain.

2. **SAOS (NB05)** works because it fits to modulus magnitude, not time-domain stress response.

3. The **ODE formulation** (relaxation, creep) has known JAX autodiff issues with Diffrax's checkpointing. The scipy fallback is the recommended workaround until Diffrax updates.

4. All **return mapping protocols** (startup, LAOS) must receive explicit strain data via `strain=` kwarg or as part of a (2, N) array.

5. For **Bayesian inference**, the test_mode must be explicitly passed to ensure the correct closure is used in the model_function.
