# IKH Notebooks Fix Summary

## Executive Summary

All 6 MIKH notebooks (01-06) have been analyzed. **NB01 (flow curve) works correctly**. The other 5 notebooks require API updates to align with the IKH model's return mapping formulation which requires explicit strain data for time-domain protocols.

## Status by Notebook

| Notebook | Status | Priority | Issue |
|----------|--------|----------|-------|
| NB01: Flow Curve | ✓ Working | - | No issues |
| NB02: Startup Shear | ✗ Broken | HIGH | Missing strain data |
| NB03: Relaxation | ✗ Broken | MEDIUM | JAX autodiff issue with ODE |
| NB04: Creep | ✗ Broken | MEDIUM | JAX autodiff issue with ODE |
| NB05: SAOS | ✓ Working | - | No issues |
| NB06: LAOS | ✗ Broken | HIGH | Missing strain data |

## Root Cause Analysis

### Issue 1: Missing Strain Data (NB02, NB06)

**Root Cause:** IKH models use incremental return mapping formulation for plasticity, which requires both time and strain history as inputs. The `_extract_time_strain()` method enforces this requirement.

**Affected Protocols:**
- Startup shear (constant shear rate → strain = γ̇ × t)
- LAOS (oscillatory → strain from data or computed)

**Not Affected:**
- Flow curve (uses `predict_flow_curve` specialized method)
- SAOS (fits to modulus, not time-domain stress)

### Issue 2: JAX Autodiff with Diffrax ODE (NB03, NB04)

**Root Cause:** NLSQ optimizer uses forward-mode autodiff (jvp) to compute Jacobians, but Diffrax's checkpointed while loops use custom_vjp, which are incompatible.

**Error Message:**
```
TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function.
```

**Affected Protocols:**
- Relaxation (ODE formulation)
- Creep (ODE formulation)

**Workaround:** Use `method='scipy'` which falls back to scipy.optimize with finite-difference Jacobians.

## Detailed Fixes

### NB02: 02_mikh_startup_shear.ipynb

**Cell 9 (NLSQ Fitting):**
```python
# BEFORE:
model.fit(t_data, stress_data, test_mode="startup", gamma_dot=ref_rate)

# AFTER:
# Compute strain from constant shear rate
strain_data = ref_rate * t_data
model.fit(t_data, stress_data, test_mode="startup", strain=strain_data)
```

**Cell 15 (Bayesian Inference):**
```python
# BEFORE:
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="startup",
    gamma_dot=ref_rate,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=42,
)

# AFTER:
# Compute strain from constant shear rate
strain_data = ref_rate * t_data
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="startup",
    strain=strain_data,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=42,
)
```

**Also update Cell 13** (rate-dependent analysis predictions - these use `predict_startup` which already handles gamma_dot correctly).

---

### NB03: 03_mikh_stress_relaxation.ipynb

**Cell 12 (NLSQ Fitting):**
```python
# BEFORE:
model_fit.fit(t_data, stress_data, test_mode='relaxation', sigma_0=sigma_0)

# AFTER:
# Use scipy fallback to avoid JAX autodiff issue with Diffrax ODE
model_fit.fit(t_data, stress_data, test_mode='relaxation', sigma_0=sigma_0, method='scipy')
```

**Cell 16 (Bayesian Inference):** No changes needed - Bayesian uses different code path.

---

### NB04: 04_mikh_creep.ipynb

**Cell 9 (NLSQ Fitting):**
```python
# BEFORE:
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied)

# AFTER:
# Use scipy fallback to avoid JAX autodiff issue with Diffrax ODE
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied, method='scipy')
```

**Cell 15 (Bayesian Inference):** No changes needed.

---

### NB05: 05_mikh_saos.ipynb

**No changes required** - Already working correctly.

---

### NB06: 06_mikh_laos.ipynb

**Cell 6 (Load Data):**
```python
# BEFORE:
t_data, strain_data, stress_data = load_pnas_laos(omega=omega, strain_amplitude_index=strain_amp_idx)
gamma_0 = np.max(np.abs(strain_data))

# AFTER (no change, just note that strain_data is available):
t_data, strain_data, stress_data = load_pnas_laos(omega=omega, strain_amplitude_index=strain_amp_idx)
gamma_0 = np.max(np.abs(strain_data))
```

**Cell 9 (NLSQ Fitting):**
```python
# BEFORE:
model.fit(t_data, stress_data, test_mode="laos", gamma_0=gamma_0, omega=omega)

# AFTER:
# Use the actual strain data loaded in Cell 6
model.fit(t_data, stress_data, test_mode="laos", strain=strain_data)
```

**Cell 20 (Bayesian Inference):**
```python
# BEFORE:
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="laos",
    gamma_0=gamma_0,
    omega=omega,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=42,
)

# AFTER:
# Use the actual strain data
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="laos",
    strain=strain_data,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=42,
)
```

**Cell 17 (Multi-Frequency Analysis):**
```python
# In the loop, update predictions:
# BEFORE:
gamma_0_w = np.max(np.abs(d["strain"]))
stress_pred_w = model.predict_laos(d["time"], gamma_0=gamma_0_w, omega=w)

# AFTER (predict_laos may need strain too, check signature):
# For prediction, gamma_0/omega might be sufficient - check model implementation
# If prediction also needs strain:
stress_pred_w = model.predict_laos(d["time"], strain=d["strain"])
```

## Implementation Plan

### Phase 1: High Priority (Blocking)
1. Fix NB02 (startup shear) - Add strain computation
2. Fix NB06 (LAOS) - Use loaded strain data

### Phase 2: Medium Priority (Workaround available)
3. Fix NB03 (relaxation) - Add method='scipy'
4. Fix NB04 (creep) - Add method='scipy'

### Phase 3: Verification
5. Run test_mikh_notebooks.py to verify all fixes
6. Update notebooks with explanatory comments about the API

## Technical Notes

### Strain Computation for Startup Shear
```python
# For constant shear rate startup:
strain = gamma_dot * time

# This is because:
# γ̇ = dγ/dt → γ(t) = ∫₀ᵗ γ̇ dt' = γ̇ × t (for constant γ̇)
```

### Why SAOS Works Without Time-Domain Strain
SAOS (NB05) fits to complex modulus G*(ω) = G'(ω) + iG''(ω), which is frequency-domain data. The fit uses `test_mode='oscillation'` with frequency input, not time-domain stress, so it doesn't trigger the return mapping path that requires strain history.

### Diffrax ODE Issue
The Diffrax library uses `custom_vjp` with checkpointing for memory-efficient ODE solving. JAX's forward-mode autodiff (jvp) cannot differentiate through custom_vjp functions, causing the error. The scipy fallback uses finite differences instead of autodiff for the Jacobian.

## Testing Strategy

After implementing fixes, test each notebook individually:

```bash
cd /Users/b80985/Projects/rheojax/examples/ikh

# Test NB02
python -c "
import sys
sys.path.insert(0, '../utils')
from rheojax.models.ikh import MIKH
from ikh_tutorial_utils import load_pnas_startup
import numpy as np

t, stress = load_pnas_startup(gamma_dot=1.0)
strain = 1.0 * t  # FIX
model = MIKH()
model.fit(t, stress, test_mode='startup', strain=strain)  # FIX
print('NB02 FIX VERIFIED')
"

# Test NB03
python -c "
from rheojax.models.ikh import MIKH
import numpy as np

t = np.logspace(-1, 2, 100)
stress = 100 * np.exp(-t / 10)
model = MIKH()
model.fit(t, stress, test_mode='relaxation', sigma_0=100.0, method='scipy')  # FIX
print('NB03 FIX VERIFIED')
"

# Test NB06
python -c "
import sys
sys.path.insert(0, '../utils')
from rheojax.models.ikh import MIKH
from ikh_tutorial_utils import load_pnas_laos

t, strain, stress = load_pnas_laos(omega=1.0, strain_amplitude_index=8)
model = MIKH()
model.fit(t, stress, test_mode='laos', strain=strain)  # FIX
print('NB06 FIX VERIFIED')
"
```

## Success Criteria

All tests pass:
```
NB01_flow_curve           ✓ PASS
NB02_startup_shear        ✓ PASS
NB03_relaxation           ✓ PASS
NB04_creep                ✓ PASS
NB05_saos                 ✓ PASS
NB06_laos                 ✓ PASS

Total: 6/6 passed
```
