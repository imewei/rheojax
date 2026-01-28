# IKH Notebooks (01-06) - Fix Implementation Report

## Executive Summary

I have analyzed all 6 MIKH notebooks (01_mikh_flow_curve through 06_mikh_laos) and identified the API alignment issues preventing them from executing correctly with RheoJAX v0.4.0+.

**Status:**
- ✓ **2 notebooks working** (NB01, NB05)
- ✗ **4 notebooks need fixes** (NB02, NB03, NB04, NB06)

---

## Root Causes

### Issue 1: Missing Strain Data (NB02, NB06)

**Problem:** IKH models use incremental return mapping formulation for elastoplasticity, which requires both time AND strain history as inputs. The `_extract_time_strain()` method validates this requirement.

**Error:**
```
ValueError: IKH models require both time and strain history.
Pass RheoData, or X of shape (2, N), or X=time with strain=gamma kwarg.
```

**Affected:** NB02 (startup shear), NB06 (LAOS)

### Issue 2: JAX Autodiff with Diffrax ODE (NB03, NB04)

**Problem:** NLSQ optimizer uses forward-mode autodiff (jvp) to compute Jacobians, but Diffrax's checkpointed ODE solver uses custom_vjp, which are incompatible.

**Error:**
```
TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function.
```

**Affected:** NB03 (relaxation), NB04 (creep)

---

## Detailed Fixes

### NB02: 02_mikh_startup_shear.ipynb

**Priority:** HIGH (blocking execution)

**Cell 9 - NLSQ Fitting:**
```python
# BEFORE:
model.fit(t_data, stress_data, test_mode="startup", gamma_dot=ref_rate)

# AFTER:
# Compute strain from constant shear rate (γ = γ̇ × t)
strain_data = ref_rate * t_data
model.fit(t_data, stress_data, test_mode="startup", strain=strain_data)
```

**Cell 15 - Bayesian Inference:**
```python
# BEFORE:
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="startup",
    gamma_dot=ref_rate,  # <-- Not used correctly
    ...
)

# AFTER:
# Compute strain from constant shear rate
strain_data = ref_rate * t_data
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="startup",
    strain=strain_data,  # <-- Explicit strain required
    ...
)
```

**Explanation:** For constant shear rate startup, strain is γ(t) = γ̇ × t. The return mapping formulation needs this explicit strain history to track elastic and plastic components incrementally.

---

### NB03: 03_mikh_stress_relaxation.ipynb

**Priority:** MEDIUM (scipy fallback available)

**Cell 12 - NLSQ Fitting:**
```python
# BEFORE:
model_fit.fit(t_data, stress_data, test_mode='relaxation', sigma_0=sigma_0)

# AFTER:
# Use scipy fallback to avoid JAX autodiff issue with Diffrax ODE solver
model_fit.fit(t_data, stress_data, test_mode='relaxation', sigma_0=sigma_0, method='scipy')
```

**Explanation:** The ODE formulation for relaxation uses Diffrax's `diffeqsolve` with checkpointing. When NLSQ tries to compute the Jacobian using forward-mode autodiff (jvp), it fails because Diffrax uses custom_vjp. The scipy fallback uses finite differences instead.

**Note:** Bayesian inference (Cell 16) works fine as-is because it uses a different code path without Jacobian computation.

---

### NB04: 04_mikh_creep.ipynb

**Priority:** MEDIUM (scipy fallback available)

**Cell 9 - NLSQ Fitting:**
```python
# BEFORE:
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied)

# AFTER:
# Use scipy fallback to avoid JAX autodiff issue with Diffrax ODE solver
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied, method='scipy')
```

**Explanation:** Same as NB03 - the ODE formulation for creep has the same JAX autodiff incompatibility.

---

### NB06: 06_mikh_laos.ipynb

**Priority:** HIGH (blocking execution)

**Cell 9 - NLSQ Fitting:**
```python
# BEFORE:
model.fit(t_data, stress_data, test_mode="laos", gamma_0=gamma_0, omega=omega)

# AFTER:
# Use the strain data already loaded in Cell 6
# IKH models require explicit strain for return mapping formulation
model.fit(t_data, stress_data, test_mode="laos", strain=strain_data)
```

**Cell 20 - Bayesian Inference:**
```python
# BEFORE:
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="laos",
    gamma_0=gamma_0,
    omega=omega,  # <-- Not sufficient
    ...
)

# AFTER:
# Use the strain data already loaded in Cell 6
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="laos",
    strain=strain_data,  # <-- Explicit strain required
    ...
)
```

**Explanation:** The `load_pnas_laos()` function already returns `(time, strain, stress)`, so the strain data is available in `strain_data` from Cell 6. The return mapping formulation needs this explicit strain history.

---

## Why NB01 and NB05 Work Without Changes

### NB01: Flow Curve
- Uses `predict_flow_curve(gamma_dot)` specialized method
- Flow curves are steady-state (no time/strain history needed)
- Directly solves algebraic equations for stress(γ̇)

### NB05: SAOS
- Fits to frequency-domain complex modulus G*(ω)
- Uses `test_mode='oscillation'` with frequency input
- Doesn't trigger return mapping path (no time-domain stress)
- Uses Maxwell analytical expressions for G'(ω) and G''(ω)

---

## Implementation Checklist

### Step 1: Apply Fixes
- [ ] Edit NB02 Cell 9 - Add strain computation
- [ ] Edit NB02 Cell 15 - Add strain to Bayesian
- [ ] Edit NB03 Cell 12 - Add method='scipy'
- [ ] Edit NB04 Cell 9 - Add method='scipy'
- [ ] Edit NB06 Cell 9 - Use strain_data
- [ ] Edit NB06 Cell 20 - Use strain_data in Bayesian

### Step 2: Verify Fixes
```bash
cd /Users/b80985/Projects/rheojax/examples/ikh
python test_mikh_notebooks.py
```

### Step 3: Expected Output
```
============================================================
NB01_flow_curve           ✓ PASS
NB02_startup_shear        ✓ PASS
NB03_relaxation           ✓ PASS
NB04_creep                ✓ PASS
NB05_saos                 ✓ PASS
NB06_laos                 ✓ PASS

Total: 6/6 passed
============================================================
```

---

## Files Created

1. **test_mikh_notebooks.py** - Automated validation script
2. **FIXES_NEEDED.md** - Detailed fix documentation
3. **FIX_SUMMARY.md** - Comprehensive fix summary
4. **apply_fixes.py** - Quick reference for exact code changes
5. **THIS_REPORT.md** - Final implementation report

---

## Technical Notes

### Strain Computation for Startup Shear
```python
# For constant shear rate startup:
# γ̇ = dγ/dt → γ(t) = ∫₀ᵗ γ̇ dt' = γ̇ × t
strain = gamma_dot * time
```

### Why method='scipy' Works
The scipy.optimize.least_squares fallback:
- Uses finite-difference approximation for Jacobian
- Doesn't require differentiating through Diffrax
- Slightly slower but numerically stable
- Already implemented in rheojax.utils.optimization

### Return Mapping Formulation
IKH models track state incrementally:
```python
for i in range(n_steps):
    # Elastic predictor
    σ_trial = σ_n + G * Δγ

    # Plastic corrector (if yield)
    if |σ_trial - α| > σ_y:
        σ_n+1 = σ_trial - G * Δγ_p
        α_n+1 = α_n + C * Δγ_p - ...

    # Structure evolution
    λ_n+1 = λ_n + (1-λ_n)/τ - Γλ_n|γ̇_p|
```

This requires explicit Δγ = strain[i+1] - strain[i] at each step.

---

## Conclusion

All 4 failing notebooks can be fixed with minimal changes:
- **NB02, NB06:** Add explicit strain arrays (1-2 lines per cell)
- **NB03, NB04:** Add method='scipy' parameter (1 parameter per fit call)

The fixes preserve the scientific content and pedagogical flow of the notebooks while aligning with the current RheoJAX API.

---

## Next Steps

1. **Apply the fixes** to the 4 failing notebooks using the exact code snippets above
2. **Run the validation script** to verify all notebooks pass
3. **Optionally add explanatory comments** in the notebooks about why strain is required
4. **Consider updating the IKH tutorial utilities** to provide helper functions for strain computation

---

**Report Generated:** 2026-01-28
**RheoJAX Version:** v0.4.0+
**Python Version:** 3.12+
**JAX Version:** 0.8.0
