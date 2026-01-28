#!/usr/bin/env python
"""
Apply fixes to IKH notebooks 01-06 to align with RheoJAX v0.4.0+ API.

This script documents the exact changes needed for each failing notebook.
The changes are minimal and focused on API alignment.
"""

# ==============================================================================
# SUMMARY OF FIXES NEEDED
# ==============================================================================

"""
WORKING (No fixes needed):
- NB01: 01_mikh_flow_curve.ipynb ✓
- NB05: 05_mikh_saos.ipynb ✓

BROKEN (Fixes required):
- NB02: 02_mikh_startup_shear.ipynb - Missing strain data
- NB03: 03_mikh_stress_relaxation.ipynb - JAX autodiff issue
- NB04: 04_mikh_creep.ipynb - JAX autodiff issue
- NB06: 06_mikh_laos.ipynb - Missing strain data
"""

# ==============================================================================
# NB02: 02_mikh_startup_shear.ipynb
# ==============================================================================

# Cell 9 - NLSQ Fitting
# OLD CODE:
"""
model.fit(t_data, stress_data, test_mode="startup", gamma_dot=ref_rate)
"""

# NEW CODE:
"""
# Compute strain from constant shear rate (γ = γ̇ × t)
strain_data = ref_rate * t_data
model.fit(t_data, stress_data, test_mode="startup", strain=strain_data)
"""

# Cell 15 - Bayesian Inference
# OLD CODE:
"""
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
"""

# NEW CODE:
"""
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
"""

# ==============================================================================
# NB03: 03_mikh_stress_relaxation.ipynb
# ==============================================================================

# Cell 12 - NLSQ Fitting
# OLD CODE:
"""
model_fit.fit(t_data, stress_data, test_mode='relaxation', sigma_0=sigma_0)
"""

# NEW CODE:
"""
# Use scipy fallback to avoid JAX autodiff issue with Diffrax ODE solver
model_fit.fit(t_data, stress_data, test_mode='relaxation', sigma_0=sigma_0, method='scipy')
"""

# ==============================================================================
# NB04: 04_mikh_creep.ipynb
# ==============================================================================

# Cell 9 - NLSQ Fitting
# OLD CODE:
"""
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied)
"""

# NEW CODE:
"""
# Use scipy fallback to avoid JAX autodiff issue with Diffrax ODE solver
model.fit(t_data, gamma_dot_data, test_mode="creep", sigma_applied=sigma_applied, method='scipy')
"""

# ==============================================================================
# NB06: 06_mikh_laos.ipynb
# ==============================================================================

# Cell 9 - NLSQ Fitting
# OLD CODE:
"""
model.fit(t_data, stress_data, test_mode="laos", gamma_0=gamma_0, omega=omega)
"""

# NEW CODE:
"""
# Use the strain data already loaded in Cell 6
# IKH models require explicit strain for return mapping formulation
model.fit(t_data, stress_data, test_mode="laos", strain=strain_data)
"""

# Cell 20 - Bayesian Inference
# OLD CODE:
"""
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
"""

# NEW CODE:
"""
# Use the strain data already loaded in Cell 6
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
"""

# ==============================================================================
# EXPLANATION OF FIXES
# ==============================================================================

print("""
IKH Notebooks Fix Summary
=========================

ROOT CAUSE 1: Missing Strain Data (NB02, NB06)
----------------------------------------------
IKH models use incremental return mapping formulation for elastoplasticity.
This requires both time AND strain arrays as inputs.

For constant shear rate startup:
    strain = gamma_dot * time

For LAOS:
    Use the strain data already loaded from load_pnas_laos()

ROOT CAUSE 2: JAX Autodiff Issue (NB03, NB04)
---------------------------------------------
NLSQ optimizer uses forward-mode autodiff (jvp) for Jacobians.
Diffrax ODE solver uses custom_vjp with checkpointing.
These are incompatible: "can't apply jvp to custom_vjp function"

Workaround:
    Add method='scipy' to use finite-difference Jacobians instead

NOTEBOOKS THAT WORK:
-------------------
- NB01: Uses predict_flow_curve() specialized method (no strain needed)
- NB05: Fits to frequency-domain modulus (no time-domain strain needed)

TESTING:
--------
After applying fixes, run:
    cd /Users/b80985/Projects/rheojax/examples/ikh
    python test_mikh_notebooks.py

Expected: All 6 tests pass
""")

# ==============================================================================
# QUICK FIX VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    print("\nTo apply these fixes:")
    print("1. Edit each notebook manually (cells listed above)")
    print("2. OR use a notebook editing tool to update the code cells")
    print("3. Run test_mikh_notebooks.py to verify")
    print("\nKey principle:")
    print("  IKH models need BOTH time and strain for return mapping protocols")
    print("  Use method='scipy' for ODE protocols (relaxation, creep)")
